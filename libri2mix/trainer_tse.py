"""SEF-PNet's SiSnrTrainer, driving the Keras enhancer.

Matches nnet/libs/trainer_unet_tse_steplr_clip.py and train.sh:

- Adam learning rate 5e-4, weight decay 1e-5, global-norm clip 1
- through epoch 100, multiply the learning rate by 0.98 every 2 epochs
- after that, multiply it by 0.9 every epoch
- loss is the mean of negative waveform SI-SNR, with samples past valid_len zeroed
- plus a small penalty on the fraction of estimate energy that lies along mix - target

The chunk loader yields waveforms. This loop converts them with the convolution
STFT, runs the enhancer, and scores the waveform, so the schedule counts
optimizer steps on chunks rather than utterances.
"""

import queue
import threading
import time

import tensorflow as tf

from .audio import conv_stft_frames
from .conf_unet_tse_32ms import adam_kwargs

# How strongly to punish estimate energy that lies along the other speaker.
INTERFERER_WEIGHT = 0.1
# Longest auxiliary in this corpus is 17.61 s. A fixed width keeps the compiled step.
_MAX_AUX_SECONDS = 18


def _sisnr(estimate, reference, eps=1e-8):
    """SEF-PNet sisnr: 20 log10 of the amplitude ratio. One value per batch item."""
    estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
    reference = reference - tf.reduce_mean(reference, axis=-1, keepdims=True)
    reference_energy = tf.reduce_sum(tf.square(reference), axis=-1, keepdims=True)
    dot = tf.reduce_sum(estimate * reference, axis=-1, keepdims=True)
    projection = dot * reference / (reference_energy + eps)
    noise = estimate - projection
    ratio = tf.norm(projection, axis=-1) / (tf.norm(noise, axis=-1) + eps)
    return 20.0 * tf.math.log(ratio + eps) / tf.math.log(10.0)


def _interferer_leak(estimate, interferer, eps=1e-8):
    """10 log10 of the fraction of estimate energy lying along the interferer.

    More negative means less of the other speaker remains. Scale-invariant.
    """
    estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
    interferer = interferer - tf.reduce_mean(interferer, axis=-1, keepdims=True)
    denom = tf.reduce_sum(tf.square(interferer), axis=-1, keepdims=True) + eps
    dot = tf.reduce_sum(estimate * interferer, axis=-1, keepdims=True)
    leak = (dot / denom) * interferer
    ratio = tf.reduce_sum(tf.square(leak), axis=-1) / (
        tf.reduce_sum(tf.square(estimate), axis=-1) + eps
    )
    return 10.0 * tf.math.log(ratio + eps) / tf.math.log(10.0)


def _mask_by_length(waveforms, lengths):
    positions = tf.range(tf.shape(waveforms)[-1])
    mask = tf.cast(positions[None, :] < tf.cast(lengths, tf.int32)[:, None], waveforms.dtype)
    return waveforms * mask


def _enrollment_batch(toolkit, aux, aux_len, max_frames):
    """Pad a batch of enrollment waveforms to the static spectrogram length."""
    spectrum = toolkit.stft(tf.convert_to_tensor(aux, tf.float32))
    frames = tf.shape(spectrum)[1]
    bins = spectrum.shape[-1]
    pad = tf.maximum(0, max_frames - frames)
    spectrum = tf.pad(spectrum, [[0, 0], [0, pad], [0, 0]])[:, :max_frames, :]
    spectrum.set_shape([None, max_frames, bins])
    spec_2ch = tf.stack([tf.math.real(spectrum), tf.math.imag(spectrum)], axis=-1)

    win = toolkit.frame_length
    hop = toolkit.frame_step
    edge = win - hop
    index = tf.range(max_frames)
    start = index * hop
    end = start + win
    real_end = edge + tf.cast(aux_len, tf.int32)
    valid = tf.logical_and(start[None, :] < real_end[:, None], end[None, :] > edge)
    # Frames past this batch's spectrogram are padding.
    valid = tf.logical_and(valid, index[None, :] < frames)
    return spec_2ch, tf.cast(valid, tf.float32)


class SiSnrTrainer(object):
    def __init__(
        self,
        model,
        toolkit,
        max_ref_frames,
        checkpoint,
        logging_period=200,
        no_impr=150,
        strategy=None,
    ):
        self.model = model
        self.toolkit = toolkit
        self.max_ref_frames = max_ref_frames
        self.checkpoint = checkpoint
        self.logging_period = logging_period
        self.no_impr = no_impr
        self.cur_epoch = 0
        self.strategy = strategy or tf.distribute.get_strategy()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(adam_kwargs["lr"]),
            weight_decay=float(adam_kwargs["weight_decay"]),
            clipnorm=1.0,
        )
        # Create momentum slots now. Doing it inside the compiled training step
        # leaves an unfed initializer placeholder on the replica GPUs.
        self.optimizer.build(list(self.model.trainable_variables))
        for variable in self.optimizer.variables:
            if hasattr(variable, "numpy"):
                variable.numpy()

    def _waveforms(self, mix, reference, aux_spec, aux_mask, training):
        mix_spec = self.toolkit.stft(mix)
        predicted = self.model(
            {
                "noisy_main": tf.stack([tf.math.real(mix_spec), tf.math.imag(mix_spec)], axis=-1),
                "noisy_ref": aux_spec,
                "ref_mask": aux_mask,
            },
            training=training,
        )
        estimate = self.toolkit.istft(tf.complex(predicted[..., 0], predicted[..., 1]))
        # Score against the chunk waveform, as SiSnrTrainer does, not a second STFT.
        target = reference
        if estimate.shape[-1] is not None and reference.shape[-1] is not None:
            width = min(int(estimate.shape[-1]), int(reference.shape[-1]))
            estimate = estimate[..., :width]
            target = target[..., :width]
        else:
            width = tf.minimum(tf.shape(estimate)[-1], tf.shape(target)[-1])
            estimate = estimate[..., :width]
            target = target[..., :width]
        return estimate, target

    def _replica_batch(self, mix, reference, valid_len, aux, aux_len):
        """Each replica keeps an equal slice of the global batch."""
        ctx = tf.distribute.get_replica_context()
        n = 1 if ctx is None else ctx.num_replicas_in_sync
        if n == 1:
            return mix, reference, valid_len, aux, aux_len
        per = tf.shape(mix)[0] // n
        start = ctx.replica_id_in_sync_group * per
        stop = start + per
        return (
            mix[start:stop],
            reference[start:stop],
            valid_len[start:stop],
            aux[start:stop],
            aux_len[start:stop],
        )

    def compute_loss(self, mix, reference, valid_len, aux, aux_len, training):
        mix, reference, valid_len, aux, aux_len = self._replica_batch(
            mix, reference, valid_len, aux, aux_len
        )
        aux_spec, aux_mask = _enrollment_batch(self.toolkit, aux, aux_len, self.max_ref_frames)
        estimate, target = self._waveforms(mix, reference, aux_spec, aux_mask, training)
        mix = mix[..., : tf.shape(estimate)[-1]]
        estimate = _mask_by_length(estimate, valid_len)
        target = _mask_by_length(target, valid_len)
        mix = _mask_by_length(mix, valid_len)
        # mix_clean is target + interferer, so the other speaker is this residual.
        interferer = mix - target
        sisdr = _sisnr(estimate, target)
        leak = _interferer_leak(estimate, interferer)
        objective = -tf.reduce_mean(sisdr) + INTERFERER_WEIGHT * tf.reduce_mean(leak)
        return objective, tf.reduce_mean(sisdr)

    def _batch_tensors(self, batch):
        return (
            tf.convert_to_tensor(batch["mix"], tf.float32),
            tf.convert_to_tensor(batch["ref"], tf.float32),
            tf.convert_to_tensor(batch["valid_len"], tf.int32),
            self._pad_aux_wave(tf.convert_to_tensor(batch["aux"], tf.float32)),
            tf.convert_to_tensor(batch["aux_len"], tf.int32),
        )

    def _pad_aux_wave(self, aux):
        width = _MAX_AUX_SECONDS * int(self.toolkit.target_sr)
        if aux.shape[1] is not None and int(aux.shape[1]) > width:
            raise ValueError(
                f"enrollment is {int(aux.shape[1])} samples, longer than {width}"
            )
        aux = tf.pad(aux, [[0, 0], [0, tf.maximum(width - tf.shape(aux)[1], 0)]])[:, :width]
        aux.set_shape([None, width])
        return aux

    def _reduce_step(self, per_loss, per_sisdr):
        # Each replica returns local_mean / num_replicas. SUM restores the global mean.
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss, axis=None)
        sisdr = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_sisdr, axis=None)
        return loss, sisdr

    def train_step(self, batch):
        return self._distributed_train(*self._batch_tensors(batch))

    @tf.function
    def _distributed_train(self, mix, reference, valid_len, aux, aux_len):
        replicas = tf.cast(self.strategy.num_replicas_in_sync, tf.float32)

        def step_fn(mix, reference, valid_len, aux, aux_len):
            with tf.GradientTape() as tape:
                loss, sisdr = self.compute_loss(
                    mix, reference, valid_len, aux, aux_len, training=True
                )
                # MirroredStrategy sums gradients. Divide so that sum matches
                # the mean over the global batch.
                scaled = loss / replicas
            gradients = tape.gradient(scaled, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return scaled, sisdr

        per_loss, per_sisdr = self.strategy.run(
            step_fn, args=(mix, reference, valid_len, aux, aux_len)
        )
        return self._reduce_step(per_loss, per_sisdr)

    def eval_step(self, batch):
        return self._distributed_eval(*self._batch_tensors(batch))

    @tf.function
    def _distributed_eval(self, mix, reference, valid_len, aux, aux_len):
        replicas = tf.cast(self.strategy.num_replicas_in_sync, tf.float32)

        def step_fn(mix, reference, valid_len, aux, aux_len):
            loss, sisdr = self.compute_loss(
                mix, reference, valid_len, aux, aux_len, training=False
            )
            return loss / replicas, sisdr

        per_loss, per_sisdr = self.strategy.run(
            step_fn, args=(mix, reference, valid_len, aux, aux_len)
        )
        return self._reduce_step(per_loss, per_sisdr)

    def _run_epoch(self, loader, training):
        losses = []
        sisdrs = []
        started = time.time()
        step = self.train_step if training else self.eval_step
        phase = "train" if training else "dev"
        print(f"{phase}: loading the first batch, then compiling the first step...", flush=True)
        # The loader reads whole enrollment wavs on the CPU. Fill the next
        # batch while the GPU runs this one, instead of waiting out the read.
        batches = queue.Queue(maxsize=4)
        def produce():
            try:
                for batch in loader:
                    batches.put(batch)
            finally:
                batches.put(None)
        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        batch_index = 0
        while True:
            batch = batches.get()
            if batch is None:
                break
            batch_index += 1
            if batch_index == 1:
                print(f"{phase}: first batch ready, compiling...", flush=True)
            loss, sisdr = step(batch)
            losses.append(float(loss))
            sisdrs.append(float(sisdr))
            if batch_index == 1:
                print(
                    f"{phase}: first step done "
                    f"(loss = {losses[-1]:+.2f}, SI-SDR {sisdrs[-1]:.2f} dB)",
                    flush=True,
                )
            if batch_index % self.logging_period == 0:
                recent = sum(losses[-self.logging_period :]) / self.logging_period
                recent_sisdr = sum(sisdrs[-self.logging_period :]) / self.logging_period
                print(
                    f"Processed {batch_index} batches "
                    f"(loss = {recent:+.2f}, SI-SDR {recent_sisdr:.2f} dB)...",
                    flush=True,
                )
        producer.join()
        mean_loss = sum(losses) / len(losses)
        mean_sisdr = sum(sisdrs) / len(sisdrs)
        minutes = (time.time() - started) / 60.0
        return {"loss": mean_loss, "sisdr": mean_sisdr, "batches": len(losses), "cost": minutes}

    def _learning_rate(self):
        value = self.optimizer.learning_rate
        # Under MirroredStrategy this is one variable per GPU. .numpy() reads
        # the primary replica. float() does not, because .value stays mirrored.
        if hasattr(value, "numpy"):
            value = value.numpy()
        elif callable(value):
            value = value()
        return float(value)

    def _set_learning_rate(self, value):
        current = self.optimizer.learning_rate
        if hasattr(current, "assign"):
            current.assign(value)
        else:
            self.optimizer.learning_rate = value

    def _decay_learning_rate(self):
        # StepLR(step_size=2, gamma=0.98) while epoch <= 100, else every epoch * 0.9.
        if self.cur_epoch <= 100 and self.cur_epoch % 2 != 0:
            return
        gamma = 0.98 if self.cur_epoch <= 100 else 0.9
        self._set_learning_rate(self._learning_rate() * gamma)

    def run(self, train_loader, dev_loader, num_epochs=200):
        print(
            f"SiSnrTrainer: lr={self._learning_rate():.3e}, "
            f"weight_decay={float(adam_kwargs['weight_decay']):.1e}, clipnorm=1",
            flush=True,
        )
        dev = self._run_epoch(dev_loader, training=False)
        best_loss = dev["loss"]
        print(
            f"START FROM EPOCH {self.cur_epoch}, LOSS = {best_loss:.4f} "
            f"(SI-SDR {dev['sisdr']:.2f} dB)",
            flush=True,
        )
        no_impr = 0
        while self.cur_epoch < num_epochs:
            self.cur_epoch += 1
            current_lr = self._learning_rate()
            train = self._run_epoch(train_loader, training=True)
            dev = self._run_epoch(dev_loader, training=False)
            note = ""
            if dev["loss"] < best_loss:
                best_loss = dev["loss"]
                no_impr = 0
                self.model.save(self.checkpoint)
                note = f"| saved {self.checkpoint}"
            else:
                no_impr += 1
                note = f"| no impr, best = {best_loss:.4f}"
            print(
                f"Loss(time/N, lr={current_lr:.3e}) - Epoch {self.cur_epoch:2d}: "
                f"train = {train['loss']:+.4f} (SI-SDR {train['sisdr']:.2f} dB, "
                f"{train['cost']:.2f}m/{train['batches']}) | "
                f"dev = {dev['loss']:+.4f} (SI-SDR {dev['sisdr']:.2f} dB, "
                f"{dev['cost']:.2f}m/{dev['batches']}) {note}",
                flush=True,
            )
            self._decay_learning_rate()
            if no_impr == self.no_impr:
                print(f"Stop training cause no impr for {no_impr} epochs", flush=True)
                break
        print(f"Training for {self.cur_epoch}/{num_epochs} epochs done!", flush=True)


def chunk_frames_match(chunk_size, frame_length, frame_step, n_frames):
    """The model input time axis has to equal the convolution STFT of one chunk."""
    return conv_stft_frames(chunk_size, frame_length, frame_step) == n_frames
