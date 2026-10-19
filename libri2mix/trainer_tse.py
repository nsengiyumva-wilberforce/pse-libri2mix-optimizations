"""SEF-PNet's SiSnrTrainer, driving the Keras enhancer.

Matches the SEF-PNet paper schedule:

- Adam learning rate 5e-4, weight decay 1e-5, global-norm clip 1
- through epoch 100, multiply the learning rate by 0.98 every 2 epochs
- for the last 20 epochs, multiply it by 0.9 every epoch
- stop at 120 epochs
- loss is the mean of negative waveform SI-SNR, with samples past valid_len zeroed

The chunk loader yields waveforms. This loop converts them with the convolution
STFT, runs the enhancer, and scores the waveform, so the schedule counts
optimizer steps on chunks rather than utterances. Each enrollment is padded to
a fixed width, and frames past that utterance's own length are masked off.
"""

import queue
import threading
import time

import numpy as np
import tensorflow as tf

from .audio import conv_stft_frames
from .conf_unet_tse_32ms import adam_kwargs

# bfloat16 compute, float32 weights. The 4090 tensor cores run this without
# loss scaling. The policy has to be set before the enhancer is built, and
# this module is imported before _build_model().
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

# Longest auxiliary in this corpus is 17.61 s. A fixed width keeps the compiled step.
_MAX_AUX_SECONDS = 18


def _sisnr(estimate, reference, eps=1e-8):
    """SEF-PNet sisnr: 10 log10 of the power ratio. One value per batch item.

    Epsilon is inside both energies. ``tf.norm`` differentiates as ``x / ||x||``,
    which is NaN at the origin, and a zero mask makes the estimate exactly that.
    """
    estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
    reference = reference - tf.reduce_mean(reference, axis=-1, keepdims=True)
    reference_energy = tf.reduce_sum(tf.square(reference), axis=-1, keepdims=True)
    dot = tf.reduce_sum(estimate * reference, axis=-1, keepdims=True)
    projection = dot * reference / (reference_energy + eps)
    noise = estimate - projection
    projection_energy = tf.reduce_sum(tf.square(projection), axis=-1)
    noise_energy = tf.reduce_sum(tf.square(noise), axis=-1)
    ratio = (projection_energy + eps) / (noise_energy + eps)
    return 10.0 * tf.math.log(ratio) / tf.math.log(10.0)


def _mask_by_length(waveforms, lengths):
    positions = tf.range(tf.shape(waveforms)[-1])
    mask = tf.cast(
        positions[None, :] < tf.cast(lengths, tf.int32)[:, None], waveforms.dtype
    )
    return waveforms * mask


def _enrollment_batch(toolkit, aux, aux_lengths, max_frames):
    """STFT enrollments padded to a fixed width.

    ``aux_lengths`` is each utterance's sample count before padding. Frames past
    that length are masked off, including the zeros added to match a longer
    enrollment in the same batch.
    """
    spectrum = toolkit.stft(tf.convert_to_tensor(aux, tf.float32))
    frames = tf.shape(spectrum)[1]
    bins = spectrum.shape[-1]
    pad = tf.maximum(0, max_frames - frames)
    spectrum = tf.pad(spectrum, [[0, 0], [0, pad], [0, 0]])[:, :max_frames, :]
    spectrum.set_shape([None, max_frames, bins])
    spec_2ch = tf.stack([tf.math.real(spectrum), tf.math.imag(spectrum)], axis=-1)

    win = toolkit.frame_length
    hop = toolkit.frame_step
    n_valid = (tf.cast(aux_lengths, tf.int32) + 2 * (win - hop) - win) // hop + 1
    n_valid = tf.reshape(n_valid, [-1])
    index = tf.range(max_frames)
    valid = tf.logical_and(index[None, :] < n_valid[:, None], index[None, :] < frames)
    return spec_2ch, tf.cast(valid, tf.float32)


def _spec_augment_enrollment(spec, valid):
    """ECAPA-TDNN SpecAugment on the enrollment spectrogram only.

    One time mask of 0–5 frames and one frequency mask of 0–10 bins
    (Desplanques et al., Interspeech 2020, arXiv:2005.07143). Masked bins
    are set to zero. The mixture spectrogram is left alone.
    """
    batch = tf.shape(spec)[0]
    frames = tf.shape(spec)[1]
    bins = tf.shape(spec)[2]

    time_width = tf.random.uniform([batch], minval=0, maxval=6, dtype=tf.int32)
    n_valid = tf.reduce_sum(tf.cast(valid > 0.0, tf.int32), axis=1)
    time_room = tf.maximum(n_valid - time_width, 0)
    time_start = tf.cast(
        tf.random.uniform([batch]) * tf.cast(time_room + 1, tf.float32),
        tf.int32,
    )
    time_index = tf.range(frames)[None, :]
    drop_time = tf.logical_and(
        time_index >= time_start[:, None],
        time_index < time_start[:, None] + time_width[:, None],
    )

    freq_width = tf.random.uniform([batch], minval=0, maxval=11, dtype=tf.int32)
    freq_room = tf.maximum(bins - freq_width, 0)
    freq_start = tf.cast(
        tf.random.uniform([batch]) * tf.cast(freq_room + 1, tf.float32),
        tf.int32,
    )
    freq_index = tf.range(bins)[None, :]
    drop_freq = tf.logical_and(
        freq_index >= freq_start[:, None],
        freq_index < freq_start[:, None] + freq_width[:, None],
    )

    keep = tf.logical_not(tf.logical_or(drop_time[:, :, None], drop_freq[:, None, :]))
    return spec * tf.cast(keep[..., None], spec.dtype)


class SiSnrTrainer:
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
        # XLA compiles the replica step only. Compiling the function that calls
        # strategy.run makes GPU 0 read replica variables that live on GPU 1.
        self._xla_train = tf.function(self._train_replica, jit_compile=True)
        self._xla_eval = tf.function(self._eval_replica, jit_compile=True)

    def _waveforms(self, mix, reference, aux_spec, aux_mask, training):
        mix_spec = self.toolkit.stft(mix)
        predicted = self.model(
            {
                "noisy_main": tf.stack(
                    [tf.math.real(mix_spec), tf.math.imag(mix_spec)], axis=-1
                ),
                "noisy_ref": aux_spec,
                "ref_mask": aux_mask,
            },
            training=training,
        )
        # The network may emit bfloat16. The convolution ISTFT and SI-SNR stay float32.
        predicted = tf.cast(predicted, tf.float32)
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
        if ctx is None:
            return mix, reference, valid_len, aux, aux_len
        n = ctx.num_replicas_in_sync
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
        aux_spec, aux_mask = _enrollment_batch(
            self.toolkit, aux, aux_len, self.max_ref_frames
        )
        if training:
            aux_spec = _spec_augment_enrollment(aux_spec, aux_mask)
        estimate, target = self._waveforms(mix, reference, aux_spec, aux_mask, training)
        estimate = _mask_by_length(estimate, valid_len)
        target = _mask_by_length(target, valid_len)
        sisdr = _sisnr(estimate, target)
        # SEF-PNet: -sum(sisnr) / N, which is the mean of the per-clip scores.
        objective = -tf.reduce_mean(sisdr)
        return objective, tf.reduce_mean(sisdr)

    def _batch_tensors(self, batch):
        aux = tf.convert_to_tensor(batch["aux"], tf.float32)
        # True length of each enrollment, before the batch pad and the fixed-width pad.
        aux_len = tf.cast(tf.convert_to_tensor(batch["aux_len"]), tf.int32)
        return (
            tf.convert_to_tensor(batch["mix"], tf.float32),
            tf.convert_to_tensor(batch["ref"], tf.float32),
            tf.convert_to_tensor(batch["valid_len"], tf.int32),
            self._pad_aux_wave(aux),
            aux_len,
        )

    def _pad_aux_wave(self, aux):
        width = _MAX_AUX_SECONDS * int(self.toolkit.target_sr)
        if aux.shape[1] is not None and int(aux.shape[1]) > width:
            raise ValueError(
                f"enrollment is {int(aux.shape[1])} samples, longer than {width}"
            )
        aux = tf.pad(aux, [[0, 0], [0, tf.maximum(width - tf.shape(aux)[1], 0)]])[
            :, :width
        ]
        aux.set_shape([None, width])
        return aux

    def _reduce_step(self, per_loss, per_sisdr):
        # Each replica returns local_mean / num_replicas. SUM restores the global mean.
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss, axis=None)
        sisdr = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_sisdr, axis=None)
        return loss, sisdr

    def _train_replica(self, mix, reference, valid_len, aux, aux_len):
        # Gradients only. The all-reduce inside apply_gradients cannot run in
        # this compiled function while it is called from strategy.run.
        replicas = tf.cast(self.strategy.num_replicas_in_sync, tf.float32)
        with tf.GradientTape() as tape:
            loss, sisdr = self.compute_loss(
                mix, reference, valid_len, aux, aux_len, training=True
            )
            # MirroredStrategy sums gradients. Divide so that sum matches
            # the mean over the global batch.
            scaled = loss / replicas
        gradients = tape.gradient(scaled, self.model.trainable_variables)
        return scaled, sisdr, gradients

    def _eval_replica(self, mix, reference, valid_len, aux, aux_len):
        replicas = tf.cast(self.strategy.num_replicas_in_sync, tf.float32)
        loss, sisdr = self.compute_loss(
            mix, reference, valid_len, aux, aux_len, training=False
        )
        return loss / replicas, sisdr

    def train_step(self, batch):
        return self._distributed_train(*self._batch_tensors(batch))

    @tf.function
    def _distributed_train(self, mix, reference, valid_len, aux, aux_len):
        def step_fn(mix, reference, valid_len, aux, aux_len):
            mix, reference, valid_len, aux, aux_len = self._replica_batch(
                mix, reference, valid_len, aux, aux_len
            )
            scaled, sisdr, gradients = self._xla_train(
                mix, reference, valid_len, aux, aux_len
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            return scaled, sisdr

        per_loss, per_sisdr = self.strategy.run(
            step_fn, args=(mix, reference, valid_len, aux, aux_len)
        )
        return self._reduce_step(per_loss, per_sisdr)

    def eval_step(self, batch):
        return self._distributed_eval(*self._batch_tensors(batch))

    @tf.function
    def _distributed_eval(self, mix, reference, valid_len, aux, aux_len):
        def step_fn(mix, reference, valid_len, aux, aux_len):
            mix, reference, valid_len, aux, aux_len = self._replica_batch(
                mix, reference, valid_len, aux, aux_len
            )
            return self._xla_eval(mix, reference, valid_len, aux, aux_len)

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
        print(
            f"{phase}: loading the first batch, then compiling the first step...",
            flush=True,
        )
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
        return {
            "loss": mean_loss,
            "sisdr": mean_sisdr,
            "batches": len(losses),
            "cost": minutes,
        }

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
        # First 100 epochs: x0.98 every 2 epochs. Epochs 101-120: x0.9 every epoch.
        if self.cur_epoch <= 100 and self.cur_epoch % 2 != 0:
            return
        gamma = 0.98 if self.cur_epoch <= 100 else 0.9
        self._set_learning_rate(self._learning_rate() * gamma)

    def run(self, train_loader, dev_loader, num_epochs=120):
        print(
            f"SiSnrTrainer: lr={self._learning_rate():.3e}, "
            f"weight_decay={float(adam_kwargs['weight_decay']):.1e}, clipnorm=1, "
            f"XLA jit_compile=True, policy={tf.keras.mixed_precision.global_policy().name}",
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
                _save_model(self.model, self.checkpoint)
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


def _host_array(value):
    """A real ndarray for a weight.

    ``model.save`` uses ``np.array(variable)``. A scalar mirrored weight's
    ``__array__`` returns a NumPy scalar, and NumPy raises
    ``object __array__ method not producing an array``. Reading ``.numpy()``
    and wrapping that skips ``__array__``.
    """
    if isinstance(value, tf.distribute.DistributedValues):
        value = value.values[0]
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _save_model(model, path):
    import keras.src.backend as keras_backend

    original = keras_backend.convert_to_numpy

    def convert_to_numpy(value):
        if isinstance(value, tf.distribute.DistributedValues) or hasattr(
            value, "numpy"
        ):
            return _host_array(value)
        return original(value)

    keras_backend.convert_to_numpy = convert_to_numpy
    try:
        model.save(path)
    finally:
        keras_backend.convert_to_numpy = original


def chunk_frames_match(chunk_size, frame_length, frame_step, n_frames):
    """The model input time axis has to equal the convolution STFT of one chunk."""
    return conv_stft_frames(chunk_size, frame_length, frame_step) == n_frames
