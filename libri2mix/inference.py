from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import tensorflow as tf

from .audio import AudioToolkit, complex_to_2ch, conv_stft_frames


def _full_utterance_forward(model, noisy, ref, mask):
    """One forward on the whole mixture. The time axis stays symbolic so every file reuses one graph."""
    infer = getattr(model, "_full_pass", None)
    if infer is None:
        n_bins = int(noisy.shape[-2])
        ref_frames = int(ref.shape[1])

        @tf.function(
            input_signature=[
                tf.TensorSpec([1, None, n_bins, 2], tf.float32),
                tf.TensorSpec([1, ref_frames, n_bins, 2], tf.float32),
                tf.TensorSpec([1, ref_frames], tf.float32),
            ],
            jit_compile=False,
        )
        def infer(noisy_spec, ref_spec, ref_mask):
            output = model([noisy_spec, ref_spec, ref_mask], training=False)
            if isinstance(output, (list, tuple)):
                output = output[0]
            return output

        model._full_pass = infer
    output = infer(noisy, ref, mask)
    return output[0]


@dataclass
class WaveformEnhancer:
    audio_toolkit: AudioToolkit
    chunk_size: int
    frame_length: int
    frame_step: int
    n_fft: int

    def sample_reference_segments_full(self, wav, K, segment_len):
        wav_len = tf.shape(wav)[0]
        if wav_len < segment_len:
            pad_len = segment_len - wav_len
            wav_pad = tf.pad(wav, [[0, pad_len]])
            return tf.tile(tf.expand_dims(wav_pad, 0), [K, 1])
        starts = tf.linspace(0.0, tf.cast(wav_len - segment_len, tf.float32), K)
        starts = tf.cast(starts, tf.int32)
        return tf.map_fn(
            lambda s: wav[s : s + segment_len],
            starts,
            fn_output_signature=tf.TensorSpec(shape=[segment_len], dtype=tf.float32),
        )

    def enhance_audio_consistent(self, noisy_wav, ref_wav, model, K=4, overlap=0.5):
        if isinstance(noisy_wav, np.ndarray):
            noisy_wav = tf.convert_to_tensor(noisy_wav, tf.float32)
        if isinstance(ref_wav, np.ndarray):
            ref_wav = tf.convert_to_tensor(ref_wav, tf.float32)

        noisy_wav = noisy_wav.numpy()
        ref_wav = ref_wav.numpy()

        if not (0.0 < overlap < 1.0):
            raise ValueError("overlap must be in the open interval (0, 1)")

        chunk_len = self.chunk_size
        hop_len = int(chunk_len * (1 - overlap))
        if hop_len <= 0:
            raise ValueError("overlap produces an invalid hop length")

        ref_len = 16600
        # compute expected spectrogram shape for reference segments so fn_output_signature matches
        fft_bins = self.n_fft // 2 + 1
        # number of frames produced by tf.signal.stft for a signal of length ref_len
        if ref_len < self.frame_length:
            expected_ref_frames = 1
        else:
            expected_ref_frames = conv_stft_frames(ref_len, self.frame_length, self.frame_step)
        total_len = len(noisy_wav)
        if total_len == 0:
            return noisy_wav.astype(np.float32)

        enhanced_output = np.zeros(total_len)
        window_sum = np.zeros(total_len)
        window = np.ones(chunk_len, dtype=np.float32)

        ref_segments = self.sample_reference_segments_full(ref_wav, K=K, segment_len=ref_len)
        ref_specs_complex = tf.map_fn(
            lambda x: self.audio_toolkit.stft(x),
            ref_segments,
            fn_output_signature=tf.TensorSpec(shape=[expected_ref_frames, fft_bins], dtype=tf.complex64),
        )
        ref_specs = complex_to_2ch(ref_specs_complex)[None, ...]
        if not np.isfinite(ref_specs.numpy()).all():
            raise ValueError("reference spectrogram contains NaN or Inf")

        for start in range(0, total_len, hop_len):
            end = start + chunk_len
            chunk = noisy_wav[start:end]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))

            chunk_tf = tf.convert_to_tensor(chunk, tf.float32)
            noisy_spec = self.audio_toolkit.stft(chunk_tf)
            noisy_2ch = complex_to_2ch(noisy_spec)[None, ...]
            if not np.isfinite(noisy_2ch.numpy()).all():
                raise ValueError("input spectrogram contains NaN or Inf")

            enhanced_2ch = model.predict([noisy_2ch, ref_specs], verbose=0)[0]
            enhanced_2ch = np.asarray(enhanced_2ch)
            if enhanced_2ch.ndim != 3 or enhanced_2ch.shape[-1] != 2:
                raise ValueError(f"unexpected model output shape: {enhanced_2ch.shape}")
            if not np.isfinite(enhanced_2ch).all():
                raise ValueError("model output contains NaN or Inf")

            enhanced_complex = tf.complex(enhanced_2ch[..., 0], enhanced_2ch[..., 1])
            enhanced_chunk = self.audio_toolkit.istft(enhanced_complex).numpy()
            if not np.isfinite(enhanced_chunk).all():
                raise ValueError("inverse STFT produced NaN or Inf")

            valid_end = min(start + len(enhanced_chunk), total_len)
            valid_len = valid_end - start
            enhanced_output[start:valid_end] += enhanced_chunk[:valid_len] * window[:valid_len]
            window_sum[start:valid_end] += window[:valid_len]

            if end >= total_len:
                break

        window_sum[window_sum == 0] = 1e-8
        enhanced_output /= window_sum
        if not np.isfinite(enhanced_output).all():
            raise ValueError("enhanced output contains NaN or Inf")

        return enhanced_output.astype(np.float32)

    def enhance_audio_full_utterance(self, noisy_wav, ref_wav, model, overlap=0.5, max_ref_frames=1280):
        """Enhance with the whole auxiliary utterance, not sampled reference segments."""
        if isinstance(noisy_wav, np.ndarray):
            noisy_wav = tf.convert_to_tensor(noisy_wav, tf.float32)
        if isinstance(ref_wav, np.ndarray):
            ref_wav = tf.convert_to_tensor(ref_wav, tf.float32)

        noisy_wav = noisy_wav.numpy()
        ref_wav = tf.cast(ref_wav, tf.float32)

        total_len = len(noisy_wav)
        if total_len == 0:
            return noisy_wav.astype(np.float32)

        ref_spec, ref_mask = self.audio_toolkit.full_reference_spectrogram(
            ref_wav, max_frames=max_ref_frames
        )
        ref_specs = complex_to_2ch(ref_spec)[None, ...]
        ref_mask = ref_mask[None, ...]
        if not np.isfinite(ref_specs.numpy()).all():
            raise ValueError("reference spectrogram contains NaN or Inf")

        noisy_tf = tf.convert_to_tensor(noisy_wav, tf.float32)
        noisy_spec = self.audio_toolkit.stft(noisy_tf)
        noisy_2ch = complex_to_2ch(noisy_spec)[None, ...]
        if not np.isfinite(noisy_2ch.numpy()).all():
            raise ValueError("input spectrogram contains NaN or Inf")

        enhanced = _full_utterance_forward(model, noisy_2ch, ref_specs, ref_mask)
        enhanced = np.asarray(enhanced)
        if enhanced.ndim != 3 or enhanced.shape[-1] != 2:
            raise ValueError(f"unexpected model output shape: {enhanced.shape}")
        if not np.isfinite(enhanced).all():
            raise ValueError("model output contains NaN or Inf")

        enhanced_complex = tf.complex(enhanced[..., 0], enhanced[..., 1])
        enhanced_wav = self.audio_toolkit.istft(enhanced_complex).numpy()
        if not np.isfinite(enhanced_wav).all():
            raise ValueError("inverse STFT produced NaN or Inf")
        return enhanced_wav[:total_len].astype(np.float32)


_DEFAULT_ENHANCER = WaveformEnhancer(AudioToolkit(), chunk_size=31000, frame_length=400, frame_step=160, n_fft=510)

sample_reference_segments_full = _DEFAULT_ENHANCER.sample_reference_segments_full
enhance_audio_consistent = _DEFAULT_ENHANCER.enhance_audio_consistent
enhance_audio_full_utterance = _DEFAULT_ENHANCER.enhance_audio_full_utterance
