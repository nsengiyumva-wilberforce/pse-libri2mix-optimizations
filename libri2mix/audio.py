from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf


def conv_stft_frames(num_samples, frame_length, frame_step):
    """Frame count of the SEF-PNet convolution STFT, including its edge pad."""
    pad = frame_length - frame_step
    return (int(num_samples) + 2 * pad - frame_length) // frame_step + 1


_CONV_STFT_FILTERS = {}


def _conv_stft_filters(frame_length, frame_step, n_fft):
    """Analysis and synthesis filters from SEF-PNet's ConvSTFT / ConviSTFT."""
    key = (int(frame_length), int(frame_step), int(n_fft))
    cached = _CONV_STFT_FILTERS.get(key)
    if cached is not None:
        return cached
    samples = np.arange(frame_length, dtype=np.float64)
    window = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * samples / frame_length))
    basis = np.fft.rfft(np.eye(n_fft, dtype=np.float64))[:frame_length]
    kernel = np.concatenate([basis.real, basis.imag], axis=1).T
    forward = kernel * window
    inverse = np.linalg.pinv(kernel).T * window

    def _as_filters(matrix):
        # (channels, window) -> conv layout (window, 1, channels)
        laid_out = np.transpose(matrix, (1, 0))[:, None, :]
        return tf.constant(laid_out.astype(np.float32))

    # Build eager tensors. Creating them inside a tf.data trace captures a
    # symbolic tensor and the pipeline cannot serialize it.
    with tf.init_scope():
        cached = (
            _as_filters(forward),
            _as_filters(inverse),
            tf.constant(window.astype(np.float32)),
        )
    _CONV_STFT_FILTERS[key] = cached
    return cached


@dataclass(frozen=True)
class AudioToolkit:
    target_sr: int = 8000
    frame_length: int = 400
    frame_step: int = 160
    n_fft: int = 510

    def load_audio_py(self, path):
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        elif isinstance(path, np.ndarray):
            path = path.item().decode("utf-8") if path.dtype.type is np.bytes_ else path.item()
        audio, sr = sf.read(path)
        audio = audio.astype("float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio

    def load_audio_tf(self, path):
        audio = tf.numpy_function(self.load_audio_py, [path], tf.float32)
        audio.set_shape([None])
        return audio

    def stft(self, wav):
        """SEF-PNet complex STFT. Accepts (samples,) or (batch, samples)."""
        squeezed = wav.shape.rank == 1
        if squeezed:
            wav = wav[None, :]
        filters, _, _ = _conv_stft_filters(self.frame_length, self.frame_step, self.n_fft)
        pad = self.frame_length - self.frame_step
        wav = tf.pad(wav, [[0, 0], [pad, pad]])[:, :, None]
        transformed = tf.nn.conv1d(wav, filters, stride=self.frame_step, padding="VALID")
        n_freq = self.n_fft // 2 + 1
        spectrum = tf.complex(transformed[:, :, :n_freq], transformed[:, :, n_freq:])
        if squeezed:
            spectrum = spectrum[0]
        return spectrum

    def istft(self, spectrum):
        """Inverse of ``stft``. Accepts (frames, bins) or (batch, frames, bins)."""
        squeezed = spectrum.shape.rank == 2
        if squeezed:
            spectrum = spectrum[None, ...]
        _, filters, window = _conv_stft_filters(self.frame_length, self.frame_step, self.n_fft)
        packed = tf.concat([tf.math.real(spectrum), tf.math.imag(spectrum)], axis=-1)
        frames = tf.shape(packed)[1]
        win = self.frame_length
        hop = self.frame_step
        out_len = (frames - 1) * hop + win
        batch = tf.shape(packed)[0]
        waveform = tf.nn.conv1d_transpose(
            packed,
            filters,
            output_shape=tf.stack([batch, out_len, 1]),
            strides=hop,
            padding="VALID",
        )[:, :, 0]
        positions = tf.range(win)[None, :] + tf.range(frames)[:, None] * hop
        weights = tf.tile(window[None, :] ** 2, [frames, 1])
        norm = tf.tensor_scatter_nd_add(
            tf.zeros([out_len], dtype=waveform.dtype),
            positions[:, :, None],
            weights,
        )
        waveform = waveform / (norm[None, :] + 1e-8)
        edge = win - hop
        waveform = waveform[:, edge:-edge]
        if squeezed:
            waveform = waveform[0]
        return waveform

    @tf.function
    def preprocess_tf(self, filepath):
        # LibriMix stores mix_clean = s1 + s2. Keep that scale; do not peak-normalize.
        return self.load_audio_tf(filepath)

    @tf.function
    def split_into_chunks(self, wav, chunk_size, stride):
        length = tf.shape(wav)[0]
        tf.debugging.assert_positive(chunk_size, message="chunk_size must be positive")
        tf.debugging.assert_positive(stride, message="stride must be positive")

        def pad_to_chunk():
            pad_len = chunk_size - length
            return tf.pad(wav, [[0, pad_len]])

        def pad_to_stride():
            remainder = tf.math.floormod(length - chunk_size, stride)
            pad_len = tf.math.floormod(stride - remainder, stride)
            return tf.pad(wav, [[0, pad_len]])

        wav = tf.cond(length < chunk_size, pad_to_chunk, pad_to_stride)
        length = tf.shape(wav)[0]
        starts = tf.range(0, length - chunk_size + 1, stride)

        def get_chunk(s):
            return wav[s : s + chunk_size]

        chunks = tf.map_fn(get_chunk, starts, fn_output_signature=tf.float32)
        tf.debugging.assert_greater(tf.shape(chunks)[0], 0, message="chunking produced no chunks")
        return chunks

    @tf.function
    def tf_rms(self, x, eps=1e-8):
        return tf.sqrt(tf.reduce_mean(tf.square(x)) + eps)

    @tf.function
    def convert_to_spectrogram(self, wav_corr, wav_ref, wavclean):
        spectrogram_corr = self.stft(wav_corr)
        spectrogram_ref = self.stft(wav_ref)
        spectrogram = self.stft(wavclean)
        spectrogram_corr = tf.expand_dims(spectrogram_corr, axis=2)
        spectrogram_ref = tf.expand_dims(spectrogram_ref, axis=2)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram_corr, spectrogram_ref, spectrogram

    @staticmethod
    def complex_to_2ch(spec):
        return tf.stack([tf.math.real(spec), tf.math.imag(spec)], axis=-1)

    @tf.function
    def sample_reference_segments(self, wav, K, segment_len):
        wav_len = tf.shape(wav)[0]

        def pad():
            pad_len = segment_len - wav_len
            wav_pad = tf.pad(wav, [[0, pad_len]])
            return tf.tile(tf.expand_dims(wav_pad, 0), [K, 1])

        def sample():
            max_start = wav_len - segment_len
            starts = tf.random.uniform([K], 0, max_start + 1, dtype=tf.int32)
            return tf.map_fn(lambda s: wav[s : s + segment_len], starts, fn_output_signature=tf.float32)

        return tf.cond(wav_len < segment_len, pad, sample)

    def full_reference_spectrogram(self, wav, max_frames=None):
        """Spectrogram of a whole auxiliary utterance, with a frame mask of ones.

        Unlike ``sample_reference_segments``, the utterance is not cropped into
        random views. When ``max_frames`` is set, the time axis is padded to that
        fixed length so the reference encoder can use a static chunk count.
        """
        spectrogram = self.stft(wav)
        n_bins = self.n_fft // 2 + 1
        if max_frames is None:
            spectrogram.set_shape([None, n_bins])
            mask = tf.ones([tf.shape(spectrogram)[0]], dtype=tf.float32)
            mask.set_shape([None])
            return spectrogram, mask

        length = tf.shape(spectrogram)[0]
        spectrogram = spectrogram[:max_frames]
        valid = tf.minimum(length, max_frames)
        mask = tf.ones([valid], dtype=tf.float32)
        pad = max_frames - tf.shape(spectrogram)[0]
        spectrogram = tf.pad(spectrogram, [[0, pad], [0, 0]])
        mask = tf.pad(mask, [[0, pad]])
        spectrogram.set_shape([max_frames, n_bins])
        mask.set_shape([max_frames])
        return spectrogram, mask


_DEFAULT_TOOLKIT = AudioToolkit()


load_audio_py = _DEFAULT_TOOLKIT.load_audio_py
load_audio_tf = _DEFAULT_TOOLKIT.load_audio_tf
preprocess_tf = _DEFAULT_TOOLKIT.preprocess_tf
split_into_chunks = _DEFAULT_TOOLKIT.split_into_chunks
tf_rms = _DEFAULT_TOOLKIT.tf_rms
convert_to_spectrogram = _DEFAULT_TOOLKIT.convert_to_spectrogram
complex_to_2ch = AudioToolkit.complex_to_2ch
sample_reference_segments = _DEFAULT_TOOLKIT.sample_reference_segments
full_reference_spectrogram = _DEFAULT_TOOLKIT.full_reference_spectrogram
