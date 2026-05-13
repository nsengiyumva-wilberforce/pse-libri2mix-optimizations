from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf


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

    @tf.function
    def preprocess_tf(self, filepath):
        wav = self.load_audio_tf(filepath)
        wav = wav / (tf.reduce_max(tf.abs(wav)) + 1e-8)
        return wav

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
        spectrogram_corr = tf.signal.stft(
            wav_corr, frame_length=self.frame_length, fft_length=self.n_fft, frame_step=self.frame_step
        )
        spectrogram_ref = tf.signal.stft(
            wav_ref, frame_length=self.frame_length, fft_length=self.n_fft, frame_step=self.frame_step
        )
        spectrogram = tf.signal.stft(
            wavclean, frame_length=self.frame_length, fft_length=self.n_fft, frame_step=self.frame_step
        )
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


_DEFAULT_TOOLKIT = AudioToolkit()


load_audio_py = _DEFAULT_TOOLKIT.load_audio_py
load_audio_tf = _DEFAULT_TOOLKIT.load_audio_tf
preprocess_tf = _DEFAULT_TOOLKIT.preprocess_tf
split_into_chunks = _DEFAULT_TOOLKIT.split_into_chunks
tf_rms = _DEFAULT_TOOLKIT.tf_rms
convert_to_spectrogram = _DEFAULT_TOOLKIT.convert_to_spectrogram
complex_to_2ch = AudioToolkit.complex_to_2ch
sample_reference_segments = _DEFAULT_TOOLKIT.sample_reference_segments
