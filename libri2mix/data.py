from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from audiomentations import AddGaussianNoise, Compose, Gain, PolarityInversion, Shift
import tensorflow as tf

from .audio import AudioToolkit, conv_stft_frames


@dataclass
class LibriSpeechDatasetBuilder:
    audio_toolkit: AudioToolkit
    chunk_size: int
    stride: int

    def __post_init__(self):
        pass



    def load_scp(self, mix_path, ref_path, tgt_path):
        def get_dict(path):
            d = {}
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    key, value = parts
                    if not value:
                        continue
                    d[key] = value
            return d

        mix_d = get_dict(mix_path)
        ref_d = get_dict(ref_path)
        tgt_d = get_dict(tgt_path)

        common_keys = sorted(set(mix_d) & set(ref_d) & set(tgt_d))
        if len(common_keys) == 0:
            raise ValueError(
                "No overlapping keys found between SCP files.\n"
                "Check that all SCPs use identical utterance IDs."
            )

        mix_list = [mix_d[k] for k in common_keys]
        ref_list = [ref_d[k] for k in common_keys]
        tgt_list = [tgt_d[k] for k in common_keys]

        if ref_list[0] == tgt_list[0]:
            print("\n[WARNING] REF == TARGET → conditioning will collapse!")

        return mix_list, ref_list, tgt_list

    def load_libri_speech_triplet_multiview(self, mix_path, ref_path, tgt_path, K=4, ref_len=16600, is_train=False):
        clean = self.audio_toolkit.preprocess_tf(tgt_path)
        noisy = self.audio_toolkit.preprocess_tf(mix_path)
        ref = self.audio_toolkit.preprocess_tf(ref_path)
        clean.set_shape([None])
        noisy.set_shape([None])
        ref.set_shape([None])



        mix_chunks, clean_chunks, valid = self.split_pair(noisy, clean, is_train)
        ref_segments = self.audio_toolkit.sample_reference_segments(ref, K, ref_len)
        return mix_chunks, ref_segments, clean_chunks, valid

    def split_pair(self, mix, clean, training):
        """SEF-PNet chunks: 50% hop, train-time jitter, tail dropped, short files padded.

        ``valid`` is the number of real samples in each chunk. Padded samples stay
        out of the SI-SDR loss.
        """
        chunk = int(self.chunk_size)
        hop = int(self.stride)
        length = tf.shape(mix)[0]
        tf.debugging.assert_equal(
            length,
            tf.shape(clean)[0],
            message="mixture and target lengths differ",
        )
        training_flag = tf.constant(bool(training), dtype=tf.bool)

        def one_padded():
            pad = chunk - length
            mix_p = tf.reshape(tf.pad(mix, [[0, pad]]), [1, chunk])
            clean_p = tf.reshape(tf.pad(clean, [[0, pad]]), [1, chunk])
            valid = tf.reshape(length, [1])
            return mix_p, clean_p, valid

        def many():
            offset = tf.math.floormod(length, hop)
            start = tf.cond(
                tf.logical_and(training_flag, offset > 0),
                lambda: tf.random.uniform([], 0, offset + 1, dtype=tf.int32),
                lambda: tf.zeros([], dtype=tf.int32),
            )
            starts = tf.range(start, length - chunk + 1, hop)
            mix_c = tf.map_fn(
                lambda s: mix[s : s + chunk], starts, fn_output_signature=tf.float32
            )
            clean_c = tf.map_fn(
                lambda s: clean[s : s + chunk], starts, fn_output_signature=tf.float32
            )
            mix_c.set_shape([None, chunk])
            clean_c.set_shape([None, chunk])
            valid = tf.fill(tf.shape(starts), chunk)
            valid.set_shape([None])
            return mix_c, clean_c, valid

        mix_c, clean_c, valid = tf.cond(length < chunk, one_padded, many)
        mix_c.set_shape([None, chunk])
        clean_c.set_shape([None, chunk])
        valid.set_shape([None])
        return mix_c, clean_c, valid

    def _with_valid_channel(self, clean_2ch, valid):
        marker = tf.fill(tf.shape(clean_2ch)[:2], tf.cast(valid, clean_2ch.dtype))
        return tf.concat([clean_2ch, marker[..., None]], axis=-1)

    def convert_to_spectrogram_multiview(self, wav_corr, wav_ref_segments, wavclean):
        spectrogram_corr = self.audio_toolkit.stft(wav_corr)
        spectrogram_clean = self.audio_toolkit.stft(wavclean)
        spectrogram_refs = tf.map_fn(
            lambda x: self.audio_toolkit.stft(x),
            wav_ref_segments,
            fn_output_signature=tf.complex64,
        )
        return spectrogram_corr, spectrogram_refs, spectrogram_clean

    def configure_dataset(self, mixture_files, reference_files, target_files, is_train=True, K=4):
        ds = tf.data.Dataset.from_tensor_slices((mixture_files, reference_files, target_files))
        if is_train:
            ds = ds.shuffle(buffer_size=len(mixture_files))
        ds = ds.map(
            lambda n, r, t: self.load_libri_speech_triplet_multiview(n, r, t, K, is_train=is_train),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.interleave(
            lambda mix_chunks, ref_segments, clean_chunks, valid: tf.data.Dataset.from_tensor_slices(
                (mix_chunks, clean_chunks, valid)
            ).map(lambda m, c, v: (m, ref_segments, c, v)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if is_train:
            ds = ds.shuffle(2000)
        ds = ds.map(
            lambda mix, ref, clean, valid: (
                *self.convert_to_spectrogram_multiview(mix, ref, clean),
                valid,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            lambda spec_noisy, spec_refs, spec_clean, valid: (
                {
                    "noisy_main": self.audio_toolkit.complex_to_2ch(spec_noisy),
                    "noisy_ref": self.audio_toolkit.complex_to_2ch(spec_refs),
                },
                self._with_valid_channel(self.audio_toolkit.complex_to_2ch(spec_clean), valid),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    def load_libri_speech_triplet_full_utterance(
        self, mix_path, ref_path, tgt_path, max_ref_frames=1280, is_train=False
    ):
        clean = self.audio_toolkit.preprocess_tf(tgt_path)
        noisy = self.audio_toolkit.preprocess_tf(mix_path)
        ref = self.audio_toolkit.preprocess_tf(ref_path)
        clean.set_shape([None])
        noisy.set_shape([None])
        ref.set_shape([None])

        mix_chunks, clean_chunks, valid = self.split_pair(noisy, clean, is_train)
        ref_spec, ref_mask = self.audio_toolkit.full_reference_spectrogram(
            ref, max_frames=max_ref_frames
        )
        return mix_chunks, ref_spec, ref_mask, clean_chunks, valid

    def configure_dataset_full_utterance(
        self, mixture_files, reference_files, target_files, is_train=True, max_ref_frames=1280
    ):
        """Same mixture chunking as ``configure_dataset``, but each auxiliary file is kept whole."""
        ds = tf.data.Dataset.from_tensor_slices((mixture_files, reference_files, target_files))
        if is_train:
            ds = ds.shuffle(buffer_size=len(mixture_files))
        ds = ds.map(
            lambda n, r, t: self.load_libri_speech_triplet_full_utterance(
                n, r, t, max_ref_frames, is_train
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.interleave(
            lambda mix_chunks, ref_spec, ref_mask, clean_chunks, valid: tf.data.Dataset.from_tensor_slices(
                (mix_chunks, clean_chunks, valid)
            ).map(lambda m, c, v: (m, ref_spec, ref_mask, c, v)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if is_train:
            ds = ds.shuffle(2000)

        n_bins = self.audio_toolkit.n_fft // 2 + 1
        n_main = conv_stft_frames(
            self.chunk_size,
            self.audio_toolkit.frame_length,
            self.audio_toolkit.frame_step,
        )

        def to_example(mix, ref_spec, ref_mask, clean, valid):
            spec_noisy = self.audio_toolkit.stft(mix)
            spec_clean = self.audio_toolkit.stft(clean)
            spec_noisy.set_shape([n_main, n_bins])
            spec_clean.set_shape([n_main, n_bins])
            ref_spec.set_shape([max_ref_frames, n_bins])
            ref_mask.set_shape([max_ref_frames])
            noisy_main = self.audio_toolkit.complex_to_2ch(spec_noisy)
            noisy_ref = self.audio_toolkit.complex_to_2ch(ref_spec)
            clean_2ch = self._with_valid_channel(self.audio_toolkit.complex_to_2ch(spec_clean), valid)
            noisy_main.set_shape([n_main, n_bins, 2])
            noisy_ref.set_shape([max_ref_frames, n_bins, 2])
            clean_2ch.set_shape([n_main, n_bins, 3])
            return (
                {"noisy_main": noisy_main, "noisy_ref": noisy_ref, "ref_mask": ref_mask},
                clean_2ch,
            )

        ds = ds.map(to_example, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)



def load_scp(mix_path, ref_path, tgt_path):
    builder = LibriSpeechDatasetBuilder(AudioToolkit(), chunk_size=1, stride=1)
    return builder.load_scp(mix_path, ref_path, tgt_path)
