from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from .audio import AudioToolkit


@dataclass
class LibriSpeechDatasetBuilder:
    audio_toolkit: AudioToolkit
    chunk_size: int
    stride: int

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

    def load_libri_speech_triplet_multiview(self, mix_path, ref_path, tgt_path, K=4, ref_len=16600):
        clean = self.audio_toolkit.preprocess_tf(tgt_path)
        noisy = self.audio_toolkit.preprocess_tf(mix_path)
        ref = self.audio_toolkit.preprocess_tf(ref_path)
        mix_chunks = self.audio_toolkit.split_into_chunks(noisy, self.chunk_size, self.stride)
        clean_chunks = self.audio_toolkit.split_into_chunks(clean, self.chunk_size, self.stride)
        ref_segments = self.audio_toolkit.sample_reference_segments(ref, K, ref_len)
        return mix_chunks, ref_segments, clean_chunks

    def convert_to_spectrogram_multiview(self, wav_corr, wav_ref_segments, wavclean):
        spectrogram_corr = tf.signal.stft(
            wav_corr,
            frame_length=self.audio_toolkit.frame_length,
            frame_step=self.audio_toolkit.frame_step,
            fft_length=self.audio_toolkit.n_fft,
        )
        spectrogram_clean = tf.signal.stft(
            wavclean,
            frame_length=self.audio_toolkit.frame_length,
            frame_step=self.audio_toolkit.frame_step,
            fft_length=self.audio_toolkit.n_fft,
        )
        spectrogram_refs = tf.map_fn(
            lambda x: tf.signal.stft(
                x,
                frame_length=self.audio_toolkit.frame_length,
                frame_step=self.audio_toolkit.frame_step,
                fft_length=self.audio_toolkit.n_fft,
            ),
            wav_ref_segments,
            fn_output_signature=tf.complex64,
        )
        return spectrogram_corr, spectrogram_refs, spectrogram_clean

    def configure_dataset(self, mixture_files, reference_files, target_files, is_train=True, K=4):
        ds = tf.data.Dataset.from_tensor_slices((mixture_files, reference_files, target_files))
        if is_train:
            ds = ds.shuffle(buffer_size=len(mixture_files))
        ds = ds.map(
            lambda n, r, t: self.load_libri_speech_triplet_multiview(n, r, t, K),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.interleave(
            lambda mix_chunks, ref_segments, clean_chunks: tf.data.Dataset.from_tensor_slices(
                (mix_chunks, clean_chunks)
            ).map(lambda m, c: (m, ref_segments, c)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if is_train:
            ds = ds.shuffle(2000)
        ds = ds.map(
            lambda mix, ref, clean: self.convert_to_spectrogram_multiview(mix, ref, clean),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            lambda spec_noisy, spec_refs, spec_clean: (
                {
                    "noisy_main": self.audio_toolkit.complex_to_2ch(spec_noisy),
                    "noisy_ref": self.audio_toolkit.complex_to_2ch(spec_refs),
                },
                self.audio_toolkit.complex_to_2ch(spec_clean),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.prefetch(tf.data.AUTOTUNE)



def load_scp(mix_path, ref_path, tgt_path):
    builder = LibriSpeechDatasetBuilder(AudioToolkit(), chunk_size=1, stride=1)
    return builder.load_scp(mix_path, ref_path, tgt_path)
