"""Waveform scp reader used by the SEF-PNet chunk loader.

Files are returned as stored. LibriMix already satisfies mix_clean = s1 + s2,
so nothing is peak-normalized here.
"""

import numpy as np
import soundfile as sf


def _parse_scripts(scp_path):
    scp = {}
    with open(scp_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            tokens = raw_line.strip().split()
            if len(tokens) != 2:
                raise RuntimeError(f"format error in {scp_path}:{line_number}: {raw_line!r}")
            key, value = tokens
            if key in scp:
                raise ValueError(f"duplicated key {key!r} in {scp_path}")
            scp[key] = value
    return scp


class WaveReader(object):
    def __init__(self, wav_scp, sample_rate=None):
        self.index_dict = _parse_scripts(wav_scp)
        self.index_keys = list(self.index_dict.keys())
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.index_dict)

    def _load(self, key):
        samples, rate = sf.read(self.index_dict[key], dtype="float32")
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1).astype(np.float32)
        if self.sample_rate is not None and rate != self.sample_rate:
            import librosa

            samples = librosa.resample(samples, orig_sr=rate, target_sr=self.sample_rate)
            samples = np.asarray(samples, dtype=np.float32)
        return samples

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.index_keys[index]
        if index not in self.index_dict:
            raise KeyError(f"missing utterance {index}")
        return self._load(index)
