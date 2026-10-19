from __future__ import annotations

import librosa
import numpy as np
from pesq import pesq
from pystoi import stoi

SR = 8000


def si_sdr(est, ref, eps=1e-8):
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    scale = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    ref_scaled = scale * ref
    noise = est - ref_scaled
    return 10 * np.log10((np.sum(ref_scaled**2) + eps) / (np.sum(noise**2) + eps))


def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def pesq_score(clean, enhanced):
    try:
        clean, enhanced = normalize(clean), normalize(enhanced)
        mode = "nb" if SR == 8000 else "wb"
        return pesq(SR, clean, enhanced, mode)
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return np.nan


def stoi_score(clean, enhanced):
    try:
        clean, enhanced = normalize(clean), normalize(enhanced)
        return stoi(clean, enhanced, SR, extended=False)
    except Exception as e:
        print(f"Error calculating STOI: {e}")
        return np.nan


def load_resample_8k(path):
    audio, _ = librosa.load(path, sr=8000, mono=True)
    return audio


def sanitize(x):
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x
