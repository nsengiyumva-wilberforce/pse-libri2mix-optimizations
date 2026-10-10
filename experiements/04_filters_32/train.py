"""Experiment: 04 encoder starts at 32 filters.

This file is a full training script. It does not import another experiment.
"""
import argparse
import base64
import os
import zlib

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*unable to load libtensorflow_io_plugins.so.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*file system plugins are not loaded.*")
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)
import time
import logging
import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from pystoi.utils import thirdoct
import csv

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import onnxruntime as ort
import sys
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display, HTML
from pesq import pesq, NoUtterancesError
from pystoi import stoi
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_io as tfio
import keras
from keras import ops
from keras.models import Sequential
import tensorflow_io as tfio
from libri2mix import AudioToolkit, LibriSpeechDatasetBuilder, WaveformEnhancer
from libri2mix.audio import conv_stft_frames
from libri2mix.metrics import (
    load_resample_8k as _load_resample_8k,
    normalize as _normalize,
    pesq_score as _pesq_score,
    sanitize as _sanitize,
    stoi_score as _stoi_score,
)
from collections import defaultdict
import warnings
from glob import glob
import math
import time
import random
import soundfile as sf
from tqdm import tqdm
import soundfile as sf
from tensorflow.keras.layers import (
    BatchNormalization,
    AveragePooling2D,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Cropping2D,
    Dropout,
    Lambda,
    SpatialDropout2D,
    LayerNormalization,
    UpSampling2D,
    ZeroPadding2D,
    RNN,
    DepthwiseConv2D,
    Add,
    Multiply,
    Bidirectional,
    TimeDistributed,
    concatenate,
    RepeatVector,
    Input,
    Layer,
    GlobalAveragePooling1D,
    Reshape,
    MultiHeadAttention,
    GRU,
    Dense,
    GlobalAveragePooling2D,
    Resizing,
    Concatenate,
    multiply,
    add,
    Activation,
    Rescaling,
)

# XLA stays off until training. Evaluation uses one graph for every file length.


def _loss_choice(value):
    key = value.strip().lower().replace("_", "-")
    aliases = {
        "time-mse": "time-mse",
        "td-mse": "time-mse",
        "stsa-mse": "stsa-mse",
        "stsa": "stsa-mse",
        "stoi": "stoi",
        "estoi": "estoi",
        "si-sdr": "si-sdr",
        "sisdr": "si-sdr",
        "pmsqe": "pmsqe",
    }
    if key not in aliases:
        raise argparse.ArgumentTypeError(
            "unknown loss {!r}. Choose from: time-mse, stsa-mse, stoi, estoi, si-sdr, pmsqe".format(
                value
            )
        )
    return aliases[key]


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Libri2Mix enhancer with a loss from Kolbaek et al., "
            "IEEE/ACM TASLP 2020, or evaluate the saved baseline."
        )
    )
    parser.add_argument(
        "--l",
        type=_loss_choice,
        default=None,
        metavar="LOSS",
        help=(
            "Training loss: time-mse, stsa-mse, stoi, estoi, si-sdr, or pmsqe. "
            "Omit to evaluate the saved baseline."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Defaults to the SEF-PNet paper schedule of 120.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Clips per GPU. One GPU trains at this size. Three GPUs train at three times this size.",
    )
    return parser.parse_args()


args = _parse_args()
if args.epochs is not None and args.epochs < 1:
    raise SystemExit("--epochs must be at least 1.")
if args.batch < 1:
    raise SystemExit("--batch must be at least 1.")

# Clips on each GPU. Three cards train at three times this. One card trains at this size.
PER_GPU_BATCH = args.batch


def _checkpoint_filename(loss_tag):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "model_weights_04_filters_32_" + loss_tag + ".keras")


#--------------------------------
# HELPERS FOR DATA LOADING
def load_scp(mix_path, ref_path, tgt_path):
    return _DATASET_BUILDER.load_scp(mix_path, ref_path, tgt_path)


# -------------------------------------------------------
# OPTIONAL: helper if you later want folder-based loading
# -------------------------------------------------------
def load_from_folder(data_root, split):
    """
    Alternative loader for:
        data_root/train/{auxs1.scp, mix_clean.scp, ref.scp}
    """
    base = os.path.join(data_root, split)
    mix_path = os.path.join(base, "mix_clean.scp")
    ref_path = os.path.join(base, "auxs1.scp")  # enrollment
    tgt_path = os.path.join(base, "ref.scp")  # clean target
    return _DATASET_BUILDER.load_scp(mix_path, ref_path, tgt_path)


sr = 8000


TARGET_SR = 8000
total_length = 4.0
trim_length = 32000

# SEF-PNet frontend: 32 ms window, 8 ms hop, 256-point FFT, square-root Hann.
n_fft = 256
frame_length = 256
frame_step = 64

CHUNK_SIZE = 4 * TARGET_SR
STRIDE = CHUNK_SIZE // 2
REF_CHUNK_FRAMES = 128
# Longest auxiliary in this corpus is 17.61 s. 18 s at an 8 ms hop, rounded up
# to a whole number of reference chunks, keeps the enrollment length static.
_MAX_AUX_FRAMES = conv_stft_frames(18 * TARGET_SR, frame_length, frame_step)
MAX_REF_FRAMES = int(math.ceil(_MAX_AUX_FRAMES / REF_CHUNK_FRAMES) * REF_CHUNK_FRAMES)
N_BINS = n_fft // 2 + 1
N_FRAMES = conv_stft_frames(CHUNK_SIZE, frame_length, frame_step)
# SEF-PNet power-law compression on complex spectra (Li et al., beta = 0.5).
MAG_COMPRESS = 0.5

# SEF-PNet paper: 120 epochs. x0.98 every 2 epochs for the first 100, then x0.9 for the last 20.
EPOCHS = 120

_AUDIO_TOOLKIT = AudioToolkit(
    target_sr=TARGET_SR,
    frame_length=frame_length,
    frame_step=frame_step,
    n_fft=n_fft,
)
_DATASET_BUILDER = LibriSpeechDatasetBuilder(
    audio_toolkit=_AUDIO_TOOLKIT,
    chunk_size=CHUNK_SIZE,
    stride=STRIDE,
)
_WAVEFORM_ENHANCER = WaveformEnhancer(
    audio_toolkit=_AUDIO_TOOLKIT,
    chunk_size=CHUNK_SIZE,
    frame_length=frame_length,
    frame_step=frame_step,
    n_fft=n_fft,
)
TRAIN_MIX, TRAIN_REF, TRAIN_TGT = load_scp(
    "data/train/mix_clean.scp",
    "data/train/auxs1.scp",  # enrollment
    "data/train/ref.scp",  # clean target
)
DEV_MIX, DEV_REF, DEV_TGT = load_scp(
    "data/dev/mix_clean.scp",
    "data/dev/auxs1.scp",  # enrollment
    "data/dev/ref.scp",  # clean target
)
TEST_MIX, TEST_REF, TEST_TGT = load_scp(
    "data/test/mix_clean.scp",
    "data/test/auxs1.scp",  # enrollment
    "data/test/ref.scp",  # clean target
)
TARGET_SR = 8000


def load_audio_py(path):
    return _AUDIO_TOOLKIT.load_audio_py(path)


def load_audio_tf(path):
    return _AUDIO_TOOLKIT.load_audio_tf(path)


@tf.function
def preprocess_tf(filepath):
    return _AUDIO_TOOLKIT.preprocess_tf(filepath)


@tf.function
def split_into_chunks(wav, chunk_size, stride):
    return _AUDIO_TOOLKIT.split_into_chunks(wav, chunk_size, stride)


@tf.function
def tf_rms(x, eps=1e-8):
    return _AUDIO_TOOLKIT.tf_rms(x, eps)


@tf.function
def convert_to_spectrogram(wav_corr, wav_ref, wavclean):
    return _AUDIO_TOOLKIT.convert_to_spectrogram(wav_corr, wav_ref, wavclean)


@tf.function
def convert_to_spectrogram_multiview(wav_corr, wav_ref_segments, wavclean):
    # main mixture
    spectrogram_corr = tf.signal.stft(
        wav_corr, frame_length=frame_length, frame_step=frame_step, fft_length=n_fft
    )
    # clean target
    spectrogram_clean = tf.signal.stft(
        wavclean, frame_length=frame_length, frame_step=frame_step, fft_length=n_fft
    )
    # reference: vectorized over K
    # wav_ref_segments: (K, N)
    spectrogram_refs = tf.map_fn(
        lambda x: tf.signal.stft(
            x, frame_length=frame_length, frame_step=frame_step, fft_length=n_fft
        ),
        wav_ref_segments,
        fn_output_signature=tf.complex64,
    )
    # spectrogram_refs: (K, T, F)
    return spectrogram_corr, spectrogram_refs, spectrogram_clean


@tf.function
def spectrogram_abs(spectrogram_corr, spectrogram):
    spectrogram = tf.abs(spectrogram)
    spectrogram_corr = tf.abs(spectrogram_corr)
    return spectrogram_corr, spectrogram


@tf.function
def augment(spectrogram_corr, spectrogram):
    real_c, imag_c = spectrogram_corr[..., 0], spectrogram_corr[..., 1]
    real_t, imag_t = spectrogram[..., 0], spectrogram[..., 1]
    # apply the same augmentation to real and imaginary parts
    real_c = tfio.audio.freq_mask(real_c, 10)
    real_c = tfio.audio.freq_mask(real_c, 10)
    real_c = tfio.audio.time_mask(real_c, 20)
    real_c = tfio.audio.time_mask(real_c, 20)
    spectrogram_corr = tf.stack([real_c, imag_c], axis=-1)
    spectrogram_clean = tf.stack([real_t, imag_t], axis=-1)

    return spectrogram_corr, spectrogram_clean


@tf.function
def expand_dims(spectrogram_corr, spectrogram):
    spectrogram_corr = tf.expand_dims(spectrogram_corr, axis=2)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram_corr, spectrogram


def complex_to_2ch(spec):
    return _AUDIO_TOOLKIT.complex_to_2ch(spec)


@tf.function
def sample_reference_segments(wav, K, segment_len):
    return _AUDIO_TOOLKIT.sample_reference_segments(wav, K, segment_len)


def load_libri_speech_triplet_multiview(
    mix_path, ref_path, tgt_path, K=4, ref_len=8000 * 2
):
    return _DATASET_BUILDER.load_libri_speech_triplet_multiview(
        mix_path, ref_path, tgt_path, K=K, ref_len=ref_len
    )


def configure_libri_speech_dataset(
    mixture_files, reference_files, target_files, is_train=True, K=4, full_utterance=False
):
    if full_utterance:
        return _DATASET_BUILDER.configure_dataset_full_utterance(
            mixture_files,
            reference_files,
            target_files,
            is_train=is_train,
            max_ref_frames=MAX_REF_FRAMES,
        )
    return _DATASET_BUILDER.configure_dataset(
        mixture_files, reference_files, target_files, is_train=is_train, K=K
    )


print("Reference mode: full auxiliary utterance")
print("Injection: early fusion, guided STFT concatenated at the input")
print("Experiment: 04 encoder starts at 32 filters")
print(
    f"Frontend: {frame_length}-point sqrt-Hann, hop {frame_step}, "
    f"chunk {CHUNK_SIZE}, stride {STRIDE}, no peak norm"
)
print(f"Utterances: train {len(TRAIN_MIX)} dev {len(DEV_MIX)} test {len(TEST_MIX)}")
print("Loader: SEF-PNet chunk dataloader (one optimizer step per chunk batch)")


# ============ sLSTM implementation ============
class sLSTMCell(Layer):
    def __init__(self, units, forget_gate_type="sigmoid", **kwargs):
        """
        sLSTM cell with proper state handling.
        """
        super(sLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.forget_gate_type = forget_gate_type

        # Input projections
        self.W_z = Dense(units, use_bias=True)  # Cell input
        self.W_i = Dense(units, use_bias=True)  # Input gate
        self.W_f = Dense(units, use_bias=True)  # Forget gate
        self.W_o = Dense(units, use_bias=True)  # Output gate

        # Recurrent projections (for h_{t-1})
        self.U_z = Dense(units, use_bias=False)
        self.U_i = Dense(units, use_bias=False)
        self.U_f = Dense(units, use_bias=False)
        self.U_o = Dense(units, use_bias=False)

    @property
    def state_size(self):
        # Return a tuple of state sizes: (h, c, n, m)
        return [self.units, self.units, self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def _stabilize_gates(self, i_tilde, f_tilde, m_prev):
        """Stabilize exponential gates."""
        # m_t = max(log(i_t) + m_{t-1}, log(i_t))
        # where log(i_t) = i_tilde
        m_t = tf.maximum(i_tilde + m_prev, i_tilde)

        # i_t' = exp(i_tilde - m_t)
        i_t_stabilized = tf.math.exp(i_tilde - m_t)

        # f_t' handling
        if self.forget_gate_type == "exponential":
            # f_t' = exp(f_tilde + m_{t-1} - m_t)
            f_t_stabilized = tf.math.exp(f_tilde + m_prev - m_t)
        else:
            f_t_stabilized = tf.math.sigmoid(f_tilde)

        return i_t_stabilized, f_t_stabilized, m_t

    def build(self, input_shape):
        """
        Build weights once input shape is known.
        input_shape: (batch_size, input_dim)
        """
        input_dim = input_shape[-1]

        # Input projections
        self.W_z.build((None, input_dim))
        self.W_i.build((None, input_dim))
        self.W_f.build((None, input_dim))
        self.W_o.build((None, input_dim))

        # Recurrent projections (from h_{t-1})
        self.U_z.build((None, self.units))
        self.U_i.build((None, self.units))
        self.U_f.build((None, self.units))
        self.U_o.build((None, self.units))

        self.built = True

    def call(self, inputs, states):
        h_prev, c_prev, n_prev, m_prev = states

        # Compute all gate logits
        z_tilde = self.W_z(inputs) + self.U_z(h_prev)
        i_tilde = self.W_i(inputs) + self.U_i(h_prev)
        f_tilde = self.W_f(inputs) + self.U_f(h_prev)
        o_tilde = self.W_o(inputs) + self.U_o(h_prev)

        # Stabilize gates
        i_t, f_t, m_t = self._stabilize_gates(i_tilde, f_tilde, m_prev)

        # Output gate (sigmoid)
        o_t = tf.math.sigmoid(o_tilde)

        # Cell input (tanh)
        z_t = tf.math.tanh(z_tilde)

        # Update cell state: c_t = f_t * c_{t-1} + i_t * z_t
        c_t = f_t * c_prev + i_t * z_t

        # Update normalizer state: n_t = f_t * n_{t-1} + i_t
        n_t = f_t * n_prev + i_t

        # Hidden state: h̃_t = c_t / (n_t + epsilon), h_t = o_t * h̃_t
        epsilon = 1e-8
        h_tilde = c_t / (n_t + epsilon)
        h_t = o_t * h_tilde

        return h_t, [h_t, c_t, n_t, m_t]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "forget_gate_type": self.forget_gate_type,
            }
        )
        return config

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        return [
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32),  # h
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32),  # c
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32),  # n
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32),  # m
        ]


# ============ mLSTM CELL implementation ============
class mLSTMCell(Layer):
    def __init__(self, units, forget_gate_type="sigmoid", **kwargs):
        super(mLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.forget_gate_type = forget_gate_type

        # Layer normalization for keys/values
        self.ln_kv = LayerNormalization(epsilon=1e-6)

        # Key scaling: 1/√d
        self.key_scale = 1.0 / tf.math.sqrt(tf.cast(units, tf.float32))

        # Projections
        self.W_q = Dense(units, use_bias=True)  # Query
        self.W_k = Dense(units, use_bias=True)  # Key
        self.W_v = Dense(units, use_bias=True)  # Value
        self.W_i = Dense(units, use_bias=True)  # Input gate
        self.W_f = Dense(units, use_bias=True)  # Forget gate
        self.W_o = Dense(units, activation="sigmoid", use_bias=True)  # Output gate

    @property
    def state_size(self):
        # Return a tuple of state sizes: (h, C_flat, n, m)
        return [self.units, self.units * self.units, self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def _stabilize_gates(self, i_tilde, f_tilde, m_prev):
        """Same stabilization as sLSTM."""
        m_t = tf.maximum(i_tilde + m_prev, i_tilde)
        i_t_stabilized = tf.math.exp(i_tilde - m_t)

        if self.forget_gate_type == "exponential":
            f_t_stabilized = tf.math.exp(f_tilde + m_prev - m_t)
        else:
            f_t_stabilized = tf.math.sigmoid(f_tilde)

        return i_t_stabilized, f_t_stabilized, m_t

    def build(self, input_shape):
        """
        Build weights once input shape is known.
        input_shape: (batch_size, input_dim)
        """
        input_dim = input_shape[-1]

        # LayerNorm for keys/values
        self.ln_kv.build((None, input_dim))

        # Projections
        self.W_q.build((None, input_dim))
        self.W_k.build((None, input_dim))
        self.W_v.build((None, input_dim))
        self.W_i.build((None, input_dim))
        self.W_f.build((None, input_dim))
        self.W_o.build((None, input_dim))

        self.built = True

    def call(self, inputs, states):
        h_prev, C_flat_prev, n_prev, m_prev = states
        batch_size = tf.shape(inputs)[0]

        # Reshape C matrix
        C_prev = tf.reshape(C_flat_prev, [batch_size, self.units, self.units])

        # Layer norm for keys/values
        inputs_norm = self.ln_kv(inputs)

        # Queries, keys, values
        q_t = self.W_q(inputs)  # Query
        k_t = self.W_k(inputs_norm) * self.key_scale  # Scaled key
        v_t = self.W_v(inputs_norm)  # Value

        # Gate logits
        i_tilde = self.W_i(inputs)
        f_tilde = self.W_f(inputs)

        # Stabilize gates
        i_t, f_t, m_t = self._stabilize_gates(i_tilde, f_tilde, m_prev)

        # Output gate
        o_t = self.W_o(inputs)

        # Update cell state: C_t = f_t C_{t-1} + i_t v_t k_t^T
        f_t_exp = tf.expand_dims(f_t, axis=-1)  # [B, d, 1]
        v_t_exp = tf.expand_dims(v_t, axis=-1)  # [B, d, 1]
        k_t_exp = tf.expand_dims(k_t, axis=1)  # [B, 1, d]

        new_memory = tf.expand_dims(i_t, axis=-1) * v_t_exp * k_t_exp
        C_t = f_t_exp * C_prev + new_memory

        # Update normalizer state: n_t = f_t n_{t-1} + i_t k_t
        n_t = f_t * n_prev + i_t * k_t

        # Compute hidden state: h̃_t = C_t q_t / max(|n_t^T q_t|, 1)
        q_t_exp = tf.expand_dims(q_t, axis=-1)  # [B, d, 1]
        numerator = tf.squeeze(C_t @ q_t_exp, axis=-1)  # [B, d]

        dot_product = tf.reduce_sum(n_t * q_t, axis=-1, keepdims=True)  # [B, 1]
        denominator = tf.maximum(tf.abs(dot_product), 1.0)  # [B, 1]

        h_tilde = numerator / denominator
        h_t = o_t * h_tilde

        # Flatten C for next step
        C_flat_t = tf.reshape(C_t, [batch_size, self.units * self.units])

        return h_t, [h_t, C_flat_t, n_t, m_t]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "forget_gate_type": self.forget_gate_type,
            }
        )
        return config

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        return [
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32),  # h
            tf.zeros(
                (batch_size, self.units * self.units), dtype=dtype or tf.float32
            ),  # C
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32),  # n
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32),  # m
        ]

@tf.keras.utils.register_keras_serializable()
class LearnableScale(layers.Layer):
    def __init__(self, initial_value=2.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value
        self.scale = self.add_weight(
            name="scale_factor",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_value),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

    def call(self, x):
        # We add 1e-6 to ensure the multiplier is never exactly 0
        return (self.scale + 1e-6) * ops.tanh(x)

    def get_config(self):
        config = super().get_config()
        config.update({"initial_value": self.initial_value})
        return config




def add_xlstm_block(x, hidden_dim=256, num_layers=2, block_types=None, prefix="xlstm"):
    """
    Add xLSTM blocks with unique names using the prefix argument.
    """
    if block_types is None:
        block_types = ["sLSTM", "mLSTM"] * ((num_layers + 1) // 2)
        block_types = block_types[:num_layers]
    
    for i, block_type in enumerate(block_types):
        residual = x
        
        if block_type == 'sLSTM':
            cell = sLSTMCell(hidden_dim, forget_gate_type='sigmoid')
        elif block_type == 'mLSTM':
            cell = mLSTMCell(hidden_dim, forget_gate_type='sigmoid')
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
        # Unique name using the prefix and index
        x = RNN(cell, return_sequences=True, 
                name=f'{prefix}_{block_type}_{i}')(x)
        
        if residual.shape[-1] == x.shape[-1]:
            # Names for operations like Add and LayerNorm are usually auto-generated,
            # but you can name them too if you want total safety:
            x = Add(name=f'{prefix}_add_{i}')([residual, x])
            x = LayerNormalization(epsilon=1e-6, name=f'{prefix}_ln_{i}')(x)
        
        if i < num_layers - 1:
            x = Dropout(0.2, name=f'{prefix}_do_{i}')(x)
    
    return x


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


@tf.keras.utils.register_keras_serializable()
class PowerLawComplexSpec(Layer):
    """Compress or expand complex spectrogram magnitude while keeping phase.

    factor=0.5 matches SEF-PNet's FeaCompression. factor=2 undoes it.
    """

    def __init__(self, factor=0.5, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.factor = float(factor)
        self.eps = float(eps)

    def call(self, spec):
        real = spec[..., 0]
        imag = spec[..., 1]
        magnitude = tf.sqrt(tf.square(real) + tf.square(imag) + self.eps)
        scale = tf.pow(magnitude, self.factor) / magnitude
        return tf.stack([real * scale, imag * scale], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor, "eps": self.eps})
        return config


def tf_alternating_block(x, filters, activation="relu", use_bn=True, name_prefix="tfb"):
    orginal_x = x
    # ---- Frequency branch (1 x 3) for the first branch, we get time convolutions over frequency
    f_branch = Conv2D(filters, (3, 3), padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name_prefix}_fconv")(x)
    if use_bn:
        f_branch  = BatchNormalization(name=f"{name_prefix}_fbn")(f_branch)
    f_branch = Activation(activation)(f_branch )

    # ---- Time branch (3 x 1) ----
    t_branch = Conv2D(filters, (5,5), padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name_prefix}_tconv")(x)
    if use_bn:
        t_branch  = BatchNormalization(name=f"{name_prefix}_tbn")(t_branch)
    t_branch = Activation(activation)(t_branch )

    # ---- Merge ----
    x = Concatenate(name=f"{name_prefix}_concat")([f_branch, t_branch])

    # sencond branch, we get frequency convolutions over time
    t_branch_2 = Conv2D(filters, (5, 5), padding="same",
                        kernel_initializer="he_normal",
                        name=f"{name_prefix}_tconv2")(orginal_x)
    if use_bn:
        t_branch_2  = BatchNormalization(name=f"{name_prefix}_tbn2")(t_branch_2)
    t_branch_2  = Activation(activation)(t_branch_2)

    f_branch_2 = Conv2D(filters, (3, 3), padding="same",
                        kernel_initializer="he_normal",
                        name=f"{name_prefix}_fconv2")(orginal_x)
    if use_bn:
        f_branch_2  = BatchNormalization(name=f"{name_prefix}_fbn2")(f_branch_2)
    f_branch_2 = Activation(activation)(f_branch_2)

    # Merge again
    x = Concatenate(name=f"{name_prefix}_concat2")([t_branch_2, f_branch_2])
    # merge the two branches  for x and original_x
    x = Concatenate(name=f"{name_prefix}_final_concat")([x, orginal_x])

    # ---- Separable TF mixing ----
    x = DepthwiseConv2D((3,3), padding="same",
                        depthwise_initializer="he_normal",
                        name=f"{name_prefix}_dw")(x)

    x = Conv2D(filters, 1, padding="same",
               kernel_initializer="he_normal",
               name=f"{name_prefix}_pw")(x)

    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_pw_bn")(x)

    x = Activation(activation)(x)

    # ---- Joint TF modeling ----
    x = Conv2D(filters, (3, 3), padding="same",
               kernel_initializer="he_normal",
               name=f"{name_prefix}_joint")(x)
    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_jbn")(x)
    x = Activation(activation)(x)

    return x
def eca_kernel_size(channels, gamma=2, b=1):
    """Odd 1D kernel from ECA-Net: k = |log2(C)/gamma + b/gamma|, forced odd."""
    t = int(abs((math.log2(max(int(channels), 1)) + b) / gamma))
    return t if t % 2 else t + 1


def eca_block(x, name="eca"):
    """Efficient channel attention. A few 1D-conv weights replace LCA's two squeeze-excitation branches."""
    channels = int(x.shape[-1])
    y = GlobalAveragePooling2D(name=f"{name}_gap")(x)
    y = Reshape((channels, 1), name=f"{name}_seq")(y)
    y = layers.Conv1D(
        1,
        eca_kernel_size(channels),
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(y)
    y = Activation("sigmoid", name=f"{name}_gate")(y)
    y = Reshape((1, 1, channels), name=f"{name}_map")(y)
    return Multiply(name=name)([x, y])


@tf.keras.utils.register_keras_serializable()
class ASFFFusion(Layer):
    """Two-input adaptively spatial feature fusion.

    Both maps already share height, width, and channels. Each is projected with a
    1x1 convolution, a second 1x1 scores the pair, and a per-bin softmax mixes
    them. The score kernel starts at zero. With prefer_left False the bias is
    zero too, so the mix starts equal. With prefer_left True the bias starts at
    [2, -2], about 98% on the main map, so a newly injected cue does not replace
    the features the mask is reading.
    """

    def __init__(self, compress=8, prefer_left=False, **kwargs):
        super().__init__(**kwargs)
        self.compress = int(compress)
        self.prefer_left = bool(prefer_left)
        # Created here so Keras tracks them. build() is too late for the functional model.
        self.compress_left = Conv2D(
            self.compress, 1, padding="same", use_bias=False, kernel_initializer="he_normal"
        )
        self.compress_right = Conv2D(
            self.compress, 1, padding="same", use_bias=False, kernel_initializer="he_normal"
        )
        bias = (
            keras.initializers.Constant([2.0, -2.0]) if self.prefer_left else "zeros"
        )
        self.level = Conv2D(
            2, 1, padding="same", kernel_initializer="zeros", bias_initializer=bias
        )

    def build(self, input_shape):
        left_shape = input_shape[0]
        self.compress_left.build(left_shape)
        self.compress_right.build(left_shape)
        score_shape = tuple(left_shape[:-1]) + (2 * self.compress,)
        self.level.build(score_shape)
        super().build(input_shape)

    def call(self, inputs):
        left, right = inputs
        scores = self.level(
            tf.concat([self.compress_left(left), self.compress_right(right)], axis=-1)
        )
        weights = tf.nn.softmax(scores, axis=-1)
        left_w, right_w = tf.split(weights, num_or_size_splits=2, axis=-1)
        return left_w * left + right_w * right

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"compress": self.compress, "prefer_left": self.prefer_left})
        return config

class STFTFrameSimilarity(Layer):
    """SEF-PNet similarity: guidance = E @ softmax(E^T @ Y), per real and imag.

    Frames with mask 0 are left out of the softmax and the enrollment mean.
    Training marks the whole batch-padded enrollment as valid, so the zeros
    that lengthen a short enrollment stay in. The second half of the returned
    channels is that mean, tiled over mixture time.
    """

    def call(self, inputs):
        mix, enroll, mask = inputs
        mix_f = tf.transpose(mix, [0, 2, 3, 1])
        enr_f = tf.transpose(enroll, [0, 2, 3, 1])
        guides = []
        valid = tf.cast(mask, mix.dtype)
        for channel in range(2):
            enrollment = enr_f[:, :, channel, :]
            mixture = mix_f[:, :, channel, :]
            logits = tf.matmul(enrollment, mixture, transpose_a=True)
            logits = logits + (valid[:, :, None] - 1.0) * 1e9
            weights = tf.nn.softmax(logits, axis=1)
            guides.append(tf.matmul(enrollment, weights))
        guided = tf.transpose(tf.stack(guides, axis=-1), [0, 2, 1, 3])

        valid_frames = valid[:, :, None, None]
        denom = tf.reduce_sum(valid_frames, axis=1, keepdims=True) + 1e-8
        mean = tf.reduce_sum(enroll * valid_frames, axis=1, keepdims=True) / denom
        mixture_time = mix.shape[1]
        if mixture_time is None:
            mixture_time = tf.shape(mix)[1]
        mean = tf.tile(mean, [1, mixture_time, 1, 1])

        out = tf.concat([guided, mean], axis=-1)
        freq = mix.shape[2]
        if freq is not None:
            out.set_shape([None, mix.shape[1], freq, 4])
        return out


class SliceChannels(Layer):
    """Keep the frequency axis static when the time axis is the whole utterance."""

    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.end = int(end)

    def call(self, x):
        sliced = x[..., self.start : self.end]
        sliced.set_shape([None, x.shape[1], x.shape[2], self.end - self.start])
        return sliced

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.end - self.start)

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


class PadTimeToMultiple(Layer):
    """Pad the time axis so four stride-2 pools land on whole bins."""

    def __init__(self, multiple=16, **kwargs):
        super().__init__(**kwargs)
        self.multiple = int(multiple)

    def call(self, x):
        time = tf.shape(x)[1]
        pad = (self.multiple - time % self.multiple) % self.multiple
        return tf.pad(x, [[0, 0], [0, pad], [0, 0], [0, 0]])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"multiple": self.multiple})
        return config


class CropToReference(Layer):
    """Drop the pool padding and return the original time and frequency size."""

    def call(self, inputs):
        features, reference = inputs
        time = tf.shape(reference)[1]
        freq = tf.shape(reference)[2]
        return features[:, :time, :freq, :]

    def compute_output_shape(self, input_shape):
        features, reference = input_shape
        return (features[0], reference[1], reference[2], features[3])


def custom_unet(
    input_shape,
    activation="relu",
    use_batch_norm=True,
    filters=16,
    num_layers=4,
    max_ref_frames=1280,
):
    """Early fusion: concatenate the guided enrollment STFT onto the mixture."""
    upsample = upsample_conv
    main_input = Input(input_shape, name="noisy_main")
    ref_input = Input((max_ref_frames, input_shape[1], input_shape[2]), name="noisy_ref")
    ref_mask_input = Input((max_ref_frames,), name="ref_mask")
    x = PowerLawComplexSpec(factor=MAG_COMPRESS, name="compress_main")(main_input)
    ref_x = PowerLawComplexSpec(factor=MAG_COMPRESS, name="compress_ref")(ref_input)
    main_input_copy = x
    sim_and_mean = STFTFrameSimilarity(name="stft_sim")([x, ref_x, ref_mask_input])
    variable_time = x.shape[1] is None
    if variable_time:
        guided = SliceChannels(0, 2, name="stft_guided")(sim_and_mean)
        enroll_mean = SliceChannels(2, 4, name="stft_mean")(sim_and_mean)
    else:
        time_bins, freq_bins = int(x.shape[1]), int(x.shape[2])
        guided = Reshape((time_bins, freq_bins, 2), name="stft_guided")(sim_and_mean[..., 0:2])
        enroll_mean = Reshape((time_bins, freq_bins, 2), name="stft_mean")(sim_and_mean[..., 2:4])
    guided = ASFFFusion(name="stft_asff")([guided, enroll_mean])
    x = Concatenate(axis=-1, name="stft_fuse")([x, guided])
    mixture_map = x

    # Four stride-2 pools need both axes divisible by 16. The 8 ms hop does not
    # land on that grid (497 x 129), so pad here and crop before the mask.
    unet_multiple = 2 ** num_layers
    spec_freq = int(x.shape[2])
    pad_freq = (unet_multiple - spec_freq % unet_multiple) % unet_multiple
    if variable_time:
        if pad_freq:
            x = ZeroPadding2D(padding=((0, 0), (0, pad_freq)), name="unet_pad_freq")(x)
        x = PadTimeToMultiple(unet_multiple, name="unet_pad_time")(x)
        pad_time = None
    else:
        spec_time = int(x.shape[1])
        pad_time = (unet_multiple - spec_time % unet_multiple) % unet_multiple
        if pad_time or pad_freq:
            x = ZeroPadding2D(padding=((0, pad_time), (0, pad_freq)), name="unet_pad")(x)

    down_layers = []
    for l in range(num_layers):
        x = tf_alternating_block(x, filters, activation, use_bn=True, name_prefix=f"tfb_{l}")
        x = eca_block(x, name=f"enc_eca_{l}")
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        filters = int(filters * 1.5)

    T_small, F_small, C_small = x.shape[1], x.shape[2], x.shape[3]
    flat = int(F_small) * int(C_small)
    if variable_time:
        x_seq = Reshape((-1, flat))(x)
    else:
        x_seq = Reshape((T_small, flat))(x)
    x_seq = add_xlstm_block(x_seq, hidden_dim=128, num_layers=1, prefix="main_bottleneck")
    x_expanded = TimeDistributed(Dense(flat))(x_seq)
    if variable_time:
        x_reshaped = Reshape((-1, int(F_small), int(C_small)))(x_expanded)
    else:
        x_reshaped = Reshape((T_small, int(F_small), int(C_small)))(x_expanded)
    x = Conv2D(C_small, (1, 1), padding="same", activation=activation, name="bn_reconstruct")(x_reshaped)
    for conv in reversed(down_layers):
        filters = conv.shape[-1]
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = concatenate([x, conv])
        x = tf_alternating_block(
            x, filters, activation, use_bn=use_batch_norm, name_prefix=f"up_conv_{filters}"
        )
    if variable_time:
        x = CropToReference(name="unet_crop")([x, mixture_map])
    elif pad_time or pad_freq:
        x = Cropping2D(cropping=((0, pad_time), (0, pad_freq)), name="unet_crop")(x)
    input_r = main_input_copy[..., 0:1]
    input_i = main_input_copy[..., 1:2]
    # A zero or near-zero kernel makes the waveform the mixture and scales every
    # gradient behind the mask by that kernel. he_normal keeps the path open.
    mask_r = Conv2D(1, (1, 1), activation=None, kernel_initializer="he_normal", name="mask_real")(x)
    mask_i = Conv2D(1, (1, 1), activation=None, kernel_initializer="he_normal", name="mask_imag")(x)
    mask_r = LearnableScale(initial_value=2.0, name="scale_r")(mask_r)
    mask_i = LearnableScale(initial_value=2.0, name="scale_i")(mask_i)
    out_r = layers.Subtract()([
        layers.Multiply()([mask_r, input_r]),
        layers.Multiply()([mask_i, input_i])
    ])
    out_i = layers.Add()([
        layers.Multiply()([mask_r, input_i]),
        layers.Multiply()([mask_i, input_r])
    ])
    out_r = layers.Add()([input_r, out_r])
    out_i = layers.Add()([input_i, out_i])

    outputs = Concatenate(axis=-1)([out_r, out_i])
    outputs = PowerLawComplexSpec(factor=1.0 / MAG_COMPRESS, name="decompress_out")(outputs)
    return Model(inputs=[main_input, ref_input, ref_mask_input], outputs=[outputs])


model_filename = _checkpoint_filename(args.l.replace("-", "_") if args.l is not None else "si_sdr")


def _build_model(variable_time=False):
    frames = None if variable_time else N_FRAMES
    return custom_unet(
        input_shape=(frames, N_BINS, 2),
        use_batch_norm=True,
        filters=32,
        num_layers=4,
        max_ref_frames=MAX_REF_FRAMES,
    )


# Kolbaek, Tan, Jensen, Jensen, "On Loss Functions for Supervised Monaural
# Time-Domain Speech Enhancement", IEEE/ACM TASLP, 2020.
# The network predicts a complex spectrogram. These losses invert it with the
# training STFT and apply the paper's waveform criteria.
_STOI_FS = 10000
_STOI_FRAME = 256
_STOI_NFFT = 512
_STOI_HOP = 128
_STOI_N = 30
_STOI_BETA = -15.0
_STSA_FFT = 256
_STSA_HOP = 128
_STOI_OBM = tf.constant(thirdoct(_STOI_FS, _STOI_NFFT, 15, 150)[0], dtype=tf.float32)


def _matlab_hann_window(window_length, dtype=tf.float32):
    # np.hanning(window_length + 2)[1:-1], the analysis window used by STOI.
    n = tf.range(1, window_length + 1, dtype=dtype)
    return tf.cast(0.5, dtype) - tf.cast(0.5, dtype) * tf.cos(
        tf.cast(2.0 * math.pi, dtype) * n / tf.cast(window_length + 1, dtype)
    )


def _spec_to_waveform(spec_2ch):
    spec_2ch = spec_2ch[..., 0:2]
    spectrum = tf.complex(spec_2ch[..., 0], spec_2ch[..., 1])
    return _AUDIO_TOOLKIT.istft(spectrum)


def _resample_8k_to_10k(wav):
    length = tf.shape(wav)[1]
    new_length = tf.cast(tf.round(tf.cast(length, tf.float32) * (_STOI_FS / TARGET_SR)), tf.int32)
    image = wav[:, :, None, None]
    resized = tf.image.resize(image, [new_length, 1], method="bilinear")
    return resized[:, :, 0, 0]


def _third_octave_envelopes(wav):
    wav = _resample_8k_to_10k(wav)
    spectrum = tf.signal.stft(
        wav,
        frame_length=_STOI_FRAME,
        frame_step=_STOI_HOP,
        fft_length=_STOI_NFFT,
        window_fn=_matlab_hann_window,
    )
    band_energy = tf.matmul(tf.square(tf.abs(spectrum)), _STOI_OBM, transpose_b=True)
    envelopes = tf.sqrt(band_energy + 1e-8)
    return tf.transpose(envelopes, [0, 2, 1])


def _envelope_segments(envelopes):
    return tf.signal.frame(envelopes, _STOI_N, 1, axis=-1)


def time_domain_mse(y_true, y_pred):
    """Time-domain MSE, Kolbaek et al. Eq. (3)."""
    target = _spec_to_waveform(y_true)
    estimate = _spec_to_waveform(y_pred)
    return tf.reduce_mean(tf.square(estimate - target))


def stsa_mse(y_true, y_pred):
    """Short-time spectral amplitude MSE, Kolbaek et al. Eq. (4). K=256, hop=128."""
    target = _spec_to_waveform(y_true)
    estimate = _spec_to_waveform(y_pred)

    def amplitude(wav):
        spectrum = tf.signal.stft(
            wav,
            frame_length=_STSA_FFT,
            frame_step=_STSA_HOP,
            fft_length=_STSA_FFT,
            window_fn=_matlab_hann_window,
        )
        return tf.abs(spectrum)

    return tf.reduce_mean(tf.square(amplitude(estimate) - amplitude(target)))


def _intelligibility_stoi(y_true, y_pred):
    target = _envelope_segments(_third_octave_envelopes(_spec_to_waveform(y_true)))
    estimate = _envelope_segments(_third_octave_envelopes(_spec_to_waveform(y_pred)))
    target_norm = tf.sqrt(tf.reduce_sum(tf.square(target), axis=-1, keepdims=True) + 1e-8)
    estimate_norm = tf.sqrt(tf.reduce_sum(tf.square(estimate), axis=-1, keepdims=True) + 1e-8)
    estimate = estimate * (target_norm / estimate_norm)
    clip = 1.0 + 10 ** (-_STOI_BETA / 20.0)
    estimate = tf.minimum(estimate, target * clip)
    target = target - tf.reduce_mean(target, axis=-1, keepdims=True)
    estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
    target = target / tf.sqrt(tf.reduce_sum(tf.square(target), axis=-1, keepdims=True) + 1e-8)
    estimate = estimate / tf.sqrt(tf.reduce_sum(tf.square(estimate), axis=-1, keepdims=True) + 1e-8)
    return tf.reduce_mean(tf.reduce_sum(target * estimate, axis=-1))


def _row_col_normalize(segments):
    segments = segments - tf.reduce_mean(segments, axis=-1, keepdims=True)
    segments = segments / tf.sqrt(tf.reduce_sum(tf.square(segments), axis=-1, keepdims=True) + 1e-8)
    segments = segments - tf.reduce_mean(segments, axis=-2, keepdims=True)
    segments = segments / tf.sqrt(tf.reduce_sum(tf.square(segments), axis=-2, keepdims=True) + 1e-8)
    return segments


def _intelligibility_estoi(y_true, y_pred):
    target = _envelope_segments(_third_octave_envelopes(_spec_to_waveform(y_true)))
    estimate = _envelope_segments(_third_octave_envelopes(_spec_to_waveform(y_pred)))
    # (batch, bands, segments, frames) -> (batch, segments, bands, frames)
    target = tf.transpose(target, [0, 2, 1, 3])
    estimate = tf.transpose(estimate, [0, 2, 1, 3])
    target = _row_col_normalize(target)
    estimate = _row_col_normalize(estimate)
    column_dots = tf.reduce_sum(target * estimate, axis=[2, 3]) / tf.cast(_STOI_N, tf.float32)
    return tf.reduce_mean(column_dots)


def stoi_loss(y_true, y_pred):
    """Negative STOI, Kolbaek et al. Eq. (10). Voice-activity detection is omitted."""
    return -_intelligibility_stoi(y_true, y_pred)


def estoi_loss(y_true, y_pred):
    """Negative ESTOI, Kolbaek et al. Eq. (16)."""
    return -_intelligibility_estoi(y_true, y_pred)


def si_sdr_loss(y_true, y_pred, eps=1e-8):
    """Negative SI-SDR, Kolbaek et al. Eq. (20). Both signals are zero-mean.

    When the label carries a third channel, that channel is the number of real
    samples in the chunk. Padded samples are zeroed on both waveforms first,
    matching SEF-PNet's valid_len mask.
    """
    target = _spec_to_waveform(y_true)
    estimate = _spec_to_waveform(y_pred)
    if y_true.shape[-1] == 3:
        valid_len = tf.cast(tf.round(y_true[:, 0, 0, 2]), tf.int32)
        positions = tf.range(tf.shape(target)[-1])
        sample_mask = tf.cast(positions[None, :] < valid_len[:, None], target.dtype)
        target = target * sample_mask
        estimate = estimate * sample_mask
    target = target - tf.reduce_mean(target, axis=-1, keepdims=True)
    estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
    dot = tf.reduce_sum(estimate * target, axis=-1, keepdims=True)
    target_energy = tf.reduce_sum(tf.square(target), axis=-1, keepdims=True)
    projection = dot * target / (target_energy + eps)
    noise = estimate - projection
    ratio = (tf.reduce_sum(tf.square(projection), axis=-1) + eps) / (
        tf.reduce_sum(tf.square(noise), axis=-1) + eps
    )
    si_sdr = 10.0 * tf.math.log(ratio) / tf.math.log(10.0)
    return -tf.reduce_mean(si_sdr)


# PMSQE at 8 kHz (Martin-Donas et al., IEEE SPL 2018), the setting used by Kolbaek et al.
_PMSQE_FFT = 256
_PMSQE_HOP = 128
_PMSQE_ALPHA = 0.1
_PMSQE_BETA = 0.309 * _PMSQE_ALPHA
_PMSQE_SL = 1.866055e-1
_PMSQE_ABS_THRESH = tf.constant(
    [
        51286152.0,
        2454709.500,
        70794.593750,
        4897.788574,
        1174.897705,
        389.045166,
        104.712860,
        45.708820,
        17.782795,
        9.772372,
        4.897789,
        3.090296,
        1.905461,
        1.258925,
        0.977237,
        0.724436,
        0.562341,
        0.457088,
        0.389045,
        0.331131,
        0.295121,
        0.269153,
        0.257040,
        0.251189,
        0.251189,
        0.251189,
        0.251189,
        0.263027,
        0.288403,
        0.309030,
        0.338844,
        0.371535,
        0.398107,
        0.436516,
        0.467735,
        0.489779,
        0.501187,
        0.501187,
        0.512861,
        0.524807,
        0.524807,
        0.524807,
    ],
    dtype=tf.float32,
)
_PMSQE_ZWICKER = tf.constant(
    [
        0.25520097857560436,
        0.25520097857560436,
        0.25520097857560436,
        0.25520097857560436,
        0.25168783742879913,
        0.24806665731869609,
        0.244767379124259,
        0.24173800119368227,
        0.23893798876066405,
        0.23633516221479894,
        0.23390360348392067,
        0.23162209128929445,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
    ],
    dtype=tf.float32,
)
_PMSQE_WIDTH = tf.constant(
    [
        0.157344,
        0.317994,
        0.322441,
        0.326934,
        0.331474,
        0.336061,
        0.340697,
        0.345381,
        0.350114,
        0.354897,
        0.359729,
        0.364611,
        0.369544,
        0.374529,
        0.379565,
        0.384653,
        0.389794,
        0.394989,
        0.400236,
        0.405538,
        0.410894,
        0.416306,
        0.421773,
        0.427297,
        0.432877,
        0.438514,
        0.444209,
        0.449962,
        0.455774,
        0.461645,
        0.467577,
        0.473569,
        0.479621,
        0.485736,
        0.491912,
        0.498151,
        0.504454,
        0.510819,
        0.517250,
        0.523745,
        0.530308,
        0.536934,
    ],
    dtype=tf.float32,
)
_PMSQE_SQRT_WIDTH = tf.sqrt(tf.reduce_sum(_PMSQE_WIDTH))
_PMSQE_BARK = tf.constant(
    np.frombuffer(
        zlib.decompress(
            base64.b64decode(
                "eNrt2c0qxFEYx/FDM8gVWNvaSNnrV0osLNjYWxBlM2PrvRg1pSmapVLKRlPuwaxE2VFWZmNjJ5G/zi2MZ+o8zvd7BZ+ezkunE0JbwUlFcevGGhzNtQ9rT+pnb/Wkb0fWwTAjnPmerZ7Wqqfz6tPRXIcczXVyfFE4bRsIz26sna83R/fArnDaNnFwJJy2fRTHwmlb/akunLaVw6lw2nZ/3RRO20rv58Jp28NsSzhtG61uCWd+zlht/lA483PGNkdOhDM/Z2xj7Ew483PG5vZXhRNnyj12fLxJcNpXmd4WTpwpV64tCSdOnH/v7qIhnDhTrv3i427CiTP1WjfrwokTZx7O2OvwnnDixJmHM7bWrAonTpw4U2unsSKcOHHixNldl6Vl4cSJEyfO/+uMXU35+F/AiRMnTpzdt/Dj418JJ06cOHHiTKVfSbLdIA=="
            )
        ),
        dtype=np.float32,
    ).reshape(129, 42)
)
_PMSQE_SLL_MASK = tf.constant(
    np.concatenate(
        [
            np.zeros(11, dtype=np.float32),
            np.array([0.5 * 25.0 / 31.25], dtype=np.float32),
            np.ones(92, dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            np.zeros(24, dtype=np.float32),
        ]
    )
    * np.float32(2.666666666666754 * (256.0 + 2.0) / 256.0**2),
    dtype=tf.float32,
)


def _pmsqe_power(wav):
    spectrum = tf.signal.stft(
        wav,
        frame_length=_PMSQE_FFT,
        frame_step=_PMSQE_HOP,
        fft_length=_PMSQE_FFT,
        window_fn=tf.signal.hann_window,
    )
    return tf.square(tf.abs(spectrum))


def _pmsqe_at_sll(spectra):
    masked = spectra * _PMSQE_SLL_MASK
    freq_mean = tf.reduce_mean(masked, axis=-1, keepdims=True)
    mean_power = tf.reduce_mean(freq_mean, axis=-2, keepdims=True)
    return 10000000.0 * spectra / (mean_power + 1e-8)


def _pmsqe_audible_power(bark_spectra, factor):
    threshold = _PMSQE_ABS_THRESH * factor
    audible = tf.where(bark_spectra > threshold, bark_spectra, tf.zeros_like(bark_spectra))
    return tf.reduce_sum(audible, axis=-1, keepdims=True)


def _pmsqe_frequency_equalization(reference, degraded):
    active = _pmsqe_audible_power(reference, 100.0) >= 1.0e7
    above = reference >= _PMSQE_ABS_THRESH * 100.0
    reference_bands = tf.where(above, reference, tf.zeros_like(reference))
    degraded_bands = tf.where(above, degraded, tf.zeros_like(degraded))
    reference_power = tf.reduce_sum(
        tf.where(active, reference_bands, tf.zeros_like(reference_bands)), axis=-2, keepdims=True
    )
    degraded_power = tf.reduce_sum(
        tf.where(active, degraded_bands, tf.zeros_like(degraded_bands)), axis=-2, keepdims=True
    )
    equalizer = (reference_power + 1000.0) / (degraded_power + 1000.0)
    equalizer = tf.clip_by_value(equalizer, 0.01, 100.0)
    return equalizer * degraded


def _pmsqe_gain_equalization(reference, degraded):
    gain = (_pmsqe_audible_power(reference, 1.0) + 5.0e3) / (
        _pmsqe_audible_power(degraded, 1.0) + 5.0e3
    )
    gain = tf.clip_by_value(gain, 3.0e-4, 5.0)
    return gain * degraded


def _pmsqe_loudness(bark_spectra):
    loudness = _PMSQE_SL * tf.pow(_PMSQE_ABS_THRESH / 0.5, _PMSQE_ZWICKER) * (
        tf.pow(
            0.5 + 0.5 * bark_spectra / _PMSQE_ABS_THRESH,
            _PMSQE_ZWICKER,
        )
        - 1.0
    )
    return tf.where(bark_spectra < _PMSQE_ABS_THRESH, tf.zeros_like(loudness), loudness)


def _pmsqe_from_power(reference_power, degraded_power):
    reference = _pmsqe_at_sll(reference_power)
    degraded = _pmsqe_at_sll(degraded_power)
    reference = 2.764344e-5 * tf.matmul(reference, _PMSQE_BARK)
    degraded = 2.764344e-5 * tf.matmul(degraded, _PMSQE_BARK)
    degraded = _pmsqe_frequency_equalization(reference, degraded)
    degraded = _pmsqe_gain_equalization(reference, degraded)
    reference_loudness = _pmsqe_loudness(reference)
    degraded_loudness = _pmsqe_loudness(degraded)
    difference = tf.abs(degraded_loudness - reference_loudness)
    masking = 0.25 * tf.minimum(reference_loudness, degraded_loudness)
    symmetric = tf.maximum(difference - masking, 1e-8)
    asymmetry = tf.pow((degraded + 50.0) / (reference + 50.0), 1.2)
    asymmetry = tf.where(asymmetry < 3.0, tf.zeros_like(asymmetry), tf.minimum(asymmetry, 12.0))
    asymmetric = asymmetry * symmetric
    symmetric_frame = tf.sqrt(
        tf.reduce_sum(tf.square(symmetric * _PMSQE_WIDTH) + 1e-8, axis=-1, keepdims=True)
    )
    symmetric_frame = symmetric_frame * _PMSQE_SQRT_WIDTH
    asymmetric_frame = tf.reduce_sum(asymmetric * _PMSQE_WIDTH, axis=-1, keepdims=True)
    audible = _pmsqe_audible_power(reference, 1.0)
    weights = tf.pow((audible + 1e5) / 1e7, 0.04)
    symmetric_frame = tf.minimum(symmetric_frame / weights, 45.0)
    asymmetric_frame = tf.minimum(asymmetric_frame / weights, 45.0)
    return tf.reduce_mean(_PMSQE_ALPHA * symmetric_frame + _PMSQE_BETA * asymmetric_frame)


def pmsqe_loss(y_true, y_pred):
    """Perceptual metric for speech quality evaluation at 8 kHz."""
    target = _pmsqe_power(_spec_to_waveform(y_true))
    estimate = _pmsqe_power(_spec_to_waveform(y_pred))
    return _pmsqe_from_power(target, estimate)


LOSS_FUNCTIONS = {
    "time-mse": time_domain_mse,
    "stsa-mse": stsa_mse,
    "stoi": stoi_loss,
    "estoi": estoi_loss,
    "si-sdr": si_sdr_loss,
    "pmsqe": pmsqe_loss,
}
LOSS_LABELS = {
    "time-mse": "Time-Domain Mean Square Error",
    "stsa-mse": "Short-Time Spectral Amplitude Mean Square Error",
    "stoi": "Short-Time Objective Intelligibility",
    "estoi": "Extended Short-Time Objective Intelligibility",
    "si-sdr": "Scale-Invariant Signal-to-Distortion Ratio",
    "pmsqe": "Perceptual Metric for Speech Quality Evaluation",
}


def complex_enhancement_loss_pc(y_true, y_pred, gamma=0.5, eps=1e-8):
    # Split Real and Imaginary
    # Shape expected: (Batch, Time, Freq, 2)
    r_t, i_t = y_true[..., 0], y_true[..., 1]
    r_p, i_p = y_pred[..., 0], y_pred[..., 1]

    # 1. Compressed Magnitude Loss
    mag_t = tf.sqrt(r_t**2 + i_t**2 + eps)
    mag_p = tf.sqrt(r_p**2 + i_p**2 + eps)
    mag_loss = tf.reduce_mean(tf.abs(mag_t**gamma - mag_p**gamma))

    # 2. Compressed Complex Loss (Handles Phase implicitly and stably)
    # This transforms the complex values into the compressed domain
    # Formula: (r + ji) / |mag| * |mag|^gamma = (r + ji) * |mag|^(gamma-1)
    factor_t = mag_t**(gamma - 1)
    factor_p = mag_p**(gamma - 1)
    
    c_real_t, c_imag_t = r_t * factor_t, i_t * factor_t
    c_real_p, c_imag_p = r_p * factor_p, i_p * factor_p
    
    complex_loss = tf.reduce_mean(tf.abs(c_real_t - c_real_p) + tf.abs(c_imag_t - c_imag_p))

    # 3. Temporal Consistency (Delta Loss)
    # Using the compressed magnitude for delta often yields better PESQ
    delta_mag_t = mag_t[:, 1:, :] - mag_t[:, :-1, :]
    delta_mag_p = mag_p[:, 1:, :] - mag_p[:, :-1, :]
    consistency_loss = tf.reduce_mean(tf.square(delta_mag_t - delta_mag_p))

    # 4. Scale-Invariant Signal-to-Noise Ratio (SI-SNR) 
    # Much more stable than a custom SI-L1 loss
    t_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    p_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    dot = tf.reduce_sum(t_flat * p_flat, axis=1, keepdims=True)
    snr_norm = tf.reduce_sum(t_flat**2, axis=1, keepdims=True) + eps
    target_proj = (dot / snr_norm) * t_flat
    
    noise_res = p_flat - target_proj
    si_snr = 10 * tf.math.log(tf.reduce_sum(target_proj**2, axis=1) / 
                             (tf.reduce_sum(noise_res**2, axis=1) + eps) + eps) / tf.math.log(10.0)
    
    si_loss = -tf.reduce_mean(si_snr) # Negative because we want to maximize SNR

    return (1.0 * mag_loss + 
            1.0 * complex_loss + 
            0.5 * consistency_loss + 
            2.0 * si_loss) # SI-SNR scale is much larger, so weight it lower


SR = 8000


# ==========================================================
# 🔢 METRICS
# ==========================================================
def si_sdr(est, ref, eps=1e-8):
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    scale = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    ref_scaled = scale * ref
    noise = est - ref_scaled
    return 10 * np.log10((np.sum(ref_scaled**2) + eps) / (np.sum(noise**2) + eps))


def sample_reference_segments_full(wav, K, segment_len):
    return _WAVEFORM_ENHANCER.sample_reference_segments_full(wav, K, segment_len)


def enhance_audio_consistent(noisy_wav, ref_wav, model, K=4, overlap=0.5):
    return _WAVEFORM_ENHANCER.enhance_audio_full_utterance(
        noisy_wav, ref_wav, model, overlap=overlap, max_ref_frames=MAX_REF_FRAMES
    )


def normalize(x):
    return _normalize(x)


def pesq_score(clean, enhanced):
    return _pesq_score(clean, enhanced)


def stoi_score(clean, enhanced):
    return _stoi_score(clean, enhanced)


def load_resample_8k(path):
    return _load_resample_8k(path)


def sanitize(x):
    return _sanitize(x)


# ==========================================================
# 📊 MAIN EVALUATION
# ==========================================================
def main():
    print(f"Evaluating {len(TEST_MIX)} samples")

    results = []

    for mix_path, ref_path, tgt_path in tqdm(
        zip(TEST_MIX, TEST_REF, TEST_TGT), total=len(TEST_MIX)
    ):
        # -------- Load audio --------
        noisy = load_resample_8k(mix_path)
        clean = load_resample_8k(tgt_path)
        ref = load_resample_8k(ref_path)

        # -------- Enhance --------
        enhanced = enhance_audio_consistent(noisy, ref, model)

        # -------- Align lengths --------
        L = min(len(clean), len(enhanced), len(noisy))
        clean, enhanced, noisy = clean[:L], enhanced[:L], noisy[:L]

        # -------- Metrics --------
        mix_sisdr = si_sdr(noisy, clean)
        enh_sisdr = si_sdr(enhanced, clean)

        mix_pesq = pesq_score(clean, noisy)
        enh_pesq = pesq_score(clean, enhanced)

        mix_stoi = stoi_score(clean, noisy)
        enh_stoi = stoi_score(clean, enhanced)

        results.append(
            [
                os.path.basename(mix_path),
                mix_sisdr,
                enh_sisdr,
                mix_pesq,
                enh_pesq,
                mix_stoi,
                enh_stoi,
                enh_sisdr - mix_sisdr,
                enh_pesq - mix_pesq,
                enh_stoi - mix_stoi,
            ]
        )

    arr = np.array(results, dtype=object)

    print("\n========== FINAL RESULTS ==========")
    print(f"SI-SDR (mix): {np.mean(arr[:,1].astype(float)):.2f}")
    print(f"SI-SDR (enh): {np.mean(arr[:,2].astype(float)):.2f}")
    print(f"SI-SDRi:      {np.mean(arr[:,7].astype(float)):.2f}")

    print(f"PESQ (mix):   {np.nanmean(arr[:,3].astype(float)):.3f}")
    print(f"PESQ (enh):   {np.nanmean(arr[:,4].astype(float)):.3f}")
    print(f"PESQi:        {np.nanmean(arr[:,8].astype(float)):.3f}")

    print(f"STOI (mix):   {np.nanmean(arr[:,5].astype(float)):.3f}")
    print(f"STOI (enh):   {np.nanmean(arr[:,6].astype(float)):.3f}")
    print(f"STOIi:        {np.nanmean(arr[:,9].astype(float)):.3f}")

    # -------- Save CSV --------
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results_full.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "mix_sisdr",
                "enh_sisdr",
                "mix_pesq",
                "enh_pesq",
                "mix_stoi",
                "enh_stoi",
                "sisdr_i",
                "pesq_i",
                "stoi_i",
            ]
        )
        writer.writerows(results)

    print("\nSaved results to", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results_full.csv"))


# ==========================================================
if args.l is None:
    model = _build_model(variable_time=True)
    model.summary()
    eval_weights = _checkpoint_filename("si_sdr")
    model.load_weights(eval_weights)
    model.trainable = False
    print("Model loaded for inference")
    main()
else:
    tf.config.optimizer.set_jit(True)
    from libri2mix.conf_unet_tse_32ms import chunk_size, dev_data, train_data, trainer_conf
    from libri2mix.dataset_tse import make_dataloader
    from libri2mix.trainer_tse import SiSnrTrainer, chunk_frames_match

    if not chunk_frames_match(chunk_size, frame_length, frame_step, N_FRAMES):
        raise SystemExit(
            f"Chunk STFT frames {conv_stft_frames(chunk_size, frame_length, frame_step)} "
            f"do not match the model input {N_FRAMES}."
        )
    visible = tf.config.list_physical_devices("GPU")
    if not visible:
        raise SystemExit("Early fusion training needs a GPU.")
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    if num_gpus != len(visible):
        raise SystemExit(
            f"MirroredStrategy has {num_gpus} replicas, but {len(visible)} GPUs are visible."
        )
    batch_size = PER_GPU_BATCH * num_gpus
    with strategy.scope():
        model = _build_model()
        trainer = SiSnrTrainer(
            model,
            _AUDIO_TOOLKIT,
            MAX_REF_FRAMES,
            model_filename,
            logging_period=trainer_conf["logging_period"],
            strategy=strategy,
        )
    model.summary()
    print(f"Training with SEF-PNet SiSnrTrainer ({LOSS_LABELS[args.l]})")
    print(f"Checkpoint: {model_filename}")
    run_epochs = EPOCHS if args.epochs is None else args.epochs
    print(
        f"Adam lr={trainer_conf['optimizer_kwargs']['lr']}, "
        f"batch {batch_size} ({PER_GPU_BATCH} per GPU x {num_gpus}), epochs {run_epochs}"
    )
    trainer.run(
        make_dataloader(
            train=True, data_kwargs=train_data, chunk_size=chunk_size, batch_size=batch_size
        ),
        make_dataloader(
            train=False, data_kwargs=dev_data, chunk_size=chunk_size, batch_size=batch_size
        ),
        num_epochs=run_epochs,
    )