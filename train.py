import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*unable to load libtensorflow_io_plugins.so.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*file system plugins are not loaded.*")
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import soundfile as sf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Dropout,
    Lambda,
    SpatialDropout2D,
    LayerNormalization,
    UpSampling2D,
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

tf.config.optimizer.set_jit(True)


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
n_fft = 510  # 128 freq bins
frame_length = 400
frame_step = 160
trim_length = 31000  # 384 time frames after STFT
total_length = 3.855  # seconds
batch_size = 4
EPOCHS = 300
CHUNK_SIZE = 31000 # chunk into 4s
STRIDE = CHUNK_SIZE // 2  # 50% overlap
TARGET_SR = 8000
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
    mixture_files, reference_files, target_files, is_train=True, K=4
):
    return _DATASET_BUILDER.configure_dataset(
        mixture_files, reference_files, target_files, is_train=is_train, K=K
    )


train_ds = configure_libri_speech_dataset(
    TRAIN_MIX, TRAIN_REF, TRAIN_TGT, is_train=True
)
val_ds = configure_libri_speech_dataset(DEV_MIX, DEV_REF, DEV_TGT, is_train=False)
test_ds = configure_libri_speech_dataset(TEST_MIX, TEST_REF, TEST_TGT, is_train=False)


# # 4. Batch and Prefetch
train_dataset = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
for noise, clean in train_dataset.take(1):
    print(noise["noisy_main"].shape, noise["noisy_ref"].shape, clean.shape)
total_train_samples = len(TRAIN_MIX)
val_size = len(DEV_MIX)
steps_per_epoch = total_train_samples // batch_size
validation_steps = val_size // batch_size
print(f"Train steps: {steps_per_epoch}")
print(f"Val steps: {validation_steps}")


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


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
    Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection"""
    below_filters = conv_below.shape[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def film(x, speaker_embedding):
    """
    Personalized Feature-wise Linear Modulation.
    """
    C = x.shape[-1]

    # In 2026 research, zero-initializing these layers is standard
    # so the model starts by doing nothing and learns to modulate.
    gamma = Dense(C, kernel_initializer="zeros")(speaker_embedding)
    beta = Dense(C, kernel_initializer="zeros")(speaker_embedding)

    # Reshape for broadcasting (B, 1, 1, C)
    gamma = Reshape((1, 1, C))(gamma)
    beta = Reshape((1, 1, C))(beta)

    # Modern FiLM: x = x * (1 + gamma) + beta
    return Multiply()([x, 1.0 + gamma]) + beta


def upsample_bilinear_personalized(x, speaker_embedding, filters, kernel_size=(3, 3)):
    """
    The Bilinear + FiLM replacement for Conv2DTranspose.
    """
    # 1. Spatial Expansion (Bilinear is the 2026 'artifact-free' choice)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    # 2. Refinement Convolution
    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # 3. Inject Speaker conditioning
    x = film(x, speaker_embedding)

    return x


def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def cross_attention_cond(x, speaker_embedding, num_heads=4, key_dim=64):
    """
    Cross-attention conditioning.

    x: (B, T, F, C)
    speaker_embedding: (B, D)
    returns: (B, T, F, C)
    """
    B, T, F, C = x.shape
    # Flatten spatial dims: (B, T*F, C)
    x_flat = Reshape((T * F, C))(x)
    # Speaker as query: (B, 1, D)
    q = Reshape((1, speaker_embedding.shape[-1]))(speaker_embedding)
    # Project to match channels
    q = Dense(C)(q)
    # Cross-attention
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        query=q, value=x_flat, key=x_flat
    )
    # Broadcast back to all positions
    attn = RepeatVector(T * F)(attn[:, 0, :])
    # Reshape back to feature map
    attn_map = Reshape((T, F, C))(attn)
    # Residual modulation
    return x + attn_map
    
def tf_alternating_block(x, filters, activation="relu", use_bn=True, name_prefix="tfb"):
    orginal_x = x
    # ---- Frequency branch (1 x 3) for the first branch, we get time convolutions over frequency
    f_branch = Conv2D(filters, (1, 3), padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name_prefix}_fconv")(x)
    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_fbn")(f_branch)
    x = Activation(activation)(x)

    # ---- Time branch (3 x 1) ----
    t_branch = Conv2D(filters, (3, 1), padding="same",
                      kernel_initializer="he_normal",
                      name=f"{name_prefix}_tconv")(x)
    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_tbn")(t_branch)
    x = Activation(activation)(x)

    # ---- Merge ----
    x = Concatenate(name=f"{name_prefix}_concat")([f_branch, t_branch])

    # sencond branch, we get frequency convolutions over time
    t_branch_2 = Conv2D(filters, (3, 1), padding="same",
                        kernel_initializer="he_normal",
                        name=f"{name_prefix}_tconv2")(orginal_x)
    if use_bn:
        orginal_x = BatchNormalization(name=f"{name_prefix}_tbn2")(t_branch_2)
    orginal_x = Activation(activation)(t_branch_2)

    f_branch_2 = Conv2D(filters, (1, 3), padding="same",
                        kernel_initializer="he_normal",
                        name=f"{name_prefix}_fconv2")(orginal_x)
    if use_bn:
        orginal_x = BatchNormalization(name=f"{name_prefix}_fbn2")(f_branch_2)
    orginal_x = Activation(activation)(f_branch_2)

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
def broadcast_speaker(embed, target_tensor):
        target_shape = target_tensor.shape # (Batch, T, F, C)
        s = layers.Dense(target_shape[-1])(embed)
        s = layers.Reshape((1, 1, target_shape[-1]))(s)
        # Tile/Broadcast to (T, F, C)
        return layers.UpSampling2D(size=(target_shape[1], target_shape[2]))(s)

def lca_block(x, channels, r=4, name="lca"):
    inter_channels = int(channels // r)
    
    # Local Attention
    xl = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name}_l1")(x)
    xl = layers.BatchNormalization()(xl)
    xl = layers.Activation("relu")(xl)
    xl = layers.Conv2D(channels, 1, padding="same", name=f"{name}_l2")(xl)
    xl = layers.BatchNormalization()(xl)
    
    # Global Attention
    xg = layers.GlobalAveragePooling2D(keepdims=True)(x)
    xg = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name}_g1")(xg)
    xg = layers.BatchNormalization()(xg)
    xg = layers.Activation("relu")(xg)
    xg = layers.Conv2D(channels, 1, padding="same", name=f"{name}_g2")(xg)
    xg = layers.BatchNormalization()(xg)
    
    # Combine
    xlg = layers.Add()([xl, xg])
    wei = layers.Activation("sigmoid")(xlg)
    return layers.Multiply()([x, wei])

def ifi_block(x, residual, channels, r=4, name="ifi"):
    # First Interaction
    xa = layers.Add()([x, residual])
    # LCA logic inside IFI
    xl = layers.Conv2D(channels // r, 1, padding="same")(xa)
    xl = layers.BatchNormalization()(xl)
    xl = layers.Activation("relu")(xl)
    xl = layers.Conv2D(channels, 1, padding="same")(xl)
    xl = layers.BatchNormalization()(xl)
    
    xg = layers.GlobalAveragePooling2D(keepdims=True)(xa)
    xg = layers.Conv2D(channels // r, 1, padding="same")(xg)
    xg = layers.BatchNormalization()(xg)
    xg = layers.Activation("relu")(xg)
    xg = layers.Conv2D(channels, 1, padding="same")(xg)
    xg = layers.BatchNormalization()(xg)
    
    wei = layers.Activation("sigmoid")(layers.Add()([xl, xg]))
    # Soft gating: xi = x * wei + residual * (1 - wei)
    xi = layers.Add()([
        layers.Multiply()([x, wei]),
        layers.Multiply()([residual, layers.Lambda(lambda x: 1.0 - x)(wei)])
    ])
    
    # Second Interaction (Iterative)
    xl2 = layers.Conv2D(channels // r, 1, padding="same")(xi)
    xl2 = layers.BatchNormalization()(xl2)
    xl2 = layers.Activation("relu")(xl2)
    xl2 = layers.Conv2D(channels, 1, padding="same")(xl2)
    xl2 = layers.BatchNormalization()(xl2)
    
    xg2 = layers.GlobalAveragePooling2D(keepdims=True)(xi)
    xg2 = layers.Conv2D(channels // r, 1, padding="same")(xg2)
    xg2 = layers.BatchNormalization()(xg2)
    xg2 = layers.Activation("relu")(xg2)
    xg2 = layers.Conv2D(channels, 1, padding="same")(xg2)
    xg2 = layers.BatchNormalization()(xg2)
    
    wei2 = layers.Activation("sigmoid")(layers.Add()([xl2, xg2]))
    # Final Output
    return layers.Add(name=name)([
        layers.Multiply()([x, wei2]),
        layers.Multiply()([residual, layers.Lambda(lambda x: 1.0 - x)(wei2)])
    ])

def attentive_pooling(x, name="att_pool"):
    """
    Self-attention pooling (Attentive Statistics Pooling lite).
    Input shape: (Batch, Time, Channels)
    Output shape: (Batch, Channels)
    """
    att_weights = layers.Dense(1, activation=None, name=f"{name}_dense")(x)
    

    att_weights = layers.Softmax(axis=1, name=f"{name}_softmax")(att_weights)
    

    context = layers.Lambda(
        lambda inputs: ops.sum(inputs[0] * inputs[1], axis=1),
        name=f"{name}_weighted_sum"
    )([x, att_weights])
    
    return context

def custom_unet(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=4,
    output_activation="sigmoid",
):
    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    main_input = Input(input_shape, name="noisy_main")  # (T, F, C)
    ref_input = Input((4, 98, 256, 2), name="noisy_ref")
    main_input_copy = ops.copy(main_input)

    x = main_input / (ops.std(main_input) + 1e-5)
    ref_x = ref_input / (ops.std(ref_input) + 1e-5)
    # plot the first element of the batch for both main and reference inputs

    down_layers = []
    for l in range(num_layers):
        x=tf_alternating_block(x, filters, activation, use_bn=True, name_prefix=f"tfb_{l}")
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2

    ref_enc = ref_x
    filters_ref = filters // (2**num_layers)
    for l in range(num_layers):
        original_ref_enc = ref_enc
        # first branch to obtain time using frequency
        ref_enc_f = TimeDistributed(
            Conv2D(filters_ref, (1, 3), activation=activation, padding="same")
        )(ref_enc)
        if use_batch_norm:
            ref_enc = TimeDistributed(BatchNormalization())(ref_enc_f)
        # then look at the time dimension by applying a 3x1 convolution
        ref_enc__t = TimeDistributed(
            Conv2D(filters_ref, (3, 1), activation=activation, padding="same")
        )(ref_enc)
        if use_batch_norm:
            ref_enc = TimeDistributed(BatchNormalization())(ref_enc__t)
        # concatenate them
        ref_enc = Concatenate()([ref_enc_f, ref_enc__t])


        # second branch to obtain frequency using time

        ref_enc_t_1 = TimeDistributed(
            Conv2D(filters_ref, (3, 1), activation=activation, padding="same")
        )(original_ref_enc)

        if use_batch_norm:
            original_ref_enc = TimeDistributed(BatchNormalization())(ref_enc_t_1)

        ref_enc_f_1 = TimeDistributed(
            Conv2D(filters_ref, (1, 3), activation=activation, padding="same")
        )(original_ref_enc)

        if use_batch_norm:
            original_ref_enc = TimeDistributed(BatchNormalization())(ref_enc_f_1)

        # concatenate them second branch
        ref_enc_branch2 = Concatenate()([ref_enc_t_1, ref_enc_f_1])

        # now concatenate both branches to get a richer representation that captures both time and frequency interactions
        ref_enc = Concatenate()([ref_enc, ref_enc_branch2])

        # then look at both time and frequency with a 3x3 convolution
        ref_enc = TimeDistributed(
            Conv2D(filters_ref, (3, 3), activation=activation, padding="same")
        )(ref_enc)
        if use_batch_norm:
            ref_enc = TimeDistributed(BatchNormalization())(ref_enc)
        ref_enc = TimeDistributed(MaxPooling2D((1, 2)))(ref_enc)
        filters_ref *= 2

    ref_seq = TimeDistributed(GlobalAveragePooling2D())(ref_enc)
    # Apply GRU Block
    # ref_seq = add_gru_block(
    #     ref_seq, hidden_dim=ref_seq.shape[-1], num_layers=2, prefix="ref"
    # )
    ref_seq = add_xlstm_block(ref_seq, hidden_dim=ref_seq.shape[-1], num_layers=2, prefix="ref")
    
    # NEW: Attentive Pooling
    speaker_embed = attentive_pooling(ref_seq, name="speaker_att_pool")
    
    # Optional: A final projection to stabilize the embedding
    speaker_embed = layers.Dense(256, activation="tanh", name="speaker_final_proj")(speaker_embed)

    T_small, F_small, C_small = x.shape[1], x.shape[2], x.shape[3]  # (24, 16, 256)

    # 1. Flatten the spatial (F_small) and channel (C_small) dimensions
    # Current x: (None, 24, 16, 256) -> Reshape to: (None, 24, 4096)
 # --- 1. Dimensionality Reduction (The "Compression") ---
    # Instead of Reshape(4096), we reduce the depth (channels) to make it manageable
    bottleneck_dim = 128 # Much more efficient than 512 or 4096
    x_compressed = Conv2D(bottleneck_dim, (1, 1), padding="same", name="bn_depth_reduce")(x)

    # --- 2. Temporal Processing Preparation ---
    # Reshape to (Time, Frequency * Compressed_Channels) 
    # Shape: (24, 16 * 128) = (24, 2048) -> Still high, but much better
    x_seq = Reshape((T_small, F_small * bottleneck_dim))(x_compressed)

    # --- 3. Speaker Injection (Residual Style) ---
    # Project speaker embed to match the seq dimension
    spk_proj = Dense(F_small * bottleneck_dim)(speaker_embed) # (None, 2048)
    spk_proj = RepeatVector(T_small)(spk_proj)               # (None, 24, 2048)

    # Add is often more stable than Concat for large feature vectors
    x_seq = layers.Add(name="bn_spk_fusion")([x_seq, spk_proj])

    x_seq = layers.LayerNormalization(name="bn_spk_norm")(x_seq)

    # --- 4. The xLSTM Core ---
    # Process the sequence with the xLSTM
    x_seq = add_xlstm_block(x_seq, hidden_dim=512, num_layers=2, prefix="main_bottleneck")

    # --- 5. Expansion & Reconstruction ---
    # Map back to the compressed spatial shape
    x_expanded = TimeDistributed(Dense(F_small * bottleneck_dim))(x_seq)
    x_reshaped = Reshape((T_small, F_small, bottleneck_dim))(x_expanded)

    # Restore original channel depth (C_small = 256)
    x = Conv2D(C_small, (1, 1), padding="same", activation=activation, name="bn_reconstruct")(x_reshaped)

    # Now proceed to FiLM and upsampling
    spk_res = broadcast_speaker(speaker_embed, x)
    x = ifi_block(x, spk_res, channels=x.shape[-1], name="bn_ifi")

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        # Apply FiLM conditioning after upsampling but before concatenation
        x = film(x, speaker_embed)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])

        spk_res_up = broadcast_speaker(speaker_embed, x)
        x = ifi_block(x, spk_res_up, channels=x.shape[-1], name=f"up_ifi_{filters}")

        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        # x = tf_alternating_block(x, filters, activation, use_bn=use_batch_norm, name_prefix=f"up_conv_{filters}")
    x = lca_block(x, channels=x.shape[-1], name="final_refinement")
    # --- Split noisy input into real / imag ---
    input_r = main_input_copy[..., 0:1]
    input_i = main_input_copy[..., 1:2]

    # --- Predict complex mask ---
    # Kernel init "zeros" helps stability at start of training
    mask_r = Conv2D(1, (1, 1), activation=None, kernel_initializer="zeros", name="mask_real")(x)
    mask_i = Conv2D(1, (1, 1), activation=None, kernel_initializer="zeros", name="mask_imag")(x)

    # --- Apply Learnable Scaling ---
    mask_r = LearnableScale(initial_value=2.0, name="scale_r")(mask_r)
    mask_i = LearnableScale(initial_value=2.0, name="scale_i")(mask_i)

    # --- Complex multiplication ---
    out_r = layers.Subtract()([
        layers.Multiply()([mask_r, input_r]),
        layers.Multiply()([mask_i, input_i])
    ])

    out_i = layers.Add()([
        layers.Multiply()([mask_r, input_i]),
        layers.Multiply()([mask_i, input_r])
    ])

    # --- Residual connection (very important) ---
    out_r = layers.Add()([input_r, out_r])
    out_i = layers.Add()([input_i, out_i])

    # --- Merge back to 2-channel ---
    outputs = Concatenate(axis=-1)([out_r, out_i])

    return Model(inputs=[main_input, ref_input], outputs=[outputs])


model_filename = (
    "model_weights_final_version_hard_convolution_baseline_LIBRIMIX.keras"
)
model = custom_unet(
    input_shape=(192, 256, 2),
    use_batch_norm=True,
    num_classes=2,
    filters=32,
    use_dropout_on_upsampling=False,
    num_layers=4,
    use_attention=False,
    upsample_mode="deconv",
    dropout=0.2,
    output_activation="sigmoid",
)
callbacks = [
    ModelCheckpoint(
        model_filename,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    # EarlyStopping(
    #     monitor="val_loss",
    #     patience=100,
    #     min_delta=0.00001,
    #     restore_best_weights=True,
    #     verbose=1,
    # ),
]


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


total_steps = steps_per_epoch * EPOCHS
warmup_steps = steps_per_epoch * 10
initial_lr = 1e-4
alpha = 0.05  # final lr fraction


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, total_steps, warmup_steps, alpha=0.0):
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        def warmup_lr():
            return self.initial_lr * step / warmup_steps

        def decay_lr():
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(math.pi) * progress))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.initial_lr * decayed

        return tf.cond(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }


class LrLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(tf.cast(self.model.optimizer.iterations, tf.float32))
        print(f"Epoch {epoch+1}: Learning rate = {lr.numpy():.6f}")


lr_schedule = WarmupCosineDecay(
    initial_lr=initial_lr,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
    alpha=alpha,
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule, weight_decay=1e-3, clipnorm=1.0
)

model.summary()

model.compile(optimizer=optimizer, loss=complex_enhancement_loss_pc)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    # steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    # validation_steps=validation_steps,
    callbacks=callbacks + [LrLogger(), reduce_lr]
)


# Evaluate the model on test set
# model.load_weights(
#     "model_weights_final_version_hard_convolution_baseline_LIBRIMIX.weights.
# )
# model.trainable = False
# print("Model loaded for inference")

# load the keras model for inference
model = tf.keras.models.load_model(
    "model_weights_final_version_hard_convolution_baseline_LIBRIMIX.keras",
    custom_objects={
        "sLSTMCell": sLSTMCell,
        "mLSTMCell": mLSTMCell,
        "complex_enhancement_loss_pc": complex_enhancement_loss_pc,
    }
)
model.trainable = False
print("Model loaded for inference")


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
    return _WAVEFORM_ENHANCER.enhance_audio_consistent(
        noisy_wav, ref_wav, model, K=K, overlap=overlap
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
    with open("evaluation_results_full.csv", "w", newline="") as f:
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

    print("\nSaved results to evaluation_results_full.csv")


# ==========================================================
main()