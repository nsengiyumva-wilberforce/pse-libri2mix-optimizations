import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # turn off oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # hide AVX log message
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
    Conv2DTranspose,
    MaxPooling2D,
    Dropout, Lambda,
    SpatialDropout2D,
    LayerNormalization,
    MaxPooling2D,
    UpSampling2D,
    RNN,
    Softmax,
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
   Rescaling
)
def load_scp(mix_path, ref_path, tgt_path):
    """
    Loads three SCP files and returns three aligned lists:
    (mix, ref, tgt)
    Each SCP file must have format:
        <utt_id> <filepath>
    """
    def get_dict(path):
        d = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                key = parts[0]
                value = parts[1]
                if not os.path.isabs(value):
                    value = os.path.abspath(value)
                d[key] = value
        return d
    # ---- Load all SCPs ----
    mix_d = get_dict(mix_path)
    ref_d = get_dict(ref_path)
    tgt_d = get_dict(tgt_path)
    # ---- Debug counts ----
    print("\n=== SCP LOAD DEBUG ===")
    print(f"Mix entries: {len(mix_d)}")
    print(f"Ref entries: {len(ref_d)}")
    print(f"Tgt entries: {len(tgt_d)}")
    # ---- Align keys ----
    common_keys = sorted(set(mix_d) & set(ref_d) & set(tgt_d))
    if len(common_keys) == 0:
        raise ValueError(
            "No overlapping keys found between SCP files.\n"
            "Check that all SCPs use identical utterance IDs."
        )
    print(f"Aligned samples: {len(common_keys)}")
    # ---- Build aligned lists ----
    mix_list = [mix_d[k] for k in common_keys]
    ref_list = [ref_d[k] for k in common_keys]
    tgt_list = [tgt_d[k] for k in common_keys]
    # ---- Sanity check (first sample) ----
    print("\n=== SAMPLE CHECK ===")
    print("Mix:", mix_list[0])
    print("Ref:", ref_list[0])
    print("Tgt:", tgt_list[0])
    # ---- Critical warning ----
    if ref_list[0] == tgt_list[0]:
        print("\n[WARNING] REF == TARGET → conditioning will collapse!")
    return mix_list, ref_list, tgt_list
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
    ref_path = os.path.join(base, "auxs1.scp") # enrollment
    tgt_path = os.path.join(base, "ref.scp") # clean target
    return load_scp(mix_path, ref_path, tgt_path)
   
    sr = 8000
n_fft = 510 # 128 freq bins
frame_length = 400
frame_step = 160
trim_length = 61680 # 384 time frames after STFT
total_length = 3.855 # seconds
batch_size = 4
CHUNK_SIZE = 32300 # chunk into 4s
STRIDE = CHUNK_SIZE // 2 # 50% overlap
TARGET_SR = 8000
TRAIN_MIX, TRAIN_REF, TRAIN_TGT = load_scp(
    "data/train/mix_clean.scp",
    "data/train/auxs1.scp", # enrollment
    "data/train/ref.scp" # clean target
)
DEV_MIX, DEV_REF, DEV_TGT = load_scp(
    "data/dev/mix_clean.scp",
    "data/dev/auxs1.scp", # enrollment
    "data/dev/ref.scp" # clean target
)
TEST_MIX, TEST_REF, TEST_TGT = load_scp(
    "data/test/mix_clean.scp",
    "data/test/auxs1.scp", # enrollment
    "data/test/ref.scp" # clean target
)
TARGET_SR = 8000
def load_audio_py(path):
    # handle all possible types safely
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    elif isinstance(path, np.ndarray):
        path = path.item().decode("utf-8") if path.dtype.type is np.bytes_ else path.item()
    audio, sr = sf.read(path)
    audio = audio.astype("float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio
def load_audio_tf(path):
    audio = tf.numpy_function(load_audio_py, [path], tf.float32)
    audio.set_shape([None]) # important for TF graph
    return audio
# def load_audio_python(filename):
# # This part runs in standard Python, not Graph mode
# # filename comes in as a byte string from TF
# data, samplerate = sf.read(filename.numpy().decode('utf-8'))
   
# # Ensure it's float32 and mono
# if len(data.shape) > 1:
# print("converted to mono")
# data = np.mean(data, axis=1)
# return data.astype(np.float32)
# def normalize(wav):
# return wav / (tf.reduce_max(tf.abs(wav)) + 1e-8)
# @tf.function
# def load_wav(filename):
# # Wrap the python function so TF can use it in a graph
# audio = tf.py_function(
# func=load_audio_python,
# inp=[filename],
# Tout=tf.float32
# )
   
# # py_function loses shape information, so we set it manually
# # if you know your input size, or just leave it partially known
# audio.set_shape([None])
# return audio
   
@tf.function
def preprocess_tf(filepath):
    wav = load_audio_tf(filepath)
    # normalize
    wav = wav / (tf.reduce_max(tf.abs(wav)) + 1e-8)
    return wav
@tf.function
def split_into_chunks(wav, chunk_size, stride):
    length = tf.shape(wav)[0]
    # pad RIGHT if too short
    def pad():
        pad_len = chunk_size - length
        return tf.pad(wav, [[0, pad_len]])
    wav = tf.cond(length < chunk_size, pad, lambda: wav)
    length = tf.shape(wav)[0]
    starts = tf.range(0, length - chunk_size + 1, stride)
    def get_chunk(s):
        return wav[s:s+chunk_size]
    chunks = tf.map_fn(get_chunk, starts, fn_output_signature=tf.float32)
    return chunks # (num_chunks, chunk_size)
@tf.function
def tf_rms(x, eps=1e-8):
    return tf.sqrt(tf.reduce_mean(tf.square(x)) + eps)
@tf.function
def convert_to_spectrogram(wav_corr,wav_ref, wavclean):
    spectrogram_corr = tf.signal.stft(wav_corr, frame_length=frame_length, fft_length=n_fft,
                                      frame_step=frame_step)
    spectrogram_ref = tf.signal.stft(wav_ref, frame_length=frame_length, fft_length=n_fft,
                                      frame_step=frame_step)
    spectrogram = tf.signal.stft(wavclean, frame_length=frame_length, fft_length=n_fft,
                                      frame_step=frame_step)
    return spectrogram_corr, spectrogram_ref, spectrogram
@tf.function
def convert_to_spectrogram_multiview(wav_corr, wav_ref_segments, wavclean):
    # main mixture
    spectrogram_corr = tf.signal.stft(
        wav_corr,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=n_fft
    )
    # clean target
    spectrogram_clean = tf.signal.stft(
        wavclean,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=n_fft
    )
    # reference: vectorized over K
    # wav_ref_segments: (K, N)
    spectrogram_refs = tf.map_fn(
        lambda x: tf.signal.stft(
            x,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=n_fft
        ),
        wav_ref_segments,
        fn_output_signature=tf.complex64
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
    return tf.stack([tf.math.real(spec), tf.math.imag(spec)], axis=-1)
   
@tf.function
def sample_reference_segments(wav, K, segment_len):
    wav_len = tf.shape(wav)[0]
    # If too short: pad
    def pad():
        pad_len = segment_len - wav_len
        wav_pad = tf.pad(wav, [[0, pad_len]])
        return tf.tile(tf.expand_dims(wav_pad, 0), [K, 1])
    # If long enough: sample
    def sample():
        max_start = wav_len - segment_len
        starts = tf.random.uniform(
            [K], 0, max_start + 1, dtype=tf.int32
        )
        return tf.map_fn(
            lambda s: wav[s:s + segment_len],
            starts,
            fn_output_signature=tf.float32
        )
    return tf.cond(wav_len < segment_len, pad, sample)
def load_libri_speech_triplet_multiview(mix_path, ref_path, tgt_path, K=4, ref_len=8000*2):
    clean = preprocess_tf(tgt_path)
    noisy = preprocess_tf(mix_path)
    ref = preprocess_tf(ref_path)
    mix_chunks = split_into_chunks(noisy, CHUNK_SIZE, STRIDE)
    clean_chunks = split_into_chunks(clean, CHUNK_SIZE, STRIDE)
    ref_segments = sample_reference_segments(ref, K, ref_len)
    return mix_chunks, ref_segments, clean_chunks
   
def configure_libri_speech_dataset(mixture_files, reference_files, target_files, is_train=True, K=4):
    ds = tf.data.Dataset.from_tensor_slices(
        (mixture_files, reference_files, target_files)
    ).cache()
    if is_train:
        ds = ds.shuffle(batch_size * 1000)
    # --------------------------------------------------
    # 1. Load + chunk
    # --------------------------------------------------
    ds = ds.map(
        lambda n, r, t: load_libri_speech_triplet_multiview(n, r, t, K),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # --------------------------------------------------
    # 2. FLATTEN chunks (CRITICAL)
    # --------------------------------------------------
    ds = ds.flat_map(
        lambda mix_chunks, ref_segments, clean_chunks:
            tf.data.Dataset.from_tensor_slices((mix_chunks, clean_chunks))
            .map(lambda m, c: (m, ref_segments, c))
    )
    # Now each element is:
    # (chunk, same_ref_segments, chunk)
    # --------------------------------------------------
    # 3. STFT
    # --------------------------------------------------
    ds = ds.map(
        lambda mix, ref, clean: convert_to_spectrogram_multiview(mix, ref, clean),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # --------------------------------------------------
    # 4. Convert to 2-channel (real/imag)
    # --------------------------------------------------
    ds = ds.map(
        lambda spec_noisy, spec_refs, spec_clean: (
            {
                "noisy_main": complex_to_2ch(spec_noisy),
                "noisy_ref": complex_to_2ch(spec_refs),
            },
            complex_to_2ch(spec_clean),
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # --------------------------------------------------
    # 5. Batch LAST
    # --------------------------------------------------
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = configure_libri_speech_dataset(
    TRAIN_MIX, TRAIN_REF, TRAIN_TGT, is_train=True
)
val_ds = configure_libri_speech_dataset(
    DEV_MIX, DEV_REF, DEV_TGT, is_train=False
)
test_ds = configure_libri_speech_dataset(
    TEST_MIX, TEST_REF, TEST_TGT, is_train=False
)



# # 4. Batch and Prefetch
train_dataset = train_ds.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
for noise, clean in train_dataset.take(1):
    print(noise["noisy_main"].shape,noise["noisy_ref"].shape,clean. shape)
total_train_samples = len(TRAIN_MIX)
val_size = len(DEV_MIX)
steps_per_epoch = total_train_samples // batch_size
validation_steps = val_size// batch_size
print(f"Train steps: {steps_per_epoch}")
print(f"Val steps: {validation_steps}")

# ============ sLSTM implementation ============
class sLSTMCell(Layer):
    def __init__(self, units, forget_gate_type='sigmoid', **kwargs):
        """
        sLSTM cell with proper state handling.
        """
        super(sLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.forget_gate_type = forget_gate_type
       
        # Input projections
        self.W_z = Dense(units, use_bias=True) # Cell input
        self.W_i = Dense(units, use_bias=True) # Input gate
        self.W_f = Dense(units, use_bias=True) # Forget gate
        self.W_o = Dense(units, use_bias=True) # Output gate
       
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
        if self.forget_gate_type == 'exponential':
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
        config.update({
            "units": self.units,
            "forget_gate_type": self.forget_gate_type,
        })
        return config
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        return [
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32), # h
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32), # c
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32), # n
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32) # m
        ]
       
       
# ============ mLSTM CELL implementation ============
class mLSTMCell(Layer):
    def __init__(self, units, forget_gate_type='sigmoid', **kwargs):
        super(mLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.forget_gate_type = forget_gate_type
       
        # Layer normalization for keys/values
        self.ln_kv = LayerNormalization(epsilon=1e-6)
       
        # Key scaling: 1/√d
        self.key_scale = 1.0 / tf.math.sqrt(tf.cast(units, tf.float32))
       
        # Projections
        self.W_q = Dense(units, use_bias=True) # Query
        self.W_k = Dense(units, use_bias=True) # Key
        self.W_v = Dense(units, use_bias=True) # Value
        self.W_i = Dense(units, use_bias=True) # Input gate
        self.W_f = Dense(units, use_bias=True) # Forget gate
        self.W_o = Dense(units, activation='sigmoid', use_bias=True) # Output gate
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
       
        if self.forget_gate_type == 'exponential':
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
        q_t = self.W_q(inputs) # Query
        k_t = self.W_k(inputs_norm) * self.key_scale # Scaled key
        v_t = self.W_v(inputs_norm) # Value
       
        # Gate logits
        i_tilde = self.W_i(inputs)
        f_tilde = self.W_f(inputs)
       
        # Stabilize gates
        i_t, f_t, m_t = self._stabilize_gates(i_tilde, f_tilde, m_prev)
       
        # Output gate
        o_t = self.W_o(inputs)
       
        # Update cell state: C_t = f_t C_{t-1} + i_t v_t k_t^T
        f_t_exp = tf.expand_dims(f_t, axis=-1) # [B, d, 1]
        v_t_exp = tf.expand_dims(v_t, axis=-1) # [B, d, 1]
        k_t_exp = tf.expand_dims(k_t, axis=1) # [B, 1, d]
       
        new_memory = tf.expand_dims(i_t, axis=-1) * v_t_exp * k_t_exp
        C_t = f_t_exp * C_prev + new_memory
       
        # Update normalizer state: n_t = f_t n_{t-1} + i_t k_t
        n_t = f_t * n_prev + i_t * k_t
       
        # Compute hidden state: h̃_t = C_t q_t / max(|n_t^T q_t|, 1)
        q_t_exp = tf.expand_dims(q_t, axis=-1) # [B, d, 1]
        numerator = tf.squeeze(C_t @ q_t_exp, axis=-1) # [B, d]
       
        dot_product = tf.reduce_sum(n_t * q_t, axis=-1, keepdims=True) # [B, 1]
        denominator = tf.maximum(tf.abs(dot_product), 1.0) # [B, 1]
       
        h_tilde = numerator / denominator
        h_t = o_t * h_tilde
       
        # Flatten C for next step
        C_flat_t = tf.reshape(C_t, [batch_size, self.units * self.units])
       
        return h_t, [h_t, C_flat_t, n_t, m_t]
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "forget_gate_type": self.forget_gate_type,
        })
        return config
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        return [
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32), # h
            tf.zeros((batch_size, self.units * self.units), dtype=dtype or tf.float32), # C
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32), # n
            tf.zeros((batch_size, self.units), dtype=dtype or tf.float32) # m
        ]
       
       
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
   
def respath(x, filters, depth=4, prefix="rp"):
    """
    Exact ResPath as described in the paper:
    No residual shortcut, just dual-branch conv summation.
    """
    for i in range(depth):
        # 3x3 branch
        conv3 = Conv2D(
            filters, (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name=f"{prefix}_conv3_{i}"
        )(x)
        conv3 = BatchNormalization(name=f"{prefix}_bn3_{i}")(conv3)
        conv3 = Activation("relu")(conv3)
        # 1x1 branch
        conv1 = Conv2D(
            filters, (1, 1),
            padding="same",
            kernel_initializer="he_normal",
            name=f"{prefix}_conv1_{i}"
        )(x)
        conv1 = BatchNormalization(name=f"{prefix}_bn1_{i}")(conv1)
        conv1 = Activation("relu")(conv1)
        # combine (this is the ONLY merge)
        x = Add(name=f"{prefix}_add_{i}")([conv3, conv1])
    return x
def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
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
    gamma = Dense(C, kernel_initializer='zeros')(speaker_embedding)
    beta = Dense(C, kernel_initializer='zeros')(speaker_embedding)
   
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
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
   
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
    attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(query=q, value=x_flat, key=x_flat)
    # Broadcast back to all positions
    attn = RepeatVector(T * F)(attn[:, 0, :])
    # Reshape back to feature map
    attn_map = Reshape((T, F, C))(attn)
    # Residual modulation
    return x + attn_map
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
    main_input = Input(input_shape, name="noisy_main") # (T, F, C)
    ref_input = Input((4, 98, 256, 2), name="noisy_ref")
    main_input_copy = ops.copy(main_input)
   
    x = main_input / (ops.std(main_input) + 1e-5)
    ref_x = ref_input / (ops.std(ref_input) + 1e-5)
   
    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(x, filters=filters, use_batch_norm=use_batch_norm,
                         dropout=dropout, dropout_type=dropout_type, activation=activation)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2
    # --- Reference Encoder with xLSTM ---
    # ref_enc = ref_x
    # filters_ref = filters // (2 ** num_layers)
    # for l in range(num_layers):
    # ref_enc = TimeDistributed(Conv2D(filters_ref, (3,3), activation=activation, padding="same"))(ref_enc)
    # if use_batch_norm:
    # ref_enc = TimeDistributed(BatchNormalization())(ref_enc)
    # ref_enc = TimeDistributed(MaxPooling2D((2,2)))(ref_enc)
    # filters_ref *= 2
    ref_enc = ref_x
    filters_ref = filters // (2 ** num_layers)
    for l in range(num_layers):
        # first look at the frequency bins by applying a 1x3 convolution
        ref_enc_f = TimeDistributed(
            Conv2D(filters_ref, (1,3), activation=activation, padding="same")
        )(ref_enc)
        if use_batch_norm:
            ref_enc = TimeDistributed(BatchNormalization())(ref_enc_f)
        # then look at the time dimension by applying a 3x1 convolution
        ref_enc__t = TimeDistributed(
            Conv2D(filters_ref, (3,1), activation=activation, padding="same")
        )(ref_enc)
        if use_batch_norm:
            ref_enc = TimeDistributed(BatchNormalization())(ref_enc__t)
        # concatenate them
        ref_enc = Concatenate()([ref_enc_f, ref_enc__t])
        #then look at both time and frequency with a 3x3 convolution
        ref_enc = TimeDistributed(
            Conv2D(filters_ref, (3,3), activation=activation, padding="same")
        )(ref_enc)
        if use_batch_norm:
            ref_enc = TimeDistributed(BatchNormalization())(ref_enc)
        ref_enc = TimeDistributed(MaxPooling2D((1,2)))(ref_enc)
        filters_ref *= 2
   
    ref_seq = TimeDistributed(GlobalAveragePooling2D())(ref_enc)
    # Replaced GRU with xLSTM Block
    ref_seq = add_xlstm_block(ref_seq, hidden_dim=ref_seq.shape[-1], num_layers=2, prefix="ref")
    speaker_embed = GlobalAveragePooling1D()(ref_seq)
    # # keep the original from the decoder
    # B, T, F, C = x.shape
    # # # 1. Flatten the spatial (F_small) and channel (C_small) dimensions
    # # # Current x: (None, 24, 16, 256) -> Reshape to: (None, 24, 4096)
    # # x_flat = Reshape((T_small, F_small * C_small), name="bottleneck_flatten")(x)
   
    # # # 2. Project 4096 down to 256 (This is where you save millions of parameters!) 128 small, 256 medium, 512 large
    # # x_reduced = TimeDistributed(Dense(2048, activation=activation), name="bottleneck_projection")(x_flat)
    # # --------------------------------------------------
    # # 1. Frequency compression (light, structured)
    # # --------------------------------------------------
    # x = Conv2D(C, (1, 3), padding="same", activation=activation)(x)
    # x = BatchNormalization()(x)
    # # --------------------------------------------------
    # # 2. Temporal modeling (xLSTM ONLY on time)
    # # --------------------------------------------------
    # # reshape: (B, T, F*C)
    # x_seq = Reshape((T, F * C))(x)
    # # compress channels (important for memory)
    # x_seq = TimeDistributed(Dense(256, activation=activation))(x_seq)
   
    # # 3. Add Speaker Embedding
    # # x_reduced: (None, 24, 256), speaker_embed: (None, 256)
    # spk = RepeatVector(T)(speaker_embed)
    # x_seq = Concatenate()([x_seq, spk])
   
    # # 4. Apply xLSTM on the compressed dimension (hidden_dim=512 is plenty, medium, small-256, large, 1024)
    # x_seq = add_xlstm_block(x_seq, hidden_dim=256, num_layers=2, prefix="main", block_types=["sLSTM", "mLSTM", "mLSTM", "mLSTM", "mLSTM", "mLSTM"])
   
    # # 5. Project back up to the original flattened size (4096)
    # x_seq = TimeDistributed(Dense(F * C, activation=activation))(x_seq)
    # x = Reshape((T, F, C))(x_seq)
    # x: (B, T, F, C)
    B, T, F, C = x.shape
   
    # --------------------------------------------------
    # 1. Frequency-aware compression (NO flattening)
    # --------------------------------------------------
    # compress channels only, keep (T, F)
    x = Conv2D(C // 2, (1, 1), padding="same", activation=activation)(x)
    x = BatchNormalization()(x)
   
    # optional: lightweight frequency mixing
    x = DepthwiseConv2D((1, 3), padding="same", activation=activation)(x)
    x = BatchNormalization()(x)
   
    C_reduced = x.shape[-1]
   
    # --------------------------------------------------
    # 2. Reshape for temporal modeling (preserve F)
    # --------------------------------------------------
    # (B, T, F, C) → (B, T, F*C_reduced but structured)
    x_seq = Reshape((T, F * C_reduced))(x)
   
    # LIGHT projection (not too aggressive)
    x_seq = TimeDistributed(Dense(1024, activation=activation))(x_seq)
   
    # --------------------------------------------------
    # 3. Inject speaker BEFORE temporal modeling
    # --------------------------------------------------
    spk = RepeatVector(T)(speaker_embed)
    x_seq = Concatenate()([x_seq, spk])
   
    # --------------------------------------------------
    # 4. xLSTM (now operates on better representation)
    # --------------------------------------------------
    x_seq = add_xlstm_block(
        x_seq,
        hidden_dim=512,
        num_layers=2,
        prefix="main"
    )
   
    # --------------------------------------------------
    # 5. Speaker re-injection (critical)
    # --------------------------------------------------
    x_seq = Concatenate()([x_seq, spk])
   
    # --------------------------------------------------
    # 6. Project back (less lossy than before)
    # --------------------------------------------------
    x_seq = TimeDistributed(Dense(F * C_reduced, activation=activation))(x_seq)
   
    x = Reshape((T, F, C_reduced))(x_seq)
   
    # --------------------------------------------------
    # 7. Restore channels
    # --------------------------------------------------
    x = Conv2D(C, (1, 1), padding="same", activation=activation)(x)
    x = BatchNormalization()(x)
   
    # Now proceed to FiLM and upsampling
    x = cross_attention_cond(x, speaker_embed)
   
    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
    for n, conv in enumerate(reversed(down_layers)):
        filters //= 2
        dropout -= dropout_change_per_layer
        x = upsample_bilinear_personalized(x, speaker_embed, filters)
        conv = respath(conv, filters=conv.shape[-1], depth=4, prefix=f"rp_{n}")
        # target_h = x.shape[1]
        # target_w = x.shape[2]
        # conv = Resizing(target_h, target_w)(conv)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
           
        x = cross_attention_cond(x, speaker_embed)
        x = conv2d_block(x, filters=filters, use_batch_norm=use_batch_norm,
                         dropout=dropout, dropout_type=dropout_type, activation=activation)
    output_mask = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    outputs = keras.layers.Multiply()([output_mask, main_input_copy])
   
    return Model(inputs=[main_input, ref_input], outputs=[outputs])
   
   
model_filename = 'model_weights_final_version_hard_convolution_baseline_LIBRIMIX.weights.h5'
model = custom_unet(
    input_shape=(200, 256, 2),
    use_batch_norm=True,
    num_classes=2,
    filters=32,
    use_dropout_on_upsampling=False,
    num_layers=3,
    use_attention=False,
    upsample_mode='deconv',
    dropout=0.2,
    output_activation='sigmoid')
callbacks = [
    ModelCheckpoint(
        model_filename,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1),
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        min_delta=0.00001,
        restore_best_weights=True,
        verbose=1),
]

def complex_enhancement_loss_pc(y_true, y_pred, gamma=0.5, eps=1e-8):
    # 1. Extract Components
    real_t, imag_t = y_true[..., 0], y_true[..., 1]
    real_p, imag_p = y_pred[..., 0], y_pred[..., 1]
    # 2. Power Compressed Magnitude Loss
    # High gamma (0.5) helps the model see quiet speech parts
    mag_t = tf.sqrt(real_t**2 + imag_t**2 + eps)
    mag_p = tf.sqrt(real_p**2 + imag_p**2 + eps)
   
    mag_loss = tf.reduce_mean(tf.abs(mag_t**gamma - mag_p**gamma))
    # 3. Phase-Aware Complex Loss
    # We compress the complex values themselves to preserve the phase angle
    # while scaling the amplitude
    real_t_c = real_t * (mag_t**(gamma-1))
    imag_t_c = imag_t * (mag_t**(gamma-1))
    real_p_c = real_p * (mag_p**(gamma-1))
    imag_p_c = imag_p * (mag_p**(gamma-1))
   
    complex_loss = tf.reduce_mean(tf.abs(real_t_c - real_p_c) + tf.abs(imag_t_c - imag_p_c))
    # 4. Temporal Consistency Term (The "Flicker" Fix)
    # This penalizes sharp, unnatural changes between adjacent time frames
    # y_shape: (Batch, Time, Freq, Complex)
    delta_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    delta_pred = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    consistency_loss = tf.reduce_mean(tf.abs(delta_true - delta_pred))
    # Total Loss: Balanced for 16kHz Speaker Extraction
    # 0.7 (Complex/Mag) + 0.3 (Temporal Smoothness)
    return (4 * mag_loss + 4 * complex_loss + 2 * consistency_loss)

total_steps  = steps_per_epoch * 100
warmup_steps = steps_per_epoch * 15
initial_lr   = 1e-4
alpha        = 0.05  # final lr fraction

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
    alpha=alpha
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    weight_decay=1e-3,
    clipnorm=1.0
)

model.compile(optimizer=optimizer, loss=complex_enhancement_loss_pc)

history = model.fit(
    train_dataset,            
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks + [LrLogger()]
)
