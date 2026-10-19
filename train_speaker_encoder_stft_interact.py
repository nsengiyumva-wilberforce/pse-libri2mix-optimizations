# Keras RNN cells set attributes in build(); not in __init__.
# pylint: disable=attribute-defined-outside-init
import csv
import math
import os
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from keras import constraints, initializers, layers, ops, saving
from keras.layers import (
    RNN,
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Layer,
    LayerNormalization,
    MaxPooling2D,
    MultiHeadAttention,
    Multiply,
    Reshape,
    Resizing,
    SpatialDropout2D,
    UpSampling2D,
    add,
    concatenate,
    multiply,
)
from keras.models import Model
from tqdm import tqdm

from libri2mix import AudioToolkit, LibriSpeechDatasetBuilder, WaveformEnhancer
from libri2mix.audio import conv_stft_frames
from libri2mix.conf_unet_tse_32ms import dev_data, train_data
from libri2mix.dataset_tse import make_dataloader
from libri2mix.metrics import (
    load_resample_8k as _load_resample_8k,
    normalize as _normalize,
    pesq_score as _pesq_score,
    sanitize as _sanitize,
    stoi_score as _stoi_score,
)
from libri2mix.trainer_tse import SiSnrTrainer, chunk_frames_match

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*unable to load libtensorflow_io_plugins.so.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*file system plugins are not loaded.*"
)
warnings.filterwarnings("ignore")

tf.config.optimizer.set_jit(False)


def make_distribution_strategy():
    """Use every visible GPU. One GPU, or none, keeps a single replica."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not enable memory growth on {gpu.name}: {exc}")
    if len(gpus) > 1:
        dist_strategy = tf.distribute.MirroredStrategy()
    else:
        dist_strategy = tf.distribute.get_strategy()
    names = ", ".join(gpu.name for gpu in gpus) if gpus else "CPU"
    print(
        f"Distribution: {dist_strategy.num_replicas_in_sync} replica(s) on {names}"
    )
    return dist_strategy, gpus


strategy, visible_gpus = make_distribution_strategy()


# --------------------------------
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


TARGET_SR = 8000
total_length = 4.0
trim_length = 32000

# Clean power-of-two STFT settings for smaller features
n_fft = 256  # 129 bins; Nyquist is dropped so the U-Net sees 128
frame_length = 256  # 32ms window
frame_step = 64  # 16ms stride (50% overlap)

# Four seconds at 8 kHz. The convolution STFT produces 503 frames for this chunk.
CHUNK_SIZE = 32000
STRIDE = CHUNK_SIZE // 2


N_BINS = n_fft // 2 + 1
N_FRAMES = conv_stft_frames(CHUNK_SIZE, frame_length, frame_step)
_MAX_AUX_FRAMES = conv_stft_frames(18 * TARGET_SR, frame_length, frame_step)
MAX_REF_FRAMES = int(math.ceil(_MAX_AUX_FRAMES / 128) * 128)

# Full-utterance reference encoding uses substantially more memory than the
# segmented reference path. Keep this at one to fit the longest padded refs.
PER_REPLICA_BATCH_SIZE = 16
global_batch_size = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync
print(
    f"Per-replica batch {PER_REPLICA_BATCH_SIZE}, global batch {global_batch_size}."
)
EPOCHS = 150
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
def convert_to_spectrogram_multiview(wav_corr, wav_ref, wavclean):
    return _DATASET_BUILDER.convert_to_spectrogram_multiview(
        wav_corr, wav_ref, wavclean
    )


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


def load_libri_speech_triplet_multiview(mix_path, ref_path, tgt_path):
    return _DATASET_BUILDER.load_libri_speech_triplet_multiview(
        mix_path, ref_path, tgt_path
    )


def configure_libri_speech_dataset(
    mixture_files, reference_files, target_files, is_train=True
):
    return _DATASET_BUILDER.configure_dataset(
        mixture_files, reference_files, target_files, is_train=is_train
    )


print(
    f"Mixture frames {N_FRAMES}, frequency bins {N_BINS}, "
    f"enrollment frames {MAX_REF_FRAMES}."
)


# ============ sLSTM implementation ============
class sLSTMCell(Layer):
    def __init__(self, units, forget_gate_type="sigmoid", **kwargs):
        """
        sLSTM cell with proper state handling.
        """
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)
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
        _h_prev, C_flat_prev, n_prev, m_prev = states
        batch_dim = tf.shape(inputs)[0]

        # Reshape C matrix
        C_prev = tf.reshape(C_flat_prev, [batch_dim, self.units, self.units])

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
        C_flat_t = tf.reshape(C_t, [batch_dim, self.units * self.units])

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


@saving.register_keras_serializable()
class LearnableScale(layers.Layer):
    def __init__(self, initial_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value
        self.scale = self.add_weight(
            name="scale_factor",
            shape=(),
            initializer=initializers.Constant(initial_value),
            trainable=True,
            constraint=constraints.NonNeg(),
        )

    def call(self, x):
        # Positive gain on a linear mask. tanh saturated the bins that have to
        # move toward 0 or 1, and one scalar could not undo that.
        return (self.scale + 1e-6) * x

    def get_config(self):
        config = super().get_config()
        config.update({"initial_value": self.initial_value})
        return config


@saving.register_keras_serializable()
class PowerLawComplexSpec(layers.Layer):
    """Compress or expand complex magnitude and keep phase.

    factor=0.5 takes the square root of the magnitude. factor=2 undoes it.
    The mask is applied in the compressed domain, where the targets are smaller.
    """

    def __init__(self, factor=0.5, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.factor = float(factor)
        self.eps = float(eps)

    def call(self, spec):
        real = spec[..., 0]
        imag = spec[..., 1]
        magnitude = ops.sqrt(ops.square(real) + ops.square(imag) + self.eps)
        scale = ops.power(magnitude, self.factor) / magnitude
        return ops.stack([real * scale, imag * scale], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor, "eps": self.eps})
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

        if block_type == "sLSTM":
            cell = sLSTMCell(hidden_dim, forget_gate_type="sigmoid")
        elif block_type == "mLSTM":
            cell = mLSTMCell(hidden_dim, forget_gate_type="sigmoid")
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        # Unique name using the prefix and index
        x = RNN(cell, return_sequences=True, name=f"{prefix}_{block_type}_{i}")(x)

        if residual.shape[-1] == x.shape[-1]:
            # Names for operations like Add and LayerNorm are usually auto-generated,
            # but you can name them too if you want total safety:
            x = Add(name=f"{prefix}_add_{i}")([residual, x])
            x = LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln_{i}")(x)

        if i < num_layers - 1:
            x = Dropout(0.2, name=f"{prefix}_do_{i}")(x)

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


def upsample_simple(_filters, _kernel_size, strides, _padding):
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

    # Small weights so the speaker modulates from the first step. Zero weights
    # leave the decoder ignoring the enrollment until those layers wake up.
    speaker_init = initializers.RandomNormal(stddev=0.02)
    gamma = Dense(C, kernel_initializer=speaker_init)(speaker_embedding)
    beta = Dense(C, kernel_initializer=speaker_init)(speaker_embedding)

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


@saving.register_keras_serializable()
class ASFFFusion(Layer):
    """Two-input adaptively spatial feature fusion.

    Both maps already share height, width, and channels. Each is projected with a
    1x1 convolution, a second 1x1 scores the pair, and a per-bin softmax mixes
    them. The score kernel starts at zero. With prefer_left False the bias is
    zero too, so the mix starts equal. With prefer_left True the bias starts at
    [0.5, -0.5], about 73% on the main map and 27% on the speaker map.
    """

    def __init__(self, compress=8, prefer_left=False, **kwargs):
        super().__init__(**kwargs)
        self.compress = int(compress)
        self.prefer_left = bool(prefer_left)
        # Created here so Keras tracks them. build() is too late for the functional model.
        self.compress_left = Conv2D(
            self.compress,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.compress_right = Conv2D(
            self.compress,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        bias = initializers.Constant([0.5, -0.5]) if self.prefer_left else "zeros"
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


@saving.register_keras_serializable()
class STFTFrameSimilarity(Layer):
    """Create the enrollment-to-mixture STFT similarity cue used by train.py."""

    def call(self, inputs):
        mix, enroll, mask = inputs  # (B,T,F,2), (B,R,F,2), (B,R)
        mix_f = tf.transpose(mix, [0, 2, 3, 1])
        enr_f = tf.transpose(enroll, [0, 2, 3, 1])
        valid = tf.cast(mask, mix.dtype)
        guided_channels = []
        for channel in range(2):
            enrollment = enr_f[:, :, channel, :]
            mixture = mix_f[:, :, channel, :]
            logits = tf.matmul(enrollment, mixture, transpose_a=True)
            logits += (valid[:, :, None] - 1.0) * 1e9
            weights = tf.nn.softmax(logits, axis=1)
            guided_channels.append(tf.matmul(enrollment, weights))
        guided = tf.transpose(tf.stack(guided_channels, axis=-1), [0, 2, 1, 3])
        valid_frames = valid[:, :, None, None]
        denom = tf.reduce_sum(valid_frames, axis=1, keepdims=True) + 1e-8
        mean = tf.reduce_sum(enroll * valid_frames, axis=1, keepdims=True) / denom
        mean = tf.tile(mean, [1, tf.shape(mix)[1], 1, 1])
        return guided, mean

    def compute_output_shape(self, input_shape):
        mix_shape, _, _ = input_shape
        return mix_shape, mix_shape


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


def tf_alternating_block(
    x, filters, activation="relu", use_bn=True, name_prefix="tfb", dropout=0.15
):
    del dropout  # reserved for optional regularization; kept for call-site compatibility
    original_x = x
    # ---- Frequency branch (1 x 3) for the first branch, we get time convolutions over frequency
    f_branch = Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_fconv",
    )(x)
    if use_bn:
        f_branch = BatchNormalization(name=f"{name_prefix}_fbn")(f_branch)
    f_branch = Activation(activation)(f_branch)

    # ---- Time branch (3 x 1) ----
    t_branch = Conv2D(
        filters,
        (5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_tconv",
    )(f_branch)
    if use_bn:
        t_branch = BatchNormalization(name=f"{name_prefix}_tbn")(t_branch)
    t_branch = Activation(activation)(t_branch)

    # ---- Merge ----
    branch_1 = Concatenate(name=f"{name_prefix}_concat")([f_branch, t_branch])

    # sencond branch, we get frequency convolutions over time
    t_branch_2 = Conv2D(
        filters,
        (5, 5),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_tconv2",
    )(original_x)
    if use_bn:
        t_branch_2 = BatchNormalization(name=f"{name_prefix}_tbn2")(t_branch_2)
    t_branch_2 = Activation(activation)(t_branch_2)

    f_branch_2 = Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_fconv2",
    )(t_branch_2)
    if use_bn:
        f_branch_2 = BatchNormalization(name=f"{name_prefix}_fbn2")(f_branch_2)
    f_branch_2 = Activation(activation)(f_branch_2)

    # Merge again
    branch_2 = Concatenate(name=f"{name_prefix}_concat2")([t_branch_2, f_branch_2])
    # merge the two branches  for x and original_x
    x = Concatenate(name=f"{name_prefix}_final_concat")([branch_1, branch_2])

    # ---- Separable TF mixing ----
    x = DepthwiseConv2D(
        (3, 3),
        padding="same",
        depthwise_initializer="he_normal",
        name=f"{name_prefix}_dw",
    )(x)
    x = eca_block(x, name=f"enc_eca_{name_prefix}_ec")

    x = Conv2D(
        filters,
        1,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_pw",
    )(x)

    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_pw_bn")(x)

    x = Activation(activation)(x)

    # ---- Joint TF modeling ----
    x = Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_joint",
    )(x)
    if use_bn:
        x = BatchNormalization(name=f"{name_prefix}_jbn")(x)
    x = Activation(activation)(x)
    # x = Dropout(dropout)(x)

    return x


def make_shared_tfb_convolutions(filters, name_prefix):
    """Create convolution layers that can be reused by mixture and reference paths."""
    return {
        "fconv": Conv2D(
            filters,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_shared_fconv",
        ),
        "tconv": Conv2D(
            filters,
            (5, 5),
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_shared_tconv",
        ),
        "tconv2": Conv2D(
            filters,
            (5, 5),
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_shared_tconv2",
        ),
        "fconv2": Conv2D(
            filters,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_shared_fconv2",
        ),
        "dw": DepthwiseConv2D(
            (3, 3),
            padding="same",
            depthwise_initializer="he_normal",
            name=f"{name_prefix}_shared_dw",
        ),
        "pw": Conv2D(
            filters,
            1,
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_shared_pw",
        ),
        "joint": Conv2D(
            filters,
            (3, 3),
            padding="same",
            groups=4,
            dilation_rate=(2, 2),
            kernel_initializer="he_normal",
            name=f"{name_prefix}_shared_joint",
        ),
    }


def shared_tf_alternating_block(
    x, convs, _filters, activation, use_bn, name_prefix, branch, dropout
):
    """Apply shared convolution weights with branch-specific normalization."""
    del _filters
    original_x = x
    f_branch = convs["fconv"](x)
    if use_bn:
        f_branch = BatchNormalization(name=f"{branch}_{name_prefix}_fbn")(f_branch)
    f_branch = Activation(activation, name=f"{branch}_{name_prefix}_fact")(f_branch)

    t_branch = convs["tconv"](f_branch)
    if use_bn:
        t_branch = BatchNormalization(name=f"{branch}_{name_prefix}_tbn")(t_branch)
    t_branch = Activation(activation, name=f"{branch}_{name_prefix}_tact")(t_branch)
    branch_1 = Concatenate(name=f"{branch}_{name_prefix}_concat")([f_branch, t_branch])

    t_branch_2 = convs["tconv2"](original_x)
    if use_bn:
        t_branch_2 = BatchNormalization(name=f"{branch}_{name_prefix}_tbn2")(t_branch_2)
    t_branch_2 = Activation(activation, name=f"{branch}_{name_prefix}_tact2")(
        t_branch_2
    )

    f_branch_2 = convs["fconv2"](t_branch_2)
    if use_bn:
        f_branch_2 = BatchNormalization(name=f"{branch}_{name_prefix}_fbn2")(f_branch_2)
    f_branch_2 = Activation(activation, name=f"{branch}_{name_prefix}_fact2")(
        f_branch_2
    )
    branch_2 = Concatenate(name=f"{branch}_{name_prefix}_concat2")(
        [t_branch_2, f_branch_2]
    )
    x = Concatenate(name=f"{branch}_{name_prefix}_final_concat")([branch_1, branch_2])

    x = convs["dw"](x)
    x = eca_block(x, name=f"{branch}_{name_prefix}_eca")
    x = convs["pw"](x)
    if use_bn:
        x = BatchNormalization(name=f"{branch}_{name_prefix}_pw_bn")(x)
    x = Activation(activation, name=f"{branch}_{name_prefix}_pw_act")(x)

    x = convs["joint"](x)
    if use_bn:
        x = BatchNormalization(name=f"{branch}_{name_prefix}_jbn")(x)
    x = Activation(activation, name=f"{branch}_{name_prefix}_jact")(x)
    return Dropout(dropout, name=f"{branch}_{name_prefix}_dropout")(x)


def broadcast_speaker(embed, target_tensor):
    target_shape = target_tensor.shape  # (Batch, T, F, C)
    s = layers.Dense(target_shape[-1])(embed)
    s = layers.Reshape((1, 1, target_shape[-1]))(s)
    # Tile/Broadcast to (T, F, C)
    return layers.UpSampling2D(size=(target_shape[1], target_shape[2]))(s)


def attentive_pooling(x, mask=None, name="att_pool"):
    """
    Self-attention pooling (Attentive Statistics Pooling lite).
    Input shape: (Batch, Time, Channels)
    Output shape: (Batch, Channels)

    ``mask`` is 1 on valid enrollment frames and 0 on padding. Padded frames are
    removed from the softmax so silence does not take the attention mass.
    """
    att_weights = layers.Dense(1, activation=None, name=f"{name}_dense")(x)
    if mask is not None:
        mask_col = layers.Reshape((x.shape[1], 1), name=f"{name}_mask")(mask)
        mask_col = ops.cast(mask_col, att_weights.dtype)
        att_weights = att_weights + (1.0 - mask_col) * ops.cast(
            -1.0e4, att_weights.dtype
        )

    att_weights = layers.Softmax(axis=1, name=f"{name}_softmax")(att_weights)

    context = layers.Lambda(
        lambda inputs: ops.sum(inputs[0] * inputs[1], axis=1),
        name=f"{name}_weighted_sum",
    )([x, att_weights])

    return context


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


def _masked_unit_scale(x, mask, eps=1e-5):
    """Divide each example by its standard deviation over valid frames only."""
    mask = ops.cast(mask, x.dtype)
    counts = ops.sum(mask, axis=[1, 2, 3], keepdims=True)
    counts = counts * ops.cast(ops.shape(x)[2] * ops.shape(x)[3], x.dtype)
    counts = ops.maximum(counts, ops.cast(1.0, x.dtype))
    mean = ops.sum(x, axis=[1, 2, 3], keepdims=True) / counts
    variance = (
        ops.sum(ops.square((x - mean) * mask), axis=[1, 2, 3], keepdims=True) / counts
    )
    return x / (ops.sqrt(variance) + eps)


def masked_attentive_pooling(features, mask, name="speaker_pool"):
    """Learn a masked, utterance-level summary of reference frame features."""
    scores = Dense(128, activation="tanh", name=f"{name}_hidden")(features)
    scores = Dense(1, name=f"{name}_score")(scores)
    valid = ops.expand_dims(ops.cast(mask > 0, "bool"), axis=-1)
    scores = ops.where(valid, scores, ops.cast(-1e4, scores.dtype))
    weights = ops.softmax(scores, axis=1) * ops.cast(valid, scores.dtype)
    weights = weights / ops.maximum(
        ops.sum(weights, axis=1, keepdims=True), ops.cast(1e-6, scores.dtype)
    )
    return ops.sum(features * weights, axis=1)


def cross_attention_cond(
    x, reference_sequence, reference_mask, num_heads=4, key_dim=64, name="bn_mask"
):
    """Inject reference context into mixture features with cross-attention.

    Args:
        x: Mixture sequence (B, T_mix, D_mix).
        reference_sequence: Frame sequence from the enrollment encoder
            (B, T_ref, D_ref), before attentive speaker pooling.
        reference_mask: Valid-frame mask (B, T_ref), with 1 for valid frames.

    Returns:
        Mixture sequence with a scaled residual from reference cross-attention.

    Mixture features are used as cross-attention queries, so the injected
    speaker context stays aligned with the mixture even when enrollment and
    mixture lengths differ.
    """
    model_dim = 256
    x_residual = Dense(model_dim, name=f"{name}_mix_proj")(x)
    ref = Dense(model_dim, name=f"{name}_ref_proj")(reference_sequence)

    # The reference-side self-attention sublayer from the SEF-Net decoder.
    ref_valid = ops.cast(reference_mask > 0, "bool")
    ref_self_mask = ops.logical_and(
        ops.expand_dims(ref_valid, axis=1),
        ops.expand_dims(ref_valid, axis=2),
    )
    ref_self = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_ref_self_attn"
    )(query=ref, value=ref, key=ref, attention_mask=ref_self_mask)
    ref = LayerNormalization(name=f"{name}_ref_self_norm")(
        Add(name=f"{name}_ref_self_res")([ref, ref_self])
    )

    # Cross-attention lets every mixture position retrieve relevant enrollment
    # frames. Mask padded enrollment frames from the attention keys/values.
    cross_mask = ops.expand_dims(ref_valid, axis=1)
    cross = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_cross_attn"
    )(query=x_residual, value=ref, key=ref, attention_mask=cross_mask)

    # Preserve the mixture path and give the separator a direct, learnable
    # residual from the reference-conditioned attention output.
    speaker_context = Dense(
        int(x.shape[-1]),
        kernel_initializer=initializers.RandomNormal(stddev=0.02),
        name=f"{name}_context_projection",
    )(cross)
    speaker_context = LearnableScale(initial_value=0.1, name=f"{name}_context_scale")(
        speaker_context
    )
    return Add(name=f"{name}_residual_fusion")([x, speaker_context])


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


def _drop_nyquist_freq_bin(x):
    return x[:, :, :-1, :]


def custom_unet(
    input_shape,
    ref_frames,
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
    # Legacy U-Net kwargs kept so older build_enhancement_model() calls stay valid.
    del num_classes, dropout_type, num_layers, output_activation
    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    main_input = Input(input_shape, name="noisy_main")  # (T, F, C)
    ref_input = Input((ref_frames, input_shape[-2], input_shape[-1]), name="noisy_ref")
    ref_mask_input = Input((ref_frames,), name="ref_mask")
    # The convolution STFT has one extra Nyquist bin. The U-Net sees the rest.
    main_body = layers.Lambda(_drop_nyquist_freq_bin, name="drop_nyquist_main")(
        main_input
    )
    ref_body = layers.Lambda(_drop_nyquist_freq_bin, name="drop_nyquist_ref")(
        ref_input
    )
    ref_mask = layers.Reshape((ref_frames, 1, 1))(ref_mask_input)
    ref_body = layers.Multiply()([ref_body, ref_mask])
    # Square-root the magnitude before the mask. The residual below is added
    # in this compressed domain and expanded again after the multiply.
    main_body = PowerLawComplexSpec(factor=0.5, name="compress_main")(main_body)
    ref_body = PowerLawComplexSpec(factor=0.5, name="compress_ref")(ref_body)
    main_input_copy = main_body

    # One scale per example. A single std over the whole batch moves with the
    # amount of enrollment padding and rescales every speaker together.
    x = main_body / (ops.std(main_body, axis=[1, 2, 3], keepdims=True) + 1e-5)
    # The STFT interaction cue is computed from the compressed, unnormalized
    # spectra, while the separator keeps the speaker encoder's normalized input.
    guided, enroll_mean = STFTFrameSimilarity(name="stft_sim")(
        [main_body, ref_body, ref_mask_input]
    )
    guided = ASFFFusion(name="stft_asff")([guided, enroll_mean])
    guided = Conv2D(2, 1, padding="same", name="stft_guided_projection")(guided)
    # Keep every skip connection aligned through the six down/up-sampling
    # stages, then crop the decoder result back to the original spectrogram.
    filter_list = [16, 32, 32, 32, 64, 128]
    unet_multiple = 2 ** len(filter_list)
    spec_time, spec_freq = int(main_body.shape[1]), int(main_body.shape[2])
    pad_time = (unet_multiple - spec_time % unet_multiple) % unet_multiple
    pad_freq = (unet_multiple - spec_freq % unet_multiple) % unet_multiple
    if pad_time or pad_freq:
        x = layers.ZeroPadding2D(
            padding=((0, pad_time), (0, pad_freq)), name="unet_pad"
        )(x)
        guided = layers.ZeroPadding2D(
            padding=((0, pad_time), (0, pad_freq)), name="stft_guided_pad"
        )(guided)
    x = layers.Add(name="stft_early_fusion")([x, guided])
    ref_x = _masked_unit_scale(ref_body, ref_mask)

    # MAIN ENCODER
    down_layers = []
    num_shared_stages = 4
    shared_tfb_convs = [
        make_shared_tfb_convolutions(
            stage_filters, name_prefix=f"shared_tfb_{stage}_{stage_filters}"
        )
        for stage, stage_filters in enumerate(filter_list[:num_shared_stages])
    ]
    for stage, l in enumerate(filter_list):
        if stage < num_shared_stages:
            x = shared_tf_alternating_block(
                x,
                shared_tfb_convs[stage],
                l,
                activation,
                use_bn=True,
                name_prefix=f"shared_tfb_{stage}_{l}",
                branch="mix",
                dropout=0.15,
            )
        else:
            x = tf_alternating_block(
                x,
                l,
                activation,
                use_bn=True,
                name_prefix=f"tfb_{stage}_{l}",
                dropout=0.15,
            )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer

    # Reuse the first four mixture stages for enrollment. The convolution
    # kernels are shared; batch-normalization and ECA parameters stay branch-
    # specific so each path can adapt to its own feature distribution.
    ref_enc = ref_x
    for stage, fil in enumerate(filter_list[:num_shared_stages]):
        ref_enc = shared_tf_alternating_block(
            ref_enc,
            shared_tfb_convs[stage],
            fil,
            activation,
            use_bn=True,
            name_prefix=f"shared_tfb_{stage}_{fil}",
            branch="ref",
            dropout=0.0,
        )
        ref_enc = MaxPooling2D((2, 2), name=f"ref_shared_pool_{stage}")(ref_enc)
    guided_bn = guided
    for stage in range(len(filter_list)):
        guided_bn = layers.AveragePooling2D((2, 2), name=f"stft_guided_pool_{stage}")(
            guided_bn
        )
    # BOTTLENECK
    T_small, F_small, C_small = x.shape[1], x.shape[2], x.shape[3]  # (24, 16, 256)
    x_seq = Reshape((T_small, F_small * C_small), name="main_bottleneck_sequence")(x)
    ref_time = ref_enc.shape[1]
    ref_freq = ref_enc.shape[2]
    ref_channels = ref_enc.shape[3]
    ref_seq = Reshape(
        (ref_time, ref_freq * ref_channels), name="reference_bottleneck_sequence"
    )(ref_enc)
    # Pool the binary validity mask by the same temporal stride as the
    # auxiliary encoder. A frame is valid when any contributing input frame is.
    ref_attention_mask = ref_mask_input
    for p in range(num_shared_stages):
        ref_attention_mask = layers.MaxPooling1D(
            pool_size=2, strides=2, padding="valid", name=f"ref_mask_pool_{p}"
        )(ops.expand_dims(ref_attention_mask, axis=-1))
        ref_attention_mask = ops.squeeze(ref_attention_mask, axis=-1)

    # A learned, padding-aware summary conditions every decoder stage.
    speaker_embed = masked_attentive_pooling(
        ref_seq, ref_attention_mask, name="speaker_attentive_pool"
    )

    x_seq = cross_attention_cond(
        x_seq,
        reference_sequence=ref_seq,
        reference_mask=ref_attention_mask,
        num_heads=4,
        key_dim=64,
        name="bn_mask",
    )

    x_seq = add_xlstm_block(
        x_seq, hidden_dim=256, num_layers=1, prefix="main_bottleneck"
    )

    x_seq = Dense(F_small * C_small, name="main_bottleneck_restore_projection")(x_seq)
    x = Reshape((T_small, F_small, C_small), name="main_bottleneck_restore_map")(x_seq)
    guided_bn = Conv2D(C_small, 1, padding="same", name="stft_bn_projection")(guided_bn)
    x = ASFFFusion(name="stft_bn_fusion")([x, guided_bn])

    # DECODER
    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
    for stage, conv in enumerate(reversed(down_layers)):
        filters = conv.shape[-1]
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        guided_here = Resizing(
            int(x.shape[1]),
            int(x.shape[2]),
            interpolation="bilinear",
            name=f"stft_guided_up_{stage}",
        )(guided)
        guided_here = Conv2D(
            int(x.shape[-1]), 1, padding="same", name=f"stft_decoder_projection_{stage}"
        )(guided_here)
        x = ASFFFusion(name=f"stft_decoder_fusion_{stage}")([x, guided_here])
        # Apply FiLM conditioning after upsampling but before concatenation
        x = film(x, speaker_embed)
        # apply film in skip connections
        conv = film(conv, speaker_embed)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = tf_alternating_block(
            x,
            filters,
            activation,
            use_bn=use_batch_norm,
            name_prefix=f"up_conv_{stage}_{filters}",
        )

    # spk_res_final = broadcast_speaker(speaker_embed, x)
    # x = ifi_block(x, spk_res_final, channels=x.shape[-1], name="final_refinement")
    if pad_time or pad_freq:
        x = layers.Cropping2D(
            cropping=((0, pad_time), (0, pad_freq)), name="unet_crop"
        )(x)
    # --- Split noisy input into real / imag ---
    input_r = main_input_copy[..., 0:1]
    input_i = main_input_copy[..., 1:2]

    # --- Predict complex mask ---
    # he_normal keeps a gradient into the U-Net. A zero kernel would make this
    # residual exactly the mixture and would scale every upstream gradient by
    # that kernel. The scale stays linear so bins are not squashed by tanh.
    mask_r = Conv2D(
        1,
        (1, 1),
        activation=None,
        kernel_initializer="he_normal",
        name="mask_real",
    )(x)
    mask_i = Conv2D(
        1,
        (1, 1),
        activation=None,
        kernel_initializer="he_normal",
        name="mask_imag",
    )(x)

    # --- Apply Learnable Scaling ---
    mask_r = LearnableScale(initial_value=1.0, name="scale_r")(mask_r)
    mask_i = LearnableScale(initial_value=1.0, name="scale_i")(mask_i)

    # --- Complex multiplication ---
    out_r = layers.Subtract()(
        [layers.Multiply()([mask_r, input_r]), layers.Multiply()([mask_i, input_i])]
    )

    out_i = layers.Add()(
        [layers.Multiply()([mask_r, input_i]), layers.Multiply()([mask_i, input_r])]
    )

    # Correction on top of the compressed mixture, then undo the square root.
    out_r = layers.Add()([input_r, out_r])
    out_i = layers.Add()([input_i, out_i])

    # --- Merge back to 2-channel ---
    outputs = Concatenate(axis=-1)([out_r, out_i])
    outputs = PowerLawComplexSpec(factor=2.0, name="decompress_out")(outputs)
    nyquist = ops.zeros_like(outputs[:, :, :1, :])
    outputs = ops.concatenate([outputs, nyquist], axis=2)

    return Model(inputs=[main_input, ref_input, ref_mask_input], outputs=[outputs])


model_filename = "model_weights_final_version_hard_convolution_baseline_LIBRIMIX_shared_encoder_stft.keras"


def build_enhancement_model():
    return custom_unet(
        input_shape=(N_FRAMES, N_BINS, 2),
        ref_frames=MAX_REF_FRAMES,
        use_batch_norm=True,
        num_classes=2,
        filters=16,
        use_dropout_on_upsampling=False,
        num_layers=4,
        use_attention=False,
        upsample_mode="deconv",
        dropout=0.5,
        output_activation="elu",
    )


with strategy.scope():
    enhancement_model = build_enhancement_model()


enhancement_model.summary()


if not chunk_frames_match(CHUNK_SIZE, frame_length, frame_step, N_FRAMES):
    raise SystemExit(
        f"Chunk STFT frames {conv_stft_frames(CHUNK_SIZE, frame_length, frame_step)} "
        f"do not match the model input {N_FRAMES}."
    )
with strategy.scope():
    trainer = SiSnrTrainer(
        enhancement_model,
        _AUDIO_TOOLKIT,
        MAX_REF_FRAMES,
        model_filename,
        logging_period=200,
        strategy=strategy,
    )
trainer.run(
    make_dataloader(
        train=True,
        data_kwargs=train_data,
        chunk_size=CHUNK_SIZE,
        batch_size=global_batch_size,
    ),
    make_dataloader(
        train=False,
        data_kwargs=dev_data,
        chunk_size=CHUNK_SIZE,
        batch_size=global_batch_size,
    ),
    num_epochs=EPOCHS,
)
enhancement_model.load_weights(model_filename)
if strategy.num_replicas_in_sync > 1:
    # Evaluation calls the model one utterance at a time. A mirrored model
    # rejects a batch that is not divisible by the replica count.
    print("Copying trained weights onto one device for evaluation.")
    device = "/GPU:0" if visible_gpus else "/CPU:0"
    with tf.device(device):
        single_device_model = build_enhancement_model()
    single_device_model.set_weights(enhancement_model.get_weights())
    enhancement_model = single_device_model

# load the keras model for inference
# enhancement_model = tf.keras.models.load_model(
#     "model_weights_final_version_hard_convolution_baseline_LIBRIMIX.keras",
#     custom_objects={
#         "sLSTMCell": sLSTMCell,
#         "mLSTMCell": mLSTMCell,
#         "complex_enhancement_loss_pc": complex_enhancement_loss_pc,
#     }
# )
enhancement_model.trainable = False
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


def enhance_audio_consistent(noisy_wav, ref_wav, inference_model, _K=4, overlap=0.5):
    # This model consumes a padded full-reference spectrogram and its mask.
    return _WAVEFORM_ENHANCER.enhance_audio_full_utterance(
        noisy_wav,
        ref_wav,
        inference_model,
        overlap=overlap,
        max_ref_frames=MAX_REF_FRAMES,
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
        enhanced = enhance_audio_consistent(noisy, ref, enhancement_model)

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
    print(f"SI-SDR (mix): {np.mean(arr[:, 1].astype(float)):.2f}")
    print(f"SI-SDR (enh): {np.mean(arr[:, 2].astype(float)):.2f}")
    print(f"SI-SDRi:      {np.mean(arr[:, 7].astype(float)):.2f}")

    print(f"PESQ (mix):   {np.nanmean(arr[:, 3].astype(float)):.3f}")
    print(f"PESQ (enh):   {np.nanmean(arr[:, 4].astype(float)):.3f}")
    print(f"PESQi:        {np.nanmean(arr[:, 8].astype(float)):.3f}")

    print(f"STOI (mix):   {np.nanmean(arr[:, 5].astype(float)):.3f}")
    print(f"STOI (enh):   {np.nanmean(arr[:, 6].astype(float)):.3f}")
    print(f"STOIi:        {np.nanmean(arr[:, 9].astype(float)):.3f}")

    # -------- Save CSV --------
    with open("evaluation_results_full.csv", "w", newline="", encoding="utf-8") as f:
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
