import tensorflow as tf
import numpy as np

# ==============================================================================
# 1. ROBUST WARMUP COSINE DECAY SCHEDULE
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package="CustomSchedules")
class PerformantWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A truly performant learning rate schedule featuring a linear warmup phase 
    followed by a smooth cosine decay down to a hard floor value (min_lr).
    """
    def __init__(self, initial_lr: float, total_steps: int, warmup_steps: int, min_lr: float = 1e-6):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        initial_lr = tf.cast(self.initial_lr, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)

        # 1. Warmup Phase logic
        def warmup_fn():
            # Linear scaling up to initial_lr
            return initial_lr * (step / tf.maximum(warmup_steps, 1.0))

        # 2. Cosine Decay Phase logic
        def decay_fn():
            decay_steps = tf.maximum(total_steps - warmup_steps, 1.0)
            completed_fraction = (step - warmup_steps) / decay_steps
            completed_fraction = tf.clip_by_value(completed_fraction, 0.0, 1.0)
            
            cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * completed_fraction))
            decayed = (initial_lr - min_lr) * cosine_decay + min_lr
            return decayed

        # Dynamically switch branches based on global step threshold
        return tf.cond(step < warmup_steps, warmup_fn, decay_fn)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr
        }


# ==============================================================================
# 2. MASTER FACTORY WRAPPER
# ==============================================================================
def configure_optimization(
    optimizer_name: str,
    base_lr: float,
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epochs: int = 3,
    weight_decay: float = 1e-4,
    clip_norm: float = 1.0,
    min_lr: float = 1e-6
):
    """
    Factory function returning a compiled performance-tuned optimizer and schedule.
    
    Args:
        optimizer_name: 'adamw' or 'lion'
        base_lr: Initial peak learning rate after warmup
        steps_per_epoch: Total steps/batches per training epoch
        total_epochs: Upper bound epoch limit (e.g., 150)
        warmup_epochs: Duration of initial structural learning rate rise
        weight_decay: Explicit decoupled penalty constraint
        clip_norm: Max global gradient threshold (essential to avoid NaN in RNNs/xLSTMs)
        min_lr: Hard terminal limit for training stability
    """
    optimizer_name = optimizer_name.lower()
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    # Instantiate the performant scheduler
    lr_schedule = PerformantWarmupCosineDecay(
        initial_lr=base_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr
    )

    # Factory Selection
    if optimizer_name == "adamw":
        # Decoupled AdamW with deep learning stability hyper-parameters
        return tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.98,       # 0.98 optimized for Sequence models / Attention variants
            epsilon=1e-8,
            global_clipnorm=clip_norm # Strongly preserves gradient paths safely across architectures
        )

    elif optimizer_name == "lion":
        # Google's Evo-optimizer: 3x less memory overhead, strict vector adjustments
        # CRITICAL ADVICE: If switching to Lion, reduce your base_lr by roughly 3x to 5x!
        return tf.keras.optimizers.Lion(
            learning_rate=lr_schedule,
            weight_decay=weight_decay * 10, # Lion demands an order of magnitude higher weight decay
            beta_1=0.9,
            beta_2=0.99,
            global_clipnorm=clip_norm
        )

    else:
        raise ValueError(f"Unsupported performance optimizer variant target: {optimizer_name}")