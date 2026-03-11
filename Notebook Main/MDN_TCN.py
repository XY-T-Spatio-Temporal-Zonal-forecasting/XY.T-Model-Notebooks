

import os
import math
import random
from typing import Tuple, List, Dict  # Only for type hints, not for instantiation
import copy
import time
import psutil
import inspect
import gc

import h5py
import joblib
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from TCN import TCN


try:
    # pylint: disable=E0611,E0401
    from tensorflow.keras.utils import register_keras_serializable
  # For recent Keras
except ImportError:
    # pylint: disable=E0611,E0401
    from tensorflow.keras.utils import register_keras_serializable  # For older versions

# pylint: disable=E0611,E0401
from tensorflow.keras import backend as K, Model, Input, optimizers
# pylint: disable=E0611,E0401
from tensorflow.keras import layers
# pylint: disable=E0611,E0401
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
# pylint: disable=E0611,E0401
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization

import tensorflow as tf

MAX_LR = 1e-3                # Peak learning rate
MIN_LR = 1e-5                  # Final learning rate

# ═══════════════════════════════════════════════════════════════════════════════
# MDN-SPECIFIC CONFIGURATION (only used when MODEL_TYPE == "MDN_TCN")
# ═══════════════════════════════════════════════════════════════════════════════
MDN_NUM_MIXTURES = 10     # Number of Gaussian components (K)
MDN_SIGMA_MIN = 1e-4      # Minimum sigma to prevent collapse
MDN_LEARNING_RATE = 5e-5  # Lower LR for MDN stability


@register_keras_serializable(package='MDN')
class MDNLayer(tf.keras.layers.Layer):
    """
    Mixture Density Network output layer.
    
    Converts encoder output into GMM parameters for probabilistic prediction.
    
    Args:
        num_mixtures: Number of Gaussian components (K)
        output_dim: Dimension of output space (2 for x, y coordinates)
        sigma_min: Minimum std deviation to prevent collapse
    
    Outputs:
        pi: (batch, K) mixture weights summing to 1
        mu: (batch, K, output_dim) means per component
        sigma: (batch, K, output_dim) std deviations (positive)
    """
    
    def __init__(self, num_mixtures=5, output_dim=2, sigma_min=1e-4, **kwargs):
        super(MDNLayer, self).__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        self.sigma_min = sigma_min
        
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        
        # π head: outputs K logits → softmax for mixture weights
        self.pi_dense = tf.keras.layers.Dense(
            self.num_mixtures,
            kernel_initializer='glorot_uniform',
            name='mdn_pi_logits'
        )
        
        # μ head: outputs K * output_dim values → reshape to (K, output_dim)
        self.mu_dense = tf.keras.layers.Dense(
            self.num_mixtures * self.output_dim,
            kernel_initializer='glorot_uniform',
            name='mdn_mu_raw'
        )
        
        # σ head: outputs K * output_dim values → softplus for positive values
        self.sigma_dense = tf.keras.layers.Dense(
            self.num_mixtures * self.output_dim,
            kernel_initializer='glorot_uniform',
            name='mdn_sigma_raw'
        )
        
        super(MDNLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Forward pass through MDN layer.
        
        Args:
            inputs: (batch, hidden_dim) encoded features from TCN
            
        Returns:
            Tuple of (pi, mu, sigma)
        """
        batch_size = tf.shape(inputs)[0]
        
        # ─────────────────────────────────────────────────────────────────────
        # Mixture weights π: softmax ensures sum = 1
        # ─────────────────────────────────────────────────────────────────────
        pi_logits = self.pi_dense(inputs)  # (B, K)
        pi = tf.nn.softmax(pi_logits, axis=-1)  # (B, K), sum = 1
        
        # ─────────────────────────────────────────────────────────────────────
        # Means μ: unconstrained, reshaped to (B, K, output_dim)
        # ─────────────────────────────────────────────────────────────────────
        mu_raw = self.mu_dense(inputs)  # (B, K * output_dim)
        mu = tf.reshape(mu_raw, (batch_size, self.num_mixtures, self.output_dim))
        
        # ─────────────────────────────────────────────────────────────────────
        # Std deviations σ: softplus ensures positive, add minimum
        # ─────────────────────────────────────────────────────────────────────
        sigma_raw = self.sigma_dense(inputs)  # (B, K * output_dim)
        sigma = tf.nn.softplus(sigma_raw) + self.sigma_min  # Always positive
        sigma = tf.reshape(sigma, (batch_size, self.num_mixtures, self.output_dim))
        # mu_flat = tf.reshape(mu, (batch_size, self.num_mixtures * self.output_dim))
        # sigma_flat = tf.reshape(sigma, (batch_size, self.num_mixtures * self.output_dim))
        
        # Concatenate into a single tensor for Keras compatibility
        # Shape: (B, K + (K * output_dim) + (K * output_dim))
        # For K=5, dim=2, this is 5 + 10 + 10 = 25
        return pi, mu, sigma
        # return tf.concat([pi, mu_flat, sigma_flat], axis=-1)
    
    def get_config(self):
        config = super(MDNLayer, self).get_config()
        config.update({
            'num_mixtures': self.num_mixtures,
            'output_dim': self.output_dim,
            'sigma_min': self.sigma_min
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


print("✅ MDNLayer class defined")
print(f"   • Outputs: π (B, K), μ (B, K, 2), σ (B, K, 2)")
print(f"   • Uses softmax for weights, softplus for std devs")


def mdn_loss_fn(y_true, pi, mu, sigma):
    """
    Compute Negative Log Likelihood for Gaussian Mixture Model.
    
    This is the core MDN loss function that measures how well the 
    predicted distribution matches the true coordinates.
    
    Args:
        y_true: (B, 2) actual (x, y) coordinates
        pi: (B, K) mixture weights (sum to 1)
        mu: (B, K, 2) mixture means
        sigma: (B, K, 2) mixture std deviations
        
    Returns:
        Scalar loss (mean NLL over batch)
        
    Mathematical Details:
        log N(y|μ,σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((y-μ)/σ)²
        
        We use log-sum-exp for numerical stability:
        log(Σ exp(xᵢ)) = max(x) + log(Σ exp(xᵢ - max(x)))
    """
    # Ensure correct dtypes
    y_true = tf.cast(y_true, tf.float32)
    pi = tf.cast(pi, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    
    # Expand y_true for broadcasting: (B, 2) → (B, 1, 2)
    y_true_expanded = tf.expand_dims(y_true, axis=1)  # (B, 1, 2)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Compute log probability of Gaussian for each component
    # log N(y|μ,σ) = -0.5*log(2π) - log(σ) - 0.5*((y-μ)/σ)²
    # ─────────────────────────────────────────────────────────────────────────
    
    # Variance
    var = tf.square(sigma)  # (B, K, 2)
    
    # Log probability (per dimension)
    log_2pi = tf.constant(np.log(2 * np.pi), dtype=tf.float32)
    
    log_prob_per_dim = (
        -0.5 * log_2pi
        - tf.math.log(sigma + 1e-10)  # Add epsilon for stability
        - 0.5 * tf.square(y_true_expanded - mu) / (var + 1e-10)
    )  # (B, K, 2)
    
    # Sum over dimensions (x, y) to get joint log probability
    log_prob_gaussian = tf.reduce_sum(log_prob_per_dim, axis=-1)  # (B, K)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Add log mixture weights
    # ─────────────────────────────────────────────────────────────────────────
    log_pi = tf.math.log(pi + 1e-10)  # (B, K)
    log_prob_weighted = log_pi + log_prob_gaussian  # (B, K)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Log-sum-exp for numerical stability
    # ─────────────────────────────────────────────────────────────────────────
    log_likelihood = tf.reduce_logsumexp(log_prob_weighted, axis=-1)  # (B,)
    
    # Negative log likelihood
    nll = -log_likelihood
    
    # Return mean over batch
    return tf.reduce_mean(nll)
    # return nll  # Return per-sample loss for Keras compatibility


class MDNLoss(tf.keras.losses.Loss):
    """
    Keras Loss class wrapper for MDN NLL loss.
    
    This allows using model.compile() with MDN models.
    Note: Requires model to output concatenated [pi, mu, sigma] tensor.
    """
    
    def __init__(self, num_mixtures=5, output_dim=2, name='mdn_loss', **kwargs):
        super(MDNLoss, self).__init__(name=name, **kwargs)
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        
    def call(self, y_true, y_pred):
        """
        Compute MDN loss from concatenated predictions.
        
        Args:
            y_true: (B, 2) true coordinates
            y_pred: (B, K + K*2 + K*2) concatenated [pi, mu_flat, sigma_flat]
        """
        K = self.num_mixtures
        D = self.output_dim
        
        # Split predictions
        pi = y_pred[:, :K]  # (B, K)
        mu_flat = y_pred[:, K:K + K*D]  # (B, K*D)
        sigma_flat = y_pred[:, K + K*D:]  # (B, K*D)
        
        # Reshape
        mu = tf.reshape(mu_flat, (-1, K, D))  # (B, K, 2)
        sigma = tf.reshape(sigma_flat, (-1, K, D))  # (B, K, 2)
        
        return mdn_loss_fn(y_true, pi, mu, sigma)


print("✅ MDN Loss functions defined")
print("   • mdn_loss_fn: Functional API for custom training")
print("   • MDNLoss: Keras Loss class for model.compile()")


# MDN MODEL BUILDER - TCN ENCODER + MDN HEAD

# Architecture:
#   Input (B, SEQ_LEN, features)
#      │
#      ▼
#   TCN Encoder (temporal feature extraction)
#      │
#      ▼
#   Dense layers (optional refinement)
#      │
#      ▼
#   MDN Layer → (π, μ, σ)

def build_tcn_mdn_model(
    seq_len,
    num_features,
    num_mixtures=5,
    tcn_filters=128,              # Changed to single int (required for skip connections)
    tcn_kernel_size=7,
    tcn_dilations=[1, 2, 4, 8, 16, 32],
    tcn_stacks=2,
    dropout_rate=0.3,
    use_batch_norm=True,
    dense_units=256,
    learning_rate=1e-4
):
    """
    Build complete TCN + MDN model for probabilistic coordinate prediction.
    
    Args:
        seq_len: Length of input sequences (e.g., 150)
        num_features: Number of input features (e.g., 26)
        num_mixtures: Number of Gaussian mixture components (K)
        tcn_filters: Number of filters for TCN (single int for skip connections)
        tcn_kernel_size: Kernel size for temporal convolutions
        tcn_dilations: Dilation rates for TCN
        tcn_stacks: Number of TCN stacks
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
        dense_units: Hidden units before MDN layer
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model with MDN outputs
    """
    
    print("═" * 60)
    print("🏗️  Building TCN + MDN Model")
    print("═" * 60)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Input layer
    # ─────────────────────────────────────────────────────────────────────────
    inputs = tf.keras.Input(
        shape=(seq_len, num_features), 
        name='sequence_input'
    )
    print(f"   Input shape: (B, {seq_len}, {num_features})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TCN Encoder
    # Note: Use single filter count (int) for skip_connections compatibility
    # ─────────────────────────────────────────────────────────────────────────
    
    # If list provided, check if all equal; if not, use last value
    if isinstance(tcn_filters, list):
        if len(set(tcn_filters)) > 1:
            print(f"   ⚠️  Converting varying filters {tcn_filters} to uniform {tcn_filters[-1]}")
            print(f"       (Required for skip connections)")
        tcn_filters = tcn_filters[-1]  # Use the last (largest) value
    
    x = TCN(
        nb_filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        nb_stacks=tcn_stacks,
        dilations=tcn_dilations,
        padding='causal',
        use_skip_connections=True,
        dropout_rate=dropout_rate,
        return_sequences=False,  # Get last timestep encoding
        use_batch_norm=use_batch_norm,
        use_layer_norm=False,
        name='tcn_encoder'
    )(inputs)
    
    print(f"   TCN output: (B, {tcn_filters})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dense layers for feature refinement
    # ─────────────────────────────────────────────────────────────────────────
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = tf.keras.layers.Dense(dense_units // 2, activation='relu', name='dense_2')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout_2')(x)
    
    print(f"   Dense output: (B, {dense_units // 2})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # MDN Output Layer
    # ─────────────────────────────────────────────────────────────────────────
    mdn_layer = MDNLayer(
        num_mixtures=num_mixtures, 
        output_dim=2,  # (x, y)
        sigma_min=MDN_SIGMA_MIN,
        name='mdn_output'
    )
    pi, mu, sigma = mdn_layer(x)
    
    print(f"   MDN outputs:")
    print(f"      π (weights): (B, {num_mixtures})")
    print(f"      μ (means):   (B, {num_mixtures}, 2)")
    print(f"      σ (stdevs):  (B, {num_mixtures}, 2)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Create model with named outputs
    # ─────────────────────────────────────────────────────────────────────────
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            'pi': pi,
            'mu': mu, 
            'sigma': sigma
        },
        name='TCN_MDN_Model'
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Model summary
    # ─────────────────────────────────────────────────────────────────────────
    total_params = model.count_params()
    print(f"\n   Total parameters: {total_params:,}")
    print("═" * 60)
    
    return model


def compile_mdn_model(model, learning_rate=1e-4):
    """
    Compile MDN model with appropriate optimizer.
    
    Note: MDN uses custom training loop, so we don't compile with loss.
    This function sets up the optimizer for manual training.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Gradient clipping for stability
    )
    return optimizer




# MDN CUSTOM TRAINING LOOP

# MDN requires custom training because:
#   1. Loss depends on multiple outputs (π, μ, σ)
#   2. Standard model.fit() doesn't handle GMM NLL easily
#   3. We need careful gradient management
def save_mdn_model(model, filepath='mdn_model.keras'):
    model.save(filepath)
    print(f"✅ Model saved to {filepath}")


class MDNTrainer:
    """
    Custom trainer for Mixture Density Network models.
    
    Handles:
        - Custom NLL loss computation
        - Gradient clipping for stability
        - Early stopping with validation
        - Keras-style progress bar with live metrics
        - Cosine decay LR scheduling with optional warmup
        - Training history tracking
    """
    
    def __init__(self, model, learning_rate=1e-4, clipnorm=1.0,
                 lr_schedule='cosine', warmup_epochs=2, min_lr_ratio=0.01):
        """
        Args:
            model: Keras model with MDN outputs
            learning_rate: Initial (peak) learning rate
            clipnorm: Max gradient norm for clipping
            lr_schedule: 'cosine' | 'constant' | 'step'
                - cosine: Cosine decay from learning_rate → learning_rate*min_lr_ratio
                - constant: Fixed learning rate throughout
                - step: Halve LR every 3 stagnant epochs
            warmup_epochs: Epochs to linearly warm up LR (cosine only)
            min_lr_ratio: Minimum LR as fraction of initial LR (cosine/step)
        """
        self.model = model
        self.initial_lr = learning_rate
        self.lr_schedule_type = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.min_lr = learning_rate * min_lr_ratio
        
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm
        )
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }
        
    def _get_lr(self, epoch, total_epochs):
        """Compute learning rate for current epoch based on schedule."""
        if self.lr_schedule_type == 'constant':
            return self.initial_lr
        
        elif self.lr_schedule_type == 'cosine':
            # Linear warmup phase
            if epoch < self.warmup_epochs:
                return self.initial_lr * (epoch + 1) / self.warmup_epochs
            # Cosine decay phase
            progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        elif self.lr_schedule_type == 'step':
            # Handled dynamically in fit() — return current optimizer LR
            return float(self.optimizer.learning_rate.numpy())
        
        return self.initial_lr
        
    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            outputs = self.model(x_batch, training=True)
            loss = mdn_loss_fn(
                y_batch,
                outputs['pi'],
                outputs['mu'],
                outputs['sigma']
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss
    
    @tf.function
    def val_step(self, x_batch, y_batch):
        outputs = self.model(x_batch, training=False)
        loss = mdn_loss_fn(
            y_batch,
            outputs['pi'],
            outputs['mu'],
            outputs['sigma']
        )
        return loss
    
    def fit(self, train_loader, val_loader, epochs=20, patience=5, 
            verbose=1, steps_per_epoch=None, validation_steps=None):
        """
        Train the MDN model with LR scheduling, progress bar, and early stopping.
        
        Args:
            train_loader: tf.data.Dataset (may be .repeat() so steps_per_epoch is REQUIRED)
            val_loader: tf.data.Dataset for validation
            epochs: Maximum epochs
            patience: Early stopping patience
            verbose: 0=silent, 1=progress bar
            steps_per_epoch: REQUIRED if train_loader uses .repeat().
            validation_steps: Optional max validation batches.
        """
        print("═" * 60)
        print("🚀 Starting MDN Training")
        print("═" * 60)
        
        if steps_per_epoch is None:
            card = tf.data.experimental.cardinality(train_loader).numpy()
            if card > 0:
                steps_per_epoch = int(card)
                print(f"   Auto-detected steps_per_epoch: {steps_per_epoch}")
            else:
                raise ValueError(
                    "train_loader is infinite (uses .repeat()). "
                    "You MUST pass steps_per_epoch = len(train_df) // BATCH_SIZE"
                )
        
        if validation_steps is None:
            card = tf.data.experimental.cardinality(val_loader).numpy()
            if card > 0:
                validation_steps = int(card)
            else:
                validation_steps = None
        
        print(f"   Steps/epoch: {steps_per_epoch}")
        print(f"   Val steps:   {validation_steps or 'auto'}")
        print(f"   Patience:    {patience}")
        print(f"   Initial LR:  {self.initial_lr:.6f}")
        print(f"   LR Schedule: {self.lr_schedule_type}")
        if self.lr_schedule_type == 'cosine':
            print(f"   Warmup:      {self.warmup_epochs} epochs")
            print(f"   Min LR:      {self.min_lr:.2e}")
        print("═" * 60)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        step_decay_counter = 0  # for 'step' schedule
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # ─── Update learning rate ────────────────────────────────────
            current_lr = self._get_lr(epoch, epochs)
            self.optimizer.learning_rate.assign(current_lr)
            
            # ─── Training phase ──────────────────────────────────────────
            print(f"\nEpoch {epoch+1}/{epochs}  (lr={current_lr:.2e})")
            progbar = tf.keras.utils.Progbar(
                steps_per_epoch, 
                width=30,
                stateful_metrics=['loss']
            )
            
            train_losses = []
            step = 0
            
            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)
                loss_val = np.asarray(loss).item()
                train_losses.append(loss_val)
                step += 1
                
                running_loss = np.mean(train_losses[-100:])
                progbar.update(step, values=[
                    ('loss', loss_val),
                    ('avg_loss', running_loss),
                ])
                
                if step >= steps_per_epoch:
                    break
            
            train_loss = np.mean(train_losses)
            
            # ─── Validation phase ────────────────────────────────────────
            val_losses = []
            val_step_count = 0
            
            for x_batch, y_batch in val_loader:
                v_loss = self.val_step(x_batch, y_batch)
                val_losses.append(float(v_loss.numpy()))
                val_step_count += 1
                if validation_steps and val_step_count >= validation_steps:
                    break
            
            val_loss = np.mean(val_losses)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # ─── Early stopping check ────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience_counter = 0
                step_decay_counter = 0
                marker = "✓ BEST"
                # 💾 SAVE TO DISK HERE
                checkpoint_name = "best_mdn_model.weights.h5"
                save_mdn_model(self.model,checkpoint_name)
                self.model.save_weights(checkpoint_name)
                print(f"✓ BEST - Saved weights to {checkpoint_name}")
            else:
                patience_counter += 1
                step_decay_counter += 1
                marker = f"patience {patience_counter}/{patience}"
                
                # Step schedule: halve LR after 3 epochs of no improvement
                if self.lr_schedule_type == 'step' and step_decay_counter >= 3:
                    new_lr = max(self.min_lr, current_lr * 0.5)
                    self.optimizer.learning_rate.assign(new_lr)
                    step_decay_counter = 0
                    marker += f" → LR halved to {new_lr:.2e}"
            
            # Print epoch summary
            print(f"  {epoch_time:.0f}s - "
                  f"train_nll: {train_loss:.4f} - "
                  f"val_nll: {val_loss:.4f} - "
                  f"{marker}")
            
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
            
            gc.collect()
        
        # Restore best weights
        if best_weights is not None:
            self.model.set_weights(best_weights)
            print(f"\n✅ Restored best weights (Val NLL: {best_val_loss:.4f})")
        
        return self.history


print("✅ MDNTrainer class defined")
print("   • LR schedules: 'cosine' (default), 'constant', 'step'")
print("   • Cosine: linear warmup → cosine decay to min_lr")
print("   • Step: halves LR after 3 stagnant epochs")
print("   • Keras-style Progbar with live loss tracking")


# MDN INFERENCE UTILITIES

# Functions for extracting predictions from MDN:
#   - Get multiple future positions with probabilities
#   - Sample from the predicted distribution
#   - Select top-K most likely futures


def mdn_predict(model, x_sequence):
    """
    Get MDN predictions from input sequence(s), auto-trimming or padding to model's expected SEQ_LEN.
    
    Args:
        model: Trained MDN model
        x_sequence: Input data
            - Single: (SEQ_LEN, features) or (1, SEQ_LEN, features)
            - Batch: (B, SEQ_LEN, features)
            
    Returns:
        dict with:
            - 'pi': (B, K) mixture weights
            - 'mu': (B, K, 2) means
            - 'sigma': (B, K, 2) std deviations
    """

    # Ensure batch dimension
    if len(x_sequence.shape) == 2:
        x_sequence = np.expand_dims(x_sequence, axis=0)

    # Get expected input shape from model
    expected_timesteps = model.input_shape[1]  # model.input_shape = (None, SEQ_LEN, features)
    n_features = model.input_shape[2]

    # Trim or pad sequences
    if x_sequence.shape[1] > expected_timesteps:
        x_sequence = x_sequence[:, -expected_timesteps:, :]  # take last SEQ_LEN timesteps
    elif x_sequence.shape[1] < expected_timesteps:
        pad_width = expected_timesteps - x_sequence.shape[1]
        x_sequence = np.pad(x_sequence, ((0, 0), (pad_width, 0), (0, 0)), mode='constant')

    # Check final shape
    assert x_sequence.shape[1:] == (expected_timesteps, n_features), \
        f"Input shape mismatch after padding/trimming: {x_sequence.shape[1:]} != {(expected_timesteps, n_features)}"

    # Get predictions
    outputs = model(x_sequence, training=False)

    return {
        'pi': outputs['pi'].numpy(),
        'mu': outputs['mu'].numpy(),
        'sigma': outputs['sigma'].numpy()
    }


def get_top_k_predictions(pi, mu, sigma, k=3):
    """
    Extract top-K most probable future positions.
    
    Args:
        pi: (K,) mixture weights for one sample
        mu: (K, 2) mixture means
        sigma: (K, 2) mixture std deviations
        k: Number of top predictions to return
        
    Returns:
        List of dicts with keys:
            - 'probability': Mixture weight
            - 'x': Predicted x coordinate
            - 'y': Predicted y coordinate
            - 'x_std': Uncertainty in x
            - 'y_std': Uncertainty in y
    """
    # Sort by probability (descending)
    sorted_indices = np.argsort(pi)[::-1][:k]
    
    results = []
    for idx in sorted_indices:
        results.append({
            'probability': float(pi[idx]),
            'x': float(mu[idx, 0]),
            'y': float(mu[idx, 1]),
            'x_std': float(sigma[idx, 0]),
            'y_std': float(sigma[idx, 1]),
            'component': int(idx)
        })
    
    return results


def sample_from_mdn(pi, mu, sigma, num_samples=100):
    """
    Sample future positions from the predicted GMM distribution.
    
    This generates samples that represent the full predictive distribution,
    useful for Monte Carlo analysis or uncertainty visualization.
    
    Args:
        pi: (K,) mixture weights
        mu: (K, 2) means
        sigma: (K, 2) std deviations
        num_samples: Number of samples to draw
        
    Returns:
        samples: (num_samples, 2) array of sampled (x, y) coordinates
        component_ids: (num_samples,) which component each sample came from
    """
    samples = []
    component_ids = []
    
    for _ in range(num_samples):
        # Choose mixture component based on weights
        k = np.random.choice(len(pi), p=pi)
        
        # Sample from that component's Gaussian
        x = np.random.normal(mu[k, 0], sigma[k, 0])
        y = np.random.normal(mu[k, 1], sigma[k, 1])
        
        samples.append([x, y])
        component_ids.append(k)
    
    return np.array(samples), np.array(component_ids)


def get_expected_position(pi, mu):
    """
    Compute expected (mean) position from mixture.
    
    E[x] = Σ πₖ · μₖ
    
    This is useful for comparison with deterministic predictions.
    
    Args:
        pi: (K,) mixture weights
        mu: (K, 2) means
        
    Returns:
        (2,) array with expected (x, y)
    """
    # Weighted sum of means
    expected = np.sum(pi[:, np.newaxis] * mu, axis=0)
    return expected


def get_most_likely_position(pi, mu):
    """
    Get the single most likely position (mode).
    
    This is simply the mean of the highest-weight component.
    
    Args:
        pi: (K,) mixture weights
        mu: (K, 2) means
        
    Returns:
        (2,) array with mode (x, y)
    """
    best_k = np.argmax(pi)
    return mu[best_k]


def batch_predictions_to_dataframe(predictions, sample_ids=None):
    """
    Convert batch MDN predictions to a pandas DataFrame for analysis.
    
    Args:
        predictions: dict with 'pi', 'mu', 'sigma' from mdn_predict()
        sample_ids: Optional list of sample identifiers
        
    Returns:
        DataFrame with one row per (sample, component)
    """
    pi = predictions['pi']
    mu = predictions['mu']
    sigma = predictions['sigma']
    
    B, K, _ = mu.shape
    
    records = []
    for b in range(B):
        sample_id = sample_ids[b] if sample_ids is not None else b
        for k in range(K):
            records.append({
                'sample_id': sample_id,
                'component': k,
                'probability': pi[b, k],
                'x': mu[b, k, 0],
                'y': mu[b, k, 1],
                'x_std': sigma[b, k, 0],
                'y_std': sigma[b, k, 1]
            })
    
    return pd.DataFrame(records)



# MDN EVALUATION METRICS

# Why MAE/RMSE are insufficient for MDN:
#   - They only measure distance to a single prediction
#   - They don't reward capturing multiple modes
#   - They ignore uncertainty calibration

# Better metrics for probabilistic predictions:
#   - NLL (Negative Log Likelihood) - primary metric
#   - Best-of-K MAE - rewards multi-modal predictions
#   - Calibration - checks if uncertainty matches reality
#   - Component diversity - detects mode collapse

def evaluate_mdn_nll(model, test_loader):
    """
    Compute mean Negative Log Likelihood on test set.
    
    Lower NLL = better probabilistic predictions.
    
    Args:
        model: Trained MDN model
        test_loader: tf.data.Dataset with (x, y) pairs
        
    Returns:
        mean_nll: Average NLL over all samples
    """
    total_nll = 0.0
    total_samples = 0
    
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch, training=False)
        
        batch_nll = mdn_loss_fn(
            y_batch,
            outputs['pi'],
            outputs['mu'],
            outputs['sigma']
        )
        
        batch_size = x_batch.shape[0]
        total_nll += batch_nll.numpy() * batch_size
        total_samples += batch_size
    
    mean_nll = total_nll / total_samples
    return mean_nll


def evaluate_best_of_k_mae(model, test_loader, k=3):
    """
    Compute Best-of-K Mean Absolute Error.
    
    For each sample, compute distance to ALL K mixture means,
    then take the MINIMUM. This rewards multi-modal predictions
    that capture different possible futures.
    
    Args:
        model: Trained MDN model
        test_loader: tf.data.Dataset
        k: Number of top components to consider
        
    Returns:
        best_of_k_mae: Mean of minimum errors
        all_mae: Regular MAE using expected position
    """
    best_errors = []
    expected_errors = []
    mode_errors = []
    
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch, training=False)
        
        pi = outputs['pi'].numpy()    # (B, K)
        mu = outputs['mu'].numpy()    # (B, K, 2)
        y_true = y_batch.numpy()      # (B, 2)
        
        batch_size = y_true.shape[0]
        
        for i in range(batch_size):
            # Distance to each mixture mean
            distances = np.linalg.norm(mu[i] - y_true[i], axis=-1)  # (K,)
            
            # Best-of-K: minimum distance among top-k by probability
            top_k_indices = np.argsort(pi[i])[::-1][:k]
            best_error = np.min(distances[top_k_indices])
            best_errors.append(best_error)
            
            # Expected position MAE
            expected_pos = get_expected_position(pi[i], mu[i])
            expected_errors.append(np.linalg.norm(expected_pos - y_true[i]))
            
            # Mode MAE
            mode_pos = get_most_likely_position(pi[i], mu[i])
            mode_errors.append(np.linalg.norm(mode_pos - y_true[i]))
    
    return {
        'best_of_k_mae': np.mean(best_errors),
        'expected_mae': np.mean(expected_errors),
        'mode_mae': np.mean(mode_errors)
    }


def evaluate_component_diversity(model, test_loader, threshold=0.1):
    """
    Measure diversity of mixture components.
    
    Detects mode collapse (when one component dominates).
    
    Args:
        model: Trained MDN model
        test_loader: tf.data.Dataset
        threshold: Minimum weight to count component as "active"
        
    Returns:
        dict with diversity metrics
    """
    all_active_counts = []
    all_entropies = []
    all_max_weights = []
    
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch, training=False)
        pi = outputs['pi'].numpy()  # (B, K)
        
        for i in range(pi.shape[0]):
            # Count active components
            num_active = np.sum(pi[i] > threshold)
            all_active_counts.append(num_active)
            
            # Entropy of mixture weights (higher = more diverse)
            entropy = -np.sum(pi[i] * np.log(pi[i] + 1e-10))
            all_entropies.append(entropy)
            
            # Maximum weight (lower = more diverse)
            all_max_weights.append(np.max(pi[i]))
    
    return {
        'mean_active_components': np.mean(all_active_counts),
        'mean_entropy': np.mean(all_entropies),
        'mean_max_weight': np.mean(all_max_weights),
        'mode_collapse_ratio': np.mean(np.array(all_max_weights) > 0.9)
    }


def evaluate_calibration(model, test_loader, percentiles=[10, 25, 50, 75, 90]):
    """
    Check calibration of uncertainty estimates.
    
    For a well-calibrated model, X% of true values should fall
    within the X% confidence region of the predicted distribution.
    
    Args:
        model: Trained MDN model
        test_loader: tf.data.Dataset
        percentiles: Confidence levels to check
        
    Returns:
        dict mapping percentile to actual coverage
    """
    # For GMM, we use sampling-based calibration
    num_samples_per_prediction = 1000
    coverage = {p: [] for p in percentiles}
    
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch, training=False)
        
        pi = outputs['pi'].numpy()
        mu = outputs['mu'].numpy()
        sigma = outputs['sigma'].numpy()
        y_true = y_batch.numpy()
        
        batch_size = y_true.shape[0]
        
        for i in range(min(batch_size, 100)):  # Limit for speed
            # Sample from GMM
            samples, _ = sample_from_mdn(pi[i], mu[i], sigma[i], num_samples_per_prediction)
            
            # Compute distance from true point to all samples
            distances_from_true = np.linalg.norm(samples - y_true[i], axis=-1)
            
            for p in percentiles:
                # What fraction of samples are further than true point?
                # If calibrated, this should be (100-p)%
                threshold = np.percentile(distances_from_true, 100 - p)
                is_within = np.linalg.norm(y_true[i] - np.mean(samples, axis=0)) <= threshold
                coverage[p].append(float(is_within))
    
    # Average coverage at each percentile
    calibration = {p: np.mean(v) * 100 for p, v in coverage.items()}
    return calibration


def full_mdn_evaluation(model, test_loader, k=3, verbose=True):
    """
    Complete MDN evaluation with all metrics.
    
    Args:
        model: Trained MDN model
        test_loader: tf.data.Dataset
        k: For best-of-k MAE
        verbose: Print results
        
    Returns:
        dict with all metrics
    """
    if verbose:
        print("═" * 60)
        print("📊 MDN Evaluation")
        print("═" * 60)
    
    # NLL
    nll = evaluate_mdn_nll(model, test_loader)
    if verbose:
        print(f"   Negative Log Likelihood: {nll:.4f}")
    
    # MAE variants
    mae_metrics = evaluate_best_of_k_mae(model, test_loader, k=k)
    if verbose:
        print(f"\n   MAE Metrics:")
        print(f"      Best-of-{k} MAE:  {mae_metrics['best_of_k_mae']:.4f}")
        print(f"      Expected MAE:    {mae_metrics['expected_mae']:.4f}")
        print(f"      Mode MAE:        {mae_metrics['mode_mae']:.4f}")
    
    # Diversity
    diversity = evaluate_component_diversity(model, test_loader)
    if verbose:
        print(f"\n   Component Diversity:")
        print(f"      Active components: {diversity['mean_active_components']:.2f}")
        print(f"      Entropy:           {diversity['mean_entropy']:.3f}")
        print(f"      Mode collapse:     {diversity['mode_collapse_ratio']*100:.1f}%")
    
    if verbose:
        print("═" * 60)
    
    return {
        'nll': nll,
        **mae_metrics,
        **diversity
    }


print("✅ MDN Evaluation metrics defined")
print("   • evaluate_mdn_nll: Negative Log Likelihood")
print("   • evaluate_best_of_k_mae: Multi-modal MAE")
print("   • evaluate_component_diversity: Mode collapse detection")
print("   • evaluate_calibration: Uncertainty calibration")
print("   • full_mdn_evaluation: Complete evaluation suite")


# MDN VISUALIZATION FUNCTIONS

# Visualization tools for multi-modal predictions:
#   - Plot predictions on football field
#   - Show uncertainty ellipses
#   - Training loss curves
#   - Component distribution analysis

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

def plot_mdn_prediction_on_field(pi, mu, sigma, true_pos=None, current_pos=None,
                                  field_size=(105, 68), ax=None, title=None):
    """
    Visualize MDN predictions on a football field.
    
    Shows each mixture component as an ellipse with:
        - Size proportional to standard deviation
        - Color intensity proportional to mixture weight
        - Label showing probability
    
    Args:
        pi: (K,) mixture weights
        mu: (K, 2) means (normalized predictions)
        sigma: (K, 2) std deviations (normalized)
        true_pos: Optional (2,) true future position (normalized)
        current_pos: Optional (2,) current position (normalized)
        field_size: (width, height) of the football field in meters
        ax: Optional matplotlib axes
        title: Optional plot title
        
    Returns:
        fig, ax: matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    else:
        fig = ax.figure

    # ─────────────────────────────────────────────────────────────────────────
    # Draw football field
    # ─────────────────────────────────────────────────────────────────────────
    ax.set_xlim(-0.02, field_size[0] + 0.02)  # Set x limits (width of the field)
    ax.set_ylim(-0.02, field_size[1] + 0.02)  # Set y limits (length of the field)
    ax.set_facecolor('#2d8a2d')  # Green field
    
    # Field outline
    ax.add_patch(plt.Rectangle((0, 0), field_size[0], field_size[1], 
                                 fill=False, edgecolor='white', linewidth=2))
    
    # Center line
    ax.axvline(x=0.5 * field_size[0], color='white', linewidth=1, linestyle='--', alpha=0.5)
    
    # Center circle
    center_circle = plt.Circle((0.5 * field_size[0], 0.5 * field_size[1]), 
                                 9.15, fill=False, color='white', linewidth=1, alpha=0.5)
    ax.add_patch(center_circle)
    
    # Penalty areas (simplified, for a full field they would be 16.5m long)
    ax.add_patch(plt.Rectangle((0, 24.84), 5.5, 16.5, 
                                 fill=False, edgecolor='white', linewidth=1, alpha=0.5))
    ax.add_patch(plt.Rectangle((field_size[0] - 5.5, 24.84), 5.5, 16.5, 
                                 fill=False, edgecolor='white', linewidth=1, alpha=0.5))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plot mixture components as ellipses
    # ─────────────────────────────────────────────────────────────────────────
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(pi)))
    
    # Sort by probability for layering (smallest first, so highest is on top)
    sorted_indices = np.argsort(pi)
    
    for idx in sorted_indices:
        if pi[idx] > 0.02:  # Only show significant components
            # Scale mu (normalized predictions) to real-world dimensions
            x_pos = mu[idx, 0] * field_size[0]
            y_pos = mu[idx, 1] * field_size[1]
            
            # Scale sigma (standard deviations) to real-world dimensions
            sigma_x = sigma[idx, 0] * field_size[0]
            sigma_y = sigma[idx, 1] * field_size[1]
            
            # Create ellipse (2 standard deviations)
            ellipse = Ellipse(
                xy=(x_pos, y_pos),
                width=4 * sigma_x,   # 2σ on each side
                height=4 * sigma_y,
                angle=0,
                alpha=min(0.8, pi[idx] + 0.2),
                facecolor=colors[np.searchsorted(np.sort(pi), pi[idx])],
                edgecolor='white',
                linewidth=2
            )
            ax.add_patch(ellipse)
            
            # Probability label
            ax.annotate(
                f'{pi[idx]*100:.0f}%',
                xy=(x_pos, y_pos),
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plot current position
    # ─────────────────────────────────────────────────────────────────────────
    if current_pos is not None:
        # Scale current position to field dimensions
        y_current = current_pos[1] 
        x_current = current_pos[0]


        print(f"True position (scaled): ({y_current:.2f}, {x_current:.2f})")
        ax.scatter(
            x_current, y_current,
            c='yellow', s=200, marker='o',
            edgecolors='black', linewidth=2,
            zorder=15, label='Current Position'
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plot true future position
    # ─────────────────────────────────────────────────────────────────────────
    if true_pos is not None:
        # Scale true position to field dimensions
        y_true = true_pos[1]
        x_true = true_pos[0]
        print(f"True position (scaled): ({y_true:.2f}, {x_true:.2f})")
        ax.scatter(
            x_true, y_true,
            c='red', s=250, marker='X',
            edgecolors='white', linewidth=2,
            zorder=20, label='True Future Position'
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Labels and legend
    # ─────────────────────────────────────────────────────────────────────────
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title or 'MDN Multi-Modal Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    
    return fig, ax


def plot_mdn_training_history(history, figsize=(12, 5)):
    """
    Plot MDN training loss curves.
    
    Args:
        history: dict with 'train_loss' and 'val_loss' lists
        figsize: Figure size
        
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # NLL Loss
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train NLL')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val NLL')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Negative Log Likelihood', fontsize=12)
    ax1.set_title('MDN Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    ax2 = axes[1]
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(epochs, 0, loss_diff, where=loss_diff > 0, 
                      alpha=0.3, color='red', label='Overfitting')
    ax2.fill_between(epochs, 0, loss_diff, where=loss_diff < 0, 
                      alpha=0.3, color='green', label='Good fit')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val NLL - Train NLL', fontsize=12)
    ax2.set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mixture_weights_distribution(model, test_loader, num_samples=500):
    """
    Analyze distribution of mixture weights across predictions.
    
    Helps detect mode collapse or unused components.
    
    Args:
        model: Trained MDN model
        test_loader: tf.data.Dataset
        num_samples: Number of samples to analyze
        
    Returns:
        fig: matplotlib figure
    """
    all_pi = []
    count = 0
    
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch, training=False)
        all_pi.extend(outputs['pi'].numpy())
        count += len(x_batch)
        if count >= num_samples:
            break
    
    all_pi = np.array(all_pi[:num_samples])
    K = all_pi.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot of weights per component
    ax1 = axes[0]
    bp = ax1.boxplot([all_pi[:, k] for k in range(K)], 
                      labels=[f'K={k}' for k in range(K)])
    ax1.set_xlabel('Mixture Component', fontsize=12)
    ax1.set_ylabel('Weight (π)', fontsize=12)
    ax1.set_title('Mixture Weight Distribution per Component', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Histogram of max weight (mode collapse indicator)
    ax2 = axes[1]
    max_weights = np.max(all_pi, axis=1)
    ax2.hist(max_weights, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.9, color='red', linestyle='--', linewidth=2, 
                 label='Mode collapse threshold')
    ax2.set_xlabel('Maximum Component Weight', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Mode Collapse Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_prediction_samples(model, x_sample, y_true=None, num_samples=200):
    """
    Plot sampled predictions from MDN to visualize full distribution.
    
    Args:
        model: Trained MDN model
        x_sample: (SEQ_LEN, features) single input sequence
        y_true: Optional (2,) true coordinates
        num_samples: Number of samples to draw
        
    Returns:
        fig, ax: matplotlib figure and axes
    """
    # Get MDN outputs
    preds = mdn_predict(model, x_sample)
    pi = preds['pi'][0]
    mu = preds['mu'][0]
    sigma = preds['sigma'][0]
    
    # Sample from distribution
    samples, component_ids = sample_from_mdn(pi, mu, sigma, num_samples)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Color by component
    scatter = ax.scatter(
        samples[:, 0], samples[:, 1],
        c=component_ids, cmap='Set1',
        alpha=0.5, s=30
    )
    
    # Plot component means
    for k in range(len(pi)):
        if pi[k] > 0.05:
            ax.scatter(
                mu[k, 0], mu[k, 1],
                c='black', s=200, marker='+',
                linewidth=3, label=f'μ_{k} ({pi[k]*100:.0f}%)'
            )
    
    # Plot true position
    if y_true is not None:
        ax.scatter(
            y_true[0], y_true[1],
            c='red', s=300, marker='X',
            edgecolor='white', linewidth=2,
            zorder=10, label='True'
        )
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'MDN Samples ({num_samples} draws)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.colorbar(scatter, ax=ax, label='Component')
    
    return fig, ax


print("✅ MDN Visualization functions defined")
print("   • plot_mdn_prediction_on_field: Field visualization")
print("   • plot_mdn_training_history: Loss curves")
print("   • plot_mixture_weights_distribution: Component analysis")
print("   • plot_prediction_samples: Sample visualization")


