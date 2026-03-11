
import tensorflow as tf
import numpy as np
try:
    # pylint: disable=E0611,E0401
    from tensorflow.keras.utils import register_keras_serializable
  # For recent Keras
except ImportError:
    # pylint: disable=E0611,E0401
    from tensorflow.keras.utils import register_keras_serializable  # For older versions
    

# zone_coords = np.zeros((NUM_ZONES, 2), dtype=np.float32)

# for z in range(NUM_ZONES):
#     r = z // N_COLS
#     c = z % N_COLS
#     zone_coords[z] = [r, c]

# ZONE_COORDS = tf.constant(zone_coords)  # (144, 2)

# MAX_DIST = tf.constant(
#     np.sqrt((N_ROWS - 1)**2 + (N_COLS - 1)**2),
#     dtype=tf.float32
# )

# Spatially weighted CE loss
# register_keras_serializable()
# class SpatialCrossEntropy(tf.keras.losses.Loss):
#     def __init__(self, alpha=2.0, name="spatial_ce"):
#         super().__init__(name=name)
#         self.alpha = alpha
#         self.ce = tf.keras.losses.SparseCategoricalCrossentropy(
#             from_logits=False,
#             reduction="none"
#         )

#     def call(self, y_true, y_pred):
#         # y_true: (batch,)
#         # y_pred: (batch, num_zones)

#         y_true = tf.cast(y_true, tf.int32)

#         # 1️⃣ Standard CE
#         ce_loss = self.ce(y_true, y_pred)  # (batch,)

#         # 2️⃣ Predicted zone
#         pred_zone = tf.argmax(y_pred, axis=1, output_type=tf.int32)

#         # 3️⃣ Look up grid coordinates
#         true_xy = tf.gather(ZONE_COORDS, y_true)
#         pred_xy = tf.gather(ZONE_COORDS, pred_zone)

#         # 4️⃣ Grid distance
#         dist = tf.norm(true_xy - pred_xy, axis=1)  # Euclidean

#         # 5️⃣ Normalize + weight
#         weight = 1.0 + self.alpha * (dist / MAX_DIST)

#         return ce_loss * weight



# # Focal Loss with per-class alpha support
# def focal_loss_sparse_per_class(alpha=None, gamma=2.0):
#     """
#     Sparse Focal Loss with per-class alpha support.
#     Works for:
#         y_pred: (batch, num_classes)
#         y_pred: (batch, time, num_classes)
#     """

#     alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

#     def loss(y_true, y_pred):

#         # If sequence output, flatten time dimension
#         if len(tf.shape(y_pred)) == 3:
#             num_classes = tf.shape(y_pred)[-1]
#             y_pred = tf.reshape(y_pred, [-1, num_classes])
#             y_true = tf.reshape(y_true, [-1])

#         y_true = tf.cast(y_true, tf.int32)

#         # Clip predictions
#         y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

#         num_classes = tf.shape(y_pred)[-1]

#         # One-hot
#         y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

#         # p_t
#         p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

#         # Focal weight
#         focal_weight = tf.pow(1.0 - p_t, gamma)

#         # Alpha per sample
#         if alpha is not None:
#             sample_alpha = tf.gather(alpha, y_true)
#         else:
#             sample_alpha = 1.0

#         # Cross entropy
#         ce = -tf.math.log(p_t)

#         loss_val = sample_alpha * focal_weight * ce

#         # return tf.reduce_mean(loss_val)
#         return loss_val

#     return loss


# # Spatially weighted CE loss (simple function version)
# register_keras_serializable()
# def spatial_loss(y_true, y_pred, alpha=0.1):
#     y_true = tf.squeeze(y_true)
#     y_true = tf.cast(y_true, tf.int32)

#     # Cross-entropy (stable, handled by TF)
#     ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

#     # Cast CE to float32 explicitly (recommended with mixed precision)
#     ce = tf.cast(ce, tf.float32)

#     # True row/col
#     true_r = tf.cast(y_true // N_COLS, tf.float32)
#     true_c = tf.cast(y_true % N_COLS, tf.float32)

#     # ---- IMPORTANT FIX HERE ----
#     num_zones = tf.shape(y_pred)[-1]

#     zones = tf.range(num_zones, dtype=tf.int32)   # ✅ int
#     zones = tf.cast(zones, tf.float32)             # → float

#     zone_r = tf.floor(zones / tf.cast(N_COLS, tf.float32))
#     zone_c = tf.math.floormod(zones, tf.cast(N_COLS, tf.float32))

#     # Expected predicted row/col
#     y_pred_f = tf.cast(y_pred, tf.float32)         # align dtypes

#     pred_r = tf.reduce_sum(y_pred_f * zone_r, axis=-1)
#     pred_c = tf.reduce_sum(y_pred_f * zone_c, axis=-1)

#     spatial_dist = tf.abs(true_r - pred_r) + tf.abs(true_c - pred_c)

#     return ce + alpha * spatial_dist



#Merged spatial and focal loss for better performance
def get_spatio_temporal_focal_loss(n_cols, class_alphas=None, gamma=2.0, spatial_weight=0.1):
    """
    Merged Loss: Per-Class Focal Loss + Differentiable Manhattan Spatial Penalty.
    
    Args:
        n_cols: Number of columns in the pitch grid (9 for a 3x9 grid).
        class_alphas: Array or list of weights for each class to handle imbalance.
        gamma: Focusing parameter for Focal Loss.
        spatial_weight: Multiplier to scale the spatial penalty's impact.
    """
    # Pre-convert class_alphas to a tensor if provided
    if class_alphas is not None:
        class_alphas = tf.constant(class_alphas, dtype=tf.float32)

    def loss(y_true, y_pred):
        # 1️⃣ Sequence Handling: Flatten time dimension if 3D (for TCNs)
        if y_pred.shape.rank == 3:
            num_classes = tf.shape(y_pred)[-1]
            y_pred = tf.reshape(y_pred, [-1, num_classes])
            y_true = tf.reshape(y_true, [-1])
        else:
            num_classes = tf.shape(y_pred)[-1]

        y_true = tf.cast(y_true, tf.int32)
        
        # Clip predictions for log stability
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # 2️⃣ Focal Loss Calculation
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred_clipped, axis=-1)
        
        focal_weight = tf.pow(1.0 - p_t, gamma)
        sample_alpha = tf.gather(class_alphas, y_true) if class_alphas is not None else 1.0
        
        ce_loss = -tf.math.log(p_t)
        focal_loss_val = sample_alpha * focal_weight * ce_loss

        # 3️⃣ Differentiable Spatial Penalty Calculation
        true_r = tf.cast(y_true // n_cols, tf.float32)
        true_c = tf.cast(y_true % n_cols, tf.float32)

        # Generate expected/soft coordinates based on probabilities
        zones = tf.cast(tf.range(num_classes), tf.float32)
        n_cols_f = tf.cast(n_cols, tf.float32)
        
        zone_r = tf.floor(zones / n_cols_f)
        # zone_c = tf.math.floormod(zones, n_cols_f)
        zone_c = tf.math.floormod(zones, n_cols)

        pred_r = tf.reduce_sum(y_pred_clipped * zone_r, axis=-1)
        pred_c = tf.reduce_sum(y_pred_clipped * zone_c, axis=-1)

        # Manhattan distance
        spatial_dist = tf.abs(true_r - pred_r) + tf.abs(true_c - pred_c)

        # 4️⃣ Combine and return (UNREDUCED for sequence masking)
        return focal_loss_val + (spatial_weight * spatial_dist)

    return loss


def label_smoothing_sparse_categorical_crossentropy(smoothing=0.1):
    """
    Label Smoothing for classification - reduces overconfidence.
    
    Instead of [0, 0, 1, 0] → [0.025, 0.025, 0.925, 0.025]
    
    Benefits:
        - Prevents model from becoming overconfident
        - Improves generalization to unseen data
        - Acts as regularization
    """
    def loss(y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)

        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        # Apply label smoothing
        y_true_smooth = y_true_one_hot * (1.0 - smoothing) + (smoothing / num_classes)
        
        # Categorical crossentropy
        return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
    
    return loss




#CUSTOM LOSS FUNCTIONS FOR BETTER COORDINATE PREDICTION

def huber_loss_tf(delta=0.1):
    """
    Huber loss - less sensitive to outliers than MSE
    Better for coordinate prediction with occasional large errors
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        is_small = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(is_small, squared_loss, linear_loss)
    return loss

def weighted_mse_loss():
    def loss(y_true, y_pred):
        # Use ... to handle (batch, 2) OR (batch, time, 2) safely
        x_error = tf.square(y_true[..., 0] - y_pred[..., 0])
        y_error = tf.square(y_true[..., 1] - y_pred[..., 1])
        return 1.2 * x_error + 0.8 * y_error  
    return loss



