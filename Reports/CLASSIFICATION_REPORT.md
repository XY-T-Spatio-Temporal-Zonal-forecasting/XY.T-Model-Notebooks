# Classification Report: Zone-Based Player Movement Prediction

## 1. Problem Statement

Predict which **zone** on a football pitch a player will occupy after a future time horizon (3 seconds), framed as a **64-class classification** problem over a discretized pitch grid.

---

## 2. Pitch Discretization

| Parameter | Value |
|-----------|-------|
| Grid rows (`N_ROWS`) | 4 |
| Grid columns (`N_COLS`) | 16 |
| Total zones (`NUM_ZONES`) | 64 |

**Zone mapping** via `xy_to_zone_vectorized()`:

- Normalized coordinates $(x \in [0,1],\ y \in [0,1])$ are discretized:
  - $\text{row} = \text{clip}(y \times N\_ROWS,\ 0,\ N\_ROWS - 1)$
  - $\text{col} = \text{clip}(x \times N\_COLS,\ 0,\ N\_COLS - 1)$
  - $\text{zone\_id} = \text{row} \times N\_COLS + \text{col}$

Each zone has a pre-computed center coordinate stored in `ZONE_COORDS` (shape `(64, 2)`) used for spatial distance calculations in the loss function.

---

## 3. Data Pipeline

### 3.1 Data Source & Loading

- **File**: `combined_matches.csv` (multi-match football tracking data)
- **Loading**: Chunked reading (`chunk_size=500,000`) with `psutil` memory monitoring (halts at 85% RAM usage)
- **Memory optimization**:
  - `float64` → `float32` downcasting
  - `int64` → smallest integer type via `pd.to_numeric(downcast='integer')`
  - Timestamp string columns auto-detected and dropped

### 3.2 Feature Engineering (24 Features)

Two functions transform raw positional data into a rich 24-feature vector per frame:

#### `add_velocity_features(df)` — grouped by `(match_id, player_id)`

| Category | Features | Derivation |
|----------|----------|------------|
| **Position** | `x_normalized`, `y_normalized` | Raw normalized coordinates |
| **Basic velocity** | `dx`, `dy` | Frame-to-frame diff, smoothed with rolling mean (window=3) |
| **Speed / acceleration** | `speed_normalized`, `acceleration`, `movement_angle` | Derived from dx/dy |
| **Multi-window velocity** | `dx_avg_3/5/10`, `dy_avg_3/5/10`, `speed_avg_3/5/10` | Rolling means over 3, 5, 10 frame windows (9 features total) |
| **Movement trends** | `acceleration_trend`, `angle_change`, `angle_stability`, `speed_change_rate` | Rolling stats of acceleration and angle |
| **Spatial context** | `distance_from_center`, `distance_from_goal_home`, `distance_from_goal_away`, `distance_from_sideline` | Euclidean / Manhattan distances to pitch landmarks |

#### `add_contextual_features(df)`

| Feature | Derivation |
|---------|------------|
| `period_progress` | Timestamp normalized within each match period |
| `position_encoded` | Ordinal encoding: GK=0, DEF=1, MID=2, FWD=3 |

### 3.3 Data Splitting

**`split_by_match_df(fixed_split=True)`** — match-level split to prevent data leakage:

- 3 matches → training set
- 1 match → validation set
- 1 match → test set
- Random seed: 42

### 3.4 Feature Scaling

- `sklearn.preprocessing.StandardScaler` fitted on **training data only**
- Applied per-sample during sequence generation (not globally)

### 3.5 Sequence Generation

**`keras_sequence_generator()`** yields `(sequence, zone_id)` tuples:

1. Groups frames by `(match_id, player_id)`
2. Slides a window of `SEQ_LEN=150` frames across each player's track
3. **Target**: `zone_id` at frame $t_{\text{current}} + \text{HORIZON\_FRAMES}$
4. Minimum track length required: $\text{SEQ\_LEN} + \text{HORIZON\_FRAMES} = 225$ frames
5. Scaler is applied to each sequence's 24 features

| Parameter | Value |
|-----------|-------|
| `SEQ_LEN` | 150 frames (6 seconds at 25 FPS) |
| `FPS` | 25 |
| `HORIZON_SECONDS` | 3 seconds |
| `HORIZON_FRAMES` | 75 frames |

### 3.6 tf.data Pipeline

**`make_tf_dataset()`** wraps the generator into a high-performance `tf.data.Dataset`:

| Setting | Value |
|---------|-------|
| Input shape | `(150, 24)` |
| Target shape | `()` scalar zone ID |
| Shuffle buffer | 4,096 |
| `drop_remainder` | `True` |
| Prefetch | `tf.data.AUTOTUNE` |
| Training repeat | `.repeat()` with explicit `steps_per_epoch` |

---

## 4. Model Architecture: Keras TCN (Classification)

### 4.1 Temporal Convolutional Network (WaveNet-style)

Based on Philippe Remy's `keras-tcn` — custom `ResidualBlock` and `TCN` classes included directly in the notebook.

**`ResidualBlock`** (one per dilation rate):

1. **2× dilated causal `Conv1D`** — same filter count, kernel size, and dilation rate
2. After each conv: optional `BatchNormalization` → `Activation('relu')` → `SpatialDropout1D`
3. **1×1 `Conv1D`** residual shortcut if the channel dimensions differ
4. **Final activation** on the element-wise sum of the residual path and the shortcut

**`TCN` layer** aggregates stacked residual blocks with additive skip connections:

| Parameter | Value |
|-----------|-------|
| `nb_filters` | 128 |
| `kernel_size` | 7 |
| `dilations` | [1, 2, 4, 8, 16, 32] |
| `nb_stacks` | 2 |
| `padding` | `'causal'` |
| `use_skip_connections` | True |
| `return_sequences` | False (outputs last timestep only) |
| `dropout_rate` | 0.1 |
| `activation` | `'relu'` |
| `use_batch_norm` | True |
| `kernel_initializer` | `'he_normal'` |

**Receptive field**:

$$RF = 1 + 2 \times (k - 1) \times S \times \sum_{i} d_i = 1 + 2 \times 6 \times 2 \times 63 = 1513 \text{ frames}$$

This far exceeds the 150-frame input length, ensuring the network can attend to the entire sequence.

### 4.2 Classification Head

```
Dense(64, activation='softmax')
```

Applied to the TCN's last-timestep output: `(batch, 128) → (batch, 64)`.

### 4.3 Full Architecture Diagram

```
Input  (B, 150, 24)
   │
   ▼
TCN Encoder
   │  128 filters, 2 stacks × 6 dilations
   │  causal padding, skip connections, BatchNorm
   │  return_sequences=False
   │  output: (B, 128)
   ▼
Dense(64, softmax) → (B, 64)   [zone probabilities]
```

### 4.4 Compilation (via `compiled_tcn()`)

| Component | Value |
|-----------|-------|
| **Optimizer** | `AdamW(learning_rate=5e-4, clipnorm=1.0)` |
| **Loss** | `spatial_loss` (custom — see §5) |
| **Metrics** | `sparse_categorical_accuracy`, `SparseTopKCategoricalAccuracy(k=5)` |

---

## 5. Loss Functions

### 5.1 `spatial_loss(y_true, y_pred, alpha=0.1)` — Active

Combines standard cross-entropy with a **spatial distance penalty** that discourages predictions that are far from the true zone on the pitch grid:

$$\mathcal{L} = \text{CE}(y, \hat{y}) + \alpha \cdot \left( \left| r_{\text{true}} - \mathbb{E}[r_{\text{pred}}] \right| + \left| c_{\text{true}} - \mathbb{E}[c_{\text{pred}}] \right| \right)$$

Where:
- $\text{CE}$ = `sparse_categorical_crossentropy`
- $\mathbb{E}[r_{\text{pred}}] = \sum_{z=0}^{63} p(z) \cdot \text{row}(z)$ — probability-weighted expected predicted row
- $\mathbb{E}[c_{\text{pred}}] = \sum_{z=0}^{63} p(z) \cdot \text{col}(z)$ — probability-weighted expected predicted column
- $\alpha = 0.1$

This ensures that even when the exact zone prediction is wrong, the model is still penalized proportionally to how far away the prediction is spatially.

### 5.2 `SpatialCrossEntropy(alpha=2.0)` — Available, Not Active

Alternative loss that uses **Euclidean distance** between the true zone center and the argmax-predicted zone center:

$$\text{weight} = 1 + \alpha \cdot \frac{d}{d_{\max}}$$

This weight is multiplied against per-sample cross-entropy loss. Currently available but **commented out** in the model compilation.

---

## 6. Training Configuration

### 6.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| `EPOCHS` | 20 |
| `BATCH_SIZE` | 256 |
| `LR` | 5e-4 |
| `steps_per_epoch` | `len(train_df) // BATCH_SIZE` |

### 6.2 Callbacks

| Callback | Configuration |
|----------|---------------|
| `GarbageCollectorCallback` | Runs `gc.collect()` + `tf.keras.backend.clear_session()` after each epoch |
| `EarlyStopping` | Monitor: `val_sparse_top_k_categorical_accuracy`, mode: `max`, patience: 5, restores best weights |
| `ReduceLROnPlateau` | Same monitor, factor: 0.5, patience: 3, min_lr: 1e-6 |
| `ModelCheckpoint` | Saves best model to `best_model_mae.keras`, same monitor |

### 6.3 Class Weights

- Computed via `sklearn.utils.class_weight.compute_class_weight("balanced")` on `train_df['zone']`
- Applied to `model.fit(class_weight=class_weights)` to mitigate zone imbalance
- Not used when training in coordinate regression mode

### 6.4 Mixed Precision

- Global policy: `mixed_float16` via `tf.keras.mixed_precision`
- Loss values explicitly cast to `float32` inside `spatial_loss` for numerical stability

### 6.5 Multi-GPU Support

- `tf.distribute.MirroredStrategy` when more than 1 GPU is detected
- Global batch size = `BATCH_SIZE × num_replicas`
- Model compiled inside `strategy.scope()`

---

## 7. Evaluation

### 7.1 Validation Batch Inspection

Inline code that runs after training:

1. Predicts on one validation batch
2. Shows first 20 true vs. predicted zones
3. Reports number of unique predicted zones (checks for mode collapse)
4. Computes **grid distance error** (Manhattan distance in row/col space)
5. Shows mean probability assigned to the true class

### 7.2 Classification Metrics

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | Exact zone match |
| **Top-5 Accuracy** | True zone in top 5 predicted zones |
| **Grid Distance Error** | Manhattan distance between predicted and true zone: $\|r_p - r_t\| + \|c_p - c_t\|$ |
| **True Class Probability** | Mean softmax probability assigned to the correct zone |
| **Classification Report** | Per-zone precision / recall / F1 via `sklearn.classification_report` |

### 7.3 Spatial Error Analysis

```python
true_rc = zone_to_rc(true_zones, N_COLS)   # (N, 2) — row/col of true zones
pred_rc = zone_to_rc(pred_zones, N_COLS)   # (N, 2) — row/col of predicted zones
grid_distances = |true_rc - pred_rc|.sum(axis=1)  # Manhattan distance per sample
```

This provides a continuous error measure even for classification, showing how "close" wrong predictions are.

### 7.4 Evaluation Functions

| Function | Backend | Purpose |
|----------|---------|---------|
| `evaluate_with_metrics()` | PyTorch | Returns accuracy, top-3 accuracy, probability matrix |
| `evaluate_detailed()` | PyTorch | Full classification report with per-zone breakdown |
| `evaluate_model()` | PyTorch | Unified eval for both classification and regression |

For Keras, evaluation is done inline using `model.predict()` on test batches with NumPy metric computation.

---

## 8. Prediction & Visualization

### 8.1 Player-Level Prediction

- `predict_next_zones_for_player()`: Extracts a player's last 150 frames → scales features → model inference → returns top-K zones with probabilities
- `predict_for_multiple_players()`: Batch prediction for all players in a team at a given frame

### 8.2 Zone Heatmap Visualization

- `visualize_zone_predictions()`: Renders an `N_ROWS × N_COLS` grid colored by predicted zone probabilities
- Highlights the true zone vs. predicted zone

---

## 9. Alternative Architectures (Defined but Not Active)

The notebook contains several PyTorch model definitions that are commented out but available:

| Model | Description | Backend |
|-------|-------------|---------|
| `EnhancedFootballModel` | Linear → BiLSTM (2 layers, 128 hidden) → MultiheadAttention (8 heads) → classifier | PyTorch |
| `TemporalConvNet` | 3-layer Conv1D [64, 128, 256] with increasing dilation → AdaptiveAvgPool | PyTorch |
| `LocusLabTCN` / `DeltaTCN` | Proper TCN residual blocks with weight normalization, channels [64, 128, 256, 512] | PyTorch |

---

## 10. End-to-End Data Flow

```
combined_matches.csv
    │
    ▼  chunked load (500K rows, memory-monitored)
    │
    ▼  add_velocity_features()  →  24 features
    │
    ▼  add_contextual_features()
    │
    ▼  dropna(FEATURE_COLS)
    │
    ▼  split_by_match_df(fixed_split=True)  →  3/1/1 matches
    │
    ▼  StandardScaler.fit(train_df)
    │
    ▼  keras_sequence_generator(coordinate_targets=False)
    │      target = zone_id at t + 75 frames
    │
    ▼  make_tf_dataset(shuffle=4096, batch=256, repeat for train)
    │
    ▼  compiled_tcn(regression=False, nb_filters=128, kernel_size=7)
    │      TCN(128) → Dense(64, softmax)
    │      compiled with spatial_loss + AdamW(lr=5e-4)
    │
    ▼  train_model()
    │      EarlyStopping, ReduceLROnPlateau, ClassWeights
    │
    ▼  Evaluate
           Top-1/5 accuracy, grid distance, classification report
```

---

## 11. Model Save / Load

| Function | Format |
|----------|--------|
| `ModelCheckpoint` callback | Saves best model as `best_model_mae.keras` during training |
| `model.save()` | Keras `.keras` format |
| `tf.keras.models.load_model()` | Requires custom objects: `TCN`, `ResidualBlock` |

---

*Report generated: February 2026*
*Source notebook: `fyp-ml-improved.ipynb`*
