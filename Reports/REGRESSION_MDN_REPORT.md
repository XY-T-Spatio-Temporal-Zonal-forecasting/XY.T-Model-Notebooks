# Regression & MDN Report: Probabilistic Coordinate Prediction

## 1. Problem Statement

Predict the **future (x, y) coordinates** of a football player after a 3-second time horizon. Two sub-approaches exist:

1. **Deterministic regression**: Predict a single `(x, y)` point using MSE loss
2. **Mixture Density Network (MDN)**: Predict a **probability distribution** over future positions as a Gaussian Mixture Model — captures multi-modal uncertainty (e.g., a player could cut left or continue right)

The MDN approach (`MODEL_TYPE = "MDN_TCN"`) is the primary focus.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| `MODEL_TYPE` | `"MDN_TCN"` |
| `BACKEND` | `"keras"` |
| `CO_ORDINATES` | `True` (auto-forced when MDN) |
| `SEQ_LEN` | 150 frames (6 seconds) |
| `FPS` | 25 |
| `HORIZON_SECONDS` | 3 seconds |
| `HORIZON_FRAMES` | 75 |
| `BATCH_SIZE` | 256 |
| `MDN_LEARNING_RATE` | 1e-4 |
| `MDN_NUM_MIXTURES` (K) | 5 |
| `MDN_SIGMA_MIN` | 1e-4 |
| `MDN_OUTPUT_DIM` | 2 (x, y) |
| `EPOCHS` | 20 |
| `PATIENCE` | 3 |

When `MODEL_TYPE == "MDN_TCN"`, the notebook automatically sets `CO_ORDINATES = True` to ensure the data pipeline produces coordinate targets instead of zone IDs.

---

## 3. Data Pipeline

The data loading, feature engineering (24 features), splitting, and scaling are **identical** to the classification pipeline (see Classification Report §3). The key differences are in target construction and dataset types.

### 3.1 Coordinate Targets

In `keras_sequence_generator()` when `coordinate_targets=True` (auto-set for MDN):

- **Target**: absolute `(x_normalized, y_normalized)` at frame $t_{\text{current}} + \text{HORIZON\_FRAMES}$
- **Output shape per sample**: `(2,)` float32
- These are **absolute positions** in normalized $[0, 1]$ coordinates, not deltas or displacements

### 3.2 tf.data Pipeline Differences from Classification

| Setting | Classification | Regression / MDN |
|---------|---------------|------------------|
| Target dtype | `tf.int32` (zone ID) | `tf.float32` (coordinates) |
| Target shape | `()` scalar | `(2,)` vector |
| Class weights | Applied | Not used |
| Training repeat | `.repeat()` | `.repeat()` |
| `steps_per_epoch` | `len(train_df) // BATCH_SIZE` | Same |

---

## 4. Deterministic Regression Mode (Keras TCN)

When `MODEL_TYPE = "Keras_tcn"` and `CO_ORDINATES = True`:

### 4.1 Architecture

Same TCN backbone as classification, but with a linear regression head:

```
Input (B, 150, 24) → TCN(128) → Dense(2, linear) → (B, 2)
```

### 4.2 Compilation

| Component | Value |
|-----------|-------|
| **Optimizer** | `Adam(lr=1e-4, clipnorm=1.0)` + `LossScaleOptimizer` for mixed precision |
| **Loss** | `'mse'` (Mean Squared Error) |
| **Metrics** | `'mae'`, `RootMeanSquaredError` |

### 4.3 Available Custom Losses (Not Default)

| Loss | Description |
|------|-------------|
| `huber_loss_tf(delta=0.1)` | Huber loss — uses linear penalty beyond delta, less sensitive to outliers than MSE |
| `weighted_mse_loss()` | Weights x-axis errors 1.2× and y-axis 0.8× (x-direction more critical in football) |

### 4.4 Training Callbacks

| Callback | Configuration |
|----------|---------------|
| `EarlyStopping` | Monitor: `val_mae`, mode: `min`, patience: 5 |
| `ReduceLROnPlateau` | Monitor: `val_mae`, factor: 0.5, patience: 3, min_lr: 1e-6 |
| `ModelCheckpoint` | Monitor: `val_mae`, mode: `min`, saves to `best_model_mae.keras` |

No class weights are used in regression mode.

---

## 5. MDN Architecture: TCN Encoder + Mixture Density Head

### 5.1 Overall Architecture

```
Input (B, 150, 24)
    │
    ▼
TCN Encoder
    │  128 filters, 2 stacks × 6 dilations
    │  causal padding, skip connections, BatchNorm
    │  return_sequences=False
    │  output: (B, 128)
    ▼
Dense(256, relu) → Dropout(0.3)
    │
    ▼
Dense(128, relu) → Dropout(0.3)
    │
    ▼
MDNLayer
    ├── π:  (B, 5)      mixture weights     [softmax]
    ├── μ:  (B, 5, 2)   component means     [unconstrained]
    └── σ:  (B, 5, 2)   component std devs  [softplus + σ_min]
```

### 5.2 TCN Encoder

Identical architecture to the classification TCN — uses the same `TCN` and `ResidualBlock` classes.

| Parameter | Value |
|-----------|-------|
| `nb_filters` | 128 |
| `kernel_size` | 7 |
| `dilations` | [1, 2, 4, 8, 16, 32] |
| `nb_stacks` | 2 |
| `padding` | `'causal'` |
| `use_skip_connections` | True |
| `return_sequences` | False |
| `dropout_rate` | 0.3 |
| `use_batch_norm` | True |
| `kernel_initializer` | `'he_normal'` |

### 5.3 Dense Refinement Layers

Two fully-connected layers transform the TCN's temporal summary into a representation suitable for predicting mixture parameters:

- `Dense(256, activation='relu')` → `Dropout(0.3)`
- `Dense(128, activation='relu')` → `Dropout(0.3)`

### 5.4 MDN Output Layer (`MDNLayer`)

Custom Keras layer that outputs parameters for $K = 5$ Gaussian mixture components:

| Output | Shape | Activation | Purpose |
|--------|-------|------------|---------|
| $\boldsymbol{\pi}$ **(pi)** | `(B, 5)` | `softmax` on logits | Mixture weights (sum to 1) |
| $\boldsymbol{\mu}$ **(mu)** | `(B, 5, 2)` | None (unconstrained) | Component means in (x, y) space |
| $\boldsymbol{\sigma}$ **(sigma)** | `(B, 5, 2)` | `softplus` $+ \sigma_{\min}$ | Component std deviations (strictly positive) |

**Internal dense layers:**

| Layer | Units | Post-processing |
|-------|-------|-----------------|
| `pi_dense` | $K = 5$ | `softmax` → `(B, 5)` |
| `mu_dense` | $K \times 2 = 10$ | `reshape` → `(B, 5, 2)` |
| `sigma_dense` | $K \times 2 = 10$ | `softplus` → $+ 10^{-4}$ → `reshape` → `(B, 5, 2)` |

The model returns outputs as a **dictionary**: `{'pi': ..., 'mu': ..., 'sigma': ...}`

### 5.5 Model Construction (`build_tcn_mdn_model()`)

Built using the Keras Functional API:

1. `Input(shape=(150, 24))`
2. `TCN(...)` encoder block
3. Dense refinement: 256 → 128 with dropout
4. `MDNLayer(num_mixtures=5, output_dim=2)`
5. `Model(inputs, outputs={'pi', 'mu', 'sigma'})`

The model is **not compiled** — it uses a custom training loop via `MDNTrainer` because the loss depends on three separate output tensors.

---

## 6. MDN Loss Function

### 6.1 Negative Log-Likelihood (NLL) of a Gaussian Mixture

The MDN is trained by minimizing the negative log-likelihood:

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(y_i \mid \mu_k, \sigma_k)$$

**`mdn_loss_fn(y_true, pi, mu, sigma)` — step by step:**

1. **Expand targets**: $y_{\text{true}}$ from `(B, 2)` → `(B, 1, 2)` for broadcasting against $K$ components

2. **Per-dimension log Gaussian**:

$$\log \mathcal{N}(y \mid \mu, \sigma) = -\frac{1}{2}\log(2\pi) - \log(\sigma + \epsilon) - \frac{(y - \mu)^2}{2\sigma^2}$$

3. **Sum over dimensions** (x, y) → joint log probability per component: `(B, K)`

4. **Add log mixture weights**: $\log p_k = \log \mathcal{N}_k + \log(\pi_k + \epsilon)$

5. **Log-sum-exp trick** for numerical stability:

$$\log \sum_k \exp(a_k) = a_{\max} + \log \sum_k \exp(a_k - a_{\max})$$

6. **Return**: $\text{mean}(-\log p)$ over the batch

$\epsilon = 10^{-10}$ is added to `log(sigma)` and `log(pi)` to guard against $\log(0)$.

### 6.2 Mixed Precision Safety

All inputs are explicitly cast to `float32` at the start of `mdn_loss_fn()` to avoid FP16 instability in log/exp operations.

### 6.3 `MDNLoss` Keras Wrapper

A `tf.keras.losses.Loss` subclass exists that expects concatenated predictions `[pi, mu_flat, sigma_flat]`, splits and reshapes internally. Available for `model.compile()` but **not used** — the custom training loop applies `mdn_loss_fn` directly, which is cleaner for dict-output models.

---

## 7. Custom Training Loop (`MDNTrainer`)

MDN uses a custom training loop because the loss depends on **three separate output tensors** ($\pi$, $\mu$, $\sigma$) returned as a dictionary — this is not easily handled by `model.fit()`.

### 7.1 Optimizer

- `Adam(learning_rate=1e-4, clipnorm=1.0)`
- Gradient clipping at norm 1.0 prevents exploding gradients from the NLL loss

### 7.2 Learning Rate Schedules

Three options via the `lr_schedule` parameter:

| Schedule | Behavior |
|----------|----------|
| **`'cosine'`** (default) | Linear warmup for 2 epochs → cosine decay to $lr \times 0.01$ |
| **`'constant'`** | Fixed LR throughout training |
| **`'step'`** | Halves LR after 3 consecutive epochs without val improvement |

#### Cosine Schedule (Default) — Mathematical Details

**Warmup phase** (epochs 0–1):

$$lr = lr_{\text{init}} \times \frac{\text{epoch} + 1}{\text{warmup\_epochs}}$$

**Decay phase** (epochs 2–19):

$$lr = lr_{\min} + (lr_{\max} - lr_{\min}) \times \frac{1 + \cos(\pi \cdot p)}{2}$$

Where:
- $p = \frac{\text{epoch} - \text{warmup\_epochs}}{\text{total\_epochs} - \text{warmup\_epochs}}$ (normalized progress)
- $lr_{\min} = lr_{\text{init}} \times 0.01 = 10^{-6}$
- $lr_{\max} = lr_{\text{init}} = 10^{-4}$

### 7.3 Training Step (`@tf.function` decorated)

```python
with GradientTape() as tape:
    outputs = model(x_batch, training=True)  # → {pi, mu, sigma}
    loss = mdn_loss_fn(y_batch, outputs['pi'], outputs['mu'], outputs['sigma'])
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### 7.4 Session Management

Before training, `tf.keras.backend.clear_session()` + `gc.collect()` clears stale graph resources, then the model is **rebuilt fresh** to prevent `AlreadyExistsError` from `@tf.function`-traced named graph resources.

### 7.5 Early Stopping

- Monitors validation NLL (lower is better)
- Patience: 3 epochs
- Saves best weights via `model.get_weights()` and restores them at the end via `model.set_weights()`

### 7.6 Progress Tracking

- `tf.keras.utils.Progbar` with live `loss` and `avg_loss` (running average of last 100 batches)
- Epoch summary printed after each epoch: elapsed time, train NLL, val NLL, current LR, best/patience marker
- History dictionary: `{'train_loss': [...], 'val_loss': [...], 'lr': [...]}`

---

## 8. Training Pipeline (`train_mdn_model()`)

Unified function orchestrating the full MDN training process:

1. **Auto-detect** `num_features` from first batch (if not provided)
2. **Build model** via `build_tcn_mdn_model()` or use pre-built model
3. **Create trainer**: `MDNTrainer(lr_schedule='cosine', warmup_epochs=2, min_lr_ratio=0.01)`
4. **Train**: `trainer.fit(steps_per_epoch=len(train_df) // BATCH_SIZE)`
5. **Print summary**: final/best NLL, epochs trained, final LR

### Actual Training Cell Parameters

| Parameter | Value |
|-----------|-------|
| `tcn_filters` | 128 |
| `tcn_kernel_size` | 7 |
| `tcn_dilations` | [1, 2, 4, 8, 16, 32] |
| `tcn_stacks` | 2 |
| `dropout_rate` | 0.3 |
| `use_batch_norm` | True |
| `dense_units` | 256 |
| `learning_rate` | 1e-4 |
| `lr_schedule` | `'cosine'` |
| `warmup_epochs` | 2 |
| `min_lr_ratio` | 0.01 |
| `epochs` | 20 |
| `patience` | 3 |

---

## 9. MDN Inference Utilities

### 9.1 Core Prediction Functions

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `mdn_predict(model, x)` | `(150, 24)` or `(B, 150, 24)` | `{pi, mu, sigma}` numpy | Forward pass wrapper |
| `get_expected_position(pi, mu)` | `(K,), (K,2)` | `(2,)` | Weighted mean: $\mathbb{E}[x] = \sum_k \pi_k \mu_k$ |
| `get_most_likely_position(pi, mu)` | `(K,), (K,2)` | `(2,)` | Mode: $\mu$ of the highest-weight component |
| `get_top_k_predictions(pi, mu, sigma, k)` | Single sample params | List of dicts | Top-K positions with probability and uncertainty |
| `sample_from_mdn(pi, mu, sigma, N)` | Single sample params | `(N, 2)` | Monte Carlo sampling from the GMM |
| `batch_predictions_to_dataframe(preds)` | Batch dict | DataFrame | Exploded view: one row per (sample, component) |

### 9.2 Top-K Prediction Output Format

Each prediction dictionary contains:

```python
{
    'probability': float,   # mixture weight π_k
    'x': float,             # predicted x position (normalized)
    'y': float,             # predicted y position (normalized)
    'x_std': float,         # uncertainty in x (σ_x)
    'y_std': float,         # uncertainty in y (σ_y)
    'component': int        # which Gaussian component index
}
```

### 9.3 Sampling from the GMM

`sample_from_mdn()` draws N samples from the learned distribution:

1. For each sample, choose a component $k$ with probability $\pi_k$
2. Draw $(x, y)$ from $\mathcal{N}(\mu_k, \sigma_k)$
3. Return `(N, 2)` array of sampled positions

This is useful for Monte Carlo uncertainty estimation and calibration checks.

---

## 10. MDN Evaluation Metrics

### 10.1 Metric Suite

| Metric | Function | Description |
|--------|----------|-------------|
| **NLL** | `evaluate_mdn_nll()` | Mean Negative Log-Likelihood on test set — **primary metric** |
| **Best-of-K MAE** | `evaluate_best_of_k_mae(k=3)` | Minimum Euclidean distance among top-K component means to true position — rewards multi-modality |
| **Expected MAE** | (same function) | $\lVert \mathbb{E}[x] - x_{\text{true}} \rVert$ — error of probability-weighted mean |
| **Mode MAE** | (same function) | Distance from highest-weight component mean to truth |
| **Component Diversity** | `evaluate_component_diversity(threshold=0.1)` | Active components ($\pi > 0.1$), entropy of $\pi$, max weight, mode-collapse ratio ($\%$ with max $\pi > 0.9$) |
| **Calibration** | `evaluate_calibration()` | Sampling-based: checks if X% of true positions fall within X% confidence regions (1000 MC samples per prediction, capped at 100 test samples) |

### 10.2 `full_mdn_evaluation()`

Orchestrates all evaluation metrics in sequence with formatted output:

1. NLL evaluation on test set
2. Best-of-K, Expected, and Mode MAE
3. Component diversity analysis
4. Calibration check at percentiles [10, 25, 50, 75, 90]

### 10.3 Validation Batch Inspection

An inline cell that runs after training:

- Gets $\pi$, $\mu$, $\sigma$ from the trained model on a validation batch
- Shows top-3 predictions per sample with coordinates, probabilities, and uncertainties  
- Computes batch-level Best-of-3 MAE, Expected MAE, Mode MAE
- Reports average active components and average max weight (mode collapse check)

### 10.4 Why Traditional Regression Metrics Are Insufficient

| Metric | Problem for MDN |
|--------|----------------|
| **MAE** | Only measures distance to a single prediction; ignores multi-modality |
| **RMSE** | Penalizes outliers but doesn't reward correct uncertainty estimates |
| **MSE** | Cannot evaluate whether the predicted distribution is well-calibrated |
| **Accuracy** | Not applicable to continuous coordinate outputs |

NLL is the proper metric because it evaluates the **entire predicted distribution**, rewarding both accurate means and well-calibrated uncertainties.

---

## 11. MDN Visualization

### 11.1 Available Functions

| Function | Purpose |
|----------|---------|
| `plot_mdn_prediction_on_field()` | Football field (green background, lines, penalty areas) with mixture components rendered as **2σ confidence ellipses**. Color intensity proportional to weight. Probability labels per component. True position marked with red X. |
| `plot_mdn_training_history()` | 2-panel figure: (1) NLL loss curves for train/val, (2) overfitting monitor showing val − train gap over epochs |
| `plot_mixture_weights_distribution()` | Box plot of $\pi$ per component + histogram of max weight across the test set (mode collapse analysis) |
| `plot_prediction_samples()` | Scatter of 200 Monte Carlo samples from the GMM, colored by component. Component means marked with `+`, true position as `X` |

### 11.2 Field Visualization Details

`plot_mdn_prediction_on_field()` draws:

- Green pitch background with white boundary lines
- Center circle and center line
- Penalty areas and goal areas (both ends)
- Penalty spots
- For each mixture component ($\pi > 0.05$):
  - Ellipse at $(\mu_x, \mu_y)$ with width $= 2\sigma_x$ and height $= 2\sigma_y$
  - Alpha transparency proportional to $\pi_k$
  - Probability label in percent

---

## 12. Model Save / Load

| Function | Details |
|----------|---------|
| `save_mdn_model(model, path)` | `model.save()` in Keras `.keras` format |
| `load_mdn_model(path)` | Loads with custom objects: `MDNLayer`, `TCN`, `ResidualBlock` |

---

## 13. Deterministic Regression vs. MDN Comparison

| Aspect | Deterministic Regression | MDN (Gaussian Mixture) |
|--------|--------------------------|------------------------|
| **Output** | Single `(x, y)` point | K=5 weighted Gaussians |
| **Loss** | MSE | NLL of GMM |
| **Training** | `model.fit()` | Custom `MDNTrainer` with `GradientTape` |
| **Optimizer** | Adam + `LossScaleOptimizer` (FP16) | Adam (clipnorm=1.0) + cosine LR schedule |
| **Captures uncertainty** | No | Yes ($\sigma$ per component) |
| **Multi-modal** | No | Yes (multiple possible futures with probabilities) |
| **Primary metric** | MAE, RMSE | NLL, Best-of-K MAE, calibration |
| **Visualization** | Single predicted point | Probability ellipses on pitch |
| **Key advantage** | Simple, fast training | Models ambiguous situations (e.g., player at a decision point) |

---

## 14. End-to-End Data Flow (MDN)

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
    ▼  keras_sequence_generator(coordinate_targets=True)
    │      target = absolute (x, y) at t + 75 frames
    │
    ▼  make_tf_dataset(batch=256, train repeats)
    │
    ▼  build_tcn_mdn_model(K=5, filters=128)
    │      TCN(128) → Dense(256) → Dense(128) → MDNLayer
    │
    ▼  MDNTrainer(cosine LR, warmup=2, clipnorm=1.0)
    │      train with NLL loss, early stopping (patience=3)
    │
    ▼  full_mdn_evaluation()
    │      NLL, Best-of-K MAE, diversity, calibration
    │
    ▼  Visualize
           plot_mdn_prediction_on_field()
           plot_training_history()
           plot_mixture_weights_distribution()
```

---

## 15. Mixed Precision

- Global policy: `mixed_float16` via `tf.keras.mixed_precision`
- MDN loss function explicitly casts all inputs to `float32` to avoid FP16 instability in log/exp operations
- Deterministic regression uses `LossScaleOptimizer` for FP16 training speedup

---

*Report generated: February 2026*
*Source notebook: `fyp-ml-improved.ipynb`*
