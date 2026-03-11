# 🎯 MDN Football Player Position Prediction Model

A Mixture Density Network (MDN) for predicting football player trajectories with uncertainty quantification.

## 📌 Project Overview

This model predicts **where a football player will be in 3 seconds** based on their movement history (last 6 seconds). Unlike traditional regression models that output a single prediction, this MDN outputs **multiple possible futures with probabilities**.

> "There's a 45% chance they'll be here, 32% chance they'll be there, and 23% chance they'll be over there"

---

## 🏗️ Model Architecture

| Component | Specification |
|-----------|---------------|
| **Type** | Mixture Density Network (MDN) with TCN backbone |
| **Parameters** | ~2.7 million |
| **Input** | 150 frames × 24 features (6 seconds @ 25 FPS) |
| **Output** | 5 Gaussian components (π, μ, σ for each) |
| **Prediction Horizon** | 3 seconds (75 frames) |

---

## 📊 Understanding the Output Metrics

### 1️⃣ π (Pi) - Mixture Weights

**What it is:** Probability/confidence for each predicted position.

```
┌─────────────────────────────────────────────────────────────┐
│  🎯 "Where will the player go?"                             │
│                                                             │
│  Prediction 1: (0.554, 0.355)  →  43.2% confident (π₁)     │
│  Prediction 2: (0.613, 0.440)  →  21.3% confident (π₂)     │
│  Prediction 3: (0.639, 0.388)  →  16.0% confident (π₃)     │
│  Prediction 4: (0.730, 0.482)  →  10.7% confident (π₄)     │
│  Prediction 5: (0.416, 0.363)  →   8.8% confident (π₅)     │
│                                    ─────────                │
│                          Total:   100.0%                    │
└─────────────────────────────────────────────────────────────┘
```

**Good sign:** Model uses multiple components, recognizing players can go different directions.

---

### 2️⃣ μ (Mu) - Predicted Positions

**What it is:** The actual (x, y) coordinates on the field (normalized 0-1).

```
Field (normalized 0-1):
    
    0,0 ┌──────────────────────────────┐ 1,0
        │                              │
        │       μ₁ ●  ← Most likely    │
        │              (43.2%)         │
        │                   μ₂ ●       │
        │                   (21.3%)    │
        │                              │
    0,1 └──────────────────────────────┘ 1,1
```

**Conversion to real coordinates:**
- x_real = μ_x × 105m (pitch length)
- y_real = μ_y × 68m (pitch width)

---

### 3️⃣ σ (Sigma) - Uncertainty

**What it is:** How "spread out" or uncertain each prediction is.

```
Low σ (0.05):                    High σ (0.25):
┌─────────────┐                  ┌─────────────┐
│             │                  │  ░░░░░░░░░  │
│      ●      │  "I'm pretty    │  ░░░●░░░░░  │  "Could be
│             │   sure here"    │  ░░░░░░░░░  │   anywhere
└─────────────┘                  └─────────────┘   in this area"
```

**Typical values:**
- σ ≈ 0.05-0.10: High confidence (~5-10m uncertainty)
- σ ≈ 0.15-0.25: Moderate confidence (~15-25m uncertainty)
- σ > 0.30: Low confidence (~30m+ uncertainty)

---

## 📈 Performance Metrics

### Negative Log Likelihood (NLL)

The primary training metric for MDN models.

| NLL Value | Interpretation |
|-----------|----------------|
| < 0       | Very good (model is confident and correct) |
| 0 - 2     | Good (predictions are reasonable) |
| > 3       | Needs improvement |

**Lower is better.**

---

### Mean Absolute Error (MAE)

Average distance between predicted and actual positions.

```
Normalized Error → Real-World Distance
─────────────────────────────────────────
   0.10          →    ~10-11 meters
   0.15          →    ~16-17 meters  
   0.20          →    ~21-22 meters
   0.25          →    ~26-27 meters
```

**For 3-second predictions:**

| MAE (normalized) | Verdict |
|------------------|---------|
| < 0.10 | Excellent |
| 0.10 - 0.20 | Good |
| 0.20 - 0.30 | Acceptable |
| > 0.30 | Needs improvement |

---

### Best-of-K MAE

Measures multi-modal prediction quality by taking the closest of K predictions.

```
Best-of-1 MAE:   Only uses the top prediction
Best-of-3 MAE:   Uses the closest of top 3 predictions
Best-of-5 MAE:   Uses the closest of top 5 predictions
```

**If Best-of-K << Best-of-1:** Model successfully captures multiple valid futures.

---

### Component Diversity Metrics

| Metric | Good Range | Bad Sign |
|--------|------------|----------|
| Active components | 2-5 | Only 1 active |
| Mean max weight | < 90% | > 95% (overconfident) |
| Mode collapse rate | < 20% | > 50% |

**Mode collapse** = Model always predicts the same thing (bad).

---

## 🏆 Model Performance Summary

| Metric | Typical Value | Status |
|--------|---------------|--------|
| Active components | 5 | ✅ Good |
| Top prediction confidence | ~40-50% | ✅ Realistic |
| Position MAE | ~0.15-0.25 | ⚠️ Acceptable |
| Uncertainty estimates | Provided | ✅ Valuable |
| Multi-modal outputs | Yes | ✅ Excellent |

---

## 🎮 Example Prediction Output

```
══════════════════════════════════════════════════════════════════════
📍 PLAYER TRAJECTORY: CURRENT → PREDICTED → ACTUAL
══════════════════════════════════════════════════════════════════════

┌─ CURRENT STATE (t = 0s)
│  Position: (0.623, 0.412)
│  [Last frame of 6.0s input sequence]
│
├─ PREDICTED @ t + 3s
│
│  🎯 Best Prediction (Mode):
│     Position: (0.554, 0.355)
│
│  📊 Expected Position (Mean):
│     Position: (0.587, 0.393)
│
│  🔮 Top-5 Possible Futures:
│     1. (0.554, 0.355)  43.2%  ±(0.198, 0.213)
│     2. (0.613, 0.440)  21.3%  ±(0.123, 0.159)
│     3. (0.639, 0.388)  16.0%  ±(0.107, 0.120)
│     4. (0.730, 0.482)  10.7%  ±(0.108, 0.126)
│     5. (0.416, 0.363)   8.8%  ±(0.269, 0.278)
│
└─ ACTUAL @ t + 3s (Ground Truth)
   Position: (0.682, 0.141)

   📏 Prediction Error: 0.2489 (mode) | 0.2687 (expected)
```

---

## 📚 Glossary

| Term | Definition |
|------|------------|
| **MDN** | Mixture Density Network - neural network that outputs probability distributions instead of single values |
| **TCN** | Temporal Convolutional Network - backbone architecture for processing sequential data |
| **π (Pi)** | Mixture weight - probability assigned to each Gaussian component |
| **μ (Mu)** | Mean - the predicted (x, y) position for each component |
| **σ (Sigma)** | Standard deviation - uncertainty/spread of each prediction |
| **GMM** | Gaussian Mixture Model - the output distribution type |
| **NLL** | Negative Log Likelihood - probabilistic loss function |
| **MAE** | Mean Absolute Error - average prediction error distance |
| **Mode** | The most likely prediction (component with highest π) |
| **Expected** | Probability-weighted average of all component means |
| **Mode Collapse** | Failure mode where model only uses one component |
| **Calibration** | How well predicted uncertainties match actual errors |

---

## 🛠️ Usage

### Loading the Model

```python
from your_mdn_module import load_mdn_model, mdn_predict

# Load saved model
model = load_mdn_model('best_mdn_model.keras')

# Run inference
predictions = mdn_predict(model, input_sequence)
pi = predictions['pi']      # (batch, K) - mixture weights
mu = predictions['mu']      # (batch, K, 2) - positions
sigma = predictions['sigma'] # (batch, K, 2) - uncertainties
```

### Getting Top-K Predictions

```python
from your_mdn_module import get_top_k_predictions

top_k = get_top_k_predictions(pi[0], mu[0], sigma[0], k=5)
for pred in top_k:
    print(f"Position: ({pred['x']:.3f}, {pred['y']:.3f})")
    print(f"Probability: {pred['probability']*100:.1f}%")
    print(f"Uncertainty: ±({pred['x_std']:.3f}, {pred['y_std']:.3f})")
```

### Sampling from Distribution

```python
from your_mdn_module import sample_from_mdn

# Generate 100 Monte Carlo samples
samples, component_ids = sample_from_mdn(pi[0], mu[0], sigma[0], num_samples=100)
print(f"Sample mean: ({samples[:, 0].mean():.3f}, {samples[:, 1].mean():.3f})")
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `fyp-ml-improved.ipynb` | Main training and evaluation notebook |
| `best_mdn_model.keras` | Saved trained model |
| `preprocessed_data/sequences.h5` | Preprocessed training data |
| `combined_matches.csv` | Raw match data |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Configuration

Key hyperparameters in the notebook:

```python
# Sequence configuration
SEQ_LEN = 150           # Input frames (6 seconds)
HORIZON_FRAMES = 75     # Prediction horizon (3 seconds)
FPS = 25                # Frame rate

# MDN configuration
MDN_NUM_MIXTURES = 5    # Number of Gaussian components
MDN_OUTPUT_DIM = 2      # Output dimensions (x, y)
MDN_SIGMA_MIN = 0.01    # Minimum sigma (prevents collapse)

# Training configuration
BATCH_SIZE = 256
EPOCHS = 50
LR = 0.001
PATIENCE = 5            # Early stopping patience
```

---

## 🎯 Practical Applications

| Use Case | Suitability |
|----------|-------------|
| Zone-level movement prediction | ✅ Excellent |
| Tactical analysis (runs, coverage) | ✅ Very good |
| Player tracking assistance | ✅ Good |
| Real-time decision support | ⚠️ Moderate (latency) |
| Exact positioning (offside calls) | ❌ Not precise enough |

---

## 📝 Notes

- **3-second predictions** are inherently uncertain due to player decision-making
- ~25m average error is **reasonable** for this horizon in a dynamic sport
- The multi-modal output is the **key advantage** - it captures multiple valid futures
- Uncertainty estimates help you **trust predictions appropriately**

---

## 📄 License

[Your License Here]

## 👤 Author

[Your Name/Info Here]
