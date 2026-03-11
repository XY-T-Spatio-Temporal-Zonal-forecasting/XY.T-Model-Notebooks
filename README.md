# XY.T — Football Player Trajectory Forecasting

A full-stack sports analytics system that predicts where football players will be **3 seconds into the future** using deep learning. The system combines Temporal Convolutional Networks (TCN), Graph Neural Networks (GNN), and Mixture Density Networks (MDN) to classify player positions into a **3×9 zone grid** (27 zones) across a full pitch.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  XY.T Prediction Pipeline                   │
│                                                             │
│  Raw Tracking Data  →  Preprocessing  →  ML Models         │
│         ↓                                                   │
│  FastAPI Backend  ←→  PostgreSQL DB                         │
│         ↓                                                   │
│  React Dashboard  (pitch viz + tactical AI via Gemini)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
MODEL/
├── Notebook Main/              # Training notebooks & preprocessing scripts
│   ├── Data_preprocessing.py   # Feature engineering pipeline
│   ├── TCN_Notebook.ipynb      # TCN zone-classification training
│   ├── GNN_Notebook.ipynb      # GNN-TCN spatio-temporal training
│   ├── MDN-TCN.ipynb           # Mixture Density Network training
│   └── preprocessed_data/      # Cached train/val/test DataFrames

├── Reports/                    # Model evaluation reports
└── requirements.txt            # Python dependencies
```

---

Datasets can be found in https://www.kaggle.com/hashirhalaldeen/datasets

## Models

### 1. TCN Zone Classifier (`TCN-Classification/`)
Predicts the most likely zone(s) a player will occupy 3 seconds ahead.

| Parameter | Value |
|---|---|
| Architecture | Temporal Convolutional Network |
| Input | 30 frames (3s history) × 26 features |
| Output | 27-class softmax (3×9 zone grid) |
| Best MC Loss | 0.050 |

### 2. GNN-TCN Spatio-Temporal Model (`Notebook Main/GNN_Notebook.ipynb`)
Models the full team as a graph — each player is a node with edges to teammates and opponents.

| Parameter | Value |
|---|---|
| Architecture | GraphSAGE (2 layers) + TCN |
| Input Nodes | 10 (5 teammates + 5 nearest opponents) |
| Input Features | 26 per node |
| Output | 27 zones × 5 players |
| Sequence Length | 30 frames @ 10 FPS |

### 3. MDN-TCN (`MDN/`)
Mixture Density Network for probabilistic coordinate regression — outputs a distribution over future positions rather than a single point.

| Parameter | Value |
|---|---|
| Architecture | TCN encoder + MDN output head |
| Output | Gaussian mixture (coordinates) |
| Best NLL | 1.22 |

---

## Features

- **Zone Classification** — Top-K probable zones 3s ahead on a 3×9 pitch grid
- **Timeline Forecasting** — Per-second forecast sequence for animated playback
- **Tactical Analysis** — Google Gemini LLM integration for real-time coaching insights
- **Interactive Dashboard** — SVG pitch visualization with player trajectories and zone heatmaps
- **Match & Player Data** — Full match roster, tracking data, and historical timelines from PostgreSQL

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL database with match tracking data
- Preprocessed DataFrames from [Kaggle](https://www.kaggle.com/datasets/hashirhalaldeen/dataframes)

---

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Preprocessed Data

Download the preprocessed DataFrames from Kaggle(https://www.kaggle.com/hashirhalaldeen/datasets) and place them in:

```
backend/app/data/preprocessed_data/
Notebook Main/preprocessed_data/
```

---

## Training

All training notebooks are in `Notebook Main/`. Run them in Jupyter after generating preprocessed data via `Data_preprocessing.ipynb`.

| Notebook | Model |
|---|---|
| `TCN_Notebook.ipynb` | TCN zone classifier |
| `GNN_Notebook.ipynb` | GNN-TCN spatio-temporal |
| `MDN-TCN.ipynb` | Mixture Density Network |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | TensorFlow 2.10 / Keras |
| Backend | FastAPI + SQLAlchemy + Pydantic v2 |
| Database | PostgreSQL |
| Frontend | React 19 + Vite + Tailwind CSS |
| Animation | Framer Motion |
| Tactical AI | Google Gemini (REST) |
| Data | NumPy, Pandas, Scikit-learn |

---

## Reports

Detailed model evaluation reports are in the [`Reports/`](Reports/) directory:

- [Classification Report](Reports/CLASSIFICATION_REPORT.md)
- [MDN Report](Reports/MDN_REPORT.md)
- [Regression MDN Report](Reports/REGRESSION_MDN_REPORT.md)
