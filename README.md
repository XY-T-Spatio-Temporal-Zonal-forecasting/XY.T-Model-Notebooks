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
│
├── backend/                    # FastAPI serving layer
│   ├── main.py                 # App entry point & lifespan startup
│   └── app/
│       ├── core/               # Settings & env config
│       ├── data/               # Data loading & pipeline
│       ├── inference/          # InferenceService + custom TCN layers
│       ├── models/             # Pydantic schemas
│       ├── routers/            # REST endpoints
│       ├── services/           # DB, data, model loader, tactical AI
│       └── utils/
│
├── football-dashboard/         # React + Tailwind CSS frontend
│   └── src/
│       └── components/         # Pitch viz, player tracking, tactical coach
│
├── current_best_models/        # Production-ready saved models
├── TCN-Classification/         # TCN classification model checkpoints
├── TCN-Regression/             # TCN regression model checkpoints
├── MDN/                        # MDN model checkpoints
├── GRU-Ensemble/               # GRU ensemble experiments
├── preprocessed_data/          # Root-level preprocessed data cache
├── Reports/                    # Model evaluation reports
├── combined_matches.csv        # Raw combined match tracking data
└── requirements.txt            # Python dependencies
```

---

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

For the backend specifically:

```bash
cd backend
pip install -r requirements.txt
```

---

### 2. Set Up Preprocessed Data

Download the preprocessed DataFrames from Kaggle and place them in:

```
backend/app/data/preprocessed_data/
Notebook Main/preprocessed_data/
```

---

### 3. Configure Environment Variables

Create a `.env` file inside `backend/`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/football_db
TCN_MODEL_PATH=app/inference/models/TCN-Classification/<model>.keras
GEMINI_API_KEY=your_gemini_api_key
DEBUG=true
```

---

### 4. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

### 5. Start the Frontend

```bash
cd football-dashboard
npm install
npm run dev
```

Dashboard available at: `http://localhost:5173`

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
