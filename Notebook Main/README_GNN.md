# GNN-TCN Spatio-Temporal Model

This model implements a hybrid Graph Neural Network (GNN) and Temporal Convolutional Network (TCN) architecture to predict football player movement (zones) 3 seconds into the future, incorporating both teammate and opponent positions.

| Parameter           | Value                                        |
| ------------------- | -------------------------------------------- |
| Model Type          | GNN (GraphSAGE-style) + TCN                  |
| Input Nodes         | 10 (5 Teammates + 5 Nearest Opponents)      |
| Input Features      | 26 (Normalized coords, speed, accel, etc.)   |
| Output Classes      | 27 (3x9 Zone Grid)                           |
| Prediction Horizon  | 3.0 Seconds (30 frames @ 10 FPS)             |
| Sequence Length     | 30 Frames (3.0s history)                     |

## Architecture Details

| Component                      | Value                         |
| ------------------------------ | ----------------------------- |
| **GNN Layers**                 | 2 Stacks (Hidden: 32, Out: 32) |
| GNN Aggregation                | Mean (GraphSAGE neighbor/self) |
| **TCN Filters**                | 64                            |
| TCN Kernel Size                | 3                             |
| TCN Dilations                  | [1, 2, 4, 8, 16]              |
| **Regularization**             | Dropout (0.4) + L2 (1e-4)     |
| **Output Head**                | Dense(32, ReLU) → Dense(27)   |

## Data Preparation (Overfitting Mitigation)

- **Window Stride:** `WINDOW_STEP = 10` (1.0s). Previously used stride-1, which created ~47k highly correlated sequences. Moving to stride-10 reduces data redundancy and improves validation generalization.
- **Class Balancing:** Implements calculated class weights for the 27 zones to address spatial imbalance (higher weights for rare sideline zones).

## Node & Graph Configuration

- **Teammates (Nodes 0-4):** Primary prediction targets.
- **Opponents (Nodes 5-9):** Contextual nodes (nearest 5 opponents) used to capture defensive pressure.
- **Adjacency:** K-Nearest Neighbors ($k=3$) based on global average training positions to define spatial information flow.

## Training Configuration

| Setting         | Value                                          |
| --------------- | ---------------------------------------------- |
| Loss Function   | Sparse Categorical Crossentropy                |
| Label Smoothing | 0.1 (Anti-imbalance)                          |
| From Logits     | True (during training) / False (inference)     |
| Optimizer       | Adam                                           |
| Learning Rate   | 1e-3 (ReduceLROnPlateau factor 0.5)            |
| Batch Size      | 64                                             |
| Epochs          | 100 (EarlyStopping patience: 20)               |

## Data Preparation (Overfitting Mitigation)

- **Window Stride:** `WINDOW_STEP = 10` (1.0s). Previously used stride-1, which created ~47k highly correlated sequences. Stride-10 significantly reduces sample redundancy.
- **Label Smoothing:** Replaced explicit class weights with label smoothing (0.1) to handle class imbalance more robustly and prevent gradient scale distortion.
- **Improved LR Schedule:** Increased `ReduceLROnPlateau` patience to 6 to allow the model more time to find local minima.

## Evaluation Metrics

- **Top-1 Accuracy:** Standard zone prediction.
- **Top-3/Top-5 Accuracy:** Spatial proximity success (is the player in the likely neighborhood).
- **Inference:** Supports per-player, per-match, and per-second trajectory queries.
