# Testing Models Comprehensive Report
*Auto-generated from experiment folders under `testings/testings`*

## Executive Summary

**Total Models Evaluated:**
- Classification: 13 models (15 runs)
- Regression: 2 models (2 runs)
- Total Dataset Samples: 5,017,029
- Sequence Length: 100 frames | Time Horizon: 3s

## Classification Models (Run-level Summary with Configuration)

| model_folder | run_id | seq_len | filters | Features | best_epoch | best_val_top1 | best_val_top3 | best_val_top5 | best_val_loss | final_overfit_gap_top1 |
|:---|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| classi_model_11_1s_best | 1 | 50 | - | ok | 11 | 0.7412 | 0.9206 | 0.9503 | 0.9791 | 0.1269 |
| classi_model_9_best | 1 | 50 | - | ok | 19 | 0.6266 | 0.9046 | 0.9552 | 1.4667 | -0.0196 |
| classi_model_2 | 1 | 50 | - | ok | 15 | 0.6181 | 0.9045 | 0.9557 | 1.5858 | -0.0095 |
| classi_model_2_val_acc_61 | 1 | 50 | - | ok | 15 | 0.6181 | 0.9045 | 0.9557 | 1.5858 | -0.0095 |
| classi_model_10 | 1 | 50 | - | 42 | 4 | 0.6134 | 0.9000 | 0.9535 | 0.0317 | -0.0053 |
| classi_model_7 | 1 | 50 | - | ok | 4 | 0.6083 | 0.9012 | 0.9556 | 1.5304 | -0.0003 |
| classi_model_8 | 1 | 50 | - | ok | 4 | 0.6083 | 0.9012 | 0.9556 | 1.5304 | -0.0003 |
| classi_model_5 | 1 | 50 | - | ok | 0 | 0.5745 | 0.8866 | 0.9513 | 3.8378 | 0.0050 |
| classi_model_6 | 1 | 50 | - | ok | 0 | 0.5745 | 0.8866 | 0.9513 | 3.8378 | 0.0050 |
| classi_model_4 | 3 | 50 | - | ok | 0 | 0.5600 | 0.8759 | 0.9479 | 0.0217 | 0.0093 |
| classi_model_5 | 3 | 50 | - | ok | 0 | 0.5600 | 0.8759 | 0.9479 | 0.0217 | 0.0093 |
| classi_model_3 | 1 | 50 | - | ok | 6 | 0.5254 | 0.8711 | 0.9440 | 1.8099 | 0.0406 |
| classi_model_4 | 1 | 50 | - | ok | 3 | 0.5158 | 0.8569 | 0.9400 | 1.4691 | -0.0049 |
| classi_0 | classification-focal_loss | 50 | - | 42 | 0 | 0.1276 | 0.3447 | 0.4775 | 0.6903 | 0.1045 |
| classi_model_1 | 1 | 50 | - | 42 | 0 | 0.0659 | 0.1861 | 0.2993 | 8.1205 | 0.0009 |

### Classification Models (Best Run Per Model with Configuration)

| model_folder | run_id | seq_len | filters | kernel | stacks | best_epoch | best_val_top1 | best_val_top3 | best_val_top5 | best_val_loss | Features | Loss type | Batch| Accuracy | Macro-Average | Weighted average 
|:---|:---|:---|:---|:---|:---|---:|---:|---:|---:|---:|:---|:---|---:|:---|:---|---:|
| classi_model_11_1second_best | 1 | 50 | 128 | 5 | 2 | 11 | 0.7412 | 0.9206 | 0.9503 | 0.9791 | 26 | S CCE | 1024 | 26 | S CCE | 1024 |
| classi_model_9_best | 1 | 50 | 96 | 9 | 2 | 19 | 0.6266 | 0.9046 | 0.9552 | 1.4667 | 24 | Label S S CE | 1024 | 26 | S CCE | 1024 |
| classi_model_2 | 1 | 50 | 64 | 5 | 3 | 15 | 0.6181 | 0.9045 | 0.9557 | 1.5858 | 42 | Label S S CE | 1024 | 26 | S CCE | 1024 |
| classi_model_2_val_acc_61 | 1 | 50 | - | - | - | 15 | 0.6181 | 0.9045 | 0.9557 | 1.5858 | 42 | - | - | 26 | S CCE | 1024 |
| classi_model_10 | 1 | 50 | 96 | 9 | 2 | 4 | 0.6134 | 0.9000 | 0.9535 | 0.0317 | 24 | focal loss | 512 | 26 | S CCE | 1024 |
| classi_model_7 | 1 | 50 | 96 | 5 | 3 | 4 | 0.6083 | 0.9012 | 0.9556 | 1.5304 | 42 | Focal loss | 512 | 26 | S CCE | 1024 |
| classi_model_8 | 1 | 50 | 96 | 5 | 3 | 4 | 0.6083 | 0.9012 | 0.9556 | 1.5304 | 42 | Focal loss | 512 | 26 | S CCE | 1024 |
| classi_model_5 | 1 | 50 | 96 | 5 | 3 | 0 | 0.5745 | 0.8866 | 0.9513 | 3.8378 | 42 | Focal loss | 512 | 26 | S CCE | 1024 |
| classi_model_6 | 1 | 50 | 96 | 5 | 3 | 0 | 0.5745 | 0.8866 | 0.9513 | 3.8378 | 42 | Focal loss | 512 | 26 | S CCE | 1024 |
| classi_model_4 | 3 | 50 | 96 | 5 | 3 | 0 | 0.5600 | 0.8759 | 0.9479 | 0.0217 | 42 | Focal loss | 512 | 26 | S CCE | 1024 |
| classi_model_3 | 1 | 50 | 64 | 5 | 3 | 6 | 0.5254 | 0.8711 | 0.9440 | 1.8099 | 42 | Label S S CE | 1024 | 26 | S CCE | 1024 |
| classi_0 | classification-focal_loss | 50 | 64 | 5 | 3 | 0 | 0.1276 | 0.3447 | 0.4775 | 0.6903 | 42 | Label S S CE | 1024 | 26 | S CCE | 1024 |
| classi_model_1 | 1 | 50 | 64 | 5 | 3 | 0 | 0.0659 | 0.1861 | 0.2993 | 8.1205 | 42 | - | 1024 | 26 | S CCE | 1024 |

## Regression Models (Run-level Summary with Configuration)

| model_folder | run_id | seq_len | filters | status | best_epoch | best_val_mae | best_val_loss | final_overfit_gap_mae | kernel |
|:---|---:|:---|:---|:---|---:|---:|---:|---:|---:|
| regress_model_12_1s | 1 | 50 | 128 | ok | 7 | 0.0424 | 0.0060 | 0.0213 | 5 |
| regress_model_13_3s | 1 | 50 | 128 | ok | 13 | 0.0849 | 0.0157 | 0.0296 | 5 |

### Regression Models (Best Run Per Model with Configuration)

| model_folder | run_id | seq_len | filters | kernel | best_epoch | best_val_mae | best_val_loss |MAE | RMSE |
|:---|---:|:---|:---|:---|---:|---:|---:|---:|---:|
| regress_model_12_1second | 1 | - | - | - | 7 | 0.0424 | 0.0060 |0.0266 | 0.0415 |
| regress_model_13_3second | 1 | - | - | - | 13 | 0.0849 | 0.0157 |0.0672  | 0.0929|

## Global Configuration (Used Across All Models)

| Parameter | Value |
|---|---|
| **Input Context** | 50 frames (10.0s) |
| **Prediction Horizon** | 30 frames (3s) |
| **Skip Connections** | True |
| **Dialations** | [1, 2, 4, 8, 16, 32, 64, 128] |
| **Input Features** | 24 |
| **Pitch Grid** | 4×16 zones |
| **Batch Size** | 256 |
| **Learning Rate** | 5.00e-04 | cosine with minimum 1e-5
| **Max Epochs** | 60 |
| **Backend** | keras |
| **Activation** | Relu |

| **Batch Normalization**| True|


