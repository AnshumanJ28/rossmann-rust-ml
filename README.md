# Rossmann Store Sales Forecasting (Rust)

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF9900?style=for-the-badge&logo=scikit-learn&logoColor=white)
![WGPU](https://img.shields.io/badge/WGPU-GPU%20Accelerated-blue?style=for-the-badge)

A fully from-scratch, GPU-accelerated time-series forecasting pipeline built entirely in Rust. This project tackles the Kaggle **Rossmann Store Sales** challenge by predicting 6 weeks of daily store sales using a hybrid deep learning and ensemble architecture.

## My Rust Learning Notes

I built this project as a personal way to learn how to do Machine Learning in Rust coming from a Python background (Pandas, PyTorch, Scikit-Learn). 

At the bottom of every `.rs` file in the `src/` directory, I have included a **"Python Equivalent"** block comment. These blocks map the Rust code directly back to the Python code I was already familiar with to help me bridge the gap!

For example, the notes show how:
- `polars` in Rust compares to `pandas` in Python (`src/data.rs`)
- `burn` in Rust compares to `PyTorch` in Python (`src/lstm_model.rs`)
- `linfa` & `smartcore` in Rust compare to `scikit-learn` in Python (`src/additive.rs` & `src/meta_model.rs`)

## Architecture Approach

This model employs a **Stacking Ensemble** approach, mirroring winning Kaggle architectures, while enforcing strict **Expanding-Window Cross Validation** to mathematically prevent future-data leakage.

1. **Custom Additive Model (Prophet-style):** Uses Ordinary Least Squares (via `linfa`) with embedded Fourier seasonality terms to capture weekly and yearly sales cycles.
2. **LSTM Neural Network:** A PyTorch-style LSTM built with `burn` and accelerated via `wgpu`. It uses a 30-day historical lookback window across 3 channels (Sales, Promo, DayOfWeek) to capture short-term temporal dependencies.
3. **Meta-Learner (Random Forest):** A Random Forest regressor (via `smartcore`) that takes the Out-Of-Fold (OOF) predictions from the base models and learns how to optimally combine them for the final forecast.

## Why We Aren't Overfitting

Preventing data leakage and overfitting is critical in time-series forecasting. This project guarantees generalization through the following measures:
- **Expanding-Window Cross Validation:** The dataset is split sequentially through time (never randomly). A model trained on past data is only ever evaluated on strictly future, unseen data windows.
- **Out-of-Fold (OOF) Stacking:** The Meta-Learner (Random Forest) is only trained on predictions that the base models generated for validation data they had *never seen during training*. This prevents the meta-learner from simply memorizing the training set.
- **Strict Regularization:** The LSTM features heavy dropout (`p=0.2`), while the Random Forest Meta-Learner has a highly constrained tree depth (`max_depth=8`) to explicitly prevent it from fitting to noise.

## Results

The model was evaluated using strict time-series cross-validation. The primary evaluation metric is **RMSE** on scaled data, alongside an estimated **RMSPE** (Root Mean Square Percentage Error) based on the true sales values.

| Model Stage | Model Type | Fold 1 RMSE | Fold 2 RMSE | Average Daily Error |
|-------------|------------|-------------|-------------|---------------------|
| **Raw Baseline** | LSTM (Untuned) | 0.0991 | 0.2002 | - |
| **Raw Baseline** | Meta-Learner (RF) | 0.0729 | 0.0669 | ~$2,781 |
| **Tuned** | LSTM (30-day, 64-hidden) | 0.1422 | 0.0929 | - |
| **Tuned** | Meta-Learner (Depth 8) | **0.0625** | **0.0627** | **~$2,605** |

> **Estimated RMSPE:** ~0.38 (38%) 
*(A solid architectural baseline for a custom-built Rust pipeline using only 3 core features without extensive external datasets!)*

---
**Author:** Anshuman
