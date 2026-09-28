/*
meta_model.rs - The Random Forest Meta-Learner (Stacking)
This is the final stage of our pipeline. We take the predictions made by the
Additive model and the LSTM, and we feed them into a Random Forest.
Think of this like an XGBoost or scikit-learn RandomForestRegressor.
The tree learns when to trust the LSTM (e.g. short-term spikes) and when
to trust the Additive model (e.g. predictable holidays).
*/

use anyhow::{Context, Result};
use polars::prelude::*;
use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::cv::{self, TsFold};

// Configuration

const TARGET_COL: &str = "Sales_scaled";

const STATIC_FEATURES: &[&str] = &[
    "Store",
    "DayOfWeek",
    "Promo",
    "StateHoliday_enc",
    "SchoolHoliday",
    "StoreType_enc",
    "Assortment_enc",
    "CompetitionDistance",
];

// Public API

pub fn train_and_evaluate_meta_learner(
    df: &DataFrame,
    folds: &[TsFold],
    additive_oof: &Float64Chunked,
    lstm_oof: &Float64Chunked,
) -> Result<()> {
    // 1. Add OOF predictions to the DataFrame
    let mut df_meta = df.clone();
    df_meta.with_column(additive_oof.clone().into_series())?;
    df_meta.with_column(lstm_oof.clone().into_series())?;

    let mut feature_cols: Vec<&str> = vec!["additive_oof", "lstm_oof"];
    feature_cols.extend_from_slice(STATIC_FEATURES);

    println!("\n[Step 5] Training Random Forest Meta-Learner across {} folds …", folds.len());

    for fold in folds {
        // 2. Split data using the CV folds
        let train_df = cv::apply_mask(&df_meta, &fold.train_mask)?;
        let val_df = cv::apply_mask(&df_meta, &fold.val_mask)?;

        // Filter out rows where OOF predictions are NaN
        let train_df = train_df
            .lazy()
            .filter(col("additive_oof").is_not_nan().and(col("lstm_oof").is_not_nan()))
            .collect()?;
            
        let val_df = val_df
            .lazy()
            .filter(col("additive_oof").is_not_nan().and(col("lstm_oof").is_not_nan()))
            .collect()?;

        if train_df.height() == 0 || val_df.height() == 0 {
            println!(
                "  Fold {}: skipped (no valid OOF predictions in train or val)",
                fold.fold_index
            );
            continue;
        }

        // 3. Extract features and target
        let (x_train, y_train) = extract_xy(&train_df, &feature_cols)?;
        let (x_val, y_val) = extract_xy(&val_df, &feature_cols)?;

        // 4. Create DenseMatrix for smartcore
        let num_rows_train = train_df.height();
        let num_rows_val = val_df.height();
        let num_cols = feature_cols.len();

        let dtrain = DenseMatrix::new(num_rows_train, num_cols, x_train, false);
        let dval = DenseMatrix::new(num_rows_val, num_cols, x_val, false);

        // 5. Configure Random Forest parameters (Tuned)
        let params = RandomForestRegressorParameters::default()
            .with_m(5)
            .with_n_trees(100)
            .with_max_depth(8);

        // 6. Train the model
        let model = RandomForestRegressor::fit(&dtrain, &y_train, params)
            .context("Failed to fit Random Forest")?;

        // 7. Predict on validation set
        let val_preds = model.predict(&dval).context("Failed to predict on val set")?;

        // 8. Compute RMSE & RMSPE
        let rmse = compute_rmse(&y_val, &val_preds);
        let rmspe = compute_rmspe(&y_val, &val_preds);
        println!(
            "  Fold {}: Meta-Learner RMSE(scaled) = {:.6}, RMSPE = {:.4}  |  train_rows={}, val_rows={}",
            fold.fold_index, rmse, rmspe, train_df.height(), val_df.height()
        );
    }

    Ok(())
}

// Helpers

/**
Extract features into a flat f64 vector (row-major) and target into f64 vector
*/
fn extract_xy(
    df: &DataFrame,
    feature_names: &[&str],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = df.height();
    let d = feature_names.len();

    let mut x = vec![0.0f64; n * d];

    for (j, &col_name) in feature_names.iter().enumerate() {
        let series = df.column(col_name)?;
        let ca = series
            .cast(&DataType::Float64)
            .context(format!("Cannot cast {col_name} to f64"))?;
        let ca = ca.f64().unwrap();
        
        for (i, val) in ca.iter().enumerate() {
            x[i * d + j] = val.unwrap_or(0.0);
        }
    }

    let y = extract_target(df)?;
    Ok((x, y))
}

/**
Extract the target column as a f64 vector
*/
fn extract_target(df: &DataFrame) -> Result<Vec<f64>> {
    let series = df.column(TARGET_COL)?;
    let ca = series.f64().context("Sales_scaled is not f64")?;
    let y: Vec<f64> = ca.iter().map(|v| v.unwrap_or(0.0)).collect();
    Ok(y)
}

/**
Root Mean Square Error.
*/
fn compute_rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len() as f64;
    let mse: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (*a - *p).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

/**
Root Mean Square Percentage Error (RMSPE)
This is the official Kaggle evaluation metric.
*/
fn compute_rmspe(actual: &[f64], predicted: &[f64]) -> f64 {
    let mut sum_sq_pct_err = 0.0;
    let mut count = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        if *a > 1e-6 { // Avoid division by zero for days with 0 sales
            let pct_err = (a - p) / a;
            sum_sq_pct_err += pct_err * pct_err;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return 0.0;
    }
    (sum_sq_pct_err / count).sqrt()
}

/* 
   Python Equivalent
If you were to write this in Python using scikit-learn, it would look like:

from sklearn.ensemble import RandomForestRegressor

def train_and_evaluate_meta_learner(df, folds, additive_oof, lstm_oof):
    df['additive_oof'] = additive_oof
    df['lstm_oof'] = lstm_oof
    
    features = ['additive_oof', 'lstm_oof', 'Store', 'DayOfWeek', 'Promo', ...]
    
    for train_idx, val_idx in folds:
        train_df = df.iloc[train_idx].dropna(subset=['additive_oof', 'lstm_oof'])
        val_df = df.iloc[val_idx].dropna(subset=['additive_oof', 'lstm_oof'])
        
        X_train, y_train = train_df[features], train_df[TARGET_COL]
        X_val, y_val = val_df[features], val_df[TARGET_COL]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=8, max_features=5)
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        
        # Compute RMSPE (which translates seamlessly on scaled data!)
        rmspe = np.sqrt(np.mean(((y_val - val_preds) / y_val) ** 2))
        print(f"Meta-Learner RMSPE: {rmspe}")
*/
