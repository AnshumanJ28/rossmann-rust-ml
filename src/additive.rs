/*
additive.rs - The Prophet Clone
Facebook's Prophet is basically a linear regression with sine/cosine waves
added as features to capture seasonality (e.g. weekly or yearly cycles).
We are doing exactly that here using `linfa` (Rust's scikit-learn).
We fit an Ordinary Least Squares (OLS) model on the time features.
*/

use anyhow::{Context, Result};
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use polars::prelude::*;

use crate::cv::{self, TsFold};

// Configuration

/**
Columns used as features for the additive model.
*/
const FEATURE_COLS: &[&str] = &[
    /*
    Trend (added dynamically below)
    Weekly Fourier
    */
    "weekly_sin_1",
    "weekly_cos_1",
    "weekly_sin_2",
    "weekly_cos_2",
    "weekly_sin_3",
    "weekly_cos_3",
    // Yearly Fourier
    "yearly_sin_1",
    "yearly_cos_1",
    "yearly_sin_2",
    "yearly_cos_2",
    "yearly_sin_3",
    "yearly_cos_3",
    "yearly_sin_4",
    "yearly_cos_4",
    "yearly_sin_5",
    "yearly_cos_5",
    // Known-in-advance regressors
    "Promo",
    "SchoolHoliday",
    "StateHoliday_enc",
];

const TARGET_COL: &str = "Sales_scaled";

// Public API

/**
Train the additive model on each CV fold and produce OOF predictions.

Returns a `Float64Chunked` of length `df.height()` where validation rows
contain the model's prediction and non-validation rows are `NaN`.
This is later consumed by the meta-learner.
*/
pub fn train_and_predict_oof(
    df: &DataFrame,
    folds: &[TsFold],
) -> Result<Float64Chunked> {
    // 1. Add a `trend` column: days since earliest date.
    let df = add_trend_column(df)?;

    // All feature column names (trend + the constant list).
    let mut all_feature_names: Vec<&str> = vec!["trend"];
    all_feature_names.extend_from_slice(FEATURE_COLS);

    // 2. Initialise the OOF prediction vector with NaN.
    let n = df.height();
    let mut oof_preds = vec![f64::NAN; n];

    println!("[Step 3] Training Custom Additive Model across {} folds …", folds.len());

    for fold in folds {
        // 3. Split data.
        let train_df = cv::apply_mask(&df, &fold.train_mask)?;
        let val_df = cv::apply_mask(&df, &fold.val_mask)?;

        // 4. Extract feature matrices and target vectors.
        let (x_train, y_train) = extract_xy(&train_df, &all_feature_names)?;
        let (x_val, _y_val) = extract_xy(&val_df, &all_feature_names)?;

        // 5. Build linfa dataset and fit OLS.
        let train_dataset = Dataset::new(x_train, y_train);
        let model = LinearRegression::default()
            .fit(&train_dataset)
            .context("Additive model fitting failed")?;

        // 6. Predict on validation set.
        let val_preds = model.predict(&x_val);

        /*
        7. Write predictions back into the OOF vector at the correct
           indices.  We use the boolean mask to find original row positions.
        */
        let mut val_idx = 0;
        for (row_idx, is_val) in fold.val_mask.iter().enumerate() {
            if is_val == Some(true) {
                oof_preds[row_idx] = val_preds[val_idx];
                val_idx += 1;
            }
        }

        // 8. Compute fold-level RMSE for logging.
        let _y_val_arr = extract_target(&val_df)?;
        let rmse = compute_rmse(&_y_val_arr, &val_preds);
        println!(
            "  Fold {}: RMSE(scaled) = {:.6}  |  train={}, val={}",
            fold.fold_index, rmse, fold.n_train, fold.n_val,
        );
    }

    // 9. Wrap into a Polars chunked array.
    let ca = Float64Chunked::new("additive_oof".into(), &oof_preds);
    Ok(ca)
}

// Helpers

/**
Add a `trend` column = (Date epoch-days) − min(Date epoch-days).
*/
fn add_trend_column(df: &DataFrame) -> Result<DataFrame> {
    let date_col = df.column("Date")?.date().context("Date not Date dtype")?;
    let min_date = date_col.min().context("empty Date")? as f64;

    df.clone()
        .lazy()
        .with_column(
            (col("Date").cast(DataType::Float64) - lit(min_date)).alias("trend"),
        )
        .collect()
        .context("Failed to add trend column")
}

/**
Extract an (n_rows × n_features) Array2 and a (n_rows,) Array1 target.
*/
fn extract_xy(
    df: &DataFrame,
    feature_names: &[&str],
) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = df.height();
    let d = feature_names.len();

    let mut x = Array2::<f64>::zeros((n, d));

    for (j, &col_name) in feature_names.iter().enumerate() {
        let series = df.column(col_name)?;
        let ca = series
            .cast(&DataType::Float64)
            .context(format!("Cannot cast {col_name} to f64"))?;
        let ca = ca.f64().unwrap();
        for (i, val) in ca.iter().enumerate() {
            x[[i, j]] = val.unwrap_or(0.0);
        }
    }

    let y = extract_target(df)?;
    Ok((x, y))
}

/**
Extract the target column as an ndarray Array1.
*/
fn extract_target(df: &DataFrame) -> Result<Array1<f64>> {
    let series = df.column(TARGET_COL)?;
    let ca = series.f64().context("Sales_scaled is not f64")?;
    let y: Vec<f64> = ca.iter().map(|v| v.unwrap_or(0.0)).collect();
    Ok(Array1::from(y))
}

/**
Root Mean Square Error.
*/
fn compute_rmse(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let diff = actual - predicted;
    let mse = diff.mapv(|v| v * v).mean().unwrap_or(0.0);
    mse.sqrt()
}

/* Python Equivalent
If you were to write this exact file in Python, it would look like this:

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

def train_and_predict_oof(df, folds):
    oof_preds = pd.Series(index=df.index, dtype=float)
    
    for train_idx, val_idx in folds:
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # X and y
        X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
        X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]
        
        # Fit OLS model (linfa's LinearRegression equivalent)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict and save to OOF
        val_preds = model.predict(X_val)
        oof_preds.iloc[val_idx] = val_preds
        
        # Log RMSE
        print(f"Fold RMSE: {root_mean_squared_error(y_val, val_preds)}")
        
    return oof_preds
*/
