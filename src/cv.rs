/*
cv.rs - Time-Series Cross-Validation
Think of this like `sklearn.model_selection.TimeSeriesSplit`.
Instead of random shuffles (which would leak future data into the past),
we use an expanding window.
Fold 1: Train on year 1, Test on month 13
Fold 2: Train on year 1 + month 13, Test on month 14
We return boolean masks (like pandas boolean indexing) to filter the rows later.
*/

use polars::prelude::*;
use anyhow::{Context, Result};

// Public types

/**
Configuration for time-series CV.
*/
pub struct TsCvConfig {
    /**
    Number of validation folds.
    */
    pub n_folds: usize,
    /**
    Length of each validation window in days.
    */
    pub val_days: i32,
    /**
    Gap in days between end of training and start of validation.
    Set to 0 for no gap.
    */
    pub gap_days: i32,
}

impl Default for TsCvConfig {
    fn default() -> Self {
        Self {
            n_folds: 3,
            val_days: 48, // matches Rossmann competition test period
            gap_days: 0,
        }
    }
}

/**
A single train/validation fold expressed as boolean masks over the
original DataFrame rows.
*/
pub struct TsFold {
    pub fold_index: usize,
    /**
    Inclusive start date of the validation window.
    */
    pub val_start: i32,
    /**
    Inclusive end date of the validation window.
    */
    pub val_end: i32,
    /**
    Training cutoff date (last training date, exclusive of gap).
    */
    pub train_end: i32,
    /**
    Boolean mask: `true` for training rows.
    */
    pub train_mask: BooleanChunked,
    /**
    Boolean mask: `true` for validation rows.
    */
    pub val_mask: BooleanChunked,
    /**
    Number of training rows.
    */
    pub n_train: usize,
    /**
    Number of validation rows.
    */
    pub n_val: usize,
}

// Public API

/**
Generate time-series CV folds from a **sorted** DataFrame that contains
a `Date` column of dtype `Date`.

The folds are placed at the *end* of the date range and work backwards,
so the last fold's validation window ends on the dataset's max date.

Returns a `Vec<TsFold>` ordered from earliest to latest.
*/
pub fn generate_ts_folds(df: &DataFrame, config: &TsCvConfig) -> Result<Vec<TsFold>> {
    // 1. Extract the Date column as i32 (days since epoch).
    let date_col = df
        .column("Date")?
        .date()
        .context("Date column is not Date dtype")?
        .clone();

    let min_date = date_col
        .min()
        .context("Date column is empty")?;
    let max_date = date_col
        .max()
        .context("Date column is empty")?;

    let total_days = max_date - min_date;

    println!(
        "[Step 2] Date range: epoch-day {} → {} ({} days)",
        min_date, max_date, total_days
    );

    /*
    2. Compute fold boundaries, working backwards from max_date.
    
    Layout (latest fold first in construction, reversed at the end):
    
      val_end   = max_date - (i * stride)
      val_start = val_end  - val_days + 1
      train_end = val_start - gap_days - 1
    
    We space folds by `stride` days, chosen so all folds fit within the
    date range while leaving at least 180 days for the first fold's
    training set.
    */
    let min_train_days = 180;
    let fold_span = config.val_days + config.gap_days;
    let available = total_days - min_train_days;

    if available < fold_span {
        anyhow::bail!(
            "Not enough data for even one fold: need {} days but only {} available after \
             reserving {} for initial training.",
            fold_span,
            available,
            min_train_days,
        );
    }

    // Stride between successive validation windows.
    let stride = if config.n_folds <= 1 {
        0
    } else {
        (available - fold_span) / (config.n_folds as i32 - 1)
    };

    let mut folds = Vec::with_capacity(config.n_folds);

    for i in 0..config.n_folds {
        let offset = i as i32 * stride;

        let val_end = max_date - offset;
        let val_start = val_end - config.val_days + 1;
        let train_end = val_start - config.gap_days - 1;

        // Sanity: training must start at or after min_date.
        if train_end < min_date {
            println!(
                "  ⚠ Fold {} skipped — train_end ({}) < min_date ({})",
                i, train_end, min_date
            );
            continue;
        }

        // 3. Build boolean masks.
        let train_mask = date_col.lt_eq(train_end);

        let val_gte = date_col.gt_eq(val_start);
        let val_lte = date_col.lt_eq(val_end);
        let val_mask = &val_gte & &val_lte;

        let n_train = train_mask.sum().unwrap_or(0) as usize;
        let n_val = val_mask.sum().unwrap_or(0) as usize;

        folds.push(TsFold {
            fold_index: i,
            val_start,
            val_end,
            train_end,
            train_mask,
            val_mask,
            n_train,
            n_val,
        });
    }

    // Reverse so fold 0 has the earliest validation window.
    folds.reverse();
    // Re-index after reversing.
    for (i, fold) in folds.iter_mut().enumerate() {
        fold.fold_index = i;
    }

    // 4. Print summary.
    println!("[Step 2] Generated {} folds:", folds.len());
    for f in &folds {
        println!(
            "  Fold {}: train ≤ day {}  |  val [{}, {}]  |  train_rows={}, val_rows={}",
            f.fold_index, f.train_end, f.val_start, f.val_end, f.n_train, f.n_val,
        );
    }

    Ok(folds)
}

/**
Convenience: apply a boolean mask to a DataFrame to extract the subset.
*/
pub fn apply_mask(df: &DataFrame, mask: &BooleanChunked) -> Result<DataFrame> {
    df.filter(mask).context("Failed to apply mask to DataFrame")
}

   Python Equivalent
If you were to write this in Python, it's similar to scikit-learn's
TimeSeriesSplit, but structured around specific date cutoffs so you don't
accidentally split a single day in half across different stores:

import pandas as pd

def generate_ts_folds(df, n_folds=3, val_days=48):
    max_date = df['Date'].max()
    folds = []
    
    for i in range(n_folds):
        val_end = max_date - pd.Timedelta(days = i * val_days)
        val_start = val_end - pd.Timedelta(days = val_days - 1)
        train_end = val_start - pd.Timedelta(days = 1)
        
        train_idx = df.index[df['Date'] <= train_end]
        val_idx = df.index[(df['Date'] >= val_start) & (df['Date'] <= val_end)]
        folds.append((train_idx, val_idx))
        
    folds.reverse()
    return folds
*/
