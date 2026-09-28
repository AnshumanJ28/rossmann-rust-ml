/*
data.rs - Data Loading & Preprocessing
This is our data pipeline. If you're coming from Python, this is essentially
everything you'd normally do with pandas (read_csv, merge, fillna, apply).
We're using Polars here because it is ridiculously fast in Rust.
*/

use polars::prelude::*;
use anyhow::{Context, Result};
use std::f64::consts::PI;

// Public configuration & result types

/**
Paths to the raw CSV files.
*/
pub struct DataPaths {
    pub train_csv: String,
    pub store_csv: String,
}

/**
Holds the fully preprocessed dataframe along with scaling parameters so we
can invert the transform later (for evaluation / submission).
*/
pub struct PreprocessedData {
    pub df: DataFrame,
    pub sales_min: f64,
    pub sales_max: f64,
}

// Constants

/**
Number of Fourier pairs for weekly seasonality (period = 7 days).
*/
const WEEKLY_FOURIER_ORDER: usize = 3;
/**
Number of Fourier pairs for yearly seasonality (period = 365.25 days).
*/
const YEARLY_FOURIER_ORDER: usize = 5;

// Main entry point

/**
Full preprocessing pipeline.  Returns a sorted, feature-rich DataFrame
ready for model consumption.
*/
pub fn load_and_preprocess(paths: &DataPaths) -> Result<PreprocessedData> {
    println!("[Step 1] Loading CSVs …");
    let train_df = load_train(&paths.train_csv)?;
    let store_df = load_store(&paths.store_csv)?;

    println!("[Step 1] Merging train + store on 'Store' …");
    let merged = merge_datasets(train_df, store_df)?;

    println!("[Step 1] Extracting temporal features …");
    let with_time = add_temporal_features(merged)?;

    println!("[Step 1] Encoding categoricals & holidays …");
    let with_cats = encode_categoricals(with_time)?;

    println!("[Step 1] Adding Fourier seasonality terms …");
    let with_fourier = add_fourier_features(with_cats)?;

    println!("[Step 1] Imputing missing values …");
    let imputed = impute_missing(with_fourier)?;

    println!("[Step 1] Filtering closed stores (Open != 1) …");
    let filtered = filter_open_stores(imputed)?;

    println!("[Step 1] Sorting by (Store, Date) …");
    let sorted = sort_by_store_date(filtered)?;

    println!("[Step 1] Scaling numeric columns …");
    let result = scale_sales(sorted)?;

    println!(
        "[Step 1] Done.  Shape = {:?}, Sales range = [{:.2}, {:.2}]",
        result.df.shape(),
        result.sales_min,
        result.sales_max,
    );

    Ok(result)
}

// Sub-steps (private helpers)

/**
Load train.csv.  We force `StateHoliday` to String because it mixes "0"
with "a", "b", "c" — without the override Polars infers i64 and chokes.
*/
fn load_train(path: &str) -> Result<DataFrame> {
    let mut schema = Schema::default();
    schema.insert("StateHoliday".into(), DataType::String);

    CsvReadOptions::default()
        .with_has_header(true)
        .with_schema_overwrite(Some(schema.into()))
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()
        .context("Failed to read train.csv")
}

/**
Load store.csv.
*/
fn load_store(path: &str) -> Result<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()
        .context("Failed to read store.csv")
}

/**
Left-join train with store on the `Store` column.
*/
fn merge_datasets(train: DataFrame, store: DataFrame) -> Result<DataFrame> {
    train
        .lazy()
        .left_join(store.lazy(), col("Store"), col("Store"))
        .collect()
        .context("Failed to join train and store DataFrames")
}

/**
Parse the `Date` string column → Date dtype, then extract Year, Month,
Day, DayOfYear, and WeekNumber.
*/
fn add_temporal_features(df: DataFrame) -> Result<DataFrame> {
    df.lazy()
        .with_columns(vec![
            col("Date")
                .str()
                .to_date(StrptimeOptions {
                    format: Some("%Y-%m-%d".into()),
                    ..Default::default()
                })
                .alias("Date"),
        ])
        .with_columns(vec![
            col("Date").dt().year().alias("Year"),
            col("Date").dt().month().alias("Month"),
            col("Date").dt().day().alias("Day"),
            col("Date").dt().ordinal_day().alias("DayOfYear"),
            col("Date").dt().week().alias("WeekNumber"),
        ])
        .collect()
        .context("Failed to extract temporal features")
}

/**
One-hot / ordinal-encode categorical columns:
  - StateHoliday: "0" → 0, "a" → 1, "b" → 2, "c" → 3
  - StoreType:     "a" → 0, "b" → 1, "c" → 2, "d" → 3
  - Assortment:    "a" → 0, "b" → 1, "c" → 2
*/
fn encode_categoricals(df: DataFrame) -> Result<DataFrame> {
    df.lazy()
        .with_columns(vec![
            // StateHoliday → integer code
            when(col("StateHoliday").eq(lit("a")))
                .then(lit(1i32))
                .when(col("StateHoliday").eq(lit("b")))
                .then(lit(2i32))
                .when(col("StateHoliday").eq(lit("c")))
                .then(lit(3i32))
                .otherwise(lit(0i32))
                .alias("StateHoliday_enc"),
            // StoreType → integer code
            when(col("StoreType").eq(lit("a")))
                .then(lit(0i32))
                .when(col("StoreType").eq(lit("b")))
                .then(lit(1i32))
                .when(col("StoreType").eq(lit("c")))
                .then(lit(2i32))
                .when(col("StoreType").eq(lit("d")))
                .then(lit(3i32))
                .otherwise(lit(-1i32))
                .alias("StoreType_enc"),
            // Assortment → integer code
            when(col("Assortment").eq(lit("a")))
                .then(lit(0i32))
                .when(col("Assortment").eq(lit("b")))
                .then(lit(1i32))
                .when(col("Assortment").eq(lit("c")))
                .then(lit(2i32))
                .otherwise(lit(-1i32))
                .alias("Assortment_enc"),
        ])
        .collect()
        .context("Failed to encode categoricals")
}

/**
Generate Fourier sine/cosine pairs for weekly and yearly seasonality.

For period P and order K we create:
  sin(2π·k·t / P)  and  cos(2π·k·t / P)   for k = 1 … K

where t = DayOfWeek (1-7) for weekly, DayOfYear (1-365) for yearly.
*/
fn add_fourier_features(df: DataFrame) -> Result<DataFrame> {
    let mut lf = df.lazy();

    // Weekly Fourier terms  (period = 7, t = DayOfWeek)
    for k in 1..=WEEKLY_FOURIER_ORDER {
        let coeff = 2.0 * PI * (k as f64) / 7.0;
        lf = lf.with_columns(vec![
            (col("DayOfWeek").cast(DataType::Float64) * lit(coeff))
                .sin()
                .alias(&format!("weekly_sin_{k}")),
            (col("DayOfWeek").cast(DataType::Float64) * lit(coeff))
                .cos()
                .alias(&format!("weekly_cos_{k}")),
        ]);
    }

    // Yearly Fourier terms  (period = 365.25, t = DayOfYear)
    for k in 1..=YEARLY_FOURIER_ORDER {
        let coeff = 2.0 * PI * (k as f64) / 365.25;
        lf = lf.with_columns(vec![
            (col("DayOfYear").cast(DataType::Float64) * lit(coeff))
                .sin()
                .alias(&format!("yearly_sin_{k}")),
            (col("DayOfYear").cast(DataType::Float64) * lit(coeff))
                .cos()
                .alias(&format!("yearly_cos_{k}")),
        ]);
    }

    lf.collect()
        .context("Failed to add Fourier features")
}

/**
Fill nulls in numeric columns with sensible defaults.
*/
fn impute_missing(df: DataFrame) -> Result<DataFrame> {
    df.lazy()
        .with_columns(vec![
            col("CompetitionDistance").fill_null(lit(0.0)),
            col("CompetitionOpenSinceMonth").fill_null(lit(0)),
            col("CompetitionOpenSinceYear").fill_null(lit(0)),
            col("Promo2SinceWeek").fill_null(lit(0)),
            col("Promo2SinceYear").fill_null(lit(0)),
        ])
        .collect()
        .context("Failed to impute missing values")
}

/**
Keep only rows where the store was open.
*/
fn filter_open_stores(df: DataFrame) -> Result<DataFrame> {
    df.lazy()
        .filter(col("Open").eq(lit(1)))
        .collect()
        .context("Failed to filter open stores")
}

/**
Sort by Store then Date – critical for time-series operations later.
*/
fn sort_by_store_date(df: DataFrame) -> Result<DataFrame> {
    df.lazy()
        .sort(
            ["Store", "Date"],
            SortMultipleOptions::default()
                .with_order_descending_multi([false, false]),
        )
        .collect()
        .context("Failed to sort by Store, Date")
}

/**
Min-Max scale the `Sales` column to [0, 1].
Returns the original min/max so we can invert later.
*/
fn scale_sales(df: DataFrame) -> Result<PreprocessedData> {
    // Cast Sales to Float64 first so we can reliably read it.
    let df = df
        .lazy()
        .with_column(col("Sales").cast(DataType::Float64).alias("Sales"))
        .collect()
        .context("Failed to cast Sales to Float64")?;

    let sales_col = df
        .column("Sales")?
        .f64()
        .context("Could not read Sales as f64")?;

    let sales_min = sales_col.min().unwrap_or(0.0);
    let sales_max = sales_col.max().unwrap_or(1.0);
    let range = if (sales_max - sales_min).abs() < f64::EPSILON {
        1.0
    } else {
        sales_max - sales_min
    };

    let scaled = df
        .lazy()
        .with_column(
            ((col("Sales") - lit(sales_min)) / lit(range))
                .alias("Sales_scaled"),
        )
        .collect()
        .context("Failed to scale Sales")?;

    Ok(PreprocessedData {
        df: scaled,
        sales_min,
        sales_max,
    })
}

/*   Python Equivalent
If you were to write this in Python with pandas, it would look like:

import pandas as pd
import numpy as np

def load_and_preprocess(train_path, store_path):
    train_df = pd.read_csv(train_path, dtype={'StateHoliday': str})
    store_df = pd.read_csv(store_path)
    
    # Merge
    df = train_df.merge(store_df, on='Store', how='left')
    
    # Dates
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Fourier terms
    for k in range(1, 4):
        df[f'weekly_sin_{k}'] = np.sin(2 * np.pi * k * df['DayOfWeek'] / 7)
        df[f'weekly_cos_{k}'] = np.cos(2 * np.pi * k * df['DayOfWeek'] / 7)
        
    # Filter Open
    df = df[df['Open'] == 1].copy()
    
    # Sort & Scale
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    sales_min, sales_max = df['Sales'].min(), df['Sales'].max()
    df['Sales_scaled'] = (df['Sales'] - sales_min) / (sales_max - sales_min)
    
    return df
*/
