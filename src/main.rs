/*
main.rs - The orchestrator
Think of this like your main Jupyter notebook or Python script. We're going to load
the data, build some cross-validation folds, and train our models step-by-step.
*/

mod data;
mod cv;
mod additive;
mod lstm_model;
mod meta_model;

use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Rossmann Hybrid Forecasting Model ===\n");

    /*
    1. Data Loading & Preprocessing
    This is basically our `pandas.read_csv()` and `fit_transform()` step.
    */
    let paths = data::DataPaths {
        train_csv: "Cdataset/train.csv".to_string(),
        store_csv: "Cdataset/store.csv".to_string(),
    };

    let preprocessed = data::load_and_preprocess(&paths)?;

    println!("\n--- Column names ({}) ---", preprocessed.df.width());
    for name in preprocessed.df.get_column_names() {
        print!("{name}  ");
    }
    println!("\n\n--- First 5 rows ---");
    println!("{}", preprocessed.df.head(Some(5)));

    println!("\n--- Sales scaling ---");
    println!("  min = {:.2}", preprocessed.sales_min);
    println!("  max = {:.2}", preprocessed.sales_max);

    // Step 2 — Time-Series Cross-Validation
    println!("\n");
    let cv_config = cv::TsCvConfig {
        n_folds: 3,
        val_days: 48,
        gap_days: 0,
    };

    let folds = cv::generate_ts_folds(&preprocessed.df, &cv_config)?;

    // Step 3 — Custom Additive Model (Prophet Clone)
    println!();
    let additive_oof = additive::train_and_predict_oof(&preprocessed.df, &folds)?;

    let n_oof_add = additive_oof.iter().filter(|v| v.map_or(false, |x| !x.is_nan())).count();
    println!(
        "\n[Step 3] OOF predictions generated: {} / {} rows",
        n_oof_add,
        preprocessed.df.height(),
    );

    // Step 4 — LSTM Network
    println!();
    let lstm_oof = lstm_model::train_and_predict_oof(&preprocessed.df, &folds)?;

    let n_oof_lstm = lstm_oof.iter().filter(|v| v.map_or(false, |x| !x.is_nan())).count();
    println!(
        "\n[Step 4] OOF predictions generated: {} / {} rows",
        n_oof_lstm,
        preprocessed.df.height(),
    );

    // Step 5 — Random Forest Meta-Learner
    println!();
    meta_model::train_and_evaluate_meta_learner(
        &preprocessed.df,
        &folds,
        &additive_oof,
        &lstm_oof,
    )?;

    Ok(())
}

   Python Equivalent
If you were to write this orchestrator in Python, it would look like this:

if __name__ == "__main__":
    print("=== Rossmann Hybrid Forecasting Model ===\n")
    
    # Step 1: Pandas data loading
    df = data.load_and_preprocess("train.csv", "store.csv")
    
    # Step 2: TimeSeriesSplit
    folds = cv.generate_ts_folds(df, n_folds=3)
    
    # Step 3: Scikit-learn Linear Regression + Fourier
    additive_oof = additive.train_and_predict_oof(df, folds)
    
    # Step 4: PyTorch LSTM
    lstm_oof = lstm_model.train_and_predict_oof(df, folds)
    
    # Step 5: XGBoost / Scikit-learn Random Forest Meta Learner
    meta_model.train_and_evaluate_meta_learner(df, folds, additive_oof, lstm_oof)
*/
