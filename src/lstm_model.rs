/*
lstm_model.rs - The Deep Learning Sequence Model
This is our PyTorch equivalent. We use the `burn` crate to build an LSTM
and train it on the GPU via WGPU.
It looks at a 30-day sliding window of (Sales, Promo, DayOfWeek) to predict
the 31st day. We use mini-batch gradient descent with the Adam optimizer.
*/

use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::MseLoss;
use burn::nn::lstm::{Lstm, LstmConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::ElementConversion;
use burn::tensor::{backend::Backend, Tensor};
use polars::prelude::*;

use crate::cv::TsFold;

// Type aliases

type B = burn::backend::Wgpu;
type AB = Autodiff<B>;

// Constants

const WINDOW_SIZE: usize = 30; // Increased to 1 month of context
const CHANNELS: usize = 3;     // Sales, Promo, DayOfWeek
const HIDDEN_SIZE: usize = 64; // Increased capacity
const DROPOUT_RATE: f64 = 0.2;
const LEARNING_RATE: f64 = 1e-3;
const BATCH_SIZE: usize = 512;
const EPOCHS: usize = 10;      // Increased training time

// Model definition

#[derive(Module, Debug, Clone)]
pub struct SalesLstm<Be: Backend> {
    lstm: Lstm<Be>,
    dropout: Dropout,
    linear: Linear<Be>,
}

#[derive(Config)]
pub struct SalesLstmConfig {
    #[config(default = 3)] // 3 channels now
    input_size: usize,
    #[config(default = 64)]
    hidden_size: usize,
    #[config(default = 0.2)]
    dropout_rate: f64,
}

impl SalesLstmConfig {
    pub fn init_model<Be: Backend>(&self, device: &Be::Device) -> SalesLstm<Be> {
        SalesLstm {
            lstm: LstmConfig::new(self.input_size, self.hidden_size, true).init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            linear: LinearConfig::new(self.hidden_size, 1).init(device),
        }
    }
}

impl<Be: Backend> SalesLstm<Be> {
    /**
    Forward pass.
    `x`: [batch, seq_len, channels]  →  Returns: [batch, 1]
    */
    pub fn forward(&self, x: Tensor<Be, 3>) -> Tensor<Be, 2> {
        let (output, _state) = self.lstm.forward(x, None);

        let dims = output.dims();
        let batch = dims[0];
        let hidden = dims[2];

        // Take last time step → [batch, hidden_size]
        let last = output.slice([0..batch, (dims[1] - 1)..dims[1], 0..hidden]);
        let last = last.reshape([batch, hidden]);
        let last = self.dropout.forward(last);
        self.linear.forward(last)
    }
}

// Window data structures

struct WindowDataset {
    /**
    Flattened: length = count × WINDOW_SIZE × CHANNELS
    */
    windows: Vec<f32>,
    /**
    length = count
    */
    targets: Vec<f32>,
    /**
    Index into original DataFrame for each target row
    */
    original_indices: Vec<usize>,
    count: usize,
}

// Public API

pub fn train_and_predict_oof(
    df: &DataFrame,
    folds: &[TsFold],
) -> Result<Float64Chunked> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let n = df.height();
    let mut oof_preds = vec![f64::NAN; n];

    let store_data = extract_store_series(df)?;

    println!("[Step 4] Training TUNED LSTM across {} folds …", folds.len());

    for fold in folds {
        let train_windows = build_windows(&store_data, &fold.train_mask)?;
        let val_windows = build_windows(&store_data, &fold.val_mask)?;

        println!(
            "  Fold {}: {} train windows, {} val windows",
            fold.fold_index, train_windows.count, val_windows.count
        );

        let config = SalesLstmConfig::new();
        let mut model: SalesLstm<AB> = config.init_model(&device);
        let mut optim = AdamConfig::new().init();

        for epoch in 0..EPOCHS {
            let epoch_loss =
                train_one_epoch(&mut model, &mut optim, &train_windows, &device);
            if epoch == 0 || epoch == EPOCHS - 1 {
                println!(
                    "    Epoch {}/{}: MSE = {:.6}",
                    epoch + 1,
                    EPOCHS,
                    epoch_loss,
                );
            }
        }

        let model_valid = <SalesLstm<AB> as AutodiffModule<AB>>::valid(&model);
        let val_preds = predict_all(&model_valid, &val_windows, &device);

        for (i, &orig_idx) in val_windows.original_indices.iter().enumerate() {
            oof_preds[orig_idx] = val_preds[i] as f64;
        }

        let rmse = compute_rmse_vecs(
            &val_windows.targets,
            &val_preds,
        );
        println!("  Fold {}: RMSE(scaled) = {:.6}", fold.fold_index, rmse);
    }

    Ok(Float64Chunked::new("lstm_oof".into(), &oof_preds))
}

// Training helpers

fn train_one_epoch<O: Optimizer<SalesLstm<AB>, AB>>(
    model: &mut SalesLstm<AB>,
    optim: &mut O,
    data: &WindowDataset,
    device: &<AB as Backend>::Device,
) -> f32 {
    let n = data.count;
    if n == 0 {
        return 0.0;
    }

    let n_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;
    let mut total_loss = 0.0f32;
    let loss_fn = MseLoss::new();

    for batch_idx in 0..n_batches {
        let start = batch_idx * BATCH_SIZE;
        let end = (start + BATCH_SIZE).min(n);
        let bs = end - start;

        let mut x_data = Vec::with_capacity(bs * WINDOW_SIZE * CHANNELS);
        let mut y_data = Vec::with_capacity(bs);
        for i in start..end {
            let off = i * WINDOW_SIZE * CHANNELS;
            x_data.extend_from_slice(&data.windows[off..off + WINDOW_SIZE * CHANNELS]);
            y_data.push(data.targets[i]);
        }

        let x = Tensor::<AB, 1>::from_floats(x_data.as_slice(), device)
            .reshape([bs, WINDOW_SIZE, CHANNELS]);
        let y = Tensor::<AB, 1>::from_floats(y_data.as_slice(), device)
            .reshape([bs, 1]);

        let pred = model.forward(x);
        let loss = loss_fn.forward(pred, y.clone(), burn::nn::loss::Reduction::Mean);

        let loss_val: f32 = loss.clone().into_scalar().elem();
        total_loss += loss_val;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads::<AB, SalesLstm<AB>>(grads, model);
        *model = optim.step(LEARNING_RATE.into(), model.clone(), grads);
    }

    total_loss / n_batches as f32
}

fn predict_all<Be: Backend>(
    model: &SalesLstm<Be>,
    data: &WindowDataset,
    device: &Be::Device,
) -> Vec<f32> {
    let n = data.count;
    if n == 0 {
        return vec![];
    }

    let mut all_preds = Vec::with_capacity(n);
    let n_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;

    for batch_idx in 0..n_batches {
        let start = batch_idx * BATCH_SIZE;
        let end = (start + BATCH_SIZE).min(n);
        let bs = end - start;

        let mut x_data = Vec::with_capacity(bs * WINDOW_SIZE * CHANNELS);
        for i in start..end {
            let off = i * WINDOW_SIZE * CHANNELS;
            x_data.extend_from_slice(&data.windows[off..off + WINDOW_SIZE * CHANNELS]);
        }

        let x = Tensor::<Be, 1>::from_floats(x_data.as_slice(), device)
            .reshape([bs, WINDOW_SIZE, CHANNELS]);

        let pred = model.forward(x);
        let pred_data = pred.reshape([bs]).into_data();
        let pred_vec: Vec<f32> = pred_data.to_vec().unwrap();
        all_preds.extend(pred_vec);
    }

    all_preds
}

// Data helpers

struct StoreSeries {
    sales: Vec<f32>,
    promo: Vec<f32>,
    dow: Vec<f32>,
    row_indices: Vec<usize>,
}

fn extract_store_series(df: &DataFrame) -> Result<Vec<StoreSeries>> {
    let store_col = df.column("Store")?.i64()?;
    let sales_col = df.column("Sales_scaled")?.f64()?;
    let promo_col = df.column("Promo")?.i64()?;
    let dow_col = df.column("DayOfWeek")?.i64()?;

    let store_vals: Vec<i64> = store_col.into_no_null_iter().collect();
    let sales_vals: Vec<f64> = sales_col.into_no_null_iter().collect();
    let promo_vals: Vec<i64> = promo_col.into_no_null_iter().collect();
    let dow_vals: Vec<i64> = dow_col.into_no_null_iter().collect();
    let n = store_vals.len();

    let mut result = Vec::new();
    let mut i = 0;
    while i < n {
        let current = store_vals[i];
        let start = i;
        while i < n && store_vals[i] == current {
            i += 1;
        }
        
        result.push(StoreSeries {
            sales: sales_vals[start..i].iter().map(|&v| v as f32).collect(),
            promo: promo_vals[start..i].iter().map(|&v| v as f32).collect(),
            dow: dow_vals[start..i].iter().map(|&v| v as f32 / 7.0).collect(), // Normalize DayOfWeek
            row_indices: (start..i).collect(),
        });
    }

    Ok(result)
}

fn build_windows(
    store_data: &[StoreSeries],
    mask: &BooleanChunked,
) -> Result<WindowDataset> {
    let mut windows = Vec::new();
    let mut targets = Vec::new();
    let mut original_indices = Vec::new();

    for store in store_data {
        let slen = store.sales.len();
        if slen <= WINDOW_SIZE {
            continue;
        }
        for i in WINDOW_SIZE..slen {
            let orig_idx = store.row_indices[i];
            if mask.get(orig_idx) != Some(true) {
                continue;
            }
            
            // Build the multi-channel window [WINDOW_SIZE * CHANNELS]
            for j in (i - WINDOW_SIZE)..i {
                windows.push(store.sales[j]);
                windows.push(store.promo[j]);
                windows.push(store.dow[j]);
            }
            
            targets.push(store.sales[i]);
            original_indices.push(orig_idx);
        }
    }

    let count = targets.len();
    Ok(WindowDataset {
        windows,
        targets,
        original_indices,
        count,
    })
}

fn compute_rmse_vecs(actual: &[f32], predicted: &[f32]) -> f64 {
    let n = actual.len() as f64;
    let mse: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (*a as f64 - *p as f64).powi(2))
        .sum::<f64>()
        / n;
    mse.sqrt()
}

   Python Equivalent
If you were to write this in Python using PyTorch, it would look like:

import torch
import torch.nn as nn

class SalesLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: [batch, window=30, channels=3]
        out, (hn, cn) = self.lstm(x)
        
        # Take last hidden state from the sequence
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)

# Training loop using torch.optim.Adam and nn.MSELoss()...
*/
