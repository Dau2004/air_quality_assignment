# Beijing Air Quality Forecasting

Forecast hourly PM2.5 levels in Beijing using sequence models (GRU/LSTM/BiLSTM/CNN-LSTM) with engineered time, weather, and interaction features. The workflow trains multiple architectures over sliding windows, selects the best via validation RMSE, retrains with longer patience, and generates a Kaggle-ready submission.

- Main notebook (best, clean, end-to-end): [Notebook/air_quality_forecasting_starter_code.ipynb](Notebook/air_quality_forecasting_starter_code.ipynb)
- Additional notebooks: [Beijing_Air_Quality_Forecasting.ipynb](Beijing_Air_Quality_Forecasting.ipynb), [AQ.ipynb](AQ.ipynb), [AQ2.ipynb](AQ2.ipynb)
- Data: [train.csv](train.csv), [test.csv](test.csv), [sample_submission .csv](sample_submission .csv)
- Example outputs: [submission_seq_models.csv](submission_seq_models.csv), [submission_seq_models2.csv](submission_seq_models2.csv)

## Contents
- Overview and objectives
- Project structure
- Setup
- End-to-end workflow
- Model architectures and experiments
- Outputs and submissions
- Reproducibility

## Overview
This project applies sequence models (RNN/GRU/LSTM) to forecast PM2.5. It:
- Engineers robust time and meteorological features
- Builds sliding windows (24/48/72h) without leakage
- Trains 16 experiments across architectures and LRs
- Selects best config by validation RMSE
- Retrains best model with longer patience (85/15 split)
- Predicts test windows and writes Kaggle submission

Primary helpers in the main notebook:
- Feature engineering: [`create_beijing_features`](Notebook/air_quality_forecasting_starter_code.ipynb)
- Windowing: [`make_windows`](Notebook/air_quality_forecasting_starter_code.ipynb), [`train_val_split_windows`](Notebook/air_quality_forecasting_starter_code.ipynb)
- Models: [`build_gru`](Notebook/air_quality_forecasting_starter_code.ipynb), [`build_lstm`](Notebook/air_quality_forecasting_starter_code.ipynb), [`build_cnn_lstm`](Notebook/air_quality_forecasting_starter_code.ipynb)

## Project structure
- Notebooks
  - [Notebook/air_quality_forecasting_starter_code.ipynb](Notebook/air_quality_forecasting_starter_code.ipynb) – main pipeline and submission
  - [Beijing_Air_Quality_Forecasting.ipynb](Beijing_Air_Quality_Forecasting.ipynb) – exploratory LSTM experiments
  - [AQ.ipynb](AQ.ipynb), [AQ2.ipynb](AQ2.ipynb) – alternative baselines, analyses
- Data files
  - [train.csv](train.csv), [test.csv](test.csv), [sample_submission .csv](sample_submission .csv)
- Artifacts
  - [submission_seq_models.csv](submission_seq_models.csv), [submission_seq_models2.csv](submission_seq_models2.csv)
  - best models (e.g., best_model_improved.h5)

## Setup
- Python 3.9+ recommended

Install packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

Optional: set a virtual environment before installing.

## End-to-end workflow
Run cells top-to-bottom in [Notebook/air_quality_forecasting_starter_code.ipynb](Notebook/air_quality_forecasting_starter_code.ipynb):

1) Load data
- Reads [train.csv](train.csv) and [test.csv](test.csv), parses datetime index.

2) EDA
- Summary, missingness, distributions, correlations, sample time series.

3) Missing values and leakage-safe cleaning
- Forward/backward fill features; drop rows with missing target pm2.5.

4) Advanced feature engineering
- Time features with cyclical encodings; weather interactions; wind transforms and precipitation via [`create_beijing_features`](Notebook/air_quality_forecasting_starter_code.ipynb).

5) Robust scaling
- Fit scaler on train; transform train/test consistently.

6) Sliding windows and time-aware split
- Build 24/48/72h windows using [`make_windows`](Notebook/air_quality_forecasting_starter_code.ipynb). Split via [`train_val_split_windows`](Notebook/air_quality_forecasting_starter_code.ipynb) (no future leakage).

7) Model builders
- [`build_gru`](Notebook/air_quality_forecasting_starter_code.ipynb), [`build_lstm`](Notebook/air_quality_forecasting_starter_code.ipynb), [`build_cnn_lstm`](Notebook/air_quality_forecasting_starter_code.ipynb), BiLSTM option.

8) Systematic experiments (16 total)
- Grid across sequence lengths × architectures, then LR ablation on the best.
- EarlyStopping + ReduceLROnPlateau; collect RMSE/MAE/R²/time.

9) Results summary and visualizations
- Tables and plots to pick the best configuration.

10) Retrain best model
- 85/15 split with longer patience for a stronger final model.

11) Build test windows and predict
- Concatenate scaled train/test features, create windows ending in test, predict.

12) Save Kaggle submission
- Writes submission CSV in the format of [sample_submission .csv](sample_submission .csv).

## Model architectures and experiments
Architectures:
- GRU(96), LSTM(96), BiLSTM(64), CNN-LSTM(Conv1D+Pooling+LSTM)
- Sequence lengths: 24h, 48h, 72h
- Learning rates: 0.002, 0.0015, 0.0010 + ablation [0.003, 0.0012, 0.0008, 0.0005]
- Loss: MSE; Metric: RMSE (reported from validation)

The notebook automatically selects the best by lowest validation RMSE.

## Outputs and submissions
- Final sequence-model submission written to [submission_seq_models2.csv](submission_seq_models2.csv)
- Columns:
  - row ID: timestamp string
  - pm2.5: non-negative integer predictions
- Sample format in [sample_submission .csv](sample_submission .csv)

## Reproducibility
- Fixed random seeds are set in notebooks.
- Time-aware splits avoid leakage.
- Train-only fit for scalers, consistent transforms for test.

## Tips
- If GPU is available, enable it for faster training in TensorFlow.
- To shorten iteration time, reduce epochs or the search space, then scale back up.

## License
This project uses the provided datasets and is intended for educational and competition purposes.

## References
- Main pipeline: [Notebook/air_quality_forecasting_starter_code.ipynb](Notebook/air_quality_forecasting_starter_code.ipynb)
- Supporting experiments: [Beijing_Air_Quality_Forecasting.ipynb](Beijing_Air_Quality_Forecasting.ipynb),