# Beijing Air Quality Forecasting (PM2.5)

Forecast hourly PM2.5 levels in Beijing using sequence models (GRU/LSTM/BiLSTM/CNN‑LSTM) with engineered time and weather features. The pipeline builds sliding windows, runs 16 experiments, selects the best by validation RMSE, retrains, and produces a Kaggle‑ready submission.

## Project structure
- Main notebook: [Notebook/air_quality_forecasting_starter_code.ipynb](Notebook/air_quality_forecasting_starter_code.ipynb)
- Data files (place at repo root):
  - [train.csv](train.csv)
  - [test.csv](test.csv)
- Output (created after running the notebook):
  - [submission_seq_models2.csv](submission_seq_models2.csv)

## Setup
- Python 3.9+ recommended

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## How to run
1) Open the main notebook: [Notebook/air_quality_forecasting_starter_code.ipynb](Notebook/air_quality_forecasting_starter_code.ipynb)
2) Ensure [train.csv](train.csv) and [test.csv](test.csv) are present at the repository root.
3) Run all cells top-to-bottom:
   - EDA and cleaning (time-aware fills, drop missing targets)
   - Advanced feature engineering
   - Sliding windows (24/48/72h) and time-aware splits
   - 16 experiments across architectures and learning rates
   - Retrain best model and generate predictions
4) The submission file is saved as [submission_seq_models2.csv](submission_seq_models2.csv)

## Notes
- The notebook uses only local files; no Colab mount required.
- For faster training, enable GPU if available.
- To iterate quickly, reduce epochs or the search grid, then scale up.

## License
For educational and competition use with the
