# Bitcoin Realized Volatility Forecasting

Comparative study of classical econometric models and deep learning for **1-day-ahead Bitcoin realized volatility** under Protocol B.

Main writeup: [`reports/report.pdf`](reports/report.pdf). Notes: [`NOTES.md`](NOTES.md).

## Layout

```text
data/
  raw/
  processed/
src/                          # importable library
  data/                      # fetch, clean, returns/RV
  features/                  # panel + EDA plots
  models/                    # naive, har, arima, garch, lstm, hybrid
  metrics/                   # rmse/mae/mape/qlike, DM, split-compare
  splits/                    # Protocol B + freeze runner
scripts/                     # thin CLIs only
results/                     # forecasts, stats JSON, ALL plots
  figures/
  forecasts/
  stats/
  splits/
reports/
  report.tex / report.pdf
  figures/                 # copy of results/figures for LaTeX
  auxiliary/                 # side PDFs
```

## Problem

- **Target:** \(y_t = \log\mathrm{RV}_{t+1}\)
- **Metrics:** RMSE / MAE on log-RV; MAPE / QLIKE on RV; Diebold–Mariano (Newey–West)
- **Protocol B:** train → val (tune) → refit train+val → score test once
- **Ladder:** Naive → HAR → ARIMA → GARCH(1,1) → LSTM → HAR+LSTM hybrid

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src   # or: pip install -e .
```

## Pipeline

```bash
python scripts/fetch_binance_ohlcv.py
python scripts/clean_ohlcv.py
python scripts/build_returns_rv.py
python scripts/run_eda.py
python scripts/build_features.py
python scripts/run_freeze.py
python scripts/run_baselines.py
python scripts/run_lstm.py
python scripts/run_hybrid.py
python scripts/run_split_compare.py
python scripts/run_significance.py
```

All pipeline plots land in `results/figures/`; CLIs also copy them into `reports/figures/` for LaTeX.
