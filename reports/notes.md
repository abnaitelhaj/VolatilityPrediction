# Running project notes

Capture bullets here during each step; polish the report only after comparison.

## Step 1 — Problem lock
- Predict **1-day-ahead log realized vol**, not price.
- Metrics: RMSE/MAE + QLIKE; time-ordered OOS only.
- Ladder: Naive → HAR → SARIMA → GARCH → EGARCH/GJR → LSTM.
- Data intent: Binance hourly → daily RV.
- Dropped: DE lakehouse, trading bot, indicator soup by default.

## Step 2 — Data
- Source: Binance spot via CCXT (`scripts/fetch_binance_ohlcv.py`)
- Pair / bar: `BTC/USDT`, `1h`
- Files: `data/raw/btcusdt_1h_binance.{csv,parquet,meta.json}`
- Range: 2017-08-17 → 2026-09-27 (UTC), **79,754** rows
- Raw policy: concat pages only; sort + drop exact duplicate timestamps (0 dropped); **no gap fill**
- Sanity: 0 nulls, 0 bad OHLC; **28** non-hourly steps (exchange gaps / outages) — leave for step 3
- Env: project `.venv` + `requirements.txt`


## Step 3 — Minimal prep
- 

## Step 4 — EDA
- 
