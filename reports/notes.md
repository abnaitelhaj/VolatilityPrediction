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


## Step 3 — Clean + wrangle
- Script: `scripts/clean_ohlcv.py` (raw file not modified; **no imputation**)
- Hourly: `data/processed/btcusdt_1h_clean.parquet` — OHLCV, `date_utc`, `gap_before_hours`
- Daily QC: `data/processed/btcusdt_1d_qc.parquet` — `n_bars`, `is_complete_day`, `missing_hours` (not repeated on hourly rows)
- Gaps: `data/processed/btcusdt_1h_gaps.parquet` (28 breaks)
- Result: 79,754 rows kept, **0 rejected**; **3,298 / 3,329** UTC days complete (24 hourly bars); 31 incomplete (partial first/last day + gap days)

## Step 4 — Minimal prep
- Script: `scripts/build_returns_rv.py`
- Hourly: `data/processed/btcusdt_1h_returns.parquet` — `log_return` only if the previous bar is exactly 1h earlier (79,725 valid; 29 skipped = 28 gaps + first bar)
- Daily: `data/processed/btcusdt_1d_rv.parquet` — `rv = sqrt(sum r^2)`, `log_rv`, `is_complete_day`, `rv_uses_24_returns`
- Incomplete days kept and flagged, not dropped. All 3,298 complete days also have 24 valid returns.
- Median daily RV on those days ≈ 0.024; no model features yet

## Step 5 — EDA
- 
