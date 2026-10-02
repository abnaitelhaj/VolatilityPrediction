# Running project notes

Capture bullets here during each step; polish the report only after comparison.

## Step 1 — Problem lock
- Predict **1-day-ahead log realized vol**, not price.
- Metrics: RMSE/MAE + QLIKE; time-ordered OOS only.
- Ladder: Naive → HAR → ARIMA → GARCH → LSTM.
- Data: Binance hourly bars aggregated to daily realized volatility.
- Outside the study: price prediction, trading rules, and technical-indicator feature sets.

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
- Sample: 3,298 complete days; report `reports/auxiliary/eda_report.pdf` (updated plots)
- Close/volume plain (no MA overlays, no stress shading); daily log returns; hourly return hist
- ACF/PACF of daily returns and log-RV (lag 0 shown, y in [-1,1])
- Rolling 30d skew/kurt, volume z, Bollinger, RSI (exploration only)
- Decision: target stays log-RV; core features start at lagged RV; extras optional later

## Step 6–7 — Target + feature engineering
- Target locked: \(y_t = \log\mathrm{RV}_{t+1}\)
- Script/report: `scripts/build_features.py`, `reports/auxiliary/fe_report.pdf`
- Panel: `data/processed/btcusdt_1d_features.parquet` (3,268 model-ready rows)
- Features: `log_rv_d`, `log_rv_w` (7d), `log_rv_m` (22d), `ret_d`, `ret_d_sq`, `volume_z30`
- Excluded from models: RSI, Bollinger, rolling skew/kurt, price MAs, min/max (EDA only)
- Corr with y (approx): HAR block ~0.62–0.67; `ret_d_sq` ~0.26; `volume_z30` ~0.31; `ret_d` ~−0.06

## Step 8 — Eval protocol frozen
- Protocol: `src/splits/protocol.py`; runner `scripts/run_freeze.py`; report `reports/auxiliary/freeze_report.pdf`
- This step only freezes protocol (no model baselines yet)
- Protocol B: train ≤2021-12-31 (1,540), val 2022–2023 (729, LSTM tune only), **final fit on train+val** (2,269), test ≥2024-01-01 (999)
- No shuffle; test never used for fitting/tuning
- Metrics: RMSE/MAE on log-RV; MAPE and QLIKE on RV
- Classical series model locked as **ARIMA**
- Lag-1 sanity on test: RMSE≈0.560, MAE≈0.444, QLIKE≈0.161

## Step 9 — Classical baselines
- `reports/auxiliary/model_plan.pdf` (if present) / model ladder in main report — per-model steps only (Naive, HAR, ARIMA, GARCH(1,1), LSTM)
- Shared: diagnose lightly → tune on train/val → refit train+val → test metrics
- Results: `reports/auxiliary/baselines_report.pdf` + `scripts/run_baselines.py`
- Classical test scores: HAR lowest error (RMSE 0.458 / QLIKE 0.101); ARIMA below Naive; GARCH higher log-error than HAR

## Step 10 — LSTM
- Script: `scripts/run_lstm.py`; report: `reports/auxiliary/lstm_report.pdf`
- Standalone LSTM. Inputs: length-$L$ window of log RV, 7-day mean, 22-day mean, squared return, and volume $z$.
- Grid on val (16 configs). Selected: L=60, hidden=128, depth=1, dropout=0.1, lr=1e-3, epoch=29
- Test: RMSE 0.454 / MAE 0.354 / MAPE 39.2\% / QLIKE 0.103. HAR: 0.458 / 0.354 / 40.7\% / 0.101.
- Artifacts: `results/stats/lstm_stats.json`, `results/forecasts/lstm_forecasts.parquet`, `results/figures/lstm_*.png`

## Step 11–12 — Comparison and writeup
- No Diebold–Mariano test and no high-vol / low-vol split.
- Source reports copied to `reports/auxiliary/`.
- Project report: `reports/report.pdf`.
- Ladder order: Naive → GARCH (exit) → ARIMA → HAR → LSTM → Hybrid.
- Split compare (HAR vs LSTM): `reports/split_compare_report.pdf`.
- Hybrid: `scripts/run_hybrid.py`; test RMSE 0.436 / MAE 0.342 / MAPE 38.3% / QLIKE 0.094 (w=0.8).
 
