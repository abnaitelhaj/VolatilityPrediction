# Bitcoin Realized Volatility Forecasting

Comparative study of classical econometric models and deep learning for **short-term Bitcoin realized volatility** forecasting.

## 1. Problem definition (locked)

### Research question
Can deep learning (LSTM) outperform standard quant baselines (Naive, HAR, SARIMA, GARCH-family) for **1-day-ahead Bitcoin realized volatility**, under a strict time-series out-of-sample protocol?

### Target
- **Variable:** log realized volatility, \(\log(\mathrm{RV}_t)\)
- **Construction (provisional, finalize after EDA):** daily RV from **intraday** (hourly) Binance BTC/USDT returns:
  \[
  \mathrm{RV}_t = \sqrt{\sum_{i \in \mathrm{day}\, t} r_{t,i}^2}
  \]
- **Horizon:** 1-step ahead (forecast \(\log(\mathrm{RV}_{t+1})\) using information available at \(t\))

### Why volatility (not price)
Volatility is the natural object for risk, sizing, and options. Price direction is a different (harder, noisier) problem and is **out of scope**.

### Metrics
- **RMSE** and **MAE** on \(\log(\mathrm{RV})\) (or on RV — pick one scale and keep it fixed)
- **QLIKE** (volatility-forecasting standard)
- Optional: Diebold–Mariano test for pairwise forecast superiority

### Evaluation protocol
- **No shuffled splits** — time-ordered only
- Primary: expanding or rolling window OOS (same protocol for every model)
- Secondary: error breakdown in **high-vol vs low-vol** regimes

### Model ladder (in order)
1. Naive (e.g. \(\widehat{\mathrm{RV}}_{t+1} = \mathrm{RV}_t\))
2. HAR-RV
3. SARIMA on \(\log(\mathrm{RV})\)
4. GARCH(1,1) on returns → variance forecast aligned to RV horizon
5. EGARCH or GJR-GARCH (asymmetry)
6. LSTM / GRU (same target, horizon, OOS)

### Explicitly out of scope
- Price prediction / trading bot / RL
- Lakehouse / Spark / MinIO theater
- Indicator soup (RSI, Bollinger, etc.) unless EDA clearly justifies it
- Claiming live PnL or “alpha”

### Data (intent)
- **Primary:** Binance BTC/USDT OHLCV via CCXT (hourly → daily RV)
- yfinance only as a fallback, not the preferred source

### Success bar
A serious project means: locked target, leakage-safe features, strong baselines before DL, honest OOS comparison, clear writeup of where models fail.

---

## Status
- [x] Step 1 — Problem locked
- [x] Step 2 — Data (`data/raw/btcusdt_1h_binance.*`, Binance `BTC/USDT` 1h, 2017-08 → 2026-09)
- [x] Step 3 — Clean + wrangle (`btcusdt_1h_clean`, `btcusdt_1d_qc`; gaps flagged, not filled)
- [x] Step 4 — Minimal prep (`btcusdt_1h_returns`, `btcusdt_1d_rv`)
- [x] Step 5 — EDA (`eda/eda_report.pdf`)
- [x] Step 6 — Finalize target (\(y_t=\log\mathrm{RV}_{t+1}\))
- [x] Step 7 — Feature engineering (`feature_eng/`, `btcusdt_1d_features.parquet`)
- [ ] Step 8 — Freeze eval protocol in code
- [ ] Step 9 — Baselines
- [ ] Step 10 — LSTM
- [ ] Step 11 — Compare
- [ ] Step 12 — Writeup

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_binance_ohlcv.py   # refreshes data/raw/
python scripts/clean_ohlcv.py          # writes data/processed/ clean + daily QC
python scripts/build_returns_rv.py     # hourly log returns + daily RV
python feature_eng/build_features.py   # daily feature panel + FE figures
```
