"""
Minimal prep: hourly log returns and provisional daily realized volatility.

A return is kept only when the previous bar is exactly one hour earlier.
Missing hours are not filled, so a return never jumps a gap.
Daily RV uses valid returns whose ending bar falls on that UTC date.
Incomplete days stay in the daily table and are flagged, not dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
HOURLY_PATH = OUT_DIR / "btcusdt_1h_clean.parquet"
DAILY_QC_PATH = OUT_DIR / "btcusdt_1d_qc.parquet"


def main() -> None:
    hourly = pd.read_parquet(HOURLY_PATH).sort_values("timestamp").reset_index(drop=True)
    daily_qc = pd.read_parquet(DAILY_QC_PATH)

    prev_close = hourly["close"].shift(1)
    gap_ok = hourly["gap_before_hours"].eq(1.0)
    prices_ok = hourly["close"].gt(0) & prev_close.gt(0)
    hourly["return_valid"] = gap_ok & prices_ok
    hourly["log_return"] = np.where(
        hourly["return_valid"],
        np.log(hourly["close"] / prev_close),
        np.nan,
    )

    valid = hourly.loc[hourly["return_valid"], ["date_utc", "log_return"]]
    grouped = valid.groupby("date_utc", as_index=False).agg(
        n_valid_returns=("log_return", "size"),
        rv=("log_return", lambda s: float(np.sqrt(np.square(s.to_numpy()).sum()))),
    )

    daily = daily_qc[["date_utc", "n_bars", "missing_hours", "is_complete_day"]].merge(
        grouped, on="date_utc", how="left"
    )
    daily["n_valid_returns"] = daily["n_valid_returns"].fillna(0).astype(int)
    daily["rv_uses_24_returns"] = daily["n_valid_returns"].eq(24)
    daily["log_rv"] = np.where(daily["rv"].gt(0), np.log(daily["rv"]), np.nan)

    hourly_out = hourly[[
        "timestamp", "datetime_utc", "date_utc", "close", "log_return", "return_valid",
    ]]
    hourly_path = OUT_DIR / "btcusdt_1h_returns.parquet"
    daily_path = OUT_DIR / "btcusdt_1d_rv.parquet"
    summary_path = OUT_DIR / "returns_rv_summary.json"

    hourly_out.to_parquet(hourly_path, index=False)
    daily.to_parquet(daily_path, index=False)

    summary = {
        "hourly_rows": int(len(hourly_out)),
        "valid_returns": int(hourly_out["return_valid"].sum()),
        "skipped_returns": int((~hourly_out["return_valid"]).sum()),
        "daily_rows": int(len(daily)),
        "days_with_rv": int(daily["rv"].notna().sum()),
        "days_log_rv": int(daily["log_rv"].notna().sum()),
        "complete_days": int(daily["is_complete_day"].sum()),
        "days_with_24_valid_returns": int(daily["rv_uses_24_returns"].sum()),
        "rv_definition": "sqrt(sum of squared hourly log returns ending on that UTC date)",
        "return_rule": "log(close_t / close_t-1) only if gap_before_hours == 1 and both closes > 0",
        "incomplete_days": "kept and flagged via is_complete_day; not used to fill prices",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
