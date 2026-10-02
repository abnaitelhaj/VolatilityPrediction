"""
Clean and wrangle Binance hourly OHLCV.

Cleaning: types, sort, duplicate check, OHLC flags. No price imputation.
Wrangling: UTC day key + gap size on the hourly table; day-level QC on its own table.
Raw files are read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paths import DATA_PROCESSED, RAW_OHLCV

RAW_PATH = RAW_OHLCV
OUT_DIR = DATA_PROCESSED
HOUR_MS = 60 * 60 * 1000

PRICE_COLS = ["open", "high", "low", "close"]


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(RAW_PATH)
    expected = {"timestamp", "datetime_utc", "open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"raw file missing columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = df.copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("int64")
    out["datetime_utc"] = pd.to_datetime(out["datetime_utc"], utc=True)
    for col in PRICE_COLS + ["volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values("timestamp").reset_index(drop=True)
    n_dupes = int(out["timestamp"].duplicated().sum())
    if n_dupes:
        raise ValueError(f"duplicate timestamps in raw: {n_dupes}")

    not_on_hour = out["datetime_utc"].dt.minute.ne(0) | out["datetime_utc"].dt.second.ne(0)
    null_price = out[PRICE_COLS + ["volume"]].isna().any(axis=1)
    nonpositive = (out[PRICE_COLS] <= 0).any(axis=1)
    high_lt_low = out["high"] < out["low"]
    high_below_body = out["high"] < out[["open", "close"]].max(axis=1)
    low_above_body = out["low"] > out[["open", "close"]].min(axis=1)
    bad_volume = out["volume"] < 0

    out["ohlc_ok"] = ~(
        not_on_hour | null_price | nonpositive | high_lt_low | high_below_body | low_above_body | bad_volume
    )

    rejected = out.loc[~out["ohlc_ok"]].copy()
    clean_df = out.loc[out["ohlc_ok"]].drop(columns=["ohlc_ok"]).reset_index(drop=True)

    summary = {
        "rows_in": int(len(df)),
        "rows_out": int(len(clean_df)),
        "rows_rejected": int(len(rejected)),
        "duplicate_timestamps": n_dupes,
        "reject_reasons": {
            "not_on_hour": int(not_on_hour.sum()),
            "null_price_or_volume": int(null_price.sum()),
            "nonpositive_price": int(nonpositive.sum()),
            "high_lt_low": int(high_lt_low.sum()),
            "high_below_open_close": int(high_below_body.sum()),
            "low_above_open_close": int(low_above_body.sum()),
            "negative_volume": int(bad_volume.sum()),
        },
        "imputation": "none",
    }
    return clean_df, rejected, summary


def wrangle(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly = clean_df.copy()
    delta_ms = hourly["timestamp"].diff()
    hourly["gap_before_hours"] = delta_ms / HOUR_MS
    hourly["date_utc"] = hourly["datetime_utc"].dt.floor("D")

    gaps = hourly.loc[hourly["gap_before_hours"].gt(1), [
        "datetime_utc", "date_utc", "gap_before_hours"
    ]].copy()
    gaps["prev_datetime_utc"] = hourly["datetime_utc"].shift(1).loc[gaps.index]
    gaps = gaps.reset_index(drop=True)

    daily = (
        hourly.groupby("date_utc", as_index=False)
        .agg(n_bars=("timestamp", "size"), volume=("volume", "sum"))
    )
    daily["is_complete_day"] = daily["n_bars"].eq(24)
    daily["missing_hours"] = (24 - daily["n_bars"]).clip(lower=0)

    hourly = hourly[[
        "timestamp", "datetime_utc", "date_utc",
        "open", "high", "low", "close", "volume",
        "gap_before_hours",
    ]]
    return hourly, daily, gaps


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    clean_df, rejected, summary = clean(raw)
    hourly, daily, gaps = wrangle(clean_df)

    hourly_path = OUT_DIR / "btcusdt_1h_clean.parquet"
    daily_path = OUT_DIR / "btcusdt_1d_qc.parquet"
    gaps_path = OUT_DIR / "btcusdt_1h_gaps.parquet"
    summary_path = OUT_DIR / "clean_summary.json"

    hourly.to_parquet(hourly_path, index=False)
    daily.to_parquet(daily_path, index=False)
    gaps.to_parquet(gaps_path, index=False)

    summary.update({
        "timestamp_min_utc": hourly["datetime_utc"].iloc[0].isoformat(),
        "timestamp_max_utc": hourly["datetime_utc"].iloc[-1].isoformat(),
        "n_days": int(len(daily)),
        "n_complete_days": int(daily["is_complete_day"].sum()),
        "n_incomplete_days": int((~daily["is_complete_day"]).sum()),
        "n_gaps": int(len(gaps)),
        "rules": {
            "timezone": "UTC",
            "complete_day": "exactly 24 distinct hourly bars on that UTC date",
            "gaps": "not filled; gap_before_hours on hourly table; gap list saved separately",
            "is_complete_day": "daily QC table only, not repeated on hourly rows",
        },
    })
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(rejected):
        rejected.to_parquet(OUT_DIR / "btcusdt_1h_rejected.parquet", index=False)


if __name__ == "__main__":
    main()
