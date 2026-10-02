"""
Step 2: fetch Binance OHLCV into data/raw/.

Raw policy:
- Concatenate paginated API responses only.
- Sort by timestamp and drop exact duplicate timestamps from page overlap
  (documented in the sidecar metadata). No gap-filling or value edits.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

from paths import DATA_RAW

RAW_DIR = DATA_RAW
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
EXCHANGE_ID = "binance"
# Binance spot BTCUSDT roughly from mid-Aug 2017
SINCE_MS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
LIMIT = 1000
SLEEP_S = 0.2


def fetch_all() -> tuple[pd.DataFrame, dict]:
    exchange = ccxt.binance({"enableRateLimit": True})
    all_rows: list[list] = []
    since = SINCE_MS
    pages = 0

    while True:
        batch = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=LIMIT)
        pages += 1
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        # advance past last candle to avoid tight loops on the same page
        next_since = last_ts + 1
        if next_since <= since:
            break
        since = next_since
        # last incomplete candle may still be forming; stop when page is short
        if len(batch) < LIMIT:
            break
        time.sleep(SLEEP_S)

    df_api = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    n_before = len(df_api)
    df = df_api.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    n_after = len(df)
    df = df.reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    meta = {
        "exchange": EXCHANGE_ID,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "since_requested_utc": datetime.fromtimestamp(SINCE_MS / 1000, tz=timezone.utc).isoformat(),
        "pages": pages,
        "rows_concatenated": n_before,
        "rows_after_dedupe": n_after,
        "duplicate_timestamps_dropped": n_before - n_after,
        "timestamp_min_utc": df["datetime_utc"].iloc[0].isoformat() if n_after else None,
        "timestamp_max_utc": df["datetime_utc"].iloc[-1].isoformat() if n_after else None,
        "raw_policy": (
            "Paginated CCXT fetch_ohlcv concatenated; sorted by timestamp; "
            "exact duplicate timestamps from page overlap dropped (keep last). "
            "No gap filling, no price/volume edits."
        ),
    }
    return df, meta


def sanity(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"empty": True}

    ts = df["timestamp"]
    deltas = ts.diff().dropna()
    expected_ms = 60 * 60 * 1000
    gap_mask = deltas != expected_ms
    n_gaps = int(gap_mask.sum())
    gap_examples = []
    if n_gaps:
        gap_idx = deltas[gap_mask].index[:5]
        for i in gap_idx:
            gap_examples.append(
                {
                    "prev_utc": df.loc[i - 1, "datetime_utc"].isoformat(),
                    "curr_utc": df.loc[i, "datetime_utc"].isoformat(),
                    "delta_hours": float(deltas.loc[i] / expected_ms),
                }
            )

    return {
        "rows": int(len(df)),
        "null_counts": {c: int(df[c].isna().sum()) for c in ["open", "high", "low", "close", "volume"]},
        "nonpositive_close": int((df["close"] <= 0).sum()),
        "high_lt_low": int((df["high"] < df["low"]).sum()),
        "duplicate_timestamps_remaining": int(df["timestamp"].duplicated().sum()),
        "non_hourly_steps": n_gaps,
        "gap_examples": gap_examples,
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {EXCHANGE_ID} {SYMBOL} {TIMEFRAME} since {SINCE_MS}...")
    df, meta = fetch_all()
    report = sanity(df)
    meta["sanity"] = report

    stem = "btcusdt_1h_binance"
    csv_path = RAW_DIR / f"{stem}.csv"
    parquet_path = RAW_DIR / f"{stem}.parquet"
    meta_path = RAW_DIR / f"{stem}.meta.json"

    # Persist API fields + utc datetime helper column
    out = df[["timestamp", "datetime_utc", "open", "high", "low", "close", "volume"]]
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    meta["files"] = {"csv": str(csv_path.name), "parquet": str(parquet_path.name)}
    meta_path.write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {parquet_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
