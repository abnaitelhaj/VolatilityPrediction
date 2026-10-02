"""Canonical project paths."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FORECASTS = RESULTS / "forecasts"
STATS = RESULTS / "stats"
SPLITS_DIR = RESULTS / "splits"
REPORTS = ROOT / "reports"
REPORT_FIGURES = REPORTS / "figures"
FEATURES_PATH = DATA_PROCESSED / "btcusdt_1d_features.parquet"
RAW_OHLCV = DATA_RAW / "btcusdt_1h_binance.parquet"
HOURLY_CLEAN = DATA_PROCESSED / "btcusdt_1h_clean.parquet"
HOURLY_RETURNS = DATA_PROCESSED / "btcusdt_1h_returns.parquet"
DAILY_RV = DATA_PROCESSED / "btcusdt_1d_rv.parquet"
DAILY_QC = DATA_PROCESSED / "btcusdt_1d_qc.parquet"


def ensure_result_dirs() -> None:
    for p in (FIGURES, FORECASTS, STATS, SPLITS_DIR, REPORT_FIGURES):
        p.mkdir(parents=True, exist_ok=True)


def sync_report_figures() -> None:
    """Copy results/figures → reports/figures for LaTeX (no symlinks)."""
    ensure_result_dirs()
    for src in FIGURES.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(FIGURES)
        dst = REPORT_FIGURES / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
