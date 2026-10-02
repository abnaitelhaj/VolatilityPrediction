"""Train / val / test metrics for HAR and LSTM from saved forecasts.

Protocol B final fits use train+val, so train and val scores are in-sample.
Test is the only clean OOS window.

One figure per split. X-axis = metrics; grouped bars = HAR vs LSTM.
MAPE is plotted as a fraction (percent / 100) so it shares the axis with
RMSE, MAE, and QLIKE. Figures land in reports/figures/ (consumed by untouched TeX).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import FIGURES, FORECASTS, STATS, ensure_result_dirs
from metrics.scores import metrics_from_log_forecast

FIG = FIGURES

C_HAR = "#2563EB"
C_LSTM = "#DC2626"
METRIC_KEYS = ["rmse_log", "mae_log", "mape_rv", "qlike"]
METRIC_LABELS = ["RMSE", "MAE", "MAPE/100", "QLIKE"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def load_panel() -> pd.DataFrame:
    base = pd.read_parquet(FORECASTS / "baseline_forecasts.parquet")
    lstm = pd.read_parquet(FORECASTS / "lstm_forecasts.parquet")
    return base.merge(lstm[["date_utc", "yhat_lstm"]], on="date_utc", how="left")


def score_panel(df: pd.DataFrame) -> dict:
    specs = {
        "HAR": "yhat_har",
        "LSTM": "yhat_lstm",
    }
    board: dict = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        board[split] = {}
        for name, col in specs.items():
            m = part.dropna(subset=["y", col])
            s = metrics_from_log_forecast(m["y"].to_numpy(), m[col].to_numpy())
            board[split][name] = {k: float(v) for k, v in s.items()}
    return board


def plot_split(board: dict, split: str, outfile: str) -> None:
    x = np.arange(len(METRIC_KEYS))
    width = 0.36
    har_vals = []
    lstm_vals = []
    for key in METRIC_KEYS:
        h = board[split]["HAR"][key]
        l = board[split]["LSTM"][key]
        if key == "mape_rv":
            h, l = h / 100.0, l / 100.0
        har_vals.append(h)
        lstm_vals.append(l)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(
        x - width / 2,
        har_vals,
        width=width,
        color=C_HAR,
        edgecolor="white",
        linewidth=0.6,
        label="HAR",
    )
    ax.bar(
        x + width / 2,
        lstm_vals,
        width=width,
        color=C_LSTM,
        edgecolor="white",
        linewidth=0.6,
        label="LSTM",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_title(f"{split.capitalize()} scores")
    ax.legend(frameon=False, ncol=2)
    save(fig, outfile)


def run() -> None:
    ensure_result_dirs()
    df = load_panel()
    board = score_panel(df)
    out = {
        "note": (
            "Protocol B final fits use train+val. Train and val metrics are "
            "in-sample for HAR and LSTM. Test is clean OOS. "
            "LSTM drops the first L-1 train rows (lookback). "
            "Figures plot MAPE as a fraction (percent/100)."
        ),
        "splits": board,
        "lstm_train_n": int(board["train"]["LSTM"]["n"]),
        "lstm_train_dropped": int(board["train"]["HAR"]["n"] - board["train"]["LSTM"]["n"]),
    }
    (STATS / "split_compare_stats.json").write_text(json.dumps(out, indent=2))

    for split in ("train", "val", "test"):
        plot_split(board, split, f"split_{split}.png")

    for old in ("split_rmse.png", "split_mae.png", "split_mape.png", "split_qlike.png"):
        p = FIG / old
        if p.exists():
            p.unlink()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
