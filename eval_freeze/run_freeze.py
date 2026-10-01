"""Freeze eval protocol: write splits, sanity-check metrics, figures for freeze_report.pdf.

This is not the baselines step — only protocol lock + a lag-1 metric sanity check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator

OUT = Path(__file__).resolve().parent
if str(OUT) not in sys.path:
    sys.path.insert(0, str(OUT))

from protocol import (
    LOG_RV_COL,
    TARGET_COL,
    TRAIN_END,
    VAL_END,
    add_split_column,
    evaluate_split,
    load_model_ready,
    naive_log_rv_forecast,
    split_summaries,
)
FIG = OUT / "figures"

NAVY = "#1B3A4B"
TEAL = "#3D7A8C"
CORAL = "#C45C26"
GOLD = "#D4A017"
SLATE = "#5C6B73"
BG = "#FAFBFC"
GRID = "#E6EAEF"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
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
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = add_split_column(load_model_ready())
    specs = split_summaries(df)
    yhat = naive_log_rv_forecast(df)

    split_metrics = {
        s.name: evaluate_split(df, yhat, s.name) for s in specs
    }

    # Persist split labels for later models
    split_table = df[["date_utc", "split", TARGET_COL, LOG_RV_COL, "rv"]].copy()
    split_table["yhat_naive"] = yhat
    split_path = OUT / "splits.parquet"
    split_table.to_parquet(split_path, index=False)

    stats = {
        "protocol": "B_train_val_test_refit_train_val",
        "train_end": TRAIN_END.date().isoformat(),
        "val_end": VAL_END.date().isoformat(),
        "val_usage": "LSTM hyperparameters / early stopping only; then final refit on train+val",
        "refit_on_train_val": True,
        "target": "y = log_rv_{t+1}",
        "metrics": ["rmse_log", "mae_log", "qlike"],
        "leaderboard_split": "test",
        "final_fit_rows": int(df["split"].isin(["train", "val"]).sum()),
        "splits": {
            s.name: {
                "start": s.start.date().isoformat(),
                "end": s.end.date().isoformat(),
                "n": s.n,
            }
            for s in specs
        },
        "naive": {
            "definition": "yhat_t = log_rv_t",
            "note": "Naive has no parameters; refit does not change forecasts",
            "by_split": {
                k: {m: v[m] for m in ("rmse_log", "mae_log", "qlike", "n")}
                for k, v in split_metrics.items()
            },
            "test_leaderboard": {
                m: split_metrics["test"][m] for m in ("rmse_log", "mae_log", "qlike", "n")
            },
        },
    }
    (OUT / "freeze_stats.json").write_text(json.dumps(stats, indent=2))

    # --- figures ---
    colors = {"train": NAVY, "val": GOLD, "test": CORAL}
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for name, part in df.groupby("split", sort=False):
        ax.plot(part["date_utc"], part[TARGET_COL], color=colors[name], lw=0.85, label=name)
    ax.axvline(TRAIN_END, color=SLATE, lw=1.0, ls="--")
    ax.axvline(VAL_END, color=SLATE, lw=1.0, ls="--")
    ax.set_ylabel(r"$y_t=\log\mathrm{RV}_{t+1}$")
    ax.set_title("Frozen chronological splits")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.grid(True)
    save(fig, "splits_timeline.png")

    # bar metrics on test (and all splits) for Naive
    metric_names = ["rmse_log", "mae_log", "qlike"]
    labels = ["RMSE (log)", "MAE (log)", "QLIKE"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, m, lab in zip(axes, metric_names, labels):
        vals = [split_metrics[s][m] for s in ("train", "val", "test")]
        ax.set_facecolor("white")
        ax.bar(["train", "val", "test"], vals, color=[NAVY, GOLD, CORAL], width=0.65)
        ax.set_title(lab)
        ax.set_ylabel(lab)
    fig.suptitle("Lag-1 sanity metrics by split", y=1.02, fontsize=13, fontweight="bold")
    save(fig, "naive_metrics.png")

    # test-period forecast vs actual
    test = df.loc[df["split"] == "test"].copy()
    test_hat = yhat.loc[test.index]
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(test["date_utc"], test[TARGET_COL], color=NAVY, lw=0.9, label="Actual $y$")
    ax.plot(test["date_utc"], test_hat, color=CORAL, lw=0.9, alpha=0.85, label=r"Naive $\hat y$")
    ax.set_ylabel(r"$\log\mathrm{RV}$")
    ax.set_title("Test set sanity: actual vs lag-1 log-RV")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.grid(True)
    save(fig, "naive_test_path.png")

    # residual hist on test
    resid = test[TARGET_COL].to_numpy() - test_hat.to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(resid, bins=45, color=TEAL, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=CORAL, lw=1.2)
    ax.set_xlabel(r"$y - \hat y$")
    ax.set_ylabel("Days")
    ax.set_title("Lag-1 test residuals (sanity)")
    save(fig, "naive_test_resid.png")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
