"""Diebold–Mariano tests on the test window (HAR, LSTM, Hybrid).

One-step-ahead loss differentials with Newey–West HAC standard errors.
Figures only for this significance note.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
MODELS = ROOT / "models"

C_HAR = "#2563EB"
C_LSTM = "#7C3AED"
C_HYB = "#059669"
C_SIG = "#059669"
C_NS = "#9CA3AF"
C_ZERO = "#6B7280"

LOSS_KEYS = ["se", "ae", "ape", "qlike"]
LOSS_SHORT = ["RMSE", "MAE", "MAPE", "QLIKE"]
LOSS_NAMES = {
    "se": "Squared log-error (RMSE)",
    "ae": "Absolute log-error (MAE)",
    "ape": "Absolute % RV error (MAPE)",
    "qlike": "QLIKE",
}
PAIRS = [("LSTM", "HAR"), ("HAR", "LSTM"), ("Hybrid", "HAR"), ("Hybrid", "LSTM")]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 140,
    "savefig.dpi": 170,
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


def nw_mean_se(d: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    n = len(d)
    mu = float(np.mean(d))
    if lag is None:
        lag = int(np.floor(n ** (1.0 / 3.0)))
    gamma0 = float(np.mean((d - mu) ** 2))
    var = gamma0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        cov = float(np.mean((d[j:] - mu) * (d[:-j] - mu)))
        var += 2.0 * w * cov
    se = float(np.sqrt(max(var, 0.0) / n))
    return mu, se


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    mu, se = nw_mean_se(d)
    if se < 1e-15:
        stat, p = 0.0, 1.0
    else:
        stat = mu / se
        p = float(2.0 * stats.norm.sf(abs(stat)))
    return {
        "mean_diff": float(mu),
        "nw_se": float(se),
        "dm_stat": float(stat),
        "p_value": float(p),
        "n": int(len(d)),
        "better": "A" if mu < 0 else ("B" if mu > 0 else "tie"),
        "d": d,
    }


def qlike_series(rv: np.ndarray, rv_hat: np.ndarray) -> np.ndarray:
    ratio = rv / rv_hat
    return ratio - np.log(ratio) - 1.0


def load_test() -> pd.DataFrame:
    base = pd.read_parquet(MODELS / "baseline_forecasts.parquet")
    lstm = pd.read_parquet(MODELS / "lstm_forecasts.parquet")
    hyb = pd.read_parquet(MODELS / "hybrid_forecasts.parquet")
    df = base.loc[base["split"] == "test", ["date_utc", "y", "yhat_har"]].copy()
    df = df.merge(lstm.loc[lstm["split"] == "test", ["date_utc", "yhat_lstm"]], on="date_utc")
    df = df.merge(hyb.loc[hyb["split"] == "test", ["date_utc", "yhat_hybrid"]], on="date_utc")
    df = df.dropna(subset=["y", "yhat_har", "yhat_lstm", "yhat_hybrid"]).reset_index(drop=True)
    df["rv"] = np.exp(df["y"].to_numpy(dtype=float))
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
    return df


def losses(y: np.ndarray, yhat: np.ndarray, rv: np.ndarray) -> dict[str, np.ndarray]:
    e = yhat - y
    rv_hat = np.exp(yhat)
    return {
        "se": e**2,
        "ae": np.abs(e),
        "ape": np.abs(rv_hat - rv) / rv,
        "qlike": qlike_series(rv, rv_hat),
    }


def plot_mean_losses(L: dict) -> None:
    # MAPE shown as fraction for shared axis with other means that are small;
    # SE mean is also small; use separate panels per loss for honesty.
    models = ["HAR", "LSTM", "Hybrid"]
    colors = [C_HAR, C_LSTM, C_HYB]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.2))
    for ax, key, title in zip(axes, LOSS_KEYS, LOSS_SHORT):
        vals = [float(np.mean(L[m][key])) for m in models]
        if key == "ape":
            vals = [100.0 * v for v in vals]
            title = "MAPE (%)"
        ax.bar(models, vals, color=colors, width=0.68, edgecolor="white", linewidth=0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Test mean loss by model", y=1.03, fontsize=13, fontweight="bold")
    save(fig, "significance_mean_loss.png")


def _dm_panel(ax, board: dict, key: str, title: str) -> None:
    tests = board[key]["tests"]
    x = np.arange(len(LOSS_KEYS))
    stats_ = [tests[k]["dm_stat"] for k in LOSS_KEYS]
    ps = [tests[k]["p_value"] for k in LOSS_KEYS]
    cols = [C_SIG if p < 0.05 else C_NS for p in ps]
    ax.bar(x, stats_, color=cols, width=0.72, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=C_ZERO, lw=1.0)
    ax.axhline(1.96, color=C_ZERO, lw=0.8, ls="--")
    ax.axhline(-1.96, color=C_ZERO, lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(LOSS_SHORT)
    ax.set_title(title)
    ax.set_ylim(-6.2, 6.2)


def plot_dm_bars(board: dict) -> None:
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.2), sharey=True)
    titles = ["LSTM vs HAR", "HAR vs LSTM", "Hybrid vs HAR", "Hybrid vs LSTM"]
    keys = ["LSTM_vs_HAR", "HAR_vs_LSTM", "Hybrid_vs_HAR", "Hybrid_vs_LSTM"]
    flat = axes.ravel()
    for ax, title, key in zip(flat, titles, keys):
        _dm_panel(ax, board, key, title)
    flat[0].set_ylabel("DM statistic")
    flat[2].set_ylabel("DM statistic")
    fig.suptitle(
        "Diebold–Mariano stats (green = p < 0.05; dashed = ±1.96)",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    flat[1].legend(
        handles=[
            Patch(facecolor=C_SIG, edgecolor="white", label="Significant"),
            Patch(facecolor=C_NS, edgecolor="white", label="Not significant"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    save(fig, "significance_dm_bars.png")

    # Focused figures for the main project report
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _dm_panel(ax, board, "LSTM_vs_HAR", "LSTM vs HAR")
    ax.set_ylabel("DM statistic")
    ax.legend(
        handles=[
            Patch(facecolor=C_SIG, edgecolor="white", label="Significant"),
            Patch(facecolor=C_NS, edgecolor="white", label="Not significant"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower left",
    )
    fig.suptitle(
        "Diebold–Mariano (green = p < 0.05; dashed = ±1.96)",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "dm_lstm_vs_har.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    _dm_panel(axes[0], board, "Hybrid_vs_HAR", "Hybrid vs HAR")
    _dm_panel(axes[1], board, "Hybrid_vs_LSTM", "Hybrid vs LSTM")
    axes[0].set_ylabel("DM statistic")
    axes[1].legend(
        handles=[
            Patch(facecolor=C_SIG, edgecolor="white", label="Significant"),
            Patch(facecolor=C_NS, edgecolor="white", label="Not significant"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )
    fig.suptitle(
        "Diebold–Mariano after hybrid (green = p < 0.05; dashed = ±1.96)",
        y=1.03,
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "dm_hybrid_vs_both.png")


def plot_diff_paths(df: pd.DataFrame, L: dict) -> None:
    """Cumulative loss advantage: negative slope means first model better."""
    dates = df["date_utc"]
    d_lh = L["LSTM"]["se"] - L["HAR"]["se"]
    series = [
        ("LSTM − HAR", d_lh, C_LSTM),
        ("HAR − LSTM", -d_lh, C_HAR),
        ("Hybrid − HAR", L["Hybrid"]["se"] - L["HAR"]["se"], C_HYB),
        ("Hybrid − LSTM", L["Hybrid"]["se"] - L["LSTM"]["se"], "#CA8A04"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.8), sharex=True)
    for ax, (label, d, color) in zip(axes.ravel(), series):
        ax.plot(dates, np.cumsum(d), color=color, lw=1.35)
        ax.axhline(0, color=C_ZERO, lw=0.9)
        ax.set_ylabel("Cumulative $d_t$")
        ax.set_title(f"Squared log-error: {label}")
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    for ax in axes[1, :]:
        ax.set_xlabel("Test date")
    fig.suptitle(
        "Cumulative loss differential (down = left model better on RMSE loss)",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "significance_cumdiff.png")


def main() -> None:
    df = load_test()
    y = df["y"].to_numpy(dtype=float)
    rv = df["rv"].to_numpy(dtype=float)
    L = {
        "HAR": losses(y, df["yhat_har"].to_numpy(dtype=float), rv),
        "LSTM": losses(y, df["yhat_lstm"].to_numpy(dtype=float), rv),
        "Hybrid": losses(y, df["yhat_hybrid"].to_numpy(dtype=float), rv),
    }

    board = {}
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        board[key] = {"A": a, "B": b, "tests": {}}
        for loss_key, loss_label in LOSS_NAMES.items():
            res = dm_test(L[a][loss_key], L[b][loss_key])
            d = res.pop("d")
            res["loss"] = loss_label
            res["better_model"] = a if res["better"] == "A" else (b if res["better"] == "B" else "tie")
            board[key]["tests"][loss_key] = res
            board[key]["tests"][loss_key]["_d_mean_check"] = float(np.mean(d))

    plot_mean_losses(L)
    plot_dm_bars(board)
    plot_diff_paths(df, L)

    # strip helper keys before save
    for key in board:
        for loss_key in board[key]["tests"]:
            board[key]["tests"][loss_key].pop("_d_mean_check", None)

    stats = {
        "n_test": int(len(df)),
        "hac": "Newey-West, lag = floor(n^(1/3))",
        "null": "Equal predictive accuracy (E[L_A - L_B] = 0)",
        "sign": "mean_diff = mean(L_A - L_B); negative => A better",
        "pairs": board,
        "figures": [
            "figures/significance_mean_loss.png",
            "figures/significance_dm_bars.png",
            "figures/significance_cumdiff.png",
            "figures/dm_lstm_vs_har.png",
            "figures/dm_hybrid_vs_both.png",
        ],
    }
    (OUT / "significance_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("n_test", "figures")}, indent=2))
    for key, block in board.items():
        print(key, {lk: round(t["p_value"], 4) for lk, t in block["tests"].items()})


if __name__ == "__main__":
    main()
