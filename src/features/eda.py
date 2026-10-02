"""EDA for complete UTC days — EDA figures → results/figures/."""

from __future__ import annotations

from paths import FIGURES, FORECASTS, STATS, SPLITS_DIR, ensure_result_dirs
from splits.protocol import (
    LOG_RV_COL,
    TARGET_COL,
    TRAIN_END,
    VAL_END,
    add_split_column,
    fit_mask,
    load_model_ready,
    naive_log_rv_forecast,
    split_summaries,
)
from metrics.scores import (
    evaluate_split,
    metrics_from_log_forecast,
    metrics_from_sigma_forecast,
)

FIG = FIGURES
SPLITS = SPLITS_DIR



import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator
from statsmodels.tsa.stattools import acf, adfuller, pacf

from paths import DAILY_RV, HOURLY_CLEAN, HOURLY_RETURNS

DAILY_PATH = DAILY_RV
HOURLY_RET = HOURLY_RETURNS

NAVY = "#1B3A4B"
TEAL = "#3D7A8C"
CORAL = "#C45C26"
SLATE = "#5C6B73"
GOLD = "#D4A017"
BG = "#FAFBFC"
GRID = "#E6EAEF"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.grid": True,
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


def moments(x: pd.Series) -> dict:
    x = x.dropna()
    return {
        "n": int(x.shape[0]),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "skew": float(x.skew()),
        "excess_kurtosis": float(x.kurtosis()),
        "min": float(x.min()),
        "p50": float(x.median()),
        "max": float(x.max()),
    }


def adf_summary(x: pd.Series) -> dict:
    stat, pvalue, usedlag, nobs, crit, _ = adfuller(
        x.dropna(), autolag="AIC", result_object=False
    )
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "crit_1pct": float(crit["1%"]),
        "crit_5pct": float(crit["5%"]),
    }


PLOT_START = pd.Timestamp("2025-01-01", tz="UTC")


def style_time_axis(ax, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Tight x-limits so the axis ends at the last observation (no empty next year)."""
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=0)


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def _corr_on_ax(ax, series: pd.Series, *, kind: str, lags: int, title: str, color: str) -> None:
    x = series.dropna().to_numpy()
    n = len(x)
    vals = acf(x, nlags=lags, fft=True) if kind == "acf" else pacf(x, nlags=lags, method="ywm")
    half = 1.96 / np.sqrt(n)
    lags_x = np.arange(0, lags + 1)
    ax.fill_between([-0.5, lags + 0.5], -half, half, color=TEAL, alpha=0.22, lw=0, zorder=0)
    ax.axhline(half, color=TEAL, lw=1.0, ls="--", alpha=0.9)
    ax.axhline(-half, color=TEAL, lw=1.0, ls="--", alpha=0.9)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.vlines(lags_x, 0, vals, color=color, lw=1.3)
    ax.plot(lags_x, vals, "o", color=color, ms=3.8)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.5, lags + 0.5)
    ax.set_title(title)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.text(
        0.98, 0.96, f"95% ≈ ±{half:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=SLATE,
    )


def plot_acf_pacf_pair(
    series: pd.Series,
    *,
    lags: int,
    title_acf: str,
    title_pacf: str,
    color: str,
    outfile: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    _corr_on_ax(axes[0], series, kind="acf", lags=lags, title=title_acf, color=color)
    _corr_on_ax(axes[1], series, kind="pacf", lags=lags, title=title_pacf, color=color)
    save(fig, outfile)


def run() -> None:
    ensure_result_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(DAILY_PATH).sort_values("date_utc")
    hourly_ret = pd.read_parquet(HOURLY_RET).sort_values("datetime_utc")
    hourly = pd.read_parquet(HOURLY_CLEAN).sort_values("datetime_utc")

    day = daily.loc[daily["rv_uses_24_returns"]].copy()
    day["date_utc"] = pd.to_datetime(day["date_utc"], utc=True)
    day["ann_rv"] = day["rv"] * np.sqrt(365.0)
    dates = set(day["date_utc"])

    h = hourly.loc[hourly["date_utc"].isin(dates)].copy()
    h["datetime_utc"] = pd.to_datetime(h["datetime_utc"], utc=True)
    h["date_utc"] = pd.to_datetime(h["date_utc"], utc=True)

    ret_h = hourly_ret.loc[
        hourly_ret["return_valid"] & hourly_ret["date_utc"].isin(dates)
    ].copy()
    ret_h["datetime_utc"] = pd.to_datetime(ret_h["datetime_utc"], utc=True)
    ret_h["date_utc"] = pd.to_datetime(ret_h["date_utc"], utc=True)

    last = h.sort_values("datetime_utc").groupby("date_utc", as_index=False).tail(1)
    panel = last[["date_utc", "close"]].merge(
        h.groupby("date_utc", as_index=False).agg(volume=("volume", "sum")),
        on="date_utc",
        how="inner",
    )
    panel = panel.merge(
        ret_h.groupby("date_utc", as_index=False).agg(log_return=("log_return", "sum")),
        on="date_utc",
        how="inner",
    )
    panel = panel.merge(
        day[["date_utc", "rv", "ann_rv", "log_rv"]], on="date_utc", how="inner"
    )
    panel = panel.sort_values("date_utc").reset_index(drop=True)

    vol_mu = panel["volume"].rolling(30, min_periods=30).mean()
    vol_sd = panel["volume"].rolling(30, min_periods=30).std()
    panel["volume_z30"] = (panel["volume"] - vol_mu) / vol_sd

    ret_h = ret_h.sort_values("datetime_utc")
    ret_h["roll_skew_30d"] = ret_h["log_return"].rolling(30 * 24, min_periods=30 * 24).skew()
    ret_h["roll_kurt_30d"] = ret_h["log_return"].rolling(30 * 24, min_periods=30 * 24).kurt()
    roll_daily = (
        ret_h.groupby("date_utc", as_index=False)
        .tail(1)[["date_utc", "roll_skew_30d", "roll_kurt_30d"]]
    )
    panel = panel.merge(roll_daily, on="date_utc", how="left")

    mid = panel["close"].rolling(20, min_periods=20).mean()
    sd = panel["close"].rolling(20, min_periods=20).std()
    panel["bb_mid"] = mid
    panel["bb_up"] = mid + 2 * sd
    panel["bb_low"] = mid - 2 * sd
    panel["rsi14"] = rsi_wilder(panel["close"], 14)

    view = panel.loc[panel["date_utc"] >= PLOT_START].copy()
    if view.empty:
        raise ValueError(f"no rows on/after {PLOT_START.date()}")
    t0 = view["date_utc"].iloc[0]
    t1 = view["date_utc"].iloc[-1]

    acf_ret = acf(panel["log_return"].dropna(), nlags=40, fft=True)
    acf_lrv = acf(panel["log_rv"].dropna(), nlags=40, fft=True)

    stats = {
        "sample": "UTC days with 24 valid hourly log returns",
        "n_days": int(len(panel)),
        "n_hourly_returns": int(len(ret_h)),
        "plot_window_start": t0.date().isoformat(),
        "plot_window_end": t1.date().isoformat(),
        "start": panel["date_utc"].iloc[0].date().isoformat(),
        "end": panel["date_utc"].iloc[-1].date().isoformat(),
        "hourly_log_return": moments(ret_h["log_return"]),
        "daily_log_return": moments(panel["log_return"]),
        "rv": moments(panel["rv"]),
        "ann_rv": moments(panel["ann_rv"]),
        "log_rv": moments(panel["log_rv"]),
        "acf_daily_return_lag1": float(acf_ret[1]),
        "acf_log_rv_lag1": float(acf_lrv[1]),
        "acf_log_rv_lag5": float(acf_lrv[5]),
        "acf_log_rv_lag22": float(acf_lrv[22]),
        "adf_daily_log_return": adf_summary(panel["log_return"]),
        "adf_log_rv": adf_summary(panel["log_rv"]),
    }
    (STATS / "eda_stats.json").write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True)
    axes[0].plot(view["date_utc"], view["close"], color=NAVY, lw=1.15)
    axes[0].set_ylabel("USDT")
    axes[0].set_title("Bitcoin daily close")
    axes[1].bar(view["date_utc"], view["volume"], color=TEAL, alpha=0.45, width=1.0, linewidth=0)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("BTC volume")
    axes[1].set_title("Daily volume")
    style_time_axis(axes[1], t0, t1)
    save(fig, "price_volume.png")

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    colors = np.where(view["log_return"].to_numpy() >= 0, "#2E7D32", "#C62828")
    ax.bar(view["date_utc"], view["log_return"], color=colors, width=1.0, linewidth=0, alpha=0.85)
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_ylabel("Daily log return")
    ax.set_title("Daily log returns")
    style_time_axis(ax, t0, t1)
    save(fig, "daily_log_returns.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    xs = ret_h["log_return"].to_numpy()
    lo, hi = np.quantile(xs, [0.002, 0.998])
    clipped = xs[(xs >= lo) & (xs <= hi)]
    ax.hist(
        clipped, bins=90, density=True, color=NAVY, alpha=0.78,
        edgecolor="white", linewidth=0.3,
    )
    grid = np.linspace(lo, hi, 400)
    mu, sigma = xs.mean(), xs.std()
    ax.plot(
        grid,
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((grid - mu) / sigma) ** 2),
        color=CORAL,
        lw=2.0,
        label="Matched normal",
    )
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Hourly log return")
    ax.set_ylabel("Density")
    ax.set_title("Hourly return distribution vs a matched normal")
    ax.legend(frameon=False)
    save(fig, "return_hist.png")

    plot_acf_pacf_pair(
        panel["log_return"],
        lags=40,
        title_acf="ACF of daily log returns",
        title_pacf="PACF of daily log returns",
        color=NAVY,
        outfile="acf_pacf_returns.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    axes[0].plot(view["date_utc"], view["ann_rv"] * 100.0, color=NAVY, lw=0.9)
    axes[0].set_ylabel("Ann. RV (%)")
    axes[0].set_title(r"Annualized realized volatility ($\mathrm{RV}\times\sqrt{365}$)")
    axes[1].plot(view["date_utc"], view["log_rv"], color=TEAL, lw=0.9)
    axes[1].set_ylabel("log RV")
    axes[1].set_title("Daily log realized volatility")
    style_time_axis(axes[1], t0, t1)
    save(fig, "rv_timeseries.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].hist(panel["rv"], bins=55, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].set_title("Daily RV")
    axes[0].set_xlabel("RV")
    axes[1].hist(panel["log_rv"], bins=55, color=TEAL, alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].set_title("Daily log RV")
    axes[1].set_xlabel("log RV")
    for a in axes:
        a.set_ylabel("Days")
    save(fig, "rv_hist.png")

    plot_acf_pacf_pair(
        panel["log_rv"],
        lags=40,
        title_acf="ACF of daily log RV",
        title_pacf="PACF of daily log RV",
        color=TEAL,
        outfile="acf_pacf_log_rv.png",
    )

    # pandas .kurt() is already excess kurtosis (Fisher: subtracts 3)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(view["date_utc"], view["roll_skew_30d"], color=NAVY, lw=1.15, label="Skewness")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.set_ylabel("Skewness", color=NAVY)
    ax.tick_params(axis="y", labelcolor=NAVY)
    ax2 = ax.twinx()
    ax2.plot(view["date_utc"], view["roll_kurt_30d"], color=CORAL, lw=1.15, label="Excess kurtosis")
    ax2.set_ylabel("Excess kurtosis", color=CORAL)
    ax2.tick_params(axis="y", labelcolor=CORAL)
    ax.set_title("30-day rolling skewness and excess kurtosis")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, ["Skewness", "Excess kurtosis"], frameon=False, loc="upper left")
    style_time_axis(ax, t0, t1)
    save(fig, "rolling_skew_kurt.png")

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(view["date_utc"], view["volume_z30"], color=TEAL, lw=0.95)
    ax.axhline(0, color=SLATE, lw=0.9)
    ax.set_ylabel("z-score")
    ax.set_title("30-day volume z-score")
    style_time_axis(ax, t0, t1)
    save(fig, "volume_z.png")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.fill_between(view["date_utc"], view["bb_low"], view["bb_up"], color=TEAL, alpha=0.28, lw=0)
    ax.plot(view["date_utc"], view["close"], color=NAVY, lw=1.15, label="Close")
    ax.plot(view["date_utc"], view["bb_up"], color=CORAL, lw=1.1, label="Upper band")
    ax.plot(view["date_utc"], view["bb_low"], color=TEAL, lw=1.1, label="Lower band")
    ax.set_ylabel("USDT")
    ax.set_title("Bollinger bands (20-day, ±2σ)")
    ax.legend(frameon=False, loc="upper left")
    style_time_axis(ax, t0, t1)
    save(fig, "bollinger.png")

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.plot(view["date_utc"], view["rsi14"], color=NAVY, lw=1.0)
    ax.axhline(70, color=CORAL, lw=0.9, ls="--")
    ax.axhline(30, color=TEAL, lw=0.9, ls="--")
    ax.fill_between(view["date_utc"], 70, 100, color=CORAL, alpha=0.08, lw=0)
    ax.fill_between(view["date_utc"], 0, 30, color=TEAL, alpha=0.08, lw=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI")
    ax.set_title("14-day RSI")
    style_time_axis(ax, t0, t1)
    save(fig, "rsi.png")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run()
