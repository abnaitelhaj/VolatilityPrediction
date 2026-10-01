"""Build daily feature panel + figures for feature_eng/fe_report.pdf."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
DAILY_RV = ROOT / "data" / "processed" / "btcusdt_1d_rv.parquet"
HOURLY_RET = ROOT / "data" / "processed" / "btcusdt_1h_returns.parquet"
HOURLY_CLEAN = ROOT / "data" / "processed" / "btcusdt_1h_clean.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "btcusdt_1d_features.parquet"

NAVY = "#1B3A4B"
TEAL = "#3D7A8C"
CORAL = "#C45C26"
SLATE = "#5C6B73"
BG = "#FAFBFC"
GRID = "#E6EAEF"

PLOT_START = pd.Timestamp("2025-01-01", tz="UTC")
HAR_W = 7
HAR_M = 22
VOL_Z_WIN = 30

FEATURE_COLS = [
    "log_rv_d",
    "log_rv_w",
    "log_rv_m",
    "ret_d",
    "ret_d_sq",
    "volume_z30",
]

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


def style_time_axis(ax, start: pd.Timestamp, end: pd.Timestamp) -> None:
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def build_panel() -> pd.DataFrame:
    daily = pd.read_parquet(DAILY_RV).sort_values("date_utc")
    hourly_ret = pd.read_parquet(HOURLY_RET).sort_values("datetime_utc")
    hourly = pd.read_parquet(HOURLY_CLEAN).sort_values("datetime_utc")

    day = daily.loc[daily["rv_uses_24_returns"]].copy()
    day["date_utc"] = pd.to_datetime(day["date_utc"], utc=True)
    dates = set(day["date_utc"])

    ret_h = hourly_ret.loc[
        hourly_ret["return_valid"] & hourly_ret["date_utc"].isin(dates)
    ].copy()
    ret_h["date_utc"] = pd.to_datetime(ret_h["date_utc"], utc=True)

    h = hourly.loc[hourly["date_utc"].isin(dates)].copy()
    h["date_utc"] = pd.to_datetime(h["date_utc"], utc=True)

    panel = day[["date_utc", "rv", "log_rv"]].copy()
    panel = panel.merge(
        ret_h.groupby("date_utc", as_index=False).agg(ret_d=("log_return", "sum")),
        on="date_utc",
        how="inner",
    )
    panel = panel.merge(
        h.groupby("date_utc", as_index=False).agg(volume=("volume", "sum")),
        on="date_utc",
        how="inner",
    )
    panel = panel.sort_values("date_utc").reset_index(drop=True)

    # Target: next day's log RV (known only after day t+1 closes)
    panel["y"] = panel["log_rv"].shift(-1)

    # HAR-style features on log RV (crypto calendar: 1 / 7 / 22)
    panel["log_rv_d"] = panel["log_rv"]
    panel["log_rv_w"] = panel["log_rv"].rolling(HAR_W, min_periods=HAR_W).mean()
    panel["log_rv_m"] = panel["log_rv"].rolling(HAR_M, min_periods=HAR_M).mean()

    panel["ret_d_sq"] = panel["ret_d"] ** 2

    vol_mu = panel["volume"].rolling(VOL_Z_WIN, min_periods=VOL_Z_WIN).mean()
    vol_sd = panel["volume"].rolling(VOL_Z_WIN, min_periods=VOL_Z_WIN).std()
    panel["volume_z30"] = (panel["volume"] - vol_mu) / vol_sd

    panel["model_ready"] = panel[["y", *FEATURE_COLS]].notna().all(axis=1)
    return panel


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    ready = panel.loc[panel["model_ready"]].copy()

    panel.to_parquet(FEATURES_PATH, index=False)

    view = ready.loc[ready["date_utc"] >= PLOT_START].copy()
    t0, t1 = view["date_utc"].iloc[0], view["date_utc"].iloc[-1]

    corr = {c: float(ready[c].corr(ready["y"])) for c in FEATURE_COLS}
    corr_feat = ready[FEATURE_COLS].corr()

    stats = {
        "n_complete_days": int(len(panel)),
        "n_model_ready": int(len(ready)),
        "n_dropped_warmup_or_tail": int((~panel["model_ready"]).sum()),
        "har_w": HAR_W,
        "har_m": HAR_M,
        "volume_z_window": VOL_Z_WIN,
        "feature_cols": FEATURE_COLS,
        "target": "y = log_rv_{t+1}",
        "start_all": panel["date_utc"].iloc[0].date().isoformat(),
        "end_all": panel["date_utc"].iloc[-1].date().isoformat(),
        "start_ready": ready["date_utc"].iloc[0].date().isoformat(),
        "end_ready": ready["date_utc"].iloc[-1].date().isoformat(),
        "corr_with_y": corr,
        "y_moments": {
            "mean": float(ready["y"].mean()),
            "std": float(ready["y"].std()),
            "skew": float(ready["y"].skew()),
            "excess_kurtosis": float(ready["y"].kurtosis()),
        },
    }
    (OUT / "fe_stats.json").write_text(json.dumps(stats, indent=2))

    # --- figures ---
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(view["date_utc"], view["y"], color=TEAL, lw=0.95)
    ax.set_ylabel(r"$\log\mathrm{RV}_{t+1}$")
    ax.set_title("Target: next-day log realized volatility")
    style_time_axis(ax, t0, t1)
    save(fig, "target_y.png")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    axes[0].plot(view["date_utc"], view["log_rv_d"], color=NAVY, lw=0.9)
    axes[0].set_ylabel(r"$\log\mathrm{RV}^{(d)}$")
    axes[0].set_title("HAR-style log-RV components")
    axes[1].plot(view["date_utc"], view["log_rv_w"], color=TEAL, lw=0.9)
    axes[1].set_ylabel(r"$\log\mathrm{RV}^{(w)}$")
    axes[2].plot(view["date_utc"], view["log_rv_m"], color=CORAL, lw=0.9)
    axes[2].set_ylabel(r"$\log\mathrm{RV}^{(m)}$")
    style_time_axis(axes[2], t0, t1)
    save(fig, "har_components.png")

    labels = [
        r"$\log\mathrm{RV}^{(d)}$",
        r"$\log\mathrm{RV}^{(w)}$",
        r"$\log\mathrm{RV}^{(m)}$",
        r"$R_t$",
        r"$R_t^2$",
        r"$Z^{(30)}_t$",
    ]
    vals = [corr[c] for c in FEATURE_COLS]
    colors = [TEAL if v >= 0 else CORAL for v in vals]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.set_facecolor("white")
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.65)
    ax.axvline(0, color=SLATE, lw=0.9)
    ax.set_xlabel(r"Correlation with $y_t=\log\mathrm{RV}_{t+1}$")
    ax.set_title("Feature–target correlation (model-ready sample)")
    save(fig, "corr_with_y.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.set_facecolor("white")
    im = ax.imshow(corr_feat.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            ax.text(j, i, f"{corr_feat.to_numpy()[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Feature correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "feature_corr.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(ready["log_rv_d"], ready["y"], s=10, alpha=0.28, color=NAVY, linewidths=0)
    lims = [
        float(np.nanmin([ready["log_rv_d"].min(), ready["y"].min()])),
        float(np.nanmax([ready["log_rv_d"].max(), ready["y"].max()])),
    ]
    ax.plot(lims, lims, color=CORAL, lw=1.0, alpha=0.85)
    ax.set_xlabel(r"$\log\mathrm{RV}_t$")
    ax.set_ylabel(r"$\log\mathrm{RV}_{t+1}$")
    ax.set_title(rf"Lag-1 persistence (corr = {corr['log_rv_d']:.2f})")
    save(fig, "lag1_scatter.png")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)
    axes[0].plot(view["date_utc"], view["ret_d"], color=NAVY, lw=0.8)
    axes[0].axhline(0, color=SLATE, lw=0.8)
    axes[0].set_ylabel(r"$R_t$")
    axes[0].set_title("Daily return and squared return")
    axes[1].plot(view["date_utc"], view["ret_d_sq"], color=CORAL, lw=0.8)
    axes[1].set_ylabel(r"$R_t^2$")
    axes[1].set_ylim(bottom=0)
    style_time_axis(axes[1], t0, t1)
    save(fig, "return_features.png")

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.plot(view["date_utc"], view["volume_z30"], color=TEAL, lw=0.95)
    ax.axhline(0, color=SLATE, lw=0.9)
    ax.set_ylabel("z-score")
    ax.set_title("30-day volume z-score feature")
    style_time_axis(ax, t0, t1)
    save(fig, "volume_z_feature.png")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
