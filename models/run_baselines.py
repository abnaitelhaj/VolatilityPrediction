"""Classical baselines under eval_freeze Protocol B (no LSTM).

Naive, HAR, ARIMA, GARCH(1,1) → train(+AIC) → refit train+val → test metrics.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from matplotlib.dates import DateFormatter
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
sys.path.insert(0, str(ROOT / "eval_freeze"))

from protocol import (  # noqa: E402
    LOG_RV_COL,
    TARGET_COL,
    add_split_column,
    fit_mask,
    load_model_ready,
    metrics_from_log_forecast,
    metrics_from_sigma_forecast,
    naive_log_rv_forecast,
)

# Distinct, readable series colors (white plot area, no grid)
C_ACTUAL = "#111827"
C_NAIVE = "#6B7280"
C_HAR = "#2563EB"
C_ARIMA = "#FFEB3B"
C_GARCH = "#DC2626"
BAR_COLORS = [C_NAIVE, C_HAR, C_ARIMA, C_GARCH]

HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]

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


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(False)


def save(fig: plt.Figure, name: str) -> None:
    for ax in fig.axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def plot_test_forecasts(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    yhat_h: pd.Series,
    yhat_a: pd.Series,
    sig_g: pd.Series,
    title: str,
    outfile: str,
) -> None:
    tdf = df.loc[mask]
    if tdf.empty:
        raise ValueError(f"empty window for {outfile}")
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(tdf["date_utc"], tdf[TARGET_COL], color=C_ACTUAL, lw=1.35, label="Actual $y$")
    ax.plot(tdf["date_utc"], yhat_h.loc[tdf.index], color=C_HAR, lw=1.15, label="HAR")
    ax.plot(tdf["date_utc"], yhat_a.loc[tdf.index], color=C_ARIMA, lw=1.15, label="ARIMA")
    ax.plot(
        tdf["date_utc"],
        np.log(sig_g.loc[tdf.index]),
        color=C_GARCH,
        lw=1.15,
        label=r"GARCH $\log\hat\sigma$",
    )
    ax.set_ylabel(r"$\log\mathrm{RV}$")
    ax.set_title(title)
    ax.set_xlim(tdf["date_utc"].iloc[0], tdf["date_utc"].iloc[-1])
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    save(fig, outfile)


def score_log(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return metrics_from_log_forecast(y, yhat)


def run_naive(df: pd.DataFrame) -> pd.Series:
    return naive_log_rv_forecast(df)


def run_har(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    final = fit_mask(df["split"], stage="final")
    X = np.column_stack([np.ones(final.sum()), df.loc[final, HAR_COLS].to_numpy()])
    y = df.loc[final, TARGET_COL].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    X_all = np.column_stack([np.ones(len(df)), df[HAR_COLS].to_numpy()])
    yhat = pd.Series(X_all @ beta, index=df.index, name="yhat_har")
    info = {
        "intercept": float(beta[0]),
        "beta_d": float(beta[1]),
        "beta_w": float(beta[2]),
        "beta_m": float(beta[3]),
        "fit_rows": int(final.sum()),
    }
    return yhat, info


def select_arima_order(series: pd.Series, *, max_p: int = 3, max_q: int = 3) -> tuple[int, int, int]:
    best = None
    best_aic = np.inf
    for d in (0, 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == q == 0 and d == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = ARIMA(series, order=(p, d, q)).fit()
                    if res.aic < best_aic:
                        best_aic = float(res.aic)
                        best = (p, d, q)
                except Exception:
                    continue
    if best is None:
        raise RuntimeError("ARIMA order search failed")
    return best


def run_arima(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    tune = fit_mask(df["split"], stage="tune")
    final = fit_mask(df["split"], stage="final")
    test = df["split"].eq("test")

    order = select_arima_order(df.loc[tune, LOG_RV_COL])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(df.loc[final, LOG_RV_COL], order=order).fit()

    yhat = pd.Series(np.nan, index=df.index, name="yhat_arima")
    # In-sample one-step predictions on final fit window (optional diagnostics)
    yhat.loc[final] = res.predict()

    # Walk-forward 1-step on test with frozen params (append actuals)
    hist_res = res
    test_idx = df.index[test].tolist()
    for i, idx in enumerate(test_idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = hist_res.forecast(1)
        yhat.loc[idx] = float(np.asarray(f)[0])
        # append today's log_rv so next forecast conditions on it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist_res = hist_res.append([df.loc[idx, LOG_RV_COL]], refit=False)

    info = {
        "order": list(order),
        "aic_train_select": float(best_aic_safe(df.loc[tune, LOG_RV_COL], order)),
        "aic_final": float(res.aic),
        "fit_rows": int(final.sum()),
    }
    return yhat, info


def best_aic_safe(series: pd.Series, order: tuple[int, int, int]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(ARIMA(series, order=order).fit().aic)


def run_garch(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Fit GARCH(1,1) on train+val returns; walk-forward sigma_{t+1} on full sample path."""
    tune = fit_mask(df["split"], stage="tune")
    final = fit_mask(df["split"], stage="final")

    # arch prefers non-tiny scale; use percent returns, convert sigma back
    ret_pct = df["ret_d"] * 100.0

    # Confirm (1,1) is reasonable vs a tiny grid on train AIC
    candidates = [(1, 1), (1, 2), (2, 1)]
    best_order = (1, 1)
    best_aic = np.inf
    for p, q in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(ret_pct.loc[tune], mean="Constant", vol="GARCH", p=p, q=q, dist="normal")
                res_t = am.fit(disp="off")
            if res_t.aic < best_aic:
                best_aic = float(res_t.aic)
                best_order = (p, q)
        except Exception:
            continue

    p, q = best_order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(ret_pct.loc[final], mean="Constant", vol="GARCH", p=p, q=q, dist="normal")
        res = am.fit(disp="off")

    # Fixed-param recursion over the full return path for 1-step sigma forecasts
    # sigmahat_t in output aligns with row t → forecast for day t+1 (matches y_t)
    mu = float(res.params.get("mu", 0.0))
    omega = float(res.params["omega"])
    # arch names: alpha[1], beta[1], ...
    alphas = [float(res.params[f"alpha[{i}]"]) for i in range(1, p + 1)]
    betas = [float(res.params[f"beta[{i}]"]) for i in range(1, q + 1)]

    r = ret_pct.to_numpy(dtype=float)
    n = len(r)
    # unconditional var start (percent^2)
    den = 1.0 - sum(alphas) - sum(betas)
    unc = omega / den if den > 1e-8 else omega / 0.05
    sig2 = np.empty(n)
    sig2[0] = unc
    for t in range(1, n):
        s = omega
        for i, a in enumerate(alphas):
            if t - 1 - i >= 0:
                eps = r[t - 1 - i] - mu
                s += a * eps * eps
        for j, b in enumerate(betas):
            if t - 1 - j >= 0:
                s += b * sig2[t - 1 - j]
        sig2[t] = max(s, 1e-12)

    # At time t we know r_t after day t closes; one-step forecast sigma_{t+1}
    # Use sig2[t] as var_t, then:
    # var_{t+1} = omega + alpha*(r_t-mu)^2 + beta*var_t  (for 1,1; general below)
    sig2_f = np.empty(n)
    for t in range(n):
        s = omega
        for i, a in enumerate(alphas):
            if t - i >= 0:
                eps = r[t - i] - mu
                s += a * eps * eps
        for j, b in enumerate(betas):
            if t - j >= 0:
                s += b * sig2[t - j]
        sig2_f[t] = max(s, 1e-12)

    # convert percent vol → decimal daily sigma
    sigma_hat = pd.Series(np.sqrt(sig2_f) / 100.0, index=df.index, name="sigma_garch")

    info = {
        "order": [p, q],
        "aic_train_select": float(best_aic),
        "aic_final": float(res.aic),
        "params": {k: float(v) for k, v in res.params.items()},
        "fit_rows": int(final.sum()),
    }
    return sigma_hat, info


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = add_split_column(load_model_ready())
    test = df["split"].eq("test")
    y_test = df.loc[test, TARGET_COL].to_numpy()
    rv_test = np.exp(y_test)

    forecasts: dict[str, pd.Series] = {}
    meta: dict = {"protocol": "B", "models": {}}

    # Naive
    yhat_n = run_naive(df)
    forecasts["naive"] = yhat_n
    meta["models"]["naive"] = {"type": "lag1_log_rv"}
    m_naive = score_log(y_test, yhat_n.loc[test].to_numpy())

    # HAR
    yhat_h, har_info = run_har(df)
    forecasts["har"] = yhat_h
    meta["models"]["har"] = har_info
    m_har = score_log(y_test, yhat_h.loc[test].to_numpy())

    # ARIMA
    yhat_a, arima_info = run_arima(df)
    forecasts["arima"] = yhat_a
    meta["models"]["arima"] = arima_info
    m_arima = score_log(y_test, yhat_a.loc[test].to_numpy())

    # GARCH
    sig_g, garch_info = run_garch(df)
    forecasts["garch_sigma"] = sig_g
    meta["models"]["garch"] = garch_info
    m_garch = metrics_from_sigma_forecast(rv_test, sig_g.loc[test].to_numpy(), y_true=y_test)

    leaderboard = {
        "naive": m_naive,
        "har": m_har,
        "arima": m_arima,
        "garch": m_garch,
    }
    meta["test_metrics"] = leaderboard
    meta["test_n"] = int(test.sum())
    meta["test_start"] = df.loc[test, "date_utc"].iloc[0].date().isoformat()
    meta["test_end"] = df.loc[test, "date_utc"].iloc[-1].date().isoformat()

    # Save forecasts
    out_df = df[["date_utc", "split", TARGET_COL, LOG_RV_COL, "rv", "ret_d"]].copy()
    out_df["yhat_naive"] = forecasts["naive"]
    out_df["yhat_har"] = forecasts["har"]
    out_df["yhat_arima"] = forecasts["arima"]
    out_df["sigma_garch"] = forecasts["garch_sigma"]
    out_df["yhat_garch_log"] = np.log(forecasts["garch_sigma"].clip(lower=1e-12))
    out_df.to_parquet(OUT / "baseline_forecasts.parquet", index=False)
    (OUT / "baseline_stats.json").write_text(json.dumps(meta, indent=2))

    # --- figures ---
    # leaderboard bars
    models = ["naive", "har", "arima", "garch"]
    labels = ["Naive", "HAR", "ARIMA", "GARCH"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8))
    for ax, key, title in zip(
        axes,
        ["rmse_log", "mae_log", "mape_rv", "qlike"],
        ["RMSE (log)", "MAE (log)", "MAPE (RV, %)", "QLIKE"],
    ):
        vals = [leaderboard[m][key] for m in models]
        ax.bar(labels, vals, color=BAR_COLORS, width=0.68, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Test-set baseline metrics", y=1.02, fontsize=13, fontweight="bold")
    save(fig, "baseline_metrics.png")

    plot_2026 = test & (df["date_utc"] >= pd.Timestamp("2026-01-01", tz="UTC"))
    plot_test_forecasts(
        df,
        mask=plot_2026,
        yhat_h=yhat_h,
        yhat_a=yhat_a,
        sig_g=sig_g,
        title="Test forecasts vs actual (2026)",
        outfile="baseline_test_paths.png",
    )

    test_start = df.loc[test, "date_utc"].iloc[0]
    plot_2mo = test & (df["date_utc"] >= test_start) & (
        df["date_utc"] < test_start + pd.DateOffset(months=2)
    )
    plot_test_forecasts(
        df,
        mask=plot_2mo,
        yhat_h=yhat_h,
        yhat_a=yhat_a,
        sig_g=sig_g,
        title="Test forecasts vs actual (first two months)",
        outfile="baseline_test_paths_2mo.png",
    )

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
