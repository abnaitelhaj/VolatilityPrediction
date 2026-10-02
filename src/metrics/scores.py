"""Forecast scoring: RMSE, MAE, MAPE, QLIKE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from splits.protocol import TARGET_COL, add_split_column


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean(e**2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(e)))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error, in percent, on a positive scale."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if np.any(actual <= 0):
        raise ValueError("MAPE requires a strictly positive actual series")
    return float(np.mean(np.abs(forecast - actual) / actual) * 100.0)


def qlike(rv: np.ndarray, rv_hat: np.ndarray) -> float:
    """Average QLIKE on the RV scale (not log).

    QLIKE = RV/RVhat - log(RV/RVhat) - 1.
    """
    rv = np.asarray(rv, dtype=float)
    rv_hat = np.asarray(rv_hat, dtype=float)
    if np.any(rv <= 0) or np.any(rv_hat <= 0):
        raise ValueError("QLIKE requires strictly positive RV and RV_hat")
    ratio = rv / rv_hat
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def metrics_from_log_forecast(
    y: np.ndarray,
    yhat: np.ndarray,
    *,
    rv: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a forecast of log-RV."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    rv_true = np.exp(y) if rv is None else np.asarray(rv, dtype=float)
    rv_hat = np.exp(yhat)
    return {
        "rmse_log": rmse(y, yhat),
        "mae_log": mae(y, yhat),
        "mape_rv": mape(rv_true, rv_hat),
        "qlike": qlike(rv_true, rv_hat),
        "n": float(len(y)),
    }


def metrics_from_sigma_forecast(
    rv_true: np.ndarray,
    sigma_hat: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a GARCH-style sigma forecast against realized RV."""
    rv_true = np.asarray(rv_true, dtype=float)
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    y = np.log(rv_true) if y_true is None else np.asarray(y_true, dtype=float)
    yhat = np.log(sigma_hat)
    return {
        "rmse_log": rmse(y, yhat),
        "mae_log": mae(y, yhat),
        "mape_rv": mape(rv_true, sigma_hat),
        "qlike": qlike(rv_true, sigma_hat),
        "n": float(len(rv_true)),
    }


def evaluate_split(
    df: pd.DataFrame,
    yhat: pd.Series | np.ndarray,
    split: str,
) -> dict[str, float]:
    """Evaluate log-RV forecasts on one frozen split."""
    d = add_split_column(df)
    mask = d["split"].to_numpy() == split
    y = d.loc[mask, TARGET_COL].to_numpy()
    yh = np.asarray(yhat)[mask] if not isinstance(yhat, pd.Series) else yhat.loc[mask].to_numpy()
    out = metrics_from_log_forecast(y, yh)
    out["split"] = split  # type: ignore[assignment]
    return out
