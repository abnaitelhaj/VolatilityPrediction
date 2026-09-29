"""Frozen evaluation protocol shared by all forecasting models.

Protocol B (chronological train / val / test + final refit):
  - TRAIN: fit while tuning (LSTM early stopping; optional classical AIC)
  - VAL:   LSTM hyperparameter / early-stopping only (non-DL skips for tuning)
  - FINAL FIT: refit all models on train+val with frozen hyperparameters
  - TEST:  final RMSE, MAE, QLIKE for every model (never used for fitting/tuning)

Chronological splits only. No shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "btcusdt_1d_features.parquet"

# Locked calendar cuts (UTC midnights). Inclusive start of each regime below.
TRAIN_END = pd.Timestamp("2021-12-31", tz="UTC")  # last train day
VAL_END = pd.Timestamp("2023-12-31", tz="UTC")  # last val day
# test: everything after VAL_END

TARGET_COL = "y"
LOG_RV_COL = "log_rv"  # log RV on day t (same as log_rv_d)

# Final leaderboard fits use all pre-test history
REFIT_ON_TRAIN_VAL = True


def fit_mask(split: pd.Series, *, stage: str) -> pd.Series:
    """Boolean mask for fitting.

    stage='tune'  -> train only (used while selecting LSTM hyperparams / AIC)
    stage='final' -> train+val (Protocol B refit before test scoring)
    """
    if stage == "tune":
        return split.eq("train")
    if stage == "final":
        return split.isin(["train", "val"])
    raise ValueError("stage must be 'tune' or 'final'")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n: int


def load_model_ready(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.loc[df["model_ready"]].copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
    df = df.sort_values("date_utc").reset_index(drop=True)
    if df[TARGET_COL].isna().any():
        raise ValueError("model_ready rows must have non-null y")
    return df


def assign_split(dates: pd.Series) -> pd.Series:
    """Return 'train' | 'val' | 'test' for each date."""
    out = pd.Series(index=dates.index, dtype="object")
    out.loc[dates <= TRAIN_END] = "train"
    out.loc[(dates > TRAIN_END) & (dates <= VAL_END)] = "val"
    out.loc[dates > VAL_END] = "test"
    return out


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = assign_split(out["date_utc"])
    return out


def split_summaries(df: pd.DataFrame) -> list[SplitSpec]:
    df = add_split_column(df)
    specs: list[SplitSpec] = []
    for name in ("train", "val", "test"):
        part = df.loc[df["split"] == name]
        specs.append(
            SplitSpec(
                name=name,
                start=part["date_utc"].iloc[0],
                end=part["date_utc"].iloc[-1],
                n=int(len(part)),
            )
        )
    return specs


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean(e**2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(e)))


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
    """Score a forecast of log-RV.

    If ``rv`` (true next-day RV) is omitted, it is recovered as exp(y).
    Forecast RV is exp(yhat).
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    rv_true = np.exp(y) if rv is None else np.asarray(rv, dtype=float)
    rv_hat = np.exp(yhat)
    return {
        "rmse_log": rmse(y, yhat),
        "mae_log": mae(y, yhat),
        "qlike": qlike(rv_true, rv_hat),
        "n": float(len(y)),
    }


def metrics_from_sigma_forecast(
    rv_true: np.ndarray,
    sigma_hat: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a GARCH-style sigma forecast against realized RV.

    RMSE/MAE are computed on log(sigma_hat) vs log(RV) (or vs y_true if given).
    """
    rv_true = np.asarray(rv_true, dtype=float)
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    y = np.log(rv_true) if y_true is None else np.asarray(y_true, dtype=float)
    yhat = np.log(sigma_hat)
    return {
        "rmse_log": rmse(y, yhat),
        "mae_log": mae(y, yhat),
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
    # next-day RV for QLIKE: exp(y)
    yh = np.asarray(yhat)[mask] if not isinstance(yhat, pd.Series) else yhat.loc[mask].to_numpy()
    out = metrics_from_log_forecast(y, yh)
    out["split"] = split  # type: ignore[assignment]
    return out


def naive_log_rv_forecast(df: pd.DataFrame) -> pd.Series:
    """Naive: predict tomorrow's log-RV with today's log-RV."""
    return df[LOG_RV_COL].astype(float)
