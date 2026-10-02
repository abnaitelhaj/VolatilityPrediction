"""Frozen evaluation protocol (Protocol B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from paths import FEATURES_PATH

TRAIN_END = pd.Timestamp("2021-12-31", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31", tz="UTC")
TARGET_COL = "y"
LOG_RV_COL = "log_rv"
REFIT_ON_TRAIN_VAL = True


def fit_mask(split: pd.Series, *, stage: str) -> pd.Series:
    """Boolean mask for fitting.

    stage='tune'  -> train only
    stage='final' -> train+val
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


def naive_log_rv_forecast(df: pd.DataFrame) -> pd.Series:
    """Naive: predict tomorrow's log-RV with today's log-RV."""
    return df[LOG_RV_COL].astype(float)
