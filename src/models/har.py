"""HAR-RV on log realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from splits.protocol import TARGET_COL, fit_mask

HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]


def forecast(df: pd.DataFrame) -> tuple[pd.Series, dict]:
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
