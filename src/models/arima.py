"""ARIMA on log-RV (order selected by AIC on train)."""

from models.baselines import run_arima as forecast

__all__ = ["forecast"]
