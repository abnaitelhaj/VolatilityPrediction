"""GARCH(1,1) on daily returns → sigma forecast."""

from models.baselines import run_garch as forecast

__all__ = ["forecast"]
