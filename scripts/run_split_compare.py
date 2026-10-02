"""Thin CLI — logic lives under ``src/``."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import sync_report_figures
from metrics.split_compare import run

if __name__ == "__main__":
    run()
    sync_report_figures()
