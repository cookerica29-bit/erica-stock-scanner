#!/usr/bin/env python3
"""Regression tests for best-contract enrichment gating."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def result(stage: str, **overrides) -> dict:
    base = {
        "direction": "LONG",
        "entry": 100.0,
        "sl": 96.0,
        "tp1": 108.0,
        "trade_eval": {"trade_stage": stage},
    }
    base.update(overrides)
    return base


CASES = [
    ("C grade blocks A+ READY", result("A+ READY"), "C", "Tradeable", False),
    ("C grade blocks B+ TRADEABLE", result("B+ TRADEABLE"), "C", "Tradeable", False),
    ("A grade allows A+ READY", result("A+ READY"), "A", "Waiting", True),
    ("B grade allows B+ TRADEABLE", result("B+ TRADEABLE"), "B", "Waiting", True),
    ("A grade non-ready near-entry still enriches", result("BUILDING / WATCHLIST"), "A", "Near Entry", True),
    ("B grade non-ready tradeable still enriches", result("BUILDING / WATCHLIST"), "B", "Tradeable", True),
    ("A grade non-ready waiting does not enrich", result("BUILDING / WATCHLIST"), "A", "Waiting", False),
    ("B grade non-ready too-far does not enrich", result("BUILDING / WATCHLIST"), "B", "Too Far", False),
]


def main() -> int:
    for name, setup, grade, entry_status, expected in CASES:
        actual = scanner._should_enrich_best_contract(setup, grade, entry_status)
        if actual is not expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
    print("Best contract enrichment gate v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
