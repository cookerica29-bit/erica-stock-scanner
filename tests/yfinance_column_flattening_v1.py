#!/usr/bin/env python3
"""Regression tests for yfinance MultiIndex OHLCV flattening."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def sample_frame(columns) -> pd.DataFrame:
    return pd.DataFrame(
        [[1, 2, 3, 4, 100] for _ in range(12)],
        columns=columns,
    )


def main() -> int:
    ticker_first = sample_frame(pd.MultiIndex.from_product([["SCHW"], FIELDS]))
    flattened = scanner._flatten_columns(ticker_first.copy())
    assert list(flattened.columns) == FIELDS
    assert "Close" in flattened.columns

    field_first = sample_frame(pd.MultiIndex.from_product([FIELDS, ["SCHW"]]))
    flattened = scanner._flatten_columns(field_first.copy())
    assert list(flattened.columns) == FIELDS
    assert "Close" in flattened.columns

    original_download = scanner.yf.download
    try:
        scanner.yf.download = lambda *args, **kwargs: ticker_first.copy()
        result = scanner._download_price_batch_raw(["SCHW"], period="1y", interval="1d")
        assert "SCHW" in result
        assert list(result["SCHW"].columns) == FIELDS
        assert "Close" in result["SCHW"].columns
    finally:
        scanner.yf.download = original_download

    print("Yfinance column flattening v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
