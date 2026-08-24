import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router


def _leading_bars(count, *, base=100.0, volume=1000):
    return [
        {
            "time": f"2026-08-23T{hour:02d}:00:00Z",
            "open": base + (idx * 0.05),
            "high": base + 1.0 + (idx * 0.05),
            "low": base - 1.0 + (idx * 0.05),
            "close": base + 0.25 + (idx * 0.05),
            "volume": volume,
        }
        for idx, hour in enumerate(range(count))
    ]


def test_execution_shadow_passes_clean_bullish_reaction():
    candidate = {"ticker": "GOOD", "signal": "long", "ema21_4h": 100.0}
    preview = {"entry_price": 100.0, "atr14": 4.0}
    bars = _leading_bars(11, base=99.0, volume=1000) + [
        {"time": "2026-08-24T04:00:00Z", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000},
        {"time": "2026-08-24T08:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1100},
        {"time": "2026-08-24T12:00:00Z", "open": 100.5, "high": 102.0, "low": 99.2, "close": 101.0, "volume": 1200},
        {"time": "2026-08-24T16:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.2, "volume": 1000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_checked"] is True
    assert result["execution_shadow_ok"] is True
    assert result["execution_shadow_reason"] == "Latest 4H snapshot confirms reaction"


def test_execution_shadow_fails_when_latest_bar_slices_hold_zone():
    candidate = {"ticker": "BKR", "signal": "long", "ema21_4h": 63.2556}
    preview = {"entry_price": 61.58, "atr14": 1.5844}
    bars = _leading_bars(11, base=61.0, volume=200000) + [
        {"time": "2026-08-24T04:00:00Z", "open": 61.0, "high": 61.9, "low": 61.13, "close": 61.48},
        {"time": "2026-08-24T08:00:00Z", "open": 61.48, "high": 62.2, "low": 61.5, "close": 61.9},
        {"time": "2026-08-24T12:00:00Z", "open": 61.9, "high": 62.4, "low": 61.6, "close": 62.0},
        {"time": "2026-08-24T16:00:00Z", "open": 61.485, "high": 62.02, "low": 61.29, "close": 61.87},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "sliced zone" in result["execution_shadow_reason"]


def test_execution_shadow_fails_without_bullish_reaction():
    candidate = {"ticker": "BAC", "signal": "long", "ema21_4h": 62.7178}
    preview = {"entry_price": 62.145, "atr14": 1.1181}
    bars = _leading_bars(11, base=61.0, volume=900000) + [
        {"time": "2026-08-24T04:00:00Z", "open": 61.0, "high": 61.8, "low": 61.52, "close": 61.9},
        {"time": "2026-08-24T08:00:00Z", "open": 61.9, "high": 62.6, "low": 61.8, "close": 62.2},
        {"time": "2026-08-24T12:00:00Z", "open": 62.2, "high": 62.8, "low": 62.1, "close": 62.385},
        {"time": "2026-08-24T16:00:00Z", "open": 62.385, "high": 62.41, "low": 62.0, "close": 62.245},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "no bullish reaction" in result["execution_shadow_reason"]


def test_execution_shadow_fails_low_conviction_positive_reaction():
    candidate = {"ticker": "SLBISH", "signal": "long", "ema21_4h": 53.0}
    preview = {"entry_price": 53.39, "atr14": 2.0}
    bars = _leading_bars(10, base=52.8, volume=300000) + [
        {"time": "2026-08-24T00:00:00Z", "open": 53.0, "high": 53.5, "low": 52.7, "close": 53.2, "volume": 300000},
        {"time": "2026-08-24T04:00:00Z", "open": 53.2, "high": 53.6, "low": 52.8, "close": 53.1, "volume": 280000},
        {"time": "2026-08-24T08:00:00Z", "open": 53.1, "high": 53.7, "low": 52.9, "close": 53.3, "volume": 260000},
        {"time": "2026-08-24T12:00:00Z", "open": 53.3, "high": 53.8, "low": 52.9, "close": 53.4, "volume": 240000},
        {"time": "2026-08-24T16:00:00Z", "open": 53.35, "high": 53.55, "low": 53.0, "close": 53.48, "volume": 76000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "reaction only" in result["execution_shadow_reason"]
    assert "thin live volume" in result["execution_shadow_reason"]


def test_execution_shadow_fails_flat_range_positive_reaction():
    candidate = {"ticker": "AGNCISH", "signal": "long", "ema21_4h": 10.9}
    preview = {"entry_price": 10.895, "atr14": 0.24}
    bars = [
        {
            "time": f"2026-08-23T{hour:02d}:00:00Z",
            "open": 10.90,
            "high": 10.96,
            "low": 10.88,
            "close": 10.91,
            "volume": 100000,
        }
        for hour in range(10)
    ] + [
        {"time": "2026-08-24T00:00:00Z", "open": 10.90, "high": 10.96, "low": 10.88, "close": 10.92, "volume": 100000},
        {"time": "2026-08-24T04:00:00Z", "open": 10.92, "high": 10.97, "low": 10.89, "close": 10.93, "volume": 110000},
        {"time": "2026-08-24T08:00:00Z", "open": 10.93, "high": 10.96, "low": 10.90, "close": 10.92, "volume": 105000},
        {"time": "2026-08-24T12:00:00Z", "open": 10.92, "high": 10.95, "low": 10.88, "close": 10.91, "volume": 98000},
        {"time": "2026-08-24T16:00:00Z", "open": 10.91, "high": 10.94, "low": 10.89, "close": 10.93, "volume": 99000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "reaction only" in result["execution_shadow_reason"]
    assert "recent range only" in result["execution_shadow_reason"]


def test_execution_shadow_requires_full_range_window():
    candidate = {"ticker": "SHORTDATA", "signal": "long", "ema21_4h": 100.0}
    preview = {"entry_price": 100.0, "atr14": 4.0}
    bars = _leading_bars(14, base=100.0, volume=1000)

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "Need 15 recent 4H bars" in result["execution_shadow_reason"]
