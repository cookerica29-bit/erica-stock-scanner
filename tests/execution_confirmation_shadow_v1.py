import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router


def test_execution_shadow_passes_clean_bullish_reaction():
    candidate = {"ticker": "GOOD", "signal": "long", "ema21_4h": 100.0}
    preview = {"entry_price": 100.0, "atr14": 4.0}
    bars = [
        {"time": "2026-08-24T04:00:00Z", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0},
        {"time": "2026-08-24T08:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
        {"time": "2026-08-24T12:00:00Z", "open": 100.5, "high": 102.0, "low": 99.2, "close": 101.0},
        {"time": "2026-08-24T16:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.2},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_checked"] is True
    assert result["execution_shadow_ok"] is True
    assert result["execution_shadow_reason"] == "Latest 4H snapshot confirms reaction"


def test_execution_shadow_fails_when_latest_bar_slices_hold_zone():
    candidate = {"ticker": "BKR", "signal": "long", "ema21_4h": 63.2556}
    preview = {"entry_price": 61.58, "atr14": 1.5844}
    bars = [
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
    bars = [
        {"time": "2026-08-24T04:00:00Z", "open": 61.0, "high": 61.8, "low": 61.52, "close": 61.9},
        {"time": "2026-08-24T08:00:00Z", "open": 61.9, "high": 62.6, "low": 61.8, "close": 62.2},
        {"time": "2026-08-24T12:00:00Z", "open": 62.2, "high": 62.8, "low": 62.1, "close": 62.385},
        {"time": "2026-08-24T16:00:00Z", "open": 62.385, "high": 62.41, "low": 62.0, "close": 62.245},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "no bullish reaction" in result["execution_shadow_reason"]
