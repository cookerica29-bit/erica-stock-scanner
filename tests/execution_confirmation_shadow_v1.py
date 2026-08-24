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
        {"time": "2026-08-24T16:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.6, "volume": 1000},
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
        {"time": "2026-08-24T00:00:00Z", "open": 53.0, "high": 53.5, "low": 52.7, "close": 53.2, "volume": 76000},
        {"time": "2026-08-24T04:00:00Z", "open": 53.2, "high": 53.6, "low": 52.8, "close": 53.1, "volume": 76000},
        {"time": "2026-08-24T08:00:00Z", "open": 53.1, "high": 53.7, "low": 52.9, "close": 53.3, "volume": 76000},
        {"time": "2026-08-24T12:00:00Z", "open": 53.3, "high": 53.8, "low": 52.9, "close": 53.4, "volume": 76000},
        {"time": "2026-08-24T16:00:00Z", "open": 53.35, "high": 53.55, "low": 53.0, "close": 53.48, "volume": 76000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "reaction only" in result["execution_shadow_reason"]
    assert "directional expansion only" in result["execution_shadow_reason"]
    assert "thin bullish confirmation volume" in result["execution_shadow_reason"]


def test_execution_shadow_passes_quiet_consolidation_after_volume_push():
    candidate = {"ticker": "DASHLIKE", "signal": "long", "ema21_4h": 100.0}
    preview = {"entry_price": 100.0, "atr14": 2.0}
    bars = _leading_bars(10, base=97.0, volume=140000) + [
        {"time": "2026-08-24T00:00:00Z", "open": 98.0, "high": 100.0, "low": 97.8, "close": 99.7, "volume": 160000},
        {"time": "2026-08-24T04:00:00Z", "open": 99.7, "high": 102.0, "low": 99.4, "close": 101.4, "volume": 220000},
        {"time": "2026-08-24T08:00:00Z", "open": 101.4, "high": 102.6, "low": 100.9, "close": 102.1, "volume": 180000},
        {"time": "2026-08-24T12:00:00Z", "open": 102.1, "high": 102.8, "low": 101.6, "close": 102.4, "volume": 60000},
        {"time": "2026-08-24T16:00:00Z", "open": 102.2, "high": 102.7, "low": 101.8, "close": 102.5, "volume": 24000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is True
    assert result["execution_shadow_reason"] == "Latest 4H snapshot confirms reaction"


def test_execution_shadow_ignores_high_volume_red_candle_for_longs():
    candidate = {"ticker": "SLBREGRESS", "signal": "long", "ema21_4h": 52.0}
    preview = {"entry_price": 53.0, "atr14": 1.0}
    bars = _leading_bars(10, base=52.0, volume=250000) + [
        {"time": "2026-08-24T00:00:00Z", "open": 53.85, "high": 54.0, "low": 52.52, "close": 53.49, "volume": 481794},
        {"time": "2026-08-24T04:00:00Z", "open": 53.49, "high": 53.8, "low": 53.1, "close": 53.7, "volume": 76000},
        {"time": "2026-08-24T08:00:00Z", "open": 53.7, "high": 53.75, "low": 53.3, "close": 53.55, "volume": 80000},
        {"time": "2026-08-24T12:00:00Z", "open": 53.55, "high": 53.9, "low": 53.4, "close": 53.75, "volume": 90000},
        {"time": "2026-08-24T16:00:00Z", "open": 53.7, "high": 54.0, "low": 53.5, "close": 53.9, "volume": 76000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "thin bullish confirmation volume" in result["execution_shadow_reason"]


def test_execution_shadow_ignores_high_volume_weak_close_candle_for_longs():
    candidate = {"ticker": "SLBWEAK", "signal": "long", "ema21_4h": 52.0}
    preview = {"entry_price": 53.0, "atr14": 1.0}
    bars = _leading_bars(10, base=52.0, volume=250000) + [
        {"time": "2026-08-24T00:00:00Z", "open": 53.40, "high": 53.75, "low": 53.10, "close": 53.57, "volume": 90000},
        {"time": "2026-08-24T04:00:00Z", "open": 54.01, "high": 54.12, "low": 53.20, "close": 53.60, "volume": 425987},
        {"time": "2026-08-24T08:00:00Z", "open": 53.70, "high": 53.75, "low": 53.30, "close": 53.55, "volume": 80000},
        {"time": "2026-08-24T12:00:00Z", "open": 53.55, "high": 53.90, "low": 53.40, "close": 53.75, "volume": 90000},
        {"time": "2026-08-24T16:00:00Z", "open": 53.70, "high": 54.00, "low": 53.50, "close": 53.90, "volume": 76000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "thin bullish confirmation volume" in result["execution_shadow_reason"]


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
    assert "directional expansion only" in result["execution_shadow_reason"]


def test_execution_shadow_rejects_decline_with_dead_cat_bounce():
    candidate = {"ticker": "TWLOLIKE", "signal": "long", "ema21_4h": 224.0}
    preview = {"entry_price": 223.0, "atr14": 7.0}
    bars = [
        {"time": "2026-08-21T00:00:00Z", "open": 250.5, "high": 252.0, "low": 248.0, "close": 249.52, "volume": 300000},
        {"time": "2026-08-21T04:00:00Z", "open": 249.4, "high": 250.0, "low": 238.5, "close": 239.94, "volume": 310000},
        {"time": "2026-08-21T08:00:00Z", "open": 239.8, "high": 241.0, "low": 233.5, "close": 234.29, "volume": 295000},
        {"time": "2026-08-21T12:00:00Z", "open": 234.1, "high": 235.0, "low": 228.2, "close": 228.70, "volume": 305000},
        {"time": "2026-08-21T16:00:00Z", "open": 228.5, "high": 229.5, "low": 223.5, "close": 224.10, "volume": 315000},
        {"time": "2026-08-22T00:00:00Z", "open": 224.0, "high": 226.0, "low": 220.5, "close": 221.20, "volume": 290000},
        {"time": "2026-08-22T04:00:00Z", "open": 221.1, "high": 223.2, "low": 218.0, "close": 219.40, "volume": 300000},
        {"time": "2026-08-22T08:00:00Z", "open": 219.3, "high": 221.0, "low": 216.2, "close": 217.10, "volume": 305000},
        {"time": "2026-08-22T12:00:00Z", "open": 217.0, "high": 219.5, "low": 215.0, "close": 216.40, "volume": 310000},
        {"time": "2026-08-22T16:00:00Z", "open": 216.2, "high": 218.0, "low": 213.5, "close": 214.50, "volume": 300000},
        {"time": "2026-08-24T00:00:00Z", "open": 214.4, "high": 217.0, "low": 213.0, "close": 216.20, "volume": 320000},
        {"time": "2026-08-24T04:00:00Z", "open": 216.0, "high": 219.5, "low": 215.0, "close": 218.80, "volume": 410000},
        {"time": "2026-08-24T08:00:00Z", "open": 218.7, "high": 221.0, "low": 217.5, "close": 220.40, "volume": 390000},
        {"time": "2026-08-24T12:00:00Z", "open": 220.2, "high": 223.0, "low": 219.0, "close": 222.10, "volume": 380000},
        {"time": "2026-08-24T16:00:00Z", "open": 222.0, "high": 224.5, "low": 221.0, "close": 223.08, "volume": 360000},
    ]

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "directional expansion only" in result["execution_shadow_reason"]


def test_execution_shadow_requires_full_range_window():
    candidate = {"ticker": "SHORTDATA", "signal": "long", "ema21_4h": 100.0}
    preview = {"entry_price": 100.0, "atr14": 4.0}
    bars = _leading_bars(14, base=100.0, volume=1000)

    result = candidates_router._execution_shadow_from_bars(candidate, preview, bars)

    assert result["execution_shadow_ok"] is False
    assert "Need 15 recent 4H bars" in result["execution_shadow_reason"]
