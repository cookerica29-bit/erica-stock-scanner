"""Kairos Watch Contract -- pullback copy PARITY tests (2026-09 session,
pre-deploy hardening pass, section 3).

watch_contract_sensors.py's resolve_pullback_reference/windowed_clear_and_return
are a deliberate, narrow COPY (not an import) of hybrid_shadow_engine.py's
_resolve_pullback_reference/_windowed_clear_and_return -- see this
session's own architecture audit for why (the Watch Contract must never
import that module, a separate autonomous-thesis research engine). A copy
can silently drift from its original during transcription; these tests
exist SPECIFICALLY to catch that, by running IDENTICAL inputs through
BOTH the copy and the real, unmodified hybrid_shadow_engine functions and
asserting IDENTICAL outputs. hybrid_shadow_engine.py itself is imported
ONLY here, in a test, purely for this comparison -- never in any
production watch_contract_*.py module.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hybrid_shadow_engine as hse  # noqa: E402 -- TEST-ONLY reference implementation
import watch_contract_sensors as wcs  # noqa: E402

_CONFIG = hse.HYBRID_STRATEGY_CONFIGS[hse.DEFAULT_HYBRID_STRATEGY_VERSION]


def _bars(rows):
    times = pd.date_range(start="2026-01-05T14:00:00Z", periods=len(rows), freq="30min", tz="UTC")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=times)


# ---------------------------------------------------------------------------
# windowed_clear_and_return parity
# ---------------------------------------------------------------------------

def test_parity_clear_then_return():
    # entry (confirmation) bar at index 0; bars 1-3 clear above the zone
    # [99, 100]; bar 4 dips back into the zone -- a genuine return.
    df = _bars([
        (99.3, 99.5, 99.2, 99.3, 1),
        (100.6, 101.2, 100.6, 101.0, 1),
        (101.0, 101.5, 101.0, 101.4, 1),
        (101.4, 101.8, 101.2, 101.6, 1),
        (100.3, 100.6, 100.0, 100.3, 1),
        (99.4, 99.6, 99.2, 99.4, 1),
    ])
    copy_result = wcs.windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", window_bars=60)
    real_result = hse._windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", 60)
    assert copy_result == real_result == (True, 1, 4)


def test_parity_no_clear():
    # Price stays inside [99, 100] the entire window -- never clears.
    rows = [(99.5, 99.7, 99.3, 99.5, 1)] * 10
    df = _bars(rows)
    copy_result = wcs.windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", window_bars=60)
    real_result = hse._windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", 60)
    assert copy_result == real_result == (False, None, None)


def test_parity_clear_but_no_return():
    # Clears above the zone at bar 1, then runs away and never comes back.
    df = _bars([
        (99.3, 99.5, 99.2, 99.3, 1),
        (100.6, 101.2, 100.6, 101.0, 1),
        (102.0, 102.5, 101.8, 102.3, 1),
        (103.0, 103.5, 102.8, 103.2, 1),
        (104.0, 104.5, 103.8, 104.2, 1),
    ])
    copy_result = wcs.windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", window_bars=60)
    real_result = hse._windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", 60)
    assert copy_result == real_result == (False, 1, None)


def test_parity_bounded_window_expiry():
    # A return that lands EXACTLY at entry_index + window_bars must be
    # found; one bar later (outside the window) must not be -- and both
    # implementations must agree at both boundaries, not just in general.
    window_bars = 5
    rows = [(99.3, 99.5, 99.2, 99.3, 1)]  # bar 0: confirmation
    rows += [(101.0, 101.5, 100.8, 101.2, 1)] * window_bars  # bars 1..window_bars: cleared, holding away
    rows[1] = (100.6, 101.2, 100.6, 101.0, 1)  # bar 1 is the actual clear
    rows.append((99.5, 99.8, 99.3, 99.5, 1))  # bar window_bars+1: the return, one bar too late
    df = _bars(rows)

    copy_late = wcs.windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", window_bars=window_bars)
    real_late = hse._windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", window_bars)
    assert copy_late == real_late == (False, 1, None), "a return one bar past the window must be missed by BOTH implementations identically"

    copy_ontime = wcs.windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", window_bars=window_bars + 1)
    real_ontime = hse._windowed_clear_and_return(df, 0, 99.0, 100.0, "LONG", window_bars + 1)
    assert copy_ontime == real_ontime
    assert copy_ontime[0] is True, "the same return, now inside a window one bar wider, must be found by BOTH implementations identically"


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_parity_both_directions_on_a_shared_pattern(direction):
    sign = 1 if direction == "LONG" else -1
    df = _bars([
        (100.0, 100.0 + sign * 0.2, 100.0 - sign * 0.2, 100.0, 1),
        (100.0 + sign * 1.5, 100.0 + sign * 1.8, 100.0 + sign * 1.3, 100.0 + sign * 1.6, 1),
        (100.0 + sign * 1.6, 100.0 + sign * 2.0, 100.0 + sign * 1.5, 100.0 + sign * 1.9, 1),
        (100.0 + sign * 0.7, 100.0 + sign * 0.9, 100.0 + sign * 0.3, 100.0 + sign * 0.5, 1),
    ])
    zone_low, zone_high = (99.5, 100.5) if direction == "LONG" else (99.5, 100.5)
    copy_result = wcs.windowed_clear_and_return(df, 0, zone_low, zone_high, direction, window_bars=60)
    real_result = hse._windowed_clear_and_return(df, 0, zone_low, zone_high, direction, 60)
    assert copy_result == real_result


# ---------------------------------------------------------------------------
# resolve_pullback_reference parity
# ---------------------------------------------------------------------------

def test_parity_persisted_arbitrary_location_bounds():
    df30 = _bars([(100.0, 100.5, 99.5, 100.2, 1)] * 5)
    location = {"level_bounds": {"high": 105.25, "low": 103.75}, "level_id": "OVV:4H:swing_pivot:2026-08-01T13:30:00Z"}
    unresolved_copy, unresolved_real = [], []
    copy_result = wcs.resolve_pullback_reference(df30, None, 0, location, "long", unresolved_copy)
    real_result = hse._resolve_pullback_reference(df30, None, 0, location, "long", _CONFIG, unresolved_real)
    assert copy_result == real_result == {
        "high": 105.25, "low": 103.75, "reference_type": "location_level", "reference_provenance": location["level_id"],
    }
    assert unresolved_copy == unresolved_real == []


def test_parity_confirmation_candle_body_fallback():
    # No location bounds supplied -- both must fall back to the
    # confirmation candle's own body (tier 2), identically.
    rows = [(100.0, 100.2, 99.9, 100.1, 1)] * 20  # enough bars for a real ATR
    rows.append((99.0, 101.0, 98.8, 100.8, 1))  # confirming bar: a real, sizeable body
    df30 = _bars(rows)
    location = {"level_bounds": None, "level_id": None}
    unresolved_copy, unresolved_real = [], []
    copy_result = wcs.resolve_pullback_reference(df30, None, len(df30) - 1, location, "long", unresolved_copy)
    real_result = hse._resolve_pullback_reference(df30, None, len(df30) - 1, location, "long", _CONFIG, unresolved_real)
    assert copy_result == real_result
    assert copy_result["reference_type"] == "confirmation_candle_body"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
