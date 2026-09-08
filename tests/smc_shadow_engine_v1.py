#!/usr/bin/env python3
"""Kairos Sprint 4 -- Shadow SMC Strategy/State Engine (2026-09 session).
Unit tests for smc_shadow_engine.evaluate_smc_shadow -- pure, synthetic-data
regression tests, no network, no DB.

Swing detection is monkeypatched per-scenario, keyed by the exact DataFrame
object (id(df)), mirroring the established convention in
tests/stock_mtf_structure_shadow_v1.py (which patches scanner._find_swings
directly rather than trying to coax real swing detection out of hand-built
candles) -- this makes each scenario's structural facts (HH/HL, prior
highs/lows for BOS/CHoCH) explicit and independent of the swing-detection
algorithm's own tuning, while every OTHER primitive this engine calls
(_detect_bos, _detect_choch, _find_order_block, _compute_atr,
score_displacement, score_location) runs for REAL against the hand-built
OHLCV data -- these tests exercise the real reused functions, not mocks of
them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402
from smc_shadow_engine import evaluate_smc_shadow, SmcShadowConfig, STRATEGY_VERSION  # noqa: E402


def bars(rows, freq="30min", start="2026-01-01"):
    """rows: list of (open, high, low, close[, volume]) tuples."""
    data = []
    for r in rows:
        o, h, l, c = r[0], r[1], r[2], r[3]
        v = r[4] if len(r) > 4 else 1_000_000
        data.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
    idx = pd.date_range(start, periods=len(data), freq=freq, tz="UTC")
    return pd.DataFrame(data, index=idx)


class patch_swings:
    """Context manager: scanner._find_swings(df, ...) returns swing_map[id(df)]
    for any df passed in, regardless of margin/tolerance."""
    def __init__(self, swing_map):
        self.swing_map = swing_map
        self.original = None

    def __enter__(self):
        self.original = scanner._find_swings

        def fake(df, margin=4, tolerance=None):
            return list(self.swing_map.get(id(df), []))

        scanner._find_swings = fake
        return self

    def __exit__(self, *exc):
        scanner._find_swings = self.original


def quiet_bars(n, base=95.0, freq="30min", start="2026-01-01"):
    """n small-bodied, small-range, normal-volume bars oscillating gently
    around `base` -- background noise a real displacement/confirmation
    candle should stand out against."""
    rows = []
    price = base
    for i in range(n):
        wig = 0.4 if i % 2 == 0 else -0.4
        o = price
        c = price + wig
        h = max(o, c) + 0.5
        l = min(o, c) - 0.5
        rows.append((o, h, l, c, 1_000_000))
        price = c
    return bars(rows, freq=freq, start=start)


# ---------------------------------------------------------------------------
# Scenario A: a clean full pass, long, all the way to ENTRY_READY.
# ---------------------------------------------------------------------------

def _scenario_a_frames():
    # ---- 4H: HTF thesis (LONG, via HH/HL swings) + a real order block. ----
    rows_4h = [(94, 97, 93, 96, 1_000_000) for _ in range(20)]
    rows_4h[12] = (97, 98, 93, 94, 1_000_000)  # the one bearish candle -> order block
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "low", "index": 2, "price": 90.0},
        {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 95.0},
        {"type": "high", "index": 15, "price": 110.0},
    ]

    # ---- 30M: zone touch, then a real BOS with strong displacement,
    # then expansion, then a genuine pullback. ----
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]  # bars 0-24, touches [93,98] zone, quiet
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # confirming BOS candle (closes above prev_high 99)
    rows_30m += [
        (103, 106, 102, 105, 1_000_000),  # 26
        (105, 108, 104, 107, 1_000_000),  # 27
        (107, 109, 106, 108, 1_000_000),  # 28 -- post-confirmation extreme close 108
        (108, 108, 106, 107, 1_000_000),  # 29
        (107, 107, 103, 104, 1_000_000),  # 30
        (104, 104, 101, 102, 1_000_000),  # 31
        (102, 102, 100, 101, 1_000_000),  # 32
        (101, 101, 99, 100, 1_000_000),   # 33
        (100, 101, 99, 100.5, 1_000_000),  # 34
        (100.5, 101, 99.5, 100, 1_000_000),  # 35
        (100, 101, 99, 100.5, 1_000_000),   # 36
        (100.5, 101.5, 100, 101, 1_000_000),  # 37
        (101, 101.5, 100, 101, 1_000_000),    # 38
        (101, 101.5, 100.5, 101, 1_000_000),  # 39 -- retraced, well above zone low 93
    ]
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = [
        {"type": "low", "index": 3, "price": 94.0},
        {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0},
        {"type": "high", "index": 20, "price": 99.5},
    ]

    # ---- 5M: execution trigger with strong displacement, no invalidation. ----
    rows_5m = [(101, 102.5, 100.5, 101.5, 1_000_000) for _ in range(20)]  # bars 0-19, quiet
    rows_5m.append((102, 107, 101, 106, 5_000_000))  # execution BOS candle (closes above prev_high 103)
    rows_5m += [
        (106, 107, 105, 106.5, 1_000_000),
        (106.5, 107.5, 106, 107, 1_000_000),
        (107, 108, 106.5, 107.5, 1_000_000),
        (107.5, 108, 107, 107.8, 1_000_000),
        (107.8, 108.2, 107.3, 108, 1_000_000),
        (108, 108.5, 107.5, 108.2, 1_000_000),
        (108.2, 108.7, 107.8, 108.4, 1_000_000),
        (108.4, 108.9, 108, 108.6, 1_000_000),
        (108.6, 109, 108.2, 108.8, 1_000_000),
    ]
    df_5m = bars(rows_5m, freq="5min", start="2026-02-05")
    swings_5m = [
        {"type": "low", "index": 2, "price": 99.0},
        {"type": "high", "index": 6, "price": 103.0},
        {"type": "low", "index": 12, "price": 100.0},
        {"type": "high", "index": 18, "price": 104.0},
    ]

    swing_map = {id(df_4h): swings_4h, id(df_30m): swings_30m, id(df_5m): swings_5m}
    return df_4h, df_30m, df_5m, swing_map


def test_scenario_a_clean_pass_reaches_entry_ready():
    df_4h, df_30m, df_5m, swing_map = _scenario_a_frames()
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, df_5m)

    assert result["strategy_version"] == STRATEGY_VERSION
    assert result["htf_direction"] == "LONG"
    assert result["htf_direction_aligned"] is True
    assert result["zone"] == {"high": 98.0, "low": 93.0}
    assert result["confirmation_event"]["detected"] is True
    assert result["confirmation_event"]["type"] == "BOS"
    assert result["displacement_evidence"]["confirmation"]["label"] in ("MODERATE", "STRONG")
    assert result["pullback_status"] == "observed"
    assert result["execution_event"]["detected"] is True
    assert result["displacement_evidence"]["execution"]["label"] in ("MODERATE", "STRONG")
    assert result["invalidation"]["invalidated"] is False
    assert result["state"] == "ENTRY_READY", result


def test_scenario_a_mirrors_for_short():
    # Mirror image of scenario A around 100 -- SHORT thesis, bearish order
    # block, bearish BOS confirmation, pullback UP toward the level, then a
    # bearish 5M execution trigger.
    rows_4h = [(96, 97, 93, 94, 1_000_000) for _ in range(20)]  # bearish drift
    rows_4h[12] = (94, 98, 93, 97, 1_000_000)  # the one BULLISH candle -> order block for SHORT
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "high", "index": 2, "price": 110.0},
        {"type": "low", "index": 5, "price": 100.0},
        {"type": "high", "index": 10, "price": 105.0},
        {"type": "low", "index": 15, "price": 90.0},
    ]

    rows_30m = [(96, 97, 94, 95, 1_000_000) for _ in range(25)]  # touches [93,98] zone
    rows_30m.append((93, 94, 87, 88, 5_000_000))  # bearish BOS candle (closes below prev_low 92)
    rows_30m += [
        (88, 89, 85, 86, 1_000_000),   # 26
        (86, 87, 83, 84, 1_000_000),   # 27
        (84, 85, 82, 83, 1_000_000),   # 28 -- extreme close 83
        (83, 85, 82.5, 84, 1_000_000),  # 29
        (84, 87, 83.5, 86, 1_000_000),  # 30
        (86, 89, 85.5, 88, 1_000_000),  # 31
        (88, 90, 87.5, 89, 1_000_000),  # 32
        (89, 90.5, 88.5, 90, 1_000_000),  # 33 -- retraced back up, well below invalidation (zone high 98)
        (90, 90.5, 89.5, 90, 1_000_000),
        (90, 90.5, 89.5, 90, 1_000_000),
        (90, 90.5, 89.5, 90, 1_000_000),
        (90, 90.5, 89.5, 90, 1_000_000),
        (90, 90.5, 89.5, 90, 1_000_000),
        (90, 90.5, 89.5, 90, 1_000_000),
    ]
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = [
        {"type": "high", "index": 3, "price": 96.0},
        {"type": "low", "index": 8, "price": 92.0},
        {"type": "high", "index": 14, "price": 95.0},
        {"type": "low", "index": 20, "price": 92.5},
    ]

    rows_5m = [(90, 90.5, 88.5, 89.5, 1_000_000) for _ in range(20)]
    rows_5m.append((89, 90, 84, 85, 5_000_000))  # bearish execution BOS (closes below prev_low 87)
    rows_5m += [(85 - i * 0.1, 85.5 - i * 0.1, 84 - i * 0.1, 84.5 - i * 0.1, 1_000_000) for i in range(9)]
    df_5m = bars(rows_5m, freq="5min", start="2026-02-05")
    swings_5m = [
        {"type": "high", "index": 2, "price": 91.0},
        {"type": "low", "index": 6, "price": 87.0},
        {"type": "high", "index": 12, "price": 90.0},
        {"type": "low", "index": 18, "price": 86.0},
    ]

    swing_map = {id(df_4h): swings_4h, id(df_30m): swings_30m, id(df_5m): swings_5m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "short", df_4h, df_30m, df_5m)

    assert result["htf_direction"] == "SHORT"
    assert result["zone"] == {"high": 98.0, "low": 93.0}
    assert result["confirmation_event"]["type"] == "BOS"
    assert result["pullback_status"] == "observed"
    assert result["execution_event"]["detected"] is True
    assert result["state"] == "ENTRY_READY", result


# ---------------------------------------------------------------------------
# Scenario B: weak/noisy 30M break must NOT count as a real confirmation.
# ---------------------------------------------------------------------------

def test_scenario_b_weak_displacement_does_not_confirm():
    df_4h, df_30m_clean, df_5m, swing_map_clean = _scenario_a_frames()
    # Replace the confirming candle (index 25) with a technically-triggering
    # but tiny, quiet candle -- same close level, negligible body/range/volume.
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((99.5, 99.7, 99.4, 99.6, 900_000))  # barely closes above prev_high 99, tiny body
    rows_30m += [(99.6, 99.8, 99.4, 99.5, 900_000) for _ in range(14)]  # stays flat afterward
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = [
        {"type": "low", "index": 3, "price": 94.0},
        {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0},
        {"type": "high", "index": 20, "price": 99.5},
    ]
    swing_map = {id(df_4h): swing_map_clean[id(df_4h)], id(df_30m): swings_30m}

    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)

    assert result["confirmation_event"]["detected"] is True, "the structural break itself still gets reported"
    assert result["confirmation_event"]["type"] == "BOS"
    assert result["displacement_evidence"]["confirmation"]["label"] == "WEAK"
    assert "weak/noisy break" in result["confirmation_event"]["reason"]
    # Must NOT advance to CONFIRMED -- a weak/noisy break stays at LOCATION_REACHED.
    assert result["state"] == "LOCATION_REACHED", result


# ---------------------------------------------------------------------------
# Scenario C: "do not chase expansion" -- confirmed, but price has expanded
# hard away from the confirmation level with no retracement at all yet.
# ---------------------------------------------------------------------------

def test_scenario_c_chased_expansion_does_not_reach_execution_ready():
    df_4h, _, df_5m, swing_map = _scenario_a_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # same strong confirming candle as scenario A
    # Straight-line expansion afterward, NO retracement at all.
    price = 103
    for i in range(14):
        price += 1.5
        rows_30m.append((price - 1.4, price + 0.3, price - 1.5, price, 1_000_000))
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = {
        id(df_30m): [
            {"type": "low", "index": 3, "price": 94.0},
            {"type": "high", "index": 8, "price": 99.0},
            {"type": "low", "index": 14, "price": 95.0},
            {"type": "high", "index": 20, "price": 99.5},
        ],
    }
    swing_map = {**swing_map, **swings_30m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, df_5m)

    assert result["confirmation_event"]["detected"] is True
    assert result["displacement_evidence"]["confirmation"]["label"] in ("MODERATE", "STRONG")
    assert result["pullback_status"] == "chased_no_pullback", result["pullback_detail"]
    # Must NOT reach EXECUTION_READY/ENTRY_READY just because price moved
    # favorably -- exactly the "do not chase expansion" rule.
    assert result["state"] == "WAITING_FOR_PULLBACK", result


def test_scenario_c2_confirmation_bar_itself_is_not_yet_assessable():
    # Zero bars have elapsed since the confirming candle -- must read
    # CONFIRMED, not WAITING_FOR_PULLBACK (there's been no time to wait yet).
    df_4h, _, _, swing_map = _scenario_a_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = {
        id(df_30m): [
            {"type": "low", "index": 3, "price": 94.0},
            {"type": "high", "index": 8, "price": 99.0},
            {"type": "low", "index": 14, "price": 95.0},
            {"type": "high", "index": 20, "price": 99.5},
        ],
    }
    swing_map = {id(df_4h): swing_map[id(df_4h)], **swings_30m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)

    assert result["pullback_status"] == "not_yet_assessable"
    assert result["state"] == "CONFIRMED", result


# ---------------------------------------------------------------------------
# Scenario D: invalidation overrides everything, from any prior state.
# ---------------------------------------------------------------------------

def test_scenario_d_invalidation_overrides_confirmed_state():
    df_4h, _, _, swing_map = _scenario_a_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # confirms
    # Then price reverses hard and closes BELOW the 4H zone low (93).
    rows_30m += [
        (103, 103, 95, 96, 1_000_000),
        (96, 96, 88, 89, 1_000_000),
        (89, 89, 84, 85, 1_000_000),
        (85, 85, 80, 82, 1_000_000),  # closes at 82, below zone low 93
    ]
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = {
        id(df_30m): [
            {"type": "low", "index": 3, "price": 94.0},
            {"type": "high", "index": 8, "price": 99.0},
            {"type": "low", "index": 14, "price": 95.0},
            {"type": "high", "index": 20, "price": 99.5},
        ],
    }
    swing_map = {id(df_4h): swing_map[id(df_4h)], **swings_30m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)

    assert result["invalidation"]["invalidated"] is True
    assert result["invalidation"]["level"] == 93.0
    assert result["state"] == "INVALIDATED", result


def test_scenario_d2_invalidation_from_plain_watching():
    # Price never even reaches the zone, but still trades below it --
    # invalidation must still be reported honestly (state stays WATCHING
    # here since location was never reached, but the invalidation FIELD
    # itself is independent evidence -- see the field-level assertion).
    # This proves invalidation is checked against a real level whenever
    # one exists, not only once a setup has progressed further.
    df_4h, _, _, swing_map = _scenario_a_frames()
    df_30m = bars([(90, 90, 80, 82, 1_000_000)], freq="30min", start="2026-02-01")
    with patch_swings({id(df_4h): swing_map[id(df_4h)]}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)

    assert result["zone"] == {"high": 98.0, "low": 93.0}
    assert result["invalidation"]["invalidated"] is True
    assert result["state"] == "INVALIDATED", result


# ---------------------------------------------------------------------------
# Scenario E: ambiguous/insufficient data is exposed as unresolved, never
# silently guessed.
# ---------------------------------------------------------------------------

def test_scenario_e_missing_4h_data_is_unresolved_not_guessed():
    result = evaluate_smc_shadow("TEST", "long", None, None, None)
    assert result["state"] == "WATCHING"
    assert any("no 4H data" in u for u in result["unresolved"])
    assert any("invalidation" in u for u in result["unresolved"])
    assert result["location"]["percentile"] is None
    assert result["zone"] is None
    assert result["invalidation"]["invalidated"] is False


def test_scenario_e2_insufficient_htf_swing_history_is_unresolved():
    df_4h = quiet_bars(10, base=100, freq="4h")
    with patch_swings({id(df_4h): []}):  # no swings at all -> NEUTRAL trend
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)

    assert result["htf_direction"] == "NEUTRAL"
    assert result["htf_direction_aligned"] is False
    assert any("insufficient 4H swing history" in u for u in result["unresolved"])
    assert result["state"] == "WATCHING"


def test_scenario_e3_no_order_block_falls_back_to_premium_discount_location_sprint_4_1():
    # HTF direction aligns, but no bearish candle exists before the last
    # swing low -- _find_order_block legitimately returns None here.
    # Sprint 4.1: this must NOT cap at WATCHING the way Sprint 4 did --
    # price sits at a real 30th-percentile discount (favorable for long)
    # within the active 4H range, which is its own valid, deterministic
    # HTF location type now. This is the exact case Sprint 4.1 exists to
    # fix ("HTF location must not require an order block").
    rows_4h = [(94, 97, 93, 96, 1_000_000) for _ in range(20)]  # every candle bullish
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "low", "index": 2, "price": 90.0},
        {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 95.0},
        {"type": "high", "index": 15, "price": 110.0},
    ]
    with patch_swings({id(df_4h): swings_4h}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)

    assert result["htf_direction_aligned"] is True
    assert any("no deterministic 4H order block" in u for u in result["unresolved"]), "the order-block miss is still honestly recorded as evidence, just no longer fatal"
    # A swing low sits at 95 (near the current price of 96) AND price is
    # at a 30th-percentile discount -- either non-order-block location
    # type is a legitimate resolution here; structural_level wins by this
    # config's priority order (checked before premium_discount), which is
    # itself the point: some OTHER deterministic type resolved instead of
    # capping at WATCHING for lack of an order block.
    assert result["location"]["location_type"] in ("structural_level", "premium_discount")
    assert result["zone"] is not None
    # result["zone"] now generally mirrors location["location_bounds"] --
    # whichever type resolved -- not "order block or nothing" as in Sprint 4.
    assert result["zone"] == result["location"]["location_bounds"]
    assert result["state"] == "LOCATION_REACHED", "Sprint 4.1: a real non-order-block location must be enough on its own"


def test_scenario_e3b_genuinely_no_htf_location_of_any_type_caps_at_watching():
    # A true negative: price sits at dead midrange (50th percentile,
    # "neutral" alignment -- neither favorable nor unfavorable), no order
    # block exists, and no swing pivot sits anywhere near current price.
    # NONE of the three deterministic location types can resolve here --
    # this must still cap at WATCHING, proving the broadened gate isn't
    # simply "always pass".
    rows_4h = [(99, 101, 98, 100, 1_000_000) for _ in range(20)]  # every candle bullish, tight range, price ~100
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "low", "index": 2, "price": 90.0},
        {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 95.0},
        {"type": "high", "index": 15, "price": 110.0},
    ]
    with patch_swings({id(df_4h): swings_4h}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)

    assert result["location"]["location_type"] is None
    assert result["location"]["label"] == "midrange"
    assert result["location"]["alignment"] == "neutral"
    assert result["zone"] is None
    assert any("no deterministic HTF location" in u for u in result["unresolved"])
    assert any("supply/demand" in u for u in result["unresolved"]), "the reserved, always-unresolved location type must be named explicitly when nothing else resolves either"
    assert result["state"] == "WATCHING"


def test_scenario_e4_htf_direction_conflicts_with_thesis_caps_at_watching():
    # 4H structure is actually bearish (LH/LL) while the requested thesis
    # is LONG -- must not proceed past WATCHING even if a zone exists.
    rows_4h = [(96, 97, 93, 94, 1_000_000) for _ in range(20)]
    rows_4h[12] = (94, 98, 93, 97, 1_000_000)
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "high", "index": 2, "price": 110.0},
        {"type": "low", "index": 5, "price": 100.0},
        {"type": "high", "index": 10, "price": 105.0},
        {"type": "low", "index": 15, "price": 90.0},
    ]
    with patch_swings({id(df_4h): swings_4h}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)

    assert result["htf_direction"] == "SHORT"
    assert result["htf_direction_aligned"] is False
    assert result["state"] == "WATCHING", "a thesis fighting the 4H structure must never progress past WATCHING"


def test_missing_direction_is_unresolved():
    result = evaluate_smc_shadow("TEST", "sideways", None, None, None)
    assert result["state"] == "WATCHING"
    assert any("direction" in u for u in result["unresolved"])


def test_config_rejects_invalid_displacement_label():
    import pytest  # local import: only this one test needs it
    try:
        SmcShadowConfig(min_confirmation_displacement_label="ULTRA")
        assert False, "should have raised"
    except ValueError:
        pass


if __name__ == "__main__":
    import inspect
    failures = []
    module = sys.modules[__name__]
    tests = [obj for name, obj in vars(module).items() if name.startswith("test_") and inspect.isfunction(obj)]
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{fn.__name__}: {exc.__class__.__name__}: {exc}")
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print(f"smc_shadow_engine_v1: {len(tests)}/{len(tests)} passed.")
