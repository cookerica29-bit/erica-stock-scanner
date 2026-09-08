#!/usr/bin/env python3
"""Kairos Sprint 4.1 -- Shadow Strategy Fidelity Calibration (2026-09
session). Deterministic scenario tests proving the recalibrated
smc_shadow_engine.py behaves per the intended playbook, not the
accidentally-overconstrained Sprint 4 implementation:

  1. A valid HTF location resolves WITHOUT an order block.
  2. No meaningful HTF location of any type -> stays WATCHING.
  3. 30M CHoCH with weak displacement -> no confirmation.
  4. 30M CHoCH with meaningful displacement -> CONFIRMED.
  5. Confirmation followed by expansion without pullback -> no entry/chase.
  6. Pullback followed by a 5M CHoCH -> eligible execution path.
  7. Pullback followed by a 5M BOS -> eligible execution path.
  8. Pullback followed by a 5M REJECTION -> eligible execution path (bonus
     -- not one of the 9 named scenarios, but section 2 explicitly
     requires REJECTION be a supported execution event type, so it gets
     its own dedicated proof here too).
  9. No clean 30M execution zone, but a valid 5M displacement-created
     execution area (the literal "Silver lesson").
 10. Invalidation at any intermediate stage.

Where a literally-named scenario has no deterministic primitive behind it
(supply/demand zone detection specifically), that limitation is reported
explicitly via its own test rather than approximated -- see
test_supply_demand_location_type_has_no_deterministic_primitive_and_is_never_resolved.

Same swing-detection-mocking convention as tests/smc_shadow_engine_v1.py
(scanner._find_swings patched per-scenario by exact DataFrame identity;
every other primitive -- _detect_bos, _detect_choch, _find_order_block,
detect_liquidity_sweep, detect_rejection, levels_near_target,
_compute_atr, score_displacement, score_location -- runs for real).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402
from smc_shadow_engine import evaluate_smc_shadow, SmcShadowConfig, LOCATION_TYPES, UNRESOLVED_LOCATION_TYPES  # noqa: E402
from smc_shadow_engine_v1 import bars, patch_swings, quiet_bars, _scenario_a_frames  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Valid HTF location without an order block.
# ---------------------------------------------------------------------------

def test_1_valid_htf_location_without_order_block_structural_level():
    # Every 4H candle is bullish (no order block possible), but price
    # sits right on top of a real, strong swing-low pivot -- a genuine
    # structural support location, resolved without any order block.
    rows_4h = [(94, 97, 93, 96, 1_000_000) for _ in range(20)]
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "low", "index": 2, "price": 90.0},
        {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 95.0},   # near current price 96 -- support
        {"type": "high", "index": 15, "price": 110.0},
    ]
    with patch_swings({id(df_4h): swings_4h}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)

    assert result["location"]["location_type"] == "structural_level"
    assert result["location"]["location_source"] == "structural_resistance.levels_near_target"
    assert result["location"]["location_score"] > 0
    assert result["zone"] == result["location"]["location_bounds"]
    assert any("no deterministic 4H order block" in u for u in result["unresolved"])
    assert result["state"] == "LOCATION_REACHED"


def test_1b_valid_htf_location_without_order_block_premium_discount():
    # Same idea, but no swing pivot happens to sit near price either --
    # only the broader premium/discount read resolves.
    rows_4h = [(94, 97, 93, 96, 1_000_000) for _ in range(20)]
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "low", "index": 2, "price": 85.0},
        {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 90.0},
        {"type": "high", "index": 15, "price": 150.0},  # far range -- no pivot near price 96
    ]
    with patch_swings({id(df_4h): swings_4h}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)

    assert result["location"]["location_type"] == "premium_discount"
    assert result["location"]["alignment"] == "favorable"
    assert result["state"] == "LOCATION_REACHED"


# ---------------------------------------------------------------------------
# 2. No meaningful HTF location of any type -> stays WATCHING.
# ---------------------------------------------------------------------------

def test_2_no_meaningful_htf_location_remains_watching():
    rows_4h = [(99, 101, 98, 100, 1_000_000) for _ in range(20)]  # dead midrange
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
    assert result["location"]["alignment"] == "neutral"
    assert result["state"] == "WATCHING"
    assert any("no deterministic HTF location" in u for u in result["unresolved"])


def test_supply_demand_location_type_has_no_deterministic_primitive_and_is_never_resolved():
    """Explicit limitation report, per this sprint's own instruction: if a
    named scenario has no deterministic backing, say so rather than invent
    logic. "supply_demand" is a real name in the schema (LOCATION_TYPES)
    but is never resolved by any code path in smc_shadow_v1 -- there is
    no deterministic supply/demand zone detector in this codebase distinct
    from the order-block concept already covered."""
    assert "supply_demand" in LOCATION_TYPES
    assert "supply_demand" in UNRESOLVED_LOCATION_TYPES
    config = SmcShadowConfig()
    assert "supply_demand" not in config.accepted_htf_location_types

    rows_4h = [(99, 101, 98, 100, 1_000_000) for _ in range(20)]
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "low", "index": 2, "price": 90.0}, {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 95.0}, {"type": "high", "index": 15, "price": 110.0},
    ]
    with patch_swings({id(df_4h): swings_4h}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)
    assert result["location"]["location_type"] != "supply_demand"
    assert any("supply/demand zone detection has no deterministic primitive" in u for u in result["unresolved"])


# ---------------------------------------------------------------------------
# 3 & 4. 30M CHoCH confirmation -- weak displacement rejected, meaningful
# displacement confirms. (BOS-based confirmation is already covered
# exhaustively in tests/smc_shadow_engine_v1.py; these prove CHoCH,
# specifically, works as its own independent confirmation path.)
# ---------------------------------------------------------------------------

def _choch_confirmation_frames(confirming_body):
    """4H location (reused from scenario A) + a 30M frame whose swings
    form a bullish CHoCH pattern (2 highs, 2 lows, HH after a HL) with
    price NEVER closing back above the broken high -- so BOS never also
    fires, isolating CHoCH as the confirming event type."""
    df_4h, _, _, swing_map = _scenario_a_frames()

    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(12)]  # bars 0-11, quiet, well inside [93,98] zone
    rows_30m.append(confirming_body)  # bar 12 -- the CHoCH-triggering candle
    rows_30m += [(96, 97, 95, 96, 1_000_000) for _ in range(15)]  # bars 13-27, stay quiet/below 100 (no BOS)
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = [
        {"type": "low", "index": 2, "price": 90.0},
        {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 8, "price": 95.0},   # HL vs the prior low (90)
        {"type": "high", "index": 12, "price": 105.0},  # HH vs the prior high (100) -- bullish CHoCH here
    ]
    swing_map = {id(df_4h): swing_map[id(df_4h)], id(df_30m): swings_30m}
    return df_4h, df_30m, swing_map


def test_3_30m_choch_weak_displacement_no_confirmation():
    # A tiny, quiet candle at the CHoCH swing index -- technically forms
    # the pattern, but nothing displaced.
    df_4h, df_30m, swing_map = _choch_confirmation_frames((95.5, 96.0, 95.3, 95.8, 900_000))
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)

    assert result["confirmation_event"]["detected"] is True
    assert result["confirmation_event"]["type"] == "CHoCH"
    assert result["displacement_evidence"]["confirmation"]["label"] == "WEAK"
    assert result["state"] == "LOCATION_REACHED", "a weak CHoCH must not confirm, same rule as a weak BOS"


def test_4_30m_choch_meaningful_displacement_confirmed():
    # A genuinely big-bodied, high-volume candle at the same swing index --
    # closes at 99, well BELOW the broken high of 100, so BOS never
    # independently fires; only CHoCH does.
    df_4h, df_30m, swing_map = _choch_confirmation_frames((94.0, 105.0, 93.0, 99.0, 5_000_000))
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)

    assert result["confirmation_event"]["detected"] is True
    assert result["confirmation_event"]["type"] == "CHoCH"
    assert result["displacement_evidence"]["confirmation"]["label"] in ("MODERATE", "STRONG")
    assert result["state"] in ("CONFIRMED", "WAITING_FOR_PULLBACK")


# ---------------------------------------------------------------------------
# 5. Confirmation followed by expansion without pullback -> no entry/chase.
# (Already covered thoroughly in tests/smc_shadow_engine_v1.py's
# scenario C; this is the fidelity-list's own dedicated pointer to it,
# re-asserted here directly so this file is a complete, self-contained
# record of all 9 required scenarios.)
# ---------------------------------------------------------------------------

def test_5_confirmation_followed_by_expansion_without_pullback_no_chase():
    df_4h, _, df_5m, swing_map = _scenario_a_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # confirms
    price = 103
    for _ in range(14):
        price += 1.5
        rows_30m.append((price - 1.4, price + 0.3, price - 1.5, price, 1_000_000))  # straight expansion, no pullback
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = {id(df_30m): [
        {"type": "low", "index": 3, "price": 94.0}, {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0}, {"type": "high", "index": 20, "price": 99.5},
    ]}
    swing_map = {id(df_4h): swing_map[id(df_4h)], **swings_30m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, df_5m)

    assert result["pullback_status"] == "chased_no_pullback"
    assert result["pullback_detail"]["returned_to_execution_area"] is False
    assert result["state"] == "WAITING_FOR_PULLBACK"
    assert result["next_condition"]["waiting_on"] == "pullback"


# ---------------------------------------------------------------------------
# 6, 7, 8. Pullback followed by a 5M CHoCH / BOS / REJECTION -- each an
# independently eligible execution path.
# ---------------------------------------------------------------------------

def _pulled_back_setup():
    """Scenario A's clean 4H+30M setup, through a genuine pullback, with
    NO 5M data yet -- the exact state needed to test each 5M execution
    event type independently."""
    df_4h, df_30m, _, swing_map = _scenario_a_frames()
    return df_4h, df_30m, swing_map


def test_6_pullback_followed_by_5m_choch_is_an_eligible_execution_path():
    df_4h, df_30m, swing_map = _pulled_back_setup()
    rows_5m = [(101, 102.5, 100.5, 101.5, 1_000_000) for _ in range(10)]  # quiet, no BOS setup
    # Closes at 99 -- BELOW H0's price (103), so BOS's "close > prev_high"
    # check never fires -- only CHoCH's swing-pattern check (which reads
    # the patched swing list, not this candle's close) does.
    rows_5m.append((93, 107, 92, 99, 5_000_000))
    rows_5m += [(99, 100, 98, 99.5, 1_000_000) for _ in range(9)]
    df_5m = bars(rows_5m, freq="5min", start="2026-02-05")
    swings_5m = {id(df_5m): [
        {"type": "low", "index": 2, "price": 99.0},
        {"type": "high", "index": 6, "price": 103.0},   # H0
        {"type": "low", "index": 8, "price": 100.0},    # HL between H0 and H1
        {"type": "high", "index": 10, "price": 107.0},  # H1 (HH) -- bullish CHoCH here, level=103
    ]}
    swing_map = {**swing_map, **swings_5m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, df_5m)

    assert result["execution_event"]["detected"] is True
    assert result["execution_event"]["type"] == "CHoCH", result["execution_event"]
    assert result["state"] in ("EXECUTION_READY", "ENTRY_READY")


def test_7_pullback_followed_by_5m_bos_is_an_eligible_execution_path():
    # The exact same real BOS-triggering 4H/30M/5M scenario A already
    # uses -- one call, so every DataFrame's identity is consistent with
    # the swing_map that was built from it.
    df_4h, df_30m, df_5m, swing_map = _scenario_a_frames()
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, df_5m)

    assert result["execution_event"]["detected"] is True
    assert result["execution_event"]["type"] == "BOS"
    assert result["state"] == "ENTRY_READY"


def test_8_pullback_followed_by_5m_rejection_is_an_eligible_execution_path():
    df_4h, df_30m, swing_map = _pulled_back_setup()
    # A liquidity sweep of the most recent swing low (scanner.
    # detect_liquidity_sweep reads swings[-1] of the relevant type --
    # here that's the low at 100.0), then a real rejection candle
    # (closes back above the swept level) -- both existing, real
    # scanner.py primitives. Only 4 bars after the quiet run so both the
    # sweep candle and the rejection candle fall inside
    # detect_liquidity_sweep's (lookback=12) and detect_rejection's
    # (lookback=5) own trailing windows, matching their real semantics.
    rows_5m = [(101, 102, 100.5, 101.5, 1_000_000) for _ in range(10)]
    rows_5m.append((101, 101.5, 97.0, 97.5, 1_000_000))     # sweeps below the swing low (100) intrabar
    rows_5m.append((97.5, 103.0, 97.0, 102.0, 4_000_000))   # rejection: closes back above 100, strong body
    rows_5m += [(102, 103, 101.5, 102.5, 1_000_000) for _ in range(2)]
    df_5m = bars(rows_5m, freq="5min", start="2026-02-05")
    swings_5m = {id(df_5m): [
        {"type": "low", "index": 2, "price": 99.0},
        {"type": "high", "index": 6, "price": 103.5},  # only 1 high -- BOS/CHoCH structurally cannot fire
        {"type": "low", "index": 9, "price": 100.0},
    ]}
    swing_map = {**swing_map, **swings_5m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, df_5m)

    assert result["execution_event"]["detected"] is True
    assert result["execution_event"]["type"] == "REJECTION", result["execution_event"]
    assert result["execution_event"]["level"] == 100.0
    # REJECTION has no displacement score of its own -- scanner.
    # detect_rejection's wick/close-through-level test IS its own quality
    # bar (a meaningful-vs-noise distinction already built into that
    # function), so unlike BOS/CHoCH there's no separate displacement
    # gate between EXECUTION_READY and ENTRY_READY for this event type --
    # a detected rejection goes straight to ENTRY_READY.
    assert result["state"] == "ENTRY_READY", result
    assert result["displacement_evidence"]["execution"] is None


# ---------------------------------------------------------------------------
# 9. No clean 30M execution zone, but a valid 5M displacement-created
# execution area -- the literal "Silver lesson".
# ---------------------------------------------------------------------------

def test_9_no_clean_30m_execution_zone_but_valid_5m_displacement_created_execution_area():
    df_4h, _, _, swing_map = _scenario_a_frames()

    # The 30M confirmation candle is a near-doji -- real BOS, real
    # STRONG displacement (huge range + volume so the score clears the
    # bar), but its OWN body [open, close] is razor-thin -- too small to
    # serve as a meaningful pullback target on its own.
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((99.9, 106.0, 93.0, 100.1, 6_000_000))  # body is only 0.2 wide; range/volume huge
    rows_30m += [
        (100.1, 101, 99.5, 100.5, 1_000_000),
        (100.5, 101, 99.8, 100.2, 1_000_000),
        (100.2, 101, 99.9, 100.6, 1_000_000),
    ]
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = {id(df_30m): [
        {"type": "low", "index": 3, "price": 94.0}, {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0}, {"type": "high", "index": 20, "price": 99.5},
    ]}

    # A 5M candle inside the SAME 30-minute window as the confirmation bar
    # (2026-02-01T12:30 .. 13:00) shows real, meaningfully-displaced
    # bullish price action -- this is the "5M execution zone created by
    # displacement" the Silver lesson describes.
    confirmation_bar_start = df_30m.index[25]
    rows_5m = [(96, 97, 95.5, 96.5, 1_000_000) for _ in range(3)]  # quiet bars just before, different window
    rows_5m += [
        (97, 98, 96.5, 97.5, 1_000_000),        # within the 30M window, quiet
        (97.5, 105, 96.5, 102, 6_000_000),      # within the window -- the real displacement candle
        (102, 103, 101, 102.5, 1_000_000),      # within the window, quiet
    ]
    df_5m = bars(rows_5m, freq="5min", start=confirmation_bar_start - pd.Timedelta(minutes=15))

    swing_map = {id(df_4h): swing_map[id(df_4h)], **swings_30m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, df_5m)

    assert result["confirmation_event"]["type"] == "BOS"
    assert result["displacement_evidence"]["confirmation"]["label"] in ("MODERATE", "STRONG")
    assert result["execution_area"] is not None, "must not silently give up just because the 30M body was too thin"
    assert result["execution_area"]["source"] == "5m_displacement_candle_body", result["execution_area"]
    assert not any("execution_area:" in u for u in result["unresolved"])


def test_9b_no_30m_zone_and_no_5m_data_is_honestly_unresolved_not_guessed():
    df_4h, _, _, swing_map = _scenario_a_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((99.9, 106.0, 93.0, 100.1, 6_000_000))  # same razor-thin body as test 9
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = {id(df_30m): [
        {"type": "low", "index": 3, "price": 94.0}, {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0}, {"type": "high", "index": 20, "price": 99.5},
    ]}
    swing_map = {id(df_4h): swing_map[id(df_4h)], **swings_30m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)  # no 5M data at all

    assert result["execution_area"] is None
    assert any("no 5M data is available" in u for u in result["unresolved"])
    assert result["pullback_status"] == "not_yet_assessable"


# ---------------------------------------------------------------------------
# 10. Invalidation at any intermediate stage.
# ---------------------------------------------------------------------------

def test_10_invalidation_at_location_reached_stage():
    df_4h, _, _, swing_map = _scenario_a_frames()
    # Zone from scenario A is [93, 98] -- price closes well below it, no
    # 30M confirmation ever attempted.
    df_30m = bars([(90, 90, 80, 82, 1_000_000)], freq="30min", start="2026-02-01")
    with patch_swings({id(df_4h): swing_map[id(df_4h)]}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)
    assert result["invalidation"]["invalidated"] is True
    assert result["state"] == "INVALIDATED"


def test_10b_invalidation_at_waiting_for_pullback_stage():
    df_4h, _, _, swing_map = _scenario_a_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # confirms
    rows_30m += [
        (103, 103, 95, 96, 1_000_000),
        (96, 96, 88, 89, 1_000_000),
        (89, 89, 84, 85, 1_000_000),
        (85, 85, 80, 82, 1_000_000),  # closes at 82, below zone low 93 -- invalidated
    ]
    df_30m = bars(rows_30m, freq="30min", start="2026-02-01")
    swings_30m = {id(df_30m): [
        {"type": "low", "index": 3, "price": 94.0}, {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0}, {"type": "high", "index": 20, "price": 99.5},
    ]}
    swing_map = {id(df_4h): swing_map[id(df_4h)], **swings_30m}
    with patch_swings(swing_map):
        result = evaluate_smc_shadow("TEST", "long", df_4h, df_30m, None)

    assert result["confirmation_event"]["detected"] is True
    assert result["invalidation"]["invalidated"] is True
    assert result["state"] == "INVALIDATED", result["state"]
    assert result["next_condition"]["waiting_on"] == "none"


def test_10c_invalidation_via_a_non_order_block_location_bounds():
    # Invalidation must work off WHICHEVER location type resolved, not
    # only an order block -- here the anchor is premium_discount's own
    # range bounds (range_high/range_low from the swing range itself,
    # fixed regardless of current price -- unlike structural_level's
    # nearest-pivot-to-current-price bounds, which are deliberately NOT
    # used for this test; see the Sprint 4.1 report's disclosed
    # limitation on structural_level's invalidation anchor).
    rows_4h = [(94, 97, 93, 96, 1_000_000) for _ in range(20)]
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01")
    swings_4h = [
        {"type": "low", "index": 2, "price": 85.0}, {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 90.0}, {"type": "high", "index": 15, "price": 150.0},
    ]
    with patch_swings({id(df_4h): swings_4h}):
        result = evaluate_smc_shadow("TEST", "long", df_4h, None, None)
    assert result["location"]["location_type"] == "premium_discount"
    bounds = result["location"]["location_bounds"]
    assert result["invalidation"]["level"] == bounds["low"] == 85.0

    # Now re-evaluate with a later close that has dropped below that bound.
    rows_4h_invalidated = rows_4h + [(96, 96, 80.0, 82.0, 1_000_000)]
    df_4h_2 = bars(rows_4h_invalidated, freq="4h", start="2026-01-01")
    with patch_swings({id(df_4h_2): swings_4h}):
        result2 = evaluate_smc_shadow("TEST", "long", df_4h_2, None, None)
    assert result2["invalidation"]["invalidated"] is True
    assert result2["state"] == "INVALIDATED"


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
    print(f"smc_shadow_engine_fidelity_v1: {len(tests)}/{len(tests)} passed.")
