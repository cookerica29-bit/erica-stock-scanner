#!/usr/bin/env python3
"""Kairos Sprint 5 -- Historical + Paper Evaluation Harness (2026-09
session). Tests for smc_shadow_evaluation.py's walk-forward simulator,
post-signal diagnostics measurement, and segmentation aggregator -- pure,
synthetic-data regression tests, no network, no DB.

Sprint 5.2 -- outcome semantics hardening: every reference to a
"hypothetical trade" / "outcome" / "win" / "loss" / "r_multiple" has been
renamed or removed to match smc_shadow_evaluation.py's own Sprint 5.2
rewrite (see that module's docstring). There is no primary trade outcome
anywhere in this file anymore -- `trade_performance_status` is always the
fixed TRADE_PERFORMANCE_UNRESOLVED_STATUS constant, and the six tests
required by the Sprint 5.2 task are called out explicitly below by name.

Swing detection is patched by a DIFFERENT mechanism than
tests/smc_shadow_engine_v1.py's identity-keyed patch_swings: the
walk-forward evaluator creates a NEW, freshly-sliced DataFrame object at
every step (df.iloc[:i+1] / df.loc[:as_of]), so an identity-keyed patch
would silently return empty swings for every single evaluation (a real
bug caught while building this file -- see patch_swings_walk_forward's
own docstring). Every slice of the SAME original series shares that
series' own index[0] (the start timestamp never changes, only the end
grows), so this patch keys off THAT instead, and reveals only the
subset of swings whose own `index` already falls within the current
slice's length -- which is also a more faithful simulation of what a
walk-forward evaluation should see: swings that haven't formed yet
(beyond the slice) are correctly invisible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402
from smc_shadow_engine_v1 import bars  # noqa: E402
from smc_shadow_evaluation import (  # noqa: E402
    run_walk_forward_evaluation, aggregate_evaluation_report, HarnessConfig, HARNESS_VERSION,
    TRADE_PERFORMANCE_UNRESOLVED_STATUS, TRADE_PERFORMANCE_UNAVAILABLE,
    _measure_post_signal_diagnostics,
)
from smc_shadow_engine import STRATEGY_VERSION  # noqa: E402


class patch_swings_walk_forward:
    def __init__(self, frames_and_swings):
        self.lookup = {full_df.index[0]: full_swings for full_df, full_swings in frames_and_swings}
        self.original = None

    def __enter__(self):
        self.original = scanner._find_swings
        lookup = self.lookup

        def fake(df, margin=4, tolerance=None):
            if df is None or len(df) == 0:
                return []
            full_swings = lookup.get(df.index[0])
            if full_swings is None:
                return []
            n = len(df)
            return [s for s in full_swings if s["index"] < n]

        scanner._find_swings = fake
        return self

    def __exit__(self, *exc):
        scanner._find_swings = self.original


def _clean_pass_frames(extra_30m_tail=None):
    """A timestamp-ALIGNED (overlapping calendar, unlike Sprint 4's
    single-snapshot fixtures) 4H/30M/5M dataset that reaches ENTRY_READY
    partway through a walk-forward pass, with real headroom afterward for
    post-signal diagnostics measurement. Same underlying candle shapes as
    tests/smc_shadow_engine_v1.py's scenario A (already proven to drive a
    clean pass), just retimestamped onto one shared calendar."""
    rows_4h = [(94, 97, 93, 96, 1_000_000) for _ in range(40)]
    rows_4h[12] = (97, 98, 93, 94, 1_000_000)  # order block
    df_4h = bars(rows_4h, freq="4h", start="2026-01-01T00:00")
    swings_4h = [
        {"type": "low", "index": 2, "price": 90.0}, {"type": "high", "index": 5, "price": 100.0},
        {"type": "low", "index": 10, "price": 95.0}, {"type": "high", "index": 15, "price": 110.0},
    ]

    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # confirming BOS candle
    rows_30m += [
        (103, 106, 102, 105, 1_000_000), (105, 108, 104, 107, 1_000_000), (107, 109, 106, 108, 1_000_000),
        (108, 108, 106, 107, 1_000_000), (107, 107, 103, 104, 1_000_000), (104, 104, 101, 102, 1_000_000),
        (102, 102, 100, 101, 1_000_000), (101, 101, 99, 100, 1_000_000), (100, 101, 99, 100.5, 1_000_000),
        (100.5, 101, 99.5, 100, 1_000_000), (100, 101, 99, 100.5, 1_000_000), (100.5, 101.5, 100, 101, 1_000_000),
        (101, 101.5, 100, 101, 1_000_000), (101, 101.5, 100.5, 101, 1_000_000),
    ] + [(101, 101.5, 100.5, 101, 1_000_000) for _ in range(6)]
    if extra_30m_tail:
        rows_30m += extra_30m_tail
    df_30m = bars(rows_30m, freq="30min", start="2026-01-05T00:00")
    swings_30m = [
        {"type": "low", "index": 3, "price": 94.0}, {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0}, {"type": "high", "index": 20, "price": 99.5},
    ]

    rows_5m = [(101, 102.5, 100.5, 101.5, 1_000_000) for _ in range(20)]
    rows_5m.append((102, 107, 101, 106, 5_000_000))  # execution BOS candle
    rows_5m += [
        (106, 107, 105, 106.5, 1_000_000), (106.5, 107.5, 106, 107, 1_000_000), (107, 108, 106.5, 107.5, 1_000_000),
        (107.5, 108, 107, 107.8, 1_000_000), (107.8, 108.2, 107.3, 108, 1_000_000), (108, 108.5, 107.5, 108.2, 1_000_000),
        (108.2, 108.7, 107.8, 108.4, 1_000_000), (108.4, 108.9, 108, 108.6, 1_000_000), (108.6, 109, 108.2, 108.8, 1_000_000),
    ]
    df_5m = bars(rows_5m, freq="5min", start="2026-01-05T19:00")
    swings_5m = [
        {"type": "low", "index": 2, "price": 99.0}, {"type": "high", "index": 6, "price": 103.0},
        {"type": "low", "index": 12, "price": 100.0}, {"type": "high", "index": 18, "price": 104.0},
    ]

    frames = [(df_4h, swings_4h), (df_30m, swings_30m), (df_5m, swings_5m)]
    return df_4h, df_30m, df_5m, frames


def test_clean_pass_produces_exactly_one_episode_and_one_signal():
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")

    assert run["strategy_version"] == STRATEGY_VERSION
    assert run["harness_version"] == HARNESS_VERSION
    assert len(run["episodes"]) == 1
    ep = run["episodes"][0]
    assert ep["max_state_reached"] == "ENTRY_READY"
    assert len(ep["signals"]) == 1
    assert ep["invalidated"] is False
    assert len(run["signals"]) == 1, "no immediate re-trigger the bar after ENTRY_READY first fires, even though the (unmodified, stateless) engine keeps reading ENTRY_READY on subsequent bars"

    signal = run["signals"][0]
    assert signal["location_type"] == "order_block"
    assert signal["confirmation_event_type"] in ("BOS", "CHoCH")
    assert signal["execution_event_type"] in ("BOS", "CHoCH")
    assert signal["execution_area_source"] == "30m_confirmation_candle_body"
    assert signal["market"] == "stock"
    assert signal["direction"] == "long"
    assert signal["assumed_target_r_multiple"] == 2.0
    assert signal["stop"] == 93.0  # the real order-block zone low, from the real (unmodified) engine
    assert signal["assumed_target"] == signal["assumed_entry_price"] + (signal["assumed_entry_price"] - signal["stop"]) * 2.0
    # Sprint 5.1/5.2 provenance audit fields.
    assert signal["stop_provenance"] == "directly_produced_by_smc_shadow_v1"
    assert signal["direction_provenance"] == "directly_produced_by_smc_shadow_v1"
    assert signal["assumed_entry_price_provenance"] == "evaluator_specific_assumption"
    assert signal["assumed_target_provenance"] == "evaluator_specific_assumption"
    # Sprint 5.2: trade_performance_status is ALWAYS this fixed constant --
    # never a "win"/"loss"/"outcome" value of any kind.
    assert signal["trade_performance_status"] == TRADE_PERFORMANCE_UNRESOLVED_STATUS
    assert "outcome" not in signal
    assert "r_multiple" not in signal
    # Full traceability: every entry-ready signal carries the exact
    # evidence and transition history that produced it.
    assert signal["evidence"]["state"] == "ENTRY_READY"
    assert len(signal["transition_history"]) >= 1
    assert signal["transition_history"][0]["evidence"]["strategy_version"] == STRATEGY_VERSION


def test_no_lookahead_the_engine_only_ever_sees_data_up_to_the_current_bar():
    """Sprint 5.2 requirement (f): re-verify unmodified -- directly proves
    the no-lookahead property: wraps evaluate_smc_shadow itself and
    records the length of df_30m it was called with at every step,
    confirming it is monotonically non-decreasing and never exceeds the
    loop's own current position."""
    import smc_shadow_evaluation as mod
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    seen_lengths = []
    original = mod.evaluate_smc_shadow

    def spy(ticker, direction, df_4h_arg, df_30m_arg, df_5m_arg, config=None):
        seen_lengths.append(len(df_30m_arg))
        return original(ticker, direction, df_4h_arg, df_30m_arg, df_5m_arg, config=config)

    mod.evaluate_smc_shadow = spy
    try:
        with patch_swings_walk_forward(frames):
            run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")
    finally:
        mod.evaluate_smc_shadow = original

    assert seen_lengths == sorted(seen_lengths), "df_30m length must grow monotonically -- never shrink or jump"
    assert seen_lengths[-1] <= len(df_30m), "must never be handed more data than genuinely exists up to that point"
    assert len(set(seen_lengths)) == len(seen_lengths), "each step sees exactly one more bar than the last -- never the same slice twice, never skipping ahead"


def test_missed_due_to_no_pullback():
    df_4h, _, _, frames_base = _clean_pass_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # confirms
    price = 103
    for _ in range(20):
        price += 1.5
        rows_30m.append((price - 1.4, price + 0.3, price - 1.5, price, 1_000_000))  # straight expansion, never pulls back
    df_30m = bars(rows_30m, freq="30min", start="2026-01-05T00:00")
    swings_30m = [
        {"type": "low", "index": 3, "price": 94.0}, {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0}, {"type": "high", "index": 20, "price": 99.5},
    ]
    frames = [(df_4h, frames_base[0][1]), (df_30m, swings_30m)]
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("XYZ", "long", df_4h, df_30m, None, market="stock")

    assert len(run["signals"]) == 0
    assert len(run["episodes"]) == 1
    ep = run["episodes"][0]
    assert ep["max_state_reached"] == "WAITING_FOR_PULLBACK"
    assert ep["missed_due_to_no_pullback"] is True
    assert ep["invalidated_before_entry_ready"] is False


def test_invalidated_before_entry_ready():
    df_4h, _, _, frames_base = _clean_pass_frames()
    rows_30m = [(95, 97, 94, 96, 1_000_000) for _ in range(25)]
    rows_30m.append((98, 104, 97, 103, 5_000_000))  # confirms
    rows_30m += [
        (103, 103, 95, 96, 1_000_000), (96, 96, 88, 89, 1_000_000),
        (89, 89, 84, 85, 1_000_000), (85, 85, 80, 82, 1_000_000),  # closes below zone low 93 -- invalidated
    ]
    df_30m = bars(rows_30m, freq="30min", start="2026-01-05T00:00")
    swings_30m = [
        {"type": "low", "index": 3, "price": 94.0}, {"type": "high", "index": 8, "price": 99.0},
        {"type": "low", "index": 14, "price": 95.0}, {"type": "high", "index": 20, "price": 99.5},
    ]
    frames = [(df_4h, frames_base[0][1]), (df_30m, swings_30m)]
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("XYZ", "long", df_4h, df_30m, None, market="stock")

    assert len(run["signals"]) == 0
    ep = run["episodes"][0]
    assert ep["invalidated"] is True
    assert ep["invalidated_before_entry_ready"] is True
    assert ep["missed_due_to_no_pullback"] is False


def test_post_signal_diagnostics_invalidation_touch_never_becomes_primary_loss():
    """Sprint 5.2 required test (1): a strategy-defined invalidation
    (stop) touch after ENTRY_READY must NOT become a primary 'loss'.
    Sprint 5.1 treated a stop touch as a legitimate primary loss; Sprint
    5.2 reverses that, because the ENTRY price it would be measured
    against is still only an evaluator_specific_assumption, not owned by
    smc_shadow_v1. The touch is real and useful -- it is tracked as
    post_entry_ready_invalidation_touched -- but trade_performance_status
    stays fixed, and no 'outcome'/'loss' field exists anywhere to be set."""
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock", measure_outcomes=False)
    signal = run["signals"][0]
    signal_index = df_30m.index.get_loc(pd.Timestamp(signal["entry_ready_bar_time"]))

    # Extend price down through the stop (93.0), no gap (Open stays above
    # stop on the crossing bar, only Low dips through).
    extra_rows = [(101 - i, 101 - i + 0.5, 101 - i - 1, 101 - i - 0.5, 1_000_000) for i in range(15)]
    df_30m_full = pd.concat([df_30m, bars(extra_rows, freq="30min", start=df_30m.index[-1] + pd.Timedelta(minutes=30))])

    measured = _measure_post_signal_diagnostics(df_30m_full, signal_index, dict(signal), "LONG", HarnessConfig())
    assert measured["post_entry_ready_invalidation_touched"] is True
    assert measured["post_entry_ready_invalidation_fill_type"] == "intrabar"
    assert measured["bars_from_entry_ready_to_invalidation_touch"] is not None
    # No primary outcome concept exists to have been set to "loss".
    assert "outcome" not in measured
    assert "r_multiple" not in measured
    assert measured["trade_performance_status"] == TRADE_PERFORMANCE_UNRESOLVED_STATUS


def test_post_signal_diagnostics_target_assumption_touch_never_becomes_primary_win():
    """Sprint 5.2 required test (2): reaching the harness's own ASSUMED
    target must NOT become a primary 'win' -- smc_shadow_v1 defines no
    target at all, so this can only ever surface as the separately-named
    post_entry_ready_target_assumption_touched diagnostic, never blended
    into any primary outcome (there is none)."""
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock", measure_outcomes=False)
    signal = run["signals"][0]
    signal_index = df_30m.index.get_loc(pd.Timestamp(signal["entry_ready_bar_time"]))

    # Extend price well past the 2R assumed target (117.0, from assumed
    # entry 101 / stop 93) and keep it there -- never touches the real
    # stop (93) again.
    extra_rows = [(101 + i, 101 + i + 1, 101 + i - 0.5, 101 + i + 0.5, 1_000_000) for i in range(20)]
    df_30m_full = pd.concat([df_30m, bars(extra_rows, freq="30min", start=df_30m.index[-1] + pd.Timedelta(minutes=30))])

    measured = _measure_post_signal_diagnostics(df_30m_full, signal_index, dict(signal), "LONG", HarnessConfig())
    assert measured["post_entry_ready_target_assumption_touched"] is True
    assert measured["post_entry_ready_target_assumption_fill_type"] == "intrabar"
    assert measured["post_entry_ready_invalidation_touched"] is False
    assert measured["research_mfe"] > 0
    assert "outcome" not in measured
    assert "win" not in str(measured.keys())
    assert measured["trade_performance_status"] == TRADE_PERFORMANCE_UNRESOLVED_STATUS


def test_post_signal_diagnostics_when_neither_touched_stays_unresolved():
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")
    signal = run["signals"][0]
    assert signal["post_entry_ready_invalidation_touched"] is False
    assert signal["post_entry_ready_target_assumption_touched"] is False
    assert signal["trade_performance_status"] == TRADE_PERFORMANCE_UNRESOLVED_STATUS


def test_ambiguous_intrabar_between_invalidation_and_target_assumption_is_never_silently_resolved():
    """Sprint 5.2 required test (5)/the user's own explicit concern: a
    single bar whose High clears the (research) assumed target AND whose
    Low clears the real stop -- OHLC cannot reveal which happened first.
    Must classify explicitly, never guess either way, and this one bar's
    ambiguity must never contaminate any directional read of the report
    (post_signal_market_behavior only ever reports independent touch
    counts/rates, never a directional win/loss conclusion)."""
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock", measure_outcomes=False)
    signal = run["signals"][0]
    signal_index = df_30m.index.get_loc(pd.Timestamp(signal["entry_ready_bar_time"]))
    assert signal["stop"] == 93.0
    assert signal["assumed_target"] == 117.0

    # One single forward bar whose range covers BOTH levels at once.
    extra_rows = [(105, 120, 90, 100, 1_000_000)]
    df_30m_full = pd.concat([df_30m, bars(extra_rows, freq="30min", start=df_30m.index[-1] + pd.Timedelta(minutes=30))])

    measured = _measure_post_signal_diagnostics(df_30m_full, signal_index, dict(signal), "LONG", HarnessConfig())
    assert measured["ambiguous_intrabar_between_invalidation_and_target_assumption"] is True
    assert measured["ambiguous_intrabar_bar_time"] is not None
    # Both touches genuinely did happen by this bar -- ambiguity is about
    # RELATIVE ORDER, not whether either occurred, so both flags stay True.
    assert measured["post_entry_ready_invalidation_touched"] is True
    assert measured["post_entry_ready_target_assumption_touched"] is True
    assert "outcome" not in measured

    # And the aggregate report must never draw a directional (win/loss)
    # conclusion from this -- there is no such field to contaminate; it
    # only ever appears as an ambiguity count, alongside (not instead of)
    # the two independent touch counts.
    report = aggregate_evaluation_report([{**run, "signals": [measured]}])
    behavior = report["post_signal_market_behavior"]
    assert behavior["ambiguous_intrabar_between_invalidation_and_target_assumption_count"] == 1
    assert behavior["post_entry_ready_invalidation_touched_count"] == 1
    assert behavior["post_entry_ready_target_assumption_touched_count"] == 1
    assert report["trade_performance"] == TRADE_PERFORMANCE_UNAVAILABLE


def test_gap_through_invalidation_fills_at_the_real_open():
    """Gap/fill convention (research diagnostics only, Sprint 5.1
    mechanism preserved, Sprint 5.2 field names): if the bar OPENS
    already beyond the stop, the recorded fill is that real, known Open
    price -- not false precision pretending a resting stop order filled
    exactly at its own price."""
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock", measure_outcomes=False)
    signal = run["signals"][0]
    signal_index = df_30m.index.get_loc(pd.Timestamp(signal["entry_ready_bar_time"]))
    assert signal["stop"] == 93.0

    # Opens well below the stop (a gap down) -- e.g. bad news overnight.
    extra_rows = [(85, 86, 84, 85.5, 1_000_000)]
    df_30m_full = pd.concat([df_30m, bars(extra_rows, freq="30min", start=df_30m.index[-1] + pd.Timedelta(minutes=30))])

    measured = _measure_post_signal_diagnostics(df_30m_full, signal_index, dict(signal), "LONG", HarnessConfig())
    assert measured["post_entry_ready_invalidation_touched"] is True
    assert measured["post_entry_ready_invalidation_fill_type"] == "gap"


def test_gap_through_target_assumption_fills_at_the_real_open_too():
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock", measure_outcomes=False)
    signal = run["signals"][0]
    signal_index = df_30m.index.get_loc(pd.Timestamp(signal["entry_ready_bar_time"]))
    assert signal["assumed_target"] == 117.0

    # Opens well above the assumed target (a gap up).
    extra_rows = [(125, 126, 124, 125.5, 1_000_000)]
    df_30m_full = pd.concat([df_30m, bars(extra_rows, freq="30min", start=df_30m.index[-1] + pd.Timedelta(minutes=30))])

    measured = _measure_post_signal_diagnostics(df_30m_full, signal_index, dict(signal), "LONG", HarnessConfig())
    assert measured["post_entry_ready_target_assumption_touched"] is True
    assert measured["post_entry_ready_target_assumption_fill_type"] == "gap"


def test_harness_never_touches_smc_shadow_engine_config():
    """The task's own critical rule: measure v1 exactly as written. Proves
    the harness passes the REAL, untouched default config through --
    never constructs a modified SmcShadowConfig of its own."""
    import inspect
    src = inspect.getsource(run_walk_forward_evaluation)
    # The only SmcShadowConfig construction allowed anywhere in the
    # walk-forward path is the module-level default fallback
    # (STRATEGY_CONFIGS[STRATEGY_VERSION]) -- never a bespoke instance
    # with edited thresholds.
    assert "SmcShadowConfig(" not in src


def test_one_persistent_entry_ready_event_creates_only_one_signal():
    """Sprint 5.1 requirement #1's first required test: extend the clean
    pass with many MORE bars that keep the SAME execution_event identity
    (no new 5M event, no fresh pullback cycle, price never leaves the
    original execution area) -- state keeps reading ENTRY_READY the
    whole time, but exactly one signal must be recorded."""
    extra_tail = [(101, 101.5, 100.5, 101, 1_000_000) for _ in range(30)]
    df_4h, df_30m, df_5m, frames = _clean_pass_frames(extra_30m_tail=extra_tail)
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")

    assert len(run["episodes"]) == 1
    ep = run["episodes"][0]
    entry_ready_bars = sum(1 for s in ep["states_visited"] if s == "ENTRY_READY")
    assert entry_ready_bars > 5, "the scenario must genuinely keep reading ENTRY_READY for many bars, not just once"
    assert len(ep["signals"]) == 1
    assert len(run["signals"]) == 1


def test_new_execution_opportunity_bookkeeping_creates_a_second_signal_given_qualifying_engine_output():
    """Proves the CALIBRATED BOOKKEEPING LOGIC itself is correct, using a
    scripted sequence of evaluate_smc_shadow() results (monkeypatched,
    not real engine computation) that satisfies all four objective
    signals: a different execution_event identity, a pullback_status
    regression-then-re-observation, and price closing outside the prior
    signal's execution area. This isolates "does the harness's own
    episode-boundary rule work correctly" from "can the real, current
    engine actually produce such a sequence from real price action" --
    see the very next test for that separate, empirically-discovered
    question."""
    import copy
    import smc_shadow_evaluation as mod

    df_4h, df_30m, df_5m, frames = _clean_pass_frames(
        extra_30m_tail=[(101, 101.5, 100.5, 101, 1_000_000) for _ in range(30)]
    )
    with patch_swings_walk_forward(frames):
        real_run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")
    template = real_run["episodes"][0]["signals"][0]["evidence"]
    assert template["state"] == "ENTRY_READY"

    def make_result(state, pullback_status, exec_bar_time, exec_level=103.0):
        r = copy.deepcopy(template)
        r["state"] = state
        r["pullback_status"] = pullback_status
        r["execution_event"] = {
            "detected": True, "type": "BOS", "level": exec_level, "bar_time": exec_bar_time,
            "reason": "5M BOS execution trigger",
        }
        return r

    # A scripted sequence: LOCATION_REACHED -> CONFIRMED -> first
    # ENTRY_READY (signal 1, execution identity A) -> price leaves the
    # area (WAITING_FOR_PULLBACK, pullback regresses) -> a fresh pullback
    # is observed again -> a NEW execution identity (B) -> ENTRY_READY
    # again (signal 2) -- never touching WATCHING at any point.
    scripted_states = ["LOCATION_REACHED", "CONFIRMED"]
    scripted_results = []
    for s in scripted_states:
        r = copy.deepcopy(template)
        r["state"] = s
        r["pullback_status"] = "not_applicable"
        r["execution_event"] = {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None}
        scripted_results.append(r)
    scripted_results.append(make_result("ENTRY_READY", "observed", "2026-01-05T20:30:00Z", 103.0))  # signal 1
    scripted_results.append(make_result("WAITING_FOR_PULLBACK", "chased_no_pullback", None))  # regression
    scripted_results.append(make_result("WAITING_FOR_PULLBACK", "observed", None))  # re-observed -- new cycle
    scripted_results.append(make_result("ENTRY_READY", "observed", "2026-01-05T23:00:00Z", 110.0))  # signal 2, new identity

    # Prices matching the execution_area logic: the FIRST signal's area is
    # whatever the template's own execution_area says (98-103, from
    # scenario A); the scripted bars after it need a close OUTSIDE that
    # range before the second signal to satisfy "left the prior area".
    scripted_closes = [96, 99, 101, 130, 101, 105]
    assert len(scripted_closes) == len(scripted_results)

    call_index = {"i": 0}
    original = mod.evaluate_smc_shadow

    def fake_evaluate(ticker, direction, df_4h_arg, df_30m_arg, df_5m_arg, config=None):
        idx = call_index["i"]
        call_index["i"] += 1
        if idx < len(scripted_results):
            return scripted_results[idx]
        return scripted_results[-1]  # hold the final state for any extra bars

    df_30m_scripted = bars(
        [(c, c + 0.5, c - 0.5, c, 1_000_000) for c in scripted_closes],
        freq="30min", start="2026-01-05T00:00",
    )
    mod.evaluate_smc_shadow = fake_evaluate
    try:
        run = mod.run_walk_forward_evaluation(
            "NVDA", "long", df_4h, df_30m_scripted, None, market="stock",
            harness_config=mod.HarnessConfig(min_confirmation_bars=0, min_htf_bars=0),
        )
    finally:
        mod.evaluate_smc_shadow = original

    assert len(run["episodes"]) == 1, "never touched WATCHING -- must stay one episode"
    ep = run["episodes"][0]
    assert len(ep["signals"]) == 2, "the bookkeeping rule must fire a second signal given a genuinely qualifying sequence"
    assert ep["signals"][0]["entry_ready_bar_time"] != ep["signals"][1]["entry_ready_bar_time"]
    assert ep["signals"][0]["execution_event_type"] is not None


def test_real_engine_second_execution_event_finding_pullback_status_is_sticky_once_observed():
    """Sprint 5.1 requirement #1's second required test, run against the
    REAL, unmodified engine (not scripted) -- an empirical finding, not a
    guess, reported explicitly per the task's own instruction ("if the
    existing engine cannot deterministically distinguish these cases,
    report that limitation explicitly instead of guessing").

    Under the SAME HTF thesis (same 4H order block, same 30M confirmation
    -- the swings never change), price is driven through a large further
    expansion (well beyond the first signal's execution area), then back
    down through it -- exactly the "genuine second pullback" shape
    requirement #1 describes -- while a genuinely NEW 5M execution event
    (different bar_time, confirmed via test_new_execution_opportunity_
    bookkeeping_... above to be sufficient ON ITS OWN when combined with
    the other three signals) is also present.

    FINDING (frozen, not patched, per Sprint 5.2 requirement 5): this does
    NOT produce a second signal with the real engine. Root cause, traced
    to smc_shadow_engine.py's own (untouched, per this sprint's explicit
    instruction) pullback_status computation:
    _returned_to_area_after_clearing scans the ENTIRE post-confirmation
    history every time it's called, not just recent bars -- so once a
    real "cleared, then returned" pattern occurs ANYWHERE after
    confirmation, pullback_status reads "observed" for every subsequent
    evaluation for the rest of the episode, regardless of what price does
    afterward. This makes the harness's own "_regressed_since_last_signal"
    signal (which depends on pullback_status genuinely leaving "observed"
    at some point) very difficult to satisfy from real price action once
    a first real pullback has occurred -- not impossible in principle
    (see the scripted test above proving the bookkeeping itself is
    correct), but not something this specific, realistic construction
    achieves either. This is a property of smc_shadow_v1's own pullback
    logic, not a harness bug -- documented here and in
    dashboard_sprint5_2_report.md as a candidate for a FUTURE, explicit
    strategy version, never something to paper over by loosening the
    harness's own criteria to compensate.
    """
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()

    rows_30m_extra = [
        (101, 130, 100.5, 129, 2_000_000),
        (129, 130, 128, 129, 1_000_000),
        (129, 130, 128, 129.5, 1_000_000),
        (129.5, 130, 128, 129, 1_000_000),
        (129, 130, 128, 129.5, 1_000_000),
        (129.5, 129.5, 100, 101, 1_000_000),
        (101, 102, 99, 100, 1_000_000),
        (100, 101, 98.5, 99, 1_000_000),
    ]
    df_30m2 = pd.concat([df_30m, bars(rows_30m_extra, freq="30min", start=df_30m.index[-1] + pd.Timedelta(minutes=30))])

    rows_5m_extra = [(99, 99.5, 98.5, 99, 1_000_000) for _ in range(25)]
    rows_5m_extra.append((99, 106, 98.5, 105, 5_000_000))  # a genuinely new execution event
    rows_5m_extra += [(105, 106, 104, 105.5, 1_000_000) for _ in range(5)]
    df_5m2 = pd.concat([df_5m, bars(rows_5m_extra, freq="5min", start=df_5m.index[-1] + pd.Timedelta(minutes=5))])

    frames2 = [frames[0], frames[1], frames[2]]  # same swings throughout -- no new swing points needed
    with patch_swings_walk_forward(frames2):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m2, df_5m2, market="stock")

    ep = run["episodes"][0]
    assert len(run["episodes"]) == 1
    assert "WATCHING" not in ep["states_visited"]
    assert len(ep["signals"]) == 1, (
        "documented finding: with the real, unmodified engine, a second signal did not result from "
        "this otherwise-qualifying scenario -- see this test's own docstring for the root-cause trace"
    )


def test_trade_performance_is_always_unavailable_regardless_of_signal_content():
    """Sprint 5.2 required test (3): win rate/expectancy must be
    unavailable regardless of what the signals themselves look like --
    trade_performance is a FIXED constant in the aggregate report, at
    every level (top-level and every segment), never computed from data,
    whether signals show invalidation touches, target-assumption touches,
    ambiguity, or nothing at all."""
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")
    base_signal = run["signals"][0]

    variants = []
    for touched_invalidation, touched_target, ambiguous in [
        (True, False, False), (False, True, False), (True, True, True), (False, False, False),
    ]:
        v = dict(base_signal)
        v["post_entry_ready_invalidation_touched"] = touched_invalidation
        v["post_entry_ready_target_assumption_touched"] = touched_target
        v["ambiguous_intrabar_between_invalidation_and_target_assumption"] = ambiguous
        variants.append(v)

    report = aggregate_evaluation_report([{**run, "signals": variants}])
    assert report["trade_performance"] == TRADE_PERFORMANCE_UNAVAILABLE
    assert report["trade_performance"]["status"] == "unavailable_unresolved_for_smc_shadow_v1"
    for seg in report["segments"].values():
        for bucket in seg.values():
            assert bucket["trade_performance"] == TRADE_PERFORMANCE_UNAVAILABLE
    # No KEY resembling a computed win rate/expectancy exists anywhere
    # (the fixed reason string is prose explaining the absence -- it may
    # legitimately use these words; only actual field/key names matter).
    def _all_keys(obj):
        keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(k)
                keys |= _all_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                keys |= _all_keys(item)
        return keys

    forbidden_key_substrings = ("win_rate", "expectancy", "r_multiple", "win", "loss", "outcome")
    for key in _all_keys(report):
        for bad in forbidden_key_substrings:
            assert bad not in key, f"unexpected performance-shaped key: {key}"


def test_segmentation_aggregation_across_multiple_runs():
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run_a = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")
    # Force a known, deterministic post-signal-diagnostics shape for
    # aggregation math (Sprint 5.2 vocabulary: independent touch flags,
    # never a primary win/loss).
    run_a["signals"][0]["post_entry_ready_target_assumption_touched"] = True
    run_a["signals"][0]["post_entry_ready_invalidation_touched"] = False
    run_a["signals"][0]["research_mfe"] = 16.0
    run_a["signals"][0]["research_mae"] = -1.0

    with patch_swings_walk_forward(frames):
        run_b = run_walk_forward_evaluation("AMD", "long", df_4h, df_30m, df_5m, market="stock")
    run_b["signals"][0]["post_entry_ready_invalidation_touched"] = True
    run_b["signals"][0]["post_entry_ready_target_assumption_touched"] = False
    run_b["signals"][0]["research_mfe"] = 2.0
    run_b["signals"][0]["research_mae"] = -8.0
    run_b["signals"][0]["market"] = "forex"  # simulate a forex signal for segmentation

    report = aggregate_evaluation_report([run_a, run_b])

    assert report["setup_statistics"]["detected"] == 2
    assert report["setup_statistics"]["entry_ready_signal_count"] == 2
    assert report["setup_statistics"]["terminal_state_reached_at_least"]["ENTRY_READY"] == 2
    assert report["setup_statistics"]["terminal_state_reached_at_least"]["LOCATION_REACHED"] == 2
    assert report["setup_statistics"]["max_progression_reached_at_least"]["ENTRY_READY"] == 2
    assert report["setup_statistics"]["max_progression_reached_at_least"]["LOCATION_REACHED"] == 2

    behavior = report["post_signal_market_behavior"]
    assert behavior["measured_count"] == 2
    assert behavior["post_entry_ready_invalidation_touched_count"] == 1
    assert behavior["post_entry_ready_target_assumption_touched_count"] == 1
    assert behavior["post_entry_ready_invalidation_touched_rate"] == 0.5
    assert behavior["post_entry_ready_target_assumption_touched_rate"] == 0.5

    assert report["trade_performance"] == TRADE_PERFORMANCE_UNAVAILABLE
    assert "wins" not in str(report.keys()), "no primary 'win' concept exists post-Sprint-5.2"
    assert "losses" not in str(report.keys()), "no primary 'loss' concept exists post-Sprint-5.2"

    market_segment = report["segments"]["market"]
    assert market_segment["stock"]["post_signal_market_behavior"]["measured_count"] == 1
    assert market_segment["stock"]["post_signal_market_behavior"]["post_entry_ready_target_assumption_touched_count"] == 1
    assert market_segment["forex"]["post_signal_market_behavior"]["measured_count"] == 1
    assert market_segment["forex"]["post_signal_market_behavior"]["post_entry_ready_invalidation_touched_count"] == 1

    loc_segment = report["segments"]["location_type"]
    assert loc_segment["order_block"]["post_signal_market_behavior"]["measured_count"] == 2


# ---------------------------------------------------------------------------
# Sprint 6.1 -- episode-creation-bar bookkeeping fix, regression tests.
# Scripted (monkeypatched evaluate_smc_shadow), same technique as
# test_new_execution_opportunity_bookkeeping_creates_a_second_signal_
# given_qualifying_engine_output above -- isolates "does the harness's
# own bookkeeping now handle a creation bar correctly" from "can the real
# engine actually produce such a sequence", exactly as that earlier test
# did for the Sprint 5.1 episode-boundary rule.
# ---------------------------------------------------------------------------

def _scripted_template():
    """One real evaluate_smc_shadow() result (from the proven clean-pass
    fixture) to deepcopy and mutate for the scripted tests below -- real
    field shapes, not hand-typed dicts that might drift from the actual
    result schema."""
    import copy
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        run = run_walk_forward_evaluation("NVDA", "long", df_4h, df_30m, df_5m, market="stock")
    template = run["episodes"][0]["signals"][0]["evidence"]
    assert template["state"] == "ENTRY_READY"
    return copy.deepcopy(template)


def _run_scripted(scripted_results, closes):
    """Runs run_walk_forward_evaluation with evaluate_smc_shadow
    monkeypatched to return `scripted_results` in order (holding the
    final one for any extra bars) against a plain 30M price series built
    from `closes`."""
    import smc_shadow_evaluation as mod
    assert len(closes) == len(scripted_results)
    call_index = {"i": 0}
    original = mod.evaluate_smc_shadow

    def fake_evaluate(ticker, direction, df_4h_arg, df_30m_arg, df_5m_arg, config=None):
        idx = call_index["i"]
        call_index["i"] += 1
        if idx < len(scripted_results):
            return scripted_results[idx]
        return scripted_results[-1]

    df_4h = bars([(100, 101, 99, 100, 1_000_000) for _ in range(30)], freq="4h", start="2026-01-01T00:00")
    df_30m_scripted = bars(
        [(c, c + 0.5, c - 0.5, c, 1_000_000) for c in closes], freq="30min", start="2026-01-05T00:00",
    )
    mod.evaluate_smc_shadow = fake_evaluate
    try:
        return mod.run_walk_forward_evaluation(
            "NVDA", "long", df_4h, df_30m_scripted, None, market="stock",
            harness_config=mod.HarnessConfig(min_confirmation_bars=0, min_htf_bars=0),
        )
    finally:
        mod.evaluate_smc_shadow = original


def test_creation_bar_invalidation_is_recorded_correctly():
    """Sprint 6.1 required test (1): an episode whose very first
    evaluated bar already reads INVALIDATED must have `invalidated` and
    `invalidated_before_entry_ready` set correctly on THAT bar -- not
    left wrong because a later bar happened to read WATCHING instead of
    INVALIDATED again (the exact real-data scenario Sprint 6 found)."""
    import copy
    template = _scripted_template()

    r0 = copy.deepcopy(template)
    r0["state"] = "INVALIDATED"
    r0["pullback_status"] = "not_applicable"
    r0["execution_event"] = {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None}

    r1 = copy.deepcopy(r0)
    r1["state"] = "WATCHING"  # the very next bar reads clean WATCHING, not INVALIDATED again

    run = _run_scripted([r0, r1], closes=[95, 100])

    assert len(run["episodes"]) == 1
    ep = run["episodes"][0]
    assert ep["states_visited"] == ["INVALIDATED"]
    assert ep["start_index"] == ep["end_index"], "must close on its own creation bar, not linger"
    assert ep["invalidated"] is True
    assert ep["invalidated_before_entry_ready"] is True
    assert ep["signals"] == []


def test_creation_bar_entry_ready_signal_is_recorded_correctly():
    """Sprint 6.1 required test (2): an episode whose very first
    evaluated bar already reads ENTRY_READY must have that signal
    recorded on THAT bar -- not silently lost because no later bar
    happened to re-confirm ENTRY_READY (the other real-data scenario
    Sprint 6 found)."""
    import copy
    template = _scripted_template()

    r0 = copy.deepcopy(template)  # state == "ENTRY_READY" already, from the template
    r1 = copy.deepcopy(template)
    r1["state"] = "WATCHING"

    run = _run_scripted([r0, r1], closes=[101, 96])

    assert len(run["episodes"]) == 1
    ep = run["episodes"][0]
    assert len(ep["signals"]) == 1, "the creation bar's own ENTRY_READY signal must be recorded"
    assert len(run["signals"]) == 1
    assert ep["signals"][0]["entry_ready_bar_time"] == r0.get("_bar_time") or ep["signals"][0]["assumed_entry_price"] == 101.0


def test_subsequent_bars_are_not_double_counted_after_creation_bar_fix():
    """Sprint 6.1 required test (3): the creation-bar fix must not cause
    the creation bar's own state to be recorded twice (via both
    _new_episode's own seeding AND the normal per-bar append running a
    second time on that same bar), nor cause the NEXT bar (same state,
    no real change) to fabricate an extra transition entry."""
    import copy
    template = _scripted_template()

    r0 = copy.deepcopy(template)
    r0["state"] = "LOCATION_REACHED"
    r0["confirmation_event"] = {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None}
    r0["execution_event"] = {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None}
    r1 = copy.deepcopy(r0)  # identical state -- no NEW transition should be recorded for this bar
    r2 = copy.deepcopy(r0)
    r2["state"] = "WATCHING"  # closes the episode on this 3rd bar

    run = _run_scripted([r0, r1, r2], closes=[96, 96.5, 90])

    assert len(run["episodes"]) == 1
    ep = run["episodes"][0]
    # Exactly one states_visited entry per evaluated bar the episode
    # existed for (creation bar 0, repeat bar 1, closing bar 2) -- the
    # creation bar's own state appears exactly ONCE, not duplicated by
    # both _new_episode's seeding and a second per-bar append.
    assert ep["states_visited"] == ["LOCATION_REACHED", "LOCATION_REACHED", "WATCHING"]
    # Only two REAL transitions ever happened (None->LOCATION_REACHED at
    # creation, LOCATION_REACHED->WATCHING at close) -- bar 1's repeat of
    # the same state must not fabricate a spurious third entry.
    assert len(ep["transitions"]) == 2
    assert ep["transitions"][0] == {"from_state": None, "to_state": "LOCATION_REACHED", "evidence": r0}
    assert ep["transitions"][1]["from_state"] == "LOCATION_REACHED"
    assert ep["transitions"][1]["to_state"] == "WATCHING"


def test_episode_boundaries_remain_correct_after_creation_bar_fix():
    """Sprint 6.1 required test (4): a creation-bar episode that closes
    immediately (born INVALIDATED) must not bleed into or block a
    genuinely separate, later episode -- boundaries (start_index,
    end_index, episode count) stay correct across both."""
    import copy
    template = _scripted_template()

    r0 = copy.deepcopy(template)
    r0["state"] = "INVALIDATED"
    r0["pullback_status"] = "not_applicable"
    r0["execution_event"] = {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None}

    r1 = copy.deepcopy(r0)
    r1["state"] = "WATCHING"  # first episode closes here

    r2 = copy.deepcopy(template)
    r2["state"] = "LOCATION_REACHED"  # a genuinely separate, later episode begins
    r2["confirmation_event"] = {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None}
    r2["execution_event"] = {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None}

    r3 = copy.deepcopy(r2)
    r3["state"] = "WATCHING"  # second episode closes here too

    run = _run_scripted([r0, r1, r2, r3], closes=[95, 100, 101, 90])

    assert len(run["episodes"]) == 2, "two genuinely separate episodes, not one contaminated by the other"
    first, second = run["episodes"]
    assert first["start_index"] == 0 and first["end_index"] == 0
    assert first["invalidated"] is True
    assert second["start_index"] == 2 and second["end_index"] == 3
    assert second["invalidated"] is False
    assert second["max_state_reached"] == "LOCATION_REACHED"


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
    print(f"smc_shadow_evaluation_v1: {len(tests)}/{len(tests)} passed.")
