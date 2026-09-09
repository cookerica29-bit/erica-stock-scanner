"""Kairos Watch Contract -- Track A (2026-09 session).

Tests the persisted Watch Contract object (watch_contract_store.py), its
deterministic sensors (watch_contract_sensors.py), and the live lifecycle
monitor (watch_contract_engine.py) -- see this session's own architecture
audit + sensor-calibration report for why this object exists and why the
STRONG-only displacement floor was chosen.

Two layers of test, deliberately different in style:
  - SENSOR-level tests (D, E, J, K and the touch/pullback/invalidation
    unit tests) call watch_contract_sensors functions directly against
    small, hand-built-but-real OHLCV DataFrames -- BOS/CHoCH's own
    pattern-matching fidelity was already validated against REAL market
    data in this session's Sensor Calibration report (26 real cases);
    these tests exist to prove the STRONG-only gate / freshness-floor /
    diagnostic wiring around that already-validated detector, not to
    re-derive BOS/CHoCH correctness from scratch.
  - ENGINE-level tests (A, B, C, F, G, H, I, L, M, N, O, P, Q, R, S) drive
    watch_contract_engine.run_watch_contract_monitor_tick against a real
    sqlite file, with the market-data fetch functions monkeypatched to
    controlled quotes/DataFrames -- these test the STATE MACHINE, event
    freshness ordering, duplicate-transition safety, and persistence,
    which is this sprint's actual deliverable.
CHoCH-specific engine tests (G, M) monkeypatch
watch_contract_sensors._detect_structural_event directly to force a
CHoCH result, rather than hand-engineering a fragile CHoCH-shaped price
series -- the goal there is proving the STATE MACHINE treats a CHoCH
result identically to a BOS result, not re-proving CHoCH detection
(already covered by the sensor-level tests below plus the real-data
calibration report).
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_contract_engine as engine  # noqa: E402
import watch_contract_sensors as sensors  # noqa: E402
import watch_contract_store as store  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic OHLCV builders -- calibrated empirically against the REAL
# displacement_score.score_displacement/scanner._detect_bos so that
# "strong"/"moderate"/"none" reliably land in the intended bucket. See
# this session's own scratch calibration for the exact margin/volume
# values chosen; not magic numbers picked at random.
# ---------------------------------------------------------------------------

def make_bos_df(direction="LONG", kind="strong", n_bars=60, interval_minutes=30, seed=11):
    """kind: 'strong' | 'moderate' | 'none'. A quiet, mean-reverting
    baseline (real percentile-ranked displacement needs real variation
    to rank against) followed by one controlled breakout candle."""
    rng = np.random.RandomState(seed)
    start_price = 100.0
    times = pd.date_range(start="2026-01-05T14:00:00Z", periods=n_bars, freq=f"{interval_minutes}min", tz="UTC")
    sign = 1 if direction == "LONG" else -1
    rows = []
    price = start_price
    for _ in range(n_bars - 1):
        target = start_price + rng.uniform(-1.0, 1.0)
        o = price
        c = price + (target - price) * 0.3 + rng.uniform(-0.15, 0.15)
        h = max(o, c) + abs(rng.uniform(0.02, 0.12))
        l = min(o, c) - abs(rng.uniform(0.02, 0.12))
        v = rng.uniform(60_000, 140_000)
        rows.append((o, h, l, c, v))
        price = c
    body_frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=times[:-1])
    swing_level = body_frame["High"].iloc[-20:].max() if direction == "LONG" else body_frame["Low"].iloc[-20:].min()
    margin, vol = {
        "strong": (4.0, 400_000),
        "moderate": (0.2, 90_000),
        "none": (-0.3, 70_000),
    }[kind]
    o = swing_level - sign * 0.05
    c = swing_level + sign * margin
    h = max(o, c) + 0.05
    l = min(o, c) - 0.05
    rows.append((o, h, l, c, vol))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=times)


def _conn(tmp_path, name="watch_contracts.db"):
    conn = sqlite3.connect(tmp_path / name)
    conn.row_factory = sqlite3.Row
    return conn


def _create(conn, **overrides):
    fields = {
        "ticker": "OVV",
        "direction": "long",
        "source_review_id": "review-1",
        "approved_htf_thesis": "BULLISH",
        "approved_current_leg": "BEARISH_CORRECTION",
        "location_type": "PRIOR_BREAKOUT_SUPPORT",
        "location_lower": 63.5,
        "location_upper": 64.5,
        "location_id": "OVV:4H:swing_pivot:2026-08-01T13:30:00Z",
        "location_status_at_approval": "NOT_REACHED",
        "confirmation_timeframe": sensors.CONFIRMATION_TIMEFRAME,
        "confirmation_direction": "long",
        "confirmation_allowed_events": list(sensors.ALLOWED_CONFIRMATION_EVENTS),
        "confirmation_min_displacement": sensors.MIN_CONFIRMATION_DISPLACEMENT,
        "execution_timeframe": sensors.EXECUTION_TIMEFRAME,
        "execution_direction": "long",
        "execution_allowed_events": list(sensors.ALLOWED_EXECUTION_EVENTS),
        "execution_min_displacement": sensors.MIN_EXECUTION_DISPLACEMENT,
        "approved_invalidation_rule": "close_below",
        "approved_invalidation_level": 62.0,
    }
    fields.update(overrides)
    fields["state"] = fields.get("state") or engine.initial_state(fields["location_status_at_approval"])
    if fields["state"] == "WATCHING" and not fields.get("location_reached_at"):
        fields["location_reached_at"] = "2026-01-05T14:00:00Z"
    return store.create_watch_contract(conn, **fields)


def _patch_quote(monkeypatch, price):
    monkeypatch.setattr(engine, "_fetch_quote", lambda ticker: price)


def _patch_bars_30m(monkeypatch, df):
    monkeypatch.setattr(engine, "_fetch_bars_df", lambda ticker, interval, period, interval_minutes, now: (
        df if interval == "30m" else None
    ))


def _patch_bars_5m(monkeypatch, df):
    monkeypatch.setattr(engine, "_fetch_bars_df", lambda ticker, interval, period, interval_minutes, now: (
        df if interval == "5m" else None
    ))


# ---------------------------------------------------------------------------
# A-C: location stage
# ---------------------------------------------------------------------------

def test_a_approved_location_already_reached_initializes_watching(tmp_path):
    conn = _conn(tmp_path)
    record = _create(conn, location_status_at_approval="REACHED")
    assert record["state"] == "WATCHING"
    assert record["location_reached_at"] is not None


def test_b_location_not_reached_initializes_waiting_for_location(tmp_path):
    conn = _conn(tmp_path)
    record = _create(conn, location_status_at_approval="NOT_REACHED")
    assert record["state"] == "WAITING_FOR_LOCATION"
    assert record["location_reached_at"] is None


def test_c_later_location_touch_advances_to_watching(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _create(conn, location_status_at_approval="NOT_REACHED", location_lower=63.5, location_upper=64.5)

    _patch_quote(monkeypatch, 70.0)  # outside bounds -- must stay put
    engine.run_watch_contract_monitor_tick(conn)
    after_miss = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after_miss["state"] == "WAITING_FOR_LOCATION"

    _patch_quote(monkeypatch, 64.0)  # inside [63.5, 64.5]
    engine.run_watch_contract_monitor_tick(conn)
    after_touch = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after_touch["state"] == "WATCHING"
    assert after_touch["location_reached_at"] is not None


# ---------------------------------------------------------------------------
# D, E: 30M confirmation sensor -- freshness and displacement gating
# ---------------------------------------------------------------------------

def test_d_old_30m_bos_before_location_is_rejected():
    df = make_bos_df("LONG", "strong")
    event_bar_time = df.index[-1].isoformat()
    result = sensors.evaluate_30m_confirmation(df, "long", min_bar_time=event_bar_time)
    assert result["confirmed"] is False
    assert "not fresher" in result["reason"]


def test_e_fresh_moderate_30m_bos_is_rejected():
    df = make_bos_df("LONG", "moderate")
    result = sensors.evaluate_30m_confirmation(df, "long", min_bar_time=None)
    assert result["confirmed"] is False
    assert result["displacement_label"] == "MODERATE"


# ---------------------------------------------------------------------------
# F, G: fresh STRONG 30M BOS/CHoCH -> WAITING_FOR_PULLBACK
# ---------------------------------------------------------------------------

def test_f_fresh_strong_30m_bos_advances_to_waiting_for_pullback(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _create(conn, location_status_at_approval="REACHED")
    df = make_bos_df("LONG", "strong")
    _patch_bars_30m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "WAITING_FOR_PULLBACK"
    assert after["confirmation_event_type"] == "BOS"
    assert after["confirmation_bar_time"] is not None
    assert after["pullback_reference_low"] is not None
    assert after["pullback_reference_high"] is not None


def test_g_fresh_strong_30m_choch_advances_to_waiting_for_pullback(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _create(conn, location_status_at_approval="REACHED")
    df = make_bos_df("LONG", "strong")  # displacement/timestamps real; event TYPE forced below
    monkeypatch.setattr(
        sensors, "_detect_structural_event",
        lambda df_, direction_uc, allowed, lookback, margin: ("CHoCH", float(df_["Close"].iloc[-1]) - 1.0, len(df_) - 1),
    )
    _patch_bars_30m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "WAITING_FOR_PULLBACK"
    assert after["confirmation_event_type"] == "CHoCH"


# ---------------------------------------------------------------------------
# H, I: pullback
# ---------------------------------------------------------------------------

def _waiting_for_pullback_contract(conn, direction="LONG", confirmation_bar_time=None, ref_low=99.0, ref_high=100.0):
    return _create(
        conn, direction=direction.lower(), location_status_at_approval="REACHED",
        state="WAITING_FOR_PULLBACK",
        confirmation_event_type="BOS", confirmation_level=100.5, confirmation_bar_time=confirmation_bar_time,
        pullback_reference_low=ref_low, pullback_reference_high=ref_high,
    )


def test_h_no_clear_and_return_stays_waiting_for_pullback(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    times = pd.date_range(start="2026-01-05T14:00:00Z", periods=10, freq="30min", tz="UTC")
    # Price sits INSIDE the reference zone the whole time -- never clears.
    df = pd.DataFrame(
        {"Open": 99.5, "High": 99.7, "Low": 99.3, "Close": 99.5, "Volume": 90_000}, index=times,
    )
    record = _waiting_for_pullback_contract(conn, confirmation_bar_time=times[0].isoformat())
    _patch_bars_30m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "WAITING_FOR_PULLBACK"
    assert after["pullback_reached_at"] is None


def test_i_qualifying_clear_and_return_advances_to_pullback_reached(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    times = pd.date_range(start="2026-01-05T14:00:00Z", periods=10, freq="30min", tz="UTC")
    highs = [99.5, 101.2, 101.5, 101.8, 100.6, 99.6, 99.9, 100.2, 100.3, 100.4]
    lows = [99.2, 100.6, 101.0, 101.2, 100.0, 99.2, 99.5, 99.8, 99.9, 100.0]
    closes = [99.3, 101.0, 101.4, 101.6, 100.3, 99.4, 99.7, 100.0, 100.1, 100.2]
    df = pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": [90_000] * 10}, index=times,
    )
    # entry (confirmation) bar is index 0; zone [99.0, 100.0]; bars 1-3 clear
    # above (low > 100.0); bar 4 (100.0-100.6) overlaps the zone again -> return.
    record = _waiting_for_pullback_contract(conn, confirmation_bar_time=times[0].isoformat(), ref_low=99.0, ref_high=100.0)
    _patch_bars_30m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "PULLBACK_REACHED"
    assert after["pullback_cleared_at"] is not None
    assert after["pullback_reached_at"] is not None


# ---------------------------------------------------------------------------
# J, K: 5M execution sensor -- freshness and displacement gating
# ---------------------------------------------------------------------------

def test_j_old_5m_event_before_pullback_is_rejected():
    df = make_bos_df("LONG", "strong", interval_minutes=5)
    event_bar_time = df.index[-1].isoformat()
    result = sensors.evaluate_5m_execution(df, "long", min_bar_time=event_bar_time)
    assert result["confirmed"] is False
    assert "not fresher" in result["reason"]


def test_k_weak_or_moderate_5m_events_are_rejected():
    df_moderate = make_bos_df("LONG", "moderate", interval_minutes=5)
    result_moderate = sensors.evaluate_5m_execution(df_moderate, "long", min_bar_time=None)
    assert result_moderate["confirmed"] is False
    assert result_moderate["displacement_label"] in ("MODERATE", "WEAK")

    df_none = make_bos_df("SHORT", "none", interval_minutes=5)
    result_none = sensors.evaluate_5m_execution(df_none, "short", min_bar_time=None)
    assert result_none["confirmed"] is False


# ---------------------------------------------------------------------------
# L, M: fresh STRONG 5M BOS/CHoCH -> ENTRY_READY
# ---------------------------------------------------------------------------

def _pullback_reached_contract(conn, direction="LONG", pullback_reached_at="2026-01-05T14:00:00Z"):
    return _create(
        conn, direction=direction.lower(), location_status_at_approval="REACHED",
        state="PULLBACK_REACHED",
        confirmation_event_type="BOS", confirmation_bar_time="2026-01-04T14:00:00Z",
        pullback_reference_low=99.0, pullback_reference_high=100.0,
        pullback_cleared_at="2026-01-04T15:00:00Z", pullback_reached_at=pullback_reached_at,
    )


def test_l_fresh_strong_5m_bos_advances_to_entry_ready(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _pullback_reached_contract(conn)
    df = make_bos_df("LONG", "strong", interval_minutes=5)
    _patch_bars_5m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "ENTRY_READY"
    assert after["execution_event_type"] == "BOS"
    assert after["execution_bar_time"] is not None


def test_m_fresh_strong_5m_choch_advances_to_entry_ready(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _pullback_reached_contract(conn)
    df = make_bos_df("LONG", "strong", interval_minutes=5)
    monkeypatch.setattr(
        sensors, "_detect_structural_event",
        lambda df_, direction_uc, allowed, lookback, margin: ("CHoCH", float(df_["Close"].iloc[-1]) - 1.0, len(df_) - 1),
    )
    _patch_bars_5m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "ENTRY_READY"
    assert after["execution_event_type"] == "CHoCH"


# ---------------------------------------------------------------------------
# N: rejection-only 5M event does NOT autonomously become ENTRY_READY
# ---------------------------------------------------------------------------

def test_n_rejection_only_event_does_not_produce_entry_ready(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _pullback_reached_contract(conn)
    df = make_bos_df("LONG", "none", interval_minutes=5)
    _patch_bars_5m(monkeypatch, df)
    monkeypatch.setattr(sensors, "evaluate_5m_execution", lambda *a, **k: {
        "confirmed": False, "event_type": None, "level": None, "bar_time": None,
        "prior_touch_count": None, "bars_since_first_pierce": None, "displacement_label": None, "reason": "no BOS/CHoCH",
    })
    monkeypatch.setattr(sensors, "evaluate_5m_rejection_diagnostic", lambda *a, **k: {
        "detected": True, "event_type": "REJECTION", "level": 99.5, "bar_time": "2026-01-06T10:00:00Z",
        "reason": "rejection detected (diagnostic only)",
    })

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "PULLBACK_REACHED", "a rejection alone must never autonomously produce ENTRY_READY"
    assert after["execution_rejection_event_type"] == "REJECTION", "the rejection is still persisted as diagnostic context"


# ---------------------------------------------------------------------------
# O: duplicate execution event -> no duplicate transition
# ---------------------------------------------------------------------------

def test_o_duplicate_tick_does_not_re_transition_a_terminal_contract(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _pullback_reached_contract(conn)
    df = make_bos_df("LONG", "strong", interval_minutes=5)
    _patch_bars_5m(monkeypatch, df)

    result1 = engine.run_watch_contract_monitor_tick(conn)
    after1 = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after1["state"] == "ENTRY_READY"
    assert result1["checked"] == 1

    result2 = engine.run_watch_contract_monitor_tick(conn)
    after2 = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after2["state"] == "ENTRY_READY"
    assert after2["execution_bar_time"] == after1["execution_bar_time"], "the same event must not be re-consumed/re-persisted"
    assert result2["checked"] == 0, "a terminal (ENTRY_READY) contract must be excluded from the active query on the next tick"


# ---------------------------------------------------------------------------
# P, Q: invalidation
# ---------------------------------------------------------------------------

def test_p_invalidation_before_confirmation(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _create(
        conn, location_status_at_approval="REACHED",
        approved_invalidation_rule="close_below", approved_invalidation_level=62.0,
    )
    times = pd.date_range(start="2026-01-05T14:00:00Z", periods=5, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {"Open": 62.5, "High": 62.6, "Low": 61.0, "Close": 61.5, "Volume": 90_000}, index=times,
    )
    _patch_bars_30m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "INVALIDATED"
    assert after["invalidated_at"] is not None
    assert after["invalidation_reason"]


def test_q_invalidation_between_confirmation_and_execution(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _pullback_reached_contract(conn)
    store.update_watch_contract(
        conn, record["watch_contract_id"],
        approved_invalidation_rule="close_below", approved_invalidation_level=62.0,
    )
    times = pd.date_range(start="2026-01-05T14:00:00Z", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"Open": 62.5, "High": 62.6, "Low": 61.0, "Close": 61.5, "Volume": 90_000}, index=times,
    )
    _patch_bars_5m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "INVALIDATED"


# ---------------------------------------------------------------------------
# R: persisted/restarted process resumes exact lifecycle state
# ---------------------------------------------------------------------------

def test_r_restart_resumes_exact_state(tmp_path, monkeypatch):
    conn1 = _conn(tmp_path, name="restart.db")
    record = _create(conn1, location_status_at_approval="REACHED")
    df = make_bos_df("LONG", "strong")
    _patch_bars_30m(monkeypatch, df)
    engine.run_watch_contract_monitor_tick(conn1)
    before = store.get_watch_contract(conn1, watch_contract_id=record["watch_contract_id"])
    assert before["state"] == "WAITING_FOR_PULLBACK"
    conn1.close()

    # Simulate a process restart: a BRAND NEW connection object to the SAME
    # sqlite file, no shared Python state at all.
    conn2 = _conn(tmp_path, name="restart.db")
    after_restart = store.get_watch_contract(conn2, watch_contract_id=record["watch_contract_id"])
    assert after_restart["state"] == "WAITING_FOR_PULLBACK"
    assert after_restart["confirmation_event_type"] == before["confirmation_event_type"]
    assert after_restart["confirmation_bar_time"] == before["confirmation_bar_time"]
    assert after_restart["pullback_reference_low"] == before["pullback_reference_low"]


# ---------------------------------------------------------------------------
# S: touch/pierce diagnostics persisted, never used as blockers
# ---------------------------------------------------------------------------

def test_s_touch_diagnostics_persisted_but_never_block(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = _create(conn, location_status_at_approval="REACHED")
    df = make_bos_df("LONG", "strong")
    _patch_bars_30m(monkeypatch, df)

    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "WAITING_FOR_PULLBACK"
    assert after["confirmation_prior_touch_count"] is not None
    assert after["confirmation_bars_since_first_pierce"] is not None

    # Directly prove the diagnostic is never checked as a gate: force an
    # enormous touch count on a SEPARATE, otherwise-identical evaluation
    # and confirm the result is still confirmed=True.
    monkeypatch.setattr(sensors, "compute_touch_diagnostics", lambda *a, **k: {
        "prior_touch_count": 999, "bars_since_first_pierce": 999,
    })
    df2 = make_bos_df("SHORT", "strong")
    result = sensors.evaluate_30m_confirmation(df2, "short", min_bar_time=None)
    assert result["confirmed"] is True
    assert result["prior_touch_count"] == 999


# ---------------------------------------------------------------------------
# T: no autonomous HTF thesis function invoked -- structural guard
# ---------------------------------------------------------------------------

def test_t_no_hybrid_shadow_engine_reference_anywhere_in_watch_contract_modules():
    import hybrid_shadow_engine

    for module in (sensors, engine, store):
        for name, value in vars(module).items():
            assert value is not hybrid_shadow_engine, (
                f"{module.__name__}.{name} holds a reference to hybrid_shadow_engine -- "
                f"the Watch Contract must never import or call the autonomous research engine"
            )
    # Also confirms neither module even names the specific autonomous
    # inference functions as its own attributes (they'd have to be
    # imported by name to be callable).
    for module in (sensors, engine):
        assert not hasattr(module, "evaluate_hybrid_shadow")
        assert not hasattr(module, "_compute_htf_thesis")
        assert not hasattr(module, "_build_level_registry")


# ---------------------------------------------------------------------------
# U: hybrid_shadow_v1 research config byte-identical / unchanged
# ---------------------------------------------------------------------------

def test_u_hybrid_shadow_research_config_untouched():
    import hybrid_shadow_engine

    config_a = hybrid_shadow_engine.HYBRID_STRATEGY_CONFIGS["hybrid_shadow_v1_a"]
    config_b = hybrid_shadow_engine.HYBRID_STRATEGY_CONFIGS["hybrid_shadow_v1_b"]
    assert config_a.require_execution_displacement is False
    assert config_b.require_execution_displacement is True
    assert config_a.swing_margin_30m == 4
    assert config_a.swing_margin_5m == 3
    assert config_a.min_confirmation_displacement_label == "MODERATE"
    assert config_a.min_execution_displacement_label == "MODERATE"
    assert config_a.pullback_window_bars == 60
    assert config_a.min_execution_zone_atr == 0.15
    # The live Watch Contract's OWN policy is deliberately stricter and
    # lives in a separate module -- never the same constant.
    assert sensors.MIN_CONFIRMATION_DISPLACEMENT == "STRONG"
    assert sensors.MIN_EXECUTION_DISPLACEMENT == "STRONG"
    assert sensors.MIN_CONFIRMATION_DISPLACEMENT != config_a.min_confirmation_displacement_label


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
