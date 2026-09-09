"""Kairos Watch Contract -- real historical end-to-end replay (2026-09
session, pre-deploy hardening pass).

Drives the ACTUAL watch_contract_store + watch_contract_engine +
watch_contract_sensors stack -- the sensor boundary is NEVER mocked here
(only the market-data FETCH function is monkeypatched, to hand the engine
a real historical DataFrame slice instead of making a live network call,
exactly the same substitution point tests/watch_contract_v1.py's own
engine-level tests already use). Every OHLCV value below comes from
research/hybrid_visual_validation_1/raw/<SYMBOL>.json -- the same real,
already-fetched 30M/5M data this session's Sensor Calibration report used,
no new fetch, no synthetic candles.

Exact bar times/levels were found by scanning this real data for a
genuinely fresh, freshness-ordered STRONG confirmation -> pullback ->
STRONG execution sequence (see this session's own search script) --
this file hardcodes the real, found values, it does not re-search.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_contract_engine as engine  # noqa: E402
import watch_contract_sensors as sensors  # noqa: E402
import watch_contract_store as store  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[1] / "research" / "hybrid_visual_validation_1" / "raw"


def load_df(symbol: str, timeframe: str, as_of: str | None = None) -> pd.DataFrame:
    data = json.loads((RAW_DIR / f"{symbol}.json").read_text())
    df = pd.DataFrame(data["ohlcv"][timeframe])
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.set_index("Datetime")
    if as_of:
        df = df.loc[: pd.Timestamp(as_of)]
    return df


def _conn(tmp_path, name="replay.db"):
    conn = sqlite3.connect(tmp_path / name)
    conn.row_factory = sqlite3.Row
    return conn


def _patch_bars_30m(monkeypatch, df):
    monkeypatch.setattr(engine, "_fetch_bars_df", lambda ticker, interval, period, interval_minutes, now: (
        df if interval == "30m" else None
    ))


def _patch_bars_5m(monkeypatch, df):
    monkeypatch.setattr(engine, "_fetch_bars_df", lambda ticker, interval, period, interval_minutes, now: (
        df if interval == "5m" else None
    ))


# ---------------------------------------------------------------------------
# Real fixture data -- DRAM (LONG) and TSM (SHORT), both genuinely fresh,
# STRONG-confirmed, real clear-and-return, real STRONG execution.
# ---------------------------------------------------------------------------

DRAM_LONG = dict(
    ticker="DRAM", direction="long",
    location_lower=59.5759994506836, location_upper=60.06399993896484,
    location_reached_at="2026-08-15T13:30:00Z",
    confirmation_bar_time="2026-08-17T13:30:00Z",
    pullback_clear_time="2026-08-17T14:00:00Z",
    pullback_return_time="2026-08-17T19:00:00Z",
    execution_bar_time="2026-09-04T18:40:00Z",
)

TSM_SHORT = dict(
    ticker="TSM", direction="short",
    location_lower=413.21949462890626, location_upper=414.52049560546874,
    location_reached_at="2026-08-22T13:30:00Z",
    confirmation_bar_time="2026-08-24T13:30:00Z",
    pullback_clear_time="2026-08-24T14:00:00Z",
    pullback_return_time="2026-08-25T13:30:00Z",
    execution_bar_time="2026-09-04T19:05:00Z",
)


def _create_watching(conn, fixture):
    return store.create_watch_contract(
        conn,
        ticker=fixture["ticker"], direction=fixture["direction"],
        approved_htf_thesis="BULLISH" if fixture["direction"] == "long" else "BEARISH",
        approved_current_leg="WITH_THESIS",
        location_type="REPLAY_FIXTURE", location_lower=fixture["location_lower"], location_upper=fixture["location_upper"],
        location_status_at_approval="REACHED", location_reached_at=fixture["location_reached_at"],
        confirmation_timeframe=sensors.CONFIRMATION_TIMEFRAME, confirmation_direction=fixture["direction"],
        confirmation_allowed_events=list(sensors.ALLOWED_CONFIRMATION_EVENTS),
        confirmation_min_displacement=sensors.MIN_CONFIRMATION_DISPLACEMENT,
        execution_timeframe=sensors.EXECUTION_TIMEFRAME, execution_direction=fixture["direction"],
        execution_allowed_events=list(sensors.ALLOWED_EXECUTION_EVENTS),
        execution_min_displacement=sensors.MIN_EXECUTION_DISPLACEMENT,
        # Generous buffer -- real price genuinely moves a lot over a
        # multi-week replay window; this must stay clear of every real
        # close this fixture's own ticks will feed, not just the ones
        # near the confirmation/pullback bars.
        approved_invalidation_rule="close_below" if fixture["direction"] == "long" else "close_above",
        approved_invalidation_level=(fixture["location_lower"] - 30) if fixture["direction"] == "long" else (fixture["location_upper"] + 30),
        state="WATCHING",
    )


@pytest.mark.parametrize("fixture", [DRAM_LONG, TSM_SHORT], ids=["LONG_DRAM", "SHORT_TSM"])
def test_real_replay_reaches_entry_ready_via_all_four_real_stages(tmp_path, monkeypatch, fixture):
    conn = _conn(tmp_path, name=f"replay_{fixture['ticker']}.db")
    record = _create_watching(conn, fixture)

    # Stage 1: WATCHING -> WAITING_FOR_PULLBACK, off the real confirmation bar.
    df30_conf = load_df(fixture["ticker"], "30m", fixture["confirmation_bar_time"])
    _patch_bars_30m(monkeypatch, df30_conf)
    engine.run_watch_contract_monitor_tick(conn)
    after_conf = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after_conf["state"] == "WAITING_FOR_PULLBACK"
    assert after_conf["confirmation_event_type"] == "BOS"
    assert pd.Timestamp(after_conf["confirmation_bar_time"]) == pd.Timestamp(fixture["confirmation_bar_time"])
    assert after_conf["confirmation_prior_touch_count"] is not None
    assert after_conf["pullback_reference_low"] == pytest.approx(fixture["location_lower"])
    assert after_conf["pullback_reference_high"] == pytest.approx(fixture["location_upper"])

    # Stage 2: WAITING_FOR_PULLBACK -> PULLBACK_REACHED, off the real clear+return.
    df30_pullback = load_df(fixture["ticker"], "30m", fixture["pullback_return_time"])
    _patch_bars_30m(monkeypatch, df30_pullback)
    engine.run_watch_contract_monitor_tick(conn)
    after_pullback = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after_pullback["state"] == "PULLBACK_REACHED"
    assert pd.Timestamp(after_pullback["pullback_cleared_at"]) == pd.Timestamp(fixture["pullback_clear_time"])
    assert pd.Timestamp(after_pullback["pullback_reached_at"]) == pd.Timestamp(fixture["pullback_return_time"])

    # Stage 3: PULLBACK_REACHED -> ENTRY_READY, off the real 5M execution.
    df5_exec = load_df(fixture["ticker"], "5m")  # execution bar sits at/near the dataset's own end
    _patch_bars_5m(monkeypatch, df5_exec)
    engine.run_watch_contract_monitor_tick(conn)
    after_exec = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after_exec["state"] == "ENTRY_READY"
    assert after_exec["execution_event_type"] == "BOS"
    assert pd.Timestamp(after_exec["execution_bar_time"]) == pd.Timestamp(fixture["execution_bar_time"])
    assert after_exec["execution_prior_touch_count"] is not None


def test_real_replay_restart_between_every_stage(tmp_path, monkeypatch):
    """Section 5's explicit requirement: destroy and reopen the DB
    connection between EVERY stage transition, proving no in-memory
    hidden state is required to resume the real replay correctly."""
    fixture = DRAM_LONG
    db_name = "restart_replay.db"

    conn = _conn(tmp_path, name=db_name)
    record = _create_watching(conn, fixture)
    conn.close()

    # Stage 1, fresh connection.
    conn = _conn(tmp_path, name=db_name)
    df30_conf = load_df(fixture["ticker"], "30m", fixture["confirmation_bar_time"])
    _patch_bars_30m(monkeypatch, df30_conf)
    engine.run_watch_contract_monitor_tick(conn)
    conn.close()

    # Confirm the state survived the "restart" with a THIRD connection,
    # before continuing to the next stage.
    conn = _conn(tmp_path, name=db_name)
    resumed = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert resumed["state"] == "WAITING_FOR_PULLBACK", "must resume at WAITING_FOR_PULLBACK with zero in-memory state"
    conn.close()

    # Stage 2, yet another fresh connection.
    conn = _conn(tmp_path, name=db_name)
    df30_pullback = load_df(fixture["ticker"], "30m", fixture["pullback_return_time"])
    _patch_bars_30m(monkeypatch, df30_pullback)
    engine.run_watch_contract_monitor_tick(conn)
    conn.close()

    conn = _conn(tmp_path, name=db_name)
    resumed2 = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert resumed2["state"] == "PULLBACK_REACHED"
    conn.close()

    # Stage 3, final fresh connection -- ENTRY_READY must only come from a
    # genuinely later 5M event (the real execution bar, well after
    # pullback_reached_at).
    conn = _conn(tmp_path, name=db_name)
    df5_exec = load_df(fixture["ticker"], "5m")
    _patch_bars_5m(monkeypatch, df5_exec)
    engine.run_watch_contract_monitor_tick(conn)
    final = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert final["state"] == "ENTRY_READY"
    assert pd.Timestamp(final["execution_bar_time"]) > pd.Timestamp(final["pullback_reached_at"])
    conn.close()


# ---------------------------------------------------------------------------
# The remaining real, non-happy-path scenarios -- same real data, no mocked
# sensors.
# ---------------------------------------------------------------------------

def test_real_moderate_confirmation_is_rejected_stays_watching(tmp_path, monkeypatch):
    conn = _conn(tmp_path)
    record = store.create_watch_contract(
        conn, ticker="LQD", direction="short",
        approved_htf_thesis="BEARISH", approved_current_leg="WITH_THESIS",
        location_type="REPLAY_FIXTURE", location_lower=104.0, location_upper=106.0,
        location_status_at_approval="REACHED", location_reached_at="2026-08-20T13:30:00Z",
        confirmation_timeframe="30m", confirmation_direction="short",
        confirmation_allowed_events=list(sensors.ALLOWED_CONFIRMATION_EVENTS),
        confirmation_min_displacement=sensors.MIN_CONFIRMATION_DISPLACEMENT,
        execution_timeframe="5m", execution_direction="short",
        execution_allowed_events=list(sensors.ALLOWED_EXECUTION_EVENTS),
        execution_min_displacement=sensors.MIN_EXECUTION_DISPLACEMENT,
        approved_invalidation_rule="close_above", approved_invalidation_level=110.0,
        state="WATCHING",
    )
    # Real LQD data, real MODERATE-confirmed CHoCH (Sensor Calibration
    # report case_002) -- must NOT advance under the STRONG-only policy.
    df30 = load_df("LQD", "30m", "2026-09-03T14:30:00Z")
    _patch_bars_30m(monkeypatch, df30)
    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "WATCHING"
    assert after["confirmation_event_type"] is None


def test_real_no_pullback_yet_remains_waiting_for_pullback(tmp_path, monkeypatch):
    fixture = DRAM_LONG
    conn = _conn(tmp_path)
    record = _create_watching(conn, fixture)

    df30_conf = load_df(fixture["ticker"], "30m", fixture["confirmation_bar_time"])
    _patch_bars_30m(monkeypatch, df30_conf)
    engine.run_watch_contract_monitor_tick(conn)
    assert store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])["state"] == "WAITING_FOR_PULLBACK"

    # Real data, but truncated BETWEEN the real clear (14:00) and the real
    # return (19:00) -- the pullback genuinely has not completed yet as of
    # this point in real history.
    df30_mid = load_df(fixture["ticker"], "30m", "2026-08-17T16:00:00Z")
    _patch_bars_30m(monkeypatch, df30_mid)
    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "WAITING_FOR_PULLBACK"
    assert after["pullback_reached_at"] is None


def test_real_weak_5m_execution_is_rejected_stays_pullback_reached(tmp_path, monkeypatch):
    # Real DRAM data, real WEAK-displacement CHoCH (Sensor Calibration
    # report case_013) -- must NOT autonomously advance.
    conn = _conn(tmp_path)
    record = store.create_watch_contract(
        conn, ticker="DRAM", direction="long",
        approved_htf_thesis="BULLISH", approved_current_leg="WITH_THESIS",
        location_type="REPLAY_FIXTURE", location_lower=57.0, location_upper=58.0,
        location_status_at_approval="REACHED", location_reached_at="2026-08-19T13:30:00Z",
        confirmation_timeframe="30m", confirmation_direction="long",
        confirmation_allowed_events=list(sensors.ALLOWED_CONFIRMATION_EVENTS),
        confirmation_min_displacement=sensors.MIN_CONFIRMATION_DISPLACEMENT,
        execution_timeframe="5m", execution_direction="long",
        execution_allowed_events=list(sensors.ALLOWED_EXECUTION_EVENTS),
        execution_min_displacement=sensors.MIN_EXECUTION_DISPLACEMENT,
        approved_invalidation_rule="close_below", approved_invalidation_level=50.0,
        state="PULLBACK_REACHED",
        confirmation_event_type="BOS", confirmation_bar_time="2026-08-20T13:30:00Z",
        pullback_reference_low=57.0, pullback_reference_high=58.0,
        pullback_cleared_at="2026-08-20T14:00:00Z", pullback_reached_at="2026-08-20T15:00:00Z",
    )
    df5 = load_df("DRAM", "5m", "2026-08-21T17:00:00Z")
    _patch_bars_5m(monkeypatch, df5)
    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "PULLBACK_REACHED"
    assert after["execution_event_type"] is None


def test_real_invalidation_fires_on_real_price_data(tmp_path, monkeypatch):
    fixture = TSM_SHORT
    conn = _conn(tmp_path)
    record = store.create_watch_contract(
        conn, ticker=fixture["ticker"], direction=fixture["direction"],
        approved_htf_thesis="BEARISH", approved_current_leg="WITH_THESIS",
        location_type="REPLAY_FIXTURE", location_lower=fixture["location_lower"], location_upper=fixture["location_upper"],
        location_status_at_approval="REACHED", location_reached_at=fixture["location_reached_at"],
        confirmation_timeframe="30m", confirmation_direction=fixture["direction"],
        confirmation_allowed_events=list(sensors.ALLOWED_CONFIRMATION_EVENTS),
        confirmation_min_displacement=sensors.MIN_CONFIRMATION_DISPLACEMENT,
        execution_timeframe="5m", execution_direction=fixture["direction"],
        execution_allowed_events=list(sensors.ALLOWED_EXECUTION_EVENTS),
        execution_min_displacement=sensors.MIN_EXECUTION_DISPLACEMENT,
        # A real, human-approved invalidation level TSM's own real 13:30
        # close (407.36) genuinely closes above.
        approved_invalidation_rule="close_above", approved_invalidation_level=406.0,
        state="WATCHING",
    )
    df30 = load_df(fixture["ticker"], "30m", "2026-08-24T13:30:00Z")
    _patch_bars_30m(monkeypatch, df30)
    engine.run_watch_contract_monitor_tick(conn)
    after = store.get_watch_contract(conn, watch_contract_id=record["watch_contract_id"])
    assert after["state"] == "INVALIDATED"
    assert after["invalidated_at"] is not None
    assert "406.0" in after["invalidation_reason"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
