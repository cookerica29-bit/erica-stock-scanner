"""Execution Layer V1 -- server-side monitor
(run_approved_setup_monitor_tick and its helpers in candidates_router.py).
See execution_layer_v1_design_audit.md (sections 5-13) and
execution_layer_v1_implementation_plan.md (sections 5-9) for the full
design this implements.

Scope discipline exercised throughout: no candidate_promotions writes, no
ENTER_NOW, no option hydration, never touches the scanner universe, never
revives a terminal memory. Every test exercises the real
candidates_router.py code via a real FastAPI TestClient for setup (the
same visual-review flow a human would use) plus a direct call to
run_approved_setup_monitor_tick() -- not a reimplementation.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


def _daily_frame():
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1
        rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000})
    rows[10]["Low"] = 90.0
    rows[10]["Close"] = 91.0
    rows[10]["Open"] = 92.0
    rows[20]["High"] = 110.0
    rows[20]["Close"] = 109.0
    rows[20]["Open"] = 108.0
    return pd.DataFrame(rows, index=index)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-scanner-key"}


@pytest.fixture(autouse=True)
def _mock_network(monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(tickers[0]).upper(): _daily_frame()})
    monkeypatch.setattr(router, "_latest_quote_for_ticker", lambda ticker: {
        "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
    })
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(p.get("ticker") or "").upper(): {
            "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
        }
        for p in previews
    })
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [
        {
            "time": f"2026-08-19T{hour:02d}:00:00Z",
            "open": 99.0 + (idx * 0.05), "high": 101.0 + (idx * 0.05),
            "low": 98.0 + (idx * 0.05), "close": 100.0 + (idx * 0.05), "volume": 1000,
        }
        for idx, hour in enumerate(range(11))
    ] + [
        {"time": "2026-08-20T02:00:00Z", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000},
        {"time": "2026-08-20T06:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1100},
        {"time": "2026-08-20T10:00:00Z", "open": 100.5, "high": 102.0, "low": 99.2, "close": 101.0, "volume": 1200},
        {"time": "2026-08-20T14:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.2, "volume": 1000},
    ])


@pytest.fixture(autouse=True)
def _forbid_option_hydration(monkeypatch):
    def _spy(ticker, direction, entry, **kwargs):
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)


def _seed(client, headers, ticker="AMD"):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": ticker, "signal": "long", "entry_price": 100.0, "ema21_4h": 99.0,
            "daily_regime": "bullish", "confidence": "high",
            "sma50_daily": 106.0, "sma200_daily": 104.0,
        }],
    }
    created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert created.status_code == 200


def _review(client, headers, ticker, decision, source="ma_pipeline", **overrides):
    body = {
        "source": source,
        "market_structure": "bullish",
        "location_read": "good",
        "clear_path_to_target": "yes",
        "lower_tf_confirmation": "not_yet",
        "decision": decision,
        "note": "test note",
        **overrides,
    }
    resp = client.post(f"/api/v1/scanner/candidates/{ticker}/visual-review", headers=headers, json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _memory_record(client, headers, ticker="AMD"):
    resp = client.get("/api/v1/scanner/candidates/approved-setup-memory", headers=headers, params={"ticker": ticker})
    assert resp.status_code == 200, resp.text
    records = resp.json()
    assert len(records) == 1
    return records[0]


def _monitor_state_row(db_path, monitor_state_id):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM approved_setup_monitor_state WHERE id=?", (monitor_state_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _events(db_path, approved_memory_id):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM approved_setup_monitor_events WHERE approved_memory_id=? ORDER BY id", (approved_memory_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _backdate(db_path, table, row_id, **fields):
    import sqlite3
    conn = sqlite3.connect(db_path)
    set_clause = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE {table} SET {set_clause} WHERE id=?", (*fields.values(), row_id))
    conn.commit()
    conn.close()


# ==========================================================================
# Scope discipline: only real approved memories are ever touched
# ==========================================================================

def test_no_active_memories_is_a_cheap_noop(client, headers):
    result = router.run_approved_setup_monitor_tick("test")
    assert result == {"checked": 0, "updated": 0, "reason": "test"}


def test_terminal_memory_never_touched(client, headers, tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    db_path = str(tmp_path / "candidates.db")
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet",
            trigger_rule="close_above", trigger_level=999.0)
    record = _memory_record(client, headers, "AMD")
    monitor_state_id = record["monitor_state"]["id"]
    before = _monitor_state_row(db_path, monitor_state_id)

    # Manually mark it INVALIDATED (simulating a prior tick) -- terminal.
    _backdate(db_path, "approved_setup_monitor_state", monitor_state_id, state="INVALIDATED", terminal_at="2026-08-20T00:00:00Z")

    result = router.run_approved_setup_monitor_tick("test")
    assert result["checked"] == 0, "a terminal-state row must not even be fetched by the monitor"

    after = _monitor_state_row(db_path, monitor_state_id)
    assert after["state"] == "INVALIDATED"
    assert after["last_checked_at"] is None, "a terminal row must never be updated by a later tick"


# ==========================================================================
# Invalidation -- intrabar, frozen approved_stop, symmetric
# ==========================================================================

def test_long_invalidated_when_price_at_or_below_approved_stop(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")
    approved_stop = record["memory"]["approved_stop"]
    memory_id = record["memory"]["id"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": approved_stop - 0.01, "timestamp": "2026-08-20T18:30:00Z",
                "source": "mock_latest_quote", "price_branch": "mid"},
    })
    result = router.run_approved_setup_monitor_tick("test")
    assert result["checked"] == 1
    assert result["updated"] == 1

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "INVALIDATED"
    assert after["terminal_at"] is not None
    assert "approved stop" in after["invalidation_reason"]

    events = _events(db_path, memory_id)
    assert len(events) == 1
    assert events[0]["to_state"] == "INVALIDATED"
    assert events[0]["from_state"] == "CONFIRMED"


def test_short_invalidated_when_price_at_or_above_approved_stop():
    assert router._is_invalidated("short", 105.0, 105.0) is True
    assert router._is_invalidated("short", 104.99, 105.0) is False
    assert router._is_invalidated("long", 95.0, 95.0) is True
    assert router._is_invalidated("long", 95.01, 95.0) is False


def test_invalidation_uses_frozen_stop_not_live_recomputed_stop(client, headers, tmp_path, monkeypatch):
    """The exact "Finding A" bug this feature exists to fix, at the
    engine level -- a live/recomputed stop must never override the frozen
    approved_stop for the invalidation decision."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")
    approved_stop = record["memory"]["approved_stop"]

    # Price is comfortably ABOVE the frozen approved_stop -- must NOT invalidate,
    # regardless of what a live-recomputed stop might say elsewhere in the app.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": approved_stop + 5.0, "timestamp": "2026-08-20T18:30:00Z",
                "source": "mock_latest_quote", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] != "INVALIDATED"


# ==========================================================================
# Type A (CONFIRMED) -- current-price R:R gates ACTIONABLE/EXTENDED
# ==========================================================================

def test_confirmed_memory_becomes_actionable_when_rr_holds(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")
    assert record["monitor_state"]["state"] == "CONFIRMED"
    entry = record["memory"]["approved_entry"]

    # Price near entry -- R:R should still comfortably clear RR_WARNING_THRESHOLD.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "2026-08-20T18:30:00Z",
                "source": "mock_latest_quote", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "ACTIONABLE"
    assert after["current_rr_at_last_check"] is not None
    assert after["current_rr_at_last_check"] >= router.RR_WARNING_THRESHOLD


def test_confirmed_memory_becomes_extended_when_rr_degrades(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")
    target = record["memory"]["approved_target"]

    # Price already very close to target -- reward shrinks, risk grows,
    # current R:R must fall below RR_WARNING_THRESHOLD.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": target - 0.05, "timestamp": "2026-08-20T18:30:00Z",
                "source": "mock_latest_quote", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "EXTENDED"


def test_actionable_can_flip_back_to_extended_as_price_moves(client, headers, tmp_path, monkeypatch):
    """The live window verdict is NOT sticky -- state is UPDATEd freely
    between ACTIONABLE/EXTENDED as price moves, unlike the one-way
    APPROVED->CONFIRMED->... evidence-gate progression."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")
    entry = record["memory"]["approved_entry"]
    target = record["memory"]["approved_target"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["state"] == "ACTIONABLE"

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": target - 0.05, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["state"] == "EXTENDED"


def test_legacy_memory_with_no_evidence_never_becomes_actionable(client, headers, tmp_path, monkeypatch):
    """A memory with neither trigger_rule nor confirmation_rule (legacy/
    incomplete evidence, e.g. FFIV/NVDA/CLH's real backfilled rows) must
    stay APPROVED forever, regardless of price -- never silently promoted
    to ACTIONABLE from price alone."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet")  # no trigger given
    record = _memory_record(client, headers, "AMD")
    assert record["monitor_state"]["state"] == "APPROVED"
    entry = record["memory"]["approved_entry"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "APPROVED", "no evidence anchor -- must never reach ACTIONABLE from price alone"


# ==========================================================================
# Type B (WAITING_FOR_TRIGGER -> TRIGGER_SATISFIED) -- completed 30m
# candles only, persisted exactly once
# ==========================================================================

def _bars(*closes_with_times):
    return [{"time": t, "open": c, "high": c + 0.5, "low": c - 0.5, "close": c} for t, c in closes_with_times]


def test_trigger_not_satisfied_by_a_forming_bar_during_market_hours(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet",
            trigger_rule="close_above", trigger_level=100.0)
    record = _memory_record(client, headers, "AMD")
    assert record["monitor_state"]["state"] == "WAITING_FOR_TRIGGER"

    import datetime as dt
    now = dt.datetime(2026, 8, 20, 15, 15, tzinfo=dt.timezone.utc)  # 11:15am ET Thursday -- regular session
    # Latest bar (15:00 UTC = 11:00am ET) is STILL FORMING at 11:15am ET
    # (a 30m bar starting 11:00 covers 11:00-11:30) -- even though its
    # close already qualifies, it must NOT count while still forming.
    forming_bars = _bars(("2026-08-20T14:30:00Z", 99.0), ("2026-08-20T15:00:00Z", 105.0))
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: forming_bars)
    monkeypatch.setattr(router, "datetime", type("_dt", (dt.datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", dt.datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "WAITING_FOR_TRIGGER", "a still-forming bar's close must never satisfy a trigger"
    assert after["trigger_satisfied_at"] is None


def test_trigger_satisfied_by_a_completed_bar(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet",
            trigger_rule="close_above", trigger_level=100.0)
    record = _memory_record(client, headers, "AMD")
    memory_id = record["memory"]["id"]

    import datetime as dt
    now = dt.datetime(2026, 8, 20, 16, 5, tzinfo=dt.timezone.utc)  # 12:05pm ET -- the 15:30 bar has closed
    completed_bars = _bars(("2026-08-20T14:30:00Z", 99.0), ("2026-08-20T15:00:00Z", 99.5), ("2026-08-20T15:30:00Z", 101.25))
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: completed_bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 101.25, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (dt.datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", dt.datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] in ("ACTIONABLE", "EXTENDED"), \
        "trigger satisfied -> immediately evaluated for the execution window in the SAME tick"
    assert after["trigger_satisfied_at"] is not None
    assert after["trigger_satisfied_bar_time"] == "2026-08-20T15:30:00Z"
    assert after["trigger_satisfied_price"] == 101.25

    events = _events(db_path, memory_id)
    to_states = [e["to_state"] for e in events]
    assert "TRIGGER_SATISFIED" not in to_states or to_states[-1] in ("ACTIONABLE", "EXTENDED"), \
        "the logged event history must reflect real transitions, ending at the current state"


def test_trigger_satisfaction_persisted_exactly_once_across_ticks(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet",
            trigger_rule="close_above", trigger_level=100.0)
    record = _memory_record(client, headers, "AMD")

    import datetime as dt
    now = dt.datetime(2026, 8, 20, 16, 5, tzinfo=dt.timezone.utc)
    completed_bars = _bars(("2026-08-20T14:30:00Z", 99.0), ("2026-08-20T15:00:00Z", 99.5), ("2026-08-20T15:30:00Z", 101.25))
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: completed_bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 101.25, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (dt.datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
        first = _monitor_state_row(db_path, record["monitor_state"]["id"])
        assert first["trigger_satisfied_at"] is not None
        first_satisfied_at = first["trigger_satisfied_at"]

        # A second tick, same bars, same everything -- must NOT re-fire /
        # overwrite the original satisfaction timestamp.
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", dt.datetime)

    second = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert second["trigger_satisfied_at"] == first_satisfied_at, \
        "trigger satisfaction must be persisted exactly once, never overwritten by a later tick"


def test_extended_before_trigger_can_skip_actionable_in_one_tick(client, headers, tmp_path, monkeypatch):
    """A gap that jumps price past both the trigger AND the execution
    window in one move -- must go straight to EXTENDED, never briefly
    read ACTIONABLE."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet",
            trigger_rule="close_above", trigger_level=100.0)
    record = _memory_record(client, headers, "AMD")
    target = record["memory"]["approved_target"]

    import datetime as dt
    now = dt.datetime(2026, 8, 20, 16, 5, tzinfo=dt.timezone.utc)
    gapped_bars = _bars(("2026-08-20T14:30:00Z", 99.0), ("2026-08-20T15:00:00Z", 99.5), ("2026-08-20T15:30:00Z", target - 0.05))
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: gapped_bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": target - 0.05, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (dt.datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", dt.datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "EXTENDED"
    assert after["trigger_satisfied_at"] is not None, "the trigger fact is still real and still recorded"


# ==========================================================================
# Staleness -- a real anchor that is too old must decay, never silently
# read as fresh forever
# ==========================================================================

def test_confirmed_memory_goes_stale_after_freshness_window_elapses(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0,
            confirmed_candle_time="2020-01-01T00:00:00Z")  # deliberately ancient
    record = _memory_record(client, headers, "AMD")
    entry = record["memory"]["approved_entry"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "STALE"


def test_stale_memory_does_not_auto_revive_to_actionable(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0,
            confirmed_candle_time="2020-01-01T00:00:00Z")
    record = _memory_record(client, headers, "AMD")
    entry = record["memory"]["approved_entry"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["state"] == "STALE"

    # A second tick, same fresh price, same everything -- STALE must not
    # flip back to ACTIONABLE on its own.
    router.run_approved_setup_monitor_tick("test")
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["state"] == "STALE"


def test_no_confirmed_candle_time_falls_back_to_approved_at_for_freshness(client, headers, tmp_path, monkeypatch):
    """confirmed_candle_time is optional -- when absent, approved_at (the
    review's own timestamp) is the best available anchor, never a
    fabricated one."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)  # no confirmed_candle_time
    record = _memory_record(client, headers, "AMD")
    entry = record["memory"]["approved_entry"]

    # approved_at was set moments ago (this test's own runtime) -- must
    # read fresh, not stale.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "ACTIONABLE"


# ==========================================================================
# Missing/failed data -- never guess, never force a wrong verdict
# ==========================================================================

def test_missing_quote_leaves_row_untouched(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})  # no quote at all
    result = router.run_approved_setup_monitor_tick("test")
    assert result["checked"] == 1
    assert result["updated"] == 0

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "CONFIRMED", "must stay exactly where it was -- never guess from missing data"
    assert after["last_checked_at"] is None


# ==========================================================================
# Alert-event dedup -- only meaningful transitions logged, never per-tick
# ==========================================================================

def test_event_log_does_not_duplicate_the_same_transition(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")
    memory_id = record["memory"]["id"]
    entry = record["memory"]["approved_entry"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    router.run_approved_setup_monitor_tick("test")
    router.run_approved_setup_monitor_tick("test")

    events = _events(db_path, memory_id)
    assert len(events) == 1, "three ticks producing the SAME ACTIONABLE verdict must log exactly one event, not three"
    assert events[0]["to_state"] == "ACTIONABLE"


# ==========================================================================
# Scope: never touches candidate_promotions, never hydrates options
# ==========================================================================

def test_no_candidate_promotions_writes_or_option_hydration(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes",
            confirmation_rule="close_above", confirmation_level=100.0)
    record = _memory_record(client, headers, "AMD")
    entry = record["memory"]["approved_entry"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": entry, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")

    import sqlite3
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM candidate_promotions").fetchone()[0]
    conn.close()
    assert count == 0
    # No AssertionError from the _forbid_option_hydration spy means this passed too.
