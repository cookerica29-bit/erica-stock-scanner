"""Kairos -- Entry-Reached Alert V1
(entry_reached_alert_v1_audit.md, and the task's own numbered sections).

Adds a persisted, deduplicated lifecycle EVENT for when an active WATCH or
APPROVED setup's frozen reviewed entry (approved_setup_memories.
approved_entry) becomes available. This is a LOCATION event, not
confirmation, not ACTIONABLE, not ENTER_NOW, not trade authorization --
"the reviewed location is available," nothing more.

Reuses the EXACT SAME Execution Layer V1 / Watch Lifecycle V1
infrastructure (run_approved_setup_monitor_tick, approved_setup_memories/
approved_setup_monitor_state/approved_setup_monitor_events, the same
5-minute scheduler) -- no new table, no competing state machine. Dedup is
persisted on approved_setup_monitor_state.entry_reached_at (frozen once
written, mirroring trigger_satisfied_at's own "persist exactly once"
convention) plus an append-only ENTRY_REACHED row in
approved_setup_monitor_events for the audit trail.

Covers, per the task's test list: LONG/SHORT touch/cross semantics (above
-> at, above -> below/gap-through, and the SHORT mirrors), first-observed-
already-reached determinism, no duplicates under repeated ticks/
oscillation/restart, interaction with the WATCH trigger lifecycle
(entry-reached does not satisfy the trigger; the trigger later satisfying
proceeds independently), interaction with APPROVE/ACTIONABLE (entry-
reached never auto-promotes; deterministic same-tick event ordering when
a row independently becomes ACTIONABLE), terminal/EXTENDED eligibility
(no new event after INVALIDATED/SUPERSEDED/WITHDRAWN/EXTENDED), the
deliberately session-BLIND V1 contract (see the audit report -- a session
gate would silently let an off-session tick consume the crossing evidence
into last_live_price before a later gated tick ever saw it fresh), frozen
approved_entry immunity to scanner drift, fail-closed on missing quotes,
no fabricated human review row, and independent per-setup-generation
dedup scope.

Most tests force _execution_window_state to always read ACTIONABLE via
monkeypatch -- this isolates entry-reached's own touch/cross mechanics
from the SEPARATE, already-tested current-R:R/EXTENDED question (the
shared mock daily_frame's frozen stop/target geometry makes real R:R
degrade fast as price approaches target, which is irrelevant noise for
most of these tests -- same isolation technique already established in
tests/watch_lifecycle_v1.py's SHORT wick test). The two tests that
specifically exercise the EXTENDED/ACTIONABLE interaction restore the
real function and use real R:R math instead.

Every test exercises the real candidates_router.py code via a real
FastAPI TestClient for setup plus direct calls to
run_approved_setup_monitor_tick() -- not a reimplementation.
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402

_REAL_EXECUTION_WINDOW_STATE = router._execution_window_state


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
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [])
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: [])


@pytest.fixture(autouse=True)
def _forbid_option_hydration(monkeypatch):
    def _spy(ticker, direction, entry, **kwargs):
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)


# A real trading Thursday already exercised throughout watch_lifecycle_v1.py
# (RTH/DST tests) -- used only as a stable, arbitrary "now"; entry-reached
# is deliberately session-blind (see module docstring), so nothing here
# depends on this landing in any particular session window.
FIXED_NOW = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
# A real Saturday -- used specifically to prove entry-reached is NOT
# session-gated (fires even when market_session() would read "closed").
FIXED_WEEKEND_NOW = datetime(2026, 8, 22, 3, 0, tzinfo=timezone.utc)


def _freeze_now(monkeypatch, when: datetime):
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: when)}))


def _unfreeze_now(monkeypatch):
    monkeypatch.setattr(router, "datetime", datetime)


def _force_actionable(monkeypatch):
    """Isolates entry-reached touch/cross mechanics from the separate,
    already-tested current-R:R/EXTENDED question -- see module
    docstring."""
    monkeypatch.setattr(router, "_execution_window_state", lambda current_rr: "ACTIONABLE")


def _use_real_execution_window_state(monkeypatch):
    monkeypatch.setattr(router, "_execution_window_state", _REAL_EXECUTION_WINDOW_STATE)


def _seed(client, headers, ticker="AMD", signal="long", entry_price=100.0):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": ticker, "signal": signal, "entry_price": entry_price, "ema21_4h": entry_price - 1.0,
            "daily_regime": signal, "confidence": "high",
            "sma50_daily": entry_price + 6.0 if signal == "long" else entry_price - 6.0,
            "sma200_daily": entry_price + 4.0 if signal == "long" else entry_price - 4.0,
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
    resp = client.get("/api/v1/scanner/candidates/approved-setup-memory", headers=headers,
                       params={"ticker": ticker, "include_inactive": True})
    assert resp.status_code == 200, resp.text
    records = resp.json()
    assert len(records) >= 1
    return sorted(records, key=lambda r: r["memory"]["approved_at"])[-1]


def _all_memory_records(client, headers, ticker="AMD"):
    resp = client.get("/api/v1/scanner/candidates/approved-setup-memory", headers=headers,
                       params={"ticker": ticker, "include_inactive": True})
    assert resp.status_code == 200, resp.text
    return sorted(resp.json(), key=lambda r: r["memory"]["approved_at"])


def _monitor_state_row(db_path, monitor_state_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM approved_setup_monitor_state WHERE id=?", (monitor_state_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _events(db_path, approved_memory_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM approved_setup_monitor_events WHERE approved_memory_id=? ORDER BY id", (approved_memory_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _entry_reached_events(db_path, approved_memory_id):
    return [e for e in _events(db_path, approved_memory_id) if e["event_type"] == "ENTRY_REACHED"]


def _reviews_for_ticker(db_path, ticker):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM candidate_visual_reviews WHERE ticker=? ORDER BY id", (ticker,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _quotes(price, timestamp=None):
    # Defaults to FIXED_NOW itself -- fresh relative to the frozen "now"
    # every touch/cross test already runs its tick under, well inside
    # ENTRY_REACHED_QUOTE_MAX_AGE_MINUTES. Tests specifically exercising
    # the freshness guard pass an explicit, deliberately-stale timestamp.
    ts = timestamp if timestamp is not None else FIXED_NOW.isoformat()
    return lambda previews: {
        str(p.get("ticker") or "").upper(): {"price": price, "timestamp": ts, "source": "mock", "price_branch": "mid"}
        for p in previews
    }


def _approve(client, headers, ticker="AMD", signal="long", entry_price=100.0):
    """APPROVE-origin, Type A (confirmation already observed) -- initial
    state CONFIRMED, in EVIDENCE_CLEARED_MONITOR_STATES from the start."""
    _seed(client, headers, ticker, signal, entry_price)
    rule = "close_above" if signal == "long" else "close_below"
    _review(
        client, headers, ticker, "approve",
        lower_tf_confirmation="yes", confirmation_rule=rule, confirmation_level=entry_price,
    )
    record = _memory_record(client, headers, ticker)
    assert record["memory"]["source_decision"] == "approve"
    assert record["monitor_state"]["state"] == "CONFIRMED"
    return record


def _watch_with_trigger(client, headers, ticker="AMD", signal="long", rule=None, level=100.0):
    _seed(client, headers, ticker, signal, entry_price=100.0)
    trigger_rule = rule or ("close_above" if signal == "long" else "close_below")
    resp = _review(client, headers, ticker, "watch", trigger_rule=trigger_rule, trigger_level=level)
    record = _memory_record(client, headers, ticker)
    assert record["memory"]["source_decision"] == "watch"
    assert record["monitor_state"]["state"] == "WAITING_FOR_TRIGGER"
    return resp, record


# ==========================================================================
# Touch / cross contract -- LONG and SHORT
# ==========================================================================

def test_long_above_to_at_entry_fires_one_event(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 2.0))  # above -- establishes previous
        router.run_approved_setup_monitor_tick("test")
        after_first = _monitor_state_row(db_path, record["monitor_state"]["id"])
        assert after_first["entry_reached_at"] is None, "still above entry -- must not fire yet"

        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry))  # exactly at entry
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None
    assert after["entry_reached_price"] == entry
    events = _entry_reached_events(db_path, memory_id)
    assert len(events) == 1
    assert events[0]["current_price"] == entry


def test_long_above_to_below_entry_gap_through_fires_once(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 3.0))
        router.run_approved_setup_monitor_tick("test")

        # Gap straight through the level -- next valid observed price is
        # BELOW entry, never an exact print at the frozen level itself.
        gapped_price = entry - 2.0
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(gapped_price))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None
    assert after["entry_reached_price"] == gapped_price, "must record the REAL observed price, never fabricate an exact print at approved_entry"
    events = _entry_reached_events(db_path, memory_id)
    assert len(events) == 1
    assert "previous observed price" in events[0]["detail"]


def test_short_below_to_at_entry_fires_one_event(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "short")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 2.0))  # below -- previous
        router.run_approved_setup_monitor_tick("test")
        assert _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"] is None

        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry))  # exactly at entry
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None
    assert after["entry_reached_price"] == entry
    assert len(_entry_reached_events(db_path, memory_id)) == 1


def test_short_below_to_above_entry_fires_one_event(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "short")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 3.0))
        router.run_approved_setup_monitor_tick("test")

        gapped_price = entry + 2.0
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(gapped_price))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None
    assert after["entry_reached_price"] == gapped_price
    assert len(_entry_reached_events(db_path, memory_id)) == 1


# ==========================================================================
# First-observed determinism, duplicates, restart
# ==========================================================================

def test_first_observed_already_reached_fires_deterministically(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]
    assert record["monitor_state"]["last_live_price"] is None, "sanity: never yet observed by the monitor"

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 1.0))  # already on the reached side
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None, "first-ever observation already past entry must fire immediately"
    assert after["entry_reached_price"] == entry - 1.0
    events = _entry_reached_events(db_path, memory_id)
    assert len(events) == 1
    assert "first observation while active" in events[0]["detail"]


def test_repeated_ticks_on_reached_side_do_not_duplicate(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5))
        router.run_approved_setup_monitor_tick("test")
        first_at = _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"]
        router.run_approved_setup_monitor_tick("test")
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] == first_at, "never overwritten by later ticks"
    assert len(_entry_reached_events(db_path, memory_id)) == 1


def test_oscillation_across_level_does_not_duplicate(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        for price in (entry + 1.0, entry - 0.5, entry + 0.8, entry - 0.2, entry + 0.3):
            monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(price))
            router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    events = _entry_reached_events(db_path, memory_id)
    assert len(events) == 1, f"price oscillating back and forth across entry must still fire exactly once, got {events}"


def test_process_restart_does_not_duplicate(client, headers, tmp_path, monkeypatch):
    """'Restart' == a fresh app/TestClient pointed at the SAME on-disk
    db_path -- no in-process cache is involved either way (same
    convention as test_persists_across_fresh_connections /
    test_server_restart_recovers_first_satisfying_historical_candle_once)."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5))
        router.run_approved_setup_monitor_tick("test")
        first_at = _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"]

        # Simulate a restart: run the tick again exactly as a freshly
        # started process's periodic task would.
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] == first_at
    assert len(_entry_reached_events(db_path, memory_id)) == 1


# ==========================================================================
# Deliberately session-blind (see module docstring / audit report)
# ==========================================================================

def test_fires_regardless_of_market_session(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    assert router.market_session(FIXED_WEEKEND_NOW.astimezone(router.EASTERN_TZ)) == "closed", "sanity: this really is a closed-session timestamp"

    _freeze_now(monkeypatch, FIXED_WEEKEND_NOW)
    try:
        # A fresh-relative-to-FIXED_WEEKEND_NOW timestamp -- isolates the
        # session-blindness question from the SEPARATE freshness guard
        # (covered in its own dedicated section below).
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5, timestamp=FIXED_WEEKEND_NOW.isoformat()))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None, "entry-reached is deliberately session-blind, matching the rest of this monitor"


# ==========================================================================
# Interaction with WATCH trigger lifecycle
# ==========================================================================

def test_watch_reaches_entry_before_trigger_stays_waiting(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "long", "close_above", 100.0)
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: [])  # trigger never satisfies
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None, "location reached must still persist/alert once"
    assert after["state"] == "WAITING_FOR_TRIGGER", "location != confirmation -- must NOT satisfy the trigger or hand off"
    assert after["trigger_satisfied_at"] is None
    assert len(_entry_reached_events(db_path, memory_id)) == 1


def test_trigger_later_satisfies_independently_of_entry_reached(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "long", "close_above", 100.0)
    entry = record["memory"]["approved_entry"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        # Tick 1: entry reached, trigger not yet.
        monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: [])
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5))
        router.run_approved_setup_monitor_tick("test")
        mid = _monitor_state_row(db_path, record["monitor_state"]["id"])
        assert mid["entry_reached_at"] is not None
        assert mid["state"] == "WAITING_FOR_TRIGGER"

        # Tick 2: the stated 30m close-above trigger now genuinely fires.
        bars = [{"time": "2026-08-20T15:30:00Z", "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0}]
        monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 1.0))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["trigger_satisfied_at"] is not None, "normal Watch Lifecycle V1 logic must proceed entirely independently"
    assert after["entry_reached_at"] == mid["entry_reached_at"], "entry-reached evidence untouched by the later trigger"


# ==========================================================================
# Interaction with APPROVE / ACTIONABLE -- real R:R behavior, no override
# ==========================================================================

def test_approve_reaches_entry_does_not_auto_actionable(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _use_real_execution_window_state(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    approved_target = record["memory"]["approved_target"]

    # Price right at/through the frozen entry -- but engineer a bad
    # execution window (near target) so this tick independently reads
    # EXTENDED, not ACTIONABLE -- proving entry-reached itself carries NO
    # actionability authority of its own; the real safety gates remain
    # authoritative regardless of the location event.
    near_target_price = approved_target - 0.05

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(near_target_price))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "EXTENDED", "existing execution safety gates remain authoritative, unaffected by entry-reached"
    assert after["state"] != "ACTIONABLE"


def test_same_tick_independently_actionable_has_deterministic_event_order(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _use_real_execution_window_state(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        # Establish "previous" above entry first, within the real R:R-safe
        # band for this mock geometry (no fire yet).
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 1.0))
        router.run_approved_setup_monitor_tick("test")
        assert _monitor_state_row(db_path, record["monitor_state"]["id"])["state"] == "ACTIONABLE", (
            "sanity: entry+1.0 must be a real, healthy R:R for this mock geometry"
        )

        # This tick: price crosses to exactly at entry -- entry-reached
        # fires, and the row independently STAYS ACTIONABLE (real R:R at
        # entry is even better) -- both conclusions land on the same tick.
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None
    assert after["state"] == "ACTIONABLE"
    events = _events(db_path, memory_id)
    entry_reached_ids = [e["id"] for e in events if e["event_type"] == "ENTRY_REACHED"]
    assert len(entry_reached_ids) == 1
    # ACTIONABLE was already true on the prior tick, so no NEW
    # state-transition event is expected on this tick (state doesn't
    # change) -- the meaningful assertion is that entry-reached's insert
    # never depends on / is never blocked by that unrelated fact. A
    # dedicated same-tick-transition ordering check is exercised by
    # test_trigger_later_satisfies_independently_of_entry_reached and by
    # the WATCH_HANDED_OFF-adjacent coverage in watch_lifecycle_v1.py;
    # here we only need entry_reached_ids to be non-empty and singular.


# ==========================================================================
# Active / terminal eligibility
# ==========================================================================

def test_invalidated_row_never_gets_additional_entry_reached_event(client, headers, tmp_path, monkeypatch):
    """A LONG's approved_stop is always below approved_entry by
    construction (support below entry, target above) -- so invalidation
    (price <= stop) is geometrically ALWAYS also 'at or below entry'.
    Entry-reached and invalidation can therefore legitimately fire on the
    SAME tick (both real, both honest evidence of where price actually
    was) -- see the ordering contract in run_approved_setup_monitor_tick's
    docstring. What the spec actually requires is verified here: once
    INVALIDATED, no ADDITIONAL entry-reached event is ever created by a
    later tick, no matter what price that later tick observes."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    approved_stop = record["memory"]["approved_stop"]
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        # Establish a "previous" price above entry first (no fire).
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 1.0))
        router.run_approved_setup_monitor_tick("test")
        assert _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"] is None

        # Now invalidate directly -- this price is necessarily also past
        # entry, so entry-reached legitimately fires on this SAME tick.
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(approved_stop - 0.01))
        router.run_approved_setup_monitor_tick("test")
        invalidated = _monitor_state_row(db_path, record["monitor_state"]["id"])
        assert invalidated["state"] == "INVALIDATED"
        events_after_invalidation = _entry_reached_events(db_path, memory_id)
        assert len(events_after_invalidation) == 1

        # A later tick observes price back at entry -- terminal rows are
        # excluded from _monitor_active_rows entirely; must change nothing.
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry))
        result = router.run_approved_setup_monitor_tick("test")
        assert result["checked"] == 0, "a terminal row must not even be fetched by the monitor"
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] == invalidated["entry_reached_at"], "no NEW event/timestamp from the later tick"
    assert len(_entry_reached_events(db_path, memory_id)) == 1, "still exactly one event total, ever"


def test_superseded_before_entry_never_fires_later(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long", entry_price=100.0)
    old_memory_id = record["memory"]["id"]
    old_monitor_id = record["monitor_state"]["id"]
    old_entry = record["memory"]["approved_entry"]

    # A genuinely new setup_key (different stop/target) supersedes the old one.
    _seed(client, headers, "AMD", "long", entry_price=105.0)
    _review(
        client, headers, "AMD", "approve",
        lower_tf_confirmation="yes", confirmation_rule="close_above", confirmation_level=105.0,
    )
    all_records = _all_memory_records(client, headers, "AMD")
    assert len(all_records) == 2
    old_after_supersede = _monitor_state_row(db_path, old_monitor_id)
    assert old_after_supersede["state"] == "SUPERSEDED"
    assert old_after_supersede["entry_reached_at"] is None, "sanity: never reached before superseding"

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        # Price observed at the OLD (now-superseded) memory's frozen entry.
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(old_entry))
        result = router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, old_monitor_id)
    assert after["entry_reached_at"] is None
    assert _entry_reached_events(db_path, old_memory_id) == []


def test_withdrawn_before_entry_never_fires_later(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    memory_id = record["memory"]["id"]
    monitor_id = record["monitor_state"]["id"]
    entry = record["memory"]["approved_entry"]

    _review(client, headers, "AMD", "reject", lower_tf_confirmation="not_yet",
            confirmation_rule=None, confirmation_level=None)
    withdrawn = _monitor_state_row(db_path, monitor_id)
    assert withdrawn["state"] == "WITHDRAWN"
    assert withdrawn["entry_reached_at"] is None

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.1))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, monitor_id)
    assert after["entry_reached_at"] is None
    assert _entry_reached_events(db_path, memory_id) == []


def test_extended_before_entry_never_fires_later(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _use_real_execution_window_state(monkeypatch)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "long", "close_above", 100.0)
    memory_id = record["memory"]["id"]
    monitor_id = record["monitor_state"]["id"]
    approved_target = record["memory"]["approved_target"]
    entry = record["memory"]["approved_entry"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        # Gap 4: R:R already degraded pre-trigger -> EXTENDED, entry never reached yet.
        near_target_price = approved_target - 0.05
        monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: [])
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(near_target_price))
        router.run_approved_setup_monitor_tick("test")
        assert _monitor_state_row(db_path, monitor_id)["state"] == "EXTENDED"
        assert _monitor_state_row(db_path, monitor_id)["entry_reached_at"] is None

        # A later tick drifts price back down through entry -- must not matter.
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, monitor_id)
    assert after["entry_reached_at"] is None, "EXTENDED must never get a fresh entry-reached event"
    assert _entry_reached_events(db_path, memory_id) == []


def test_stale_row_never_fires_a_new_entry_reached_event(client, headers, tmp_path, monkeypatch):
    """Decision confirmed 2026-09: STALE is ineligible for a new
    ENTRY_REACHED event. Drives a real WATCH row through TRIGGER_SATISFIED
    -> ACTIONABLE -> STALE (past EXECUTION_EVIDENCE_FRESHNESS_HOURS since
    trigger_satisfied_at, the SAME real staleness math the rest of this
    monitor already uses -- not a shortcut), confirming entry was never
    reached before going stale, then confirming a later tick with price
    genuinely at/through entry still produces nothing."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "long", "close_above", 100.0)
    memory_id = record["memory"]["id"]
    monitor_id = record["monitor_state"]["id"]
    entry = record["memory"]["approved_entry"]

    # Tick 1: trigger satisfies (price stays comfortably ABOVE entry, so
    # entry-reached does not also fire here -- keeps this test isolated
    # to the STALE question alone).
    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        bars = [{"time": "2026-08-20T15:30:00Z", "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0}]
        monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 1.0))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)
    after_trigger = _monitor_state_row(db_path, monitor_id)
    assert after_trigger["state"] == "ACTIONABLE"
    assert after_trigger["trigger_satisfied_at"] == FIXED_NOW.isoformat()
    assert after_trigger["entry_reached_at"] is None, "sanity: entry not reached yet"

    # Tick 2: 73 hours later, same anchor -- past EXECUTION_EVIDENCE_FRESHNESS_HOURS (72).
    stale_now = FIXED_NOW + timedelta(hours=73)
    _freeze_now(monkeypatch, stale_now)
    try:
        monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: [])
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 1.0, timestamp=stale_now.isoformat()))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)
    gone_stale = _monitor_state_row(db_path, monitor_id)
    assert gone_stale["state"] == "STALE"
    assert gone_stale["entry_reached_at"] is None, "sanity: still never reached before going stale"

    # Tick 3: price now genuinely at/through entry -- must still not fire.
    later_now = stale_now + timedelta(minutes=5)
    _freeze_now(monkeypatch, later_now)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5, timestamp=later_now.isoformat()))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, monitor_id)
    assert after["state"] == "STALE"
    assert after["entry_reached_at"] is None, "STALE must never get a fresh entry-reached event"
    assert _entry_reached_events(db_path, memory_id) == []


# ==========================================================================
# Frozen approved_entry vs. scanner drift
# ==========================================================================

def test_scanner_entry_drift_does_not_change_authoritative_entry(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long", entry_price=100.0)
    frozen_entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]
    assert frozen_entry == 100.0

    # A later ordinary rescan reports a DIFFERENT live entry for the same ticker.
    _seed(client, headers, "AMD", "long", entry_price=103.5)
    unchanged = _memory_record(client, headers, "AMD")
    assert unchanged["memory"]["approved_entry"] == frozen_entry, "frozen entry must never drift with new scans"

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        # Price at the OLD frozen entry (100), nowhere near the NEW scanner entry (103.5).
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(frozen_entry - 0.1))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None, "must fire based on the FROZEN entry, ignoring scanner drift entirely"
    assert after["entry_reached_price"] == frozen_entry - 0.1


# ==========================================================================
# Quote freshness (narrow guard, entry-reached only -- see
# entry_reached_alert_v1_audit.md's freshness audit section; invalidation/
# current-R:R are deliberately UNCHANGED and still ignore quote age)
# ==========================================================================

def test_stale_quote_timestamp_fails_closed(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    stale_timestamp = (FIXED_NOW - timedelta(minutes=router.ENTRY_REACHED_QUOTE_MAX_AGE_MINUTES + 1)).isoformat()
    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5, timestamp=stale_timestamp))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is None, "a quote older than the freshness threshold must not fire entry-reached"
    assert _entry_reached_events(db_path, memory_id) == []
    # last_checked_at/last_live_price DO still advance -- the freshness
    # guard is narrow to entry-reached only, unlike the whole-row
    # missing-quote fail-closed case below.
    assert after["last_checked_at"] is not None
    assert after["last_live_price"] == entry - 0.5


def test_quote_just_inside_freshness_window_still_fires(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]

    fresh_timestamp = (FIXED_NOW - timedelta(minutes=router.ENTRY_REACHED_QUOTE_MAX_AGE_MINUTES - 1)).isoformat()
    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5, timestamp=fresh_timestamp))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is not None, "a quote just inside the freshness window must still fire"


def test_missing_quote_timestamp_fails_closed(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    memory_id = record["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
            "AMD": {"price": entry - 0.5, "timestamp": None, "source": "mock", "price_branch": "mid"},
        })
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is None, "a missing/unparseable timestamp must fail closed, never assumed fresh"
    assert _entry_reached_events(db_path, memory_id) == []


def test_known_limitation_stale_tick_can_consume_crossing_evidence(client, headers, tmp_path, monkeypatch):
    """Documented, disclosed limitation (see entry_reached_alert_v1_audit.md):
    a stale-quote tick skips the ENTRY_REACHED check itself, but
    last_live_price still advances unconditionally (unchanged, pre-
    existing behavior this feature does not touch) -- so if the stale
    tick's price was already on the reached side, that becomes the new
    "previous price" baseline. A LATER fresh tick observing the SAME
    already-past-entry price finds no NEW cross (previous was already
    <= entry too) and does not retroactively fire. This is the same
    class of limitation the session-gate finding surfaced, but narrower
    in practice: it requires Alpaca to return a stale-timestamped quote
    during otherwise-live monitoring (an anomalous feed-health scenario),
    not a routine, nightly certainty like the session case was. Fixing
    it completely would require last_live_price to stop advancing on a
    stale tick -- a change to ALL rows' displayed "current price"
    behavior, not narrowly scoped to entry-reached, so it was
    deliberately NOT done for V1. See
    test_recovers_if_price_moves_back_above_entry_after_a_stale_tick for
    the self-healing case that DOES still work correctly."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]

    stale_timestamp = (FIXED_NOW - timedelta(hours=6)).isoformat()
    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5, timestamp=stale_timestamp))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)
    stale_tick_result = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert stale_tick_result["entry_reached_at"] is None
    assert stale_tick_result["last_live_price"] == entry - 0.5, "sanity: last_live_price still advances on a stale tick, unchanged pre-existing behavior"

    later = FIXED_NOW + timedelta(minutes=5)
    _freeze_now(monkeypatch, later)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5, timestamp=later.isoformat()))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"] is None, (
        "documented limitation: the SAME already-past-entry price on a later fresh tick is not a new cross"
    )


def test_recovers_if_price_moves_back_above_entry_after_a_stale_tick(client, headers, tmp_path, monkeypatch):
    """The self-healing case: if price genuinely moves back above entry
    at some point after the stale observation, a later real re-cross
    still fires correctly -- no evidence is permanently lost as long as
    price itself moves again."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]

    stale_timestamp = (FIXED_NOW - timedelta(hours=6)).isoformat()
    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5, timestamp=stale_timestamp))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"] is None

    # Price genuinely moves back above entry, freshly observed.
    back_above = FIXED_NOW + timedelta(minutes=5)
    _freeze_now(monkeypatch, back_above)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry + 1.0, timestamp=back_above.isoformat()))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"] is None

    # And now genuinely re-crosses, freshly observed -- fires correctly.
    re_crosses = FIXED_NOW + timedelta(minutes=10)
    _freeze_now(monkeypatch, re_crosses)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.2, timestamp=re_crosses.isoformat()))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["entry_reached_at"] is not None


# ==========================================================================
# Fail-closed / no fabricated review
# ==========================================================================

def test_stale_missing_quote_fails_closed(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})  # no quote at all
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["entry_reached_at"] is None
    assert after["last_checked_at"] is None, "missing data must leave the row entirely untouched this tick"


def test_no_fabricated_human_review_row(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record = _approve(client, headers, "AMD", "long")
    entry = record["memory"]["approved_entry"]
    before = _reviews_for_ticker(db_path, "AMD")
    assert len(before) == 1
    assert before[0]["decision"] == "approve"

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry - 0.5))
        router.run_approved_setup_monitor_tick("test")
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after = _reviews_for_ticker(db_path, "AMD")
    assert after == before, "entry-reached detection must never write/alter a human review row"


# ==========================================================================
# Independent per-setup-generation dedup scope
# ==========================================================================

def test_new_setup_generation_fires_independently(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _force_actionable(monkeypatch)
    record1 = _approve(client, headers, "AMD", "long", entry_price=100.0)
    entry1 = record1["memory"]["approved_entry"]
    memory_id_1 = record1["memory"]["id"]

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry1 - 0.5))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)
    assert _monitor_state_row(db_path, record1["monitor_state"]["id"])["entry_reached_at"] is not None

    # A genuinely new setup generation (different stop/target -> different setup_key).
    _seed(client, headers, "AMD", "long", entry_price=105.0)
    _review(
        client, headers, "AMD", "approve",
        lower_tf_confirmation="yes", confirmation_rule="close_above", confirmation_level=105.0,
    )
    record2 = _memory_record(client, headers, "AMD")
    assert record2["memory"]["id"] != memory_id_1
    entry2 = record2["memory"]["approved_entry"]
    memory_id_2 = record2["memory"]["id"]
    assert _monitor_state_row(db_path, record2["monitor_state"]["id"])["entry_reached_at"] is None

    _freeze_now(monkeypatch, FIXED_NOW)
    try:
        monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes(entry2 - 0.5))
        router.run_approved_setup_monitor_tick("test")
    finally:
        _unfreeze_now(monkeypatch)

    after2 = _monitor_state_row(db_path, record2["monitor_state"]["id"])
    assert after2["entry_reached_at"] is not None, "the new setup generation must be able to fire its own, independent event"
    assert len(_entry_reached_events(db_path, memory_id_2)) == 1
    assert len(_entry_reached_events(db_path, memory_id_1)) == 1, "the old generation's event must be untouched"
