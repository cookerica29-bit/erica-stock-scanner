"""Kairos Dashboard Sprint 1 -- Strategy State Read Model (2026-09 session).
Endpoint-level tests for GET /api/v1/scanner/candidates/dashboard-state,
exercised via a real FastAPI TestClient against the real candidates_router.py
route -- not a reimplementation.

Setup rows are inserted directly into approved_setup_memories /
approved_setup_monitor_state (rather than driving the full
scan -> approve -> monitor-tick flow) so each test can pin an exact
monitor_state permutation without depending on run_approved_setup_monitor_tick's
own timing/network behavior -- that flow already has its own dedicated
coverage in tests/approved_setup_monitor_v1.py, tests/watch_lifecycle_v1.py,
and tests/entry_reached_alert_v1.py. This file is only about the NEW
read-model layer on top of whatever state those rows already carry.
"""

import json
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402
import journal_store  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.db"))
    # The dashboard-state journal overlay lazily builds and caches a
    # SQLiteJournalRepository singleton (see _dashboard_journal_repository
    # in candidates_router.py) -- reset it so each test's own
    # JOURNAL_DB_PATH is actually the one used, not a prior test's.
    router._dashboard_journal_repository_singleton = None
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-scanner-key"}


def _db_path():
    return router.default_candidates_db_path()


def _insert_setup(
    ticker="AMD",
    direction="long",
    setup_key=None,
    monitor_state="WAITING_FOR_TRIGGER",
    entry_reached_at=None,
    entry_reached_price=None,
    approved_entry=100.0,
    approved_stop=95.0,
    approved_target=110.0,
    source="ma_pipeline",
    invalidation_reason=None,
    current_stop=None,
    breakeven_set_at=None,
    partial_profit_suggested_at=None,
):
    setup_key = setup_key or f"{ticker}-{uuid.uuid4().hex[:8]}"
    # Route through the real _get_db() (rather than a bare sqlite3.connect)
    # so the real _initialize_candidates_schema runs first -- these tables
    # are created lazily on first use, exactly as in production.
    conn = router._get_db()
    try:
        cur = conn.execute(
            """
            INSERT INTO approved_setup_memories(
                ticker, source, direction, setup_key, approved_at,
                approved_entry, approved_stop, approved_target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, source, direction, setup_key, "2026-08-20T14:00:00Z", approved_entry, approved_stop, approved_target),
        )
        memory_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO approved_setup_monitor_state(
                approved_memory_id, setup_key, ticker, source, state,
                created_at, updated_at, entry_reached_at, entry_reached_price,
                invalidation_reason, current_stop, breakeven_set_at, partial_profit_suggested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (memory_id, setup_key, ticker, source, monitor_state, "2026-08-20T14:00:00Z", "2026-08-20T15:00:00Z", entry_reached_at, entry_reached_price, invalidation_reason, current_stop, breakeven_set_at, partial_profit_suggested_at),
        )
        conn.commit()
    finally:
        conn.close()
    return memory_id, setup_key


def test_requires_api_key(client):
    resp = client.get("/api/v1/scanner/candidates/dashboard-state")
    assert resp.status_code == 401


def test_empty_when_no_setups(client, headers):
    resp = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 0
    assert body["setups"] == []
    assert body["mechanism"] == "dashboard_state_read_model_v1"
    # 11 states as of the TARGET_HIT addition (candidates_router.py's
    # _is_target_hit, same session) -- see dashboard_state_v1.py's own
    # test_dashboard_states_is_the_exact_eleven_state_schema_requested.
    assert set(body["supported_states"]) == {
        "DISCOVERED", "WATCHING", "LOCATION_REACHED", "CONFIRMED",
        "WAITING_FOR_PULLBACK", "EXECUTION_READY", "ENTRY_READY",
        "INVALIDATED", "TARGET_HIT", "POSITION_OPEN", "CLOSED",
    }


def test_waiting_for_trigger_without_entry_reached_is_watching(client, headers):
    _insert_setup(ticker="AMD", monitor_state="WAITING_FOR_TRIGGER", entry_reached_at=None)
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    assert body["count"] == 1
    row = body["setups"][0]
    assert row["symbol"] == "AMD"
    assert row["state"] == "WATCHING"
    assert row["state_label"] == "Watching"
    assert row["legacy_state"] == "WAITING_FOR_TRIGGER"
    assert row["market"] == "stock"
    assert row["planned_entry"] == 100.0
    assert row["stop"] == 95.0
    assert row["target"] == 110.0
    assert row["source"] == "ma_pipeline"


def test_waiting_for_trigger_with_entry_reached_is_location_reached(client, headers):
    _insert_setup(ticker="MSFT", monitor_state="WAITING_FOR_TRIGGER", entry_reached_at="2026-08-20T16:00:00Z", entry_reached_price=100.5)
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "MSFT")
    assert row["state"] == "LOCATION_REACHED"
    assert row["entry_reached_at"] == "2026-08-20T16:00:00Z"
    assert row["entry_reached_price"] == 100.5


def test_actionable_is_entry_ready(client, headers):
    _insert_setup(ticker="NVDA", monitor_state="ACTIONABLE")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "NVDA")
    assert row["state"] == "ENTRY_READY"
    assert row["state_label"] == "Entry Ready"


def test_invalidated_is_included_not_hidden(client, headers):
    # The key architectural finding this sprint disclosed: ACTIVE_MONITOR_STATES
    # / _monitor_active_rows treat INVALIDATED as TERMINAL and exclude it --
    # the dashboard endpoint must NOT reuse that helper, or this test fails.
    _insert_setup(ticker="TSLA", monitor_state="INVALIDATED", invalidation_reason="stop_hit")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next((r for r in body["setups"] if r["symbol"] == "TSLA"), None)
    assert row is not None, "INVALIDATED setup was hidden from the dashboard-state endpoint"
    assert row["state"] == "INVALIDATED"
    assert row["invalidation_reason"] == "stop_hit"


# ---------------------------------------------------------------------------
# Trade-Management Automation (2026-09 session): breakeven/current_stop/
# partial-profit surfaced on the dashboard row, not folded into `state`
# (independent flags, not a single-value enum like TARGET_HIT).
# ---------------------------------------------------------------------------

def test_no_breakeven_or_partial_by_default(client, headers):
    _insert_setup(ticker="AMD", monitor_state="ACTIONABLE", entry_reached_at="2026-08-20T15:00:00Z")
    row = next(r for r in client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()["setups"] if r["symbol"] == "AMD")
    assert row["breakeven_set"] is False
    assert row["partial_profit_suggested"] is False
    assert row["stop"] == 95.0, "with no current_stop set, Stop/Exit falls back to the original approved_stop"
    assert row["original_stop"] == 95.0


def test_breakeven_set_surfaces_the_new_effective_stop(client, headers):
    _insert_setup(
        ticker="AMD", monitor_state="ACTIONABLE", entry_reached_at="2026-08-20T15:00:00Z",
        approved_entry=100.0, approved_stop=95.0, current_stop=100.0, breakeven_set_at="2026-08-21T10:00:00Z",
    )
    row = next(r for r in client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()["setups"] if r["symbol"] == "AMD")
    assert row["breakeven_set"] is True
    assert row["breakeven_set_at"] == "2026-08-21T10:00:00Z"
    assert row["stop"] == 100.0, "Stop/Exit must reflect the CURRENT (breakeven) stop, not the stale original"
    assert row["exit"] == 100.0
    assert row["original_stop"] == 95.0, "the original approved_stop stays available separately, not lost"


def test_partial_profit_suggested_is_independent_of_breakeven(client, headers):
    # A setup can have a pending partial-profit suggestion with or without
    # breakeven having fired -- these are independent flags, tested here
    # with ONLY partial set, to prove neither implies the other.
    _insert_setup(
        ticker="OXY", monitor_state="ACTIONABLE", entry_reached_at="2026-08-20T15:00:00Z",
        partial_profit_suggested_at="2026-08-22T09:00:00Z",
    )
    row = next(r for r in client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()["setups"] if r["symbol"] == "OXY")
    assert row["partial_profit_suggested"] is True
    assert row["partial_profit_suggested_at"] == "2026-08-22T09:00:00Z"
    assert row["breakeven_set"] is False, "partial-profit alone must not imply breakeven"


def test_withdrawn_is_excluded(client, headers):
    _insert_setup(ticker="AAPL", monitor_state="WITHDRAWN")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    assert all(r["symbol"] != "AAPL" for r in body["setups"])


def test_superseded_is_excluded(client, headers):
    _insert_setup(ticker="GOOG", monitor_state="SUPERSEDED")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    assert all(r["symbol"] != "GOOG" for r in body["setups"])


def test_stale_is_included_with_null_state_and_preserved_legacy_state(client, headers):
    _insert_setup(ticker="AMZN", monitor_state="STALE")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "AMZN")
    assert row["state"] is None
    assert row["state_label"] is None
    assert row["legacy_state"] == "STALE"


def test_extended_is_waiting_for_pullback(client, headers):
    _insert_setup(ticker="META", monitor_state="EXTENDED")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "META")
    assert row["state"] == "WAITING_FOR_PULLBACK"


def test_market_field_reflects_source_stock_vs_forex(client, headers):
    # Dashboard Sprint 2: market is derived from the real .source column
    # (dashboard_state.market_for_source), not hardcoded to "stock".
    _insert_setup(ticker="AMD", monitor_state="WAITING_FOR_TRIGGER", source="ma_pipeline")
    _insert_setup(ticker="EURUSD", monitor_state="WAITING_FOR_TRIGGER", source="smc_forex")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    stock_row = next(r for r in body["setups"] if r["symbol"] == "AMD")
    forex_row = next(r for r in body["setups"] if r["symbol"] == "EURUSD")
    assert stock_row["market"] == "stock"
    assert forex_row["market"] == "forex"


def test_trade_type_and_option_fields_from_a_real_stored_preview(client, headers):
    # Dashboard Sprint 3: candidate_plan_previews is a REAL, existing table
    # (written by the existing review-queue preview flow) -- this proves
    # the endpoint reads a genuine stored row, not a mock.
    _insert_setup(ticker="AMD", direction="long", monitor_state="ACTIONABLE", source="ma_pipeline")
    conn = router._get_db()
    conn.execute(
        """
        INSERT INTO candidate_plan_previews(ticker, source, signal, computed_at, option_contract_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("AMD", "ma_pipeline", "long", "2026-09-06T14:00:00Z", json.dumps({
            "available": True, "type": "CALL", "strike": 105.0, "expiry": "2026-09-19",
        })),
    )
    conn.commit()
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "AMD")
    assert row["trade_type"] == "CALL"
    assert row["option_strike"] == 105.0
    assert row["option_expiration"] == "2026-09-19"


def test_trade_type_falls_back_to_direction_with_no_stored_preview(client, headers):
    _insert_setup(ticker="MSFT", direction="short", monitor_state="WAITING_FOR_TRIGGER")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "MSFT")
    assert row["trade_type"] == "SHORT"
    assert row["option_strike"] is None
    assert row["option_expiration"] is None


def test_trade_type_falls_back_when_stored_preview_has_no_available_contract(client, headers):
    _insert_setup(ticker="NVDA", direction="long", monitor_state="ACTIONABLE", source="ma_pipeline")
    conn = router._get_db()
    conn.execute(
        """
        INSERT INTO candidate_plan_previews(ticker, source, signal, computed_at, option_contract_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("NVDA", "ma_pipeline", "long", "2026-09-06T14:00:00Z", json.dumps({
            "available": False, "execution": "No Options Chain",
        })),
    )
    conn.commit()
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "NVDA")
    assert row["trade_type"] == "LONG"
    assert row["option_strike"] is None
    assert row["option_expiration"] is None


def test_multiple_setups_all_returned(client, headers):
    _insert_setup(ticker="AMD", monitor_state="WAITING_FOR_TRIGGER")
    _insert_setup(ticker="MSFT", monitor_state="ACTIONABLE")
    _insert_setup(ticker="NVDA", monitor_state="CONFIRMED")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    assert body["count"] == 3
    assert {r["symbol"] for r in body["setups"]} == {"AMD", "MSFT", "NVDA"}


# ---------------------------------------------------------------------------
# Journal position overlay -- real SQLiteJournalRepository against a
# scratch DB, not a mock.
# ---------------------------------------------------------------------------

def _open_journal_entry(**overrides):
    base = {
        "journal_id": f"j-{uuid.uuid4().hex[:8]}",
        "position_id": f"p-{uuid.uuid4().hex[:8]}",
        "ticker": "AMD",
        "direction": "LONG",
        "result": "Open",
        "outcome": "Open",
        "tracking_status": "active",
        "tracking_started_at": "2026-08-20T14:00:00Z",
        "planned_underlying_entry": 100.0,
        "original_stop": 95.0,
        "original_tp1": 110.0,
        "scanner_timeframe": "4H",
        "setup_grade": "A",
    }
    base.update(overrides)
    return base


def test_open_journal_entry_overrides_setup_lifecycle_state(client, headers):
    _insert_setup(ticker="AMD", monitor_state="WAITING_FOR_TRIGGER")
    # Use the exact path the endpoint itself will resolve via
    # default_journal_db_path() (JOURNAL_DB_PATH env var set by the client
    # fixture), not a path we invent ourselves.
    journal = journal_store.SQLiteJournalRepository(journal_store.default_journal_db_path())
    journal.create_entry(_open_journal_entry(ticker="AMD"))

    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "AMD")
    assert row["state"] == "POSITION_OPEN"
    assert row["state_label"] == "Position Open"
    # legacy_state still reflects the real underlying monitor state --
    # the overlay changes the presented `state`, it does not erase history.
    assert row["legacy_state"] == "WAITING_FOR_TRIGGER"


def test_closed_journal_entry_overrides_setup_lifecycle_state(client, headers):
    _insert_setup(ticker="MSFT", monitor_state="ACTIONABLE")
    journal = journal_store.SQLiteJournalRepository(journal_store.default_journal_db_path())
    journal.create_entry(_open_journal_entry(
        ticker="MSFT", result="Win", outcome="Win", tracking_status="completed",
        tracking_completed_at="2026-08-21T14:00:00Z",
    ))

    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "MSFT")
    assert row["state"] == "CLOSED"


def test_no_journal_entry_leaves_setup_lifecycle_state_untouched(client, headers):
    _insert_setup(ticker="NVDA", monitor_state="ACTIONABLE")
    body = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers).json()
    row = next(r for r in body["setups"] if r["symbol"] == "NVDA")
    assert row["state"] == "ENTRY_READY"


# ---------------------------------------------------------------------------
# Requirement #10: preserve all current API behavior -- prove the new
# endpoint's existence doesn't perturb the pre-existing review-queue /
# ranked endpoints' own contract (a light smoke check; the exhaustive
# review-queue tests live in tests/review_queue_v1.py and
# tests/review_queue_asset_type_v1.py, untouched by this sprint).
# ---------------------------------------------------------------------------

def test_review_queue_endpoint_still_responds_normally(client, headers):
    resp = client.get("/api/v1/scanner/candidates/review-queue", headers=headers)
    assert resp.status_code == 200, resp.text


def test_ranked_endpoint_still_responds_normally(client, headers):
    resp = client.get("/api/v1/scanner/candidates/ranked", headers=headers)
    assert resp.status_code == 200, resp.text
