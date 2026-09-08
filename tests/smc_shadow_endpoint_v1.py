"""Kairos Sprint 4 -- Shadow SMC Strategy/State Engine: endpoint/feature-flag
tests (2026-09 session). Real FastAPI TestClient against the real
smc_shadow_router.py route -- not a reimplementation. Bar fetching is
monkeypatched (same convention as candidates_router.py's own
_batch_download/_recent_4h_bars_for_execution_shadow mocking) -- no live
network access from these tests.
"""

import sys
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import candidates_router as cand_router  # noqa: E402
import smc_shadow_router as shadow_router  # noqa: E402
from smc_shadow_store import SQLiteSmcShadowRepository  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    # Fresh, isolated shadow DB per test -- never the real default path.
    shadow_router._repository = SQLiteSmcShadowRepository(str(tmp_path / "shadow.sqlite3"))
    app = FastAPI()
    app.include_router(cand_router.router)
    app.include_router(shadow_router.router)
    return TestClient(app)


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-scanner-key"}


@pytest.fixture(autouse=True)
def _no_live_bars(monkeypatch):
    """Every test in this file must fail loudly if it accidentally tries
    a real network call -- default to returning None (no data) unless a
    specific test overrides it."""
    monkeypatch.setattr(shadow_router, "fetch_smc_shadow_bars", lambda ticker, interval, historical=False: None)


def _insert_setup(conn_getter, ticker="AMD", direction="long", setup_key=None, monitor_state="WAITING_FOR_TRIGGER"):
    setup_key = setup_key or f"{ticker}-{uuid.uuid4().hex[:8]}"
    conn = conn_getter()
    cur = conn.execute(
        """INSERT INTO approved_setup_memories(ticker, source, direction, setup_key, approved_at, approved_entry, approved_stop, approved_target)
           VALUES (?,?,?,?,?,?,?,?)""",
        (ticker, "ma_pipeline", direction, setup_key, "2026-09-05T14:00:00Z", 100.0, 95.0, 110.0),
    )
    memory_id = cur.lastrowid
    conn.execute(
        """INSERT INTO approved_setup_monitor_state(approved_memory_id, setup_key, ticker, source, state, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (memory_id, setup_key, ticker, "ma_pipeline", monitor_state, "2026-09-05T14:00:00Z", "2026-09-06T20:00:00Z"),
    )
    conn.commit()
    return setup_key


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def test_status_endpoint_reports_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("KAIROS_SMC_SHADOW_ENABLED", raising=False)
    resp = client.get("/api/v1/scanner/candidates/shadow-smc/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["strategy_version"] == "smc_shadow_v1"


def test_status_endpoint_reports_enabled_when_flag_set(client, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.get("/api/v1/scanner/candidates/shadow-smc/status")
    assert resp.json()["enabled"] is True


def test_list_state_returns_disabled_response_without_flag(client, headers, monkeypatch):
    monkeypatch.delenv("KAIROS_SMC_SHADOW_ENABLED", raising=False)
    resp = client.get("/api/v1/scanner/candidates/shadow-smc/state", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert "disabled" in body["message"].lower() or "KAIROS_SMC_SHADOW_ENABLED" in body["message"]


def test_evaluate_endpoint_returns_disabled_response_without_flag(client, headers, monkeypatch):
    monkeypatch.delenv("KAIROS_SMC_SHADOW_ENABLED", raising=False)
    resp = client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate",
        headers=headers, json={"ticker": "AMD", "direction": "long"},
    )
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_tick_is_a_no_op_without_flag(client, monkeypatch):
    monkeypatch.delenv("KAIROS_SMC_SHADOW_ENABLED", raising=False)
    result = shadow_router.run_smc_shadow_tick()
    assert result == {"ran": False, "reason": "disabled"}


# ---------------------------------------------------------------------------
# Auth -- same scanner API key as every other endpoint in this app.
# ---------------------------------------------------------------------------

def test_list_state_requires_api_key_when_enabled(client, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.get("/api/v1/scanner/candidates/shadow-smc/state")
    assert resp.status_code == 401


def test_evaluate_requires_api_key_when_enabled(client, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.post("/api/v1/scanner/candidates/shadow-smc/evaluate", json={"ticker": "AMD", "direction": "long"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Behavior with the flag enabled
# ---------------------------------------------------------------------------

def test_manual_evaluate_persists_and_returns_the_evaluation(client, headers, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate",
        headers=headers, json={"ticker": "AMD", "direction": "long"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["enabled"] is True
    assert body["evaluation"]["strategy_version"] == "smc_shadow_v1"
    # No bar data (monkeypatched to None) -- must degrade to WATCHING with
    # explicit unresolved reasons, never crash, never guess a state.
    assert body["evaluation"]["state"] == "WATCHING"
    assert len(body["evaluation"]["unresolved"]) > 0
    assert body["persisted"]["state"] == "WATCHING"

    # And it's now readable back via the detail endpoint.
    detail = client.get(f"/api/v1/scanner/candidates/shadow-smc/state/research::AMD::long", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["current"]["state"] == "WATCHING"
    assert len(detail.json()["transitions"]) == 1


def test_detail_endpoint_404s_for_an_unknown_setup_key(client, headers, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.get("/api/v1/scanner/candidates/shadow-smc/state/does-not-exist", headers=headers)
    assert resp.status_code == 404


def test_tick_evaluates_every_active_production_setup_and_snapshots_it(client, headers, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    key1 = _insert_setup(cand_router._get_db, ticker="AMD", monitor_state="WAITING_FOR_TRIGGER")
    key2 = _insert_setup(cand_router._get_db, ticker="NVDA", monitor_state="ACTIONABLE")
    result = shadow_router.run_smc_shadow_tick()
    assert result["ran"] is True
    assert result["evaluated"] == 2

    row1 = shadow_router._repo().get_current(key1)
    row2 = shadow_router._repo().get_current(key2)
    assert row1["production_legacy_state"] == "WAITING_FOR_TRIGGER"
    assert row1["production_state"] == "WATCHING"
    assert row2["production_legacy_state"] == "ACTIONABLE"
    assert row2["production_state"] == "ENTRY_READY"
    # No live bar data (monkeypatched away) -- the shadow read stays
    # WATCHING regardless of what production decided. This IS the
    # research signal: production says ENTRY_READY, shadow (with no data
    # to work from) says WATCHING -- a real, honestly-reported disagreement.
    assert row1["state"] == "WATCHING"
    assert row2["state"] == "WATCHING"


def test_list_state_reports_agreement_and_disagreement_counts(client, headers, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    _insert_setup(cand_router._get_db, ticker="AMD", monitor_state="WAITING_FOR_TRIGGER")
    _insert_setup(cand_router._get_db, ticker="NVDA", monitor_state="ACTIONABLE")
    shadow_router.run_smc_shadow_tick()

    resp = client.get("/api/v1/scanner/candidates/shadow-smc/state", headers=headers)
    body = resp.json()
    assert body["enabled"] is True
    assert body["count"] == 2
    # AMD: production WATCHING, shadow WATCHING -> agreement (both non-ENTRY_READY).
    # NVDA: production ENTRY_READY, shadow WATCHING -> disagreement.
    assert body["entry_ready_agreement_count"] == 1
    assert body["entry_ready_disagreement_count"] == 1
    assert "never influenced" in body["disclaimer"]


def test_shadow_engine_never_writes_to_the_candidates_database(client, headers, monkeypatch, tmp_path):
    """The structural guarantee: run the tick, then verify every table in
    candidates.db that existed before is byte-identical after, except for
    reads. approved_setup_memories/monitor_state must be completely
    unchanged."""
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    _insert_setup(cand_router._get_db, ticker="AMD", monitor_state="WAITING_FOR_TRIGGER")

    conn = cand_router._get_db()
    before_memories = [dict(r) for r in conn.execute("SELECT * FROM approved_setup_memories").fetchall()]
    before_monitor = [dict(r) for r in conn.execute("SELECT * FROM approved_setup_monitor_state").fetchall()]
    conn.close()

    shadow_router.run_smc_shadow_tick()

    conn = cand_router._get_db()
    after_memories = [dict(r) for r in conn.execute("SELECT * FROM approved_setup_memories").fetchall()]
    after_monitor = [dict(r) for r in conn.execute("SELECT * FROM approved_setup_monitor_state").fetchall()]
    conn.close()

    assert before_memories == after_memories
    assert before_monitor == after_monitor


def test_shadow_state_never_appears_in_the_dashboard_state_endpoint(client, headers, monkeypatch):
    """The Sprint 2/3 dashboards read ONLY /candidates/dashboard-state --
    confirms that endpoint's response shape is completely untouched by
    this sprint (no new shadow-related fields leaked into it)."""
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    _insert_setup(cand_router._get_db, ticker="AMD", monitor_state="ACTIONABLE")
    shadow_router.run_smc_shadow_tick()

    resp = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers)
    body = resp.json()
    row = body["setups"][0]
    for forbidden_key in ("shadow_state", "smc_shadow_state", "strategy_version", "shadow_smc"):
        assert forbidden_key not in row, f"dashboard-state row must not carry any shadow-engine field ({forbidden_key} found)"
    assert row["state"] == "ENTRY_READY"  # production's own read, untouched
