"""Kairos Watch Contract -- router-level feature-flag tests (2026-09
session, pre-deploy hardening pass).

Section 1: dev-create must fail closed in production (behind its own
explicit flag, independent of normal API-key auth). Section 8: confirms
KAIROS_WATCH_CONTRACT_ENABLED (periodic monitor tick) and
KAIROS_WATCH_CONTRACT_DEV_CREATE_ENABLED (this create route) are
genuinely independent controls.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402
import watch_contract_engine as engine  # noqa: E402
import watch_contract_router as wc_router  # noqa: E402
import watch_contract_store as store  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    app = FastAPI()
    app.include_router(router.router)
    app.include_router(wc_router.router)
    return TestClient(app)


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-scanner-key"}


VALID_BODY = {
    "ticker": "OVV",
    "direction": "long",
    "approved_htf_thesis": "BULLISH",
    "approved_current_leg": "BEARISH_CORRECTION",
    "location": {"type": "PRIOR_BREAKOUT_SUPPORT", "lower": 63.5, "upper": 64.5, "status_at_approval": "REACHED"},
    "invalidation": {"rule": "close_below", "level": 62.0},
}


def test_dev_create_flag_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, raising=False)
    assert wc_router.dev_create_enabled() is False


def test_dev_create_returns_404_when_flag_disabled(client, headers, monkeypatch):
    monkeypatch.delenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, raising=False)
    resp = client.post("/api/v1/scanner/candidates/watch-contracts/dev-create", headers=headers, json=VALID_BODY)
    assert resp.status_code == 404


def test_dev_create_404_even_with_a_valid_api_key(client, monkeypatch):
    """The flag check happens BEFORE auth -- a correct API key must not
    reveal that the route exists when the flag is off."""
    monkeypatch.delenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, raising=False)
    resp = client.post(
        "/api/v1/scanner/candidates/watch-contracts/dev-create",
        headers={"X-API-Key": "test-scanner-key"}, json=VALID_BODY,
    )
    assert resp.status_code == 404


def test_dev_create_cannot_manufacture_a_contract_when_disabled(client, headers, monkeypatch, tmp_path):
    monkeypatch.delenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, raising=False)
    client.post("/api/v1/scanner/candidates/watch-contracts/dev-create", headers=headers, json=VALID_BODY)
    conn = router._get_db()
    try:
        assert store.list_watch_contracts(conn) == [], "no row must be persisted when the dev-create flag is off"
    finally:
        conn.close()


def test_dev_create_works_when_flag_enabled(client, headers, monkeypatch):
    monkeypatch.setenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, "1")
    resp = client.post("/api/v1/scanner/candidates/watch-contracts/dev-create", headers=headers, json=VALID_BODY)
    assert resp.status_code == 200, resp.text
    assert resp.json()["ticker"] == "OVV"


def test_get_list_is_unaffected_by_the_dev_create_flag(client, headers, monkeypatch):
    """Read functionality (GET) must never be gated by the create flag."""
    monkeypatch.delenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, raising=False)
    resp = client.get("/api/v1/scanner/candidates/watch-contracts", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"watch_contracts": []}


def test_monitor_tick_is_unaffected_by_the_dev_create_flag(tmp_path, monkeypatch):
    """The monitor's own tick function must not reference or be blocked
    by the dev-create flag at all -- it is a completely separate control
    (KAIROS_WATCH_CONTRACT_ENABLED gates the periodic *registration* in
    main.py, not this function itself)."""
    monkeypatch.delenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, raising=False)
    import sqlite3
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.row_factory = sqlite3.Row
    result = engine.run_watch_contract_monitor_tick(conn)
    assert result == {"checked": 0, "reason": "periodic"}


def test_watch_contract_enabled_and_dev_create_enabled_are_independent_flags(monkeypatch):
    monkeypatch.delenv(engine.WATCH_CONTRACT_ENABLED_ENV_VAR, raising=False)
    monkeypatch.setenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, "1")
    assert engine.watch_contract_enabled() is False
    assert wc_router.dev_create_enabled() is True

    monkeypatch.setenv(engine.WATCH_CONTRACT_ENABLED_ENV_VAR, "1")
    monkeypatch.delenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, raising=False)
    assert engine.watch_contract_enabled() is True
    assert wc_router.dev_create_enabled() is False


# ---------------------------------------------------------------------------
# Section 6: location-already-reached freshness-anchor semantics.
# ---------------------------------------------------------------------------

def test_explicit_reached_at_is_used_verbatim_not_overridden(client, headers, monkeypatch):
    monkeypatch.setenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, "1")
    body = dict(VALID_BODY, location=dict(VALID_BODY["location"], reached_at="2026-08-01T13:30:00Z"))
    resp = client.post("/api/v1/scanner/candidates/watch-contracts/dev-create", headers=headers, json=body)
    assert resp.status_code == 200
    assert resp.json()["location_reached_at"] == "2026-08-01T13:30:00Z"


def test_omitted_reached_at_falls_back_to_now_not_an_earlier_timestamp(client, headers, monkeypatch):
    """The safe-direction fallback (section 6): omitting reached_at on a
    REACHED contract must anchor freshness at the CREATION moment, never
    at some earlier, already-chart-visible instant -- this is what
    guarantees a structural event Erica could already see at review time
    is always correctly excluded as stale, not accidentally consumed."""
    monkeypatch.setenv(wc_router.WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR, "1")
    before = pd.Timestamp.now(tz="UTC")
    body = dict(VALID_BODY, location=dict(VALID_BODY["location"]))
    body["location"].pop("reached_at", None)
    resp = client.post("/api/v1/scanner/candidates/watch-contracts/dev-create", headers=headers, json=body)
    after = pd.Timestamp.now(tz="UTC")
    assert resp.status_code == 200
    anchor = pd.Timestamp(resp.json()["location_reached_at"])
    assert before <= anchor <= after, "the fallback anchor must be 'now' at creation time, not an earlier chart-visible instant"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
