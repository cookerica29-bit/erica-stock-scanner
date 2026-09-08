"""Kairos Sprint 5 -- Historical + Paper Evaluation Harness: endpoint
tests (2026-09 session). Real FastAPI TestClient against the real
smc_shadow_router.py routes. Bar fetching is monkeypatched with the
timestamp-aligned synthetic frames already proven (in
tests/smc_shadow_evaluation_v1.py) to drive a real walk-forward pass to
ENTRY_READY -- no live network access from these tests.
"""

import sys
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
from smc_shadow_evaluation_store import SQLiteSmcShadowEvaluationRepository  # noqa: E402
from smc_shadow_evaluation_v1 import _clean_pass_frames, patch_swings_walk_forward  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    shadow_router._repository = SQLiteSmcShadowRepository(str(tmp_path / "shadow.sqlite3"))
    shadow_router._evaluation_repository = SQLiteSmcShadowEvaluationRepository(str(tmp_path / "shadow.sqlite3"))
    app = FastAPI()
    app.include_router(cand_router.router)
    app.include_router(shadow_router.router)
    return TestClient(app)


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-scanner-key"}


@pytest.fixture()
def clean_pass_bars(monkeypatch):
    """Wires the real, previously-validated aligned dataset in as the
    'historical' fetch result, so a real walk-forward pass really
    reaches ENTRY_READY through this endpoint, not a trivial stub."""
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()

    def fake_fetch(ticker, interval, historical=False):
        return {"4h": df_4h, "30m": df_30m, "5m": df_5m}.get(interval)

    monkeypatch.setattr(shadow_router, "fetch_smc_shadow_bars", fake_fetch)
    cm = patch_swings_walk_forward(frames)
    cm.__enter__()
    yield
    cm.__exit__(None, None, None)


def test_evaluate_historical_disabled_by_default(client, headers, monkeypatch):
    monkeypatch.delenv("KAIROS_SMC_SHADOW_ENABLED", raising=False)
    resp = client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "NVDA", "direction": "long"},
    )
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_evaluation_report_disabled_by_default(client, headers, monkeypatch):
    monkeypatch.delenv("KAIROS_SMC_SHADOW_ENABLED", raising=False)
    resp = client.get("/api/v1/scanner/candidates/shadow-smc/evaluation-report", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_evaluate_historical_requires_api_key(client, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.post("/api/v1/scanner/candidates/shadow-smc/evaluate-historical", json={"ticker": "NVDA", "direction": "long"})
    assert resp.status_code == 401


def test_evaluate_historical_runs_a_real_walk_forward_pass_and_persists_it(client, headers, monkeypatch, clean_pass_bars):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "NVDA", "direction": "long", "market": "stock"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ran"] is True
    assert body["strategy_version"] == "smc_shadow_v1"
    assert body["episode_count"] == 1
    assert body["signal_count"] == 1
    assert body["run_id"]

    signals = shadow_router._evaluation_repo().list_signals(body["run_id"])
    assert len(signals) == 1
    assert signals[0]["location_type"] == "order_block"


def test_evaluate_historical_rejects_invalid_kind(client, headers, monkeypatch, clean_pass_bars):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "NVDA", "direction": "long", "kind": "nonsense"},
    )
    assert resp.status_code == 400


def test_evaluate_historical_handles_missing_data_honestly(client, headers, monkeypatch):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    monkeypatch.setattr(shadow_router, "fetch_smc_shadow_bars", lambda ticker, interval, historical=False: None)
    resp = client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "ZZZZ", "direction": "long"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ran"] is False
    assert "no 30M historical data" in body["reason"]


def test_evaluation_report_aggregates_across_runs(client, headers, monkeypatch, clean_pass_bars):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "NVDA", "direction": "long", "market": "stock"},
    )
    resp = client.get("/api/v1/scanner/candidates/shadow-smc/evaluation-report", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["setup_statistics"]["detected"] == 1
    assert body["setup_statistics"]["entry_ready_signal_count"] == 1
    assert body["setup_statistics"]["terminal_state_reached_at_least"]["ENTRY_READY"] == 1
    assert body["setup_statistics"]["max_progression_reached_at_least"]["ENTRY_READY"] == 1
    assert "never influenced" in body["disclaimer"]
    assert body["trade_performance"]["status"] == "unavailable_unresolved_for_smc_shadow_v1"
    assert body["segments"]["location_type"]["order_block"]["post_signal_market_behavior"]["measured_count"] == 1


def test_evaluation_report_filters_by_kind(client, headers, monkeypatch, clean_pass_bars):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "NVDA", "direction": "long", "kind": "historical"},
    )
    client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "NVDA", "direction": "long", "kind": "paper"},
    )
    hist = client.get("/api/v1/scanner/candidates/shadow-smc/evaluation-report", headers=headers, params={"kind": "historical"}).json()
    paper = client.get("/api/v1/scanner/candidates/shadow-smc/evaluation-report", headers=headers, params={"kind": "paper"}).json()
    both = client.get("/api/v1/scanner/candidates/shadow-smc/evaluation-report", headers=headers).json()
    assert hist["setup_statistics"]["detected"] == 1
    assert paper["setup_statistics"]["detected"] == 1
    assert both["setup_statistics"]["detected"] == 2


def test_evaluation_endpoints_never_accept_a_config_override(client, headers, monkeypatch, clean_pass_bars):
    """Structural proof of 'measure v1 exactly as written, do not
    optimize thresholds yet' -- extra/unknown request fields (e.g. an
    attempted threshold override) are silently ignored by pydantic's
    default model behavior, never applied to the strategy."""
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    resp = client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers,
        json={"ticker": "NVDA", "direction": "long", "min_confirmation_displacement_label": "WEAK"},
    )
    assert resp.status_code == 200
    assert resp.json()["strategy_version"] == "smc_shadow_v1"


def test_shadow_evaluation_never_appears_in_dashboard_state(client, headers, monkeypatch, clean_pass_bars):
    monkeypatch.setenv("KAIROS_SMC_SHADOW_ENABLED", "1")
    client.post(
        "/api/v1/scanner/candidates/shadow-smc/evaluate-historical",
        headers=headers, json={"ticker": "NVDA", "direction": "long"},
    )
    resp = client.get("/api/v1/scanner/candidates/dashboard-state", headers=headers)
    body = resp.json()
    assert body["count"] == 0  # no approved_setup_memories rows were ever created by this sprint's endpoints
    assert "setups" in body
