"""PATCH /candidate-promotions/{id}/taken -- the tri-state "taken" flag.

Scope: the endpoint itself and the tri-state (None/True/False) semantics
through storage and the API response. The outcome resolver that reads
taken=1 promotions is covered separately in
tests/candidate_promotion_outcome_watcher_v1.py.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


@pytest.fixture()
def client(router):
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


def _promotion_payload(ticker="AAPL", promoted_at="2026-08-20T14:00:00Z"):
    return {
        "ticker": ticker, "source": "ma_pipeline", "direction": "long", "entry_price": 100.0,
        "stop": 95.0, "target": 110.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": promoted_at, "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 110.0, "raw_risk_reward": 2.0, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 95.0, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": {"body_percentile": 60.0}, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable", "bos_confirmed": False, "bos_details": None,
    }


def _seed_promotion(router, **kwargs) -> int:
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    new_id = router._store_promotion(conn, _promotion_payload(**kwargs))
    conn.commit()
    conn.close()
    return new_id


def test_new_promotion_defaults_taken_to_none_not_false(router, client):
    promotion_id = _seed_promotion(router)

    rows = client.get("/api/v1/scanner/candidate-promotions", headers={"X-API-Key": "test-scanner-key"}).json()
    row = next(r for r in rows if r["id"] == promotion_id)

    assert row["taken"] is None, "undecided must stay None, not collapse to False"
    assert row["taken_at"] is None


def test_marking_taken_true_persists_and_stamps_taken_at(router, client):
    promotion_id = _seed_promotion(router)

    resp = client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["taken"] is True
    assert body["taken_at"] is not None

    rows = client.get("/api/v1/scanner/candidate-promotions", headers={"X-API-Key": "test-scanner-key"}).json()
    row = next(r for r in rows if r["id"] == promotion_id)
    assert row["taken"] is True


def test_marking_taken_false_is_distinct_from_undecided(router, client):
    promotion_id = _seed_promotion(router)

    resp = client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["taken"] is False  # explicit skip, not None/undecided
    assert body["taken_at"] is not None


def test_marking_taken_null_resets_to_undecided(router, client):
    promotion_id = _seed_promotion(router)
    client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": True},
    )

    resp = client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": None},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["taken"] is None
    assert body["taken_at"] is None  # fully back to "never touched", not just taken=None with a stale timestamp

    rows = client.get("/api/v1/scanner/candidate-promotions", headers={"X-API-Key": "test-scanner-key"}).json()
    row = next(r for r in rows if r["id"] == promotion_id)
    assert row["taken"] is None
    assert row["taken_at"] is None


def test_null_reset_from_false_also_works(router, client):
    promotion_id = _seed_promotion(router)
    client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": False},
    )
    resp = client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": None},
    )
    assert resp.status_code == 200
    assert resp.json()["taken"] is None


def test_null_reset_leaves_already_computed_outcome_untouched(router, client):
    """Scoping decision, tested explicitly: resetting taken back to
    undecided does NOT erase outcome/outcome_* -- those are independent
    historical facts about what real bars showed, not something a
    taken-flag correction should silently wipe."""
    promotion_id = _seed_promotion(router)
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.execute(
        "UPDATE candidate_promotions SET taken=1, outcome='hit_target', outcome_hit_at=? WHERE id=?",
        ("2026-08-25T14:00:00Z", promotion_id),
    )
    conn.commit()
    conn.close()

    resp = client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": None},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["taken"] is None
    assert body["outcome"] == "hit_target"
    assert body["outcome_hit_at"] == "2026-08-25T14:00:00Z"


def test_taken_key_is_still_required_omitting_it_is_rejected(router, client):
    """null is a supported explicit value, not the same as leaving the key
    out entirely -- omitting it must still be a 422, not a silent reset."""
    promotion_id = _seed_promotion(router)
    resp = client.patch(
        f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={},
    )
    assert resp.status_code == 422


def test_taken_endpoint_requires_api_key(router, client):
    promotion_id = _seed_promotion(router)
    resp = client.patch(f"/api/v1/scanner/candidate-promotions/{promotion_id}/taken", json={"taken": True})
    assert resp.status_code == 401


def test_taken_endpoint_404s_for_unknown_promotion_id(router, client):
    resp = client.patch(
        "/api/v1/scanner/candidate-promotions/999999/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": True},
    )
    assert resp.status_code == 404


def test_taken_targets_specific_promotion_row_not_all_of_a_tickers_rows(router, client):
    """Append-only means a ticker can have multiple promotion rows -- marking
    one taken must not affect a different row for the same ticker."""
    first_id = _seed_promotion(router, ticker="AAPL", promoted_at="2026-08-20T14:00:00Z")
    second_id = _seed_promotion(router, ticker="AAPL", promoted_at="2026-08-24T14:00:00Z")
    assert first_id != second_id

    client.patch(
        f"/api/v1/scanner/candidate-promotions/{first_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": True},
    )

    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    first_row = conn.execute("SELECT taken FROM candidate_promotions WHERE id=?", (first_id,)).fetchone()
    second_row = conn.execute("SELECT taken FROM candidate_promotions WHERE id=?", (second_id,)).fetchone()
    conn.close()
    assert first_row["taken"] == 1
    assert second_row["taken"] is None


def test_null_reset_targets_specific_promotion_row_not_all_of_a_tickers_rows(router, client):
    first_id = _seed_promotion(router, ticker="AAPL", promoted_at="2026-08-20T14:00:00Z")
    second_id = _seed_promotion(router, ticker="AAPL", promoted_at="2026-08-24T14:00:00Z")
    for pid in (first_id, second_id):
        client.patch(
            f"/api/v1/scanner/candidate-promotions/{pid}/taken",
            headers={"X-API-Key": "test-scanner-key"},
            json={"taken": True},
        )

    client.patch(
        f"/api/v1/scanner/candidate-promotions/{first_id}/taken",
        headers={"X-API-Key": "test-scanner-key"},
        json={"taken": None},
    )

    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    first_row = conn.execute("SELECT taken FROM candidate_promotions WHERE id=?", (first_id,)).fetchone()
    second_row = conn.execute("SELECT taken FROM candidate_promotions WHERE id=?", (second_id,)).fetchone()
    conn.close()
    assert first_row["taken"] is None
    assert second_row["taken"] == 1
