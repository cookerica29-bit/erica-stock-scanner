"""Stage-3 tracking infrastructure -- candidates_router.py.

Two new append-only tables, both modeled on patterns already proven in this
codebase, not invented from scratch:

- candidate_visual_reviews: a human's stage-3 visual-review verdict
  (approved/rejected), captured independent of promotion/taken -- so a
  review is recorded even for a candidate never promoted. Tri-state
  (approved/rejected/not-yet-reviewed), but "not yet reviewed" is the
  absence of any row, not a stored null -- consistent with this being an
  append-only event log (a review either happened, with a real verdict, or
  it hasn't). GET /candidate-visual-reviews returns full history; "current
  verdict" is each ticker's latest row, the same pattern GET
  /candidate-promotions already uses for its own append-only table.

- candidate_ranking_snapshots: one row per ranked candidate per ranking
  computation, all sharing one snapshot_id -- so what the ranking actually
  surfaced at a given moment can be compared later against what was
  visually approved and what was taken, without re-deriving ranking state
  from live data that's since moved (the same execution_shadow_ok
  volatility lesson from earlier this session, applied to ranking).
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


def _promotion_daily_frame():
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1
        rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000})
    rows[5]["Low"] = 99.0
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


# ---------------------------------------------------------------------------
# candidate_visual_reviews
# ---------------------------------------------------------------------------

def test_visual_review_requires_auth(client):
    response = client.post(
        "/api/v1/scanner/candidates/AAPL/visual-review",
        json={"source": "ma_pipeline", "verdict": "approved"},
    )
    assert response.status_code == 401


def test_visual_review_records_a_verdict(client, headers):
    response = client.post(
        "/api/v1/scanner/candidates/aapl/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "verdict": "approved", "note": "Clean structure, taking it."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["source"] == "ma_pipeline"
    assert body["verdict"] == "approved"
    assert body["note"] == "Clean structure, taking it."
    assert body["reviewed_at"]
    assert isinstance(body["id"], int)


def test_visual_review_rejects_invalid_verdict(client, headers):
    response = client.post(
        "/api/v1/scanner/candidates/AAPL/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "verdict": "maybe"},
    )
    assert response.status_code == 422


def test_visual_review_does_not_require_a_promotion_or_candidate_row(client, headers):
    # Deliberately no POST /candidates first -- a review can happen for any
    # ticker/source, independent of ENTER_NOW status.
    response = client.post(
        "/api/v1/scanner/candidates/NEVERPROMOTED/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "verdict": "rejected"},
    )
    assert response.status_code == 200
    assert response.json()["verdict"] == "rejected"


def test_visual_review_is_append_only_not_overwritten(client, headers):
    first = client.post(
        "/api/v1/scanner/candidates/AAPL/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "verdict": "rejected"},
    )
    second = client.post(
        "/api/v1/scanner/candidates/AAPL/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "verdict": "approved"},
    )
    assert first.json()["id"] != second.json()["id"]

    listed = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    aapl_reviews = [r for r in listed if r["ticker"] == "AAPL"]
    assert len(aapl_reviews) == 2
    verdicts = {r["verdict"] for r in aapl_reviews}
    assert verdicts == {"rejected", "approved"}
    # Newest first.
    assert listed[0]["ticker"] == "AAPL"
    assert listed[0]["verdict"] == "approved"


def test_visual_review_list_requires_auth(client):
    response = client.get("/api/v1/scanner/candidate-visual-reviews")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# candidate_ranking_snapshots
# ---------------------------------------------------------------------------

def _seed_one_ranked_candidate(client, headers):
    previous = {
        "_batch_download": router._batch_download,
        "_best_contract": router._best_contract,
        "_latest_quote_for_ticker": router._latest_quote_for_ticker,
        "_latest_quotes_for_previews": router._latest_quotes_for_previews,
        "_recent_4h_bars_for_execution_shadow": router._recent_4h_bars_for_execution_shadow,
    }
    router._batch_download = lambda tickers, period, interval: {str(tickers[0]).upper(): _promotion_daily_frame()}
    router._best_contract = lambda ticker, direction, entry, **kwargs: {
        "available": True, "execution": "Fair", "type": "CALL", "strike": 100.0,
        "expiry": "2026-09-18", "dte": 29, "symbol": "MOCK", "source": "option_chain",
        "bid": 1.10, "ask": 1.20, "mid": 1.15, "mark": 1.15, "estimated_contract_cost": 120.0,
    }
    router._latest_quote_for_ticker = lambda ticker: {
        "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
    }
    router._latest_quotes_for_previews = lambda previews: {
        str(preview.get("ticker") or "").upper(): {
            "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
        }
        for preview in previews
    }
    router._recent_4h_bars_for_execution_shadow = lambda ticker: [
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
    ]
    try:
        payload = {
            "source": "ma_pipeline",
            "scanned_at": "2026-08-20T14:30:00Z",
            "candidates": [{
                "ticker": "NVDA", "signal": "long", "entry_price": 100.0, "ema21_4h": 99.0,
                "daily_regime": "bullish", "confidence": "high", "sma50_daily": 106.0, "sma200_daily": 104.0,
            }],
        }
        created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
        assert created.status_code == 200
        return previous
    except Exception:
        for name, fn in previous.items():
            setattr(router, name, fn)
        raise


def test_ranked_call_writes_a_ranking_snapshot(client, headers):
    previous = _seed_one_ranked_candidate(client, headers)
    try:
        response = client.get("/api/v1/scanner/candidates/ranked", headers=headers)
        assert response.status_code == 200
        snapshot_id = response.json()["snapshot_id"]
        assert snapshot_id

        snapshots = client.get("/api/v1/scanner/candidate-ranking-snapshots", headers=headers).json()
        rows = [row for row in snapshots if row["snapshot_id"] == snapshot_id]
        assert len(rows) == 1
        assert rows[0]["ticker"] == "NVDA"
        assert rows[0]["rank"] == 1
        assert rows[0]["mechanism"] == router.RANKING_MECHANISM_VERSION
        assert rows[0]["computed_at"]
    finally:
        for name, fn in previous.items():
            setattr(router, name, fn)


def test_each_ranked_call_writes_a_distinct_snapshot(client, headers):
    previous = _seed_one_ranked_candidate(client, headers)
    try:
        first = client.get("/api/v1/scanner/candidates/ranked", headers=headers).json()
        second = client.get("/api/v1/scanner/candidates/ranked", headers=headers).json()
        assert first["snapshot_id"] != second["snapshot_id"]

        all_snapshots = client.get("/api/v1/scanner/candidate-ranking-snapshots", headers=headers).json()
        distinct_ids = {row["snapshot_id"] for row in all_snapshots}
        assert {first["snapshot_id"], second["snapshot_id"]} <= distinct_ids
    finally:
        for name, fn in previous.items():
            setattr(router, name, fn)


def test_ranking_snapshots_filterable_by_snapshot_id(client, headers):
    previous = _seed_one_ranked_candidate(client, headers)
    try:
        result = client.get("/api/v1/scanner/candidates/ranked", headers=headers).json()
        snapshot_id = result["snapshot_id"]

        filtered = client.get(
            "/api/v1/scanner/candidate-ranking-snapshots",
            params={"snapshot_id": snapshot_id},
            headers=headers,
        ).json()
        assert len(filtered) == 1
        assert filtered[0]["snapshot_id"] == snapshot_id
    finally:
        for name, fn in previous.items():
            setattr(router, name, fn)


def test_ranking_snapshots_list_requires_auth(client):
    response = client.get("/api/v1/scanner/candidate-ranking-snapshots")
    assert response.status_code == 401
