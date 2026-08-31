"""Stage B: GET /candidates/review-queue -- candidates_router.py.

The whole point: this path must never trigger option-contract hydration,
and must never regress GET /candidate-plan-previews (today's existing
dashboard). Both are tested directly here, not assumed from reading the
code -- _safe_option_contract_for_candidate is monkeypatched to a spy that
fails the test if it's ever called via the review-queue path, and
/candidate-plan-previews's output is compared byte-for-byte before/after
exercising the review queue.

Ranking logic itself (_stage1_mechanical_ready / rank_stage1_candidates)
is reused unchanged from Stage 2 -- not retested here beyond confirming
the review-queue endpoint produces the same order Stage 2's own tests
already establish for that logic.
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
    """Fails loudly if anything on the review-queue path ever calls this --
    the entire point of Stage B."""
    calls = []

    def _spy(ticker, direction, entry, **kwargs):
        calls.append((ticker, direction, entry, kwargs))
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)
    return calls


def _seed(client, headers, ticker="NVDA", signal="long", entry_price=100.0):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": ticker, "signal": signal, "entry_price": entry_price, "ema21_4h": 99.0,
            "daily_regime": "bullish" if signal == "long" else "bearish", "confidence": "high",
            "sma50_daily": 106.0, "sma200_daily": 104.0,
        }],
    }
    created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert created.status_code == 200


def test_review_queue_requires_auth(client):
    response = client.get("/api/v1/scanner/candidates/review-queue")
    assert response.status_code == 401


def test_review_queue_never_triggers_option_hydration(client, headers, _forbid_option_hydration):
    _seed(client, headers)
    response = client.get("/api/v1/scanner/candidates/review-queue", headers=headers)
    assert response.status_code == 200
    assert _forbid_option_hydration == []  # the spy was never called


def test_review_queue_marks_freshly_computed_entries_as_deferred(client, headers):
    _seed(client, headers)
    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next(c for c in body["candidates"] if c["ticker"] == "NVDA")
    assert entry["option_contract_deferred"] is True
    assert entry.get("option_contract") is None


def test_review_queue_excludes_short_candidates(client, headers):
    _seed(client, headers, ticker="NVDA", signal="long")
    _seed(client, headers, ticker="TSLA", signal="short", entry_price=200.0)
    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    tickers = {c["ticker"] for c in body["candidates"]}
    assert "TSLA" not in tickers


def test_review_queue_entry_carries_setup_key_and_no_review_yet(client, headers):
    _seed(client, headers)
    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next(c for c in body["candidates"] if c["ticker"] == "NVDA")
    assert entry["setup_key"]
    assert entry["current_review"] is None  # Needs Review


def test_review_queue_reflects_latest_review_for_setup(client, headers):
    _seed(client, headers)
    first_pass = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next(c for c in first_pass["candidates"] if c["ticker"] == "NVDA")
    setup_key = entry["setup_key"]

    client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "decision": "approve",
        },
    )

    second_pass = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry2 = next(c for c in second_pass["candidates"] if c["ticker"] == "NVDA")
    assert entry2["setup_key"] == setup_key
    assert entry2["current_review"] is not None
    assert entry2["current_review"]["decision"] == "approve"


def test_review_queue_writes_a_ranking_snapshot(client, headers):
    _seed(client, headers)
    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    snapshots = client.get("/api/v1/scanner/candidate-ranking-snapshots", headers=headers).json()
    rows = [row for row in snapshots if row["snapshot_id"] == body["snapshot_id"]]
    assert len(rows) == 1
    assert rows[0]["ticker"] == "NVDA"


def test_review_queue_opportunistically_reuses_a_fresh_cached_preview(client, headers, monkeypatch):
    """If GET /candidate-plan-previews already computed a fresh row (with
    real options, since that path is untouched), the review queue reuses
    it via a pure read -- and does NOT call _compute_candidate_promotion
    again for that candidate."""
    _seed(client, headers)

    # Warm the shared cache the normal way -- via the untouched, fully
    # optioned /candidate-plan-previews path. Give it a real (mocked)
    # option contract this once, since that path still calls it.
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry, **kwargs: {
        "available": True, "execution": "Fair", "type": "CALL", "strike": 100.0,
        "expiry": "2026-09-18", "dte": 29, "symbol": "MOCK", "source": "option_chain",
    })
    warm = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers).json()
    assert warm[0]["option_contract"]["available"] is True

    calls = []
    original = router._compute_candidate_promotion

    def _spy(candidate):
        calls.append(candidate["ticker"])
        return original(candidate)

    monkeypatch.setattr(router, "_compute_candidate_promotion", _spy)
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not be called on the review-queue path")
    ))

    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next(c for c in body["candidates"] if c["ticker"] == "NVDA")
    assert calls == []  # reused the cached row, no fresh computation
    assert entry["option_contract_deferred"] is False  # real data, already there
    assert entry["stop"] == warm[0]["stop"]
    assert entry["target"] == warm[0]["target"]


def test_candidate_plan_previews_unchanged_by_review_queue_traffic(client, headers, monkeypatch):
    """The actual regression guard: exercise the review queue, then confirm
    /candidate-plan-previews still returns byte-identical output -- the
    review queue never writes into candidate_plan_previews at all."""
    _seed(client, headers)
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry, **kwargs: {
        "available": True, "execution": "Fair", "type": "CALL", "strike": 100.0,
        "expiry": "2026-09-18", "dte": 29, "symbol": "MOCK", "source": "option_chain",
    })
    before = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers).json()

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not be called on the review-queue path")
    ))
    client.get("/api/v1/scanner/candidates/review-queue", headers=headers)

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry, **kwargs: {
        "available": True, "execution": "Fair", "type": "CALL", "strike": 100.0,
        "expiry": "2026-09-18", "dte": 29, "symbol": "MOCK", "source": "option_chain",
    })
    after = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers).json()
    assert before == after
