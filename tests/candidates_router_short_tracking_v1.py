"""POST /candidates/{ticker}/track-short-outcome and its ripple effects on
GET /candidate-promotions -- the additive short-side outcome-evidence path
that deliberately does NOT touch _promotion_block_reason (the real
ENTER_NOW/dashboard gate, which stays fully blocked for shorts).

Scope: the router-level wiring (endpoint behavior, promotion_kind storage,
default-listing exclusion, independence from the real promotion path).
Whether a tracking_only row flows through the hourly outcome resolver is
covered separately in
tests/candidate_promotion_outcome_watcher_v1.py::test_tracking_only_short_promotion_resolves_exactly_like_a_real_one
-- that resolver query has never filtered on direction or kind, so nothing
here needs to re-prove that; this file proves the row gets created and kept
out of the dashboard-facing listing correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client():
    import candidates_router

    app = FastAPI()
    app.include_router(candidates_router.router)
    return TestClient(app)


def _daily_frame():
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1
        rows.append({
            "Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5,
            "Close": close, "Volume": 1_000_000,
        })
    rows[5]["Low"] = 99.0
    rows[10]["Low"] = 90.0
    rows[10]["Close"] = 91.0
    rows[10]["Open"] = 92.0
    rows[20]["High"] = 110.0
    rows[20]["Close"] = 109.0
    rows[20]["Open"] = 108.0
    return pd.DataFrame(rows, index=index)


def _seed_env(tmp):
    db_path = os.path.join(tmp, "candidates.db")
    os.environ["KAIROS_CANDIDATES_DB"] = db_path
    os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"
    import candidates_router
    previous_download = candidates_router._batch_download
    candidates_router._batch_download = lambda tickers, period, interval: {
        str(tickers[0]).upper(): _daily_frame()
    }
    return candidates_router, previous_download


def _ingest(client, headers, ticker, signal, regime="bearish"):
    return client.post(
        "/api/v1/scanner/candidates",
        headers=headers,
        json={
            "source": "ma_pipeline",
            "scanned_at": "2026-08-28T13:57:00Z",
            "candidates": [{
                "ticker": ticker, "signal": signal, "entry_price": 100.0,
                "ema21_4h": 101.0, "daily_regime": regime, "confidence": "high",
                "sma50_daily": 98.0, "sma200_daily": 120.0,
            }],
        },
    )


def test_track_short_outcome_creates_tracking_only_row_with_real_plan_math():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "ORCL", "short").status_code == 200

            resp = client.post(
                "/api/v1/scanner/candidates/ORCL/track-short-outcome?source=ma_pipeline",
                headers=headers,
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["ticker"] == "ORCL"
            assert body["direction"] == "short"
            assert body["promotion_kind"] == "tracking_only"
            # Real plan math, not a stub -- same computation every real
            # promotion uses.
            assert body["stop"] is not None
            assert body["atr14"] > 0
            assert isinstance(body["id"], int)
        finally:
            router._batch_download = previous_download


def test_track_short_outcome_rejects_a_long_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "AAPL", "long", regime="bullish").status_code == 200

            resp = client.post(
                "/api/v1/scanner/candidates/AAPL/track-short-outcome?source=ma_pipeline",
                headers=headers,
            )
            assert resp.status_code == 422
            assert "short candidates only" in resp.json()["detail"]
        finally:
            router._batch_download = previous_download


def test_track_short_outcome_404s_for_unknown_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            resp = client.post(
                "/api/v1/scanner/candidates/NOPE/track-short-outcome?source=ma_pipeline",
                headers=headers,
            )
            assert resp.status_code == 404
        finally:
            router._batch_download = previous_download


def test_track_short_outcome_does_not_change_candidate_status():
    """This is a promotion-row event, not a status change -- the candidate
    must stay exactly where it was (still 'new'), unlike the real
    status=active path which flips status to 'active' as part of the same
    request."""
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "ORCL", "short").status_code == 200

            client.post(
                "/api/v1/scanner/candidates/ORCL/track-short-outcome?source=ma_pipeline",
                headers=headers,
            )
            listed = client.get("/api/v1/scanner/candidates", headers=headers).json()
            assert listed[0]["status"] == "new"
        finally:
            router._batch_download = previous_download


def test_tracking_only_row_excluded_from_default_promotions_listing():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "ORCL", "short").status_code == 200
            client.post(
                "/api/v1/scanner/candidates/ORCL/track-short-outcome?source=ma_pipeline",
                headers=headers,
            )

            default_listing = client.get("/api/v1/scanner/candidate-promotions", headers=headers).json()
            assert not any(p["ticker"] == "ORCL" for p in default_listing)

            opted_in = client.get(
                "/api/v1/scanner/candidate-promotions?include_tracking_only=true", headers=headers
            ).json()
            assert any(p["ticker"] == "ORCL" and p["promotion_kind"] == "tracking_only" for p in opted_in)
        finally:
            router._batch_download = previous_download


def test_tracking_only_row_does_not_bypass_the_real_enter_now_gate():
    """Proves the two paths are genuinely independent: having a
    tracking-only row on file for a ticker must not grant (or even touch)
    ENTER_NOW/dashboard eligibility for that same ticker. The real gate --
    PATCH /candidates/{ticker} with status=active -- must still reject the
    short exactly as it always has."""
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "ORCL", "short").status_code == 200
            client.post(
                "/api/v1/scanner/candidates/ORCL/track-short-outcome?source=ma_pipeline",
                headers=headers,
            )

            promoted = client.patch(
                "/api/v1/scanner/candidates/ORCL?source=ma_pipeline",
                headers=headers,
                json={"status": "active"},
            )
            assert promoted.status_code == 422
            assert "research-only" in promoted.json()["detail"]
        finally:
            router._batch_download = previous_download


def test_taken_flag_works_on_a_tracking_only_row_id_with_no_direction_check():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "ORCL", "short").status_code == 200
            created = client.post(
                "/api/v1/scanner/candidates/ORCL/track-short-outcome?source=ma_pipeline",
                headers=headers,
            ).json()

            taken = client.patch(
                f"/api/v1/scanner/candidate-promotions/{created['id']}/taken",
                headers=headers,
                json={"taken": True},
            )
            assert taken.status_code == 200
            assert taken.json()["taken"] is True
            assert taken.json()["promotion_kind"] == "tracking_only"
        finally:
            router._batch_download = previous_download


def test_a_newer_tracking_only_row_never_hides_an_existing_real_promotion():
    """The documented edge case from list_candidate_promotions' comment:
    "latest" is computed over real (enter_now) promotions only, so a
    tracking-only row added later for the same ticker can't displace or
    hide a genuine promotion already on record -- it was never a real
    promotion to begin with. Verified here as actual behavior, not just
    asserted in a comment."""
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            import sqlite3
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "RYN", "short").status_code == 200

            # Seed an OLD real ("enter_now") promotion directly (bypassing
            # the still-blocked real gate, same technique the outcome
            # watcher tests use to set up historical rows).
            conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
            conn.row_factory = sqlite3.Row
            router._initialize_candidates_schema(conn)
            old_promo = router._compute_candidate_promotion(
                conn.execute("SELECT * FROM candidates WHERE ticker='RYN'").fetchone()
            )
            router._store_promotion(conn, old_promo)
            conn.commit()
            conn.close()

            # Confirm the old row is visible before the newer tracking-only
            # one exists.
            before = client.get("/api/v1/scanner/candidate-promotions", headers=headers).json()
            assert any(p["ticker"] == "RYN" for p in before)

            # Now add the newer tracking-only row.
            client.post(
                "/api/v1/scanner/candidates/RYN/track-short-outcome?source=ma_pipeline",
                headers=headers,
            )

            after = client.get("/api/v1/scanner/candidate-promotions", headers=headers).json()
            ryn_rows = [p for p in after if p["ticker"] == "RYN"]
            assert len(ryn_rows) == 1
            assert ryn_rows[0]["promotion_kind"] == "enter_now"

            # include_tracking_only=true drops the exclusion entirely, so
            # the endpoint's real contract ("one row per ticker/source, the
            # latest") applies uniformly again -- for RYN that's now the
            # tracking-only row, since it has the higher id. Confirms the
            # tracking-only row genuinely exists (it's not lost), just
            # invisible in the default view.
            opted_in = client.get(
                "/api/v1/scanner/candidate-promotions?include_tracking_only=true", headers=headers
            ).json()
            ryn_opted_in = [p for p in opted_in if p["ticker"] == "RYN"]
            assert len(ryn_opted_in) == 1
            assert ryn_opted_in[0]["promotion_kind"] == "tracking_only"
        finally:
            router._batch_download = previous_download
