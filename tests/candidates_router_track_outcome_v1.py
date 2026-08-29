"""POST /candidates/{ticker}/track-outcome and its ripple effects on
GET /candidate-promotions -- the additive outcome-evidence path that
deliberately does NOT touch _promotion_block_reason (the real ENTER_NOW/
dashboard gate).

Originally shipped 2026-08-28 as track-short-outcome (short-only).
Generalized 2026-08-29 when a second category of "has a real plan, worth
tracking, but not ENTER_NOW-eligible" candidate appeared: a long that's
mechanically ready but confluence_label=="conflicted" (see
_promotion_block_reason's own confluence-conflicted branch for the real
DASH-driven rationale). One endpoint for the concept, not two
near-duplicates.

Scope: the router-level wiring (endpoint behavior for both the short and
conflicted-long cases, promotion_kind storage, default-listing exclusion,
independence from the real promotion path, rejecting candidates that have
no real plan to track or are already genuinely ENTER_NOW-eligible). Whether
a tracking_only row flows through the hourly outcome resolver is covered
separately in
tests/candidate_promotion_outcome_watcher_v1.py::test_tracking_only_short_promotion_resolves_exactly_like_a_real_one
-- that resolver query has never filtered on direction or kind, so nothing
here needs to re-prove that.
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


def _controlled_promotion(ticker="GOOD", direction="long", **overrides):
    """A fully controlled _compute_candidate_promotion() stand-in -- lets
    long-candidate tests set exact mechanical/confluence conditions instead
    of reverse-engineering real signal combinations from a price frame.
    Every key _store_promotion/_mechanical_promotion_block_reason actually
    reads is present with a clean-pass default; override just what a given
    test cares about."""
    plan = {
        "ticker": ticker, "source": "ma_pipeline", "direction": direction,
        "entry_price": 100.0, "stop": 95.0, "target": 110.0, "risk_reward": 2.0,
        "rr_warning": False, "no_valid_target": False,
        "promoted_at": "2026-08-29T12:00:00Z", "position_size": None,
        "atr14": 2.0, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 110.0, "raw_risk_reward": 2.0, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 95.0, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": None, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable", "bos_confirmed": True, "bos_details": None,
        "macro_bias": "Macro Neutral", "macro_conflict": False, "choch_conflict": False,
        "choch_details": None, "sweep_confirmed": False, "sweep_details": None,
        "rejection_confirmed": False, "rejection_details": None,
        "location_percentile": 50.0, "location_label": "midrange", "location_alignment": "neutral",
        "confluence_signals": None, "confluence_counts": None, "confluence_label": "some confluence",
        "entry_proximity_ok": True, "entry_proximity_reason": None,
        "execution_shadow_checked": True, "execution_shadow_ok": True,
        "execution_shadow_reason": "Recent 4H confirmation remains structurally intact",
    }
    plan.update(overrides)
    return plan


def _with_mocked_promotion(router, promotion_dict):
    """Context-manager-free monkeypatch: returns (restore_fn,) after
    replacing _compute_candidate_promotion with one that always returns
    promotion_dict, regardless of the real candidate row passed in."""
    previous = router._compute_candidate_promotion
    router._compute_candidate_promotion = lambda candidate: dict(promotion_dict)
    return previous


def _with_stubbed_live_gate_context(router):
    """track_candidate_outcome's LONG branch calls
    _promotion_with_live_gate_context to get real entry_proximity_ok/
    execution_shadow_ok (see that endpoint's own comment for why -- these
    are never part of _compute_candidate_promotion's own output). That
    function makes real network calls (_latest_quote_for_ticker,
    _attach_execution_shadow's 4h-bar fetch) this test suite has no
    business making. Stubs both to clean-pass values so tests can isolate
    the confluence/mechanical logic they actually care about. Returns a
    restore function."""
    previous_quote = router._latest_quote_for_ticker
    previous_shadow = router._attach_execution_shadow
    router._latest_quote_for_ticker = lambda ticker: {
        "price": 100.0, "price_branch": "midpoint", "source": "test",
    }
    router._attach_execution_shadow = lambda candidate, preview: {
        **preview, "execution_shadow_checked": True, "execution_shadow_ok": True,
        "execution_shadow_reason": "ok",
    }

    def restore():
        router._latest_quote_for_ticker = previous_quote
        router._attach_execution_shadow = previous_shadow

    return restore


# --- short candidates (original behavior, unchanged, URL renamed) ---------

def test_track_outcome_creates_tracking_only_row_for_a_short_with_real_plan_math():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "ORCL", "short").status_code == 200

            resp = client.post(
                "/api/v1/scanner/candidates/ORCL/track-outcome?source=ma_pipeline",
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


def test_track_outcome_404s_for_unknown_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            resp = client.post(
                "/api/v1/scanner/candidates/NOPE/track-outcome?source=ma_pipeline",
                headers=headers,
            )
            assert resp.status_code == 404
        finally:
            router._batch_download = previous_download


def test_track_outcome_does_not_change_candidate_status():
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
                "/api/v1/scanner/candidates/ORCL/track-outcome?source=ma_pipeline",
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
                "/api/v1/scanner/candidates/ORCL/track-outcome?source=ma_pipeline",
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


def test_tracking_only_short_does_not_bypass_the_real_enter_now_gate():
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
                "/api/v1/scanner/candidates/ORCL/track-outcome?source=ma_pipeline",
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
                "/api/v1/scanner/candidates/ORCL/track-outcome?source=ma_pipeline",
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
                "/api/v1/scanner/candidates/RYN/track-outcome?source=ma_pipeline",
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


# --- conflicted longs (new 2026-08-29 case) ---------------------------------

def test_track_outcome_accepts_a_mechanically_ready_conflicted_long():
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "DASH", "long", regime="bullish").status_code == 200
            previous_promo = _with_mocked_promotion(
                router, _controlled_promotion(ticker="DASH", confluence_label="conflicted")
            )
            restore_gate_context = _with_stubbed_live_gate_context(router)
            try:
                resp = client.post(
                    "/api/v1/scanner/candidates/DASH/track-outcome?source=ma_pipeline",
                    headers=headers,
                )
                assert resp.status_code == 201
                body = resp.json()
                assert body["ticker"] == "DASH"
                assert body["direction"] == "long"
                assert body["confluence_label"] == "conflicted"
                assert body["promotion_kind"] == "tracking_only"
            finally:
                router._compute_candidate_promotion = previous_promo
                restore_gate_context()
        finally:
            router._batch_download = previous_download


def test_track_outcome_rejects_an_already_enter_now_eligible_long():
    """No reason to track-only something that's already fully eligible --
    use the real promotion path instead."""
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "CLEAN", "long", regime="bullish").status_code == 200
            previous_promo = _with_mocked_promotion(
                router, _controlled_promotion(ticker="CLEAN", confluence_label="some confluence")
            )
            restore_gate_context = _with_stubbed_live_gate_context(router)
            try:
                resp = client.post(
                    "/api/v1/scanner/candidates/CLEAN/track-outcome?source=ma_pipeline",
                    headers=headers,
                )
                assert resp.status_code == 422
                assert "already ENTER_NOW eligible" in resp.json()["detail"]
            finally:
                router._compute_candidate_promotion = previous_promo
                restore_gate_context()
        finally:
            router._batch_download = previous_download


def test_track_outcome_rejects_a_long_with_no_valid_plan():
    """A mechanical failure (no valid target here) means there's no real
    plan to track an outcome against -- rejected regardless of direction or
    confluence, short or long, research-only or not."""
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "BROKEN", "long", regime="bullish").status_code == 200
            previous_promo = _with_mocked_promotion(
                router,
                _controlled_promotion(ticker="BROKEN", no_valid_target=True, target=None, risk_reward=None),
            )
            try:
                resp = client.post(
                    "/api/v1/scanner/candidates/BROKEN/track-outcome?source=ma_pipeline",
                    headers=headers,
                )
                assert resp.status_code == 422
                assert "no valid plan to track" in resp.json()["detail"]
            finally:
                router._compute_candidate_promotion = previous_promo
        finally:
            router._batch_download = previous_download


def test_tracking_only_conflicted_long_does_not_bypass_the_real_enter_now_gate():
    """Same independence guarantee as the short case above, for the new
    conflicted-long case: tracking it must not grant real ENTER_NOW
    eligibility -- PATCH /candidates/{ticker} status=active must still
    reject it for being conflicted."""
    with tempfile.TemporaryDirectory() as tmp:
        router, previous_download = _seed_env(tmp)
        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            assert _ingest(client, headers, "DASH", "long", regime="bullish").status_code == 200
            previous_promo = _with_mocked_promotion(
                router, _controlled_promotion(ticker="DASH", confluence_label="conflicted")
            )
            # update_candidate_status re-enriches with LIVE entry-proximity/
            # execution-shadow context on top of _compute_candidate_promotion
            # (_promotion_with_live_gate_context) regardless of what's in the
            # mocked promotion dict -- stub those two specifically so this
            # test isolates the confluence check, not real market-data
            # plumbing this test doesn't care about.
            previous_quote = router._latest_quote_for_ticker
            previous_shadow = router._attach_execution_shadow
            router._latest_quote_for_ticker = lambda ticker: {
                "price": 100.0, "price_branch": "midpoint", "source": "test",
            }
            router._attach_execution_shadow = lambda candidate, preview: {
                **preview, "execution_shadow_checked": True, "execution_shadow_ok": True,
                "execution_shadow_reason": "ok",
            }
            try:
                client.post(
                    "/api/v1/scanner/candidates/DASH/track-outcome?source=ma_pipeline",
                    headers=headers,
                )
                promoted = client.patch(
                    "/api/v1/scanner/candidates/DASH?source=ma_pipeline",
                    headers=headers,
                    json={"status": "active"},
                )
                assert promoted.status_code == 422
                assert "confluence is conflicted" in promoted.json()["detail"].lower()
            finally:
                router._compute_candidate_promotion = previous_promo
                router._latest_quote_for_ticker = previous_quote
                router._attach_execution_shadow = previous_shadow
        finally:
            router._batch_download = previous_download
