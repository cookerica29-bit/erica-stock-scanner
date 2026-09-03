"""Kairos -- Review Queue Evolution Sprint 1 (review_queue_evolution_audit.md
+ the sprint's own numbered sections).

Reclassifies entry_proximity_ok/execution_shadow_ok from hard elimination
gates into Review Value ranking evidence, scoped ENTIRELY to
GET /candidates/review-queue -- _stage1_mechanical_ready/
rank_stage1_candidates/_confluence_sort_key (GET /candidates/ranked,
"Stage 2") and _mechanical_promotion_block_reason/_promotion_block_reason
(the separate ENTER_NOW-dashboard gate) are untouched; regression coverage
for those lives in tests/review_queue_v1.py and is not duplicated here.

Covers: reclassification (a candidate failing ONLY entry_proximity or
execution_shadow is now included, not eliminated), the new fresh-price
sanity layer (target_reached_or_passed hard exclusion, extension/target-
room/staleness penalties), the transparent Review Value score formula,
global ranking before truncation, the Top-10/expand-to-15 display cap
(never force-filled, previously-reviewed candidates never consume a new-
review slot), and the diagnostics block.

Every test exercises the real candidates_router.py code via a real
FastAPI TestClient -- not a reimplementation.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


def _daily_frame():
    # Real, verified values for this exact frame (2026-09 sprint):
    # entry=100 -> stop~96.4977, target~109.7977, atr14~2.0226, R:R=2.8
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


STOP = 96.4977
TARGET = 109.7977
ATR14 = 2.0226


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
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(t).upper(): _daily_frame() for t in tickers})
    monkeypatch.setattr(router, "_latest_quote_for_ticker", lambda ticker: {
        "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
    })
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(p.get("ticker") or "").upper(): {
            "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
        }
        for p in previews
    })
    # Deliberately NOT a passing execution-shadow shape by default -- most
    # tests want to control this explicitly per-ticker via _quotes_for/
    # _shadow_for below. Empty bars -> execution_shadow_ok False.
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [])


@pytest.fixture(autouse=True)
def _forbid_option_hydration(monkeypatch):
    def _spy(ticker, direction, entry, **kwargs):
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)


PASSING_SHADOW_BARS = [
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


def _seed(client, headers, ticker="AMD", entry_price=100.0, scanned_at="2026-08-20T14:30:00Z"):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": scanned_at,
        "candidates": [{
            "ticker": ticker, "signal": "long", "entry_price": entry_price, "ema21_4h": entry_price - 1.0,
            "daily_regime": "long", "confidence": "high",
            "sma50_daily": entry_price + 6.0, "sma200_daily": entry_price + 4.0,
        }],
    }
    created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert created.status_code == 200


def _quotes_map(prices: dict[str, float]):
    def _fn(previews):
        return {
            ticker.upper(): {"price": price, "timestamp": "2026-08-20T18:30:00Z", "source": "mock", "price_branch": "mid"}
            for ticker, price in prices.items()
        }
    return _fn


def _get_queue(client, headers):
    resp = client.get("/api/v1/scanner/candidates/review-queue", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _find(payload, ticker):
    for c in payload["candidates"]:
        if c["ticker"] == ticker:
            return c
    return None


# ==========================================================================
# Reclassification: entry_proximity / execution_shadow no longer eliminate
# ==========================================================================

def test_entry_proximity_failure_no_longer_eliminates(client, headers, monkeypatch):
    _seed(client, headers, "AMD")
    # Far from entry (>1.5%/0.5xATR) -- would have been eliminated before this sprint.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": 105.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    entry = _find(payload, "AMD")
    assert entry is not None, "a candidate failing ONLY entry proximity must still appear in the review queue"
    assert entry["entry_proximity_ok"] is False, "sanity: this really did fail the old proximity gate"
    assert "review_value_score" in entry
    assert entry["review_value_breakdown"]["entry_proximity_points"] == 0.0


def test_execution_shadow_failure_no_longer_eliminates(client, headers, monkeypatch):
    _seed(client, headers, "AMD")
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": 100.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [])  # fails execution shadow

    payload = _get_queue(client, headers)
    entry = _find(payload, "AMD")
    assert entry is not None, "a candidate failing ONLY execution shadow must still appear in the review queue"
    assert entry["execution_shadow_ok"] is not True
    assert entry["review_value_breakdown"]["execution_shadow_points"] == 0.0


def test_promoted_from_near_miss_diagnostic_counts_it(client, headers, monkeypatch):
    _seed(client, headers, "AMD")
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": 105.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [])

    payload = _get_queue(client, headers)
    assert payload["diagnostics"]["promoted_from_near_miss_count"] == 1


# ==========================================================================
# Hard exclusions preserved (direction/regime/target/R:R) + new
# target_reached_or_passed exclusion
# ==========================================================================

def test_short_still_excluded(client, headers):
    payload = {
        "source": "ma_pipeline", "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": "SHRT", "signal": "short", "entry_price": 100.0, "ema21_4h": 101.0,
            "daily_regime": "short", "confidence": "high", "sma50_daily": 94.0, "sma200_daily": 96.0,
        }],
    }
    resp = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert resp.status_code == 200
    result = _get_queue(client, headers)
    assert _find(result, "SHRT") is None, "shorts remain excluded -- unchanged this sprint"


def test_target_reached_is_a_hard_exclusion(client, headers, monkeypatch):
    _seed(client, headers, "AMD")
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": TARGET + 1.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    assert _find(payload, "AMD") is None, "a candidate whose live price has passed target must be excluded, not merely penalized"
    assert payload["diagnostics"]["excluded_target_reached_count"] == 1


def test_target_exactly_at_is_excluded(client, headers, monkeypatch):
    _seed(client, headers, "AMD")
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": TARGET}))
    payload = _get_queue(client, headers)
    assert _find(payload, "AMD") is None, "equality counts -- at target is 'reached', not merely close"


def test_near_but_not_at_target_is_penalized_not_excluded(client, headers, monkeypatch):
    _seed(client, headers, "AMD")
    near_target_price = 108.0  # < TARGET (109.7977)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": near_target_price}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    entry = _find(payload, "AMD")
    assert entry is not None, "not yet at target -- must still be visible"
    assert entry["review_value_breakdown"]["target_room_penalty"] < 0, "very little remaining room must be penalized"


def test_no_valid_target_still_excluded(client, headers, monkeypatch):
    # A daily frame with no clean swing structure -> no_valid_target True.
    flat = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    flat_rows = [{"Open": 100.0, "High": 100.2, "Low": 99.8, "Close": 100.0, "Volume": 1_000_000} for _ in range(30)]
    flat_frame = pd.DataFrame(flat_rows, index=flat)
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(t).upper(): flat_frame for t in tickers})
    _seed(client, headers, "FLAT")
    payload = _get_queue(client, headers)
    assert _find(payload, "FLAT") is None


def test_rr_below_threshold_still_excluded_this_sprint(client, headers, monkeypatch):
    """R:R is treated conservatively this sprint (not loosened) -- a
    candidate whose SCAN-TIME R:R is below RR_WARNING_THRESHOLD must
    still be excluded, exactly as before."""
    def _low_rr_frame(tickers, period, interval):
        idx = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
        rows = []
        for i in range(30):
            close = 100.0 - i * 0.02  # much flatter -> small ATR/target distance -> low R:R
            rows.append({"Open": close + 0.05, "High": close + 0.1, "Low": close - 0.1, "Close": close, "Volume": 1_000_000})
        rows[10]["Low"] = 98.5
        rows[10]["Close"] = 98.8
        rows[20]["High"] = 101.0
        rows[20]["Close"] = 100.8
        return {str(t).upper(): pd.DataFrame(rows, index=idx) for t in tickers}

    monkeypatch.setattr(router, "_batch_download", _low_rr_frame)
    _seed(client, headers, "LOWRR")
    payload = _get_queue(client, headers)
    entry = _find(payload, "LOWRR")
    # Either excluded outright (no_valid_target/rr_warning) or simply not
    # present -- either way it must not appear as a ranked candidate.
    assert entry is None


# ==========================================================================
# Review Value formula -- transparent, additive, spot-checked
# ==========================================================================

def test_review_value_score_matches_hand_computed_formula(client, headers, monkeypatch):
    _seed(client, headers, "AMD")
    extended_price = 103.0  # entry+3.0 -- past the 0.5xATR/1.5% proximity band, not yet at target
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": extended_price}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    entry = _find(payload, "AMD")
    assert entry is not None
    breakdown = entry["review_value_breakdown"]

    counts = entry["confluence_counts"]
    expected_favorable = router.REVIEW_VALUE_FAVORABLE_POINTS * counts["favorable"]
    expected_unfavorable = router.REVIEW_VALUE_UNFAVORABLE_POINTS * counts["unfavorable"]
    expected_rr_points = router.REVIEW_VALUE_RR_POINT_WEIGHT * min(entry["risk_reward"], router.REVIEW_VALUE_RR_CAP_FOR_SCORING)

    assert breakdown["favorable_points"] == pytest.approx(expected_favorable, abs=0.01)
    assert breakdown["unfavorable_points"] == pytest.approx(expected_unfavorable, abs=0.01)
    assert breakdown["rr_points"] == pytest.approx(expected_rr_points, abs=0.01)
    assert breakdown["extension_penalty"] < 0, "price extended past entry beyond the proximity band must be penalized"

    hand_total = (
        breakdown["favorable_points"] + breakdown["unfavorable_points"]
        + breakdown["entry_proximity_points"] + breakdown["execution_shadow_points"]
        + breakdown["rr_points"] + breakdown["extension_penalty"]
        + breakdown["target_room_penalty"] + breakdown["staleness_penalty"]
    )
    assert entry["review_value_score"] == pytest.approx(hand_total, abs=0.01)


def test_staleness_penalty_applied_for_old_candidate(client, headers, monkeypatch):
    old_scanned_at = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    _seed(client, headers, "OLD", scanned_at=old_scanned_at)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"OLD": 100.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    entry = _find(payload, "OLD")
    assert entry is not None
    assert entry["fresh_price_state"]["candidate_age_hours"] > router.REVIEW_VALUE_STALENESS_GRACE_HOURS
    assert entry["review_value_breakdown"]["staleness_penalty"] < 0


def test_fresh_candidate_no_staleness_penalty(client, headers, monkeypatch):
    fresh_scanned_at = datetime.now(timezone.utc).isoformat()
    _seed(client, headers, "FRESH", scanned_at=fresh_scanned_at)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"FRESH": 100.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    entry = _find(payload, "FRESH")
    assert entry["review_value_breakdown"]["staleness_penalty"] == 0.0


# ==========================================================================
# Global ranking, never first-pass
# ==========================================================================

def test_ranking_is_global_not_first_pass(client, headers, monkeypatch):
    """A structurally stronger candidate seeded/updated LATER must still
    outrank a weaker one seeded earlier -- proves sort is by score, not
    insertion order."""
    _seed(client, headers, "WEAK", scanned_at="2026-08-20T09:00:00Z")
    _seed(client, headers, "STRONG", scanned_at="2026-08-20T14:00:00Z")
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"WEAK": 105.0, "STRONG": 100.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS if ticker == "STRONG" else [])

    payload = _get_queue(client, headers)
    weak = _find(payload, "WEAK")
    strong = _find(payload, "STRONG")
    assert strong["review_value_score"] > weak["review_value_score"]
    assert strong["rank"] < weak["rank"], "higher score must mean a lower (better) rank number"


# ==========================================================================
# Top-10 / expand-to-15 display cap
# ==========================================================================

def _seed_many(client, headers, n, prefix="TKR"):
    for i in range(n):
        _seed(client, headers, f"{prefix}{i:02d}", scanned_at="2026-08-20T14:30:00Z")


def test_never_force_fills_below_ten(client, headers, monkeypatch):
    _seed_many(client, headers, 4)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(p.get("ticker") or "").upper(): {"price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock", "price_branch": "mid"}
        for p in previews
    })
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    assert payload["diagnostics"]["needs_review_displayed_count"] == 4
    assert len(payload["candidates"]) == 4


def test_caps_at_ten_by_default(client, headers, monkeypatch):
    _seed_many(client, headers, 14)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(p.get("ticker") or "").upper(): {"price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock", "price_branch": "mid"}
        for p in previews
    })
    # All 14 identical, all clearing every gate/bonus equally -- a real tie.
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [])  # no execution-shadow bonus for any -> all tie

    payload = _get_queue(client, headers)
    assert payload["diagnostics"]["needs_review_total_count"] == 14
    assert payload["diagnostics"]["needs_review_displayed_count"] == 10, "identical scores must not expand past 10 -- no meaningful quality gap to justify it"
    assert len(payload["candidates"]) == 10


def test_never_exceeds_fifteen(client, headers, monkeypatch):
    _seed_many(client, headers, 20)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(p.get("ticker") or "").upper(): {"price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock", "price_branch": "mid"}
        for p in previews
    })
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    assert payload["diagnostics"]["needs_review_displayed_count"] <= 15
    assert len(payload["candidates"]) <= 15


def test_weak_eleventh_candidate_does_not_expand_past_ten(client, headers, monkeypatch):
    """10 strong candidates (execution-shadow bonus, at entry) + 4 weak
    ones (no execution-shadow bonus, extended past entry) -- the weak
    ones must not fill slots 11-15."""
    strong = [f"STR{i:02d}" for i in range(10)]
    weak = [f"WEK{i:02d}" for i in range(4)]
    for t in strong:
        _seed(client, headers, t, scanned_at="2026-08-20T14:30:00Z")
    for t in weak:
        _seed(client, headers, t, scanned_at="2026-08-20T14:30:00Z")

    prices = {t: 100.0 for t in strong}
    prices.update({t: 103.5 for t in weak})  # extended, no near-entry bonus
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map(prices))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS if ticker in strong else [])

    payload = _get_queue(client, headers)
    displayed_tickers = {c["ticker"] for c in payload["candidates"] if c.get("current_review") is None}
    assert displayed_tickers == set(strong), "weak, low-scoring candidates must not consume slots 11-15"
    assert payload["diagnostics"]["needs_review_displayed_count"] == 10


def test_strong_eleventh_candidate_can_expand_to_fifteen(client, headers, monkeypatch):
    """12 candidates, all clearing every bonus and at entry (near-
    identical, strong scores) -- close enough to #10 that expansion
    should include them, up to the 15 ceiling."""
    tickers = [f"EXP{i:02d}" for i in range(12)]
    for t in tickers:
        _seed(client, headers, t, scanned_at="2026-08-20T14:30:00Z")
    prices = {t: 100.0 for t in tickers}
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map(prices))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    assert payload["diagnostics"]["needs_review_total_count"] == 12
    assert payload["diagnostics"]["needs_review_displayed_count"] == 12, "near-identical strong scores must be allowed to expand past 10"
    assert len(payload["candidates"]) == 12


# ==========================================================================
# Previously-reviewed candidates never consume a new-review slot
# ==========================================================================

def test_reviewed_candidates_do_not_consume_new_review_slots(client, headers, monkeypatch):
    strong = [f"REV{i:02d}" for i in range(11)]  # 11 needs-review candidates
    for t in strong:
        _seed(client, headers, t, scanned_at="2026-08-20T14:30:00Z")
    prices = {t: 100.0 for t in strong}
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map(prices))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    # Review (approve) the very top-ranked candidate BEFORE fetching the queue.
    first_pass = _get_queue(client, headers)
    top_ticker = first_pass["candidates"][0]["ticker"]
    client.post(f"/api/v1/scanner/candidates/{top_ticker}/visual-review", headers=headers, json={
        "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
        "clear_path_to_target": "yes", "lower_tf_confirmation": "yes",
        "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
    })

    second_pass = _get_queue(client, headers)
    displayed_needs_review = [c for c in second_pass["candidates"] if c.get("current_review") is None]
    assert len(displayed_needs_review) == 10, "the reviewed candidate freeing up must let an 11th still-needs-review candidate in, not shrink the queue"
    assert top_ticker not in {c["ticker"] for c in displayed_needs_review}
    # The reviewed candidate must still be present in the response (existing edit/re-review UX), just not counted against the cap.
    reviewed_entry = next((c for c in second_pass["candidates"] if c["ticker"] == top_ticker), None)
    assert reviewed_entry is not None
    assert reviewed_entry["current_review"]["decision"] == "approve"
    assert second_pass["diagnostics"]["needs_review_total_count"] == 10
    assert second_pass["diagnostics"]["already_reviewed_count"] == 1


# ==========================================================================
# Diagnostics block
# ==========================================================================

def test_diagnostics_counts_are_internally_consistent(client, headers, monkeypatch):
    _seed(client, headers, "GOOD", scanned_at="2026-08-20T14:30:00Z")
    _seed(client, headers, "FARAWAY", scanned_at="2026-08-20T14:30:00Z")
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"GOOD": 100.0, "FARAWAY": 105.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    payload = _get_queue(client, headers)
    diag = payload["diagnostics"]
    assert diag["total_candidate_count"] >= 2
    assert diag["eligible_pool_count"] == 2
    assert diag["needs_review_total_count"] == 2
    assert diag["needs_review_displayed_count"] == 2
    assert diag["already_reviewed_count"] == 0
    assert diag["promoted_from_near_miss_count"] == 1  # FARAWAY only
    assert diag["top_cutoff_score"] is not None


# ==========================================================================
# Downstream unaffected: GET /candidates/ranked (Stage 2) untouched
# ==========================================================================

def test_candidates_ranked_endpoint_unaffected(client, headers, monkeypatch):
    """GET /candidates/ranked must still use the OLD mechanism/gates --
    a candidate failing entry_proximity must still be ELIMINATED there,
    proving the two gate systems didn't get conflated. This endpoint
    (unlike the review-queue path) legitimately DOES hydrate options by
    design -- unrelated to this sprint -- so the option-hydration spy is
    overridden to a benign stub for this one test only."""
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: None)
    _seed(client, headers, "AMD")
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _quotes_map({"AMD": 105.0}))
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)

    ranked_resp = client.get("/api/v1/scanner/candidates/ranked", headers=headers)
    assert ranked_resp.status_code == 200
    ranked_payload = ranked_resp.json()
    assert ranked_payload["mechanism"] == router.RANKING_MECHANISM_VERSION
    assert not any(c["ticker"] == "AMD" for c in ranked_payload["candidates"]), (
        "Stage 2 must still hard-exclude on entry_proximity -- unchanged this sprint"
    )
