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
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
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


# ---------------------------------------------------------------------------
# Performance pass (2026-09-01 session): the reordered gate funnel.
# _compute_candidate_promotion is monkeypatched directly (dispatching per
# ticker) so each scenario can control exactly which cheap gate a candidate
# fails, independent of real daily-bar fixture engineering -- the point of
# these tests is proving the FUNNEL ordering, not re-testing structural_
# resistance's own math (already covered elsewhere).
# ---------------------------------------------------------------------------

def _fake_promotion(ticker, *, valid_target=True, risk_reward=2.5, rr_warning=False, favorable=2, unfavorable=1):
    return {
        "ticker": ticker, "source": "ma_pipeline", "direction": "long",
        "entry_price": 100.0, "stop": 96.0,
        "target": 110.0 if valid_target else None,
        "risk_reward": risk_reward if valid_target else None,
        "rr_warning": rr_warning, "no_valid_target": not valid_target,
        "promoted_at": "2026-09-01T00:00:00Z", "position_size": None,
        "atr14": 2.0, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 110.0 if valid_target else None, "raw_risk_reward": risk_reward if valid_target else None,
        "target_clamped": False, "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 95.0, "stop_source": "order_block",
        "displacement_score": 50.0, "displacement_label": "MODERATE", "displacement_components": {},
        "raw_magnitude_score": 50.0, "displacement_read": "neutral",
        "bos_confirmed": False, "bos_details": None,
        "macro_bias": "Macro Neutral", "macro_conflict": False, "choch_conflict": False, "choch_details": None,
        "sweep_confirmed": False, "sweep_details": None, "rejection_confirmed": False, "rejection_details": None,
        "location_percentile": 50.0, "location_label": "midrange", "location_alignment": "neutral",
        "confluence_signals": {"risk_reward": "favorable" if valid_target and not rr_warning else "unfavorable"},
        "confluence_counts": {"favorable": favorable, "unfavorable": unfavorable, "neutral": 3, "applicable": 7},
        "confluence_label": "some confluence",
        "last_daily_close": 100.0, "last_daily_close_at": "2026-08-31T21:00:00Z",
    }


@pytest.fixture()
def _funnel_spies(monkeypatch):
    """Records exactly which tickers reached each expensive call, without
    touching real network/data logic -- _compute_candidate_promotion is
    replaced outright (dispatch table below), quotes/execution-shadow are
    still real functions, just spied on."""
    promotions = {}  # filled per-test

    def _dispatch_promotion(candidate):
        ticker = str(candidate["ticker"]).upper()
        if ticker not in promotions:
            raise router.HTTPException(status_code=422, detail=f"no fixture for {ticker}")
        return dict(promotions[ticker])

    monkeypatch.setattr(router, "_compute_candidate_promotion", _dispatch_promotion)

    calls = {"batch_download": [], "latest_quote_single": [], "latest_quotes_batch": [], "execution_shadow": []}

    def _spy_batch_download(tickers, period, interval):
        calls["batch_download"].append(list(tickers))
        return {}

    def _spy_latest_quote_single(ticker):
        # Records rather than forbids: POST /visual-review's single-
        # candidate wrapper legitimately uses this path (no batching
        # benefit for one ticker). Tests that exercise ONLY the bulk
        # GET /candidates/review-queue endpoint assert this list stays
        # empty; tests that also call POST /visual-review don't.
        calls["latest_quote_single"].append(ticker)
        return {"price": 100.5, "timestamp": "2026-08-20T18:30:00Z", "source": "mock", "price_branch": "mid"}

    def _spy_latest_quotes_batch(previews):
        tickers = sorted({str(p.get("ticker") or "").upper() for p in previews})
        calls["latest_quotes_batch"].append(tickers)
        return {t: {"price": 100.5, "timestamp": "2026-08-20T18:30:00Z", "source": "mock", "price_branch": "mid"} for t in tickers}

    def _spy_execution_shadow(ticker):
        # Same proven-good bars shape as _mock_network above (a real
        # bullish reaction, not flat/boring bars) -- this needs to
        # actually PASS the real, unmocked _execution_shadow_from_bars
        # math, not just supply 15 bars.
        calls["execution_shadow"].append(ticker)
        return [
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

    monkeypatch.setattr(router, "_batch_download", _spy_batch_download)
    monkeypatch.setattr(router, "_latest_quote_for_ticker", _spy_latest_quote_single)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", _spy_latest_quotes_batch)
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", _spy_execution_shadow)

    return promotions, calls


def _seed_many(client, headers, tickers, signal="long", daily_regime="bullish"):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [
            {
                "ticker": ticker, "signal": signal, "entry_price": 100.0, "ema21_4h": 99.0,
                "daily_regime": daily_regime, "confidence": "high", "sma50_daily": 106.0, "sma200_daily": 104.0,
            }
            for ticker in tickers
        ],
    }
    created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert created.status_code == 200


def test_regime_misaligned_never_reaches_daily_bar_fetch(client, headers, _funnel_spies):
    promotions, calls = _funnel_spies
    # BEARTICKER is signal='long' but daily_regime is bearish -- a real,
    # if unusual, candidate shape (mirrors ma_pipeline's own confidence
    # split). Must be filtered before any network call at all.
    _seed_many(client, headers, ["GOODTICKER"], daily_regime="bullish")
    _seed_many(client, headers, ["BEARTICKER"], daily_regime="bearish")
    promotions["GOODTICKER"] = _fake_promotion("GOODTICKER")
    # No promotions["BEARTICKER"] entry at all -- if this ticker is ever
    # dispatched to _compute_candidate_promotion, the fixture raises.

    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    tickers_in_queue = {c["ticker"] for c in body["candidates"]}
    assert "BEARTICKER" not in tickers_in_queue
    assert "GOODTICKER" in tickers_in_queue
    assert all("BEARTICKER" not in chunk for chunk in calls["batch_download"])


def test_invalid_target_and_rr_warning_never_reach_quote_fetch(client, headers, _funnel_spies):
    promotions, calls = _funnel_spies
    _seed_many(client, headers, ["QUALIFIES", "NOTARGET", "BADRR"])
    promotions["QUALIFIES"] = _fake_promotion("QUALIFIES", valid_target=True, risk_reward=2.5, rr_warning=False)
    promotions["NOTARGET"] = _fake_promotion("NOTARGET", valid_target=False)
    promotions["BADRR"] = _fake_promotion("BADRR", valid_target=True, risk_reward=1.1, rr_warning=True)

    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()

    # All three DID reach the daily-bar fetch (that's how we learn they're
    # invalid in the first place) -- but only QUALIFIES should ever reach
    # the live-quote batch.
    all_prewarmed = {t for chunk in calls["batch_download"] for t in chunk}
    assert {"QUALIFIES", "NOTARGET", "BADRR"} <= all_prewarmed

    quoted_tickers = {t for chunk in calls["latest_quotes_batch"] for t in chunk}
    assert quoted_tickers == {"QUALIFIES"}
    assert calls["latest_quote_single"] == []  # the forbidden single-ticker path was never used

    tickers_in_queue = {c["ticker"] for c in body["candidates"]}
    assert tickers_in_queue == {"QUALIFIES"}


def test_qualifying_candidate_reaches_execution_shadow_and_is_ranked(client, headers, _funnel_spies):
    promotions, calls = _funnel_spies
    _seed_many(client, headers, ["WINNER"])
    promotions["WINNER"] = _fake_promotion("WINNER")

    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    assert calls["execution_shadow"] == ["WINNER"]
    entry = next(c for c in body["candidates"] if c["ticker"] == "WINNER")
    assert entry["rank"] == 1
    assert entry["execution_shadow_ok"] is True
    assert entry["entry_proximity_ok"] is True


def test_ranking_order_unchanged_for_equivalent_inputs(client, headers, _funnel_spies):
    """Same relative confluence/R:R inputs as Stage 2's own pure-function
    tests (favorable desc, unfavorable asc, R:R desc tiebreak) -- proving
    the reordered funnel produces the SAME order through the real endpoint,
    not just that rank_stage1_candidates itself is unchanged in isolation."""
    promotions, calls = _funnel_spies
    _seed_many(client, headers, ["ALPHA", "BETA", "GAMMA"])
    promotions["ALPHA"] = _fake_promotion("ALPHA", risk_reward=2.0, favorable=5, unfavorable=1)
    promotions["BETA"] = _fake_promotion("BETA", risk_reward=5.83, favorable=3, unfavorable=1)
    promotions["GAMMA"] = _fake_promotion("GAMMA", risk_reward=4.07, favorable=3, unfavorable=1)

    body = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    ordered = [c["ticker"] for c in sorted(body["candidates"], key=lambda c: c["rank"])]
    assert ordered == ["ALPHA", "BETA", "GAMMA"]


def test_setup_key_and_current_review_still_bind_correctly_after_reordering(client, headers, _funnel_spies):
    promotions, calls = _funnel_spies
    _seed_many(client, headers, ["BINDTEST"])
    promotions["BINDTEST"] = _fake_promotion("BINDTEST")

    first_pass = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next(c for c in first_pass["candidates"] if c["ticker"] == "BINDTEST")
    setup_key = entry["setup_key"]
    assert setup_key == "BINDTEST|ma_pipeline|long|96.00|110.00"
    assert entry["current_review"] is None

    client.post(
        "/api/v1/scanner/candidates/BINDTEST/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
        },
    )

    second_pass = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry2 = next(c for c in second_pass["candidates"] if c["ticker"] == "BINDTEST")
    assert entry2["setup_key"] == setup_key
    assert entry2["current_review"]["decision"] == "approve"


def test_batched_quote_fetch_used_instead_of_per_ticker_calls(client, headers, _funnel_spies):
    """The direct fix for the ~45s/364-call category profiling found:
    confirms a SINGLE batched quote call covers every precheck survivor,
    not one call per candidate."""
    promotions, calls = _funnel_spies
    tickers = [f"T{i}" for i in range(10)]
    _seed_many(client, headers, tickers)
    for t in tickers:
        promotions[t] = _fake_promotion(t)

    client.get("/api/v1/scanner/candidates/review-queue", headers=headers)
    assert len(calls["latest_quotes_batch"]) == 1  # one batched call, not ten
    assert set(calls["latest_quotes_batch"][0]) == set(tickers)
    assert calls["latest_quote_single"] == []
