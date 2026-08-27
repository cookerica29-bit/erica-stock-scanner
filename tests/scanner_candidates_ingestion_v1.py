import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client():
    import candidates_router

    app = FastAPI()
    app.include_router(candidates_router.router)
    return TestClient(app)


def _promotion_daily_frame():
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1
        rows.append({
            "Open": close + 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1_000_000,
        })
    rows[5]["Low"] = 99.0
    rows[10]["Low"] = 90.0
    rows[10]["Close"] = 91.0
    rows[10]["Open"] = 92.0
    rows[20]["High"] = 110.0
    rows[20]["Close"] = 109.0
    rows[20]["Open"] = 108.0
    return pd.DataFrame(rows, index=index)


def test_scanner_candidate_ingestion_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "candidates.db")
        os.environ["KAIROS_CANDIDATES_DB"] = db_path
        os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"

        import candidates_router

        previous_download = candidates_router._batch_download
        previous_alpaca_bars = candidates_router._alpaca_daily_bars_for_review
        previous_call_anthropic = candidates_router._call_anthropic_chart_review
        previous_best_contract = candidates_router._best_contract
        previous_latest_quote = candidates_router._latest_quote_for_ticker
        previous_latest_quotes = candidates_router._latest_quotes_for_previews
        previous_execution_bars = candidates_router._recent_4h_bars_for_execution_shadow
        candidates_router._batch_download = lambda tickers, period, interval: {
            str(tickers[0]).upper(): _promotion_daily_frame()
        }
        candidates_router._alpaca_daily_bars_for_review = lambda ticker: _promotion_daily_frame()
        candidates_router._best_contract = lambda ticker, direction, entry, **kwargs: {
            "available": True,
            "execution": "Fair",
            "type": "PUT" if direction == "SHORT" else "CALL",
            "strike": 100.0,
            "expiry": "2026-09-18",
            "dte": 29,
            "symbol": "MOCK",
            "source": "option_chain",
            "bid": 1.10,
            "ask": 1.20,
            "mid": 1.15,
            "mark": 1.15,
            "estimated_contract_cost": 120.0,
        }
        candidates_router._latest_quote_for_ticker = lambda ticker: {
            "price": 100.0,
            "timestamp": "2026-08-20T18:30:00Z",
            "source": "mock_latest_quote",
            "price_branch": "mid",
        }
        candidates_router._latest_quotes_for_previews = lambda previews: {
            str(preview.get("ticker") or "").upper(): {
                "price": 100.0,
                "timestamp": "2026-08-20T18:30:00Z",
                "source": "mock_latest_quote",
                "price_branch": "mid",
            }
            for preview in previews
        }
        candidates_router._recent_4h_bars_for_execution_shadow = lambda ticker: [
            {
                "time": f"2026-08-19T{hour:02d}:00:00Z",
                "open": 99.0 + (idx * 0.05),
                "high": 101.0 + (idx * 0.05),
                "low": 98.0 + (idx * 0.05),
                "close": 100.0 + (idx * 0.05),
                "volume": 1000,
            }
            for idx, hour in enumerate(range(11))
        ] + [
            {"time": "2026-08-20T02:00:00Z", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000},
            {"time": "2026-08-20T06:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1100},
            {"time": "2026-08-20T10:00:00Z", "open": 100.5, "high": 102.0, "low": 99.2, "close": 101.0, "volume": 1200},
            {"time": "2026-08-20T14:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.2, "volume": 1000},
        ]
        candidates_router._call_anthropic_chart_review = lambda prompt: (
            {
                "classification": "fresh_clean_structural_break",
                "rationale": "Price has broken structure cleanly and is holding above the reclaimed level. Treat this as an informational second opinion only.",
            },
            '{"content":[{"type":"text","text":"mock"}]}',
            "mock-claude",
        )

        client = _client()
        headers = {"X-API-Key": "test-scanner-key"}
        payload = {
            "source": "ma_pipeline",
            "scanned_at": "2026-08-20T14:30:00Z",
            "candidates": [
                {
                    "ticker": "nvda",
                    "signal": "long",
                    "entry_price": 217.25,
                    "ema21_4h": 214.1,
                    "daily_regime": "bullish",
                    "confidence": "high",
                    "sma50_daily": 207.3,
                    "sma200_daily": 195.08,
                },
                {"ticker": "BAD/FORMAT", "signal": "short"},
            ],
        }

        try:
            missing_key = client.post("/api/v1/scanner/candidates", json=payload)
            assert missing_key.status_code == 401

            wrong_key = client.post(
                "/api/v1/scanner/candidates",
                headers={"X-API-Key": "wrong"},
                json=payload,
            )
            assert wrong_key.status_code == 401

            created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
            assert created.status_code == 200
            assert created.json() == {
                "received": 2,
                "created": 1,
                "updated": 0,
                "rejected": [{"ticker": "BAD/FORMAT", "reason": "invalid ticker format"}],
            }

            listed = client.get("/api/v1/scanner/candidates", headers=headers).json()
            assert len(listed) == 1
            assert listed[0] == {
                "ticker": "NVDA",
                "source": "ma_pipeline",
                "signal": "long",
                "entry_price": 217.25,
                "ema21_4h": 214.1,
                "daily_regime": "bullish",
                "confidence": "high",
                "sma50_daily": 207.3,
                "sma200_daily": 195.08,
                "status": "new",
                "scanned_at": "2026-08-20T14:30:00+00:00",
                "updated_at": listed[0]["updated_at"],
                "source_universe": None,
            }

            update_payload = {
                "source": "ma_pipeline",
                "scanned_at": "2026-08-20T18:30:00Z",
                "candidates": [
                    {
                        "ticker": "NVDA",
                        "signal": "long",
                        "entry_price": 100.0,
                        "ema21_4h": 99.0,
                        "daily_regime": "bullish",
                        "confidence": "medium",
                        "sma50_daily": 106.0,
                        "sma200_daily": 104.0,
                    },
                ],
            }
            updated = client.post("/api/v1/scanner/candidates", headers=headers, json=update_payload)
            assert updated.status_code == 200
            assert updated.json() == {"received": 1, "created": 0, "updated": 1, "rejected": []}

            after_update = client.get("/api/v1/scanner/candidates", headers=headers).json()
            assert len(after_update) == 1
            assert after_update[0]["ticker"] == "NVDA"
            assert after_update[0]["signal"] == "long"
            assert after_update[0]["entry_price"] == 100.0
            assert after_update[0]["daily_regime"] == "bullish"

            import sqlite3

            conn = sqlite3.connect(db_path)
            try:
                current_count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
                history_count = conn.execute("SELECT COUNT(*) FROM candidate_history").fetchone()[0]
            finally:
                conn.close()
            assert current_count == 1
            assert history_count == 2

            previews = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)
            assert previews.status_code == 200
            preview_rows = previews.json()
            assert len(preview_rows) == 1
            preview = preview_rows[0]
            assert preview["ticker"] == "NVDA"
            assert preview["source"] == "ma_pipeline"
            assert preview["signal"] == "long"
            # The raw structural target (110.0) sits exactly on _promotion_daily_frame's
            # bar-20 gap-spike high (a ~10% gap up that gave back most of the move
            # within 3 sessions) -- structural_resistance correctly flags that as an
            # unreliable "gap_extreme" level and clamps the target just below it.
            assert preview["raw_target"] == 110.0
            assert preview["target"] < 110.0
            assert preview["target"] == pytest.approx(109.7977, abs=1e-3)
            assert preview["target_clamped"] is True
            assert preview["target_clamp_badge"] == "TARGET NEAR GAP SPIKE"
            assert preview["stop"] < 100.0
            assert preview["risk_reward"] > 0
            assert preview["raw_risk_reward"] > preview["risk_reward"]
            assert preview["rr_warning"] is False
            assert preview["no_valid_target"] is False
            assert preview["option_contract"]["type"] == "CALL"
            assert preview["option_contract"]["strike"] == 100.0
            assert preview["current_price"] == 100.0
            assert preview["entry_proximity_ok"] is True

            promoted = client.patch(
                "/api/v1/scanner/candidates/NVDA?source=ma_pipeline",
                headers=headers,
                json={"status": "active"},
            )
            assert promoted.status_code == 200
            promoted_payload = promoted.json()
            assert promoted_payload["ticker"] == "NVDA"
            assert promoted_payload["source"] == "ma_pipeline"
            assert promoted_payload["status"] == "active"
            promotion = promoted_payload["promotion"]
            assert promotion["direction"] == "long"
            assert promotion["entry_price"] == 100.0
            # Same gap-spike clamp as the plan-preview assertion above.
            assert promotion["raw_target"] == 110.0
            assert promotion["target"] == pytest.approx(109.7977, abs=1e-3)
            assert promotion["target_clamped"] is True
            assert promotion["stop"] < 100.0
            assert promotion["risk_reward"] > 0
            assert promotion["rr_warning"] is False
            assert promotion["no_valid_target"] is False
            assert promotion["min_target_atr_multiple"] == 2.0
            assert promotion["position_size"] is None
            assert promotion["stop"] == preview["stop"]
            assert promotion["target"] == preview["target"]
            assert promotion["risk_reward"] == preview["risk_reward"]

            chart_review = client.post(
                "/api/v1/scanner/candidates/NVDA/ai-chart-review?source=ma_pipeline",
                headers=headers,
            )
            assert chart_review.status_code == 200
            review_payload = chart_review.json()
            assert review_payload["ticker"] == "NVDA"
            assert review_payload["source"] == "ma_pipeline"
            assert review_payload["signal"] == "long"
            assert review_payload["classification"] == "fresh_clean_structural_break"
            assert "informational" in review_payload["caveat"].lower()
            assert review_payload["data_source"] == "alpaca_adjusted_daily_ohlcv"
            assert review_payload["model"] == "mock-claude"

            conn = sqlite3.connect(db_path)
            try:
                promotion_count = conn.execute("SELECT COUNT(*) FROM candidate_promotions").fetchone()[0]
                status_history = conn.execute(
                    """
                    SELECT ticker, source, previous_status, new_status, trigger
                    FROM candidate_status_history
                    ORDER BY id
                    """
                ).fetchall()
            finally:
                conn.close()
            assert promotion_count == 1
            assert status_history == [("NVDA", "ma_pipeline", "new", "active", "api_status_update")]

            active = client.get("/api/v1/scanner/candidates?status=active", headers=headers).json()
            assert len(active) == 1
            assert active[0]["ticker"] == "NVDA"

            promotions = client.get("/api/v1/scanner/candidate-promotions", headers=headers)
            assert promotions.status_code == 200
            promotion_rows = promotions.json()
            assert len(promotion_rows) == 1
            assert promotion_rows[0]["ticker"] == "NVDA"
            assert promotion_rows[0]["source"] == "ma_pipeline"
            assert promotion_rows[0]["direction"] == "long"
            assert promotion_rows[0]["rr_warning"] is False
            assert promotion_rows[0]["no_valid_target"] is False
            assert promotion_rows[0]["position_size"] is None

            restarted_client = _client()
            after_restart = restarted_client.get("/api/v1/scanner/candidates", headers=headers).json()
            assert len(after_restart) == 1
            assert after_restart[0]["ticker"] == "NVDA"
            assert after_restart[0]["status"] == "active"

            after_restart_promotions = restarted_client.get("/api/v1/scanner/candidate-promotions", headers=headers).json()
            assert len(after_restart_promotions) == 1
            assert after_restart_promotions[0]["ticker"] == "NVDA"

            chart_reviews = client.get("/api/v1/scanner/candidate-chart-reviews", headers=headers)
            assert chart_reviews.status_code == 200
            review_rows = chart_reviews.json()
            assert len(review_rows) == 1
            assert review_rows[0]["ticker"] == "NVDA"
            assert review_rows[0]["classification"] == "fresh_clean_structural_break"

            previews_after_review = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)
            assert previews_after_review.status_code == 200

            conn = sqlite3.connect(db_path)
            try:
                status_after_read_only_calls = conn.execute(
                    "SELECT status FROM candidates WHERE ticker='NVDA' AND source='ma_pipeline'"
                ).fetchone()[0]
                status_history_after_read_only_calls = conn.execute(
                    """
                    SELECT ticker, source, previous_status, new_status, trigger
                    FROM candidate_status_history
                    ORDER BY id
                    """
                ).fetchall()
            finally:
                conn.close()
            assert status_after_read_only_calls == "active"
            assert status_history_after_read_only_calls == [
                ("NVDA", "ma_pipeline", "new", "active", "api_status_update")
            ]
        finally:
            candidates_router._batch_download = previous_download
            candidates_router._alpaca_daily_bars_for_review = previous_alpaca_bars
            candidates_router._call_anthropic_chart_review = previous_call_anthropic
            candidates_router._best_contract = previous_best_contract
            candidates_router._latest_quote_for_ticker = previous_latest_quote
            candidates_router._latest_quotes_for_previews = previous_latest_quotes
            candidates_router._recent_4h_bars_for_execution_shadow = previous_execution_bars


def test_chart_review_parser_accepts_fenced_json():
    import candidates_router

    parsed = candidates_router._parse_chart_review_text(
        '```json\n{"classification":"choppy_range_bound","rationale":"Range-bound and overlapping."}\n```'
    )
    assert parsed == {
        "classification": "choppy_range_bound",
        "rationale": "Range-bound and overlapping.",
    }


def test_preview_contract_normalizes_no_options_chain():
    import candidates_router

    normalized = candidates_router._normalize_preview_option_contract({
        "available": False,
        "chain_available": False,
        "execution": "No Clean Contract",
        "reason": "No options chain available",
        "source": "unavailable",
    })
    assert normalized["execution"] == "No Options Chain"
    assert normalized["reason"] == "No options chain available"
    assert normalized["source"] == "unavailable"
    assert normalized.get("transient_unavailable") is None


def test_preview_contract_keeps_suggested_strike_when_chain_available():
    import candidates_router

    normalized = candidates_router._normalize_preview_option_contract({
        "available": True,
        "chain_available": True,
        "clean": False,
        "execution": "Suggested",
        "reason": "Best contract spread is too wide or unavailable",
        "source": "option_chain",
        "strike": 106.0,
        "type": "CALL",
        "expiry": "2026-09-18",
        "dte": 25,
    })
    assert normalized["execution"] == "Suggested"
    assert normalized["strike"] == 106.0
    assert normalized["available"] is True


def test_entry_proximity_uses_percent_or_atr_threshold():
    import candidates_router

    near = candidates_router._entry_proximity(
        entry_price=100.0,
        atr14=2.0,
        quote={"price": 101.0},
    )
    assert near["entry_proximity_ok"] is True
    assert near["entry_distance_pct"] == 1.0
    assert near["entry_distance_atr"] == 0.5

    far = candidates_router._entry_proximity(
        entry_price=100.0,
        atr14=2.0,
        quote={"price": 104.0},
    )
    assert far["entry_proximity_ok"] is False
    assert "away from entry" in far["entry_proximity_reason"]


def test_entry_proximity_distrusts_one_sided_quotes():
    """A bid_only/ask_only quote is real for display (see
    tests/current_quote_price_v1.py) but not reliable enough to gate a
    pass/fail decision on -- confirmed live via BRK.B showing a bogus
    5.81%/3.94 ATR "away from entry" off a bid_only quote of 473.31 while
    the real price was ~0.35% from entry. _entry_proximity() must refuse to
    compute a distance from either one-sided branch, same as no quote at all."""
    import candidates_router

    for branch in ("bid_only", "ask_only"):
        result = candidates_router._entry_proximity(
            entry_price=502.495,
            atr14=7.4005,
            quote={"price": 473.31, "price_branch": branch, "source": "alpaca_latest_quote"},
        )
        assert result["entry_proximity_ok"] is False
        assert result["current_price"] is None
        assert result["entry_distance_pct"] is None
        assert result["entry_distance_atr"] is None
        assert "one-sided" in result["entry_proximity_reason"]

    # A proper two-sided (midpoint) quote must still be trusted normally.
    clean = candidates_router._entry_proximity(
        entry_price=100.0,
        atr14=2.0,
        quote={"price": 101.0, "price_branch": "midpoint"},
    )
    assert clean["entry_proximity_ok"] is True
    assert clean["current_price"] == 101.0


def _near_miss_candidate(**overrides):
    row = {"ticker": "NEAR", "source": "ma_pipeline", "signal": "long", "daily_regime": "bullish"}
    row.update(overrides)
    return row


def test_gate_gap_report_excludes_categorical_short():
    import candidates_router

    report = candidates_router._gate_gap_report(
        _near_miss_candidate(signal="short", daily_regime="bearish"),
        {"signal": "short", "no_valid_target": False, "target": 90.0, "risk_reward": 3.0,
         "rr_warning": False, "entry_proximity_ok": True},
    )
    assert report["categorical_blocked"] is True
    assert report["categorical_reason"] == "Shorts are research-only"
    assert report["failing_count"] is None
    assert report["gaps"] == []


def test_gate_gap_report_tier1_rr_only():
    import candidates_router

    preview = {
        "signal": "long",
        "no_valid_target": False,
        "target": 110.0,
        "risk_reward": 1.47,
        "rr_warning": True,
        "entry_proximity_ok": True,
        "entry_proximity_threshold_pct": 1.5,
        "entry_proximity_threshold_atr": 0.5,
        "execution_shadow_checked": False,
        "execution_shadow_ok": None,
    }
    report = candidates_router._gate_gap_report(_near_miss_candidate(), preview)
    assert report["categorical_blocked"] is False
    assert report["failing_count"] == 1
    assert report["gaps"] == [{
        "condition": "risk_reward",
        "detail": "R:R 1.47 -- needs 1.50 (off by 0.03)",
    }]


def test_gate_gap_report_tier2_rr_and_proximity():
    import candidates_router

    preview = {
        "signal": "long",
        "no_valid_target": False,
        "target": 110.0,
        "risk_reward": 1.40,
        "rr_warning": True,
        "entry_proximity_ok": False,
        "entry_distance_pct": 1.2,
        "entry_distance_atr": 0.8,
        "entry_proximity_threshold_pct": 1.5,
        "entry_proximity_threshold_atr": 0.5,
        "execution_shadow_checked": False,
        "execution_shadow_ok": None,
    }
    report = candidates_router._gate_gap_report(_near_miss_candidate(), preview)
    assert report["failing_count"] == 2
    conditions = {gap["condition"] for gap in report["gaps"]}
    assert conditions == {"risk_reward", "entry_proximity"}
    proximity_gap = next(g for g in report["gaps"] if g["condition"] == "entry_proximity")
    assert proximity_gap["detail"] == (
        "Entry moved 1.20% / 0.80 ATR from scan -- outside proximity tolerance (max 1.5% / 0.50 ATR)"
    )


def test_gate_gap_report_execution_shadow_directional_expansion():
    import candidates_router

    preview = {
        "signal": "long",
        "no_valid_target": False,
        "target": 110.0,
        "risk_reward": 2.0,
        "rr_warning": False,
        "entry_proximity_ok": True,
        "execution_shadow_checked": True,
        "execution_shadow_ok": False,
        "execution_shadow_reason": "directional expansion only -0.22 ATR",
        "execution_shadow_diagnostics": {
            "has_recent_confirmation": True,
            "direction_expanded": False,
            "directional_expansion_atr": -0.22,
            "directional_expansion_required_atr": 0.75,
            "volume_confirmed": True,
            "holds_zone": True,
            "no_fresh_lower_low": True,
            "low_vol_bucket": False,
            "low_vol_net_move_ok": True,
        },
    }
    report = candidates_router._gate_gap_report(_near_miss_candidate(), preview)
    assert report["failing_count"] == 1
    assert report["gaps"] == [{
        "condition": "directional_expansion",
        "detail": "Execution: directional expansion -0.22 ATR -- needs +0.75 ATR (off by 0.97)",
    }]


def test_contract_gate_blocks_tiny_option_premium():
    import candidates_router

    reason = candidates_router._contract_block_reason({
        "available": True,
        "execution": "Excellent",
        "type": "CALL",
        "strike": 10.0,
        "expiry": "2026-09-18",
        "ask": 0.16,
        "mid": 0.15,
        "estimated_contract_cost": 16.0,
    })
    assert reason is not None
    assert "too thin" in reason
    assert "$0.16" in reason


def test_contract_gate_allows_clean_option_premium():
    import candidates_router

    reason = candidates_router._contract_block_reason({
        "available": True,
        "execution": "Good",
        "type": "CALL",
        "strike": 100.0,
        "expiry": "2026-09-18",
        "ask": 0.65,
        "mid": 0.60,
        "estimated_contract_cost": 65.0,
    })
    assert reason is None


def test_promotion_and_preview_ignore_contract_quality():
    """Contract quality (spread/liquidity/DTE/delta) is informational only --
    a low-confidence "Suggested" contract must not block ENTER_NOW preview
    status or promotion, as long as the other gates pass. This is the
    dual-enforcement check for the contract-quality-to-informational change:
    the same low-quality mock must not block the preview OR the PATCH."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "candidates.db")
        os.environ["KAIROS_CANDIDATES_DB"] = db_path
        os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"

        import candidates_router

        previous_download = candidates_router._batch_download
        previous_best_contract = candidates_router._best_contract
        previous_latest_quote = candidates_router._latest_quote_for_ticker
        previous_latest_quotes = candidates_router._latest_quotes_for_previews
        previous_execution_bars = candidates_router._recent_4h_bars_for_execution_shadow
        candidates_router._batch_download = lambda tickers, period, interval: {
            str(tickers[0]).upper(): _promotion_daily_frame()
        }
        # Deliberately a low-confidence, not-"clean" contract -- this is what
        # _best_contract now returns for a real chain that failed spread/
        # liquidity/DTE/score thresholds, instead of hiding the strike.
        candidates_router._best_contract = lambda ticker, direction, entry, **kwargs: {
            "available": True,
            "chain_available": True,
            "clean": False,
            "execution": "Suggested",
            "reason": "Best contract spread is too wide or unavailable",
            "type": "CALL",
            "strike": 100.0,
            "expiry": "2026-09-18",
            "dte": 25,
            "symbol": "MOCK",
            "source": "option_chain",
            "bid": None,
            "ask": None,
            "mid": None,
            "mark": None,
            "estimated_contract_cost": None,
        }
        candidates_router._latest_quote_for_ticker = lambda ticker: {
            "price": 100.0,
            "timestamp": "2026-08-24T14:30:00Z",
            "source": "mock_latest_quote",
            "price_branch": "mid",
        }
        candidates_router._latest_quotes_for_previews = lambda previews: {
            str(preview.get("ticker") or "").upper(): {
                "price": 100.0,
                "timestamp": "2026-08-24T14:30:00Z",
                "source": "mock_latest_quote",
                "price_branch": "mid",
            }
            for preview in previews
        }
        candidates_router._recent_4h_bars_for_execution_shadow = lambda ticker: [
            {
                "time": f"2026-08-19T{hour:02d}:00:00Z",
                "open": 99.0 + (idx * 0.05),
                "high": 101.0 + (idx * 0.05),
                "low": 98.0 + (idx * 0.05),
                "close": 100.0 + (idx * 0.05),
                "volume": 1000,
            }
            for idx, hour in enumerate(range(11))
        ] + [
            {"time": "2026-08-20T02:00:00Z", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000},
            {"time": "2026-08-20T06:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1100},
            {"time": "2026-08-20T10:00:00Z", "open": 100.5, "high": 102.0, "low": 99.2, "close": 101.0, "volume": 1200},
            {"time": "2026-08-20T14:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.2, "volume": 1000},
        ]

        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            created = client.post(
                "/api/v1/scanner/candidates",
                headers=headers,
                json={
                    "source": "ma_pipeline",
                    "scanned_at": "2026-08-20T14:30:00Z",
                    "candidates": [
                        {
                            "ticker": "LQD",
                            "signal": "long",
                            "entry_price": 100.0,
                            "ema21_4h": 99.0,
                            "daily_regime": "bullish",
                            "confidence": "high",
                            "sma50_daily": 90.0,
                            "sma200_daily": 80.0,
                        }
                    ],
                },
            )
            assert created.status_code == 200

            previews = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)
            assert previews.status_code == 200
            preview = previews.json()[0]
            # The low-quality contract must not block entry proximity / the
            # base ENTER_NOW-readiness the execution shadow check depends on.
            assert preview["entry_proximity_ok"] is True
            assert preview["execution_shadow_checked"] is True
            assert preview["option_contract"]["available"] is True
            assert preview["option_contract"]["execution"] == "Suggested"
            assert preview["option_contract"]["strike"] == 100.0
            assert preview["option_contract"]["clean"] is False

            promoted = client.patch(
                "/api/v1/scanner/candidates/LQD?source=ma_pipeline",
                headers=headers,
                json={"status": "active"},
            )
            assert promoted.status_code == 200, promoted.text
            assert promoted.json()["status"] == "active"
            assert promoted.json()["option_contract"]["execution"] == "Suggested"
        finally:
            candidates_router._batch_download = previous_download
            candidates_router._best_contract = previous_best_contract
            candidates_router._latest_quote_for_ticker = previous_latest_quote
            candidates_router._latest_quotes_for_previews = previous_latest_quotes
            candidates_router._recent_4h_bars_for_execution_shadow = previous_execution_bars


def test_near_miss_endpoint_ranks_by_tier_and_excludes_categorical():
    """End-to-end check of GET /candidate-near-misses: a long candidate
    failing exactly 2 execution-shadow sub-conditions shows up as tier 2
    with both real gaps, a short candidate (categorically blocked, not
    gradable) never appears at all, and the strict "Actionable only"
    concept (0 gaps) is untouched -- this is an additive ranking, not a
    replacement of the existing gate."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "candidates.db")
        os.environ["KAIROS_CANDIDATES_DB"] = db_path
        os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"

        import candidates_router

        previous_download = candidates_router._batch_download
        previous_best_contract = candidates_router._best_contract
        previous_latest_quote = candidates_router._latest_quote_for_ticker
        previous_latest_quotes = candidates_router._latest_quotes_for_previews
        previous_execution_bars = candidates_router._recent_4h_bars_for_execution_shadow
        candidates_router._batch_download = lambda tickers, period, interval: {
            str(tickers[0]).upper(): _promotion_daily_frame()
        }
        candidates_router._best_contract = lambda ticker, direction, entry, **kwargs: {
            "available": False, "chain_available": False, "execution": "No Options Chain",
            "reason": "No options chain available", "source": "unavailable",
        }
        candidates_router._latest_quote_for_ticker = lambda ticker: {
            "price": 100.0, "timestamp": "2026-08-24T14:30:00Z",
            "source": "mock_latest_quote", "price_branch": "midpoint",
        }
        candidates_router._latest_quotes_for_previews = lambda previews: {
            str(preview.get("ticker") or "").upper(): {
                "price": 100.0, "timestamp": "2026-08-24T14:30:00Z",
                "source": "mock_latest_quote", "price_branch": "midpoint",
            }
            for preview in previews
        }
        # One clean qualifying confirmation bar (strong reaction + expansion
        # + volume), then a pullback: fails exactly "fresh lower low" and
        # "directional expansion" (verified directly against
        # _execution_shadow_from_bars before baking into this test) -- 2
        # gaps -> tier 2.
        baseline = [
            {
                "time": f"2026-08-23T{hour:02d}:00:00Z",
                "open": 99.9, "high": 100.1, "low": 99.8, "close": 100.0, "volume": 100000,
            }
            for hour in range(10)
        ]
        confirmation = [
            {"time": "2026-08-24T00:00:00Z", "open": 100.0, "high": 102.3, "low": 99.9, "close": 102.2, "volume": 250000},
            {"time": "2026-08-24T04:00:00Z", "open": 102.2, "high": 102.4, "low": 101.8, "close": 102.0, "volume": 120000},
            {"time": "2026-08-24T08:00:00Z", "open": 102.0, "high": 102.1, "low": 101.0, "close": 101.2, "volume": 110000},
            {"time": "2026-08-24T12:00:00Z", "open": 101.2, "high": 101.3, "low": 100.3, "close": 100.5, "volume": 105000},
            {"time": "2026-08-24T16:00:00Z", "open": 100.5, "high": 100.6, "low": 99.9, "close": 100.1, "volume": 90000},
        ]
        near_miss_bars = baseline + confirmation
        candidates_router._recent_4h_bars_for_execution_shadow = lambda ticker: near_miss_bars

        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            created = client.post(
                "/api/v1/scanner/candidates",
                headers=headers,
                json={
                    "source": "ma_pipeline",
                    "scanned_at": "2026-08-20T14:30:00Z",
                    "candidates": [
                        {
                            "ticker": "NEARMISS",
                            "signal": "long",
                            "entry_price": 100.0,
                            "ema21_4h": 99.0,
                            "daily_regime": "bullish",
                            "confidence": "high",
                            "sma50_daily": 90.0,
                            "sma200_daily": 80.0,
                        },
                        {
                            "ticker": "SHORTONE",
                            "signal": "short",
                            "entry_price": 100.0,
                            "ema21_4h": 101.0,
                            "daily_regime": "bearish",
                            "confidence": "high",
                            "sma50_daily": 110.0,
                            "sma200_daily": 120.0,
                        },
                    ],
                },
            )
            assert created.status_code == 200

            resp = client.get("/api/v1/scanner/candidate-near-misses", headers=headers)
            assert resp.status_code == 200, resp.text
            rows = resp.json()
            tickers_shown = {row["ticker"] for row in rows}
            assert "SHORTONE" not in tickers_shown, "categorically blocked (short) candidate must never appear"
            near_miss = next((row for row in rows if row["ticker"] == "NEARMISS"), None)
            assert near_miss is not None
            assert near_miss["tier"] == 2
            assert near_miss["failing_count"] == 2
            conditions = {gap["condition"] for gap in near_miss["gaps"]}
            assert conditions == {"fresh_lower_low", "directional_expansion"}
            for gap in near_miss["gaps"]:
                assert gap["detail"], "every gap must state a real reason, not just a badge name"
        finally:
            candidates_router._batch_download = previous_download
            candidates_router._best_contract = previous_best_contract
            candidates_router._latest_quote_for_ticker = previous_latest_quote
            candidates_router._latest_quotes_for_previews = previous_latest_quotes
            candidates_router._recent_4h_bars_for_execution_shadow = previous_execution_bars


def test_promotion_blocks_when_price_is_not_near_entry():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "candidates.db")
        os.environ["KAIROS_CANDIDATES_DB"] = db_path
        os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"

        import candidates_router

        previous_download = candidates_router._batch_download
        previous_best_contract = candidates_router._best_contract
        previous_latest_quote = candidates_router._latest_quote_for_ticker
        candidates_router._batch_download = lambda tickers, period, interval: {
            str(tickers[0]).upper(): _promotion_daily_frame()
        }
        candidates_router._best_contract = lambda ticker, direction, entry, **kwargs: {
            "available": True,
            "execution": "Good",
            "type": "CALL",
            "strike": 100.0,
            "expiry": "2026-09-18",
            "dte": 25,
            "symbol": "MOCK",
            "source": "option_chain",
            "bid": 1.10,
            "ask": 1.20,
            "mid": 1.15,
            "mark": 1.15,
            "estimated_contract_cost": 120.0,
        }
        candidates_router._latest_quote_for_ticker = lambda ticker: {
            "price": 110.0,
            "timestamp": "2026-08-24T14:30:00Z",
            "source": "mock_latest_quote",
            "price_branch": "mid",
        }

        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            created = client.post(
                "/api/v1/scanner/candidates",
                headers=headers,
                json={
                    "source": "ma_pipeline",
                    "scanned_at": "2026-08-24T13:57:00Z",
                    "candidates": [
                        {
                            "ticker": "AAPL",
                            "signal": "long",
                            "entry_price": 100.0,
                            "ema21_4h": 99.0,
                            "daily_regime": "bullish",
                            "confidence": "high",
                            "sma50_daily": 90.0,
                            "sma200_daily": 80.0,
                        }
                    ],
                },
            )
            assert created.status_code == 200

            promoted = client.patch(
                "/api/v1/scanner/candidates/AAPL?source=ma_pipeline",
                headers=headers,
                json={"status": "active"},
            )
            assert promoted.status_code == 422
            assert "away from entry" in promoted.json()["detail"]
        finally:
            candidates_router._batch_download = previous_download
            candidates_router._best_contract = previous_best_contract
            candidates_router._latest_quote_for_ticker = previous_latest_quote


def test_short_candidate_cannot_promote_to_clean_dashboard():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "candidates.db")
        os.environ["KAIROS_CANDIDATES_DB"] = db_path
        os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"

        import candidates_router

        previous_download = candidates_router._batch_download
        candidates_router._batch_download = lambda tickers, period, interval: {
            str(tickers[0]).upper(): _promotion_daily_frame()
        }

        try:
            client = _client()
            headers = {"X-API-Key": "test-scanner-key"}
            created = client.post(
                "/api/v1/scanner/candidates",
                headers=headers,
                json={
                    "source": "ma_pipeline",
                    "scanned_at": "2026-08-24T13:57:00Z",
                    "candidates": [
                        {
                            "ticker": "ORCL",
                            "signal": "short",
                            "entry_price": 100.0,
                            "ema21_4h": 101.0,
                            "daily_regime": "bearish",
                            "confidence": "high",
                            "sma50_daily": 98.0,
                            "sma200_daily": 120.0,
                        }
                    ],
                },
            )
            assert created.status_code == 200

            promoted = client.patch(
                "/api/v1/scanner/candidates/ORCL?source=ma_pipeline",
                headers=headers,
                json={"status": "active"},
            )
            assert promoted.status_code == 422
            assert "research-only" in promoted.json()["detail"]

            listed = client.get("/api/v1/scanner/candidates", headers=headers).json()
            assert listed[0]["status"] == "new"
        finally:
            candidates_router._batch_download = previous_download


if __name__ == "__main__":
    test_scanner_candidate_ingestion_lifecycle()
    test_chart_review_parser_accepts_fenced_json()
    test_preview_contract_normalizes_expiration_data_unavailable()
    test_entry_proximity_uses_percent_or_atr_threshold()
    test_contract_gate_blocks_tiny_option_premium()
    test_contract_gate_allows_clean_option_premium()
    test_promotion_blocks_when_price_is_not_near_entry()
    test_short_candidate_cannot_promote_to_clean_dashboard()
    print("scanner_candidates_ingestion_v1 passed")
