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
            assert preview["target"] == 110.0
            assert preview["stop"] < 100.0
            assert preview["risk_reward"] > 0
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
            assert promotion["target"] == 110.0
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


def test_preview_contract_normalizes_expiration_data_unavailable():
    import candidates_router

    normalized = candidates_router._normalize_preview_option_contract({
        "available": False,
        "execution": "No Clean Contract",
        "reason": "No option expirations available",
        "source": "unavailable",
    })
    assert normalized["execution"] == "Contract Data Unavailable"
    assert "retry later" in normalized["reason"]
    assert normalized["source"] == "data_unavailable"
    assert normalized["transient_unavailable"] is True


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
