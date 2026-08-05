#!/usr/bin/env python3
"""Regression tests for non-breaking scanner summary payloads."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


GENERATION = "2026-08-03T22:00:00Z"


def row(
    ticker: str,
    rank: int,
    status: str,
    grade: str,
    ask: float | None,
    *,
    pricing_status: str = "ready",
    pricing_quality: str = "live_ask",
    heavy_note: str = "debug payload",
) -> dict:
    return {
        "ticker": ticker,
        "timeframe": "4H",
        "direction": "LONG",
        "price": 25.0,
        "entry": 24.5,
        "sl": 23.0,
        "tp1": 27.0,
        "tp2": 28.5,
        "tp3": 30.0,
        "setupGrade": grade,
        "setupGradeReason": f"{grade} setup summary",
        "setupStatus": "Early Confirmation",
        "setupStatusReason": "structure confirmation is forming",
        "entryStatus": "Near Entry",
        "stockTrend": "Bullish",
        "stockLocation": "Discount",
        "confirmationStarted": True,
        "confirmationReason": "bullish reaction from demand/support",
        "signal_timestamp": "2026-08-03T17:30:00Z",
        "ranking": {
            "rank": rank,
            "tier": "TOP_OPPORTUNITY" if rank <= 3 else "HIGH_PRIORITY",
            "score": 100 - rank,
            "status_bucket": status,
            "priority_bucket": rank,
            "positive_reasons": ["A-grade setup"],
            "cautions": [],
            "ranking_components": {"heavy": True},
            "version": "opportunity-ranking-v1",
        },
        "earnings": {"status": "NO_EARNINGS_RISK", "date": None, "days_until": None, "source": "test"},
        "option": {
            "type": "CALL",
            "strike": 25.0,
            "expiry": "2026-09-11",
            "requested_expiration": "2026-09-11",
            "resolved_expiration": "2026-09-18",
            "bid": ask - 0.1 if ask else None,
            "ask": ask,
            "mid": ask - 0.05 if ask else None,
            "mark": ask - 0.05 if ask else None,
            "spread": 0.1 if ask else None,
            "open_interest": 300,
            "volume": 50,
            "pricing_status": pricing_status,
            "pricing_quality": pricing_quality,
            "estimated_contract_cost": round(ask * 100, 2) if ask else None,
        },
        "option_pricing": {
            "status": pricing_status,
            "quality": pricing_quality,
            "estimated_contract_cost": round(ask * 100, 2) if ask else None,
            "bid": ask - 0.1 if ask else None,
            "ask": ask,
            "mid": ask - 0.05 if ask else None,
            "mark": ask - 0.05 if ask else None,
            "type": "CALL",
            "strike": 25.0,
            "requested_expiration": "2026-09-11",
            "resolved_expiration": "2026-09-18",
            "expiration": "2026-09-18",
            "expiry": "2026-09-18",
            "source": "yfinance_option_chain",
        },
        "option_plan": {
            "available": True,
            "type": "CALL",
            "preferred_strike": 25.0,
            "planned_entry": 24.5,
            "tp1": 27.0,
            "suggested_expiration": {"min_dte": 21, "max_dte": 35, "label": "21-35 DTE"},
            "source": "kairos_trade_plan",
        },
        "trade_eval": {"full": heavy_note, "coaching": ["long explanation"]},
        "quality": {"full": heavy_note, "coach_note": "long coaching text"},
        "market_regime_details": {"full": heavy_note},
        "marketRegimeDetails": {"full": heavy_note},
        "_scan_timing": {"total_ticker_ms": 123.4},
        "checklist": {"trendConfirmed": True},
    }


def full_payload() -> dict:
    rows = [
        row("AAA", 1, "ENTER_NOW", "A", 8.0),
        row("BBB", 2, "EARLY_ENTRY", "A", 2.0),
        row("CCC", 3, "ALMOST_READY", "B", 0.75),
        row("DDD", 4, "WAITING", "B", None, pricing_status="not_requested", pricing_quality="not_requested"),
    ]
    return {
        "rows": rows,
        "near_miss": [],
        "meta": {
            "generated_at": GENERATION,
            "ranking_generated_at": GENERATION,
            "qualified_rows": len(rows),
            "near_miss_rows": 0,
        },
    }


def best_overall(payload: dict) -> str:
    return sorted(payload["rows"], key=lambda item: item["ranking"]["rank"])[0]["ticker"]


def best_budget(payload: dict, budget: float) -> str | None:
    eligible = []
    for item in payload["rows"]:
        cost = ((item.get("option") or {}).get("estimated_contract_cost"))
        if isinstance(cost, (int, float)) and cost <= budget:
            eligible.append(item)
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: item["ranking"]["rank"])[0]["ticker"]


def test_full_response_remains_backward_compatible_and_summary_is_additive():
    original_scan_cached = main.scan_cached
    original_attach = main._attach_notification_metrics

    def fake_scan_cached(*_args, **_kwargs):
        return full_payload()

    main.scan_cached = fake_scan_cached
    main._attach_notification_metrics = lambda payload: payload
    try:
        client = TestClient(main.app)
        full = client.get("/api/scan?universe=default").json()
        summary = client.get("/api/scan?universe=default&view=summary").json()
        assert full["rows"][0]["trade_eval"]["full"] == "debug payload"
        assert "trade_eval" not in summary["rows"][0]
        assert "quality" not in summary["rows"][0]
        assert "market_regime_details" not in summary["rows"][0]
        assert "marketRegimeDetails" not in summary["rows"][0]
        assert "_scan_timing" not in summary["rows"][0]
        required = {
            "ticker", "timeframe", "direction", "current_price", "planned_entry",
            "stop", "tp1", "tp2", "tp3", "setupGrade", "display_status",
            "normalized_status_bucket", "ranking", "earnings", "option",
            "option_pricing", "pricing_status", "pricing_quality",
            "accessibility", "lazy_hydration", "setup_id", "scan_generation",
            "ranking_status_bucket", "execution_lifecycle",
            "execution_lifecycle_presentation_enabled", "mission_identity",
            "mission_workflow_bucket", "mission_workflow_enabled",
        }
        assert required.issubset(summary["rows"][0].keys())
        assert summary["rows"][0]["ranking_status_bucket"] == "ENTER_NOW"
        assert summary["rows"][0]["execution_lifecycle"]["ranking_status_bucket"] == "ENTER_NOW"
        assert summary["meta"]["view"] == "summary"
        assert summary["meta"]["summary_version"] == main.SUMMARY_SCAN_VIEW_VERSION
        assert summary["meta"]["stock_event_memory_presentation_v1"] is False
        assert summary["meta"]["stock_mission_workflow_v1"] is False
        assert summary["meta"]["mission_workflow"]["enabled"] is False
    finally:
        main.scan_cached = original_scan_cached
        main._attach_notification_metrics = original_attach


def test_summary_preserves_budget_and_market_outcomes():
    summary = main._summarize_scan_response(full_payload())
    assert best_overall(summary) == best_overall(full_payload())
    assert best_budget(summary, 100) == best_budget(full_payload(), 100) == "CCC"
    assert best_budget(summary, 250) == best_budget(full_payload(), 250) == "BBB"
    counts = summary["meta"]["today_market_counts"]
    assert counts["enter_now"] == 1
    assert counts["early_entry"] == 1
    assert counts["almost_ready"] == 1
    assert counts["waiting"] == 1
    assert summary["meta"]["qualified_count"] == 4
    assert summary["meta"]["near_miss_count"] == 0
    assert summary["rows"][0]["lazy_hydration"]["ticker"] == "AAA"
    assert summary["rows"][0]["lazy_hydration"]["expiration"] == "2026-09-18"


def test_detail_lookup_returns_matching_full_row_and_rejects_stale_generation():
    original_snapshot = main.analysis_cache_snapshot
    generated_at = datetime(2026, 8, 3, 22, 0, tzinfo=timezone.utc)
    payload = full_payload()
    summary = main._summarize_scan_response(payload)
    setup_id = summary["rows"][1]["setup_id"]

    def fake_snapshot(*_args, **_kwargs):
        return {
            "rows": payload["rows"],
            "near_miss": [],
            "generated_at": generated_at,
            "scan_meta": {},
        }

    main.analysis_cache_snapshot = fake_snapshot
    try:
        client = TestClient(main.app)
        ok = client.get(f"/api/scan/BBB?universe=default&detail=full&setup_id={setup_id}&generation={GENERATION}")
        assert ok.status_code == 200
        assert ok.json()["setup"]["ticker"] == "BBB"
        assert ok.json()["setup"]["trade_eval"]["full"] == "debug payload"
        stale = client.get(f"/api/scan/BBB?universe=default&detail=full&setup_id={setup_id}&generation=2026-08-03T21:00:00Z")
        assert stale.status_code == 409
        assert stale.json()["detail"]["reason"] == "stale_generation"
    finally:
        main.analysis_cache_snapshot = original_snapshot


if __name__ == "__main__":
    test_full_response_remains_backward_compatible_and_summary_is_additive()
    test_summary_preserves_budget_and_market_outcomes()
    test_detail_lookup_returns_matching_full_row_and_rejects_stale_generation()
    print("scan_summary_payload_v1 passed")
