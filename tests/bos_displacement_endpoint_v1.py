#!/usr/bin/env python3
"""Endpoint regression tests for BOS displacement shadow diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_endpoint_returns_warming_without_cache():
    original_auth = main._require_journal_admin_token
    original_snapshot = main.analysis_cache_snapshot
    main._require_journal_admin_token = lambda token: None
    main.analysis_cache_snapshot = lambda *args, **kwargs: None
    try:
        response = main.api_dev_bos_displacement_shadow(x_kairos_admin_token="ok")
    finally:
        main._require_journal_admin_token = original_auth
        main.analysis_cache_snapshot = original_snapshot
    assert response["ready"] is False
    assert response["status"] == "not_ready"
    assert "does not trigger scanning" in response["message"]


def test_endpoint_uses_cached_rows_and_does_not_trigger_work():
    calls = {"snapshot": 0, "candles": 0, "report": 0}
    original_auth = main._require_journal_admin_token
    original_snapshot = main.analysis_cache_snapshot
    original_candles = main._shadow_candle_data_for_rows
    original_report = main.build_bos_displacement_shadow_report

    def fake_snapshot(*args, **kwargs):
        calls["snapshot"] += 1
        return {
            "rows": [{"ticker": "AAPL", "timeframe": "1D", "trade_eval": {"displacement": "NONE"}}],
            "near_miss": [{"ticker": "MSFT", "timeframe": "4H", "trade_eval": {"displacement": "WEAK"}}],
            "generated_at": None,
            "scan_meta": {"configured_universe_count": 2},
        }

    def fake_candles(rows, limit):
        calls["candles"] += 1
        assert [row["ticker"] for row in rows] == ["AAPL", "MSFT"]
        assert limit == 2
        return {}

    def fake_report(rows, candle_data):
        calls["report"] += 1
        assert len(rows) == 2
        assert candle_data == {}
        return {
            "status": "ready",
            "message": "Shadow study only. Live strategy unchanged.",
            "processed_setups": 2,
            "all_traces": [],
        }

    main._require_journal_admin_token = lambda token: None
    main.analysis_cache_snapshot = fake_snapshot
    main._shadow_candle_data_for_rows = fake_candles
    main.build_bos_displacement_shadow_report = fake_report
    try:
        response = main.api_dev_bos_displacement_shadow(
            universe="default",
            include_all_traces=False,
            limit=2,
            x_kairos_admin_token="ok",
        )
    finally:
        main._require_journal_admin_token = original_auth
        main.analysis_cache_snapshot = original_snapshot
        main._shadow_candle_data_for_rows = original_candles
        main.build_bos_displacement_shadow_report = original_report

    assert response["ready"] is True
    assert response["processed_setups"] == 2
    assert response["live_strategy_changed"] is False
    assert "all_traces" not in response
    assert calls == {"snapshot": 1, "candles": 1, "report": 1}


def main_test() -> int:
    test_endpoint_returns_warming_without_cache()
    test_endpoint_uses_cached_rows_and_does_not_trigger_work()
    print("BOS displacement endpoint v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
