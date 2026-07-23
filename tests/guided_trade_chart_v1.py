#!/usr/bin/env python3
"""Regression tests for guided trade chart candle delivery."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


class FakeRow(dict):
    pass


class FakeCandles:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def tail(self, limit):
        return FakeCandles(self.rows[-limit:])

    def iterrows(self):
        yield from self.rows


class FakeProvider:
    name = "fake-provider"

    def __init__(self, calls):
        self.calls = calls

    def download(self, ticker, **kwargs):
        self.calls.append((ticker, kwargs))
        return FakeCandles([
            ("2026-07-20T14:00:00Z", FakeRow(Open=100, High=104, Low=99, Close=103, Volume=1000)),
            ("2026-07-20T18:00:00Z", FakeRow(Open=103, High=106, Low=102, Close=105, Volume=1200)),
        ])


class FailingProvider:
    name = "failing-provider"

    def download(self, ticker, **kwargs):
        raise RuntimeError("provider unavailable")


def test_guided_chart_endpoint_serializes_supported_timeframes():
    calls = []
    original_builder = main.build_market_data_provider
    original_router = main.provider_name_for_timeframe
    main.provider_name_for_timeframe = lambda timeframe: f"provider-{timeframe}"
    main.build_market_data_provider = lambda name=None: FakeProvider(calls)
    try:
        client = TestClient(main.app)
        response = client.get("/api/chart/candles?symbol=oxy&timeframe=30M&limit=20")
    finally:
        main.build_market_data_provider = original_builder
        main.provider_name_for_timeframe = original_router

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["chart_component_version"] == "guided-trade-chart-v1"
    assert payload["symbol"] == "OXY"
    assert payload["timeframe"] == "30M"
    assert payload["period"] == "60d"
    assert payload["interval"] == "30m"
    assert payload["provider"] == "fake-provider"
    assert payload["candles_loaded"] == 2
    assert payload["candles"][0]["timestamp"] == "2026-07-20T14:00:00Z"
    assert payload["candles"][0]["open"] == 100.0
    assert calls == [("OXY", {"period": "60d", "interval": "30m", "progress": False, "auto_adjust": True, "group_by": "ticker"})]


def test_guided_chart_endpoint_failure_isolated_from_trade_plan():
    original_builder = main.build_market_data_provider
    main.build_market_data_provider = lambda name=None: FailingProvider()
    try:
        client = TestClient(main.app)
        response = client.get("/api/chart/candles?symbol=DOW&timeframe=4H")
    finally:
        main.build_market_data_provider = original_builder

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["symbol"] == "DOW"
    assert payload["timeframe"] == "4H"
    assert payload["candles"] == []
    assert payload["candles_loaded"] == 0
    assert payload["cache_status"] == "provider_error"
    assert payload["error"] == "RuntimeError"


def main_test() -> int:
    test_guided_chart_endpoint_serializes_supported_timeframes()
    test_guided_chart_endpoint_failure_isolated_from_trade_plan()
    print("Guided trade chart v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
