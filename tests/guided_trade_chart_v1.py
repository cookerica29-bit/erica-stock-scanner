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
    def __init__(self, calls, name="fake-provider", rows=None):
        self.name = name
        self.calls = calls
        self.rows = rows if rows is not None else [
            ("2026-07-20T14:00:00Z", FakeRow(Open=100, High=104, Low=99, Close=103, Volume=1000)),
            ("2026-07-20T18:00:00Z", FakeRow(Open=103, High=106, Low=102, Close=105, Volume=1200)),
        ]

    def download(self, ticker, **kwargs):
        self.calls.append((self.name, ticker, kwargs))
        return FakeCandles(self.rows)


class FailingProvider:
    def __init__(self, name="failing-provider"):
        self.name = name

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
    assert payload["selected_provider"] == "fake-provider"
    assert payload["requested_timeframe"] == "30M"
    assert payload["normalized_timeframe"] == "30M"
    assert payload["fallback_used"] is False
    assert payload["provider_attempts"][0]["status"] == "ready"
    assert payload["candles_loaded"] == 2
    assert payload["candles"][0]["timestamp"] == "2026-07-20T14:00:00Z"
    assert payload["candles"][0]["open"] == 100.0
    assert calls == [("fake-provider", "OXY", {"period": "60d", "interval": "30m", "progress": False, "auto_adjust": True, "group_by": "ticker"})]


def test_guided_chart_4h_falls_back_to_alpaca_when_configured_provider_has_no_candles():
    calls = []
    original_builder = main.build_market_data_provider
    original_router = main.provider_name_for_timeframe

    def fake_builder(name=None):
        if name == main.YAHOO_PROVIDER_NAME:
            return FakeProvider(calls, name=main.YAHOO_PROVIDER_NAME, rows=[])
        return FakeProvider(calls, name=main.ALPACA_PROVIDER_NAME)

    main.provider_name_for_timeframe = lambda timeframe: main.YAHOO_PROVIDER_NAME
    main.build_market_data_provider = fake_builder
    try:
        client = TestClient(main.app)
        response = client.get("/api/chart/candles?symbol=XOM&timeframe=4H&limit=20")
    finally:
        main.build_market_data_provider = original_builder
        main.provider_name_for_timeframe = original_router

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["timeframe"] == "4H"
    assert payload["interval"] == "4h"
    assert payload["selected_provider"] == main.ALPACA_PROVIDER_NAME
    assert payload["fallback_used"] is True
    assert [attempt["provider"] for attempt in payload["provider_attempts"]] == [main.YAHOO_PROVIDER_NAME, main.ALPACA_PROVIDER_NAME]
    assert payload["provider_attempts"][0]["status"] == "unavailable"
    assert payload["provider_attempts"][0]["failure_reason"] == "no_candles"
    assert payload["provider_attempts"][1]["status"] == "ready"


def test_guided_chart_all_provider_failure_returns_clean_unavailable():
    original_builder = main.build_market_data_provider
    original_router = main.provider_name_for_timeframe
    main.provider_name_for_timeframe = lambda timeframe: main.YAHOO_PROVIDER_NAME
    main.build_market_data_provider = lambda name=None: FailingProvider(name=name or "unknown")
    try:
        client = TestClient(main.app)
        response = client.get("/api/chart/candles?symbol=DOW&timeframe=4H")
    finally:
        main.build_market_data_provider = original_builder
        main.provider_name_for_timeframe = original_router

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["symbol"] == "DOW"
    assert payload["timeframe"] == "4H"
    assert payload["candles"] == []
    assert payload["candles_loaded"] == 0
    assert payload["cache_status"] == "provider_unavailable"
    assert payload["failure_reason"] == "RuntimeError"
    assert [attempt["status"] for attempt in payload["provider_attempts"]] == ["error", "error"]


def test_chart_provider_fallback_does_not_change_scanner_provider_routing():
    original_router = main.provider_name_for_timeframe
    main.provider_name_for_timeframe = lambda timeframe: main.YAHOO_PROVIDER_NAME
    try:
        assert main.provider_name_for_timeframe("4H") == main.YAHOO_PROVIDER_NAME
        assert main._chart_provider_candidates("4H") == [main.YAHOO_PROVIDER_NAME, main.ALPACA_PROVIDER_NAME]
    finally:
        main.provider_name_for_timeframe = original_router


def main_test() -> int:
    test_guided_chart_endpoint_serializes_supported_timeframes()
    test_guided_chart_4h_falls_back_to_alpaca_when_configured_provider_has_no_candles()
    test_guided_chart_all_provider_failure_returns_clean_unavailable()
    test_chart_provider_fallback_does_not_change_scanner_provider_routing()
    print("Guided trade chart v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
