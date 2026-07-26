#!/usr/bin/env python3
"""Developer-only Alpaca migration audit tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import market_data  # noqa: E402
import provider_migration_audit as audit  # noqa: E402


TOKEN = "audit-token"


def sample_validation(max_ohlc=0.0, missing=0, yahoo_status="success", alpaca_status="success"):
    return {
        "providers": {
            "yahoo": {
                "status": yahoo_status,
                "duplicate_timestamps": 0,
                "out_of_order_timestamps": 0,
            },
            "alpaca": {
                "status": alpaca_status,
                "duplicate_timestamps": 0,
                "out_of_order_timestamps": 0,
            },
        },
        "comparison": {
            "completed_timestamp_matches": 20,
            "missing_in_alpaca_count": missing,
            "missing_in_yahoo_count": 0,
            "stats": {
                "max_ohlc_percent_difference": max_ohlc,
            },
        },
    }


def test_dependency_map_documents_trust_critical_consumers():
    rows = audit.consumer_dependency_map()
    names = {row["consumer"] for row in rows}
    assert "production stock scanner" in names
    assert "Guided Trade Charts" in names
    assert "replay engine" in names
    assert "Verified History" in names
    assert "Trade Intelligence" in names
    scanner = next(row for row in rows if row["consumer"] == "production stock scanner")
    assert scanner["controlling_flag"] == "STOCK_DATA_PROVIDER_PROFILE"
    assert scanner["trust_critical"] is True
    assert "scanner._price_cache" in scanner["cache"]


def test_migration_state_reports_profile_without_changing_it():
    previous = os.environ.get("STOCK_DATA_PROVIDER_PROFILE")
    try:
        os.environ["STOCK_DATA_PROVIDER_PROFILE"] = market_data.PROVIDER_PROFILE_PROPOSED_HYBRID
        report = audit.migration_state_report()
        assert report["production_routing_changed"] is False
        assert report["provider_profile"]["active_profile"] == market_data.PROVIDER_PROFILE_PROPOSED_HYBRID
        assert report["provider_profile"]["current_timeframe_routing"]["1D"] == market_data.ALPACA_PROVIDER_NAME
        assert any(item["consumer"] == "option chains and contract selection" for item in report["remaining_yahoo_paths"])
    finally:
        if previous is None:
            os.environ.pop("STOCK_DATA_PROVIDER_PROFILE", None)
        else:
            os.environ["STOCK_DATA_PROVIDER_PROFILE"] = previous


def test_raw_discrepancy_classification():
    assert audit._classify_raw_discrepancy(sample_validation(max_ohlc=0.0)) == "MATCH"
    assert audit._classify_raw_discrepancy(sample_validation(max_ohlc=0.01)) == "HARMLESS_DIFFERENCE"
    assert audit._classify_raw_discrepancy(sample_validation(max_ohlc=0.2)) == "EXPECTED_PROVIDER_VARIATION"
    assert audit._classify_raw_discrepancy(sample_validation(max_ohlc=1.2)) == "STRATEGY_SIGNIFICANT"
    assert audit._classify_raw_discrepancy(sample_validation(max_ohlc=0.0, missing=6)) == "DATA_QUALITY_CONCERN"
    missing_creds = sample_validation()
    missing_creds["providers"]["alpaca"]["error_classification"] = "missing_credentials"
    assert audit._classify_raw_discrepancy(missing_creds) == "UNRESOLVED"


def test_provider_comparison_report_uses_mocked_provider_calls():
    previous_validate = market_data.validate_candle_pair
    previous_strategy = audit._strategy_comparison_rows
    try:
        calls = []

        def fake_validate(ticker, period, interval):
            calls.append((ticker, period, interval))
            return sample_validation(max_ohlc=0.0)

        market_data.validate_candle_pair = fake_validate
        audit._strategy_comparison_rows = lambda symbols: [{
            "symbol": symbols[0],
            "classification": "MATCH",
            "comparison": {"differences": [], "material_differences": []},
        }]
        report = audit.provider_comparison_report(symbols=["SPY"], timeframes=["1D"], include_strategy=True, limit=1)
        assert report["production_routing_changed"] is False
        assert calls == [("SPY", "1y", "1d")]
        assert report["raw_candle_comparison"]["classification_counts"] == {"MATCH": 1}
        assert report["downstream_interpretation_comparison"]["classification_counts"] == {"MATCH": 1}
    finally:
        market_data.validate_candle_pair = previous_validate
        audit._strategy_comparison_rows = previous_strategy


def test_protected_audit_endpoints():
    previous_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    previous_report = main.migration_state_report
    previous_compare = main.provider_comparison_report
    try:
        os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN
        main.migration_state_report = lambda: {"version": audit.AUDIT_VERSION, "production_routing_changed": False}
        main.provider_comparison_report = lambda **kwargs: {
            "version": audit.AUDIT_VERSION,
            "production_routing_changed": False,
            "kwargs": kwargs,
        }
        client = TestClient(main.app)
        assert client.get("/api/dev/alpaca-migration-audit").status_code == 403
        assert client.get("/api/dev/alpaca-migration-audit", headers={"X-Kairos-Admin-Token": "wrong"}).status_code == 403
        audit_response = client.get("/api/dev/alpaca-migration-audit", headers={"X-Kairos-Admin-Token": TOKEN})
        assert audit_response.status_code == 200
        assert audit_response.json()["version"] == audit.AUDIT_VERSION
        compare_response = client.get(
            "/api/dev/provider-comparison?symbols=SPY&timeframes=1D&include_strategy=false&limit=1",
            headers={"X-Kairos-Admin-Token": TOKEN},
        )
        assert compare_response.status_code == 200
        kwargs = compare_response.json()["kwargs"]
        assert kwargs["symbols"] == ["SPY"]
        assert kwargs["timeframes"] == ["1D"]
        assert kwargs["include_strategy"] is False
        assert kwargs["limit"] == 1
    finally:
        main.migration_state_report = previous_report
        main.provider_comparison_report = previous_compare
        if previous_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_token


def main_test() -> int:
    test_dependency_map_documents_trust_critical_consumers()
    test_migration_state_reports_profile_without_changing_it()
    test_raw_discrepancy_classification()
    test_provider_comparison_report_uses_mocked_provider_calls()
    test_protected_audit_endpoints()
    print("Provider migration audit v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())

