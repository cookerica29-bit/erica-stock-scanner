"""Regression coverage for bounded market-data failure diagnostics."""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import scanner


class PartialProvider:
    name = "yahoo"

    def download(self, tickers, **kwargs):
        index = pd.date_range("2026-01-01", periods=10, freq="D")
        columns = pd.MultiIndex.from_product([["AAPL"], ["Open", "High", "Low", "Close", "Volume"]])
        return pd.DataFrame(
            [[1, 2, 0.5, 1.5, 1000] for _ in range(10)],
            index=index,
            columns=columns,
        )


class RateLimitedProvider:
    name = "yahoo"

    def download(self, tickers, **kwargs):
        raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")


def reset_diagnostics():
    scanner._cache_snapshot(reset=True)


def test_records_missing_symbol_from_batch_response():
    reset_diagnostics()
    result = scanner._download_price_batch_raw(
        ["AAPL", "SQ"],
        period="60d",
        interval="4h",
        provider=PartialProvider(),
    )
    diagnostics = scanner._market_data_failure_diagnostics()

    assert "AAPL" in result
    assert "SQ" not in result
    assert diagnostics["total"] == 1
    assert diagnostics["summary"] == {"provider_error": 1}
    failure = diagnostics["failures"][0]
    assert failure["symbol"] == "SQ"
    assert failure["provider"] == "yahoo"
    assert failure["timeframe"] == "4H"
    assert failure["endpoint"] == "yahoo:download:4H"
    assert failure["retry_count"] == 0


def test_rate_limited_failures_are_bounded():
    reset_diagnostics()
    symbols = [f"T{i:02d}" for i in range(30)]
    result = scanner._download_price_batch_raw(
        symbols,
        period="60d",
        interval="4h",
        provider=RateLimitedProvider(),
    )
    diagnostics = scanner._market_data_failure_diagnostics()

    assert result == {}
    assert diagnostics["total"] == 30
    assert diagnostics["summary"] == {"rate_limited": 30}
    assert len(diagnostics["failures"]) == scanner.MAX_MARKET_DATA_FAILURE_DETAILS
    assert diagnostics["failures_truncated"] is True
    assert diagnostics["failures"][0]["raw_error_summary"].startswith("RuntimeError: Too Many Requests")


def test_partial_reason_uses_market_data_failure_count():
    reset_diagnostics()
    scanner._record_market_data_failure(
        symbol="SQ",
        provider="yahoo",
        interval="4h",
        endpoint="yahoo:download:4H",
        error_summary="Too Many Requests",
    )
    reasons = scanner._scan_partial_reasons(
        attempted=1,
        processed=0,
        tradeability_skipped=0,
        provider_metrics={},
        market_data_failures=scanner._market_data_failure_diagnostics(),
    )

    assert reasons == [{"stage": "market_data", "reason": "provider_symbol_fetch_failed", "count": 1}]


if __name__ == "__main__":
    test_records_missing_symbol_from_batch_response()
    test_rate_limited_failures_are_bounded()
    test_partial_reason_uses_market_data_failure_count()
    print("market_data_failure_diagnostics_v1 passed")
