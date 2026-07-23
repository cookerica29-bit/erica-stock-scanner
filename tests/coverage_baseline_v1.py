#!/usr/bin/env python3
"""Regression tests for discovered-universe coverage baseline metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def assert_utc_z_timestamp(value):
    assert isinstance(value, str)
    assert value.endswith("Z")
    parsed = __import__("datetime").datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def row(**overrides):
    base = {
        "ticker": "AAA",
        "setupGrade": "A",
        "entryStatus": "Tradeable",
        "setupStatus": "Trend Resumption Confirmed",
        "trade_eval": {
            "trade_stage": "A+ READY",
            "trigger_confirmed": True,
            "no_trade_reasons": [],
        },
        "best_contract": {
            "available": True,
            "source": "option_chain",
        },
        "option_plan": {
            "available": True,
            "type": "CALL",
            "preferred_strike": 105,
            "suggested_expiration": {"label": "21–35 DTE"},
            "expected_hold": {"label": "7–12 Trading Days", "fallback_used": True},
            "confidence": {"label": "★★★★☆"},
            "strike_rounding_increment": 2.5,
        },
    }
    base.update(overrides)
    return base


def base_meta(**overrides):
    meta = {
        "configured_universe_count": 6,
        "symbols_successfully_processed": 4,
        "symbols_terminally_evaluated": 6,
        "symbols_with_setup": 4,
        "symbols_without_setup": 1,
        "symbols_intentionally_rejected": 1,
        "symbols_operationally_failed": 0,
        "symbols_not_evaluated": 0,
        "evaluation_coverage": 1.0,
        "evaluation_coverage_percent": 100.0,
        "result_yield": 0.6667,
        "result_yield_percent": 66.67,
        "tradeability_skipped": 1,
        "tradeability_skip_reasons": {"options data unknown": 1},
        "no_setup_or_failed_count": 1,
        "symbols_omitted_or_rejected": 2,
        "scan_started_at": "2026-07-22T12:00:00Z",
        "scan_completed_at": "2026-07-22T12:00:05Z",
        "scan_duration_ms": 5000.0,
        "partial_result": False,
        "partial_result_reasons": [],
        "performance": {
            "market_data_fetch_ms": 1200.0,
            "strategy_evaluation_ms": 800.0,
            "symbols_per_second": 0.8,
            "cache_hit_rate": 0.5,
            "peak_worker_count": 4,
        },
    }
    meta.update(overrides)
    return meta


def base_context():
    return {
        "universe_source": "discovered",
        "universe_generated_at": "2026-07-22T11:55:00Z",
        "universe_symbol_count": 550,
        "discovery": {
            "final_admitted_symbol_count": 550,
            "effective_cap": 550,
        },
    }


def test_discovered_scan_metrics_count_canonical_fields_independently():
    rows = [
        row(ticker="AAA", setupGrade="A"),
        row(
            ticker="BBB",
            setupGrade="B",
            entryStatus="Near Entry",
            trade_eval={"trade_stage": "B+ TRADEABLE", "b_plus_tradeable": True, "no_trade_reasons": []},
            best_contract={"available": False, "source": "fallback", "fallback_reason": "option chain unavailable"},
        ),
    ]
    near_miss = [
        row(
            ticker="CCC",
            setupGrade="C",
            entryStatus="Too Far",
            setupStatus="Pullback Active",
            trade_eval={"trade_stage": "RANGE / NO TRADE", "no_trade_reasons": ["Poor premium/discount location"]},
            best_contract={"available": False, "source": "not_evaluated", "reason": "Best contract not evaluated for non-actionable setup"},
        ),
        row(
            ticker="DDD",
            setupGrade=None,
            entryStatus="Waiting",
            setupStatus="Pullback Active",
            trade_eval={"trade_stage": "BUILDING / WATCHLIST", "no_trade_reasons": []},
            best_contract={"available": False, "source": "loading", "loading": True, "reason": "Options data is loading in the background"},
        ),
    ]
    snapshot = scanner.build_discovered_scan_coverage_snapshot(rows, near_miss, base_meta(), base_context())
    assert snapshot["scan"]["universe_source"] == "discovered"
    assert snapshot["scan"]["universe_symbol_count"] == 550
    assert snapshot["scan"]["symbols_requested"] == 6
    assert snapshot["scan"]["symbols_processed"] == 4
    assert snapshot["scan"]["symbols_returned"] == 4
    assert snapshot["scan"]["symbols_terminally_evaluated"] == 6
    assert snapshot["scan"]["symbols_with_setup"] == 4
    assert snapshot["scan"]["symbols_without_setup"] == 1
    assert snapshot["scan"]["symbols_intentionally_rejected"] == 1
    assert snapshot["scan"]["symbols_operationally_failed"] == 0
    assert snapshot["scan"]["symbols_not_evaluated"] == 0
    assert snapshot["scan"]["evaluation_coverage_percent"] == 100.0
    assert snapshot["scan"]["result_yield_percent"] == 66.67
    assert snapshot["scan"]["symbols_skipped"] == 1
    assert snapshot["scan"]["symbols_failed"] == 0
    assert snapshot["scan"]["partial_result"] is False
    assert snapshot["scan"]["partial_result_reasons"] == []
    assert snapshot["scan"]["performance"]["symbols_per_second"] == 0.8
    assert_utc_z_timestamp(snapshot["generated_at"])
    assert_utc_z_timestamp(snapshot["scan"]["scan_started_at"])
    assert_utc_z_timestamp(snapshot["scan"]["scan_completed_at"])

    assert snapshot["stage_distribution"]["A+ READY"] == 1
    assert snapshot["stage_distribution"]["B+ TRADEABLE"] == 1
    assert snapshot["stage_distribution"]["RANGE / NO TRADE"] == 1
    assert snapshot["stage_distribution"]["BUILDING / WATCHLIST"] == 1
    assert snapshot["grade_distribution"] == {"A": 1, "B": 1, "C": 1, "unknown": 1}
    assert snapshot["canonical_field_distributions"]["trade_stage"]["A+ READY"] == 1
    assert snapshot["canonical_field_distributions"]["entry_status"]["Too Far"] == 1

    assert snapshot["contract_distribution"]["suggested contract available"] == 1
    assert snapshot["contract_distribution"]["potential/fallback contract only"] == 1
    assert snapshot["contract_distribution"]["contract unavailable"] == 1
    assert snapshot["contract_distribution"]["option data unknown or temporarily failed"] == 1
    assert snapshot["option_plan_diagnostics"]["option_plans_generated"] == 4
    assert snapshot["option_plan_diagnostics"]["expected_hold_fallback_used"] == 4
    assert snapshot["option_plan_diagnostics"]["strike_rounding_distribution"]["2.5"] == 4
    assert snapshot["option_plan_diagnostics"]["expiration_window_distribution"]["21–35 DTE"] == 4
    assert snapshot["option_plan_diagnostics"]["confidence_distribution"]["★★★★☆"] == 4


def test_legitimate_tradeability_skip_does_not_make_scan_partial():
    snapshot = scanner.build_discovered_scan_coverage_snapshot(
        [],
        [],
        base_meta(
            configured_universe_count=2,
            symbols_successfully_processed=0,
            tradeability_skipped=2,
            symbols_skipped=2,
            no_setup_or_failed_count=0,
            symbols_failed=0,
            symbols_omitted_or_rejected=2,
            tradeability_skip_reasons={"no price data": 2},
            partial_result=False,
            partial_result_reasons=[],
        ),
        base_context(),
    )
    assert snapshot["scan"]["symbols_requested"] == 2
    assert snapshot["scan"]["symbols_skipped"] == 2
    assert snapshot["scan"]["symbols_failed"] == 0
    assert snapshot["scan"]["partial_result"] is False
    assert snapshot["scan"]["partial_result_reasons"] == []


def test_partial_result_reason_classifies_processing_failures():
    reasons = scanner._scan_partial_reasons(
        attempted=4,
        processed=2,
        tradeability_skipped=1,
        processing_failures=[{"ticker": "BAD", "reason": "internal exception"}],
        provider_metrics={"alpaca_bar_symbols_failed": 0, "alpaca_max_pages_exceeded_count": 0},
    )
    assert reasons == [{"stage": "strategy_evaluation", "reason": "internal exception", "count": 1}]


def test_no_setup_outcomes_do_not_make_scan_partial():
    reasons = scanner._scan_partial_reasons(
        attempted=100,
        processed=90,
        tradeability_skipped=0,
        processing_failures=[
            {"ticker": f"NOSETUP{i}", "reason": "symbol returned no setup"}
            for i in range(10)
        ],
        provider_metrics={"alpaca_bar_symbols_failed": 0, "alpaca_max_pages_exceeded_count": 0},
    )
    assert reasons == []

    snapshot = scanner.build_discovered_scan_coverage_snapshot(
        [row(ticker=f"ROW{i}") for i in range(90)],
        [],
        base_meta(
            configured_universe_count=100,
            symbols_successfully_processed=90,
            symbols_terminally_evaluated=100,
            symbols_with_setup=90,
            symbols_without_setup=10,
            symbols_intentionally_rejected=0,
            symbols_operationally_failed=0,
            symbols_not_evaluated=0,
            evaluation_coverage=1.0,
            evaluation_coverage_percent=100.0,
            result_yield=0.9,
            result_yield_percent=90.0,
            tradeability_skipped=0,
            symbols_skipped=0,
            no_setup_or_failed_count=10,
            symbols_failed=0,
            symbols_omitted_or_rejected=10,
            partial_result=False,
            partial_result_reasons=[],
        ),
        base_context(),
    )
    assert snapshot["scan"]["partial_result"] is False
    assert snapshot["scan"]["symbols_terminally_evaluated"] == 100
    assert snapshot["scan"]["symbols_without_setup"] == 10
    assert snapshot["scan"]["symbols_operationally_failed"] == 0
    assert snapshot["scan"]["evaluation_coverage_percent"] == 100.0
    assert snapshot["scan"]["result_yield_percent"] == 90.0


def test_provider_failures_keep_scan_partial():
    reasons = scanner._scan_partial_reasons(
        attempted=100,
        processed=98,
        tradeability_skipped=0,
        processing_failures=[],
        provider_metrics={"alpaca_bar_symbols_failed": 2, "alpaca_max_pages_exceeded_count": 0},
    )
    assert reasons == [{"stage": "market_data", "reason": "provider_symbol_fetch_failed", "count": 2}]


def test_unknown_option_data_is_not_counted_as_confirmed_no_options():
    snapshot = scanner.build_discovered_scan_coverage_snapshot(
        [],
        [
            row(
                ticker="UNK",
                setupGrade="B",
                entryStatus="Waiting",
                trade_eval={"trade_stage": "BUILDING / WATCHLIST", "no_trade_reasons": []},
                best_contract={"available": False, "source": "loading", "loading": True, "reason": "Options data is loading in the background"},
            )
        ],
        base_meta(tradeability_skip_reasons={"options data unknown": 1}),
        base_context(),
    )
    assert snapshot["contract_distribution"]["option data unknown or temporarily failed"] == 1
    assert snapshot["blocker_distribution"].get("no options", 0) == 0
    assert snapshot["provider_failures"]["options data unknown"] == 1
    assert snapshot["provider_failures"]["Options data is loading in the background"] == 1


def test_coverage_snapshot_store_returns_latest_completed_snapshot():
    previous = scanner.coverage_baseline_snapshot()
    snapshot = scanner.build_discovered_scan_coverage_snapshot(
        [row()],
        [],
        base_meta(configured_universe_count=1, symbols_successfully_processed=1, tradeability_skipped=0, no_setup_or_failed_count=0, symbols_omitted_or_rejected=0),
        base_context(),
    )
    try:
        scanner._store_coverage_baseline_snapshot(snapshot)
        latest = scanner.coverage_baseline_snapshot()
        assert latest["scan"]["symbols_requested"] == 1
        assert latest["discovery"]["effective_cap"] == 550
    finally:
        scanner._store_coverage_baseline_snapshot(previous or {})


def main() -> int:
    test_discovered_scan_metrics_count_canonical_fields_independently()
    test_legitimate_tradeability_skip_does_not_make_scan_partial()
    test_partial_result_reason_classifies_processing_failures()
    test_no_setup_outcomes_do_not_make_scan_partial()
    test_provider_failures_keep_scan_partial()
    test_unknown_option_data_is_not_counted_as_confirmed_no_options()
    test_coverage_snapshot_store_returns_latest_completed_snapshot()
    print("Coverage baseline v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
