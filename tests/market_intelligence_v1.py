"""Regression tests for derived Market Intelligence summaries."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def row(ticker, *, direction="LONG", grade="B", sector="Technology", stage="B+ TRADEABLE", entry_status="Tradeable", earnings_days=30):
    return {
        "ticker": ticker,
        "direction": direction,
        "setupGrade": grade,
        "entryStatus": entry_status,
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 112.0,
        "rr": 2.4,
        "sector": sector,
        "stockLocation": "Discount" if direction == "LONG" else "Premium",
        "quality": {"grade": grade, "score": 70, "freshness": "Fresh OB"},
        "trade_eval": {
            "trade_stage": stage,
            "a_plus_ready": stage == "A+ READY",
            "trigger_confirmed": stage == "A+ READY",
            "b_plus_tradeable": stage == "B+ TRADEABLE",
            "structure_quality": "CLEAN BOS",
            "displacement": "STRONG",
            "htf_aligned": True,
            "no_trade_reasons": [],
        },
        "option_plan": {"available": True},
        "earnings": {"loaded": True, "days_until": earnings_days},
    }


def scan_meta():
    return {
        "configured_universe_count": 4,
        "symbols_attempted": 4,
        "symbols_terminally_evaluated": 4,
        "symbols_with_setup": 4,
        "symbols_without_setup": 0,
        "symbols_operationally_failed": 0,
        "evaluation_coverage": 1,
        "evaluation_coverage_percent": 100,
        "result_yield": 1,
        "result_yield_percent": 100,
        "partial_result": False,
        "partial_result_reasons": [],
    }


def ranked_rows():
    rows = [
        row("AAA", direction="SHORT", grade="A", sector="Energy", stage="A+ READY", earnings_days=6),
        row("BBB", direction="SHORT", grade="B", sector="Energy", stage="B+ TRADEABLE", earnings_days=12),
        row("CCC", direction="LONG", grade="B", sector="Technology", stage="B+ TRADEABLE", entry_status="Near Entry"),
        row("DDD", direction="SHORT", grade="C", sector=None, stage="RANGE / NO TRADE"),
    ]
    scanner.apply_opportunity_ranking(rows, [])
    return rows


def test_market_intelligence_matches_scan_totals():
    rows = ranked_rows()
    result = scanner.build_market_intelligence(rows, scan_meta())
    intel = result["market_intelligence"]

    assert result["market_intelligence_version"] == scanner.MARKET_INTELLIGENCE_VERSION
    assert intel["ranked_opportunities"] == 4
    assert intel["coverage"]["symbols_attempted"] == 4
    assert intel["coverage"]["evaluation_coverage_percent"] == 100
    assert intel["direction_distribution"]["SHORT"]["count"] == 3
    assert intel["direction_distribution"]["LONG"]["count"] == 1
    assert sum(item["count"] for item in intel["sector_distribution"].values()) == 4
    assert intel["grade_distribution"]["B"]["count"] == 2
    assert intel["status_distribution"]["EARLY_ENTRY"]["count"] == 1
    assert intel["status_distribution"]["LOW_PRIORITY"]["count"] == 1


def test_market_intelligence_is_deterministic_and_preserves_ranking():
    rows = ranked_rows()
    before = copy.deepcopy([item["ranking"] for item in rows])
    generated_at = scanner._utc_now()

    first = scanner.build_market_intelligence(rows, scan_meta(), generated_at=generated_at)
    second = scanner.build_market_intelligence(rows, scan_meta(), generated_at=generated_at)

    assert first["market_intelligence"] == second["market_intelligence"]
    assert [item["ranking"] for item in rows] == before


def test_market_intelligence_today_take_and_earnings_summary():
    rows = ranked_rows()
    intel = scanner.build_market_intelligence(rows, scan_meta())["market_intelligence"]

    assert len(intel["todays_take"]) <= 5
    assert any("ranked opportunities" in item for item in intel["todays_take"])
    assert intel["market_tone"] == "Bearish Lean"
    assert intel["earnings_summary"]["within_7_days"] == 1
    assert intel["earnings_summary"]["within_14_days"] == 1


def test_cached_scan_meta_preserves_market_intelligence():
    rows = ranked_rows()
    meta = scan_meta()
    meta.update(scanner.build_market_intelligence(rows, meta))
    cached = {
        "generated_at": scanner._utc_now(),
        "rows": rows,
        "near_miss": [],
        "scan_meta": meta,
    }

    cached_meta = scanner._analysis_cache_meta(("default", None), cached, refreshing=False)

    assert cached_meta["market_intelligence_version"] == scanner.MARKET_INTELLIGENCE_VERSION
    assert cached_meta["market_intelligence"]["ranked_opportunities"] == 4
    assert cached_meta["market_intelligence_diagnostics"]["input_rows"] == 4


if __name__ == "__main__":
    test_market_intelligence_matches_scan_totals()
    test_market_intelligence_is_deterministic_and_preserves_ranking()
    test_market_intelligence_today_take_and_earnings_summary()
    test_cached_scan_meta_preserves_market_intelligence()
    print("Market Intelligence v1 tests passed")
