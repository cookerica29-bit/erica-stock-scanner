"""Regression tests for presentation-only opportunity ranking."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def row(ticker, *, grade="B", stage="BUILDING / WATCHLIST", entry_status="Waiting", **overrides):
    base = {
        "ticker": ticker,
        "direction": "LONG",
        "setupGrade": grade,
        "entryStatus": entry_status,
        "price": 100.0,
        "distanceFromEntryAtr": 0.0 if entry_status == "Tradeable" else 0.4 if entry_status == "Near Entry" else 0.8,
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 112.0,
        "rr": 2.4,
        "stockLocation": "Discount",
        "in_ob": True,
        "confirmationStarted": stage == "A+ READY",
        "quality": {"grade": grade, "score": 71, "freshness": "Fresh OB"},
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
        "earnings": {"loaded": True, "days_until": 30},
    }
    base.update(overrides)
    return base


STRATEGY_FIELDS = [
    "ticker",
    "direction",
    "setupGrade",
    "entryStatus",
    "entry",
    "sl",
    "tp1",
    "tp2",
    "tp3",
    "rr",
    "trade_eval",
    "option_plan",
]


def stripped(rows):
    return [{field: copy.deepcopy(item.get(field)) for field in STRATEGY_FIELDS} for item in rows]


def test_ranking_preserves_rows_and_strategy_fields():
    rows = [
        row("AAA", grade="A", stage="A+ READY", entry_status="Tradeable"),
        row("BBB", grade="B", stage="B+ TRADEABLE", entry_status="Tradeable"),
    ]
    near_miss = [row("CCC", grade="C", stage="RANGE / NO TRADE")]
    before_rows = stripped(rows)
    before_near = stripped(near_miss)

    meta = scanner.apply_opportunity_ranking(rows, near_miss)

    assert meta["ranking_version"] == scanner.OPPORTUNITY_RANKING_VERSION
    assert meta["ranked_setup_count"] == 3
    assert {item["ticker"] for item in rows + near_miss} == {"AAA", "BBB", "CCC"}
    assert stripped(sorted(rows, key=lambda item: item["ticker"])) == sorted(before_rows, key=lambda item: item["ticker"])
    assert stripped(sorted(near_miss, key=lambda item: item["ticker"])) == sorted(before_near, key=lambda item: item["ticker"])
    assert [item["ranking"]["rank"] for item in rows + near_miss] == [1, 2, 3]


def test_priority_buckets_dominate_scores():
    rows = [
        row("WAITA", grade="A", stage="BUILDING / WATCHLIST", entry_status="Waiting", rr=8.0),
        row("NOWB", grade="B", stage="A+ READY", entry_status="Tradeable", rr=1.5),
        row("ALMB", grade="B", stage="B+ TRADEABLE", entry_status="Near Entry", rr=9.0),
        row("NOWA", grade="A", stage="A+ READY", entry_status="Tradeable", rr=1.5),
    ]

    scanner.apply_opportunity_ranking(rows, [])
    ordered = [item["ticker"] for item in rows]

    assert ordered.index("NOWA") < ordered.index("NOWB")
    assert ordered.index("NOWB") < ordered.index("ALMB")
    assert ordered.index("ALMB") < ordered.index("WAITA")
    assert rows[0]["ranking"]["status_bucket"] == "ENTER_NOW"


def test_enter_now_requires_executable_entry_state():
    beyond_short = row(
        "WFC",
        grade="A",
        stage="A+ READY",
        entry_status="Waiting",
        direction="SHORT",
        entry=89.22,
        sl=90.21,
        tp1=87.24,
        price=88.00,
        distanceFromEntryAtr=0.8,
    )
    before_short = row(
        "WYNN",
        grade="B",
        stage="A+ READY",
        entry_status="Tradeable",
        direction="SHORT",
        entry=99.92,
        sl=102.99,
        tp1=93.78,
        price=100.10,
        distanceFromEntryAtr=0.05,
    )
    executable_short = row(
        "EXEC",
        grade="A",
        stage="A+ READY",
        entry_status="Tradeable",
        direction="SHORT",
        entry=99.92,
        sl=102.99,
        tp1=93.78,
        price=99.90,
        distanceFromEntryAtr=0.02,
    )
    early_touch = row(
        "RETEST",
        grade="A",
        stage="A+ READY",
        entry_status="Tradeable",
        direction="SHORT",
        entry=99.92,
        sl=102.99,
        tp1=93.78,
        price=99.90,
        distanceFromEntryAtr=0.02,
        early_entry_shadow={"state": "WAITING_FOR_RETEST"},
    )
    missed_short = row(
        "MISS",
        grade="A",
        stage="A+ READY",
        entry_status="Too Far",
        direction="SHORT",
        entry=99.92,
        sl=102.99,
        tp1=93.78,
        price=98.00,
        distanceFromEntryAtr=1.1,
    )

    assert scanner._ranking_status_bucket(beyond_short) == "ALMOST_READY"
    assert scanner._ranking_status_bucket(before_short) == "ALMOST_READY"
    assert scanner._ranking_status_bucket(executable_short) == "ENTER_NOW"
    assert scanner._ranking_status_bucket(early_touch) == "WAITING_FOR_RETEST"
    assert scanner._ranking_status_bucket(missed_short) == "MISSED_ENTRY"


def test_grade_a_beats_grade_b_inside_equivalent_status():
    rows = [
        row("BSET", grade="B", stage="B+ TRADEABLE", entry_status="Tradeable", rr=5.0),
        row("ASET", grade="A", stage="B+ TRADEABLE", entry_status="Tradeable", rr=1.5),
    ]

    scanner.apply_opportunity_ranking(rows, [])

    assert rows[0]["ticker"] == "ASET"
    assert rows[0]["ranking"]["status_bucket"] == "EARLY_ENTRY"
    assert rows[0]["ranking"]["priority_bucket"] < rows[1]["ranking"]["priority_bucket"]


def test_ranking_is_deterministic_and_input_order_independent():
    fixtures = [
        row("CCC", grade="B", stage="B+ TRADEABLE", entry_status="Tradeable", rr=2.2),
        row("AAA", grade="B", stage="B+ TRADEABLE", entry_status="Tradeable", rr=2.2),
        row("BBB", grade="B", stage="B+ TRADEABLE", entry_status="Tradeable", rr=2.2),
    ]
    rows_a = copy.deepcopy(fixtures)
    rows_b = list(reversed(copy.deepcopy(fixtures)))

    scanner.apply_opportunity_ranking(rows_a, [])
    scanner.apply_opportunity_ranking(rows_b, [])

    assert [(item["ticker"], item["ranking"]["rank"]) for item in rows_a] == [
        ("AAA", 1),
        ("BBB", 2),
        ("CCC", 3),
    ]
    assert [(item["ticker"], item["ranking"]["rank"]) for item in rows_b] == [
        ("AAA", 1),
        ("BBB", 2),
        ("CCC", 3),
    ]


def test_missing_optional_fields_are_neutral_and_reported():
    rows = [
        row(
            "MISS",
            rr=None,
            earnings={},
            quality={"grade": "B", "score": 50},
            option_plan={"available": False, "reason": "missing planned entry"},
        )
    ]

    meta = scanner.apply_opportunity_ranking(rows, [])

    assert rows[0]["ranking"]["rank"] == 1
    assert rows[0]["ranking"]["score"] >= 0
    assert "Option Plan unavailable" in rows[0]["ranking"]["cautions"]
    missing = meta["ranking_diagnostics"]["missing_field_counts"]
    assert missing["earnings"] == 1


def test_earnings_cautions_do_not_change_strategy_fields():
    rows = [row("ERN", grade="A", stage="A+ READY", entry_status="Tradeable", earnings={"days_until": 6})]
    before = stripped(rows)

    scanner.apply_opportunity_ranking(rows, [])

    assert "Earnings in 6 days" in rows[0]["ranking"]["cautions"]
    assert rows[0]["ranking"]["ranking_components"]["earnings"] == -5
    assert stripped(rows) == before


def test_duplicate_reasons_and_cautions_are_deduplicated():
    rows = [
        row(
            "DUP",
            grade="B",
            stage="B+ TRADEABLE",
            entry_status="Tradeable",
            trade_eval={
                "trade_stage": "B+ TRADEABLE",
                "b_plus_tradeable": True,
                "structure_quality": "CHOPPY / INTERNAL ONLY",
                "no_trade_reasons": ["Choppy/internal structure", "Choppy/internal structure"],
            },
        )
    ]

    scanner.apply_opportunity_ranking(rows, [])

    assert rows[0]["ranking"]["cautions"].count("Choppy/internal structure") == 1


def test_cached_scan_meta_preserves_ranking_diagnostics():
    rows = [row("AAA", grade="A", stage="A+ READY", entry_status="Tradeable")]
    ranking_meta = scanner.apply_opportunity_ranking(rows, [])
    cached = {
        "generated_at": scanner._utc_now(),
        "rows": rows,
        "near_miss": [],
        "scan_meta": {
            "configured_universe_count": 1,
            "symbols_attempted": 1,
            **ranking_meta,
        },
    }

    meta = scanner._analysis_cache_meta(("default", None), cached, refreshing=False)

    assert meta["ranking_version"] == scanner.OPPORTUNITY_RANKING_VERSION
    assert meta["ranked_setup_count"] == 1
    assert meta["ranking_diagnostics"]["top_20"][0]["ticker"] == "AAA"


if __name__ == "__main__":
    test_ranking_preserves_rows_and_strategy_fields()
    test_priority_buckets_dominate_scores()
    test_grade_a_beats_grade_b_inside_equivalent_status()
    test_ranking_is_deterministic_and_input_order_independent()
    test_missing_optional_fields_are_neutral_and_reported()
    test_earnings_cautions_do_not_change_strategy_fields()
    test_duplicate_reasons_and_cautions_are_deduplicated()
    test_cached_scan_meta_preserves_ranking_diagnostics()
    print("Opportunity ranking v1 tests passed")
