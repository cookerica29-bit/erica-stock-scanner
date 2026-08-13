import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner
import smart_notifications


def setup(**overrides):
    row = {
        "ticker": "ABC",
        "timeframe": "1H",
        "direction": "LONG",
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 110.0,
        "price": 99.8,
        "current_quote_price": 100.1,
        "current_bar_high": 99.9,
        "current_bar_low": 99.7,
        "entryStatus": "Tradeable",
        "setupGrade": "A",
        "setup_status": "QUALIFIED",
        "setupStatus": "Early Confirmation",
        "signal_timestamp": "2026-08-05T10:00:00Z",
        "trade_eval": {
            "trade_stage": "B+ TRADEABLE",
            "b_plus_tradeable": True,
            "trigger_confirmed": False,
            "a_plus_ready": False,
        },
    }
    row.update(overrides)
    if "trade_eval" in overrides:
        merged = {
            "trade_stage": "B+ TRADEABLE",
            "b_plus_tradeable": True,
            "trigger_confirmed": False,
            "a_plus_ready": False,
        }
        merged.update(overrides["trade_eval"])
        row["trade_eval"] = merged
    return row


def run(rows, at):
    meta = {"ranking_generated_at": at, "scan_completed_at": at}
    return scanner.update_stock_early_entry_shadow(rows, [], meta, generated_at=at)


def reset_memory():
    tmp = tempfile.NamedTemporaryFile(delete=True)
    scanner.STOCK_EARLY_ENTRY_MEMORY_PATH = tmp.name
    scanner.reset_stock_early_entry_shadow_memory()


def assert_state(row, state):
    assert row["early_entry_shadow"]["state"] == state, row["early_entry_shadow"]


def test_confirmation_before_entry_touch_enters_without_changing_bucket():
    reset_memory()
    first = setup(current_bar=1, price=99.8, current_bar_low=100.1, current_bar_high=100.2)
    run([first], "2026-08-05T10:00:00Z")
    assert scanner._ranking_status_bucket(first) == "EARLY_ENTRY"
    assert_state(first, "EARLY_ENTRY_BUILDING")

    second = setup(
        current_bar=2,
        price=100.1,
        current_bar_low=99.9,
        current_bar_high=100.3,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([second], "2026-08-05T11:00:00Z")
    assert scanner._ranking_status_bucket(second) == "ENTER_NOW"
    assert_state(second, "ENTRY_TRIGGERED")


def test_entry_touch_before_confirmation_waits_for_later_retest():
    reset_memory()
    early = setup(current_bar=1, price=100.2, current_bar_low=99.8, current_bar_high=100.4)
    run([early], "2026-08-05T10:00:00Z")
    assert_state(early, "EARLY_TOUCH")

    same_bar_confirm = setup(
        current_bar=1,
        price=100.2,
        current_bar_low=99.8,
        current_bar_high=100.4,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([same_bar_confirm], "2026-08-05T10:30:00Z")
    assert_state(same_bar_confirm, "WAITING_FOR_RETEST")

    later_retest = setup(
        current_bar=2,
        price=100.05,
        current_bar_low=99.95,
        current_bar_high=100.2,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([later_retest], "2026-08-05T11:00:00Z")
    assert_state(later_retest, "ENTRY_TRIGGERED")


def test_no_retest_becomes_missed_entry():
    reset_memory()
    run([setup(current_bar=1, price=100.2, current_bar_low=99.8, current_bar_high=100.4)], "2026-08-05T10:00:00Z")
    confirmed = setup(
        current_bar=2,
        price=103.0,
        current_bar_low=102.8,
        current_bar_high=103.2,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([confirmed], "2026-08-05T11:00:00Z")
    assert_state(confirmed, "WAITING_FOR_RETEST")
    missed = setup(
        current_bar=8,
        price=104.0,
        current_bar_low=103.8,
        current_bar_high=104.4,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([missed], "2026-08-05T17:00:00Z")
    assert_state(missed, "MISSED_ENTRY")


def test_tp1_before_confirmation_and_stop_before_entry_are_terminal():
    reset_memory()
    tp = setup(current_bar=1, price=111.0, current_bar_low=99.8, current_bar_high=111.2)
    run([tp], "2026-08-05T10:00:00Z")
    assert_state(tp, "TP1_BEFORE_CONFIRMATION")
    confirmed = setup(
        current_bar=2,
        price=100.0,
        current_bar_low=99.9,
        current_bar_high=100.1,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([confirmed], "2026-08-05T11:00:00Z")
    assert_state(confirmed, "TP1_BEFORE_CONFIRMATION")

    reset_memory()
    stopped = setup(current_bar=1, price=94.5, current_bar_low=94.4, current_bar_high=99.0)
    run([stopped], "2026-08-05T10:00:00Z")
    assert_state(stopped, "INVALIDATED")


def test_plan_change_creates_new_memory_and_short_touch_rules():
    reset_memory()
    first = setup(current_bar=1, entry=100.0, sl=95.0, tp1=110.0)
    second = setup(current_bar=1, entry=101.0, sl=96.0, tp1=111.0)
    run([first], "2026-08-05T10:00:00Z")
    run([second], "2026-08-05T11:00:00Z")
    diag = scanner.stock_early_entry_shadow_diagnostics()
    assert diag["total_memories"] == 1
    assert diag["states"]["EARLY_TOUCH"] == 1
    assert second["early_entry_shadow"]["original_plan"]["entry"] == 100.0
    assert second["early_entry_shadow"]["current_plan"]["entry"] == 101.0

    reset_memory()
    short = setup(
        direction="SHORT",
        entry=50.0,
        sl=52.0,
        tp1=46.0,
        price=49.8,
        current_bar_low=49.7,
        current_bar_high=49.9,
        current_bar=1,
    )
    run([short], "2026-08-05T10:00:00Z")
    assert_state(short, "EARLY_TOUCH")


def test_new_generation_creates_new_memory_and_replaces_old_generation():
    reset_memory()
    first = setup(current_bar=1, entry=100.0, sl=95.0, tp1=110.0, signal_timestamp="2026-08-05T10:00:00Z")
    second = setup(current_bar=2, entry=101.0, sl=96.0, tp1=111.0, signal_timestamp="2026-08-05T11:00:00Z")
    run([first], "2026-08-05T10:00:00Z")
    run([second], "2026-08-05T11:00:00Z")
    diag = scanner.stock_early_entry_shadow_diagnostics()
    assert diag["total_memories"] == 2
    assert diag["states"]["PLAN_REPLACED"] == 1
    assert diag["states"]["EARLY_TOUCH"] == 1


def test_raw_enter_now_first_print_receives_lifecycle_tracking():
    reset_memory()
    raw_enter = setup(
        current_bar=2,
        price=100.05,
        current_quote_price=100.05,
        current_bar_low=99.9,
        current_bar_high=100.2,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    assert scanner._ranking_status_bucket(raw_enter) == "ENTER_NOW"
    run([raw_enter], "2026-08-05T11:00:00Z")
    assert_state(raw_enter, "ENTRY_TRIGGERED")


def test_almost_ready_direct_jump_to_raw_enter_now_keeps_lifecycle_coverage():
    reset_memory()
    almost = setup(
        current_bar=1,
        price=99.4,
        current_quote_price=99.4,
        current_bar_low=100.2,
        current_bar_high=100.4,
        trade_eval={"b_plus_tradeable": False, "trigger_confirmed": False, "a_plus_ready": False},
    )
    assert scanner._ranking_status_bucket(almost) == "ALMOST_READY"
    run([almost], "2026-08-05T10:00:00Z")
    assert_state(almost, "EARLY_ENTRY_BUILDING")

    raw_enter = setup(
        current_bar=2,
        price=100.05,
        current_quote_price=100.05,
        current_bar_low=99.9,
        current_bar_high=100.2,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    assert scanner._ranking_status_bucket(raw_enter) == "ENTER_NOW"
    run([raw_enter], "2026-08-05T11:00:00Z")
    assert_state(raw_enter, "ENTRY_TRIGGERED")


def test_opposite_direction_does_not_inherit_lifecycle_state():
    reset_memory()
    long_row = setup(
        current_bar=1,
        price=100.05,
        current_quote_price=100.05,
        current_bar_low=99.9,
        current_bar_high=100.2,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([long_row], "2026-08-05T10:00:00Z")
    assert_state(long_row, "ENTRY_TRIGGERED")

    short_row = setup(
        direction="SHORT",
        entry=100.0,
        sl=105.0,
        tp1=90.0,
        price=99.95,
        current_quote_price=99.95,
        current_bar_low=99.8,
        current_bar_high=100.1,
        current_bar=2,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([short_row], "2026-08-05T11:00:00Z")
    assert_state(short_row, "ENTRY_TRIGGERED")
    assert short_row["early_entry_shadow"]["candidate_id"] != long_row["early_entry_shadow"]["candidate_id"]


def test_new_generation_does_not_inherit_entry_triggered_state():
    reset_memory()
    triggered = setup(
        current_bar=1,
        price=100.05,
        current_quote_price=100.05,
        current_bar_low=99.9,
        current_bar_high=100.2,
        signal_timestamp="2026-08-05T10:00:00Z",
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    run([triggered], "2026-08-05T10:00:00Z")
    assert_state(triggered, "ENTRY_TRIGGERED")

    later_generation = setup(
        current_bar=2,
        price=99.4,
        current_quote_price=99.4,
        current_bar_low=100.2,
        current_bar_high=100.4,
        signal_timestamp="2026-08-05T11:00:00Z",
        trade_eval={"b_plus_tradeable": False, "trigger_confirmed": False, "a_plus_ready": False},
    )
    run([later_generation], "2026-08-05T11:00:00Z")
    assert_state(later_generation, "EARLY_ENTRY_BUILDING")
    assert later_generation["early_entry_shadow"]["candidate_id"] != triggered["early_entry_shadow"]["candidate_id"]


def test_expiration_for_non_confirming_setup():
    reset_memory()
    row = setup(current_bar=1, price=99.8)
    run([row], "2026-08-05T10:00:00Z")
    later = setup(current_bar=30, price=99.8)
    run([later], "2026-08-07T10:00:00Z")
    assert_state(later, "EXPIRED")


def test_scheduled_scans_preserve_memory_without_recreating():
    reset_memory()
    first = setup(current_bar=1, price=100.2, current_bar_low=99.8, current_bar_high=100.4)
    run([first], "2026-08-05T10:00:00Z")
    first_shadow = first["early_entry_shadow"]
    assert first_shadow["state"] == "EARLY_TOUCH"
    assert first_shadow["first_seen_at"] == "2026-08-05T10:00:00Z"

    second = setup(current_bar=2, price=100.15, current_bar_low=99.9, current_bar_high=100.3)
    run([second], "2026-08-05T11:00:00Z")
    second_shadow = second["early_entry_shadow"]
    assert second_shadow["state"] == "EARLY_TOUCH"
    assert second_shadow["first_seen_at"] == "2026-08-05T10:00:00Z"
    assert second_shadow["first_entry_touch_at"] == "2026-08-05T10:00:00Z"
    assert second_shadow["last_seen_at"] == "2026-08-05T11:00:00Z"
    assert len(second_shadow["transitions"]) == len(first_shadow["transitions"])


def test_execution_lifecycle_presentation_is_separate_from_ranking_bucket():
    reset_memory()
    previous_flag = scanner.STOCK_EVENT_MEMORY_PRESENTATION_V1
    try:
        scanner.STOCK_EVENT_MEMORY_PRESENTATION_V1 = False
        row = setup(current_bar=1, price=100.2, current_bar_low=99.8, current_bar_high=100.4)
        run([row], "2026-08-05T10:00:00Z")
        presentation = scanner.stock_execution_lifecycle_presentation(row)
        assert presentation["enabled"] is False
        assert presentation["ranking_status_bucket"] == "EARLY_ENTRY"
        assert presentation["state"] == "EARLY_TOUCH"
        assert presentation["display"] == "EARLY TOUCH"
        assert scanner._ranking_status_bucket(row) == "EARLY_ENTRY"

        scanner.STOCK_EVENT_MEMORY_PRESENTATION_V1 = True
        enabled = scanner.stock_execution_lifecycle_presentation(row)
        assert enabled["enabled"] is True
        assert enabled["bucket"] == "EARLY_TOUCH"
        assert enabled["actionable"] is False
    finally:
        scanner.STOCK_EVENT_MEMORY_PRESENTATION_V1 = previous_flag


def test_stale_early_touch_blocks_enter_now_notifications_until_valid_retest():
    row = setup(
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
        early_entry_shadow={"state": "WAITING_FOR_RETEST"},
    )
    assert smart_notifications.setup_bucket(row) == "ENTER_NOW"
    assert smart_notifications.stale_early_touch_blocks_enter_now(row) is True

    row["early_entry_shadow"] = {"state": "ENTRY_TRIGGERED"}
    assert smart_notifications.stale_early_touch_blocks_enter_now(row) is False


def with_shadow(row, state, *, generation=None):
    shadow_row = dict(row)
    if generation is not None:
        shadow_row["signal_timestamp"] = generation
    key = scanner.stock_early_entry_memory_key(shadow_row)
    row["early_entry_shadow"] = {"candidate_id": key, "state": state}
    return row


def test_new_entry_signal_requires_ranking_and_lifecycle_agreement():
    ranking_executable = setup(
        current_bar=2,
        price=100.1,
        current_quote_price=100.1,
        current_bar_low=99.9,
        current_bar_high=100.3,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    with_shadow(ranking_executable, "WAITING_FOR_CONFIRMATION")
    assert scanner._ranking_status_bucket(ranking_executable) == "ENTER_NOW"
    signal = scanner.stock_new_entry_signal(ranking_executable)
    assert signal["bucket"] == "WAITING_FOR_CONFIRMATION"
    assert signal["actionable"] is False
    assert signal["current_strategy_executable"] is True
    assert signal["lifecycle_entry_triggered"] is False

    lifecycle_only = setup(
        setupGrade="C",
        setup_status="SKIPPED",
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    with_shadow(lifecycle_only, "ENTRY_TRIGGERED")
    signal = scanner.stock_new_entry_signal(lifecycle_only)
    assert signal["bucket"] == "NO_CURRENT_ENTRY"
    assert signal["actionable"] is False
    assert lifecycle_only["early_entry_shadow"]["state"] == "ENTRY_TRIGGERED"

    both_agree = setup(
        price=100.05,
        current_quote_price=100.05,
        current_bar_low=99.9,
        current_bar_high=100.2,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    with_shadow(both_agree, "ENTRY_TRIGGERED")
    signal = scanner.stock_new_entry_signal(both_agree)
    assert signal["bucket"] == "ENTER_NOW"
    assert signal["actionable"] is True

    invalidated = setup(
        price=100.05,
        current_quote_price=100.05,
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    with_shadow(invalidated, "INVALIDATED")
    signal = scanner.stock_new_entry_signal(invalidated)
    assert signal["bucket"] == "INVALIDATED"
    assert signal["actionable"] is False

    new_generation = setup(
        price=100.05,
        current_quote_price=100.05,
        signal_timestamp="2026-08-05T12:00:00Z",
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    with_shadow(new_generation, "ENTRY_TRIGGERED", generation="2026-08-05T10:00:00Z")
    signal = scanner.stock_new_entry_signal(new_generation)
    assert signal["bucket"] == "ALMOST_READY"
    assert signal["same_setup_identity"] is False
    assert signal["actionable"] is False


def test_daily_lifecycle_report_counts_today_transitions_only():
    memories = {
        "aaa": {
            "ticker": "AAA",
            "timeframe": "1H",
            "direction": "LONG",
            "candidate_id": "aaa",
            "transitions": [
                {"from": None, "to": "EARLY_ENTRY_BUILDING", "timestamp": "2026-08-05T10:00:00Z", "reason": "early_entry_first_printed"},
                {"from": "EARLY_ENTRY_BUILDING", "to": "EARLY_TOUCH", "timestamp": "2026-08-05T11:00:00Z", "reason": "planned_entry_touched_before_confirmation"},
                {"from": "EARLY_TOUCH", "to": "WAITING_FOR_RETEST", "timestamp": "2026-08-04T12:00:00Z", "reason": "yesterday"},
            ],
        },
        "bbb": {
            "ticker": "BBB",
            "timeframe": "4H",
            "direction": "SHORT",
            "candidate_id": "bbb",
            "transitions": [
                {"from": None, "to": "EARLY_ENTRY_BUILDING", "timestamp": "2026-08-05T10:30:00Z", "reason": "early_entry_first_printed"},
                {"from": "WAITING_FOR_RETEST", "to": "ENTRY_TRIGGERED", "timestamp": "2026-08-05T14:00:00Z", "reason": "valid_retest_after_confirmation"},
                {"from": "ENTRY_TRIGGERED", "to": "MISSED_ENTRY", "timestamp": "not-a-date", "reason": "ignored"},
            ],
        },
    }
    report = scanner._stock_early_daily_lifecycle_report(memories, now=datetime(2026, 8, 5, tzinfo=timezone.utc))
    assert report["report_date_utc"] == "2026-08-05"
    assert report["counts"]["early_entry"] == 2
    assert report["counts"]["early_touch"] == 1
    assert report["counts"]["waiting_for_retest"] == 0
    assert report["counts"]["enter_now"] == 1
    assert report["counts"]["missed_entry"] == 0
    assert [item["key"] for item in report["ordered_counts"]][:4] == [
        "early_entry",
        "early_touch",
        "waiting_for_retest",
        "enter_now",
    ]


if __name__ == "__main__":
    test_confirmation_before_entry_touch_enters_without_changing_bucket()
    test_entry_touch_before_confirmation_waits_for_later_retest()
    test_no_retest_becomes_missed_entry()
    test_tp1_before_confirmation_and_stop_before_entry_are_terminal()
    test_plan_change_creates_new_memory_and_short_touch_rules()
    test_new_generation_creates_new_memory_and_replaces_old_generation()
    test_raw_enter_now_first_print_receives_lifecycle_tracking()
    test_almost_ready_direct_jump_to_raw_enter_now_keeps_lifecycle_coverage()
    test_opposite_direction_does_not_inherit_lifecycle_state()
    test_new_generation_does_not_inherit_entry_triggered_state()
    test_expiration_for_non_confirming_setup()
    test_scheduled_scans_preserve_memory_without_recreating()
    test_execution_lifecycle_presentation_is_separate_from_ranking_bucket()
    test_stale_early_touch_blocks_enter_now_notifications_until_valid_retest()
    test_new_entry_signal_requires_ranking_and_lifecycle_agreement()
    test_daily_lifecycle_report_counts_today_transitions_only()
    print("stock_early_entry_memory_v1 passed")
