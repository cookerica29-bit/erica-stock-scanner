import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner


def setup(**overrides):
    row = {
        "ticker": "ABC",
        "timeframe": "1H",
        "direction": "LONG",
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 110.0,
        "price": 99.8,
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
    assert diag["total_memories"] == 2

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


if __name__ == "__main__":
    test_confirmation_before_entry_touch_enters_without_changing_bucket()
    test_entry_touch_before_confirmation_waits_for_later_retest()
    test_no_retest_becomes_missed_entry()
    test_tp1_before_confirmation_and_stop_before_entry_are_terminal()
    test_plan_change_creates_new_memory_and_short_touch_rules()
    test_expiration_for_non_confirming_setup()
    test_scheduled_scans_preserve_memory_without_recreating()
    print("stock_early_entry_memory_v1 passed")
