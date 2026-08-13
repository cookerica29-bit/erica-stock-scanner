import tempfile
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scanner
import main


def reset_memory():
    tmp = tempfile.NamedTemporaryFile(delete=True)
    scanner.STOCK_EARLY_ENTRY_MEMORY_PATH = tmp.name
    scanner.reset_stock_early_entry_shadow_memory()


def reset_memory_path(path):
    scanner.STOCK_EARLY_ENTRY_MEMORY_PATH = str(path)
    scanner._stock_early_entry_memory_loaded = False
    scanner._stock_early_entry_memory = {}


def setup(**overrides):
    row = {
        "ticker": "ABC",
        "timeframe": "1H",
        "direction": "LONG",
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 110.0,
        "atr": 4.0,
        "price": 99.5,
        "entryStatus": "Tradeable",
        "setupGrade": "A",
        "setup_status": "QUALIFIED",
        "setupStatus": "Early Confirmation",
        "signal_timestamp": "2026-08-05T09:00:00Z",
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


def bar(ts, open_=99.0, high=100.2, low=99.8, close=100.05):
    return {"time": ts, "open": open_, "high": high, "low": low, "close": close}


def run(row, at):
    meta = {"ranking_generated_at": at, "scan_completed_at": at}
    scanner.update_stock_early_entry_shadow([row], [], meta, generated_at=at)
    return row["early_entry_shadow"]


def test_ohlc_schema_normalizes_uppercase_and_lowercase():
    idx = pd.to_datetime(["2026-08-05T04:00:00Z", "2026-08-06T04:00:00Z"])
    upper = pd.DataFrame({"Open": [1, 2], "High": [3, 4], "Low": [0.5, 1.5], "Close": [2.5, 3.5]}, index=idx)
    lower = pd.DataFrame({"open": [1, 2], "high": [3, 4], "low": [0.5, 1.5], "close": [2.5, 3.5]}, index=idx)
    for frame in (upper, lower):
        bars = scanner._stock_completed_bar_records(frame, "1D", "2026-08-06T21:00:00Z")
        assert bars[-1]["high"] == 4.0
        assert bars[-1]["low"] == 1.5
        assert bars[-1]["close"] == 3.5


def test_incomplete_latest_candle_excluded_and_latest_completed_included():
    idx = pd.to_datetime(["2026-08-06T13:30:00Z", "2026-08-06T17:30:00Z"])
    frame = pd.DataFrame({"Open": [1, 2], "High": [3, 9], "Low": [0.5, 8], "Close": [2.5, 8.5]}, index=idx)
    bars = scanner._stock_completed_bar_records(frame, "4H", "2026-08-06T20:00:00Z")
    assert [item["time"] for item in bars] == ["2026-08-06T13:30:00Z"]
    assert bars[0]["high"] == 3.0


def test_enrichment_populates_completed_current_bar_fields_and_summary_preserves_them():
    idx = pd.to_datetime(["2026-08-06T13:30:00Z", "2026-08-06T17:30:00Z"])
    h4 = pd.DataFrame({"Open": [1, 2], "High": [3, 9], "Low": [0.5, 8], "Close": [2.5, 8.5]}, index=idx)
    row = setup(ticker="H4T", timeframe="4H", price=2.5)
    enriched = scanner._enrich_stock_scout_fields(
        row,
        row,
        row,
        h4,
        h4,
        observed_at=pd.Timestamp("2026-08-06T20:00:00Z").to_pydatetime(),
    )
    assert enriched["current_bar_open"] == 1.0
    assert enriched["current_bar_high"] == 3.0
    assert enriched["current_bar_low"] == 0.5
    assert enriched["current_bar_close"] == 2.5
    assert enriched["current_candle_time"] == "2026-08-06T13:30:00Z"
    summary = main._summary_row(enriched, "gen")
    assert summary["current_bar_high"] == 3.0
    assert summary["current_bar_low"] == 0.5
    assert summary["current_candle_time"] == "2026-08-06T13:30:00Z"


def test_cached_capitalized_bars_replay_between_refresh_retest_after_confirmation():
    reset_memory()
    first = setup(_lifecycle_completed_bars=[bar("2026-08-05T10:00:00Z")])
    shadow = run(first, "2026-08-05T10:05:00Z")
    assert shadow["state"] == "EARLY_TOUCH"

    confirmed = setup(
        price=102.0,
        _lifecycle_completed_bars=[bar("2026-08-05T11:00:00Z", high=102.4, low=101.4, close=102.0)],
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    shadow = run(confirmed, "2026-08-05T11:05:00Z")
    assert shadow["state"] == "WAITING_FOR_RETEST"

    retest = setup(
        price=100.05,
        _lifecycle_completed_bars=[
            bar("2026-08-05T11:00:00Z", high=102.4, low=101.4, close=102.0),
            bar("2026-08-05T12:00:00Z", high=100.4, low=99.9, close=100.05),
        ],
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    shadow = run(retest, "2026-08-05T12:05:00Z")
    assert shadow["state"] == "ENTRY_TRIGGERED"
    assert shadow["retest_at"] == "2026-08-05T12:00:00Z"


def test_same_bar_confirmation_retest_blocked_and_later_wick_outside_window_rejected():
    reset_memory()
    run(setup(_lifecycle_completed_bars=[bar("2026-08-05T10:00:00Z")]), "2026-08-05T10:05:00Z")
    same_bar = setup(
        price=100.05,
        _lifecycle_completed_bars=[bar("2026-08-05T11:00:00Z", high=100.4, low=99.9, close=100.05)],
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    shadow = run(same_bar, "2026-08-05T11:05:00Z")
    assert shadow["state"] == "WAITING_FOR_RETEST"

    outside = setup(
        price=103.0,
        _lifecycle_completed_bars=[
            bar("2026-08-05T11:00:00Z", high=100.4, low=99.9, close=100.05),
            bar("2026-08-05T12:00:00Z", high=103.4, low=99.9, close=103.0),
        ],
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    shadow = run(outside, "2026-08-05T12:05:00Z")
    assert shadow["state"] == "WAITING_FOR_RETEST"
    assert not shadow.get("retest_at")


def test_short_valid_retest_uses_same_direction_window():
    reset_memory()
    first = setup(
        direction="SHORT",
        entry=50.0,
        sl=52.0,
        tp1=46.0,
        price=49.8,
        _lifecycle_completed_bars=[bar("2026-08-05T10:00:00Z", high=50.2, low=49.6, close=49.8)],
    )
    run(first, "2026-08-05T10:05:00Z")
    confirmed = setup(
        direction="SHORT",
        entry=50.0,
        sl=52.0,
        tp1=46.0,
        price=49.0,
        _lifecycle_completed_bars=[bar("2026-08-05T11:00:00Z", high=49.4, low=48.8, close=49.0)],
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    assert run(confirmed, "2026-08-05T11:05:00Z")["state"] == "WAITING_FOR_RETEST"
    retest = setup(
        direction="SHORT",
        entry=50.0,
        sl=52.0,
        tp1=46.0,
        price=49.95,
        _lifecycle_completed_bars=[
            bar("2026-08-05T11:00:00Z", high=49.4, low=48.8, close=49.0),
            bar("2026-08-05T12:00:00Z", high=50.1, low=49.7, close=49.95),
        ],
        trade_eval={"trigger_confirmed": True, "a_plus_ready": True, "b_plus_tradeable": False},
    )
    assert run(retest, "2026-08-05T12:05:00Z")["state"] == "ENTRY_TRIGGERED"


def test_reprocessing_is_idempotent_and_migration_is_bounded():
    reset_memory()
    row = setup(_lifecycle_completed_bars=[bar("2026-08-05T10:00:00Z")])
    first = run(row, "2026-08-05T10:05:00Z")
    second = run(row, "2026-08-05T10:05:00Z")
    assert len(first["transitions"]) == len(second["transitions"])
    assert second["evidence_migration_policy"] == "bounded_initial_setup_generation_replay"
    assert second["last_evaluated_bar_time"] == "2026-08-05T10:00:00Z"


def test_same_ticker_timeframe_direction_memories_coexist_and_only_matching_plan_replaced():
    reset_memory()
    one_day = setup(ticker="COL", timeframe="1D", direction="LONG", entry=100, sl=95, tp1=110, current_bar=1, price=100.2, current_bar_low=99.8, current_bar_high=100.4)
    h4 = setup(ticker="COL", timeframe="4H", direction="LONG", entry=101, sl=96, tp1=111, current_bar=1, price=101.2, current_bar_low=100.8, current_bar_high=101.4)
    short = setup(ticker="COL", timeframe="1D", direction="SHORT", entry=102, sl=107, tp1=92, current_bar=1, price=101.8, current_bar_low=101.5, current_bar_high=102.2)
    scanner.update_stock_early_entry_shadow([one_day, h4, short], [], {"ranking_generated_at": "2026-08-05T10:00:00Z"}, generated_at="2026-08-05T10:00:00Z")
    diag = scanner.stock_early_entry_shadow_diagnostics()
    assert diag["total_memories"] == 3
    assert diag["states"]["EARLY_TOUCH"] == 3

    replacement = setup(ticker="COL", timeframe="1D", direction="LONG", entry=100.5, sl=95, tp1=110, current_bar=2, price=100.6, current_bar_low=100.3, current_bar_high=100.8)
    scanner.update_stock_early_entry_shadow([replacement], [], {"ranking_generated_at": "2026-08-05T11:00:00Z"}, generated_at="2026-08-05T11:00:00Z")
    diag = scanner.stock_early_entry_shadow_diagnostics()
    assert diag["total_memories"] == 3
    assert diag["states"]["EARLY_TOUCH"] == 3
    assert replacement["early_entry_shadow"]["original_plan"]["entry"] == 100
    assert replacement["early_entry_shadow"]["current_plan"]["entry"] == 100.5

    new_generation = setup(
        ticker="COL",
        timeframe="1D",
        direction="LONG",
        entry=101.5,
        sl=96,
        tp1=111,
        signal_timestamp="2026-08-05T11:00:00Z",
        current_bar=3,
        price=101.6,
        current_bar_low=101.3,
        current_bar_high=101.8,
    )
    scanner.update_stock_early_entry_shadow([new_generation], [], {"ranking_generated_at": "2026-08-05T12:00:00Z"}, generated_at="2026-08-05T12:00:00Z")
    diag = scanner.stock_early_entry_shadow_diagnostics()
    assert diag["total_memories"] == 4
    assert diag["states"]["PLAN_REPLACED"] == 1
    assert diag["states"]["EARLY_TOUCH"] == 3


def test_stock_early_memory_atomic_write_creates_valid_primary_and_backup():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        reset_memory_path(path)
        scanner._stock_early_entry_memory = {
            "one": {"candidate_id": "one", "state": "EARLY_TOUCH", "transitions": [{"to": "EARLY_TOUCH"}]},
        }
        scanner._save_stock_early_entry_memory_locked()
        primary = json.loads(path.read_text())
        assert primary["memories"]["one"]["state"] == "EARLY_TOUCH"

        scanner._stock_early_entry_memory["two"] = {"candidate_id": "two", "state": "WAITING_FOR_CONFIRMATION", "transitions": []}
        scanner._save_stock_early_entry_memory_locked()
        backup = json.loads(scanner._stock_early_entry_memory_backup_path(path).read_text())
        primary = json.loads(path.read_text())
        assert "one" in backup["memories"]
        assert "two" not in backup["memories"]
        assert "two" in primary["memories"]


def test_stock_early_memory_failed_replace_leaves_existing_file_intact():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        reset_memory_path(path)
        original_payload = {
            "version": scanner.STOCK_EARLY_ENTRY_MEMORY_VERSION,
            "updated_at": "2026-08-05T10:00:00Z",
            "memories": {"old": {"candidate_id": "old", "state": "EARLY_TOUCH", "transitions": [{"to": "EARLY_TOUCH"}]}},
        }
        path.write_text(json.dumps(original_payload, sort_keys=True))
        scanner._stock_early_entry_memory = {
            "new": {"candidate_id": "new", "state": "WAITING_FOR_CONFIRMATION", "transitions": []},
        }
        real_replace = scanner.os.replace

        def failing_replace(src, dst):
            if Path(dst) == path:
                raise OSError("simulated final replace failure")
            return real_replace(src, dst)

        try:
            scanner.os.replace = failing_replace
            scanner._save_stock_early_entry_memory_locked()
        finally:
            scanner.os.replace = real_replace

        assert json.loads(path.read_text()) == original_payload
        assert not any(item.name.endswith(".tmp") for item in path.parent.iterdir())


def test_stock_early_memory_malformed_primary_recovers_from_backup():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        backup = scanner._stock_early_entry_memory_backup_path(path)
        backup_payload = {
            "version": scanner.STOCK_EARLY_ENTRY_MEMORY_VERSION,
            "updated_at": "2026-08-05T10:00:00Z",
            "memories": {
                "abc": {
                    "candidate_id": "abc",
                    "state": "WAITING_FOR_RETEST",
                    "transitions": [{"to": "EARLY_TOUCH"}, {"to": "WAITING_FOR_RETEST"}],
                }
            },
        }
        path.write_text("{bad json")
        backup.write_text(json.dumps(backup_payload, sort_keys=True))
        reset_memory_path(path)
        loaded = scanner._load_stock_early_entry_memory_locked()
        assert loaded == backup_payload["memories"]
        assert loaded["abc"]["transitions"][1]["to"] == "WAITING_FOR_RETEST"


def test_stock_early_memory_restart_preserves_identities_histories_without_events():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        reset_memory_path(path)
        row = setup(_lifecycle_completed_bars=[bar("2026-08-05T10:00:00Z")])
        shadow = run(row, "2026-08-05T10:05:00Z")
        key = shadow["candidate_id"]
        assert shadow["state"] == "EARLY_TOUCH"
        before_transitions = list(shadow["transitions"])

        scanner._stock_early_entry_memory_loaded = False
        scanner._stock_early_entry_memory = {}
        reloaded = scanner._load_stock_early_entry_memory_locked()
        assert key in reloaded
        assert reloaded[key]["transitions"] == before_transitions

        diag = scanner.stock_early_entry_shadow_diagnostics()
        assert diag["total_memories"] == 1
        assert diag["states"] == {"EARLY_TOUCH": 1}
        assert diag["valid_enter_now_transitions"] == 0
        assert diag["plan_replaced"] == 0
        assert len(reloaded[key]["transitions"]) == len(before_transitions)


if __name__ == "__main__":
    test_ohlc_schema_normalizes_uppercase_and_lowercase()
    test_incomplete_latest_candle_excluded_and_latest_completed_included()
    test_enrichment_populates_completed_current_bar_fields_and_summary_preserves_them()
    test_cached_capitalized_bars_replay_between_refresh_retest_after_confirmation()
    test_same_bar_confirmation_retest_blocked_and_later_wick_outside_window_rejected()
    test_short_valid_retest_uses_same_direction_window()
    test_reprocessing_is_idempotent_and_migration_is_bounded()
    test_same_ticker_timeframe_direction_memories_coexist_and_only_matching_plan_replaced()
    test_stock_early_memory_atomic_write_creates_valid_primary_and_backup()
    test_stock_early_memory_failed_replace_leaves_existing_file_intact()
    test_stock_early_memory_malformed_primary_recovers_from_backup()
    test_stock_early_memory_restart_preserves_identities_histories_without_events()
    print("stock_lifecycle_evidence_integrity_v1 passed")
