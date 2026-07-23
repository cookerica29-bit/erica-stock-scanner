#!/usr/bin/env python3
"""Position Intelligence replay validation tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import journal_store  # noqa: E402
import main  # noqa: E402
from position_intelligence import build_position_intelligence, replay_position_intelligence, aggregate_replays  # noqa: E402


TOKEN = "journal-secret"


def position(**overrides):
    base = {
        "journal_id": "j-1",
        "position_id": "p-1",
        "ticker": "UAL",
        "direction": "LONG",
        "result": "Open",
        "entry_timestamp": "2026-07-01T14:00:00Z",
        "actual_underlying_entry": 100,
        "planned_underlying_entry": 99,
        "original_stop": 95,
        "original_tp1": 110,
        "original_tp2": 115,
        "original_tp3": 120,
        "grade": "A",
        "position_last_state": "EXIT",
        "position_best_price": 150,
        "position_state_history": [{"event_id": "live-history-must-not-change"}],
    }
    base.update(overrides)
    return base


def candle(ts, high, low, close):
    return {"timestamp": ts, "high": high, "low": low, "close": close}


def test_shared_builder_and_long_short_replay_math():
    live = build_position_intelligence(position(position_best_price=None), {"current_price": 104, "timestamp": "2026-07-01T18:00:00Z"})
    assert live["state"] == "HEALTHY"
    assert round(live["current_r"], 2) == 0.8
    assert live["progress_to_tp1"]["display_percent"] == 40

    long = replay_position_intelligence(position(position_best_price=None), [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T14:00:00Z", 106, 101, 104),
        candle("2026-07-03T14:00:00Z", 111, 103, 108),
    ])
    assert long["entry_price"] == 100
    assert long["targets"]["tp1"] == 110
    assert long["tp1_timestamp"] == "2026-07-03T14:00:00Z"
    assert round(long["maximum_progress"]) == 110
    assert round(long["MFE"], 2) == 2.2
    assert round(long["MAE"], 2) == -0.2
    assert long["final_state"] == "PROTECT"

    short = replay_position_intelligence(position(
        direction="SHORT",
        actual_underlying_entry=100,
        original_stop=105,
        original_tp1=90,
        original_tp2=85,
        original_tp3=80,
        position_best_price=None,
    ), [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T14:00:00Z", 99, 94, 96),
        candle("2026-07-03T14:00:00Z", 98, 89, 92),
    ])
    assert short["direction"] == "SHORT"
    assert short["tp1_timestamp"] == "2026-07-03T14:00:00Z"
    assert round(short["maximum_progress"]) == 110
    assert round(short["MFE"], 2) == 2.2
    assert round(short["MAE"], 2) == -0.2


def test_no_lookahead_best_price_and_state_transitions():
    replay = replay_position_intelligence(position(position_best_price=999), [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T14:00:00Z", 103, 97, 98),
        candle("2026-07-03T14:00:00Z", 105, 99, 102),
    ])
    assert replay["maximum_progress"] == 50
    assert replay["final_state"] == "WATCH"
    transitions = [event for event in replay["timeline"] if event["event_type"] == "STATE_TRANSITION"]
    assert [event["previous_state"] for event in transitions] == ["HEALTHY"]
    assert [event["new_state"] for event in transitions] == ["WATCH"]
    assert replay["first_watch_timestamp"] == "2026-07-02T14:00:00Z"
    assert replay["watch_recovery_count"] == 0
    assert replay["time_in_each_state"]["WATCH"]["candle_count"] == 2


def test_replay_ignores_final_journal_target_touch_state_until_reached_chronologically():
    replay = replay_position_intelligence(position(
        ticker="OXY",
        direction="SHORT",
        entry_timestamp="2026-07-16T16:13:17.667006Z",
        exit_timestamp="2026-07-21T00:29:30.725000Z",
        actual_underlying_entry=54.28,
        original_stop=55.9,
        original_tp1=51.04,
        original_tp2=49.42,
        original_tp3=47.8,
        position_last_state="PROTECT",
        position_best_price=51.04,
        position_tp1_reached=True,
        first_target_touch_at="2026-07-21T00:29:30.724Z",
    ), [
        candle("2026-07-17T04:00:00Z", 55.17, 54.005, 54.86),
        candle("2026-07-20T04:00:00Z", 55.6699, 54.0, 55.19),
    ])
    assert replay["tp1_timestamp"] is None
    assert round(replay["maximum_progress"], 2) == 8.64
    assert round(replay["maximum_r"], 2) == 0.17
    assert replay["final_state"] == "WATCH"
    assert replay["time_in_each_state"]["PROTECT"]["candle_count"] == 0
    assert replay["time_in_each_state"]["WATCH"]["candle_count"] == 2


def test_watch_recovery_protect_exit_and_churn_metrics():
    replay = replay_position_intelligence(position(position_best_price=None), [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-01T15:00:00Z", 102, 97, 98),
        candle("2026-07-01T16:00:00Z", 103, 99, 102),
        candle("2026-07-01T17:00:00Z", 108, 101, 103),
        candle("2026-07-01T18:00:00Z", 108, 96, 102),
        candle("2026-07-01T19:00:00Z", 103, 94, 94),
    ])
    states = [event["new_state"] for event in replay["timeline"] if event["event_type"] == "STATE_TRANSITION"]
    assert "WATCH" in states
    assert "HEALTHY" in states
    assert "PROTECT" in states
    assert "EXIT" in states
    assert replay["same_day_state_changes"] > 2
    assert replay["high_churn"] is True
    assert replay["stop_timestamp"] == "2026-07-01T19:00:00Z"

    rapid = replay_position_intelligence(position(position_best_price=None), [
        candle("2026-07-02T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T15:00:00Z", 102, 97, 98),
        candle("2026-07-02T16:00:00Z", 103, 99, 102),
    ])
    assert rapid["rapid_reversal_count"] == 1


def test_milestone_dedup_and_target_stop_ambiguity():
    replay = replay_position_intelligence(position(position_best_price=None), [
        candle("2026-07-01T14:00:00Z", 111, 94, 100),
        candle("2026-07-02T14:00:00Z", 112, 96, 108),
    ])
    event_types = [event["event_type"] for event in replay["timeline"]]
    assert event_types.count("PROGRESS_25") == 1
    assert event_types.count("PROGRESS_50") == 1
    assert event_types.count("PROGRESS_75") == 1
    assert event_types.count("TP1_REACHED") == 1
    assert event_types.count("STOP_INVALIDATED") == 1
    assert "AMBIGUOUS_TARGET_STOP" in event_types
    assert replay["outcome_order_ambiguous"] is True
    assert replay["outcome_category"] == "AMBIGUOUS"


def test_data_needed_and_original_plan_not_overwritten():
    original = position(actual_underlying_entry=None, planned_underlying_entry=None, entry_price=None, entry=None)
    before = dict(original)
    replay = replay_position_intelligence(original, [candle("2026-07-01T14:00:00Z", 101, 99, 100)])
    assert replay["final_state"] == "DATA_NEEDED"
    assert replay["outcome_category"] == "DATA_INCOMPLETE"
    assert original == before
    assert original["position_last_state"] == "EXIT"
    assert original["position_state_history"] == [{"event_id": "live-history-must-not-change"}]


def test_multiple_trades_same_ticker_long_and_short():
    long = replay_position_intelligence(position(journal_id="long", position_id="long-pos", ticker="OXY", direction="LONG", position_best_price=None), [
        candle("2026-07-01T14:00:00Z", 104, 99, 103),
    ])
    short = replay_position_intelligence(position(journal_id="short", position_id="short-pos", ticker="OXY", direction="SHORT", actual_underlying_entry=100, original_stop=105, original_tp1=90, position_best_price=None), [
        candle("2026-07-01T14:00:00Z", 101, 96, 97),
    ])
    assert long["journal_id"] == "long"
    assert short["journal_id"] == "short"
    assert long["position_id"] != short["position_id"]
    assert long["direction"] == "LONG"
    assert short["direction"] == "SHORT"


def test_aggregate_sample_sizes_and_provider_failure_isolation():
    complete = replay_position_intelligence(position(position_best_price=None), [candle("2026-07-01T14:00:00Z", 104, 99, 103)])
    incomplete = replay_position_intelligence(position(journal_id="j-2", position_id="p-2"), [])
    aggregate = aggregate_replays([complete, incomplete])
    assert aggregate["positions_replayed"] == 2
    assert aggregate["complete_replays"] == 1
    assert aggregate["incomplete_replays"] == 1
    assert aggregate["percent_entered_watch"]["sample_size"] == 2
    assert "DATA_INCOMPLETE" in aggregate["outcome_category_distribution"]


def test_dev_endpoint_auth_empty_message_and_no_side_effects():
    tmp = tempfile.TemporaryDirectory()
    previous_repo = main._journal_repository
    previous_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    previous_submit_discovery = main._submit_discovery_universe_job
    previous_scan_cached = main.scan_cached
    try:
        repo = journal_store.SQLiteJournalRepository(str(Path(tmp.name) / "journal.sqlite3"))
        main._journal_repository = repo
        os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN
        main._submit_discovery_universe_job = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay must not trigger discovery"))
        main.scan_cached = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replay must not trigger scanner refresh"))
        client = TestClient(main.app)
        assert client.get("/api/dev/position-replay").status_code == 403
        empty = client.get("/api/dev/position-replay", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert empty["status"] == "not_ready"
        assert "No server-backed positions are available for replay yet" in empty["message"]

        repo.create_entry(position(journal_id="j-endpoint", position_id="p-endpoint", position_best_price=None))
        previous_fetch = main._fetch_replay_candles
        main._fetch_replay_candles = lambda entry: ([candle("2026-07-01T14:00:00Z", 104, 99, 103)], {
            "provider": "test",
            "timeframe": "4H",
            "timeframe_source": "inferred_default",
            "period": "60d",
            "interval": "4h",
            "candles_returned": 1,
            "cache_status": "test",
            "fetch_failure": None,
        })
        try:
            payload = client.get("/api/dev/position-replay?summary_only=false", headers={"X-Kairos-Admin-Token": TOKEN}).json()
            assert payload["ready"] is True
            assert payload["replays"][0]["timeline"][0]["event_type"] == "ENTRY"
            assert payload["synthetic_results_included"] is False
            single = client.get("/api/dev/position-replay/p-endpoint?summary_only=true", headers={"X-Kairos-Admin-Token": TOKEN}).json()
            assert single["ready"] is True
            assert single["replays"][0]["position_id"] == "p-endpoint"
            assert "timeline" not in single["replays"][0]
        finally:
            main._fetch_replay_candles = previous_fetch
    finally:
        main._journal_repository = previous_repo
        main._submit_discovery_universe_job = previous_submit_discovery
        main.scan_cached = previous_scan_cached
        if previous_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_token
        tmp.cleanup()


if __name__ == "__main__":
    test_shared_builder_and_long_short_replay_math()
    test_no_lookahead_best_price_and_state_transitions()
    test_watch_recovery_protect_exit_and_churn_metrics()
    test_milestone_dedup_and_target_stop_ambiguity()
    test_data_needed_and_original_plan_not_overwritten()
    test_multiple_trades_same_ticker_long_and_short()
    test_aggregate_sample_sizes_and_provider_failure_isolation()
    test_dev_endpoint_auth_empty_message_and_no_side_effects()
    print("Position replay v1 tests passed")
