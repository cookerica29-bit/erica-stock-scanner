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
from position_intelligence import (  # noqa: E402
    RANGE_BOUND_ONLY,
    RECOMPUTE_CHRONOLOGICALLY,
    SAFE_STATIC_INPUT,
    STRIP_FROM_REPLAY,
    aggregate_replays,
    build_position_intelligence,
    classify_journal_replay_parity,
    evidence_guard,
    evidence_log_from_replays,
    real_evidence_counts,
    replay_position_intelligence,
    replay_readiness,
    REPLAY_FIELD_CLASSIFICATION,
)


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


def test_replay_field_classification_documents_no_lookahead_inputs():
    expected = {
        "entry_timestamp": SAFE_STATIC_INPUT,
        "planned_underlying_entry": SAFE_STATIC_INPUT,
        "actual_underlying_entry": SAFE_STATIC_INPUT,
        "original_stop": SAFE_STATIC_INPUT,
        "original_tp1": SAFE_STATIC_INPUT,
        "exit_timestamp": RANGE_BOUND_ONLY,
        "tracking_completed_at": RANGE_BOUND_ONLY,
        "first_target_touch_at": RECOMPUTE_CHRONOLOGICALLY,
        "second_target_touch_at": RECOMPUTE_CHRONOLOGICALLY,
        "third_target_touch_at": RECOMPUTE_CHRONOLOGICALLY,
        "target_hit_at": RECOMPUTE_CHRONOLOGICALLY,
        "stop_hit_at": RECOMPUTE_CHRONOLOGICALLY,
        "position_best_price": RECOMPUTE_CHRONOLOGICALLY,
        "position_max_progress_percent": RECOMPUTE_CHRONOLOGICALLY,
        "position_tp1_reached": RECOMPUTE_CHRONOLOGICALLY,
        "maximum_favorable_excursion": RECOMPUTE_CHRONOLOGICALLY,
        "maximum_adverse_excursion": RECOMPUTE_CHRONOLOGICALLY,
        "mfe": RECOMPUTE_CHRONOLOGICALLY,
        "mae": RECOMPUTE_CHRONOLOGICALLY,
        "best_price": RECOMPUTE_CHRONOLOGICALLY,
        "best_progress": RECOMPUTE_CHRONOLOGICALLY,
        "current_progress": RECOMPUTE_CHRONOLOGICALLY,
        "current_r": RECOMPUTE_CHRONOLOGICALLY,
        "last_market_price": RECOMPUTE_CHRONOLOGICALLY,
        "last_evaluated_at": RECOMPUTE_CHRONOLOGICALLY,
        "position_last_state": STRIP_FROM_REPLAY,
        "position_state_history": STRIP_FROM_REPLAY,
        "exit_price": STRIP_FROM_REPLAY,
        "recorded_outcome": STRIP_FROM_REPLAY,
        "closed_reason": STRIP_FROM_REPLAY,
    }
    for field, classification in expected.items():
        assert REPLAY_FIELD_CLASSIFICATION[field] == classification


def test_replay_ignores_final_journal_target_touch_state_until_reached_chronologically():
    replay = replay_position_intelligence(position(
        ticker="OXY",
        direction="SHORT",
        timeframe="1D",
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
    assert replay["data_complete"] is False
    assert replay["outcome_category"] == "DATA_INCOMPLETE"
    assert "insufficient candles through recorded TP1 timestamp" in replay["data_gaps"]
    assert replay["time_in_each_state"]["PROTECT"]["candle_count"] == 0
    assert replay["time_in_each_state"]["WATCH"]["candle_count"] == 2


def test_daily_replay_includes_completion_date_and_can_refute_journal_tp1():
    oxy = position(
        ticker="OXY",
        direction="SHORT",
        timeframe="1D",
        entry_timestamp="2026-07-16T16:13:17.667006Z",
        exit_timestamp="2026-07-21T00:29:30.725000Z",
        actual_underlying_entry=54.28,
        original_stop=55.9,
        original_tp1=51.04,
        original_tp2=49.42,
        original_tp3=47.8,
        result="Win",
        outcome="TP1",
        completion_reason="target",
        first_target_touch_at="2026-07-21T00:29:30.724Z",
        position_last_state="PROTECT",
        position_best_price=51.04,
        position_tp1_reached=True,
    )
    replay = replay_position_intelligence(oxy, [
        candle("2026-07-16T04:00:00Z", 54.2, 53.53, 53.65),
        candle("2026-07-17T04:00:00Z", 55.17, 54.005, 54.86),
        candle("2026-07-20T04:00:00Z", 55.6699, 54.0, 55.19),
        candle("2026-07-21T04:00:00Z", 56.5, 55.36, 56.5),
        candle("2026-07-22T04:00:00Z", 57.82, 57.09, 57.5),
    ], timeframe="1D")
    assert replay["replay_window_filter"] == "date_inclusive"
    assert replay["candles_evaluated"] == 4
    assert replay["last_evaluated_candle_timestamp"] == "2026-07-21T04:00:00Z"
    assert replay["tp1_timestamp"] is None
    assert replay["stop_timestamp"] == "2026-07-21T04:00:00Z"
    assert replay["data_complete"] is True
    parity = classify_journal_replay_parity(oxy, replay)
    assert parity["status"] == "JOURNAL_EVENT_UNSUPPORTED"
    assert "TP1" in parity["reason"]


def test_complete_multi_day_replay_reaches_tp1_after_entry_date():
    replay = replay_position_intelligence(position(
        result="Win",
        outcome="TP1",
        completion_reason="target",
        first_target_touch_at="2026-07-03T20:00:00Z",
        exit_timestamp="2026-07-03T20:00:00Z",
    ), [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T14:00:00Z", 106, 101, 104),
        candle("2026-07-03T14:00:00Z", 111, 103, 108),
    ])
    assert replay["candles_evaluated"] == 3
    assert replay["tp1_timestamp"] == "2026-07-03T14:00:00Z"
    assert replay["data_complete"] is True
    assert classify_journal_replay_parity(position(result="Win", outcome="TP1", first_target_touch_at="2026-07-03T20:00:00Z"), replay)["status"] == "MATCH"


def test_long_and_short_tp1_touch_are_direction_aware():
    long = replay_position_intelligence(position(result="Win", first_target_touch_at="2026-07-02T14:00:00Z"), [
        candle("2026-07-01T14:00:00Z", 105, 99, 104),
        candle("2026-07-02T14:00:00Z", 110, 104, 108),
    ])
    assert long["tp1_timestamp"] == "2026-07-02T14:00:00Z"

    short = replay_position_intelligence(position(
        direction="SHORT",
        actual_underlying_entry=100,
        original_stop=105,
        original_tp1=90,
        result="Win",
        first_target_touch_at="2026-07-02T14:00:00Z",
    ), [
        candle("2026-07-01T14:00:00Z", 101, 95, 96),
        candle("2026-07-02T14:00:00Z", 96, 90, 92),
    ])
    assert short["tp1_timestamp"] == "2026-07-02T14:00:00Z"


def test_replay_output_ignores_misleading_final_values_for_long_and_short():
    final_values = {
        "position_last_state": "EXIT",
        "position_best_price": 999,
        "best_price": 999,
        "position_max_progress_percent": 999,
        "best_progress": 999,
        "current_progress": 999,
        "current_r": -99,
        "position_tp1_reached": True,
        "tp1_reached": True,
        "first_target_touch_at": "2026-07-10T14:00:00Z",
        "tp1_reached_at": "2026-07-10T14:00:00Z",
        "second_target_touch_at": "2026-07-10T14:00:00Z",
        "third_target_touch_at": "2026-07-10T14:00:00Z",
        "target_hit_at": "2026-07-10T14:00:00Z",
        "first_stop_touch_at": "2026-07-10T14:00:00Z",
        "stop_hit_at": "2026-07-10T14:00:00Z",
        "position_state_history": [{"event_id": "future-state"}],
        "maximum_favorable_excursion": 999,
        "maximum_adverse_excursion": -999,
        "maximum_favorable_excursion_r": 999,
        "maximum_adverse_excursion_r": -999,
        "mfe": 999,
        "mae": -999,
        "last_market_price": 1,
        "last_evaluated_at": "2026-07-10T14:00:00Z",
        "exit_price": 1,
        "recorded_outcome": "Win",
        "result": "Win",
        "outcome": "TP3",
        "completion_reason": "target",
        "closed_reason": "target",
        "reviewResult": "Winner",
    }

    long_candles = [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T14:00:00Z", 103, 97, 98),
    ]
    clean_long = replay_position_intelligence(position(position_best_price=None), long_candles)
    polluted_long = replay_position_intelligence(position(**final_values), long_candles)
    assert polluted_long["final_state"] == clean_long["final_state"]
    assert polluted_long["maximum_progress"] == clean_long["maximum_progress"]
    assert polluted_long["maximum_r"] == clean_long["maximum_r"]
    assert polluted_long["tp1_timestamp"] == clean_long["tp1_timestamp"]
    assert polluted_long["stop_timestamp"] == clean_long["stop_timestamp"]
    assert all(event.get("event_id") != "future-state" for event in polluted_long["timeline"])

    short_position = position(
        direction="SHORT",
        actual_underlying_entry=100,
        original_stop=105,
        original_tp1=90,
        original_tp2=85,
        original_tp3=80,
        position_best_price=None,
    )
    short_candles = [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T14:00:00Z", 103, 97, 102),
    ]
    clean_short = replay_position_intelligence(short_position, short_candles)
    polluted_short = replay_position_intelligence({**short_position, **final_values}, short_candles)
    assert polluted_short["final_state"] == clean_short["final_state"]
    assert polluted_short["maximum_progress"] == clean_short["maximum_progress"]
    assert polluted_short["maximum_r"] == clean_short["maximum_r"]
    assert polluted_short["tp1_timestamp"] == clean_short["tp1_timestamp"]
    assert polluted_short["stop_timestamp"] == clean_short["stop_timestamp"]


def test_recorded_exit_timestamp_only_bounds_replay_range():
    replay = replay_position_intelligence(position(
        exit_timestamp="2026-07-02T14:00:00Z",
        result="Win",
        outcome="TP3",
        completion_reason="target",
        first_target_touch_at="2026-07-03T14:00:00Z",
    ), [
        candle("2026-07-01T14:00:00Z", 101, 99, 100),
        candle("2026-07-02T14:00:00Z", 102, 99, 101),
        candle("2026-07-03T14:00:00Z", 120, 99, 115),
    ])
    assert replay["candles_evaluated"] == 2
    assert replay["recorded_outcome"] == "Win"
    assert replay["tp1_timestamp"] is None
    assert replay["final_state"] == "HEALTHY"
    assert replay["data_complete"] is False
    assert replay["outcome_category"] == "DATA_INCOMPLETE"


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


def test_replay_readiness_and_evidence_counts():
    ready = replay_readiness(position(scanner_timeframe="4H", actual_strike=100, actual_expiration="2026-08-21", actual_option_premium=1.2), candles_available=True, timeframe_source="journaled")
    assert ready["status"] == "REPLAY_READY"
    partial = replay_readiness(position(scanner_timeframe="", original_tp2=None, actual_expiration=""), candles_available=True, timeframe_source="inferred_default")
    assert partial["status"] == "PARTIALLY_READY"
    assert "setup timeframe" in partial["missing_optional"]
    assert "TP2" in partial["missing_optional"]
    missing_tp1 = replay_readiness(position(original_tp1=None, target_price=None, plannedTp1=None), candles_available=True)
    assert missing_tp1["status"] == "NOT_REPLAYABLE"
    invalid = replay_readiness(position(direction="SHORT", actual_underlying_entry=100, original_stop=95, original_tp1=90), candles_available=True)
    assert invalid["status"] == "NOT_REPLAYABLE"
    assert "invalid stop geometry" in invalid["invalid"]

    complete = replay_position_intelligence(position(position_best_price=None, result="Win"), [candle("2026-07-01T14:00:00Z", 104, 99, 103)])
    incomplete = replay_position_intelligence(position(journal_id="j-2", position_id="p-2"), [])
    counts = real_evidence_counts([complete, incomplete])
    assert counts["real_positions_replayed"] == 2
    assert counts["complete_real_replays"] == 1
    assert counts["incomplete_real_replays"] == 1
    guard = evidence_guard({"closed_complete_real_replays": 1})
    assert "No threshold recommendations" in guard["message"]
    evidence = evidence_log_from_replays([complete, incomplete, incomplete])
    assert any(item["observation_type"] == "DATA_GAP" for item in evidence)
    assert len([item for item in evidence if item["observation_type"] == "DATA_GAP" and item["position_id"] == "p-2"]) == 1


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
        assert empty["evidence_readiness"]["total_durable_positions"] == 0

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
            assert payload["evidence_readiness"]["total_durable_positions"] == 1
            assert payload["evidence_readiness"]["partially_ready"] == 1
            assert "Evidence sample is still developing" in payload["evidence_guard"]["message"]
            assert payload["synthetic_results_included"] is False
            single = client.get("/api/dev/position-replay/p-endpoint?summary_only=true", headers={"X-Kairos-Admin-Token": TOKEN}).json()
            assert single["ready"] is True
            assert single["replays"][0]["position_id"] == "p-endpoint"
            assert "timeline" not in single["replays"][0]
            before = repo.get_entry("j-endpoint")
            refresh = client.post("/api/dev/position-replay/refresh", headers={"X-Kairos-Admin-Token": TOKEN}, json={"mode": "stale"}).json()
            after = repo.get_entry("j-endpoint")
            assert refresh["refreshed_positions"] == 1
            assert after["replay_cache_status"] == "ready"
            assert after["result"] == before["result"]
            assert after["position_state_history"] == before["position_state_history"]
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
    test_replay_field_classification_documents_no_lookahead_inputs()
    test_replay_ignores_final_journal_target_touch_state_until_reached_chronologically()
    test_daily_replay_includes_completion_date_and_can_refute_journal_tp1()
    test_complete_multi_day_replay_reaches_tp1_after_entry_date()
    test_long_and_short_tp1_touch_are_direction_aware()
    test_replay_output_ignores_misleading_final_values_for_long_and_short()
    test_recorded_exit_timestamp_only_bounds_replay_range()
    test_watch_recovery_protect_exit_and_churn_metrics()
    test_milestone_dedup_and_target_stop_ambiguity()
    test_data_needed_and_original_plan_not_overwritten()
    test_multiple_trades_same_ticker_long_and_short()
    test_aggregate_sample_sizes_and_provider_failure_isolation()
    test_replay_readiness_and_evidence_counts()
    test_dev_endpoint_auth_empty_message_and_no_side_effects()
    print("Position replay v1 tests passed")
