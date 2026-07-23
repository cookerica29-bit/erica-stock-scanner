import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import journal_store
import main
from trade_intelligence import (
    TRADE_INTELLIGENCE_VERSION,
    build_trade_intelligence_snapshot,
    build_verified_trade_records,
    group_metrics,
    similar_trade_insight,
    trade_intelligence_eligibility_funnel,
)


TOKEN = "trade-intelligence-test-token"


def entry(index=0, **overrides):
    base = {
        "journal_id": f"j-{index}",
        "position_id": f"p-{index}",
        "ticker": overrides.pop("ticker", "OXY"),
        "direction": overrides.pop("direction", "SHORT"),
        "setup_grade": overrides.pop("grade", "A"),
        "scanner_status_normalized": overrides.pop("status", "ENTER_NOW"),
        "scanner_timeframe": overrides.pop("timeframe", "4H"),
        "setupLocation": overrides.pop("location", "premium"),
        "confirmation_status": overrides.pop("confirmation", "confirmed"),
        "market_regime": overrides.pop("market_regime", "bearish"),
        "entry_timestamp": overrides.pop("entry_timestamp", f"2026-07-{(index % 20) + 1:02d}T14:00:00Z"),
        "planned_underlying_entry": overrides.pop("planned_underlying_entry", 54.28),
        "original_stop": overrides.pop("original_stop", 55.90),
        "original_tp1": overrides.pop("original_tp1", 51.04),
        "result": overrides.pop("result", "Win"),
        "outcome": overrides.pop("outcome", "TP1"),
    }
    base.update(overrides)
    return base


def replay(index=0, **overrides):
    base = {
        "journal_id": f"j-{index}",
        "position_id": f"p-{index}",
        "ticker": overrides.pop("ticker", "OXY"),
        "direction": overrides.pop("direction", "SHORT"),
        "data_complete": overrides.pop("data_complete", True),
        "entry_timestamp": overrides.pop("entry_timestamp", f"2026-07-{(index % 20) + 1:02d}T14:00:00Z"),
        "tp1_timestamp": overrides.pop("tp1_timestamp", f"2026-07-{(index % 20) + 1:02d}T18:00:00Z"),
        "tp2_timestamp": overrides.pop("tp2_timestamp", None),
        "tp3_timestamp": overrides.pop("tp3_timestamp", None),
        "stop_timestamp": overrides.pop("stop_timestamp", None),
        "outcome_category": overrides.pop("outcome_category", "CALM_WINNER"),
        "journal_replay_parity": overrides.pop("journal_replay_parity", {"status": "MATCH", "reason": "journal TP1 outcome reproduced by replay"}),
        "final_r": overrides.pop("final_r", 1.0),
        "maximum_r": overrides.pop("maximum_r", 1.5),
        "minimum_r": overrides.pop("minimum_r", -0.2),
        "MFE": overrides.pop("MFE", 1.5),
        "MAE": overrides.pop("MAE", -0.2),
    }
    base.update(overrides)
    return base


def analytics_record(index=0, status="VERIFIED"):
    return {
        "position_id": f"p-{index}",
        "analytics_verification": {"status": status},
    }


def independent_metrics(records):
    total = len(records)
    tp1_count = sum(1 for item in records if item["replay"].get("tp1_timestamp"))
    tp2_count = sum(1 for item in records if item["replay"].get("tp2_timestamp"))
    tp3_count = sum(1 for item in records if item["replay"].get("tp3_timestamp"))
    stop_count = sum(1 for item in records if item["replay"].get("stop_timestamp"))
    final_rs = sorted(float(item["replay"]["final_r"]) for item in records)
    mid = len(final_rs) // 2
    median_r = final_rs[mid] if len(final_rs) % 2 else (final_rs[mid - 1] + final_rs[mid]) / 2
    return {
        "tp1_rate": round(tp1_count / total * 100, 1),
        "tp2_rate": round(tp2_count / total * 100, 1),
        "tp3_rate": round(tp3_count / total * 100, 1),
        "stop_rate": round(stop_count / total * 100, 1),
        "average_r": round(sum(float(item["replay"]["final_r"]) for item in records) / total, 2),
        "median_r": round(median_r, 2),
        "average_maximum_favorable_excursion_r": round(sum(float(item["replay"]["MFE"]) for item in records) / total, 2),
        "average_maximum_drawdown_r": round(sum(abs(float(item["replay"]["MAE"])) for item in records) / total, 2),
    }


def verified_records(count=30, **entry_overrides):
    entries = [entry(i, **entry_overrides) for i in range(count)]
    replays = [replay(i, **entry_overrides) for i in range(count)]
    analytics = [analytics_record(i) for i in range(count)]
    return build_verified_trade_records(entries, replays, analytics)


def test_insufficient_sample_returns_clear_message():
    records = verified_records(2)
    insight = similar_trade_insight(entry(99), records, exact_min_trades=30, broad_min_trades=100)
    assert insight["available"] is False
    assert insight["message"] == "Not enough verified historical data yet."
    assert insight["exact_match_count"] == 2
    assert insight["broader_match_count"] == 2


def test_exact_group_metrics_use_replay_verified_trades():
    records = verified_records(30)
    insight = similar_trade_insight(entry(99), records)
    assert insight["available"] is True
    assert insight["group_type"] == "exact"
    assert insight["verified_trades"] == 30
    assert insight["metrics"]["tp1_rate"] == 100.0
    assert insight["metrics"]["average_r"] == 1.0
    assert insight["metrics"]["average_time_to_tp1_trading_days"] is not None
    assert insight["evidence"]["sample_size"] == 30


def test_broader_group_used_when_exact_group_is_too_small():
    records = []
    records.extend(verified_records(2, ticker="OXY"))
    records.extend(verified_records(4, ticker="DVN"))
    insight = similar_trade_insight(entry(99, ticker="OXY"), records, exact_min_trades=3, broad_min_trades=6)
    assert insight["available"] is True
    assert insight["group_type"] == "broader"
    assert insight["exact_match_count"] == 2
    assert insight["broader_match_count"] == 6


def test_journal_only_and_mismatched_outcomes_are_excluded():
    entries = [entry(0), entry(1), entry(2)]
    replays = [replay(0), replay(1), replay(2, data_complete=False)]
    analytics = [
        analytics_record(0, "VERIFIED"),
        analytics_record(1, "JOURNAL_REPLAY_MISMATCH"),
        analytics_record(2, "INSUFFICIENT_REPLAY_DATA"),
    ]
    records = build_verified_trade_records(entries, replays, analytics)
    assert len(records) == 1
    snapshot = build_trade_intelligence_snapshot(records)
    assert snapshot["version"] == TRADE_INTELLIGENCE_VERSION
    assert snapshot["verified_trade_count"] == 1


def test_dashboard_summaries_display_sample_sizes():
    records = verified_records(30)
    snapshot = build_trade_intelligence_snapshot(records)
    first_symbol = snapshot["dashboard"]["most_reliable_symbols"][0]
    assert first_symbol["key"] == "OXY"
    assert first_symbol["total_verified_trades"] == 30
    assert first_symbol["tp1_rate"] == 100.0


def test_eligibility_funnel_reconciles_included_and_excluded_records():
    entries = [
        entry(0, result="Open", outcome=""),
        entry(1),
        entry(2),
        entry(3),
        entry(4),
    ]
    replays = [
        replay(1),
        replay(2, data_complete=False),
        replay(3, tp1_timestamp=None, stop_timestamp="2026-07-03T18:00:00Z", outcome_category="STOP_DETECTED"),
    ]
    analytics = [
        analytics_record(0, "NOT_APPLICABLE"),
        analytics_record(1, "VERIFIED"),
        analytics_record(2, "INSUFFICIENT_REPLAY_DATA"),
        analytics_record(3, "JOURNAL_REPLAY_MISMATCH"),
        analytics_record(4, "REPLAY_PENDING"),
    ]
    funnel = trade_intelligence_eligibility_funnel(entries, replays, analytics)
    assert funnel["completed_journal_count"] == 4
    assert funnel["replay_available_count"] == 3
    assert funnel["replay_data_complete_count"] == 2
    assert funnel["eligible_trade_intelligence_count"] == 1
    assert funnel["journal_replay_mismatch_count"] == 1
    assert funnel["replay_pending_count"] == 1
    assert funnel["insufficient_replay_data_count"] == 1
    assert funnel["reconciliation"]["journal_records_reconciled"] is True
    assert funnel["reconciliation"]["completed_records_reconciled"] is True
    assert funnel["exclusion_reasons"]["NOT_COMPLETED"] == 1
    assert funnel["exclusion_reasons"]["JOURNAL_REPLAY_MISMATCH"] == 1


def test_knowledge_growth_and_group_progress_do_not_publish_below_threshold():
    records = verified_records(7)
    snapshot = build_trade_intelligence_snapshot(records)
    growth = snapshot["knowledge_growth"]
    assert growth["verified_trades_collected"] == 7
    assert growth["exact_threshold_progress"]["label"] == "7 of 30"
    assert growth["broader_threshold_progress"]["label"] == "7 of 100"
    assert growth["eligible_exact_groups"] == 0
    assert growth["eligible_broader_groups"] == 0
    assert growth["closest_exact_groups"][0]["verified_trades"] == 7
    assert growth["closest_exact_groups"][0]["eligible"] is False


def test_exact_and_broader_progress_messages_are_reported_without_rates():
    exact_records = verified_records(18)
    exact_insight = similar_trade_insight(entry(99), exact_records)
    assert exact_insight["available"] is False
    assert exact_insight["exact_progress"]["label"] == "18 of 30"
    assert "metrics" not in exact_insight

    broad_records = []
    broad_records.extend(verified_records(8, ticker="OXY"))
    for index in range(8, 64):
        broad_records.extend(verified_records(1, ticker=f"T{index}"))
    broad_insight = similar_trade_insight(entry(99, ticker="OXY"), broad_records)
    assert broad_insight["available"] is False
    assert broad_insight["exact_progress"]["label"] == "8 of 30"
    assert broad_insight["broader_progress"]["label"] == "64 of 100"
    assert "metrics" not in broad_insight


def test_eligible_exact_and_broader_thresholds_remain_unchanged():
    exact = similar_trade_insight(entry(99), verified_records(30))
    assert exact["available"] is True
    assert exact["group_type"] == "exact"
    assert exact["minimum_required"] == 30

    records = []
    records.extend(verified_records(2, ticker="OXY"))
    for index in range(2, 100):
        records.extend(verified_records(1, ticker=f"B{index}"))
    broader = similar_trade_insight(entry(99, ticker="OXY"), records)
    assert broader["available"] is True
    assert broader["group_type"] == "broader"
    assert broader["minimum_required"] == 100


def test_group_progress_is_sorted_capped_and_labeled():
    records = []
    records.extend(verified_records(18, ticker="AAA"))
    records.extend(verified_records(13, ticker="BBB", direction="LONG", market_regime="bullish"))
    records.extend(verified_records(4, ticker="CCC", grade="B"))
    snapshot = build_trade_intelligence_snapshot(records)
    exact_rows = snapshot["knowledge_growth"]["closest_exact_groups"]
    broad_rows = snapshot["knowledge_growth"]["closest_broader_groups"]
    assert len(exact_rows) <= 5
    assert len(broad_rows) <= 5
    assert exact_rows[0]["verified_trades"] == 18
    assert "AAA" in exact_rows[0]["label"]
    assert "|" not in exact_rows[0]["label"]


def test_independent_metric_validation_matches_group_metrics():
    records = verified_records(4)
    records[1]["replay"]["tp2_timestamp"] = "2026-07-02T20:00:00Z"
    records[2]["replay"]["tp1_timestamp"] = None
    records[2]["replay"]["stop_timestamp"] = "2026-07-03T18:00:00Z"
    records[2]["replay"]["final_r"] = -1.0
    records[2]["replay"]["MFE"] = 0.2
    records[2]["replay"]["MAE"] = -1.0
    records[3]["replay"]["tp3_timestamp"] = "2026-07-04T22:00:00Z"
    expected = independent_metrics(records)
    actual = group_metrics(records)
    for key, value in expected.items():
        assert actual[key] == value, f"{key}: {actual[key]} != {value}"


def test_protected_endpoint_uses_verified_analytics_and_replay():
    tmp = tempfile.TemporaryDirectory()
    previous_repo = main._journal_repository
    previous_replay = main._replay_positions
    previous_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    previous_cache = dict(main._trade_intelligence_cache)
    try:
        main._journal_repository = journal_store.SQLiteJournalRepository(str(Path(tmp.name) / "journal.sqlite3"))
        for i in range(3):
            main._journal_repository.create_entry(entry(i))
        os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN
        replay_calls = {"count": 0}

        def fake_replay(entries, summary_only=True):
            replay_calls["count"] += 1
            return [replay(index, position_id=item["position_id"]) for index, item in enumerate(entries)]

        main._replay_positions = fake_replay
        main._trade_intelligence_cache.update({"signature": None, "snapshot": None, "verified_records": []})
        client = TestClient(main.app)
        assert client.get("/api/dev/trade-intelligence").status_code == 403
        assert client.get("/api/dev/trade-intelligence", headers={"X-Kairos-Admin-Token": "wrong"}).status_code == 403
        payload = client.get("/api/dev/trade-intelligence", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert payload["version"] == TRADE_INTELLIGENCE_VERSION
        assert payload["verified_trade_count"] == 3
        assert payload["cache_signature"]
        assert payload["eligibility_funnel"]["eligible_trade_intelligence_count"] == 3
        assert payload["eligibility_funnel"]["reconciliation"]["journal_records_reconciled"] is True
        assert payload["data_quality"]["eligible_verified_trades"] == 3
        assert payload["knowledge_growth"]["exact_threshold_progress"]["label"] == "3 of 30"
        assert replay_calls["count"] == 1
        cached = client.get("/api/dev/trade-intelligence", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert cached["diagnostics"]["cache_status"] == "hit"
        assert cached["cache_signature"] == payload["cache_signature"]
        assert replay_calls["count"] == 1
        similar = client.post(
            "/api/dev/trade-intelligence/similar?exact_min_trades=3&broad_min_trades=3",
            headers={"X-Kairos-Admin-Token": TOKEN},
            json={"setup": entry(99)},
        ).json()
        assert similar["available"] is True
        assert similar["group_type"] == "exact"
        main._journal_repository.create_entry(entry(10, ticker="DVN"))
        refreshed = client.get("/api/dev/trade-intelligence", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert refreshed["verified_trade_count"] == 4
        assert replay_calls["count"] == 2
    finally:
        main._journal_repository = previous_repo
        main._replay_positions = previous_replay
        main._trade_intelligence_cache.update(previous_cache)
        if previous_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_token
        tmp.cleanup()


if __name__ == "__main__":
    test_insufficient_sample_returns_clear_message()
    test_exact_group_metrics_use_replay_verified_trades()
    test_broader_group_used_when_exact_group_is_too_small()
    test_journal_only_and_mismatched_outcomes_are_excluded()
    test_dashboard_summaries_display_sample_sizes()
    test_eligibility_funnel_reconciles_included_and_excluded_records()
    test_knowledge_growth_and_group_progress_do_not_publish_below_threshold()
    test_exact_and_broader_progress_messages_are_reported_without_rates()
    test_eligible_exact_and_broader_thresholds_remain_unchanged()
    test_group_progress_is_sorted_capped_and_labeled()
    test_independent_metric_validation_matches_group_metrics()
    test_protected_endpoint_uses_verified_analytics_and_replay()
    print("Trade Intelligence v1 tests passed")
