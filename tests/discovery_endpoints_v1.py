#!/usr/bin/env python3
"""Tests for manual discovery-universe cache endpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def reset_discovery_cache():
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": [],
            "generated_at": None,
            "expires_at": None,
            "pipeline_counts": {},
            "thresholds": {},
            "formula": {},
            "stage3": {},
            "stage4": {},
            "top_20": [],
            "bottom_20_selected": [],
            "watchlist_overlap": {},
            "last_error": None,
            "last_duration": None,
            "job_id": None,
            "running": False,
            "started_at": None,
            "completed_at": None,
            "metrics": {},
        })


def reset_handoff_state():
    with main._discovered_scan_handoff_lock:
        main._discovered_scan_handoff_state.update({
            "handoff_triggered_at": None,
            "handoff_reason": None,
            "refresh_job_id": None,
            "refresh_started_at": None,
            "refresh_completed_at": None,
            "refresh_failed_at": None,
            "refresh_attempt_count": 0,
            "stale_job_recovered": False,
            "scanner_cache_generation": None,
            "last_checked_at": None,
            "last_result": None,
        })


class InlineExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

        class Done:
            pass

        return Done()


def fake_discovery_result(static_watchlist=None):
    return {
        "symbols": ["AAPL", "MSFT", "F"],
        "pipeline_counts": {
            "raw_assets": 10,
            "tradable_optionable": 8,
            "hygiene_passed": 7,
            "dollar_volume_passed": 6,
            "options_liquidity_passed": 5,
            "ranked": 5,
            "selected": 3,
        },
        "thresholds": {"target_universe_size": 550},
        "formula": {"combined_liquidity_score": "test"},
        "stage3": {"elapsed_seconds": 1.2},
        "stage4": {"elapsed_seconds": 2.3},
        "top_20": [{"symbol": "AAPL", "rank": 1}],
        "bottom_20_selected": [{"symbol": "F", "rank": 3}],
        "watchlist_overlap": {
            "watchlist_count": len(static_watchlist or []),
            "overlap": 3,
            "missing": [],
        },
        "elapsed_seconds": 3.5,
    }


def assert_utc_z_timestamp(value):
    assert isinstance(value, str)
    assert value.endswith("Z")
    parsed = __import__("datetime").datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_discovery_status_warming_without_cache():
    previous_token = os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
    reset_discovery_cache()
    try:
        client = TestClient(main.app)
        response = client.get("/api/discovery/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "warming"
        assert payload["enabled"] is False
        assert payload["has_cache"] is False
        assert payload["selected_count"] == 0
        symbols = client.get("/api/discovery/symbols").json()
        assert symbols["symbols"] == []
        assert symbols["count"] == 0
        assert symbols["status"]["status"] == "warming"
    finally:
        if previous_token is not None:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


def test_discovery_run_disabled_when_token_unset():
    previous_token = os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
    reset_discovery_cache()
    try:
        client = TestClient(main.app)
        response = client.post("/api/discovery/run")
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"]
    finally:
        if previous_token is not None:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


def test_discovery_run_rejects_wrong_token():
    previous_token = os.environ.get("DISCOVERY_ADMIN_TOKEN")
    os.environ["DISCOVERY_ADMIN_TOKEN"] = "secret"
    reset_discovery_cache()
    try:
        client = TestClient(main.app)
        response = client.post("/api/discovery/run", headers={"X-Kairos-Admin-Token": "wrong"})
        assert response.status_code == 403
        assert "Invalid" in response.json()["detail"]
    finally:
        if previous_token is None:
            os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
        else:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


def test_discovery_run_populates_cache_with_valid_token():
    previous_token = os.environ.get("DISCOVERY_ADMIN_TOKEN")
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    os.environ["DISCOVERY_ADMIN_TOKEN"] = "secret"
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = fake_discovery_result
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    reset_discovery_cache()
    try:
        client = TestClient(main.app)
        response = client.post("/api/discovery/run?refresh=true", headers={"X-Kairos-Admin-Token": "secret"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["accepted"] is True
        assert payload["status"]["status"] == "ready"
        assert payload["status"]["selected_count"] == 3
        assert payload["status"]["pipeline_counts"]["selected"] == 3
        assert payload["status"]["metrics"]["final_admitted_symbol_count"] == 3
        assert payload["status"]["metrics"]["effective_cap"] == 550
        assert_utc_z_timestamp(payload["status"]["metrics"]["discovery_started_at"])
        assert_utc_z_timestamp(payload["status"]["metrics"]["discovery_completed_at"])
        assert payload["status"]["last_error"] is None

        symbols = client.get("/api/discovery/symbols").json()
        assert symbols["symbols"] == ["AAPL", "MSFT", "F"]
        assert symbols["count"] == 3
        assert symbols["status"]["has_cache"] is True

        status = client.get("/api/discovery/status").json()
        assert status["enabled"] is True
        assert status["status"] == "ready"
        assert status["stale"] is False
        assert status["generated_at"]
        assert status["expires_at"]
    finally:
        main._discovery_universe_executor = previous_executor
        main.build_ranked_discovery_universe = previous_builder
        main._maybe_enqueue_discovered_scan_handoff = previous_handoff
        reset_discovery_cache()
        if previous_token is None:
            os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
        else:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


def test_scan_discovered_universe_returns_warming_when_cache_missing():
    reset_discovery_cache()
    client = TestClient(main.app)
    response = client.get("/api/scan?universe=discovered")
    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == []
    assert payload["near_miss"] == []
    assert payload["meta"]["status"] == "warming"
    assert payload["meta"]["universe"] == "discovered"
    assert payload["meta"]["cache_key"] == "discovered"
    assert "Discovery universe is not ready" in payload["meta"]["message"]


def test_scan_discovered_universe_uses_cached_symbols_without_touching_default_or_finviz():
    original_scan_cached = main.scan_cached
    calls = []
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL", "F", "KMI"],
            "generated_at": __import__("datetime").datetime.utcnow(),
            "expires_at": __import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "cache_key": kwargs.get("universe") or "default",
                "configured_universe_count": len(watchlist or []),
            },
        }

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        response = client.get("/api/scan?universe=discovered")
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == ["AAPL", "F", "KMI"]
        assert calls[0][1]["force_refresh"] is False
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        assert calls[0][1]["coverage_context"]["universe_source"] == "discovered"
        assert calls[0][1]["coverage_context"]["universe_symbol_count"] == 3
        assert response.json()["meta"]["cache_key"] == "discovered"
    finally:
        main.scan_cached = original_scan_cached
        reset_discovery_cache()


def test_scan_discovered_universe_passes_full_cached_symbol_list_without_truncation():
    original_scan_cached = main.scan_cached
    calls = []
    symbols = [f"T{i}" for i in range(550)]
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": symbols,
            "generated_at": __import__("datetime").datetime.utcnow(),
            "expires_at": __import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "cache_key": kwargs.get("universe") or "default",
                "configured_universe_count": len(watchlist or []),
            },
        }

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        response = client.get("/api/scan?universe=discovered")
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == symbols
        assert calls[0][1]["force_refresh"] is False
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        assert calls[0][1]["coverage_context"]["universe_symbol_count"] == 550
        assert response.json()["meta"]["configured_universe_count"] == 550
    finally:
        main.scan_cached = original_scan_cached
        reset_discovery_cache()


def test_scan_default_route_uses_discovered_universe_by_default():
    original_scan_cached = main.scan_cached
    calls = []
    symbols = [f"T{i}" for i in range(750)]
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": symbols,
            "generated_at": __import__("datetime").datetime.utcnow(),
            "expires_at": __import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "cache_key": kwargs.get("universe") or "default",
                "configured_universe_count": len(watchlist or []),
            },
        }

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        response = client.get("/api/scan")
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == symbols
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        assert calls[0][1]["coverage_context"]["universe_symbol_count"] == 750
        assert response.json()["meta"]["configured_universe_count"] == 750
    finally:
        main.scan_cached = original_scan_cached
        reset_discovery_cache()


def test_scan_explicit_default_and_finviz_modes_remain_available():
    original_scan_cached = main.scan_cached
    calls = []

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {"rows": [], "near_miss": [], "meta": {}}

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        client.get("/api/scan?universe=default")
        client.get("/api/scan?discover=true")
        client.get("/api/scan?universe=finviz")
        assert calls == [
            (None, {"force_refresh": False, "discover": False}),
            (None, {"force_refresh": False, "discover": True}),
            (None, {"force_refresh": False, "discover": True}),
        ]
    finally:
        main.scan_cached = original_scan_cached


def test_discovery_cache_refresh_needed_for_missing_and_stale_cache():
    reset_discovery_cache()
    assert main._discovery_cache_needs_refresh() is True

    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })
    assert main._discovery_cache_needs_refresh() is False

    with main._discovery_universe_lock:
        main._discovery_universe_cache["expires_at"] = now - __import__("datetime").timedelta(seconds=1)
    assert main._discovery_cache_needs_refresh() is True
    reset_discovery_cache()


def test_discovery_auto_submit_skips_fresh_cache_and_running_job():
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = fake_discovery_result
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    reset_discovery_cache()
    try:
        accepted, job_id = main._submit_discovery_universe_job_if_needed()
        assert accepted is True
        assert job_id.startswith("discovery:")
        assert main._discovery_status_snapshot()["status"] == "ready"

        accepted, reason = main._submit_discovery_universe_job_if_needed()
        assert accepted is False
        assert reason == "cache fresh"

        with main._discovery_universe_lock:
            main._discovery_universe_cache.update({
                "running": True,
                "symbols": [],
                "generated_at": None,
                "expires_at": None,
            })
        accepted, reason = main._submit_discovery_universe_job_if_needed()
        assert accepted is False
        assert reason == "already running"
    finally:
        main._discovery_universe_executor = previous_executor
        main.build_ranked_discovery_universe = previous_builder
        main._maybe_enqueue_discovered_scan_handoff = previous_handoff
        reset_discovery_cache()


def test_startup_registers_and_submits_discovery_refresh():
    previous_register = main.register_background_periodic_task
    previous_submit = main._submit_discovery_universe_job_if_needed
    previous_start_market_cache = main.start_market_cache_refresh
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    calls = []
    main.register_background_periodic_task = lambda key, ttl, callback: calls.append(("register", key, ttl, callback))
    main._submit_discovery_universe_job_if_needed = lambda: calls.append(("submit",)) or (True, "job")
    main.start_market_cache_refresh = lambda: calls.append(("market_cache",))
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": calls.append(("handoff", reason)) or (False, "stubbed")
    try:
        main.startup_market_cache_refresh()
        assert calls[0][0:3] == ("register", "discovery_universe", main.DISCOVERY_UNIVERSE_TTL_SECONDS)
        assert callable(calls[0][3])
        assert calls[1][0:3] == ("register", "discovered_scan_handoff", 30)
        assert callable(calls[1][3])
        assert calls[2] == ("market_cache",)
        assert calls[3] == ("submit",)
        assert calls[4] == ("handoff", "startup_discovery_ready_no_scanner_cache")
    finally:
        main.register_background_periodic_task = previous_register
        main._submit_discovery_universe_job_if_needed = previous_submit
        main.start_market_cache_refresh = previous_start_market_cache
        main._maybe_enqueue_discovered_scan_handoff = previous_handoff


def test_discovered_scan_handoff_queues_when_discovery_ready_and_scan_cache_missing():
    reset_discovery_cache()
    reset_handoff_state()
    previous_status = main.analysis_cache_status
    previous_scan_cached = main.scan_cached
    calls = []
    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL", "MSFT"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_status(watchlist=None, **kwargs):
        return {
            "has_cache": False,
            "status": "warming",
            "refreshing": False,
            "generated_at": None,
            "last_refresh_error": None,
        }

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "has_cache": False,
                "status": "warming",
                "refreshing": True,
                "refresh_job_id": "universe:discovered:1:test",
                "refresh_started_at": "2026-08-03T00:00:00Z",
            },
        }

    main.analysis_cache_status = fake_status
    main.scan_cached = fake_scan_cached
    try:
        accepted, job = main._maybe_enqueue_discovered_scan_handoff("test_missing_cache")
        assert accepted is True
        assert job == "universe:discovered:1:test"
        assert len(calls) == 1
        assert calls[0][0] == ["AAPL", "MSFT"]
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        snapshot = main._discovered_scan_handoff_snapshot()
        assert snapshot["handoff_reason"] == "test_missing_cache"
        assert snapshot["refresh_attempt_count"] == 1
        assert snapshot["refresh_job_id"] == "universe:discovered:1:test"
    finally:
        main.analysis_cache_status = previous_status
        main.scan_cached = previous_scan_cached
        reset_discovery_cache()
        reset_handoff_state()


def test_discovered_scan_handoff_does_not_duplicate_running_or_ready_cache():
    reset_discovery_cache()
    reset_handoff_state()
    previous_status = main.analysis_cache_status
    previous_scan_cached = main.scan_cached
    calls = []
    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    main.scan_cached = lambda *args, **kwargs: calls.append((args, kwargs))
    try:
        main.analysis_cache_status = lambda *args, **kwargs: {
            "has_cache": False,
            "status": "warming",
            "refreshing": True,
            "refresh_job_id": "already-running",
        }
        accepted, reason = main._maybe_enqueue_discovered_scan_handoff("test_running")
        assert accepted is False
        assert reason == "scanner refresh already running"
        assert calls == []

        main.analysis_cache_status = lambda *args, **kwargs: {
            "has_cache": True,
            "status": "fresh",
            "refreshing": False,
            "generated_at": "2026-08-03T00:00:00Z",
        }
        accepted, reason = main._maybe_enqueue_discovered_scan_handoff("test_ready")
        assert accepted is False
        assert reason == "scanner cache ready"
        assert calls == []
        assert main._discovered_scan_handoff_snapshot()["scanner_cache_generation"] == "2026-08-03T00:00:00Z"
    finally:
        main.analysis_cache_status = previous_status
        main.scan_cached = previous_scan_cached
        reset_discovery_cache()
        reset_handoff_state()


def test_discovered_cache_status_triggers_handoff_and_exposes_diagnostics():
    reset_discovery_cache()
    reset_handoff_state()
    previous_status = main.analysis_cache_status
    previous_scan_cached = main.scan_cached
    calls = []
    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_status(watchlist=None, **kwargs):
        return {
            "has_cache": False,
            "status": "warming",
            "refreshing": bool(calls),
            "generated_at": None,
            "refresh_job_id": "handoff-job" if calls else None,
            "refresh_started_at": "2026-08-03T00:00:00Z" if calls else None,
        }

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "has_cache": False,
                "status": "warming",
                "refreshing": True,
                "refresh_job_id": "handoff-job",
                "refresh_started_at": "2026-08-03T00:00:00Z",
            },
        }

    main.analysis_cache_status = fake_status
    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        payload = client.get("/api/cache/status?universe=discovered").json()
        assert len(calls) == 1
        assert payload["status"] == "scanner_warming"
        assert payload["refreshing"] is True
        assert payload["handoff"]["handoff_reason"] == "cache_status_discovery_ready_no_scanner_cache"
        assert payload["handoff"]["refresh_job_id"] == "handoff-job"
        assert payload["message"] == "Discovery ready; scanning the 1,000-symbol universe."
    finally:
        main.analysis_cache_status = previous_status
        main.scan_cached = previous_scan_cached
        reset_discovery_cache()
        reset_handoff_state()


def test_coverage_baseline_endpoint_warms_without_completed_discovered_scan():
    reset_discovery_cache()
    previous_snapshot = main.coverage_baseline_snapshot
    main.coverage_baseline_snapshot = lambda: None
    try:
        client = TestClient(main.app)
        payload = client.get("/api/coverage/baseline").json()
        assert payload["status"] == "warming"
        assert payload["ready"] is False
        assert payload["generated_at"] is None
        assert payload["scan"] == {}
    finally:
        main.coverage_baseline_snapshot = previous_snapshot


def test_coverage_baseline_endpoint_returns_latest_snapshot_without_starting_work():
    previous_snapshot = main.coverage_baseline_snapshot
    previous_scan_cached = main.scan_cached
    calls = []
    snapshot = {
        "generated_at": "2026-07-22T12:00:00Z",
        "discovery": {"final_admitted_symbol_count": 550},
        "scan": {"universe_source": "discovered", "symbols_requested": 550},
        "stage_distribution": {"Enter Now": 1},
        "grade_distribution": {"A": 1, "B": 0, "C": 0, "unknown": 0},
        "contract_distribution": {"suggested contract available": 1},
        "blocker_distribution": {},
        "provider_failures": {},
    }
    main.coverage_baseline_snapshot = lambda: snapshot
    main.scan_cached = lambda *args, **kwargs: calls.append((args, kwargs))
    try:
        client = TestClient(main.app)
        payload = client.get("/api/coverage/baseline").json()
        assert payload["status"] == "ready"
        assert payload["ready"] is True
        assert payload["scan"]["symbols_requested"] == 550
        assert payload["stage_distribution"]["Enter Now"] == 1
        assert calls == []
    finally:
        main.coverage_baseline_snapshot = previous_snapshot
        main.scan_cached = previous_scan_cached


def main_test() -> int:
    test_discovery_status_warming_without_cache()
    test_discovery_run_disabled_when_token_unset()
    test_discovery_run_rejects_wrong_token()
    test_discovery_run_populates_cache_with_valid_token()
    test_scan_discovered_universe_returns_warming_when_cache_missing()
    test_scan_discovered_universe_uses_cached_symbols_without_touching_default_or_finviz()
    test_scan_discovered_universe_passes_full_cached_symbol_list_without_truncation()
    test_scan_default_route_uses_discovered_universe_by_default()
    test_scan_explicit_default_and_finviz_modes_remain_available()
    test_discovery_cache_refresh_needed_for_missing_and_stale_cache()
    test_discovery_auto_submit_skips_fresh_cache_and_running_job()
    test_startup_registers_and_submits_discovery_refresh()
    test_discovered_scan_handoff_queues_when_discovery_ready_and_scan_cache_missing()
    test_discovered_scan_handoff_does_not_duplicate_running_or_ready_cache()
    test_discovered_cache_status_triggers_handoff_and_exposes_diagnostics()
    test_coverage_baseline_endpoint_warms_without_completed_discovered_scan()
    test_coverage_baseline_endpoint_returns_latest_snapshot_without_starting_work()
    print("Discovery endpoints v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
