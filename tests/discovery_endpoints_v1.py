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
    os.environ["DISCOVERY_ADMIN_TOKEN"] = "secret"
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = fake_discovery_result
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
        assert calls == [(["AAPL", "F", "KMI"], {"force_refresh": False, "universe": "discovered"})]
        assert response.json()["meta"]["cache_key"] == "discovered"
    finally:
        main.scan_cached = original_scan_cached
        reset_discovery_cache()


def test_scan_default_and_finviz_modes_remain_unchanged():
    original_scan_cached = main.scan_cached
    calls = []

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {"rows": [], "near_miss": [], "meta": {}}

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        client.get("/api/scan")
        client.get("/api/scan?universe=default")
        client.get("/api/scan?discover=true")
        client.get("/api/scan?universe=finviz")
        assert calls == [
            (None, {"force_refresh": False, "discover": False}),
            (None, {"force_refresh": False, "discover": False}),
            (None, {"force_refresh": False, "discover": True}),
            (None, {"force_refresh": False, "discover": True}),
        ]
    finally:
        main.scan_cached = original_scan_cached


def main_test() -> int:
    test_discovery_status_warming_without_cache()
    test_discovery_run_disabled_when_token_unset()
    test_discovery_run_rejects_wrong_token()
    test_discovery_run_populates_cache_with_valid_token()
    test_scan_discovered_universe_returns_warming_when_cache_missing()
    test_scan_discovered_universe_uses_cached_symbols_without_touching_default_or_finviz()
    test_scan_default_and_finviz_modes_remain_unchanged()
    print("Discovery endpoints v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
