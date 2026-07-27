#!/usr/bin/env python3
"""Scanner health cache-status universe routing tests."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def test_cache_status_uses_discovered_universe_key():
    calls = []
    original_analysis_cache_status = main.analysis_cache_status
    original_discovery_symbols_ready = main._discovery_symbols_ready
    try:
        main._discovery_symbols_ready = lambda: (True, ["DOW", "UNP"], {"has_cache": True})

        def fake_analysis_cache_status(watchlist=None, **kwargs):
            calls.append({"watchlist": watchlist, **kwargs})
            return {
                "cache": "hit",
                "generated_at": "2026-07-27T20:47:53Z",
                "age_seconds": 60,
                "stale": False,
                "refreshing": False,
                "last_refresh_success_at": "2026-07-27T20:47:53Z",
                "last_refresh_duration": 12.8,
                "last_refresh_error": None,
                "symbols_attempted": 750,
                "symbols_terminally_evaluated": 750,
                "qualified_rows": 545,
                "near_miss_rows": 179,
                "partial_result": False,
            }

        main.analysis_cache_status = fake_analysis_cache_status
        response = TestClient(main.app).get("/api/cache/status?universe=discovered")
        assert response.status_code == 200
        payload = response.json()
        assert calls == [{"watchlist": ["DOW", "UNP"], "universe": "discovered"}]
        assert payload["universe"] == "discovered"
        assert payload["fresh"] is True
        assert payload["cache_age_seconds"] == 60
        assert payload["last_successful_refresh"] == "2026-07-27T20:47:53Z"
        assert payload["last_refresh_duration_ms"] == 12800.0
        assert payload["qualified_count"] == 545
        assert payload["near_miss_count"] == 179
    finally:
        main.analysis_cache_status = original_analysis_cache_status
        main._discovery_symbols_ready = original_discovery_symbols_ready


def test_cache_status_default_universe_stays_default():
    calls = []
    original_analysis_cache_status = main.analysis_cache_status
    try:
        def fake_analysis_cache_status(watchlist=None, **kwargs):
            calls.append({"watchlist": watchlist, **kwargs})
            return {"cache": "hit", "generated_at": "2026-07-27T20:40:00Z", "age_seconds": 30, "stale": False}

        main.analysis_cache_status = fake_analysis_cache_status
        response = TestClient(main.app).get("/api/cache/status?universe=default")
        assert response.status_code == 200
        assert response.json()["universe"] == "default"
        assert calls == [{"watchlist": None, "universe": "default"}]
    finally:
        main.analysis_cache_status = original_analysis_cache_status


def test_cache_status_rejects_unknown_universe():
    response = TestClient(main.app).get("/api/cache/status?universe=unknown")
    assert response.status_code == 422


if __name__ == "__main__":
    test_cache_status_uses_discovered_universe_key()
    test_cache_status_default_universe_stays_default()
    test_cache_status_rejects_unknown_universe()
    print("scanner_health_universe_api_v1 passed")
