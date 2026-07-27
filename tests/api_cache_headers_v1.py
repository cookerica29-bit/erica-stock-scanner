#!/usr/bin/env python3
"""Regression tests for no-store headers on live-data API responses."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


def assert_no_store(response):
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"


original_scan_cached = main.scan_cached
original_analysis_cache_status = main.analysis_cache_status
try:
    main.scan_cached = lambda *args, **kwargs: {
        "rows": [],
        "near_miss": [],
        "meta": {"cache": "test"},
    }
    main.analysis_cache_status = lambda *args, **kwargs: {"status": "ready"}

    client = TestClient(main.app)

    scan = client.get("/api/scan")
    assert scan.status_code == 200
    assert_no_store(scan)

    cache_status = client.get("/api/cache/status")
    assert cache_status.status_code == 200
    assert_no_store(cache_status)

    watchlist = client.get("/api/watchlist")
    assert watchlist.status_code == 200
    assert_no_store(watchlist)

    index = client.get("/")
    assert index.status_code == 200
    assert_no_store(index)
finally:
    main.scan_cached = original_scan_cached
    main.analysis_cache_status = original_analysis_cache_status

print("API cache headers v1 tests passed")
