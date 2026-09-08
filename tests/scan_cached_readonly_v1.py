"""scan_cached_readonly() (2026-09 session, scanner.py) -- the genuinely
side-effect-free sibling of scan_cached(), added specifically because
scan_cached() itself unconditionally submits a real background refresh
job on a stale or missing cache (verified: it never BLOCKS its caller,
but it is not side-effect-free). This file proves scan_cached_readonly()
never does that, under either cache condition, directly against the real
function -- not a reimplementation, and not inferred from
candidates_router.py's own (mocked) usage.
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scanner  # noqa: E402


def _reset_cache():
    scanner._analysis_cache.clear()


def test_missing_cache_returns_empty_and_submits_no_refresh(monkeypatch):
    _reset_cache()
    calls = []
    monkeypatch.setattr(scanner, "_submit_analysis_refresh", lambda *a, **k: calls.append((a, k)))
    result = scanner.scan_cached_readonly()
    assert result["rows"] == []
    assert result["near_miss"] == []
    assert result["meta"]["cache"] == "miss"
    assert calls == [], "a missing cache must never trigger a background refresh submission"


def test_stale_cache_returns_stale_rows_and_submits_no_refresh(monkeypatch):
    _reset_cache()
    key = scanner._analysis_cache_key(None, discover=False, universe="default")
    ancient = datetime.now(timezone.utc) - timedelta(seconds=100_000)  # far past ANALYSIS_CACHE_STALE_SECONDS
    scanner._analysis_cache[key] = {
        "rows": [{"ticker": "AMD"}], "near_miss": [], "scan_meta": {},
        "generated_at": ancient,
    }
    calls = []
    monkeypatch.setattr(scanner, "_submit_analysis_refresh", lambda *a, **k: calls.append((a, k)))
    result = scanner.scan_cached_readonly()
    assert [r.get("ticker") for r in result["rows"]] == ["AMD"], "a stale cache must still return its (stale) rows, not be treated as empty"
    assert result["meta"]["stale"] is True
    assert calls == [], "a STALE cache must never trigger a background refresh submission -- this is the exact bug scan_cached() itself has"
    _reset_cache()


def test_fresh_cache_returns_rows_and_submits_no_refresh(monkeypatch):
    _reset_cache()
    key = scanner._analysis_cache_key(None, discover=False, universe="default")
    scanner._analysis_cache[key] = {
        "rows": [{"ticker": "NVDA"}], "near_miss": [], "scan_meta": {},
        "generated_at": datetime.now(timezone.utc),
    }
    calls = []
    monkeypatch.setattr(scanner, "_submit_analysis_refresh", lambda *a, **k: calls.append((a, k)))
    result = scanner.scan_cached_readonly()
    assert [r.get("ticker") for r in result["rows"]] == ["NVDA"]
    assert result["meta"]["stale"] is False
    assert calls == []
    _reset_cache()


def test_scan_cached_itself_still_submits_a_refresh_on_a_stale_cache():
    """Contrast case, against the REAL, unmodified scan_cached() -- proves
    the distinction this task is about is real, not asserted. This is
    exactly why the dashboard was corrected to call scan_cached_readonly()
    instead."""
    _reset_cache()
    key = scanner._analysis_cache_key(None, discover=False, universe="default")
    ancient = datetime.now(timezone.utc) - timedelta(seconds=100_000)
    scanner._analysis_cache[key] = {
        "rows": [], "near_miss": [], "scan_meta": {},
        "generated_at": ancient,
    }
    submitted = {"called": False}
    real_submit = scanner._submit_background_job

    def _spy(*args, **kwargs):
        submitted["called"] = True
        return False  # pretend submission failed/no-ops -- we only care whether it was ATTEMPTED
    scanner._submit_background_job = _spy
    try:
        scanner.scan_cached()
    finally:
        scanner._submit_background_job = real_submit
        _reset_cache()
    assert submitted["called"] is True, "scan_cached() itself DOES attempt a background job submission on a stale cache -- the real behavior scan_cached_readonly() exists to avoid"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
