from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def reset_refresh_state():
    with scanner._background_jobs_lock:
        scanner._background_jobs.clear()
        scanner._analysis_refresh_state.clear()
    with scanner._cache_lock:
        scanner._analysis_cache.clear()


def assert_between(value, low, high, label):
    assert low <= value <= high, f"{label}: expected {low} <= {value} <= {high}, got {value}"


reset_refresh_state()

# Worker-count configuration parsing is safe and preserves the production default.
assert scanner._parse_background_analysis_scan_workers(None) == 4
assert scanner._parse_background_analysis_scan_workers("") == 4
assert scanner._parse_background_analysis_scan_workers("bad") == 4
assert scanner._parse_background_analysis_scan_workers("0") == 1
assert scanner._parse_background_analysis_scan_workers("-3") == 1
assert scanner._parse_background_analysis_scan_workers("7") == 7

def active_worker_count_with_env(value):
    env = os.environ.copy()
    env["BACKGROUND_ANALYSIS_SCAN_WORKERS"] = value
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import scanner; print(scanner.BACKGROUND_ANALYSIS_SCAN_WORKERS)",
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    return int(output.strip().splitlines()[-1])


assert active_worker_count_with_env("6") == 6
assert active_worker_count_with_env("not-a-number") == 4

# Cache timestamps normalize aware UTC values and preserve the public Z format.
key = scanner._analysis_cache_key(["AAPL"])
aware_generated_at = datetime.now(timezone.utc) - timedelta(seconds=45)
meta = scanner._analysis_cache_meta(key, {"generated_at": aware_generated_at, "rows": [], "near_miss": []}, False)
assert meta["generated_at"].endswith("Z")
assert_between(meta["age_seconds"], 40, 60, "aware UTC cache age")
assert meta["stale"] is False

# Legacy naive UTC cache timestamps continue to calculate age without aware/naive comparison failures.
legacy_naive_generated_at = datetime.utcnow() - timedelta(seconds=scanner.ANALYSIS_CACHE_STALE_SECONDS + 15)
meta = scanner._analysis_cache_meta(key, {"generated_at": legacy_naive_generated_at, "rows": [], "near_miss": []}, False)
assert meta["generated_at"].endswith("Z")
assert meta["age_seconds"] >= scanner.ANALYSIS_CACHE_STALE_SECONDS
assert meta["stale"] is True

# Legacy naive UTC timestamps round-trip through cache metadata serialization without changing the effective age.
legacy_roundtrip_generated_at = datetime.utcnow() - timedelta(seconds=75)
meta = scanner._analysis_cache_meta(key, {"generated_at": legacy_roundtrip_generated_at, "rows": [], "near_miss": []}, False)
serialized = meta["generated_at"]
assert serialized.endswith("Z")
parsed_roundtrip = scanner._coerce_utc_datetime(serialized)
assert parsed_roundtrip.tzinfo is not None
fixed_now = datetime.now(timezone.utc)
original_age = scanner._age_seconds(legacy_roundtrip_generated_at, now=fixed_now)
roundtrip_age = scanner._age_seconds(parsed_roundtrip, now=fixed_now)
assert abs(original_age - roundtrip_age) < 0.01

# Non-UTC aware timestamps convert to the equivalent UTC instant before formatting and age math.
offset_timestamp = "2026-07-16T20:00:00+05:00"
converted = scanner._coerce_utc_datetime(offset_timestamp)
assert converted == datetime(2026, 7, 16, 15, 0, 0, tzinfo=timezone.utc)
assert scanner._format_utc_timestamp(offset_timestamp) == "2026-07-16T15:00:00Z"
age_from_offset = scanner._age_seconds(offset_timestamp, now=datetime(2026, 7, 16, 16, 30, 0, tzinfo=timezone.utc))
assert age_from_offset == 90 * 60

# ISO strings from older serialized cache paths also normalize safely.
iso_generated_at = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
meta = scanner._analysis_cache_meta(key, {"generated_at": iso_generated_at, "rows": [], "near_miss": []}, False)
assert_between(meta["age_seconds"], 20, 45, "ISO string cache age")
assert meta["stale"] is False

# Stored cache entries now use timezone-aware UTC datetimes.
cached = scanner._store_analysis_cache(key, [{"ticker": "AAPL"}], [], {"symbols_attempted": 1})
assert cached["generated_at"].tzinfo is not None
assert cached["generated_at"].utcoffset() == timedelta(0)

# Refresh submission success explicitly returns True and records started state.
reset_refresh_state()
original_submit_background_job = scanner._submit_background_job
try:
    calls = []

    def submit_success(refresh_key, fn, *args, **kwargs):
        calls.append((refresh_key, args))
        return True

    scanner._submit_background_job = submit_success
    assert scanner._submit_analysis_refresh(key, ["AAPL"], reason="test") is True
    assert calls and calls[0][0] == scanner._analysis_refresh_key(key)
    started_state = scanner._analysis_refresh_state[scanner._analysis_state_key(key)]
    assert started_state["refresh_started_at"].tzinfo is not None
finally:
    scanner._submit_background_job = original_submit_background_job

# Already-active refreshes explicitly return False and do not submit a duplicate.
reset_refresh_state()
refresh_key = scanner._analysis_refresh_key(key)
with scanner._background_jobs_lock:
    scanner._background_jobs.add(refresh_key)
    scanner._analysis_refresh_state[scanner._analysis_state_key(key)] = {
        "refresh_job_id": "existing",
        "refresh_started_at": datetime.now(timezone.utc),
    }
assert scanner._submit_analysis_refresh(key, ["AAPL"], reason="duplicate") is False
with scanner._background_jobs_lock:
    scanner._background_jobs.discard(refresh_key)

# Executor submission failure returns False and _submit_background_job cleans the job key.
class FailingExecutor:
    def submit(self, _fn):
        raise RuntimeError("executor closed")


reset_refresh_state()
original_executor = scanner._background_executor
try:
    scanner._background_executor = FailingExecutor()
    failed_key = ("refresh-safety", "failure")
    assert scanner._submit_background_job(failed_key, lambda: None) is False
    with scanner._background_jobs_lock:
        assert failed_key not in scanner._background_jobs
finally:
    scanner._background_executor = original_executor

# scan_cached treats a failed refresh submission as "serve cached data", not success or already-running metadata.
reset_refresh_state()
stale_cached = {
    "rows": [{"ticker": "AAPL"}],
    "near_miss": [],
    "generated_at": datetime.now(timezone.utc) - timedelta(seconds=scanner.ANALYSIS_CACHE_STALE_SECONDS + 20),
    "scan_meta": {},
}
with scanner._cache_lock:
    scanner._analysis_cache[key] = stale_cached
original_submit_analysis_refresh = scanner._submit_analysis_refresh
try:
    scanner._submit_analysis_refresh = lambda *args, **kwargs: False
    result = scanner.scan_cached(["AAPL"], force_refresh=True)
    assert result["rows"] == [{"ticker": "AAPL"}]
    assert result["meta"]["refresh_requested"] is True
    assert result["meta"]["stale"] is True
finally:
    scanner._submit_analysis_refresh = original_submit_analysis_refresh

print("Production refresh safety v1 tests passed")
