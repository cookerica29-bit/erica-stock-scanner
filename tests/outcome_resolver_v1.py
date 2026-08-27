"""outcome_resolver.resolve_outcome -- pure bar-replay classification.

No DB/network here; see tests/candidate_promotion_outcome_watcher_v1.py for
the periodic-job orchestration (fetching real bars, updating the DB) built
on top of this.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from outcome_resolver import (
    DEFAULT_MAX_TRACKING_DAYS,
    OUTCOME_AMBIGUOUS,
    OUTCOME_EXPIRED,
    OUTCOME_HIT_STOP,
    OUTCOME_HIT_TARGET,
    OUTCOME_STILL_OPEN,
    resolve_outcome,
)

PROMOTED_AT = datetime(2026, 8, 1, 14, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 8, 10, 14, 0, tzinfo=timezone.utc)  # 9 days later, well within window


def _bars(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {time, high, low} (open/close filled in arbitrarily,
    unused by resolve_outcome but kept for realism)."""
    index = pd.to_datetime([r["time"] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "Open": [r.get("open", r["low"]) for r in rows],
            "High": [r["high"] for r in rows],
            "Low": [r["low"] for r in rows],
            "Close": [r.get("close", r["low"]) for r in rows],
        },
        index=index,
    )


def test_long_hits_target_first():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0},   # neither
        {"time": "2026-08-03T14:00:00Z", "high": 106.0, "low": 100.0},  # target=105 hit
    ])
    result = resolve_outcome("long", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_HIT_TARGET
    assert result["hit_at"] == "2026-08-03T14:00:00+00:00"
    assert result["note"] is None


def test_long_hits_stop_first():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0},
        {"time": "2026-08-03T14:00:00Z", "high": 102.0, "low": 94.0},  # stop=95 hit
        {"time": "2026-08-04T14:00:00Z", "high": 106.0, "low": 100.0},  # would've hit target -- too late
    ])
    result = resolve_outcome("long", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_HIT_STOP
    assert result["hit_at"] == "2026-08-03T14:00:00+00:00"


def test_short_hits_target_first():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0},
        {"time": "2026-08-03T14:00:00Z", "high": 101.0, "low": 94.0},  # target=95 hit (short: low <= target)
    ])
    result = resolve_outcome("short", stop=105.0, target=95.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_HIT_TARGET


def test_short_hits_stop_first():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 106.0, "low": 99.0},  # stop=105 hit (short: high >= stop)
    ])
    result = resolve_outcome("short", stop=105.0, target=95.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_HIT_STOP


def test_same_bar_touches_both_is_ambiguous_not_guessed():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 106.0, "low": 94.0},  # both target=105 and stop=95 inside range
    ])
    result = resolve_outcome("long", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_AMBIGUOUS
    assert result["hit_at"] is None
    assert "can't be determined" in result["note"]


def test_no_resolution_within_window_is_still_open():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0},
        {"time": "2026-08-03T14:00:00Z", "high": 102.0, "low": 98.0},
    ])
    result = resolve_outcome("long", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_STILL_OPEN
    assert result["hit_at"] is None


def test_no_resolution_past_max_tracking_days_is_expired():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0},
    ])
    far_future = PROMOTED_AT + timedelta(days=DEFAULT_MAX_TRACKING_DAYS + 1)
    result = resolve_outcome("long", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=far_future)
    assert result["outcome"] == OUTCOME_EXPIRED
    assert "45-day" in result["note"] or "tracking window" in result["note"]


def test_no_valid_target_can_still_resolve_hit_stop():
    """The no_valid_target promotion case (target is None): hit_target must
    never trigger, no matter how favorable price action looks."""
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 500.0, "low": 99.0},  # huge favorable move, but no target to hit
        {"time": "2026-08-03T14:00:00Z", "high": 200.0, "low": 94.0},  # stop=95 hit
    ])
    result = resolve_outcome("long", stop=95.0, target=None, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_HIT_STOP


def test_no_valid_target_stays_still_open_indefinitely_within_window():
    bars = _bars([
        {"time": "2026-08-02T14:00:00Z", "high": 500.0, "low": 99.0},
    ])
    result = resolve_outcome("long", stop=95.0, target=None, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_STILL_OPEN


def test_empty_bars_with_no_resolution_yet_is_still_open():
    bars = _bars([])
    result = resolve_outcome("long", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] == OUTCOME_STILL_OPEN


def test_unsupported_direction_returns_none_outcome_with_note():
    bars = _bars([{"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0}])
    result = resolve_outcome("sideways", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW)
    assert result["outcome"] is None
    assert "unsupported direction" in result["note"]


def test_bar_source_is_threaded_through():
    bars = _bars([{"time": "2026-08-02T14:00:00Z", "high": 106.0, "low": 100.0}])
    result = resolve_outcome("long", stop=95.0, target=105.0, bars=bars, promoted_at=PROMOTED_AT, now=NOW, bar_source="daily")
    assert result["bar_source"] == "daily"
