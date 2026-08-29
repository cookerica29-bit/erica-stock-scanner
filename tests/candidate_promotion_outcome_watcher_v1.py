"""main._watch_candidate_promotion_outcomes -- the hourly bar-replay resolver.

Structural precedent: _momentum_short_lifecycle_watch_open_records
(main.py) -- load open records, skip terminal ones, fetch fresh data for
what's still pending, update state. Scope here: the orchestration (which
rows get picked up, how bars get fetched/filtered/applied, what gets
written back, failure handling) -- the pure classification logic itself is
tests/outcome_resolver_v1.py's job.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    import main
    return main, candidates_router


def _promotion_payload(ticker="AAPL", promoted_at="2026-08-01T14:00:00Z", direction="long",
                        entry_price=100.0, stop=95.0, target=110.0, promotion_kind="enter_now"):
    return {
        "ticker": ticker, "source": "ma_pipeline", "direction": direction, "entry_price": entry_price,
        "stop": stop, "target": target, "risk_reward": 2.0, "rr_warning": False,
        "promotion_kind": promotion_kind,
        "no_valid_target": target is None, "promoted_at": promoted_at, "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": target, "raw_risk_reward": 2.0, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": stop, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": {"body_percentile": 60.0}, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable", "bos_confirmed": False, "bos_details": None,
    }


def _seed(candidates_router, taken=None, **kwargs) -> int:
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    candidates_router._initialize_candidates_schema(conn)
    new_id = candidates_router._store_promotion(conn, _promotion_payload(**kwargs))
    if taken is not None:
        conn.execute("UPDATE candidate_promotions SET taken=? WHERE id=?", (1 if taken else 0, new_id))
    conn.commit()
    conn.close()
    return new_id


def _row(candidates_router, promotion_id):
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM candidate_promotions WHERE id=?", (promotion_id,)).fetchone()
    conn.close()
    return row


def _bars_df(rows: list[dict]) -> pd.DataFrame:
    index = pd.to_datetime([r["time"] for r in rows], utc=True)
    return pd.DataFrame(
        {"Open": [r["low"] for r in rows], "High": [r["high"] for r in rows],
         "Low": [r["low"] for r in rows], "Close": [r["low"] for r in rows],
         "Volume": [1_000_000] * len(rows)},
        index=index,
    )


def test_untaken_and_undecided_promotions_are_never_examined(env, monkeypatch):
    main, router = env
    called = {"count": 0}
    monkeypatch.setattr(main, "_batch_download", lambda *a, **k: (called.__setitem__("count", called["count"] + 1), {})[1])

    _seed(router, taken=None)  # undecided
    _seed(router, taken=False)  # explicitly skipped

    metrics = main._watch_candidate_promotion_outcomes("test")

    assert called["count"] == 0, "no fetch should happen when nothing is taken=1"
    assert metrics["promotions_checked"] == 0


def test_taken_promotion_resolves_hit_target(env, monkeypatch):
    main, router = env
    promo_id = _seed(router, taken=True, direction="long", stop=95.0, target=110.0,
                      promoted_at="2026-08-01T14:00:00Z")

    bars = _bars_df([
        {"time": "2026-08-02T14:00:00Z", "high": 108.0, "low": 99.0},
        {"time": "2026-08-03T14:00:00Z", "high": 112.0, "low": 105.0},  # target=110 hit
    ])
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": bars})

    metrics = main._watch_candidate_promotion_outcomes("test")

    assert metrics["promotions_checked"] == 1
    assert metrics["promotions_resolved"] == 1
    row = _row(router, promo_id)
    assert row["outcome"] == "hit_target"
    assert row["outcome_hit_at"] is not None
    assert row["outcome_bar_source"] == "4h"
    assert row["outcome_resolved_at"] is not None


def test_taken_promotion_stays_still_open_and_gets_rechecked(env, monkeypatch):
    main, router = env
    promo_id = _seed(router, taken=True, direction="long", stop=95.0, target=110.0,
                      promoted_at="2026-08-01T14:00:00Z")

    bars = _bars_df([{"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0}])
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": bars})

    main._watch_candidate_promotion_outcomes("test")
    row = _row(router, promo_id)
    assert row["outcome"] == "still_open"

    # A still_open promotion must be picked up again on the next cycle.
    metrics2 = main._watch_candidate_promotion_outcomes("test")
    assert metrics2["promotions_checked"] == 1


def test_resolved_promotion_is_not_rechecked(env, monkeypatch):
    main, router = env
    _seed(router, taken=True, direction="long", stop=95.0, target=110.0,
          promoted_at="2026-08-01T14:00:00Z")

    bars = _bars_df([{"time": "2026-08-02T14:00:00Z", "high": 112.0, "low": 99.0}])
    fetch_calls = {"count": 0}

    def _fake_download(tickers, period, interval):
        fetch_calls["count"] += 1
        return {"AAPL": bars}

    monkeypatch.setattr(main, "_batch_download", _fake_download)

    main._watch_candidate_promotion_outcomes("test")  # resolves to hit_target
    assert fetch_calls["count"] == 1

    metrics2 = main._watch_candidate_promotion_outcomes("test")
    assert metrics2["promotions_checked"] == 0
    assert fetch_calls["count"] == 1, "a resolved promotion must not trigger another fetch"


def test_fetch_failure_does_not_crash_or_falsely_resolve(env, monkeypatch):
    main, router = env
    promo_id = _seed(router, taken=True, promoted_at="2026-08-01T14:00:00Z")

    def _raise(tickers, period, interval):
        raise RuntimeError("simulated Alpaca outage")

    monkeypatch.setattr(main, "_batch_download", _raise)

    metrics = main._watch_candidate_promotion_outcomes("test")

    assert metrics["fetch_failures"] >= 1
    row = _row(router, promo_id)
    assert row["outcome"] is None, "a fetch failure must never leave a false outcome behind"


def test_empty_or_missing_ticker_data_leaves_outcome_untouched(env, monkeypatch):
    main, router = env
    promo_id = _seed(router, taken=True, promoted_at="2026-08-01T14:00:00Z")
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": pd.DataFrame()})

    metrics = main._watch_candidate_promotion_outcomes("test")

    assert metrics["fetch_failures"] >= 1
    row = _row(router, promo_id)
    assert row["outcome"] is None


def test_bars_before_promoted_at_are_excluded_from_resolution(env, monkeypatch):
    """A bar dated BEFORE promoted_at that would've hit the stop must not
    count -- only price action strictly after the promotion is real
    evidence of what happened to a taken trade."""
    main, router = env
    _seed(router, taken=True, direction="long", stop=95.0, target=110.0,
          promoted_at="2026-08-05T14:00:00Z")

    bars = _bars_df([
        {"time": "2026-08-01T14:00:00Z", "high": 101.0, "low": 90.0},  # before promotion -- stop would trigger, must be ignored
        {"time": "2026-08-06T14:00:00Z", "high": 101.0, "low": 99.0},  # after promotion -- neither hit
    ])
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": bars})

    main._watch_candidate_promotion_outcomes("test")
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM candidate_promotions").fetchone()
    conn.close()
    assert row["outcome"] == "still_open"


def test_two_promotions_same_ticker_resolved_independently(env, monkeypatch):
    """Append-only means the same ticker can have multiple taken promotions
    (different promoted_at, possibly different stop/target) -- each must be
    resolved against its own window of the same downloaded bars."""
    main, router = env
    early_id = _seed(router, taken=True, direction="long", stop=95.0, target=110.0,
                      promoted_at="2026-08-01T14:00:00Z")
    late_id = _seed(router, taken=True, direction="long", stop=95.0, target=110.0,
                     promoted_at="2026-08-04T14:00:00Z")

    bars = _bars_df([
        {"time": "2026-08-02T14:00:00Z", "high": 112.0, "low": 99.0},  # hits target -- only visible to `early`
        {"time": "2026-08-05T14:00:00Z", "high": 101.0, "low": 99.0},  # neither -- visible to both, but early already resolved
    ])
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": bars})

    main._watch_candidate_promotion_outcomes("test")

    early_row = _row(router, early_id)
    late_row = _row(router, late_id)
    assert early_row["outcome"] == "hit_target"
    assert late_row["outcome"] == "still_open"  # its window starts after the target-hit bar


def test_no_valid_target_promotion_only_resolves_to_stop_or_open(env, monkeypatch):
    main, router = env
    promo_id = _seed(router, taken=True, direction="long", stop=95.0, target=None,
                      promoted_at="2026-08-01T14:00:00Z")

    bars = _bars_df([{"time": "2026-08-02T14:00:00Z", "high": 500.0, "low": 99.0}])  # big favorable move, no target
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": bars})

    main._watch_candidate_promotion_outcomes("test")
    row = _row(router, promo_id)
    assert row["outcome"] == "still_open"


def test_state_snapshot_reflects_last_run(env, monkeypatch):
    main, router = env
    _seed(router, taken=True, promoted_at="2026-08-01T14:00:00Z")
    bars = _bars_df([{"time": "2026-08-02T14:00:00Z", "high": 101.0, "low": 99.0}])
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": bars})

    main._watch_candidate_promotion_outcomes("test")
    snapshot = main._promotion_outcome_state_snapshot()

    assert snapshot["running"] is False
    assert snapshot["last_completed_at"] is not None
    assert snapshot["promotions_checked"] == 1
    assert snapshot["last_error"] is None


def test_tracking_only_short_promotion_resolves_exactly_like_a_real_one(env, monkeypatch):
    """The whole point of promotion_kind='tracking_only' (see
    candidates_router.track_candidate_outcome): it must flow through this
    resolver with zero special-casing, identical to a real 'enter_now'
    promotion. This query has never filtered on direction or kind -- taken=1
    is the only thing that matters -- so this is a regression guard on that
    staying true, not a new capability being added to the resolver itself."""
    main, router = env
    promo_id = _seed(router, taken=True, direction="short", stop=105.0, target=90.0,
                      promoted_at="2026-08-01T14:00:00Z", promotion_kind="tracking_only")

    bars = _bars_df([
        {"time": "2026-08-02T14:00:00Z", "high": 102.0, "low": 95.0},
        {"time": "2026-08-03T14:00:00Z", "high": 96.0, "low": 88.0},  # short target=90 hit
    ])
    monkeypatch.setattr(main, "_batch_download", lambda tickers, period, interval: {"AAPL": bars})

    metrics = main._watch_candidate_promotion_outcomes("test")

    assert metrics["promotions_checked"] == 1
    assert metrics["promotions_resolved"] == 1
    row = _row(router, promo_id)
    assert row["outcome"] == "hit_target"
    assert row["promotion_kind"] == "tracking_only"
