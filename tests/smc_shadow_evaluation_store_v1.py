#!/usr/bin/env python3
"""Kairos Sprint 5 -- Historical + Paper Evaluation Harness: persistence
tests (2026-09 session). Real SQLiteSmcShadowEvaluationRepository against
a scratch DB file.

Sprint 5.2: schema/vocabulary renamed to match smc_shadow_evaluation.py's
own Sprint 5.2 rewrite -- "trade" -> "signal" throughout, no outcome/
r_multiple column, trade_performance always the fixed unavailable dict.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_shadow_evaluation_store import SQLiteSmcShadowEvaluationRepository  # noqa: E402
from smc_shadow_evaluation_v1 import _clean_pass_frames, patch_swings_walk_forward  # noqa: E402
from smc_shadow_evaluation import (  # noqa: E402
    run_walk_forward_evaluation, TRADE_PERFORMANCE_UNAVAILABLE, TRADE_PERFORMANCE_UNRESOLVED_STATUS,
)


def repo():
    return SQLiteSmcShadowEvaluationRepository(tempfile.mktemp(suffix=".sqlite3"))


def _real_run(ticker="NVDA", market="stock"):
    df_4h, df_30m, df_5m, frames = _clean_pass_frames()
    with patch_swings_walk_forward(frames):
        return run_walk_forward_evaluation(ticker, "long", df_4h, df_30m, df_5m, market=market)


def test_record_run_persists_run_metadata_and_every_signal():
    run = _real_run()
    r = repo()
    run_id = r.record_run(
        kind="historical", ticker="NVDA", direction="long", market="stock",
        strategy_version=run["strategy_version"], harness_version=run["harness_version"],
        episodes=run["episodes"], signals=run["signals"],
    )
    runs = r.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["kind"] == "historical"
    assert runs[0]["signal_count"] == 1
    assert runs[0]["episode_count"] == 1

    signals = r.list_signals(run_id)
    assert len(signals) == 1
    assert signals[0]["trade_performance_status"] == TRADE_PERFORMANCE_UNRESOLVED_STATUS
    assert signals[0]["location_type"] == "order_block"
    assert signals[0]["stop"] == 93.0
    # No outcome/r_multiple column exists on a signal row at all.
    assert "outcome" not in signals[0]
    assert "r_multiple" not in signals[0]
    # Tri-state booleans round-trip as real Python bool/None, not raw 0/1.
    assert signals[0]["post_entry_ready_invalidation_touched"] is False
    assert signals[0]["post_entry_ready_target_assumption_touched"] is False


def test_get_signal_evidence_round_trips_full_traceability():
    run = _real_run()
    r = repo()
    run_id = r.record_run(
        kind="historical", ticker="NVDA", direction="long", market="stock",
        strategy_version=run["strategy_version"], harness_version=run["harness_version"],
        episodes=run["episodes"], signals=run["signals"],
    )
    signals = r.list_signals(run_id)
    evidence = r.get_signal_evidence(signals[0]["id"])
    assert evidence["evidence"]["state"] == "ENTRY_READY"
    assert evidence["evidence"]["strategy_version"] == run["strategy_version"]
    assert len(evidence["transition_history"]) >= 1
    assert evidence["transition_history"][0]["evidence"]["ticker"] == "NVDA"


def test_get_signal_evidence_returns_none_for_unknown_id():
    r = repo()
    assert r.get_signal_evidence(999999) is None


def test_multiple_runs_and_kinds_are_independently_listable():
    r = repo()
    run1 = _real_run(ticker="NVDA", market="stock")
    run2 = _real_run(ticker="EURUSD", market="forex")
    r.record_run(kind="historical", ticker="NVDA", direction="long", market="stock",
                  strategy_version=run1["strategy_version"], harness_version=run1["harness_version"],
                  episodes=run1["episodes"], signals=run1["signals"])
    r.record_run(kind="paper", ticker="EURUSD", direction="long", market="forex",
                  strategy_version=run2["strategy_version"], harness_version=run2["harness_version"],
                  episodes=run2["episodes"], signals=run2["signals"])

    assert len(r.list_runs()) == 2
    assert len(r.list_runs(kind="historical")) == 1
    assert len(r.list_runs(kind="paper")) == 1
    assert len(r.list_signals()) == 2


def test_report_rebuilds_full_aggregate_from_persisted_data_alone():
    r = repo()
    run1 = _real_run(ticker="NVDA", market="stock")
    run2 = _real_run(ticker="EURUSD", market="forex")
    r.record_run(kind="historical", ticker="NVDA", direction="long", market="stock",
                  strategy_version=run1["strategy_version"], harness_version=run1["harness_version"],
                  episodes=run1["episodes"], signals=run1["signals"])
    r.record_run(kind="historical", ticker="EURUSD", direction="long", market="forex",
                  strategy_version=run2["strategy_version"], harness_version=run2["harness_version"],
                  episodes=run2["episodes"], signals=run2["signals"])

    report = r.report()
    assert report["runs_included"] == 2
    assert report["setup_statistics"]["detected"] == 2
    assert report["setup_statistics"]["entry_ready_signal_count"] == 2
    assert report["setup_statistics"]["terminal_state_reached_at_least"]["LOCATION_REACHED"] == 2
    assert report["setup_statistics"]["terminal_state_reached_at_least"]["ENTRY_READY"] == 2
    assert report["setup_statistics"]["max_progression_reached_at_least"]["LOCATION_REACHED"] == 2
    assert report["setup_statistics"]["max_progression_reached_at_least"]["ENTRY_READY"] == 2
    assert report["post_signal_market_behavior"]["measured_count"] == 2
    assert report["trade_performance"] == TRADE_PERFORMANCE_UNAVAILABLE
    market_segment = report["segments"]["market"]
    assert market_segment["stock"]["post_signal_market_behavior"]["measured_count"] == 1
    assert market_segment["forex"]["post_signal_market_behavior"]["measured_count"] == 1
    assert market_segment["stock"]["trade_performance"] == TRADE_PERFORMANCE_UNAVAILABLE

    # kind filtering
    r.record_run(kind="paper", ticker="AMD", direction="long", market="stock",
                  strategy_version=run1["strategy_version"], harness_version=run1["harness_version"],
                  episodes=[], signals=[])
    assert r.report(kind="historical")["runs_included"] == 2
    assert r.report(kind="paper")["runs_included"] == 1
    assert r.report()["runs_included"] == 3


def test_run_with_zero_signals_still_persists_run_metadata():
    r = repo()
    run_id = r.record_run(
        kind="historical", ticker="ZZZ", direction="long", market="stock",
        strategy_version="smc_shadow_v1", harness_version="smc_shadow_evaluation_harness_v1_2",
        episodes=[], signals=[],
    )
    runs = r.list_runs()
    assert runs[0]["signal_count"] == 0
    assert r.list_signals(run_id) == []


if __name__ == "__main__":
    import inspect
    failures = []
    module = sys.modules[__name__]
    tests = [obj for name, obj in vars(module).items() if name.startswith("test_") and inspect.isfunction(obj)]
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{fn.__name__}: {exc.__class__.__name__}: {exc}")
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print(f"smc_shadow_evaluation_store_v1: {len(tests)}/{len(tests)} passed.")
