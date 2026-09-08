#!/usr/bin/env python3
"""Kairos Sprint 4 -- Shadow SMC Strategy/State Engine: persistence tests
(2026-09 session). Real SQLiteSmcShadowRepository against a scratch DB
file -- no mocking of sqlite3 itself.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_shadow_store import SQLiteSmcShadowRepository, TERMINAL_SHADOW_STATES  # noqa: E402
from smc_shadow_engine import STRATEGY_VERSION  # noqa: E402


def repo():
    return SQLiteSmcShadowRepository(tempfile.mktemp(suffix=".sqlite3"))


def fake_eval(state, **overrides):
    base = {
        "state": state,
        "strategy_version": STRATEGY_VERSION,
        "htf_direction": "LONG",
        "htf_direction_aligned": True,
        "location": {"percentile": 20.0, "label": "discount", "alignment": "favorable", "range_high": 110, "range_low": 90},
        "zone": {"high": 100.0, "low": 95.0},
        "confirmation_event": {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None},
        "displacement_evidence": {"confirmation": None, "execution": None},
        "pullback_status": "not_applicable",
        "pullback_detail": None,
        "execution_event": {"detected": False, "level": None, "bar_time": None},
        "invalidation": {"invalidated": False, "level": 95.0, "reason": None},
        "unresolved": [],
    }
    base.update(overrides)
    return base


def test_first_evaluation_creates_a_row_and_a_transition_from_none():
    r = repo()
    row = r.record_evaluation(setup_key="AMD-1", ticker="amd", direction="long", evaluation=fake_eval("WATCHING"), production_state="WATCHING", production_legacy_state="WAITING_FOR_TRIGGER")
    assert row["state"] == "WATCHING"
    assert row["ticker"] == "AMD"  # uppercased
    assert row["strategy_version"] == STRATEGY_VERSION
    assert row["production_state"] == "WATCHING"
    assert row["production_legacy_state"] == "WAITING_FOR_TRIGGER"
    assert row["created_at"] == row["updated_at"] == row["state_since"]

    transitions = r.get_transitions("AMD-1")
    assert len(transitions) == 1
    assert transitions[0]["from_state"] is None
    assert transitions[0]["to_state"] == "WATCHING"


def test_unchanged_state_does_not_log_a_new_transition_or_move_state_since():
    r = repo()
    row1 = r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("WATCHING"))
    row2 = r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("WATCHING"))
    assert row1["state_since"] == row2["state_since"]
    assert row1["updated_at"] <= row2["updated_at"]
    assert len(r.get_transitions("AMD-1")) == 1


def test_state_change_logs_a_new_transition_and_bumps_state_since():
    r = repo()
    r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("WATCHING"))
    row2 = r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("LOCATION_REACHED"))
    assert row2["state"] == "LOCATION_REACHED"
    transitions = r.get_transitions("AMD-1")
    assert [t["to_state"] for t in transitions] == ["WATCHING", "LOCATION_REACHED"]
    assert transitions[1]["from_state"] == "WATCHING"


def test_full_progression_logs_every_transition_in_order():
    r = repo()
    sequence = ["WATCHING", "LOCATION_REACHED", "CONFIRMED", "WAITING_FOR_PULLBACK", "EXECUTION_READY", "ENTRY_READY"]
    for s in sequence:
        r.record_evaluation(setup_key="NVDA-1", ticker="NVDA", direction="long", evaluation=fake_eval(s))
    transitions = r.get_transitions("NVDA-1")
    assert [t["to_state"] for t in transitions] == sequence
    current = r.get_current("NVDA-1")
    assert current["state"] == "ENTRY_READY"


def test_invalidated_is_terminal_and_further_evaluations_are_a_no_op():
    r = repo()
    r.record_evaluation(setup_key="TSLA-1", ticker="TSLA", direction="short", evaluation=fake_eval("CONFIRMED"))
    invalidated_eval = fake_eval("INVALIDATED", invalidation={"invalidated": True, "level": 250.0, "reason": "closed above stop"})
    row = r.record_evaluation(setup_key="TSLA-1", ticker="TSLA", direction="short", evaluation=invalidated_eval)
    assert row["state"] == "INVALIDATED"
    assert row["invalidation_level"] == 250.0

    # A later re-evaluation reading WATCHING must NOT resurrect it.
    row2 = r.record_evaluation(setup_key="TSLA-1", ticker="TSLA", direction="short", evaluation=fake_eval("WATCHING"))
    assert row2["state"] == "INVALIDATED", "INVALIDATED must be terminal -- never silently overwritten"
    transitions = r.get_transitions("TSLA-1")
    assert transitions[-1]["to_state"] == "INVALIDATED"
    assert "INVALIDATED" in TERMINAL_SHADOW_STATES


def test_reset_state_allows_a_fresh_cycle_after_invalidation():
    r = repo()
    r.record_evaluation(setup_key="TSLA-1", ticker="TSLA", direction="short", evaluation=fake_eval("INVALIDATED", invalidation={"invalidated": True, "level": 250.0, "reason": "x"}))
    r.reset_state("TSLA-1")
    assert r.get_current("TSLA-1") is None
    row = r.record_evaluation(setup_key="TSLA-1", ticker="TSLA", direction="short", evaluation=fake_eval("WATCHING"))
    assert row["state"] == "WATCHING"
    # Transition history from BEFORE the reset is preserved for research.
    transitions = r.get_transitions("TSLA-1")
    assert transitions[0]["to_state"] == "INVALIDATED"
    assert transitions[-1]["to_state"] == "WATCHING"


def test_list_current_returns_every_setup_independently():
    r = repo()
    r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("WATCHING"))
    r.record_evaluation(setup_key="NVDA-1", ticker="NVDA", direction="long", evaluation=fake_eval("ENTRY_READY"))
    listed = r.list_current()
    assert {row["setup_key"] for row in listed} == {"AMD-1", "NVDA-1"}


def test_different_strategy_versions_are_fully_independent_rows():
    r = repo()
    r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("WATCHING"), strategy_version="smc_shadow_v1")
    r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("ENTRY_READY"), strategy_version="smc_shadow_v2_experimental")
    assert r.get_current("AMD-1", strategy_version="smc_shadow_v1")["state"] == "WATCHING"
    assert r.get_current("AMD-1", strategy_version="smc_shadow_v2_experimental")["state"] == "ENTRY_READY"


def test_evidence_json_round_trips_the_full_evaluation():
    import json
    r = repo()
    evaluation = fake_eval("CONFIRMED", confirmation_event={"detected": True, "type": "BOS", "level": 101.5, "bar_time": "2026-09-01T14:30:00Z", "reason": "BOS confirmed with STRONG displacement (score 88.0)"})
    row = r.record_evaluation(setup_key="AMD-1", ticker="AMD", direction="long", evaluation=evaluation)
    stored = json.loads(row["evidence_json"])
    assert stored["confirmation_event"]["type"] == "BOS"
    assert stored["confirmation_event"]["reason"].startswith("BOS confirmed")


def test_production_snapshot_is_stored_verbatim_never_recomputed():
    r = repo()
    row = r.record_evaluation(
        setup_key="AMD-1", ticker="AMD", direction="long", evaluation=fake_eval("WATCHING"),
        production_state="ENTRY_READY", production_legacy_state="ACTIONABLE",
    )
    assert row["production_state"] == "ENTRY_READY"
    assert row["production_legacy_state"] == "ACTIONABLE"
    # The shadow state and the production snapshot can legitimately
    # disagree -- that disagreement IS the research signal this sprint
    # exists to surface, not something this module resolves or hides.
    assert row["state"] == "WATCHING"


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
    print(f"smc_shadow_store_v1: {len(tests)}/{len(tests)} passed.")
