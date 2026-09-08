"""Kairos Dashboard Sprint 1 -- Strategy State Read Model (2026-09 session).
Pure unit tests for dashboard_state.py's mapping functions.

Every mapping table entry gets its own assertion (no loops hiding a
silently-skipped case) -- this file is the single place proving the
mapping table quoted in the Sprint 1 report is exactly what the code
does, including the deliberately EXCLUDED and deliberately UNMAPPED
cases (WITHDRAWN/SUPERSEDED/PLAN_REPLACED/SKIP excluded; STALE unmapped).

The ENTER_NOW -> ENTRY_READY case (the task's own explicit requirement)
is proven twice: once as a pure dict-in/dict-out unit test, and once
end-to-end against the REAL, unmodified
scanner.stock_execution_lifecycle_presentation() -- see
test_enter_now_maps_to_entry_ready_end_to_end. That second test is the
one that would actually catch a scanner.py refactor silently changing
what "ENTER_NOW" looks like on the wire.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard_state as ds  # noqa: E402
import scanner  # noqa: E402


# ---------------------------------------------------------------------------
# Mapping 1: approved/watch monitor state
# ---------------------------------------------------------------------------

def test_confirmed_maps_to_confirmed():
    assert ds.map_approved_monitor_state("CONFIRMED", None) == "CONFIRMED"


def test_actionable_maps_to_entry_ready():
    assert ds.map_approved_monitor_state("ACTIONABLE", None) == "ENTRY_READY"


def test_invalidated_maps_to_invalidated():
    assert ds.map_approved_monitor_state("INVALIDATED", None) == "INVALIDATED"


def test_trigger_satisfied_maps_to_execution_ready():
    assert ds.map_approved_monitor_state("TRIGGER_SATISFIED", None) == "EXECUTION_READY"


def test_approved_maps_to_confirmed_judgment_call():
    assert ds.map_approved_monitor_state("APPROVED", None) == "CONFIRMED"


def test_extended_maps_to_waiting_for_pullback_judgment_call():
    assert ds.map_approved_monitor_state("EXTENDED", None) == "WAITING_FOR_PULLBACK"


def test_stale_is_deliberately_unmapped():
    assert ds.map_approved_monitor_state("STALE", None) is None


def test_withdrawn_is_excluded():
    assert ds.map_approved_monitor_state("WITHDRAWN", None) is None


def test_superseded_is_excluded():
    assert ds.map_approved_monitor_state("SUPERSEDED", None) is None


def test_waiting_for_trigger_without_entry_reached_is_watching():
    assert ds.map_approved_monitor_state("WAITING_FOR_TRIGGER", None) == "WATCHING"
    assert ds.map_approved_monitor_state("WAITING_FOR_TRIGGER", "") == "WATCHING"


def test_waiting_for_trigger_with_entry_reached_is_location_reached():
    assert ds.map_approved_monitor_state("WAITING_FOR_TRIGGER", "2026-08-20T14:00:00Z") == "LOCATION_REACHED"


def test_state_matching_is_case_insensitive_and_whitespace_tolerant():
    assert ds.map_approved_monitor_state("  confirmed  ", None) == "CONFIRMED"


def test_unknown_state_is_unmapped_not_guessed():
    assert ds.map_approved_monitor_state("SOME_FUTURE_STATE_NOBODY_WROTE_YET", None) is None


def test_none_state_is_unmapped():
    assert ds.map_approved_monitor_state(None, None) is None


# ---------------------------------------------------------------------------
# Mapping 2: scanner.py ENTER_NOW / early-entry-shadow lifecycle
# ---------------------------------------------------------------------------

def test_enter_now_bucket_maps_to_entry_ready():
    result = ds.map_scanner_lifecycle_result({"ranking_status_bucket": "ENTER_NOW"})
    assert result == {"state": "ENTRY_READY", "excluded": False, "raw_shadow_state": None, "raw_bucket": "ENTER_NOW"}


def test_early_entry_bucket_maps_to_execution_ready():
    assert ds.map_scanner_lifecycle_result({"ranking_status_bucket": "EARLY_ENTRY"})["state"] == "EXECUTION_READY"


def test_almost_ready_bucket_maps_to_execution_ready():
    assert ds.map_scanner_lifecycle_result({"ranking_status_bucket": "ALMOST_READY"})["state"] == "EXECUTION_READY"


def test_waiting_bucket_maps_to_discovered():
    assert ds.map_scanner_lifecycle_result({"ranking_status_bucket": "WAITING"})["state"] == "DISCOVERED"


def test_skip_bucket_is_excluded():
    result = ds.map_scanner_lifecycle_result({"ranking_status_bucket": "SKIP"})
    assert result["state"] is None
    assert result["excluded"] is True


def test_plan_replaced_bucket_is_excluded():
    result = ds.map_scanner_lifecycle_result({"ranking_status_bucket": "PLAN_REPLACED"})
    assert result["state"] is None and result["excluded"] is True


def test_early_entry_building_shadow_maps_to_watching():
    assert ds.map_scanner_lifecycle_result({"state": "EARLY_ENTRY_BUILDING"})["state"] == "WATCHING"


def test_waiting_for_confirmation_shadow_maps_to_watching():
    assert ds.map_scanner_lifecycle_result({"state": "WAITING_FOR_CONFIRMATION"})["state"] == "WATCHING"


def test_early_touch_shadow_maps_to_location_reached():
    assert ds.map_scanner_lifecycle_result({"state": "EARLY_TOUCH"})["state"] == "LOCATION_REACHED"


def test_waiting_for_retest_shadow_maps_to_waiting_for_pullback():
    assert ds.map_scanner_lifecycle_result({"state": "WAITING_FOR_RETEST"})["state"] == "WAITING_FOR_PULLBACK"


def test_missed_entry_shadow_maps_to_invalidated():
    assert ds.map_scanner_lifecycle_result({"state": "MISSED_ENTRY"})["state"] == "INVALIDATED"


def test_tp1_before_confirmation_shadow_maps_to_invalidated():
    assert ds.map_scanner_lifecycle_result({"state": "TP1_BEFORE_CONFIRMATION"})["state"] == "INVALIDATED"


def test_invalidated_shadow_maps_to_invalidated():
    assert ds.map_scanner_lifecycle_result({"state": "INVALIDATED"})["state"] == "INVALIDATED"


def test_expired_shadow_maps_to_invalidated():
    assert ds.map_scanner_lifecycle_result({"state": "EXPIRED"})["state"] == "INVALIDATED"


def test_plan_replaced_shadow_is_excluded():
    result = ds.map_scanner_lifecycle_result({"state": "PLAN_REPLACED"})
    assert result["state"] is None and result["excluded"] is True


def test_entry_triggered_shadow_falls_through_to_bucket():
    # scanner.py's own comment: "current ENTER NOW authority is resolved
    # separately" -- the shadow state alone must NOT decide the dashboard
    # state; the ranking bucket underneath it does.
    result = ds.map_scanner_lifecycle_result({"state": "ENTRY_TRIGGERED", "ranking_status_bucket": "ENTER_NOW"})
    assert result["state"] == "ENTRY_READY"
    assert result["raw_shadow_state"] == "ENTRY_TRIGGERED"


def test_empty_presentation_maps_to_none():
    assert ds.map_scanner_lifecycle_result({}) == {"state": None, "excluded": False, "raw_shadow_state": None, "raw_bucket": None}
    assert ds.map_scanner_lifecycle_result(None)["state"] is None


def test_map_scanner_lifecycle_convenience_wrapper_matches_result():
    assert ds.map_scanner_lifecycle({"ranking_status_bucket": "ENTER_NOW"}) == "ENTRY_READY"


def test_enter_now_maps_to_entry_ready_end_to_end():
    """Proves the task's explicit requirement ("ENTER_NOW must map to
    ENTRY_READY") against the REAL, unmodified
    scanner.stock_execution_lifecycle_presentation -- not a
    reimplementation or a synthetic dict shaped like its output."""
    row = {
        "direction": "LONG",
        "setupGrade": "A",
        "entryStatus": "Tradeable",
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 110.0,
        "current_price": 100.05,
        "current_quote_price": 100.05,
        "trade_eval": {"trigger_confirmed": True},
    }
    presentation = scanner.stock_execution_lifecycle_presentation(row)
    assert presentation["ranking_status_bucket"] == "ENTER_NOW", presentation
    assert ds.map_scanner_lifecycle(presentation) == "ENTRY_READY"


# ---------------------------------------------------------------------------
# Mapping 3: journal position overlay
# ---------------------------------------------------------------------------

def test_open_entry_overlay_wins():
    assert ds.position_overlay_state(True, False) == "POSITION_OPEN"


def test_open_entry_overlay_wins_even_if_also_closed():
    assert ds.position_overlay_state(True, True) == "POSITION_OPEN"


def test_closed_entry_overlay():
    assert ds.position_overlay_state(False, True) == "CLOSED"


def test_no_entry_overlay_is_none():
    assert ds.position_overlay_state(False, False) is None


# ---------------------------------------------------------------------------
# Labels / next-step text / schema completeness
# ---------------------------------------------------------------------------

def test_every_dashboard_state_has_a_label_and_next_step():
    for state in ds.DASHBOARD_STATES:
        assert ds.dashboard_state_label(state), state
        assert ds.dashboard_state_next_step(state), state


def test_label_and_next_step_are_none_for_none_state():
    assert ds.dashboard_state_label(None) is None
    assert ds.dashboard_state_next_step(None) is None


# ---------------------------------------------------------------------------
# Dashboard Sprint 2: market classification
# ---------------------------------------------------------------------------

def test_ma_pipeline_source_is_stock():
    assert ds.market_for_source("ma_pipeline") == "stock"


def test_smc_forex_source_is_forex():
    assert ds.market_for_source("smc_forex") == "forex"


def test_market_classification_is_case_insensitive():
    assert ds.market_for_source("SMC_FOREX") == "forex"


def test_unknown_or_missing_source_defaults_to_stock():
    assert ds.market_for_source(None) == "stock"
    assert ds.market_for_source("") == "stock"
    assert ds.market_for_source("some_future_source") == "stock"


def test_dashboard_states_is_the_exact_eleven_state_schema_requested():
    # Originally the exact 10-state schema Dashboard Sprint 1 requested;
    # extended to 11 the session TARGET_HIT was added to the backend
    # monitor (candidates_router.py's _is_target_hit) -- the mirror-image
    # outcome of INVALIDATED needed a dashboard-side home too, same
    # reasoning as INVALIDATED's own original inclusion here.
    assert ds.DASHBOARD_STATES == (
        "DISCOVERED", "WATCHING", "LOCATION_REACHED", "CONFIRMED",
        "WAITING_FOR_PULLBACK", "EXECUTION_READY", "ENTRY_READY",
        "INVALIDATED", "TARGET_HIT", "POSITION_OPEN", "CLOSED",
    )


def test_target_hit_maps_to_target_hit():
    assert ds.map_approved_monitor_state("TARGET_HIT", None) == "TARGET_HIT"


# ---------------------------------------------------------------------------
# Dashboard Sprint 3: trade-type / option-contract fields
# ---------------------------------------------------------------------------

def test_trade_type_prefers_call_from_a_real_available_contract():
    contract = {"available": True, "type": "CALL", "strike": 105.0, "expiry": "2026-09-19"}
    assert ds.trade_type_for("long", contract) == "CALL"


def test_trade_type_prefers_put_from_a_real_available_contract():
    contract = {"available": True, "type": "PUT", "strike": 95.0, "expiry": "2026-09-19"}
    assert ds.trade_type_for("short", contract) == "PUT"


def test_trade_type_falls_back_to_long_when_contract_unavailable():
    contract = {"available": False, "execution": "No Clean Contract", "reason": "Missing candidate entry price"}
    assert ds.trade_type_for("long", contract) == "LONG"


def test_trade_type_falls_back_to_short_when_contract_unavailable():
    contract = {"available": False, "execution": "No Options Chain"}
    assert ds.trade_type_for("short", contract) == "SHORT"


def test_trade_type_falls_back_to_direction_when_no_contract_at_all():
    assert ds.trade_type_for("long", None) == "LONG"
    assert ds.trade_type_for("short", None) == "SHORT"


def test_trade_type_never_fabricates_a_call_put_from_a_suggested_but_unavailable_contract():
    # A "Suggested" (clean=False) contract is still available=True with a
    # real type/strike/expiry -- scanner.py's own quality tier, not a
    # fabrication. This must still surface as CALL/PUT.
    contract = {"available": True, "clean": False, "type": "CALL", "strike": 100.0, "expiry": "2026-10-16", "execution": "Suggested"}
    assert ds.trade_type_for("long", contract) == "CALL"


def test_option_contract_fields_present_when_available():
    contract = {"available": True, "type": "CALL", "strike": 105.0, "expiry": "2026-09-19"}
    assert ds.option_contract_fields(contract) == {"strike": 105.0, "expiration": "2026-09-19"}


def test_option_contract_fields_absent_when_unavailable():
    assert ds.option_contract_fields({"available": False}) == {"strike": None, "expiration": None}


def test_option_contract_fields_absent_when_no_contract_at_all():
    assert ds.option_contract_fields(None) == {"strike": None, "expiration": None}


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
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print(f"dashboard_state_v1: {len(tests)}/{len(tests)} passed.")
