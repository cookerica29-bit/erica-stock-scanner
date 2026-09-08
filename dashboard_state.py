"""Kairos Dashboard Sprint 1 -- Strategy State Read Model (2026-09 session).

Additive, read-only normalization layer. This module NEVER computes a
trading decision, NEVER calls a live network/data provider, and NEVER
writes to any table -- it only remaps the OUTPUT of already-existing,
already-decided lifecycle states (candidates_router.py's Approved/Watch
monitor, scanner.py's ENTER_NOW/early-entry-shadow lifecycle, and
journal_store.py's position tracking) into one stable vocabulary a future
spreadsheet dashboard can read without needing to understand three
separate state machines.

Preserved, untouched by this module:
  - scanner.py's frozen strategy functions (see tests/stock_strategy_freeze_v1.py)
  - candidates_router.py's ApprovedSetupMonitorStateName state machine and
    run_approved_setup_monitor_tick
  - journal_store.py's entry_status/JournalRepository
  - every existing API response shape

DASHBOARD_STATES is intentionally a superset of what today's systems can
actually distinguish -- see DASHBOARD_STATE_SUPPORTED_TODAY. A state not
in that set is schema-valid but never emitted by this sprint's mapping
functions; this is deliberate ("map only what is supported today... leave
unsupported states unused/reserved" -- see the task's own instruction),
not an oversight.
"""

from __future__ import annotations

from typing import Any, Optional

DASHBOARD_STATE_MECHANISM_VERSION = "dashboard_state_read_model_v1"

# The full stable schema requested. Order matches the requested lifecycle
# progression (roughly early -> late), not a priority/sort order.
DASHBOARD_STATES = (
    "DISCOVERED",
    "WATCHING",
    "LOCATION_REACHED",
    "CONFIRMED",
    "WAITING_FOR_PULLBACK",
    "EXECUTION_READY",
    "ENTRY_READY",
    "INVALIDATED",
    # TARGET_HIT (2026-09 session): the mirror-image outcome of INVALIDATED
    # at the backend (candidates_router.py's _is_target_hit / TARGET_HIT
    # state, added the same session as the manual-candidate-submission
    # feature this module's own mapping addition below serves). Placed
    # right after INVALIDATED -- the other terminal setup-lifecycle outcome
    # -- not before it, matching this tuple's own "roughly early -> late"
    # ordering comment.
    "TARGET_HIT",
    "POSITION_OPEN",
    "CLOSED",
)

# Every state in DASHBOARD_STATES has a genuine, real data source
# SOMEWHERE in today's systems (see the mapping tables below and the
# accompanying report for exactly which one produces each state) -- none
# of them are purely aspirational placeholders. That said, this sprint's
# ACTUAL endpoint (candidates_router.list_dashboard_state) only reads from
# the Approved/Watch monitor + journal overlay, not scanner.py's
# scan_cached() output (a deliberate, disclosed scope decision -- see the
# report's "architectural risks" section for why). Within that endpoint's
# actual scope, DISCOVERED is schema-valid but never emitted (every row in
# approved_setup_memories already has at least a watch/approve decision
# behind it, by construction) and STALE has no mapping at all (see
# map_approved_monitor_state) -- both correctly fall under "leave
# unsupported states unused/reserved," not silently invented.
DASHBOARD_STATE_SUPPORTED_TODAY = frozenset(DASHBOARD_STATES)

DASHBOARD_STATE_LABELS: dict[str, str] = {
    "DISCOVERED": "Discovered",
    "WATCHING": "Watching",
    "LOCATION_REACHED": "Entry Location Reached",
    "CONFIRMED": "Confirmed",
    "WAITING_FOR_PULLBACK": "Waiting for Pullback",
    "EXECUTION_READY": "Execution Ready",
    "ENTRY_READY": "Entry Ready",
    "INVALIDATED": "Invalidated",
    "TARGET_HIT": "Target Hit",
    "POSITION_OPEN": "Position Open",
    "CLOSED": "Closed",
}

# Purely descriptive of the ALREADY-decided condition behind each state --
# never a new trading instruction. "Kairos is monitoring for the stated
# trigger" restates a fact already true today (the monitor tick already
# does this); it does not tell a human what to do.
DASHBOARD_STATE_NEXT_STEP: dict[str, str] = {
    "DISCOVERED": "Awaiting human chart review.",
    "WATCHING": "Kairos is monitoring for the stated trigger/entry condition.",
    "LOCATION_REACHED": "Price reached the reviewed entry; the stated confirmation is not yet satisfied.",
    "CONFIRMED": "Confirmation was observed; awaiting the execution-window/safety-gate check.",
    "WAITING_FOR_PULLBACK": "The execution window has extended; awaiting a pullback/retest.",
    "EXECUTION_READY": "Near-ready; awaiting the final confirmation/lifecycle check before entry.",
    "ENTRY_READY": "This setup has cleared all current safety gates.",
    "INVALIDATED": "This setup is no longer valid.",
    "TARGET_HIT": "Price reached the approved target.",
    "POSITION_OPEN": "A journal entry shows an open position for this ticker.",
    "CLOSED": "A journal entry shows this position as closed.",
}


# ---------------------------------------------------------------------------
# Mapping 1: candidates_router.py's Approved/Watch monitor
# (approved_setup_monitor_state.state, ApprovedSetupMonitorStateName) --
# see review_queue_evolution_audit.md / watch_lifecycle_v1_audit.md /
# entry_reached_alert_v1_audit.md for how each of these states already
# behaves; this function only RENAMES, it re-derives nothing.
# ---------------------------------------------------------------------------

# States excluded entirely from the dashboard read model (not merely
# "unmapped" -- genuinely not a CURRENT setup generation worth showing):
#   WITHDRAWN  -- the human reversed/rejected this decision.
#   SUPERSEDED -- a newer setup generation replaced this exact row; that
#                 newer row (a different approved_setup_memories id) is
#                 the one to show, not this one.
APPROVED_MONITOR_EXCLUDED_STATES = frozenset({"WITHDRAWN", "SUPERSEDED"})

# Direct/clean mappings, independent of entry_reached_at.
_APPROVED_MONITOR_STATE_MAP: dict[str, str] = {
    "CONFIRMED": "CONFIRMED",
    "ACTIONABLE": "ENTRY_READY",
    "INVALIDATED": "INVALIDATED",
    # TARGET_HIT (2026-09 session): identity mapping, same pattern as
    # INVALIDATED just above -- the backend state name IS the dashboard
    # state name, no renaming needed.
    "TARGET_HIT": "TARGET_HIT",
    # TRIGGER_SATISFIED is transient in practice (the monitor tick
    # immediately re-evaluates it to ACTIONABLE/EXTENDED/STALE within the
    # same tick -- see run_approved_setup_monitor_tick's own docstring;
    # confirmed via tests/watch_lifecycle_v1.py that it is never itself
    # the logged to_state at rest). If ever observed, it means the
    # objective trigger just fired and safety gates are being evaluated
    # right now -- the closest real match is EXECUTION_READY.
    "TRIGGER_SATISFIED": "EXECUTION_READY",
    # APPROVED is the legacy/rare case where a human approved a setup
    # with NEITHER a confirmation_rule NOR a trigger ever recorded (no
    # objective evidence beyond the approval itself). Treated as CONFIRMED
    # -- a human approval IS a form of confirmation, just not the
    # objective-anchor kind Execution Layer V1 later required. Flagged
    # as a judgment call in the report, not asserted as obviously correct.
    "APPROVED": "CONFIRMED",
    # EXTENDED means the execution window degraded (current R:R below
    # threshold) -- not that anything was invalidated. The closest real
    # semantic match in the requested vocabulary is "wait for a better
    # re-entry," i.e. WAITING_FOR_PULLBACK. A genuine judgment call,
    # flagged in the report -- EXTENDED does not natively mean "pullback"
    # today, it means "current R:R math looks bad from here."
    "EXTENDED": "WAITING_FOR_PULLBACK",
}
# STALE has NO clean match in the requested vocabulary -- "evidence
# existed, human never acted for too long" is a fact about human
# responsiveness, not about location/confirmation/execution or a
# position. Deliberately left OUT of _APPROVED_MONITOR_STATE_MAP (falls
# through to None/unmapped below) rather than forced into a wrong bucket.


def map_approved_monitor_state(state: Optional[str], entry_reached_at: Optional[str]) -> Optional[str]:
    """Returns a DASHBOARD_STATES value, or None if this raw state has no
    clean mapping today (the row should still be surfaced with
    legacy_state preserved -- see build_dashboard_row -- just with
    state/state_label left null rather than guessed)."""
    state = str(state or "").strip().upper()
    if state in APPROVED_MONITOR_EXCLUDED_STATES:
        return None
    if state == "WAITING_FOR_TRIGGER":
        # Entry-Reached Alert V1: entry_reached_at is set the instant
        # price reaches the frozen entry, entirely independent of whether
        # the stated confirmation/trigger has fired yet (see
        # entry_reached_alert_v1_audit.md's own product invariant --
        # "the reviewed location is available" is not "confirmed"). This
        # is exactly the LOCATION_REACHED / WATCHING distinction the new
        # schema asks for, and the ONLY existing signal that can make it.
        return "LOCATION_REACHED" if entry_reached_at else "WATCHING"
    return _APPROVED_MONITOR_STATE_MAP.get(state)


# ---------------------------------------------------------------------------
# Mapping 2: scanner.py's ENTER_NOW / early-entry-shadow lifecycle
# (scan_cached/scan_all row output). Reuses scanner.stock_execution_
# lifecycle_presentation(row) and _ranking_status_bucket(row) EXACTLY as
# they exist -- this module calls them, it does not reimplement them, and
# it never mutates the row passed in beyond what those functions already
# do to their own copy.
# ---------------------------------------------------------------------------

# early_entry_shadow.state -> dashboard state, when that shadow state is
# present and reliable (see scanner._stock_authoritative_lifecycle_state
# for how "reliable" -- key match -- is decided; this module trusts
# whatever stock_execution_lifecycle_presentation already resolved).
_SCANNER_SHADOW_STATE_MAP: dict[str, str] = {
    "EARLY_ENTRY_BUILDING": "WATCHING",
    "WAITING_FOR_CONFIRMATION": "WATCHING",
    # "Price reached the entry before confirmation. Wait - this is not
    # Enter Now." -- scanner.py's own presentation text for this state,
    # word-for-word the LOCATION_REACHED concept.
    "EARLY_TOUCH": "LOCATION_REACHED",
    # "Confirmation completed after an early touch. Wait for a fresh
    # retest of the entry area." -- scanner.py's own presentation text,
    # word-for-word the WAITING_FOR_PULLBACK concept.
    "WAITING_FOR_RETEST": "WAITING_FOR_PULLBACK",
    "MISSED_ENTRY": "INVALIDATED",
    "TP1_BEFORE_CONFIRMATION": "INVALIDATED",
    "INVALIDATED": "INVALIDATED",
    "EXPIRED": "INVALIDATED",
    # ENTRY_TRIGGERED and PLAN_REPLACED are deliberately excluded from
    # this map -- see SCANNER_SHADOW_EXCLUDED_STATES below.
}
# ENTRY_TRIGGERED: scanner.py's own presentation says "current ENTER NOW
# authority is resolved separately" -- i.e. this shadow state alone does
# NOT tell you the current state; the ranking bucket is the authority.
# Falls through to the bucket map below rather than being force-mapped.
# PLAN_REPLACED: "the trade plan changed materially; this memory was
# closed" -- a newer plan is the current one, same reasoning as
# SUPERSEDED above -- excluded from the dashboard entirely, not "unmapped".
SCANNER_SHADOW_FALLTHROUGH_STATES = frozenset({"ENTRY_TRIGGERED"})
SCANNER_SHADOW_EXCLUDED_STATES = frozenset({"PLAN_REPLACED"})

# _ranking_status_bucket(row) -> dashboard state, used when no shadow
# state is present (or it fell through) -- see scanner._ranking_status_
# bucket's own docstring/return values; this map is exhaustive over every
# value that function can return.
_SCANNER_RANKING_BUCKET_MAP: dict[str, str] = {
    "ENTER_NOW": "ENTRY_READY",  # the mapping explicitly requested
    "EARLY_ENTRY": "EXECUTION_READY",
    "ALMOST_READY": "EXECUTION_READY",
    "WAITING": "DISCOVERED",
    "WAITING_FOR_RETEST": "WAITING_FOR_PULLBACK",
    "MISSED_ENTRY": "INVALIDATED",
    "TP1_BEFORE_CONFIRMATION": "INVALIDATED",
    "INVALIDATED": "INVALIDATED",
    "EXPIRED": "INVALIDATED",
}
# SKIP and PLAN_REPLACED buckets are excluded entirely -- SKIP means "not
# a real setup" (grade C / RANGE-NO-TRADE / choppy, per
# scanner._ranking_status_bucket's own logic), nothing to show on a
# setup-state dashboard.
SCANNER_BUCKET_EXCLUDED = frozenset({"SKIP", "PLAN_REPLACED"})


def map_scanner_lifecycle(presentation: dict[str, Any]) -> Optional[str]:
    """presentation = scanner.stock_execution_lifecycle_presentation(row)'s
    own return value (already computed, read-only). Returns a
    DASHBOARD_STATES value, "EXCLUDED" is signaled by returning None with
    is_excluded also True in the caller's own check -- see
    map_scanner_lifecycle_result for the richer, disambiguated version
    used by the actual endpoint."""
    result = map_scanner_lifecycle_result(presentation)
    return result["state"]


def map_scanner_lifecycle_result(presentation: dict[str, Any]) -> dict[str, Any]:
    """Richer version distinguishing 'excluded from the dashboard
    entirely' from 'no clean mapping, show unmapped' -- the two
    situations _APPROVED_MONITOR_STATE_MAP's caller also needs to tell
    apart. Returns {"state": str|None, "excluded": bool, "raw_shadow_state":
    str|None, "raw_bucket": str|None}."""
    presentation = presentation or {}
    shadow_state = str(presentation.get("state") or "").strip().upper() or None
    bucket = str(presentation.get("ranking_status_bucket") or "").strip().upper() or None

    if shadow_state and shadow_state in SCANNER_SHADOW_EXCLUDED_STATES:
        return {"state": None, "excluded": True, "raw_shadow_state": shadow_state, "raw_bucket": bucket}
    if shadow_state and shadow_state not in SCANNER_SHADOW_FALLTHROUGH_STATES and shadow_state in _SCANNER_SHADOW_STATE_MAP:
        return {"state": _SCANNER_SHADOW_STATE_MAP[shadow_state], "excluded": False, "raw_shadow_state": shadow_state, "raw_bucket": bucket}

    if bucket and bucket in SCANNER_BUCKET_EXCLUDED:
        return {"state": None, "excluded": True, "raw_shadow_state": shadow_state, "raw_bucket": bucket}
    mapped = _SCANNER_RANKING_BUCKET_MAP.get(bucket) if bucket else None
    return {"state": mapped, "excluded": False, "raw_shadow_state": shadow_state, "raw_bucket": bucket}


# ---------------------------------------------------------------------------
# Mapping 3: journal_store.py position tracking (real, existing --
# JournalRepository.list_entries({"ticker": ..., "status": "open"/"closed"}),
# journal_store.entry_status). Overlays POSITION_OPEN/CLOSED on top of
# whichever setup-lifecycle state mapping 1/2 produced, since an actual
# open/closed position is more senior information than setup-monitoring
# state. Ticker-only correlation -- see the report for the disclosed
# limitation (no native journal-entry <-> setup_key link exists today).
# ---------------------------------------------------------------------------

def position_overlay_state(has_open_entry: bool, has_closed_entry: bool) -> Optional[str]:
    if has_open_entry:
        return "POSITION_OPEN"
    if has_closed_entry:
        return "CLOSED"
    return None


def dashboard_state_label(state: Optional[str]) -> Optional[str]:
    return DASHBOARD_STATE_LABELS.get(state) if state else None


def dashboard_state_next_step(state: Optional[str]) -> Optional[str]:
    return DASHBOARD_STATE_NEXT_STEP.get(state) if state else None


# ---------------------------------------------------------------------------
# Dashboard Sprint 2 -- market classification (2026-09 session).
#
# Sprint 1 hardcoded every row's "market" field to the literal "stock".
# Sprint 2's spec requires the dashboard to support both stocks and forex,
# so this derives the real value from approved_setup_memories.source
# instead -- the same field candidates_router.py's own schema already
# documents as carrying "e.g. 'ma_pipeline', 'smc_forex'" (see
# CandidatePayload.source). This is asset-CLASSIFICATION, not a trading
# decision -- no scanner/strategy behavior is touched, and it lives here
# (the read-model module) rather than in frontend JS, consistent with
# "use the read model as the sole source of truth, do not recompute
# anything in frontend JavaScript."
#
# Judgment call, disclosed: no live forex source has ever been observed
# flowing through approved_setup_memories in this codebase as of this
# session (smc_forex is a documented, forward-looking schema example, not
# yet a populated production source -- confirmed via grep, zero matches
# outside that one docstring). This classifier is therefore honest about
# being a naming-convention heuristic ("forex" appears in the source
# string, case-insensitively) rather than a verified mapping -- it will
# classify smc_forex correctly the moment that source is real, and
# defaults to "stock" for every source in production use today (ma_pipeline
# and its own variants), which is a safe, unchanged default.
def market_for_source(source: Optional[str]) -> str:
    return "forex" if "forex" in str(source or "").lower() else "stock"


# ---------------------------------------------------------------------------
# Dashboard Sprint 3 -- trade-type / option-contract fields for the minimal
# stock signal dashboard (2026-09 session).
#
# "Prefer CALL/PUT when valid option-expression data already exists" --
# already-computed, already-stored data, per this module's own never-fetch-
# live rule. The only such data in this codebase is
# candidate_plan_previews.option_contract_json: a snapshot computed by the
# EXISTING review-queue preview flow (_safe_option_contract_for_candidate /
# scanner._best_contract), keyed by (ticker, source), read here exactly as
# stored -- no live options-chain fetch, no re-scoring, no new selection
# logic. See _dashboard_option_contract in candidates_router.py for the
# read-only lookup this function's input comes from.
#
# Disclosed limitations (same spirit as the Sprint 1 journal overlay's own
# disclosed ticker-only limitation):
#   - candidate_plan_previews is keyed by (ticker, source), not by
#     setup_key -- a preview computed for an earlier/different setup
#     generation on the same ticker could be stale relative to the
#     CURRENT approved setup by the time it reaches ENTRY_READY. This
#     module does not attempt to freshness-check it; it is presented as
#     "the last computed suggestion available", not a guarantee.
#   - Only a contract with available=True is treated as real (this
#     includes scanner.py's "Suggested" quality tier -- clean=False, but a
#     genuine chain-derived strike/expiry, not a fabricated one) --
#     available=False (no chain, no matching contracts near entry, a
#     stored transient/loading placeholder) always falls back to the
#     plain LONG/SHORT direction, never a guessed strike/expiration.
def trade_type_for(direction: Optional[str], option_contract: Optional[dict]) -> str:
    """Returns 'CALL'/'PUT' when a real, already-computed, available
    option contract exists; otherwise the plain underlying direction
    ('LONG'/'SHORT'). Never fabricates option data that doesn't exist."""
    if isinstance(option_contract, dict) and option_contract.get("available") and option_contract.get("type") in ("CALL", "PUT"):
        return option_contract["type"]
    return "SHORT" if str(direction or "").strip().lower() == "short" else "LONG"


def option_contract_fields(option_contract: Optional[dict]) -> dict:
    """Returns {'strike': float|None, 'expiration': str|None} -- both None
    whenever there is no real, available contract (see trade_type_for's
    own docstring for what counts as "real" here). Never invents a value
    for either field independently of the other."""
    if not isinstance(option_contract, dict) or not option_contract.get("available"):
        return {"strike": None, "expiration": None}
    return {"strike": option_contract.get("strike"), "expiration": option_contract.get("expiry")}
