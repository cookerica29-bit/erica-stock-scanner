"""Kairos Sprint 5 -- Historical + Paper Evaluation Harness (2026-09
session). Recalibrated in Sprint 5.1 -- see that section of this
docstring below, and dashboard_sprint5_1_report.md for the full
old-vs-new comparison, provenance audit, and regression results.

Measures strategy_version="smc_shadow_v1" -- imported from
smc_shadow_engine.py and called EXACTLY as written, with its default
config -- exactly as it already exists. This module never edits, tunes,
or overrides any SmcShadowConfig threshold. If a rule looks weak once
real data runs through it, that observation belongs in this module's own
report output as a candidate for a FUTURE strategy_version (e.g.
"smc_shadow_v1_1"), never as a silent in-place edit to v1's own values --
see smc_shadow_engine.STRATEGY_CONFIGS's own docstring for why that
registry exists.

Two genuinely different kinds of "lookahead" appear in this file, and
they are NOT the same thing:
  1. The WALK-FORWARD STATE EVALUATION (run_walk_forward_evaluation) calls
     evaluate_smc_shadow() at each historical point using ONLY bars up to
     and including that point -- exactly what a live tick would have seen
     "as of" that moment, no lookahead into the future at all. This is
     what makes the resulting state-transition history a genuine
     backtest of the state machine's own behavior, not a hindsight
     read.
  2. The POST-SIGNAL DIAGNOSTICS measurement (_measure_post_signal_
     diagnostics, renamed from _measure_trade_outcome in Sprint 5.2)
     deliberately DOES look forward from the entry-ready signal bar --
     that is the entire point of asking "what did price do after this
     signal". This is a separate, later step from (1), never fed back
     into the state evaluation itself.

HARNESS_VERSION identifies THIS module's own measurement conventions --
separate from STRATEGY_VERSION, because v1's own strategy logic never
defines a take-profit target at all (it only gates entry conditions and
an invalidation level).

---------------------------------------------------------------------------
Sprint 5.1 -- evaluation fidelity calibration (what changed and why)
---------------------------------------------------------------------------
Sprint 5 got the no-lookahead property right but was imprecise about two
separate things: what counts as "a new trade" within a still-open thesis,
and what the harness is honestly entitled to claim about outcomes it
partly invented. Both are calibrated here, in the HARNESS only --
smc_shadow_engine.py is untouched by this sprint.

1. EPISODE BOUNDARIES. Sprint 5 required a full return to WATCHING before
   any second trade could be recorded -- correct for suppressing
   duplicate trades out of one persistent ENTRY_READY read, but too
   restrictive for a genuinely new execution opportunity arising under
   the SAME still-valid HTF/30M thesis (price pulls back, executes,
   continues, pulls back AGAIN, executes again -- without the location or
   30M confirmation ever being lost). A second trade within the SAME open
   episode is now permitted, but only when ALL of these are objectively
   true from fields evaluate_smc_shadow() already produces -- no
   arbitrary cooldown bar count anywhere:
     - state reads ENTRY_READY again, AND
     - the current execution_event's (type, level, bar_time) identity is
       DIFFERENT from the execution_event behind the prior trade (the
       prior event has genuinely been "consumed", not just re-read), AND
     - price has closed outside the prior trade's own execution_area
       bounds at some point since that trade (it genuinely left the
       zone), AND
     - pullback_status regressed away from "observed" and then returned
       to "observed" again since the prior trade (a genuine new pullback
       cycle, not the same one still in progress).
   See _new_episode/_execution_event_identity/the per-bar tracking in
   run_walk_forward_evaluation. Disclosed limitation: this is still a
   HARNESS interpretation of "new opportunity" -- evaluate_smc_shadow()
   itself has no concept of "episode" or "trade count" at all; requiring
   all four signals together (rather than any one alone) is itself a
   judgment call about how much agreement should count as "new", not
   something smc_shadow_v1 defines. See test_new_execution_opportunity_*
   and test_persistent_entry_ready_still_produces_only_one_trade.

2. ENTRY/STOP/TARGET/DIRECTION PROVENANCE. Every hypothetical trade now
   carries an explicit provenance tag for each field:
     - direction:        directly_produced_by_smc_shadow_v1 (echoed from
                          evaluate_smc_shadow()'s own result)
     - stop:              directly_produced_by_smc_shadow_v1 (== result
                          ["invalidation"]["level"], a real value the
                          engine itself computes)
     - entry / entry_bar_time: evaluator_specific_assumption (the ENGINE
                          never outputs an "entry price" -- the harness
                          chooses "the 30M close at the bar ENTRY_READY
                          was first read")
     - target:            evaluator_specific_assumption (v1 defines NO
                          target at all; target_r_multiple is a harness
                          convention, see HarnessConfig)
   Because the primary win/loss determination requires BOTH a stop and a
   target, and only the stop is real, the PRIMARY `outcome` field can
   only ever resolve to "loss" (the real stop was hit -- a legitimate,
   strategy-grounded event, r_multiple always exactly computable) or stay
   unresolved. It is NEVER computed as a "win" from the invented target --
   see _measure_trade_outcome. The assumed-target read is still measured
   and kept, but as clearly separate, clearly labeled research_* fields
   (research_target_reached, research_r_multiple) and a separate
   research_summary block in aggregation -- never blended into the
   primary outcome/r_multiple/win-rate.

3. SAME-BAR AMBIGUITY. When a single bar's range covers both the real
   stop AND the (research-only) assumed target, OHLC data cannot reveal
   which was touched first. This is now its own explicit outcome,
   "ambiguous_intrabar" -- excluded entirely from every win-rate/
   expectancy computation, never silently resolved either way. See
   _measure_trade_outcome.

4. GAP / FILL CONVENTION, explicitly disclosed: if a bar's OPEN already
   satisfies a stop/target condition (the market gapped through it), the
   simulated fill is at that OPEN price (a worse fill for a stop, a
   better one for a reached target) rather than the nominal stop/target
   level -- tagged exit_fill_type="gap" vs "intrabar". This is standard,
   disclosed backtesting practice, not a guess: the Open price is a real,
   known value, unlike the genuinely unknowable intrabar sequencing in
   (3) above.

5. NO-LOOKAHEAD is unchanged and re-verified this sprint (same test,
   still passing) -- state evaluation at bar t still only ever receives
   data through bar t; only the SEPARATE, later outcome-measurement walk
   looks forward, exactly as before.

6. SYNTHETIC DATA remains validation-only. Every synthetic scenario run
   in this codebase's test suite or report is labeled exactly that --
   proof the harness computes correctly, never folded into any claim
   about smc_shadow_v1's real-world performance. See
   dashboard_sprint5_1_report.md's own explicit statement on this.

---------------------------------------------------------------------------
Sprint 5.2 -- outcome semantics hardening (what changed and why)
---------------------------------------------------------------------------
Sprint 5.1 got the STOP right (a real, strategy-produced value; a stop
touch is a legitimate primary "loss") but didn't go far enough: a
strategy-defined stop being touched relative to an EVALUATOR-ASSUMED
entry price is still not a fully strategy-defined trade. Per the Sprint
5.1 provenance audit itself: direction and stop are
directly_produced_by_smc_shadow_v1, but entry price, entry timestamp,
and target are ALL evaluator_specific_assumption. A "trade" requires
entry too -- so smc_shadow_v1, as currently specified, cannot yet produce
a fully-owned trade plan at all, and NEITHER a stop touch NOR a target
touch may be classified as a primary win/loss/R.

What changed:
  - The former "hypothetical trade" concept is renamed, throughout this
    module, the store, the router, and every test, to an "entry-ready
    SIGNAL" -- episode["trades"]/run["trades"] are now episode["signals"]/
    run["signals"]. This is not cosmetic: "trade" implies a P&L-bearing
    position; "signal" does not, and the rename makes the field names
    themselves (assumed_entry_price, post_entry_ready_invalidation_touched,
    research_mfe, ...) impossible to mistake for realized performance
    even without reading the provenance tags.
  - There is NO primary outcome/r_multiple/win/loss field anywhere on a
    signal record anymore. `trade_performance_status` is a fixed,
    hardcoded constant ("unresolved_incomplete_strategy_provenance") on
    every single signal -- never computed from data, never anything else,
    by construction (see _build_entry_ready_signal).
  - R-multiple (primary OR research) no longer exists at all -- an R
    figure is itself an expectancy-shaped number that needs both a real
    entry and a real target to mean anything; since neither is fully
    strategy-owned, this sprint removes R-multiple from the vocabulary
    entirely rather than keep a "research_r_multiple" that could still be
    misread as edge.
  - What DOES still get measured and kept -- clearly separate research
    diagnostics, independent of each other, never combined into a single
    "outcome": post_entry_ready_invalidation_touched (+ its bar_time,
    fill_type, and bars-since-signal) and post_entry_ready_target_
    assumption_touched (+ the same three), plus research_mfe/research_mae
    (pure price-excursion measurements, no target/stop needed to compute).
    Both touch flags are tracked independently and DO NOT stop the
    measurement walk early -- there is no "trade is over" concept to stop
    for anymore, so the harness now walks the full window and reports
    both facts regardless of which (if either) happened "first" in
    spirit. See _measure_post_signal_diagnostics.
  - Same-bar ambiguity is preserved as a real, distinct concept -- if
    invalidation and the assumed target are BOTH touched for the first
    time on the identical bar, ambiguous_intrabar_between_invalidation_
    and_target_assumption is set True for that bar. This does not block
    either touch flag from being recorded (both genuinely did happen by
    that bar) -- it only means their RELATIVE ORDER within that one bar
    is unknowable from OHLC data, exactly the same honesty principle as
    Sprint 5.1's original ambiguity handling, now scoped correctly to
    "we don't know the order" rather than "we don't know the outcome"
    (there is no single outcome to not-know anymore).
  - The gap/fill convention from Sprint 5.1 is UNCHANGED in mechanism
    (Open-price fill when a level is gapped through, vs the nominal level
    when merely touched intrabar) -- only the field names carry it now
    (post_entry_ready_invalidation_fill_type / post_entry_ready_target_
    assumption_fill_type), and it is explicitly documented as a RESEARCH
    fill convention, not a live-execution guarantee -- see requirement 4
    of the Sprint 5.2 task and dashboard_sprint5_2_report.md.
  - The performance report (aggregate_evaluation_report /
    SQLiteSmcShadowEvaluationRepository.report) now returns three
    explicitly separate top-level blocks: setup_statistics (the state
    funnel), post_signal_market_behavior (the touch/MFE/MAE/ambiguity/gap
    diagnostics, segmented), and trade_performance -- which is ALWAYS the
    fixed, hardcoded
    {"status": "unavailable_unresolved_for_smc_shadow_v1", "reason": ...}
    dict, never computed from the signals at all. There is no code path
    in this module that can produce a win_rate or expectancy number for
    smc_shadow_v1 -- the concept was removed, not merely hidden.
  - RE-ENTRY LIMITATION, frozen and documented rather than patched:
    Sprint 5.1 discovered that smc_shadow_v1's own pullback_status
    computation (_returned_to_area_after_clearing, scanning the entire
    post-confirmation history every call) makes its "pullback genuinely
    regressed" signal effectively permanent once first satisfied --
    meaning smc_shadow_v1, as currently specified, behaves as ONE
    execution cycle per continuing setup in practice, even though the
    harness's own episode-boundary bookkeeping (Sprint 5.1) is capable of
    recording more than one signal per episode when the underlying
    evidence allows it. This sprint does NOT patch, work around, or paper
    over that engine behavior in the evaluator -- it is documented here,
    in the Sprint 5.2 report, and flagged explicitly as a candidate for a
    FUTURE, EXPLICIT strategy version (e.g. one with a windowed rather
    than whole-history pullback check) to address, never a silent v1 edit.

---------------------------------------------------------------------------
Sprint 6.1 -- baseline integrity fix + terminal_state/max_progression
---------------------------------------------------------------------------
Sprint 6's real-data baseline found a genuine HARNESS bookkeeping defect
(not a smc_shadow_v1 issue) in run_walk_forward_evaluation's main loop:
on the bar an episode is CREATED, the loop did
    current_episode = _new_episode(i, result)
    _update_episode_evidence_fields(current_episode, result)
    continue
-- that `continue` skipped the invalidation-check block AND the
ENTRY_READY signal-recording block for that SAME bar (both live later in
the loop body, unreached via the early `continue`). Usually this
"self-healed" (a later bar re-running the normal path caught up), but
when it didn't -- e.g. an episode born already INVALIDATED whose very
next bar reads WATCHING directly, or an episode born already ENTRY_READY
with no later bar re-confirming it -- the episode's own `invalidated`/
`signals` fields ended up wrong, undiscoverable from synthetic fixtures
(all of which start clean from WATCHING) and only surfaced by Sprint 6's
real cohort.

FIXED HERE, in the harness itself (not a reporting-script workaround):
the loop no longer `continue`s past a creation bar. A `just_created` flag
suppresses ONLY the states_visited/transitions bookkeeping _new_episode()
already seeded for that one bar (preventing a duplicate append) -- every
other per-bar check (pullback-opportunity tracking, ENTRY_READY signal
recording, the INVALIDATED check, and the episode-close check) now runs
UNCONDITIONALLY on every eligible bar, including the bar an episode is
born on. A consequence, correctly: an episode CAN now close on its own
creation bar (e.g. born already INVALIDATED) -- start_index == end_index
is now a valid, correctly-recorded episode shape, not skipped.
HARNESS_VERSION bumped to smc_shadow_evaluation_harness_v1_3 accordingly.
See dashboard_sprint6_1_report.md for the full before/after delta against
the exact Sprint 6 cohort/windows, and the 5 regression tests
(test_creation_bar_invalidation_is_recorded_correctly,
test_creation_bar_entry_ready_signal_is_recorded_correctly,
test_subsequent_bars_are_not_double_counted_after_creation_bar_fix,
test_episode_boundaries_remain_correct_after_creation_bar_fix,
test_no_lookahead_the_engine_only_ever_sees_data_up_to_the_current_bar --
re-run unmodified) proving it.

TERMINAL_STATE vs MAX_PROGRESSION -- formally distinguished, precisely
named, in this module's own report output (not left to each report
script's own ad hoc choice):
  - terminal_state_reached_at_least (in _setup_statistics, renamed from
    the Sprint 5.2 "states_reached" key): "where was the setup when its
    episode ended" -- built from episode["max_state_reached"] /
    episode["invalidated"], i.e. the STATE MACHINE'S OWN returned `state`
    label, which respects smc_shadow_v1's own "invalidation is checked
    LAST and overrides everything above it" design (smc_shadow_engine.py
    step 6). If price already violates the invalidation level on the
    same bar a location/confirmation was found, this view reports
    INVALIDATED, not the intermediate stage -- exactly matching what the
    strategy itself returned, nothing invented.
  - max_progression_reached_at_least (new): "how far through the
    strategy funnel did the setup EVER progress" -- built from
    episode["location_type"] / episode["confirmation_event_type"] /
    episode["execution_event_type"] / (episode["entry_ready_bar_time"]
    or episode["signals"]), all of which _update_episode_evidence_fields
    populates independently of whatever the `state` field's invalidation
    override later does to that SAME bar. This recovers "how far did the
    underlying evidence get" even on a bar where invalidation masked the
    returned state -- a property of smc_shadow_v1's own evidence-dict
    design (each stage's evidence lives in its own dict slot, only
    `state` itself gets overridden), not a harness limitation, and NOT
    eliminated by the Sprint 6.1 bookkeeping fix above (the two views
    will keep differing after the fix, for this legitimate reason, not a
    bug -- see the Sprint 6.1 report's own explanation of the remaining
    gap between them).
  Both are now computed unconditionally, every report, every segment --
  never an implicit choice a future report script makes on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

import scanner
from smc_shadow_engine import (
    STRATEGY_VERSION,
    SHADOW_STATES,
    SmcShadowConfig,
    STRATEGY_CONFIGS,
    evaluate_smc_shadow,
)

HARNESS_VERSION = "smc_shadow_evaluation_harness_v1_3"

# Non-terminal progression order, used only to track "the most advanced
# state this episode ever reached" -- INVALIDATED is tracked separately
# (as episode["invalidated"]), not as a position in this ordering.
_PROGRESSION_ORDER = {
    "WATCHING": 0, "LOCATION_REACHED": 1, "CONFIRMED": 2,
    "WAITING_FOR_PULLBACK": 3, "EXECUTION_READY": 4, "ENTRY_READY": 5,
}


@dataclass
class HarnessConfig:
    """This module's OWN measurement conventions -- distinct from
    SmcShadowConfig, which belongs entirely to the strategy under test.
    Every value here is a harness-methodology choice, disclosed as such,
    never a strategy threshold.
    """
    version: str = HARNESS_VERSION
    # smc_shadow_v1 gates entry conditions and defines an invalidation
    # level, but never defines an entry price or a take-profit target --
    # neither is a deterministic concept anywhere in the strategy under
    # test. assumed_target_r_multiple is still used to compute a
    # RESEARCH-only reference level (a fixed multiple off the assumed
    # entry/real-stop distance) purely so post_entry_ready_target_
    # assumption_touched has something concrete to check against -- it is
    # NOT used to compute any R-multiple, win rate, or expectancy (Sprint
    # 5.2 removed R-multiple from this module's vocabulary entirely). This
    # is NOT part of smc_shadow_v1 and must never be described as though
    # it were.
    assumed_target_r_multiple: float = 2.0
    # How many bars forward (on the SAME timeframe the signal was
    # evaluated on -- 30M) the post-signal diagnostic walk will look
    # before giving up.
    max_bars_to_diagnostics: int = 200
    # Minimum bars of 4H/30M history required before the walk-forward
    # loop starts evaluating at all (avoids evaluating against
    # near-empty slices at the very start of a historical window).
    min_htf_bars: int = 20
    min_confirmation_bars: int = 20


def _direction_upper(direction: str) -> str:
    d = str(direction or "").strip().upper()
    return d if d in ("LONG", "SHORT") else ""


def _slice_up_to(df: Optional[pd.DataFrame], as_of) -> Optional[pd.DataFrame]:
    """All rows of `df` whose index is <= `as_of` -- the no-lookahead
    truncation every walk-forward step applies to every timeframe."""
    if df is None or len(df) == 0:
        return df
    return df.loc[:as_of]


def _new_episode(index: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_index": index,
        "end_index": None,
        "states_visited": [result["state"]],
        "transitions": [{"from_state": None, "to_state": result["state"], "evidence": result}],
        "max_state_reached": result["state"],
        "signals": [],  # Sprint 5.2: renamed from "trades" -- an open
                         # episode can produce more than one entry-ready
                         # SIGNAL (Sprint 5.1's episode-boundary rule,
                         # unchanged this sprint); none of them are a
                         # "trade" -- see module docstring.
        "invalidated": False,
        "invalidated_before_entry_ready": False,
        "incomplete": False,
        "detected_bar_time": result.get("_bar_time"),
        "location_type": result["location"]["location_type"],
        "confirmation_event_type": None,
        "confirmation_displacement_label": None,
        "execution_event_type": None,
        "execution_area_source": None,
        "confirmed_bar_time": None,
        "entry_ready_bar_time": None,
        # Sprint 5.1 new-execution-opportunity tracking -- see module
        # docstring. Internal bookkeeping, not part of the public episode
        # shape callers should rely on beyond `signals`.
        "_last_execution_identity": None,
        "_last_signal_execution_area": None,
        "_regressed_since_last_signal": False,
        "_left_prior_area_since_last_signal": False,
    }


def _execution_event_identity(result: dict[str, Any]) -> Optional[tuple]:
    """The objective identity of the CURRENT execution event, if any --
    (type, level, bar_time). Two evaluations describing genuinely the
    same 5M event will always produce the same identity; a genuinely NEW
    event changes at least the bar_time (and usually the level)."""
    ev = result.get("execution_event") or {}
    if not ev.get("detected"):
        return None
    return (ev.get("type"), ev.get("level"), ev.get("bar_time"))


def _more_advanced(a: str, b: str) -> str:
    return b if _PROGRESSION_ORDER.get(b, -1) > _PROGRESSION_ORDER.get(a, -1) else a


def _update_episode_evidence_fields(episode: dict[str, Any], result: dict[str, Any]) -> None:
    """Records the FIRST time each key piece of evidence appears -- this
    is what "full traceability from every hypothetical trade back to its
    exact state evidence" means in practice: not just the final
    snapshot, but which event types actually produced the progression."""
    if result["confirmation_event"]["detected"] and episode["confirmation_event_type"] is None:
        episode["confirmation_event_type"] = result["confirmation_event"]["type"]
        disp = result["displacement_evidence"]["confirmation"]
        episode["confirmation_displacement_label"] = disp.get("label") if disp else None
        episode["confirmed_bar_time"] = result.get("_bar_time")
    if result["execution_event"]["detected"] and episode["execution_event_type"] is None:
        episode["execution_event_type"] = result["execution_event"]["type"]
    if result.get("execution_area") is not None and episode["execution_area_source"] is None:
        episode["execution_area_source"] = result["execution_area"]["source"]
    if result["state"] == "ENTRY_READY" and episode["entry_ready_bar_time"] is None:
        episode["entry_ready_bar_time"] = result.get("_bar_time")


# Sprint 5.2: the ONLY value trade_performance_status is ever set to,
# anywhere in this module -- a hardcoded constant, never computed from
# data. See module docstring's Sprint 5.2 section.
TRADE_PERFORMANCE_UNRESOLVED_STATUS = "unresolved_incomplete_strategy_provenance"


def _build_entry_ready_signal(
    ticker: str, direction_uc: str, result: dict[str, Any], assumed_entry_price: float,
    entry_ready_bar_time: Optional[str], harness_config: HarnessConfig,
) -> Optional[dict[str, Any]]:
    """Renamed from _build_hypothetical_trade (Sprint 5.2) -- this builds
    an ENTRY-READY SIGNAL record, not a trade. See module docstring."""
    stop = result["invalidation"]["level"]
    if stop is None or assumed_entry_price is None:
        return None  # no deterministic stop -- never fabricate a signal without one
    assumed_risk = abs(assumed_entry_price - stop)
    if assumed_risk <= 0:
        return None
    assumed_target = (
        assumed_entry_price + assumed_risk * harness_config.assumed_target_r_multiple if direction_uc == "LONG"
        else assumed_entry_price - assumed_risk * harness_config.assumed_target_r_multiple
    )
    return {
        "ticker": ticker,
        "direction": "long" if direction_uc == "LONG" else "short",
        "entry_ready_bar_time": entry_ready_bar_time,
        "assumed_entry_price": assumed_entry_price,
        "stop": stop,
        "assumed_target": assumed_target,
        "assumed_target_r_multiple": harness_config.assumed_target_r_multiple,
        "assumed_risk": round(assumed_risk, 4),
        # Sprint 5.1/5.2 provenance audit -- see module docstring and
        # dashboard_sprint5_2_report.md's own provenance table.
        "direction_provenance": "directly_produced_by_smc_shadow_v1",
        "entry_ready_bar_time_provenance": "directly_produced_by_smc_shadow_v1",
        "stop_provenance": "directly_produced_by_smc_shadow_v1",
        "assumed_entry_price_provenance": "evaluator_specific_assumption",
        "assumed_target_provenance": "evaluator_specific_assumption",
        # Sprint 5.2: fixed, hardcoded, never computed -- see
        # TRADE_PERFORMANCE_UNRESOLVED_STATUS's own comment.
        "trade_performance_status": TRADE_PERFORMANCE_UNRESOLVED_STATUS,
        # Research diagnostics -- filled in by
        # _measure_post_signal_diagnostics. Explicit, unambiguous names --
        # never a "win"/"loss"/"outcome"/"r_multiple" field anywhere here.
        "post_entry_ready_invalidation_touched": None,
        "post_entry_ready_invalidation_touched_bar_time": None,
        "post_entry_ready_invalidation_fill_type": None,
        "bars_from_entry_ready_to_invalidation_touch": None,
        "post_entry_ready_target_assumption_touched": None,
        "post_entry_ready_target_assumption_touched_bar_time": None,
        "post_entry_ready_target_assumption_fill_type": None,
        "bars_from_entry_ready_to_target_assumption_touch": None,
        "ambiguous_intrabar_between_invalidation_and_target_assumption": False,
        "ambiguous_intrabar_bar_time": None,
        "research_mfe": None,
        "research_mae": None,
    }


def _measure_post_signal_diagnostics(
    df_30m_full: pd.DataFrame, signal_index: int, signal: dict[str, Any], direction_uc: str,
    harness_config: HarnessConfig,
) -> dict[str, Any]:
    """Renamed from _measure_trade_outcome (Sprint 5.2). Walks FORWARD
    from signal_index+1 in the FULL 30M series (including bars the
    no-lookahead state walk had not yet seen at signal time -- deliberate
    here, see module docstring's point 1 on the two different kinds of
    "lookahead" in this file).

    There is no primary outcome to resolve anymore -- this walks the
    FULL measurement window (never stops early) and records, entirely
    INDEPENDENTLY of each other:
      - post_entry_ready_invalidation_touched: did price ever touch the
        real, strategy-produced stop level after the signal?
      - post_entry_ready_target_assumption_touched: did price ever touch
        the harness's own assumed research target?
      - research_mfe / research_mae: pure price-excursion measurements
        relative to the assumed entry, needing no target or stop at all.

    Same-bar ambiguity: if BOTH touch flags become true for the FIRST
    time on the identical bar, their relative order within that bar is
    unknowable from OHLC data -- ambiguous_intrabar_between_invalidation_
    and_target_assumption is set True for that bar. This does not
    suppress either touch flag (both genuinely happened by that bar); it
    only flags that this one bar's internal ordering is unresolved.

    Gap/fill convention (research diagnostics only -- see requirement 4):
    if a bar's OPEN already satisfies a touch condition (a gap through),
    the recorded fill_type is "gap" (the real Open price is the more
    honest reference point) vs "intrabar" (the nominal level was merely
    touched, not gapped through).
    """
    entry = signal["assumed_entry_price"]
    stop = signal["stop"]
    target = signal["assumed_target"]
    mfe = 0.0
    mae = 0.0
    invalidation_touched = False
    invalidation_touched_index = None
    invalidation_fill_type = None
    target_touched = False
    target_touched_index = None
    target_fill_type = None
    ambiguous = False
    ambiguous_index = None

    end = min(signal_index + 1 + harness_config.max_bars_to_diagnostics, len(df_30m_full))
    for j in range(signal_index + 1, end):
        bar = df_30m_full.iloc[j]
        o, h, l = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
        if direction_uc == "LONG":
            mfe = max(mfe, h - entry)
            mae = min(mae, l - entry)
            stop_hit = l <= stop
            target_hit = (h >= target) if target is not None else False
            stop_gapped = o <= stop
            target_gapped = o >= target if target is not None else False
        else:
            mfe = max(mfe, entry - l)
            mae = min(mae, entry - h)
            stop_hit = h >= stop
            target_hit = (l <= target) if target is not None else False
            stop_gapped = o >= stop
            target_gapped = o <= target if target is not None else False

        newly_invalidation = stop_hit and not invalidation_touched
        newly_target = target_hit and not target_touched

        if newly_invalidation and newly_target:
            ambiguous, ambiguous_index = True, j
        if newly_invalidation:
            invalidation_touched, invalidation_touched_index = True, j
            invalidation_fill_type = "gap" if stop_gapped else "intrabar"
        if newly_target:
            target_touched, target_touched_index = True, j
            target_fill_type = "gap" if target_gapped else "intrabar"
        # Deliberately no `break` -- there is no "position closed" concept
        # to stop for; the full window is walked so both diagnostics are
        # as complete as the available data allows.

    signal.update({
        "post_entry_ready_invalidation_touched": invalidation_touched,
        "post_entry_ready_invalidation_touched_bar_time": (
            scanner._timestamp_at(df_30m_full, invalidation_touched_index)
            if invalidation_touched_index is not None else None
        ),
        "post_entry_ready_invalidation_fill_type": invalidation_fill_type,
        "bars_from_entry_ready_to_invalidation_touch": (
            (invalidation_touched_index - signal_index) if invalidation_touched_index is not None else None
        ),
        "post_entry_ready_target_assumption_touched": target_touched,
        "post_entry_ready_target_assumption_touched_bar_time": (
            scanner._timestamp_at(df_30m_full, target_touched_index)
            if target_touched_index is not None else None
        ),
        "post_entry_ready_target_assumption_fill_type": target_fill_type,
        "bars_from_entry_ready_to_target_assumption_touch": (
            (target_touched_index - signal_index) if target_touched_index is not None else None
        ),
        "ambiguous_intrabar_between_invalidation_and_target_assumption": ambiguous,
        "ambiguous_intrabar_bar_time": (
            scanner._timestamp_at(df_30m_full, ambiguous_index) if ambiguous_index is not None else None
        ),
        "research_mfe": round(mfe, 4),
        "research_mae": round(mae, 4),
    })
    return signal


def run_walk_forward_evaluation(
    ticker: str,
    direction: str,
    df_4h: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_5m: Optional[pd.DataFrame],
    strategy_config: Optional[SmcShadowConfig] = None,
    harness_config: Optional[HarnessConfig] = None,
    market: str = "stock",
    measure_outcomes: bool = True,
) -> dict[str, Any]:
    """The historical (and, identically, paper -- see module docstring)
    walk-forward evaluator. Steps through df_30m bar by bar; at each
    step, truncates df_4h/df_30m/df_5m to bars available as of that 30M
    bar's own timestamp (no lookahead) and calls the REAL, unmodified
    evaluate_smc_shadow() with strategy_config exactly as given (default:
    smc_shadow_v1's own registered config, untouched).

    Returns {ticker, direction, market, strategy_version, harness_version,
    episodes: [...], signals: [...]} -- `signals` (renamed from `trades`
    in Sprint 5.2, see module docstring) is the flat list of every
    entry-ready signal across all episodes, each carrying its own
    `evidence` (the full evaluate_smc_shadow() result at the moment
    ENTRY_READY was reached) and `transition_history` (every state change
    in that episode leading up to it) for full traceability.
    """
    strategy_config = strategy_config or STRATEGY_CONFIGS[STRATEGY_VERSION]
    harness_config = harness_config or HarnessConfig()
    direction_uc = _direction_upper(direction)

    episodes: list[dict[str, Any]] = []
    current_episode: Optional[dict[str, Any]] = None
    start = harness_config.min_confirmation_bars

    for i in range(start, len(df_30m)):
        as_of = df_30m.index[i]
        df_30m_slice = df_30m.iloc[: i + 1]
        df_4h_slice = _slice_up_to(df_4h, as_of)
        df_5m_slice = _slice_up_to(df_5m, as_of)
        if df_4h_slice is None or len(df_4h_slice) < harness_config.min_htf_bars:
            continue

        result = evaluate_smc_shadow(ticker, direction, df_4h_slice, df_30m_slice, df_5m_slice, config=strategy_config)
        result["_bar_time"] = scanner._timestamp_at(df_30m, i)
        new_state = result["state"]

        # Sprint 5.1: episodes now close ONLY on a genuine return to
        # WATCHING or a terminal INVALIDATED -- a trade no longer closes
        # the episode on its own (see module docstring's "episode
        # boundaries" section). A brand-new episode (a completely new
        # setup/thesis) can only start once the prior one has genuinely
        # closed this way.
        #
        # Sprint 6.1 fix (see module docstring): this bar's OWN
        # invalidation-check and signal-recording (below) must run even
        # when this is the very bar a new episode is created on -- an
        # earlier unconditional `continue` here skipped both for that
        # bar, which Sprint 6's real-data baseline found left some
        # episodes with a wrong invalidated/signal record whenever the
        # episode didn't happen to "self-heal" on a later bar.
        # `just_created` below ONLY suppresses the states_visited/
        # transitions bookkeeping _new_episode() already seeded for this
        # exact bar (preventing a duplicate entry) -- every other check
        # further down now runs unconditionally for every eligible bar,
        # including this one.
        just_created = False
        if current_episode is None:
            if new_state == "WATCHING":
                continue
            current_episode = _new_episode(i, result)
            _update_episode_evidence_fields(current_episode, result)
            just_created = True

        if not just_created:
            if new_state != current_episode["states_visited"][-1]:
                current_episode["transitions"].append({
                    "from_state": current_episode["states_visited"][-1], "to_state": new_state, "evidence": result,
                })
            current_episode["states_visited"].append(new_state)
            current_episode["max_state_reached"] = _more_advanced(current_episode["max_state_reached"], new_state)
            _update_episode_evidence_fields(current_episode, result)

        # Track the two objective signals a genuinely new execution
        # opportunity (vs. continued observation of the same one) needs,
        # independent of whether a signal has been recorded yet this bar.
        pullback_status = result.get("pullback_status")
        if pullback_status not in (None, "observed", "not_applicable"):
            current_episode["_regressed_since_last_signal"] = True
        if current_episode["_last_signal_execution_area"] is not None:
            area = current_episode["_last_signal_execution_area"]
            close = float(df_30m_slice["Close"].iloc[-1])
            if close < area["low"] or close > area["high"]:
                current_episode["_left_prior_area_since_last_signal"] = True

        if new_state == "ENTRY_READY":
            current_identity = _execution_event_identity(result)
            if not current_episode["signals"]:
                should_record = True  # the first signal in this episode
            else:
                # A second (or later) signal under the SAME still-open
                # episode requires ALL of: a genuinely different
                # execution-event identity, a real pullback cycle since
                # the last signal, and price having left the prior
                # signal's execution area -- see module docstring. Any
                # one condition alone (e.g. just a new bar_time) is not
                # treated as sufficient on its own.
                should_record = (
                    current_identity is not None
                    and current_identity != current_episode["_last_execution_identity"]
                    and current_episode["_regressed_since_last_signal"]
                    and current_episode["_left_prior_area_since_last_signal"]
                )
            if should_record:
                assumed_entry_price = float(df_30m_slice["Close"].iloc[-1])
                signal = _build_entry_ready_signal(
                    ticker, direction_uc, result, assumed_entry_price, result.get("_bar_time"), harness_config,
                )
                if signal is not None:
                    signal["evidence"] = result
                    signal["transition_history"] = list(current_episode["transitions"])
                    signal["market"] = market
                    signal["location_type"] = current_episode["location_type"]
                    # Confirmation is thesis-level -- the SAME 30M event
                    # applies to every signal in this episode by
                    # definition (a second signal only exists because the
                    # HTF thesis/30M confirmation never reset). Execution
                    # is per-opportunity, though -- Sprint 5.1: read from
                    # THIS bar's own result, not the episode's "first
                    # ever" value, since a second signal's execution event
                    # can genuinely differ in type from the first's.
                    signal["confirmation_event_type"] = current_episode["confirmation_event_type"]
                    signal["confirmation_displacement_label"] = current_episode["confirmation_displacement_label"]
                    signal["execution_event_type"] = result["execution_event"]["type"]
                    signal["execution_area_source"] = result["execution_area"]["source"] if result.get("execution_area") else None
                    signal["time_to_confirmation_bars"] = _bar_gap(current_episode, "confirmed_bar_time", df_30m, i)
                    signal["time_to_entry_ready_bars"] = i - current_episode["start_index"]
                    if measure_outcomes:
                        signal = _measure_post_signal_diagnostics(df_30m, i, signal, direction_uc, harness_config)
                    current_episode["signals"].append(signal)
                    current_episode["_last_execution_identity"] = current_identity
                    current_episode["_last_signal_execution_area"] = result.get("execution_area")
                    current_episode["_regressed_since_last_signal"] = False
                    current_episode["_left_prior_area_since_last_signal"] = False

        if new_state == "INVALIDATED":
            current_episode["invalidated"] = True
            current_episode["invalidated_before_entry_ready"] = len(current_episode["signals"]) == 0

        episode_should_close = new_state == "WATCHING" or new_state == "INVALIDATED"
        if episode_should_close:
            current_episode["end_index"] = i
            episodes.append(current_episode)
            current_episode = None

    if current_episode is not None:
        current_episode["end_index"] = len(df_30m) - 1
        current_episode["incomplete"] = True
        episodes.append(current_episode)

    signals = [s for ep in episodes for s in ep["signals"]]
    for ep in episodes:
        ep["missed_due_to_no_pullback"] = (
            not ep["signals"]
            and not ep["invalidated"]
            and ep["max_state_reached"] in ("CONFIRMED", "WAITING_FOR_PULLBACK")
        )

    return {
        "ticker": str(ticker or "").upper(),
        "direction": direction,
        "market": market,
        "strategy_version": strategy_config.version,
        "harness_version": harness_config.version,
        "episodes": episodes,
        "signals": signals,
    }


def _bar_gap(episode: dict[str, Any], field: str, df_30m: pd.DataFrame, current_index: int) -> Optional[int]:
    if episode.get(field) is None:
        return None
    return current_index - episode["start_index"]


# ---------------------------------------------------------------------------
# Segmentation / aggregate reporting
# ---------------------------------------------------------------------------

SEGMENTATION_DIMENSIONS = (
    "market",                       # stock vs forex
    "direction",                    # long vs short
    "location_type",                # HTF location type
    "confirmation_event_type",      # 30M event type: CHoCH vs BOS
    "confirmation_displacement_label",  # displacement strength
    "execution_event_type",         # 5M execution type: CHoCH vs BOS vs REJECTION
    "execution_area_source",        # whether a 30M execution zone existed vs a 5M-displacement fallback
)


# Sprint 5.2: the ONLY value trade_performance ever takes anywhere in
# this module's aggregation output -- a hardcoded dict, never computed
# from signals. There is no code path here that can produce a win_rate
# or expectancy number for smc_shadow_v1 -- see module docstring.
TRADE_PERFORMANCE_UNAVAILABLE = {
    "status": "unavailable_unresolved_for_smc_shadow_v1",
    "reason": (
        "smc_shadow_v1 directly produces direction and an invalidation "
        "(stop) level, but does not define an entry price, entry "
        "timestamp, or target -- all three are evaluator_specific_"
        "assumption (see each signal's own provenance fields). A "
        "primary win/loss/R-multiple/win-rate/expectancy requires ALL "
        "trade-plan fields to be owned by the strategy version under "
        "test; smc_shadow_v1 does not yet own entry+target semantics, "
        "so trade performance is unresolved by construction, not "
        "computed and hidden. See post_signal_market_behavior for the "
        "available, separately-labeled research diagnostics."
    ),
}


def _max_progression_state(ep: dict[str, Any]) -> str:
    """Sprint 6.1: 'how far through the strategy funnel did this setup
    EVER progress' -- built from the episode's own evidence fields
    (populated by _update_episode_evidence_fields independently of
    whatever the `state` field's same-bar invalidation override later
    does), NOT from max_state_reached/invalidated. See module docstring's
    Sprint 6.1 section for exactly why these two views are DIFFERENT
    questions and both stay -- this is deliberately NOT "corrected" to
    match terminal_state; they measure different things by design."""
    if ep["entry_ready_bar_time"] is not None or ep["signals"]:
        return "ENTRY_READY"
    if ep["execution_event_type"] is not None:
        return "EXECUTION_READY"
    if ep["confirmation_event_type"] is not None:
        # No standalone episode-level flag distinguishes "confirmed" from
        # "confirmed, pullback assessment underway" -- WAITING_FOR_PULLBACK
        # and CONFIRMED share this bucket on the evidence view (disclosed,
        # not silently conflated -- see report's own unresolved-assumptions
        # list).
        return "WAITING_FOR_PULLBACK"
    if ep["location_type"] is not None:
        return "LOCATION_REACHED"
    return "WATCHING"


def _setup_statistics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Layer 1 -- the state funnel, as TWO formally-distinguished views
    (Sprint 6.1) -- see module docstring:
      - terminal_state_reached_at_least: "where was the setup when its
        episode ended" -- the state machine's own returned `state` label
        (via max_state_reached/invalidated), respecting smc_shadow_v1's
        own invalidation-overrides-everything design.
      - max_progression_reached_at_least: "how far through the funnel did
        the setup EVER progress" -- from the episode's own evidence
        fields, independent of a same-bar invalidation override.
    Neither is "the corrected version of the other" -- they answer
    different questions and are both always reported, precisely named,
    so no downstream report can implicitly pick one."""
    terminal_state_reached = {s: 0 for s in SHADOW_STATES}
    max_progression_reached = {s: 0 for s in SHADOW_STATES}
    for ep in episodes:
        for s in SHADOW_STATES:
            if s == "INVALIDATED":
                if ep["invalidated"]:
                    terminal_state_reached[s] += 1
            elif _PROGRESSION_ORDER.get(ep["max_state_reached"], -1) >= _PROGRESSION_ORDER.get(s, 999):
                terminal_state_reached[s] += 1

        if ep["invalidated"]:
            max_progression_reached["INVALIDATED"] += 1
        max_prog = _max_progression_state(ep)
        for s in SHADOW_STATES:
            if s == "INVALIDATED":
                continue
            if _PROGRESSION_ORDER.get(max_prog, -1) >= _PROGRESSION_ORDER.get(s, 999):
                max_progression_reached[s] += 1

    return {
        "detected": len(episodes),
        "terminal_state_reached_at_least": terminal_state_reached,
        "max_progression_reached_at_least": max_progression_reached,
        "entry_ready_signal_count": sum(len(ep["signals"]) for ep in episodes),
        "invalidated_before_entry_ready_count": len(
            [ep for ep in episodes if ep["invalidated_before_entry_ready"]]
        ),
        "missed_due_to_no_pullback_count": len([ep for ep in episodes if ep["missed_due_to_no_pullback"]]),
        "incomplete_episode_count": len([ep for ep in episodes if ep["incomplete"]]),
    }


def _post_signal_market_behavior(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Layer 2 -- what price actually did AFTER each entry-ready signal,
    reported as independently-tracked touch diagnostics, never combined
    into a single outcome. See module docstring's Sprint 5.2 section and
    _measure_post_signal_diagnostics."""
    measured = [s for s in signals if s["post_entry_ready_invalidation_touched"] is not None]
    invalidation_touched = [s for s in measured if s["post_entry_ready_invalidation_touched"] is True]
    target_assumption_touched = [s for s in measured if s["post_entry_ready_target_assumption_touched"] is True]
    ambiguous = [
        s for s in measured if s["ambiguous_intrabar_between_invalidation_and_target_assumption"] is True
    ]
    stop_gapped = [s for s in measured if s.get("post_entry_ready_invalidation_fill_type") == "gap"]
    target_gapped = [s for s in measured if s.get("post_entry_ready_target_assumption_fill_type") == "gap"]
    mfe_values = [s["research_mfe"] for s in measured if s["research_mfe"] is not None]
    mae_values = [s["research_mae"] for s in measured if s["research_mae"] is not None]

    return {
        "measured_count": len(measured),
        "post_entry_ready_invalidation_touched_count": len(invalidation_touched),
        "post_entry_ready_invalidation_touched_rate": (
            round(len(invalidation_touched) / len(measured), 3) if measured else None
        ),
        "post_entry_ready_target_assumption_touched_count": len(target_assumption_touched),
        "post_entry_ready_target_assumption_touched_rate": (
            round(len(target_assumption_touched) / len(measured), 3) if measured else None
        ),
        "ambiguous_intrabar_between_invalidation_and_target_assumption_count": len(ambiguous),
        "post_entry_ready_invalidation_gap_count": len(stop_gapped),
        "post_entry_ready_target_assumption_gap_count": len(target_gapped),
        "avg_research_mfe": round(sum(mfe_values) / len(mfe_values), 4) if mfe_values else None,
        "avg_research_mae": round(sum(mae_values) / len(mae_values), 4) if mae_values else None,
    }


def _signal_summary(episodes: list[dict[str, Any]], signals: list[dict[str, Any]]) -> dict[str, Any]:
    """A single segment's full 3-layer summary -- used both for the
    overall report and each per-dimension segment bucket below."""
    return {
        "setup_statistics": _setup_statistics(episodes),
        "post_signal_market_behavior": _post_signal_market_behavior(signals),
        "trade_performance": dict(TRADE_PERFORMANCE_UNAVAILABLE),
    }


def aggregate_evaluation_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combines any number of run_walk_forward_evaluation() outputs (one
    per ticker/direction) into the required 3-layer top-level report
    (setup_statistics / post_signal_market_behavior / trade_performance)
    plus the same 3-layer breakdown per segment across every dimension in
    SEGMENTATION_DIMENSIONS. Pure aggregation -- no re-evaluation, no
    threshold access at all. trade_performance is ALWAYS the fixed
    TRADE_PERFORMANCE_UNAVAILABLE dict, at every level -- there is no
    code path here that computes a win_rate or expectancy for
    smc_shadow_v1."""
    all_episodes = [ep for run in runs for ep in run["episodes"]]
    all_signals = [s for run in runs for s in run["signals"]]

    segments: dict[str, dict[str, Any]] = {}
    for dim in SEGMENTATION_DIMENSIONS:
        # Segment episodes by their own location_type (episode-level) or,
        # for dims that only exist on signals (confirmation/execution
        # event type, displacement label, execution area source), bucket
        # episodes by the dim value carried by any of their signals --
        # an episode with no signals simply cannot be segmented by a
        # signal-only dimension and is excluded from that dimension's
        # setup_statistics bucketing, though its signals list stays empty
        # regardless.
        episode_buckets: dict[Any, list[dict[str, Any]]] = {}
        signal_buckets: dict[Any, list[dict[str, Any]]] = {}
        for ep in all_episodes:
            if dim == "location_type":
                key = ep.get(dim)
                episode_buckets.setdefault(key, []).append(ep)
            elif dim in ("market", "direction"):
                # Carried on signals, but conceptually episode-level for
                # a given run -- fall back to the run's own signals if
                # the episode itself has none, so still-open/invalidated
                # episodes aren't silently dropped from setup_statistics.
                keys = {s.get(dim) for s in ep["signals"]}
                if not keys:
                    continue
                for key in keys:
                    episode_buckets.setdefault(key, []).append(ep)
            else:
                keys = {s.get(dim) for s in ep["signals"]}
                for key in keys:
                    episode_buckets.setdefault(key, []).append(ep)
        for s in all_signals:
            key = s.get(dim)
            signal_buckets.setdefault(key, []).append(s)

        keys = set(episode_buckets) | set(signal_buckets)
        segments[dim] = {
            str(k): _signal_summary(episode_buckets.get(k, []), signal_buckets.get(k, []))
            for k in keys
        }

    return {
        "harness_version": HARNESS_VERSION,
        "strategy_version": runs[0]["strategy_version"] if runs else STRATEGY_VERSION,
        "tickers_evaluated": sorted({run["ticker"] for run in runs}),
        "setup_statistics": _setup_statistics(all_episodes),
        "post_signal_market_behavior": _post_signal_market_behavior(all_signals),
        "trade_performance": dict(TRADE_PERFORMANCE_UNAVAILABLE),
        "segments": segments,
    }
