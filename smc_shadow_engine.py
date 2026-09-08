"""Kairos Sprint 4 -- Shadow SMC Strategy/State Engine (2026-09 session).
Recalibrated in Sprint 4.1 to remove accidental overconstraints found by
auditing this file's exact trading semantics against the intended
playbook -- see the module-level "Sprint 4.1" section below and
dashboard_sprint4_1_report.md for the full old-vs-new gate comparison.

A separate, feature-flagged RESEARCH engine that evaluates a setup against
the new discretionary Kairos SMC workflow (4H thesis/location -> 30M
CHoCH/BOS confirmation -> wait for pullback, don't chase -> 5M execution
confirmation) and produces a shadow state, entirely independent of the
frozen production strategy in scanner.py. This module:

  - NEVER writes to approved_setup_memories, approved_setup_monitor_state,
    candidate_promotions, or journal_entries.
  - NEVER calls a live trade/alert/notification function.
  - NEVER feeds scanner._ranking_status_bucket / ENTER_NOW / any dashboard
    read model (dashboard_state.py, candidates_router.py's
    /candidates/dashboard-state) -- the Erica/Sprint-3 dashboards cannot
    turn green or show any signal from this engine's output because they
    do not read it at all.
  - Is PURE with respect to strategy computation: evaluate_smc_shadow()
    takes already-fetched OHLCV DataFrames and returns a result dict. It
    makes no network call and touches no database. (Persistence and live
    bar-fetching for the periodic research tick live in
    smc_shadow_store.py / smc_shadow_router.py, kept deliberately
    separate so this file stays trivially unit-testable with synthetic
    data and has zero side effects of its own.)

Reuse, not reinvention: every structural primitive below is an EXISTING,
already-tested, deterministic function already used by the frozen
production strategy or its own informational-only scoring modules --
none of it is new SMC logic invented for this sprint:
  - scanner._find_swings / _get_trend                    (swing/HTF trend)
  - scanner._find_order_block                             (order-block location type)
  - structural_resistance.levels_near_target              (structural support/
                                                             resistance location type,
                                                             reused against swing
                                                             pivots -- Sprint 4.1)
  - location_score.score_location                         (premium/discount
                                                             location type)
  - scanner._detect_bos / _detect_choch / _first_bos_close_index
                                                            (structural break --
                                                             30M confirmation AND,
                                                             since Sprint 4.1, one
                                                             of three 5M execution
                                                             event types)
  - scanner.detect_liquidity_sweep / scanner.detect_rejection
                                                            (rejection/reclaim --
                                                             the third 5M execution
                                                             event type, Sprint 4.1)
  - scanner._compute_atr / _timestamp_at                   (ATR, bar->ISO time)
  - displacement_score.score_displacement                  (candle-quality /
                                                             weak-vs-meaningful
                                                             displacement)

Where existing code/data cannot deterministically resolve a concept (no
HTF location of any accepted type found; insufficient swing history for
HTF direction; not enough bars since confirmation to judge a pullback; no
deterministic supply/demand zone detector at all), this module records
that fact explicitly in the result's `unresolved` list and the relevant
evidence field is left None/"unresolved" -- it is never silently guessed.
See SmcShadowConfig's own docstring for the handful of genuinely
unvalidated threshold placeholders this naturally requires (same
disclosure convention as displacement_score.py's own weights).

---------------------------------------------------------------------------
Sprint 4.1 -- fidelity calibration (what changed and why)
---------------------------------------------------------------------------
Sprint 4 accidentally over-constrained two of the five stages by picking
ONE deterministic primitive per concept and treating it as though it were
the ONLY valid one:

  1. HTF LOCATION previously required scanner._find_order_block to return
     a real zone -- but an order block is only ONE of several location
     types Erica's playbook actually uses (supply/demand, structural
     support/resistance, premium/discount within the active range, or an
     order block when one genuinely exists). LOCATION_REACHED is now
     gated by whichever of THREE deterministic location types resolves
     (order_block, structural_level, premium_discount), evaluated in that
     priority order and reported via an explicit location_type/
     location_source/location_score/location_bounds/location_reason
     schema (see _evaluate_htf_location). A fourth type, supply_demand, is
     named in the schema (LOCATION_TYPES) but deliberately never resolved
     in smc_shadow_v1 -- there is no deterministic supply/demand zone
     detector in this codebase distinct from the order-block concept
     already covered above, and this module will not manufacture one just
     to fill in the state machine. It is exposed as research-only/
     reserved, and its absence is called out explicitly in `unresolved`
     whenever no location type resolves at all.
  2. 5M EXECUTION previously required a BOS specifically -- now a generic
     execution-event abstraction (_execution_trigger) accepts BOS, CHoCH,
     or REJECTION (a liquidity sweep of the most recent opposing swing
     followed by an objective rejection candle -- scanner.
     detect_liquidity_sweep + scanner.detect_rejection, both existing,
     both already used elsewhere in this codebase; nothing new was
     invented). Which one fired is recorded explicitly as
     execution_event["type"].
  3. PULLBACK previously measured a generic ATR-based retracement
     fraction only. That diagnostic is PRESERVED (excursion/ATR-multiple/
     retracement_fraction still computed and still drives the
     "chased_no_pullback" read), but a pullback is now ALSO recognized
     the more literal way the playbook describes it -- price returning to
     an objectively-defined execution area created by the confirmation
     displacement itself (its own candle body), with an explicit,
     disclosed, deterministic fallback to a materially-displaced 5M
     candle's body when the 30M confirmation candle's own body is too
     small to function as a meaningful revisit zone (the literal "Silver
     lesson" case: 4H location -> 30M confirmation/displacement -> 5M
     execution zone CREATED BY that displacement -> pullback -> 5M
     confirmation). See _confirmation_execution_area. pullback_status is
     "observed" when EITHER signal says so -- this is a genuine widening
     of what counts as a real pullback, not a narrowing, and every
     Sprint-4 scenario that already passed via the retracement-fraction
     path continues to pass unchanged.
  4. 30M CONFIRMATION (CHoCH-or-BOS candidate, triggering candle located,
     displacement quality measured, weak/noisy breaks rejected) and the
     displacement threshold itself are UNCHANGED, per this sprint's own
     explicit instruction not to touch either.
  5. STRATEGY VERSIONING: SmcShadowConfig now carries its own `version`
     field and an explicit accepted-types schema
     (accepted_htf_location_types / accepted_confirmation_events /
     accepted_execution_events / invalidation_source_priority), and
     evaluate_smc_shadow() stamps result["strategy_version"] from
     config.version rather than a hardcoded module constant -- a future
     rule variant is a NEW SmcShadowConfig(version=...), never a silent
     mutation of smc_shadow_v1's own params. See STRATEGY_CONFIGS.
  6. EXPLAINABILITY: every result now also carries next_condition
     (structured, not prose) answering "what is this setup waiting for
     next", alongside the existing per-stage `reason` strings that answer
     "why did it advance/not advance".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

import scanner
from displacement_score import score_displacement
from location_score import score_location
from structural_resistance import levels_near_target

STRATEGY_VERSION = "smc_shadow_v1"

# Feature flag -- OFF unless explicitly set. Checked by smc_shadow_router.py
# before running any evaluation and by main.py before registering the
# periodic background tick at all -- when this is false, nothing in this
# whole feature computes, persists, or schedules anything.
SMC_SHADOW_ENABLED_ENV_VAR = "KAIROS_SMC_SHADOW_ENABLED"


def smc_shadow_enabled() -> bool:
    return str(os.environ.get(SMC_SHADOW_ENABLED_ENV_VAR) or "").strip().lower() in ("1", "true", "yes", "on")


# The exact required state sequence (plus the terminal INVALIDATED state,
# reachable from any non-terminal state). Order here is the natural
# progression, not a priority ranking.
SHADOW_STATES = (
    "WATCHING",
    "LOCATION_REACHED",
    "CONFIRMED",
    "WAITING_FOR_PULLBACK",
    "EXECUTION_READY",
    "ENTRY_READY",
    "INVALIDATED",
)

# Every HTF location type this schema knows about. "supply_demand" is
# permanently reserved/research-only in smc_shadow_v1 -- see the Sprint
# 4.1 module docstring section above for why it is named but never
# resolved. Order here also fixes the evaluation priority (most specific
# claim first).
LOCATION_TYPES = ("order_block", "structural_level", "premium_discount", "supply_demand")
UNRESOLVED_LOCATION_TYPES = frozenset({"supply_demand"})

# Every 30M confirmation event type this schema knows about (unchanged
# from Sprint 4 -- see the task's own instruction to preserve this stage
# as-is).
CONFIRMATION_EVENT_TYPES = ("BOS", "CHoCH")

# Every 5M execution event type this schema knows about, in evaluation
# priority order.
EXECUTION_EVENT_TYPES = ("BOS", "CHoCH", "REJECTION")


@dataclass
class SmcShadowConfig:
    """Version-specific, disclosed-as-unvalidated tuning knobs. Every
    threshold here is a genuine strategy judgment call this module does
    NOT claim is "correct" -- exactly the same disclosure convention
    displacement_score.py already uses for its own component weights.

    `version` is this config's own identity -- evaluate_smc_shadow()
    stamps its result's strategy_version from THIS field, not from a
    hardcoded module constant. A future rule variant must be a NEW
    SmcShadowConfig with a new `version` string (see STRATEGY_CONFIGS
    below), never a silent in-place change to an existing one -- "do not
    silently mutate a strategy version later" (Sprint 4.1 requirement).
    """
    version: str = STRATEGY_VERSION

    # ---- accepted-type schema (Sprint 4.1) ----
    # Which HTF location types this version's LOCATION_REACHED gate will
    # accept, in priority order. Defaults to every deterministically
    # supported type (i.e. everything except "supply_demand", which is
    # never resolved regardless of whether it's listed here).
    accepted_htf_location_types: tuple[str, ...] = ("order_block", "structural_level", "premium_discount")
    # Which 30M events count as a confirmation candidate. Unchanged from
    # Sprint 4 by explicit instruction.
    accepted_confirmation_events: tuple[str, ...] = ("BOS", "CHoCH")
    # Which 5M events count as an execution trigger.
    accepted_execution_events: tuple[str, ...] = ("BOS", "CHoCH", "REJECTION")
    # Which evidence sources, in priority order, may anchor the
    # invalidation level. "location_bounds" = the resolved HTF location's
    # own bounds (whichever type produced them); "confirmation_level" =
    # the 30M confirmation event's own broken level, used only when no
    # location bounds exist.
    invalidation_source_priority: tuple[str, ...] = ("location_bounds", "confirmation_level")

    # Swing detection margin (bars each side) -- same parameter shape as
    # scanner._find_swings's own default (margin=4), applied on 4H/30M.
    swing_margin_htf: int = 4
    swing_margin_confirmation: int = 4
    # A tighter margin on 5M -- the execution timeframe is intentionally
    # more sensitive to short-lived structure than 4H/30M.
    swing_margin_execution: int = 3

    bos_lookback_confirmation: int = 40
    bos_lookback_execution: int = 20

    # Confirmation displacement gate: the 30M CHoCH/BOS candle's
    # score_displacement() label must be at least this -- "WEAK" alone is
    # exactly the "noisy break" case the task says to distinguish from a
    # meaningful one. ("WEAK" < "MODERATE" < "STRONG" per
    # displacement_score.py's own label cutoffs.) UNCHANGED in Sprint 4.1
    # by explicit instruction.
    min_confirmation_displacement_label: str = "MODERATE"
    # Same idea for the 5M execution trigger candle (BOS/CHoCH path only
    # -- REJECTION's own quality bar is scanner.detect_rejection's own
    # candle-shape test, not a displacement score).
    min_execution_displacement_label: str = "MODERATE"

    # How close (in ATR) price must sit to a swing pivot for the
    # "structural_level" HTF location type to resolve (Sprint 4.1).
    structural_level_proximity_atr: float = 0.5

    # Pullback / chase-prevention thresholds (see module docstring's
    # "ambiguous SMC concept" handling). `chase_max_extension_atr`: once
    # price has moved this many 30M-ATRs beyond the confirmation candle's
    # close without retracing, it counts as "expansion we should not
    # chase" rather than "still building". `pullback_min_retracement_fraction`:
    # fraction of that post-confirmation excursion price must retrace
    # back toward the broken level before a pullback counts as "observed"
    # via the retracement-fraction signal specifically (the
    # execution-area return signal, added in Sprint 4.1, is independent
    # of this fraction -- see _confirmation_execution_area).
    chase_max_extension_atr: float = 1.5
    pullback_min_retracement_fraction: float = 0.30
    # Minimum size (in 30M ATR) the confirmation candle's own body must
    # be to serve as the execution area on its own; below this, the
    # engine looks for a materially-displaced 5M candle within that same
    # 30M bar's window instead (Sprint 4.1's "Silver lesson" fallback).
    min_execution_zone_atr: float = 0.15

    # REJECTION execution-event parameters (Sprint 4.1) -- passed straight
    # through to scanner.detect_liquidity_sweep / scanner.detect_rejection.
    rejection_sweep_lookback: int = 12
    rejection_lookback: int = 5

    def __post_init__(self) -> None:
        if self.min_confirmation_displacement_label not in ("WEAK", "MODERATE", "STRONG"):
            raise ValueError("min_confirmation_displacement_label must be WEAK/MODERATE/STRONG")
        if self.min_execution_displacement_label not in ("WEAK", "MODERATE", "STRONG"):
            raise ValueError("min_execution_displacement_label must be WEAK/MODERATE/STRONG")
        for t in self.accepted_htf_location_types:
            if t not in LOCATION_TYPES:
                raise ValueError(f"unknown HTF location type: {t}")
        for t in self.accepted_confirmation_events:
            if t not in CONFIRMATION_EVENT_TYPES:
                raise ValueError(f"unknown confirmation event type: {t}")
        for t in self.accepted_execution_events:
            if t not in EXECUTION_EVENT_TYPES:
                raise ValueError(f"unknown execution event type: {t}")


# Registry of named strategy versions -- "separate playbook from
# implementation" (Sprint 4.1 requirement #5). A future research variant
# is added HERE as a new entry, never by editing smc_shadow_v1's own
# values in place.
STRATEGY_CONFIGS: dict[str, SmcShadowConfig] = {
    STRATEGY_VERSION: SmcShadowConfig(version=STRATEGY_VERSION),
}


_DISPLACEMENT_LABEL_RANK = {"WEAK": 0, "MODERATE": 1, "STRONG": 2}


def _displacement_meets_bar(label: Optional[str], minimum_label: str) -> bool:
    if label is None:
        return False
    return _DISPLACEMENT_LABEL_RANK.get(label, -1) >= _DISPLACEMENT_LABEL_RANK.get(minimum_label, 99)


def _direction_upper(direction: str) -> str:
    d = str(direction or "").strip().upper()
    return d if d in ("LONG", "SHORT") else ""


def _direction_lower(direction: str) -> str:
    d = str(direction or "").strip().lower()
    return d if d in ("long", "short") else ""


def _latest_close(*frames: Optional[pd.DataFrame]) -> Optional[float]:
    """The freshest available close across the given frames, finest
    granularity first (5M, then 30M, then 4H is the typical call order)."""
    for df in frames:
        if df is not None and len(df) > 0:
            return float(df["Close"].iloc[-1])
    return None


def _zone_touched(df: Optional[pd.DataFrame], zone_low: float, zone_high: float, lookback: int = 60) -> bool:
    """Has price traded into [zone_low, zone_high] at any point in the
    trailing `lookback` bars of `df`? A plain range-overlap check -- not a
    new SMC concept, just "did a candle's high/low range cross this
    price band"."""
    if df is None or len(df) == 0:
        return False
    window = df.iloc[-lookback:]
    return bool(((window["Low"] <= zone_high) & (window["High"] >= zone_low)).any())


def _returned_to_area_after_clearing(df: Optional[pd.DataFrame], zone_low: float, zone_high: float, direction_uc: str) -> bool:
    """Has price fully cleared [zone_low, zone_high] (moved decisively
    beyond it) and THEN come back to touch it? Deliberately distinct from
    _zone_touched: the bar immediately after a confirmation candle
    typically still overlaps that candle's own body just by virtue of
    starting there -- that is not a pullback, it's the starting point.
    This requires a genuine "left, then came back" round trip before
    counting a "return"."""
    if df is None or len(df) == 0:
        return False
    cleared = False
    for i in range(len(df)):
        low = float(df["Low"].iloc[i])
        high = float(df["High"].iloc[i])
        if direction_uc == "LONG":
            if not cleared:
                if low > zone_high:
                    cleared = True
                continue
            if low <= zone_high and high >= zone_low:
                return True
        else:
            if not cleared:
                if high < zone_low:
                    cleared = True
                continue
            if high >= zone_low and low <= zone_high:
                return True
    return False


# ---------------------------------------------------------------------------
# Sprint 4.1: HTF location -- generalized beyond "order block only".
# ---------------------------------------------------------------------------

def _evaluate_htf_location(
    df_4h: Optional[pd.DataFrame],
    swings_4h: list,
    direction_uc: str,
    direction_lc: str,
    current_price: Optional[float],
    config: SmcShadowConfig,
    unresolved: list[str],
) -> dict[str, Any]:
    """Returns the full location evidence dict: location_type,
    location_source, location_score, location_bounds, location_reason,
    plus the raw premium/discount fields (percentile/label/alignment/
    range_high/range_low) as supplementary evidence regardless of which
    type (if any) actually gates progression. location_bounds is None
    when nothing deterministic resolved."""
    location: dict[str, Any] = {
        "location_type": None, "location_source": None, "location_score": None,
        "location_bounds": None, "location_reason": None,
        "percentile": None, "label": None, "alignment": None, "range_high": None, "range_low": None,
    }
    if df_4h is None or len(df_4h) == 0 or current_price is None or not direction_uc:
        unresolved.append("location: no 4H data provided")
        return location

    # Premium/discount percentile is always computed as supplementary
    # evidence (unchanged from Sprint 4), independent of whether it ends
    # up being the type that gates LOCATION_REACHED.
    loc = score_location(current_price, swings_4h, direction_lc)
    location.update({
        "percentile": loc["location_percentile"], "label": loc["location_label"],
        "alignment": loc["location_alignment"], "range_high": loc["range_high"], "range_low": loc["range_low"],
    })
    if loc["location_percentile"] is None:
        unresolved.append("location: insufficient 4H swing range to compute premium/discount")

    # Type 1: order block -- the most specific claim ("this exact zone").
    if "order_block" in config.accepted_htf_location_types:
        ob = scanner._find_order_block(df_4h, direction_uc, swings_4h)
        if ob is not None:
            bounds = {"high": ob["high"], "low": ob["low"]}
            location.update({
                "location_type": "order_block",
                "location_source": "scanner._find_order_block",
                "location_score": 100.0,
                "location_bounds": bounds,
                "location_reason": f"4H order block at {bounds['low']}-{bounds['high']}",
            })
            return location
        unresolved.append("location: no deterministic 4H order block found")

    # Type 2: structural support/resistance -- reuses
    # structural_resistance.levels_near_target against the swing pivots
    # themselves (the same real function candidates_router.py already
    # uses for target clamping), evaluated at the CURRENT price rather
    # than a target price.
    if "structural_level" in config.accepted_htf_location_types:
        atr_4h = scanner._compute_atr(df_4h) if len(df_4h) >= 15 else 0.0
        if atr_4h > 0:
            # levels_near_target's own direction convention means "the
            # direction a TARGET could be blocked from" -- direction="long"
            # makes it check swing HIGHS (resistance above a long's
            # target). For an HTF LOCATION check we want the opposite
            # sense: support BELOW price for a long entry (a swing LOW),
            # resistance ABOVE price for a short entry (a swing HIGH) --
            # so the direction passed here is deliberately inverted from
            # `direction_lc`. Documented here rather than silently correct
            # by accident.
            findings = levels_near_target(
                df_4h.reset_index(drop=True), swings_4h, current_price, atr_4h,
                "short" if direction_lc == "long" else "long",
                proximity_atr=config.structural_level_proximity_atr,
            )
            swing_findings = [f for f in findings if f["kind"] == "swing_pivot"]
            if swing_findings:
                best = swing_findings[0]
                band = config.structural_level_proximity_atr * atr_4h
                bounds = {"high": best["price"] + band, "low": best["price"] - band}
                score_by_strength = {"strong": 90.0, "moderate": 65.0, "weak": 30.0}
                location.update({
                    "location_type": "structural_level",
                    "location_source": "structural_resistance.levels_near_target",
                    "location_score": score_by_strength.get(best["strength"], 50.0),
                    "location_bounds": bounds,
                    "location_reason": (
                        f"Price is within {config.structural_level_proximity_atr} ATR of a "
                        f"{best['strength']} swing level at {best['price']} -- {best['note']}"
                    ),
                })
                return location
        else:
            unresolved.append("location: insufficient 4H bars to compute ATR for the structural-level proximity check")

    # Type 3: premium/discount alignment -- the broadest, weakest form of
    # location evidence: no specific zone, just "price sits in the
    # direction-favorable half of the active 4H range".
    if "premium_discount" in config.accepted_htf_location_types and loc.get("location_alignment") == "favorable":
        bounds = {"high": loc["range_high"], "low": loc["range_low"]}
        ideal_percentile = 0.0 if direction_lc == "long" else 100.0
        location.update({
            "location_type": "premium_discount",
            "location_source": "location_score.score_location",
            "location_score": round(100.0 - abs(loc["location_percentile"] - ideal_percentile), 1),
            "location_bounds": bounds,
            "location_reason": (
                f"Price is in the {loc['location_label']} zone ({loc['location_percentile']}th percentile) "
                f"of the active 4H range -- favorable for {direction_lc}"
            ),
        })
        return location

    # Nothing deterministic resolved. supply_demand is named explicitly
    # here (rather than silently omitted) so a researcher reading
    # `unresolved` knows it exists as a concept and exactly why it never
    # fires in this strategy_version.
    unresolved.append(
        "location: no deterministic HTF location (order block, structural level, or favorable "
        "premium/discount) found for this direction -- note: supply/demand zone detection has no "
        "deterministic primitive in this codebase and is reserved as research-only in "
        f"strategy_version={config.version}, never silently approximated by another type"
    )
    return location


# ---------------------------------------------------------------------------
# Sprint 4.1: 5M execution -- generalized beyond "BOS only".
# ---------------------------------------------------------------------------

def _first_rejection_index(df: pd.DataFrame, direction: str, sweep_level: Optional[float], lookback: int = 5) -> Optional[int]:
    """Mirrors scanner.detect_rejection's own per-candle test exactly, to
    recover WHICH candle qualified (that function returns only a bool) --
    same "reimplement just enough to expose an index the original doesn't"
    precedent as location_score._latest_swing_range's own documented
    byte-for-byte mirroring of scanner._latest_swing_range."""
    if direction not in ("LONG", "SHORT") or sweep_level is None or len(df) < 2:
        return None
    n = len(df)
    start = max(0, n - lookback)
    for i in range(start, n):
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        open_ = float(df["Open"].iloc[i])
        close = float(df["Close"].iloc[i])
        body = abs(close - open_)
        candle_range = high - low
        if candle_range <= 0:
            continue
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        if direction == "LONG":
            closed_back_above = low < sweep_level and close > sweep_level
            lower_wick_failure = low < sweep_level and lower_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_above or (lower_wick_failure and close > open_):
                return i
        else:
            closed_back_below = high > sweep_level and close < sweep_level
            upper_wick_failure = high > sweep_level and upper_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_below or (upper_wick_failure and close < open_):
                return i
    return None


def _execution_trigger(
    df_5m: pd.DataFrame, swings_5m: list, direction_uc: str, config: SmcShadowConfig,
) -> tuple[Optional[str], Optional[float], Optional[int]]:
    """Returns (event_type, level, triggering_index) for the FIRST
    accepted execution event type (in config.accepted_execution_events'
    priority order) that fires, or (None, None, None). Each event type
    reuses an existing, already-tested scanner.py primitive -- see the
    module docstring's Sprint 4.1 section."""
    if "BOS" in config.accepted_execution_events:
        bos_ok, bos_level = scanner._detect_bos(df_5m, swings_5m, direction_uc, lookback=config.bos_lookback_execution)
        if bos_ok:
            idx = scanner._first_bos_close_index(df_5m, swings_5m, direction_uc, bos_level, lookback=config.bos_lookback_execution)
            if idx is not None:
                return "BOS", bos_level, idx

    if "CHoCH" in config.accepted_execution_events:
        suppress, _reason, bearish_lvl, bullish_lvl, choch_idx = scanner._detect_choch(swings_5m, direction_uc)
        choch_ok = (
            (direction_uc == "LONG" and bullish_lvl is not None and not suppress)
            or (direction_uc == "SHORT" and bearish_lvl is not None and not suppress)
        )
        if choch_ok and isinstance(choch_idx, int) and choch_idx >= 0:
            level = bullish_lvl if direction_uc == "LONG" else bearish_lvl
            return "CHoCH", level, choch_idx

    if "REJECTION" in config.accepted_execution_events:
        swept, sweep_level = scanner.detect_liquidity_sweep(df_5m, swings_5m, direction_uc, lookback=config.rejection_sweep_lookback)
        if swept and scanner.detect_rejection(df_5m, direction_uc, sweep_level, lookback=config.rejection_lookback):
            idx = _first_rejection_index(df_5m, direction_uc, sweep_level, lookback=config.rejection_lookback)
            if idx is not None:
                return "REJECTION", sweep_level, idx

    return None, None, None


# ---------------------------------------------------------------------------
# Sprint 4.1: the pullback "execution area" -- the literal Silver lesson.
# ---------------------------------------------------------------------------

def _confirmation_execution_area(
    df_30m: pd.DataFrame, df_5m: Optional[pd.DataFrame], confirming_index: int,
    direction_lc: str, config: SmcShadowConfig, unresolved: list[str],
) -> Optional[dict[str, Any]]:
    """The objectively-defined area price should return to before
    execution is considered -- primarily the 30M confirmation candle's
    own body (a real, deterministic value straight from its OHLC, not a
    new pattern). When that body is too small (< min_execution_zone_atr
    of 30M ATR) to function as a meaningful revisit target, falls back to
    the first materially-displaced 5M candle's body WITHIN that same 30M
    bar's own time window -- the literal "5M execution zone created by
    displacement" the Silver lesson describes. Returns None (and records
    why in `unresolved`) only when neither is available -- never
    fabricates a zone."""
    open_ = float(df_30m["Open"].iloc[confirming_index])
    close = float(df_30m["Close"].iloc[confirming_index])
    body_high, body_low = max(open_, close), min(open_, close)
    atr_30m = scanner._compute_atr(df_30m) if len(df_30m) >= 15 else 0.0
    body_atr_multiple = ((body_high - body_low) / atr_30m) if atr_30m > 0 else None

    if body_atr_multiple is not None and body_atr_multiple >= config.min_execution_zone_atr:
        return {"high": body_high, "low": body_low, "source": "30m_confirmation_candle_body"}

    if df_5m is None or len(df_5m) == 0:
        unresolved.append(
            "execution_area: the 30M confirmation candle's body is too small to use as a pullback "
            "target, and no 5M data is available to look for a displacement-created execution area instead"
        )
        return None

    bar_start = df_30m.index[confirming_index]
    bar_end = bar_start + pd.Timedelta(minutes=30)
    window_mask = (df_5m.index >= bar_start) & (df_5m.index < bar_end)
    window_positions = [i for i, flag in enumerate(window_mask) if flag]
    if not window_positions:
        unresolved.append(
            "execution_area: the 30M confirmation candle's body is too small, and no 5M bars fall "
            "within its own time window for a fallback execution area"
        )
        return None

    for pos in window_positions:
        disp = score_displacement(df_5m, direction_lc, index=pos)
        if _displacement_meets_bar(disp.get("label"), config.min_confirmation_displacement_label):
            o5 = float(df_5m["Open"].iloc[pos])
            c5 = float(df_5m["Close"].iloc[pos])
            return {"high": max(o5, c5), "low": min(o5, c5), "source": "5m_displacement_candle_body"}

    unresolved.append(
        "execution_area: the 30M confirmation candle's body is too small, and no meaningfully-displaced "
        "5M candle was found within its own time window either"
    )
    return None


def evaluate_smc_shadow(
    ticker: str,
    direction: str,
    df_4h: Optional[pd.DataFrame],
    df_30m: Optional[pd.DataFrame],
    df_5m: Optional[pd.DataFrame],
    config: Optional[SmcShadowConfig] = None,
) -> dict[str, Any]:
    """Pure evaluation -- no network call, no DB write. Returns a dict
    with at minimum: state, strategy_version, unresolved (list[str]),
    next_condition, and the structured evidence fields the task requires
    (htf_direction, location, execution_area, confirmation_event,
    displacement_evidence, pullback_status, execution_event,
    invalidation).

    A completely missing/empty timeframe frame is handled the same way
    as any other under-determined input: the concepts that frame would
    have resolved are marked unresolved, and the state simply cannot
    advance past whatever the remaining frames can support -- it never
    guesses to fill the gap.
    """
    config = config or STRATEGY_CONFIGS[STRATEGY_VERSION]
    direction_lc = _direction_lower(direction)
    direction_uc = _direction_upper(direction)
    unresolved: list[str] = []

    result: dict[str, Any] = {
        "ticker": str(ticker or "").upper(),
        "direction": direction_lc or None,
        "strategy_version": config.version,
        "state": "WATCHING",
        "htf_direction": None,
        "htf_direction_aligned": None,
        "location": {
            "location_type": None, "location_source": None, "location_score": None,
            "location_bounds": None, "location_reason": None,
            "percentile": None, "label": None, "alignment": None, "range_high": None, "range_low": None,
        },
        "zone": None,  # == location["location_bounds"]; kept for backward compatibility (smc_shadow_store.py's zone_high/zone_low columns)
        "confirmation_event": {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None},
        "displacement_evidence": {"confirmation": None, "execution": None},
        "execution_area": None,
        "pullback_status": "not_applicable",
        "pullback_detail": None,
        "execution_event": {"detected": False, "type": None, "level": None, "bar_time": None, "reason": None},
        "invalidation": {"invalidated": False, "level": None, "reason": None},
        "unresolved": unresolved,
        "next_condition": None,
    }

    if not direction_uc:
        unresolved.append("direction: missing or not long/short -- no evaluation possible")
        result["next_condition"] = {"waiting_on": "direction", "detail": "A valid long/short direction is required before any evaluation can proceed."}
        return result

    # ---- Step 1: 4H HTF thesis direction ----
    if df_4h is None or len(df_4h) == 0:
        unresolved.append("htf_direction: no 4H data provided")
        swings_4h: list = []
        htf_trend = "NEUTRAL"
    else:
        swings_4h = scanner._find_swings(df_4h, margin=config.swing_margin_htf)
        htf_trend = scanner._get_trend(swings_4h)
        if htf_trend == "NEUTRAL":
            unresolved.append("htf_direction: insufficient 4H swing history to establish a trend")
    result["htf_direction"] = htf_trend
    result["htf_direction_aligned"] = htf_trend == direction_uc

    # ---- Step 2: HTF "major location" -- generalized (Sprint 4.1) ----
    current_price = _latest_close(df_5m, df_30m, df_4h)
    location = _evaluate_htf_location(df_4h, swings_4h, direction_uc, direction_lc, current_price, config, unresolved)
    result["location"] = location
    location_bounds = location["location_bounds"]
    result["zone"] = location_bounds

    location_reached = bool(
        result["htf_direction_aligned"]
        and location_bounds is not None
        and _zone_touched(df_30m if df_30m is not None else df_4h, location_bounds["low"], location_bounds["high"])
    )

    state = "WATCHING"
    if location_reached:
        state = "LOCATION_REACHED"
    elif not result["htf_direction_aligned"]:
        pass  # next_condition filled in below
    elif location_bounds is not None:
        unresolved.append("location: a location was found but price has not yet traded into it")

    # ---- Step 3: 30M directional confirmation (CHoCH/BOS + displacement)
    # -- UNCHANGED from Sprint 4 by explicit instruction. ----
    confirmed = False
    confirming_index: Optional[int] = None
    if state == "LOCATION_REACHED":
        if df_30m is None or len(df_30m) == 0:
            unresolved.append("confirmation: no 30M data provided")
        else:
            swings_30m = scanner._find_swings(df_30m, margin=config.swing_margin_confirmation)
            event_type = None
            event_level = None

            if "BOS" in config.accepted_confirmation_events:
                bos_ok, bos_level = scanner._detect_bos(df_30m, swings_30m, direction_uc, lookback=config.bos_lookback_confirmation)
                if bos_ok:
                    confirming_index = scanner._first_bos_close_index(df_30m, swings_30m, direction_uc, bos_level, lookback=config.bos_lookback_confirmation)
                    event_type, event_level = "BOS", bos_level

            if event_type is None and "CHoCH" in config.accepted_confirmation_events:
                choch_suppress, choch_reason, bearish_lvl, bullish_lvl, choch_idx = scanner._detect_choch(swings_30m, direction_uc)
                choch_ok = (
                    (direction_uc == "LONG" and bullish_lvl is not None and not choch_suppress)
                    or (direction_uc == "SHORT" and bearish_lvl is not None and not choch_suppress)
                )
                if choch_ok:
                    confirming_index = choch_idx if isinstance(choch_idx, int) and choch_idx >= 0 else None
                    event_type = "CHoCH"
                    event_level = bullish_lvl if direction_uc == "LONG" else bearish_lvl

            if event_type is not None and confirming_index is not None:
                displacement = score_displacement(df_30m, direction_lc, index=confirming_index)
                result["displacement_evidence"]["confirmation"] = displacement
                meaningful = _displacement_meets_bar(displacement.get("label"), config.min_confirmation_displacement_label)
                result["confirmation_event"] = {
                    "detected": True,
                    "type": event_type,
                    "level": event_level,
                    "bar_time": scanner._timestamp_at(df_30m, confirming_index),
                    "reason": (
                        f"{event_type} confirmed with {displacement.get('label')} displacement (score {displacement.get('score')})"
                        if meaningful
                        else f"{event_type} technically triggered but displacement was {displacement.get('label')} -- treated as a weak/noisy break, not a real confirmation"
                    ),
                }
                confirmed = meaningful
            elif event_type is not None and confirming_index is None:
                unresolved.append(f"confirmation: {event_type} detected but its triggering candle could not be located within the lookback window")
                result["confirmation_event"] = {"detected": True, "type": event_type, "level": event_level, "bar_time": None, "reason": "triggering candle unresolved -- displacement not scored"}
            else:
                unresolved.append("confirmation: no accepted 30M confirmation event (CHoCH/BOS) detected yet")

    if confirmed:
        state = "CONFIRMED"

    # ---- Step 4: pullback / chase-prevention (Sprint 4.1: execution-area
    # return, OR'd with the existing ATR-retracement-fraction signal). ----
    if confirmed and confirming_index is not None:
        execution_area = _confirmation_execution_area(df_30m, df_5m, confirming_index, direction_lc, config, unresolved)
        result["execution_area"] = execution_area

        post_bars = df_30m.iloc[confirming_index + 1:]
        confirmation_close = float(df_30m["Close"].iloc[confirming_index])
        if len(post_bars) == 0:
            result["pullback_status"] = "not_yet_assessable"
        else:
            extreme = float(post_bars["Close"].max()) if direction_uc == "LONG" else float(post_bars["Close"].min())
            excursion = abs(extreme - confirmation_close)
            current_close = float(df_30m["Close"].iloc[-1])
            retracement = abs(extreme - current_close)
            retracement_fraction = (retracement / excursion) if excursion > 0 else 0.0
            atr_30m = scanner._compute_atr(df_30m) if len(df_30m) >= 15 else 0.0
            chased = (atr_30m > 0) and ((excursion / atr_30m) >= config.chase_max_extension_atr)
            pulled_back_by_fraction = retracement_fraction >= config.pullback_min_retracement_fraction

            returned_to_area = False
            if execution_area is not None:
                returned_to_area = _returned_to_area_after_clearing(post_bars, execution_area["low"], execution_area["high"], direction_uc) or (
                    df_5m is not None and _returned_to_area_after_clearing(df_5m, execution_area["low"], execution_area["high"], direction_uc)
                )

            result["pullback_detail"] = {
                "confirmation_close": confirmation_close,
                "post_confirmation_extreme": extreme,
                "excursion": round(excursion, 4),
                "excursion_atr_multiple": round(excursion / atr_30m, 2) if atr_30m > 0 else None,
                "retracement_fraction": round(retracement_fraction, 3),
                "returned_to_execution_area": returned_to_area,
            }
            if pulled_back_by_fraction or returned_to_area:
                result["pullback_status"] = "observed"
            elif chased:
                result["pullback_status"] = "chased_no_pullback"
            else:
                result["pullback_status"] = "not_yet"

        # state is "CONFIRMED" at this point (set above whenever `confirmed`
        # is True). It advances to WAITING_FOR_PULLBACK once at least one
        # bar has elapsed since the confirming candle -- "not_yet_assessable"
        # (zero bars elapsed, i.e. this IS the confirming candle) is the one
        # case that genuinely stays CONFIRMED.
        if result["pullback_status"] != "not_yet_assessable":
            state = "WAITING_FOR_PULLBACK"

    # ---- Step 5: 5M execution confirmation -- generalized beyond BOS
    # (Sprint 4.1), only after a real pullback. ----
    if state == "WAITING_FOR_PULLBACK" and result["pullback_status"] == "observed":
        if df_5m is None or len(df_5m) == 0:
            unresolved.append("execution: no 5M data provided")
        else:
            swings_5m = scanner._find_swings(df_5m, margin=config.swing_margin_execution)
            event_type, event_level, trigger_index = _execution_trigger(df_5m, swings_5m, direction_uc, config)
            if event_type is not None and trigger_index is not None:
                displacement_exec = None
                meaningful_exec = True
                if event_type in ("BOS", "CHoCH"):
                    displacement_exec = score_displacement(df_5m, direction_lc, index=trigger_index)
                    result["displacement_evidence"]["execution"] = displacement_exec
                    meaningful_exec = _displacement_meets_bar(displacement_exec.get("label"), config.min_execution_displacement_label)
                result["execution_event"] = {
                    "detected": True,
                    "type": event_type,
                    "level": event_level,
                    "bar_time": scanner._timestamp_at(df_5m, trigger_index),
                    "reason": (
                        f"5M {event_type} execution trigger"
                        + (f" with {displacement_exec.get('label')} displacement (score {displacement_exec.get('score')})" if displacement_exec else " (rejection/reclaim -- no displacement score applies)")
                    ),
                }
                state = "EXECUTION_READY"
                if meaningful_exec:
                    state = "ENTRY_READY"
            elif event_type is not None and trigger_index is None:
                unresolved.append(f"execution: 5M {event_type} detected but its triggering candle could not be located")
            else:
                unresolved.append("execution: no accepted 5M execution event (BOS/CHoCH/REJECTION) detected yet")

    # ---- Step 6: invalidation (checked last -- overrides everything above) ----
    invalidation_level: Optional[float] = None
    for source in config.invalidation_source_priority:
        if source == "location_bounds" and location_bounds is not None:
            invalidation_level = location_bounds["low"] if direction_uc == "LONG" else location_bounds["high"]
            break
        if source == "confirmation_level" and result["confirmation_event"]["level"] is not None:
            invalidation_level = result["confirmation_event"]["level"]
            break
    if invalidation_level is None:
        unresolved.append("invalidation: no deterministic anchor level available (no HTF location, no confirmation level)")

    if invalidation_level is not None and current_price is not None:
        invalidated = (current_price < invalidation_level) if direction_uc == "LONG" else (current_price > invalidation_level)
        if invalidated:
            state = "INVALIDATED"
            result["invalidation"] = {
                "invalidated": True,
                "level": invalidation_level,
                "reason": (
                    f"Price {current_price} closed {'below' if direction_uc == 'LONG' else 'above'} "
                    f"the invalidating level {invalidation_level}"
                ),
            }
        else:
            result["invalidation"] = {"invalidated": False, "level": invalidation_level, "reason": None}
    elif invalidation_level is not None:
        result["invalidation"] = {"invalidated": False, "level": invalidation_level, "reason": None}

    result["state"] = state
    result["unresolved"] = unresolved
    result["next_condition"] = _next_condition(state, result)
    return result


def _next_condition(state: str, result: dict[str, Any]) -> dict[str, Optional[str]]:
    """Structured (not prose-only) answer to "what is this setup waiting
    for next" -- Sprint 4.1 explainability requirement."""
    if state == "INVALIDATED":
        return {"waiting_on": "none", "detail": "Terminal -- this setup has been invalidated."}
    if state == "ENTRY_READY":
        return {"waiting_on": "none", "detail": "All required conditions for this strategy version are satisfied."}
    if state == "EXECUTION_READY":
        return {"waiting_on": "execution_displacement", "detail": "A 5M execution event fired, but its displacement quality has not cleared the ENTRY_READY bar yet."}
    if state == "WAITING_FOR_PULLBACK":
        if result["pullback_status"] == "observed":
            return {"waiting_on": "5m_execution", "detail": "A pullback has been observed -- waiting for a 5M BOS/CHoCH/REJECTION execution event."}
        if result["pullback_status"] == "chased_no_pullback":
            return {"waiting_on": "pullback", "detail": "Price expanded away from the confirmation level without retracing -- waiting for a genuine pullback, not chasing."}
        return {"waiting_on": "pullback", "detail": "Confirmed -- waiting for price to pull back toward the confirmation level or its execution area."}
    if state == "CONFIRMED":
        return {"waiting_on": "pullback", "detail": "Just confirmed -- waiting for at least one more bar to assess pullback."}
    if state == "LOCATION_REACHED":
        return {"waiting_on": "30m_confirmation", "detail": "Price is at a valid HTF location -- waiting for a 30M CHoCH/BOS with meaningful displacement."}
    if not result.get("htf_direction_aligned"):
        return {"waiting_on": "htf_direction", "detail": f"4H structure currently reads {result.get('htf_direction')}, not aligned with the requested direction."}
    return {"waiting_on": "htf_location", "detail": "Waiting for a valid HTF location (order block, structural level, or favorable premium/discount) to be reached."}
