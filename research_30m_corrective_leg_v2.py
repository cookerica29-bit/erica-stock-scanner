"""30m Corrective-Leg Anchoring Research -- Phase 2 (2026-09 session).

DEVELOPER-ONLY RESEARCH SCRIPT. Not imported by main.py, candidates_router.py,
or any production module. Makes zero writes to any database table. Reads
real historical 30M bars via scanner._batch_download (the same Alpaca-
backed provider production already uses) and prints/saves developer-only
diagnostic output only.

Phase 1 (commit ed88df8) established that generic swing/BOS/CHoCH
detection is not sufficient because the machine was not reliably
identifying WHICH swing belongs to the active correction. This phase does
NOT start from CHoCH. It reconstructs the corrective leg itself first --
where it began, its pivot sequence, and which pivot is "controlling" (the
level whose break would mean the correction terminated) -- and only
checks break quality AFTER a controlling swing is selected.

CRITICAL DIRECTION RULE (per this task's explicit instruction): direction
is taken from the ALREADY-APPROVED setup direction, passed in as an
argument. No independent direction classifier (_get_trend,
_mtf_shadow_structure_direction, etc.) is ever used to override it. Pivot/
correction PRIMITIVES from scanner.py (_find_swings, _compute_atr) are
reused where their semantics fit; direction is never re-derived from them.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, ".")

import scanner  # noqa: E402
import research_30m_confirmation_audit as phase1  # noqa: E402

# ---------------------------------------------------------------------------
# Tunable research parameters -- ALL explicitly unvalidated placeholders,
# flagged here (not buried in code) exactly per this codebase's own
# convention for unvalidated thresholds (see e.g. RR_WARNING_THRESHOLD's
# comments in candidates_router.py). The stability test (Part 10) exists
# specifically to measure how much the RESULT depends on these choices.
# ---------------------------------------------------------------------------
BASE_PIVOT_MARGIN = 2  # fast/sensitive raw pivot detection; confirmation delay = this many bars
MICRO_ATR_THRESHOLD = 0.5     # min-neighbor-excursion (ATR) below which a pivot is MICRO
STRUCTURAL_ATR_THRESHOLD = 1.5  # at/above which a pivot is STRUCTURAL; between = INTERNAL
MIN_IMPULSE_ATR = 1.0  # minimum ATR excursion for a swing to qualify as "the last directional impulse"


# ---------------------------------------------------------------------------
# Data structures -- research output only, never added to production models.
# ---------------------------------------------------------------------------

@dataclass
class PivotRecord:
    index: int
    timestamp: str
    price: float
    type: str  # "high" | "low"
    confirmed_at_index: int
    confirmed_at_timestamp: str
    confirmation_delay_bars: int
    excursion_prior_atr: Optional[float]
    excursion_next_atr: Optional[float]
    bars_to_prior_neighbor: Optional[int]
    bars_to_next_neighbor: Optional[int]
    significance_score: float
    classification: str  # MICRO | INTERNAL | STRUCTURAL
    can_be_reclassified_by_future_bars: bool  # see Part 6 -- always False by construction here (explained below)


@dataclass
class CorrectionLegResearch:
    thesis_direction: str
    correction_direction: str
    state: str  # NO_IMPULSE_FOUND | NO_CORRECTION | CORRECTION_DEVELOPING | CORRECTION_AMBIGUOUS
    start_timestamp: Optional[str] = None
    start_price: Optional[float] = None
    extreme_timestamp: Optional[str] = None
    extreme_price: Optional[float] = None
    pivots: list = field(default_factory=list)  # PivotRecords inside the correction window
    correction_depth_price: Optional[float] = None
    correction_depth_atr: Optional[float] = None
    correction_depth_pct: Optional[float] = None
    bars_in_correction: Optional[int] = None
    ambiguity_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Pivot detection + significance (Part 4/5) -- margin-independent scoring
# ---------------------------------------------------------------------------

def raw_pivots(df: pd.DataFrame, base_margin: int = BASE_PIVOT_MARGIN) -> list[dict]:
    """Uses the EXISTING, unmodified scanner._find_swings at a small, fast
    margin -- maximal sensitivity, minimal confirmation lag. Every pivot
    _find_swings returns from an already point-in-time-truncated df is, by
    that function's own definition, only recognized once `base_margin`
    bars exist after it WITHIN the truncated data -- so confirmation delay
    is exactly base_margin bars for every pivot here, and no pivot in this
    list could only be known because of bars after the point-in-time
    cutoff (Part 6)."""
    swings = scanner._find_swings(df, margin=base_margin)
    pivots = []
    for s in swings:
        idx = int(s["index"])
        confirmed_idx = idx + base_margin
        pivots.append({
            "index": idx,
            "timestamp": df.index[idx],
            "price": float(s["price"]),
            "type": s["type"],
            "confirmed_at_index": confirmed_idx,
            "confirmed_at_timestamp": df.index[confirmed_idx] if confirmed_idx < len(df) else df.index[-1],
            "confirmation_delay_bars": base_margin,
        })
    return pivots


def score_pivot_significance(pivots: list[dict], atr: float) -> list[dict]:
    """Significance measured INDEPENDENTLY of the discrete detection
    margin, via each pivot's relationship to its immediate opposite-type
    neighbors -- ATR-normalized excursion and bar spacing. This directly
    answers Part 5: pivot classification should not depend on a single
    arbitrary window parameter (Phase 1 found margin=3 vs 4 could flip
    results). Only information already in `pivots` (itself already
    point-in-time-safe) is used -- no future data.

    Fractal nesting / parent-child relationship (also requested in Part 5)
    is approximated here, not implemented as a full recursive tree: a
    pivot's neighbor-excursion score already captures whether it sits
    "inside" a larger nearby swing (small excursion = nested/micro) --
    this is a simplification, disclosed as such in the report, not a
    claim of a complete fractal hierarchy.
    """
    highs = [p for p in pivots if p["type"] == "high"]
    lows = [p for p in pivots if p["type"] == "low"]
    out = []
    for p in pivots:
        opposite = lows if p["type"] == "high" else highs
        prior = [o for o in opposite if o["index"] < p["index"]]
        nxt = [o for o in opposite if o["index"] > p["index"]]
        prior_n = max(prior, key=lambda o: o["index"]) if prior else None
        next_n = min(nxt, key=lambda o: o["index"]) if nxt else None

        exc_prior = abs(p["price"] - prior_n["price"]) / atr if prior_n and atr > 0 else None
        exc_next = abs(p["price"] - next_n["price"]) / atr if next_n and atr > 0 else None
        bars_prior = p["index"] - prior_n["index"] if prior_n else None
        bars_next = next_n["index"] - p["index"] if next_n else None

        # Conservative: a pivot is only as significant as its WEAKER side.
        # A pivot with a big move on one side but almost none on the other
        # is still likely a minor wiggle, not a real structural point.
        candidates = [e for e in (exc_prior, exc_next) if e is not None]
        score = min(candidates) if candidates else 0.0

        if score >= STRUCTURAL_ATR_THRESHOLD:
            classification = "STRUCTURAL"
        elif score >= MICRO_ATR_THRESHOLD:
            classification = "INTERNAL"
        else:
            classification = "MICRO"

        out.append({
            **p,
            "excursion_prior_atr": round(exc_prior, 3) if exc_prior is not None else None,
            "excursion_next_atr": round(exc_next, 3) if exc_next is not None else None,
            "bars_to_prior_neighbor": bars_prior,
            "bars_to_next_neighbor": bars_next,
            "significance_score": round(score, 3),
            "classification": classification,
        })
    return out


def prune_micro_pivots_once(pivots: list[dict], atr: float) -> list[dict]:
    """Method F's hierarchical simplification: remove MICRO pivots once,
    then RE-SCORE the survivors against each other (their neighbor
    relationships change once micro noise is removed -- a pivot that
    looked MICRO against its immediate neighbor might be INTERNAL/
    STRUCTURAL against the next real one out). This is a single
    simplification pass, not full recursion to a fixed point -- disclosed
    explicitly in the report as a scoped simplification, not a claim of
    a complete fractal reduction."""
    survivors = [p for p in pivots if p["classification"] != "MICRO"]
    return score_pivot_significance(
        [{k: v for k, v in p.items() if k not in (
            "excursion_prior_atr", "excursion_next_atr", "bars_to_prior_neighbor",
            "bars_to_next_neighbor", "significance_score", "classification")}
         for p in survivors],
        atr,
    )


# ---------------------------------------------------------------------------
# Correction-leg reconstruction (Part 2/3) -- direction-authoritative, no
# independent trend classifier ever consulted.
# ---------------------------------------------------------------------------

def find_thesis_impulse(pivots: list[dict], thesis_direction: str) -> Optional[tuple[dict, dict]]:
    """"Last directional impulse in thesis direction" -- correction_start
    must be the ACTUAL most recent same-type (high for LONG) confirmed
    pivot, full stop -- NOT "the most recent pivot whose OWN neighbor-
    excursion score clears MIN_IMPULSE_ATR". An earlier bug used the
    latter and could anchor to a stale, lower peak when a later, truly-
    most-recent peak's own immediate-neighbor excursion happened to be
    small (e.g. a choppy top), producing a self-contradictory
    "correction" whose extreme was priced BEYOND its own start --
    caught in production by reconstruct_correction's validity guard
    below, root-caused and fixed here. The MIN_IMPULSE_ATR significance
    gate is now applied to the LEG itself (start->end excursion) in
    reconstruct_correction, not to either endpoint's own neighbor score,
    which is a different, already-used-elsewhere measure."""
    end_type = "high" if thesis_direction == "LONG" else "low"
    start_type = "low" if thesis_direction == "LONG" else "high"
    ends = [p for p in pivots if p["type"] == end_type]
    if not ends:
        return None
    end_pivot = max(ends, key=lambda p: p["index"])  # the genuine most recent confirmed same-type pivot
    starts = [p for p in pivots if p["type"] == start_type and p["index"] < end_pivot["index"]]
    if not starts:
        return None
    start_pivot = max(starts, key=lambda p: p["index"])  # the swing immediately preceding it
    return start_pivot, end_pivot


def reconstruct_correction(df: pd.DataFrame, pivots: list[dict], thesis_direction: str, atr: float) -> CorrectionLegResearch:
    correction_direction = "bearish" if thesis_direction == "LONG" else "bullish"
    impulse = find_thesis_impulse(pivots, thesis_direction)
    if impulse is None:
        return CorrectionLegResearch(
            thesis_direction=thesis_direction, correction_direction=correction_direction,
            state="NO_IMPULSE_FOUND",
            ambiguity_note="No opposing-type pivot pair exists to anchor an impulse in the thesis direction.",
        )
    impulse_start, impulse_end = impulse  # impulse_end = correction's candidate start (the peak/trough)

    # Significance gate on the LEG itself (start->end excursion), not on
    # either endpoint's own neighbor-based score -- see find_thesis_impulse's
    # docstring for why those are different measures and conflating them
    # was the root cause of an earlier bug.
    leg_excursion_atr = abs(impulse_end["price"] - impulse_start["price"]) / atr if atr > 0 else 0.0
    if leg_excursion_atr < MIN_IMPULSE_ATR:
        return CorrectionLegResearch(
            thesis_direction=thesis_direction, correction_direction=correction_direction,
            state="NO_IMPULSE_FOUND",
            ambiguity_note=(
                f"Most recent same-type pivot pair only spans {leg_excursion_atr:.3f} ATR, "
                f"below MIN_IMPULSE_ATR={MIN_IMPULSE_ATR} -- too small to call a real directional impulse."
            ),
        )

    correction_extreme_type = "low" if thesis_direction == "LONG" else "high"
    pool = [p for p in pivots if p["index"] > impulse_end["index"]]
    extremes = [p for p in pool if p["type"] == correction_extreme_type]
    if not extremes:
        return CorrectionLegResearch(
            thesis_direction=thesis_direction, correction_direction=correction_direction,
            state="NO_CORRECTION",
            start_timestamp=str(impulse_end["timestamp"]), start_price=impulse_end["price"],
            ambiguity_note="A thesis-direction impulse was found, but no opposing pivot has formed since.",
        )

    is_new_extreme = (lambda price, best: price < best) if thesis_direction == "LONG" else (lambda price, best: price > best)
    extreme_pivot = extremes[0]
    for e in extremes[1:]:
        if is_new_extreme(e["price"], extreme_pivot["price"]):
            extreme_pivot = e

    # Validity guard: for a real correction, the extreme must be WORSE
    # than the start in the correction direction (LONG: extreme < start).
    # Defensive check kept even after fixing find_thesis_impulse's root
    # cause above -- reports a clear, honest failure instead of silently
    # returning a self-contradictory correction if this is ever violated
    # by some other path.
    valid_side = (extreme_pivot["price"] < impulse_end["price"]) if thesis_direction == "LONG" else (extreme_pivot["price"] > impulse_end["price"])
    if not valid_side:
        return CorrectionLegResearch(
            thesis_direction=thesis_direction, correction_direction=correction_direction,
            state="INVALID_RECONSTRUCTION",
            start_timestamp=str(impulse_end["timestamp"]), start_price=impulse_end["price"],
            extreme_timestamp=str(extreme_pivot["timestamp"]), extreme_price=extreme_pivot["price"],
            ambiguity_note=(
                f"Selected extreme ({extreme_pivot['price']}) is on the wrong side of the selected "
                f"start ({impulse_end['price']}) for a {correction_direction} correction -- "
                "reconstruction rejected rather than reported as a false signal."
            ),
        )

    depth_price = abs(impulse_end["price"] - extreme_pivot["price"])
    correction = CorrectionLegResearch(
        thesis_direction=thesis_direction, correction_direction=correction_direction,
        state="CORRECTION_DEVELOPING",
        start_timestamp=str(impulse_end["timestamp"]), start_price=impulse_end["price"],
        extreme_timestamp=str(extreme_pivot["timestamp"]), extreme_price=extreme_pivot["price"],
        pivots=[p for p in pool if p["index"] <= extreme_pivot["index"]] + [
            p for p in pool if p["index"] > extreme_pivot["index"]
        ],
        correction_depth_price=round(depth_price, 4),
        correction_depth_atr=round(depth_price / atr, 3) if atr > 0 else None,
        correction_depth_pct=round(depth_price / impulse_end["price"] * 100, 3) if impulse_end["price"] else None,
        bars_in_correction=int(len(df) - 1 - impulse_end["index"]),
    )
    # Ambiguity signal: more than one extreme candidate within a small
    # ATR band of the winner -- report rather than silently pick one.
    close_rivals = [e for e in extremes if e is not extreme_pivot and abs(e["price"] - extreme_pivot["price"]) / atr < 0.3] if atr > 0 else []
    if close_rivals:
        correction.state = "CORRECTION_AMBIGUOUS"
        correction.ambiguity_note = f"{len(close_rivals)} other candidate extreme(s) within 0.3 ATR of the selected one."
    return correction


# ---------------------------------------------------------------------------
# Controlling-swing candidate methods A-F (Part 4)
# ---------------------------------------------------------------------------

def _opposing_pivots_in_correction(correction: CorrectionLegResearch, thesis_direction: str) -> list[dict]:
    opposing_type = "high" if thesis_direction == "LONG" else "low"
    return [p for p in correction.pivots if p["type"] == opposing_type]


def method_A(correction, thesis_direction, atr) -> Optional[dict]:
    """Most recent confirmed opposing pivot BEFORE the correction's extreme."""
    before_extreme_idx = next((p["index"] for p in correction.pivots if str(p["timestamp"]) == correction.extreme_timestamp), None)
    candidates = [p for p in _opposing_pivots_in_correction(correction, thesis_direction)
                  if before_extreme_idx is None or p["index"] < before_extreme_idx]
    return max(candidates, key=lambda p: p["index"]) if candidates else None


def method_B(correction, thesis_direction, atr) -> Optional[dict]:
    """Last opposing pivot whose subsequent break produced a NEW correction extreme."""
    extreme_type = "low" if thesis_direction == "LONG" else "high"
    opposing = sorted(_opposing_pivots_in_correction(correction, thesis_direction), key=lambda p: p["index"])
    extremes = sorted([p for p in correction.pivots if p["type"] == extreme_type], key=lambda p: p["index"])
    is_new_extreme = (lambda price, best: price < best) if thesis_direction == "LONG" else (lambda price, best: price > best)

    running_best = correction.start_price
    causal = []
    for opp in opposing:
        following = [e for e in extremes if e["index"] > opp["index"]]
        if not following:
            continue
        nxt_extreme = min(following, key=lambda e: e["index"])
        if is_new_extreme(nxt_extreme["price"], running_best):
            causal.append(opp)
            running_best = nxt_extreme["price"]
    return causal[-1] if causal else None


def method_C(correction, thesis_direction, atr) -> Optional[dict]:
    """Highest (LONG) / lowest (SHORT) internal opposing pivot inside the FINAL downswing --
    the leg from the second-most-recent extreme (or correction start, if only one leg) to the current extreme."""
    extreme_type = "low" if thesis_direction == "LONG" else "high"
    extremes = sorted([p for p in correction.pivots if p["type"] == extreme_type], key=lambda p: p["index"])
    if len(extremes) >= 2:
        leg_start_idx = extremes[-2]["index"]
    else:
        leg_start_idx = None  # whole correction is the "final downswing"
    candidates = [p for p in _opposing_pivots_in_correction(correction, thesis_direction)
                  if leg_start_idx is None or p["index"] > leg_start_idx]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p["price"]) if thesis_direction == "LONG" else min(candidates, key=lambda p: p["price"])


def method_D(correction, thesis_direction, atr, df) -> Optional[dict]:
    """Last STRUCTURAL opposing pivot that remains unbroken as of cutoff."""
    candidates = [p for p in _opposing_pivots_in_correction(correction, thesis_direction) if p["classification"] == "STRUCTURAL"]
    candidates = sorted(candidates, key=lambda p: -p["index"])
    for c in candidates:
        broken = _is_broken_by_cutoff(df, c["price"], thesis_direction, c["confirmed_at_index"])
        if not broken["close_through"]:
            return c
    return None


def method_E(correction, thesis_direction, atr) -> Optional[dict]:
    """Most recent opposing pivot classified INTERNAL or STRUCTURAL (not MICRO) --
    "belongs to the most recent complete impulse", not the nearest micro wiggle."""
    candidates = [p for p in _opposing_pivots_in_correction(correction, thesis_direction) if p["classification"] != "MICRO"]
    return max(candidates, key=lambda p: p["index"]) if candidates else None


def method_F(correction, thesis_direction, atr) -> Optional[dict]:
    """Hierarchical: prune MICRO pivots once, re-score survivors against
    each other, take the highest/lowest surviving opposing pivot before
    the extreme (mirrors C's "final downswing" framing but on the
    PRUNED/re-scored sequence rather than the raw one)."""
    pruned = prune_micro_pivots_once(correction.pivots, atr)
    opposing_type = "high" if thesis_direction == "LONG" else "low"
    extreme_type = "low" if thesis_direction == "LONG" else "high"
    candidates = [p for p in pruned if p["type"] == opposing_type]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p["price"]) if thesis_direction == "LONG" else min(candidates, key=lambda p: p["price"])


METHODS = {"A": method_A, "B": method_B, "C": method_C, "E": method_E, "F": method_F}
# D takes an extra `df` argument (it needs to check break state per-candidate) -- handled separately below.


# ---------------------------------------------------------------------------
# Break checking -- ONLY run after a controlling swing is already selected (Part 11)
# ---------------------------------------------------------------------------

def _is_broken_by_cutoff(df: pd.DataFrame, level: float, thesis_direction: str, from_index: int) -> dict:
    """thesis_direction=LONG: broken means CLOSE > level (bullish reclaim
    of a bearish corrective structure). SHORT mirrors: CLOSE < level."""
    segment = df.iloc[from_index:]
    if segment.empty:
        return {"close_through": False, "wick_through": False}
    if thesis_direction == "LONG":
        close_hits = segment[segment["Close"] > level]
        wick_hits = segment[segment["High"] > level]
    else:
        close_hits = segment[segment["Close"] < level]
        wick_hits = segment[segment["Low"] < level]
    result = {"close_through": not close_hits.empty, "wick_through": not wick_hits.empty}
    if not close_hits.empty:
        idx = df.index.get_loc(close_hits.index[0])
        bar = df.iloc[idx]
        result["break_index"] = int(idx)
        result["break_timestamp"] = str(df.index[idx])
        result["break_open"] = round(float(bar["Open"]), 4)
        result["break_high"] = round(float(bar["High"]), 4)
        result["break_low"] = round(float(bar["Low"]), 4)
        result["break_close"] = round(float(bar["Close"]), 4)
    return result


def inspect_break(df: pd.DataFrame, level: float, thesis_direction: str, from_index: int, atr: float) -> dict:
    info = _is_broken_by_cutoff(df, level, thesis_direction, from_index)
    if info.get("close_through") and atr > 0:
        dist = abs(info["break_close"] - level) / atr
        info["distance_through_atr"] = round(dist, 3)
        body = abs(info["break_close"] - info["break_open"])
        rng = info["break_high"] - info["break_low"]
        info["body_range_ratio"] = round(body / rng, 3) if rng > 0 else None
        info["close_location_in_range"] = (
            round((info["break_close"] - info["break_low"]) / rng, 3) if rng > 0 else None
        )
    return info


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_for_ticker(ticker: str, direction: str, cutoff_iso: str, base_margin: int = BASE_PIVOT_MARGIN) -> dict:
    thesis_direction = "LONG" if direction.lower() == "long" else "SHORT"
    cutoff = phase1._to_utc(cutoff_iso)
    raw_30m = phase1.fetch_bars(ticker, interval="30m", period="60d")
    if raw_30m.empty:
        return {"ticker": ticker, "error": "no 30m data returned"}
    df = phase1.closed_candles_only(phase1.truncate_point_in_time(raw_30m, cutoff), cutoff, bar_minutes=30)
    if len(df) < 30:
        return {"ticker": ticker, "error": "insufficient point-in-time bars", "bars": len(df)}

    atr = scanner._compute_atr(df, period=14)
    pivots = score_pivot_significance(raw_pivots(df, base_margin), atr)
    correction = reconstruct_correction(df, pivots, thesis_direction, atr)

    result = {
        "ticker": ticker,
        "cutoff": cutoff_iso,
        "bars_point_in_time": len(df),
        "last_bar": str(df.index[-1]),
        "atr": round(atr, 4),
        "correction": asdict(correction),
        "methods": {},
    }

    if correction.state in ("CORRECTION_DEVELOPING", "CORRECTION_AMBIGUOUS"):
        for name, fn in METHODS.items():
            pivot = fn(correction, thesis_direction, atr)
            entry = {"selected_pivot": pivot}
            if pivot is not None:
                entry["break"] = inspect_break(df, pivot["price"], thesis_direction, pivot["confirmed_at_index"], atr)
            result["methods"][name] = entry
        d_pivot = method_D(correction, thesis_direction, atr, df)
        d_entry = {"selected_pivot": d_pivot}
        if d_pivot is not None:
            d_entry["break"] = inspect_break(df, d_pivot["price"], thesis_direction, d_pivot["confirmed_at_index"], atr)
        result["methods"]["D"] = d_entry
    return result


def chronological_debug_table(df: pd.DataFrame, pivots: list[dict], correction: CorrectionLegResearch, methods: dict) -> str:
    """Part 8: developer-only chronological table for visual inspection."""
    pivot_by_index = {p["index"]: p for p in pivots}
    controlling_by_index = {}
    for name, entry in methods.items():
        p = entry.get("selected_pivot")
        if p:
            controlling_by_index.setdefault(p["index"], []).append(name)
    lines = [f"{'idx':>4} {'timestamp':<20} {'O':>8} {'H':>8} {'L':>8} {'C':>8}  {'pivot':<6} {'sig':<10} {'controlling_for':<20}"]
    start = None
    if correction.start_timestamp:
        try:
            start = df.index.get_loc(pd.Timestamp(correction.start_timestamp))
        except KeyError:
            start = None
    row_start = max(0, (start or 0) - 3)
    for i in range(row_start, len(df)):
        bar = df.iloc[i]
        piv = pivot_by_index.get(i)
        pivot_str = f"{piv['type'][0].upper()}" if piv else ""
        sig_str = piv["classification"] if piv else ""
        controlling = ",".join(controlling_by_index.get(i, []))
        lines.append(
            f"{i:>4} {str(df.index[i]):<20} {bar['Open']:>8.2f} {bar['High']:>8.2f} {bar['Low']:>8.2f} {bar['Close']:>8.2f}  "
            f"{pivot_str:<6} {sig_str:<10} {controlling:<20}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ANCHORS = [
        ("NVDA", "long", "2026-09-01T01:13:30.652359+00:00"),
        ("FFIV", "long", "2026-09-01T01:08:53.668785+00:00"),
        ("CLH", "long", "2026-09-01T00:43:10.213229+00:00"),
    ]
    out = []
    for ticker, direction, cutoff in ANCHORS:
        per_margin = {}
        for m in (1, 2, 3):
            print(f"--- {ticker} margin={m} ---", file=sys.stderr, flush=True)
            per_margin[m] = run_for_ticker(ticker, direction, cutoff, base_margin=m)
        out.append({"ticker": ticker, "per_margin": per_margin})
    print(json.dumps(out, indent=2, default=str))
