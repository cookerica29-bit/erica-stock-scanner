"""Kairos-native signal confluence summary.

This is deliberately NOT a port of scanner._build_trade_stage_eval's A+
READY / B+ TRADEABLE tier logic. That function turned out to depend on
several pieces Kairos never ported (scanner._cleanliness_read's efficiency
ratio, scanner._room_to_target's separate R:R path with its own stricter
2.0 threshold, standalone in_ob/near_ob tracking) and on the legacy hard-
threshold detect_displacement, not score_displacement's continuous
replacement -- reconstructing it faithfully would mean inheriting two more
unvalidated thresholds just to earn a false sense of legitimacy from being
"the real algorithm," when legacy's own tier logic isn't even internally
consistent about which displacement function it trusts. See the session
discussion before this file was written.

Instead: a flat, equal-weighted count of how many of the 7 already-shipped
overlay signals read favorable/unfavorable/neutral for this candidate's own
direction. Equal weighting is a deliberate choice, not a default -- a
weighted scheme would mean inventing one new unvalidated number PER SIGNAL
(how much more should BOS matter than location?), which is strictly more
guessing than a flat count, not less. Every field taken alone earned its
place today by being either a raw continuous value or a clean boolean,
specifically to avoid injecting unvalidated magnitude judgments; a
weighted synthesis would be the single largest such judgment introduced
all session. If the outcome-tracking system eventually shows some signals
actually predict better than others, that's the evidence that should
justify unequal weighting later -- not a guess made today.

Per-signal reads are deliberately asymmetric, not forced into a false
favorable/unfavorable/neutral symmetry: BOS/sweep/rejection can only ever
confirm FOR a direction (never against it, by their own detection design),
so they have no "unfavorable" state. macro/CHoCH only ever flag conflicts
(never confirm alignment), so they have no "favorable" state. Location and
displacement are the only two signals with a real three-way read, because
they're the only two computed as genuinely bidirectional in the first
place.

Named "confluence" (not "grade," "tier," "A+," or "B+") specifically so
nothing here implies continuity with legacy's severity or its exact
labels. Purely descriptive: never referenced by no_valid_target,
rr_warning, or anything that affects promotion eligibility -- same rule as
every other overlay shipped today.
"""

from __future__ import annotations

from typing import Any, Optional

# Label-bucketing thresholds -- explicitly unvalidated guesses, same
# category as displacement_score's STRONG_LABEL_MIN_SCORE/weights,
# location_score's PREMIUM_THRESHOLD/DISCOUNT_THRESHOLD, and
# outcome_resolver's DEFAULT_MAX_TRACKING_DAYS. Not evidence, not tuned
# against real outcomes -- confluence_counts (the raw numbers) is the
# authoritative field; confluence_label is display sugar on top of it,
# same relationship as displacement_label/location_label to their own raw
# values. Deliberately left as-shipped rather than hand-tuned further --
# now that outcome tracking exists, adjusting these without real resolved
# evidence would just be substituting one guess for another.
CONFLICTED_UNFAVORABLE_MIN = 2
STRONG_FAVORABLE_RATIO = 0.6
SOME_FAVORABLE_RATIO = 0.4

LABEL_CONFLICTED = "conflicted"
LABEL_STRONG = "strong confluence"
LABEL_SOME = "some confluence"
LABEL_LIMITED = "limited confluence"


def summarize_confluence(
    bos_confirmed: bool,
    displacement_read: Optional[str],
    sweep_confirmed: bool,
    rejection_confirmed: bool,
    macro_conflict: bool,
    choch_conflict: bool,
    location_alignment: Optional[str],
    risk_reward: Optional[float],
    rr_warning: bool,
    no_valid_target: bool,
) -> dict[str, Any]:
    """Combine the 7 already-computed overlay signals into a transparent,
    equal-weighted summary. Every argument here is already direction-
    resolved by its own source (e.g. bos_confirmed means "confirmed in
    this candidate's own direction," location_alignment is already
    favorable/unfavorable/neutral relative to direction) -- this function
    does no direction math of its own, purely classification + counting.

    Returns {confluence_signals, confluence_counts, confluence_label}.
    confluence_signals is the full per-signal breakdown (location is None,
    not "neutral", when there's no valid swing range to read at all --
    excluded from the counts below rather than penalizing a candidate for
    a signal that simply isn't available). confluence_counts is the raw,
    authoritative tally, computed against however many signals were
    actually applicable (not a hardcoded 7) so a missing signal never
    silently makes every label harder to reach. confluence_label is the
    single glance-scannable word, explicitly flagged placeholder
    thresholds and all.
    """
    signals: dict[str, Optional[str]] = {
        "bos": "favorable" if bos_confirmed else "neutral",
        "displacement": (
            "favorable" if displacement_read == "favorable"
            else "unfavorable" if displacement_read == "adverse"
            else "neutral"
        ),
        "sweep": "favorable" if sweep_confirmed else "neutral",
        "rejection": "favorable" if rejection_confirmed else "neutral",
        "macro_choch": "unfavorable" if (macro_conflict or choch_conflict) else "neutral",
        "location": (
            "favorable" if location_alignment == "favorable"
            else "unfavorable" if location_alignment == "unfavorable"
            else "neutral" if location_alignment == "neutral"
            else None
        ),
        "risk_reward": "unfavorable" if (no_valid_target or rr_warning) else "favorable",
    }

    applicable = {name: read for name, read in signals.items() if read is not None}
    favorable_count = sum(1 for read in applicable.values() if read == "favorable")
    unfavorable_count = sum(1 for read in applicable.values() if read == "unfavorable")
    neutral_count = sum(1 for read in applicable.values() if read == "neutral")
    applicable_count = len(applicable)

    counts = {
        "favorable": favorable_count,
        "unfavorable": unfavorable_count,
        "neutral": neutral_count,
        "applicable": applicable_count,
    }

    favorable_ratio = (favorable_count / applicable_count) if applicable_count else 0.0
    if unfavorable_count >= CONFLICTED_UNFAVORABLE_MIN:
        # Real red flags take priority over the ratio -- otherwise two or
        # more genuine conflicts could get diluted into a falsely
        # reassuring "strong confluence" just because several other
        # signals happened to be favorable too.
        label = LABEL_CONFLICTED
    elif unfavorable_count == 0 and favorable_ratio >= STRONG_FAVORABLE_RATIO:
        label = LABEL_STRONG
    elif favorable_ratio >= SOME_FAVORABLE_RATIO:
        label = LABEL_SOME
    else:
        label = LABEL_LIMITED

    return {
        "confluence_signals": signals,
        "confluence_counts": counts,
        "confluence_label": label,
    }
