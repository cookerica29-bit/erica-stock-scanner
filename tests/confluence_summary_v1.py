"""confluence_summary.summarize_confluence -- pure combination logic.

Scope: the module's own math (per-signal classification, counting against
applicable signals only, label bucketing). Router wiring is covered
separately in tests/candidates_router_confluence_v1.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confluence_summary import (
    CONFLICTED_UNFAVORABLE_MIN,
    LABEL_CONFLICTED,
    LABEL_LIMITED,
    LABEL_SOME,
    LABEL_STRONG,
    SOME_FAVORABLE_RATIO,
    STRONG_FAVORABLE_RATIO,
    summarize_confluence,
)


def _all_favorable(**overrides):
    base = dict(
        bos_confirmed=True,
        displacement_read="favorable",
        sweep_confirmed=True,
        rejection_confirmed=True,
        macro_conflict=False,
        choch_conflict=False,
        location_alignment="favorable",
        risk_reward=2.5,
        rr_warning=False,
        no_valid_target=False,
    )
    base.update(overrides)
    return base


def _all_neutral(**overrides):
    # risk_reward has NO neutral state by design (favorable|unfavorable
    # only -- it's always determinate, see module docstring) -- so a
    # "neutral" baseline sets it to its unfavorable state here, same as
    # every real no_valid_target promotion does. Tests that need a true
    # all-signals-uninformative case account for this being 6 neutral + 1
    # unfavorable, not 7 neutral.
    base = dict(
        bos_confirmed=False,
        displacement_read="quiet",
        sweep_confirmed=False,
        rejection_confirmed=False,
        macro_conflict=False,
        choch_conflict=False,
        location_alignment="neutral",
        risk_reward=None,
        rr_warning=True,
        no_valid_target=True,
    )
    base.update(overrides)
    return base


# -- per-signal classification -------------------------------------------------

def test_bos_confirmed_is_favorable_not_confirmed_is_neutral_never_unfavorable():
    result_true = summarize_confluence(**_all_neutral(bos_confirmed=True))
    result_false = summarize_confluence(**_all_neutral(bos_confirmed=False))
    assert result_true["confluence_signals"]["bos"] == "favorable"
    assert result_false["confluence_signals"]["bos"] == "neutral"


def test_displacement_maps_all_three_reads():
    fav = summarize_confluence(**_all_neutral(displacement_read="favorable"))
    adv = summarize_confluence(**_all_neutral(displacement_read="adverse"))
    quiet = summarize_confluence(**_all_neutral(displacement_read="quiet"))
    none_read = summarize_confluence(**_all_neutral(displacement_read=None))
    assert fav["confluence_signals"]["displacement"] == "favorable"
    assert adv["confluence_signals"]["displacement"] == "unfavorable"
    assert quiet["confluence_signals"]["displacement"] == "neutral"
    assert none_read["confluence_signals"]["displacement"] == "neutral"


def test_sweep_and_rejection_are_favorable_or_neutral_never_unfavorable():
    result = summarize_confluence(**_all_neutral(sweep_confirmed=True, rejection_confirmed=True))
    assert result["confluence_signals"]["sweep"] == "favorable"
    assert result["confluence_signals"]["rejection"] == "favorable"
    result2 = summarize_confluence(**_all_neutral(sweep_confirmed=False, rejection_confirmed=False))
    assert result2["confluence_signals"]["sweep"] == "neutral"
    assert result2["confluence_signals"]["rejection"] == "neutral"


def test_macro_choch_is_unfavorable_or_neutral_never_favorable():
    neither = summarize_confluence(**_all_neutral(macro_conflict=False, choch_conflict=False))
    macro_only = summarize_confluence(**_all_neutral(macro_conflict=True, choch_conflict=False))
    choch_only = summarize_confluence(**_all_neutral(macro_conflict=False, choch_conflict=True))
    both = summarize_confluence(**_all_neutral(macro_conflict=True, choch_conflict=True))
    assert neither["confluence_signals"]["macro_choch"] == "neutral"
    assert macro_only["confluence_signals"]["macro_choch"] == "unfavorable"
    assert choch_only["confluence_signals"]["macro_choch"] == "unfavorable"
    assert both["confluence_signals"]["macro_choch"] == "unfavorable"


def test_location_maps_all_three_reads_and_none():
    fav = summarize_confluence(**_all_neutral(location_alignment="favorable"))
    unfav = summarize_confluence(**_all_neutral(location_alignment="unfavorable"))
    neutral = summarize_confluence(**_all_neutral(location_alignment="neutral"))
    none_align = summarize_confluence(**_all_neutral(location_alignment=None))
    assert fav["confluence_signals"]["location"] == "favorable"
    assert unfav["confluence_signals"]["location"] == "unfavorable"
    assert neutral["confluence_signals"]["location"] == "neutral"
    assert none_align["confluence_signals"]["location"] is None


def test_risk_reward_is_favorable_or_unfavorable_never_neutral():
    good = summarize_confluence(**_all_neutral(risk_reward=2.0, rr_warning=False, no_valid_target=False))
    warned = summarize_confluence(**_all_neutral(risk_reward=1.0, rr_warning=True, no_valid_target=False))
    missing = summarize_confluence(**_all_neutral(risk_reward=None, rr_warning=True, no_valid_target=True))
    assert good["confluence_signals"]["risk_reward"] == "favorable"
    assert warned["confluence_signals"]["risk_reward"] == "unfavorable"
    assert missing["confluence_signals"]["risk_reward"] == "unfavorable"


# -- counting: applicable, not hardcoded 7 -------------------------------------

def test_best_possible_case_is_six_favorable_one_neutral():
    """macro_choch has no favorable state by design (it only ever flags
    conflicts, never confirms alignment) -- so the actual best case tops
    out at 6 favorable + 1 neutral, never 7 favorable. Confirms the
    asymmetry is real, not accidentally achievable."""
    result = summarize_confluence(**_all_favorable())
    assert result["confluence_counts"] == {"favorable": 6, "unfavorable": 0, "neutral": 1, "applicable": 7}
    assert result["confluence_signals"]["macro_choch"] == "neutral"
    assert result["confluence_label"] == LABEL_STRONG


def test_counts_drop_to_six_applicable_when_location_is_none():
    """A missing location signal must not silently penalize every label --
    the denominator itself shrinks, not just the numerator."""
    result = summarize_confluence(**_all_favorable(location_alignment=None))
    assert result["confluence_counts"]["applicable"] == 6
    assert result["confluence_counts"]["favorable"] == 5
    assert result["confluence_counts"]["unfavorable"] == 0
    # 5/6 favorable, 0 unfavorable -- still strong confluence despite the
    # missing signal, because the ratio is computed against 6, not 7.
    assert result["confluence_label"] == LABEL_STRONG


def test_neutral_signals_are_counted_but_not_favorable_or_unfavorable():
    result = summarize_confluence(**_all_neutral())
    assert result["confluence_counts"] == {"favorable": 0, "unfavorable": 1, "neutral": 6, "applicable": 7}


# -- label bucketing ------------------------------------------------------------

def test_all_favorable_is_strong_confluence():
    result = summarize_confluence(**_all_favorable())
    assert result["confluence_label"] == LABEL_STRONG


def test_all_neutral_is_limited_confluence():
    result = summarize_confluence(**_all_neutral())
    assert result["confluence_label"] == LABEL_LIMITED


def test_two_unfavorable_signals_forces_conflicted_even_with_high_favorable_count():
    """The explicit design point: real red flags can't be diluted into a
    falsely reassuring label just because several other signals are
    favorable."""
    result = summarize_confluence(
        bos_confirmed=True, displacement_read="adverse", sweep_confirmed=True,
        rejection_confirmed=True, macro_conflict=True, choch_conflict=False,
        location_alignment="favorable", risk_reward=3.0, rr_warning=False, no_valid_target=False,
    )
    assert result["confluence_counts"]["unfavorable"] == 2
    assert result["confluence_counts"]["favorable"] == 5
    assert result["confluence_label"] == LABEL_CONFLICTED


def test_one_unfavorable_signal_does_not_force_conflicted():
    result = summarize_confluence(
        bos_confirmed=True, displacement_read="adverse", sweep_confirmed=True,
        rejection_confirmed=True, macro_conflict=False, choch_conflict=False,
        location_alignment="favorable", risk_reward=3.0, rr_warning=False, no_valid_target=False,
    )
    assert result["confluence_counts"]["unfavorable"] == 1
    assert result["confluence_label"] != LABEL_CONFLICTED


def test_strong_confluence_requires_zero_unfavorable():
    """Even a single unfavorable signal disqualifies "strong," regardless
    of how high the favorable ratio is otherwise."""
    result = summarize_confluence(
        bos_confirmed=True, displacement_read="adverse", sweep_confirmed=True,
        rejection_confirmed=True, macro_conflict=False, choch_conflict=False,
        location_alignment="favorable", risk_reward=3.0, rr_warning=False, no_valid_target=False,
    )
    assert result["confluence_counts"]["favorable"] == 5
    assert result["confluence_counts"]["unfavorable"] == 1
    assert result["confluence_label"] != LABEL_STRONG


def test_ratio_boundaries_match_module_constants():
    # 4 of 7 favorable = 0.571 -> below STRONG_FAVORABLE_RATIO (0.6), at/above SOME (0.4)
    four_of_seven = summarize_confluence(
        bos_confirmed=True, displacement_read="favorable", sweep_confirmed=True,
        rejection_confirmed=True, macro_conflict=False, choch_conflict=False,
        location_alignment="neutral", risk_reward=None, rr_warning=True, no_valid_target=True,
    )
    assert four_of_seven["confluence_counts"]["favorable"] == 4
    assert four_of_seven["confluence_counts"]["unfavorable"] == 1
    assert four_of_seven["confluence_label"] == LABEL_SOME

    # 2 of 7 favorable = 0.286 -> below SOME_FAVORABLE_RATIO (0.4)
    two_of_seven = summarize_confluence(
        bos_confirmed=True, displacement_read="favorable", sweep_confirmed=False,
        rejection_confirmed=False, macro_conflict=False, choch_conflict=False,
        location_alignment="neutral", risk_reward=None, rr_warning=True, no_valid_target=True,
    )
    assert two_of_seven["confluence_counts"]["favorable"] == 2
    assert two_of_seven["confluence_counts"]["unfavorable"] == 1
    assert two_of_seven["confluence_label"] == LABEL_LIMITED


def test_module_constants_are_the_documented_placeholder_values():
    """Pins the exact agreed-upon placeholder thresholds so a future
    accidental edit doesn't silently change behavior without review."""
    assert CONFLICTED_UNFAVORABLE_MIN == 2
    assert STRONG_FAVORABLE_RATIO == 0.6
    assert SOME_FAVORABLE_RATIO == 0.4
