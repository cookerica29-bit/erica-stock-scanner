"""entry_status_label -- legacy's four-tier ATR-distance bucket.

Purely descriptive, ported verbatim from scanner._stock_entry_status's
cutoffs (0.25 / 0.50 / 1.00 ATR). Deliberately NOT reconciled with the
existing, actually-gating entry_proximity_ok (a different, already-tuned
single threshold) -- see ENTRY_STATUS_TRADEABLE_MAX_ATR's comment in
candidates_router.py for why the two independently-derived fields can and
will disagree on real candidates. This file includes a concrete
disagreement case (entry_proximity_ok=True while entry_status_label reads
"Near Entry", not "Tradeable"), demonstrated with real numbers, not just
asserted -- see also the real-ticker verification in this feature's
commit message.

No DB involvement -- entry_distance/entry_proximity_ok/entry_status_label
are all ephemeral, computed fresh per request from a live quote
(_attach_entry_proximity), never persisted. Nothing to migrate, nothing to
round-trip.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router


# -- pure _entry_status_label boundary tests -----------------------------------

def test_none_distance_is_waiting():
    assert router._entry_status_label(None) == "Waiting"


def test_at_tradeable_boundary_is_tradeable():
    assert router._entry_status_label(0.25) == "Tradeable"


def test_just_past_tradeable_boundary_is_near_entry():
    assert router._entry_status_label(0.2501) == "Near Entry"


def test_at_near_entry_boundary_is_near_entry():
    assert router._entry_status_label(0.50) == "Near Entry"


def test_just_past_near_entry_boundary_is_waiting():
    assert router._entry_status_label(0.5001) == "Waiting"


def test_at_waiting_boundary_is_waiting():
    assert router._entry_status_label(1.00) == "Waiting"


def test_just_past_waiting_boundary_is_too_far():
    assert router._entry_status_label(1.0001) == "Too Far"


def test_zero_distance_is_tradeable():
    assert router._entry_status_label(0.0) == "Tradeable"


# -- _entry_proximity wiring ----------------------------------------------------

def test_no_quote_defaults_to_waiting():
    result = router._entry_proximity(entry_price=100.0, atr14=10.0, quote=None)
    assert result["entry_status_label"] == "Waiting"
    assert result["entry_distance_atr"] is None


def test_one_sided_quote_defaults_to_waiting():
    quote = {"price": 100.0, "price_branch": "bid_only", "timestamp": "2026-08-01T00:00:00Z", "source": "alpaca"}
    result = router._entry_proximity(entry_price=100.0, atr14=10.0, quote=quote)
    assert result["entry_status_label"] == "Waiting"
    assert result["entry_proximity_ok"] is False


def test_close_price_is_tradeable_and_proximity_ok_agree():
    """The common case: both fields agree when price is very close to entry."""
    quote = {"price": 101.0, "price_branch": "two_sided", "timestamp": "2026-08-01T00:00:00Z", "source": "alpaca"}
    result = router._entry_proximity(entry_price=100.0, atr14=10.0, quote=quote)
    assert result["entry_distance_atr"] == 0.1
    assert result["entry_status_label"] == "Tradeable"
    assert result["entry_proximity_ok"] is True


def test_far_price_is_too_far_and_proximity_not_ok_agree():
    """Both fields also agree when price has moved well past either threshold."""
    quote = {"price": 115.0, "price_branch": "two_sided", "timestamp": "2026-08-01T00:00:00Z", "source": "alpaca"}
    result = router._entry_proximity(entry_price=100.0, atr14=10.0, quote=quote)
    assert result["entry_distance_atr"] == 1.5
    assert result["entry_status_label"] == "Too Far"
    assert result["entry_proximity_ok"] is False


def test_entry_proximity_ok_true_while_entry_status_label_reads_near_entry():
    """The explicit disagreement case: entry=100, atr=10, price=104 ->
    distance=4.0. entry_status_label buckets on distance_atr=0.4 alone ->
    "Near Entry" (0.25 < 0.4 <= 0.50). entry_proximity_ok uses a DIFFERENT
    rule -- max(entry*1.5%, atr*0.5) = max(1.5, 5.0) = 5.0 -- and 4.0 <= 5.0,
    so it's True. Same underlying price move, two different unreconciled
    reads: one says "close enough to trade" (the actually-gating field),
    the other says "not quite there yet" (the purely descriptive one).
    This is the expected, designed-in disagreement, not a bug."""
    quote = {"price": 104.0, "price_branch": "two_sided", "timestamp": "2026-08-01T00:00:00Z", "source": "alpaca"}
    result = router._entry_proximity(entry_price=100.0, atr14=10.0, quote=quote)

    assert result["entry_distance_atr"] == 0.4
    assert result["entry_status_label"] == "Near Entry"
    assert result["entry_proximity_ok"] is True


# -- pydantic model ---------------------------------------------------------------

def test_plan_preview_out_model_accepts_entry_status_label():
    payload = {
        "ticker": "AAPL", "source": "test", "signal": "long", "entry_price": 100.0,
        "stop": 95.0, "target": 110.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "atr14": 10.0, "atr_multiplier": 1.5,
        "rr_warning_threshold": 1.5, "min_target_atr_multiple": 2.0,
        "target_source": "daily_swing_structure", "option_contract": None,
        "preview_error": None, "computed_at": "2026-08-28T00:00:00Z",
        "candidate_updated_at": "2026-08-28T00:00:00Z",
        "entry_status_label": "Near Entry",
    }
    out = router.CandidatePlanPreviewOut(**payload)
    assert out.entry_status_label == "Near Entry"


def test_plan_preview_out_model_defaults_entry_status_label_to_waiting():
    payload = {
        "ticker": "AAPL", "source": "test", "signal": "long", "entry_price": 100.0,
        "stop": 95.0, "target": 110.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "atr14": 10.0, "atr_multiplier": 1.5,
        "rr_warning_threshold": 1.5, "min_target_atr_multiple": 2.0,
        "target_source": "daily_swing_structure", "option_contract": None,
        "preview_error": None, "computed_at": "2026-08-28T00:00:00Z",
        "candidate_updated_at": "2026-08-28T00:00:00Z",
    }
    out = router.CandidatePlanPreviewOut(**payload)
    assert out.entry_status_label == "Waiting"
