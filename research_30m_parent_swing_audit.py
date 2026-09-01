"""30-Minute Execution Confirmation Research Audit -- supplementary run
(2026-09 session). Tests the EXISTING _mtf_shadow_* primitives
(scanner.py, built for /api/dev/stock-mtf-structure-shadow*, shadow/
dev-only, never production-authoritative) directly against point-in-time-
truncated real data for the three anchors, rather than raw
_find_swings+_detect_bos/_detect_choch with an arbitrary recent-bar
window (see research_30m_confirmation_audit.py's first two runs).

Reasoning: _mtf_shadow_parent_swing anchors "which swing matters" to the
4H thesis's own most recent structural leg (last swing high + the swing
low that preceded it), instead of treating every swing pair _find_swings
happens to produce as equally eligible -- this is architecturally the
closest existing candidate to what Part 4 of the audit is asking for. It
has never itself been validated against real human labels; this is that
validation, scoped to the three anchors only (not the full 24-example set,
given time budget -- the audit report is explicit that this is a smaller,
supplementary check, not a full replacement for the main comparison).

DEVELOPER-ONLY. No production imports beyond scanner.py's already-shadow
functions. No writes anywhere.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import scanner  # noqa: E402
import research_30m_confirmation_audit as base  # noqa: E402


ANCHORS = [
    ("NVDA", "2026-09-01T01:13:30.652359+00:00", "yes"),
    ("FFIV", "2026-09-01T01:08:53.668785+00:00", "yes"),
    ("CLH", "2026-09-01T00:43:10.213229+00:00", "not_yet"),
]


def fetch_truncated(ticker: str, interval: str, period: str, cutoff, bar_minutes: int):
    raw = base.fetch_bars(ticker, interval=interval, period=period)
    if raw.empty:
        return raw
    return base.closed_candles_only(base.truncate_point_in_time(raw, cutoff), cutoff, bar_minutes=bar_minutes)


def run_anchor(ticker: str, cutoff_iso: str, label: str) -> dict:
    cutoff = base._to_utc(cutoff_iso)
    h4 = fetch_truncated(ticker, "4h", "60d", cutoff, 240)
    m30 = fetch_truncated(ticker, "30m", "60d", cutoff, 30)

    if h4.empty or m30.empty or len(h4) < 10 or len(m30) < 10:
        return {"ticker": ticker, "label": label, "error": "insufficient point-in-time data", "h4_bars": len(h4), "m30_bars": len(m30)}

    h4_direction, h4_swings = scanner._mtf_shadow_structure_direction(h4, margin=4)
    h4_price = float(h4["Close"].iloc[-1])
    parent_swing = scanner._mtf_shadow_parent_swing(h4, h4_swings, h4_direction if h4_direction in ("LONG", "SHORT") else "LONG")
    h4_location = scanner._mtf_shadow_location(h4_price, parent_swing, h4_direction if h4_direction in ("LONG", "SHORT") else "LONG")

    m30_direction, m30_swings = scanner._mtf_shadow_structure_direction(m30, margin=4)
    thesis_direction = h4_direction if h4_direction in ("LONG", "SHORT") else "LONG"  # fall back to the review's own stated direction
    m30_event = scanner._mtf_shadow_structure_event(m30, m30_swings, thesis_direction)
    m30_relationship = scanner._mtf_shadow_relationship(m30_direction, thesis_direction, m30_event)

    h1 = fetch_truncated(ticker, "1h", "60d", cutoff, 60)
    h1_direction = "NEUTRAL"
    h1_relationship = "NEUTRAL"
    if not h1.empty and len(h1) >= 10:
        h1_direction, h1_swings = scanner._mtf_shadow_structure_direction(h1, margin=4)
        h1_event = scanner._mtf_shadow_structure_event(h1, h1_swings, thesis_direction)
        h1_relationship = scanner._mtf_shadow_relationship(h1_direction, thesis_direction, h1_event)

    correction_state = scanner._mtf_shadow_correction_state(h1_relationship, m30_relationship, m30_event, thesis_direction)

    return {
        "ticker": ticker,
        "label": label,
        "h4_thesis_direction": h4_direction,
        "h4_parent_swing": parent_swing,
        "h4_location": h4_location,
        "h1_direction": h1_direction,
        "h1_relationship_to_thesis": h1_relationship,
        "m30_direction": m30_direction,
        "m30_relationship_to_thesis": m30_relationship,
        "m30_event": m30_event,
        "correction_state": correction_state,
        "h4_bars": len(h4),
        "h1_bars": len(h1),
        "m30_bars": len(m30),
        "m30_last_bar": str(m30.index[-1]),
    }


def main():
    results = []
    for ticker, cutoff_iso, label in ANCHORS:
        print(f"--- {ticker} ---", file=sys.stderr, flush=True)
        results.append(run_anchor(ticker, cutoff_iso, label))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
