"""30m Corrective-Leg Anchoring Research -- Phase 2, full 24-example run
(2026-09 session). Developer-only. No writes anywhere.

Runs research_30m_corrective_leg_v2's reconstruction + controlling-swing
methods against the same 24 real, point-in-time human-labeled examples
from Phase 1 (research_30m_confirmation_audit.LABELED_EXAMPLES), at 3
base pivot margins (1, 2, 3), to directly answer Part 9 (full-set results,
reported honestly including UNKNOWN/AMBIGUOUS cases, never forced into
yes/no) and Part 10 (parameter stability -- does the SELECTED controlling
swing's PRICE LEVEL stay the same as base_margin changes?).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import research_30m_confirmation_audit as phase1  # noqa: E402
import research_30m_corrective_leg_v2 as v2  # noqa: E402

MARGINS = (1, 2, 3)
METHOD_NAMES = ("A", "B", "C", "D", "E", "F")


def run_one(ticker: str, label: str, decision: str, cutoff_iso: str) -> dict:
    per_margin = {}
    for m in MARGINS:
        r = v2.run_for_ticker(ticker, "long", cutoff_iso, base_margin=m)
        per_margin[m] = r
    return {"ticker": ticker, "label": label, "decision": decision, "cutoff": cutoff_iso, "per_margin": per_margin}


def main():
    results = []
    for ticker, label, decision, reviewed_at, _note in phase1.LABELED_EXAMPLES:
        print(f"--- {ticker} ---", file=sys.stderr, flush=True)
        results.append(run_one(ticker, label, decision, reviewed_at))
    print(json.dumps(results, default=str))


if __name__ == "__main__":
    main()
