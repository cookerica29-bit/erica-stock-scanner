"""Generates Part 8 visual debug artifacts for the three anchors, at
base_margin=1 (the finest granularity tested). Developer-only, no writes
anywhere beyond local txt files in the scratch output directory.

Chart images (candles + annotated pivots/controlling swing) were
attempted but skipped: matplotlib is not installed locally or in the
Railway environment `railway run` uses (confirmed via direct import
check on both), and it is not in requirements.txt. Installing it purely
for an optional ("if practical") research artifact was not undertaken,
to avoid any dependency/environment change -- this is disclosed
explicitly in the Phase 2 report rather than silently skipped. The
REQUIRED minimum (a chronological table) is produced below in full.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import scanner  # noqa: E402
import research_30m_confirmation_audit as phase1  # noqa: E402
import research_30m_corrective_leg_v2 as v2  # noqa: E402

ANCHORS = [
    ("NVDA", "long", "2026-09-01T01:13:30.652359+00:00"),
    ("FFIV", "long", "2026-09-01T01:08:53.668785+00:00"),
    ("CLH", "long", "2026-09-01T00:43:10.213229+00:00"),
]

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."


def generate(ticker, direction, cutoff_iso):
    thesis_direction = "LONG" if direction.lower() == "long" else "SHORT"
    cutoff = phase1._to_utc(cutoff_iso)
    raw = phase1.fetch_bars(ticker, interval="30m", period="60d")
    df = phase1.closed_candles_only(phase1.truncate_point_in_time(raw, cutoff), cutoff, bar_minutes=30)
    atr = scanner._compute_atr(df, period=14)
    pivots = v2.score_pivot_significance(v2.raw_pivots(df, v2.BASE_PIVOT_MARGIN), atr)
    correction = v2.reconstruct_correction(df, pivots, thesis_direction, atr)

    methods = {}
    if correction.state in ("CORRECTION_DEVELOPING", "CORRECTION_AMBIGUOUS"):
        for name, fn in v2.METHODS.items():
            p = fn(correction, thesis_direction, atr)
            methods[name] = {"selected_pivot": p}
        methods["D"] = {"selected_pivot": v2.method_D(correction, thesis_direction, atr, df)}

    table = v2.chronological_debug_table(df, pivots, correction, methods)
    table_path = f"{OUT_DIR}/{ticker}_chronological_debug.txt"
    with open(table_path, "w") as f:
        f.write(f"{ticker} ({thesis_direction}) -- point-in-time cutoff {cutoff_iso}\n")
        f.write(f"correction state: {correction.state}\n")
        f.write(f"start: {correction.start_price} @ {correction.start_timestamp}\n")
        f.write(f"extreme: {correction.extreme_price} @ {correction.extreme_timestamp}\n")
        f.write(f"note: {correction.ambiguity_note}\n\n")
        any_controlling = any(e.get("selected_pivot") for e in methods.values())
        f.write(f"controlling swing found by any method: {any_controlling}\n\n")
        f.write(table)
    print(f"wrote {table_path}")


if __name__ == "__main__":
    for ticker, direction, cutoff in ANCHORS:
        generate(ticker, direction, cutoff)
