"""30m Corrective-Leg Anchoring Research -- Phase 3, Batch 2 (2026-09-21
session). DEVELOPER-ONLY RESEARCH SCRIPT. Not imported by main.py,
candidates_router.py, or any production module. Makes zero writes.

Phase 3's first run (research_30m_corrective_leg_v3_reclaim.py) found a
real, directionally-correct signal (75% reclaim rate for approve-decision
examples vs 27% watch / 20% reject at margin=2) but recommended B, not A,
specifically because n=4 approve examples was too small to trust.

This batch pulls every NEW real review recorded in production's
candidate_visual_reviews table since Phase 1's original 24-example pull
(2026-09-01 session) -- fetched live via GET /api/v1/scanner/
candidate-visual-reviews against the real production database, not
fabricated -- filtered to review_type="visual" with a real
lower_tf_confirmation value (excludes "practical_rejection" rows, which
were rejected before the lower-timeframe question was ever reached --
same exclusion Phase 3's first report already applied to the
market_structure=range rejects in the original 24).

19 new on-topic examples found this way, nearly doubling the approve
count (4 -> 8) and adding a genuinely on-topic reject (BAX, rejected
specifically as "not_yet" -- a real negative case, unlike the original
dataset's HTF-range rejects which were off-topic for this question).
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import research_30m_confirmation_audit as phase1  # noqa: E402
import research_30m_corrective_leg_v3_reclaim as v3  # noqa: E402

MARGINS = (1, 2, 3)

# Pulled live 2026-09-21 via GET /api/v1/scanner/candidate-visual-reviews
# against production. Same tuple shape as phase1.LABELED_EXAMPLES:
# (ticker, lower_tf_confirmation label, decision, reviewed_at, note).
NEW_EXAMPLES = [
    ("BAX", "not_yet", "reject", "2026-09-08T04:32:52.353380+00:00", None),
    ("ASND", "yes", "approve", "2026-09-03T02:10:59.345498+00:00",
     "ASND LONG — bullish Daily trend + bullish 4H recovery + confirmed 30M bullish structure. "
     "Wait for a pullback to the $263 reviewed location while the bullish structure remains intact. "
     "Target $281.26 beneath higher-timeframe liquidity. Stop $255.38."),
    ("LYB", "not_yet", "watch", "2026-09-03T02:07:07.873747+00:00", None),
    ("PGR", "not_yet", "watch", "2026-09-03T02:05:26.418776+00:00", None),
    ("BKR", "not_yet", "watch", "2026-09-03T02:03:45.625654+00:00", None),
    ("DVN", "not_yet", "watch", "2026-09-03T02:02:08.308003+00:00", None),
    ("OXY", "not_yet", "watch", "2026-09-03T01:59:55.065165+00:00", None),
    ("FANG", "not_yet", "watch", "2026-09-03T01:58:08.966994+00:00", None),
    ("STLD", "yes", "watch", "2026-09-03T01:56:25.739411+00:00", None),
    ("OXY", "yes", "approve", "2026-09-01T23:45:54.164659+00:00",
     "Daily and 4H structure support the bullish recovery from the July low, with a clear liquidity "
     "objective near the ~$67 weak high/Kairos target. The recent 30m pullback produced bearish "
     "structure, but buyers subsequently recovered and closed back above the controlling structure "
     "around ~$60.75, providing bullish lower-TF confirmation. Current price remains near Kairos's "
     "$60.94 planned entry rather than materially extended."),
    ("MRNA", "yes", "watch", "2026-09-01T23:31:04.205002+00:00", None),
    ("BP", "not_yet", "watch", "2026-09-01T23:29:06.360035+00:00", None),
    ("XOM", "yes", "approve", "2026-09-01T17:43:39.398120+00:00",
     "Daily structure remains bullish and the recent 4H bearish move appears corrective within the "
     "larger bullish leg. The 30m pullback reached ~$155–156, then produced a bullish CHoCH and "
     "subsequent recovery through nearby structure. Lower-TF bullish confirmation is now present, "
     "with a clear path toward the ~$168 weak high and Kairos's $173.75 target. Location is less "
     "favorable after the recovery, so extension should be monitored"),
    ("EOG", "not_yet", "watch", "2026-09-01T17:37:29.660403+00:00",
     "Daily and 4H structure remain bullish with a clear liquidity objective around $153–154, "
     "aligning well with the $153.34 target. Price recently corrected from that weak high and is now "
     "recovering, but the 30m has not yet provided a clean bullish continuation confirmation. Watch "
     "for a completed 30m close above approximately $149 to confirm the recovery while the original "
     "thesis and R:R remain valid."),
    ("MTDR", "not_yet", "watch", "2026-09-01T17:35:48.845304+00:00",
     "Bullish Daily thesis with a strong 4H recovery from the ~$45 low. Price is now approaching "
     "nearby 30m liquidity/resistance around $59.5–60 while trading in the upper portion of the "
     "current range. Clear path remains toward the $62.77 target and higher-timeframe liquidity, but "
     "lower-TF continuation is not yet confirmed. Watch for a completed 30m close above ~$60."),
    ("CF", "yes", "watch", "2026-09-01T17:34:03.239650+00:00",
     "Bullish Daily/4H structure with 30m bullish continuation confirmed. Price is already extended "
     "into premium after a strong advance and is trading near/through prior 4H highs. Path toward the "
     "~$140–142 weak-high liquidity remains structurally valid, but current location and minimum 1.51 "
     "R:R make the entry unattractive. Prefer a pullback/retest rather than chasing continuation."),
    ("VG", "yes", "watch", "2026-09-01T17:32:07.626846+00:00",
     "Bullish Daily/4H structure with 30m continuation confirmed, but price is extended into the "
     "upper portion of the current leg with nearby 4H/30m weak-high liquidity around $15–15.5. "
     "Location is unfavorable despite the bullish structure. Wait for a pullback/retest or a decisive "
     "break and hold above nearby resistance before reconsidering continuation toward $16.73"),
    ("CLH", "yes", "approve", "2026-09-01T05:12:36.156603+00:00",
     "Bullish HTF structure; bearish correction pulling back into prior breakout area. Clear path "
     "toward $328/weak high. Wait for lower-TF bullish confirmation"),
    ("CLH", "not_yet", "watch", "2026-09-01T00:10:01.323585+00:00",
     "Bullish HTF structure; bearish correction pulling back into prior breakout area. Clear path "
     "toward $328/weak high. Wait for lower-TF bullish confirmation"),
]


def main():
    combined = list(phase1.LABELED_EXAMPLES) + NEW_EXAMPLES
    results = []
    for ticker, label, decision, reviewed_at, note in combined:
        print(f"--- {ticker} (label={label}, decision={decision}) ---", file=sys.stderr, flush=True)
        per_margin = {}
        for m in MARGINS:
            per_margin[m] = v3.run_for_ticker(ticker, "long", reviewed_at, base_margin=m)
        results.append({
            "ticker": ticker, "label": label, "decision": decision,
            "cutoff": reviewed_at, "note": note, "per_margin": per_margin,
            "batch": "original" if (ticker, label, decision, reviewed_at, note) in phase1.LABELED_EXAMPLES else "new",
        })
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
