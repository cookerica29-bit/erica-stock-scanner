# 30m Corrective-Leg Anchoring Research — Phase 2

**2026-09-01 session. Research/audit only. No production code, ranking,
Review Queue, Approved Setups, Approved Setup Memory, or monitor state
changed. No ENTER_NOW/ACTIONABLE/alerts. Zero database writes. Not
pushed, not deployed — committed locally only.**

This phase does not start from CHoCH. It reconstructs the corrective leg
itself first (where it began, its pivot sequence, which pivot is
"controlling") and only inspects break quality after a controlling swing
is selected. Direction is taken as authoritative input from the approved
setup — no independent direction classifier is ever consulted or allowed
to override it.

Research code: `research_30m_corrective_leg_v2.py` (core reconstruction),
`research_30m_corrective_leg_v2_fullset.py` (24-example runner),
`research_30m_corrective_leg_v2_debug_artifacts.py` (chronological debug
tables). All developer-only, zero writes, real point-in-time-truncated
Alpaca 30M data (same provider production uses).

---

## 1. Correction-leg candidate definitions

Six candidate "correction begins" framings were considered (per the
task's Part 2 list): reversal from a confirmed swing extreme, first
opposing structural break, opposing-candle-sequence with meaningful
ATR-normalized movement, pullback from a recent directional impulse,
retracement of the prior leg, and formation of an opposing swing
sequence. These are not mutually exclusive — the one actually
implemented (**"pullback from the most recent directional impulse"**)
subsumes most of the others: it requires (a) a real prior impulse in the
thesis direction (excursion ≥ `MIN_IMPULSE_ATR`, an explicitly-flagged
unvalidated placeholder, currently 1.0 ATR) and (b) at least one
opposing-type pivot forming after that impulse's peak — which is itself
"formation of an opposing swing sequence" and implies retracement. It was
chosen over "first opposing structural break" specifically because Phase
1 showed that starting from a break (BOS/CHoCH) without first
establishing what's being corrected is what produced stale, irrelevant
matches. Other framings (e.g. a pure ATR-normalized opposing-candle-run
without requiring a confirmed pivot) were not implemented — noted as a
real gap in Part 15/18.

---

## 2. Point-in-time pivot semantics

`raw_pivots()` reuses the existing, unmodified `scanner._find_swings` at
a small base margin (tested: 1, 2, 3 bars). Because this is called on an
already point-in-time-truncated dataframe (closed candles only, per
Phase 1's verified `closed_candles_only`), every pivot `_find_swings`
returns is, by that function's own definition, only recognized once
`base_margin` bars exist after it *within the truncated data* — so
**confirmation delay is exactly `base_margin` bars for every pivot**, and
no pivot in any result here could only be known because of bars after
the human's review timestamp. This was true by construction, not
separately re-verified per-pivot, and was additionally sanity-checked via
Phase 1's synthetic boundary test (still valid — the underlying
truncation function is unchanged).

**Can pivots be reclassified by later bars?** The pivot's own
existence/price/timestamp: no. Its MICRO/INTERNAL/STRUCTURAL
*classification* (Part 3 below): also no, within a single run, because
scoring only ever looks at pivots already present in the truncated data.
However, if a NEW pivot forms later (after cutoff, in a future point-in-
time run), an OLDER pivot's neighbor-based excursion score could change
because it now has a different "next neighbor" — this is disclosed
plainly, not hidden: it means a pivot's significance classification is a
snapshot valid as of a given cutoff, not a permanent fact, mirroring
Phase 1's finding about `_detect_choch`'s "most recent" instability.

---

## 3. Pivot significance methodology

Phase 1 found margin=3 vs. margin=4 could flip results. This phase
addresses that directly: pivot significance is scored **independently of
the discrete detection margin**, via each pivot's ATR-normalized
excursion to its immediate opposite-type neighbors (prior and next), and
bar-spacing to those neighbors. A pivot's score is the **minimum** of its
prior/next excursion — conservative by design: a pivot with a big move on
one side but almost none on the other is still likely a minor wiggle.
Thresholds (explicitly unvalidated, flagged in-file):
`MICRO < 0.5 ATR ≤ INTERNAL < 1.5 ATR ≤ STRUCTURAL`.

Fractal nesting / parent-child relationship (also requested) is
**approximated, not fully implemented**: a pivot's neighbor-excursion
score already partially captures "is this nested inside a bigger swing,"
and Method F additionally does a single micro-pruning-then-rescoring
pass (remove MICRO pivots, recompute neighbor relationships among
survivors) as a scoped, one-pass simplification of true hierarchical
reduction — disclosed as a simplification, not a complete fractal tree,
consistent with the time budget for this phase.

---

## 4. Controlling-swing candidate methods

All six (A–F) were implemented exactly as specified, mirrored
automatically for SHORT via the shared `thesis_direction` parameter (no
direction-specific code duplication):

- **A** — most recent confirmed opposing pivot before the correction's extreme.
- **B** — last opposing pivot whose subsequent break produced a new correction extreme (paired against the running-extreme sequence).
- **C** — highest/lowest opposing pivot inside the *final* leg (from the second-most-recent extreme, or correction start if only one leg, to the current extreme).
- **D** — last STRUCTURAL-only opposing pivot that remains unbroken as of cutoff.
- **E** — most recent opposing pivot classified INTERNAL-or-better (excludes MICRO).
- **F** — hierarchical: prune MICRO pivots once, re-score survivors, take the highest/lowest surviving opposing pivot in the final leg.

None were called "correct" a priori. Comparative results are in Part 8/13.

---

## 5–7. NVDA / CLH / FFIV reconstruction

**A real bug was found and fixed during this work**, disclosed in full
because it materially changed every downstream result: the original
`find_thesis_impulse()` selected "correction start" as the most recent
pivot whose *own* neighbor-significance score cleared `MIN_IMPULSE_ATR`
— not the true most recent same-type pivot. For FFIV this picked a
stale, earlier peak instead of the actual most recent high, producing a
self-contradictory correction (extreme priced *beyond* its own start —
caught by a new validity guard, `INVALID_RECONSTRUCTION`). Fixed at the
root: correction start is now always the genuine most recent same-type
pivot, with the impulse-significance gate applied to the *leg's own*
excursion (start→end), not either endpoint's neighbor score. All results
below are post-fix.

### NVDA — human: YES, *"30M bullish CHoCH after correction"*

At `margin=1`: a real, valid correction reconstructed — start 220.60 @
14:30 (an INTERNAL high), decline to extreme 217.71 @ 18:30 (STRUCTURAL
low), 10 bars, depth 2.10 ATR. **Zero opposing (high-type) pivots
confirmed anywhere in that window** — visually, the chronological table
(`NVDA_chronological_debug.txt`) shows real intrabar wiggling around
220.2–220.3 between 15:00–16:30, but none of it resolves into a clean,
isolated local-max pivot at any tested margin; the Low/High values
overlap too much bar-to-bar. **All six methods return None.** At
`margin≥2`, the same "0 opposing pivots" result recurs (fewer total
pivots found, same absence of the opposing type). The strong, clean
break on the final bar (close 220.88, reclaiming the 220.60 correction
start outright) is real and visible in the data — but no *internal*
controlling swing exists for a break-quality check to attach to.

### CLH — human: NOT_YET, *"Wait for lower-TF bullish confirmation"*

At `margin=1/2`: correction start 314.49 @ 18:00, extreme 313.06 @
18:30 — only **3 bars deep**. One internal high pivot exists at the
correction's own start (314.05, INTERNAL) but nothing *after* the
extreme within the window. **Zero opposing pivots after the extreme;
all six methods return None.** Separately and notably (see
`CLH_chronological_debug.txt`): the very last available bar (19:30)
closes at 315.28 — **already above the original correction start**
(314.49) — a real, visible reclaim of the pre-correction level, but not
one this phase's methods are built to detect (they all require an
*internal* opposing pivot specifically, not a simple "price closed above
the pre-correction high" check). This is flagged as a promising,
simpler, complementary signal for future work (Part 15/18), not folded
in here to avoid retroactively tuning the result.

### FFIV — human: YES (weaker label, no note)

At `margin=1`, post-fix: state is **`NO_CORRECTION`**. The genuine most
recent STRUCTURAL impulse is 394.11 (Aug 28 19:30, a low) → 399.61 (Aug
31 13:30, a high) — and the chronological table
(`FFIV_chronological_debug.txt`) shows price simply **climbing
steadily** afterward (397.70 → 401.51 → 403.98 → 407.19) with no
opposing pivot ever forming — i.e., no correction happened at all in the
window this phase looks at. This is a genuinely different pattern from
NVDA/CLH: FFIV's actual structural event, if the human's "yes" label
corresponds to something concrete, more plausibly *is* that Aug 28→31
low-to-high impulse itself (a reversal off a low into a sustained move),
not a "pause-then-reclaim inside an uptrend" pattern this phase's
correction-first methodology is built to find. Reported as a genuine
mismatch between this phase's pattern template and what may have
actually happened, not forced to fit.

---

## 8. Full 24-example results

Every one of the 24 real, point-in-time labeled examples (7 "yes" / 17
"not_yet", same set as Phase 1) was run at margins 1, 2, and 3.

**Headline result: zero opposing-type pivots were found by ANY method,
for ANY of the 24 tickers, at ANY of the three margins tested.** Every
single controlling-swing selection across the entire 72-run sweep
(24 tickers × 3 margins) returned `None`.

Correction-state distribution (how far reconstruction got, even without
ever reaching a controlling-swing candidate):

| margin | CORRECTION_DEVELOPING | CORRECTION_AMBIGUOUS | NO_CORRECTION | NO_IMPULSE_FOUND |
|---|---|---|---|---|
| 1 | 14 | 2 | 3 | 5 |
| 2 | 9 | 3 | 12 | 0 |
| 3 | 7 | 4 | 11 | 2 |

Spot-checked (not assumed): for every multi-pivot correction inspected
directly (DVN, CF, DUOL, CLH), **every pivot found was the same type**
(the extreme type) — confirming this is not a selection bug in the A–F
methods, but that `_find_swings` genuinely never confirms an opposing
pivot inside these short real corrections, at these margins. This makes
geometric sense: a `margin`-bar-confirmable opposing pivot needs enough
bars on both sides to exist within a correction that is itself often
only 2–12 bars deep — many of these real corrections are structurally
too brief for a fixed-window pivot confirmation rule to resolve internal
structure at all, regardless of which of the 6 selection methods is
applied downstream.

---

## 9. Ambiguity cases

`CORRECTION_AMBIGUOUS` (multiple candidate extremes within 0.3 ATR of
each other) occurred for 2–4 tickers per margin (VUG, DVN, NTNX, DUOL,
QQQ, QQQM at various margins) — reported as ambiguous rather than forcing
a pick, per instruction. `NO_IMPULSE_FOUND` (0–5 tickers depending on
margin) and `NO_CORRECTION` (3–12 tickers) are both honest non-signals,
not forced yes/no answers — together these non-committal states account
for **29–63% of the sample depending on margin**, before even reaching
the controlling-swing question.

---

## 10. Parameter stability

Since no controlling swing was ever selected, "does the same swing get
picked under perturbation" is vacuously true (None = None) and not a
meaningful stability measure here. The meaningful stability question at
this stage is **whether a correction is even recognized as the same kind
of thing across margins** — measured directly: for each of the 24
tickers, does the reconstruction `state` (DEVELOPING/AMBIGUOUS/
NO_CORRECTION/NO_IMPULSE_FOUND) stay the same across margins 1→2→3?

**Result: 6/24 (25%) stable.** The other 18 flip between at least two
different states depending purely on base pivot margin — e.g. NVDA goes
NO_IMPULSE_FOUND → CORRECTION_DEVELOPING → CORRECTION_DEVELOPING; FFIV
goes CORRECTION_DEVELOPING → NO_CORRECTION → NO_CORRECTION. **This is the
direct, structural counterpart of Phase 1's margin-3-vs-4 instability —
it persists even after moving to a correction-first methodology**,
because the correction reconstruction itself still ultimately depends on
`_find_swings`'s own margin-dependent pivot confirmation as its
foundation.

---

## 11–13. Break quality / false positives / false negatives

Because zero controlling swings were ever selected, there is **no break
quality to report** (Part 11's instruction — "only after selecting the
controlling swing" — was followed literally: there is nothing to
inspect). The confusion matrix collapses to a degenerate case:

| | detector: no signal (100% of cases) |
|---|---|
| human YES (7) | 7 false negatives |
| human NOT_YET (17) | 17 "true negatives" *if* silence is read as NOT_YET — but this phase's methods do not actually output a NOT_YET verdict, only "no controlling swing identified," which is a different, weaker claim (see Part 15) |

Framed honestly rather than as a flattering accuracy number: **this
methodology currently has nothing to say about any of the 24 examples**
in the specific "here is the controlling swing and whether it broke"
sense it was designed to answer. It also never produced a false
*positive* — a real, if modest, improvement in kind over Phase 1's naive
detectors (which had 13/17 false positives on the same negative-labeled
set).

---

## 14. Lookahead risks

Same guarantee as Phase 1, reverified for this phase's own code path:
`closed_candles_only` + `truncate_point_in_time` are reused unchanged
from `research_30m_confirmation_audit.py`. Every pivot's confirmation
delay is explicit and bar-counted (Part 2). No result in this report used
a bar that would not have closed by the stated review timestamp — the
new bug (Part 5–7) was an **internal logical-consistency defect**
(anchoring to a stale peak), not a lookahead defect; it never used future
data, it used a wrong *choice* among already-valid, already-past data.

---

## 15. Recommendation — **C**

**Mechanical reconstruction still does not correspond reliably enough to
human chart interpretation** — but the specific failure mode has changed
in an informative way from Phase 1, and the reason is now much better
understood:

1. **The core problem is no longer "wrong swing selected."** The A–F
   methods themselves were never actually exercised in anger — they
   uniformly had nothing to choose from. The bottleneck moved one level
   earlier: **the corrective leg itself, once honestly reconstructed
   without lookahead, is usually too short/brief for `_find_swings`-style
   fixed-window pivot confirmation to resolve any internal opposing
   structure at all** (Part 8). This is a more fundamental limitation
   than a threshold-tuning problem.
2. **A real implementation bug was found and fixed mid-phase**
   (Part 5–7) that would have produced a misleadingly clean-looking
   result for NVDA had it gone unnoticed — a caution about how easy it
   is to manufacture false confidence in this problem space, and a
   reason to trust the (less flattering, more honest) post-fix numbers
   over what a first pass might have reported.
3. **State-level stability is only 25%** (Part 10) — the same
   base-margin sensitivity Phase 1 found in swing/CHoCH detection is
   still present one layer up, in whether a correction is recognized at
   all.
4. **A concrete, promising, NOT-yet-tried direction surfaced organically
   in this phase's own data**: CLH's final bar closing back above the
   *original pre-correction level* (Part 5–7) is a real, visible,
   non-fabricated signal that this phase's methods weren't built to
   check (they all require an internal opposing pivot, not a simpler
   "reclaimed the impulse start" test). This — plus explicitly modeling
   sub-pivot-scale price action (single-candle reversal patterns, not
   requiring a full confirmed swing) — is the most concrete lead for a
   Phase 3, not "more of the same swing-based approach with better
   tuning."

**What would change this recommendation to A or B:** if a version of
this methodology that also checks "reclaimed the correction start"
directly (without requiring an internal pivot) were built and it showed
stable, non-fabricated signal on a reasonable fraction of the 24-example
set, that would justify B (more labels, narrow down between that
approach and the pivot-based one). Getting the pivot-based approach
alone to produce ANY non-degenerate result across a meaningful share of
real examples, with reasonable margin-stability, would be the bar for A.
Neither happened here.

---

## Appendix — files

- `research_30m_corrective_leg_v2.py` — core reconstruction module
  (pivot detection/significance, correction reconstruction, methods A–F,
  break inspection).
- `research_30m_corrective_leg_v2_fullset.py` — 24-example runner.
- `research_30m_corrective_leg_v2_debug_artifacts.py` — chronological
  debug table generator (Part 8's minimum required artifact). Chart
  images were attempted but skipped — matplotlib is not installed
  locally or in the Railway environment `railway run` uses (verified via
  direct import check on both) and is not in `requirements.txt`;
  installing a new dependency purely for an optional artifact was not
  undertaken.
- Generated chronological tables for the three anchors (not committed —
  regenerable via the script above against a live-configured
  environment): `NVDA_chronological_debug.txt`,
  `FFIV_chronological_debug.txt`, `CLH_chronological_debug.txt`.
- Raw JSON results are not committed (real market data, regenerable by
  re-running the scripts; none of them write anywhere).
