# 30m Corrective-Leg Anchoring Research — Phase 3, Batch 2

## 0. What changed

Phase 3's first pass (n=24, 4 approve) found a 75%/27%/20% reclaim-rate
separation (approve/watch/reject) at margin=2 and recommended **B**: real
signal, too small a sample to trust yet. This batch pulled every new real
review recorded in production since then (fetched live via
`GET /api/v1/scanner/candidate-visual-reviews` against the actual
production database — not fabricated), filtered to on-topic reviews only
(excludes `practical_rejection` rows, which never reached the
lower-timeframe question). **19 new examples**, nearly doubling the
approve count (4 → 8) and total n (24 → 43).

## 1. The separation weakened with more data

| Margin | approve | watch | reject |
|---|---|---|---|
| 1 (n=8/29/6) | 62% | 41% | 50% |
| 2 (n=8/29/6) | 50% | 28% | 17% |
| 3 (n=8/29/6) | 12% | 7% | 17% |

Margin=2 still shows the right *direction* (approve > watch > reject), but
the gap shrank substantially (75/27/20 → 50/28/17) — the approve rate
specifically dropped from 75% to 50% as more real examples were added.
Margin=1's separation nearly disappeared (reject rate 50% is now almost
indistinguishable from approve's 62%). This is the expected, honest
behavior of a small-sample finding meeting more data — not a bug, but a
real signal that the first result was more optimistic than the underlying
effect actually supports.

## 2. The harder finding: 2 of the 4 new approve examples show NO signal at ANY margin

This is the important part. Checking each new approve example across all
three margins individually, rather than just the aggregate rate:

| Ticker | margin=1 | margin=2 | margin=3 | Human note |
|---|---|---|---|---|
| CLH | reclaim=True | reclaim=True | no correction | *"bearish correction pulling back into prior breakout area... wait for lower-TF bullish confirmation"* |
| ASND | reclaim=True | no correction | reclaim=False | *(no explicit reclaim language)* |
| XOM | no correction | reclaim=False | no correction | *"30m pullback reached ~$155–156, then produced a bullish CHoCH and subsequent recovery through nearby structure"* |
| OXY | reclaim=False | no correction | no correction | *"closed back above the controlling structure around ~$60.75"* |

**OXY and XOM's human notes describe the reclaim mechanism almost in the
exact words this phase's `check_reclaim` function tests for — and the
mechanical check finds it at NO margin for either one.** CLH (the
already-known anchor) still confirms cleanly at 2 of 3 margins. ASND
partially confirms at 1 of 3. There is no single margin that catches all
four, or even most of them consistently — margin=2 (the aggregate
"best" margin) only catches 1 of these 4 human-confirmed examples.

**Working hypothesis, not yet verified**: `find_thesis_impulse` anchors
the correction's start to "the most recent same-type confirmed swing
pivot" via `_find_swings`. OXY's note references "the controlling
structure around ~$60.75" — that may not be the same level
`_find_swings` would select as the most recent confirmed swing high,
if the human is reading a finer or differently-located structural point
(e.g. an order block boundary, not a swing pivot). This would mean the
reclaim *concept* is right, but the *anchor point* (which price level
counts as "where the correction started") is still being selected
differently than a human would pick it — echoing Phase 2's original
diagnosis, just one level removed from where Phase 2 found it.

## 3. Revised recommendation — **B, downgraded confidence**

Not a clean reversal to C: the aggregate direction is still correct, and
CLH's clean confirmation across two margins is real, not cherry-picked. But
the promising 4-example approve rate was partly a small-sample favorable
draw, and — more importantly — the mechanism is silent or wrong on 2 of 4
new human-confirmed cases whose notes describe exactly the pattern it's
built to catch. That's a specific, checkable problem (anchor-point
selection), not a vague "needs more data" problem.

**Before trusting this for anything real**, the next step isn't more
labels — it's understanding why OXY and XOM's mechanically-reconstructed
correction start doesn't match what their human reviewer was clearly
looking at. That's a targeted debugging question (pull up OXY's and
XOM's actual chronological pivot table, same as Phase 2's Part 8 debug
artifacts, and see exactly which level `find_thesis_impulse` picked vs.
the ~$60.75 the human named), not another aggregate statistics pass.
