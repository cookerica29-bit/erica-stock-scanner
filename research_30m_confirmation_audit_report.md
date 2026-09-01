# 30-Minute Execution Confirmation Research Audit

**2026-09-01 session. Research/audit only — no production code changed, no
Approved Setup Memory or monitor state touched, no ENTER_NOW/ACTIONABLE/
alerts implemented. Every finding below is either a direct file/line
citation or a real number produced by running the existing, unmodified
production primitives against real point-in-time-truncated market data.**

Research harnesses used to produce this report:
- `research_30m_confirmation_audit.py` — main harness, 24 human-labeled
  examples, real 30M bars, multiple swing-margin/recent-window
  configurations.
- `research_30m_parent_swing_audit.py` — supplementary harness testing the
  existing `_mtf_shadow_*` parent-swing-anchored primitives against the
  three anchors specifically.

Both are developer-only, read real market data via the same
`scanner._batch_download` provider production already uses, and make
**zero writes** to any table.

---

## 1. Existing reusable primitives

| Primitive | File/function | Timeframe assumption | Lookback/window | Swing/pivot definition | Closed-candles only? | Repaints? | Symmetric? | Category | Safely reusable for 30M research? |
|---|---|---|---|---|---|---|---|---|---|
| Swing detection | `_find_swings`, [scanner.py:5145](scanner.py:5145) | None hardcoded — takes whatever df is passed | `margin` param, default 4 bars each side | Local extremum: a bar's High/Low is the max/min within a `±margin`-bar window | Yes — only ever reads `df["High"]`/`["Low"]`, no intrabar access | Individual confirmed pivots do **not** retroactively change; a pivot only becomes "confirmed" once `margin` bars have passed on both sides (so recognition always lags the actual extreme by `margin` bars) | Yes (high/low symmetric) | A (mechanical fact) — production-authoritative for stop/target on Daily bars | **Conditionally** — the algorithm itself is timeframe-agnostic, but `margin=4` was tuned/validated for Daily bars only; empirically found unsuitable as-is at 30M (Part 4/12) |
| Trend classification | `_get_trend`, [scanner.py:5168](scanner.py:5168) | Swing-list agnostic | Last 2 highs + last 2 lows only | HH+HL → LONG, LH+LL → SHORT, else NEUTRAL | Yes (inherits from swings) | No | Yes | A/B (mechanical fact, used as a candidate filter) | Yes, but strict — frequently returns NEUTRAL (see Part 9) |
| BOS | `_detect_bos` / public alias `detect_structure_break`, [scanner.py:5320](scanner.py:5320) / [scanner.py:8606](scanner.py:8606) (confirmed one-line pass-through, not a separate implementation) | None hardcoded | `lookback=40` bars default | Close beyond the 2nd-most-recent same-type swing, **with same-direction candle body** (a wick poke alone doesn't count) | Yes — `closes[i] > prev_high["price"] and closes[i] > opens[i]` | No (deterministic given the swing list) | Yes | A/E — production-authoritative (gates ENTER_NOW's mechanical check) on Daily bars | Yes as an algorithm; untested at 30M before this audit |
| CHoCH | `_detect_choch`, [scanner.py:5496](scanner.py:5496) | Swing-list agnostic | All swing pairs (unbounded) | HH confirmed after a genuine HL between it and the prior high (mirrored for bearish) | Yes (swing-derived) | Individual pivots don't repaint, but the function always returns only the **most recent** CHoCH — so "what counts as the active CHoCH" shifts as new swings form, even though no past fact is altered | Yes | A in legacy `scanner.py` gating; C (informational overlay) in the current ma_pipeline/candidates_router.py path — see [candidates_router.py:3036-3039](candidates_router.py:3036) | Yes as an algorithm; empirically the single **least reliable** piece at 30M in this audit (Part 9, 12, 13) |
| Order block | `_find_order_block`, [scanner.py:5343](scanner.py:5343) | None hardcoded | Scans back from the most recent swing low/high to the last opposite-colored candle | Last bearish candle before the impulse off the swing low (longs) | Yes | No | Yes | A — production-authoritative (sets the live stop) | Untested at 30M; not exercised in this audit (out of scope — stop logic is explicitly off-limits) |
| Liquidity sweep | `detect_liquidity_sweep`, [scanner.py:8553](scanner.py:8553) | None hardcoded | `lookback=12` bars | Most recent swing low/high; sweep = any bar's **wick** (Low/High) pierced it | **No** — uses intrabar Low/High, not Close | No | Yes | A/D (overlay in current path, hard gate in legacy A+ tier) | Yes as an algorithm; fired on only 3/24 examples at 30M margin=4 (Part 7) |
| Rejection | `detect_rejection` / `_evaluate_rejection` (candidates_router.py mirrors this exactly, confirmed intentional per its own docstring), [scanner.py:8576](scanner.py:8576) | None hardcoded | `lookback=5` bars | Close reclaims the sweep level, OR a long-wick failure pattern with a same-direction close | Uses Close for the reclaim test, wick ratios for the failure pattern | No | Yes | Same as sweep | Yes; fired on only 1/24 examples (TTAN) at 30M margin=4 |
| Displacement | `score_displacement`, [displacement_score.py:72](displacement_score.py:72) | None hardcoded — percentile-ranks against the **stock's own trailing history on whatever df is passed** | `DEFAULT_LOOKBACK_BARS=25` | N/A (continuous score, not a swing concept) | Yes | No | Yes | D (informational grading input, never a gate — explicit in its own docstring) | Yes, but its weights (`DEFAULT_DISPLACEMENT_WEIGHTS`) are explicitly flagged in-file as unvalidated placeholders |
| ATR | `_compute_atr`, [scanner.py:2415](scanner.py:2415) | None hardcoded | `period=14` | N/A | Yes (standard EWM True Range) | No | N/A | A — production-authoritative | Yes, directly, no caveats |
| 4H execution persistence | `_execution_shadow_from_bars`, [candidates_router.py:2010](candidates_router.py:2010) | **Hardcoded 4H** (fetched via `_recent_4h_bars_for_execution_shadow`) | 15 bars (~2.5 trading days) | N/A — not a swing/break check at all; checks hold-zone, no-fresh-lower-low, directional expansion, volume | Yes (`bar.get("close")`) | No | Asymmetric by construction (long-only checks in this codebase's live call sites) | A/E — production-authoritative, gates ENTER_NOW | **No** — already established (prior session's execution audit) that this is a narrow persistence check, not structural confirmation, and must not be reused/renamed as lower-TF confirmation. Reconfirmed here: it operates on a completely different timeframe (4H) and a completely different question ("has price held up recently") than "did the corrective swing break" |
| MTF hierarchy shadow (v1) | `stock_mtf_structure_shadow_for_setup` + `_mtf_shadow_*` helpers, [scanner.py:6369-6660](scanner.py:6369) | **Explicitly multi-timeframe**: 1D/4H/1H/30M, all via the same `_find_swings(margin=4)` | Whatever `_batch_download` returns per timeframe (60d for 4H/1H/30M in production use) | `_mtf_shadow_parent_swing`: the most recent swing high + the swing low that preceded it (an explicit answer to Part 4's "which swing" question, at the 4H level) | Yes | Same caveats as `_find_swings`/`_detect_choch` above | Yes | **D — shadow/research only.** Served via `/api/dev/stock-mtf-structure-shadow*`, gated behind `_require_journal_admin_token`, every response hardcodes `"live_strategy_changed": False` and a `"Shadow ... study only"` message. Never wired into ranking, Review Queue, Approved Setups, or ENTER_NOW | **Yes, conditionally** — architecturally the closest existing thing to what this task needs, but **never previously validated against real human labels**. This audit is the first such validation (Part 9) |
| MTF hierarchy shadow (v2/v3) | `build_stock_mtf_structure_shadow_v2_report`/`_v3_report`, [scanner.py:7034](scanner.py:7034), [7570](scanner.py:7570) | Same as v1, layered on top | Same | Builds additively on v1's output | Same | Same | Same | D — shadow/research only, same gating | Not exercised in this audit (large, ~1000+ combined lines; out of scope given time budget — flagged as a candidate for a **future** audit pass, not evaluated here) |
| `smc_scanner.py` (`reaction_confirmed`, `detect_swings`, `detect_liquidity_sweep`) | [smc_scanner.py](smc_scanner.py), whole module | **Forex-oriented** — docstring examples are EURUSD; uses forex session-level concepts (PDH/PDL, Asian session H/L) | `swing_lookback=3` (its own `ScannerConfig`, separate from `scanner.py`'s `margin`) | Its OWN swing detection (`detect_swings`), unrelated implementation to `scanner._find_swings` | Presumably yes (not verified in depth — out of scope, see below) | Not verified | Not verified | **Unused / untracked** — confirmed via `git status`: this file, `verify_zones.py`, and `tests/smc_scanner_v1.py` are all untracked, never committed, never imported by `main.py`/`candidates_router.py`/`ma_pipeline.py` | **No — do not reuse.** Different codebase, different market (forex), different config system, never integrated, never validated against equities data. Its `reaction_confirmed()` function's *concept* ("CHoCH with displacement OR engulfing close through a zone boundary") is worth knowing about as a design idea, but the implementation itself is out of scope and unvalidated for this purpose |
| `verify_zones.py` | [verify_zones.py](verify_zones.py), whole module | Forex (imports `smc_scanner.py`) | N/A | N/A | N/A | N/A | **Unused** — a matplotlib chart-rendering tool for visually verifying `smc_scanner.py`'s zone detection, CSV-driven, not a detector at all | **No** — it's a plotting/verification utility, not an algorithm; not applicable here |

**Summary of Part 2's core finding:** every genuinely reusable structural
primitive (`_find_swings`, `_detect_bos`, `_detect_choch`, sweep,
rejection) already operates on closed candles only and is directional-
symmetric. None of them were ever validated at 30M before this audit. The
one existing system that already attempts exactly this multi-timeframe
correction/confirmation question — `stock_mtf_structure_shadow_for_setup`
— is shadow/dev-only and had never been checked against real human labels
until Part 9 below.

---

## 2. Unsafe / inappropriate legacy primitives for this purpose

- **`execution_shadow_ok`** — already established (prior session) as a
  narrow 4H price-persistence check, not structural confirmation. This
  audit reconfirms it operates on an entirely different timeframe and
  question than what's being researched here. Must not be reused or
  renamed.
- **`smc_scanner.py` / `verify_zones.py`** — forex-only, untracked,
  never integrated, never validated for equities. Reusing them would
  violate the explicit instruction not to reuse legacy logic merely
  because it exists.
- **Legacy `scanner.py` hard gates** (`scanner._build_trade_stage_eval`'s
  use of BOS/sweep/rejection as mandatory AND-conditions for its "A+
  READY" tier) — these are Daily-bar gates from the pre-ma_pipeline
  system, never validated at 30M, and reusing their *thresholds*
  (as opposed to the underlying detection functions, which are shared
  and fine) would be importing an unvalidated Daily-tuned strategy
  decision into a 30M context wholesale.

---

## 3-4. Corrective structure definition and swing selection — what was tested, honestly

The task correctly identifies this as the hardest part. Three genuinely
different "which swing matters" strategies were tested empirically, not
just discussed conceptually:

**Strategy 1 — full-history swings.** Run `_find_swings` on the entire
point-in-time-truncated 30M history (up to 780 bars / 60 days), let
`_detect_bos`/`_detect_choch` find whatever prior swing pair they find
within their own internal lookback. **Result: badly over-fires** (13 false
positives out of 17 negative-labeled examples) because "prior swing" in a
780-bar window is very often an old, resolved, structurally irrelevant
level from weeks earlier — confirmed directly: NVDA's and FFIV's detected
CHoCH events under this strategy sit **11.1–11.4 ATR** away from the break
candle's close, an implausible distance for a "just happened" event, and
are timestamped 31-37 bars (16-18 trading hours) before the review moment.
This is exactly the failure mode Part 4 warned about — Kairos would be
firing on irrelevant, stale pivots.

**Strategy 2 — recency-constrained window.** Restrict the swing-detection
INPUT itself to only the most recent N bars (40/80/150 tested, ≈3.3/6.7/
12.5 trading days), scaling `margin` down for smaller windows (2/3/4).
**Result: meaningfully better precision at N=40** (8 false positives vs.
13 at full history) but still weak, and unstable — FFIV's detected event
type/level/distance changes materially between margin=3 and margin=4 on
the exact same data (Part 9), meaning the choice of an unvalidated
parameter changes the answer.

**Strategy 3 — HTF-parent-swing-anchored** (the existing
`_mtf_shadow_parent_swing` primitive). Anchor "which 4H range are we
correcting within" to the 4H thesis's own most recent structural leg
(last swing high + the swing low that formed it), then evaluate the 30M
event against that context via `_mtf_shadow_correction_state`. **Result:
timing is more sensible** (CLH's detected 30M BOS is dated the review
day itself, not weeks earlier) **but the upstream 4H direction classifier
(`_get_trend`) frequently disagrees with the actual approved thesis
direction** — it returned NEUTRAL for both NVDA and CLH (real Daily-regime
LONG setups) and returned SHORT for FFIV (a real approved LONG setup) as
of their exact review moments. Using this system correctly requires
feeding it the ALREADY-KNOWN thesis direction (from the approval itself),
not re-deriving direction independently — my supplementary harness did
this correctly for NVDA (fell back to LONG) but NOT for FFIV (used the
mechanically-derived SHORT, which is wrong for a real LONG setup) — a
methodology gap in this specific probe, disclosed rather than hidden,
and itself an important finding: reconciling "mechanically-derived HTF
direction" with "the actual approved thesis direction" is a real, unsolved
design problem, not an implementation detail to paper over.

**No single strategy tested here is ready to ship.** Strategy 2 at N=40
is the least-bad of the three by raw false-positive count; Strategy 3 has
the most sensible event timing but needs its direction-reconciliation bug
fixed before it can be fairly judged.

---

## 5. Closed-candle requirement — how it was enforced

`research_30m_confirmation_audit.py`'s `closed_candles_only()` function
requires `bar_open_time + 30min <= cutoff` for every bar used — not just
`bar_open_time <= cutoff`. Verified with a synthetic test before running
against real data: a bar whose open time exactly equals the cutoff is
correctly excluded (its close would not have happened yet). No intrabar
trigger was used anywhere in this audit.

For every detected event, the harness records: structural level, break
candle timestamp, whether the event was BOS (requires decisive
same-direction-body CLOSE beyond the level — wick alone never qualifies,
per `_detect_bos`'s own logic) vs. CHoCH (swing-sequence based, not a
single-candle close test), and the close's distance beyond the level in
ATR units. Raw wick-only breaks were never counted as confirmation by
either the BOS or CHoCH primitives — both already require a swing-sequence
or same-direction-close condition. A separate, purely wick-based signal
(liquidity sweep) was tracked and reported independently (Part 7), never
conflated with "confirmation."

---

## 6. Decisive break — measured, not decided

`event_close_distance_atr` (break candle's close distance beyond the
level, in ATR units) was recorded for every detected event. Real range
observed: from **0.009 ATR** (FFIV, margin=3, full-history — the close
barely cleared the level at all) to **20.5 ATR** (IGV, full-history — an
implausible distance indicating a stale/irrelevant swing reference, not a
real "decisive" break). No threshold was imposed on the detectors — this
is exactly the "measure first" instruction. What the data shows: distance
alone does not reliably separate real signal from artifact — very large
distances usually indicate a stale swing reference (Part 4), and very
small distances indicate a marginal break, but there is no clean
threshold visible in this sample that would separate genuine confirmation
from noise. Candle body/range, close position within candle, and volume-
relative-to-median features were *not* separately computed in this pass
(scope/time constraint, explicitly disclosed — see Part 16) beyond what
`_detect_bos` already implicitly requires (same-direction candle body).

---

## 7. Liquidity sweep + break as a separate event — measured, not required

`detect_liquidity_sweep` (wick-based, most recent swing level, 12-bar
lookback) and `detect_rejection` (close-reclaim or wick-failure pattern,
5-bar lookback) were run independently alongside BOS/CHoCH, never
required by default. Across all 24 examples at margin=4/full-history:
**sweep fired on 3/24** (NTNX, BILL, TTAN), **rejection fired on 1/24**
(TTAN, the same one). TTAN is labeled `not_yet` — so in this small sample,
requiring sweep+rejection (Detector C) is dramatically more conservative
than plain BOS/CHoCH (Detector A: 19/24 fire; Detector C: 1/24 fires) but
its one firing instance is still a disagreement with the human label, not
a match. **This sample is too small to conclude sweep+reclaim materially
improves quality** — it mostly just fires far less often. Worth
continued measurement with more examples, not adoption as a gate yet.

---

## 8. Retest — not resolved, disclosed as a gap

Given the time budget for this audit, a genuine post-break RETEST_CONFIRMED
sub-state (price breaks the level, pulls back to retest it, then holds)
was **not implemented as a separate measured state**. Only break-detection
(BOS/CHoCH) and the independent sweep/rejection check were built. This is
an explicit, disclosed gap, not a silent omission — Part 16 recommends it
as necessary future work before any detector choice, per the task's own
instruction not to decide break-vs-retest without evidence.

---

## 9. NVDA result — deliverable item 7 (positive label: "30M bullish CHoCH after correction")

Real review note, verbatim: *"Bullish HTF structure; 30M bullish CHoCH
after correction. Neutral entry location with overhead structure near
227–230."* Review timestamp `2026-09-01T01:13:30.652359+00:00`. Point-in-
time data used: 780 30M bars, last closed bar `2026-08-31 19:30:00+00:00`
(the last regular-session 30M bar of that trading day — the review
happened after market close, so this is genuinely the most recent data
that existed at approval time).

- **Strategy 1 (full history, margins 3/4/6):** CHoCH detected, but 31
  bars (≈15.5 trading hours) before cutoff, **11.35 ATR** away — almost
  certainly a stale, unrelated swing pairing, not the event described.
- **Strategy 2, N=40:** **nothing detected at all**; 30M trend classified
  **SHORT** as of the review moment. A clean **false negative** against
  the human's explicit "yes."
- **Strategy 2, N=80/150:** same stale CHoCH as Strategy 1 (31 bars back,
  11.35 ATR).
- **Strategy 3 (parent-swing-anchored):** `h4_thesis_direction` resolved
  to **NEUTRAL** (not LONG) via `_get_trend` — the mechanical 4H
  structure classifier does not agree NVDA had a clean bullish 4H
  structure at the review moment, despite the human calling it "Bullish
  HTF structure" and ma_pipeline's own Daily-SMA-based regime signal
  saying long. `correction_state` came back `NONE` (m30/h1 relationships
  both NEUTRAL).

**Conclusion for NVDA: no configuration tested found a plausible, recent,
decisive mechanical match for the human's stated event.** This is reported
as a genuine disagreement, not adjusted or explained away.

## 10. CLH result — deliverable item 8 (negative/pending label: "wait for lower-TF bullish confirmation")

Real review note, verbatim: *"Bullish HTF structure; bearish correction
pulling back into prior breakout area. Clear path toward $328/weak high.
Wait for lower-TF bullish confirmation."* Review timestamp
`2026-09-01T00:43:10.213229+00:00`. Same last closed 30M bar as NVDA/FFIV
(`2026-08-31 19:30:00+00:00`) — all three anchors evaluate the identical
final bar of that trading day, since all three reviews happened after
market close that same evening.

- **Strategy 1 (margin=4/6):** BOS detected, 12 bars before cutoff, 0.261
  ATR — a real, fairly recent, modest-distance break.
- **Strategy 1 (margin=3):** BOS, 16 bars back, 0.228 ATR — similar.
- **Strategy 2, every window (40/80/150):** an event (CHoCH or BOS) fires
  in every single configuration, 11-16 bars before cutoff, 0.228-2.218
  ATR.
- **Strategy 3 (parent-swing-anchored):** BOS detected, LONG direction,
  level 313.915, dated `2026-08-31T13:30:00Z` — the review day itself, the
  most recent and plausible-looking event timing of any anchor tested.
  `h1_relationship_to_thesis` and `m30_relationship_to_thesis` both read
  `WITH_THESIS`.

**Conclusion for CLH: every single strategy and parameter configuration
tested found SOME kind of confirming break — directly contradicting the
human's explicit "not yet, waiting" label.** This is the most consistent,
robust finding in the entire audit (not parameter-sensitive, unlike
FFIV/NVDA) and arguably the single most important result: **the existing
primitives, however configured, cannot currently distinguish CLH's "not
yet" from a genuine "yes."** Either these primitives measure something
structurally different from what this human reviewer means by "lower-TF
bullish confirmation" for this kind of setup, or the human's bar for
confirmation is higher/more holistic than any tested mechanical
definition captures (see Part 16).

## 11. FFIV result — deliverable item 9 (weaker positive label — no note describing the event)

No review note recorded — `lower_tf_confirmation=yes` with no supporting
text, so this is explicitly treated as a **weaker positive label** per
the task's own framing, not equivalent evidence to NVDA's detailed note.
Review timestamp `2026-09-01T01:08:53.668785+00:00`.

- **Strategy 1/2 (margin=3, all windows):** BOS detected, only **4 bars**
  before cutoff, but **0.009 ATR** distance — essentially zero, a
  marginal, barely-closed-through break.
- **Strategy 1/2 (margin=4/6, N=150):** a completely different event
  (CHoCH, 37 bars back, 11.1 ATR — the same stale-artifact pattern as
  NVDA) surfaces instead.
- **Strategy 3:** `h4_thesis_direction` resolved to **SHORT** — actively
  disagreeing with the real approved LONG direction (a data point this
  audit's supplementary harness mishandled by not overriding it with the
  known true direction — disclosed above in Part 4, not hidden).

**Conclusion for FFIV: highly parameter-sensitive.** A small change in
swing margin flips the answer between "a real, if very marginal, recent
break" and "a stale, implausible one." Given the weak label to begin
with, this is consistent with FFIV being a genuinely marginal/ambiguous
case rather than a clean confirmation — worth exactly this level of
skepticism, not forced into a "match."

---

## 12. Additional historical sample — deliverable item 10

**24 unique, real, human-reviewed examples** were available directly from
production `candidate_visual_reviews` (fetched via the existing
`GET /candidate-visual-reviews` endpoint, not fabricated) with a real
`lower_tf_confirmation` label: **7 labeled "yes", 17 labeled "not_yet"**.
All are direction=long (confirmed via each row's own `setup_key`). Four of
the 17 "not_yet" examples (XLK, WFC, BKR, CF) carry `market_structure=
range` — meaning the human rejected these on Daily/4H structural grounds
before lower-TF confirmation was even the deciding factor; they were kept
in the sample for completeness but are noted as less meaningful tests of
this specific question.

This is a real, if modest, sample — well beyond "three anchors," but not
large enough to support a statistically confident accuracy claim. No such
claim is made here.

No larger historical dataset with real, timestamped, human-reviewed
`lower_tf_confirmation` labels was found elsewhere in the repository or
database. `candidate_promotions` (7 rows) has no such field at all — it's
a different kind of record (mechanical ENTER_NOW promotions), not human
visual-review evidence.

---

## 13. Detector comparison — deliverable item 11

Confusion counts (detected vs. human label, `not_yet`=negative,
`yes`=positive), across the strategies tested:

| Strategy | TP | FP | TN | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| Full history (margin=4) | 6 | 13 | 4 | 1 | 32% | 86% |
| Recent window N=40 | 5 | 8 | 9 | 2 | 38% | 71% |
| Recent window N=80 | 6 | 10 | 7 | 1 | 38% | 86% |
| Recent window N=150 | 6 | 13 | 4 | 1 | 32% | 86% |

(Detector A — "any BOS or direction-favorable CHoCH" — used for this
table; Detectors B/C/D's per-ticker flags are in the raw JSON output,
`research_30m_confirmation_audit.py`'s results. Detector C — sweep+break —
has the best precision by far, 0/1 firings matching a positive label, but
an n=1 firing count is far too small to draw a conclusion from.)

**No configuration achieves usable precision.** N=40 is the least bad
(38% precision, meaning ~2 in 3 "detections" would be wrong), and even
its best case still misses NVDA — the anchor with the most explicit,
detailed human description of the event we're trying to detect.

---

## 14. False positives — deliverable item 12

At the best-performing configuration (N=40 recent window):
**KHC, OXY, LPLA, WFC, CLH, BKR, CF, DUOL** (8 of 17 negative-labeled
examples). Structural context for each is in the raw JSON
(`event_type`, `event_bars_before_cutoff`, `event_close_distance_atr`,
`trend` are recorded for every one). CLH is discussed in full in Part 8
above as the single most important, cross-methodology-consistent false
positive in this audit.

## 15. False negatives — deliverable item 13

At N=40: **VGT, NVDA**. NVDA is discussed in full in Part 7 above — the
anchor with the clearest, most detailed human description of a "30M
bullish CHoCH" event, and the one this audit's mechanical detectors were
**least** able to find under any tested configuration.

---

## 16. Trigger timing — deliverable item 14

For every detected event, `event_bars_before_cutoff` (bars between the
detected break and the review moment) was recorded. Observed range at
N=40: mostly clustered at **11-14 bars** (≈5.5-7 trading hours) before
cutoff — i.e., when a genuine-looking recent break is found at all, it
tends to be found several hours before the review, not moments before.
One notable exception: FFIV's marginal break was only **4 bars** (~2
hours) before cutoff. No example showed a detected event landing within
1-2 bars of the review — either the primitives don't cleanly find
"just-happened" events, or (more likely, per Part 3/4's findings) the
swing-pairing itself is picking up whatever break happens to exist in the
window rather than the SPECIFIC event a human would point to, making the
"timing" measurement itself unreliable until the underlying detection
question (which swing matters) is better resolved. **This audit does not
have a reliable answer to "would this be fast enough for real-time
alerting" — the detection question needs to be solved first.**

---

## 17. Lookahead / repainting risks — deliverable item 15

- **Lookahead:** actively guarded against via `closed_candles_only()`
  (Part 5) — verified correct with a synthetic boundary test before any
  real data was used. No result in this report used a bar that would not
  have been closed at the stated review timestamp.
- **Repainting:** individual confirmed swing pivots do not retroactively
  change (Part 1/2) — `_find_swings`' `margin`-bar confirmation window is
  fixed once satisfied. However, `_detect_choch`'s "most recent CHoCH"
  return value is **not stable over time** in the sense that a fresh call
  with more bars can surface a *different* CHoCH as "the current one" —
  not because the past changed, but because "most recent" is a moving
  target. Any future alerting system built on this needs to be aware
  that "CHoCH detected" is a snapshot-in-time statement, not a permanent
  fact about that price level.
- **Direction-classifier disagreement** (new finding from Part 3/9, not a
  classic repainting risk but a related trust issue): `_get_trend`'s
  strict HH/HL-vs-LH/LL classification frequently disagrees with
  ma_pipeline's Daily-SMA-based regime signal and with the human's own
  stated market structure. Any 30M detector that derives its own HTF
  "thesis direction" independently, rather than trusting the
  already-approved direction, risks silently evaluating the wrong
  direction (as happened to FFIV in this audit's Strategy 3 run).

---

## 18. Recommendation — deliverable item 16

**C — Existing data/logic is insufficient and needs a different research
design, not just more of the same.**

Reasoning, directly from the evidence above:

1. This is **not primarily a sample-size problem.** Even with a real
   24-example, non-fabricated labeled dataset (Part 10) and three
   genuinely different swing-selection strategies (Part 3/4), no
   configuration achieved usable precision (best: 38%, Part 11), and the
   two positive anchors (NVDA, FFIV) behaved in ways that suggest the
   underlying "which swing matters" question is not yet correctly
   specified — not that more labeled examples using the *same* detector
   definitions would resolve it.
2. **CLH is the most important single finding.** Every strategy and every
   parameter tested finds *some* confirming break for CLH, directly
   contradicting the human's explicit "not yet." This is not
   parameter-sensitive noise (unlike FFIV) — it's consistent across
   fundamentally different approaches. That consistency is itself
   informative: it suggests generic swing-break detection, however
   configured, measures something categorically different from what this
   reviewer means by confirmation for this kind of setup — more data
   fed into the *same* kind of detector would likely just reproduce the
   same mismatch.
3. **The most promising existing primitive
   (`_mtf_shadow_parent_swing`/`correction_state`) has a real,
   fixable-but-unfixed methodology problem** (Part 3/4/9's direction-
   reconciliation issue) that needs to be resolved and *then re-tested*
   before it can be fairly judged — that is itself "a different research
   design," not "run the same thing on more tickers."
4. Two explicitly-requested measurement dimensions (Part 6's fuller
   decisive-break feature set: body/range ratio, close-position-in-candle,
   volume-relative-to-median; Part 8's retest-vs-break distinction) were
   **not built in this pass** (time budget) and are likely necessary
   inputs to a redesigned detector, not optional extras.

**What "different research design" concretely means, as a recommended
next step:** fix the parent-swing approach's direction handling (always
use the already-known approved thesis direction, never re-derive it),
extend it to explicitly identify *the specific corrective leg back into
the parent 4H range* as its own anchored swing pair (rather than treating
any `_find_swings` pivot in a recency window as eligible), add the
missing decisive-break feature set (Part 6) and a genuine retest sub-state
(Part 8), and *then* re-run against this same 24-example set plus
whatever additional real reviews accumulate. Only after that redesigned
detector is measured should a shadow-detector build (Option A) be
considered.

---

## Appendix — files

- `research_30m_confirmation_audit.py` — main harness (24-example
  comparison, 3 swing margins × 4 recent-window sizes).
- `research_30m_parent_swing_audit.py` — supplementary harness
  (parent-swing-anchored primitives, 3 anchors).
- Raw JSON results are not committed (real market data, regenerable by
  re-running either script against a live Alpaca-configured environment;
  neither script writes anywhere).
