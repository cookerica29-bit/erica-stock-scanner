# Kairos — Review-Queue Evolution: Targeted Audit

**Audit only. No production code changed. No commits. No deployments.**

Every claim below is either a **VERIFIED FACT** (cited to a specific file/function/line, or to a real production number pulled live during this audit) or explicitly marked **RECOMMENDATION**. Where a number was measured against live production data, the query and result are shown.

---

## 1. Current production funnel / call chain

Two separate "scanner" pipelines exist in this codebase. **This audit covers the stock/Review-Queue pipeline** (`ma_pipeline.py` → `candidates_router.py`), which is what `/review-queue` actually serves. There is a second, older, options-oriented pipeline (`main.api_scan` → `scan_cached` → `smart_notifications.py`, feeding `index.html`'s "Candidates Dashboard") that is architecturally unrelated to `/review-queue` and out of scope here — flagged only so it's never confused with the funnel below.

```
Universe                    main._merge_curated_watchlist_into_universe (main.py:1373)
                             merges scanner.WATCHLIST (113 hardcoded symbols, scanner.py:49)
                             into the broker-fed "discovered" universe (~939 symbols,
                             discovery.build_ranked_discovery_universe), capped at
                             MA_PIPELINE_MAX_SYMBOLS (default ~1000)
                             │
Scheduling                  main._submit_ma_pipeline_scan_if_due (main.py:1468)
                             runs TWICE A DAY ONLY: MA_PIPELINE_SCAN_TIMES_ET,
                             default "09:45,15:30" ET, trading days only
                             │
Candidate generation        ma_pipeline.scan_ma_pipeline_candidates →
+ first hard gates           _candidate_from_frames (ma_pipeline.py:93)
                             │
Persistence                 candidates_router.upsert_candidate_shortlist
                             (candidates_router.py:4296) — UPSERT into `candidates`
                             table, keyed (ticker, source); NEVER deletes a row
                             │
Structure computation        candidates_router._compute_candidate_promotion
(BOS/CHoCH/sweep/             (candidates_router.py:3558) — calls into scanner.py's
rejection/location/           _find_swings/_detect_bos/_detect_choch/
displacement/target/stop)     detect_liquidity_sweep + this module's own
                             score_location/score_displacement/summarize_confluence
                             │
Review-queue hard gates      candidates_router.list_review_queue
+ ranking                    (candidates_router.py:4697), via
                             _preview_clears_mechanical_prechecks →
                             _preview_base_enter_now_ready → _stage1_mechanical_ready
                             → rank_stage1_candidates / _confluence_sort_key
                             │
Review Queue                 GET /candidates/review-queue → public/review_queue.js
```

**VERIFIED, live production, pulled during this audit** (`railway run` against the real API):
- `candidates` table: **636 total rows** (452 `signal=long`, 184 `signal=short`)
- `/candidates/review-queue`: **13 candidates** returned
- `/candidate-near-misses` (fails exactly 1 gate, capped at the endpoint's own `limit=50` max): **50 returned** — i.e. **at least 50** structurally-generated long candidates are sitting exactly one gate away from the queue right now, and the true count could be higher since the endpoint itself caps at 50.

So today's funnel already narrows **452 → 13** long candidates (a ~97% elimination rate) purely from gates — landing coincidentally near the 10–15 range the proposed model wants, but as a **side effect of gate strictness**, not a deliberate rank-then-truncate design (§2 confirms no truncation exists at all).

---

## 2. Hard-gate inventory

For each gate: what it tests, whether it **eliminates** the ticker outright or only affects score, and — the classification the whole audit turns on — **whether it's opportunity-discovery or trade-readiness**.

| # | Gate | Where | Condition | Eliminates or scores? | Discovery or readiness? |
|---|---|---|---|---|---|
| 1 | Daily regime stack | `ma_pipeline._candidate_from_frames`, ma_pipeline.py:112 | `latest_daily > sma50 > sma200` (long) / mirrored short | **Eliminates.** No candidate row is ever created — invisible everywhere, even to near-miss. | Discovery-adjacent (trend existence), but binary/strict — a fresh cross or basing structure is invisible even if visually promising. |
| 2 | 4h EMA21 proximity ≤3% | `ma_pipeline._candidate_from_frames`, ma_pipeline.py:119-121 | `abs(latest_4h - ema21_4h) / latest_4h <= 0.03` | **Eliminates**, at candidate-**generation** time — before the ticker exists as a row anywhere, before near-miss can even see it. | **Pure trade-readiness/timing.** This is the earliest and most invisible readiness gate in the whole funnel. |
| 3 | `signal='long'` | `list_review_queue` SQL, candidates_router.py:4751 | Direction filter | **Eliminates.** 184/636 (29%) of all candidates today. | Neither — a scope decision (shorts are "research-only," see `_promotion_block_reason`'s own comment, candidates_router.py:3883-3906), not a quality judgment. |
| 4 | Regime alignment (again) | `_candidate_regime_aligned`, candidates_router.py:3826 | `daily_regime` string contains "long"/"bull" | **Eliminates**, but mostly redundant with #1 for `ma_pipeline`-sourced rows (same value, set once). | Same as #1. |
| 5 | Daily-bar availability | `_compute_candidate_promotion`, candidates_router.py:3574-3586 | ≥20 usable daily candles, ATR14 > 0 | **Eliminates** (raises `HTTPException`, caught, row still shows in `preview_error`-tagged form but never clears prechecks) | Data-quality, not a judgment about the setup. |
| 6 | Valid structural target | `_preview_clears_mechanical_prechecks`, candidates_router.py:2762 | A swing exists ≥ `min_target_atr_multiple × ATR14` away | **Eliminates.** | Closer to fundamental viability than pure timing — "no computable target" is a real structural gap, though a genuinely tight/compressed range could still be worth a human glance. |
| 7 | R:R ≥ 1.5 | `_preview_clears_mechanical_prechecks`, candidates_router.py:2764; `RR_WARNING_THRESHOLD=1.5` | `risk_reward >= 1.5` | **Eliminates.** | **Hybrid.** R:R is computed from `entry_price` frozen at the last twice-daily scan (§5) — a structurally good setup can fail this purely because of when it was last scanned relative to where price is now. |
| 8 | Entry proximity | `_preview_base_enter_now_ready`, candidates_router.py:2778; thresholds `ENTRY_PROXIMITY_MAX_PCT_DEFAULT=1.5`, `ENTRY_PROXIMITY_MAX_ATR_MULTIPLE_DEFAULT=0.5` | Live quote within 1.5% / 0.5×ATR of the frozen entry | **Eliminates.** **Confirmed the single largest live source of elimination today** — of the sampled near-miss tickers, the overwhelming majority (AAOI, ACGL, AFRM, AME, AON, ARMK, …) fail on `entry_proximity` alone. | **Pure trade-readiness/timing.** This is exactly the kind of rule the audit was asked to find: it eliminates candidates purely because price isn't near entry *right now*, regardless of how good the underlying structure is. |
| 9 | Execution shadow | `_stage1_mechanical_ready`, candidates_router.py:2801; computed by `_execution_shadow_from_bars`, candidates_router.py:2567 | A multi-condition check over the most recent 15 4h bars: price holding above `entry - 0.5×ATR` (or the 4h EMA21 if higher), no fresh lower low, directional expansion ≥ threshold, a "strong" confirming candle with adequate reaction, volume qualification for low-volatility names | **Eliminates.** | **Pure trade-readiness/timing** — explicitly requires *very recent, currently-in-progress* price action, not just decent structure. |

**Gates that exist in the codebase but do NOT gate `/review-queue` today** (important — this directly bears on §4/§6):

| Signal | Where computed | Legacy (`scanner.py`) role | Current `_stage1_mechanical_ready` role |
|---|---|---|---|
| BOS confirmed | `_detect_bos` via `_compute_candidate_promotion` | Hard hard-cap hint (part of A+/B+ tier logic) | **Informational only** — code comment explicitly: "does NOT gate ENTER_NOW eligibility or block promotion" (candidates_router.py:3593-3606) |
| Macro bias / CHoCH conflict | `_recency_adjusted_macro_bias`, `_detect_choch` | Hard-caps score to grade "C", forces `a_plus_ready=False` | **Informational only** (candidates_router.py:3608-3620) |
| Liquidity sweep + rejection | `detect_liquidity_sweep`, `_evaluate_rejection` | Mandatory AND-condition for the top "A+ READY" tier | **Informational only** (candidates_router.py:3650-3667) |
| Location (premium/discount) | `score_location` | "The single hardest gate found in any port" (three inconsistent threshold schemes) | **Informational only**, exposed as a continuous percentile (candidates_router.py:3669-3686) |
| Displacement | `score_displacement` | Legacy `detect_displacement` was a hard threshold | **Informational only**, continuous score (candidates_router.py:3688-3693) |
| Confluence label (incl. "conflicted") | `summarize_confluence` | N/A (new, not a legacy port) | **Ranking input only** for `/review-queue` — explicitly and deliberately moved OUT of gating in a documented 2026-08-31 redesign (candidates_router.py:2781-2793) |

**One important architectural nuance not to lose**: there is a **second, parallel** eligibility function, `_promotion_block_reason` (candidates_router.py:3877), that governs a *different* concept — "ENTER_NOW dashboard-ready" for the older `candidates.html`/`index.html` promotion workflow, not `/review-queue` membership. It shares most of the same mechanical conditions as `_stage1_mechanical_ready` but **additionally** hard-blocks `confluence_label == "conflicted"` (candidates_router.py:3945-3946) — a gate that was deliberately *not* carried over when `/review-queue` was redesigned. These two gate functions can drift out of sync; any future change should be explicit about which one it's touching. See §8.

---

## 3. Current ranking formula — exactly, in full

`rank_stage1_candidates` (candidates_router.py:2846), sort key `_confluence_sort_key` (candidates_router.py:2817):

```python
(
    0 if has_confluence else 1,      # confluence-unavailable candidates sort last, as a group
    -favorable_count,                # more favorable signals first
    unfavorable_count,               # fewer unfavorable signals first
    -risk_reward,                    # higher R:R first
    entry_distance_pct,              # closer to entry first
)
```

- **Weights**: none. `favorable_count`/`unfavorable_count` are flat, equal-weighted counts of the 7 confluence signals (BOS, displacement, sweep, rejection, macro/CHoCH, location, R:R) — explicitly a deliberate choice against a weighted composite (confluence_summary.py:15-26: *"a weighted scheme would mean inventing one new unvalidated number PER SIGNAL... strictly more guessing than a flat count, not less"*).
- **Tie-breaking**: R:R descending, then entry-distance ascending, then **stable** (original `updated_at DESC` order preserved for anything still tied).
- **Ranking occurs strictly AFTER all hard gates**: `ready = [... if _stage1_mechanical_ready(...)]` filters first (candidates_router.py:2855), *then* `.sort(...)`. A candidate that fails any hard gate never enters the sort at all.
- **Ranking is global, not truncated-then-ranked**: every survivor of `_stage1_mechanical_ready` is sorted and returned — `"count": len(ranked)`, `"candidates": ranked` (candidates_router.py:4822-4823), **no Top-N cap anywhere in this code path.** Today's queue happens to be small (13) because the gates are strict, not because ranking stops early.
- **Mechanism/disclaimer are self-reported honestly**: `RANKING_MECHANISM_VERSION = "stage1_mechanical_then_confluence_count_v1"`, `RANKING_DISCLAIMER = "Ranking is unvalidated -- based on signal counts, not track record."` (candidates_router.py:2813-2814) — the codebase already discloses this is unvalidated.

### Explicit answer to your question 2's core ask

> *If 40 potentially useful charts exist, can the current architecture reliably identify the best 10–15 of those 40 for human review?*

**No, not reliably, for two independent reasons, both verified:**

1. **There is no Top-N selection mechanism at all.** The queue returns however many candidates survive the hard gates — could be 2, could be 60. Today it happens to be 13, which is a coincidence of gate strictness on this particular day, not a designed outcome.
2. **The hard gates that determine "survives at all" are dominated by trade-readiness/timing conditions** (#2, #8, #9 above), not opportunity quality. A structurally excellent setup (strong BOS, favorable displacement, clean sweep+rejection, good location) that is simply not within 1.5%/0.5×ATR of its entry *right now*, or lacks a same-week confirming 4h candle, is **completely invisible** — not ranked #40, not visible via a toggle on the Review Queue page, just gone. The near-miss data pulled live during this audit (§1) shows this happening to at least 50 real tickers today.

So of a hypothetical 40 "potentially useful charts," today's architecture would likely show some small, gate-determined subset (anywhere from a handful to all 40, unpredictably) — not a deliberately curated best-10–15.

---

## 4. Staleness / extension findings

**VERIFIED**, all from direct code reading:

1. **Candidate `entry_price` is captured at most twice a day.** `ma_pipeline._candidate_from_frames` sets `entry_price = round(latest_4h, 4)` (ma_pipeline.py:127) at whichever of the two daily scan times (default 09:45/15:30 ET) actually ran. Between scans, this value is frozen.
2. **The `candidates` table row is never deleted or expired.** `upsert_candidate_shortlist` (candidates_router.py:4296) only INSERTs/UPDATEs tickers present in the *current* scan payload — a ticker that drops out of a later scan (e.g. no longer regime-qualifies) simply keeps its old row, with a stale `entry_price`/`daily_regime`, indefinitely. `list_review_queue`'s query (`SELECT * FROM candidates WHERE signal='long' ORDER BY updated_at DESC`) has **no age/recency cutoff at all** — there is no explicit "this candidate is too old, drop it" rule anywhere in this path.
3. **R:R, stop, and target are computed from the frozen `entry_price`**, not a live price. `_compute_candidate_promotion` reads `entry = candidate["entry_price"]` (candidates_router.py:3561) directly from the DB row — `risk = abs(entry_price - stop)`, `reward = abs(target - entry_price)` (candidates_router.py:3706-3719) both use that stale value. A setup that has genuinely moved a lot since the last scan can still show an R:R computed against a now-outdated reference entry.
4. **Only entry proximity and the displayed "current price" use a live quote.** `_attach_entry_proximity` (candidates_router.py:2421, called from `list_review_queue` via `_latest_quotes_for_previews`) fetches a real live quote on **every single GET call** and compares it against the frozen `entry_price`. This is the mechanism that *does* eventually catch an extended setup — once live price drifts far enough past entry, `entry_proximity_ok` correctly flips to `False` and the candidate is dropped from the ranked list on the next fresh computation (gate #8 in §2).
5. **Ranking is recomputed fresh on every backend call, but the frontend never re-issues that call automatically.** `public/review_queue.js`'s `loadQueue()` (review_queue.js:180) is called once, on page load — grepping the whole file for `setInterval` returns nothing. There is no polling, no auto-refresh, no "stale data" banner.

### Direct explanation of the symptom you observed

The most likely explanation, given the above, is **not a backend computation bug**: the backend, if asked fresh, would correctly reflect live price via `entry_proximity_ok`/`current_price`/`entry_distance_pct`. The most likely cause is **#5 — the open browser tab was never told to refetch**, so the card kept showing whatever `list_review_queue` returned at the last page load, even as real price kept moving. A secondary, compounding factor is **#3** — even a fresh reload's R:R math still uses the twice-daily `entry_price`, not the current live price, so R:R itself can lag reality by up to ~6 hours even on a genuinely fresh call.

### What a formal staleness/extension penalty would need (not implemented, described only)

- A **distance-beyond-entry** measure (already computable — `entry_distance_pct` exists, just currently binary-gated rather than continuously scored).
- A **proximity-to-target** measure (not currently computed at all — `_current_rr`-style math exists in `candidates_router.py`'s Approved/Watch monitor for a *different* purpose (frozen approved memories, not raw candidates) and isn't wired to the raw candidate pipeline).
- An explicit **candidate-row age** cutoff, since rows never expire today (§4.2).
- A decision on whether staleness should be an **exclusion** (matching the "no candidate should occupy a Top-10 slot after the opportunity passed") or a **ranking penalty** (matching the "loose enough to discover" philosophy) — the existing `entry_proximity_ok` binary-eliminate pattern is the former; the proposed Review-Value model in §5 argues for the latter.

---

## 5. Proposed classification of existing signals

Per your instruction not to assume location/sweep/rejection/displacement should auto-eliminate — and consistent with what's **already true today** (§2's second table: none of these currently gate `/review-queue`):

| Signal | Proposed classification | Note |
|---|---|---|
| Daily regime stack (SMA50/200) | **Hard eligibility gate** (keep) | A genuine prerequisite for "this ticker has a directional thesis at all" — but see §9 risk about basing/fresh-cross tickers being invisible. |
| 4h EMA21 proximity ≤3% (ma_pipeline) | **Ranking penalty, not a gate** *(change from today)* | Currently eliminates before a candidate even exists as a row — the single most invisible readiness gate found. Converting to a penalty lets Kairos surface "good structure, currently extended from the pullback zone" for human judgment. |
| Valid structural target | **Hard eligibility gate** (keep) | Without a target there's no trade plan to show a human at all — this is closer to "not a computable opportunity" than "not trade-ready." |
| R:R < 1.5 | **Ranking penalty, not a hard gate** *(change from today)* | A structurally sound setup can fail this purely from stale-entry math (§4.3). Reward/risk quality is exactly the kind of continuous evidence a human should weigh, not a silent binary cutoff. |
| Entry proximity | **Ranking penalty (or a "watch, not review-now" bucket), not a hard gate** *(change from today)* | Confirmed the largest current source of "otherwise good, invisible" candidates. This is pure trade-readiness, and the proposed model explicitly wants review-worthy-but-not-yet-actionable charts to still surface. |
| Execution shadow (recent 4h confirmation) | **Ranking penalty, not a hard gate** *(change from today)* | Also pure trade-readiness/timing — a setup lacking a fresh confirming candle can still be worth a human's attention ("watch for confirmation"), which maps directly onto the existing Watch lifecycle rather than disappearing entirely. |
| BOS confirmed | **Ranking reward** (already true) | No change needed. |
| Macro bias / CHoCH conflict | **Ranking penalty** (already true) | No change needed. |
| Sweep confirmed | **Ranking reward** (already true) | No change needed. |
| Rejection confirmed | **Ranking reward** (already true) | No change needed. |
| Location (premium/discount) | **Ranking reward/penalty** (already true) + **human-review information** | Already continuous (`location_percentile`) — good raw material for a review-value score, and already displayed. |
| Displacement | **Ranking reward/penalty** (already true) + **human-review information** | Same — already continuous. |
| Confluence label/counts | **Ranking sort key** (already true for `/review-queue`; still a hard gate in the *separate* `_promotion_block_reason` path — see §2's nuance) | Recommend explicitly deciding whether the ENTER_NOW-dashboard path should also stop hard-gating on this, for consistency — not required for the Review-Queue redesign itself. |
| Target/path quality | **Not currently computed as a distinct signal** | `target_clamped`/`target_clamp_reason` (from `levels_near_target`/`clamp_target`) exist and are close to this — worth reusing rather than inventing new logic. |
| Freshness of confirmation | **Not currently computed as a distinct signal for raw candidates** | The Approved/Watch monitor has an analogous concept (`_evidence_freshness_anchor`/`EXECUTION_EVIDENCE_FRESHNESS_HOURS`, candidates_router.py:5971-6003) built for a *different* purpose (frozen memories) — same idea, different data, not wired to raw candidates today. |
| Contradictory/newer opposing structure | **Overlaps with macro/CHoCH conflict** (already computed) | No new computation needed; already a ranking penalty. |
| Extension beyond planned entry | **Not currently computed as a continuous measure** | `entry_distance_pct` exists but is binary-gated (§2 #8), not scored. |
| Proximity to target | **Not currently computed at all for raw candidates** | Would need new logic — see §4's staleness section. |
| Stock liquidity | **Not currently gated or scored** | No stock-side liquidity/volume check exists in this funnel today. |
| Options liquidity | **Human-review information only** (already true, informational) | Explicitly demoted already: *"Contract quality (spread/liquidity/DTE/delta) is informational only... not an ENTER_NOW/promotion gate"* (candidates_router.py:2772-2778). Consistent with your instruction that options should never determine stock-setup validity. |

---

## 6. Recommended Top-10/15 queue architecture (design-level, not implemented)

Given everything above, the pieces already exist to build this with comparatively little new logic:

1. **Widen the eligibility gate** to only the genuinely structural conditions: regime stack, valid target exist. Drop R:R/entry-proximity/execution-shadow from *elimination* — reuse them as **ranking inputs** instead (§5).
2. **Compute a single Review-Value score** per surviving candidate — a transparent, disclosed, equal-weighted-or-simple composite (matching this codebase's existing "count over invented weight" philosophy from `confluence_summary.py`) over: confluence favorable/unfavorable counts (already exist), R:R (continuous, already exists), entry-distance-as-a-penalty-curve instead of a cliff (new, small), and a staleness/extension penalty (new — §4).
3. **Global rank by that score** — reuse `rank_stage1_candidates`'s existing shape (sort, then `enumerate` for `rank`), just fed a wider `ready` set and a richer sort key.
4. **Top 10 by default, expandable to 15** only when candidates 11–15 clear a real quality floor — implementable as a threshold on the same Review-Value score (e.g. "within X% of the #10 candidate's score" or "confluence favorable_ratio ≥ some floor"), computed at query time, not stored.
5. **Rank #25-beats-#5 is already structurally true** — `list_review_queue` recomputes the *entire* ranking fresh on every call (§3); nothing caches a stale Top-10 that a later-computed #25 couldn't unseat. The only thing that needs building is the truncation-with-expansion step itself.
6. **Previously-reviewed candidates already leave the "unreviewed" default view**: `public/review_queue.js`'s `state.filter = 'unreviewed'` / `firstUnreviewedIndex` (review_queue.js:188-194) already distinguishes reviewed from unreviewed client-side, keyed by `setup_key` and `current_review` (already attached server-side in `list_review_queue`, candidates_router.py:4812-4816). This mostly already satisfies "previously reviewed candidates should not continually consume New Review slots" — worth confirming with Erica whether the *symptom* she's worried about is actually about setup_key drift (a known, pre-existing, separately-documented issue — see `project_scanner_design.md`/`stock_scanner_gates.md` memory notes from earlier work this session) rather than the ranking model itself.
7. **Reject/Watch/Approve downstream behavior requires zero changes.** Reject already fully leaves the workflow (a `candidate_visual_reviews` row with `decision='reject'`, `_withdraw_active_memory_for_setup_key` if applicable). Watch already moves to lifecycle monitoring via the Watch Lifecycle V1 work earlier this session, resurfacing only on a real state change (`ENTRY_REACHED`, `WATCH_TRIGGER_SATISFIED`, etc.) — this is *already exactly* the "resurface only on meaningful state change" behavior requested. Approve already retains the frozen `approved_setup_memories` monitoring, entirely independent of the raw `candidates` table and its ranking. None of this touches the redesign.

---

## 7. Specific files/functions that would eventually need modification

*(For implementation — not touched in this audit.)*

- **`ma_pipeline.py`** — `_candidate_from_frames`: relax or remove the 4h EMA21 ≤3% elimination (currently the earliest, most invisible gate); decide whether the SMA50/200 stack stays a hard gate or becomes graded.
- **`candidates_router.py`**:
  - `_preview_clears_mechanical_prechecks` / `_preview_base_enter_now_ready` / `_stage1_mechanical_ready` — redefine which conditions eliminate vs. score. **Do this carefully alongside `_mechanical_promotion_block_reason`/`_promotion_block_reason`** (the separate ENTER_NOW-dashboard gate) so the two don't silently diverge further than they already have (§2's nuance).
  - `_confluence_sort_key` / `rank_stage1_candidates` — extend into the full Review-Value score; add the Top-10/expand-to-15 truncation step (currently no truncation exists at all).
  - `list_review_queue` — wire in whatever new staleness/extension signal gets built; this is also the natural place to decide the age-cutoff question from §4.2.
  - New: a staleness/extension scoring function (distance-beyond-entry, proximity-to-target) — no direct analog exists for raw candidates today; the closest prior art is the Approved/Watch monitor's `_current_rr`/`_execution_window_state` (built for a different table, reusable as a pattern, not directly).
  - `RANKING_MECHANISM_VERSION`/`RANKING_DISCLAIMER` — bump/update to reflect the new mechanism honestly, matching this codebase's existing convention of disclosing unvalidated logic rather than hiding it.
- **`public/review_queue.js`** — `loadQueue()`: if the stale-frontend-snapshot issue (§4.5) matters independently of the ranking redesign, this is a separate, much smaller fix (poll or a manual refresh affordance) worth considering regardless of the larger redesign's timeline.
- **`confluence_summary.py`** — likely reusable close to as-is; may need a version bump if its role expands from "descriptive" to "directly authoritative for ranking."

---

## 8. Risks / regressions this redesign could introduce

1. **Volume risk**: loosening entry-proximity/execution-shadow/R:R from gates to penalties could meaningfully grow the eligible pool beyond today's 13 (§1's near-miss data already shows ≥50 candidates one gate away) — the Top-10/expand-to-15 truncation (§6.4) is load-bearing to prevent "dump hundreds of tickers," and needs its own validation before shipping.
2. **Drift between the two gate systems** (§2's nuance): `_stage1_mechanical_ready` (review-queue) and `_mechanical_promotion_block_reason`/`_promotion_block_reason` (ENTER_NOW dashboard) already differ in one real way (confluence-conflicted). Changing one without deciding about the other risks the two dashboards disagreeing about the same ticker in confusing ways.
3. **No outcome evidence behind any of this** — the codebase's own `RANKING_DISCLAIMER` already says as much for today's mechanism; a new Review-Value composite score would be an equally unvalidated placeholder, just a different one. Same category of risk as everything else in this file already discloses; worth carrying the same honesty forward rather than presenting the new score as more validated than the old one.
4. **Twice-daily scan cadence limits how "extended" detection can ever work without also addressing staleness** (§4) — even a perfect ranking formula fed a stale `entry_price` will misjudge R:R/extension for however many hours have passed since the last `ma_pipeline` run. The redesign and the staleness fix are coupled; shipping one without the other risks half-fixing the exact symptom that motivated this audit.
5. **Frontend has no live-refresh mechanism** (§4.5) — a beautifully-ranked backend still won't help if Erica's open tab never re-fetches it. Worth deciding whether this is in scope for the redesign or a separate, smaller fix.
6. **Setup-key drift** — already a known, separately-documented pre-existing issue in this codebase (per earlier memory notes from this project). A wider eligible pool interacting with setup-key drift could surface it more visibly; worth being aware of, not necessarily worth fixing as part of this specific redesign.

---

## 9. Minimal implementation plan (described only — no code changes made)

1. Decide, explicitly, which of R:R/entry-proximity/execution-shadow move from gate → penalty (§5's recommendation, pending your sign-off) — this is a product decision, not an engineering one, and should happen before any code changes.
2. Build the staleness/extension signal (§4's "what would be required" list) as a pure, additive computation — no existing behavior changes yet.
3. Extend `_confluence_sort_key` into the full Review-Value composite, still gated exactly as today (no eligibility changes yet) — validates the new scoring math against the *existing* small pool first, cheaply.
4. Loosen the chosen gates to penalties (per step 1), re-run against real production data, and specifically check the resulting pool size against §1's near-miss baseline (≥50+13=63+ candidates) to calibrate the Top-10/expand-to-15 threshold realistically rather than guessing.
5. Add the Top-10/expand-to-15 truncation step.
6. Address the frontend staleness gap (§4.5) — separately assess whether this ships with the redesign or independently, since it's a much smaller, unrelated fix.
7. Update `RANKING_MECHANISM_VERSION`/`RANKING_DISCLAIMER` to describe the new mechanism honestly.
8. Full regression pass on the existing Approve/Watch/Reject downstream lifecycle (§6.7 already shows this needs no changes, but should be verified, not assumed, once real code changes exist).

No step above has been implemented. This document is the audit deliverable only.
