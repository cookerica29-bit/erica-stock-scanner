# Kairos — Review Queue Evolution, Sprint 1: Implementation + Verification Report

**Status: implemented locally, fully tested, verified against real production data (read-only). NOT deployed. NOT pushed.**

---

## 1. Files / functions changed

**`candidates_router.py`**
- **New** (all additive, none replace existing functions):
  - `REVIEW_VALUE_*` constants (8 weights/caps) + `REVIEW_QUEUE_TOP_N`/`REVIEW_QUEUE_MAX_N`/`REVIEW_QUEUE_EXPANSION_MIN_RATIO`.
  - `RANKING_MECHANISM_VERSION_V2` / `RANKING_DISCLAIMER_V2` — the review-queue's new, honestly-labeled mechanism string (see the audit's own convention: never claim more validation than exists).
  - `_fresh_price_state(...)` — the fresh-price sanity layer (item 4).
  - `_review_value_score(preview, fresh)` — the transparent, additive scoring formula (item 3).
  - `_rank_review_queue_candidates_v2(rows)` — global sort + rank assignment, review-queue-only.
  - `_ensure_candidate_ranking_snapshots_schema(conn)` — additive migration (`ALTER TABLE ADD COLUMN`, same proven-safe pattern used throughout this project) for 7 new diagnostic columns on `candidate_ranking_snapshots`.
- **Rewritten**: `list_review_queue` (`GET /candidates/review-queue`) — steps 1–4 (regime alignment, batched daily-bar prewarm, base preview, `_preview_clears_mechanical_prechecks`, batched live-quote fetch) are **byte-identical** to before this sprint. Everything from live-quote attachment onward is new: entry-proximity/execution-shadow are attached but no longer eliminate; the fresh-price layer runs; Review Value scoring + global ranking replace `rank_stage1_candidates`; the Top-10/expand-to-15 cap applies only to not-yet-reviewed candidates; a `diagnostics` block is returned.
- **Extended**: `_store_ranking_snapshot` — now also persists `review_value_score`/`entry_proximity_ok`/`execution_shadow_ok`/`extension_penalty`/`target_room_penalty`/`staleness_penalty`/`included_in_display`, via `.get(...)` so the **unmodified** `GET /candidates/ranked` call site simply writes `NULL` for these (verified, not assumed — see §8).
- **Completely untouched**: `_stage1_mechanical_ready`, `_preview_base_enter_now_ready`, `rank_stage1_candidates`, `_confluence_sort_key`, `_mechanical_promotion_block_reason`, `_promotion_block_reason`, `_compute_candidate_promotion`, `_attach_entry_proximity`, `_attach_execution_shadow`, `_gate_gap_report`, `list_candidate_near_misses`, `GET /candidates/ranked`, `ma_pipeline.py` (in full), every Approved/Watch/Entry-Reached/journal/options/lifecycle function.

**`public/review_queue.js`** — `loadQueue()` now accepts `{silent}`; a silent call updates `state.queue`/`state.diagnostics`/`state.lastLoadedAt` and the freshness bar **only**, never touches `#mainContent`. New: `renderFreshnessBar()`, `relativeTimeLabel()`, `manualRefreshQueue()`, `startAutoRefresh()`/`stopAutoRefresh()`. `DOMContentLoaded` now also calls `startAutoRefresh()`.

**`public/review_queue.html`** — new `#freshnessBar` element (outside `#mainContent`, deliberately) + its CSS. Script cache-buster bumped.

**New tests**: `tests/review_queue_evolution_sprint1_v1.py` (21 tests), `tests/review_queue_freshness_v1.js` (6 test groups).

---

## 2. Exact Review Value formula

Fully transparent, additive — every term is a named, returned field (`review_value_breakdown`):

```
score = favorable_points + unfavorable_points
      + entry_proximity_points + execution_shadow_points
      + rr_points
      + extension_penalty + target_room_penalty + staleness_penalty

favorable_points        = +2.0  x confluence favorable_count
unfavorable_points      = -2.0  x confluence unfavorable_count
entry_proximity_points  = +1.5  if entry_proximity_ok else 0
execution_shadow_points = +1.5  if execution_shadow_ok else 0
rr_points                = 0.5  x min(risk_reward, 5.0)          [scan-time R:R, unchanged/trusted]

extension_penalty   = clamp(-1.0 x (extension_atr - 0.5xATR_threshold), floor -4.0)   [only beyond the existing proximity threshold]
target_room_penalty = clamp(-1.5 x (1.5 - current_rr_live),            floor -4.0)   [live price vs frozen stop/target]
staleness_penalty   = clamp(-0.05 x (candidate_age_hours - 8h),        floor -3.0)   [grace period ~ one scan cycle]
```

All eight weights/caps are disclosed, unvalidated placeholders — same category as every other threshold in this codebase (`RR_WARNING_THRESHOLD`, `ENTRY_PROXIMITY_MAX_PCT_DEFAULT`, `CONFLICTED_UNFAVORABLE_MIN`). Not tuned against outcome data; revisit once real usage exists.

---

## 3. Former gates → ranking evidence (this sprint)

| Was a hard gate | Now | Reused mechanism |
|---|---|---|
| `entry_proximity_ok` | Ranking bonus (+1.5) or absence (0) | `_attach_entry_proximity` — unchanged |
| `execution_shadow_ok` | Ranking bonus (+1.5) or absence (0) | `_attach_execution_shadow` — unchanged |

**Deliberately deferred, not touched this sprint**: `ma_pipeline.py`'s 4H EMA21≤3% candidate-**generation**-time gate. A ticker failing it never becomes a `candidates` row at all — reclassifying it would mean either duplicating candidate generation or changing behavior for every consumer of the `candidates` table (near-miss, the generic `/candidates` list), not just the review queue. This is explicitly out of scope per "do not globally remove or rewrite... unless necessary," disclosed here rather than silently skipped.

---

## 4. Hard gates that remain (unchanged)

Via `_preview_clears_mechanical_prechecks`, byte-identical to before this sprint:
- Direction (`long` only; shorts excluded, unchanged)
- Regime alignment
- Data availability (daily bars, ATR14)
- Valid structural target must exist
- **Scan-time R:R ≥ 1.5** — kept conservative exactly as instructed, since the audit found it depends on a twice-daily-stale entry price. Not loosened this sprint.

**New this sprint**: `target_reached_or_passed` — a candidate whose **live** price has reached or passed its frozen target is now hard-excluded (not merely penalized), on the reasoning that the opportunity is definitionally over, not a "score it lower" situation. Computed by the fresh-price layer, from the *same* already-fetched live quote — no new network call.

---

## 5. Fresh-price / staleness behavior

`_fresh_price_state` recomputes, per eligible candidate, from the **already-fetched** live quote (no new fetch, no swing/ATR/structure rebuild):
- `current_price` (already existed, via `_attach_entry_proximity`)
- `current_rr_live` — R:R using live price as hypothetical entry against the frozen stop/target (mirrors the existing, already-proven `_current_rr` direction math from the Approved/Watch monitor — reused, not re-derived)
- `target_reached_or_passed` — hard exclusion (§4)
- `extension_atr` — distance past entry in ATR terms, penalized only beyond the *existing* proximity threshold (no new magic number)
- `candidate_age_hours` — from `scanned_at`, penalized past an 8-hour grace period (~one `ma_pipeline` scan cycle)

**Real production confirmation** (§7): a real candidate (GM) was demoted from the display list via a live-price-driven `target_room_penalty`, using data fetched during this verification, not a synthetic test.

---

## 6. Queue-cap behavior

- Ranking is **global**, over the full post-exclusion eligible pool, before any truncation.
- Top 10 by default among candidates with `current_review is None` ("needs review").
- Slots 11–15 fill only if that candidate's score is **both** net-positive **and** ≥ 75% of the #10 candidate's score — proven with real data (§7: 31 eligible candidates, #10 scored 8.69, #15 scored 8.135, well within the 75% floor, correctly expanded).
- **Never force-filled**: `test_never_force_fills_below_ten` proves fewer than 10 renders as-is, no padding.
- **Never exceeds 15**: `test_never_exceeds_fifteen` proves a 20-candidate pool still caps at ≤15.
- **Previously-reviewed candidates never consume a new-review slot**: excluded from the cap's counting entirely, but still returned in the response (so the existing edit/re-review UX in `review_queue.js` keeps working) — proven end-to-end: approving the #1-ranked candidate out of 11 correctly lets an 11th still-needs-review candidate take its place (`test_reviewed_candidates_do_not_consume_new_review_slots`).

---

## 7. Before/after — real production data (read-only, no deployment)

Verified against **60 real, current production candidates** (pulled live from `/api/v1/scanner/candidates`, scanned today), replayed through the new local code with **real live market data** (real Alpaca credentials via `railway run`, against a local scratch DB — production database never touched, never written to).

```
total_candidate_count:              60
eligible_pool_count:                35   (was capped to whatever cleared entry_proximity+execution_shadow before)
excluded_mechanical_count:          25   (direction/regime/target/scan-time R:R -- unchanged hard gates)
excluded_target_reached_count:       0   (none of this sample happened to be at/past target right now)
promoted_from_near_miss_count:      31   <- of the 35 eligible, 31 would have been INVISIBLE before this sprint
needs_review_total_count:           35
needs_review_displayed_count:       15   (expanded past 10 -- #11-15 cleared the 0.75 ratio floor)
top_cutoff_score:                8.135
```

Displayed scores, rank 1→15 (real data): `11.055, 10.97, 10.5, 10.39, 10.345, 10.325, 10.0, 9.425, 9.24, 8.69, 8.685, 8.575, 8.475, 8.35, 8.135` — **strictly non-increasing**, confirming a genuine global sort, not insertion/first-pass order.

**A real "promoted from near-miss" example**: COF ranked **#1** with `execution_shadow_ok=False` — before this sprint, COF would not have appeared in the review queue at all despite being the single highest-scoring candidate in the entire real sample.

**A real "demoted/excluded appropriately" example**: **GM**, a real production candidate, ranked **#34 of 35** eligible (score 2.361, `target_room_penalty=-0.2` from live price vs. its frozen stop/target) and was correctly **excluded from the displayed Top-15** (`included_in_display=0` in the snapshot table) — the demotion mechanism caught a real, live case, not just a synthetic test.

**"Fewer than 10 can be returned"**: not observed in this particular 60-candidate real sample (35 eligible is comfortably above 10) — proven instead with controlled test data (`test_never_force_fills_below_ten`, 4 seeded candidates → 4 returned, no padding). Today's real production pool is large enough that a real <10 example wasn't available at verification time; the code path is identical regardless of pool size.

**Downstream unaffected**: not exercised in this specific real-data run (fresh scratch DB, no existing reviews) — confirmed instead two other ways: (a) the dedicated regression test above proves the mechanism end-to-end with a real Approve action through the new code path, and (b) the diff itself touches zero Approved/Watch/Entry-Reached/lifecycle/journal/options files (`git status --short` shows only `candidates_router.py`, `public/review_queue.html`, `public/review_queue.js`, and new test files).

---

## 8. Regression results

- **New Sprint 1 backend suite**: `tests/review_queue_evolution_sprint1_v1.py` — **21/21 passing**. Covers reclassification, all preserved hard gates, the new target-reached exclusion, the formula (hand-computed spot-check), staleness, global-sort proof, never-force-fill, caps at 10/15 with both a weak-11th-excluded and a strong-11th-included real scenario, reviewed-candidates-don't-consume-slots, diagnostics consistency, and `GET /candidates/ranked` unaffected.
- **New frontend suite**: `tests/review_queue_freshness_v1.js` — **6/6 passing** (found and fixed one real bug during this work: `manualRefreshQueue()` was missing a `return`, so `await`ing it didn't actually wait for the fetch — caught by the test, not shipped).
- **Existing review-queue suite** (`tests/review_queue_v1.py`, 15 tests) — **all still pass, unmodified** — confirms option-hydration exclusion, deferred-preview flag, short exclusion, setup_key/current_review binding, snapshot writing, cache reuse, funnel-gating order, and batched-quote-fetch behavior are all still exactly as before.
- **Full backend suite**: `python3 -m pytest tests/ -q --override-ini="python_files=*_v1.py *_v2.py *_v3.py" --ignore=tests/production_refresh_safety_v1.py` → **895 passed** (was 874 before this sprint), same **8 pre-existing, unrelated failures** present since before this sprint (alpaca discovery universe env-parsing, discovery startup registration, opportunity-ranking priority buckets, option-pricing retry classification, scanner-health cache key, verified-analytics replay parity).
- **Full frontend suite**: `review_queue_auth_v1.js`, `review_queue_persistence_v1.js`, `review_queue_freshness_v1.js`, `setup_board_v1.js`, `setup_board_selfinit_v1.js` — all passing.

---

## Notes / open items for your review before deploy

1. **The 4H EMA21≤3% generation-time gate is explicitly deferred** (§3) — not reclassified this sprint. Flag if you want this addressed in Sprint 2.
2. **R:R stays conservative this sprint** (§4) — scan-time R:R≥1.5 is still a hard gate; the fresh-price layer adds live-price-aware penalties/exclusion on top, but doesn't loosen the scan-time gate itself.
3. **All 8 Review Value weights and both cap constants (10/15, 0.75 ratio) are disclosed placeholders** — directionally sensible, transparently explainable, not outcome-validated. Same honesty convention as the rest of this codebase.
4. **Auto-refresh cadence is 60 seconds** — a judgment call balancing freshness against quote/scanner load; easy to tune (`REVIEW_QUEUE_AUTO_REFRESH_MS`) if you want it faster/slower.
5. Real verification used a 60-candidate sample of today's real production pool, not the full ~450 — chosen for speed; the code path is identical regardless of sample size, and the pool's actual size (35 eligible from 60 sampled) suggests the full pool would show materially more eligible candidates, all going through the same tested cap logic.

**Holding deployment for your review, per explicit instruction.**
