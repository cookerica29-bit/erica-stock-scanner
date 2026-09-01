# Kairos — Watch Lifecycle V1: Architecture Audit

**Status: audit complete, grounded in the exact deployed code (re-read fresh, not from memory). No code changed yet.**

## Determination

**No competing lifecycle or material Approved Setup Memory redesign is needed.** Every piece of Section 1's checklist reuses cleanly. One small additive schema extension is required (a `source_decision` discriminator), plus three real, disclosed, cross-cutting correctness gaps in the *shared* monitor primitives that block Watch from working correctly and also — quietly — affect the already-deployed Approve-Type-B path. Per your own branching instruction, proceeding directly into implementation; this report is the "chosen state model" writeup for the one schema change involved.

---

## 1. What's directly reusable, unchanged

- **Scheduler**: `register_background_periodic_task("approved_setup_monitor", MONITOR_TICK_SECONDS, ...)` — the same 5-minute in-process thread. No second monitor.
- **State enum**: `ApprovedSetupMonitorStateName` already contains `APPROVED, WAITING_FOR_TRIGGER, CONFIRMED, TRIGGER_SATISFIED, ACTIONABLE, STALE, WITHDRAWN, INVALIDATED, EXTENDED, SUPERSEDED` — **every semantic Section 5/13 asks for already has a name**. `WAITING_FOR_TRIGGER` *is* "WATCHING." No new state names needed anywhere.
- **Invalidation**: `_is_invalidated(direction, price, approved_stop)` — intrabar, frozen-stop, symmetric. Reused verbatim (Section 6).
- **R:R / execution window**: `_current_rr`, `_execution_window_state` — reused verbatim (Section 8), no Watch-specific variant.
- **Revision engine**: `_review_evidence_materially_differs` + the supersede-on-new-evidence flow in `_create_approved_setup_memory` — reused verbatim for Watch's setup_key-supersession rule (Section 10). Already keys on `(ticker, source)` + setup_key difference, not on decision — works identically for watch-originated memories with zero changes.
- **Event log**: `approved_setup_monitor_events` + `_record_monitor_event`'s dedup-on-change rule — reused verbatim for the `WATCH_*` event types (Section 14); only the `event_type`/`detail` values are new.
- **Queue-rotation independence (Section 12)**: already true by construction. `run_approved_setup_monitor_tick` reads `approved_setup_monitor_state`/`approved_setup_memories` directly — it has never touched `/candidates/review-queue`. Zero changes needed.
- **Completed-candle machinery**: `market_session`/`EASTERN_TZ` (from `market_data.py`, DST-correct via zoneinfo) — reused, but the function built on top of them has a real gap (Gap 2 below).

## 2. The one schema decision: `source_decision`, not a second table

Section 4/9 ask whether Watch needs its own memory table. It doesn't — `approved_setup_memories` already freezes exactly the fields Watch needs (`approved_stop`, `approved_target`, `approved_entry`, `direction`, `setup_key`, `visual_review_id`, `trigger_*`, `confirmation_*`) and the paired `approved_setup_monitor_state` row is already genuinely decision-agnostic. Renaming or forking the table would be the "meaningful redesign" this task says to stop for — and isn't warranted.

**Add one nullable column: `approved_setup_memories.source_decision TEXT`** (`Literal["approve", "watch"]`, defaulting to `"approve"` for every existing row via backfill-on-migration — see below), same additive `ALTER TABLE` pattern as every prior column in this table. This is the entire schema change.

- `source_decision="watch"` **is** the "machine lifecycle/handoff record" Section 9 asks for — no fabricated APPROVE row, `candidate_visual_reviews.decision` stays `"watch"` forever (append-only, untouched, exactly as it was written).
- `list_approved_setup_memories` needs no filter change — it already returns whatever's active, tagged now with real origin.
- **Existing rows** (FFIV/NVDA/CLH's backfilled memories, any real approval since) get `source_decision="approve"` via the migration — never guessed, since every row that exists today genuinely came from an approve.

## 3. `_sync_approved_setup_memory_on_review`'s dispatch — the actual gap

Confirmed by re-reading the live function: `decision in ("watch", "reject")` **only ever withdraws** an existing active memory — it never creates one. Watch today has **zero durable, frozen evidence** beyond the append-only `candidate_visual_reviews` row itself (which doesn't store `approved_stop`/`approved_target` at all — only the trigger/confirmation fields). This is exactly Section 4's suspected failure mode, confirmed real.

**Fix**: widen the `decision == "watch"` branch to mirror `"approve"`'s exact logic (same active-memory lookup, same `_review_evidence_materially_differs` revision check, same `_create_approved_setup_memory` call) **only when the review carries a complete trigger contract** (`trigger_rule` + `trigger_level` both present — `trigger_timeframe` defaults, same completeness rule already enforced at validation time). A `decision=="watch"` review *without* a complete trigger takes the **existing** withdraw-only path unchanged — correctly staying passive, no memory, no monitoring: "MANUAL REVIEW REQUIRED" (Section 3). `decision=="reject"` is untouched.

A fresh watch-with-trigger memory's initial `state` is decided by the exact same rule already in `_create_approved_setup_memory` (confirmation→`CONFIRMED`, trigger→`WAITING_FOR_TRIGGER`, neither→`APPROVED`) — for Watch, this always resolves to `WAITING_FOR_TRIGGER` in practice, since Section 3 scopes Watch monitoring to `lower_tf_confirmation="not_yet"` + a trigger, never a confirmation anchor (a "yes, already confirmed" read is a same-session-approve concern per the existing rule D, not a Watch one).

## 4. Gap 2 (real, cross-cutting): 30m completed-candle check has no RTH-window filter

`_last_completed_30m_bar` (re-read verbatim from the deployed code) only checks whether the **latest fetched bar is currently forming** — it never checks that the bar's own start time actually falls in 9:30–16:00 ET. When `market_session(now_et) == "closed"`, it returns `bars[-1]` **unconditionally**, with no check that `bars[-1]` is even an RTH bar at all. A completed pre-market (e.g. 8:00–8:30 ET) or after-hours (e.g. 16:30–17:00 ET) candle can satisfy a trigger today — directly violating Section 2's explicit contract ("Extended-hours/overnight/premarket candles MUST NOT satisfy a stock Watch trigger").

This function is shared by any Type B row, approve- or watch-originated. **Fixing it in place fixes both** — not a Watch-specific patch, a real correctness fix to the one shared primitive, disclosed here rather than silently bundled in.

**Fix**: new `_completed_rth_30m_bars(bars, now)` — filters to bars whose own ET start time is a valid RTH 30-minute slot (`09:30, 10:00, …, 15:30`, weekday only, aligned to the session open), independent of current session status, then excludes the single forming bar only when the market is currently open. Replaces `_last_completed_30m_bar`'s use inside the tick.

## 5. Gap 3 (real, cross-cutting): only the latest bar is ever checked, not "first satisfying"

Section 7 requires persisting the **first** satisfying candle's timestamp/close, and explicitly: "if the server was offline when the candle occurred, on restart it may discover the completed historical RTH candle and persist the original candle timestamp." The current tick only evaluates the single most-recent completed bar each cycle. Under normal 5-minute-tick / 30-minute-bar cadence this is harmless (at most one new bar appears between ticks, so "latest" and "first new" coincide) — but after a real outage spanning multiple bar completions, "latest completed bar" and "first bar that actually satisfied the rule" can be different bars entirely, and the wrong (later) timestamp/price would be persisted.

**Fix**: scan every completed RTH bar strictly after `last_evaluated_bar_time` (already a real column, already updated every tick) for the first one satisfying the rule, chronologically — not just the tail. For a **fresh** row (`last_evaluated_bar_time IS NULL`), keep the existing, deliberate first-tick behavior (check only the single latest completed bar, not the full history) — this preserves the already-agreed rule from the original design audit ("a real, valid signal shouldn't be missed on the first tick," without accidentally reaching back through the full 5-day fetch window for a review submitted moments ago).

## 6. Gap 4 (Section 11, deterministic recommendation): pre-trigger EXTENDED never evaluated

Confirmed: the tick's R:R/window/staleness block is gated by `if new_state in EVIDENCE_CLEARED_MONITOR_STATES` — a row still `WAITING_FOR_TRIGGER` that *didn't* just satisfy its trigger this tick never enters it. So today, a Type B setup (approve- or watch-originated) can sit in `WAITING_FOR_TRIGGER` indefinitely while price runs the entire distance to target and back, with **zero signal** that the opportunity is gone, right up until the moment the trigger (if it ever fires) immediately flips it to `EXTENDED` in the same tick via the *already-existing* override — the one piece of Option B this codebase already implements.

**Recommendation: Option A — evaluate the execution window for `WAITING_FOR_TRIGGER` rows too, every tick, not only after the evidence gate clears.** Reasoning: R:R is well-defined before a trigger fires (same formula, frozen entry/stop/target, doesn't require the trigger at all); Option B provides zero early signal and only reveals the setup was dead at the exact moment it would have mattered least; Option A matches the product goal stated twice in this task ("must not preserve WATCH indefinitely if the original execution opportunity is effectively gone") and costs nothing extra — it's the same `_current_rr`/`_execution_window_state` calls, just called one branch earlier. `STALE` stays scoped to evidence-cleared states only (staleness is specifically about "evidence existed, human never acted on it" — a concept that doesn't apply before the gate clears).

## 7. UI gap

`setup_board.js`'s `loadBoard()` only fetches `GET /candidates/approved-setup-memory` when `state.decision === 'approve'`; `stateBadge`, `stateNotice`, and `actionableBanner` all hard-return `''` when `state.decision !== 'approve'`. Every piece of Execution Layer display machinery is currently invisible on Watch Setups. Needs widening to run for both decisions, reading the new `source_decision`-tagged records the same way.

## 8. Scoping decision: handoff stays on the Watch Setups page for V1

Section 9 says the UI "may" surface a satisfied Watch on Approved/Execution Setups with a distinct origin label — phrased as optional, and Section 13's own UI spec describes "TRIGGER SATISFIED / HANDED OFF" as a **Watch Setups card state**, not a migration. Making a watch-originated `ACTIONABLE` memory *also* appear on the Approved Setups board would require changing that board's membership filter (currently `current_review.decision === 'approve'` off the live review-queue payload) — a separate, larger UI-membership decision. **Deferred for V1**: the full lifecycle, including trigger-satisfied/handoff and the explicit "Kairos is evaluating/monitoring this setup" copy, renders on the Watch Setups page itself. Flagging this explicitly rather than silently deciding it.

## 9. Section 3/13 "MANUAL REVIEW REQUIRED" needs no persisted state

A watch review without a complete trigger contract never gets a memory at all under the fix in §3 — so "MANUAL REVIEW REQUIRED" is computed purely in the UI from `current_review.decision === 'watch' && lower_tf_confirmation === 'not_yet' && (trigger_rule == null || trigger_level == null)`, exactly the same client-side pattern already used for the existing trigger/confirmation blocks. No new column, no new state.

---

Proceeding to implementation: schema (`source_decision`), the widened `_sync_approved_setup_memory_on_review` dispatch, the two monitor-tick correctness fixes (RTH-window filter, first-satisfying-bar scan) applied to the shared primitives, the pre-trigger execution-window evaluation, `WATCH_*` event types, Watch Setups UI, and full test coverage per Section 15 — before any push/deploy, per your explicit hold.
