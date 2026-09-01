# Kairos — Execution Layer V1 Design Audit

**Status: design/audit only. No code changed. Nothing pushed or deployed.**
Grounded entirely in the real codebase (`candidates_router.py`, `main.py`, `scanner.py`, `market_data.py`, `ma_pipeline.py`) as it exists today, plus one read-only production query (via `railway run`, real key never seen by the assistant) against the three live `approved_setup_memories` rows. No production writes were made.

---

## 1. Current architecture trace

Two **separate, parallel** human-approval pipelines exist in this codebase today. Execution Layer V1 is about the second one only, and must never touch the first.

**Pipeline 1 — legacy "ENTER_NOW" dashboard** (`/candidates`, Candidates Dashboard):
`candidates` row → `_compute_candidate_promotion` → `_promotion_block_reason` (hard-blocks **all** short candidates, then layers mechanical + confluence-conflict checks on longs) → `PATCH /candidates/{ticker}` with `status=active` → `candidate_promotions` (append-only, has `taken`/`outcome*` columns for a separate outcome-resolver pipeline). No visual review is required anywhere in this path.

**Pipeline 2 — Review Queue → Approved Setups** (what this audit is about):
```
candidates (ma_pipeline ingestion, entry_price = latest 4H close)
  → GET /candidates/review-queue
      Gate 1: regime alignment (candidates query itself is WHERE signal='long')
      Gate 2: batched daily-bar prewarm → _compute_review_queue_base_preview
              (= _compute_candidate_promotion: stop/target/R:R/ATR14/BOS/
               displacement/macro/sweep/rejection/location, all daily-bar-derived)
      Gate 3: _preview_clears_mechanical_prechecks
              (direction MUST be "long" — hardcoded — + regime + valid target + R:R≥1.5)
      Gate 4: batched live quote → _attach_entry_proximity (directionless, live-drifting)
      Gate 5: _attach_execution_shadow (4H-bar bullish-continuation check, LONG-shaped logic)
  → human visual review → POST /candidates/{ticker}/visual-review
      → candidate_visual_reviews (append-only, immutable per row, "latest wins")
      → _sync_approved_setup_memory_on_review
           approve  → _create_approved_setup_memory (frozen snapshot + paired
                       approved_setup_monitor_state row, state='APPROVED', in one txn)
           watch/reject → withdraws any active monitor_state (→ 'WITHDRAWN')
  → Approved Setups / Watch Setups boards (public/setup_board.js)
      fetches the SAME GET /candidates/review-queue payload again, filters
      client-side on current_review.decision, computes WAITING_FOR_ENTRY /
      EXTENDED / INVALIDATED **client-side from the LIVE candidate fields**
      (see Finding A below — this does NOT read approved_setup_memories at all)
```

**Finding A (current, live, real gap):** `GET /candidates/approved-setup-memory` exists and is fully implemented, but **nothing in the frontend calls it.** `public/setup_board.js`'s `computeDisplayState()` computes WAITING/EXTENDED/INVALIDATED from `item.entry_price` / `item.stop` / `item.current_price` — all **live, re-computed-every-request** fields off the current `candidates`/`candidate_plan_previews` row, not the frozen `approved_setup_memories.approved_stop`. Today this happens to still be correct for FFIV/NVDA/CLH only because their live stop hasn't drifted from their approved stop since approval (all three were approved within the last session, same setup_key still current). It is a real bug-in-waiting: if a rescan ever produces a different order-block stop for the same setup_key window, INVALIDATED would be computed against the wrong number. **This must be fixed as part of Execution Layer V1, not left as-is** — see §17 item 1.

The trigger-capture feature (previous task) added `trigger_timeframe/rule/level/reason` to both `candidate_visual_reviews` and `approved_setup_memories`. Nothing monitors it yet — that is exactly this audit's subject.

---

## 2. Trustworthy fields vs. unsafe fields

| Field | Source | Timeframe | Historical / Live | Frozen / Mutable | Exact / Reconstructed | Safe for gating? |
|---|---|---|---|---|---|---|
| `candidates.entry_price` | `ma_pipeline._candidate_from_frames`: `close_4h.iloc[-1]` | 4H | Live | Mutable — recomputed every scan | Exact | **No** — drifts every scan, never an authoritative trigger (confirmed prior audit) |
| `candidates.daily_regime` | same ingestion, `signal` value copied verbatim | Daily | Live | Mutable | Exact | Yes, as a coarse long/short gate only |
| `preview.stop` / `preview.target` | `_compute_candidate_promotion`: order-block / structural-swing levels off **daily** bars | Daily | Live-recomputed but structurally stable (only changes on genuine swing/OB shift — verified empirically, 4 live pulls, byte-identical) | Mutable in `candidates`/`candidate_plan_previews`; **frozen** once copied into `approved_setup_memories.approved_stop/approved_target` | Exact (approval_event) / exact-by-construction (live_backfill — setup_key match guarantees stop/target equality) | Yes — this is the correct invalidation anchor once frozen (see §5) |
| `preview.atr14` | `_compute_atr(df, 14)`, daily | Daily | Live | Mutable | Exact | Usable, but **not currently persisted into `approved_setup_memories`** — see Finding B |
| `entry_proximity_ok` / `entry_distance_pct` / `entry_proximity_threshold_pct` | `_entry_proximity`: live quote (Alpaca IEX, with one-sided-quote → daily-close fallback) vs. **live** `entry_price` | Tick / daily-close fallback | Live | Mutable | Exact, but **directionless** (`abs(current − entry)`) | Not reusable as-is for Execution Layer — tuned for a different question (see §6) |
| `execution_shadow_ok` | `_execution_shadow_from_bars`: bullish-continuation test over last 15×4H bars, 75s TTL cache | 4H | Live | Mutable | Exact | **Not** the human's lower-TF confirmation — direction-blind (bullish-shaped test with no direction parameter — harmless only because direction is hard-gated to "long" upstream) and must never be conflated with `lower_tf_confirmation` (explicit invariant, §15) |
| `candidate_visual_reviews.lower_tf_confirmation` / `trigger_*` | Human, append-only, "latest row wins" | N/A (human judgment) | Historical | Immutable per row | Exact for rows written after trigger-capture shipped; `NULL` (never fabricated) for older rows | Yes — this is the one human-authored, trustworthy signal in the whole system |
| `approved_setup_memories.*` (approval_event) | Copied verbatim from `preview` + review at the approval instant | Mixed (daily for stop/target, live-tick for price-at-approval) | Historical | **Immutable, never UPDATEd** | Exact | Yes |
| `approved_setup_memories.*` (live_backfill — FFIV/NVDA/CLH) | Reconstructed at backfill time | same | Historical (but not the true approval moment) | Immutable | `approved_stop`/`approved_target`/`setup_key` exact by construction; `approved_entry`, `current_price_at_approval`, every scanner-evidence field **reconstructed, not real approval-moment truth** (`snapshot_exact=false`, `backfill_note` spells this out) | Stop/target: yes. Everything else: display/context only, never a gate |
| `approved_setup_monitor_state.*` | Written by `_create_approved_setup_memory`, one row per memory | N/A | Live status | **Mutable, in place** — this is the one table designed to be written by a future monitor | Exact | This is the correct home for Execution Layer's live verdict — currently unused by any consumer |
| `candidate_promotions.*` | Legacy ENTER_NOW pipeline | N/A | Historical | Append-only | Exact | **Never** read or written by Execution Layer (Pipeline 1, separate concern) |

**Finding B:** `approved_setup_memories` does not store `atr14`. `_create_approved_setup_memory`'s INSERT list has no ATR column, even though `preview["atr14"]` is available at approval time and simply not copied over. If an ATR-anchored execution window is chosen (§6, option freezing ATR-at-approval), this is a one-column additive gap, not a redesign — flagged as an open question rather than assumed.

---

## 3. Type A semantics (confirmation already "yes")

Real evidence from the only two live Type A memories (NVDA, FFIV):

- NVDA's `review_note`: *"Bullish HTF structure; 30M bullish CHoCH after correction. Neutral entry location with overhead structure near 227–230."* — a real, specific chart read, but **zero objective anchor**: no price level, no candle timestamp, nothing machine-checkable was ever recorded.
- FFIV's `review_note` is `null` — even less to go on.
- Both approvals happened **minutes** before this audit (same session) — real time-decay cannot be empirically demonstrated on this data, but the design must still hold generally, since a real trader will reopen Approved Setups tomorrow or next week.

This directly explains the four options in the brief:

- **A (immediately eligible forever, gated only on thesis/price/R:R)** — matches nothing about "we cannot allow a week-old YES to create a fresh signal." Rejected.
- **B (require a fresh execution event after approval)** — effectively converts every Type A into Type B retroactively, contradicting the premise that the reviewer already saw the confirmation.
- **C (bounded actionable window, N completed bars/days after approval)** — respects that the human's "yes" IS real evidence, but only for a bounded time, without requiring a UI change.
- **D (force a reference level/event even on "yes")** — the most rigorous, but is a real product-friction decision (do you make every reviewer type a price even when they just want to say "I saw it confirm") that this capture-only session explicitly declined to force for NOT_YET reviews, and which this audit should not silently decide either.

**Recommendation: C, with D flagged as the open question for the user (see §18).** Treat a Type A "yes" as real evidence that decays: from `approved_at`/`reviewed_at`, the memory is eligible to read ACTIONABLE (subject to the same current-R:R/execution-window gate as Type B) for a bounded reconfirmation window; once that window elapses without the human ever entering, the memory falls to a new `STALE` sub-state (not INVALIDATED — the thesis was never contradicted, just unconfirmed for too long) that requires a fresh human review/trigger to reactivate. Window length is an explicit, unvalidated placeholder (propose 3 trading days, or N completed 4H bars, same honesty-about-thresholds convention this codebase already uses everywhere) — not backed by evidence, flagged for revisit.

---

## 4. Type B semantics

Straightforward given the trigger is already captured and validated:

1. Each monitor tick, for every **active** memory with a non-null `trigger_level`, fetch the latest completed 30m bar(s) for its ticker.
2. On the first completed 30m close satisfying `trigger_rule` against `trigger_level`, persist satisfaction **exactly once** (§8) — this is a historical fact about one specific bar, not a live condition re-derived each tick.
3. In the **same tick**, evaluate the execution window / current R:R (§6/§7) — a setup can go straight from WAITING to EXTENDED without ever resting in ACTIONABLE if price already blew through the window by the time the qualifying candle closed.
4. Apply the same freshness/decay concept as Type A (§3) to a satisfied-but-unacted-on trigger, so the two paths converge into one `STALE` concept rather than diverging designs.

---

## 5. Invalidation design

Three candidate rules, evaluated on their tradeoffs (not intuition):

| Rule | Speed | False-positive risk | Consistency with existing code |
|---|---|---|---|
| Intrabar breach of stop | Immediate | Real (a stop-hunt wick that reverses same-bar would trip it) | Matches what a **real broker stop order** does; matches `computeDisplayState`'s existing (if wrongly-sourced) precedent |
| Completed 30m close through stop | Up to 30 min lag | Lower | Matches the trigger's own "completed candle" philosophy |
| Completed 4H close through stop | Up to 4h lag | Lowest | Matches `execution_shadow`'s own timeframe and the daily/4H structural origin of the stop itself |

**Recommendation: intrabar breach of `approved_stop`, using the same live-quote-with-daily-close-fallback price source `_entry_proximity` already trusts.** The reasoning is an asymmetry the brief doesn't ask for but matters: for a **confirmation** signal, a false positive means a bad entry (expensive), so "completed candle only" is the right conservative choice — this is exactly why the trigger-capture task rejected wick-based confirmation. For **invalidation**, the failure directions are reversed: a false-positive invalidation just means a good setup gets prematurely marked dead and the human has to re-review (annoying, never dangerous); a false-negative (still showing tradeable after a real stop-out) is the dangerous direction. A real stop-loss order does not wait for a candle to close, so neither should Kairos's invalidation read. Symmetric and deterministic: LONG invalidated when `price ≤ approved_stop`; SHORT mirrors (`price ≥ approved_stop`) — see §15 for why SHORT is unreachable in production today regardless.

**Concrete fix required regardless of which rule is chosen:** read `approved_stop` from `approved_setup_memories`, never the live/recomputed `stop` — this corrects Finding A.

---

## 6. Acceptable execution window — alternatives

Existing `entry_proximity` (1.5% / 0.5×ATR, `_entry_proximity`) is **not reusable as-is**: it's keyed to the live-drifting `entry_price`, it's directionless (treats "ran away favorably" and "moved adversely" identically), and it was tuned for a completely different question (dashboard ENTER_NOW eligibility off a fresh 4H close), never validated against "does entering now still honor the original R:R."

Alternatives audited:

- **Frozen `approved_entry` ± pct/ATR** — better anchor (doesn't drift), but still needs to be made **directional**: for a LONG setup, price sitting *below* `approved_entry` (better fill, better R:R) is not "extended" in any sense that should block a trade — EXTENDED is inherently a one-sided concept (too far in the *chase* direction), not a symmetric distance-from-entry band like today's proximity check.
- **`trigger_level` as the anchor** — appealing for Type B (directly answers "still near where the human said to act"), but has no equivalent for Type A, which has no stored level (same tension as §3's Option D).
- **Risk-distance-based (R-multiples of `approved_entry → approved_stop`)** — ties the window's size to *this setup's own risk geometry* rather than a flat percentage tuned for something else. More setup-specific than a flat %/ATR figure.
- **Current-price R:R (§7)** — the literal, direct answer to "does entering now still make sense" — this is the actual question the brief poses, not a proxy for it.

**Recommendation:** gate ACTIONABLE primarily on **current-price R:R** (§7) — it's the only option that directly answers the stated question and needs no new frozen anchor beyond what's already frozen (`approved_stop`, `approved_target`). Use a distance-based check (favorable-direction-only, risk-distance-scaled) only as a **cheap pre-check / explanatory UI copy** ("price is now 1.2R beyond your planned entry"), not a second independent gate that could disagree with the R:R verdict — two gates that can produce contradictory answers is exactly the kind of ambiguity §1 says must not exist.

---

## 7. Current-price R:R design

```
LONG:  risk = current_price − approved_stop     reward = approved_target − current_price
SHORT: risk = approved_stop − current_price     reward = current_price − approved_target
current_rr = reward / risk
```

This is structurally identical to `_compute_candidate_promotion`'s own `risk_reward = reward / risk` pattern, just evaluated against the frozen stop/target and the live price instead of the frozen entry.

**Minimum threshold: reuse `RR_WARNING_THRESHOLD = 1.5`**, the same constant Stage 1 candidate qualification and the target-clamp's `min_viable_rr` already use. This is "most semantically consistent" per the brief's own framing — introducing a second, different number for the same underlying question ("is this R:R viable") would imply two different definitions with no evidence either should differ. Like every other threshold in this codebase, 1.5 remains unvalidated/un-backtested — that status doesn't change here, it's just not a reason to invent a *new* unvalidated number instead of reusing the one already in production use.

**Real current examples** (live read-only production data, no writes, pulled during this audit):

| Ticker | Current price | Approved stop | Approved target | Current R:R |
|---|---|---|---|---|
| NVDA | 218.84 | 216.0144 | 235.4644 | **5.88** |
| FFIV | 403.84 | 392.7308 | 433.6558 | **2.68** |
| CLH | 315.25 | 307.9376 | 328.0376 | **1.75** |

All three currently clear 1.5 comfortably — useful as a sanity check that the formula and threshold aren't obviously broken on real numbers, not as a validation of the threshold itself.

---

## 8. Trigger-satisfaction semantics

- **Completed candles only:** `market_data.py` already has `_is_forming_candle(timestamp, interval, timing_context)` + `_interval_timedelta` + `validation_timing_context` — session-aware (treats everything as completed once market is closed), already generalizes to `"30m"`. **Reusable, but currently only exercised in a diagnostics/comparison path (`_provider_diagnostics`), never in a live gating decision** — needs its own dedicated test coverage before being trusted here, not assumed correct by association.
- **Timezone/session handling:** Eastern Time, via the same `market_session`/`validation_timing_context` helpers already used elsewhere — no new convention needed.
- **Duplicate candle handling:** persist the timestamp of the last **evaluated** bar per memory (new column, §10) so a tick that re-observes the same still-latest completed bar (normal between candle closes) never re-fires a transition.
- **First satisfaction timestamp:** persist the timestamp of the *specific bar* whose close satisfied the rule, not "now" — the server may observe it up to one tick-interval late; the recorded fact should be about the market, not about Kairos's polling schedule.
- **Gap behavior:** needs no special handling — checking each newly-completed close against `trigger_level` naturally satisfies the rule the instant a gap-bar's close qualifies, exactly per the semantic contract (close-based, not path-based).
- **Trigger already satisfied before monitor starts:** the first tick for a freshly-active memory must evaluate the **latest already-completed bar**, not wait for a brand-new one — otherwise a real, already-true signal is silently missed forever. Explicit design decision, not an edge case to special-case away.
- **Trigger satisfied while server was offline:** a non-event by construction — satisfaction is derived from real historical completed bars fetched fresh every tick (no in-memory streaming state), so a restart simply resumes from "what does history say" on its next tick. This is the concrete reason a stateless poll-recent-history design is safer here than a live tick/websocket stream.
- **Trigger satisfied then reversed before next check:** per the semantic contract itself, satisfaction is a **historical fact about one completed bar** — once true, it stays true even if price later reverses. This is exactly why it must be persisted the moment it's observed, not re-derived by checking "is price currently beyond the level" (which would silently un-satisfy it on reversal — a real violation of the human's own stated contract).
- **Stale trigger:** same decay concept as §3 — a trigger satisfied long ago without action should fall into `STALE`, unifying Type A/B under one freshness rule.
- **Setup invalidated before trigger:** invalidation (§5) must be checked every tick independent of trigger logic, and once a memory's `state` is terminal, it must be excluded from the trigger check entirely — this is exactly what `ACTIVE_MONITOR_STATES` already exists to express; the monitor should only ever iterate rows in that set.
- **Setup extended before trigger:** real and possible (a gap can jump price past both the trigger and the execution window in one move) — handled naturally by evaluating the window check in the same tick as satisfaction (§4 step 3), so a setup can go straight from WAITING to EXTENDED, satisfying the invariant "never call an extended setup ACTIONABLE."

---

## 9. State machine

`ApprovedSetupMonitorStateName` (already defined in `candidates_router.py`) reserves `APPROVED, WAITING_FOR_TRIGGER, WITHDRAWN, INVALIDATED, EXTENDED, SUPERSEDED` — explicitly documented as agreed in a **prior design-only session**, with only `APPROVED/WITHDRAWN/SUPERSEDED` actually wired today. This audit continues that vocabulary rather than inventing a new one, and adds what's missing: `ACTIONABLE`, `TRIGGER_SATISFIED`, `STALE`.

```
APPROVED  (== product-facing "WAITING"; both Type A and Type B start here)
   │
   ├─ Type B: 30m trigger closes per trigger_rule ───────► TRIGGER_SATISFIED
   │                                                              │
   ├─ Type A: no trigger stored, evaluated on thesis+price ──────┤
   │          directly, no separate TRIGGER_SATISFIED hop        │
   │                                                              ▼
   │                                             execution-window check (every tick)
   │                                                    ┌─────────┴─────────┐
   │                                                    ▼                   ▼
   │                                              ACTIONABLE            EXTENDED
   │                                              (current R:R OK)   (window blown)
   │
   ├─ reconfirmation window elapses w/o action ──────────► STALE (needs fresh review)
   │
   ├─ approved_stop breached, any active state ──────────► INVALIDATED (terminal)
   ├─ human re-reviews to watch/reject ───────────────────► WITHDRAWN (terminal)
   └─ a newer setup_key gets its own active approval ─────► SUPERSEDED (terminal)
```

**Should `TRIGGER_SATISFIED` be its own persisted state, or evidence only?** Recommend **its own real, persisted state** (not evidence-only with ACTIONABLE fully derived), for three concrete reasons: (1) it's the natural place to anchor the "persisted exactly once" satisfaction timestamp from §8; (2) it gives the alert contract (§12) a real, distinct transition to fire "your trigger just fired" separately from "you can act now," which the brief's own example alert text implies matters even if the UI visually collapses the two; (3) it makes the "extended before trigger" edge case (§8) a real, auditable sequence instead of an invisible internal computation.

The practical consequence: `state` holds the **coarse, sticky lifecycle** (`APPROVED → TRIGGER_SATISFIED → {terminal}`), while the **current window verdict** (`ACTIONABLE` vs `EXTENDED`) is a separate, freely-overwritten field re-evaluated every tick once `TRIGGER_SATISFIED` (Type B) or fresh-enough evidence (Type A) is true — it is not itself a one-way state transition, since price can legitimately move a setup back and forth between "still within window" and "now extended" as it fluctuates. This keeps the lifecycle history meaningful (a setup doesn't "un-satisfy" its trigger) while letting the live verdict fluctuate honestly.

Restart-safety: since every transition is derived from a fresh DB read + fresh market data each tick (never in-memory-only), a restart is a non-event — this is already server-authoritative by construction, no extra work needed beyond writing to the DB row that already exists.

---

## 10. Persistence changes required

- `approved_setup_memories`: **optional** new nullable column `atr14_at_approval REAL` — only needed if an ATR-anchored window (§6) is chosen over the R:R-primary recommendation. Flagged as an open question, not assumed.
- `approved_setup_monitor_state`: add nullable columns —
  - `execution_window_status TEXT` (`'ACTIONABLE' | 'EXTENDED' | NULL`) — the freely-overwritten live verdict from §9
  - `trigger_satisfied_at TEXT`, `trigger_satisfied_bar_time TEXT`, `trigger_satisfied_price REAL` — frozen once written (§8)
  - `last_evaluated_bar_time TEXT` — the dedup/idempotency anchor (§8)
  - `current_rr_at_last_check REAL` — reused for both UI display and alert bodies without recomputing
  - Extend `ApprovedSetupMonitorStateName` with `ACTIONABLE`, `TRIGGER_SATISFIED`, `STALE`
- **New table** `approved_setup_monitor_events` (append-only) — one row per **meaningful transition** (not per tick), used for alert dedup (§12) and as a real audit trail of what Kairos told the user and when. Kept separate from `monitor_state` for the same B/C reason (evidence vs. current-status) the codebase already used to justify `approved_setup_memories` vs. `approved_setup_monitor_state` as two tables instead of one.
- No changes needed to `candidate_visual_reviews`, `candidates`, or `candidate_promotions`.

All additive, nullable, `ALTER TABLE ADD COLUMN`-if-missing — same proven-safe pattern used for every schema change so far this session.

---

## 11. Monitor architecture

**Reuse the existing in-process mechanism — do not build new infrastructure.** `main.py` already runs a single daemon thread (`_background_refresh_loop`, started once at `@app.on_event("startup")` via `start_market_cache_refresh()`), ticking every 30s, checking each registered task's own TTL via `_periodic_refresh_due`, and running registered callbacks **synchronously inline** (not via the thread pool) — this is the exact same pattern `candidate_promotion_outcome_watcher` already uses today for real, batched-market-data-consuming periodic work, registered via `register_background_periodic_task(key, ttl_seconds, callback)`.

- **Cadence:** register `"approved_setup_monitor"` at roughly a 5-minute TTL — checking every 30s would be wasteful against a bar that only closes every 30 minutes; 5 minutes discovers a new completed close within 5 minutes of it happening, an acceptable latency for this use case.
- **Scope:** `SELECT * FROM approved_setup_monitor_state WHERE state IN ACTIVE_MONITOR_STATES` — bounded to real approved count (5–20), a structurally smaller, separate query from anything universe/scanner-related. Never touches `candidates` or the 550+-symbol scan machinery.
- **Batching:** one batched 30m-bar download across all active Type-B tickers (mirrors the existing `CANDIDATE_PREVIEW_PREWARM_CHUNK_SIZE`/`_batch_download` precedent — trivially within the existing chunk size of 50 at this scale), plus one batched live-quote call (`_latest_quotes_for_previews`, already exists) for R:R/window evaluation across the same set.
- **Candle-close alignment:** not needed for V1 — a flat 5-minute poll against "the latest completed bar" (via `_is_forming_candle`-style logic) finds a new close within 5 minutes regardless of tick alignment; building schedule-alignment logic adds complexity for no meaningful latency win at this cadence.
- **Restart behavior:** a non-event (§8/§9) — stateless-poll-against-history design.
- **Railway scheduling mechanism:** no `railway.json`/`railway.toml` exists; `Procfile` runs a single `web` process. Reusing the existing in-process thread avoids provisioning a **new** Railway service/cron — a real infra decision this audit should not make silently.
- **Locking/concurrency:** the shared loop runs registered callbacks inline, one at a time, in a single thread — two ticks of the same periodic task cannot overlap by construction, no new lock needed. Inside the tick: fetch all market data first, then write — same lesson this codebase already learned and documented (`_enriched_previews_for_candidates`'s old long-held-transaction bug).
- **API/provider failure behavior:** same established precedent everywhere in this file — warn and skip, never guess. A tick that can't get fresh data for a ticker simply leaves that memory's `last_checked_at` unmoved; it must never force an ACTIONABLE/INVALIDATED verdict from missing data.
- **Stale data handling:** `last_checked_at` already exists precisely for this — the UI (§13) should surface it, so a human can tell monitoring has gone quiet rather than Kairos silently implying freshness it doesn't have.

---

## 12. Alert contract

**No delivery transport exists anywhere in this codebase today** (confirmed: no SMTP/SMS/Slack-webhook/push integration of any kind). V1 alerting should therefore be scoped to **data + dedup + prominent UI surfacing only** — not real delivery, which is a separate integration decision (which channel, whose contact info, rate limits) outside what this audit can responsibly decide.

- **Dedup rule:** log a new `approved_setup_monitor_events` row only when the derived verdict actually **changes** from the last logged event for that memory (compare against the most recent event before inserting) — mirrors `_sync_approved_setup_memory_on_review`'s own "approve is a no-op if already active" discipline. Never re-log the same transition every 5-minute tick.
- **Message content:** reuse exactly what's already computed fresh each tick — `current_price`, `approved_stop`, `approved_target`, `current_rr_at_last_check` — no new computation needed beyond §7.
- **Example bodies**, matching the brief's own format:
  - `ACTIONABLE`: *"NVDA is now ACTIONABLE. Approved execution conditions satisfied. Current $X · Stop $Y · Target $Z · Current R:R N"*
  - `TRIGGER_SATISFIED` (optional, internal/log-only unless the user wants it surfaced separately from ACTIONABLE): *"NVDA's 30m close above $X has occurred."*
  - `EXTENDED`: *"NVDA trigger occurred, but price is now extended. Do not chase."*
  - `INVALIDATED`: *"NVDA approved setup has invalidated."*
- **V1 delivery = UI only:** an unseen/most-recent `approved_setup_monitor_events` row for a memory renders a distinct badge/treatment on its Approved Setups card. Real push/email/SMS is §17 (deferred).

---

## 13. Approved Setups UI proposal

**Card hierarchy** (server-computed state driving the render, not client-side recomputation from live fields — fixing Finding A):

- **WAITING** (== `APPROVED`/`TRIGGER_SATISFIED`-pending-window, muted/low visual weight): *"DO NOT ENTER — Kairos is monitoring."* — **but only once monitoring actually exists.** Until then, every card must keep the exact same discipline the trigger-capture feature already established: *"Status: Not monitored yet."* This audit changes no UI; that disclaimer stays until §17 item 3–5 actually ship and are verified.
- **ACTIONABLE**: visually dominant — a full-width, success-colored banner **above** the metrics grid (not buried in the existing signal-grid/trigger-block area), explicit current price/stop/target/current R:R, one clear sentence ("Kairos has confirmed your stated execution condition and price remains within an acceptable window"), plus the same site-wide non-advice disclaimer already rendered at the top of every board page.
- **EXTENDED**: reuse the existing `--warn` styling and *"Extended — do not chase"* copy already shipped — just make it read from the frozen snapshot + live R:R, not the live-drifting proximity check.
- **INVALIDATED**: reuse the existing `--fail` styling and *"Setup invalidated"* copy — corrected to read `approved_stop` breach (Finding A fix), not live/recomputed stop.

**ENTRY → PLANNED ENTRY / REFERENCE ENTRY:** recommend **"PLANNED ENTRY"** — reads as "this was the plan" rather than a clinical audit-trail term, and pairs naturally alongside "Stop"/"Target" if the copy is ever extended to "Planned Stop"/"Planned Target." **Not implemented in this pass**, per the brief's explicit instruction — flagged for §17.

---

## 14. FFIV / NVDA / CLH walkthrough

Using real, current, read-only production data (no writes):

- **NVDA — Type A.** `lower_tf_confirmation=yes`, real qualitative rationale in `review_note` ("30M bullish CHoCH after correction"), but **no objective anchor** — no price, no timestamp. Current R:R **5.88**. Under the §3 recommendation (bounded reconfirmation window from `approved_at`, currently minutes old — comfortably fresh), NVDA would read ACTIONABLE today, purely on freshness + current R:R + distance, because there is nothing more specific to check it against. This is Type A's fundamental evidentiary ceiling, not something Execution Layer can retroactively improve.
- **FFIV — Type A.** Same pattern, `review_note` is `null` — even less to go on than NVDA. Current R:R **2.68**. Same treatment.
- **CLH — Type B, and a correction to the brief's premise.** The brief states all three currently read `lower_tf_confirmation=yes`; the real, current, live review for CLH is actually **`not_yet`**, with `trigger_timeframe/rule/level` all `NULL` (this review predates — or didn't use — the optional trigger fields). Under the Type B design, **CLH cannot become ACTIONABLE at all today** — there is no stored condition to monitor, and none may be invented. It stays in WAITING indefinitely until a new human visual review is submitted for this exact `setup_key` with an explicit trigger. Current R:R is **1.75** (still above 1.5) but is irrelevant while no trigger exists — this is precisely the scenario §8/§15's "no trigger, no monitoring" principle exists to protect.

**What cannot be known for any of the three** (spelled out explicitly, not glossed over): none has `atr14` frozen at approval (Finding B); none has an objective confirming price/timestamp behind its `yes`/`not_yet` read (Type A's evidentiary limit); all three are `snapshot_origin="live_backfill"`, `snapshot_exact=false` — `approved_entry`, `current_price_at_approval`, and every scanner-evidence field are reconstructions at backfill time, not literal approval-moment truth (though `approved_stop`/`approved_target`/`setup_key` **are** exact, by the setup_key-matching construction already documented in the backfill code). This audit does not optimize any rule around these three — they're used only as real-data grounding.

---

## 15. Safety invariants — status against this design

Every invariant in the brief is achievable under the design above; two are worth calling out explicitly:

- **"Never make ACTIONABLE from Entry touch alone"** — satisfied by construction: ACTIONABLE requires `TRIGGER_SATISFIED`/fresh-Type-A-evidence **and** the current-R:R window (§9), never a bare entry-price touch.
- **"Never overwrite immutable Approved Setup Memory"** — untouched; all new state lives in `approved_setup_monitor_state`/`approved_setup_monitor_events`, never in `approved_setup_memories`.
- **"Never carry approval across setup_key generations"** — already correctly enforced today, verified directly: `_active_monitor_state_row_for_setup_key` is keyed on exact `setup_key`, and `_create_approved_setup_memory`'s superseding logic marks any older active memory for the same `(ticker, source)` under a **different** `setup_key` as `SUPERSEDED` in the same transaction.
- **"Never treat `execution_shadow_ok` as lower-TF confirmation"** — a real risk given how similar the two concepts sound; `execution_shadow_ok` is a separate, direction-blind, 4H mechanical heuristic (§2), and must stay entirely out of the Type A/B trigger logic.
- **Currently-live violation to fix, not just avoid repeating:** Finding A — `setup_board.js`'s existing INVALIDATED/EXTENDED computation already reads live, not frozen, fields. This predates Execution Layer V1 and must be corrected as part of it (§17 item 1), not treated as already-safe.
- **"Repeatedly alert the same state transition"** — addressed structurally by the dedup rule in §12 (log-on-change-only against `approved_setup_monitor_events`).

---

## 16. Exact recommended V1 implementation scope

1. **Fix Finding A** — `setup_board.js` reads state from `GET /candidates/approved-setup-memory` (already exists), computing INVALIDATED against `approved_stop`, not the live candidate row.
2. Schema: the columns/table in §10, plus the three new `ApprovedSetupMonitorStateName` values.
3. Monitor: `register_background_periodic_task("approved_setup_monitor", ~300, ...)`, batched, scoped to `ACTIVE_MONITOR_STATES` only (§11).
4. Invalidation check: intrabar breach of frozen `approved_stop`, every tick, symmetric LONG/SHORT (§5).
5. Type B trigger-satisfaction check: completed-30m-close semantics, persisted exactly once (§8).
6. Type A: bounded reconfirmation window from `approved_at` (§3), explicit unvalidated placeholder constant.
7. Current-price R:R + execution-window verdict (`ACTIONABLE`/`EXTENDED`), gated at `RR_WARNING_THRESHOLD=1.5` (§6/§7).
8. `approved_setup_monitor_events` append-only log + dedup-on-change (§12).
9. UI: card hierarchy from §13, reading server-computed state — remove "Status: Not monitored yet" **only once 3–8 are deployed and verified against real data for several real trading days**, not on merge.

**Explicitly not in V1**, even though touched by this audit: the ENTRY rename (copy-only follow-up), SHORT support (blocked upstream by three separate long-only gates outside this feature's scope — §17), real alert delivery transport.

---

## 17. Deferred to V2

- `ENTRY` → `PLANNED ENTRY` copy change.
- SHORT direction support — requires lifting the long-only gate in `_preview_clears_mechanical_prechecks` (`if direction != "long": return False`), the review-queue's own `WHERE signal='long'` query, and `_promotion_block_reason`'s explicit short block — a larger, separate decision belonging to that pipeline, not something Execution Layer should silently enable as a side effect.
- Real alert delivery (push/email/SMS/Slack) — no transport exists today; needs its own integration decision.
- Backtesting/validating: `RR_WARNING_THRESHOLD` reuse, the reconfirmation-window length, the distance-based window backstop constant — all explicit, unvalidated placeholders, same honesty standard as every other threshold already in this file.
- Requiring a trigger on every NOT_YET review (still an open question from the trigger-capture task itself).
- Any Kairos-inferred trigger/BOS/CHoCH detection — permanently out of scope per this session's own Phase 1/2 research findings.
- Candle-close-aligned monitor scheduling (V1 uses simple TTL polling, §11).
- Freezing `atr14_at_approval` — only needed if an ATR-anchored window is chosen over the R:R-primary recommendation.

---

## 18. Recommendation

**B — one targeted research question must be answered before implementation, everything else in this audit is resolved with a concrete recommendation.**

Not A: Type A's reconfirmation-window length and Option D (force an explicit reference level even on `lower_tf_confirmation=yes`, going forward) are genuine product-taste decisions, not code-archaeology questions — the evidence (NVDA/FFIV's real but anchor-less rationale) shows the tradeoff clearly but doesn't resolve it.
Not C: the architecture is fundamentally sound and additive — every piece this design needs (immutable snapshot table, a monitor-state table with the target enum values *already reserved from a prior design session*, a working in-process scheduler with a direct precedent, a completed-candle detection helper, and the trigger itself) already exists in the codebase. No redesign is needed, only wiring what's there plus the additive columns in §10.

**The one question for the user:** for Type A (`lower_tf_confirmation=yes`, no stored trigger) — is a bounded freshness window (§3, Option C) sufficient going forward, or should Kairos start requiring an explicit reference level/event on every `yes` too (Option D), closing the evidentiary gap NVDA/FFIV illustrate, at the cost of added review friction? Everything else in this report (invalidation rule, execution-window formula, R:R minimum, monitor architecture, persistence, state machine, UI) is a specific, grounded recommendation ready to implement once that one question is answered.
