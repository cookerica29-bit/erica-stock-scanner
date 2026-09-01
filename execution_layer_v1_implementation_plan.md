# Kairos — Execution Layer V1 Implementation Plan

**Status: plan only. No code written. Nothing pushed or deployed.**
Builds on [execution_layer_v1_design_audit.md](execution_layer_v1_design_audit.md)'s Recommendation B, now that the Type-A open question is resolved. Grounded in a fresh read-only re-check of production (via `railway run`, real key never exposed to me) done specifically because the user flagged a real edit made after the audit was written.

---

## 0. Re-read of current production CLH (as instructed)

The user was right — CLH changed after the audit snapshot. Current state:

| review id | decision | lower_tf_confirmation | reviewed_at |
|---|---|---|---|
| 7 | watch | not_yet | 00:10:01Z |
| 8 | approve | not_yet | 00:43:10Z |
| **29** | **approve** | **yes** | **05:12:36Z** |

Review **id 29** is real: the reviewer submitted a fresh approve with `lower_tf_confirmation=yes` after the audit ran. But checking `approved_setup_memories` directly:

```
memory id: 3 | visual_review_id: 8 | lower_tf_confirmation: not_yet | approved_at: 00:43:10Z
```

**The active memory is still id 3, bound to review id 8 (`not_yet`) — review id 29 never produced a memory at all.**

*(Re-verified read-only, no mutation, immediately before writing §4 below: identical state — memory 3/review 8/`not_yet` is still the sole active memory, review 29/`yes` still has no corresponding memory. The bug is confirmed still live in production, exactly as expected since no code has changed yet.)*

### Finding C (new): re-approving an already-active setup_key is a silent no-op today

`_sync_approved_setup_memory_on_review`'s current rule:
```python
if decision == "approve":
    if _active_monitor_state_row_for_setup_key(conn, setup_key) is not None:
        return   # <-- no-op, regardless of what changed in the new review
```
This only checks **setup_key identity**, never whether the new review's content is materially different from what's frozen. CLH's `setup_key` (`CLH|ma_pipeline|long|307.94|328.04`) hasn't changed since review id 8, so review id 29's real, fresh `yes` was silently discarded from the memory's perspective — `candidate_visual_reviews` (append-only, "latest row wins" for *reads of the review log*) now disagrees with `approved_setup_memories` (supposed to be the authoritative frozen snapshot Execution Layer acts on).

This is not cosmetic — it directly blocks the goal of this task. Without a fix, **CLH's real observed confirmation can never reach Execution Layer**, no matter how the confirmation-anchor fields are designed, because the memory that would carry it is never created. This is addressed in **§4**.

One more honest complication: review id 29 predates the confirmation-anchor fields this plan adds — it has no `trigger_*`/`confirmation_*` data at all (both null, confirmed via the API). So even after §4's fix ships, CLH will need a **third**, genuinely new review submission through the updated form to produce a memory with a real confirmation anchor. Nothing here retroactively fabricates one for reviews 8 or 29 — consistent with "do not fabricate."

---

## 1. Field design: separate `confirmation_*` fields, not a shared field + discriminator

Investigated both options against the real schema and the real (near-)future consumers.

**Option 1 — reuse `trigger_timeframe/rule/level/reason` with an added `trigger_status`/`trigger_origin` discriminator.**
**Option 2 — new, distinct `confirmation_timeframe/rule/level/confirmed_candle_time/confirmation_note` fields.**

Recommendation: **Option 2.** Reasoning:

1. **Every `trigger_*` row ever written to date — in tests and in the (currently empty, in production) real trigger rows — was written under one explicit contract: "a future condition Kairos should wait for."** Introducing a discriminator now means every future reader of `trigger_level` (a SQL query, a report, a new dashboard someone writes in six months) must remember to also check the discriminator before drawing any conclusion, forever. Two distinctly-named columns make the meaning of `WHERE trigger_level IS NOT NULL` unambiguous by construction — exactly the ambiguity the user explicitly warned against.
2. **The two concepts are consumed by genuinely different code paths, not two flavors of the same computation.** A future trigger is *polled for* (fetch new 30m bars, check for a qualifying close). An observed confirmation is *recorded evidence* (no polling needed to detect it — it already happened). Sharing storage buys no real code reuse; the monitor branches on the concept immediately either way.
3. **A brand-new field is required regardless** — "completed confirming candle timestamp" has no equivalent in the existing four columns (`trigger_reason` is free text, not a timestamp). Since a new column is unavoidable, giving the whole group self-documenting names costs nothing extra in migration complexity (same additive-column pattern, same column count either way) while removing the discriminator-forgetting risk entirely.
4. **Mutual exclusivity becomes visible in the schema itself.** A row can never sensibly have both "I already saw this confirm" and "Kairos should wait for this to happen" — with separate columns, a row with both non-null is an obviously-wrong state to validate against; with one shared field + a status enum, that exclusivity is implicit and only ever enforced by app code someone could bypass later (a migration script, a backfill, a future endpoint).

Type reuse is fine and different from field reuse: `confirmation_timeframe`/`confirmation_rule` validate against the exact same `TriggerTimeframe`/`TriggerRule` Literal aliases already defined (same vocabulary of valid values, no ambiguity risk — it's a shared *validator*, not shared *storage*).

### Exact new fields

On both `candidate_visual_reviews` and `approved_setup_memories` (same additive-nullable-column pattern used for `trigger_*` last session):

| Column | Type | Meaning |
|---|---|---|
| `confirmation_timeframe` | `TEXT` (`Literal["30m"]`) | Same vocabulary as `trigger_timeframe` |
| `confirmation_rule` | `TEXT` (`Literal["close_above","close_below"]`) | Same vocabulary as `trigger_rule` |
| `confirmation_level` | `REAL` | The price the reviewer observed a completed candle close beyond |
| `confirmed_candle_time` | `TEXT` (ISO8601, human-entered) | **The market event's own timestamp** — when the confirming candle closed. Distinct from `reviewed_at` by design (see below) |
| `confirmation_note` | `TEXT` | Free-text context, mirrors `trigger_reason`'s role |

`reviewed_at` already exists and is untouched — it's "when the human filed this review," never conflated with `confirmed_candle_time` ("when the market event the human is describing actually happened"). These can legitimately differ by hours or days (a reviewer confirming today what they watched happen yesterday), and the user was explicit these must not be conflated — keeping them as two independently-populated columns satisfies that directly; no derivation of one from the other is ever performed.

`confirmed_candle_time` is **human-entered, not server-validated against real market data** — consistent with the original trigger-capture task's "no market-data validation calls during review submission" rule, which this plan does not relax. Stored as a plain ISO8601 string, checked only for well-formedness (parses as a timestamp), never checked against what the market actually did at that time.

---

## 2. Validation rules (exact)

Extends `record_candidate_visual_review`'s existing validation block (the `trigger_*` completeness check already there is untouched in shape, just joined by three new rules):

**A. `confirmation_rule`/`confirmation_level` completeness** — same all-or-nothing contract as `trigger_rule`/`trigger_level`: both present or both absent, 422 with a clear message otherwise. `confirmation_timeframe` defaults to `"30m"` when rule+level are given without an explicit timeframe. `confirmation_note` is always independent (can exist with or without a numeric anchor — mirrors `trigger_reason`).

**B. `confirmed_candle_time` is optional within the confirmation group, not lockstepped with rule+level.** A reviewer may be confident they saw a qualifying close without recalling the exact bar. This is a deliberate judgment call, not a given — matches the user's own "preserve **when possible**" phrasing for this specific field, unlike rule/level which the user called a hard requirement. If given, it must parse as a valid timestamp (422 if not) — no market lookup.

**C. Mutual exclusivity with `lower_tf_confirmation` (new — tightens `trigger_*` too, which has no such check today):**
- `trigger_rule`/`trigger_level`/`trigger_reason` may only be set when `lower_tf_confirmation == "not_yet"`. Setting them alongside `"yes"` is rejected (422) — a future-tense trigger makes no sense on an already-confirmed setup.
- `confirmation_rule`/`confirmation_level`/`confirmed_candle_time`/`confirmation_note` may only be set when `lower_tf_confirmation == "yes"`. Setting them alongside `"not_yet"` is rejected (422) — the reverse error, same reasoning.
- Both groups are rejected outright when `practical_rejection_reason` is set (mirrors the existing visual-field exclusion for practical rejections — a pre-chart-review disqualification has neither a future trigger nor an observed confirmation to record).

**D. New hard requirement (the user's actual decision) — scoped precisely:**
When `decision == "approve"` **and** `lower_tf_confirmation == "yes"`, `confirmation_rule` and `confirmation_level` become **required** (422 if either is missing) — this is the one genuinely new mandatory-field rule in this whole feature. It is deliberately **not** extended to `watch`/`reject` reviews with `lower_tf_confirmation=="yes"` — a watch never creates a memory, so there's no Execution Layer consequence to leaving it optional there, and forcing it everywhere would be pure friction with no corresponding safety gain (same "collect real usage before over-requiring" caution the original trigger task applied to `not_yet`, applied consistently here to the one case where it still makes sense).

`trigger_rule`/`trigger_level` on `not_yet` reviews remain **optional**, unchanged from the original task — the user's decision only concerns the `yes` path.

---

## 3. Approved Setup Memory changes

`_create_approved_setup_memory` / `_sync_approved_setup_memory_on_review`: add the same five keyword params (`confirmation_timeframe/rule/level/candle_time/note`), copied verbatim into the new columns — identical mechanical pattern to how `trigger_*` was threaded through last session, no new design needed there.

`backfill_approved_setup_memories` (FFIV/NVDA/CLH's original path): **untouched in behavior.** It already reads whatever the source `review_row` has for the four `trigger_*` columns (null, for these) and passes them through without fabricating — it will do exactly the same for the five new `confirmation_*` columns, correctly staying null for legacy rows. No special-casing needed; the existing "never fabricate, read exactly what the review row has" logic already generalizes.

**No new "legacy/incomplete" flag column is needed.** Once §2's rule D is enforced, it becomes structurally impossible for any **new** `approval_event`-origin memory to exist with `lower_tf_confirmation="yes"` and `confirmation_level IS NULL` — that combination can only ever occur on `snapshot_origin="live_backfill"` rows (FFIV, NVDA, CLH's current memory). The enforcement point itself makes "null anchor on a yes-approved memory" a reliable, self-documenting signal of "predates this requirement" — no extra bookkeeping required. This is a direct, deliberate consequence of where rule D is placed, not an accident.

---

## 4. Approved Setup Memory Revision — final design (decision made: **implement**)

Watch→re-approve is explicitly **rejected** as the documented workflow. The no-op is a real correctness bug for this product and is fixed directly. Design below satisfies all 10 requirements from the decision; each is called out inline.

### 4.1 New columns (immutable side stays immutable — requirement 1)

`approved_setup_memories` gains two nullable columns, set once at INSERT, **never UPDATEd** (no existing memory row is ever touched by this feature — requirement 1, requirement 8):

| Column | Meaning |
|---|---|
| `revision_of_memory_id INTEGER` | The id of the memory this one revises. `NULL` for a memory that starts a fresh setup_key generation (today's existing path, unchanged — requirement 9). Non-null only for a same-setup_key evidence revision (the new path). |
| `revision_reason TEXT` | `Literal["newer_approved_review_evidence"]` — the one supported value today, same "single literal now, extend later" pattern already used for `TriggerTimeframe`. Present only when `revision_of_memory_id` is set. |

`approved_setup_monitor_state` (already the mutable table — the right place for a forward pointer) gains one nullable column, set via `UPDATE` in the same transaction that already flips a row to `SUPERSEDED`:

| Column | Meaning |
|---|---|
| `superseded_by_memory_id INTEGER` | The id of the memory that made this row's memory non-authoritative. Wired into the **existing** supersede `UPDATE` statement (today it only sets `state='SUPERSEDED', updated_at, terminal_at`) — this is a small, purely additive extension of that one existing statement, used for *both* the pre-existing different-setup_key path and the new same-setup_key revision path. It does not change any currently-observable behavior; it just records what was previously untracked. |

Together these give full bidirectional lineage (requirement 5) without ever mutating a historical row: from a new memory, `revision_of_memory_id` points back; from the old memory's (mutable) monitor_state, `superseded_by_memory_id` points forward; `visual_review_id` (already exists, unchanged) identifies the originating review on each side; the new memory's own `approved_at` and the old monitor_state's existing `updated_at`/`terminal_at` together record when the revision took effect — no redundant new timestamp column needed.

`revision_of_memory_id IS NOT NULL` vs. `NULL` is exactly how a caller distinguishes "this memory is an evidence revision of the same setup_key" from "this memory is a genuinely new setup_key generation that happened to supersede an old one" — satisfying requirement 9 (the existing structural path is provably unchanged: it simply never sets this column, exactly as it never has).

### 4.2 Meaningful evidence comparison (requirement 6) — exact, deterministic

A new pure function, `_review_evidence_materially_differs(old_memory_row, new_review) -> bool`, compares **only** the fields the decision lists — nothing else, and never `reviewed_at`/`id`/any timestamp:

```
market_structure, location_read, clear_path_to_target, lower_tf_confirmation,
review_note,
trigger_timeframe, trigger_rule, trigger_level, trigger_reason,
confirmation_timeframe, confirmation_rule, confirmation_level,
confirmed_candle_time, confirmation_note
```

Deliberately **excluded** from the comparison (these are mechanical/live-computed values, not human evidence, and always get a fresh value on any new preview regardless — comparing them would make ordinary rescans look like "new evidence"): `approved_entry/stop/target/risk_reward`, `current_price_at_approval`, `entry_distance_pct_at_approval`, `entry_proximity_threshold_pct_at_approval`, `bos_confirmed`, `displacement_*`, `macro_bias`, `sweep_confirmed`, `rejection_confirmed`, `execution_shadow_ok`, `confluence_*`, `location_label`, `location_alignment`.

Comparison rule, stated exactly: each listed field is compared for equality; text fields (`review_note`, `trigger_reason`, `confirmation_note`) are compared after a trivial `.strip()` (so a stray trailing space from a form field never manufactures a false "revision") but are otherwise compared byte-for-byte — no case-folding, no fuzzy matching. Numeric fields (`trigger_level`, `confirmation_level`) are compared for exact equality, not with a rounding tolerance — unlike computed prices elsewhere in this codebase (stop/target, which DO need tolerance because they're recomputed floats), these are raw human-typed values stored and read back unchanged, so exact equality is the correct, not merely convenient, choice. `NULL` vs. any non-null value always counts as a difference. **Any single field differing is sufficient** to mark the evidence as materially different — the function returns as soon as one mismatch is found, no weighting or partial-credit logic.

This directly satisfies "do not create a new memory revision merely because `reviewed_at` changed or the same review was submitted twice" (requirement 6's own qualifier) — `reviewed_at` is never in the comparison set, and a byte-identical resubmission compares as `False` (no difference) on every listed field, so it stays a no-op exactly as an unchanged same-setup_key approve does today.

### 4.3 Revision flow

In `_sync_approved_setup_memory_on_review`, the `decision == "approve"` branch becomes:

```
active = _active_monitor_state_row_for_setup_key(conn, setup_key)
if active is None:
    create fresh memory + monitor_state   # unchanged existing path
elif _review_evidence_materially_differs(active memory's row, new review):
    create NEW memory:
        revision_of_memory_id = active.approved_memory_id
        revision_reason = "newer_approved_review_evidence"
        (all other fields from the NEW review/preview, exactly like any fresh approval)
    create NEW monitor_state row for the new memory (fresh — see 4.4)
    UPDATE the OLD monitor_state row:
        state = 'SUPERSEDED', updated_at = reviewed_at, terminal_at = reviewed_at,
        superseded_by_memory_id = <new memory id>
else:
    return   # unchanged: identical resubmission stays a no-op (requirement 6, 10's idempotency case)
```

This reuses the *exact same* `_create_approved_setup_memory` creation path a fresh, first-time approval already uses (same function, same "always insert a clean row" behavior) — it is not a new, separate code path for "revision" creation, just a new condition under which that existing function gets called again. This is what makes §4.4 true by construction rather than by extra logic.

### 4.4 Monitor state does not inherit stale execution evidence (requirement 7)

Because the new revision's monitor_state is a **brand-new INSERT** (never a copy or partial-copy of the old row), none of the old row's execution-tracking columns exist on the new row at all: `trigger_satisfied_at/bar_time/price`, `execution_window_status`, `last_evaluated_bar_time`, `current_rr_at_last_check`, `last_checked_at`, `last_live_price/entry` all start `NULL` — there is nothing to "reset" because nothing is ever copied forward in the first place. The new row's initial `state` is decided the same way any fresh memory's is (§5 of this plan): `CONFIRMED` immediately if the new review supplies `confirmation_rule/level`, otherwise `APPROVED` awaiting a future `TRIGGER_SATISFIED`.

**Alert-dedupe state** (`approved_setup_monitor_events`, keyed on `approved_memory_id`) is satisfied the same way: a fresh memory id means the event log is naturally empty for the new generation — no old alert on the retired memory can suppress or interfere with a genuinely new one on the revision.

**Deliberate simplification, stated explicitly rather than hidden:** requirement 7 allows carrying evidence forward "if the contract explicitly proves that evidence still applies" (e.g., a revision that changes `market_structure` but leaves `trigger_level` byte-identical *could*, in principle, preserve an already-observed trigger satisfaction). V1 does **not** attempt this — it always starts the new revision completely clean, which trivially satisfies "must not inherit stale evidence" without needing to build and validate a "proof that it still applies" mechanism (a nontrivial, bug-prone surface on its own). This costs nothing real in practice: if a Type-B trigger genuinely already fired historically, the very next monitor tick re-derives that fact fresh from real market history (per §8 of the design audit — satisfaction is always evaluated against actual completed bars, never assumed) and the revision reaches `TRIGGER_SATISFIED` again within one tick interval, just with an honestly fresh timestamp instead of a copied one. Same safe-failure-direction reasoning already used for invalidation in the audit: a moment of extra, unnecessary "not yet satisfied" is a safe gap, not a dangerous one.

### 4.5 Regression coverage (requirement 10, exact scenarios)

New test file section (or extension of the trigger/confirmation suite) covering, verbatim:

1. Active memory exists for `setup_key` with `lower_tf_confirmation=not_yet` (no confirmation fields).
2. A new `approve` review arrives for the **same** `setup_key` with `lower_tf_confirmation=yes` and valid `confirmation_rule/level` → assert:
   - the old memory row is byte-for-byte unchanged (re-fetch by id, compare every field to before)
   - a new memory row now exists
   - the new memory's `visual_review_id` points to the new review
   - the new memory's `confirmation_rule/level/timeframe` match exactly what was submitted
   - the new memory's `revision_of_memory_id` equals the old memory's id
   - `GET /candidates/approved-setup-memory` (default, active-only) returns the **new** memory for this setup_key, not the old one
   - the old memory's monitor_state is `SUPERSEDED` with `superseded_by_memory_id` pointing at the new memory
   - the new memory's monitor_state exists, is active, and starts with every execution-tracking column `NULL` (fresh, not copied)
3. Submitting the **identical** `approve` payload a second time (same `lower_tf_confirmation=yes`, same confirmation fields) is idempotent: no third memory is created, the second memory stays authoritative.
4. A revision triggered by a change to `market_structure`/`location_read`/`clear_path_to_target`/`review_note` alone (trigger/confirmation fields unchanged) still fires — confirms the comparison isn't accidentally scoped to only the trigger/confirmation fields.
5. A materially different `setup_key` (stop/target actually changed) still takes the pre-existing structural-supersession path — `revision_of_memory_id` stays `NULL` on the new memory, confirming requirement 9 (no regression to existing behavior).
6. `reviewed_at`-only difference (same content, re-submitted moments later) does not create a revision.

No live market-data calls, no `candidate_promotions` writes, no monitoring/ENTER_NOW logic exercised by any of the above — same discipline as every prior test file this session.

**Production:** verified read-only only (§0's re-check), never mutated. CLH's real stuck state is left exactly as-is in production until this is implemented and deployed — no manual repair performed or planned as part of this work, per your instruction.

---

## 5. Updated state machine (supersedes audit §9)

Both Type A and Type B now share one underlying question — **"has qualifying execution evidence been recorded, and is it still fresh enough to act on?"** — reached at different moments:

```
APPROVED
   │
   ├─ Type A (confirmation_rule/level present at approval) ──► CONFIRMED   (immediately, at memory creation)
   │
   ├─ Type B (trigger_rule/level present, no confirmation) ──► stays APPROVED until monitor observes
   │          a qualifying completed 30m close ───────────────► TRIGGER_SATISFIED
   │
   │   (a memory with NEITHER confirmation_* nor trigger_* populated — should not exist for any
   │    NEW approval once §2 rule D ships; legacy live_backfill rows simply stay APPROVED forever,
   │    correctly never reaching CONFIRMED/TRIGGER_SATISFIED without a fresh human review)
   │
   ▼ (from CONFIRMED or TRIGGER_SATISFIED, every monitor tick)
execution-window / current-R:R check ──► ACTIONABLE  or  EXTENDED   (freely re-evaluated, not sticky)
   │
   ├─ freshness window elapses without action (measured from confirmed_candle_time,
   │  falling back to reviewed_at if not given, for Type A; from trigger_satisfied_at
   │  for Type B) ─────────────────────────────────────────────► STALE (needs fresh review)
   │
   ├─ approved_stop breached (intrabar, any active state) ─────► INVALIDATED (terminal)
   ├─ human re-reviews to watch/reject ─────────────────────────► WITHDRAWN (terminal)
   └─ a newer setup_key gets its own active approval,
      OR §4's new-evidence supersede fires ────────────────────► SUPERSEDED (terminal)
```

`CONFIRMED` is a new state name (added alongside the already-reserved `TRIGGER_SATISFIED`/`ACTIONABLE`/`STALE` from the audit) — deliberately not reusing `TRIGGER_SATISFIED` for Type A, since "trigger satisfied" no longer reads correctly for something that was never a future trigger. Both states mean the same thing downstream (evidence gate cleared, eligible for the execution-window check) and are treated identically by every consumer past that point — keeping them separately named preserves exactly the same "why is this its own persisted state" reasoning the audit already gave for `TRIGGER_SATISFIED` (a real, distinct, auditable transition; a real anchor for the freshness clock; a real distinct alert-log entry) — restated here for Type A rather than re-argued.

This directly implements the user's core instruction: **a freshness window is no longer the sole evidence for a historical YES** — it cannot even begin (no `APPROVED → CONFIRMED` transition) without `confirmation_rule`/`confirmation_level` present; freshness only decides how long `CONFIRMED` stays actionable-eligible once it's real.

---

## 6. Monitor tick — updated cost model

Two genuinely different costs per active memory, worth stating explicitly since it affects §11 of the audit's batching plan:

- **Type A (already `CONFIRMED`) / any memory past its evidence gate:** no bar-fetch needed to detect anything — the qualifying event already happened and is frozen. Each tick only needs a **live quote** (for current-R:R / execution-window / freshness-clock evaluation), reusing the same batched `_latest_quotes_for_previews` call the audit already planned.
- **Type B (still `APPROVED`, has `trigger_*`, not yet `TRIGGER_SATISFIED`):** needs the batched 30m-bar fetch to check for a new qualifying completed close, **in addition to** the live quote once/if it transitions to `TRIGGER_SATISFIED` in that same tick.

Net effect: the 30m-bar batch call only needs to include tickers still waiting on a **future** trigger — a strict subset of the already-small (5–20) active set, making the monitor's real network cost even lighter than the audit assumed for the Type-A share of it.

---

## 7. Persistence changes (supersedes audit §10 for the affected tables)

- `candidate_visual_reviews`: add `confirmation_timeframe`, `confirmation_rule`, `confirmation_level`, `confirmed_candle_time`, `confirmation_note` (5 nullable columns) — same additive `ALTER TABLE ADD COLUMN`-if-missing pattern used for `trigger_*`.
- `approved_setup_memories`: same 5 columns, same pattern, threaded through `_create_approved_setup_memory`.
- `approved_setup_monitor_state`: everything from audit §10 (`execution_window_status`, `trigger_satisfied_at`, `trigger_satisfied_bar_time`, `trigger_satisfied_price`, `last_evaluated_bar_time`, `current_rr_at_last_check`) **plus `superseded_by_memory_id`** (§4.1), plus extend `ApprovedSetupMonitorStateName` with `CONFIRMED` (new) alongside `ACTIONABLE`, `TRIGGER_SATISFIED`, `STALE` (already planned in the audit).
- `approved_setup_memories`: **plus `revision_of_memory_id`, `revision_reason`** (§4.1), on top of the 5 confirmation columns above.
- `approved_setup_monitor_events` (new, append-only): unchanged from audit §10/§12 — keyed on `approved_memory_id`, which is exactly what makes §4.4's "no stale alert-dedupe carryover" true.
- No changes to `candidates`, `candidate_plan_previews`, or `candidate_promotions`.

---

## 8. UI changes (Review Queue form)

Under `lower_tf_confirmation = "yes"` (currently no extra fields render at all): add a **required-when-approving** confirmation block, visually and functionally distinct from the existing `not_yet` trigger block —

> **CONFIRMED — describe what you already saw**
> `[ 30m ]` `[ Above ▾ / Below ]` `[ $_______ ]` (rule/level — required to Approve)
> `When did that candle close? [ optional date/time ]`
> `Note (optional)`

Copy must make the tense difference unmistakable to a novice reviewer — `not_yet`'s block already reads "Waiting for 30m close: …" (future tense); this block should read something like "You saw a 30m close [above/below] $X — when did that happen?" (past tense, already-occurred framing). Submit is blocked client-side (mirroring the existing required-field pattern already used for the four core visual-read fields) when `decision=approve` and rule/level are empty under `yes` — with the server-side rule D as the real enforcement, client-side is just UX, not the safety boundary.

Approved/Watch Setups boards: the existing trigger-capture "Waiting for: …" / "EXECUTION TRIGGER … Status: Not monitored yet" copy is extended with a second, distinctly-worded variant for the confirmation case (e.g. "CONFIRMED — 30m close above $X, observed [confirmed_candle_time or reviewed_at]" / "Status: Not monitored yet") — same disclaimer discipline, not implemented until the monitor itself ships, exactly as the original trigger-capture task established.

---

## 9. Test plan (additive to the existing 20-case `trigger_capture_v1.py` suite)

- Confirmation completeness: rule-without-level rejected, level-without-rule rejected, timeframe-without-rule-and-level rejected (mirrors existing trigger tests exactly).
- `confirmed_candle_time` optional: rule+level with no candle time succeeds; malformed candle time string rejected.
- Rule D: `decision=approve, lower_tf_confirmation=yes, confirmation_rule/level omitted` → 422. `decision=watch, lower_tf_confirmation=yes, confirmation omitted` → succeeds (no requirement).
- Mutual exclusivity: `confirmation_*` set with `lower_tf_confirmation=not_yet` → 422. `trigger_*` set with `lower_tf_confirmation=yes` → 422. Either group set with `practical_rejection_reason` → 422.
- Approve copies `confirmation_*` verbatim into the new memory columns, frozen against later live changes (mirrors the existing trigger-freeze test exactly).
- Backfill (FFIV/NVDA/CLH) never populates `confirmation_*` — explicit regression test asserting all five stay `NULL` after a fresh backfill run.
- Finding C fix — the full §4.5 scenario list (evidence revision creates a new memory + supersedes the old one; old memory byte-for-byte unchanged; lineage columns correct; fresh monitor_state with no carried-forward execution evidence; identical resubmission stays idempotent; non-trigger evidence changes still fire a revision; a genuine setup_key change still takes the unchanged structural path).
- State machine: a fresh Type-A memory is created directly in `CONFIRMED` (not `APPROVED`) when `confirmation_rule/level` are present; a fresh Type-B memory with only `trigger_*` stays `APPROVED` until a monitor-driven test transitions it.
- No monitoring, no ENTER_NOW, no `candidate_promotions` writes, no live market-data calls anywhere in this test suite — same discipline as every prior test file this session.

---

## 10. Ordered implementation steps

1. Schema: 5 confirmation columns × 2 tables (§1/§7) + `revision_of_memory_id`/`revision_reason` on `approved_setup_memories` + `superseded_by_memory_id` on `approved_setup_monitor_state` (§4.1), extended `ApprovedSetupMonitorStateName` enum.
2. Validation rules A–D (§2) in `record_candidate_visual_review`.
3. Thread `confirmation_*` through `_create_approved_setup_memory` / `_sync_approved_setup_memory_on_review` (§3); confirm backfill stays a no-op for these columns.
4. `_review_evidence_materially_differs` (§4.2) + the revised `decision == "approve"` branch (§4.3) — this is now settled scope for V1, not an open decision.
5. Test suite (§9 + §4.5 in full).
6. `setup_board.js` Finding-A fix from the audit (§17 item 1 there) — read state from `approved_setup_memories`/`monitor_state`, not live candidate fields. This is a prerequisite for everything downstream, independent of Type A/B.
7. Monitor: register the periodic task, invalidation check, Type B trigger-satisfaction check, Type A freshness check, execution-window/current-R:R check, `approved_setup_monitor_events` logging (audit §11/§12, cost-split per §6 here).
8. UI: Review Queue confirmation block (§8), Approved/Watch board copy update — "Status: Not monitored yet" removed only once step 7 is deployed and verified against real data for several real trading days, not on merge.
9. Re-verify FFIV/NVDA/CLH against the live system once deployed: confirm the two Type-A memories still correctly show no path to `CONFIRMED` (legacy, no anchor — expected and correct), and confirm CLH's *next* real review (submitted through the new form) produces a proper `CONFIRMED` memory **that supersedes memory id 3 via the new revision path** — this is the real, live regression case §4.5 was written to cover, not a hypothetical. Production is not touched to force this — it happens naturally the next time the human reviews CLH through the updated form.

Still no code has been written for any of this. §4's decision is now resolved (implement); everything in this plan is ready to build on your go-ahead.
