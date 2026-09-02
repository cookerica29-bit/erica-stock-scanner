# Kairos — Entry-Reached Alert V1: Audit + Implementation Report

**Status: implemented locally, fully tested, NOT pushed or deployed.**

**2026-09 addendum — three decisions applied and verified, this round:**
1. **Session behavior**: confirmed and reaffirmed session-blind, exactly as
   found in the original audit. No dedicated RTH-previous-price field
   added. No code change (already correct); re-verified with the full
   regression sweep and a fresh real-browser check.
2. **STALE**: now excluded from a new `ENTRY_REACHED` event
   (`ENTRY_REACHED_INELIGIBLE_STATES` now `{"EXTENDED", "STALE"}`).
   Verified no conflict with STALE's existing semantics — see §1.9-new.
3. **Quote freshness**: fully audited (§1.2) and a narrow, entry-reached-
   only freshness guard implemented (`_entry_reached_quote_is_fresh`),
   using the real, already-present Alpaca quote timestamp that was
   previously discarded at this call site. One honest, disclosed residual
   limitation found and documented, not hidden — see §1.2 and §5.

## Product invariant (restated, unchanged)

`ENTRY_REACHED` means "the reviewed location is available." It does not mean
confirmation, ACTIONABLE, ENTER_NOW, or trade authorization. Nothing in this
implementation gives it any of that authority.

---

## 1. Audit findings (requested before any implementation)

### 1.1 Authoritative frozen-entry field
`approved_setup_memories.approved_entry` — frozen at review time, immutable
for the memory's lifetime (a revision or a genuinely new setup_key always
creates a **new** memory row; the old row's `approved_entry` is never
mutated). Confirmed by direct test
(`test_scanner_entry_drift_does_not_change_authoritative_entry`): a later
scan reporting a different live entry for the same ticker leaves the frozen
value untouched, and detection uses the frozen value exclusively.

### 1.2 Live price source, monitor cadence, and quote freshness (fully audited + implemented this round)

`_latest_quotes_for_previews` → `AlpacaMarketDataProvider.latest_quotes` in
`market_data.py` → Alpaca's `/v2/stocks/quotes/latest` (real NBBO quote
endpoint). Cadence: `MONITOR_TICK_SECONDS = 300` (5 minutes), the same
shared tick Watch Lifecycle V1 runs on. One quote per tick, no intra-tick
reconstruction — matches the spec's own framing.

**What price is actually used**: `AlpacaMarketDataProvider._quote_price`
returns the bid/ask **midpoint** when both sides are present (branch
`"midpoint"`), or a one-sided bid/ask otherwise (`"bid_only"`/`"ask_only"`).
Entry-reached does **not** special-case the one-sided branch the way
`_entry_proximity`'s tight ATR-based ENTER_NOW gate already does (distrust
+ fall back to daily close, per a real, previously-confirmed BRK.B/AER
bug in this codebase) — a coarse "did price cross a level" location
question is far less sensitive to a few cents of one-sided-quote noise
than that tight proximity gate is, and firing a tick late from a stale
one-sided print is a much smaller cost here than getting actionability
confidently wrong. What matters for entry-reached specifically is the
quote's **age**, not its sidedness.

**Does it include an authoritative timestamp — yes.** Alpaca's own `"t"`
field, a real RFC3339 UTC string, already flows all the way through
`latest_quotes()` into `quote["timestamp"]`.

**Was it being discarded — yes, confirmed, at exactly one place.**
`run_approved_setup_monitor_tick` read `quote.get("price")` and nothing
else; `quote.get("timestamp")` was fetched by the provider but never once
compared to "now" anywhere in the monitor. This was true for the **whole**
monitor, not just entry-reached — **and remains true for invalidation and
current-R:R after this change**, exactly per your instruction not to alter
unrelated scanner/monitor behavior. Only entry-reached now reads it.

**What happens when markets are closed or Alpaca returns an old quote**:
Alpaca's IEX feed (the default here — no `feed=sip` param is sent anywhere
in this codebase) only trades during the regular session. A "latest quote"
fetched pre-market/after-hours/on a weekend returns the most recent quote
IEX actually has — a real, well-formed, two-sided response object that is
nonetheless hours or days stale. **This could not be verified against live
Alpaca infrastructure from this environment** (no working API credentials
available here) — it is Alpaca's documented IEX behavior plus this
codebase's own prior finding (the BRK.B/AER case above) that a
valid-looking quote object can still be far from anything currently true,
not an empirical confirmation I obtained myself. Said plainly: I'm
reporting what the code and documentation say, not what I watched happen
live.

**The guard implemented**: `_entry_reached_quote_is_fresh(quote, now)` —
parses `quote["timestamp"]` via `_parse_iso_to_utc` (the *existing* shared
parser `_is_stale` already uses for its own freshness math — same
function, same "monitor freshness math" docstring, reused rather than
reinvented) and requires `(now - timestamp) <= timedelta(minutes=
ENTRY_REACHED_QUOTE_MAX_AGE_MINUTES)`. `ENTRY_REACHED_QUOTE_MAX_AGE_
MINUTES = 15` — an unvalidated placeholder, same honesty-about-thresholds
convention as every other constant in this file (`RR_WARNING_THRESHOLD`,
`ENTRY_PROXIMITY_MAX_PCT_DEFAULT`, `EXECUTION_EVIDENCE_FRESHNESS_HOURS`,
etc.): generous against the 5-minute tick cadence (a real IEX quote should
be seconds old during actual trading) while decisively rejecting an
hours/days-stale closed-market quote. **Fails closed** on a missing quote,
a missing/unparseable timestamp, or a too-old timestamp — the row is left
untouched for entry-reached only (not the whole row: `last_checked_at`/
`last_live_price` still advance exactly as before, invalidation/R:R still
evaluate exactly as before). Narrowly scoped — wired into entry-reached's
own eligibility check only, nothing else in the monitor was touched.

**One honest limitation found and disclosed, not hidden** (verified via a
failing test I then fixed correctly, not papered over): because
`last_live_price` still advances on a stale-quote tick (unchanged,
pre-existing behavior for the whole monitor), if that stale tick's price
was already on the reached side, it becomes the new "previous price"
baseline. A **later fresh** tick observing the exact **same** price then
finds no new cross (previous was already past entry too) and does not
retroactively fire. This is the same *class* of gap the session-gate
finding surfaced, but meaningfully narrower in practice: it requires
Alpaca to return a stale-timestamped quote *during otherwise-live
monitoring* — an anomalous feed-health scenario, not a routine nightly
certainty like the session case was. If price genuinely moves again after
the stale observation (back above entry, then a real re-cross), the event
still fires correctly — confirmed by dedicated tests for both the
limitation itself and the self-healing case. Fixing this completely would
require `last_live_price` to stop advancing on a stale tick — a change to
**every** row's displayed "current price," not narrowly scoped to
entry-reached — so, per your explicit instruction, I did **not** build
that. See §5.

### 1.3 Exact LONG/SHORT touch/cross semantics
Implemented exactly as specified in `_entry_reached()`:
- **First observation** (`last_live_price IS NULL` — a real, persisted
  column, never reset by a restart): LONG reached if `price <= entry`;
  SHORT if `price >= entry`.
- **Subsequent ticks**: LONG reached if `previous > entry and current <=
  entry`; SHORT if `previous < entry and current >= entry`.
- Equality counts on both sides. A gap-through records the **real observed
  price**, never a fabricated print at the exact entry level (verified by
  `test_long_above_to_below_entry_gap_through_fires_once` /
  the SHORT mirror).

`last_live_price` (already on `approved_setup_monitor_state`, already
updated every tick for every active row, already survives restarts as a
real DB column) turned out to be exactly the "previous price" evidence the
spec asked me to check for — no new column needed for it.

### 1.4 Gap-through behavior
Confirmed working exactly per the spec's own worked example: previous
above entry → next valid quote below entry → one event, `entry_reached_
price` = the real observed price (e.g. 60.70 in the OXY-shaped example),
never `entry`/60.94 itself.

### 1.5 Restart/outage recovery
Deterministic, by construction: `last_live_price` and `entry_reached_at`
are both real, persisted columns — a restart never resets either. A fresh
memory's first-ever tick uses the "first observation" rule (no invented
history). A row that already had prior monitor observations before a
restart recovers using the real, persisted `last_live_price` exactly as if
no restart had happened — confirmed by
`test_process_restart_does_not_duplicate` (a "restart" is modeled the same
way the rest of this test suite already does: a fresh app/TestClient
against the same on-disk db_path, since there is no in-process cache
either way).

### 1.6 Interaction with WATCH triggers
Fully independent, confirmed by two dedicated tests. Reaching entry while
`WAITING_FOR_TRIGGER` persists exactly one `ENTRY_REACHED` event and
**does not** touch `trigger_satisfied_at`, does not change `state`. The
trigger later satisfying proceeds through the unmodified Watch Lifecycle
V1 code path, using its own unrelated evidence.

### 1.7 Interaction with APPROVE/ACTIONABLE
Entry-reached carries zero actionability authority. Verified with the real
(un-mocked) R:R math: a row that reaches entry but whose execution window
has independently degraded still reads `EXTENDED`, never `ACTIONABLE`,
from entry-reached alone
(`test_approve_reaches_entry_does_not_auto_actionable`). Same-tick
ordering is deterministic: `ENTRY_REACHED` is evaluated and inserted
**before** the invalidation/trigger/actionability block runs for that row,
so its `id` is always lower than any state-transition event logged later
in the same tick (`test_invalidated_row_never_gets_additional_entry_
reached_event` proves this directly for the LONG-invalidation case, where
the two legitimately coincide on one tick — see 1.9).

### 1.8 Market-closed / extended-hours semantics — **confirmed, reaffirmed this round**

Your stated preference was regular-session-only. I built that first,
exactly as asked, and it revealed a real bug during testing: `last_live_
price` already advances on **every** tick regardless of session (true for
the whole monitor, unrelated to this feature). A session-gated
entry-reached check would let an off-session tick silently "consume" the
crossing evidence into `last_live_price` — by the time the gated,
regular-session tick finally ran, the "previous vs current" comparison no
longer showed a crossing (both sides already past entry), so the event
would **permanently never fire**, even though the reviewed location was
genuinely reached. I caught this via `test_fires_regardless_of_market_
session` before it was a production bug, not after.

**You confirmed this finding and asked to keep it session-blind, with no
dedicated RTH-previous-price field for V1.** No code change was needed
this round (already implemented that way) — re-verified with the full
regression sweep and a fresh real-browser check; `test_fires_regardless_
of_market_session` still passes, proven with a real Saturday timestamp.
This matches the rest of the monitor's existing, uniformly session-blind
cadence and needs no new schema. Quote *freshness* (§1.2) is now the
narrower, more precise guard doing the actual "don't fire off a
genuinely-not-live quote" job — it doesn't care *why* a quote is old
(closed session, a stuck feed, anything else), only *how old* it is,
which is the honest question that actually matters here.

### 1.9 Dedup key
`approved_setup_monitor_state.entry_reached_at` (frozen once written,
never overwritten — same convention as `trigger_satisfied_at`), one row
per `approved_memory_id` = one row per durable setup generation. This
doubles as the dedup gate itself (`WHERE entry_reached_at IS NULL`), no
separate lookup query needed. A genuinely new setup generation (new
memory row via revision or a different setup_key) gets its own row and
its own independent dedup scope — confirmed by
`test_new_setup_generation_fires_independently`.

I deliberately did **not** route the event-log insert through the
existing `_record_monitor_event`'s dedup rule (compares only the last
logged event's `to_state`). That rule is unsuitable here: rows in
`EVIDENCE_CLEARED_MONITOR_STATES` can legitimately revisit an earlier
`to_state` value over their lifetime (e.g. `CONFIRMED → EXTENDED →
CONFIRMED` if R:R recovers), so a genuine, never-before-logged
`ENTRY_REACHED` insert could be silently swallowed purely because its
`to_state` (the row's own *unchanged* current state — `from_state ==
to_state` deliberately, since this is an observation, not a transition)
happens to coincide with an unrelated, already-logged transition event.
`_record_entry_reached_event()` is a small, separate insert into the same
table instead, with its own event-type-scoped dedup logic.

**A real, disclosed same-tick coincidence, found and handled correctly,
not avoided**: for a LONG, `approved_stop` is always below `approved_
entry` by construction (support below entry, target above). So invalidation
(`price <= stop`) is *geometrically always* also "at or below entry" —
entry-reached and invalidation legitimately fire on the exact same tick
whenever a setup gaps straight through both levels in one move. This isn't
a bug; it's honest evidence of where price actually was. Two pre-existing
tests in `tests/approved_setup_monitor_v1.py` needed updating because they
encoded the **old** (pre-Entry-Reached) event count and didn't anticipate
this — same pattern as prior phases of this project: the tests were
verified to be checking the *old*, now-superseded contract, then rewritten
to assert the new, correct one (see §4).

### 1.10 ACTIONABLE interaction / event ordering
Contract: **persist ENTRY_REACHED → evaluate invalidation/trigger/
actionability → persist any resulting state-transition event**, exactly
the order you suggested. Implemented via insertion order within the same
per-row loop body (entry-reached evaluated immediately after `current_rr`
is computed, before the `if _is_invalidated(...)` block). Confirmed
deterministic via the id-ordering assertion in the invalidation test.

### 1.11 STALE eligibility — **changed this round, semantics verified, no conflict found**

`ENTRY_REACHED_INELIGIBLE_STATES` is now `{"EXTENDED", "STALE"}`.

**Exact current STALE semantics, verified directly against the code**:
a memory reaches `STALE` only from `EVIDENCE_CLEARED_MONITOR_STATES`
(`CONFIRMED`/`TRIGGER_SATISFIED`/`ACTIONABLE`/`EXTENDED`) — i.e. only
*after* its evidence gate has already cleared at least once — when its
freshness anchor (`trigger_satisfied_at`, else `confirmed_candle_time`,
else `approved_at`) is older than `EXECUTION_EVIDENCE_FRESHNESS_HOURS`
(72 hours). Per this monitor's own existing, pre-dating-this-feature
design: `STALE` is **deliberately excluded from re-evaluation** —
"once a memory goes stale, the monitor's job for it is done until a
fresh human review supersedes it" (its own code comment) — and it
**never auto-revives** back to `ACTIONABLE` on its own.

**No conflict found with adding it to entry-reached's exclusion list.**
Two things confirmed directly, not assumed:
1. **Invalidation still runs for a STALE row, unconditionally, every
   tick** (the invalidation check is first, always, for every active
   row, regardless of state) — excluding `STALE` from entry-reached
   eligibility doesn't touch that at all, and the two can still coincide
   on the same tick exactly like the LONG-invalidation case in §1.9 (a
   STALE row invalidating and observing price at/past entry
   simultaneously correctly still records **no** entry-reached event,
   confirmed by `test_stale_row_never_fires_a_new_entry_reached_event`,
   which drives exactly this sequence with a real, controlled staleness
   anchor).
2. **A STALE memory is not a dead end for the setup itself** — a fresh
   human review on the same setup_key creates a genuinely **new** memory
   row (a new setup generation), which gets its own, fully independent
   entry-reached eligibility and dedup scope, unaffected by the old
   row's STALE status. Nothing about excluding STALE here blocks that
   path.

This is fully consistent with the reasoning already used for `EXTENDED`
in the original spec: once Kairos has functionally stopped actively
surfacing a setup, a *new* opportunity-style alert for it (as opposed to
the underlying data quietly continuing to update) shouldn't fire either.

---

## 2. Existing "entry reached" code — audit and semantic comparison (`smart_notifications.py` / `public/stock_alerts.js`)

**These are semantically incompatible with the new feature and were left
completely untouched — Option C.**

| | Legacy (`smart_notifications.py`) | New (Entry-Reached Alert V1) |
|---|---|---|
| Entity monitored | Every raw scanner opportunity/options row (`result["rows"]`/`["near_miss"]` from every `/scan` call) — evaluated on **every** scan, whether or not a human ever reviewed it | Only `approved_setup_memories` rows a human explicitly approved or watched |
| Entry field | `setup.get("entry")`/`entry_price` — the **live, drifting** scanner value at scan time | `approved_setup_memories.approved_entry` — frozen at review time |
| Identity/dedup key | `setup_identity()` — a hash of ticker/direction/timeframe/**entry/stop/target from the raw candidate** — changes if the scanner recomputes those by even a cent | `approved_memory_id` — a durable, immutable DB row id |
| Server-side durable? | Yes (`smart_notification_entity_states`, a real SQLite table) | Yes (`approved_setup_monitor_state`/`_events`) |
| Dedup across restart? | Yes, via the persisted `entry_reached` flag per entity | Yes, via `entry_reached_at` |
| Tied to Approved/Watch lifecycle? | **No** — entirely separate candidate universe, entirely separate identity scheme | Yes, by construction |
| Implies trade readiness? | **Yes, explicitly** — `normalize_bucket()` aliases `"ENTRY_REACHED"` directly to `"ENTER_NOW"`, and the built notification message says "Review the option plan before executing" | **No** — the product invariant is enforced by construction; nothing here writes to any actionability field |
| Client dependency | `public/stock_alerts.js` runs entirely browser-side, loaded only on `index.html` (the raw Candidates Dashboard) — not loaded on `approved_setups.html`/`watch_setups.html` at all | Server computes the verdict; the browser only renders the persisted result |

Given the drifting entry, the drift-vulnerable identity hash, and the
explicit `ENTER_NOW` conflation, reusing or merging this mechanism would
either break its existing, already-deployed options-notification behavior,
or reintroduce exactly the setup_key-drift fragility this session's
Approved/Watch lifecycle work was built to get away from. **Recommendation:
C — leave it alone, entirely separate.** No file in `smart_notifications.py`
or `public/stock_alerts.js` was touched.

### Notification plumbing available today
Same answer as Watch Lifecycle V1's own audit: **no delivery transport
exists** for the Approved/Watch lifecycle (no push, no email, no external
delivery) — `approved_setup_monitor_events` *is* the alert surface, read
by a future UI. `ENTRY_REACHED` uses that exact same surface; nothing new
was built.

---

## 3. Implementation — files and functions changed

**`candidates_router.py`**
- `approved_setup_monitor_state`: two new columns, additive
  (`ALTER TABLE ... ADD COLUMN` if missing, same proven-safe pattern used
  throughout this project) — `entry_reached_at TEXT`, `entry_reached_price
  REAL`. Added to the CREATE TABLE literal, the migration function, the
  `ApprovedSetupMonitorStateOut` Pydantic model, and `_row_to_approved_
  setup_monitor_state`.
- `ENTRY_REACHED_INELIGIBLE_STATES = {"EXTENDED", "STALE"}` — audited
  against the real `ApprovedSetupMonitorStateName` values, not an
  invented alias. Terminal states (`INVALIDATED`/`WITHDRAWN`/
  `SUPERSEDED`) are already excluded structurally (`_monitor_active_rows`
  only returns `ACTIVE_MONITOR_STATES`, so a terminal row is never even
  fetched). `STALE` added this round — semantics verified, no conflict
  with existing STALE behavior (§1.11).
- `ENTRY_REACHED_QUOTE_MAX_AGE_MINUTES = 15` and
  `_entry_reached_quote_is_fresh(quote, now)` — the new, narrow
  freshness guard, entry-reached-only (§1.2). Reuses `_parse_iso_to_utc`,
  the existing shared timestamp parser.
- `_entry_reached(direction, previous_price, current_price, approved_entry)`
  — the pure touch/cross function (§1.3).
- `_record_entry_reached_event(...)` — the dedicated insert (§1.9).
- `run_approved_setup_monitor_tick`: one new block per row, inserted
  immediately after `current_rr` is computed and before the invalidation
  check (§1.10), now also gated on the freshness guard; docstring updated
  to document the reaffirmed session-blind decision (§1.8), STALE (§1.11),
  and the ordering contract.
- No changes to `_sync_approved_setup_memory_on_review`, `_create_
  approved_setup_memory`, the RTH trigger machinery, any WATCH_* event
  type, or invalidation/current-R:R's own (still freshness-blind, still
  session-blind) price handling — all reused completely unchanged.

**`public/setup_board.js`**
- `entryReachedLine(record)` — reads `monitor_state.entry_reached_at`/
  `entry_reached_price`/`state` (already fetched by both boards' existing
  `?include_inactive=true` call — no new endpoint, no new fetch). Renders
  `"Entry not reached"`, `"Entry reached <date> at $<price>"`, or with a
  `" — confirmation still pending"` suffix when `state === 
  'WAITING_FOR_TRIGGER'`. Never infers from `item.current_price` — always
  the persisted server verdict, per your explicit instruction.
- Wired into `renderCard` right under the current-price row, on both
  boards, for both `approve`- and `watch`-origin cards alike.

**`public/approved_setups.html` / `public/watch_setups.html`**
- New `.entry-reached-line` CSS rule (mirrors the existing small-metadata
  style). Script cache-buster bumped
  (`?v=20260902-entry-reached`).

**Tests**
- `tests/entry_reached_alert_v1.py` (27 tests total) — the full checklist:
  LONG/SHORT touch/at/gap-through, first-observed determinism, duplicate/
  oscillation/restart idempotency, session-blindness (proven with a real
  weekend timestamp), WATCH-trigger independence in both directions,
  APPROVE/ACTIONABLE non-authority + deterministic same-tick ordering,
  all terminal/EXTENDED/STALE exclusions, frozen-entry immunity to
  scanner drift, no fabricated review row, independent per-generation
  dedup — **plus, this round**: a real STALE scenario driven through
  `TRIGGER_SATISFIED → ACTIONABLE → STALE` with a controlled 73-hour
  anchor, and a dedicated quote-freshness section (stale-timestamp fails
  closed, a timestamp just inside the window still fires, a missing
  timestamp fails closed, the disclosed stale-tick-consumes-evidence
  limitation demonstrated directly, and the self-healing case where price
  genuinely moves again after a stale observation).
- `tests/approved_setup_monitor_v1.py` — the same two pre-existing tests
  touched last round were revisited this round: the freshness guard
  correctly fails closed on both tests' pre-existing quote timestamps
  (one hardcoded to a past date, one a dummy `"t"` string) unless "now"
  is frozen to match — one test now explicitly freezes "now" to its
  quote's real timestamp (restoring the entry-reached+invalidation
  coincidence it was checking last round); the other's dummy timestamp is
  left as-is and its assertion reverted to the simpler, original
  single-event count, since the freshness guard now naturally and
  correctly prevents entry-reached from firing there at all.
- `tests/setup_board_v1.js` — Section N (4 tests, unchanged this round):
  not-reached, reached-with-price, reached-with-pending-confirmation, and
  the no-record-at-all defensive fallback. No UI copy changes were needed
  for STALE/freshness — `entryReachedLine` already renders "Entry not
  reached" correctly regardless of *why* `entry_reached_at` is null.

No changes were made to `smart_notifications.py`, `public/stock_alerts.js`,
the Watch Lifecycle V1 trigger machinery, or any Execution Layer V1 safety
gate. Invalidation and current-R:R remain exactly as freshness-blind and
session-blind as they were before this feature existed.

---

## 4. Test results (this round)

- **New/updated**: `tests/entry_reached_alert_v1.py` — **27/27 passing**
  (was 21; +1 STALE test, +5 quote-freshness tests).
- **Watch Lifecycle V1 regression**: `tests/watch_lifecycle_v1.py` — 22/22
  passing, unchanged.
- **Approved-memory/monitor regression**: `tests/approved_setup_memory_v1.py`,
  `tests/approved_setup_memory_concurrency_v1.py`,
  `tests/approved_setup_memory_migration_v1.py`,
  `tests/approved_setup_monitor_v1.py` (19/19, both freshness-affected
  tests fixed correctly — see §3), `tests/trigger_capture_v1.py` — all
  passing.
- **Targeted regression sweep** (Watch Lifecycle + Entry-Reached +
  approved-memory/monitor together): **107/107 passing**.
- **Full backend suite**:
  `python3 -m pytest tests/ -q --override-ini="python_files=*_v1.py *_v2.py *_v3.py" --ignore=tests/production_refresh_safety_v1.py`
  → **874 passed** (was 868), same **8 pre-existing, unrelated failures**
  present before this work started (alpaca discovery universe env-parsing,
  discovery startup registration, opportunity-ranking priority buckets,
  option-pricing retry classification, scanner-health cache key,
  verified-analytics replay parity — none touch the Approved/Watch
  lifecycle).
- **Full frontend suite**: `review_queue_auth_v1.js`, `review_queue_
  persistence_v1.js`, `setup_board_v1.js` (Section N unchanged, still
  passing), `setup_board_selfinit_v1.js` — all passing.
- **Restart/idempotency**: unchanged from last round —
  `test_process_restart_does_not_duplicate` and
  `test_repeated_ticks_on_reached_side_do_not_duplicate`.
- **Concurrency**: unchanged from last round — no new concurrent-write API
  path, no new test needed.
- **Real local verification, this round**: seeded a scratch SQLite DB
  (dummy key, never production) exercising both new decisions end-to-end
  via the real HTTP API + real manually-invoked
  `run_approved_setup_monitor_tick()` calls:
  - **STALE**: a real APPROVE row driven `CONFIRMED (105, above entry) →
    ACTIONABLE, still not reached → STALE (73h later, same anchor) →` a
    third tick with price genuinely at/through entry, confirmed
    `entry_reached_at` stayed `None` throughout, even though the row
    simultaneously invalidated on that same third tick (the same honest
    same-tick coincidence as §1.9, correctly still suppressed by the
    STALE-at-the-start-of-that-tick exclusion).
  - **Freshness**: a 6-hour-stale quote at the exact entry price produced
    no event; a later fresh quote at the *same* price also produced none
    (the disclosed limitation, reproduced live, not just in a unit test).
  - Confirmed the running app still serves and renders both boards
    cleanly against the updated schema (no errors in the server's own
    log across the whole session). No production database or key was
    touched at any point.

---

## 5. Decisions I made that you may want to weigh in on before production

1. Same-tick `ENTRY_REACHED` + `INVALIDATED` coincidence for LONG setups
   (§1.9) is geometrically unavoidable and, in my view, correct — flagged
   since it wasn't spelled out in the spec's own examples. Unchanged this
   round, now also exercised for the STALE case.
2. **The one remaining open item — quote-freshness's disclosed residual
   limitation (§1.2)**: a stale-quote tick can permanently consume
   crossing evidence into `last_live_price` if price never moves again
   after the stale observation. Narrower and rarer than the original
   session-gate bug (requires an anomalous stale-quote event *during*
   otherwise-live monitoring, not a nightly certainty), and self-healing
   the moment price genuinely moves again. Fixing it completely needs a
   broader change (stop `last_live_price` from advancing on a stale
   tick, affecting the UI's displayed "current price" for every row, not
   just entry-reached) — I did not build that, per your explicit
   instruction not to invent a broader fix. Your call whether this
   residual risk is acceptable for V1 or whether you want that broader
   change scoped as separate follow-up work.

Session behavior and STALE are both now resolved and verified — nothing
outstanding on either. Nothing else in this implementation required a
judgment call outside the spec.

---

## 6. Deployment status

**Not pushed. Not deployed.** Local commit only, per the task's explicit
instruction. Ready for your review before any push/deploy authorization.
