# Kairos Watch Contract — Track A / Track B Interface Gap

*Documentation only, per this session's explicit instruction. Codex's Track B (Unified Chart Review) is NOT integrated or edited here. No prose parser was built or proposed — `location_description` is never parsed for numeric Watch Contract fields, now or as a future plan.*

## Why this exists

Track A (this session's Watch Contract) needs a small set of **structured, numeric** fields to instantiate a contract. Track B's current `chart_review_v1` result carries the human's read as **prose** for the location. These are not the same shape, and no automatic bridge should paper over that gap by guessing numbers out of text.

## Exact fields Track A requires today

From `watch_contract_router.WatchContractDevCreateIn` (the only production-shaped contract this sprint defines):

| Field | Type | Notes |
|---|---|---|
| `ticker` | string | |
| `direction` | `"long" \| "short"` | |
| `approved_htf_thesis` | string | e.g. `"BULLISH"` / `"BEARISH"` — freeform, stored verbatim |
| `approved_current_leg` | string | e.g. `"BEARISH_CORRECTION"` — freeform, stored verbatim |
| `location.type` | string | e.g. `"PRIOR_BREAKOUT_SUPPORT"` |
| `location.lower` | **float** | numeric lower bound of the approved zone |
| `location.upper` | **float** | numeric upper bound of the approved zone |
| `location.id` | string, optional | stable provenance id, if the review process has one |
| `location.status_at_approval` | `"REACHED" \| "NOT_REACHED"` | |
| `location.reached_at` | ISO timestamp, optional | the REAL, human-observed moment price touched the location — see this session's own freshness-anchor semantics writeup (`watch_contract_router.py`'s `WatchContractLocationIn.reached_at` docstring) for why this must be the review's own real timestamp, not derived from market data |
| `invalidation.rule` | `"close_above" \| "close_below"` | |
| `invalidation.level` | **float** | |

## What Track B currently has

Per this session's read of the current chart-review result shape: primarily **`location_status`** (a status label) and **`location_description`** (free text — e.g. "price rallied into the 63.5–64.5 support zone that broke down in early August").

The numbers Track A needs (`location.lower`, `location.upper`) are **inside that prose**, not in a separate field. `location_status` likely maps cleanly to `location.status_at_approval`, but that's the only field with an obvious 1:1 today.

## What is explicitly NOT being done about this

- No parser reads `location_description` to extract `63.5`/`64.5`.
- No heuristic infers `location.type` from the wording of the description.
- No default/placeholder numeric bounds are invented when the structured fields are missing.

Any of those would violate this whole project's standing "do not invent unavailable values" discipline, and a text-scraped number silently feeding a live monitor's actual price-band logic is exactly the kind of failure mode this session has been working to keep out of the Watch Contract.

## Required schema additions for Track B (a later, explicit schema version)

**Locked field list**, refined from the draft above:

1. **`location_type`** (string, small fixed vocabulary — e.g. matching the location menu already designed in `kairos_hybrid_strategy_design_v1.md` §5.1: prior swing level, structural support/resistance, premium/discount, breakout/retest)
2. **`location_lower`** (float)
3. **`location_upper`** (float)
4. **`location_id`** (string, **optional — only when reliable provenance exists**; e.g. tied to a specific chart annotation. Omit rather than fabricate one.)
5. **`location_status`** (`"REACHED" | "NOT_REACHED"`)
6. **`location_freshness_anchor`** (ISO timestamp, only when `location_status == "REACHED"`) — renamed from the draft's `location_reached_at` to make the intent explicit: this is a freshness FLOOR for the 30M confirmation check (Watch Contract confirmation must be strictly after it), not necessarily a claim about the exact instant price touched the zone. **Sourcing rule:**
   - **Normally**: the review's own `reviewed_at` timestamp — the moment Erica looked at the chart and recorded location=REACHED, confirmation=NONE. Using the review moment (rather than trying to pin the exact historical candle where price first entered the zone) is what guarantees a structural event already visible on the chart at review time is correctly excluded as stale — see Track A's own `WatchContractLocationIn.reached_at` docstring for the full reasoning (Track A's request-time "now" fallback is the same safety principle, one step later in the pipeline, for when even `reviewed_at` isn't supplied).
   - **Only when reliably known**: the exact location-touch candle's own timestamp, if Track B's tooling can genuinely identify it (e.g. from the same charting data the review was done against) — a real, sourced timestamp, never inferred from `location_description`'s prose.
7. **`invalidation_rule`** (`"close_above" | "close_below"`)
8. **`invalidation_level`** (float)

`location_description` can absolutely keep existing alongside these — it's valuable as human-readable context in its own right (e.g. surfaced in a dashboard detail panel) — it just cannot be the *only* representation of the numbers a live monitor needs to act on.

## The bridge itself, once Track B has these fields

A thin adapter — `chart_review_result -> WatchContractDevCreateIn`-shaped payload. Field mapping is direct except one translation: Track B's `location_freshness_anchor` becomes Track A's `location.reached_at` input (Track A's own internal column stays named `location_reached_at`, unchanged by this naming refinement — only the Track B-side field name and its sourcing rule were clarified here). No inference, no text parsing. Not built this sprint; both tracks need review first, per this session's own instruction.
