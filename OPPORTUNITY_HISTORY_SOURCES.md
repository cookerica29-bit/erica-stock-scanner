# Opportunity History Sources

Status: developer-only shadow analytics.

This report documents the authoritative sources available for Opportunity Remaining history. It does not change scanner strategy, card eligibility, entries, stops, targets, journal behavior, alerts, or option selection.

| Field | Current Source | Persistence | Server-side? | Browser-side? | Status |
| --- | --- | --- | --- | --- | --- |
| Scanner setup snapshots | `scannerRows`, `scannerNearMiss`, card setup payloads | Browser memory | No | Yes | Not durable server-side |
| Setup ID | `setup_id` or deterministic setup identity helpers | Local journal | No | Yes | Available |
| Ticker | Scanner row and journal `ticker` | Browser memory / localStorage | No | Yes | Available |
| Direction | Scanner row and journal `direction` | Browser memory / localStorage | No | Yes | Available |
| Grade | `setup_grade`, `setupGrade`, `scanner_status` | Browser memory / localStorage | No | Yes | Available |
| Setup family / lifecycle type | `entryType`, `setupStatus`, `phase`, `setup_family` | Browser memory / localStorage | No | Yes | Partially available |
| Planned entry | `entry_price`, `entry` | Browser memory / localStorage | No | Yes | Available |
| Stop | `stop_price`, `plannedStop`, `stop` | Browser memory / localStorage | No | Yes | Available |
| TP1 / TP2 / TP3 | `target_price`, `tp1`, `tp2`, `tp3`, `plannedTp*` | Browser memory / localStorage | No | Yes | TP1 available; TP2/TP3 partial |
| Entry-trigger time | `first_entry_touch_at`, `entry_reached_at` | Local journal | No | Yes | Available after tracking |
| Target-hit time | `first_target_touch_at`, `tp1_reached_at` | Local journal | No | Yes | Available after tracking |
| Stop or invalidation time | `first_stop_touch_at`, `tracking_completed_at`, `completion_reason` | Local journal | No | Yes | Partially available |
| MFE / MAE | `maximum_favorable_excursion`, `maximum_adverse_excursion` | Local journal | No | Yes | Available after tracking |
| Existing replay dataset | None found as a durable candle-level replay store | Missing | No | No | Missing |

## Persistence Approach

Sprint 2 adds a separate browser-local `kairos_opportunity_history_v1` dataset generated only by an explicit developer import from the existing local journal. The original journal remains unchanged. Import uses deterministic record IDs and refuses to overwrite richer existing records with poorer imported records.

## Replay Limitation

Until a candle-level replay store exists, lifecycle replay snapshots are milestone approximations from persisted journal outcomes. They are useful for shadow stability checks, but they are not a substitute for historical candle replay.
