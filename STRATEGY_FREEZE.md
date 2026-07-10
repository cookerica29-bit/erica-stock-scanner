# Stock Scanner Strategy Freeze v1.0

Frozen strategy name: Stock Scanner Strategy v1.0

Frozen baseline commit: `7441aac88d5cdf2bb479b85f0e73e4cec629ed57`

Purpose: protect the current stock scanner methodology from accidental drift while UI, speed, caching, providers, notifications, and app readiness continue to improve.

## Frozen Strategy Surface

The following must not change without an explicit strategy-change request:

- trade direction logic
- trend / HTF alignment logic
- setup qualification
- setup grades
- confirmation rules
- structure / BOS / order-block logic
- premium / discount location logic
- entry status logic
- entries
- stops
- targets
- R:R requirements
- trade-stage classification
- ENTER NOW / ALMOST READY / WAITING / SKIP mapping where it reflects strategy qualification
- alert qualification logic
- journal qualification logic

## Safe Change Surface

The following may still change safely when they do not alter scanner outcomes:

- UI styling and wording
- card layout
- caching
- background refresh
- performance
- provider abstraction
- Yahoo / Alpaca integration
- loading states
- notifications delivery
- mobile responsiveness
- logging and diagnostics
- contract display lifecycle
- execution-tab presentation
- universe inclusion rules only when explicitly requested and documented

## Change Management Rule

Any future strategy change must:

- be explicitly labeled `STRATEGY CHANGE`
- explain the trading reason
- identify affected outputs
- increment the strategy version, for example `v1.1`
- include before/after regression results
- never be bundled into a UI, cache, performance, or provider commit

## Regression Guard

Run:

```bash
python3 tests/stock_strategy_freeze_v1.py
```

The regression guard checks representative frozen outputs and source hashes for core strategy functions.

If it fails, either revert the accidental strategy drift or perform a documented strategy version change.
