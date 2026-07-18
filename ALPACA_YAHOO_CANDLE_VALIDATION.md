# Alpaca vs Yahoo Candle Validation

STATUS: 1D/1W READY FOR CONTROLLED ALPACA ROUTING; 4H REMAINS YAHOO

This document records the current provider-migration validation for Kairos
stock-scanner candles. It replaces the stale July 11 validation notes, which
covered the prior 93-symbol watchlist before the Yahoo MultiIndex flattening
fix, Alpaca pagination support, the current 113-symbol watchlist, and the
daily swing precision tolerance.

## Current Recommendation

- Route 1D and 1W stock candles to Alpaca behind a reversible provider profile.
- Keep 4H stock candles on Yahoo.
- Keep options chains and earnings on Yahoo-backed `Ticker()` behavior.
- Do not change scanner strategy, setup qualification, grades, entries, stops,
  targets, alerts, contracts, journal behavior, or card rendering.

Recommended runtime profile:

`STOCK_DATA_PROVIDER_PROFILE=proposed_hybrid_alpaca_1d_1w_yahoo_4h`

Rollback:

Unset `STOCK_DATA_PROVIDER_PROFILE` or set it to `production_yahoo`.

## Execution Context

- Current watchlist size: 113 symbols
- Validation target: current Railway production deployment
- Production comparison endpoint:
  `/api/data-provider/compare?hybrid=true&full=true`
- Production profile tested:
  - 1D: Yahoo
  - 1W: Yahoo
  - 4H: Yahoo
- Proposed hybrid profile tested:
  - 1D: Alpaca
  - 1W: Alpaca
  - 4H: Yahoo
- Validation includes fixes already deployed before routing:
  - Yahoo/yfinance MultiIndex flattening fix for single-symbol fallback
  - Alpaca historical bars pagination
  - Daily-only swing/trend precision tolerance

## Strategy Freeze

This checkpoint preserves Stock Scanner Strategy v1.0.

No changes were made to:

- scanner strategy
- indicators
- setup qualification
- setup grading
- confirmation logic
- entry calculations
- stop calculations
- target calculations
- risk/reward calculations
- status buckets
- alerts
- journaling behavior
- card rendering
- earnings logic
- option-selection logic

The approved provider routing changes only which candle provider serves each
timeframe.

## Fresh Hybrid Validation Results

Fresh Railway validation against the current 113-symbol watchlist produced:

- Exact strategy-output matches: 91
- Reviewed rounding-only differences: 20
- Structural edge case: 1
- Pre-existing data gap: 1

Raw diagnostic classification may still mark the rounding-only rows as
`material_difference` when target/stop/risk numbers differ by more than one
cent after downstream rounding. Manual review found these to be explainable
provider precision differences, not different trade decisions.

The one structural edge case after the AAL fix is BAC. It shows the same
boundary-proximity pattern: tiny provider differences around swing/trend
selection can move which timeframe wins. This is expected to shift between
tickers over time rather than remain a stable BAC-only problem.

The pre-existing data gap is SQ, which continues to return no usable price data
in this environment. This is the same cleanup category as MRO, which was later
removed after confirming it was acquired/delisted.

## AAL Discovery And Fix

AAL originally showed a material Yahoo-only vs hybrid difference:

- selected timeframe flipped from 4H to 1D
- grade flipped from C to A
- entry status flipped from Near Entry to Too Far

Both profiles used Yahoo for 4H, so the variable was the 1D/1W data source.
Investigation showed the 1D swing/trend path was sensitive to exact equality at
a daily swing boundary. Yahoo represented the relevant low as an exact tie,
while Alpaca differed by roughly half a cent. That was enough to alter the
daily swing sequence, the trend decision, and then best-timeframe selection.

Fix:

- Commit `c76aec2`
- Added daily-only `$0.006` swing/trend tolerance
- Applied only to 1D swing/trend handling
- Left 4H exact because the observed issue was daily provider precision noise

Safety validation:

- Full Yahoo-only before/after scan across 113 symbols showed zero output
  regressions with the final `$0.006` tolerance.
- AAL no longer shows the original timeframe/grade/entry-status flip in the
  live hybrid comparison.

## BOS Precision Investigation

`_detect_bos()` still uses strict `>` / `<` comparisons against prior swing
levels. That carries the same theoretical precision-sensitivity pattern as the
original AAL swing/trend issue.

Current evidence does not justify a preemptive BOS tolerance change:

- BOS candidate close deltas across the current watchlist were small.
- At BOS boundary comparisons, max observed Yahoo-vs-Alpaca close delta was
  about `$0.005`.
- The closest real BOS-threshold boundary found was about `$0.01` away.
- No current ticker showed a BOS-confirmation flip caused by provider precision.

Status:

- Known monitored risk
- Not a current migration blocker
- No BOS logic change was made

## Other Exact-Comparison Risks

CHoCH, higher-timeframe structure, and order-block zone logic contain related
exact or strict price comparisons. They were not individually investigated in
this validation pass.

Status:

- Candidate future investigation if a live mismatch surfaces
- Not a current blocker for 1D/1W routing

## 4H Session Anchoring

4H remains on Yahoo. Prior validation found Alpaca 4H bars have session
anchoring/aggregation differences that are not equivalent to the current Yahoo
4H feed used by the scanner.

Because the scanner's 4H setup logic is sensitive to candle boundaries, this is
not a safe cutover candidate yet. Re-examining 4H would be a separate project,
likely requiring explicit session-aligned aggregation validation rather than a
simple provider switch.

## Production Routing Plan

Use an environment-controlled provider profile:

- Default: `production_yahoo`
- Optional hybrid: `proposed_hybrid_alpaca_1d_1w_yahoo_4h`

Default behavior remains Yahoo-only when `STOCK_DATA_PROVIDER_PROFILE` is not
set.

To enable hybrid routing in Railway:

`STOCK_DATA_PROVIDER_PROFILE=proposed_hybrid_alpaca_1d_1w_yahoo_4h`

To roll back:

`STOCK_DATA_PROVIDER_PROFILE=production_yahoo`

or remove `STOCK_DATA_PROVIDER_PROFILE`.

## Open Follow-Ups

- Clean up SQ if it remains a persistent no-data symbol.
- Revisit BAC-like boundary cases after several live hybrid scans.
- Investigate BOS/CHoCH/HTF/order-block tolerance only if real mismatches
  surface.
- Treat 4H Alpaca migration as a separate session-anchoring project.
