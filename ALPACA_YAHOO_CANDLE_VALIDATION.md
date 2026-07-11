# Alpaca vs Yahoo Candle Validation

STATUS: VALIDATION FRAMEWORK DEPLOYMENT PENDING

This document is a deployment checkpoint for the Sprint 2 validation framework.
It is not the final Yahoo-vs-Alpaca candle validation report.

## Execution Context

- Checkpoint date: 2026-07-11
- Current local base commit: `68698da` (`Add Alpaca provider abstraction and comparison framework`)
- Railway status endpoint: Alpaca credentials configured
- Active production provider: `yahoo`
- Alpaca production activation: no
- Yahoo removal: no
- Scanner strategy changes: none

## Current Blocker

Railway has valid Alpaca environment variables, but the deployed service is still
running the older comparison path from commit `68698da`. That path returns HTTP
500 for provider comparison requests after credentials are present.

Observed live checks before this checkpoint:

- `/api/data-provider/status`: HTTP 200, `alpaca_configured: true`, `stock_data_provider: yahoo`
- `/api/data-provider/compare?ticker=AAPL&period=1y&interval=1d`: HTTP 500
- `/api/data-provider/compare?ticker=AAPL&period=2y&interval=1wk`: HTTP 500
- `/api/data-provider/compare?ticker=AAPL&period=60d&interval=4h`: HTTP 500

Because local execution does not have Railway secrets, the safer full-watchlist
diagnostics must be deployed before the real candle comparison can continue.

## What This Checkpoint Adds

- Full-watchlist comparison routing through `/api/data-provider/compare?full=true`
- Production scanner provider remains `yahoo`
- Alpaca remains comparison-only and inactive for production scanner output
- Sanitized comparison responses with:
  - `success`
  - provider status
  - ticker
  - interval
  - error classification
  - sanitized error message
  - readiness category
- Full default-watchlist diagnostics across:
  - 1D: `1y` / `1d`
  - 1W: `2y` / `1wk`
  - 4H: `60d` / `4h`
- Incomplete-candle diagnostics fields for each provider
- Missing-bar, duplicate-timestamp, out-of-order, and sufficient-history checks

## Security Constraints

The diagnostics must never expose:

- API keys
- secret keys
- authentication headers
- full upstream payloads
- stack traces
- account identifiers

This report intentionally does not include credential values.

## Strategy Freeze

This checkpoint preserves Stock Scanner Strategy v1.0.

No changes were made to:

- scanner strategy
- indicators
- trend scoring
- setup grading
- confirmation logic
- entry calculations
- stop calculations
- target calculations
- risk/reward calculations
- `ENTER NOW` logic
- alerts
- journaling
- filters
- earnings logic
- option-selection logic

## Validation Status

Real Yahoo-vs-Alpaca candle validation has not yet been completed.

No readiness result in this file should be treated as final.

The final validation report must be generated after this checkpoint is deployed
and the comparison endpoint returns sanitized structured JSON for:

- `/api/data-provider/compare?ticker=AAPL&period=1y&interval=1d`
- `/api/data-provider/compare?ticker=AAPL&period=2y&interval=1wk`
- `/api/data-provider/compare?ticker=AAPL&period=60d&interval=4h`
- `/api/data-provider/compare?full=true`

## Live-Market Validation Still Required

Because this checkpoint is occurring while the market is closed, actively
forming-candle behavior cannot be honestly passed yet.

After the framework deploys, the final report must mark forming-candle tests as:

`PENDING LIVE-MARKET VALIDATION`

until the comparison is rerun during:

- pre-market
- regular session
- after-hours

## Next Step

Deploy this validation framework checkpoint, verify the three AAPL comparison
requests return structured JSON instead of HTTP 500, then run the full
93-ticker / 279-combination comparison against the deployed service.
