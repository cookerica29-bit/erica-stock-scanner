const assert = require('assert');
const further = require('../public/further_analysis.js');
const analytics = require('../public/expected_move_analytics.js');

function setup(overrides = {}) {
  return {
    ticker: 'ATO',
    direction: 'SHORT',
    timeframe: '4H',
    setupGrade: 'B+ TRADEABLE',
    entryStatus: 'Near Entry',
    confirmationStarted: false,
    stockTrend: 'Bearish',
    setupTimeframeDirection: 'Bearish',
    stockLocation: 'Premium',
    stockPhase: 'Pullback',
    stockSetupStatusReason: 'Price is near resistance and waiting for confirmation.',
    price: 112.45,
    entry: 111.8,
    sl: 114.2,
    tp1: 106.8,
    best_contract: {
      available: true,
      type: 'PUT',
      strike: 110,
      expiry: '2026-08-21',
      dte: 38,
      bid: 2.1,
      ask: 2.25,
      spread: 0.15,
      open_interest: 1200,
      volume: 240,
      implied_volatility: 0.32,
    },
    ...overrides,
  };
}

function completed({ grade = 'B+ TRADEABLE', direction = 'SHORT', timeframe = '4H', days = 7, reason = 'target' }) {
  return {
    tracking_status: 'completed',
    completion_reason: reason,
    setup_grade: grade,
    direction,
    scanner_timeframe: timeframe,
    trading_days_to_target: reason === 'target' ? days : null,
    trading_days_to_stop: reason === 'stop' ? days : null,
    entry_price: 100,
    target_price: direction === 'SHORT' ? 90 : 110,
    maximum_favorable_excursion_r: 2,
    maximum_adverse_excursion_r: 0.5,
    target_distance_atr: 2.5,
    bars_to_target: days * 2,
    bars_to_stop: reason === 'stop' ? days * 2 : null,
    trading_days_to_entry: 1,
  };
}

function many(count, factory) {
  return Array.from({ length: count }, (_, idx) => factory(idx));
}

// Setup IDs use more than ticker, so multiple signals for one ticker remain distinct.
const signalA = setup({ entry: 111.8, signal_timestamp: '2026-07-10T14:00:00Z' });
const signalB = setup({ entry: 109.5, signal_timestamp: '2026-07-11T14:00:00Z' });
assert.notStrictEqual(further.setupIdentity(signalA), further.setupIdentity(signalB));
assert.ok(further.selectorLabel(signalA).includes('ATO'));
assert.ok(further.selectorLabel(signalA).includes('PUT'));
assert.ok(further.selectorLabel(signalA).includes('B+ TRADEABLE'));

// Expected Move remains learning below threshold.
const learningEntries = many(12, i => completed({ days: 4 + i }));
const learning = analytics.expectedMoveAnalytics(learningEntries, further.setupCriteria(signalA));
const learningReadiness = analytics.expectedMoveReleaseReadiness(learning, analytics.suggestedExpirationAnalytics(learning));
assert.strictEqual(learningReadiness.card_ready, false);
assert.strictEqual(learningReadiness.release_status, 'learning');

// Qualified exact group can produce the future two-field card contract.
const readyEntries = many(100, i => completed({ days: 6 + (i % 5) }));
const ready = analytics.expectedMoveAnalytics(readyEntries, further.setupCriteria(signalA));
const readyReadiness = analytics.expectedMoveReleaseReadiness(ready, analytics.suggestedExpirationAnalytics(ready));
const cardFields = analytics.buildExpectedMoveCardFields(readyReadiness);
assert.deepStrictEqual(Object.keys(cardFields).sort(), ['expected_move_label', 'suggested_expiration_label']);

// Fallback analytics are not treated as exact recommendations.
const fallbackEntries = [
  ...many(10, i => completed({ timeframe: '4H', days: 3 + i })),
  ...many(100, i => completed({ timeframe: 'Daily', days: 10 + (i % 5) })),
];
const fallback = analytics.expectedMoveAnalytics(fallbackEntries, further.setupCriteria(signalA));
const fallbackReady = analytics.expectedMoveReleaseReadiness(fallback, analytics.suggestedExpirationAnalytics(fallback));
assert.strictEqual(fallback.fallback_used, true);
assert.strictEqual(fallbackReady.card_ready, false);

// A+ and B+, Long and Short remain separate through the criteria.
const aLong = setup({ ticker: 'BMY', direction: 'LONG', setupGrade: 'A+ READY', best_contract: { type: 'CALL', strike: 50, expiry: '2026-08-21' } });
assert.notDeepStrictEqual(further.setupCriteria(signalA), further.setupCriteria(aLong));

// Missing option data fails safely and creates grounded risk/help/hurt statements only.
const missingContract = setup({ best_contract: {}, option: {}, suggested_contract: {} });
const model = further.buildAnalysisModel(missingContract, learningReadiness, learning.stats);
assert.ok(model.contract.type);
assert.ok(model.risks.some(line => line.includes('No complete option contract')));
assert.ok(model.helps_hurts.hurts.some(line => line.includes('Option contract details')));
assert.ok(model.thesis.every(line => !/guarantee|probability|buy now|sell now/i.test(line)));

// Expiration comparison never claims enough time while Expected Move is learning.
assert.strictEqual(model.expiration_comparison.enoughTime, 'Learning');

console.log('Further Analysis v1 tests passed');
