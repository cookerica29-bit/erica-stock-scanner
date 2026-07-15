const assert = require('assert');
const expectedMove = require('../public/expected_move_analytics.js');
const opportunity = require('../public/opportunity_analytics.js');

function completedSample(overrides = {}) {
  return {
    ticker: 'ATO',
    setup_grade: 'A+ READY',
    direction: 'LONG',
    scanner_timeframe: 'Daily',
    tracking_status: 'completed',
    completion_reason: 'target',
    first_entry_touch_at: '2026-01-02T14:30:00Z',
    first_target_touch_at: '2026-01-09T20:00:00Z',
    trading_days_to_target: 5,
    trading_days_to_entry: 1,
    entry_price: 100,
    target_price: 110,
    target_distance_atr: 2.5,
    maximum_favorable_excursion_r: 2,
    maximum_adverse_excursion_r: 0.5,
    ...overrides,
  };
}

function history(count = 100, overrides = {}) {
  return Array.from({ length: count }, (_, index) => completedSample({
    trading_days_to_target: 5 + (index % 4),
    ...overrides,
  }));
}

const readyHistory = history(120);

const longEarly = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setupGrade: 'A+ READY',
  direction: 'LONG',
  timeframe: 'Daily',
  price: 104,
  entry: 100,
  tp1: 110,
  first_entry_touch_at: '2026-07-13T14:30:00Z',
}, readyHistory, { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(longEarly.status, 'ON_TIME');
assert.strictEqual(longEarly.priceProgressPct, 40);
assert.strictEqual(longEarly.opportunityRemainingPct, 60);
assert.strictEqual(longEarly.confidence, 'MODERATE');

const shortMid = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setupGrade: 'A+ READY',
  direction: 'SHORT',
  timeframe: 'Daily',
  price: 94,
  entry: 100,
  tp1: 90,
  first_entry_touch_at: '2026-07-13T14:30:00Z',
}, history(120, { direction: 'SHORT' }), { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(shortMid.priceProgressPct, 60);
assert.strictEqual(shortMid.opportunityRemainingPct, 40);
assert.strictEqual(shortMid.status, 'MID_MOVE');

const untriggered = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setupGrade: 'A+ READY',
  direction: 'LONG',
  timeframe: 'Daily',
  price: 106,
  entry: 100,
  tp1: 110,
}, readyHistory, { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(untriggered.entryTriggered, false);
assert.strictEqual(untriggered.priceProgressPct, null);
assert.strictEqual(untriggered.moveCompletedPct, 0);
assert.strictEqual(untriggered.opportunityRemainingPct, 100);
assert.ok(untriggered.explanation.includes('Entry has not been reached'));

const retracement = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setupGrade: 'A+ READY',
  direction: 'LONG',
  timeframe: 'Daily',
  price: 99,
  entry: 100,
  tp1: 110,
  first_entry_touch_at: '2026-07-13T14:30:00Z',
}, readyHistory, { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(retracement.rawPriceProgressPct, -10);
assert.strictEqual(retracement.priceProgressPct, 0);
assert.strictEqual(retracement.opportunityRemainingPct, 100);

const completed = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setupGrade: 'A+ READY',
  direction: 'LONG',
  timeframe: 'Daily',
  price: 110,
  entry: 100,
  tp1: 110,
  first_entry_touch_at: '2026-07-13T14:30:00Z',
  first_target_touch_at: '2026-07-15T14:30:00Z',
}, readyHistory, { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(completed.status, 'COMPLETED');
assert.strictEqual(completed.opportunityRemainingPct, 0);

const invalidated = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setupGrade: 'A+ READY',
  direction: 'LONG',
  timeframe: 'Daily',
  price: 97,
  entry: 100,
  tp1: 110,
  completion_reason: 'stop',
}, readyHistory, { expectedMoveModule: expectedMove });
assert.strictEqual(invalidated.status, 'INVALIDATED');
assert.strictEqual(invalidated.opportunityRemainingPct, null);

const insufficient = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setupGrade: 'A+ READY',
  direction: 'LONG',
  timeframe: 'Daily',
  price: 104,
  entry: 100,
  tp1: 110,
  first_entry_touch_at: '2026-07-13T14:30:00Z',
}, history(12), { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(insufficient.status, 'INSUFFICIENT_DATA');
assert.strictEqual(insufficient.confidence, 'INSUFFICIENT');
assert.strictEqual(insufficient.opportunityRemainingPct, null);

assert.strictEqual(opportunity.countWeekdaysBetween('2026-07-10T14:00:00Z', '2026-07-15T14:00:00Z'), 3);

const position = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  setup_grade: 'A+ READY',
  direction: 'LONG',
  scanner_timeframe: 'Daily',
  current_price: 106,
  entry_price: 100,
  target_price: 110,
  first_entry_touch_at: '2026-07-13T14:30:00Z',
  original_opportunity_remaining_pct: 100,
}, readyHistory, { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(position.priceProgressPct, 60);
assert.strictEqual(position.opportunityRemainingPct, 40);

const goodContract = opportunity.contractHealth({ expiration_date: '2026-09-01' }, longEarly, { now: '2026-07-15T14:00:00Z' });
assert.strictEqual(goodContract.status, 'GOOD_MATCH');

const tooShort = opportunity.contractHealth({ days_to_expiration_at_entry: 2 }, longEarly, { now: '2026-07-15T14:00:00Z' });
assert.strictEqual(tooShort.status, 'EXPIRING_SOON');

const lateContract = opportunity.contractHealth({ days_to_expiration_at_entry: 30 }, {
  ...longEarly,
  status: 'LATE',
  opportunityRemainingPct: 20,
}, { now: '2026-07-15T14:00:00Z' });
assert.strictEqual(lateContract.status, 'LATE_MOVE');

const diagnostics = opportunity.diagnostics([{ ticker: 'ATO', setupGrade: 'A+ READY', direction: 'LONG', timeframe: 'Daily', price: 104, entry: 100, tp1: 110, first_entry_touch_at: '2026-07-13T14:30:00Z' }], readyHistory, { now: '2026-07-15T14:30:00Z', expectedMoveModule: expectedMove });
assert.strictEqual(diagnostics.length, 1);
assert.strictEqual(diagnostics[0].ticker, 'ATO');
assert.strictEqual(diagnostics[0].lifecycleState, 'ON_TIME');

console.log('Opportunity analytics v1 tests passed');
