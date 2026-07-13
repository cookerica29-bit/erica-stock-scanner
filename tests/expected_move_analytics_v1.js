const assert = require('assert');
const analytics = require('../public/expected_move_analytics.js');

function targetSetup({ grade = 'A+ READY', direction = 'LONG', timeframe = 'Daily', days = 7, mfe = 2, mae = 0.5, atr = 2.7 }) {
  return {
    tracking_status: 'completed',
    completion_reason: 'target',
    setup_grade: grade,
    direction,
    scanner_timeframe: timeframe,
    trading_days_to_target: days,
    first_target_touch_at: '2026-01-10T16:00:00Z',
    entry_price: 100,
    target_price: direction === 'SHORT' ? 90 : 110,
    maximum_favorable_excursion_r: mfe,
    maximum_adverse_excursion_r: mae,
    target_distance_atr: atr,
    bars_to_target: days * 2,
    trading_days_to_entry: 1,
  };
}

function stopSetup({ grade = 'A+ READY', direction = 'LONG', timeframe = 'Daily', days = 3 }) {
  return {
    tracking_status: 'completed',
    completion_reason: 'stop',
    setup_grade: grade,
    direction,
    scanner_timeframe: timeframe,
    trading_days_to_stop: days,
    first_stop_touch_at: '2026-01-08T16:00:00Z',
    entry_price: 100,
    target_price: direction === 'SHORT' ? 90 : 110,
    maximum_favorable_excursion_r: 0.4,
    maximum_adverse_excursion_r: 1,
    target_distance_atr: 2,
    bars_to_stop: days * 2,
  };
}

function many(count, factory) {
  return Array.from({ length: count }, (_, idx) => factory(idx));
}

// Legacy label normalization.
assert.strictEqual(analytics.normalizeGrade({ setup_grade: 'A' }), 'A_PLUS_READY');
assert.strictEqual(analytics.normalizeGrade({ scanner_status: 'READY' }), 'A_PLUS_READY');
assert.strictEqual(analytics.normalizeGrade({ setup_grade: 'B' }), 'B_PLUS_TRADEABLE');
assert.strictEqual(analytics.normalizeGrade({ scanner_status: 'B+ TRADEABLE' }), 'B_PLUS_TRADEABLE');
assert.strictEqual(analytics.normalizeTimeframe('1D'), 'Daily');
assert.strictEqual(analytics.normalizeTimeframe('4h'), '4H');

// A+ and B+ stay separated, as do long/short and timeframe buckets.
const separated = [
  ...many(30, i => targetSetup({ grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily', days: 5 + i })),
  ...many(30, i => targetSetup({ grade: 'B+ TRADEABLE', direction: 'LONG', timeframe: 'Daily', days: 20 + i })),
  ...many(30, i => targetSetup({ grade: 'A+ READY', direction: 'SHORT', timeframe: 'Daily', days: 40 + i })),
  ...many(30, i => targetSetup({ grade: 'A+ READY', direction: 'LONG', timeframe: '4H', days: 60 + i })),
];
const aLongDaily = analytics.expectedMoveAnalytics(separated, { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
const bLongDaily = analytics.expectedMoveAnalytics(separated, { grade: 'B+ TRADEABLE', direction: 'LONG', timeframe: 'Daily' });
const aShortDaily = analytics.expectedMoveAnalytics(separated, { grade: 'A+ READY', direction: 'SHORT', timeframe: 'Daily' });
const aLong4h = analytics.expectedMoveAnalytics(separated, { grade: 'A+ READY', direction: 'LONG', timeframe: '4H' });
assert.notStrictEqual(aLongDaily.median_days_to_target, bLongDaily.median_days_to_target);
assert.notStrictEqual(aLongDaily.median_days_to_target, aShortDaily.median_days_to_target);
assert.notStrictEqual(aLongDaily.median_days_to_target, aLong4h.median_days_to_target);

// Percentile method is deterministic linear interpolation, range rounds outward.
const percentileSet = many(30, i => targetSetup({ days: 1 + i }));
const pctResult = analytics.expectedMoveAnalytics(percentileSet, { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(pctResult.target_sample_count, 30);
assert.strictEqual(pctResult.p25_days_to_target, 8.25);
assert.strictEqual(pctResult.p75_days_to_target, 22.75);
assert.strictEqual(pctResult.expected_move_min_days, 8);
assert.strictEqual(pctResult.expected_move_max_days, 23);

// Outliers do not dominate the IQR range.
const outlierSet = [...many(30, i => targetSetup({ days: 5 + i })), targetSetup({ days: 1000 })];
const outlierResult = analytics.expectedMoveAnalytics(outlierSet, { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.ok(outlierResult.expected_move_max_days < 40);

// Stopped trades do not influence time-to-target distribution.
const withStops = [...many(30, i => targetSetup({ days: 10 + i })), ...many(30, i => stopSetup({ days: 1 + i }))];
const stopResult = analytics.expectedMoveAnalytics(withStops, { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(stopResult.target_sample_count, 30);
assert.ok(stopResult.p25_days_to_target >= 10);

// Missing and corrupted duration records are excluded and counted.
const missing = targetSetup({ days: 7 });
delete missing.trading_days_to_target;
const negative = targetSetup({ days: -1 });
const invalidTarget = targetSetup({ days: 7 });
delete invalidTarget.target_price;
const exclusionSet = [...many(30, i => targetSetup({ days: 7 + i })), missing, negative, invalidTarget];
const exclusionResult = analytics.expectedMoveAnalytics(exclusionSet, { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(exclusionResult.exclusion_count, 3);
assert.strictEqual(exclusionResult.stats.exclusion_reasons.missing_duration, 1);
assert.strictEqual(exclusionResult.stats.exclusion_reasons.negative_duration, 1);
assert.strictEqual(exclusionResult.stats.exclusion_reasons.invalid_target, 1);

// Minimum sample threshold and confidence transitions.
const twentyNine = many(29, i => targetSetup({ days: 4 + i }));
const learning = analytics.expectedMoveAnalytics(twentyNine, { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(learning.status, 'learning');
assert.strictEqual(learning.confidence, 'insufficient');
assert.strictEqual(learning.expected_move_min_days, null);

const low = analytics.expectedMoveAnalytics(many(30, i => targetSetup({ days: 4 + i })), { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(low.status, 'early_estimate');
assert.strictEqual(low.confidence, 'low');

const moderate = analytics.expectedMoveAnalytics(many(100, i => targetSetup({ days: 4 + (i % 20) })), { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(moderate.status, 'ready');
assert.strictEqual(moderate.confidence, 'moderate');

const high = analytics.expectedMoveAnalytics(many(300, i => targetSetup({ days: 4 + (i % 20) })), { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(high.status, 'ready');
assert.strictEqual(high.confidence, 'high');

// Fallback uses grade+direction before broader buckets and never mixes A+/B+.
const fallbackSet = [
  ...many(10, i => targetSetup({ grade: 'A+ READY', direction: 'LONG', timeframe: '4H', days: 2 + i })),
  ...many(35, i => targetSetup({ grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily', days: 8 + i })),
  ...many(35, i => targetSetup({ grade: 'B+ TRADEABLE', direction: 'LONG', timeframe: '4H', days: 80 + i })),
];
const fallback = analytics.expectedMoveAnalytics(fallbackSet, { grade: 'A+ READY', direction: 'LONG', timeframe: '4H' });
assert.strictEqual(fallback.fallback_used, true);
assert.strictEqual(fallback.fallback_level, 'grade_direction');
assert.deepStrictEqual(fallback.group_used, { grade: 'A_PLUS_READY', direction: 'LONG', timeframe: null });
assert.ok(fallback.median_days_to_target < 50);

// Stored weekend-safe durations are consumed as stored, not recalculated from timestamps.
const weekendStored = analytics.expectedMoveAnalytics(many(30, () => targetSetup({ days: 3 })), { grade: 'A+ READY', direction: 'LONG', timeframe: 'Daily' });
assert.strictEqual(weekendStored.median_days_to_target, 3);

// Suggested expiration remains internal and never shorter than expected-move upper bound.
const expiration = analytics.suggestedExpirationAnalytics(low);
assert.notStrictEqual(expiration.status, 'learning');
assert.ok(expiration.min_dte >= low.expected_move_max_days);
const noExpiration = analytics.suggestedExpirationAnalytics(learning);
assert.strictEqual(noExpiration.status, 'learning');

console.log('Expected Move analytics v1 tests passed');
