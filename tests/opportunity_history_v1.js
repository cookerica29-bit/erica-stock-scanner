const assert = require('assert');
const history = require('../public/opportunity_history.js');
const opportunity = require('../public/opportunity_analytics.js');

function journalRecord(overrides = {}) {
  return {
    setup_id: 'setup-1',
    ticker: 'ATO',
    direction: 'LONG',
    setup_grade: 'A+ READY',
    scanner_timeframe: 'Daily',
    entryType: 'Pullback',
    signal_timestamp: '2026-01-02T14:30:00Z',
    first_entry_touch_at: '2026-01-03T14:30:00Z',
    first_target_touch_at: '2026-01-10T20:00:00Z',
    tracking_completed_at: '2026-01-10T20:00:00Z',
    completion_reason: 'target',
    entry_price: 100,
    stop_price: 96,
    target_price: 110,
    trading_days_to_entry: 1,
    trading_days_to_target: 5,
    final_trading_days: 5,
    maximum_favorable_excursion: 11,
    maximum_adverse_excursion: 2,
    ...overrides,
  };
}

function qualifiedRecords(count, overrides = {}) {
  return Array.from({ length: count }, (_, index) => history.normalizeJournalEntry(journalRecord({
    setup_id: `setup-${overrides.ticker || 'ATO'}-${index}`,
    ticker: overrides.ticker || 'ATO',
    direction: overrides.direction || 'LONG',
    setup_grade: overrides.setup_grade || 'A+ READY',
    entryType: overrides.entryType || 'Pullback',
    signal_timestamp: `2026-01-${String((index % 20) + 1).padStart(2, '0')}T14:30:00Z`,
    first_entry_touch_at: `2026-02-${String((index % 20) + 1).padStart(2, '0')}T14:30:00Z`,
    first_target_touch_at: `2026-02-${String((index % 20) + 6).padStart(2, '0')}T20:00:00Z`,
    trading_days_to_target: 5 + (index % 4),
    final_trading_days: 5 + (index % 4),
    maximum_favorable_excursion: 10 + (index % 3),
    maximum_adverse_excursion: 1 + (index % 2),
    ...overrides,
  })));
}

// Journal import normalization: qualified, partial, and excluded are separated.
const qualified = history.normalizeJournalEntry(journalRecord());
assert.strictEqual(qualified.dataQuality, 'QUALIFIED');
assert.strictEqual(qualified.direction, 'LONG');
assert.strictEqual(qualified.tp1Hit, true);
assert.strictEqual(qualified.tradingDaysToTp1, 5);

const partial = history.normalizeJournalEntry(journalRecord({ first_entry_touch_at: null, first_target_touch_at: null, tracking_completed_at: null, completion_reason: '' }));
assert.strictEqual(partial.dataQuality, 'PARTIAL');
assert.ok(partial.exclusionReasons.includes('MISSING_ENTRY_TRIGGER'));

const excluded = history.normalizeJournalEntry(journalRecord({ direction: '', entry_price: null }));
assert.strictEqual(excluded.dataQuality, 'EXCLUDED');
assert.ok(excluded.exclusionReasons.includes('MISSING_DIRECTION'));

const preview = history.previewJournalImport([journalRecord(), partial, excluded]);
assert.strictEqual(preview.total, 3);
assert.strictEqual(preview.qualified, 1);
assert.strictEqual(preview.partial, 1);
assert.strictEqual(preview.excluded, 1);

// Outcome ordering: target before stop and stop before target are deterministic.
const targetFirst = history.normalizeJournalEntry(journalRecord({
  first_target_touch_at: '2026-01-08T20:00:00Z',
  first_stop_touch_at: '2026-01-09T15:00:00Z',
}));
assert.strictEqual(targetFirst.tp1Hit, true);
assert.strictEqual(targetFirst.stopped, false);

const stopFirst = history.normalizeJournalEntry(journalRecord({
  first_target_touch_at: '2026-01-10T20:00:00Z',
  first_stop_touch_at: '2026-01-08T15:00:00Z',
}));
assert.strictEqual(stopFirst.tp1Hit, false);
assert.strictEqual(stopFirst.stopped, true);

// Duplicate prevention is idempotent and richer records replace poorer records.
const poorer = { ...partial, recordId: 'same-record' };
const richer = { ...qualified, recordId: 'same-record' };
let merged = history.mergeOpportunityRecords([], [poorer]);
assert.strictEqual(merged.records.length, 1);
assert.strictEqual(merged.records[0].dataQuality, 'PARTIAL');
merged = history.mergeOpportunityRecords(merged.records, [richer]);
assert.strictEqual(merged.records.length, 1);
assert.strictEqual(merged.records[0].dataQuality, 'QUALIFIED');
const again = history.mergeOpportunityRecords(merged.records, [richer]);
assert.strictEqual(again.records.length, 1);
assert.strictEqual(again.duplicates, 1);

// Cohorts enforce the threshold for exact ticker evidence and select fallback transparently.
const exactSamples = qualifiedRecords(30, { ticker: 'ATO', direction: 'LONG' });
const summaries = history.buildCohortSummaries(exactSamples, { now: '2026-07-15T12:00:00Z' });
const selectedExact = history.selectCohortForSetup({ ticker: 'ATO', direction: 'LONG', setup_grade: 'A+ READY', entryType: 'Pullback' }, summaries);
assert.strictEqual(selectedExact.summary.readiness, 'QUALIFIED');
assert.strictEqual(selectedExact.exactGroupUsed, true);
assert.strictEqual(selectedExact.fallbackUsed, false);

const broadOnly = [
  ...qualifiedRecords(2, { ticker: 'ATO', direction: 'LONG' }),
  ...qualifiedRecords(30, { ticker: 'BMY', direction: 'LONG' }),
];
const broadSummaries = history.buildCohortSummaries(broadOnly, { now: '2026-07-15T12:00:00Z' });
const selectedFallback = history.selectCohortForSetup({ ticker: 'ATO', direction: 'LONG', setup_grade: 'A+ READY', entryType: 'Pullback' }, broadSummaries);
assert.strictEqual(selectedFallback.exactGroupUsed, false);
assert.strictEqual(selectedFallback.fallbackUsed, true);
assert.notStrictEqual(selectedFallback.summary.cohortLevel, 'ticker_direction_family_grade');

// Opportunity analytics can consume durable cohort summaries without browser-local journal samples.
const shadow = opportunity.opportunityAnalytics({
  ticker: 'ATO',
  direction: 'LONG',
  setup_grade: 'A+ READY',
  entryType: 'Pullback',
  price: 104,
  entry: 100,
  tp1: 110,
  first_entry_touch_at: '2026-07-13T14:30:00Z',
}, [], { now: '2026-07-15T14:30:00Z', cohortSummaries: summaries, opportunityHistoryModule: history });
assert.strictEqual(shadow.status, 'ON_TIME');
assert.strictEqual(shadow.sampleSize, 30);
assert.strictEqual(shadow.confidence, 'LOW');

// Lifecycle replay and stability diagnostics are deterministic.
const replay = history.lifecycleReplay(qualified);
assert.ok(replay.length >= 5);
assert.strictEqual(replay[0].opportunityRemainingPct, 91);
assert.strictEqual(replay[replay.length - 1].opportunityRemainingPct, 0);
const stable = history.stabilityDiagnostics([
  { opportunityRemainingPct: 91 },
  { opportunityRemainingPct: 73 },
  { opportunityRemainingPct: 68 },
  { opportunityRemainingPct: 47 },
  { opportunityRemainingPct: 22 },
]);
assert.strictEqual(stable.unstable, false);

const gap = history.stabilityDiagnostics([
  { opportunityRemainingPct: 90 },
  { opportunityRemainingPct: 40 },
  { opportunityRemainingPct: 12 },
]);
assert.strictEqual(gap.unstable, true);
assert.ok(gap.warnings.includes('LARGE_SINGLE_STEP_CHANGE'));

// Short records normalize and remain usable for historical reconstruction.
const shortRecord = history.normalizeJournalEntry(journalRecord({
  setup_id: 'short-1',
  direction: 'SHORT',
  entry_price: 100,
  stop_price: 104,
  target_price: 90,
  maximum_favorable_excursion: 12,
  maximum_adverse_excursion: 1.5,
}));
assert.strictEqual(shortRecord.direction, 'SHORT');
assert.strictEqual(shortRecord.mfe, 12);
assert.strictEqual(shortRecord.mae, 1.5);

const report = history.sourceFieldReport();
assert.ok(report.some(row => row.field === 'Replay dataset' && row.status === 'Missing'));

console.log('Opportunity history v1 tests passed');
