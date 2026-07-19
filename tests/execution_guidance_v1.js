const assert = require('assert');
const guidance = require('../public/execution_guidance.js');

function setup(overrides = {}) {
  return {
    ticker: 'NKE',
    direction: 'SHORT',
    price: 75,
    entry: 70,
    sl: 72,
    tp1: 65,
    setupGrade: 'B',
    entryStatus: 'Near Entry',
    trade_eval: { trade_stage: 'BUILDING / WATCHLIST' },
    ...overrides,
  };
}

// Label data for cards: Current Price and Planned Entry remain separate.
const rows = guidance.executionPlanRows(setup());
assert.deepStrictEqual(rows.map(row => row.label), ['Current Price', 'Planned Entry', 'Stop', 'Target']);
assert.strictEqual(rows.find(row => row.label === 'Planned Entry').key, 'planned-entry');
assert.strictEqual(rows.find(row => row.label === 'Current Price').value, 75);
assert.strictEqual(rows.find(row => row.label === 'Planned Entry').value, 70);
assert.notStrictEqual(guidance.currentPrice(setup()), guidance.plannedEntry(setup()));

const multiTargetRows = guidance.executionPlanRows(setup({ tp1: 65, tp3: 60 }));
assert.deepStrictEqual(multiTargetRows.map(row => row.label), ['Current Price', 'Planned Entry', 'Stop', 'TP1', 'TP3']);
assert.strictEqual(multiTargetRows.find(row => row.label === 'TP1').value, 65);
assert.strictEqual(multiTargetRows.find(row => row.label === 'TP3').value, 60);

// Almost Ready is not setup-confirmed; it waits for confirmation, not entry fill.
const almostReady = guidance.nextStep(setup(), { bucket: 'ALMOST_READY' }, 'pending');
assert.strictEqual(almostReady.label, 'Next Step');
assert.deepStrictEqual(almostReady.lines, ['Setup is still developing. Wait for full confirmation.']);
assert.strictEqual(guidance.executionState(setup(), { bucket: 'ALMOST_READY' }), 'SETUP_NOT_CONFIRMED');
assert.deepStrictEqual(guidance.cardStatus(setup(), { bucket: 'ALMOST_READY' }), { label: 'ALMOST READY', className: 'almost-ready' });
const almostStages = guidance.readinessStages(setup(), { bucket: 'ALMOST_READY' });
assert.ok(almostStages.some(stage => stage.label === 'Confirm' && stage.status === 'Waiting'));
assert.ok(almostStages.some(stage => stage.label === 'Execute' && stage.status === 'Not Ready'));

const earlyEntrySetup = setup({
  setupGrade: 'B',
  entryStatus: 'Tradeable',
  trade_eval: {
    trade_stage: 'B+ TRADEABLE',
    b_plus_tradeable: true,
    trigger_confirmed: false,
    a_plus_ready: false,
  },
});
assert.strictEqual(guidance.isConfirmedSetup(earlyEntrySetup, { bucket: 'EARLY_ENTRY' }), false);
assert.strictEqual(guidance.isEarlyEntrySetup(earlyEntrySetup, { bucket: 'EARLY_ENTRY' }), true);
assert.strictEqual(guidance.executionState(earlyEntrySetup, { bucket: 'EARLY_ENTRY' }), 'SETUP_EARLY_ENTRY');
assert.deepStrictEqual(guidance.cardStatus(earlyEntrySetup, { bucket: 'EARLY_ENTRY' }), { label: 'EARLY ENTRY', className: 'early-entry' });
const earlyStages = guidance.readinessStages(earlyEntrySetup, { bucket: 'EARLY_ENTRY' });
assert.deepStrictEqual(earlyStages.map(stage => `${stage.label}:${stage.status}`), ['Trend:Complete', 'Zone:Complete', 'Confirm:Early', 'Execute:Caution']);
assert.deepStrictEqual(guidance.nextStep(earlyEntrySetup, { bucket: 'EARLY_ENTRY' }, 'available').lines, ["Structure has broken, but full confirmation hasn't happened yet.", 'Consider smaller size given lower confirmation.']);

// Canonical readiness bucket wins over stale/raw trade_eval readiness fields.
const staleConfirmedRawFields = setup({
  setupGrade: 'A',
  entryStatus: 'Tradeable',
  trade_eval: {
    trade_stage: 'A+ READY',
    trigger_confirmed: true,
    a_plus_ready: true,
    b_plus_tradeable: true,
  },
});
assert.strictEqual(guidance.isConfirmedSetup(staleConfirmedRawFields, { bucket: 'SKIP' }), false);
assert.strictEqual(guidance.isEarlyEntrySetup(staleConfirmedRawFields, { bucket: 'SKIP' }), false);
assert.strictEqual(guidance.executionState(staleConfirmedRawFields, { bucket: 'SKIP' }), 'SETUP_NOT_CONFIRMED');
assert.deepStrictEqual(guidance.cardStatus(staleConfirmedRawFields, { bucket: 'SKIP' }), { label: 'NO TRADE', className: 'skip' });
assert.deepStrictEqual(guidance.nextStep(staleConfirmedRawFields, { bucket: 'SKIP' }, 'available').lines, ['No entry. The setup is not currently valid.']);

// Enter Now can remain confirmed while current price differs from Planned Entry.
const confirmedWaiting = setup({ price: 75, entry: 70, entryStatus: 'Near Entry', distanceFromEntryAtr: 0.5 });
assert.strictEqual(guidance.executionState(confirmedWaiting, { bucket: 'ENTER_NOW' }), 'SETUP_CONFIRMED_WAITING_FOR_ENTRY');
assert.deepStrictEqual(guidance.cardStatus(confirmedWaiting, { bucket: 'ENTER_NOW' }), { label: 'ENTER NOW', className: 'enter-now' });
const waitingStages = guidance.readinessStages(confirmedWaiting, { bucket: 'ENTER_NOW' });
assert.deepStrictEqual(waitingStages.map(stage => `${stage.label}:${stage.status}`), ['Trend:Complete', 'Zone:Complete', 'Confirm:Complete', 'Execute:Waiting']);
assert.ok(waitingStages.find(stage => stage.label === 'Execute').state.includes('execute-waiting-entry'));
const enterWaiting = guidance.nextStep(confirmedWaiting, { bucket: 'ENTER_NOW' }, 'available');
assert.deepStrictEqual(enterWaiting.lines, [
  'Setup confirmed. Wait for price to reach the planned entry at $70.00.',
  'Set an alert at $70.00.',
]);

// Contract guidance never overrides waiting for Planned Entry.
assert.deepStrictEqual(guidance.nextStep(confirmedWaiting, { bucket: 'ENTER_NOW' }, 'potential').lines, enterWaiting.lines);

// Entry-reached Enter Now uses execute wording only when a validated contract exists.
const reached = setup({ price: 70.02, entry: 70, entryStatus: 'Tradeable', distanceFromEntryAtr: 0.1 });
assert.strictEqual(guidance.executionState(reached, { bucket: 'ENTER_NOW' }), 'SETUP_CONFIRMED_ENTRY_REACHED');
const reachedStages = guidance.readinessStages(reached, { bucket: 'ENTER_NOW' });
assert.deepStrictEqual(reachedStages.map(stage => `${stage.label}:${stage.status}`), ['Trend:Complete', 'Zone:Complete', 'Confirm:Complete', 'Execute:Ready']);
assert.ok(reachedStages.find(stage => stage.label === 'Execute').state.includes('execute-entry-ready'));
const enterWithContract = guidance.nextStep(reached, { bucket: 'ENTER_NOW' }, 'available');
assert.deepStrictEqual(enterWithContract.lines, ['Price is at the planned entry. You can execute this trade.']);

const enterWithoutContract = guidance.nextStep(reached, { bucket: 'ENTER_NOW' }, 'potential');
assert.deepStrictEqual(enterWithoutContract.lines, ['Price is at the planned entry.', 'Verify and select the live option contract before executing.']);

// Tradeable but not Enter Now keeps the wording grounded in existing production meaning.
const tradeable = guidance.nextStep(setup({ entryStatus: 'Tradeable' }), { bucket: 'ALMOST_READY' }, 'available');
assert.deepStrictEqual(tradeable.lines, ['Monitor price near the planned entry.']);

// Existing too-far/stale fields produce do-not-chase wording without changing scanner status.
const passed = guidance.nextStep(setup({ entryStatus: 'Too Far' }), { bucket: 'ENTER_NOW' }, 'available');
assert.deepStrictEqual(passed.lines, ['Do not chase. Price moved beyond the planned entry.']);

// BAM-style / mixed-case Enter Now variants normalize into confirmed setup handling.
const bam = guidance.nextStep(setup({ status: 'Enter Now', price: 75, entry: 70 }), {}, 'available');
assert.ok(bam.lines[0].startsWith('Setup confirmed.'));
assert.ok(!bam.lines.join(' ').includes('still developing'));
const aReady = guidance.nextStep(setup({ trade_eval: { trade_stage: 'A+ READY' }, price: 75, entry: 70 }), {}, 'available');
assert.ok(aReady.lines[0].startsWith('Setup confirmed.'));

// Watchlist/building setups remain monitoring-only.
const building = guidance.nextStep(setup({ entryStatus: 'Waiting', trade_eval: { trade_stage: 'BUILDING / WATCHLIST' } }), { bucket: 'WAITING' }, 'pending');
assert.deepStrictEqual(building.lines, ['Continue monitoring. The setup is still developing.']);
assert.deepStrictEqual(guidance.cardStatus(setup({ entryStatus: 'Waiting' }), { bucket: 'WAITING' }), { label: 'BUILDING', className: 'wait' });

// Invalidated/no-trade setups explicitly say no entry.
const invalid = guidance.nextStep(setup({ trade_eval: { trade_stage: 'RANGE / NO TRADE' } }), { bucket: 'SKIP' }, 'pending');
assert.deepStrictEqual(invalid.lines, ['No entry. The setup is not currently valid.']);
assert.deepStrictEqual(guidance.cardStatus(setup({ trade_eval: { trade_stage: 'RANGE / NO TRADE' } }), { bucket: 'SKIP' }), { label: 'NO TRADE', className: 'skip' });
const skipStages = guidance.readinessStages(setup({ trade_eval: { trade_stage: 'RANGE / NO TRADE' } }), { bucket: 'SKIP' });
assert.deepStrictEqual(skipStages.map(stage => `${stage.label}:${stage.status}`), ['Trend:No Trade', 'Zone:Muted', 'Confirm:Muted', 'Execute:Not Ready']);
assert.ok(skipStages.every(stage => !stage.state.includes('current')), 'No-trade timeline should not use current-step styling');

// Nike-style unfilled planned entry remains unfilled: price and planned entry are not forced together.
const nike = setup({ ticker: 'NKE', price: 77.25, entry: 70.5, direction: 'SHORT' });
assert.strictEqual(guidance.currentPrice(nike), 77.25);
assert.strictEqual(guidance.plannedEntry(nike), 70.5);
assert.strictEqual(guidance.executionState(nike, { bucket: 'ENTER_NOW' }), 'SETUP_CONFIRMED_WAITING_FOR_ENTRY');
assert.deepStrictEqual(
  guidance.nextStep(setup({ ticker: 'NKE', price: 82, entry: 70.5, direction: 'SHORT', entryStatus: 'Too Far' }), { bucket: 'ENTER_NOW' }, 'available').lines,
  ['Do not chase. Price moved beyond the planned entry.']
);

// Helpers do not mutate scanner status or eligibility fields.
const original = setup({ progress_bucket: 'ALMOST_READY', setupGrade: 'B+ TRADEABLE' });
const before = JSON.stringify(original);
guidance.nextStep(original, { bucket: 'ALMOST_READY' }, 'pending');
guidance.executionPlanRows(original);
assert.strictEqual(JSON.stringify(original), before);

console.log('Execution guidance v1 tests passed');
