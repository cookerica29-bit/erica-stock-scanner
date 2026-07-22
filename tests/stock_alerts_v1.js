const assert = require('assert');
const alerts = require('../public/stock_alerts.js');

function setup(overrides = {}) {
  return {
    ticker: 'BKR',
    direction: 'SHORT',
    timeframe: '1D',
    signal_timestamp: '2026-07-14T16:00:00Z',
    price: 57.66,
    entry: 57.5,
    sl: 58.33,
    tp1: 56.6,
    tp2: 55.75,
    tp3: 54.9,
    rr: 2,
    option_plan: {
      available: true,
      type: 'PUT',
      preferred_strike: 57,
      suggested_expiration: { label: '21–35 DTE' },
      expected_hold: { label: '7–12 Trading Days' },
    },
    ...overrides,
  };
}

const base = setup();
const id = alerts.setupIdentity(base);
const previous = { [id]: 'ALMOST_READY' };
const sent = new Set();
const readinessResolver = s => ({ bucket: s.bucket || 'ENTER_NOW' });
const contractStateResolver = () => ({ state: 'validated_live', label: 'Aug 21 · $57 Put' });

const transition = alerts.detectSetupEvents([base], previous, sent, {
  now: '2026-07-14T17:00:00Z',
  readinessResolver,
  contractStateResolver,
  executionStateResolver: () => 'SETUP_CONFIRMED_WAITING_FOR_ENTRY',
});
assert.strictEqual(transition.events.length, 1);
assert.strictEqual(transition.events[0].type, 'ENTER_NOW');
assert.ok(transition.events[0].message.includes('KAIROS SETUP CONFIRMED'));
assert.ok(transition.events[0].message.includes('Wait for price to reach the Planned Entry.'));
assert.ok(transition.events[0].message.includes('Option Plan:'));
assert.ok(transition.events[0].message.includes('$57 Put'));
assert.ok(transition.events[0].message.includes('Suggested Expiration: 21–35 DTE'));

sent.add(transition.events[0].key);
const duplicate = alerts.detectSetupEvents([base], previous, sent, {
  readinessResolver,
  contractStateResolver,
  executionStateResolver: () => 'SETUP_CONFIRMED_WAITING_FOR_ENTRY',
});
assert.strictEqual(duplicate.events.length, 0);

const sameEnterNow = alerts.detectSetupEvents([base], { [id]: 'ENTER_NOW' }, new Set(), {
  readinessResolver,
});
assert.strictEqual(sameEnterNow.events.length, 0);

const laterSignal = setup({ signal_timestamp: '2026-07-15T16:00:00Z' });
const laterId = alerts.setupIdentity(laterSignal);
const later = alerts.detectSetupEvents([laterSignal], { [laterId]: 'BUILDING' }, new Set(), {
  readinessResolver,
  contractStateResolver,
  executionStateResolver: () => 'SETUP_CONFIRMED_WAITING_FOR_ENTRY',
});
assert.strictEqual(later.events.length, 1);
assert.notStrictEqual(later.events[0].key, transition.events[0].key);

const potential = alerts.enterNowMessage(setup({ option_plan: null }), {
  state: 'potential',
  potential: { strike: 57, type: 'PUT' },
  expiration: { label: 'Learning' },
}, false);
assert.ok(potential.includes('Option Plan:'));
assert.ok(potential.includes('$57 Put'));
assert.ok(potential.includes('Expiration Guidance: Learning'));
assert.ok(!potential.includes('You can execute this trade'));

const entryReached = alerts.detectSetupEvents([base], { [id]: 'ENTER_NOW' }, new Set(), {
  now: '2026-07-14T17:05:00Z',
  readinessResolver,
  contractStateResolver,
  executionStateResolver: () => 'SETUP_CONFIRMED_ENTRY_REACHED',
  entryReachedTimestampResolver: () => '2026-07-14T17:05:00Z',
});
assert.strictEqual(entryReached.events.length, 1);
assert.strictEqual(entryReached.events[0].type, 'ENTRY_REACHED');
assert.ok(entryReached.events[0].message.includes('KAIROS ENTRY REACHED'));

assert.strictEqual(alerts.tp1Touched({ direction: 'LONG', plannedTp1: 105 }, { high: 105, low: 99 }), true);
assert.strictEqual(alerts.tp1Touched({ direction: 'LONG', plannedTp1: 105 }, { high: 104.99, low: 99 }), false);
assert.strictEqual(alerts.tp1Touched({ direction: 'SHORT', plannedTp1: 95 }, { high: 101, low: 95 }), true);
assert.strictEqual(alerts.tp1Touched({ direction: 'SHORT', plannedTp1: 95 }, { high: 101, low: 95.01 }), false);
assert.strictEqual(alerts.tp1Touched({ direction: 'SHORT', plannedTp1: 95 }, { price: 96, high: 101, low: 95 }), true);

const tpMessage = alerts.tp1Message({
  ticker: 'ATO',
  direction: 'SHORT',
  entry: 179.2,
  plannedTp1: 176.15,
  plannedTp2: 174.8,
  plannedTp3: 173.4,
}, { price: 176.08 });
assert.ok(tpMessage.includes('KAIROS TP1 REACHED'));
assert.ok(tpMessage.includes('Consider taking partial profits.'));
assert.ok(tpMessage.includes('Manage the remainder according to your plan.'));
assert.ok(!tpMessage.includes('Sell now'));
assert.ok(!tpMessage.includes('Close the trade'));

console.log('Stock alerts v1 tests passed');
