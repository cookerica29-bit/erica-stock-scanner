const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const cardData = require('../public/card_data.js');

const html = fs.readFileSync('public/index.html', 'utf8');
const inline = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)][0][1];

function elementStub() {
  return {
    innerHTML: '',
    textContent: '',
    value: '',
    checked: false,
    dataset: {},
    style: {},
    selectedOptions: [{ textContent: '' }],
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    appendChild() {},
    remove() {},
  };
}

const storage = {};
const context = {
  console,
  Date,
  Math,
  Number,
  String,
  Boolean,
  Array,
  Object,
  JSON,
  RegExp,
  performance: { now: () => 100 },
  URLSearchParams,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => false,
  fetch: () => Promise.reject(new Error('network disabled in state test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  document: {
    body: { appendChild() {} },
    getElementById: () => elementStub(),
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => elementStub(),
    addEventListener() {},
    removeEventListener() {},
  },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
context.window = {
  KairosExecutionGuidance: executionGuidance,
  KairosCardData: cardData,
  open: () => {},
  addEventListener: () => {},
  removeEventListener: () => {},
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);

function model({ entry = {}, live = {} } = {}) {
  return {
    id: 'position-1',
    entry: {
      id: 'position-1',
      ticker: 'DOW',
      direction: 'SHORT',
      result: 'Open',
      tracking_status: 'active',
      actual_underlying_entry: 300.62,
      ...entry,
    },
    plan: {
      ticker: 'DOW',
      direction: 'SHORT',
      entryPrice: 300.62,
      stopPrice: 303,
      tp1: 296,
      tp2: 294,
      tp3: 292,
      setupGrade: 'A',
    },
    live: {
      currentPrice: 298,
      currentScannerStatus: 'NO TRADE',
      currentGrade: 'C',
      currentEntryStatus: 'Too Far',
      tradeProgressStatus: 'WAITING',
      rNow: 1.1,
      mfeR: null,
      maeR: null,
      entryTouchedAt: null,
      tp1TouchedAt: null,
      stopTouchedAt: null,
      tradingDaysElapsed: 1,
      barsElapsed: 3,
      ...live,
    },
    position_intelligence: { current_r: live.rNow ?? 1.1, state: 'HEALTHY' },
  };
}

const chart = context.renderGuidedTradeChartBlock({
  ticker: 'DOW',
  direction: 'SHORT',
  guided_status: 'Healthy',
  entry: 300.62,
  stop: 303,
  tp1: 296,
}, 'position');
assert.ok(chart.includes('Short · Healthy · Entry $300.62'));
assert.ok(!chart.includes('&lt;span class=&quot;direction-value'));

const waiting = model({
  live: {
    currentPrice: 301,
    currentEntryStatus: 'Waiting for Entry',
    currentScannerStatus: 'Waiting',
    rNow: -0.16,
  },
});
assert.strictEqual(context.positionLifecycleStatus(waiting), 'WAITING FOR ENTRY');
assert.strictEqual(context.positionEntryState(waiting), 'Waiting for entry');
assert.strictEqual(context.positionMfeMaeLabel({ entry: {}, live: {} }), 'Tracking unavailable');

const upOneR = model({ live: { rNow: 1.1, currentScannerStatus: 'NO TRADE', currentEntryStatus: 'Too Far' } });
assert.strictEqual(context.positionLifecycleStatus(upOneR), 'POSITION OPEN');
assert.strictEqual(context.positionEntryState(upOneR), 'Entry reached');
assert.strictEqual(context.currentScannerViewLabel(upOneR), 'NO TRADE · Grade C · Too Far');
assert.ok(context.positionMfeMaeLabel(upOneR).includes('+1.10R'));
assert.ok(!context.positionMfeMaeLabel(upOneR).includes('0.00R / 0.00R'));

const downHalfR = model({ live: { rNow: -0.52, currentScannerStatus: 'NO TRADE', currentEntryStatus: 'Too Far' } });
assert.strictEqual(context.positionLifecycleStatus(downHalfR), 'POSITION OPEN');
assert.strictEqual(context.positionEntryState(downHalfR), 'Entry reached');
assert.ok(context.positionMfeMaeLabel(downHalfR).includes('-0.52R'));

const tp1 = model({ entry: { first_target_touch_at: '2026-07-27T14:00:00Z' }, live: { tp1TouchedAt: '2026-07-27T14:00:00Z' } });
assert.strictEqual(context.positionLifecycleStatus(tp1), 'TP1 REACHED');
assert.ok(context.positionTargetState(tp1).startsWith('Reached'));

const stopped = model({ entry: { first_stop_touch_at: '2026-07-27T14:00:00Z' }, live: { stopTouchedAt: '2026-07-27T14:00:00Z' } });
assert.strictEqual(context.positionLifecycleStatus(stopped), 'STOPPED');
assert.strictEqual(context.positionStopState(stopped).startsWith('Reached'), true);

context.saveJournal([upOneR.entry]);
context.syncPositionIntelligenceHistory([{ ...upOneR, position_intelligence: { current_r: 1.1, state: 'HEALTHY', ticker: 'DOW' } }]);
const persistedUp = context.loadJournal()[0];
assert.strictEqual(persistedUp.position_mfe_r, 1.1);
context.syncPositionIntelligenceHistory([{ ...upOneR, entry: persistedUp, position_intelligence: { current_r: -0.52, state: 'WATCH', ticker: 'DOW' } }]);
const persistedDown = context.loadJournal()[0];
assert.strictEqual(persistedDown.position_mfe_r, 1.1);
assert.strictEqual(persistedDown.position_mae_r, -0.52);

console.log('Position Manager state v1 tests passed');
