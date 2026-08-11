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
  confirm: () => true,
  fetch: () => Promise.reject(new Error('network disabled in position handoff test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  sessionStorage: {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
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

function setup(overrides = {}) {
  return {
    ticker: 'DOW',
    direction: 'SHORT',
    timeframe: '4H',
    scan_generation: 'scan-1',
    setup_generation: 'setup-1',
    signal_timestamp: '2026-08-11T14:00:00Z',
    entry: 31.2,
    sl: 32.15,
    tp1: 29.5,
    tp2: 28.7,
    tp3: 27.9,
    price: 31.1,
    setupGrade: 'A',
    entryStatus: 'Tradeable',
    confirmationStarted: true,
    confirmationReason: 'BOS confirmed',
    trade_eval: {
      trade_stage: 'A+ READY',
      trigger_confirmed: true,
      a_plus_ready: true,
      bos: true,
      liquidity_sweep: true,
      rejection: true,
      displacement_classification: 'STRONG',
    },
    ...overrides,
  };
}

function entryFromHandoff(source, overrides = {}) {
  const ev = source.trade_eval || {};
  const handoff = context.positionHandoffSnapshot(source, ev, '2026-08-11T14:05:00Z');
  return context.ensureJournalStableIds({
    id: 'journal-1',
    result: 'Open',
    tracking_status: 'active',
    setup_id: handoff.source_setup_id,
    source_setup_id: handoff.source_setup_id,
    ticker: handoff.ticker,
    direction: handoff.direction,
    scanner_timeframe: handoff.timeframe,
    planned_underlying_entry: handoff.original_planned_entry,
    actual_underlying_entry: handoff.actual_entry,
    original_stop: handoff.original_stop,
    original_tp1: handoff.original_tp1,
    original_tp2: handoff.original_tp2,
    original_tp3: handoff.original_tp3,
    setup_grade: handoff.setup_grade,
    position_handoff_version: 'position-handoff-v1',
    position_handoff_status: 'POSITION_TAKEN',
    position_taken_at: handoff.position_taken_at,
    position_handoff: handoff,
    current_price: 30.8,
    ...overrides,
  });
}

const enterNow = setup();
const originalSetupId = context.setupIdFromSetup(enterNow);
const cardHtml = context.renderCard(enterNow);
assert.ok(cardHtml.includes('Position Taken'));

const handoff = context.positionHandoffSnapshot(enterNow, enterNow.trade_eval, '2026-08-11T14:05:00Z');
assert.strictEqual(handoff.handoff_version, 'position-handoff-v1');
assert.strictEqual(handoff.source_setup_id, originalSetupId);
assert.strictEqual(handoff.ticker, 'DOW');
assert.strictEqual(handoff.direction, 'SHORT');
assert.strictEqual(handoff.original_planned_entry, 31.2);
assert.strictEqual(handoff.original_stop, 32.15);
assert.strictEqual(handoff.original_tp1, 29.5);
assert.strictEqual(handoff.confirmation.trigger_confirmed, true);
assert.strictEqual(handoff.position_taken_at, '2026-08-11T14:05:00Z');

const journalEntry = entryFromHandoff(enterNow);
assert.strictEqual(journalEntry.position_handoff.resulting_journal_id, 'journal-1');
assert.strictEqual(journalEntry.position_handoff.resulting_position_id, 'journal-1');

let model = context.buildPositionModels([journalEntry], new Map())[0];
assert.ok(model);
assert.strictEqual(model.latestSetup, null);
assert.strictEqual(context.currentScannerViewLabel(model), 'No current new-entry signal');
assert.strictEqual(context.positionThesisStatus(model), 'THESIS INTACT');
assert.notStrictEqual(model.position_intelligence.state, 'EXIT');

const waitingSameSetup = setup({ entryStatus: 'Waiting', trade_eval: { trade_stage: 'BUILDING / WATCHLIST', trigger_confirmed: false } });
let scannerMap = new Map([[originalSetupId, waitingSameSetup], ['DOW', waitingSameSetup]]);
model = context.buildPositionModels([journalEntry], scannerMap)[0];
assert.strictEqual(model.scannerContextMatch, 'setup_identity');
assert.strictEqual(model.plan.entryPrice, 31.2);
assert.strictEqual(model.plan.stopPrice, 32.15);
assert.strictEqual(model.position_intelligence.state, 'HEALTHY');

const skippedSameSetup = setup({ setupGrade: 'C', entryStatus: 'Too Far', trade_eval: { trade_stage: 'RANGE / NO TRADE', trigger_confirmed: false } });
scannerMap = new Map([[originalSetupId, skippedSameSetup], ['DOW', skippedSameSetup]]);
model = context.buildPositionModels([journalEntry], scannerMap)[0];
assert.notStrictEqual(model.position_intelligence.state, 'EXIT');
assert.strictEqual(model.plan.tp1, 29.5);

const laterSameTicker = setup({
  direction: 'LONG',
  timeframe: '1D',
  setup_generation: 'setup-2',
  entry: 35,
  sl: 33,
  tp1: 38,
  price: 31.8,
});
scannerMap = new Map([['DOW', laterSameTicker]]);
model = context.buildPositionModels([journalEntry], scannerMap)[0];
assert.strictEqual(model.scannerContextMatch, 'ticker_context_only');
assert.strictEqual(model.entry.direction, 'SHORT');
assert.strictEqual(model.plan.entryPrice, 31.2);
assert.strictEqual(model.plan.stopPrice, 32.15);
assert.strictEqual(model.plan.tp1, 29.5);
assert.notStrictEqual(model.position_intelligence.state, 'EXIT');
assert.ok(context.currentNewEntrySignalExplanation(model).includes('Same ticker context only'));

const stopped = context.buildPositionModels([entryFromHandoff(enterNow, {
  first_entry_touch_at: '2026-08-11T14:10:00Z',
  first_stop_touch_at: '2026-08-11T15:00:00Z',
  current_price: 32.3,
})], new Map())[0];
assert.strictEqual(context.positionThesisStatus(stopped), 'STOP / THESIS INVALIDATED');

const closed = {
  ...journalEntry,
  result: 'Win',
  tracking_status: 'completed',
  completion_reason: 'target',
};
assert.strictEqual(context.buildPositionModels([closed], new Map()).length, 0);
assert.strictEqual(context.positionThesisStatus({ entry: closed, live: {}, position_intelligence: {} }), 'CLOSED');

console.log('position_handoff_v1 passed');
