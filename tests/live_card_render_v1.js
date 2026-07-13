const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const contractGuidance = require('../public/contract_guidance.js');

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
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => false,
  fetch: () => Promise.reject(new Error('network disabled in render test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  document: {
    getElementById: () => elementStub(),
    querySelectorAll: () => [],
    createElement: () => elementStub(),
  },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
context.window = {
  KairosExecutionGuidance: executionGuidance,
  KairosContractGuidance: contractGuidance,
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
    ticker: 'BAM',
    direction: 'SHORT',
    price: 46.34,
    entry: 46.05,
    sl: 47,
    tp1: 44,
    timeframe: '4H',
    entryStatus: 'Near Entry',
    distanceFromEntryAtr: 0.5,
    setupGrade: 'A+ READY',
    stockTrend: 'Bearish',
    stockPhase: 'Pullback',
    stockSetupStatus: 'Pullback Active',
    stockLocation: 'Premium',
    confirmationStarted: true,
    trade_eval: { trade_stage: 'A+ READY', trigger_confirmed: true },
    best_contract: { available: true, type: 'PUT', strike: 45, expiry: '2026-08-21', ask: 1.2, bid: 1.1, spread: 0.1 },
    ...overrides,
  };
}

function htmlFor(fixture, lifecycleContract = undefined) {
  const input = lifecycleContract ? { ...fixture, best_contract: lifecycleContract } : fixture;
  return context.renderCard(input);
}

const enterWaiting = htmlFor(setup());
assert.ok(enterWaiting.includes('data-normalized-status="ENTER_NOW"'));
assert.ok(enterWaiting.includes('data-execution-state="SETUP_CONFIRMED_WAITING_FOR_ENTRY"'));
assert.ok(enterWaiting.includes('data-execute-visual-state="waiting"'));
assert.ok(enterWaiting.includes('execute-waiting-entry'));
assert.ok(enterWaiting.includes('Setup confirmed. Wait for price to reach the planned entry at $46.05.'));
assert.ok(enterWaiting.includes('Set an alert at $46.05.'));
assert.ok(!enterWaiting.includes('Continue monitoring. The setup is still developing.'));

const reachedLive = htmlFor(setup({ price: 46.05, entryStatus: 'Tradeable', distanceFromEntryAtr: 0.1 }));
assert.ok(reachedLive.includes('data-execution-state="SETUP_CONFIRMED_ENTRY_REACHED"'));
assert.ok(reachedLive.includes('data-execute-visual-state="ready"'));
assert.ok(reachedLive.includes('execute-entry-ready'));
assert.ok(reachedLive.includes('Price is at the planned entry. You can execute this trade.'));

const reachedPotential = htmlFor(
  setup({ price: 46.05, entryStatus: 'Tradeable', distanceFromEntryAtr: 0.1 }),
  { available: false, source: 'unavailable', reason: 'no contract passed filters' }
);
assert.ok(reachedPotential.includes('execute-entry-ready'));
assert.ok(reachedPotential.includes('Price is at the planned entry.'));
assert.ok(reachedPotential.includes('Verify and select the live option contract before executing.'));

const almostReady = htmlFor(setup({ trade_eval: { trade_stage: 'BUILDING / WATCHLIST' }, setupGrade: 'B+ TRADEABLE', entryStatus: 'Near Entry', confirmationStarted: false }));
assert.ok(almostReady.includes('data-normalized-status="ALMOST_READY"'));
assert.ok(almostReady.includes('execute-not-ready'));
assert.ok(almostReady.includes('Setup is still developing. Wait for full confirmation.'));

const building = htmlFor(setup({ trade_eval: { trade_stage: 'BUILDING / WATCHLIST' }, setupGrade: 'B+ TRADEABLE', entryStatus: 'Waiting', confirmationStarted: false, direction: 'LONG' }));
assert.ok(building.includes('data-normalized-status="BUILDING"'));
assert.ok(building.includes('execute-not-ready'));
assert.ok(building.includes('Continue monitoring. The setup is still developing.'));

console.log('Live card render v1 tests passed');
