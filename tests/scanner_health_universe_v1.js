const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const cardData = require('../public/card_data.js');

const html = fs.readFileSync('public/index.html', 'utf8');
const inline = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)][0][1];

function elementStub(id = '') {
  return {
    id,
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

const elements = new Map();
function getElement(id) {
  if (!elements.has(id)) elements.set(id, elementStub(id));
  return elements.get(id);
}

const storage = {};
const errors = [];
const context = {
  console: { ...console, error: (...args) => errors.push(args.join(' ')) },
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
  AbortController: class {
    constructor() {
      this.signal = {
        addEventListener() {},
        removeEventListener() {},
      };
    }
    abort() {}
  },
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => true,
  fetch: () => Promise.reject(new Error('network disabled in scanner health universe test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  sessionStorage: {
    getItem: key => storage[`session:${key}`] || null,
    setItem: (key, value) => { storage[`session:${key}`] = String(value); },
    removeItem: key => { delete storage[`session:${key}`]; },
  },
  document: {
    body: { appendChild() {} },
    getElementById: getElement,
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

async function main() {
getElement('universeFilter').value = 'discovered';
assert.strictEqual(context.normalizedScannerUniverse(), 'discovered');
assert.strictEqual(context.scannerScanUrl(), '/api/scan');
assert.strictEqual(context.scannerCacheStatusUrl(), '/api/cache/status?universe=discovered');

context.updateDataStatus({
  universe: 'default',
  status: 'stale',
  stale: true,
  age_seconds: 900,
  generated_at: '2026-07-27T20:00:00Z',
});
assert.ok(getElement('dataStatus').innerHTML.includes('Scanner health is updating'));
assert.ok(!getElement('dataStatus').innerHTML.includes('Stale · Updated'));

context.updateDataStatus({
  universe: 'discovered',
  status: 'fresh',
  stale: false,
  cache_age_seconds: 58,
  generated_at: '2026-07-27T20:45:00Z',
});
assert.ok(getElement('dataStatus').innerHTML.includes('Fresh · Updated'));

getElement('universeFilter').value = 'default';
assert.strictEqual(context.scannerScanUrl(), '/api/scan?universe=default');
assert.strictEqual(context.scannerCacheStatusUrl(), '/api/cache/status?universe=default');
context.updateDataStatus({
  universe: 'default',
  status: 'fresh',
  stale: false,
  cache_age_seconds: 30,
  generated_at: '2026-07-27T20:45:00Z',
});
assert.ok(getElement('dataStatus').innerHTML.includes('Fresh · Updated'));

let statusUrl = '';
getElement('universeFilter').value = 'discovered';
context.fetchWithTimeout = (url) => {
  statusUrl = url;
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve({
      universe: 'discovered',
      status: 'fresh',
      stale: false,
      cache_age_seconds: 10,
      generated_at: '2026-07-27T20:47:53Z',
      symbols_attempted: 750,
      symbols_terminally_evaluated: 750,
      partial_result: false,
      refreshing: false,
    }),
  });
};
await context.loadCacheStatus();
assert.strictEqual(statusUrl, '/api/cache/status?universe=discovered');
assert.ok(getElement('dataStatus').innerHTML.includes('Fresh · Updated'));

const completedPayload = {
  rows: [{ ticker: 'DOW', direction: 'SHORT', quality: { grade: 'A' }, trade_eval: {} }],
  near_miss: [],
  meta: {
    universe: 'discovered',
    cache: 'hit',
    generated_at: '2026-07-27T20:47:53Z',
    symbols_attempted: 750,
    symbols_terminally_evaluated: 750,
    partial_result: false,
    refreshing: false,
  },
};

vm.runInContext(`
  renderScannerResults = () => { throw new Error('render exploded'); };
  renderExecutionTab = () => {};
  renderFurtherAnalysisTab = () => {};
  renderPositionsTab = () => {};
  detectStockAlertEvents = () => [];
  deliverStockAlertEvent = () => {};
  evaluatePositionAlertNotifications = () => {};
  refreshExpectedMoveTrackingFromRows = () => {};
  recordEnterNowFunnelSnapshot = () => {};
  loadNotifications = () => {};
`, context);
context.fetchWithTimeout = () => Promise.resolve({ ok: true, json: () => Promise.resolve(completedPayload) });
await context.runScan();
assert.strictEqual(getElement('scanBtn').disabled, false);
assert.ok(getElement('results').innerHTML.includes('Scan completed, but results could not be displayed.'));
assert.ok(errors.some(line => line.includes('scan response received but render failed')));

context.fetchWithTimeout = () => Promise.resolve({ ok: false, status: 500 });
await context.runScan();
assert.strictEqual(getElement('scanBtn').disabled, false);

console.log('scanner_health_universe_v1 passed');
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
