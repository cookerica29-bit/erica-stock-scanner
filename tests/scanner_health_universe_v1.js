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

const discoveryWarmingMeta = {
  universe: 'discovered',
  cache: 'miss',
  status: 'warming',
  refreshing: true,
  has_cache: false,
  discovery_status: {
    status: 'refreshing',
    running: true,
    has_cache: false,
    selected_count: 18,
    started_at: new Date(Date.now() - 125000).toISOString(),
    pipeline_counts: { raw_assets: 14162, hygiene_passed: 5683, options_liquidity_passed: 1039 },
  },
};
context.updateDataStatus(discoveryWarmingMeta);
assert.ok(getElement('dataStatus').innerHTML.includes('Discovery warming'));
assert.ok(getElement('dataStatus').innerHTML.includes('Building expanded universe'));
assert.strictEqual(context.isDiscoveryUniverseWarmup(discoveryWarmingMeta), true);
assert.strictEqual(context.isDiscoveredFirstScanWarming(discoveryWarmingMeta), false);
context.__fixtureMeta = discoveryWarmingMeta;
vm.runInContext('latestScannerMeta = __fixtureMeta; scannerRows = []; scannerNearMiss = [];', context);
vm.runInContext(`
  renderMarketCoveragePanel = () => {};
  renderMarketSnapshot = () => {};
  renderMarketIntelligencePanel = () => {};
  renderTopOpportunities = () => {};
`, context);
context.renderScannerResults();
assert.ok(getElement('results').innerHTML.includes('Discovery warming'));
assert.ok(getElement('results').innerHTML.includes('Building the expanded stock universe'));
assert.ok(getElement('results').innerHTML.includes('Raw assets'));
assert.ok(getElement('results').innerHTML.includes('14162'));
assert.ok(getElement('results').innerHTML.includes('Core scanner remains healthy'));
assert.ok(getElement('results').innerHTML.includes('View core universe while waiting'));

const discoveredFirstScanMeta = {
  universe: 'discovered',
  cache: 'miss',
  status: 'warming',
  refreshing: true,
  has_cache: false,
  refresh_duration: 49.7,
  discovery_status: { status: 'ready', running: false, has_cache: true, selected_count: 750 },
};
context.updateDataStatus(discoveredFirstScanMeta);
assert.ok(getElement('dataStatus').innerHTML.includes('Loading discovered scanner'));
assert.strictEqual(context.isDiscoveryUniverseWarmup(discoveredFirstScanMeta), false);
assert.strictEqual(context.isDiscoveredFirstScanWarming(discoveredFirstScanMeta), true);
context.__fixtureMeta = discoveredFirstScanMeta;
vm.runInContext('latestScannerMeta = __fixtureMeta; scannerRows = []; scannerNearMiss = [];', context);
context.renderScannerResults();
assert.ok(getElement('results').innerHTML.includes('Loading discovered scanner'));
assert.ok(getElement('results').innerHTML.includes('first discovered-universe scan'));

context.updateDataStatus({
  universe: 'discovered',
  status: 'fresh',
  stale: false,
  cache_age_seconds: 58,
  generated_at: '2026-07-27T20:45:00Z',
});
assert.ok(getElement('dataStatus').innerHTML.includes('Fresh · Updated'));

getElement('universeFilter').value = 'discovered';
let changedUniverse = '';
context.handleUniverseChange = () => { changedUniverse = getElement('universeFilter').value; };
context.switchScannerUniverse('default');
assert.strictEqual(changedUniverse, 'default');

getElement('universeFilter').value = 'default';
assert.strictEqual(context.scannerScanUrl(), '/api/scan?universe=default&view=summary');
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

let pollScheduled = false;
context.scheduleScannerWarmingPoll = () => { pollScheduled = true; };
context.fetchWithTimeout = (url) => {
  statusUrl = url;
  return Promise.resolve({ ok: true, json: () => Promise.resolve(discoveryWarmingMeta) });
};
vm.runInContext('scannerRows = []; scannerNearMiss = []; scannerWarmingPollTimer = null; scannerRequestController = null;', context);
await context.loadCacheStatus();
assert.strictEqual(statusUrl, '/api/cache/status?universe=discovered');
assert.strictEqual(pollScheduled, true);
assert.ok(getElement('results').innerHTML.includes('Discovery warming'));

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
