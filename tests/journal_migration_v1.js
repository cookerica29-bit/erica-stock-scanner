const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const contractGuidance = require('../public/contract_guidance.js');
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
let fetchImpl = () => Promise.reject(new Error('fetch not stubbed'));
const context = {
  console: { ...console, info() {} },
  Date,
  Math,
  Number,
  String,
  Boolean,
  Array,
  Object,
  JSON,
  RegExp,
  URLSearchParams,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => false,
  fetch: (...args) => fetchImpl(...args),
  AbortController: class {
    constructor() { this.signal = {}; }
    abort() {}
  },
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  crypto: { randomUUID: () => 'generated-id' },
  document: {
    body: { appendChild() {} },
    getElementById: () => elementStub(),
    querySelectorAll: () => [],
    createElement: () => elementStub(),
  },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
context.window = {
  KairosExecutionGuidance: executionGuidance,
  KairosContractGuidance: contractGuidance,
  KairosCardData: cardData,
  open: () => {},
  addEventListener: () => {},
  removeEventListener: () => {},
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);

function entry(overrides = {}) {
  return {
    journal_id: 'j-1',
    position_id: 'p-1',
    ticker: 'OXY',
    direction: 'LONG',
    entry_timestamp: '2026-07-20T14:00:00Z',
    actual_underlying_entry: 54.6,
    actual_strike: 55,
    actual_expiration: '2026-08-21',
    updated_at: '2026-07-20T15:00:00Z',
    position_state_history: [{ event_id: 'server-event' }],
    ...overrides,
  };
}

const stable = context.ensureJournalStableIds({ ticker: 'AAPL' });
assert.strictEqual(stable.journal_id, 'generated-id');
assert.strictEqual(stable.position_id, 'generated-id');

let analysis = context.analyzeJournalMigration([entry({ journal_id: 'local-new', position_id: 'local-pos' })], []);
assert.strictEqual(analysis.newEntries.length, 1);
assert.strictEqual(analysis.conflicts.length, 0);

analysis = context.analyzeJournalMigration(
  [entry({ position_state_history: [{ event_id: 'local-event' }] })],
  [entry({ position_state_history: [{ event_id: 'server-event' }] })]
);
assert.strictEqual(analysis.matchingEntries.length, 1);
assert.strictEqual(JSON.stringify(analysis.matchingEntries[0].merged_history.map(e => e.event_id)), JSON.stringify(['server-event', 'local-event']));

analysis = context.analyzeJournalMigration(
  [entry({ journal_id: '', id: '', position_id: '', updated_at: '' })],
  [entry({ journal_id: 'server-id', position_id: 'server-pos' })]
);
assert.strictEqual(analysis.possibleDuplicates.length, 1);
assert.strictEqual(analysis.conflicts.length, 1);

storage.stock_scanner_journal = JSON.stringify([entry()]);
const backupKey = context.backupLocalJournalBeforeMigration(context.loadJournal());
assert.ok(backupKey.startsWith('stock_scanner_journal_backup_'));
assert.ok(storage[backupKey].includes('OXY'));

storage.kairos_journal_admin_token = 'secret';
let calls = [];
fetchImpl = (url, options = {}) => {
  calls.push({ url, options });
  if (String(url).startsWith('/api/journal?')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ entries: [] }) });
  }
  if (String(url).includes('/api/journal/migrate')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ entries: [entry()], created: 1, updated: 0, conflicts: 0, conflict_ids: [] }) });
  }
  return Promise.reject(new Error('unexpected url'));
};

context.saveJournal([entry({ journal_id: 'pre-confirm' })]);
assert.strictEqual(calls.length, 0, 'local save must not silently upload before confirmed migration');

context.migrateLocalJournalToServer(() => true).then(result => {
  assert.strictEqual(result.migrated, true);
  assert.ok(calls.some(call => call.url.includes('/api/journal/migrate')));
  assert.ok(JSON.parse(storage.stock_scanner_journal_migrated_ids).includes('j-1'));
  assert.strictEqual(storage.stock_scanner_journal_server_authoritative, 'true');

  fetchImpl = () => Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) });
  return context.syncJournalEntriesToServer([entry({ journal_id: 'pending-1' })]);
}).then(result => {
  assert.strictEqual(result.synced, false);
  assert.ok(JSON.parse(storage.stock_scanner_journal_pending_sync).some(e => e.journal_id === 'pending-1'));

  fetchImpl = url => Promise.resolve({ ok: true, json: () => Promise.resolve({ entries: [entry({ journal_id: 'server-only', ticker: 'MSFT' })] }) });
  return context.initializeJournalServerSync();
}).then(result => {
  assert.strictEqual(result.enabled, true);
  assert.strictEqual(result.migration_required, true, 'local OXY backup still requires review against server-only MSFT');
  assert.ok(storage.stock_scanner_journal_server_cache.includes('MSFT'));
  console.log('Journal migration v1 tests passed');
}).catch(error => {
  console.error(error);
  process.exit(1);
});
