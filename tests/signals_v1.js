// Kairos Dashboard Sprint 3 (2026-09 session) -- public/signals.js, the
// minimal stock-only signal dashboard. Tests the real module code via the
// repo's existing plain-Node DOM/fetch fake convention (see
// tests/dashboard_v1.js / tests/setup_board_v1.js), with the same
// createElement/appendChild-capable fake DOM tests/dashboard_v1.js uses,
// since this module also manages row identity via real DOM node reuse.
'use strict';

const assert = require('assert');

function makeElement(tag) {
  const el = {
    tagName: String(tag || 'div').toUpperCase(),
    children: [],
    parentNode: null,
    hidden: false,
    colSpan: 1,
    value: '',
    style: {},
    _classes: new Set(),
    _dataset: {},
    _listeners: {},
    _html: '',
    get innerHTML() { return el._html; },
    set innerHTML(v) { el._html = v; },
    get className() { return Array.from(el._classes).join(' '); },
    set className(v) { el._classes = new Set(String(v || '').split(/\s+/).filter(Boolean)); },
    get dataset() { return el._dataset; },
    classList: {
      add: (c) => el._classes.add(c),
      remove: (c) => el._classes.delete(c),
      contains: (c) => el._classes.has(c),
      toggle: (c, force) => {
        if (force === undefined) { el._classes.has(c) ? el._classes.delete(c) : el._classes.add(c); }
        else if (force) el._classes.add(c);
        else el._classes.delete(c);
      },
    },
    appendChild(child) {
      if (child.parentNode) {
        const idx = child.parentNode.children.indexOf(child);
        if (idx >= 0) child.parentNode.children.splice(idx, 1);
      }
      child.parentNode = el;
      el.children.push(child);
      return child;
    },
    removeChild(child) {
      const idx = el.children.indexOf(child);
      if (idx >= 0) el.children.splice(idx, 1);
      child.parentNode = null;
      return child;
    },
    addEventListener(type, fn) { (el._listeners[type] = el._listeners[type] || []).push(fn); },
    dispatch(type) { (el._listeners[type] || []).forEach((fn) => fn({ target: el })); },
    focus() {},
  };
  return el;
}

const elements = {
  apiBand: makeElement('div'),
  apiKeyInput: makeElement('input'),
  statusLine: makeElement('div'),
  emptyState: makeElement('div'),
  authPanel: makeElement('div'),
  dashTableWrap: makeElement('div'),
  signalsBody: makeElement('tbody'),
  signalsCount: makeElement('div'),
  freshnessBar: makeElement('div'),
};

global.document = {
  getElementById: (id) => elements[id] || null,
  createElement: (tag) => makeElement(tag),
  addEventListener: () => {},
};
global.window = { addEventListener: () => {} };
global.localStorage = { getItem: () => 'test-key', setItem: () => {}, removeItem: () => {} };
global.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

let fetchQueue = [];
global.fetch = async (url) => {
  const next = fetchQueue.shift();
  if (!next) throw new Error(`Unexpected fetch call with nothing queued: ${url}`);
  return {
    ok: next.status >= 200 && next.status < 300,
    status: next.status,
    text: async () => JSON.stringify(next.body),
  };
};

const signals = require('../public/signals.js');

function row(overrides = {}) {
  return {
    symbol: 'AMD', market: 'stock', direction: 'long', trade_type: 'LONG',
    state: 'WATCHING', state_label: 'Watching', next_step: 'Kairos is monitoring for the stated trigger/entry condition.',
    last_change: '2026-09-06T15:00:00Z',
    planned_entry: 100.0, entry: 100.0, stop: 95.0, exit: 95.0, target: 112.0, targets: [112.0],
    option_strike: null, option_expiration: null,
    source: 'ma_pipeline', setup_key: 'AMD-1', legacy_state: 'WAITING_FOR_TRIGGER',
    ...overrides,
  };
}

function payload(rows) {
  return { mechanism: 'dashboard_state_read_model_v1', disclaimer: 'test disclaimer', count: rows.length, setups: rows, supported_states: [] };
}

function rowClassOf(setupKey) {
  const entry = signals.rowIndex.get(setupKey);
  return entry ? entry.rowEl.className : null;
}

function tbodySymbols() {
  return elements.signalsBody.children
    .filter((c) => c.className && c.className.includes('setup-row'))
    .map((c) => c.dataset.setupKey);
}

async function run() {
  // --- A. Forex rows are excluded entirely -- this dashboard is stocks
  // only, regardless of state. ---
  fetchQueue = [{ status: 200, body: payload([
    row({ setup_key: 'FX-1', symbol: 'EURUSD', market: 'forex', state: 'ENTRY_READY' }),
    row({ setup_key: 'ST-1', symbol: 'AMD', market: 'stock', state: 'WATCHING' }),
  ]) }];
  await signals.loadSignals();
  assert.ok(!signals.rowIndex.has('FX-1'), 'A: a forex row must never appear, even when ENTRY_READY');
  assert.ok(signals.rowIndex.has('ST-1'), 'A: a stock row renders normally');
  assert.strictEqual(signals.isStock({ market: 'forex' }), false, 'A: isStock() rejects forex');
  assert.strictEqual(signals.isStock({ market: 'FOREX' }), false, 'A: isStock() is case-insensitive');
  assert.strictEqual(signals.isStock({ market: 'stock' }), true, 'A: isStock() accepts stock');

  // --- B. ENTRY_READY -> entire row green. ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'ER-1', symbol: 'NVDA', state: 'ENTRY_READY', state_label: 'Entry Ready' })]) }];
  await signals.loadSignals();
  assert.strictEqual(rowClassOf('ER-1'), 'setup-row row-accent-green', 'B: ENTRY_READY row is solid green');

  // --- C. Non-ready states are neutral rows -- no full-row color at all. ---
  for (const s of ['WATCHING', 'CONFIRMED', 'LOCATION_REACHED', 'WAITING_FOR_PULLBACK', 'EXECUTION_READY', 'POSITION_OPEN', 'INVALIDATED', 'CLOSED']) {
    const key = `neutral-${s}`;
    fetchQueue = [{ status: 200, body: payload([row({ setup_key: key, symbol: 'ZZZ', state: s })]) }];
    await signals.loadSignals();
    assert.strictEqual(rowClassOf(key), 'setup-row', `C: ${s} must be a plain neutral row, no accent -- even INVALIDATED is not red on this page`);
  }

  // --- D. Technical backend states translate into the 5 simple labels
  // (plus the disclosed 6th, Closed) -- never the raw backend state name. ---
  const translations = [
    ['DISCOVERED', 'Watching'],
    ['WATCHING', 'Watching'],
    ['LOCATION_REACHED', 'Almost Ready'],
    ['CONFIRMED', 'Almost Ready'],
    ['WAITING_FOR_PULLBACK', 'Almost Ready'],
    ['EXECUTION_READY', 'Almost Ready'],
    ['ENTRY_READY', 'Entry Ready'],
    ['INVALIDATED', 'Invalid'],
    ['POSITION_OPEN', 'Position Open'],
    ['CLOSED', 'Closed'],
  ];
  for (const [backendState, label] of translations) {
    assert.strictEqual(signals.simpleStatus(row({ state: backendState })), label, `D: ${backendState} -> "${label}"`);
  }
  // The raw backend state string must never leak into the rendered row.
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'D-1', symbol: 'MSFT', state: 'WAITING_FOR_PULLBACK' })]) }];
  await signals.loadSignals();
  const dHtml = signals.rowIndex.get('D-1').rowEl.innerHTML;
  assert.ok(dHtml.includes('Almost Ready'), 'D: shows the simple label');
  assert.ok(!dHtml.includes('WAITING_FOR_PULLBACK'), 'D: never shows the raw backend state name');

  // A state with no honest simple translation (simulating something like
  // STALE, which the backend returns with state=null) is excluded rather
  // than mislabeled.
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'D-unmapped', symbol: 'QQQ', state: null, state_label: null })]) }];
  await signals.loadSignals();
  assert.ok(!signals.rowIndex.has('D-unmapped'), 'D: an unmapped/null backend state is excluded, never guessed');

  // --- E. Entry / Exit / TP render correctly when present. ---
  const eRow = row({ setup_key: 'E-1', symbol: 'AMD', planned_entry: 101.5, entry: 101.5, stop: 96.25, exit: 96.25, target: null, targets: [110.0, 120.0] });
  const eFields = Object.fromEntries(signals.detailFields(eRow));
  assert.strictEqual(eFields['ENTRY'], '$101.50', 'E: ENTRY renders formatted');
  assert.strictEqual(eFields['EXIT'], '$96.25', 'E: EXIT renders formatted');
  assert.strictEqual(eFields['TP'], '$110.00, $120.00', 'E: TP renders multiple targets joined');

  // --- F. Option fields (CALL/PUT, STRIKE, EXPIRATION) render only when
  // the backend actually resolved a real option contract. ---
  const fOptionRow = row({
    setup_key: 'F-1', symbol: 'NVDA', state: 'ENTRY_READY', trade_type: 'CALL',
    option_strike: 130.0, option_expiration: '2026-09-19',
  });
  const fFields = Object.fromEntries(signals.detailFields(fOptionRow));
  assert.strictEqual(fFields['CALL/PUT'], 'CALL', 'F: CALL/PUT renders when a real option trade_type is present');
  assert.strictEqual(fFields['STRIKE'], '$130.00', 'F: STRIKE renders');
  assert.ok(fFields['EXPIRATION'].includes('2026'), 'F: EXPIRATION renders');

  const fPlainRow = row({ setup_key: 'F-2', symbol: 'AMD', trade_type: 'LONG', option_strike: null, option_expiration: null });
  const fPlainFields = Object.fromEntries(signals.detailFields(fPlainRow));
  assert.ok(!('CALL/PUT' in fPlainFields), 'F: no CALL/PUT field for a plain LONG/SHORT trade -- never fabricated');
  assert.ok(!('STRIKE' in fPlainFields), 'F: no STRIKE field when there is no real option contract');
  assert.ok(!('EXPIRATION' in fPlainFields), 'F: no EXPIRATION field when there is no real option contract');

  // A CALL/PUT trade_type with only a strike (no expiration) shows exactly
  // what's available -- never invents the missing half.
  const fPartialRow = row({ setup_key: 'F-3', symbol: 'TSLA', trade_type: 'PUT', option_strike: 240.0, option_expiration: null });
  const fPartialFields = Object.fromEntries(signals.detailFields(fPartialRow));
  assert.strictEqual(fPartialFields['CALL/PUT'], 'PUT');
  assert.strictEqual(fPartialFields['STRIKE'], '$240.00');
  assert.ok(!('EXPIRATION' in fPartialFields), 'F: EXPIRATION omitted, not fabricated, when the backend has no value for it');

  // --- G. Missing optional fields do not break rendering. ---
  const sparse = { symbol: 'ZZZ', market: 'stock', direction: null, trade_type: null, state: 'WATCHING', state_label: null, setup_key: 'G-1' };
  assert.doesNotThrow(() => signals.buildRowCells(sparse), 'G: buildRowCells must not throw on an almost-empty row');
  assert.doesNotThrow(() => signals.buildDetailHtml(sparse), 'G: buildDetailHtml must not throw on an almost-empty row');
  assert.deepStrictEqual(signals.detailFields(sparse), [], 'G: nothing set -> zero detail fields, not fabricated placeholders');
  fetchQueue = [{ status: 200, body: payload([sparse]) }];
  await assert.doesNotReject(signals.loadSignals(), 'G: a full load with a sparse row must not throw');
  assert.ok(signals.rowIndex.get('G-1'), 'G: the sparse row still renders (falls back to LONG/Watching/Wait)');

  // --- H. Inline expand/collapse works exactly as the main dashboard's does. ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'H-1', symbol: 'AAPL' })]) }];
  await signals.loadSignals();
  const hEntry = signals.rowIndex.get('H-1');
  assert.strictEqual(hEntry.detailEl.hidden, true, 'H: detail starts collapsed');
  hEntry.rowEl.dispatch('click');
  assert.strictEqual(hEntry.detailEl.hidden, false, 'H: click expands inline');
  assert.ok(hEntry.rowEl.classList.contains('expanded'), 'H: expanded class applied');
  hEntry.rowEl.dispatch('click');
  assert.strictEqual(hEntry.detailEl.hidden, true, 'H: second click collapses');
  assert.ok(!hEntry.rowEl.classList.contains('expanded'), 'H: expanded class removed');

  // --- I. No technical vocabulary anywhere in the rendered output. ---
  fetchQueue = [{ status: 200, body: payload([
    row({ setup_key: 'I-1', symbol: 'AMD', state: 'CONFIRMED' }),
    row({ setup_key: 'I-2', symbol: 'NVDA', state: 'ENTRY_READY', trade_type: 'CALL', option_strike: 130, option_expiration: '2026-09-19' }),
  ]) }];
  await signals.loadSignals();
  const fullHtml = signals.rowIndex.get('I-1').rowEl.innerHTML + signals.rowIndex.get('I-1').detailTd.innerHTML
    + signals.rowIndex.get('I-2').rowEl.innerHTML + signals.rowIndex.get('I-2').detailTd.innerHTML;
  for (const forbidden of ['CHoCH', 'BOS', 'displacement', 'HTF', 'LTF', 'premium', 'discount', 'supply', 'demand', 'lifecycle', 'debug', 'WAITING_FOR_TRIGGER']) {
    assert.ok(!fullHtml.toLowerCase().includes(forbidden.toLowerCase()), `I: forbidden technical term "${forbidden}" must not appear anywhere in the rendered output`);
  }

  // --- J. Sort order: Entry Ready, Position Open, Almost Ready, Watching,
  // Closed, Invalid. ---
  const unsorted = [
    row({ setup_key: 'J-invalid', state: 'INVALIDATED' }),
    row({ setup_key: 'J-closed', state: 'CLOSED' }),
    row({ setup_key: 'J-watching', state: 'WATCHING' }),
    row({ setup_key: 'J-almost', state: 'CONFIRMED' }),
    row({ setup_key: 'J-position', state: 'POSITION_OPEN' }),
    row({ setup_key: 'J-entryready', state: 'ENTRY_READY' }),
  ];
  const sortedKeys = signals.sortRows(unsorted).map((r) => r.setup_key);
  assert.deepStrictEqual(sortedKeys, ['J-entryready', 'J-position', 'J-almost', 'J-watching', 'J-closed', 'J-invalid'], 'J: sort order matches Entry Ready > Position Open > Almost Ready > Watching > Closed > Invalid');

  console.log('Dashboard Sprint 3 (signals.js) tests passed');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
