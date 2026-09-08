// Kairos Dashboard Sprint 2 (2026-09 session) -- public/dashboard.js.
// Extended in Sprint 2.1 (density polish: full-row coloring reserved for
// ENTRY_READY/INVALIDATED only, individual-state color moved entirely to
// the Current State pill, shortened main-table Next Step text, debug-only
// Lifecycle field). Tests the real module code via the repo's existing
// plain-Node DOM/fetch fake convention (see tests/setup_board_v1.js),
// extended with real element creation/appendChild/classList since this
// module manages row identity via actual DOM node reuse (not just
// innerHTML templating) -- see dashboard.js's own
// render()/ensureRow()/updateRow() comments for why.
'use strict';

const assert = require('assert');

// ---- minimal fake DOM: real enough for createElement/appendChild/
// classList/dataset/hidden, which is everything dashboard.js touches. ----
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
  dashboardBody: makeElement('tbody'),
  dashboardDisclaimer: makeElement('div'),
  dashboardCount: makeElement('div'),
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

const dashboard = require('../public/dashboard.js');

function row(overrides = {}) {
  return {
    symbol: 'AMD', market: 'stock', direction: 'long',
    state: 'WATCHING', state_label: 'Watching', next_step: 'Kairos is monitoring for the stated trigger/entry condition.',
    last_change: '2026-09-06T15:00:00Z',
    planned_entry: 100.0, entry: 100.0, stop: 95.0, exit: 95.0, target: 112.0, targets: [112.0],
    source: 'ma_pipeline', setup_key: 'AMD-1', legacy_state: 'WAITING_FOR_TRIGGER',
    ...overrides,
  };
}

function payload(rows) {
  return { mechanism: 'dashboard_state_read_model_v1', disclaimer: 'test disclaimer', count: rows.length, setups: rows, supported_states: [] };
}

function rowClassOf(setupKey) {
  const entry = dashboard.rowIndex.get(setupKey);
  return entry ? entry.rowEl.className : null;
}

function tbodyOrder() {
  return elements.dashboardBody.children
    .filter((c) => c.className && c.className.includes('setup-row'))
    .map((c) => c.dataset.setupKey);
}

async function run() {
  // --- A. ENTRY_READY entire row turns green (the critical rule,
  // unchanged by Sprint 2.1 -- only the class name changed from
  // row-state-green to row-accent-green). ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'ER-1', symbol: 'NVDA', state: 'ENTRY_READY', state_label: 'Entry Ready' })]) }];
  await dashboard.loadDashboard();
  assert.strictEqual(rowClassOf('ER-1'), 'setup-row row-accent-green', 'A: ENTRY_READY row gets the row-accent-green class on the WHOLE row');

  // --- A2. Sprint 2.1: every other state now renders with a plain
  // neutral row -- no row-level color at all, only the pill carries
  // state color. Covers every state named in requirement #6 explicitly
  // (POSITION_OPEN, WAITING_FOR_PULLBACK, LOCATION_REACHED, CONFIRMED)
  // plus WATCHING/DISCOVERED for completeness. ---
  const neutralStates = ['WATCHING', 'CONFIRMED', 'POSITION_OPEN', 'WAITING_FOR_PULLBACK', 'LOCATION_REACHED', 'EXECUTION_READY', 'DISCOVERED'];
  for (const s of neutralStates) {
    const key = `neutral-${s}`;
    fetchQueue = [{ status: 200, body: payload([row({ setup_key: key, symbol: 'ZZZ', state: s })]) }];
    await dashboard.loadDashboard();
    assert.strictEqual(rowClassOf(key), 'setup-row', `A2: ${s} must use the plain neutral row background, no row-accent-* class`);
  }

  // --- A3. State pills retain their own individual colors (requirement
  // #5) even though the row itself is neutral now. ---
  const pillExpectations = [
    ['WATCHING', 'pill-blue'],
    ['CONFIRMED', 'pill-yellow'],
    ['LOCATION_REACHED', 'pill-yellow'],
    ['WAITING_FOR_PULLBACK', 'pill-orange'],
    ['EXECUTION_READY', 'pill-orange'],
    ['POSITION_OPEN', 'pill-purple'],
    ['DISCOVERED', 'pill-gray'],
    ['ENTRY_READY', 'pill-green'],
    ['INVALIDATED', 'pill-red'],
    ['TARGET_HIT', 'pill-green'],
  ];
  for (const [s, pillClass] of pillExpectations) {
    const key = `pill-${s}`;
    fetchQueue = [{ status: 200, body: payload([row({ setup_key: key, symbol: 'ZZZ', state: s, state_label: s })]) }];
    await dashboard.loadDashboard();
    assert.ok(
      dashboard.rowIndex.get(key).rowEl.innerHTML.includes(`state-pill ${pillClass}`),
      `A3: ${s}'s Current State pill must carry ${pillClass}`,
    );
  }

  // --- B. Bullish (long) but not ENTRY_READY must NOT be green. ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'B-1', symbol: 'MSFT', direction: 'long', state: 'WATCHING' })]) }];
  await dashboard.loadDashboard();
  const bClass = rowClassOf('B-1');
  assert.ok(!bClass.includes('green'), 'B: a bullish (long) non-ready row must not be green merely because direction is bullish');
  assert.strictEqual(bClass, 'setup-row', 'B: WATCHING is a neutral row -- its color lives only in the pill now');

  // --- C. Bearish (short) but not INVALIDATED must NOT be red. ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'C-1', symbol: 'TLT', direction: 'short', state: 'CONFIRMED' })]) }];
  await dashboard.loadDashboard();
  const cClass = rowClassOf('C-1');
  assert.ok(!cClass.includes('red'), 'C: a bearish (short) non-invalidated row must not be red merely because direction is bearish');
  assert.strictEqual(cClass, 'setup-row', 'C: CONFIRMED is a neutral row -- its color lives only in the pill now');

  // --- D. INVALIDATED row is a restrained red, regardless of direction. ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'D-1', symbol: 'TSLA', direction: 'short', state: 'INVALIDATED', state_label: 'Invalidated' })]) }];
  await dashboard.loadDashboard();
  assert.strictEqual(rowClassOf('D-1'), 'setup-row row-accent-red', 'D: INVALIDATED row is a restrained red');

  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'D-2', symbol: 'AMD', direction: 'long', state: 'INVALIDATED', state_label: 'Invalidated' })]) }];
  await dashboard.loadDashboard();
  assert.strictEqual(rowClassOf('D-2'), 'setup-row row-accent-red', 'D: INVALIDATED is red even for a long/bullish setup');

  // --- D2. TARGET_HIT (2026-09 session) is a resolved win -- full-row
  // green, the same reasoning as INVALIDATED's own restrained-red inclusion. ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'D-3', symbol: 'OXY', direction: 'long', state: 'TARGET_HIT', state_label: 'Target Hit' })]) }];
  await dashboard.loadDashboard();
  assert.strictEqual(rowClassOf('D-3'), 'setup-row row-accent-green', 'D2: TARGET_HIT row is a resolved-win green');

  // --- E. Row expands inline on click; collapses on a second click.
  // Uses EXECUTION_READY (no SHORT_NEXT_STEP entry) so the main-table
  // cell keeps showing the raw backend next_step text verbatim -- makes
  // the "content still updates in place" check below meaningful. ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'E-1', symbol: 'AAPL', state: 'EXECUTION_READY' })]) }];
  await dashboard.loadDashboard();
  const eEntry = dashboard.rowIndex.get('E-1');
  assert.strictEqual(eEntry.detailEl.hidden, true, 'E: detail row starts collapsed');
  eEntry.rowEl.dispatch('click');
  assert.strictEqual(eEntry.detailEl.hidden, false, 'E: clicking the row expands the inline detail panel');
  assert.ok(eEntry.rowEl.classList.contains('expanded'), 'E: expanded class applied for chevron styling');
  eEntry.rowEl.dispatch('click');
  assert.strictEqual(eEntry.detailEl.hidden, true, 'E: clicking again collapses it');
  assert.ok(!eEntry.rowEl.classList.contains('expanded'), 'E: expanded class removed on collapse');

  // Expand state must survive an unrelated data refresh (no disruptive
  // full-table redraw) -- same DOM node, not a freshly created one.
  eEntry.rowEl.dispatch('click'); // expand again
  const nodeBefore = eEntry.rowEl;
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'E-1', symbol: 'AAPL', state: 'EXECUTION_READY', next_step: 'updated next step text' })]) }];
  await dashboard.loadDashboard();
  const eEntryAfter = dashboard.rowIndex.get('E-1');
  assert.strictEqual(eEntryAfter.rowEl, nodeBefore, 'E: the SAME DOM node is reused across a refresh, not recreated');
  assert.strictEqual(eEntryAfter.detailEl.hidden, false, 'E: an expanded row stays expanded across a background refresh');
  assert.ok(eEntryAfter.rowEl.innerHTML.includes('updated next step text'), 'E: row content still updates in place');
  eEntryAfter.rowEl.dispatch('click'); // collapse for subsequent tests

  // --- F. Entry/stop/TP render correctly when present. ---
  fetchQueue = [{ status: 200, body: payload([row({
    setup_key: 'F-1', symbol: 'AMD', planned_entry: 101.5, entry: 101.5, stop: 96.25, exit: 96.25,
    target: null, targets: [110.0, 120.0],
  })]) }];
  await dashboard.loadDashboard();
  const fFields = dashboard.detailFields(dashboard.state.rows.find((r) => r.setup_key === 'F-1'));
  const fMap = Object.fromEntries(fFields);
  assert.strictEqual(fMap['Entry / Planned Entry'], '$101.50', 'F: entry renders formatted');
  assert.strictEqual(fMap['Stop / Exit'], '$96.25', 'F: stop renders formatted');
  assert.strictEqual(fMap['TP / Targets'], '$110.00, $120.00', 'F: multiple targets render joined');

  // --- G. Missing optional fields do not break rendering -- omitted, not
  // shown as a placeholder ("show available values only"). ---
  const sparse = { symbol: 'ZZZ', market: 'stock', direction: null, state: null, state_label: null, next_step: null, last_change: null, setup_key: 'G-1', legacy_state: null };
  assert.doesNotThrow(() => dashboard.buildRowCells(sparse), 'G: buildRowCells must not throw on an almost-empty row');
  assert.doesNotThrow(() => dashboard.buildDetailHtml(sparse), 'G: buildDetailHtml must not throw on an almost-empty row');
  const gFields = dashboard.detailFields(sparse);
  assert.deepStrictEqual(gFields, [], 'G: a row with nothing set produces zero detail fields, not fabricated placeholders');
  fetchQueue = [{ status: 200, body: payload([sparse]) }];
  await assert.doesNotReject(dashboard.loadDashboard(), 'G: a full load with a sparse row must not throw');
  assert.ok(dashboard.rowIndex.get('G-1').rowEl.innerHTML.includes('--'), 'G: missing direction/next_step in the ROW cells fall back to a placeholder (unlike the detail panel, which omits them)');

  // --- H. Sort order: ENTRY_READY, then execution/almost-ready, then
  // confirmed, then watching, then inactive, then invalidated -- with
  // stable ordering preserved within equal-priority states. ---
  const unsorted = [
    row({ setup_key: 'H-invalidated', state: 'INVALIDATED' }),
    row({ setup_key: 'H-targethit', state: 'TARGET_HIT' }),
    row({ setup_key: 'H-discovered', state: 'DISCOVERED' }),
    row({ setup_key: 'H-watching-1', state: 'WATCHING' }),
    row({ setup_key: 'H-confirmed', state: 'CONFIRMED' }),
    row({ setup_key: 'H-execready', state: 'EXECUTION_READY' }),
    row({ setup_key: 'H-entryready', state: 'ENTRY_READY' }),
    row({ setup_key: 'H-watching-2', state: 'WATCHING' }),
    row({ setup_key: 'H-pullback', state: 'WAITING_FOR_PULLBACK' }),
    row({ setup_key: 'H-locreached', state: 'LOCATION_REACHED' }),
    row({ setup_key: 'H-closed', state: 'CLOSED' }),
  ];
  const sortedKeys = dashboard.sortRows(unsorted).map((r) => r.setup_key);
  assert.strictEqual(sortedKeys[0], 'H-entryready', 'H: ENTRY_READY sorts first');
  assert.deepStrictEqual(new Set(sortedKeys.slice(1, 3)), new Set(['H-execready', 'H-pullback']), 'H: execution/almost-ready states are next');
  assert.deepStrictEqual(new Set(sortedKeys.slice(3, 5)), new Set(['H-confirmed', 'H-locreached']), 'H: confirmed/progressing states are next');
  assert.strictEqual(sortedKeys[5], 'H-watching-1', 'H: watching preserves stable original order (first watching row first)');
  assert.strictEqual(sortedKeys[6], 'H-watching-2', 'H: watching preserves stable original order (second watching row second)');
  assert.deepStrictEqual(new Set(sortedKeys.slice(7, 9)), new Set(['H-discovered', 'H-closed']), 'H: inactive states are next');
  assert.deepStrictEqual(new Set(sortedKeys.slice(9, 11)), new Set(['H-invalidated', 'H-targethit']), 'H: TARGET_HIT sorts alongside INVALIDATED -- both are terminal, resolved outcomes');

  // Rendering also reflects this order in the actual DOM (tbody child order).
  fetchQueue = [{ status: 200, body: payload(unsorted) }];
  await dashboard.loadDashboard();
  assert.strictEqual(tbodyOrder()[0], 'H-entryready', 'H: rendered DOM order also puts ENTRY_READY first');
  assert.deepStrictEqual(
    new Set(tbodyOrder().slice(-2)), new Set(['H-invalidated', 'H-targethit']),
    'H: rendered DOM order puts the two terminal/resolved states (INVALIDATED, TARGET_HIT) last',
  );

  // --- I. Market column reflects stock vs forex (Sprint 2 backend addition). ---
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'I-1', symbol: 'EURUSD', market: 'forex' })]) }];
  await dashboard.loadDashboard();
  assert.ok(dashboard.rowIndex.get('I-1').rowEl.innerHTML.includes('FOREX'), 'I: forex market renders');

  // --- J. Sprint 2.1 requirement #7: short, operational Next Step labels
  // render in the main table for every state the task named explicitly. ---
  const shortLabelExpectations = [
    ['ENTRY_READY', 'Review entry'],
    ['POSITION_OPEN', 'Manage position'],
    ['WAITING_FOR_PULLBACK', 'Wait for pullback'],
    ['LOCATION_REACHED', 'Wait for confirmation'],
    ['CONFIRMED', 'Wait for execution'],
    ['WATCHING', 'Keep watching'],
    ['INVALIDATED', 'Setup invalid'],
    ['TARGET_HIT', 'Target hit'],
  ];
  for (const [s, label] of shortLabelExpectations) {
    const key = `short-${s}`;
    fetchQueue = [{ status: 200, body: payload([row({
      setup_key: key, symbol: 'ZZZ', state: s,
      next_step: 'a much longer backend explanation that would not fit in a spreadsheet cell',
    })]) }];
    await dashboard.loadDashboard();
    assert.ok(
      dashboard.rowIndex.get(key).rowEl.innerHTML.includes(label),
      `J: ${s} must show the short operational phrase "${label}" in the main table`,
    );
    assert.ok(
      !dashboard.rowIndex.get(key).rowEl.innerHTML.includes('a much longer backend explanation'),
      `J: ${s}'s main-table cell must show the SHORT phrase, not the full backend explanation`,
    );
  }

  // A state with no short-phrase entry keeps showing the backend's own
  // full text verbatim in the main table ("shorten where possible", not
  // "invent a phrase for every state").
  fetchQueue = [{ status: 200, body: payload([row({
    setup_key: 'J-fallback', symbol: 'ZZZ', state: 'EXECUTION_READY', next_step: 'a genuinely unshortened backend phrase',
  })]) }];
  await dashboard.loadDashboard();
  assert.ok(
    dashboard.rowIndex.get('J-fallback').rowEl.innerHTML.includes('a genuinely unshortened backend phrase'),
    'J: a state with no short-phrase mapping falls back to the full backend next_step text',
  );

  // --- K. Requirement #8: the expanded detail panel retains the FULL
  // backend-provided explanation even though the main table shows the
  // short phrase for the same row. ---
  const kRow = row({
    setup_key: 'K-1', symbol: 'NVDA', state: 'ENTRY_READY', state_label: 'Entry Ready',
    next_step: 'This setup has cleared all current safety gates.',
  });
  fetchQueue = [{ status: 200, body: payload([kRow]) }];
  await dashboard.loadDashboard();
  assert.ok(dashboard.rowIndex.get('K-1').rowEl.innerHTML.includes('Review entry'), 'K: main row shows the short phrase');
  assert.ok(!dashboard.rowIndex.get('K-1').rowEl.innerHTML.includes('This setup has cleared'), 'K: main row does NOT show the full explanation');
  const kFields = Object.fromEntries(dashboard.detailFields(kRow));
  assert.strictEqual(kFields['Next Step'], 'This setup has cleared all current safety gates.', 'K: expanded detail keeps the FULL, unabbreviated backend explanation');

  // --- L. Requirement #11: Lifecycle (debug) is hidden from the normal
  // production UI, shown only behind an explicit debug condition. ---
  const lRow = row({ setup_key: 'L-1', legacy_state: 'WAITING_FOR_TRIGGER' });
  delete global.window.location; // default: no debug query param at all
  assert.strictEqual(dashboard.isDebugMode(), false, 'L: debug mode is off by default');
  const lFieldsProd = dashboard.detailFields(lRow);
  assert.ok(!lFieldsProd.some(([label]) => label === 'Lifecycle (debug)'), 'L: Lifecycle (debug) is absent from the normal production UI');

  global.window.location = { search: '?debug=1' };
  assert.strictEqual(dashboard.isDebugMode(), true, 'L: ?debug=1 turns on debug mode');
  const lFieldsDebug = dashboard.detailFields(lRow);
  const lDebugField = lFieldsDebug.find(([label]) => label === 'Lifecycle (debug)');
  assert.ok(lDebugField, 'L: Lifecycle (debug) appears once debug mode is explicitly on');
  assert.strictEqual(lDebugField[1], 'WAITING_FOR_TRIGGER', 'L: Lifecycle (debug) shows the real raw legacy_state value');
  delete global.window.location; // reset for any future test additions

  // --- M. Trade-Management Automation (2026-09 session): breakeven/
  // partial-profit badges shown alongside the state pill, not folded
  // into it -- both can be true at once, tested here together. ---
  const mRow = row({
    setup_key: 'M-1', symbol: 'AMD', state: 'ACTIONABLE', state_label: 'Entry Ready',
    breakeven_set: true, breakeven_set_at: '2026-09-06T10:00:00Z',
    partial_profit_suggested: true, partial_profit_suggested_at: '2026-09-06T12:00:00Z',
  });
  fetchQueue = [{ status: 200, body: payload([mRow]) }];
  await dashboard.loadDashboard();
  const mHtml = dashboard.rowIndex.get('M-1').rowEl.innerHTML;
  assert.ok(mHtml.includes('mgmt-badge breakeven'), 'M: breakeven badge renders when breakeven_set is true');
  assert.ok(mHtml.includes('mgmt-badge partial'), 'M: partial-profit badge renders when partial_profit_suggested is true');

  // Neither badge renders when neither flag is set (default `row()`).
  fetchQueue = [{ status: 200, body: payload([row({ setup_key: 'M-2', symbol: 'MSFT' })]) }];
  await dashboard.loadDashboard();
  const m2Html = dashboard.rowIndex.get('M-2').rowEl.innerHTML;
  assert.ok(!m2Html.includes('mgmt-badge'), 'M: no badge renders when neither flag is set');

  // Detail panel: current stop shown with the original alongside it once
  // they differ (breakeven/trail moved it).
  const mFields = Object.fromEntries(dashboard.detailFields(row({
    stop: 100.0, exit: 100.0, original_stop: 95.0,
  })));
  assert.strictEqual(mFields['Stop / Exit'], '$100.00 (originally $95.00)', 'M: detail panel shows both current and original stop once they differ');

  const mFieldsUnchanged = Object.fromEntries(dashboard.detailFields(row({ original_stop: 95.0 }))); // stop === original_stop === 95.0
  assert.strictEqual(mFieldsUnchanged['Stop / Exit'], '$95.00', 'M: detail panel shows a single value when the stop has never moved');

  const mFieldsBE = Object.fromEntries(dashboard.detailFields(row({ breakeven_set: true, breakeven_set_at: '2026-09-06T10:00:00Z' })));
  assert.ok(mFieldsBE['Breakeven Stop'] && mFieldsBE['Breakeven Stop'].startsWith('Set '), 'M: Breakeven Stop detail field appears once set');

  const mFieldsNoBE = Object.fromEntries(dashboard.detailFields(row({})));
  assert.ok(!('Breakeven Stop' in mFieldsNoBE), 'M: Breakeven Stop detail field is absent when not set');

  console.log('Dashboard Sprint 2 / 2.1 (dashboard.js) tests passed');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
