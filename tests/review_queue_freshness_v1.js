// Review Queue Evolution Sprint 1 -- freshness/auto-refresh (2026-09
// session). Same hand-built DOM/fetch fake convention as
// tests/review_queue_persistence_v1.js (no jsdom dependency available).
//
// The core invariant under test: a silent/auto-refresh tick must NEVER
// touch #mainContent (which may hold an in-progress, unsubmitted review
// form) -- only the standalone #freshnessBar element. A non-silent
// (manual/initial) load behaves exactly as before this sprint.
'use strict';

const assert = require('assert');

function makeElement() {
  const classes = new Set();
  return {
    _html: '',
    get innerHTML() { return this._html; },
    set innerHTML(value) { this._html = value; },
    style: {},
    classList: {
      add: (cls) => classes.add(cls),
      remove: (cls) => classes.delete(cls),
      contains: (cls) => classes.has(cls),
      toggle: (cls, on) => { if (on) classes.add(cls); else classes.delete(cls); },
    },
    focus() {},
  };
}

const elements = {
  apiBand: makeElement(),
  mainContent: makeElement(),
  apiKeyInput: makeElement(),
  freshnessBar: makeElement(),
};

global.document = {
  getElementById: (id) => elements[id] || null,
  querySelector: () => null,
  addEventListener: () => {}, // deliberately never fired -- see review_queue_auth_v1.js's own comment
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

function queueResponse(candidates, diagnostics) {
  fetchQueue.push({
    status: 200,
    body: {
      mechanism: 'review_value_v1_freshprice_sprint1',
      disclaimer: 'test disclaimer',
      snapshot_id: 'snap-1',
      count: candidates.length,
      candidates,
      diagnostics: diagnostics || {
        needs_review_total_count: candidates.length,
        needs_review_displayed_count: candidates.length,
      },
    },
  });
}

function candidate(ticker) {
  return {
    ticker, source: 'ma_pipeline', signal: 'long', rank: 1,
    entry_price: 100, stop: 95, target: 110, risk_reward: 2,
    review_value_score: 3.5, current_review: null, setup_key: `${ticker}|ma_pipeline|long|95.00|110.00`,
  };
}

const board = require('../public/review_queue.js');

async function run() {
  // --- 1. A normal (non-silent) load renders #mainContent, sets
  // lastLoadedAt, and populates the freshness bar. ---
  queueResponse([candidate('AAA')]);
  await board.loadQueue();
  assert.ok(elements.mainContent.innerHTML.length > 0, '1: normal load renders mainContent');
  assert.ok(board.state.lastLoadedAt instanceof Date, '1: lastLoadedAt set after a normal load');
  const afterFirstLoad = elements.mainContent.innerHTML;

  // --- 2. A silent load updates state.queue/diagnostics/lastLoadedAt and
  // the freshness bar, but does NOT touch #mainContent at all -- proves
  // an in-progress form on mainContent survives an auto-refresh tick. ---
  elements.mainContent.innerHTML = '<form>UNSAVED DRAFT INPUT</form>'; // simulate an in-progress review form
  const beforeSilent = board.state.lastLoadedAt;
  queueResponse([candidate('AAA'), candidate('BBB')], {
    needs_review_total_count: 2, needs_review_displayed_count: 2,
  });
  await board.loadQueue({ silent: true });
  assert.strictEqual(
    elements.mainContent.innerHTML, '<form>UNSAVED DRAFT INPUT</form>',
    '2: a silent refresh must NEVER touch #mainContent -- an in-progress form must survive it',
  );
  assert.strictEqual(board.state.queue.length, 2, '2: silent refresh still updates state.queue');
  assert.ok(board.state.lastLoadedAt.getTime() >= beforeSilent.getTime(), '2: lastLoadedAt still advances on a silent refresh');
  assert.ok(
    elements.freshnessBar.innerHTML.includes('ago') || elements.freshnessBar.innerHTML.includes('just now'),
    '2: freshness bar shows a relative-time label',
  );
  assert.ok(elements.freshnessBar.innerHTML.includes('2/2'), '2: freshness bar reflects the new diagnostics pool size');

  // --- 3. relativeTimeLabel formats sensibly for known offsets. ---
  const now = Date.now();
  assert.strictEqual(board.relativeTimeLabel(null), 'never', '3: null -> "never"');
  assert.strictEqual(board.relativeTimeLabel(new Date(now - 2000)), 'just now', '3: <5s -> "just now"');
  assert.strictEqual(board.relativeTimeLabel(new Date(now - 30000)), '30s ago', '3: 30s -> "30s ago"');
  assert.strictEqual(board.relativeTimeLabel(new Date(now - 120000)), '2m ago', '3: 120s -> "2m ago"');
  assert.strictEqual(board.relativeTimeLabel(new Date(now - 3 * 3600 * 1000)), '3h ago', '3: 3h -> "3h ago"');

  // --- 4. A silent refresh that FAILS must never surface an error over
  // an active review -- state/mainContent must be left exactly as-is. ---
  elements.mainContent.innerHTML = '<form>STILL UNSAVED</form>';
  fetchQueue.push({ status: 500, body: { detail: 'boom' } });
  await board.loadQueue({ silent: true });
  assert.strictEqual(elements.mainContent.innerHTML, '<form>STILL UNSAVED</form>', '4: a failed silent refresh must not touch mainContent');

  // --- 5. manualRefreshQueue() performs a full (non-silent) reload. ---
  queueResponse([candidate('CCC')]);
  await board.manualRefreshQueue();
  assert.notStrictEqual(elements.mainContent.innerHTML, '<form>STILL UNSAVED</form>', '5: manual refresh performs a real, full reload');
  assert.strictEqual(board.state.queue.length, 1, '5: manual refresh reflects the newly fetched queue');

  // --- 6. startAutoRefresh never stacks multiple intervals. ---
  const realSetInterval = global.setInterval;
  let setIntervalCalls = 0;
  global.setInterval = (...args) => { setIntervalCalls += 1; return realSetInterval(...args); };
  try {
    board.startAutoRefresh();
    board.startAutoRefresh();
    board.startAutoRefresh();
    assert.strictEqual(setIntervalCalls, 1, '6: calling startAutoRefresh multiple times must only ever arm one interval');
  } finally {
    board.stopAutoRefresh();
    global.setInterval = realSetInterval;
  }

  console.log('Review queue freshness v1 tests passed');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
