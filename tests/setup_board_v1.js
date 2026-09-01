// Approved Setups / Watch Setups boards (2026-09 session). Presentation
// only -- both boards call the SAME GET /candidates/review-queue payload
// Review Queue already uses (no new backend endpoint, no duplicated
// strategy calculation) and filter client-side by
// current_review.decision. This tests the real public/setup_board.js code
// via the repo's existing plain-Node DOM/fetch fake convention (see
// tests/review_queue_persistence_v1.js), not a reimplementation.
'use strict';

const assert = require('assert');

function makeElement() {
  const classes = new Set();
  return {
    _html: '',
    get innerHTML() { return this._html; },
    set innerHTML(value) { this._html = value; },
    value: '',
    classList: {
      add: (cls) => classes.add(cls),
      remove: (cls) => classes.delete(cls),
      contains: (cls) => classes.has(cls),
    },
    focus() {},
  };
}

const elements = {
  apiBand: makeElement(),
  mainContent: makeElement(),
  apiKeyInput: makeElement(),
};

global.document = {
  getElementById: (id) => elements[id] || null,
  addEventListener: () => {},
};
global.window = { addEventListener: () => {} };
global.localStorage = { getItem: () => 'test-key', setItem: () => {}, removeItem: () => {} };
global.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

let fetchCallCount = 0;
let fetchQueue = [];
global.fetch = async (url) => {
  fetchCallCount += 1;
  const next = fetchQueue.shift();
  if (!next) throw new Error(`Unexpected fetch call with nothing queued: ${url}`);
  return {
    ok: next.status >= 200 && next.status < 300,
    status: next.status,
    text: async () => JSON.stringify(next.body),
  };
};

const board = require('../public/setup_board.js');

function review(decision, overrides = {}) {
  return {
    id: 1, ticker: 'AAA', source: 'ma_pipeline', setup_key: 'AAA|ma_pipeline|long|95.00|110.00',
    review_type: 'visual', market_structure: 'bullish', location_read: 'good',
    clear_path_to_target: 'yes', lower_tf_confirmation: 'yes', decision, note: 'looks good',
    reviewed_at: '2026-09-01T10:00:00Z',
    ...overrides,
  };
}

function candidate(overrides = {}) {
  return {
    ticker: 'AAA', source: 'ma_pipeline', signal: 'long', rank: 1,
    entry_price: 100, stop: 95, target: 110, risk_reward: 2, current_price: 100,
    entry_distance_pct: 0, entry_proximity_threshold_pct: 1.5,
    confluence_available: true,
    confluence_counts: { favorable: 2, unfavorable: 1, neutral: 4, applicable: 7 },
    confluence_label: 'some confluence',
    bos_confirmed: false, displacement_label: 'WEAK', displacement_score: 20,
    location_label: 'midrange', location_alignment: 'neutral', macro_bias: 'Macro Neutral',
    sweep_confirmed: false, rejection_confirmed: false,
    execution_shadow_ok: true, execution_shadow_reason: 'ok',
    setup_key: 'AAA|ma_pipeline|long|95.00|110.00', current_review: null,
    ...overrides,
  };
}

function queuePayload(candidates) {
  return {
    mechanism: 'stage1_mechanical_then_confluence_count_v1',
    disclaimer: 'Ranking is unvalidated -- based on signal counts, not track record.',
    snapshot_id: 'test-snapshot',
    count: candidates.length,
    candidates,
  };
}

function cardTickers() {
  return (elements.mainContent.innerHTML.match(/class="setup-ticker">([A-Z]+)</g) || [])
    .map((m) => m.replace('class="setup-ticker">', '').replace('<', ''));
}

async function run() {
  // --- A. Approved filtering: approve appears, watch/reject do not ---
  const approved = candidate({ ticker: 'AAA', rank: 1, current_review: review('approve') });
  const watched = candidate({
    ticker: 'BBB', rank: 2, setup_key: 'BBB|ma_pipeline|long|50.00|60.00',
    current_review: review('watch', { ticker: 'BBB', setup_key: 'BBB|ma_pipeline|long|50.00|60.00' }),
  });
  const rejected = candidate({
    ticker: 'CCC', rank: 3, setup_key: 'CCC|ma_pipeline|long|30.00|40.00',
    current_review: review('reject', { ticker: 'CCC', setup_key: 'CCC|ma_pipeline|long|30.00|40.00' }),
  });
  const unreviewed = candidate({
    ticker: 'DDD', rank: 4, setup_key: 'DDD|ma_pipeline|long|20.00|30.00', current_review: null,
  });

  board.state.decision = 'approve';
  fetchQueue = [{ status: 200, body: queuePayload([approved, watched, rejected, unreviewed]) }];
  fetchCallCount = 0;
  await board.loadBoard();

  assert.strictEqual(fetchCallCount, 1, 'G: exactly one fetch call -- no bulk options hydration, no extra network round-trips');
  assert.strictEqual(board.state.board.length, 1, 'A: only the approved candidate should be on the Approved board');
  assert.strictEqual(board.state.board[0].ticker, 'AAA');
  assert.ok(elements.mainContent.innerHTML.includes('AAA'), 'A: approved candidate renders');
  assert.ok(!elements.mainContent.innerHTML.includes('BBB'), 'A: watch candidate must not appear on Approved board');
  assert.ok(!elements.mainContent.innerHTML.includes('CCC'), 'A: rejected candidate must not appear on Approved board');
  assert.ok(!elements.mainContent.innerHTML.includes('DDD'), 'A: unreviewed candidate must not appear on Approved board');
  assert.ok(elements.mainContent.innerHTML.includes('Entry'), 'entry/stop/target metrics render');
  assert.ok(elements.mainContent.innerHTML.includes('Current Price'), 'current price renders');

  // --- B. Watch filtering: watch appears, approve/reject do not ---
  board.state.decision = 'watch';
  fetchQueue = [{ status: 200, body: queuePayload([approved, watched, rejected, unreviewed]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 1, 'B: only the watch candidate should be on the Watch board');
  assert.strictEqual(board.state.board[0].ticker, 'BBB');
  assert.ok(elements.mainContent.innerHTML.includes('BBB'));
  assert.ok(!elements.mainContent.innerHTML.includes('AAA'), 'B: approved candidate must not appear on Watch board');
  assert.ok(!elements.mainContent.innerHTML.includes('CCC'), 'B: rejected candidate must not appear on Watch board');
  // Watch card must preserve why it's on watch: the review note and structured fields.
  assert.ok(elements.mainContent.innerHTML.includes('looks good'), 'watch card preserves the review note');
  assert.ok(elements.mainContent.innerHTML.includes('Bullish'), 'watch card preserves market_structure read');

  // --- C. setup_key isolation: old approve does not follow a new setup_key ---
  const aaaNewSetup = candidate({ ticker: 'AAA', setup_key: 'AAA|ma_pipeline|long|80.00|130.00', current_review: null });
  board.state.decision = 'approve';
  fetchQueue = [{ status: 200, body: queuePayload([aaaNewSetup]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 0, 'C: a materially new setup_key must not inherit the old approve decision');
  assert.ok(!elements.mainContent.innerHTML.includes('setup-ticker'), 'C: no card rendered for the now-unreviewed generation');

  // --- D. Review edits move tickers between boards (simulated as what a
  // reload after a Review Queue edit would show -- these boards always
  // fetch fresh, never trust client-cached state.board across a fetch). ---
  const clhWatch = candidate({
    ticker: 'CLH', rank: 1, setup_key: 'CLH|ma_pipeline|long|307.94|328.04',
    current_review: review('watch', { ticker: 'CLH', setup_key: 'CLH|ma_pipeline|long|307.94|328.04' }),
  });
  board.state.decision = 'approve';
  fetchQueue = [{ status: 200, body: queuePayload([clhWatch]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 0, 'D: CLH on Watch must not appear on Approved yet');

  const clhApproved = candidate({
    ticker: 'CLH', rank: 1, setup_key: 'CLH|ma_pipeline|long|307.94|328.04',
    current_review: review('approve', { ticker: 'CLH', setup_key: 'CLH|ma_pipeline|long|307.94|328.04', id: 8 }),
  });
  fetchQueue = [{ status: 200, body: queuePayload([clhApproved]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 1, 'D: after watch->approve, CLH must appear on Approved');
  assert.strictEqual(board.state.board[0].ticker, 'CLH');

  board.state.decision = 'watch';
  fetchQueue = [{ status: 200, body: queuePayload([clhApproved]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 0, 'D: after watch->approve, CLH must disappear from Watch');

  // reject removes it from both
  const clhRejected = candidate({
    ticker: 'CLH', rank: 1, setup_key: 'CLH|ma_pipeline|long|307.94|328.04',
    current_review: review('reject', { ticker: 'CLH', setup_key: 'CLH|ma_pipeline|long|307.94|328.04', id: 9 }),
  });
  board.state.decision = 'approve';
  fetchQueue = [{ status: 200, body: queuePayload([clhRejected]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 0, 'D: rejected CLH must not appear on Approved');
  board.state.decision = 'watch';
  fetchQueue = [{ status: 200, body: queuePayload([clhRejected]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 0, 'D: rejected CLH must not appear on Watch');

  // --- E. Reload persistence: hydrates from backend current_review only,
  // never from any client-tracked session state (there IS no session
  // counter in this module -- board.state.board is fully replaced by
  // filterAndOrder(state.queue, ...) on every loadBoard() call). ---
  board.state.decision = 'approve';
  fetchQueue = [{ status: 200, body: queuePayload([approved]) }];
  await board.loadBoard();
  assert.strictEqual(board.state.board.length, 1, 'E: fresh load reflects exactly what the server returned');
  assert.strictEqual(board.state.board[0].ticker, 'AAA');

  // --- F. No duplicate reviewed cards: render() fully replaces
  // mainContent.innerHTML on every loadBoard() call rather than appending
  // -- loading the same board twice in a row must still show exactly one
  // card per candidate, not an accumulating list. ---
  fetchQueue = [{ status: 200, body: queuePayload([approved]) }];
  await board.loadBoard();
  fetchQueue = [{ status: 200, body: queuePayload([approved]) }];
  await board.loadBoard();
  const tickers = cardTickers();
  assert.deepStrictEqual(tickers, ['AAA'], 'F: reloading the same board twice must not duplicate cards');

  // --- H. Invalidated display: long below stop, short above stop ---
  const longInvalidated = candidate({ ticker: 'LNG', signal: 'long', entry_price: 100, stop: 95, current_price: 94 });
  const shortInvalidated = candidate({ ticker: 'SHT', signal: 'short', entry_price: 100, stop: 105, current_price: 106 });
  const longWaiting = candidate({ ticker: 'LWT', signal: 'long', entry_price: 100, stop: 95, current_price: 98 });
  const longExtended = candidate({
    ticker: 'LEX', signal: 'long', entry_price: 100, stop: 95, current_price: 105,
    entry_distance_pct: 5, entry_proximity_threshold_pct: 1.5,
  });
  const shortExtended = candidate({
    ticker: 'SEX', signal: 'short', entry_price: 100, stop: 105, current_price: 95,
    entry_distance_pct: 5, entry_proximity_threshold_pct: 1.5,
  });

  assert.strictEqual(board.computeDisplayState(longInvalidated), 'INVALIDATED', 'H: long price <= stop is INVALIDATED');
  assert.strictEqual(board.computeDisplayState(shortInvalidated), 'INVALIDATED', 'H: short price >= stop is INVALIDATED');
  assert.strictEqual(board.computeDisplayState(longWaiting), 'WAITING_FOR_ENTRY', 'long price between stop and entry is WAITING_FOR_ENTRY');
  assert.strictEqual(board.computeDisplayState(longExtended), 'EXTENDED', 'long price materially beyond entry (existing proximity threshold) is EXTENDED');
  assert.strictEqual(board.computeDisplayState(shortExtended), 'EXTENDED', 'short price materially beyond entry is EXTENDED');
  // Missing data must never invent a state -- falls back to WAITING_FOR_ENTRY.
  assert.strictEqual(board.computeDisplayState(candidate({ current_price: null })), 'WAITING_FOR_ENTRY', 'missing current_price falls back to WAITING_FOR_ENTRY, not a guess');

  // Rendered invalidated/extended notices show the required copy.
  board.state.decision = 'approve';
  fetchQueue = [{
    status: 200,
    body: queuePayload([
      candidate({ ticker: 'LNG', signal: 'long', entry_price: 100, stop: 95, current_price: 94, current_review: review('approve', { ticker: 'LNG', setup_key: candidate({ ticker: 'LNG' }).setup_key }) }),
    ]),
  }];
  await board.loadBoard();
  assert.ok(elements.mainContent.innerHTML.includes('Setup invalidated'), 'H: invalidated card shows "Setup invalidated"');

  fetchQueue = [{
    status: 200,
    body: queuePayload([
      candidate({
        ticker: 'LEX', signal: 'long', entry_price: 100, stop: 95, current_price: 105,
        entry_distance_pct: 5, entry_proximity_threshold_pct: 1.5,
        current_review: review('approve', { ticker: 'LEX' }),
      }),
    ]),
  }];
  await board.loadBoard();
  assert.ok(elements.mainContent.innerHTML.includes('Extended') && elements.mainContent.innerHTML.includes('do not chase'), 'H: extended card shows "Extended -- do not chase"');

  // --- I. Execution trigger display: informational only, explicit
  // "Not monitored yet" status, never implies live monitoring. ---
  board.state.decision = 'approve';
  const withTrigger = candidate({
    ticker: 'TRG', setup_key: 'TRG|ma_pipeline|long|300.00|340.00',
    current_review: review('approve', {
      ticker: 'TRG', setup_key: 'TRG|ma_pipeline|long|300.00|340.00',
      trigger_timeframe: '30m', trigger_rule: 'close_above', trigger_level: 318.25,
      trigger_reason: 'Reclaim of prior range high',
    }),
  });
  fetchQueue = [{ status: 200, body: queuePayload([withTrigger]) }];
  await board.loadBoard();
  assert.ok(elements.mainContent.innerHTML.includes('Execution Trigger'), 'I: trigger block renders when trigger_level/trigger_rule are set');
  assert.ok(elements.mainContent.innerHTML.includes('30m'), 'I: trigger timeframe renders');
  assert.ok(elements.mainContent.innerHTML.includes('close above'), 'I: trigger rule renders in human-readable form');
  assert.ok(elements.mainContent.innerHTML.includes('318.25'), 'I: trigger level renders');
  assert.ok(elements.mainContent.innerHTML.includes('Reclaim of prior range high'), 'I: trigger_reason renders when present');
  assert.ok(elements.mainContent.innerHTML.includes('Not monitored yet'), 'I: trigger block must explicitly state Kairos is not currently monitoring it');

  // No trigger stored -> no trigger block at all (existing reviews with no trigger).
  const noTrigger = candidate({
    ticker: 'NTG', setup_key: 'NTG|ma_pipeline|long|300.00|340.00',
    current_review: review('approve', { ticker: 'NTG', setup_key: 'NTG|ma_pipeline|long|300.00|340.00' }),
  });
  fetchQueue = [{ status: 200, body: queuePayload([noTrigger]) }];
  await board.loadBoard();
  assert.ok(!elements.mainContent.innerHTML.includes('Execution Trigger'), 'I: no trigger block when no trigger is stored on the review');

  console.log('Setup board v1 tests passed');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
