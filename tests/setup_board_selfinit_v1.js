// Approved/Watch Setups self-initialization (2026-09 session, production
// bug fix). Root cause: setup_board.js used to expose an initBoard(decision)
// function that EACH board page had to call from a SEPARATE inline
// <script>initBoard('approve')</script> tag after the external
// setup_board.js tag. If that external script failed to execute for any
// reason, the inline script threw "initBoard is not defined" with nothing
// to catch it, and the page stayed frozen on its static "Loading approved
// setups..." HTML forever -- matching exactly what was reported in
// production. review_queue.js never had this problem: it wires its own
// DOMContentLoaded listener internally, as one self-contained script.
//
// This is a SEPARATE test file (not appended to setup_board_v1.js)
// because it needs a document.addEventListener that actually captures and
// invokes the DOMContentLoaded callback -- setup_board_v1.js's fixture
// deliberately uses a no-op stub (same convention as
// review_queue_auth_v1.js), and Node caches the required module by path,
// so a fresh require() with different document/window fixtures needs its
// own process.
'use strict';

const assert = require('assert');

function makeElement() {
  const classes = new Set();
  return {
    _html: '',
    get innerHTML() { return this._html; },
    set innerHTML(value) { this._html = value; },
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

let domContentLoadedCallback = null;
global.document = {
  getElementById: (id) => elements[id] || null,
  addEventListener: (type, cb) => {
    if (type === 'DOMContentLoaded') domContentLoadedCallback = cb;
  },
  body: { dataset: { decision: 'watch' } },
};
global.window = { addEventListener: () => {} };
global.localStorage = { getItem: () => 'test-key', setItem: () => {}, removeItem: () => {} };
global.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

let fetchCalls = 0;
global.fetch = async (url) => {
  fetchCalls += 1;
  // Watch Lifecycle V1: loadBoard() now also fetches
  // GET /candidates/approved-setup-memory for both boards -- that
  // endpoint returns a plain array (ApprovedSetupMemoryRecordOut[]), not
  // the review-queue's {candidates: [...]} envelope.
  if (String(url).includes('/candidates/approved-setup-memory')) {
    return { ok: true, status: 200, text: async () => JSON.stringify([]) };
  }
  return {
    ok: true,
    status: 200,
    text: async () => JSON.stringify({
      mechanism: 'stage1_mechanical_then_confluence_count_v1',
      disclaimer: 'Ranking is unvalidated -- based on signal counts, not track record.',
      snapshot_id: 'test-snapshot',
      count: 0,
      candidates: [],
    }),
  };
};

const board = require('../public/setup_board.js');

async function run() {
  assert.ok(typeof domContentLoadedCallback === 'function', 'requiring the module must register a DOMContentLoaded listener on its own, with no external caller needed');

  // --- Firing DOMContentLoaded must read the decision from <body
  // data-decision>, matching what approved_setups.html/watch_setups.html
  // actually set, and must call loadBoard() itself -- no separate inline
  // script required. This is the direct fix for the production bug: the
  // page no longer depends on a second script tag succeeding. ---
  domContentLoadedCallback(); // fire-and-forget, exactly like a real DOMContentLoaded dispatch
  await new Promise((resolve) => setTimeout(resolve, 0)); // flush the async loadBoard() it kicked off
  assert.strictEqual(board.state.decision, 'watch', 'self-init must read the decision from document.body.dataset.decision');
  assert.strictEqual(fetchCalls, 2, 'self-init must call loadBoard(), which fetches the review-queue payload and the approved-setup-memory payload exactly once each');
  assert.strictEqual(board.state.loaded, true, 'the board must actually finish loading, not stay stuck');

  console.log('Setup board self-init v1 tests passed');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
