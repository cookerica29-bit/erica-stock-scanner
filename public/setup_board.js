// Kairos Approved Setups / Watch Setups boards (2026-09 session).
//
// This is presentation/workflow only, built on top of the Review Queue's
// existing, unchanged data path -- GET /candidates/review-queue already
// returns every candidate's setup_key and current_review (the latest
// candidate_visual_reviews row for that exact setup generation, or null),
// see review_queue.js's own header comment for the full history. Both
// boards call that SAME endpoint and filter client-side by
// current_review.decision -- deliberately no new backend endpoint, no
// duplicated strategy/ranking calculation, and this is what makes the
// setup_key identity boundary (a materially new setup_key starts
// unreviewed, old review stays in history) apply here for free: it's
// already true of the field this reads.
//
// One module drives both pages -- decision ('approve' | 'watch') comes
// from <body data-decision="..."> at init time, set per-page in
// approved_setups.html / watch_setups.html.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const KEY = 'kairos_scanner_api_key';
  const API_BASE = '/api/v1/scanner';

  const state = {
    decision: 'approve', // 'approve' | 'watch' -- set by initBoard()
    queue: [], // full, unfiltered review-queue payload
    board: [], // queue filtered to state.decision (+ state-grouped for approve)
    disclaimer: '',
    loaded: false,
    authRequired: false,
  };
  let loadBoardInFlight = false;

  // ---- pure helpers, no DOM -- exported for tests ----
  function isReviewed(item) {
    return !!(item && item.current_review);
  }

  function decisionFor(item) {
    return isReviewed(item) ? item.current_review.decision : null;
  }

  // Approved-setup display state. Informational only -- never written back
  // to the server, never touches candidate_promotions or ENTER_NOW.
  //
  // INVALIDATED and EXTENDED are the only two states this computes.
  // ACTIONABLE is deliberately NOT implemented: the closest existing single
  // field (entry_proximity_ok) is only one piece of the real ENTER_NOW
  // condition (which also needs execution_shadow_ok, BOS/displacement/sweep
  // confirmation) -- using it alone here would silently approximate ENTER_NOW/
  // Stage D, which this build is explicitly scoped to not do. EXTENDED
  // reuses entry_distance_pct / entry_proximity_threshold_pct -- fields the
  // backend already computes for ENTER_NOW's own proximity gate -- instead
  // of inventing a new "materially extended" threshold.
  function computeDisplayState(item) {
    const direction = String((item && item.signal) || '').toLowerCase();
    const entry = item ? item.entry_price : null;
    const stop = item ? item.stop : null;
    const price = item ? item.current_price : null;
    if (entry == null || stop == null || price == null || (direction !== 'long' && direction !== 'short')) {
      return 'WAITING_FOR_ENTRY';
    }
    if (direction === 'long' && price <= stop) return 'INVALIDATED';
    if (direction === 'short' && price >= stop) return 'INVALIDATED';

    const distPct = item.entry_distance_pct;
    const thresholdPct = item.entry_proximity_threshold_pct;
    if (distPct != null && thresholdPct != null) {
      if (direction === 'long' && price > entry && distPct > thresholdPct) return 'EXTENDED';
      if (direction === 'short' && price < entry && distPct > thresholdPct) return 'EXTENDED';
    }
    return 'WAITING_FOR_ENTRY';
  }

  const STATE_ORDER = ['WAITING_FOR_ENTRY', 'EXTENDED', 'INVALIDATED'];

  // Filters queue to the board's decision, preserving existing rank order.
  // For 'approve', additionally groups by display state (waiting / extended
  // / invalidated) -- a stable partition, so rank order is preserved WITHIN
  // each group, per the task's explicit sorting rule. 'watch' preserves
  // plain rank order with no grouping (display state is an approved-setups
  // concept only).
  function filterAndOrder(queue, decision) {
    const matches = (queue || []).filter((item) => decisionFor(item) === decision);
    if (decision !== 'approve') return matches;
    const groups = { WAITING_FOR_ENTRY: [], EXTENDED: [], INVALIDATED: [] };
    matches.forEach((item) => { groups[computeDisplayState(item)].push(item); });
    return STATE_ORDER.reduce((acc, s) => acc.concat(groups[s]), []);
  }

  function apiKey() {
    return localStorage.getItem(KEY) || sessionStorage.getItem(KEY) || '';
  }

  function persistApiKey(value) {
    localStorage.setItem(KEY, value);
    sessionStorage.setItem(KEY, value);
  }

  function headers() {
    const key = apiKey();
    return { 'Content-Type': 'application/json', ...(key ? { 'X-API-Key': key } : {}) };
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      credentials: 'same-origin',
      headers: { ...headers(), ...(options.headers || {}) },
    });
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch { payload = { detail: text }; }
    if (!response.ok) {
      const message = payload && payload.detail ? payload.detail : `Request failed (${response.status})`;
      const err = new Error(Array.isArray(message) ? message.map(m => m.msg || String(m)).join(', ') : String(message));
      err.status = response.status;
      throw err;
    }
    return payload;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
  }

  function fmtMoney(value) {
    return value == null ? '--' : `$${Number(value).toFixed(2)}`;
  }

  function fmtNumber(value, digits = 2) {
    return value == null ? '--' : Number(value).toFixed(digits);
  }

  function fmtPct(value) {
    return value == null ? '--' : `${Number(value).toFixed(2)}%`;
  }

  async function submitApiKey() {
    const input = document.getElementById('apiKeyInput');
    const value = (input.value || '').trim();
    if (!value) return;
    try {
      await fetchJson(`${API_BASE}/session`, {
        method: 'POST',
        body: JSON.stringify({ api_key: value }),
      });
      persistApiKey(value);
      document.getElementById('apiBand').classList.add('hidden');
      await loadBoard();
    } catch (err) {
      setStatus(`Could not connect: ${err.message}`, true);
    }
  }
  window.submitApiKey = submitApiKey;

  function setStatus(message, isError = false) {
    const main = document.getElementById('mainContent');
    main.innerHTML = `<div class="status-line${isError ? ' error' : ''}">${escapeHtml(message)}</div>`;
  }

  function renderAuthRequiredPanel(boardLabel) {
    return `
      <div class="auth-required-panel" role="alert">
        <div class="auth-required-icon" aria-hidden="true">🔒</div>
        <h2>Sign in required</h2>
        <p>Your scanner API key is needed to load ${escapeHtml(boardLabel)}. Paste it in the field above and click Connect.</p>
        <p class="auth-required-hint">Already signed in on another tab? This page will pick that up automatically -- or click Retry.</p>
        <button type="button" onclick="loadBoard()">Retry</button>
      </div>`;
  }

  async function loadBoard() {
    if (loadBoardInFlight) return;
    loadBoardInFlight = true;
    const boardLabel = state.decision === 'approve' ? 'approved setups' : 'watch setups';
    setStatus(`Loading ${boardLabel}…`);
    try {
      const result = await fetchJson(`${API_BASE}/candidates/review-queue`);
      state.queue = result.candidates || [];
      state.disclaimer = result.disclaimer || '';
      state.board = filterAndOrder(state.queue, state.decision);
      state.loaded = true;
      state.authRequired = false;
      render();
    } catch (err) {
      if (err.status === 401) {
        state.authRequired = true;
        document.getElementById('apiBand').classList.remove('hidden');
        document.getElementById('mainContent').innerHTML = renderAuthRequiredPanel(boardLabel);
        const input = document.getElementById('apiKeyInput');
        if (input && input.focus) input.focus();
      } else {
        setStatus(`Could not load ${boardLabel}: ${err.message}`, true);
      }
    } finally {
      loadBoardInFlight = false;
    }
  }

  // Same cross-tab auto-retry as review_queue.js.
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('storage', (event) => {
      if (event.key === KEY && event.newValue && state.authRequired) {
        loadBoard();
      }
    });
  }

  const MARKET_STRUCTURE_LABELS = { bullish: 'Bullish', bearish: 'Bearish', range: 'Range' };
  const LOCATION_LABELS = { good: 'Good', neutral: 'Neutral', bad: 'Bad' };
  const CLEAR_PATH_LABELS = { yes: 'Yes', no: 'No' };
  const LOWER_TF_LABELS = { yes: 'Yes', not_yet: 'Not yet' };
  const STATE_LABELS = {
    WAITING_FOR_ENTRY: 'Waiting for Entry',
    EXTENDED: 'Extended',
    INVALIDATED: 'Invalidated',
  };

  function stateBadge(item) {
    if (state.decision !== 'approve') return '';
    const s = computeDisplayState(item);
    return `<span class="state-pill state-${s}">${STATE_LABELS[s]}</span>`;
  }

  function stateNotice(item) {
    if (state.decision !== 'approve') return '';
    const s = computeDisplayState(item);
    if (s === 'EXTENDED') return `<div class="state-notice notice-extended">Extended &mdash; do not chase</div>`;
    if (s === 'INVALIDATED') return `<div class="state-notice notice-invalidated">Setup invalidated</div>`;
    return '';
  }

  function editHref(item) {
    return `/review-queue?ticker=${encodeURIComponent(item.ticker)}&source=${encodeURIComponent(item.source || '')}`;
  }

  function renderCard(item) {
    const cr = item.current_review || {};
    const chartUrl = `https://www.tradingview.com/symbols/${encodeURIComponent(item.ticker)}/`;
    return `
      <div class="setup-card">
        <div class="setup-head">
          <div>
            <span class="setup-ticker">${escapeHtml(item.ticker)}</span>
            <span class="direction-pill">${escapeHtml((item.signal || '').toUpperCase())}</span>
            ${stateBadge(item)}
          </div>
          <span class="setup-rank">Rank #${item.rank} &middot; ${escapeHtml(item.source || '')}</span>
        </div>

        <div class="current-price-row">
          <span class="current-price-label">Current Price</span>
          <span class="current-price-value">${fmtMoney(item.current_price)}</span>
        </div>

        <div class="metric-grid primary-metrics">
          <div class="metric"><div class="metric-label">Entry</div><div class="metric-value">${fmtMoney(item.entry_price)}</div></div>
          <div class="metric"><div class="metric-label">Stop Loss</div><div class="metric-value">${fmtMoney(item.stop)}</div></div>
          <div class="metric"><div class="metric-label">Target</div><div class="metric-value">${fmtMoney(item.target)}</div></div>
          <div class="metric"><div class="metric-label">R:R</div><div class="metric-value">${fmtNumber(item.risk_reward)}</div></div>
        </div>
        <div class="metric-grid secondary-metrics">
          <div class="metric"><div class="metric-label">Entry distance</div><div class="metric-value">${fmtPct(item.entry_distance_pct)}</div></div>
        </div>

        ${stateNotice(item)}

        <div class="review-context">
          <div class="review-meta">Reviewed ${escapeHtml(new Date(cr.reviewed_at).toLocaleString())}</div>
          ${cr.note ? `<div class="review-note">${escapeHtml(cr.note)}</div>` : ''}
          <div class="signal-grid">
            <div class="signal-row"><span>Market structure</span><span class="read">${escapeHtml(MARKET_STRUCTURE_LABELS[cr.market_structure] || cr.market_structure || '--')}</span></div>
            <div class="signal-row"><span>Location</span><span class="read">${escapeHtml(LOCATION_LABELS[cr.location_read] || cr.location_read || '--')}</span></div>
            <div class="signal-row"><span>Clear path to target</span><span class="read">${escapeHtml(CLEAR_PATH_LABELS[cr.clear_path_to_target] || cr.clear_path_to_target || '--')}</span></div>
            <div class="signal-row"><span>Lower-TF confirmation</span><span class="read">${escapeHtml(LOWER_TF_LABELS[cr.lower_tf_confirmation] || cr.lower_tf_confirmation || '--')}</span></div>
          </div>
        </div>

        <div class="card-actions">
          <a class="nav-link" href="${chartUrl}" target="_blank" rel="noopener">Open Chart on TradingView &#8599;</a>
          <a class="nav-link" href="${editHref(item)}">Edit Review</a>
        </div>
      </div>`;
  }

  function emptyStateMessage() {
    if (state.decision === 'approve') {
      return `No approved setups yet. Approve a candidate in the <a class="nav-link" href="/review-queue">Review Queue</a> to see it here.`;
    }
    return `No setups on watch yet. Mark a candidate Watch in the <a class="nav-link" href="/review-queue">Review Queue</a> to see it here.`;
  }

  function render() {
    const main = document.getElementById('mainContent');
    if (!state.loaded) return;
    const countLabel = state.decision === 'approve' ? 'approved setup' : 'watch setup';
    const header = `<div class="disclaimer">${escapeHtml(state.disclaimer)}</div>
      <div class="progress-row"><span class="progress-count">${state.board.length} ${countLabel}${state.board.length === 1 ? '' : 's'}</span></div>`;
    if (!state.board.length) {
      main.innerHTML = `${header}<div class="status-line">${emptyStateMessage()}</div>`;
      return;
    }
    main.innerHTML = `${header}<div class="setup-list">${state.board.map(renderCard).join('')}</div>`;
  }

  // Root cause of the "Approved Setups not loading" production bug
  // (2026-09 session): this used to be an exported initBoard(decision)
  // function that EACH board page had to call itself from a SEPARATE
  // inline <script>initBoard('approve')</script> tag after the external
  // setup_board.js tag. review_queue.js never has this problem -- it
  // wires its own DOMContentLoaded listener internally, as part of the
  // SAME script, so there is nothing else that has to load or run for it
  // to work. The two-script split here meant that if the external
  // setup_board.js tag failed to execute for ANY reason (slow/flaky
  // network, a blocked/failed script load, a browser extension) -- as
  // opposed to a normal successful load, which this session's own
  // testing repeatedly reproduced working -- the second, inline script
  // would throw "initBoard is not defined" with nothing to catch it, and
  // the page would stay frozen on its static "Loading approved setups…"
  // HTML forever: no error shown, no retry offered, matching exactly
  // what was reported. Also: initBoard(decision) took the decision as an
  // explicit argument the caller had to pass correctly -- despite this
  // file's own header comment claiming decision "comes from <body
  // data-decision=...>", the code never actually read that attribute.
  // Fixed the same way review_queue.js already does it: one script,
  // self-initializing, reading <body data-decision> itself, wrapped so
  // an unexpected init-time exception is surfaced as visible text
  // instead of leaving the page stuck on its loading state.
  if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('DOMContentLoaded', () => {
      try {
        const decision = (document.body && document.body.dataset && document.body.dataset.decision) || 'approve';
        state.decision = decision;
        if (apiKey()) {
          document.getElementById('apiBand').classList.add('hidden');
        }
        loadBoard();
      } catch (err) {
        setStatus(`Could not initialize this page: ${err.message}`, true);
      }
    });
  }
  window.loadBoard = loadBoard;

  // Exposed for tests (tests/setup_board_v1.js) -- real browser usage never
  // touches this return value, it only matters under module.exports.
  return {
    state,
    loadBoard,
    renderAuthRequiredPanel,
    submitApiKey,
    isReviewed,
    decisionFor,
    computeDisplayState,
    filterAndOrder,
  };
});
