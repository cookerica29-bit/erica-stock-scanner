// Kairos Signals (Dashboard Sprint 3, 2026-09 session) -- a second,
// minimal presentation over the SAME backend dashboard-state read model
// (GET /candidates/dashboard-state) the main Erica dashboard
// (public/dashboard.js) already uses. Built for a user who does not know
// trading terminology: no CHoCH/BOS/displacement/HTF-LTF/premium-
// discount/supply-demand language anywhere, no lifecycle/debug values, no
// AI features. This module computes NO trading decision of its own --
// every state, trade type (CALL/PUT/LONG/SHORT), and price value comes
// directly from the backend; this file only TRANSLATES the backend's
// already-decided state into plain words and renders it.
//
// Deliberately does NOT touch or import public/dashboard.js -- a
// completely separate page/module, per the task's own "do not change...
// the Erica dashboard" instruction. Any shared convention below (UMD
// wrapper, DOM node reuse for row identity, apiKey/fetchJson helpers) is
// duplicated on purpose, matching this codebase's established one-file-
// per-page pattern.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const KEY = 'kairos_scanner_api_key';
  const API_BASE = '/api/v1/scanner';
  const AUTO_REFRESH_MS = 60000; // same cadence as the main dashboard/review queue

  const state = {
    rows: [],
    loaded: false,
    authRequired: false,
    lastLoadedAt: null,
  };
  let loadInFlight = false;
  let autoRefreshTimer = null;

  // setup_key -> {rowEl, detailEl, detailTd} -- same DOM-node-reuse
  // identity pattern as public/dashboard.js, so an expanded row survives
  // a background poll untouched.
  const rowIndex = new Map();

  // ---- pure helpers, no DOM -- exported for tests ----

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
  }

  function fmtMoney(value) {
    return value == null ? '--' : `$${Number(value).toFixed(2)}`;
  }

  function fmtExpiration(value) {
    if (!value) return '--';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  }

  // ---------------------------------------------------------------------
  // Filtering -- this dashboard is stocks only. Uses the backend's own
  // `market` field (dashboard_state.market_for_source, Sprint 2) rather
  // than inventing a second classification -- this is a PRESENTATION
  // filter, not a new trading-decision system.
  function isStock(row) {
    return String((row && row.market) || '').toLowerCase() !== 'forex';
  }

  // ---------------------------------------------------------------------
  // Backend state -> plain, non-technical status. The backend's 10-state
  // vocabulary (dashboard_state.DASHBOARD_STATES) is intentionally
  // collapsed to the 5 words a novice needs, per the task's own list.
  //
  // Disclosed judgment calls:
  //   - DISCOVERED -> "Watching": the closest plain word for "not yet
  //     actionable, nothing wrong with it either" -- "Discovered" itself
  //     is not in the requested vocabulary and reads as jargon.
  //   - LOCATION_REACHED, CONFIRMED, WAITING_FOR_PULLBACK, EXECUTION_READY
  //     all collapse to "Almost Ready" -- every one of these means
  //     "something happened, not fully ready yet", which is exactly what
  //     "Almost Ready" communicates to someone who shouldn't need to
  //     parse the difference between a retest wait and a confirmation
  //     wait.
  //   - CLOSED -> "Closed": not one of the 5 explicitly named statuses,
  //     but "Invalid" would be dishonest for a trade that was simply
  //     closed out (won, lost, or scratched) rather than invalidated
  //     before entry -- a plain, honest word beats forcing it into the
  //     wrong bucket.
  //   - A state the backend could not itself resolve (null/unmapped --
  //     e.g. STALE) has NO safe plain translation and is EXCLUDED from
  //     this page entirely (see filterRows) rather than mislabeled
  //     "Watching", which would incorrectly imply Kairos is still
  //     actively monitoring it.
  const SIMPLE_STATUS = {
    DISCOVERED: 'Watching',
    WATCHING: 'Watching',
    LOCATION_REACHED: 'Almost Ready',
    CONFIRMED: 'Almost Ready',
    WAITING_FOR_PULLBACK: 'Almost Ready',
    EXECUTION_READY: 'Almost Ready',
    ENTRY_READY: 'Entry Ready',
    INVALIDATED: 'Invalid',
    POSITION_OPEN: 'Position Open',
    CLOSED: 'Closed',
  };

  function simpleStatus(row) {
    const s = row && row.state;
    return (s && SIMPLE_STATUS[s]) || null;
  }

  // Status -> badge color (presentation only -- the row itself only ever
  // turns solid green for Entry Ready, see rowAccentFor below).
  const STATUS_COLOR = {
    Watching: 'blue',
    'Almost Ready': 'orange',
    'Entry Ready': 'green',
    'Position Open': 'purple',
    Invalid: 'red',
    Closed: 'gray',
  };

  function statusColor(label) {
    return STATUS_COLOR[label] || 'gray';
  }

  // Plain Next Step language, keyed by the SIMPLE status (not the raw
  // backend state) -- exactly the 5 phrases the task names, plus one
  // honest analog for the one extra status (Closed) this page also shows.
  const SIMPLE_NEXT_STEP = {
    Watching: 'Wait',
    'Almost Ready': 'Get Ready',
    'Entry Ready': 'Review Entry',
    'Position Open': 'Manage Trade',
    Invalid: 'Setup Invalid',
    Closed: 'Trade Closed',
  };

  function simpleNextStep(label) {
    return SIMPLE_NEXT_STEP[label] || '--';
  }

  // Row accent -- Entry Ready is the ONLY full-row solid treatment on
  // this page, per the task's own explicit rule.
  function rowAccentFor(row) {
    return row && row.state === 'ENTRY_READY' ? 'green' : null;
  }

  // ---------------------------------------------------------------------
  // Trade column -- CALL/PUT/LONG/SHORT comes directly from the backend's
  // own trade_type field (dashboard_state.trade_type_for, Sprint 3) --
  // this module does not decide CALL vs PUT vs LONG vs SHORT itself, it
  // only displays whatever the read model already resolved.
  function tradeLabel(row) {
    const t = String((row && row.trade_type) || '').toUpperCase();
    return t || (String((row && row.direction) || '').toLowerCase() === 'short' ? 'SHORT' : 'LONG');
  }

  function isOptionTrade(row) {
    const t = tradeLabel(row);
    return t === 'CALL' || t === 'PUT';
  }

  // ---------------------------------------------------------------------
  // Rows this page will ever show: stocks only, and only rows whose
  // backend state translates to a real, honest plain status (see
  // SIMPLE_STATUS's own comment on why an unmapped state is excluded
  // rather than guessed).
  function filterRows(rows) {
    return (rows || []).filter((row) => isStock(row) && simpleStatus(row) != null);
  }

  // Same priority ordering spirit as the main dashboard: the thing the
  // user should look at first, first. Entry Ready leads; Invalid trails.
  const STATUS_PRIORITY = {
    'Entry Ready': 0,
    'Position Open': 1,
    'Almost Ready': 2,
    Watching: 3,
    Closed: 4,
    Invalid: 5,
  };

  function sortRows(rows) {
    const buckets = {};
    (rows || []).forEach((row) => {
      const p = STATUS_PRIORITY[simpleStatus(row)];
      const key = p == null ? 99 : p;
      (buckets[key] = buckets[key] || []).push(row);
    });
    return Object.keys(buckets)
      .map(Number)
      .sort((a, b) => a - b)
      .reduce((acc, p) => acc.concat(buckets[p]), []);
  }

  // ---------------------------------------------------------------------
  // Detail panel -- ONLY the six values the task names, shown ONLY when
  // the backend actually supplied them. No direction/state/next-step/
  // source/lifecycle text here at all (unlike the main dashboard's richer
  // detail panel) -- this page's detail stays exactly as minimal as its
  // main table.
  function detailFields(row) {
    row = row || {};
    const fields = [];
    const entryVal = row.planned_entry != null ? row.planned_entry : row.entry;
    if (entryVal != null) fields.push(['ENTRY', fmtMoney(entryVal)]);
    const exitVal = row.stop != null ? row.stop : row.exit;
    if (exitVal != null) fields.push(['EXIT', fmtMoney(exitVal)]);
    const targets = Array.isArray(row.targets) ? row.targets.filter((t) => t != null) : [];
    if (targets.length) {
      fields.push(['TP', targets.map(fmtMoney).join(', ')]);
    } else if (row.target != null) {
      fields.push(['TP', fmtMoney(row.target)]);
    }
    // CALL/PUT, STRIKE, EXPIRATION only ever appear together, and only
    // when the backend resolved a real option contract -- never fabricated
    // from a plain LONG/SHORT signal.
    if (isOptionTrade(row)) {
      fields.push(['CALL/PUT', tradeLabel(row)]);
      if (row.option_strike != null) fields.push(['STRIKE', fmtMoney(row.option_strike)]);
      if (row.option_expiration) fields.push(['EXPIRATION', fmtExpiration(row.option_expiration)]);
    }
    return fields;
  }

  function buildDetailHtml(row) {
    const fields = detailFields(row);
    if (!fields.length) return '<div class="detail-empty">No additional detail available.</div>';
    return `<div class="detail-grid">${fields.map(([label, value]) => `
        <div class="detail-field"><div class="detail-label">${escapeHtml(label)}</div><div class="detail-value">${escapeHtml(String(value))}</div></div>`).join('')}</div>`;
  }

  function buildRowCells(row) {
    const label = simpleStatus(row) || 'Watching';
    return `
        <td><div class="ticker-cell"><span class="chevron"></span><span>${escapeHtml(row.symbol || '')}</span></div></td>
        <td><span class="trade-pill">${escapeHtml(tradeLabel(row))}</span></td>
        <td><span class="status-pill status-${statusColor(label)}">${escapeHtml(label)}</span></td>
        <td>${escapeHtml(simpleNextStep(label))}</td>`;
  }

  // ---------------------------------------------------------------------
  // DOM -- identical row-identity/reuse approach to public/dashboard.js
  // (a separate implementation, not a shared import, per this sprint's
  // "do not touch the Erica dashboard" instruction).

  function toggleDetail(setupKey) {
    const entry = rowIndex.get(setupKey);
    if (!entry) return;
    entry.detailEl.hidden = !entry.detailEl.hidden;
    entry.rowEl.classList.toggle('expanded', !entry.detailEl.hidden);
  }
  window.toggleDetail = toggleDetail;

  function ensureRow(row) {
    let entry = rowIndex.get(row.setup_key);
    if (entry) return entry;
    const rowEl = document.createElement('tr');
    rowEl.dataset.setupKey = row.setup_key;
    rowEl.addEventListener('click', () => toggleDetail(row.setup_key));

    const detailEl = document.createElement('tr');
    detailEl.className = 'detail-row';
    detailEl.hidden = true;
    const detailTd = document.createElement('td');
    detailTd.colSpan = 4;
    detailEl.appendChild(detailTd);

    entry = { rowEl, detailEl, detailTd };
    rowIndex.set(row.setup_key, entry);
    return entry;
  }

  function updateRow(entry, row) {
    entry.rowEl.innerHTML = buildRowCells(row);
    const expanded = !entry.detailEl.hidden;
    const accent = rowAccentFor(row);
    entry.rowEl.className = `setup-row${accent ? ` row-accent-${accent}` : ''}${expanded ? ' expanded' : ''}`;
    entry.detailTd.innerHTML = buildDetailHtml(row);
  }

  function render(tbody, rows) {
    if (!tbody) return;
    const seen = new Set();
    (rows || []).forEach((row) => {
      seen.add(row.setup_key);
      const entry = ensureRow(row);
      updateRow(entry, row);
      tbody.appendChild(entry.rowEl);
      tbody.appendChild(entry.detailEl);
    });
    Array.from(rowIndex.keys()).forEach((key) => {
      if (seen.has(key)) return;
      const entry = rowIndex.get(key);
      if (entry.rowEl.parentNode) entry.rowEl.parentNode.removeChild(entry.rowEl);
      if (entry.detailEl.parentNode) entry.detailEl.parentNode.removeChild(entry.detailEl);
      rowIndex.delete(key);
    });
  }

  // ---------------------------------------------------------------------
  // Networking / auth -- same convention as public/dashboard.js.

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

  async function submitApiKey() {
    const input = document.getElementById('apiKeyInput');
    const value = (input.value || '').trim();
    if (!value) return;
    try {
      await fetchJson(`${API_BASE}/session`, { method: 'POST', body: JSON.stringify({ api_key: value }) });
      persistApiKey(value);
      document.getElementById('apiBand').classList.add('hidden');
      await loadSignals();
    } catch (err) {
      setSection('error', `Could not connect: ${err.message}`);
    }
  }
  window.submitApiKey = submitApiKey;

  function renderAuthRequiredPanel() {
    return `
      <div role="alert">
        <div class="auth-required-icon" aria-hidden="true">&#128274;</div>
        <h2>Sign in required</h2>
        <p>Your scanner API key is needed to load this page. Paste it in the field above and click Connect.</p>
        <p class="auth-required-hint">Already signed in on another tab? This page will pick that up automatically -- or click Retry.</p>
        <button type="button" onclick="loadSignals()">Retry</button>
      </div>`;
  }

  function setSection(section, message) {
    const statusLine = document.getElementById('statusLine');
    const emptyState = document.getElementById('emptyState');
    const authPanel = document.getElementById('authPanel');
    const tableWrap = document.getElementById('dashTableWrap');
    statusLine.hidden = !(section === 'loading' || section === 'error');
    if (section === 'loading' || section === 'error') {
      statusLine.innerHTML = escapeHtml(message || 'Loading signals…');
      statusLine.classList.toggle('error', section === 'error');
    }
    emptyState.hidden = section !== 'empty';
    authPanel.hidden = section !== 'auth';
    if (section === 'auth') authPanel.innerHTML = renderAuthRequiredPanel();
    tableWrap.hidden = section !== 'table';
  }

  function relativeTimeLabel(date) {
    if (!date) return 'never';
    const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
    if (seconds < 5) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    return `${hours}h ago`;
  }

  function renderFreshnessBar() {
    const bar = document.getElementById('freshnessBar');
    if (!bar) return;
    if (!state.lastLoadedAt) { bar.style.display = 'none'; return; }
    bar.style.display = 'flex';
    const staleMs = Date.now() - state.lastLoadedAt.getTime();
    const isStale = staleMs > AUTO_REFRESH_MS * 3;
    bar.classList.toggle('stale', isStale);
    bar.innerHTML = `
      <span>Updated ${escapeHtml(relativeTimeLabel(state.lastLoadedAt))}${isStale ? ' &mdash; refresh may be paused' : ''}</span>
      <button type="button" class="freshness-refresh-btn" onclick="loadSignals()">Refresh now</button>`;
  }

  function renderTable() {
    const countEl = document.getElementById('signalsCount');
    const visible = sortRows(filterRows(state.rows));
    if (!visible.length) {
      countEl.hidden = true;
      setSection('empty');
      return;
    }
    countEl.hidden = false;
    countEl.innerHTML = `${visible.length} signal${visible.length === 1 ? '' : 's'}`;
    setSection('table');
    render(document.getElementById('signalsBody'), visible);
  }

  async function loadSignals(options = {}) {
    const silent = !!options.silent;
    if (loadInFlight) return;
    loadInFlight = true;
    if (!silent && !state.loaded) setSection('loading');
    try {
      const result = await fetchJson(`${API_BASE}/candidates/dashboard-state`);
      state.rows = result.setups || [];
      state.lastLoadedAt = new Date();
      state.loaded = true;
      state.authRequired = false;
      document.getElementById('apiBand').classList.add('hidden');
      renderTable();
      renderFreshnessBar();
    } catch (err) {
      if (silent) return;
      if (err.status === 401) {
        state.authRequired = true;
        document.getElementById('apiBand').classList.remove('hidden');
        setSection('auth');
        const input = document.getElementById('apiKeyInput');
        if (input && input.focus) input.focus();
      } else {
        setSection('error', `Could not load signals: ${err.message}`);
      }
    } finally {
      loadInFlight = false;
    }
  }
  window.loadSignals = loadSignals;

  function startAutoRefresh() {
    if (autoRefreshTimer !== null) return;
    autoRefreshTimer = setInterval(() => {
      if (state.authRequired) return;
      loadSignals({ silent: true });
    }, AUTO_REFRESH_MS);
  }
  function stopAutoRefresh() {
    if (autoRefreshTimer !== null) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
  }
  window.stopAutoRefresh = stopAutoRefresh;

  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('storage', (event) => {
      if (event.key === KEY && event.newValue && state.authRequired) {
        loadSignals();
      }
    });
  }

  if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('DOMContentLoaded', () => {
      try {
        if (apiKey()) document.getElementById('apiBand').classList.add('hidden');
        loadSignals();
        startAutoRefresh();
      } catch (err) {
        setSection('error', `Could not initialize this page: ${err.message}`);
      }
    });
  }

  // Exposed for tests (tests/signals_v1.js) -- real browser usage never
  // touches this return value.
  return {
    state,
    rowIndex,
    loadSignals,
    render,
    isStock,
    filterRows,
    sortRows,
    simpleStatus,
    statusColor,
    simpleNextStep,
    rowAccentFor,
    tradeLabel,
    isOptionTrade,
    detailFields,
    buildDetailHtml,
    buildRowCells,
    toggleDetail,
    SIMPLE_STATUS,
    SIMPLE_NEXT_STEP,
    STATUS_PRIORITY,
  };
});
