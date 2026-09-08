// Kairos Dashboard (Sprint 2, 2026-09 session; density polish in Sprint
// 2.1) -- Erica's live spreadsheet-style trading dashboard.
//
// The backend dashboard-state read model (GET /candidates/dashboard-state,
// see dashboard_state.py / candidates_router.py's Sprint 1 work) is the
// SOLE source of truth for every setup's state, label, next-step text,
// and market classification. This module never recomputes readiness or
// lifecycle logic -- it only renders whatever the server already decided,
// sorts it, colors it, and lets a row expand for more detail. Same
// UMD/state/fetchJson/apiKey convention as public/setup_board.js and
// public/review_queue.js (duplicated here on purpose, matching this
// codebase's existing convention of one self-contained file per page
// rather than a shared bundle).
//
// Sprint 2.1 changed ONLY presentation, nothing backend-facing: full-row
// solid/tinted coloring is now reserved for ENTRY_READY (green) and
// INVALIDATED (restrained red) -- see rowAccentFor. Every other state's
// row uses the plain neutral background; its individual color lives
// entirely in the Current State pill now (colorFor, unchanged). Main-
// table Next Step text is shortened to an operational phrase for named
// states (shortNextStep) while the expanded detail panel keeps the FULL
// backend-provided next_step text (detailFields reads row.next_step
// directly, never the shortened version). Lifecycle (debug) is gated
// behind isDebugMode().
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const KEY = 'kairos_scanner_api_key';
  const API_BASE = '/api/v1/scanner';
  const DASHBOARD_AUTO_REFRESH_MS = 60000; // same cadence as review_queue.js's REVIEW_QUEUE_AUTO_REFRESH_MS

  const state = {
    rows: [],
    disclaimer: '',
    loaded: false,
    authRequired: false,
    lastLoadedAt: null,
  };
  let loadInFlight = false;
  let autoRefreshTimer = null;

  // setup_key -> {rowEl, detailEl, detailTd}. Lives OUTSIDE state
  // deliberately -- these are real DOM node references that must survive
  // across polls so an expanded row stays expanded and unrelated rows are
  // never torn down/recreated (see render() below).
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

  function directionLabel(direction) {
    const d = String(direction || '').trim();
    return d ? d.toUpperCase() : '--';
  }

  function relativeTimeFromIso(iso) {
    if (!iso) return '--';
    const then = new Date(iso);
    if (Number.isNaN(then.getTime())) return '--';
    const seconds = Math.max(0, Math.round((Date.now() - then.getTime()) / 1000));
    if (seconds < 5) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.round(hours / 24);
    return `${days}d ago`;
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

  // ---------------------------------------------------------------------
  // State -> color / sort priority. The backend owns WHICH state a setup
  // is in; this module only owns how that finite, already-known vocabulary
  // (dashboard_state.DASHBOARD_STATES) is colored and ordered on screen --
  // presentation, not strategy.
  //
  // Sprint 2.1 (density polish): STATE_COLOR now drives ONLY the Current
  // State pill (colorFor) -- full-row solid/tinted backgrounds are a
  // SEPARATE, much narrower concept (rowAccentFor, just below), reserved
  // for ENTRY_READY and INVALIDATED only. Requirement #5/#6: state
  // identification for every other state lives entirely in the pill now.
  //
  // Disclosed judgment calls (the task's color rules name 5-6 states
  // explicitly; the other 4-5 states in the 10-state schema need a home):
  //   - LOCATION_REACHED colored yellow alongside CONFIRMED ("progressing")
  //     -- price arrived at the reviewed zone but nothing is confirmed yet,
  //     which reads as "progressing" more than "almost ready".
  //   - WAITING_FOR_PULLBACK colored orange alongside EXECUTION_READY
  //     ("almost ready / waiting on final execution condition") -- a
  //     pullback/retest wait IS the final condition before entry.
  //   - DISCOVERED and CLOSED (and any state with no clean mapping at all,
  //     e.g. STALE, which the backend returns with a null state) are
  //     colored gray ("inactive/reviewed").
  //   - POSITION_OPEN's sort position isn't named in the task's 6-bucket
  //     list at all -- placed second (right after ENTRY_READY, ahead of
  //     the execution/almost-ready tier) on the reasoning that a position
  //     already open is more operationally relevant than one still
  //     approaching entry. See the Sprint 2 report for the full writeup.
  const STATE_COLOR = {
    DISCOVERED: 'gray',
    WATCHING: 'blue',
    LOCATION_REACHED: 'yellow',
    CONFIRMED: 'yellow',
    WAITING_FOR_PULLBACK: 'orange',
    EXECUTION_READY: 'orange',
    ENTRY_READY: 'green',
    INVALIDATED: 'red',
    // TARGET_HIT (2026-09 session): the mirror-image outcome of
    // INVALIDATED, added the same session as manual candidate submission
    // -- a resolved win, colored green same as ENTRY_READY (this codebase's
    // only other green) rather than inventing a new color for a 6-color
    // existing palette (gray/blue/yellow/orange/green/red/purple).
    TARGET_HIT: 'green',
    POSITION_OPEN: 'purple',
    CLOSED: 'gray',
  };
  const DEFAULT_COLOR = 'gray'; // unmapped state (null, e.g. STALE) -- "inactive/reviewed"

  // Pill color -- every state keeps its own distinct identity here,
  // unchanged from Sprint 2.
  function colorFor(row) {
    const s = row && row.state;
    return (s && STATE_COLOR[s]) || DEFAULT_COLOR;
  }

  // Row accent -- Sprint 2.1: full-row coloring reserved for exactly two
  // states. Everything else (including POSITION_OPEN, WAITING_FOR_PULLBACK,
  // LOCATION_REACHED, CONFIRMED, WATCHING, DISCOVERED/CLOSED/unmapped)
  // returns null, meaning "use the normal dark/neutral row background".
  const ROW_ACCENT = {
    ENTRY_READY: 'green',
    INVALIDATED: 'red',
    // TARGET_HIT (2026-09 session): a resolved win is exactly as
    // notable/terminal as a resolved loss -- gets full-row accent too,
    // same reasoning as INVALIDATED's own inclusion here.
    TARGET_HIT: 'green',
  };

  function rowAccentFor(row) {
    const s = row && row.state;
    return (s && ROW_ACCENT[s]) || null;
  }

  // Ascending priority -- 0 sorts first. Ties preserve the server's own
  // order (approved_at DESC) via the stable bucket-concat sort below, same
  // pattern as public/setup_board.js's filterAndOrder/STATE_ORDER.
  const STATE_PRIORITY = {
    ENTRY_READY: 0,
    POSITION_OPEN: 1,
    EXECUTION_READY: 2,
    WAITING_FOR_PULLBACK: 2,
    CONFIRMED: 3,
    LOCATION_REACHED: 3,
    WATCHING: 4,
    DISCOVERED: 5,
    CLOSED: 5,
    INVALIDATED: 6,
    // TARGET_HIT (2026-09 session): grouped with INVALIDATED -- both are
    // terminal, resolved outcomes, no longer operationally actionable.
    TARGET_HIT: 6,
  };
  const DEFAULT_PRIORITY = 5; // unmapped state -- grouped with the other inactive states

  function priorityFor(row) {
    const s = row && row.state;
    return (s && Object.prototype.hasOwnProperty.call(STATE_PRIORITY, s)) ? STATE_PRIORITY[s] : DEFAULT_PRIORITY;
  }

  function sortRows(rows) {
    const buckets = {};
    (rows || []).forEach((row) => {
      const p = priorityFor(row);
      (buckets[p] = buckets[p] || []).push(row);
    });
    return Object.keys(buckets)
      .map(Number)
      .sort((a, b) => a - b)
      .reduce((acc, p) => acc.concat(buckets[p]), []);
  }

  // ---------------------------------------------------------------------
  // Debug mode -- Sprint 2.1 requirement #11: "Lifecycle (debug)" must be
  // hidden from the normal production UI behind an explicit debug/
  // development condition. Two ways to opt in, both explicit acts (never
  // on by accident): a `?debug=1` query param, or a localStorage flag
  // (so it survives a reload without retyping the query string). Wrapped
  // in try/catch -- a debug-mode check must never be the reason the
  // dashboard fails to render.
  function isDebugMode() {
    try {
      const search = (typeof window !== 'undefined' && window.location && window.location.search) || '';
      if (search && typeof URLSearchParams !== 'undefined') {
        const value = new URLSearchParams(search).get('debug');
        if (value === '1' || value === 'true') return true;
      }
    } catch (err) { /* fall through to the localStorage check */ }
    try {
      if (typeof localStorage !== 'undefined' && localStorage.getItem('kairos_dashboard_debug') === '1') return true;
    } catch (err) { /* not debug mode */ }
    return false;
  }

  // Sprint 2.1 requirement #7: short, operational main-table phrasing for
  // the states this task named explicitly. A state with no entry here
  // keeps showing the backend's own full next_step text -- "shorten...
  // where possible", not "invent a phrase for every state".
  const SHORT_NEXT_STEP = {
    ENTRY_READY: 'Review entry',
    POSITION_OPEN: 'Manage position',
    WAITING_FOR_PULLBACK: 'Wait for pullback',
    LOCATION_REACHED: 'Wait for confirmation',
    CONFIRMED: 'Wait for execution',
    WATCHING: 'Keep watching',
    INVALIDATED: 'Setup invalid',
    TARGET_HIT: 'Target hit',
  };

  function shortNextStep(row) {
    const s = row && row.state;
    if (s && SHORT_NEXT_STEP[s]) return SHORT_NEXT_STEP[s];
    return (row && row.next_step) || '--';
  }

  // ---------------------------------------------------------------------
  // Detail panel -- "show available values only": a field is omitted
  // entirely (not shown as a placeholder) when the backend didn't supply
  // it, per Sprint 2's original requirement. Sprint 2.1 requirement #8:
  // the Next Step value here is ALWAYS the full backend-provided
  // row.next_step text, never the shortened main-table phrase -- nothing
  // the backend explained is lost, only the main table's own display is
  // abbreviated (see buildRowCells/shortNextStep). Requirement #11: the
  // raw legacy_state ("Lifecycle (debug)") only appears in debug mode.
  function detailFields(row) {
    row = row || {};
    const fields = [];
    const entryVal = row.planned_entry != null ? row.planned_entry : row.entry;
    if (entryVal != null) fields.push(['Entry / Planned Entry', fmtMoney(entryVal)]);
    const stopVal = row.stop != null ? row.stop : row.exit;
    if (stopVal != null) {
      // Trade-Management Automation: once the stop has moved (breakeven
      // and/or trailed), show BOTH the current and original value -- the
      // main table's own Stop/Exit column already reflects the current
      // one via row.stop/row.exit (see candidates_router.py's own
      // effective_stop), this just makes the "was X, now Y" story visible
      // rather than silently losing the starting point.
      const stopLabel = (row.original_stop != null && row.original_stop !== stopVal)
        ? `${fmtMoney(stopVal)} (originally ${fmtMoney(row.original_stop)})`
        : fmtMoney(stopVal);
      fields.push(['Stop / Exit', stopLabel]);
    }
    const targets = Array.isArray(row.targets) ? row.targets.filter((t) => t != null) : [];
    if (targets.length) {
      fields.push(['TP / Targets', targets.map(fmtMoney).join(', ')]);
    } else if (row.target != null) {
      fields.push(['TP / Targets', fmtMoney(row.target)]);
    }
    if (row.direction) fields.push(['Direction', directionLabel(row.direction)]);
    if (row.state_label) fields.push(['Current State', row.state_label]);
    if (row.next_step) fields.push(['Next Step', row.next_step]);
    if (row.source) fields.push(['Source / Discovery Origin', row.source]);
    if (row.breakeven_set) fields.push(['Breakeven Stop', `Set ${relativeTimeFromIso(row.breakeven_set_at)}`]);
    if (row.partial_profit_suggested) {
      fields.push(['Partial Profit', `Suggested ${relativeTimeFromIso(row.partial_profit_suggested_at)} -- consider reducing size ~50% (informational only)`]);
    }
    if (row.legacy_state && isDebugMode()) fields.push(['Lifecycle (debug)', row.legacy_state]);
    return fields;
  }

  function buildDetailHtml(row) {
    const fields = detailFields(row);
    if (!fields.length) return '<div class="detail-empty">No additional detail available.</div>';
    return `<div class="detail-grid">${fields.map(([label, value]) => `
        <div class="detail-field"><div class="detail-label">${escapeHtml(label)}</div><div class="detail-value">${escapeHtml(String(value))}</div></div>`).join('')}</div>`;
  }

  // Trade-Management Automation (2026-09 session): small badges shown
  // ALONGSIDE the state pill -- see dashboard.html's own .mgmt-badge
  // comment for why these can't be folded into the pill itself.
  function mgmtBadges(row) {
    let html = '';
    if (row && row.breakeven_set) html += '<span class="mgmt-badge breakeven">BE</span>';
    if (row && row.partial_profit_suggested) html += '<span class="mgmt-badge partial">+2R</span>';
    return html;
  }

  function buildRowCells(row) {
    const stateText = row.state_label || (row.legacy_state ? `Unmapped (${row.legacy_state})` : 'Unmapped');
    return `
        <td><div class="setup-cell"><span class="chevron"></span><span>${escapeHtml(row.symbol || '')}</span></div></td>
        <td>${escapeHtml((row.market || '--').toUpperCase())}</td>
        <td><span class="direction-pill">${escapeHtml(directionLabel(row.direction))}</span></td>
        <td><span class="state-pill pill-${colorFor(row)}">${escapeHtml(stateText)}</span>${mgmtBadges(row)}</td>
        <td>${escapeHtml(shortNextStep(row))}</td>
        <td>${escapeHtml(relativeTimeFromIso(row.last_change))}</td>`;
  }

  // ---------------------------------------------------------------------
  // DOM -- row identity/reuse. A setup_key that already has a row keeps
  // its EXACT DOM nodes across a refresh (only their content/class is
  // updated) -- an expanded detail panel, or any other per-row DOM state,
  // survives a poll untouched. Nodes are only created once and only
  // destroyed when the setup genuinely drops out of the response.

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
    detailTd.colSpan = 6;
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

  // Renders `rows` (already in display order) into `tbody`, reusing
  // existing row/detail node pairs by setup_key and appending them in the
  // requested order (appendChild on an already-attached node MOVES it
  // rather than recreating it) -- this is what keeps a background poll
  // from being a disruptive full-table redraw. Exported for tests, which
  // pass a fake tbody explicitly rather than relying on document lookup.
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
  // Networking / auth -- identical convention to public/setup_board.js.

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
      await loadDashboard();
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
        <p>Your scanner API key is needed to load the dashboard. Paste it in the field above and click Connect.</p>
        <p class="auth-required-hint">Already signed in on another tab? This page will pick that up automatically -- or click Retry.</p>
        <button type="button" onclick="loadDashboard()">Retry</button>
      </div>`;
  }

  // Mutually-exclusive sections -- never wipes #dashboardBody's row nodes
  // (and therefore never loses expand state) just because of an
  // auth/loading/empty transition; only actual data changes touch rows.
  function setSection(section, message) {
    const statusLine = document.getElementById('statusLine');
    const emptyState = document.getElementById('emptyState');
    const authPanel = document.getElementById('authPanel');
    const tableWrap = document.getElementById('dashTableWrap');
    statusLine.hidden = !(section === 'loading' || section === 'error');
    if (section === 'loading' || section === 'error') {
      statusLine.innerHTML = escapeHtml(message || 'Loading dashboard…');
      statusLine.classList.toggle('error', section === 'error');
    }
    emptyState.hidden = section !== 'empty';
    authPanel.hidden = section !== 'auth';
    if (section === 'auth') authPanel.innerHTML = renderAuthRequiredPanel();
    tableWrap.hidden = section !== 'table';
  }

  function renderFreshnessBar() {
    const bar = document.getElementById('freshnessBar');
    if (!bar) return;
    if (!state.lastLoadedAt) { bar.style.display = 'none'; return; }
    bar.style.display = 'flex';
    const staleMs = Date.now() - state.lastLoadedAt.getTime();
    const isStale = staleMs > DASHBOARD_AUTO_REFRESH_MS * 3;
    bar.classList.toggle('stale', isStale);
    bar.innerHTML = `
      <span>Dashboard updated ${escapeHtml(relativeTimeLabel(state.lastLoadedAt))}${isStale ? ' &mdash; auto-refresh may be paused' : ''}</span>
      <button type="button" class="freshness-refresh-btn" onclick="loadDashboard()">Refresh now</button>`;
  }

  function renderTable() {
    const disclaimerEl = document.getElementById('dashboardDisclaimer');
    const countEl = document.getElementById('dashboardCount');
    if (state.disclaimer) {
      disclaimerEl.hidden = false;
      disclaimerEl.innerHTML = escapeHtml(state.disclaimer);
    } else {
      disclaimerEl.hidden = true;
    }
    if (!state.rows.length) {
      countEl.hidden = true;
      setSection('empty');
      return;
    }
    countEl.hidden = false;
    countEl.innerHTML = `${state.rows.length} active setup${state.rows.length === 1 ? '' : 's'}`;
    setSection('table');
    render(document.getElementById('dashboardBody'), sortRows(state.rows));
  }

  async function loadDashboard(options = {}) {
    const silent = !!options.silent;
    if (loadInFlight) return;
    loadInFlight = true;
    if (!silent && !state.loaded) setSection('loading');
    try {
      const result = await fetchJson(`${API_BASE}/candidates/dashboard-state`);
      state.rows = result.setups || [];
      state.disclaimer = result.disclaimer || '';
      state.lastLoadedAt = new Date();
      state.loaded = true;
      state.authRequired = false;
      document.getElementById('apiBand').classList.add('hidden');
      renderTable();
      renderFreshnessBar();
    } catch (err) {
      if (silent) return; // a background tick failing must never disrupt the view
      if (err.status === 401) {
        state.authRequired = true;
        document.getElementById('apiBand').classList.remove('hidden');
        setSection('auth');
        const input = document.getElementById('apiKeyInput');
        if (input && input.focus) input.focus();
      } else {
        setSection('error', `Could not load dashboard: ${err.message}`);
      }
    } finally {
      loadInFlight = false;
    }
  }
  window.loadDashboard = loadDashboard;

  function startAutoRefresh() {
    if (autoRefreshTimer !== null) return;
    autoRefreshTimer = setInterval(() => {
      if (state.authRequired) return;
      loadDashboard({ silent: true });
    }, DASHBOARD_AUTO_REFRESH_MS);
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
        loadDashboard();
      }
    });
  }

  if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('DOMContentLoaded', () => {
      try {
        if (apiKey()) document.getElementById('apiBand').classList.add('hidden');
        loadDashboard();
        startAutoRefresh();
      } catch (err) {
        setSection('error', `Could not initialize this page: ${err.message}`);
      }
    });
  }

  // Exposed for tests (tests/dashboard_v1.js) -- real browser usage never
  // touches this return value, it only matters under module.exports.
  return {
    state,
    rowIndex,
    loadDashboard,
    render,
    sortRows,
    colorFor,
    rowAccentFor,
    priorityFor,
    detailFields,
    buildDetailHtml,
    buildRowCells,
    shortNextStep,
    isDebugMode,
    toggleDetail,
    directionLabel,
    relativeTimeFromIso,
    STATE_COLOR,
    ROW_ACCENT,
    STATE_PRIORITY,
    SHORT_NEXT_STEP,
  };
});
