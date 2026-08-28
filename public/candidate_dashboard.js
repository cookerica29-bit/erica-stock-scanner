(function () {
  const KEY = 'kairos_scanner_api_key';
  const ALERT_KEY = 'kairos_candidate_alerts';
  // Active Plans keeps the exact old key/semantics (default OFF) -- this is
  // the specific, deliberate fix for RYN/LPX/CVNA disappearing from Active
  // Plans when they didn't pass the strict gate; it must not regress.
  const ACTIVE_ACTIONABLE_ONLY_KEY = 'kairos_candidate_actionable_only';
  // Inbox gets its OWN key with the opposite default (ON) -- these are two
  // independent toggle states, not one shared boolean, precisely so
  // changing Inbox's default can never affect Active Plans.
  const INBOX_READY_ONLY_KEY = 'kairos_candidate_inbox_ready_only';
  const NEAR_MISS_KEY = 'kairos_candidate_near_miss_only';
  const API_KEY_DB = 'kairos_candidate_dashboard';
  const API_KEY_STORE = 'settings';
  const ENTER_NOW_MIN_RR = 1.5;
  const ENTER_NOW_MAX_SCAN_AGE_MS = 5 * 60 * 60 * 1000;
  const ACCEPTABLE_CONTRACT_GRADES = ['excellent', 'good', 'fair'];
  const state = {
    loaded: false,
    loading: false,
    view: 'new',
    candidates: [],
    promotions: [],
    planPreviews: [],
    chartReviews: [],
    error: null,
    nearMisses: [],
    nearMissLoading: false,
    nearMissLoaded: false,
    nearMissError: null,
  };
  let knownCandidateIds = new Set();
  let candidateAlertsEnabled = storageGet(localStorage, ALERT_KEY) === 'on';
  // Active Plans: defaults OFF for every new device/session, unchanged from
  // before. Only ever flips on via explicit user click in
  // toggleActionableOnly() while viewing Active Plans.
  let activeActionableOnlyEnabled = storageGet(localStorage, ACTIVE_ACTIONABLE_ONLY_KEY) === 'on';
  // Inbox: defaults ON for every new device/session (ready-only by
  // default, matching the dashboard's own "stay in audit/legacy" intent) --
  // the inverse default of Active Plans. Only flips off via explicit user
  // click in toggleActionableOnly() while viewing Inbox.
  let inboxReadyOnlyEnabled = storageGet(localStorage, INBOX_READY_ONLY_KEY) !== 'off';
  // Same default-OFF discipline as before.
  let nearMissEnabled = storageGet(localStorage, NEAR_MISS_KEY) === 'on';
  let candidateAlertsInitialized = false;
  let apiKeyPanelExpanded = false;
  let cachedApiKey = '';
  let candidateSessionAuthenticated = false;

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmtMoney(value) {
    // Number(null) === 0 and Number('') === 0 -- both finite, so without
    // this guard a genuinely-missing price (e.g. current_price: null when
    // no reliable quote exists) silently renders as "$0.00" instead of
    // "—". That's exactly the kind of fabricated-looking number this
    // dashboard has been fixing all week (see the entry-proximity
    // one-sided-quote fix) -- guard it here too since it's shared by every
    // card, not just the near-miss view that surfaced it.
    if (value === null || value === undefined || value === '') return '—';
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function fmtNumber(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function fmtTime(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  }

  function entryDistanceLabel(preview) {
    if (preview?.entry_distance_pct == null) return '—';
    const pct = fmtNumber(preview.entry_distance_pct, 2);
    const atr = preview.entry_distance_atr == null ? '' : ` · ${fmtNumber(preview.entry_distance_atr, 2)} ATR`;
    // entry_status_label is legacy's four-tier bucket, purely descriptive
    // and deliberately NOT the same threshold as entry_proximity_ok (the
    // actual gate, shown separately as an "Also: ..." pill elsewhere on the
    // card when it fails). Kept as plain muted parenthetical text inside
    // this KV cell -- not a colored badge -- specifically so it never reads
    // as a second verdict next to the real one. The two can disagree (e.g.
    // proximity_ok: true while this reads "Near Entry" not "Tradeable")
    // and that's expected, not a bug -- see candidates_router.py's
    // ENTRY_STATUS_TRADEABLE_MAX_ATR comment for the full explanation.
    // Note: escapeHtml is applied once, by the caller, to this whole
    // return value -- not escaped again in here.
    const status = preview?.entry_status_label ? ` (${preview.entry_status_label})` : '';
    return `${pct}%${atr}${status}`;
  }

  function storageGet(storage, key) {
    try {
      return storage.getItem(key) || '';
    } catch {
      return '';
    }
  }

  function storageSet(storage, key, value) {
    try {
      storage.setItem(key, value);
    } catch {}
  }

  function storageRemove(storage, key) {
    try {
      storage.removeItem(key);
    } catch {}
  }

  function openApiKeyDb() {
    if (!('indexedDB' in window)) return Promise.resolve(null);
    return new Promise(resolve => {
      const request = indexedDB.open(API_KEY_DB, 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore(API_KEY_STORE);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
      request.onblocked = () => resolve(null);
    });
  }

  async function readApiKeyFromDb() {
    const db = await openApiKeyDb();
    if (!db) return '';
    return new Promise(resolve => {
      const tx = db.transaction(API_KEY_STORE, 'readonly');
      const request = tx.objectStore(API_KEY_STORE).get(KEY);
      request.onsuccess = () => resolve(String(request.result || ''));
      request.onerror = () => resolve('');
      tx.oncomplete = () => db.close();
      tx.onerror = () => db.close();
    });
  }

  async function writeApiKeyToDb(value) {
    const db = await openApiKeyDb();
    if (!db) return;
    await new Promise(resolve => {
      const tx = db.transaction(API_KEY_STORE, 'readwrite');
      const store = tx.objectStore(API_KEY_STORE);
      if (value) store.put(value, KEY);
      else store.delete(KEY);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
    });
  }

  function persistApiKey(value) {
    cachedApiKey = value;
    storageSet(localStorage, KEY, value);
    storageSet(sessionStorage, KEY, value);
  }

  async function hydrateApiKey() {
    const existing = apiKey();
    if (existing) return existing;
    const stored = await readApiKeyFromDb();
    if (stored) persistApiKey(stored);
    return stored;
  }

  function clearApiKey() {
    cachedApiKey = '';
    storageRemove(localStorage, KEY);
    storageRemove(sessionStorage, KEY);
  }

  function apiKey() {
    cachedApiKey = cachedApiKey || storageGet(localStorage, KEY) || storageGet(sessionStorage, KEY);
    return cachedApiKey;
  }

  function headers() {
    const key = apiKey();
    return {
      'Content-Type': 'application/json',
      ...(key ? { 'X-API-Key': key } : {}),
    };
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
      const message = payload?.detail || `Request failed (${response.status})`;
      throw new Error(Array.isArray(message) ? message.map(item => item.msg || item.reason || String(item)).join(', ') : String(message));
    }
    return payload;
  }

  function promotionKey(item) {
    return `${String(item.ticker || '').toUpperCase()}|${item.source || ''}`;
  }

  function promotionsByKey() {
    const map = new Map();
    state.promotions.forEach(item => map.set(promotionKey(item), item));
    return map;
  }

  function chartReviewsByKey() {
    const map = new Map();
    state.chartReviews.forEach(item => map.set(promotionKey(item), item));
    return map;
  }

  function planPreviewsByKey() {
    const map = new Map();
    state.planPreviews.forEach(item => map.set(promotionKey(item), item));
    return map;
  }

  function effectivePlanForCandidate(item, promoMap, previewMap) {
    const key = promotionKey(item);
    const promotion = promoMap.get(key);
    const preview = previewMap.get(key);
    if (promotion) {
      return {
        ...preview,
        ...promotion,
        option_contract: preview?.option_contract || promotion.option_contract || null,
      };
    }
    return preview || null;
  }

  function isRegimeAligned(item) {
    const direction = String(item.signal || '').toLowerCase();
    const regime = String(item.daily_regime || '').toLowerCase();
    if (direction === 'long') return regime.includes('long') || regime.includes('bull');
    if (direction === 'short') return regime.includes('short') || regime.includes('bear');
    return false;
  }

  function contractBlockReason(plan) {
    const contract = plan?.option_contract || {};
    if (!contract.available) return contract.reason || contract.execution || 'No clean options contract';
    const execution = String(contract.execution || '').toLowerCase();
    const acceptable = ACCEPTABLE_CONTRACT_GRADES.some(grade => execution.includes(grade));
    return acceptable ? '' : `Contract quality is ${contract.execution || 'unknown'}`;
  }

  function scanFreshnessBlockReason(item) {
    const rawTime = item?.scanned_at || item?.updated_at;
    if (!rawTime) return 'Scan time unavailable';
    const scannedAt = new Date(rawTime);
    if (Number.isNaN(scannedAt.getTime())) return 'Scan time unavailable';
    if (Date.now() - scannedAt.getTime() > ENTER_NOW_MAX_SCAN_AGE_MS) {
      return 'Scan is stale; refresh before entry';
    }
    return '';
  }

  function entryProximityBlockReason(plan) {
    if (!plan) return 'Plan math pending';
    if (plan.entry_proximity_ok) return '';
    return plan.entry_proximity_reason || 'Price is not near entry';
  }

  function executionConfirmationBlockReason(plan) {
    if (!plan) return 'Plan math pending';
    if (plan.execution_shadow_ok === true) return '';
    return plan.execution_shadow_reason || 'Execution confirmation is not ready';
  }

  function routeBlockReason(item, plan) {
    const direction = String(item.signal || '').toLowerCase();
    if (direction === 'short') return 'Shorts are research-only';
    if (direction !== 'long') return 'Unsupported direction';
    if (!isRegimeAligned(item)) return 'Regime is not aligned';
    if (!plan) return 'Plan math pending';
    if (plan.preview_error) return plan.preview_error;
    if (plan.no_valid_target || plan.target == null || plan.risk_reward == null) return 'No valid target';
    if (plan.rr_warning || Number(plan.risk_reward) < ENTER_NOW_MIN_RR) return `R:R is below ${ENTER_NOW_MIN_RR}:1`;
    // Contract quality (spread/liquidity/DTE/delta) is informational only --
    // demoted from a hard gate, same pattern as the AI chart-read demotion.
    // contractBlockReason() is intentionally unused here now; the contract
    // fields in renderPlanPreview() show a best-effort suggested strike
    // instead. See the backend twin, _preview_base_enter_now_ready in
    // candidates_router.py.
    const proximityReason = entryProximityBlockReason(plan);
    if (proximityReason) return proximityReason;
    const executionReason = executionConfirmationBlockReason(plan);
    if (executionReason) return executionReason;
    const freshnessReason = scanFreshnessBlockReason(item);
    if (freshnessReason) return freshnessReason;
    return '';
  }

  function isEnterNowCandidate(item, promoMap, previewMap) {
    return !routeBlockReason(
      item,
      effectivePlanForCandidate(item, promoMap, previewMap),
    );
  }

  function enterNowCandidates() {
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    return state.candidates.filter(item => isEnterNowCandidate(item, promoMap, previewMap));
  }

  function setStatus(message, kind = '') {
    const el = document.getElementById('candidateStatus');
    if (!el) return;
    el.textContent = message;
    el.style.color = kind === 'error' ? 'var(--fail)' : kind === 'ok' ? 'var(--pass)' : 'var(--muted)';
  }

  async function saveCandidateApiKey() {
    const input = document.getElementById('candidateApiKeyInput');
    const value = String(input?.value || '').trim();
    if (value) {
      persistApiKey(value);
      await fetchJson('/api/v1/scanner/session', {
        method: 'POST',
        body: JSON.stringify({ api_key: value }),
      });
      candidateSessionAuthenticated = true;
    } else {
      clearApiKey();
      await fetchJson('/api/v1/scanner/session', { method: 'DELETE' }).catch(() => null);
      candidateSessionAuthenticated = false;
    }
    await writeApiKeyToDb(value);
    state.loaded = false;
    apiKeyPanelExpanded = !value;
    updateApiKeyPanel();
    setStatus(value ? 'Scanner session saved on this device.' : 'Scanner API key cleared.', value ? 'ok' : '');
  }

  function syncInput() {
    const input = document.getElementById('candidateApiKeyInput');
    if (input && !input.value) input.value = apiKey();
    updateApiKeyPanel();
  }

  function updateApiKeyPanel() {
    const band = document.getElementById('candidateApiBand');
    const toggle = document.getElementById('candidateApiToggle');
    if (!band) return;
    const hasKey = Boolean(apiKey()) || candidateSessionAuthenticated;
    const collapsed = hasKey && !apiKeyPanelExpanded;
    band.classList.toggle('collapsed', collapsed);
    if (toggle) toggle.textContent = collapsed ? 'Change Key' : hasKey ? 'Hide Key' : 'Change Key';
  }

  function toggleCandidateApiKeyPanel() {
    apiKeyPanelExpanded = !apiKeyPanelExpanded;
    updateApiKeyPanel();
  }

  async function loadCandidateDashboard(force = false) {
    await hydrateApiKey();
    syncInput();
    if (state.loading || (state.loaded && !force)) {
      render();
      return;
    }
    state.loading = true;
    state.error = null;
    setStatus('Loading candidate inbox...');
    render();
    try {
      const [candidates, promotions, planPreviewsResult, chartReviews] = await Promise.all([
        fetchJson('/api/v1/scanner/candidates'),
        fetchJson('/api/v1/scanner/candidate-promotions'),
        fetchJson('/api/v1/scanner/candidate-plan-previews').catch((error) => ({ __previewError: error.message || String(error) })),
        fetchJson('/api/v1/scanner/candidate-chart-reviews'),
      ]);
      state.candidates = Array.isArray(candidates) ? candidates : [];
      state.promotions = Array.isArray(promotions) ? promotions : [];
      const previewError = planPreviewsResult && planPreviewsResult.__previewError;
      state.planPreviews = Array.isArray(planPreviewsResult) ? planPreviewsResult : [];
      state.chartReviews = Array.isArray(chartReviews) ? chartReviews : [];
      state.loaded = true;
      candidateSessionAuthenticated = true;
      updateApiKeyPanel();
      notifyForNewCandidates(state.candidates);
      const previewStatus = previewError
        ? ` Plan previews unavailable temporarily: ${previewError}`
        : '';
      const cleanCount = enterNowCandidates().length;
      setStatus(`Loaded ${cleanCount} ENTER_NOW cards from ${state.candidates.length} scanned candidates. Raw Medium/High candidates stay in audit/legacy.${previewStatus}`, previewError ? 'error' : 'ok');
    } catch (error) {
      state.error = error.message || String(error);
      setStatus(state.error, 'error');
    } finally {
      state.loading = false;
      render();
    }
    // If "near misses" was left on from a previous session (localStorage),
    // a fresh page load never calls toggleNearMiss() -- without this, the
    // toggle shows on but the list stays empty until the user clicks it
    // off and back on.
    if (nearMissEnabled && !state.nearMissLoaded && !state.nearMissLoading) {
      await loadNearMisses();
    }
  }

  function setCandidateView(view) {
    state.view = view || 'new';
    document.querySelectorAll('.candidate-tab[data-candidate-view]').forEach(button => {
      button.classList.toggle('active', button.dataset.candidateView === state.view);
    });
    render();
  }

  // "Actionable only" applies to Inbox and Active Plans -- the two views
  // where narrowing to ENTER_NOW-eligible candidates is meaningful. It does
  // NOT apply to Dismissed or All, and it never touches renderStats() counts.
  function actionableOnlyApplicable() {
    return state.view === 'new' || state.view === 'active';
  }

  // Two fully independent states, not one shared boolean -- Inbox and
  // Active Plans intentionally have different defaults (see the constants
  // above), and toggling one must never move the other.
  function currentActionableOnlyEnabled() {
    if (state.view === 'new') return inboxReadyOnlyEnabled;
    if (state.view === 'active') return activeActionableOnlyEnabled;
    return false;
  }

  function statusScopedCandidates() {
    if (state.view === 'all') return state.candidates;
    return state.candidates.filter(item => String(item.status || 'new').toLowerCase() === state.view);
  }

  function candidatesForView() {
    const scoped = statusScopedCandidates();
    if (!actionableOnlyApplicable() || !currentActionableOnlyEnabled()) return scoped;
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    return scoped.filter(item => isEnterNowCandidate(item, promoMap, previewMap));
  }

  function toggleActionableOnly() {
    if (state.view === 'new') {
      inboxReadyOnlyEnabled = !inboxReadyOnlyEnabled;
      storageSet(localStorage, INBOX_READY_ONLY_KEY, inboxReadyOnlyEnabled ? 'on' : 'off');
    } else if (state.view === 'active') {
      activeActionableOnlyEnabled = !activeActionableOnlyEnabled;
      storageSet(localStorage, ACTIVE_ACTIONABLE_ONLY_KEY, activeActionableOnlyEnabled ? 'on' : 'off');
    }
    if (currentActionableOnlyEnabled() && nearMissEnabled) {
      // Mutually exclusive views of the same list -- "actionable only" (0
      // gaps) and "near misses" (1-2 gaps) show disjoint candidates by
      // definition, so showing both toggles "on" would be contradictory.
      nearMissEnabled = false;
      storageSet(localStorage, NEAR_MISS_KEY, 'off');
    }
    render();
  }

  // Inbox-only, same as the strict gate this sits alongside. Ranks
  // candidates by how CLOSE they are to passing every gate (1 or 2 failing
  // conditions) instead of an all-or-nothing pass/fail -- see
  // GET /candidate-near-misses (candidates_router._gate_gap_report). Does
  // not change what "Actionable only" computes; purely additive.
  function nearMissApplicable() {
    return state.view === 'new';
  }

  async function toggleNearMiss() {
    nearMissEnabled = !nearMissEnabled;
    storageSet(localStorage, NEAR_MISS_KEY, nearMissEnabled ? 'on' : 'off');
    if (nearMissEnabled && inboxReadyOnlyEnabled) {
      // Near misses is Inbox-only, so the only actionable-only state it
      // could conflict with is Inbox's own.
      inboxReadyOnlyEnabled = false;
      storageSet(localStorage, INBOX_READY_ONLY_KEY, 'off');
    }
    if (nearMissEnabled && !state.nearMissLoaded && !state.nearMissLoading) {
      await loadNearMisses();
    }
    render();
  }

  async function loadNearMisses() {
    state.nearMissLoading = true;
    state.nearMissError = null;
    render();
    try {
      const rows = await fetchJson('/api/v1/scanner/candidate-near-misses?status=new&limit=10');
      state.nearMisses = Array.isArray(rows) ? rows : [];
      state.nearMissLoaded = true;
    } catch (error) {
      state.nearMissError = error.message || String(error);
    } finally {
      state.nearMissLoading = false;
      render();
    }
  }

  function renderNearMissFilterBand() {
    const toggle = document.getElementById('candidateNearMissToggle');
    const info = document.getElementById('candidateNearMissInfo');
    if (!toggle || !info) return;
    const applicable = nearMissApplicable();
    toggle.style.display = applicable ? '' : 'none';
    toggle.classList.toggle('on', nearMissEnabled);
    toggle.setAttribute('aria-pressed', String(nearMissEnabled));
    toggle.textContent = nearMissEnabled ? '✓ Near misses' : 'Show near misses';

    if (!applicable || !nearMissEnabled) {
      info.textContent = '';
      info.classList.remove('near-miss-active');
      return;
    }
    if (state.nearMissLoading) {
      info.textContent = 'Ranking near misses...';
      info.classList.add('near-miss-active');
      return;
    }
    if (state.nearMissError) {
      info.textContent = state.nearMissError;
      info.classList.add('near-miss-active');
      return;
    }
    const tier1 = state.nearMisses.filter(item => item.tier === 1).length;
    const tier2 = state.nearMisses.filter(item => item.tier === 2).length;
    info.textContent = `Showing ${state.nearMisses.length} near-miss${state.nearMisses.length === 1 ? '' : 'es'} (Tier 1: ${tier1}, Tier 2: ${tier2})`;
    info.classList.add('near-miss-active');
  }

  function renderNearMissList(isAutoFallback) {
    const list = document.getElementById('candidateList');
    if (!list) return;
    // Only shown when this near-miss list is standing in for an empty
    // Inbox ready-only view the user never explicitly asked to leave --
    // makes clear this is a fallback, not a silent content swap.
    const fallbackNote = isAutoFallback
      ? '<div class="candidate-empty near-miss-fallback-note">No ready candidates right now (Inbox defaults to ready-only). Showing the closest near-misses instead:</div>'
      : '';
    if (state.nearMissLoading) {
      list.innerHTML = fallbackNote + '<div class="candidate-empty">Ranking near misses...</div>';
      return;
    }
    if (state.nearMissError) {
      list.innerHTML = fallbackNote + `<div class="candidate-empty">${escapeHtml(state.nearMissError)}</div>`;
      return;
    }
    if (!state.nearMisses.length) {
      list.innerHTML = fallbackNote + '<div class="candidate-empty">No candidates are within 1-2 gates of ENTER_NOW right now.</div>';
      return;
    }
    list.innerHTML = fallbackNote + state.nearMisses.map(item => `
      <article class="candidate-card">
        <div class="candidate-card-head">
          <div>
            <div class="candidate-ticker">${escapeHtml(item.ticker)}</div>
            <div class="candidate-meta">${escapeHtml(item.source || 'source unknown')} · scanned ${escapeHtml(fmtTime(item.scanned_at))}</div>
          </div>
          <span class="candidate-near-miss-tier tier-${item.tier}">Tier ${item.tier} · ${item.failing_count} gate${item.failing_count === 1 ? '' : 's'} away</span>
        </div>
        <div class="candidate-kv">
          <div><span>Entry</span><strong>${escapeHtml(fmtMoney(item.entry_price))}</strong></div>
          <div><span>Current</span><strong>${escapeHtml(fmtMoney(item.current_price))}</strong></div>
          <div><span>R:R</span><strong>${escapeHtml(item.risk_reward == null ? '—' : fmtNumber(item.risk_reward, 2))}</strong></div>
          <div><span>Signal</span><strong>${escapeHtml(item.signal || '—')}</strong></div>
        </div>
        <div class="candidate-near-miss-gaps">
          ${(item.gaps || []).map(gap => `<div class="candidate-near-miss-gap">${escapeHtml(gap.detail)}</div>`).join('')}
        </div>
      </article>
    `).join('');
  }

  function renderActionableFilterBand() {
    const toggle = document.getElementById('candidateActionableToggle');
    const info = document.getElementById('candidateFilterInfo');
    if (!toggle || !info) return;
    const applicable = actionableOnlyApplicable();
    const enabled = currentActionableOnlyEnabled();
    toggle.style.display = applicable ? '' : 'none';
    toggle.classList.toggle('on', enabled);
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.textContent = enabled ? '✓ Actionable only' : 'Show actionable only';

    if (!applicable) {
      info.textContent = '';
      info.classList.remove('active-filter');
      return;
    }
    const total = statusScopedCandidates().length;
    if (enabled) {
      const shown = candidatesForView().length;
      info.textContent = `Showing ${shown} of ${total} (actionable only)`;
      info.classList.add('active-filter');
    } else {
      info.textContent = `Showing all ${total}`;
      info.classList.remove('active-filter');
    }
  }

  function renderStats() {
    const el = document.getElementById('candidateStats');
    if (!el) return;
    const counts = state.candidates.reduce((acc, item) => {
      const status = String(item.status || 'new').toLowerCase();
      acc[status] = (acc[status] || 0) + 1;
      return acc;
    }, {});
    const newest = state.candidates
      .map(item => item.scanned_at || item.updated_at)
      .filter(Boolean)
      .sort()
      .at(-1);
    const stats = [
      ['New', counts.new || 0],
      ['Active', counts.active || 0],
      ['Dismissed', counts.dismissed || 0],
      ['Last Scan', newest ? fmtTime(newest) : '—'],
    ];
    el.innerHTML = stats.map(([label, value]) => `
      <div class="candidate-stat">
        <div class="candidate-stat-label">${escapeHtml(label)}</div>
        <div class="candidate-stat-value">${escapeHtml(value)}</div>
      </div>
    `).join('');
  }

  function render() {
    updateCandidateAlertToggle();
    renderStats();
    renderActionableFilterBand();
    renderNearMissFilterBand();
    const list = document.getElementById('candidateList');
    if (!list) return;
    if (!apiKey() && !candidateSessionAuthenticated && !state.loaded && !state.error) {
      list.innerHTML = '<div class="candidate-empty">Paste the scanner API key to review candidates pushed by the local scanner.</div>';
      return;
    }
    if (state.loading) {
      list.innerHTML = '<div class="candidate-empty">Loading candidates...</div>';
      return;
    }
    if (state.error) {
      list.innerHTML = `<div class="candidate-empty">${escapeHtml(state.error)}</div>`;
      return;
    }
    const items = candidatesForView();
    const userWantsNearMiss = nearMissEnabled && nearMissApplicable();
    // Inbox defaults to ready-only; when that leaves nothing to show (a
    // real, expected possibility on quiet days) fall back to the ranked
    // near-miss view automatically instead of a blank "no candidates"
    // state -- but never override an explicit user choice: only kicks in
    // while ready-only is actually active and the user hasn't already
    // opted into (or out of) near misses themselves.
    const autoNearMissFallback = !userWantsNearMiss && state.view === 'new' && inboxReadyOnlyEnabled && items.length === 0;
    if (userWantsNearMiss || autoNearMissFallback) {
      if (!state.nearMissLoaded && !state.nearMissLoading) {
        // Defer so this render() pass finishes first; loadNearMisses()
        // triggers its own render() calls as it progresses.
        setTimeout(loadNearMisses, 0);
      }
      renderNearMissList(autoNearMissFallback);
      return;
    }
    if (!items.length) {
      list.innerHTML = '<div class="candidate-empty">No candidates in this view.</div>';
      return;
    }
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    const reviewMap = chartReviewsByKey();
    list.innerHTML = items.map(item => renderCandidateCard(item, promoMap.get(promotionKey(item)), previewMap.get(promotionKey(item)), reviewMap.get(promotionKey(item)))).join('');
  }

  // confluence_label -> pill tone, reusing the existing pill vocabulary
  // rather than inventing new visual language. "strong"/"conflicted" get
  // the same green/red weight as ready-primary/reason-primary elsewhere on
  // the card; "some"/"limited" stay deliberately quieter since neither is
  // a verdict, just a count.
  const CONFLUENCE_TONE = {
    'strong confluence': 'high',
    'some confluence': '',
    'limited confluence': 'secondary',
    'conflicted': 'bad',
  };

  function renderConfluencePill(plan) {
    const label = plan?.confluence_label;
    if (!label) return '';
    const tone = CONFLUENCE_TONE[label] ?? '';
    const counts = plan.confluence_counts;
    const ratio = counts ? ` · ${counts.favorable}/${counts.applicable}` : '';
    const display = label.replace(/\b\w/g, (c) => c.toUpperCase());
    const breakdown = plan.confluence_signals
      ? Object.entries(plan.confluence_signals)
        .filter(([, v]) => v != null)
        .map(([k, v]) => `${k}: ${v}`)
        .join(', ')
      : '';
    return `<span class="candidate-pill ${escapeHtml(tone)}" title="${escapeHtml(breakdown)}">${escapeHtml(display)}${escapeHtml(ratio)}</span>`;
  }

  function renderCandidateCard(item, promotion, planPreview, chartReview) {
    const status = String(item.status || 'new').toLowerCase();
    const direction = String(item.signal || '').toLowerCase();
    const confidence = String(item.confidence || 'unknown').toLowerCase();
    const regime = String(item.daily_regime || 'unknown').toLowerCase();
    const aligned = regime && direction && regime.includes(direction);
    // Single source of truth for "why isn't this ENTER_NOW-ready", computed
    // once from the SAME effective plan (preview merged under promotion)
    // routeBlockReason() itself uses, then threaded into every place a
    // reason gets displayed (primary badge, plan-preview pills, promotion
    // pills, and the actions row) so they can never disagree with each other.
    const effectivePlan = promotion
      ? { ...planPreview, ...promotion, option_contract: planPreview?.option_contract || promotion.option_contract || null }
      : planPreview;
    const blockReason = routeBlockReason(item, effectivePlan);
    // The positive counterpart to the red reason-primary badge: same
    // "single source of truth" guarantee (routeBlockReason() is the exact
    // function isEnterNowCandidate()/enterNowCandidates() call too -- the
    // same check behind the "Loaded X ENTER_NOW cards" count and the
    // default ready-only Inbox filter), just rendered as an affirmative
    // green badge instead of silence when there's nothing to warn about.
    //
    // READY is purely mechanical (regime/target/R:R/proximity/execution --
    // see routeBlockReason) and always has been; it has no awareness of
    // confluence/location/macro-CHoCH, which all shipped after this badge
    // existed. That's a real, confirmed gap (not a new gate to add) -- a
    // candidate can pass every mechanical gate while confluence_label
    // reads "conflicted" (2+ real unfavorable signals -- the same
    // CONFLICTED_UNFAVORABLE_MIN threshold confluence_label itself already
    // uses, not a new one invented here). When that happens, the badge
    // itself -- not just the separate confluence pill next to it -- needs
    // to say so, since a reader scanning the pill row hits THIS pill
    // first and may never connect it back to a pill three slots over.
    // Deliberately keyed on "conflicted" specifically, not "limited"/
    // "some": real data (2026-08-28) showed "limited confluence" is
    // actually the NORM among mechanically-ready candidates (5 of 7 live
    // that day), so flagging anything short of "strong" would dilute this
    // into noise on most cards; "conflicted" alone stayed a real minority
    // (~1 of 7) and is the one case that actually contradicts a green
    // checkmark. This is presentation only -- routeBlockReason/
    // _promotion_block_reason and every filter/count built on them are
    // completely unchanged; a conflicted-but-ready candidate is still
    // "ready" everywhere that matters, it just doesn't get to LOOK
    // uncomplicated about it anymore.
    const isConflictedReady = !blockReason && effectivePlan?.confluence_label === 'conflicted';
    let reasonBadge;
    if (blockReason) {
      reasonBadge = `<span class="candidate-pill reason-primary" title="${escapeHtml(blockReason)}">${escapeHtml(blockReason)}</span>`;
    } else if (isConflictedReady) {
      const ratio = effectivePlan.confluence_counts
        ? `${effectivePlan.confluence_counts.favorable}/${effectivePlan.confluence_counts.applicable}`
        : '';
      const title = 'Passes every ENTER_NOW gate: regime, target, R:R, entry proximity, execution confirmation. '
        + 'Confluence is conflicted (2+ real unfavorable signals) -- mechanical readiness does not account for this.';
      reasonBadge = `<span class="candidate-pill ready-primary conflicted" title="${escapeHtml(title)}">⚠ Ready — Conflicted${ratio ? ` (${escapeHtml(ratio)})` : ''}</span>`;
    } else {
      reasonBadge = '<span class="candidate-pill ready-primary" title="Passes every ENTER_NOW gate: regime, target, R:R, entry proximity, execution confirmation.">✓ Ready — ENTER_NOW eligible</span>';
    }
    return `
      <article class="candidate-card ${status === 'active' ? 'active-plan' : ''} ${status === 'dismissed' ? 'dismissed' : ''}">
        <div class="candidate-card-head">
          <div>
            <div class="candidate-ticker">${escapeHtml(item.ticker)}</div>
            <div class="candidate-meta">${escapeHtml(item.source || 'source unknown')} · scanned ${escapeHtml(fmtTime(item.scanned_at))}</div>
          </div>
          <span class="candidate-pill ${escapeHtml(direction)}">${escapeHtml(direction || '—')}</span>
        </div>
        <div class="candidate-pill-row">
          ${reasonBadge}
          ${renderConfluencePill(effectivePlan)}
          <span class="candidate-pill ${escapeHtml(confidence)}">${escapeHtml(confidence)}</span>
          <span class="candidate-pill ${aligned ? 'high' : 'medium'}">${aligned ? 'regime aligned' : `regime ${regime || 'unknown'}`}</span>
          <span class="candidate-pill">${escapeHtml(status)}</span>
        </div>
        <div class="candidate-kv">
          <div><span>Entry</span><strong>${escapeHtml(fmtMoney(item.entry_price))}</strong></div>
          <div><span>4H EMA21</span><strong>${escapeHtml(fmtMoney(item.ema21_4h))}</strong></div>
          <div><span>SMA50 Daily</span><strong>${escapeHtml(fmtMoney(item.sma50_daily))}</strong></div>
          <div><span>SMA200 Daily</span><strong>${escapeHtml(fmtMoney(item.sma200_daily))}</strong></div>
        </div>
        ${planPreview ? renderPlanPreview(planPreview, Boolean(promotion), blockReason) : ''}
        ${promotion ? renderPromotion(promotion, blockReason) : ''}
        ${chartReview ? renderChartReview(chartReview) : ''}
        ${renderActions(item, blockReason)}
      </article>
    `;
  }

  function signalRow(label, main, detail = '', tone = 'info') {
    return `
      <div class="candidate-plan-signal ${escapeHtml(tone)}">
        <div class="candidate-plan-signal-label">${escapeHtml(label)}</div>
        <div class="candidate-plan-signal-main">${main}</div>
        <div class="candidate-plan-signal-detail">${detail}</div>
      </div>`;
  }

  function renderTargetClampSignal(obj) {
    if (!obj || !obj.target_clamp_badge) return [];
    if (obj.target_clamped) {
      const rawTarget = obj.raw_target == null ? null : fmtMoney(obj.raw_target);
      const rawRr = obj.raw_risk_reward == null ? null : fmtNumber(obj.raw_risk_reward, 2);
      const detail = rawTarget
        ? `Raw target <s>${escapeHtml(rawTarget)}</s>${rawRr ? ` / raw R:R ${escapeHtml(rawRr)}` : ''}`
        : '';
      return [signalRow(
        'Target',
        `Adjusted - ${escapeHtml(obj.target_clamp_badge)}`,
        detail,
        'applied',
      )];
    }
    const reasonSuffix = obj.target_clamp_reason ? ` (${escapeHtml(obj.target_clamp_reason)})` : '';
    return [signalRow(
      'Target',
      `May be unreachable - ${escapeHtml(obj.target_clamp_badge)}`,
      `Target/R:R above are unadjusted${reasonSuffix}`,
      'warning',
    )];
  }

  function renderStopSourceSignal(obj) {
    if (!obj || obj.stop_source !== 'order_block') return [];
    const rawStop = obj.raw_stop == null ? '' : `Flat ATR ${escapeHtml(fmtMoney(obj.raw_stop))}`;
    return [signalRow('Stop', 'Order-block invalidation', rawStop, 'stop-order-block')];
  }

  function renderDisplacementSignal(obj) {
    if (!obj || !obj.displacement_read) return [];
    const read = String(obj.displacement_read || '').toLowerCase();
    const label = obj.displacement_label ? String(obj.displacement_label).toUpperCase() : '';
    const score = obj.displacement_score == null ? '' : fmtNumber(obj.displacement_score, 1);
    const magnitude = obj.raw_magnitude_score == null
      ? ''
      : `Raw magnitude ${escapeHtml(fmtNumber(obj.raw_magnitude_score, 1))}`;
    const mainParts = [
      read ? read.charAt(0).toUpperCase() + read.slice(1) : 'Displacement',
      label,
      score,
    ].filter(Boolean);
    return [signalRow(
      'Displacement',
      escapeHtml(mainParts.join(' - ')),
      magnitude,
      read === 'adverse' ? 'warning' : 'info',
    )];
  }

  function renderBosSignal(obj) {
    if (!obj || !Object.prototype.hasOwnProperty.call(obj, 'bos_confirmed')) return [];
    if (obj.bos_confirmed) {
      const breakLevel = obj.bos_details?.break_level == null
        ? ''
        : `Break level ${escapeHtml(fmtMoney(obj.bos_details.break_level))}`;
      return [signalRow('BOS', 'Confirmed', breakLevel, 'info')];
    }
    return [signalRow('BOS', 'Not yet confirmed', '', 'info')];
  }

  // macro_conflict/choch_conflict have no "favorable" state by design (see
  // confluence_summary.py's module docstring) -- they only ever flag a
  // conflict, never confirm alignment. So unlike BOS/displacement above,
  // there is nothing informative to show when clean; this row is silent
  // unless one of the two actually fires, same silent-when-not-applicable
  // pattern as target-clamp/stop-source. Warning tone (not info) is
  // deliberate -- legacy treated this signal seriously enough to warrant
  // the same visual weight as a refused target clamp or adverse displacement.
  function renderMacroChochSignal(obj) {
    if (!obj || (!obj.macro_conflict && !obj.choch_conflict)) return [];
    const parts = [];
    if (obj.macro_conflict && obj.macro_bias) {
      parts.push(`Bias conflict — ${escapeHtml(obj.macro_bias)}`);
    } else if (obj.macro_conflict) {
      parts.push('Bias conflict');
    }
    if (obj.choch_conflict) {
      const level = obj.choch_details?.level == null ? '' : ` at ${escapeHtml(fmtMoney(obj.choch_details.level))}`;
      const dir = obj.choch_details?.direction ? `${escapeHtml(obj.choch_details.direction)} ` : '';
      parts.push(`CHoCH conflict — ${dir}break${level}`);
    }
    const detail = obj.choch_conflict && obj.choch_details?.reason ? escapeHtml(obj.choch_details.reason) : '';
    return [signalRow('Macro/CHoCH', parts.join(' · '), detail, 'warning')];
  }

  // sweep/rejection have no "unfavorable" state either -- they only ever
  // confirm FOR a direction. Silent when not confirmed, same reasoning as
  // macro/CHoCH above but the opposite tone: these are confirmations, not
  // warnings, so they stay at BOS's neutral 'info' weight per design.
  function renderSweepSignal(obj) {
    if (!obj || !obj.sweep_confirmed) return [];
    const level = obj.sweep_details?.level == null ? '' : `Swept level ${escapeHtml(fmtMoney(obj.sweep_details.level))}`;
    return [signalRow('Sweep', 'Confirmed', level, 'info')];
  }

  function renderRejectionSignal(obj) {
    if (!obj || !obj.rejection_confirmed) return [];
    const d = obj.rejection_details || {};
    const detailParts = [];
    if (d.condition) detailParts.push(escapeHtml(d.condition));
    if (d.wick_body_ratio != null) detailParts.push(`wick/body ${escapeHtml(fmtNumber(d.wick_body_ratio, 2))}`);
    return [signalRow('Rejection', 'Confirmed', detailParts.join(' · '), 'info')];
  }

  // Muted/informational by design, even when location_alignment is
  // "unfavorable" -- location is deliberately the least gate-worthy signal
  // here despite being legacy's harshest gate, so it never escalates to
  // warning tone. Silent when there's no valid swing range to read at all
  // (location_percentile == null), same as confluence_summary's own
  // null-handling for this field.
  function renderLocationSignal(obj) {
    if (!obj || obj.location_percentile == null) return [];
    const label = obj.location_label ? String(obj.location_label) : 'location';
    const main = `${label.charAt(0).toUpperCase()}${label.slice(1)} (${fmtNumber(obj.location_percentile, 1)}%)`;
    const alignment = obj.location_alignment;
    const detail = (alignment === 'favorable' || alignment === 'unfavorable')
      ? `${alignment.charAt(0).toUpperCase()}${alignment.slice(1)} for this direction`
      : '';
    return [signalRow('Location', escapeHtml(main), escapeHtml(detail), 'info')];
  }

  function renderPlanSignals(obj) {
    const rows = [
      ...renderTargetClampSignal(obj),
      ...renderStopSourceSignal(obj),
      ...renderDisplacementSignal(obj),
      ...renderBosSignal(obj),
      ...renderMacroChochSignal(obj),
      ...renderSweepSignal(obj),
      ...renderRejectionSignal(obj),
      ...renderLocationSignal(obj),
    ];
    if (!rows.length) return '';
    return `
      <div class="candidate-plan-signals">
        <div class="candidate-plan-signals-title">Plan Signals</div>
        ${rows.join('')}
      </div>`;
  }

  function renderPromotion(promotion, blockReason) {
    const warnings = [];
    const rrReasonText = `R:R is below ${ENTER_NOW_MIN_RR}:1`;
    // "Also:" pills show real secondary data without implying it's the
    // reason the card is excluded -- suppressed only when it IS the reason
    // (then the primary reason-primary badge above already covers it).
    if (promotion.rr_warning && blockReason !== rrReasonText) {
      warnings.push('<span class="candidate-pill warn secondary">Also: R:R warning</span>');
    }
    if (promotion.no_valid_target && blockReason !== 'No valid target') {
      warnings.push('<span class="candidate-pill bad secondary">Also: No valid target</span>');
    }
    if (promotion.position_size == null) warnings.push('<span class="candidate-pill">Options sizing pending</span>');
    return `
      <div class="candidate-pill-row">${warnings.join('')}</div>
      <div class="candidate-kv">
        <div><span>Stop</span><strong>${escapeHtml(fmtMoney(promotion.stop))}</strong></div>
        <div><span>Target</span><strong>${escapeHtml(fmtMoney(promotion.target))}</strong></div>
        <div><span>R:R</span><strong>${escapeHtml(promotion.risk_reward == null ? '—' : fmtNumber(promotion.risk_reward, 2))}</strong></div>
        <div><span>ATR14</span><strong>${escapeHtml(fmtNumber(promotion.atr14, 2))}</strong></div>
      </div>
      ${renderPlanSignals(promotion)}
      <div class="candidate-meta">Promoted ${escapeHtml(fmtTime(promotion.promoted_at))} · target source ${escapeHtml(promotion.target_source || '—')}</div>
    `;
  }

  function renderPlanPreview(preview, hasPromotion, blockReason) {
    const warnings = [];
    // Same "Also:" demotion as renderPromotion() above -- these are real
    // data points, kept visible for context, but never allowed to look like
    // THE reason unless they actually are the operative gate (blockReason).
    const rrReasonText = `R:R is below ${ENTER_NOW_MIN_RR}:1`;
    const proximityReasonText = preview.entry_proximity_reason || 'Price is not near entry';
    const executionReasonText = preview.execution_shadow_reason || 'Execution confirmation is not ready';
    if (preview.preview_error && blockReason !== preview.preview_error) {
      warnings.push(`<span class="candidate-pill bad secondary">Also: ${escapeHtml(preview.preview_error)}</span>`);
    }
    if (preview.rr_warning && blockReason !== rrReasonText) {
      warnings.push('<span class="candidate-pill warn secondary">Also: R:R warning</span>');
    }
    if (preview.no_valid_target && blockReason !== 'No valid target') {
      warnings.push('<span class="candidate-pill bad secondary">Also: No valid target</span>');
    }
    if (!preview.entry_proximity_ok && blockReason !== proximityReasonText) {
      warnings.push(`<span class="candidate-pill bad secondary">Also: ${escapeHtml(proximityReasonText)}</span>`);
    }
    if (preview.execution_shadow_checked) {
      if (preview.execution_shadow_ok) {
        warnings.push('<span class="candidate-pill high">Execution confirmed</span>');
      } else if (blockReason !== executionReasonText) {
        warnings.push('<span class="candidate-pill warn secondary">Also: execution not ready</span>');
      }
    }
    const contract = preview.option_contract || {};
    // Optionability floor, not a quality gate: chain_available === false is
    // the ONLY genuine "not optionable" case. Everything else with a strike
    // is a best-effort suggestion to verify in the broker, not an assertion
    // of tradeable quality -- contract quality no longer hides the strike.
    const hasStrike = Boolean(contract.available && contract.strike != null);
    const noChain = contract.chain_available === false;
    const isClean = contract.clean === true;
    const contractLabel = hasStrike
      ? `${fmtMoney(contract.strike)} ${String(contract.type || '').toUpperCase()}`
      : (noChain ? 'No options chain available' : (contract.reason || contract.execution || 'No contract found near entry'));
    const contractMeta = hasStrike
      ? `${contract.expiry || contract.expiration || '—'} · ${contract.dte ?? '—'} DTE · ${isClean ? (contract.execution || '—') : 'Suggested strike (verify spread/liquidity in broker)'}`
      : '';
    return `
      <div class="candidate-pill-row">
        <span class="candidate-pill">${hasPromotion ? 'Current Plan Math' : 'Plan Preview'}</span>
        ${warnings.join('')}
      </div>
      <div class="candidate-kv">
        <div><span>Stop</span><strong>${escapeHtml(fmtMoney(preview.stop))}</strong></div>
        <div><span>Target</span><strong>${escapeHtml(fmtMoney(preview.target))}</strong></div>
        <div><span>R:R</span><strong>${escapeHtml(preview.risk_reward == null ? '—' : fmtNumber(preview.risk_reward, 2))}</strong></div>
        <div><span>ATR14</span><strong>${escapeHtml(fmtNumber(preview.atr14, 2))}</strong></div>
        <div><span>Current</span><strong>${escapeHtml(fmtMoney(preview.current_price))}</strong></div>
        <div><span>Entry Distance</span><strong>${escapeHtml(entryDistanceLabel(preview))}</strong></div>
        <div><span>Contract</span><strong>${escapeHtml(contractLabel)}</strong></div>
        <div><span>Expiry</span><strong>${escapeHtml(contractMeta)}</strong></div>
      </div>
      ${renderPlanSignals(preview)}
      <div class="candidate-meta">Preview computed ${escapeHtml(fmtTime(preview.computed_at))} · target source ${escapeHtml(preview.target_source || '—')}</div>
      ${preview.execution_shadow_checked ? `<div class="candidate-meta">Execution gate: ${escapeHtml(preview.execution_shadow_reason || '—')}</div>` : ''}
    `;
  }

  function classificationLabel(value) {
    return String(value || 'mixed_unclear').replace(/_/g, ' ');
  }

  function renderChartReview(review) {
    return `
      <div class="candidate-ai-note">
        <div class="candidate-pill-row">
          <span class="candidate-pill">AI Chart Note</span>
          <span class="candidate-pill medium">${escapeHtml(classificationLabel(review.classification))}</span>
        </div>
        <div class="candidate-meta">${escapeHtml(review.caveat || 'Informational only, not a recommendation.')}</div>
        <div class="candidate-ai-rationale">${escapeHtml(review.rationale)}</div>
        <div class="candidate-meta">Reviewed ${escapeHtml(fmtTime(review.reviewed_at))} · ${escapeHtml(review.data_source || 'price data')}</div>
      </div>
    `;
  }

  function renderActions(item, blockReason) {
    const status = String(item.status || 'new').toLowerCase();
    const source = encodeURIComponent(item.source || '');
    const ticker = encodeURIComponent(item.ticker || '');
    const promote = status !== 'active' && !blockReason
      ? `<button onclick="updateCandidateStatus('${ticker}','${source}','active','${status}')">Promote</button>`
      : '';
    const chartRead = `<button class="btn-ghost" onclick="requestChartReview('${ticker}','${source}')">Get AI Chart Read</button>`;
    const dismiss = status !== 'dismissed' && status !== 'active'
      ? `<button class="btn-secondary" onclick="updateCandidateStatus('${ticker}','${source}','dismissed','${status}')">Dismiss</button>`
      : '';
    const restore = status !== 'new'
      ? `<button class="btn-ghost" onclick="updateCandidateStatus('${ticker}','${source}','new','${status}')">Back to Inbox</button>`
      : '';
    const blocked = blockReason
      ? `<span class="candidate-pill bad" title="${escapeHtml(blockReason)}">Not dashboard-ready</span>`
      : '';
    return `<div class="candidate-actions">${promote}${chartRead}${dismiss}${restore}${blocked}</div>`;
  }

  async function updateCandidateStatus(ticker, source, status, currentStatus = '') {
    const decodedTicker = decodeURIComponent(ticker);
    const decodedSource = decodeURIComponent(source);
    if (status === 'dismissed') {
      const ok = window.confirm(`Dismiss ${decodedTicker} from the candidate inbox?`);
      if (!ok) return;
    }
    if (String(currentStatus).toLowerCase() === 'active' && status === 'new') {
      const ok = window.confirm(`Move active plan ${decodedTicker} back to the inbox? The saved promotion math will remain available.`);
      if (!ok) return;
    }
    setStatus(`${status === 'active' ? 'Promoting' : 'Updating'} ${decodedTicker}...`);
    try {
      await fetchJson(`/api/v1/scanner/candidates/${encodeURIComponent(decodedTicker)}?source=${encodeURIComponent(decodedSource)}`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      });
      state.loaded = false;
      await loadCandidateDashboard(true);
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    }
  }

  async function requestChartReview(ticker, source) {
    const decodedTicker = decodeURIComponent(ticker);
    const decodedSource = decodeURIComponent(source);
    setStatus(`Requesting informational AI chart note for ${decodedTicker}...`);
    try {
      const review = await fetchJson(`/api/v1/scanner/candidates/${encodeURIComponent(decodedTicker)}/ai-chart-review?source=${encodeURIComponent(decodedSource)}`, {
        method: 'POST',
      });
      // The POST response already carries the full review row -- splice it into
      // local state and re-render instead of reloading the whole dashboard
      // (candidates + promotions + plan-previews + chart-reviews). That refetch
      // was a full-list reload just to update one card's AI badge: it reset
      // scroll position for every open card and re-ran the plan-preview
      // computation (live option-chain lookups) for all candidates, not just
      // this one.
      const key = promotionKey(review);
      const existingIndex = state.chartReviews.findIndex(item => promotionKey(item) === key);
      if (existingIndex >= 0) {
        state.chartReviews[existingIndex] = review;
      } else {
        state.chartReviews.push(review);
      }
      setStatus(`AI chart note ready for ${decodedTicker}.`, 'ok');
      render();
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    }
  }

  function candidateAlertId(item) {
    return [
      String(item.ticker || '').toUpperCase(),
      item.source || '',
      item.signal || '',
      item.entry_price ?? '',
      item.scanned_at || item.updated_at || '',
    ].join('|');
  }

  function candidateNotificationTitle(item) {
    const direction = String(item.signal || '').toUpperCase();
    return `${String(item.ticker || 'Ticker').toUpperCase()} ${direction || 'Candidate'}`;
  }

  function candidateNotificationBody(item) {
    const confidence = item.confidence ? `${String(item.confidence).toUpperCase()} confidence` : 'New dashboard candidate';
    return `${confidence} · Entry ${fmtMoney(item.entry_price)} · scanned ${fmtTime(item.scanned_at)}`;
  }

  function playCandidateAlertTone() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = 820;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.08, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.45);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.5);
    } catch (_) {}
  }

  function sendCandidateTestNotification() {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    new Notification('Kairos Stock Alerts On', {
      body: 'You will be notified when new inbox candidates appear.',
      tag: 'kairos-stock-alerts-enabled',
    });
  }

  function updateCandidateAlertToggle() {
    const btn = document.getElementById('candidateAlertToggle');
    if (!btn) return;
    const blocked = 'Notification' in window && Notification.permission === 'denied';
    btn.className = 'alert-toggle' + (blocked ? ' blocked' : candidateAlertsEnabled ? ' on' : '');
    btn.textContent = blocked ? 'Alerts Blocked' : candidateAlertsEnabled ? 'Alerts On' : 'Alerts Off';
  }

  async function toggleCandidateAlerts() {
    if (!('Notification' in window)) {
      candidateAlertsEnabled = !candidateAlertsEnabled;
      localStorage.setItem(ALERT_KEY, candidateAlertsEnabled ? 'on' : 'off');
      updateCandidateAlertToggle();
      if (candidateAlertsEnabled) playCandidateAlertTone();
      return;
    }

    if (!candidateAlertsEnabled && Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') {
        candidateAlertsEnabled = false;
        localStorage.setItem(ALERT_KEY, 'off');
        updateCandidateAlertToggle();
        return;
      }
    }

    if (Notification.permission === 'denied') {
      candidateAlertsEnabled = false;
      localStorage.setItem(ALERT_KEY, 'off');
      updateCandidateAlertToggle();
      return;
    }

    candidateAlertsEnabled = !candidateAlertsEnabled;
    localStorage.setItem(ALERT_KEY, candidateAlertsEnabled ? 'on' : 'off');
    if (candidateAlertsEnabled) {
      playCandidateAlertTone();
      sendCandidateTestNotification();
    }
    updateCandidateAlertToggle();
  }

  function notifyForNewCandidates(candidates) {
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    const inbox = candidates.filter(item => (
      String(item.status || 'new').toLowerCase() === 'new'
      && isEnterNowCandidate(item, promoMap, previewMap)
    ));
    const nextIds = new Set(inbox.map(candidateAlertId));
    const newItems = inbox.filter(item => !knownCandidateIds.has(candidateAlertId(item)));

    if (candidateAlertsInitialized && candidateAlertsEnabled && newItems.length) {
      playCandidateAlertTone();
      document.title = `(${newItems.length}) New Kairos Candidate`;
      if ('Notification' in window && Notification.permission === 'granted') {
        newItems.slice(0, 3).forEach(item => {
          new Notification(candidateNotificationTitle(item), {
            body: candidateNotificationBody(item),
            tag: candidateAlertId(item),
          });
        });
      }
    }

    knownCandidateIds = nextIds;
    candidateAlertsInitialized = true;
  }

  window.saveCandidateApiKey = saveCandidateApiKey;
  window.toggleCandidateApiKeyPanel = toggleCandidateApiKeyPanel;
  window.toggleCandidateAlerts = toggleCandidateAlerts;
  window.loadCandidateDashboard = loadCandidateDashboard;
  window.setCandidateView = setCandidateView;
  window.toggleActionableOnly = toggleActionableOnly;
  window.toggleNearMiss = toggleNearMiss;
  window.updateCandidateStatus = updateCandidateStatus;
  window.requestChartReview = requestChartReview;
})();
