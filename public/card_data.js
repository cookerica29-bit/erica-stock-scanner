(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosCardData = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  const EARNINGS_TIMEOUT_MS = 30000;

  function firstPresent(...values) {
    return values.find(value => value !== undefined && value !== null && value !== '');
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function normalizeDateText(value) {
    if (!value) return '';
    const text = String(value).trim();
    if (/^\d{4}-\d{2}-\d{2}/.test(text)) return text.slice(0, 10);
    const parsed = new Date(text);
    if (Number.isNaN(parsed.getTime())) return text;
    return parsed.toISOString().slice(0, 10);
  }

  function formatShortDate(value) {
    if (!value) return '';
    const text = normalizeDateText(value);
    const date = new Date(`${text}T00:00:00`);
    if (Number.isNaN(date.getTime())) return text;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  function isSameLevel(a, b) {
    const left = finiteNumber(a);
    const right = finiteNumber(b);
    if (left === null || right === null) return false;
    return Math.abs(left - right) < 0.000001;
  }

  function pushDistinct(rows, label, value, key) {
    const numeric = finiteNumber(value);
    if (numeric === null) return;
    if (rows.some(row => isSameLevel(row.value, numeric))) return;
    rows.push({ label, value: numeric, key });
  }

  function targetRows(setup = {}) {
    const rows = [];
    const tp1 = finiteNumber(firstPresent(setup.tp1, setup.target, setup.target_price));
    const tp2 = finiteNumber(firstPresent(setup.tp2, setup.target2, setup.target_2));
    const tp3 = finiteNumber(firstPresent(setup.tp3, setup.target3, setup.target_3));
    const finalTarget = finiteNumber(firstPresent(setup.final_target, setup.finalTarget, setup.finalTargetPrice));
    const hasMultiple = tp2 !== null || tp3 !== null || finalTarget !== null;

    if (!hasMultiple) {
      pushDistinct(rows, 'Target', tp1, 'target');
      return rows;
    }

    pushDistinct(rows, 'TP1', tp1, 'tp1');
    pushDistinct(rows, 'TP2', tp2, 'tp2');
    pushDistinct(rows, 'TP3', tp3, 'tp3');
    if (tp2 === null && tp3 === null) {
      pushDistinct(rows, 'Final Target', finalTarget, 'final-target');
    } else if (finalTarget !== null && !rows.some(row => isSameLevel(row.value, finalTarget))) {
      pushDistinct(rows, 'Final Target', finalTarget, 'final-target');
    }
    return rows;
  }

  function executionPlanRows(setup = {}) {
    return [
      { label: 'Current Price', value: finiteNumber(firstPresent(setup.price, setup.current_price, setup.currentPrice, setup.underlying_price)), key: 'current-price' },
      { label: 'Planned Entry', value: finiteNumber(firstPresent(setup.entry, setup.entry_price)), key: 'planned-entry' },
      { label: 'Stop', value: finiteNumber(firstPresent(setup.sl, setup.stop, setup.stop_price)), key: 'stop' },
      ...targetRows(setup),
    ];
  }

  function targetOrderingWarning(setup = {}) {
    const entry = finiteNumber(firstPresent(setup.entry, setup.entry_price));
    const rows = targetRows(setup).filter(row => /^TP\d$/.test(row.label));
    if (entry === null || rows.length < 2) return null;
    const direction = String(setup.direction || '').toUpperCase();
    const values = rows.map(row => finiteNumber(row.value)).filter(value => value !== null);
    const ordered = direction === 'SHORT'
      ? values.every((value, index) => index === 0 ? entry > value : values[index - 1] >= value)
      : values.every((value, index) => index === 0 ? entry < value : values[index - 1] <= value);
    return ordered ? null : 'target_ordering_inconsistent';
  }

  function normalizeEarnings(setup = {}, now = new Date()) {
    const earnings = setup.earnings && typeof setup.earnings === 'object' ? setup.earnings : {};
    const status = String(firstPresent(earnings.status, setup.earnings_status, setup.earningsStatus, '')).toLowerCase();
    const source = String(firstPresent(earnings.source, setup.earnings_source, setup.earningsSource, '')).toLowerCase();
    const error = firstPresent(earnings.error, earnings.error_message, setup.earnings_error, setup.earningsError);
    const date = normalizeDateText(firstPresent(
      earnings.date,
      earnings.earnings_date,
      earnings.earningsDate,
      earnings.next_earnings_date,
      earnings.nextEarningsDate,
      setup.earnings_date,
      setup.earningsDate,
      setup.next_earnings_date,
      setup.nextEarningsDate
    ));
    const days = finiteNumber(firstPresent(
      earnings.days_until,
      earnings.daysUntil,
      earnings.days_until_earnings,
      earnings.daysUntilEarnings,
      setup.days_until_earnings,
      setup.daysUntilEarnings,
      setup.earnings_days_until,
      setup.earningsDaysUntil
    ));
    const startedAt = firstPresent(earnings.started_at, earnings.startedAt, setup.earnings_started_at, setup.earningsStartedAt);
    const startedMs = startedAt ? new Date(startedAt).getTime() : NaN;
    const ageMs = Number.isFinite(startedMs) ? now.getTime() - startedMs : null;
    const pending = earnings.loading === true || earnings.pending === true || status === 'loading' || status === 'pending' || status === 'refreshing';

    if (date) {
      const className = days !== null && days <= 7 ? 'earnings-high' : days !== null && days <= 14 ? 'earnings-moderate' : 'earnings-neutral';
      const dayText = days === null ? '' : days === 0 ? 'today' : days === 1 ? '1 day away' : `${days} days away`;
      return {
        state: 'date',
        className,
        main: formatShortDate(date),
        sub: dayText,
        date,
        days_until: days,
        source,
      };
    }

    if (pending && (ageMs === null || ageMs <= EARNINGS_TIMEOUT_MS)) {
      return { state: 'loading', className: 'earnings-neutral', main: 'Loading...', sub: '', source };
    }

    if (error || ['error', 'failed', 'failure', 'provider_error'].includes(status)) {
      return { state: 'failed', className: 'earnings-neutral', main: 'Data unavailable', sub: '', source, error: error ? String(error) : '' };
    }

    if (earnings.loaded === false && pending) {
      return { state: 'failed', className: 'earnings-neutral', main: 'Data unavailable', sub: '', source };
    }

    return { state: 'unavailable', className: 'earnings-neutral', main: 'Unavailable', sub: '', source };
  }

  function diagnostics(setups = []) {
    const counts = {
      earnings_requests_started: 0,
      earnings_requests_resolved: 0,
      earnings_unavailable: 0,
      earnings_failed: 0,
      earnings_stuck_beyond_timeout: 0,
      cards_with_tp1: 0,
      cards_with_tp2: 0,
      cards_with_tp3: 0,
      target_ordering_warnings: 0,
      cards_missing_targets: 0,
    };
    const now = new Date();
    setups.forEach(setup => {
      const earnings = setup.earnings || {};
      const normalized = normalizeEarnings(setup, now);
      if (earnings.loading || earnings.pending || ['loading', 'pending', 'refreshing'].includes(String(earnings.status || '').toLowerCase())) {
        counts.earnings_requests_started += 1;
      }
      if (normalized.state === 'date') counts.earnings_requests_resolved += 1;
      if (normalized.state === 'unavailable') counts.earnings_unavailable += 1;
      if (normalized.state === 'failed') counts.earnings_failed += 1;
      if (normalized.state !== 'loading' && (earnings.loading || earnings.pending || ['loading', 'pending', 'refreshing'].includes(String(earnings.status || '').toLowerCase()))) {
        counts.earnings_stuck_beyond_timeout += 1;
      }
      if (finiteNumber(setup.tp1) !== null) counts.cards_with_tp1 += 1;
      if (finiteNumber(setup.tp2) !== null) counts.cards_with_tp2 += 1;
      if (finiteNumber(setup.tp3) !== null) counts.cards_with_tp3 += 1;
      if (targetRows(setup).length === 0) counts.cards_missing_targets += 1;
      if (targetOrderingWarning(setup)) counts.target_ordering_warnings += 1;
    });
    return counts;
  }

  return {
    EARNINGS_TIMEOUT_MS,
    firstPresent,
    finiteNumber,
    normalizeDateText,
    formatShortDate,
    targetRows,
    executionPlanRows,
    targetOrderingWarning,
    normalizeEarnings,
    diagnostics,
  };
});
