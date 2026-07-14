(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosStockAlerts = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  const DIAGNOSTIC_DEFAULTS = {
    enter_now_transitions_detected: 0,
    enter_now_alerts_sent: 0,
    entry_reached_alerts_sent: 0,
    tp1_touches_detected: 0,
    tp1_alerts_sent: 0,
    duplicates_suppressed: 0,
    delivery_failures: 0,
    active_tracked_positions_without_tp1: 0,
    last_delivery_error: '',
  };

  function firstPresent(...values) {
    return values.find(value => value !== undefined && value !== null && value !== '');
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function money(value) {
    const number = finiteNumber(value);
    return number === null ? '—' : `$${number.toFixed(2)}`;
  }

  function strikeText(value) {
    const number = finiteNumber(value);
    if (number === null) return '—';
    return Number.isInteger(number) ? `$${number}` : `$${number.toFixed(2)}`;
  }

  function upper(value) {
    return String(value || '').trim().toUpperCase();
  }

  function setupIdentity(setup = {}) {
    const explicit = firstPresent(setup.setup_id, setup.setupId);
    if (explicit) return String(explicit);
    return [
      upper(setup.ticker),
      upper(setup.direction),
      firstPresent(setup.timeframe, setup.scanner_timeframe, setup.setupTf, ''),
      firstPresent(setup.signal_timestamp, setup.signalTimestamp, setup.candle_timestamp, setup.candleTime, setup.signal_market_date, setup.market_date, ''),
      finiteNumber(firstPresent(setup.entry, setup.entry_price)),
      finiteNumber(firstPresent(setup.sl, setup.stop, setup.stop_price)),
      finiteNumber(firstPresent(setup.tp1, setup.target, setup.target_price)),
      finiteNumber(setup.tp2),
      finiteNumber(setup.tp3),
      finiteNumber(firstPresent(setup.final_target, setup.finalTarget)),
    ].map(value => value === null || value === undefined || value === '' ? 'NA' : String(value)).join('|');
  }

  function signalTimestamp(setup = {}) {
    return firstPresent(setup.signal_timestamp, setup.signalTimestamp, setup.candle_timestamp, setup.candleTime, setup.signal_market_date, setup.market_date, 'no-signal-time');
  }

  function enterNowKey(setup = {}) {
    return `${setupIdentity(setup)}|enter_now|${signalTimestamp(setup)}`;
  }

  function entryReachedKey(setup = {}, touchTimestamp) {
    return `${setupIdentity(setup)}|entry_reached|${touchTimestamp || signalTimestamp(setup)}`;
  }

  function tp1Key(entry = {}, touchTimestamp) {
    return `${firstPresent(entry.setup_id, setupIdentity(entry))}|tp1|${touchTimestamp || entry.first_target_touch_at || entry.tp1_reached_at || 'no-touch-time'}`;
  }

  function isConfirmedBucket(bucket) {
    return upper(bucket).replace(/\s+/g, '_') === 'ENTER_NOW';
  }

  function isNonConfirmedBucket(bucket) {
    const normalized = upper(bucket).replace(/\s+/g, '_');
    return ['BUILDING', 'WATCHLIST', 'WAITING', 'ALMOST_READY', 'SKIP', 'NO_TRADE', ''].includes(normalized);
  }

  function setupBucket(setup = {}, readinessResolver) {
    if (typeof readinessResolver === 'function') {
      const result = readinessResolver(setup) || {};
      if (result.bucket) return result.bucket;
    }
    return firstPresent(setup.progress_bucket, setup.status_bucket, setup.status, setup.scanner_status, setup.trade_eval?.trade_stage, '');
  }

  function isEntryReached(setup = {}, executionStateResolver) {
    if (typeof executionStateResolver === 'function') {
      return executionStateResolver(setup) === 'SETUP_CONFIRMED_ENTRY_REACHED';
    }
    const entry = finiteNumber(firstPresent(setup.entry, setup.entry_price));
    const price = finiteNumber(firstPresent(setup.price, setup.current_price, setup.currentPrice, setup.underlying_price));
    if (entry === null || price === null) return false;
    return Math.abs(price - entry) < 0.000001;
  }

  function contractLines(setup = {}, contractState = {}) {
    if (contractState.state === 'validated_live' && contractState.label) {
      return [`Suggested Contract:`, contractState.label];
    }
    if (contractState.state === 'potential' && contractState.potential) {
      const type = contractState.potential.type ? `${contractState.potential.type.charAt(0)}${contractState.potential.type.slice(1).toLowerCase()}` : '';
      return [
        `Potential Contract:`,
        `${strikeText(contractState.potential.strike)} ${type}`.trim(),
        `Expiration Guidance: ${contractState.expiration?.label || 'Learning'}`,
      ];
    }
    return ['Suggested Contract:', 'Unavailable'];
  }

  function rrText(setup = {}) {
    const value = firstPresent(setup.rr, setup.riskReward, setup.reward_risk);
    if (value !== undefined && value !== null && value !== '') return String(value);
    const entry = finiteNumber(firstPresent(setup.entry, setup.entry_price));
    const stop = finiteNumber(firstPresent(setup.sl, setup.stop, setup.stop_price));
    const target = finiteNumber(firstPresent(setup.tp1, setup.target, setup.target_price));
    if (entry === null || stop === null || target === null || Math.abs(entry - stop) === 0) return '—';
    return (Math.abs(target - entry) / Math.abs(entry - stop)).toFixed(1);
  }

  function enterNowMessage(setup = {}, contractState = {}, entryReached = false) {
    const ticker = upper(setup.ticker) || 'UNKNOWN';
    const direction = upper(setup.direction) || 'N/A';
    const next = entryReached
      ? 'Price is at the Planned Entry. Verify the live option contract before executing.'
      : 'Wait for price to reach the Planned Entry.';
    return [
      `🟢 KAIROS SETUP CONFIRMED — ${ticker}`,
      '',
      `Direction: ${direction}`,
      `Timeframe: ${firstPresent(setup.timeframe, setup.scanner_timeframe, setup.setupTf, '—')}`,
      `Current Price: ${money(firstPresent(setup.price, setup.current_price, setup.currentPrice, setup.underlying_price))}`,
      `Planned Entry: ${money(firstPresent(setup.entry, setup.entry_price))}`,
      `Stop: ${money(firstPresent(setup.sl, setup.stop, setup.stop_price))}`,
      `TP1: ${money(firstPresent(setup.tp1, setup.target, setup.target_price))}`,
      setup.tp2 != null ? `TP2: ${money(setup.tp2)}` : '',
      setup.tp3 != null ? `TP3: ${money(setup.tp3)}` : '',
      `R:R: ${rrText(setup)}`,
      '',
      'Next Step:',
      next,
      '',
      ...contractLines(setup, contractState),
    ].filter(line => line !== '').join('\n');
  }

  function entryReachedMessage(setup = {}, contractState = {}) {
    const ticker = upper(setup.ticker) || 'UNKNOWN';
    const contractNote = contractState.state === 'validated_live'
      ? 'Verify the live option contract before executing.'
      : 'Verify and select the live option contract before executing.';
    return [
      `🎯 KAIROS ENTRY REACHED — ${ticker}`,
      '',
      `Price has reached the Planned Entry at ${money(firstPresent(setup.entry, setup.entry_price))}.`,
      '',
      contractNote,
    ].join('\n');
  }

  function tp1Touched(entry = {}, obs = {}) {
    const target = finiteNumber(firstPresent(entry.target_price, entry.plannedTp1, entry.tp1));
    if (target === null) return false;
    const direction = upper(entry.direction);
    const high = finiteNumber(firstPresent(obs.high, obs.price, entry.current_price));
    const low = finiteNumber(firstPresent(obs.low, obs.price, entry.current_price));
    if (direction === 'SHORT') return low !== null && low <= target;
    return high !== null && high >= target;
  }

  function tp1Message(entry = {}, obs = {}) {
    const ticker = upper(entry.ticker) || 'UNKNOWN';
    return [
      `💰 KAIROS TP1 REACHED — ${ticker}`,
      '',
      `Direction: ${upper(entry.direction) || 'N/A'}`,
      `Entry: ${money(firstPresent(entry.entry_price, entry.entry))}`,
      `TP1: ${money(firstPresent(entry.target_price, entry.plannedTp1, entry.tp1))}`,
      `Current Price: ${money(firstPresent(obs.price, entry.current_price, entry.underlying_price_at_signal))}`,
      '',
      'Trade Plan Reminder:',
      'TP1 reached.',
      'Consider taking partial profits.',
      'Manage the remainder according to your plan.',
      '',
      entry.plannedTp2 != null ? `TP2: ${money(entry.plannedTp2)}` : '',
      entry.plannedTp3 != null ? `TP3: ${money(entry.plannedTp3)}` : '',
    ].filter(line => line !== '').join('\n');
  }

  function detectSetupEvents(setups = [], previousStatuses = {}, sentKeys = new Set(), options = {}) {
    const nextStatuses = {};
    const events = [];
    setups.forEach(setup => {
      const id = setupIdentity(setup);
      const bucket = setupBucket(setup, options.readinessResolver);
      const previous = previousStatuses[id];
      nextStatuses[id] = bucket;
      if (previous !== undefined && !isConfirmedBucket(previous) && isConfirmedBucket(bucket)) {
        const key = enterNowKey(setup);
        if (!sentKeys.has(key)) {
          events.push({ type: 'ENTER_NOW', key, setup, message: enterNowMessage(setup, options.contractStateResolver?.(setup), isEntryReached(setup, options.executionStateResolver)) });
        }
      }
      if (isConfirmedBucket(bucket) && isEntryReached(setup, options.executionStateResolver)) {
        const timestamp = options.entryReachedTimestampResolver?.(setup) || options.now || new Date().toISOString();
        const key = entryReachedKey(setup, timestamp);
        if (!sentKeys.has(key)) {
          events.push({ type: 'ENTRY_REACHED', key, setup, entry_reached_at: timestamp, message: entryReachedMessage(setup, options.contractStateResolver?.(setup)) });
        }
      }
    });
    return { events, nextStatuses };
  }

  function defaultDiagnostics() {
    return { ...DIAGNOSTIC_DEFAULTS };
  }

  return {
    DIAGNOSTIC_DEFAULTS,
    firstPresent,
    finiteNumber,
    money,
    setupIdentity,
    enterNowKey,
    entryReachedKey,
    tp1Key,
    setupBucket,
    isEntryReached,
    enterNowMessage,
    entryReachedMessage,
    tp1Touched,
    tp1Message,
    detectSetupEvents,
    defaultDiagnostics,
  };
});
