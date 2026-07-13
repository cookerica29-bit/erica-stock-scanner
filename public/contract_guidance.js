(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosContractGuidance = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  const FALLBACK_STRIKE_BUCKETS = [
    { max: 25, increment: 0.5 },
    { max: 100, increment: 1 },
    { max: 200, increment: 2.5 },
    { max: Infinity, increment: 5 },
  ];

  function upper(value) {
    return String(value || '').trim().toUpperCase();
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function firstPresent(...values) {
    return values.find(value => value !== undefined && value !== null && value !== '');
  }

  function plannedEntry(setup = {}) {
    return finiteNumber(firstPresent(setup.entry, setup.entry_price, setup.planned_entry));
  }

  function direction(setup = {}) {
    const raw = upper(setup.direction);
    if (raw === 'LONG' || raw === 'CALL') return 'LONG';
    if (raw === 'SHORT' || raw === 'PUT') return 'SHORT';
    return '';
  }

  function optionTypeForDirection(value) {
    const raw = upper(value);
    if (raw === 'LONG' || raw === 'CALL') return 'CALL';
    if (raw === 'SHORT' || raw === 'PUT') return 'PUT';
    return '';
  }

  function strikeIncrementForPrice(price) {
    const value = finiteNumber(price);
    if (value === null || value <= 0) return null;
    return FALLBACK_STRIKE_BUCKETS.find(bucket => value < bucket.max || bucket.max === Infinity)?.increment || 5;
  }

  function metadataStrikeIncrement(setup = {}) {
    const sources = [
      setup.best_contract,
      setup.option,
      setup.suggested_contract,
      setup.contract,
      setup.option_metadata,
      setup.option_chain_metadata,
      setup.best_contract && setup.best_contract.diagnostics,
    ].filter(Boolean);
    for (const source of sources) {
      const value = finiteNumber(firstPresent(
        source.strike_increment,
        source.strikeIncrement,
        source.increment,
        source.strike_step,
        source.strikeStep
      ));
      if (value && value > 0) return value;
    }
    return null;
  }

  function strikeIncrement(setup = {}) {
    const entry = plannedEntry(setup);
    const real = metadataStrikeIncrement(setup);
    if (real) return { value: real, source: 'metadata' };
    const fallback = strikeIncrementForPrice(entry);
    return fallback ? { value: fallback, source: 'price_bucket' } : { value: null, source: 'unavailable' };
  }

  function roundToIncrement(value, increment, mode) {
    const raw = finiteNumber(value);
    const step = finiteNumber(increment);
    if (raw === null || !step || step <= 0) return null;
    const scaled = raw / step;
    const rounded = mode === 'ceil'
      ? Math.ceil(scaled - 1e-9) * step
      : Math.floor(scaled + 1e-9) * step;
    return Number(rounded.toFixed(step < 1 ? 2 : step % 1 === 0 ? 0 : 2));
  }

  function potentialStrike(setup = {}) {
    const entry = plannedEntry(setup);
    const dir = direction(setup);
    const increment = strikeIncrement(setup);
    if (entry === null || !dir || !increment.value) return null;
    return roundToIncrement(entry, increment.value, dir === 'SHORT' ? 'ceil' : 'floor');
  }

  function expirationGuidance(readiness = {}) {
    const min = finiteNumber(readiness.suggested_expiration_min_dte);
    const max = finiteNumber(readiness.suggested_expiration_max_dte);
    const cardReady = readiness.card_ready === true;
    if (cardReady && min !== null && max !== null && min > 0 && max >= min) {
      return {
        status: 'ready',
        min_dte: min,
        max_dte: max,
        label: `${Math.round(min)}–${Math.round(max)} DTE`,
      };
    }
    return {
      status: 'learning',
      min_dte: null,
      max_dte: null,
      label: 'Learning',
    };
  }

  function hasValidatedLiveContract(setup = {}, lifecycle = '') {
    const contract = setup.best_contract || {};
    const strike = finiteNumber(contract.strike);
    return lifecycle === 'available'
      && contract.available === true
      && strike !== null
      && Boolean(firstPresent(contract.expiry, contract.expiration))
      && Boolean(optionTypeForDirection(contract.type));
  }

  function liveFailureReason(setup = {}) {
    const contract = setup.best_contract || {};
    const reason = String(contract.reason || '').toLowerCase();
    if (reason.includes('provider') || reason.includes('yahoo')) return 'provider error';
    if (reason.includes('no option expirations') || reason.includes('no listed')) return 'option chain unavailable';
    if (reason.includes('no contracts')) return 'option chain unavailable';
    if (reason.includes('quote') || reason.includes('bid') || reason.includes('ask')) return 'missing quote';
    if (reason.includes('liquidity') || reason.includes('volume') || reason.includes('open interest')) return 'missing liquidity data';
    if (reason.includes('spread') || reason.includes('filters') || reason.includes('suitable')) return 'no contract passed filters';
    if (reason.includes('not evaluated')) return 'not evaluated';
    return reason ? contract.reason : 'contract data unavailable';
  }

  function potentialContract(setup = {}, readiness = {}) {
    const dir = direction(setup);
    const entry = plannedEntry(setup);
    if (!dir) {
      return { available: false, source: 'estimated', reason: 'missing direction' };
    }
    if (entry === null) {
      return { available: false, source: 'estimated', reason: 'missing planned entry' };
    }
    const increment = strikeIncrement(setup);
    const strike = potentialStrike(setup);
    if (strike === null) {
      return { available: false, source: 'estimated', reason: 'strike increment unavailable' };
    }
    const expiry = expirationGuidance(readiness);
    return {
      available: true,
      source: 'estimated',
      contract_guidance_source: 'estimated',
      type: optionTypeForDirection(dir),
      strike,
      planned_entry: entry,
      strike_increment: increment.value,
      strike_increment_source: increment.source,
      expiration_status: expiry.status,
      potential_expiration_min_dte: expiry.min_dte,
      potential_expiration_max_dte: expiry.max_dte,
      expiration_label: expiry.label,
      disclaimer: 'Verify the live option chain before entering.',
    };
  }

  function guidanceState(setup = {}, lifecycle = '', readiness = {}) {
    if (hasValidatedLiveContract(setup, lifecycle)) {
      return { state: 'validated_live', source: 'validated_live', live_failure_reason: null };
    }
    if (lifecycle === 'pending') {
      return { state: 'pending', source: 'pending', live_failure_reason: 'lookup pending' };
    }
    const potential = potentialContract(setup, readiness);
    if (potential.available) {
      return {
        state: 'potential',
        source: 'estimated',
        potential,
        live_failure_reason: liveFailureReason(setup),
      };
    }
    return {
      state: 'unavailable',
      source: 'unavailable',
      reason: potential.reason,
      live_failure_reason: liveFailureReason(setup),
    };
  }

  function diagnostics(rows = [], readinessResolver = () => ({}), lifecycleResolver = () => '') {
    const result = {
      validated_live_contract_count: 0,
      potential_contract_fallback_count: 0,
      expiration_learning_count: 0,
      missing_planned_entry_count: 0,
      missing_direction_count: 0,
      estimated_strike_increment_used: {},
      live_failure_reasons: {},
    };
    rows.forEach(setup => {
      const readiness = readinessResolver(setup) || {};
      const lifecycle = lifecycleResolver(setup) || '';
      const state = guidanceState(setup, lifecycle, readiness);
      if (state.state === 'validated_live') result.validated_live_contract_count += 1;
      if (state.state === 'potential') {
        result.potential_contract_fallback_count += 1;
        if (state.potential.expiration_status === 'learning') result.expiration_learning_count += 1;
        const key = `${state.potential.strike_increment} (${state.potential.strike_increment_source})`;
        result.estimated_strike_increment_used[key] = (result.estimated_strike_increment_used[key] || 0) + 1;
      }
      if (state.reason === 'missing planned entry') result.missing_planned_entry_count += 1;
      if (state.reason === 'missing direction') result.missing_direction_count += 1;
      if (state.live_failure_reason) {
        result.live_failure_reasons[state.live_failure_reason] = (result.live_failure_reasons[state.live_failure_reason] || 0) + 1;
      }
    });
    return result;
  }

  return {
    FALLBACK_STRIKE_BUCKETS,
    plannedEntry,
    direction,
    optionTypeForDirection,
    strikeIncrementForPrice,
    metadataStrikeIncrement,
    strikeIncrement,
    potentialStrike,
    expirationGuidance,
    hasValidatedLiveContract,
    liveFailureReason,
    potentialContract,
    guidanceState,
    diagnostics,
  };
});
