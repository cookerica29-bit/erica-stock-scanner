(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosOpportunityPriceMix = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  const BANDS = {
    BUDGET: { label: 'Budget', min: 0, max: 250 },
    MID_RANGE: { label: 'Mid-Range', min: 251, max: 600 },
    PREMIUM: { label: 'Premium', min: 601, max: Infinity },
    UNAVAILABLE: { label: 'Unavailable', min: null, max: null },
  };

  const STAGES = [
    'universe',
    'market_data_success',
    'technical_setup_found',
    'grade_eligible',
    'enter_now',
    'developing',
    'skip',
    'optionable',
    'contract_data_available',
    'suggested_contract_found',
    'displayed',
  ];

  const QUALITY_PROPOSALS = [
    { ticker: 'F', sector: 'Consumer Discretionary / Auto', bucket: 'Budget-likely', rationale: 'Large-cap, highly liquid, usually tight option chain, low nominal strikes.' },
    { ticker: 'CCL', sector: 'Consumer Discretionary / Travel', bucket: 'Budget-likely', rationale: 'Liquid cruise name with active options and typically low-to-moderate one-contract costs.' },
    { ticker: 'JBLU', sector: 'Industrials / Airlines', bucket: 'Budget-likely', rationale: 'Low-priced airline with listed options; requires chain-depth confirmation before production inclusion.' },
    { ticker: 'SOFI', sector: 'Financials / Fintech', bucket: 'Budget-likely', rationale: 'Liquid retail-traded options; spreads must be monitored to avoid weak contracts.' },
    { ticker: 'PLTR', sector: 'Technology / Software', bucket: 'Mid-Range-likely', rationale: 'Very liquid options and frequent technical movement, often produces sub-premium contract costs.' },
    { ticker: 'HOOD', sector: 'Financials / Brokerage', bucket: 'Mid-Range-likely', rationale: 'Active options market with practical strikes, but volatility can lift premiums.' },
    { ticker: 'AFRM', sector: 'Financials / Consumer Credit', bucket: 'Mid-Range-likely', rationale: 'Liquid options and strong movement; higher IV needs spread/quality safeguards.' },
    { ticker: 'RIVN', sector: 'Consumer Discretionary / EV', bucket: 'Budget-likely', rationale: 'Lower-priced, optionable, but should be screened for spread and open interest.' },
    { ticker: 'UBER', sector: 'Technology / Mobility', bucket: 'Mid-Range-likely', rationale: 'Deep chain and broad participation, often below mega-cap premium costs.' },
    { ticker: 'SNAP', sector: 'Communication Services', bucket: 'Budget-likely', rationale: 'Low nominal strikes and active options; quality filter must reject thin expirations.' },
    { ticker: 'PINS', sector: 'Communication Services', bucket: 'Mid-Range-likely', rationale: 'Optionable growth name with moderate contract costs and sector diversification.' },
    { ticker: 'GM', sector: 'Consumer Discretionary / Auto', bucket: 'Mid-Range-likely', rationale: 'Liquid, optionable large-cap auto exposure; often cheaper than mega-cap tech.' },
    { ticker: 'T', sector: 'Communication Services / Telecom', bucket: 'Budget-likely', rationale: 'Liquid low-priced stock with listed options; movement frequency must be validated.' },
    { ticker: 'WBD', sector: 'Communication Services / Media', bucket: 'Budget-likely', rationale: 'Low nominal strikes and listed options; chain quality varies by expiration.' },
    { ticker: 'KMI', sector: 'Energy / Midstream', bucket: 'Budget-likely', rationale: 'Lower-priced energy exposure with options; scanner compatibility needs setup-frequency review.' },
    { ticker: 'CSCO', sector: 'Technology / Networking', bucket: 'Mid-Range-likely', rationale: 'Liquid options and lower premiums than mega-cap software/semis.' },
  ];

  const QUALITY_SAFEGUARDS = [
    'Prefer 21+ DTE unless the live chain and setup timing justify shorter duration.',
    'Avoid far-OTM contracts; keep strike near planned entry and directionally ATM/slightly ITM.',
    'Require bid/ask spread percentage at or below the existing live-contract threshold.',
    'Require meaningful open interest or same-day volume; reject empty/liquidity-less strikes.',
    'Do not use a cheap weekly contract to force a Budget classification.',
    'Keep selected contract tied to Planned Entry, not current price drift.',
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

  function contractPremium(setup = {}) {
    const sources = [
      setup.best_contract,
      setup.option,
      setup.suggested_contract,
      setup.contract,
    ].filter(Boolean);
    for (const source of sources) {
      const premium = finiteNumber(firstPresent(
        source.ask,
        source.mark,
        source.mid,
        source.lastPrice,
        source.last_price,
        source.last
      ));
      if (premium !== null && premium > 0) return premium;
    }
    return null;
  }

  function contractCost(setup = {}) {
    const premium = contractPremium(setup);
    return premium === null ? null : premium * 100;
  }

  function costBand(cost) {
    const value = finiteNumber(cost);
    if (value === null || value < 0) return 'UNAVAILABLE';
    if (value <= BANDS.BUDGET.max) return 'BUDGET';
    if (value <= BANDS.MID_RANGE.max) return 'MID_RANGE';
    return 'PREMIUM';
  }

  function setupBucket(setup = {}, progressResolver = null) {
    if (typeof progressResolver === 'function') {
      const resolved = progressResolver(setup) || {};
      if (resolved.bucket) return upper(resolved.bucket);
    }
    return upper(firstPresent(setup.progress_bucket, setup.readiness_bucket, setup.simple_status, setup.status));
  }

  function gradeEligible(setup = {}) {
    const grade = upper(firstPresent(setup.setupGrade, setup.setup_grade, setup.grade, setup.scanner_status, setup.trade_eval?.trade_stage));
    return grade.includes('A') || grade.includes('B');
  }

  function hasDirection(setup = {}) {
    return ['LONG', 'SHORT'].includes(upper(setup.direction));
  }

  function hasMarketData(setup = {}) {
    return finiteNumber(firstPresent(setup.price, setup.current_price, setup.underlying_price)) !== null;
  }

  function hasTechnicalSetup(setup = {}) {
    return hasDirection(setup) && finiteNumber(firstPresent(setup.entry, setup.entry_price, setup.planned_entry)) !== null;
  }

  function isOptionable(setup = {}) {
    const contract = setup.best_contract || {};
    const source = upper(contract.source);
    const reason = upper(contract.reason);
    if (reason.includes('NO OPTION EXPIRATIONS') || reason.includes('NO LISTED')) return false;
    if (source === 'NOT_EVALUATED') return Boolean(setup.option || setup.potential_strike);
    return Boolean(setup.option || contract.available || source === 'OPTION_CHAIN' || setup.potential_strike || setup.contract_guidance_source);
  }

  function contractDataAvailable(setup = {}) {
    const contract = setup.best_contract || {};
    const source = upper(contract.source);
    if (source === 'NOT_EVALUATED') return false;
    return Boolean(contract.available || source === 'OPTION_CHAIN' || (!setup.best_contract && setup.option));
  }

  function suggestedContractFound(setup = {}) {
    const contract = setup.best_contract || {};
    if (contract.available === true) return true;
    const source = upper(contract.source);
    if (source === 'NOT_EVALUATED' || source === 'UNAVAILABLE') return false;
    return Boolean(!setup.best_contract && setup.option?.strike && setup.option?.expiry);
  }

  function stageFlags(setup = {}, progressResolver = null) {
    const bucket = setupBucket(setup, progressResolver);
    return {
      universe: true,
      market_data_success: hasMarketData(setup),
      technical_setup_found: hasTechnicalSetup(setup),
      grade_eligible: gradeEligible(setup),
      enter_now: bucket === 'ENTER_NOW',
      developing: bucket === 'ALMOST_READY' || bucket === 'WAITING',
      skip: bucket === 'SKIP',
      optionable: isOptionable(setup),
      contract_data_available: contractDataAvailable(setup),
      suggested_contract_found: suggestedContractFound(setup),
      displayed: true,
    };
  }

  function blankCounts() {
    return Object.fromEntries(Object.keys(BANDS).map(key => [key, 0]));
  }

  function incrementStage(matrix, stage, band) {
    if (!matrix[stage]) matrix[stage] = blankCounts();
    matrix[stage][band] = (matrix[stage][band] || 0) + 1;
  }

  function displaySection(setup = {}, progressResolver = null) {
    const bucket = setupBucket(setup, progressResolver);
    if (bucket === 'ENTER_NOW') return 'Enter Now';
    if (bucket === 'ALMOST_READY' || bucket === 'WAITING') return 'Developing';
    if (bucket === 'SKIP') return 'Skip';
    return 'Other';
  }

  function rowDiagnostic(setup = {}, options = {}) {
    const cost = contractCost(setup);
    const band = costBand(cost);
    const flags = stageFlags(setup, options.progressResolver);
    const contract = setup.best_contract || {};
    const source = contract.available === true
      ? 'validated_live'
      : setup.contract_guidance_source === 'estimated'
      ? 'potential_estimate'
      : upper(contract.source || 'unknown').toLowerCase();
    return {
      ticker: upper(setup.ticker),
      direction: upper(setup.direction),
      timeframe: firstPresent(setup.timeframe, setup.scanner_timeframe, setup.setupTf) || '',
      grade: firstPresent(setup.setupGrade, setup.setup_grade, setup.scanner_status, setup.trade_eval?.trade_stage) || '',
      lifecycle_state: setupBucket(setup, options.progressResolver) || 'UNKNOWN',
      entry_status: firstPresent(setup.entryStatus, setup.entry_status) || '',
      section: displaySection(setup, options.progressResolver),
      optionable: flags.optionable,
      contract_data_available: flags.contract_data_available,
      suggested_contract_found: flags.suggested_contract_found,
      estimated_contract_cost: cost,
      cost_band: band,
      contract_source: source,
      live_failure_reason: firstPresent(contract.reason, contract.live_failure_reason, ''),
      strike: firstPresent(contract.strike, setup.option?.strike),
      expiry: firstPresent(contract.expiry, contract.expiration, setup.option?.expiry),
      bid: firstPresent(contract.bid, setup.option?.bid),
      ask: firstPresent(contract.ask, setup.option?.ask),
      spread: firstPresent(contract.spread, setup.option?.spread),
      open_interest: firstPresent(contract.open_interest, contract.openInterest, setup.option?.open_interest),
      volume: firstPresent(contract.volume, setup.option?.volume),
      dte: firstPresent(contract.dte, setup.option?.dte),
      flags,
    };
  }

  function summarize(rows = [], options = {}) {
    const diagnostics = rows.map(row => rowDiagnostic(row, options));
    const matrix = Object.fromEntries(STAGES.map(stage => [stage, blankCounts()]));
    diagnostics.forEach(item => {
      STAGES.forEach(stage => {
        if (item.flags[stage]) incrementStage(matrix, stage, item.cost_band);
      });
    });
    const bySection = {};
    diagnostics.forEach(item => {
      bySection[item.section] = bySection[item.section] || blankCounts();
      bySection[item.section][item.cost_band] += 1;
    });
    const byFailureReason = {};
    diagnostics
      .filter(item => item.cost_band === 'UNAVAILABLE' || !item.suggested_contract_found)
      .forEach(item => {
        const reason = item.live_failure_reason || (item.contract_data_available ? 'contract cost unavailable' : 'contract data unavailable');
        byFailureReason[reason] = (byFailureReason[reason] || 0) + 1;
      });
    return {
      generated_at: new Date().toISOString(),
      total_rows: diagnostics.length,
      stage_matrix: matrix,
      by_section: bySection,
      by_failure_reason: byFailureReason,
      rows: diagnostics,
      proposals: QUALITY_PROPOSALS,
      safeguards: QUALITY_SAFEGUARDS,
      recommendation: recommend(matrix, byFailureReason),
    };
  }

  function recommend(matrix = {}, failureReasons = {}) {
    const enterNow = matrix.enter_now || blankCounts();
    const developing = matrix.developing || blankCounts();
    const budgetPipeline = (enterNow.BUDGET || 0) + (developing.BUDGET || 0);
    const midPipeline = (enterNow.MID_RANGE || 0) + (developing.MID_RANGE || 0);
    const unavailable = (enterNow.UNAVAILABLE || 0) + (developing.UNAVAILABLE || 0);
    const reasons = Object.entries(failureReasons).sort((a, b) => b[1] - a[1]);
    if (budgetPipeline === 0 && midPipeline <= 1) {
      return 'Current actionable/developing mix is structurally sparse below Premium. Investigate universe balance first, then audit whether cheaper acceptable contracts exist for high-premium setups.';
    }
    if (unavailable > budgetPipeline + midPipeline) {
      return `Contract data availability is the main bottleneck (${reasons[0]?.[0] || 'unknown'}). Fix coverage before changing the universe.`;
    }
    if (budgetPipeline === 0) {
      return 'Budget opportunities are absent in the current setup pipeline. Add reviewed liquid low-premium candidates before changing contract selection.';
    }
    return 'Preserve current scanner standards. Use this mix over several scans before changing universe or contract-selection policy.';
  }

  return {
    BANDS,
    STAGES,
    QUALITY_PROPOSALS,
    QUALITY_SAFEGUARDS,
    contractPremium,
    contractCost,
    costBand,
    rowDiagnostic,
    summarize,
  };
});
