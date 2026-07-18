(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosContractCandidateAudit = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function upper(value) {
    return String(value || '').trim().toUpperCase();
  }

  function costBand(cost) {
    const value = finiteNumber(cost);
    if (value === null || value < 0) return 'UNAVAILABLE';
    if (value <= 250) return 'BUDGET';
    if (value <= 600) return 'MID_RANGE';
    return 'PREMIUM';
  }

  function setupBucket(setup = {}, progressResolver = null) {
    if (typeof progressResolver === 'function') {
      const resolved = progressResolver(setup) || {};
      if (resolved.bucket) return upper(resolved.bucket);
    }
    return upper(setup.progress_bucket || setup.readiness_bucket || setup.status);
  }

  function contractCost(contract = {}) {
    if (!contract) return null;
    return finiteNumber(contract.estimated_contract_cost);
  }

  function auditForSetup(setup = {}, options = {}) {
    const audit = setup.best_contract?.candidate_audit || null;
    const bucket = setupBucket(setup, options.progressResolver);
    const current = audit?.current_selected_contract || null;
    const balanced = audit?.best_balanced_contract || null;
    const lowest = audit?.lowest_cost_acceptable_contract || null;
    const currentCost = contractCost(current);
    const balancedCost = contractCost(balanced);
    const lowestCost = contractCost(lowest);
    const potentialSavings = finiteNumber(audit?.potential_savings)
      ?? (currentCost !== null && lowestCost !== null && lowestCost < currentCost ? currentCost - lowestCost : null);
    return {
      ticker: upper(setup.ticker),
      direction: upper(setup.direction),
      timeframe: setup.timeframe || setup.scanner_timeframe || setup.setupTf || '',
      status: bucket || 'UNKNOWN',
      grade: setup.setupGrade || setup.setup_grade || '',
      audited: Boolean(audit),
      candidateDataUnavailable: !audit,
      current,
      bestQuality: audit?.best_quality_contract || null,
      balanced,
      lowest,
      rejectedCandidates: audit?.rejected_candidates || [],
      candidates: audit?.candidates || [],
      candidateCount: audit?.candidate_count || 0,
      acceptableCandidateCount: audit?.acceptable_candidate_count || 0,
      currentCost,
      balancedCost,
      lowestCost,
      potentialSavings: potentialSavings !== null ? Math.round(potentialSavings * 100) / 100 : null,
      cheaperAcceptableFound: potentialSavings !== null && potentialSavings > 0,
      currentBand: costBand(currentCost),
      lowestBand: costBand(lowestCost),
      productionSelectionChanged: false,
    };
  }

  function summarize(rows = [], options = {}) {
    const relevant = rows.filter(setup => {
      const bucket = setupBucket(setup, options.progressResolver);
      return bucket === 'ENTER_NOW' || bucket === 'EARLY_ENTRY' || bucket === 'ALMOST_READY' || bucket === 'WAITING';
    });
    const audits = relevant.map(setup => auditForSetup(setup, options));
    const audited = audits.filter(row => row.audited);
    const withCheaper = audited.filter(row => row.cheaperAcceptableFound);
    const withoutCheaper = audited.filter(row => !row.cheaperAcceptableFound);
    const unavailable = audits.filter(row => row.candidateDataUnavailable);
    const savings = withCheaper.map(row => row.potentialSavings).filter(value => value !== null);
    const distribution = { BUDGET: 0, MID_RANGE: 0, PREMIUM: 0, UNAVAILABLE: 0 };
    audited.forEach(row => {
      distribution[row.currentBand] = (distribution[row.currentBand] || 0) + 1;
    });
    return {
      setup_count: relevant.length,
      setups_audited: audited.length,
      cheaper_acceptable_candidate_found: withCheaper.length,
      no_cheaper_acceptable_candidate: withoutCheaper.length,
      candidate_data_unavailable: unavailable.length,
      average_potential_savings: savings.length
        ? Math.round((savings.reduce((sum, value) => sum + value, 0) / savings.length) * 100) / 100
        : null,
      distribution_by_current_selection_band: distribution,
      rows: audits,
    };
  }

  return {
    costBand,
    auditForSetup,
    summarize,
  };
});
