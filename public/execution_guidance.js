(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosExecutionGuidance = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  function upper(value) {
    return String(value || '').trim().toUpperCase();
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function plannedEntry(setup = {}) {
    return finiteNumber(setup.entry ?? setup.entry_price);
  }

  function currentPrice(setup = {}) {
    return finiteNumber(setup.price ?? setup.current_price ?? setup.underlying_price);
  }

  function isNoTrade(setup = {}, readiness = {}) {
    const bucket = upper(readiness.bucket);
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    const setupStatus = upper(setup.setupStatus || setup.stockSetupStatus);
    return bucket === 'SKIP'
      || tradeStage.includes('NO TRADE')
      || tradeStage.includes('INVALID')
      || setupStatus.includes('INVALID');
  }

  function isDeveloping(setup = {}, readiness = {}) {
    const bucket = upper(readiness.bucket);
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    return bucket === 'WAITING'
      || tradeStage.includes('BUILDING')
      || tradeStage.includes('WATCHLIST')
      || upper(setup.entryStatus) === 'WAITING';
  }

  function hasAvailableContract(contractLifecycle) {
    return contractLifecycle === 'available';
  }

  function nextStep(setup = {}, readiness = {}, contractLifecycle = 'pending') {
    const bucket = upper(readiness.bucket);
    const entryStatus = String(setup.entryStatus || '').trim();
    const entry = plannedEntry(setup);

    if (isNoTrade(setup, readiness)) {
      return {
        label: 'Next Step',
        lines: ['No entry. The setup is not currently valid.'],
      };
    }

    if (bucket === 'ENTER_NOW') {
      if (hasAvailableContract(contractLifecycle)) {
        return {
          label: 'Next Step',
          lines: ['Price is in the planned entry zone. You can execute this trade.'],
        };
      }
      return {
        label: 'Next Step',
        lines: ['Price is in the planned entry zone.', 'Verify and select the live option contract before executing.'],
      };
    }

    if (entryStatus === 'Tradeable') {
      return {
        label: 'Next Step',
        lines: ['Monitor price near the planned entry.'],
      };
    }

    if (bucket === 'ALMOST_READY' || entryStatus === 'Near Entry') {
      const alertLine = entry !== null ? `Set an alert at $${entry.toFixed(2)}.` : null;
      return {
        label: 'Next Step',
        lines: ['Waiting for price to reach the planned entry.', alertLine].filter(Boolean),
      };
    }

    if (isDeveloping(setup, readiness)) {
      return {
        label: 'Next Step',
        lines: ['Continue monitoring. The setup is still developing.'],
      };
    }

    return {
      label: 'Next Step',
      lines: ['Continue monitoring. The setup is still developing.'],
    };
  }

  function executionPlanRows(setup = {}) {
    return [
      { label: 'Current Price', value: currentPrice(setup) },
      { label: 'Planned Entry', value: plannedEntry(setup) },
      { label: 'Stop', value: finiteNumber(setup.sl ?? setup.stop ?? setup.stop_price) },
      { label: 'Target', value: finiteNumber(setup.tp1 ?? setup.target ?? setup.target_price) },
    ];
  }

  return {
    plannedEntry,
    currentPrice,
    nextStep,
    executionPlanRows,
  };
});
