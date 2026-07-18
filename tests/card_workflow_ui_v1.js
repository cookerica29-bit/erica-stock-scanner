const assert = require('assert');
const fs = require('fs');

const html = fs.readFileSync('public/index.html', 'utf8');

assert.ok(html.includes('Execution Plan'), 'Execution Plan label should render');
assert.ok(html.includes('Risk/Reward'), 'Risk/Reward row should render');
assert.ok(html.includes('data-plan-visual="risk-reward"'), 'Risk/reward visual bar should render when full plan data exists');
assert.ok(html.includes("openTerms('term-risk-reward')"), 'Execution Plan help should link to Terms');
assert.ok(html.includes('id="term-planned-entry"'), 'Planned Entry term anchor should exist');
assert.ok(html.includes('id="term-stop"'), 'Stop term anchor should exist');
assert.ok(html.includes('id="term-target"'), 'Target term anchor should exist');
assert.ok(html.includes('id="term-risk-reward"'), 'Risk/Reward term anchor should exist');
assert.ok(html.includes('planned-entry'), 'Planned Entry row should have a distinct class');
assert.ok(html.includes('.index-plan-row.planned-entry'), 'Planned Entry styling should exist');

assert.ok(html.includes('execute-not-ready'), 'Execute not-ready state should exist');
assert.ok(html.includes('execute-waiting-entry'), 'Execute waiting-for-entry state should exist');
assert.ok(html.includes('execute-entry-ready'), 'Execute ready state should exist');
assert.ok(html.includes('execute-entry-passed'), 'Execute passed state should exist');
assert.ok(html.includes('.timeline-step.execute-entry-ready .timeline-dot'), 'Execute ready circle styling should exist');
assert.ok(html.includes('box-shadow: 0 0 0 5px rgba(34,197,94,0.13)'), 'Execute ready state should use a restrained green glow');
assert.ok(html.includes('data-normalized-status'), 'Cards should expose normalized status diagnostics');
assert.ok(html.includes('data-execution-state'), 'Cards should expose execution-state diagnostics');
assert.ok(html.includes('data-execute-visual-state'), 'Cards should expose execute visual-state diagnostics');

assert.ok(html.includes("label: '🎯 Suggested Contract'"), 'Validated live contracts should keep Suggested Contract heading');
assert.ok(html.includes("label: '🎯 Potential Contract'"), 'Fallback estimates should use Potential Contract heading');
assert.ok(html.includes('.suggested-contract.potential'), 'Potential Contract should be visually distinct');

assert.ok(html.includes('window.KairosExecutionGuidance.cardStatus'), 'Status badge should use the execution guidance status contract');
assert.ok(html.includes('window.KairosExecutionGuidance.readinessStages'), 'Timeline should use the execution guidance stage contract');

console.log('Card workflow UI v1 tests passed');
