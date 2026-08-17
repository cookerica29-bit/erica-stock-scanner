const assert = require('assert');
const fs = require('fs');

const html = fs.readFileSync('public/index.html', 'utf8');

assert.ok(!html.includes('function renderTradePlanBlock'), 'Scanner card Execution Plan renderer should be removed');
assert.ok(!html.includes('data-plan-visual="risk-reward"'), 'Scanner card risk/reward rail should be removed');
assert.ok(html.includes('id="term-planned-entry"'), 'Planned Entry term anchor should exist');
assert.ok(html.includes('id="term-stop"'), 'Stop term anchor should exist');
assert.ok(html.includes('id="term-target"'), 'Target term anchor should exist');
assert.ok(html.includes('id="term-risk-reward"'), 'Planned TP1 term anchor should exist');

assert.ok(html.includes('execute-not-ready'), 'Execute not-ready state should exist');
assert.ok(html.includes('execute-waiting-entry'), 'Execute waiting-for-entry state should exist');
assert.ok(html.includes('execute-entry-ready'), 'Execute ready state should exist');
assert.ok(html.includes('execute-entry-passed'), 'Execute passed state should exist');
assert.ok(html.includes('.timeline-step.execute-entry-ready .timeline-dot'), 'Execute ready circle styling should exist');
assert.ok(html.includes('box-shadow: 0 0 0 5px rgba(34,197,94,0.13)'), 'Execute ready state should use a restrained green glow');
assert.ok(html.includes('data-normalized-status'), 'Cards should expose normalized status diagnostics');
assert.ok(html.includes('data-execution-state'), 'Cards should expose execution-state diagnostics');
assert.ok(html.includes('data-execute-visual-state'), 'Cards should expose execute visual-state diagnostics');

assert.ok(html.includes('🎯 Contract Candidate'), 'Scanner cards should render Contract Candidate heading');
assert.ok(html.includes('Strike'), 'Contract Candidate should show strike');
assert.ok(html.includes('Model DTE Window'), 'Contract Candidate should show DTE guidance');
assert.ok(html.includes('term-contract'), 'Contract Candidate should link to Terms');

assert.ok(html.includes('window.KairosExecutionGuidance.cardStatus'), 'Status badge should use the execution guidance status contract');
assert.ok(html.includes('window.KairosExecutionGuidance.readinessStages'), 'Timeline should use the execution guidance stage contract');

console.log('Card workflow UI v1 tests passed');
