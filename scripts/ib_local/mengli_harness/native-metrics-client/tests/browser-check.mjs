import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { startMetricsBrowserFixture } from './browser-fixture.mjs';
import { fixtureCard, fixtureRow } from './fixture-plugin.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
const output = path.join(here, '.browser-artifacts', String(Date.now()));
fs.mkdirSync(output, { recursive: true });
const fixture = await startMetricsBrowserFixture();
assert(!['8900', '8902'].includes(new URL(fixture.url).port));
const browser = spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
  '--disable-sync', '--disable-extensions', '--remote-debugging-port=0', `--user-data-dir=${path.join(output, 'edge-profile')}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
const report = { status: 'running', scope: 'Official DSH; explicit fixture telemetry; no body/model', checks: [], errors: [], requests: [] };
let socket;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let text = '';
    const timer = setTimeout(() => reject(new Error('Edge debugger timeout')), 15000);
    browser.stderr.on('data', chunk => { text += chunk; const match = text.match(/DevTools listening on (ws:\/\/[^\s]+)/); if (match) { clearTimeout(timer); resolve(match[1]); } });
    browser.once('error', reject);
  });
  socket = new WebSocket(endpoint); await once(socket, 'open');
  let seq = 0; const pending = new Map();
  socket.addEventListener('message', event => {
    const value = JSON.parse(event.data);
    if (value.id) { const waiter = pending.get(value.id); pending.delete(value.id); value.error ? waiter.reject(new Error(JSON.stringify(value.error))) : waiter.resolve(value.result); }
    else if (value.method === 'Runtime.exceptionThrown') report.errors.push(value.params.exceptionDetails.exception?.description || value.params.exceptionDetails.text);
    else if (value.method === 'Network.requestWillBeSent') report.requests.push(value.params.request.url);
  });
  const command = (method, params = {}, sessionId) => new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params, sessionId })); });
  const evaluate = async (id, expression) => {
    const value = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }, id);
    if (value.exceptionDetails) throw new Error(value.exceptionDetails.exception?.description || value.exceptionDetails.text);
    return value.result.value;
  };
  const ready = async (id, expression) => {
    for (let attempt = 0; attempt < 100; attempt++) {
      if (await evaluate(id, expression)) return;
      await new Promise(resolve => setTimeout(resolve, 80));
    }
    throw new Error('Readiness failed: ' + expression + '\n' + await evaluate(id, 'document.body.innerText.slice(-6000)'));
  };
  const capture = async (id, name) => { const { data } = await command('Page.captureScreenshot', { format: 'png' }, id); fs.writeFileSync(path.join(output, name), Buffer.from(data, 'base64')); };
  const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
  const { sessionId: id } = await command('Target.attachToTarget', { targetId, flatten: true });
  for (const domain of ['Page', 'Runtime', 'Network']) await command(domain + '.enable', {}, id);
  await command('Emulation.setDeviceMetricsOverride', { width: 1440, height: 960, deviceScaleFactor: 1, mobile: false }, id);
  await command('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] }, id);
  await command('Page.navigate', { url: fixture.url }, id);
  await ready(id, 'Boolean(document.querySelector("[aria-label=打开生命体征]"))');
  assert.equal(report.requests.filter(url => url.includes('/api/family/telemetry')).length, 0);
  await evaluate(id, 'document.querySelector("[aria-label=打开生命体征]").click()');
  await ready(id, 'Boolean(document.querySelector("[data-mengli-metrics=life]")?.textContent.includes("456,789"))');
  await capture(id, '01-life-desktop-light.png');
  const life = await evaluate(id, 'document.querySelector("[data-mengli-metrics=life]").innerText');
  assert(life.includes('3,509') && life.includes('213 / 4,096') && !life.includes('健康分数'));
  assert(life.includes('交互仪表盘') && !life.includes('此轮无绑定拓扑快照'));
  report.checks.push('lazy first GET; real official life panel, saved cycle not initial zero, raw four-layer norms');

  // Use the actual native '+' tab affordance and GuideBody entries, not direct plugin calls.
  const clickPlus = `(() => { const n=[...document.querySelectorAll('button')].find(n=>/^(New tab|新标签页)$/.test(n.getAttribute('aria-label')||n.getAttribute('title')||'')); if(!n) throw Error('Native plus not found: '+[...document.querySelectorAll('button')].map(n=>n.getAttribute('aria-label')||n.getAttribute('title')||n.textContent).join('|')); n.click(); })()`;
  await evaluate(id, clickPlus);
  await ready(id, 'Boolean(document.querySelector("[data-sidebar-right-guide-entry=mengli-learning-metrics]"))');
  assert(await evaluate(id, 'Boolean(document.querySelector("[data-sidebar-right-guide-entry=mengli-life-metrics]"))'));
  await capture(id, '02-official-plus-guide.png');
  await evaluate(id, 'document.querySelector("[data-sidebar-right-guide-entry=mengli-learning-metrics]").click()');
  await ready(id, 'Boolean(document.querySelector("[data-mengli-metrics=learning]")?.textContent.includes("仅 1 条记录，暂无趋势"))');
  assert.equal(await evaluate(id, 'document.querySelectorAll("[data-mengli-metrics=learning] svg").length'), 0);
  assert.equal(await evaluate(id, 'document.querySelectorAll("[data-mengli-metrics=learning] .mm-single-record").length'), 2);
  assert.equal(await evaluate(id, 'document.querySelectorAll("[data-mengli-metrics=learning] path").length'), 0);
  await evaluate(id, '[...document.querySelectorAll("[data-mengli-metrics=learning] button")].find(n=>n.textContent==="原始表格").click()');
  await ready(id, 'Boolean(document.querySelector("[data-mengli-metrics=learning] table")?.textContent.includes("null"))');
  await capture(id, '03-learning-desktop-light.png');
  assert.equal(report.requests.filter(url => url.includes('/api/family/telemetry')).length, 1);
  report.checks.push('native + guide contains both panels; one sample is two compact records with no SVG; raw null table; no duplicate GET');

  // Commit through precisely the same subscription contract as native-family-client.
  const event = { status: 'committed', cursor: 1, cycle: 3510, branch_id: fixtureCard.branch_id,
    checkpoint: '3509/fixture-next.json', row: { ...fixtureRow, cycle: 3510, updates: 0, response_nll: 0, joint_progress: null },
    life_card_data: { ...fixtureCard, age: { cycles: 3510, byte_events: 456917 }, latest: { updates: 0, momentum_norms: [0, 0.1, 0.08, 0.15] } } };
  await evaluate(id, `metricsFixtureCommit(${JSON.stringify(event)})`);
  await ready(id, 'document.querySelectorAll("[data-mengli-metrics=learning] .mm-hit").length===2');
  assert(await evaluate(id, 'document.querySelector("[data-mengli-metrics=learning] .mm-heroes").textContent.includes("本轮更新0")'));
  await evaluate(id, `metricsFixtureCommit(${JSON.stringify(event)})`);
  assert.equal(await evaluate(id, 'document.querySelectorAll("[data-mengli-metrics=learning] .mm-hit").length'), 2);
  await evaluate(id, 'document.querySelector("[data-mengli-metrics=learning] .mm-hit").focus()');
  await ready(id, 'document.querySelector("[data-mengli-metrics=learning] .mm-readout").textContent.includes("读数 · 3509")');
  report.checks.push('saved event creates real two-point SVG; true zero retained; null stays compact; repeated event deduplicated; keyboard tooltip');

  // Close using the native tab menu affordance, then reopen from official GuideBody.
  const buttons = await evaluate(id, '[...document.querySelectorAll("button")].map(n=>({label:n.getAttribute("aria-label"),title:n.getAttribute("title"),text:n.textContent})).filter(n=>/close|关闭/i.test(n.label||n.title||n.text))');
  report.closeButtons = buttons;
  const close = `(() => {const b=[...document.querySelectorAll('button')].find(n=>/^(Close tab|关闭标签页|Close|关闭)$/.test(n.getAttribute('aria-label')||n.getAttribute('title')||'')&&n.closest('[data-dockkit-tab]')?.textContent.includes('学习训练'));if(b){b.click();return true}return false})()`;
  const closed = await evaluate(id, close);
  if (!closed) {
    // Official rightbar shortcut closes the focused current tab.
    await evaluate(id, 'document.querySelector("[data-mengli-metrics=learning] button").focus()');
    await command('Input.dispatchKeyEvent', { type: 'keyDown', key: 'w', code: 'KeyW', modifiers: 2, windowsVirtualKeyCode: 87 }, id);
    await command('Input.dispatchKeyEvent', { type: 'keyUp', key: 'w', code: 'KeyW', modifiers: 2, windowsVirtualKeyCode: 87 }, id);
  }
  await ready(id, '!document.querySelector("[data-mengli-metrics=learning]")');
  await evaluate(id, 'document.querySelector("[aria-label=打开生命体征]").click()');
  await evaluate(id, clickPlus);
  await ready(id, 'Boolean(document.querySelector("[data-sidebar-right-guide-entry=mengli-learning-metrics]"))');
  await evaluate(id, 'document.querySelector("[data-sidebar-right-guide-entry=mengli-learning-metrics]").click()');
  await ready(id, 'Boolean(document.querySelector("[data-mengli-metrics=learning]"))');
  assert.equal(report.requests.filter(url => url.includes('/api/family/telemetry')).length, 1);
  report.checks.push('native close/reopen retains shared saved read model without refetch');

  await command('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }, { name: 'prefers-reduced-motion', value: 'reduce' }] }, id);
  await ready(id, 'document.body.hasAttribute("data-ds-dark-theme")');
  await capture(id, '04-learning-desktop-dark.png');
  await command('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true }, id);
  await ready(id, 'innerWidth===390');
  await capture(id, '05-learning-mobile-dark.png');
  assert(await evaluate(id, 'document.documentElement.scrollWidth===innerWidth'));
  assert(await evaluate(id, 'document.querySelector("[data-mengli-metrics=learning]").clientWidth>=300'));
  assert.equal(await evaluate(id, 'document.querySelectorAll("iframe").length'), 0);
  const mobileButton = await evaluate(id, `(() => {const b=document.querySelector('[aria-label="生命与学习面板"]');const r=b.getBoundingClientRect();return {width:r.width,height:r.height,display:getComputedStyle(b.parentElement).display}})()`);
  assert(mobileButton.width >= 44 && mobileButton.height >= 44);
  await evaluate(id, 'document.querySelector("[aria-label=生命与学习面板]").click()');
  await ready(id, 'Boolean([...document.querySelectorAll("[role=menuitem]")].find(n=>n.textContent.includes("生命体征")))');
  assert(await evaluate(id, '[...document.querySelectorAll("[role=menuitem]")].some(n=>n.textContent.includes("学习训练"))'));
  await capture(id, '06-mobile-native-menu.png');
  await evaluate(id, '[...document.querySelectorAll("[role=menuitem]")].find(n=>n.textContent.includes("生命体征")).click()');
  await ready(id, 'Boolean(document.querySelector("[data-mengli-metrics=life]")?.textContent.includes("456,917"))');
  await capture(id, '07-life-mobile-dark.png');
  await command('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] }, id);
  await ready(id, '!document.body.hasAttribute("data-ds-dark-theme")');
  await capture(id, '08-life-mobile-light.png');
  assert.equal(report.requests.filter(url => url.includes('/api/family/telemetry')).length, 1);
  report.checks.push('390px native pane, dark/light and reduced-motion; 44px official menu with both entries; no page overflow or iframe');
  assert.equal(report.errors.length, 0, report.errors.join('\n'));
  assert(!report.requests.some(url => /\/api\/session\/prompt|\/api\/family\/(load|say|step)|api\.deepseek/.test(url)));
  report.status = 'passed';
} catch (error) { report.status = 'failed'; report.error = error.stack; throw error; }
finally {
  socket?.close();
  if (browser.exitCode === null) { const exited = once(browser, 'exit'); browser.kill(); await exited; }
  await fixture.close();
  try { fs.rmSync(path.join(output, 'edge-profile'), { recursive: true, force: true }); } catch {}
  fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, output, checks: report.checks, error: report.error, errors: report.errors }));
}
