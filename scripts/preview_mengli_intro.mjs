// Actual local-browser preview via Chromium's built-in DevTools protocol.
// No package download, fake telemetry, model call, or training is involved.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';

const option = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const origin = option('--url') ?? 'http://127.0.0.1:8900';
const executable = option('--browser') ?? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const output = path.resolve(option('--output') ?? 'results/mengli_intro_20261113/preview');
assert(!fs.existsSync(output), 'Choose a new preview output directory');
fs.mkdirSync(output, { recursive: true });
const profile = path.resolve(`data/harness/intro-browser-${process.pid}`);
fs.mkdirSync(profile, { recursive: true });
const browser = spawn(executable, ['--headless=new', '--disable-gpu', '--no-first-run',
  '--no-default-browser-check', '--disable-background-networking', '--disable-sync',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
let socket;
const report = { status: 'running', screenshots: [], requests: [], console_errors: [] };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
try {
  const debuggerUrl = await new Promise((resolve, reject) => {
    let text = '';
    const timeout = setTimeout(() => reject(new Error('Browser debugger did not start')), 15000);
    browser.once('error', reject);
    browser.stderr.on('data', chunk => {
      text += chunk;
      const match = text.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    });
  });
  socket = new WebSocket(debuggerUrl);
  await once(socket, 'open');
  let id = 0;
  const pending = new Map();
  const listeners = new Set();
  socket.addEventListener('message', event => {
    const value = JSON.parse(event.data);
    if (value.id) {
      const job = pending.get(value.id);
      pending.delete(value.id);
      if (value.error) job?.reject(new Error(JSON.stringify(value.error)));
      else job?.resolve(value.result);
    } else {
      if (value.method === 'Network.requestWillBeSent') {
        const request = value.params.request;
        const url = new URL(request.url);
        report.requests.push({ method: request.method, url: url.origin + url.pathname });
      }
      if (value.method === 'Runtime.exceptionThrown') report.console_errors.push(value.params.exceptionDetails.text);
      for (const listener of [...listeners]) listener(value);
    }
  });
  const command = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const key = ++id;
    pending.set(key, { resolve, reject });
    socket.send(JSON.stringify({ id: key, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const event = (method, sessionId) => new Promise(resolve => {
    const listener = message => {
      if (message.method === method && message.sessionId === sessionId) { listeners.delete(listener); resolve(message.params); }
    };
    listeners.add(listener);
  });
  async function page(width, height, mobile = false, url = origin) {
    const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true });
    for (const domain of ['Page', 'Runtime', 'Network']) await command(`${domain}.enable`, {}, sessionId);
    await command('Page.bringToFront', {}, sessionId);
    await command('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile }, sessionId);
    const loaded = event('Page.loadEventFired', sessionId);
    await command('Page.navigate', { url }, sessionId);
    await loaded;
    return sessionId;
  }
  const evaluate = async (sessionId, expression) => {
    const result = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }, sessionId);
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };
  async function capture(sessionId, file) {
    const phase = await evaluate(sessionId, 'document.getElementById("mengli-intro")?.dataset.phase ?? "workbench"');
    const { data } = await command('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false }, sessionId);
    fs.writeFileSync(path.join(output, file), Buffer.from(data, 'base64'));
    report.screenshots.push({ file, phase });
  }
  const desktop = await page(1440, 960);
  await delay(350);
  if (process.argv.includes('--expect-life')) {
    report.life_binding = await evaluate(desktop, '({card:window.__mengliLife,source:document.getElementById("mengli-intro").dataset.lifeSource,activity:document.getElementById("mengli-intro").dataset.activity,momentum:document.getElementById("mengli-intro").dataset.momentum,animation:getComputedStyle(document.querySelector(".intro-mark")).animationName})');
    assert.equal(report.life_binding.card?.schema, 1);
    assert.equal(report.life_binding.source, report.life_binding.card.source.sha256);
    assert.equal(report.life_binding.activity, 'saved');
    assert.equal(report.life_binding.momentum, 'measured');
    assert.equal(report.life_binding.animation, 'none');
    await evaluate(desktop, 'window.MengliIntro.updateLife(window.__mengliLife)');
    assert.equal(await evaluate(desktop, 'document.querySelector(".mengli-brand-mark").classList.contains("has-life-update")'), false);
  }
  await capture(desktop, '01-seed.png');
  await delay(1300); await capture(desktop, '02-unfurl.png');
  await delay(2100); await capture(desktop, '03-name.png');
  await delay(3300); await capture(desktop, '04-workbench.png');
  assert.equal(await evaluate(desktop, 'Boolean(document.getElementById("mengli-intro"))'), false);
  const loaded = event('Page.loadEventFired', desktop);
  await command('Page.reload', {}, desktop); await loaded;
  assert.equal(await evaluate(desktop, 'Boolean(document.getElementById("mengli-intro"))'), false);
  await evaluate(desktop, 'document.getElementById("intro-replay").click()');
  assert.equal(await evaluate(desktop, 'document.getElementById("mengli-intro").dataset.phase'), 'seed');
  await command('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape' }, desktop);
  await command('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape' }, desktop);
  await delay(800);
  assert.equal(await evaluate(desktop, 'Boolean(document.querySelector(".workspace").inert)'), false);
  const mobile = await page(390, 844, true);
  report.mobile_viewport = await evaluate(mobile, '({width:document.getElementById("mengli-intro").getBoundingClientRect().width,height:document.getElementById("mengli-intro").getBoundingClientRect().height,innerWidth,visualWidth:visualViewport.width})');
  assert.equal(report.mobile_viewport.width, 390);
  assert.equal(report.mobile_viewport.height, 844);
  await delay(3800); await capture(mobile, '05-mobile.png');
  await command('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape' }, mobile);
  await delay(800);
  await command('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] }, mobile);
  await evaluate(mobile, 'window.MengliIntro.replay()');
  assert.equal(await evaluate(mobile, 'document.getElementById("mengli-intro").classList.contains("is-reduced")'), true);
  await delay(900);
  assert.equal(await evaluate(mobile, 'Boolean(document.getElementById("mengli-intro"))'), false);
  let motherOrigin = null;
  if (process.argv.includes('--family')) {
    const family = await evaluate(desktop, 'fetch("/api/family").then(r=>r.json())');
    assert.equal(family.mode, 'paused');
    assert.equal(family.loaded, false);
    assert(family.workbench?.url, 'Official workbench must be started');
    report.family = { cycle: family.cycle, mode: family.mode, loaded: family.loaded, management_session_id: family.management_session_id };
    await evaluate(desktop, 'window.MengliFamily.showView("care")');
    await evaluate(desktop, 'document.getElementById("care-takeover").click()');
    for (let check = 0; check < 30; check++) {
      if (await evaluate(desktop, 'document.getElementById("care-mode").textContent.includes("家长接管")')) break;
      await delay(100);
    }
    assert.equal(await evaluate(desktop, 'document.getElementById("care-mode").textContent.includes("家长接管")'), true);
    const paused = await evaluate(desktop, 'fetch("/api/family/control",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({mode:"paused"})}).then(r=>r.json())');
    assert.equal(paused.loaded, false);
    assert.equal(paused.cycle, family.cycle);
    await capture(desktop, '06-family-desktop.png');
    await evaluate(desktop, 'document.querySelector("[data-view=plan]").click()');
    assert.equal(await evaluate(desktop, 'document.getElementById("family-plan-panel").hidden'), false);
    await capture(desktop, '07-family-plan.png');
    await evaluate(mobile, 'window.MengliFamily.showView("care")');
    report.family.mobile = await evaluate(mobile, '({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,sendDisabled:document.getElementById("parent-send").disabled})');
    assert.equal(report.family.mobile.width, report.family.mobile.scrollWidth);
    assert.equal(report.family.mobile.sendDisabled, true);
    await capture(mobile, '08-family-mobile.png');
    motherOrigin = family.workbench.origin;
    const mother = await page(1440, 960, false, family.workbench.url);
    await delay(1500);
    report.family.official_surface = await evaluate(mother, '({title:document.title,text:document.body.innerText.slice(0,2000),width:innerWidth,scrollWidth:document.documentElement.scrollWidth})');
    assert(report.family.official_surface.text.length > 50, 'Official UI must render, not an empty frame');
    assert(!report.family.official_surface.text.includes('梦漓生命体监控'), 'Retired injected dashboard must not appear in the official surface');
    report.family.composer_ready = await evaluate(mother, 'Boolean(document.querySelector("[contenteditable=true],textarea:not([disabled])"))');
    assert.equal(report.family.composer_ready, true, 'The bound management workspace must have an enabled composer');
    await capture(mother, '09-official-mother.png');
    await evaluate(desktop, 'window.MengliFamily.showView("mother")');
    await delay(1000);
    await capture(desktop, '10-embedded-mother.png');
    await evaluate(mobile, 'window.MengliFamily.showView("mother")');
    await delay(1000);
    await capture(mobile, '11-embedded-mother-mobile.png');
  }
  assert.equal(report.console_errors.length, 0);
  if (motherOrigin) assert(report.requests.every(request => request.url.startsWith(origin) || request.url.startsWith(motherOrigin)), 'No external model/API requests in browser preview');
  else assert(report.requests.every(request => request.method === 'GET' && request.url.startsWith(origin)));
  report.status = 'passed';
  report.checks = ['normal completion', 'same-tab refresh skips', 'manual replay', 'Escape', 'background released', 'mobile width', 'reduced motion', motherOrigin ? 'family tabs and native DSH surface; no external API' : 'GET-only network'];
} catch (error) {
  report.status = 'failed'; report.error = error.message;
  throw error;
} finally {
  socket?.close();
  if (browser.exitCode === null) { const exited = once(browser, 'exit'); browser.kill(); await exited; }
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch { report.browser_profile_retained = profile; }
  fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2)+'\n');
  console.log(JSON.stringify({ status: report.status, screenshots: report.screenshots, checks: report.checks, error: report.error }));
}
