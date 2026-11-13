// Native DSH browser acceptance, reusing the existing Edge/CDP preview technique.
// Does not submit to a model or load/advance the real body.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';

const option = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const fixture = process.argv.includes('--fixture')
  ? await (await import('../tests/fixtures/mengli_native_browser.mjs')).startNativeBrowserFixture() : null;
const bridge = fixture?.bridge || option('--bridge') || 'http://127.0.0.1:8902';
const family = await (await fetch(bridge + '/api/family')).json();
const url = family.workbench?.url;
assert(url, 'Native workbench must be ready');
const origin = new URL(url).origin;
const output = path.resolve(option('--output') || 'results/mengli_native_20261113/browser');
assert(!fs.existsSync(output), 'Choose a new preview output directory');
fs.mkdirSync(output, { recursive: true });
const profile = path.resolve(`data/harness/native-browser-${process.pid}`);
fs.mkdirSync(profile, { recursive: true });
const browser = spawn(option('--browser') || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--disable-sync', '--disable-extensions', '--disk-cache-size=1048576',
  '--media-cache-size=1048576', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const report = { status: 'running', scope: fixture ? 'offline UI fixture; no real body or model' : 'saved real body read-only; unsent drafts only', screenshots: [], requests: [], errors: [] };
let socket;
try {
  const debuggerURL = await new Promise((resolve, reject) => {
    let text = '';
    const timeout = setTimeout(() => reject(new Error('Browser debugger did not start')), 15000);
    browser.once('error', reject);
    browser.stderr.on('data', chunk => {
      text += chunk;
      const match = text.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timeout); resolve(match[1]); }
    });
  });
  socket = new WebSocket(debuggerURL);
  await once(socket, 'open');
  let sequence = 0;
  const pending = new Map(), listeners = new Set();
  socket.addEventListener('message', event => {
    const value = JSON.parse(event.data);
    if (value.id) {
      const waiter = pending.get(value.id); pending.delete(value.id);
      if (value.error) waiter?.reject(new Error(JSON.stringify(value.error)));
      else waiter?.resolve(value.result);
    } else {
      if (value.method === 'Runtime.exceptionThrown') report.errors.push(value.params.exceptionDetails.exception?.description || value.params.exceptionDetails.text);
      if (value.method === 'Runtime.consoleAPICalled' && value.params.type === 'error') report.errors.push(value.params.args.map(arg => arg.value || arg.description || '').join(' '));
      if (value.method === 'Network.requestWillBeSent') {
        const request = value.params.request, parsed = new URL(request.url);
        const inline = ['data:', 'blob:'].includes(parsed.protocol);
        report.requests.push({ method: request.method, url: inline ? `${parsed.protocol}embedded-resource` : parsed.origin + parsed.pathname, inline });
      }
      for (const listener of [...listeners]) listener(value);
    }
  });
  const command = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const event = (method, sessionId) => new Promise(resolve => {
    const listener = value => { if (value.method === method && value.sessionId === sessionId) { listeners.delete(listener); resolve(value.params); } };
    listeners.add(listener);
  });
  const evaluate = async (sessionId, expression) => {
    const result = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }, sessionId);
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  const waitFor = async (sessionId, expression) => {
    for (let attempt = 0; attempt < 80; attempt++) {
      if (await evaluate(sessionId, expression)) return;
      await delay(100);
    }
    throw new Error(`UI did not become ready: ${expression}`);
  };
  async function page(width, height, mobile = false) {
    const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true });
    for (const domain of ['Page', 'Runtime', 'Network']) await command(`${domain}.enable`, {}, sessionId);
    await command('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile }, sessionId);
    const loaded = event('Page.loadEventFired', sessionId);
    await command('Page.navigate', { url }, sessionId);
    await loaded;
    try {
      await waitFor(sessionId, 'Boolean(document.querySelector("[contenteditable=true],textarea:not([disabled])"))');
      await evaluate(sessionId, 'window.MengliIntro?.skip?.()');
      await waitFor(sessionId, '!document.getElementById("mengli-intro")');
    } catch (error) {
      report.boot_failure = await evaluate(sessionId, '({text:document.body.innerText,html:document.body.innerHTML.slice(0,12000)})');
      await capture(sessionId, 'boot-failure.png');
      throw error;
    }
    return sessionId;
  }
  async function capture(sessionId, file) {
    const { data } = await command('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false }, sessionId);
    fs.writeFileSync(path.join(output, file), Buffer.from(data, 'base64'));
    report.screenshots.push(file);
  }
  const desktop = await page(1440, 960);
  await delay(500);
  report.initial = await evaluate(desktop, `({title:document.title,text:document.body.innerText.slice(0,6000),buttons:[...document.querySelectorAll('button,[role=tab]')].map(b=>({text:b.innerText,label:b.getAttribute('aria-label'),title:b.title})),iframes:document.querySelectorAll('iframe').length,custom:document.querySelectorAll('.dialogue-pane,#dialogue-stream,.family-tabs').length})`);
  assert.equal(report.initial.iframes, 0);
  assert.equal(report.initial.custom, 0);
  assert(!report.initial.text.includes('梦漓生命体监控'));
  assert(!report.initial.text.includes('预览版'), 'Mengli identity must not retain the upstream preview badge');
  assert(report.initial.title.includes('梦漓 MengLi'), 'Native product title must use Mengli identity');
  await capture(desktop, '01-native-mother.png');
  if (!fixture && !process.argv.includes('--inspect')) {
    await evaluate(desktop, `(()=>{const b=[...document.querySelectorAll('button')].find(n=>n.innerText.includes('DeepSeek Flash'));if(!b)throw new Error('Native model selector unavailable');b.click()})()`);
    await waitFor(desktop, `Boolean([...document.querySelectorAll('[role=menuitem]')].find(n=>n.innerText.includes('推理等级')))`);
    await evaluate(desktop, `([...document.querySelectorAll('[role=menuitem]')].find(n=>n.innerText.includes('推理等级'))).click()`);
    await waitFor(desktop, `['Off','Low','High','Max'].every(level=>document.body.innerText.includes(level))`);
    report.reasoning_levels = ['Off', 'Low', 'High', 'Max'];
    await capture(desktop, '01-reasoning-levels.png');
    for (let pane = 0; pane < 2; pane++) {
      await command('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape' }, desktop);
      await command('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape' }, desktop);
    }
  }

  if (!process.argv.includes('--inspect')) {
    const draft = '未发送的妈妈草稿 · 原生工作台稳定性检查';
    await evaluate(desktop, 'window.__nativePreviewEditor=document.querySelector("[contenteditable=true]");window.__nativePreviewEditor.focus()');
    await command('Input.insertText', { text: draft }, desktop);
    await delay(10500);
    report.draft_idle = await evaluate(desktop, `({focused:document.activeElement===window.__nativePreviewEditor,text:window.__nativePreviewEditor.textContent})`);
    assert(report.draft_idle.focused, 'Idle updates must preserve editor focus');
    assert(report.draft_idle.text.includes(draft), 'Idle updates must preserve unsent text');
    const clicked = await evaluate(desktop, `(()=>{const b=[...document.querySelectorAll('button,[role=tab]')].find(b=>b.innerText.trim()==='梦漓'&&b.getAttribute('aria-label')!=='梦漓菜单');if(!b)return false;b.click();return true})()`);
    assert(clicked, 'Native Mengli view must be available');
    if (fixture) {
      const clickCareAction = async text => {
        await evaluate(desktop, `(()=>{const b=[...document.querySelectorAll('[aria-label="家庭照护"]')].find(n=>n.getBoundingClientRect().width>0);if(!b)throw new Error('Native care menu unavailable');b.click()})()`);
        await waitFor(desktop, `Boolean([...document.querySelectorAll('[role=menuitem],button')].find(n=>n.innerText.trim()===${JSON.stringify(text)}&&n.getBoundingClientRect().width>0))`);
        await evaluate(desktop, `(()=>{const b=[...document.querySelectorAll('[role=menuitem],button')].find(n=>n.innerText.trim()===${JSON.stringify(text)}&&n.getBoundingClientRect().width>0);b.click()})()`);
      };
      await clickCareAction('连接身体');
      await waitFor(desktop, 'fetch("/api/family").then(r=>r.json()).then(v=>v.loaded)');
      await clickCareAction('家长接管');
      await waitFor(desktop, 'fetch("/api/family").then(r=>r.json()).then(v=>v.mode==="parent")');
      for (const [index, text] of ['家长界面交付 fixture', '/goal literal child-directed text'].entries()) {
        await evaluate(desktop, 'document.querySelector("[contenteditable=true]").focus()');
        await command('Input.insertText', { text }, desktop);
        if (index === 0) await evaluate(desktop, 'document.querySelector("button[aria-label=发送消息]").click()');
        else {
          await command('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 }, desktop);
          await command('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 }, desktop);
        }
        await waitFor(desktop, `fetch('/api/family').then(r=>r.json()).then(v=>v.cursor===${index + 1}&&!v.busy)`);
      }
      report.parent_delivery = { calls: fixture.requests.filter(value => value.command === 'step') };
      assert.equal(report.parent_delivery.calls.length, 2);
      assert(report.parent_delivery.calls.every(value => value.caregiver === 'parent'));
    }
    await waitFor(desktop, 'document.querySelectorAll("[data-slot=\\"conversation.chat.flow\\"] [data-chat-node-key]").length>0');
    report.care = await evaluate(desktop, `({text:document.body.innerText.slice(0,6000),nodes:document.querySelectorAll('[data-slot="conversation.chat.flow"] [data-chat-node-key]').length})`);
    assert(!report.care.text.includes('已保存历史/会话原文'), 'Saved provenance must not repeat in every message');
    report.role_sides = await evaluate(desktop, `({rightUserRows:document.querySelectorAll('[data-chat-flow-kind="user"]').length,leftSpeakerRows:document.querySelectorAll('[data-chat-flow-kind="assistant-step"]').length})`);
    assert.equal(report.role_sides.rightUserRows, fixture ? 2 : 0, 'Only the parent occupies the native user/right side');
    await capture(desktop, '02-native-mengli.png');
    // Keep real message nodes, not just equal text, across the former 2-second refresh periods.
    report.stable_setup = await evaluate(desktop, `(()=>{const nodes=[...document.querySelectorAll('[data-chat-node-key]')];window.__nativePreviewNodes=nodes;let scroller=nodes[0];while(scroller && !(scroller.scrollHeight>scroller.clientHeight && /auto|scroll/.test(getComputedStyle(scroller).overflowY)))scroller=scroller.parentElement;window.__nativePreviewScroller=scroller;if(scroller)scroller.scrollTop=Math.floor((scroller.scrollHeight-scroller.clientHeight)/3);window.__nativePreviewScrollTop=scroller?.scrollTop;return {count:nodes.length,scrollTop:scroller?.scrollTop}})()`);
    assert(report.stable_setup.count > 0, 'Saved body dialogue must use the native flow');
    await delay(200);
    report.visible_nodes = await evaluate(desktop, `(()=>{window.__nativePreviewNodes=[...document.querySelectorAll('[data-chat-node-key]')].filter(n=>{const r=n.getBoundingClientRect();return r.bottom>0&&r.top<innerHeight});window.__nativePreviewScrollTop=window.__nativePreviewScroller?.scrollTop;return window.__nativePreviewNodes.length})()`);
    assert(report.visible_nodes > 0);
    await delay(10500);
    report.stability = await evaluate(desktop, `({sameNodes:window.__nativePreviewNodes.every(n=>n.isConnected),scrollTop:window.__nativePreviewScroller?.scrollTop,expectedScrollTop:window.__nativePreviewScrollTop,iframes:document.querySelectorAll('iframe').length,text:document.body.innerText.slice(0,6000)})`);
    assert(report.stability.sameNodes, 'Unchanged native dialogue must not be rebuilt');
    assert(!report.stability.text.includes('家庭离线') && !report.stability.text.includes('家庭连接中断'), 'Family event connection must remain online');
    assert.equal(report.stability.scrollTop, report.stability.expectedScrollTop, 'Idle updates must not reset history scrolling');
    await evaluate(desktop, `(()=>{const b=[...document.querySelectorAll('button,[role=tab]')].find(b=>['聊天','对话'].includes(b.innerText.trim()));if(!b)throw new Error('Native mother Chat view unavailable');b.click()})()`);
    await waitFor(desktop, 'Boolean(document.querySelector("[contenteditable=true]"))');
    assert((await evaluate(desktop, 'document.querySelector("[contenteditable=true]").textContent')).includes(draft), 'Switching views must preserve the mother draft');
    await evaluate(desktop, `(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.innerText.trim()==='育儿计划');if(!b)throw new Error('Native care plan control unavailable');b.click()})()`);
    await waitFor(desktop, 'Boolean(document.querySelector("textarea[aria-label=育儿计划内容]"))');
    await waitFor(desktop, '(()=>{const r=document.querySelector("textarea[aria-label=育儿计划内容]").getBoundingClientRect();return r.width>=300 && r.right<=innerWidth+1})()');
    report.plan_width = await evaluate(desktop, 'document.querySelector("textarea[aria-label=育儿计划内容]").getBoundingClientRect().width');
    assert(report.plan_width >= 260, 'Native care-plan dock must open at a readable width');
    await capture(desktop, '03-native-plan.png');
    const mobile = await page(390, 844, true);
    await evaluate(mobile, `(()=>{if(document.querySelector('textarea[aria-label="育儿计划内容"]')){const close=[...document.querySelectorAll('button[aria-label]')].find(b=>b.getAttribute('aria-label').includes('关闭')&&b.getBoundingClientRect().width>0);if(!close)throw new Error('Native plan tab close control unavailable');close.click()}})()`);
    await waitFor(mobile, 'Boolean(document.querySelector("[contenteditable=true]"))');
    await evaluate(mobile, `(()=>{const b=[...document.querySelectorAll('button,[role=tab]')].find(b=>b.innerText.trim()==='梦漓'&&b.getAttribute('aria-label')!=='梦漓菜单');if(b)b.click()})()`);
    await delay(500);
    report.mobile = await evaluate(mobile, '({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})');
    assert.equal(report.mobile.width, report.mobile.scrollWidth);
    await capture(mobile, '03-native-mobile.png');
    assert.equal(report.errors.length, 0, report.errors.join('\n'));
    assert(report.requests.every(request => request.inline || request.url.startsWith(origin)), 'No external model requests allowed');
    assert(!report.requests.some(request => request.url.endsWith('/api/session/prompt')), 'Dreamli input must not use the model prompt endpoint');
  }
  report.status = process.argv.includes('--inspect') ? 'inspected' : 'passed';
} catch (error) {
  report.status = 'failed'; report.error = error.message;
  throw error;
} finally {
  socket?.close();
  if (browser.exitCode === null) { const exited = once(browser, 'exit'); browser.kill(); await exited; }
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch { report.browser_profile_retained = profile; }
  await fixture?.close();
  fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ status: report.status, screenshots: report.screenshots, error: report.error, errors: report.errors }));
}
