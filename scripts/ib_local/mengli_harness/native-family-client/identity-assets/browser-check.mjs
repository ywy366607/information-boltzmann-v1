// Isolated native-app acceptance. All data is explicitly fixture-only; no model/body calls.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const production = process.argv.includes('--production');
const fixture = production ? { bridge: 'http://127.0.0.1:8902', requests: [], close: async () => {} }
  : await (await import('../../../../../tests/fixtures/mengli_native_browser.mjs')).startNativeBrowserFixture();
let loginKey = production ? process.env.MENGLI_BROWSER_LOGIN_KEY : fixture.loginKey;
delete process.env.MENGLI_BROWSER_LOGIN_KEY;
if (!loginKey) throw new Error('Set MENGLI_BROWSER_LOGIN_KEY for the authenticated browser check');
const family = await (await fetch(fixture.bridge + '/api/family')).json();
const url = family.workbench.url;
if (!production) assert(!['8900', '8902'].includes(new URL(url).port));
const output = path.resolve(process.env.MENGLI_BROWSER_OUTPUT_ROOT || 'results/mengli_native_identity', String(Date.now()));
fs.mkdirSync(output, { recursive: true });
const profile = path.resolve(output, 'edge-profile');
const browser = spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--disable-sync', '--disable-extensions', '--renderer-process-limit=1',
  '--js-flags=--max-old-space-size=96', '--disk-cache-size=1048576', '--media-cache-size=1048576',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
const report = { origin: new URL(url).origin, scope: production ? 'saved real source, UI only; zero body loads or model submissions' : 'offline native fixture; zero body loads or model submissions', checks: [], errors: [], requests: [] };
let socket;
try {
  const debuggerURL = await new Promise((resolve, reject) => {
    let text = '';
    const timer = setTimeout(() => reject(new Error('Edge debugger timeout')), 15000);
    browser.stderr.on('data', chunk => {
      text += chunk;
      const match = text.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
    browser.once('error', reject);
  });
  socket = new WebSocket(debuggerURL);
  await once(socket, 'open');
  let sequence = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const value = JSON.parse(event.data);
    if (value.id) {
      const waiter = pending.get(value.id); pending.delete(value.id);
      value.error ? waiter.reject(new Error(JSON.stringify(value.error))) : waiter.resolve(value.result);
    } else if (value.method === 'Runtime.exceptionThrown') report.errors.push(value.params.exceptionDetails.exception?.description || value.params.exceptionDetails.text);
    else if (value.method === 'Network.requestWillBeSent') {
      const request = new URL(value.params.request.url);
      report.requests.push(['data:', 'blob:'].includes(request.protocol) ? request.protocol : request.origin + request.pathname);
    }
  });
  const command = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++sequence; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params, sessionId }));
  });
  const evaluate = async (sessionId, expression) => {
    const value = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }, sessionId);
    if (value.exceptionDetails) throw new Error(value.exceptionDetails.exception?.description || value.exceptionDetails.text);
    return value.result.value;
  };
  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
  const ready = async (sessionId, expression, timeout = 15000) => {
    for (let i = 0; i < timeout / 100; i++) { if (await evaluate(sessionId, expression)) return; await wait(100); }
    throw new Error('Native UI readiness failed: ' + expression);
  };
  const capture = async (id, filename) => {
    const { data } = await command('Page.captureScreenshot', { format: 'png' }, id);
    fs.writeFileSync(path.join(output, filename), Buffer.from(data, 'base64'));
  };
  const targets = new Map();
  async function tab(reduced = false, mobile = false) {
    const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true });
    targets.set(sessionId, targetId);
    for (const domain of ['Page', 'Runtime', 'Network']) await command(domain + '.enable', {}, sessionId);
    await command('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' },
      { name: 'prefers-reduced-motion', value: reduced ? 'reduce' : 'no-preference' }] }, sessionId);
    await command('Emulation.setDeviceMetricsOverride', { width: mobile ? 390 : 1440, height: mobile ? 844 : 960, deviceScaleFactor: 1, mobile }, sessionId);
    await command('Page.navigate', { url }, sessionId);
    await ready(sessionId, 'Boolean(document.getElementById("key") || window.MengliIntro)');
    if (await evaluate(sessionId, 'Boolean(document.getElementById("key"))')) {
      assert(await evaluate(sessionId, `!document.getElementById('root') && !window.__DSH_BOOT__ && location.search===''`));
      await capture(sessionId, mobile ? '00-login-mobile.png' : '00-login-desktop.png');
      if (!mobile) {
        await evaluate(sessionId, `document.getElementById('key').value='incorrect-offline-key';document.querySelector('form').requestSubmit()`);
        await ready(sessionId, 'document.getElementById("error").textContent.includes("密钥不正确")');
      }
      await evaluate(sessionId, `document.getElementById('key').value=${JSON.stringify(loginKey)};document.querySelector('form').requestSubmit()`);
    }
    await ready(sessionId, 'Boolean(window.MengliIntro && document.getElementById("root"))');
    assert(await evaluate(sessionId, 'document.body.hasAttribute("data-ds-dark-theme") && location.search === ""'));
    return sessionId;
  }
  const desktop = await tab();
  await ready(desktop, 'Boolean(document.getElementById("mengli-intro"))');
  assert(await evaluate(desktop, 'document.getElementById("root").inert'));
  assert(await evaluate(desktop, 'document.getElementById("mengli-intro").contains(document.activeElement)'));
  await command('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 }, desktop);
  assert(await evaluate(desktop, 'document.activeElement.classList.contains("intro-skip")'));
  await evaluate(desktop, 'window.originalCore=document.getElementById("mengli-life-core");window.entranceStarted=performance.now()');
  await capture(desktop, '01-seed.png');
  const fullOpening = !process.argv.includes('--quick');
  if (!fullOpening) {
    await wait(900);
    await capture(desktop, '02-name.png');
    await evaluate(desktop, 'MengliIntro.skip()');
    report.skipped = ['Full 45-second playback; quick mode exercises login/account and native UI'];
  } else {
  await wait(2400);
  await capture(desktop, '02-name.png');
  assert(await evaluate(desktop, 'getComputedStyle(document.getElementById("mengli-intro")).backgroundColor === "rgb(0, 0, 0)"'));
  await wait(1600);
  report.wind = await evaluate(desktop, `Array.from(document.querySelectorAll('.is-current .ml-poem-fragment')).slice(0,3).map(n=>({opacity:Number(getComputedStyle(n).opacity),transform:getComputedStyle(n).transform}))`);
  assert(report.wind.some(value => value.opacity > 0 && value.transform !== 'none'));
  assert(new Set(report.wind.map(value => value.transform)).size > 1);
  await capture(desktop, '03-poem-wind.png');
  await ready(desktop, 'document.getElementById("mengli-intro").dataset.poemGroup === "1"');
  await wait(900);
  await capture(desktop, '03-language-begins.png');
  await ready(desktop, 'document.getElementById("mengli-intro").dataset.poemGroup === "8"', 40000);
  await wait(1000);
  await capture(desktop, '04-final-stanza.png');
  await ready(desktop, 'window.MengliIntro.entered');
  report.auto_enter_ms = await evaluate(desktop, 'performance.now()-entranceStarted');
  assert(report.auto_enter_ms >= 44000 && report.auto_enter_ms < 46500);
  }
  await wait(350);
  await capture(desktop, '04-core-handoff.png');
  await ready(desktop, '!document.getElementById("mengli-intro")');
  await ready(desktop, 'document.getElementById("mengli-life-core").dataset.position === "panel"');
  assert(await evaluate(desktop, 'originalCore===document.getElementById("mengli-life-core")'));
  await wait(1200);
  report.docked = await evaluate(desktop, `(()=>{const core=originalCore.getBoundingClientRect(),dock=document.querySelector('[data-mengli-core-dock]').getBoundingClientRect();return {same:originalCore===document.getElementById('mengli-life-core'),distance:Math.hypot(core.x+core.width/2-dock.x-dock.width/2,core.y+core.height/2-dock.y-dock.height/2),width:core.width}})()`);
  assert(report.docked.same && report.docked.distance < 2 && report.docked.width >= 300);
  await capture(desktop, '05-default-life-panel.png');
  const telemetryBefore = report.requests.filter(value => new URL(value).pathname === '/api/family/telemetry').length;
  assert(await evaluate(desktop, `(()=>{const close=[...document.querySelectorAll('button')].find(n=>/^(Close tab|关闭标签页|Close|关闭)$/.test(n.getAttribute('aria-label')||n.title||'')&&n.closest('[data-dockkit-tab]')?.textContent.includes('生命体征'));if(!close)return false;close.click();return true})()`));
  await ready(desktop, 'originalCore.dataset.position === "park"');
  await evaluate(desktop, 'MengliIntro.replay();MengliIntro.skip()');
  await ready(desktop, '!document.getElementById("mengli-intro")');
  assert(await evaluate(desktop, 'originalCore.dataset.position === "park"'));
  await evaluate(desktop, 'document.querySelector("[aria-label=打开生命体征]").click()');
  await ready(desktop, 'originalCore.dataset.position === "panel"');
  assert.equal(report.requests.filter(value => new URL(value).pathname === '/api/family/telemetry').length, telemetryBefore);
  assert(await evaluate(desktop, 'originalCore===document.getElementById("mengli-life-core")'));
  report.checks.push(fullOpening ? 'pure-black nine-stanza opening, five seconds each, measured wind fragments and 45-second auto entry; same core dock, native close/park/reopen without refetch' : 'manual entrance handoff in quick mode; same core dock and native close/park/reopen without refetch');
  assert(await evaluate(desktop, '!document.getElementById("root").inert'));
  await ready(desktop, 'Boolean(document.querySelector("[contenteditable=true]"))');
  await evaluate(desktop, 'window.identityEditor=document.querySelector("[contenteditable=true]");identityEditor.focus()');
  await command('Input.insertText', { text: '未发送的原生草稿' }, desktop);
  await evaluate(desktop, 'MengliIntro.replay()');
  await ready(desktop, 'Boolean(document.getElementById("mengli-intro"))');
  await command('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 }, desktop);
  await ready(desktop, '!document.getElementById("mengli-intro")');
  assert(await evaluate(desktop, 'document.activeElement===identityEditor && identityEditor.textContent.includes("未发送的原生草稿")'));
  report.checks.push('Esc restores native root; replay returns exact editor focus and preserves unsent draft');
  await evaluate(desktop, 'document.querySelector("[aria-label=梦漓菜单]").click()');
  await ready(desktop, 'Boolean([...document.querySelectorAll("[role=menuitem]")].find(n=>n.innerText.trim()==="重播开场"))');
  await evaluate(desktop, '[...document.querySelectorAll("[role=menuitem]")].find(n=>n.innerText.trim()==="重播开场").click()');
  await ready(desktop, 'Boolean(document.getElementById("mengli-intro"))');
  await evaluate(desktop, `window.fixtureCard={schema:1,identity:{name:'梦漓',individual_id:'OFFLINE_UI_FIXTURE'},branch_id:'offline-ui',source:{cycle:1,sha256:'fixture-only',checkpoint:'NOT_A_BODY'},age:{cycles:1,byte_events:128},memory:{occupied:2},latest:{updates:0}};MengliIntro.updateLife(fixtureCard)`);
  assert(await evaluate(desktop, '!document.getElementById("mengli-intro").classList.contains("is-learning")'));
  assert(await evaluate(desktop, '!document.getElementById("mengli-intro").innerText.includes("第1次诞生")'));
  await evaluate(desktop, 'MengliIntro.updateLife({...fixtureCard,source:{...fixtureCard.source,cycle:2},age:{cycles:2},latest:{updates:0,momentum_norms:[0,null,6]}})');
  assert(await evaluate(desktop, 'document.getElementById("mengli-life-core").dataset.learning === "false"'));
  await evaluate(desktop, 'MengliIntro.updateLife({...fixtureCard,source:{...fixtureCard.source,cycle:3},age:{cycles:3},latest:{updates:1,momentum_norms:Array(8).fill(12)},dynamics:{frequency:.8}},{committed:true})');
  assert(await evaluate(desktop, 'document.getElementById("mengli-life-core").dataset.learning === "true"'));
  const continuous = await evaluate(desktop, '({same:originalCore===document.getElementById("mengli-life-core"),phase:Number(originalCore.dataset.phase)})');
  await wait(200);
  assert(await evaluate(desktop, `originalCore===document.getElementById('mengli-life-core')&&Number(originalCore.dataset.phase)>${continuous.phase}&&originalCore.dataset.layers==='8'&&originalCore.dataset.mean==='12'`));
  await evaluate(desktop, 'MengliIntro.updateLife({...fixtureCard,source:{...fixtureCard.source,cycle:4},age:{cycles:4},latest:{updates:1,momentum_norms:Array.from({length:32},(_,i)=>i+1)},dynamics:{frequency:Array.from({length:32},(_,i)=>Array(8).fill(.7+i*.01))}})');
  assert(await evaluate(desktop, 'document.querySelectorAll(".ml-nebula-layer").length===32 && Math.abs(Number(document.querySelector(".ml-nebula-layer[data-layer=\\"32\\"]").dataset.frequency)-1.01)<1e-12'));
  await wait(250);
  assert(await evaluate(desktop, 'Number(document.querySelector(".ml-nebula-layer[data-layer=\\"32\\"]").dataset.phase)!==Number(document.querySelector(".ml-nebula-layer[data-layer=\\"1\\"]").dataset.phase)'));
  await capture(desktop, '03-individual-layer-membranes.png');
  report.checks.push('32 independent depth membranes; per-layer P amplitude and nested head-frequency rhythm; preserves phase on telemetry update');
  await evaluate(desktop, 'document.querySelector(".ml-intro-enter").click()');
  await evaluate(desktop, 'MengliIntro.skip()');
  await ready(desktop, '!document.getElementById("mengli-intro")');
  await capture(desktop, '02-native-brand.png');
  const brand = await evaluate(desktop, `({title:document.title,text:document.body.innerText,iframes:document.querySelectorAll('iframe').length,manifest:document.querySelector('link[rel=manifest]').href,mark:[...document.querySelectorAll('img')].filter(n=>n.src.startsWith('data:image/svg+xml')).length})`);
  assert(brand.title.includes('梦漓 MengLi'));
  assert(brand.text.includes('梦漓 MengLi'));
  assert(!brand.text.includes('预览版'));
  assert.equal(brand.iframes, 0);
  assert(brand.mark >= 2);
  assert(brand.manifest.startsWith('data:application/manifest+json'));
  report.brand = brand;
  report.checks.push('native replay menu and central enter; no invented birth ordinal; no read/zero-update learning glow; real-commit-only glow; changing depth preserves core and phase');
  const scene = id => evaluate(id, `(()=>{
    const body=getComputedStyle(document.body), root=document.querySelector('.wSkVaW_root');
    const probe=document.createElement('span');document.body.append(probe);
    const color=token=>{probe.style.color='var('+token+')';return getComputedStyle(probe).color};
    const luminance=rgb=>{const v=rgb.match(/[\\d.]+/g).slice(0,3).map(Number).map(x=>{x/=255;return x<=.04045?x/12.92:((x+.055)/1.055)**2.4});return .2126*v[0]+.7152*v[1]+.0722*v[2]};
    const contrast=(a,b)=>{const x=luminance(a),y=luminance(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05)};
    const colors=Object.fromEntries(['--dsw-alias-bg-base','--dsw-specific-input-major','--dsw-alias-bg-layer-2','--dsw-alias-label-primary','--dsw-alias-label-secondary'].map(t=>[t,color(t)]));
    const ratios=Object.fromEntries(['--dsw-alias-label-primary','--dsw-alias-label-secondary'].flatMap(t=>['--dsw-alias-bg-base','--dsw-specific-input-major','--dsw-alias-bg-layer-2'].map(b=>[t+' / '+b,contrast(colors[t],colors[b])])));
    probe.remove();
    const editor=document.querySelector('[contenteditable=true]'),r=editor.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
    return {dark:document.body.hasAttribute('data-ds-dark-theme'),source:document.documentElement.getAttribute('data-ds-theme-source'),colors,ratios,
      grain:getComputedStyle(root,'::before').backgroundImage,mask:getComputedStyle(root,'::before').maskImage,
      pointerEvents:getComputedStyle(root,'::after').pointerEvents,animation:getComputedStyle(root,'::after').animationName,
      nativeInputHit:editor===hit||editor.contains(hit),overflow:document.documentElement.scrollWidth>innerWidth};
  })()`);
  report.default_dark_scene = await scene(desktop);
  assert(report.default_dark_scene.dark);
  assert.equal(report.default_dark_scene.colors['--dsw-alias-bg-base'], 'rgb(21, 21, 23)');
  assert.equal(report.default_dark_scene.colors['--dsw-specific-input-major'], 'rgb(44, 44, 46)');
  await capture(desktop, '04-default-dark-native-scene.png');
  // Palette inspection only; do not overwrite the user's saved theme preference.
  await evaluate(desktop, 'document.body.removeAttribute("data-ds-dark-theme")');
  report.light_scene = await scene(desktop);
  assert(!report.light_scene.dark);
  assert(report.light_scene.grain.includes('repeating-linear-gradient'));
  assert.equal(report.light_scene.pointerEvents, 'none');
  assert.equal(report.light_scene.animation, 'none');
  assert(report.light_scene.nativeInputHit);
  assert(Object.values(report.light_scene.ratios).every(value=>value>=4.5), 'Light primary/secondary text contrast');
  await capture(desktop, '04-light-native-scene.png');
  await evaluate(desktop, 'document.body.setAttribute("data-ds-dark-theme", "")');
  await ready(desktop, 'document.body.hasAttribute("data-ds-dark-theme")');
  report.dark_scene = await scene(desktop);
  assert(Object.values(report.dark_scene.ratios).every(value=>value>=4.5), 'Dark primary/secondary text contrast');
  assert(report.dark_scene.nativeInputHit);
  await capture(desktop, '05-dark-native-scene.png');
  await evaluate(desktop, 'MengliIntro.replay()');
  await ready(desktop, 'Boolean(document.getElementById("mengli-intro"))');
  await wait(3800);
  await capture(desktop, '06-dark-entrance.png');
  await evaluate(desktop, 'MengliIntro.skip()');
  await ready(desktop, '!document.getElementById("mengli-intro")');
  report.checks.push('native default-dark under system light, official neutral surfaces with jade edge glow; DOM-only light/dark palette inspection preserves saved preference, input hit and contrast >=4.5');
  await command('Page.reload', {}, desktop);
  await ready(desktop, 'Boolean(window.MengliIntro && document.querySelector("[aria-label=梦漓菜单]"))');
  await ready(desktop, 'Boolean(document.querySelector("#mengli-intro .ml-poem-stage"))');
  assert(await evaluate(desktop, `document.querySelectorAll('.ml-poem-group').length===9 && document.querySelectorAll('.ml-poem-line').length===16 && document.querySelector('.ml-poem-line').textContent==='不要温柔地走进于那个良夜' && [...document.querySelectorAll('.ml-poem-line')].at(-1).textContent==='活下去！' && document.querySelector('.ml-poem-author').textContent==='——心智NOUS解放阵线' && !document.querySelector('.ml-intro-footer,.ml-poem-title,.ml-poem-dedication')`));
  await capture(desktop, '09-refresh-poem.png');
  await evaluate(desktop, 'MengliIntro.skip()');
  await ready(desktop, '!document.getElementById("mengli-intro")');
  await evaluate(desktop, 'window.dispatchEvent(new PageTransitionEvent("pageshow",{persisted:true}))');
  await ready(desktop, 'Boolean(document.getElementById("mengli-intro"))');
  await evaluate(desktop, 'MengliIntro.skip()');
  await ready(desktop, '!document.getElementById("mengli-intro")');
  const lifeRequests = report.requests.filter(value => new URL(value).pathname === '/api/family/life');
  assert.equal(lifeRequests.length, 1);
  report.checks.push('refresh replays entrance; restored pageshow replays; original 16 poem lines and signature preserved; exactly one initial life GET per tab');
  const mobile = await tab(true, true);
  await ready(mobile, 'Boolean(document.getElementById("mengli-intro"))');
  assert(await evaluate(mobile, 'document.getElementById("mengli-intro").classList.contains("is-reduced")'));
  await wait(900);
  assert(await evaluate(mobile, `getComputedStyle(document.querySelector('.is-current .ml-poem-fragment')).animationName==='none' && getComputedStyle(document.getElementById('mengli-intro')).backgroundColor==='rgb(0, 0, 0)'`));
  await capture(mobile, '03-reduced-mobile-entrance.png');
  await evaluate(mobile, 'MengliIntro.skip()');
  await ready(mobile, '!document.getElementById("mengli-intro")');
  assert(await evaluate(mobile, '!document.getElementById("root").inert && document.documentElement.scrollWidth===innerWidth'));
  await ready(mobile, 'document.querySelector("[data-mengli-metrics=life]")?.getBoundingClientRect().width > 0 && document.getElementById("mengli-life-core").dataset.position === "panel"');
  const staticPhase = await evaluate(mobile, 'document.getElementById("mengli-life-core").dataset.phase');
  await wait(200);
  assert.equal(await evaluate(mobile, 'document.getElementById("mengli-life-core").dataset.phase'), staticPhase);
  assert(await evaluate(mobile, 'document.body.hasAttribute("data-ds-dark-theme")'));
  await capture(mobile, '07-default-dark-mobile.png');
  await evaluate(mobile, 'document.body.removeAttribute("data-ds-dark-theme")');
  await capture(mobile, '07-light-mobile-scene.png');
  await evaluate(mobile, 'document.body.setAttribute("data-ds-dark-theme", "")');
  await ready(mobile, 'document.body.hasAttribute("data-ds-dark-theme")');
  await capture(mobile, '08-dark-mobile-scene.png');
  report.checks.push('pure-black reduced-motion mobile entrance: soft text without wind, manual enter restores root, static core and no overflow; light/dark workspace screenshots');
  await command('Target.activateTarget', { targetId: targets.get(desktop) });
  await evaluate(desktop, 'document.querySelector("[aria-label=账号设置]").click()');
  await ready(desktop, 'Boolean(document.querySelector("input[aria-label=显示名]"))');
  const fillField = async (label, text) => {
    await evaluate(desktop, `(()=>{const input=document.querySelector('input[aria-label="${label}"]');input.focus();input.select()})()`);
    await command('Input.insertText', { text }, desktop);
  };
  if (!production) {
    await fillField('显示名', '预览家长');
    await evaluate(desktop, '[...document.querySelectorAll("button")].find(n=>n.textContent==="保存显示名").click()');
    await ready(desktop, 'document.body.innerText.includes("显示名已保存")');
    assert(await evaluate(desktop, 'document.querySelector("[aria-label=账号设置]").textContent === "预览家长"'));
    await fillField('当前密钥', loginKey);
    await fillField('新密钥', 'offline-preview-changed-key');
    await fillField('再次输入新密钥', 'offline-preview-changed-key');
    await evaluate(desktop, '[...document.querySelectorAll("button")].find(n=>n.textContent==="保存新密钥").click()');
    await ready(desktop, 'document.body.innerText.includes("密钥已更改")');
    loginKey = 'offline-preview-changed-key';
  }
  await capture(desktop, '10-native-account-settings.png');
  await evaluate(desktop, '[...document.querySelectorAll("button")].find(n=>n.textContent==="退出登录").click()');
  await ready(desktop, 'Boolean(document.getElementById("key"))');
  assert(await evaluate(desktop, '!window.MengliIntro && location.search === ""'));
  await capture(desktop, '11-logged-out.png');
  await evaluate(desktop, `document.getElementById('key').value=${JSON.stringify(loginKey)};document.querySelector('form').requestSubmit()`);
  await ready(desktop, 'Boolean(window.MengliIntro)');
  await evaluate(desktop, 'MengliIntro.skip()');
  report.checks.push(production ? 'native account panel, logout and password re-login without changing the real name/key' : 'native display-name update, key change, logout and new-key re-login; no model/body commands');
  assert.equal(report.errors.length, 0, report.errors.join('\n'));
  assert.equal(fixture.requests.length, 0, 'No body loads or worker commands');
  assert(!report.requests.some(value => /\/api\/session\/prompt/.test(value)));
  report.status = 'passed';
} catch (error) { report.status = 'failed'; report.error = error.stack; throw error; }
finally {
  socket?.close();
  if (browser.exitCode === null) { const exit = once(browser, 'exit'); browser.kill(); await exit; }
  await fixture.close();
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
  fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, output, checks: report.checks, error: report.error, errors: report.errors }));
}
