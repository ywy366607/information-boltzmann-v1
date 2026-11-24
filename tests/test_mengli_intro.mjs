import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../scripts/ib_local/mengli_harness/web/intro.js', import.meta.url), 'utf8');

function fixture({ reduced = false, storage = new Map(), blockedStorage = false, stylesheet = true, life = null } = {}) {
  let document;
  class Element {
    constructor(tag) {
      this.tagName = tag; this.children = []; this.parentNode = null; this.className = '';
      this.dataset = {}; this.attributes = {}; this.inert = false; this.handlers = new Map();
      const properties = new Map();
      this.style = { setProperty: (key, value) => properties.set(key, value), getPropertyValue: key => properties.get(key) };
      this.classList = { add: name => {
        this.className = [...new Set([...this.className.split(' ').filter(Boolean), name])].join(' ');
      }, remove: name => { this.className = this.className.split(' ').filter(value => value !== name).join(' '); },
      contains: name => this.className.split(' ').includes(name) };
    }
    appendChild(node) { this.children.push(node); node.parentNode = this; return node; }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener(name, handler) { this.handlers.set(name, handler); }
    removeEventListener(name) { this.handlers.delete(name); }
    click() { this.handlers.get('click')?.({}); }
    remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(node => node !== this); this.parentNode = null; }
    get isConnected() { return this === document.body || Boolean(this.parentNode?.isConnected); }
    focus() { document.activeElement = this; }
  }
  const walk = node => [node, ...node.children.flatMap(walk)];
  const handlers = new Map();
  document = {
    readyState: 'complete', createElement: tag => new Element(tag),
    addEventListener: (name, callback) => handlers.set(name, callback),
    removeEventListener: name => handlers.delete(name),
  };
  document.body = new Element('body');
  document.activeElement = document.body;
  const header = document.body.appendChild(new Element('header'));
  const brand = header.appendChild(new Element('div')); brand.className = 'mengli-brand-mark';
  const lifeName = header.appendChild(new Element('span')); lifeName.id = 'life-name';
  const lifeReading = header.appendChild(new Element('span')); lifeReading.id = 'life-reading';
  const workspace = document.body.appendChild(new Element('div'));
  workspace.className = 'workspace';
  const replay = header.appendChild(new Element('button'));
  replay.id = 'intro-replay'; replay.hidden = true;
  const styles = { sheet: stylesheet ? {} : null };
  document.getElementById = id => id === 'mengli-intro-styles' ? styles : walk(document.body).find(node => node.id === id) ?? null;
  document.querySelector = selector => selector === 'header' ? header : selector === '.workspace' ? workspace : selector === '.mengli-brand-mark' ? brand : null;
  let now = 0, next = 0;
  const timers = new Map(), events = [];
  const window = {
    __mengliLife: life,
    setTimeout: (callback, delay) => { const id = ++next; timers.set(id, { at: now + delay, callback }); return id; },
    clearTimeout: id => timers.delete(id),
    matchMedia: () => ({ matches: reduced }),
    sessionStorage: {
      getItem(key) { if (blockedStorage) throw new Error('blocked'); return storage.get(key) ?? null; },
      setItem(key, value) { if (blockedStorage) throw new Error('blocked'); storage.set(key, value); },
    },
    dispatchEvent: event => events.push(event),
  };
  vm.runInNewContext(source, { window, document, CustomEvent: class {
    constructor(type, options) { this.type = type; this.detail = options.detail; }
  } });
  return { window, document, header, workspace, replay, events, timers, storage, brand,
    root: () => document.getElementById('mengli-intro'),
    key: value => handlers.get('keydown')?.({ key: value, preventDefault() {} }),
    async advance(duration) {
      const end = now + duration;
      while (true) {
        const due = [...timers].filter(([, task]) => task.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        const [id, task] = due;
        timers.delete(id); now = task.at; task.callback();
        await Promise.resolve(); await Promise.resolve();
      }
      now = end;
    },
  };
}

test('four visual stages complete without touching dashboard or model state', async () => {
  const f = fixture();
  assert.equal(f.root().dataset.phase, 'seed');
  assert.equal(f.header.inert, true);
  assert.equal(f.workspace.inert, true);
  await f.advance(900); assert.equal(f.root().dataset.phase, 'unfurl');
  await f.advance(1900); assert.equal(f.root().dataset.phase, 'name');
  await f.advance(1900); assert.equal(f.root().dataset.phase, 'handoff');
  await f.advance(1500); assert.equal(f.root().classList.contains('is-leaving'), true);
  await f.advance(650);
  assert.equal(f.root(), null);
  assert.equal(f.header.inert, false); assert.equal(f.workspace.inert, false);
  assert.equal(f.document.activeElement, f.replay);
  assert.equal(f.events[0].detail.reason, 'complete');
  assert.equal(f.timers.size, 0);
});

test('Escape and the skip button both release the curtain and all waits', async () => {
  for (const mode of ['escape', 'button']) {
    const f = fixture();
    if (mode === 'escape') f.key('Escape');
    else f.root().children.find(node => node.className === 'intro-bottom').children[1].click();
    await f.advance(650);
    assert.equal(f.root(), null); assert.equal(f.timers.size, 0);
    assert.equal(f.events[0].detail.reason, 'skipped');
    assert.equal(f.workspace.inert, false);
  }
});

test('same-tab refresh skips automatically, manual replay remains available', async () => {
  const first = fixture();
  first.key('Escape'); await first.advance(650);
  const second = fixture({ storage: first.storage });
  assert.equal(second.root(), null);
  assert.equal(second.replay.hidden, false);
  second.replay.click();
  assert.equal(second.root().dataset.phase, 'seed');
  second.window.MengliIntro.skip(); await second.advance(650);
  assert.equal(second.root(), null);
});

test('replaying mid-entrance replaces the old instance without leaving background inert', async () => {
  const f = fixture();
  await f.advance(1200);
  const old = f.root();
  f.window.MengliIntro.replay();
  assert.notEqual(f.root(), old); assert.equal(old.isConnected, false);
  await f.advance(7000);
  assert.equal(f.root(), null); assert.equal(f.events.length, 1);
  assert.equal(f.workspace.inert, false); assert.equal(f.timers.size, 0);
});

test('reduced motion is a brief static identity, with no long timeline', async () => {
  const f = fixture({ reduced: true });
  assert.equal(f.root().dataset.phase, 'name');
  assert.equal(f.root().classList.contains('is-reduced'), true);
  await f.advance(760);
  assert.equal(f.root(), null); assert.equal(f.timers.size, 0);
});

test('blocked browser storage or missing styles cannot lock the workbench', async () => {
  const blocked = fixture({ blockedStorage: true });
  await blocked.advance(7000);
  assert.equal(blocked.root(), null); assert.equal(blocked.workspace.inert, false);
  const missing = fixture({ stylesheet: false });
  assert.equal(missing.root(), null); assert.equal(missing.timers.size, 0);
  assert.equal(missing.workspace.inert, false); assert.equal(missing.replay.hidden, true);
});

function card(cycle = 3509, branch = 'fixture-branch', updates = 4) {
  return { schema: 1, identity: { name: '梦漓', individual_id: 'fixture-individual' }, branch_id: branch,
    source: { checkpoint: 'offline-fixture.pt.gz', sha256: String(cycle), cycle },
    age: { cycles: cycle, byte_events: 4717536, born_at: null, active_seconds: null },
    memory: { occupied: 4096, capacity: 4096, episodes_committed: 4789, oldest_since_write_events: 4715584 },
    latest: { updates, state_change: .408542, momentum_norms: [7.698, 12.869, 37.815, 20.601] } };
}
const withClass = (root, name) => {
  if (root.classList?.contains(name)) return root;
  for (const child of root.children) { const found = withClass(child, name); if (found) return found; }
  return null;
};

test('checkpoint facts and four relative layer magnitudes are exposed without an initial learning pulse', () => {
  const f = fixture({ life: card() });
  assert.equal(f.root().dataset.life, 'bound');
  assert.equal(f.root().dataset.activity, 'saved');
  assert.equal(f.root().dataset.momentum, 'measured');
  assert(withClass(f.root(), 'intro-life-age').textContent.includes('4,717,536'));
  assert(withClass(f.root(), 'intro-life-memory').textContent.includes('4,789'));
  assert(withClass(f.root(), 'intro-life-oldest').textContent.includes('4,715,584'));
  assert(withClass(f.root(), 'intro-life-layers').textContent.includes('7.70 / 12.87 / 37.81 / 20.60'));
  assert.equal(Number(withClass(f.root(), 'orbit-c').style.getPropertyValue('--layer-drive')), 1);
  assert.equal(f.brand.classList.contains('has-life-update'), false);
});

test('only a genuine new committed cycle on the same branch emits an update pulse', async () => {
  const f = fixture({ life: card() });
  f.window.MengliIntro.updateLife(card());
  assert.equal(f.brand.classList.contains('has-life-update'), false);
  f.window.MengliIntro.replay();
  assert.equal(f.brand.classList.contains('has-life-update'), false);
  f.window.MengliIntro.updateLife(card(3510));
  assert.equal(f.brand.classList.contains('has-life-update'), true);
  assert.equal(f.root().classList.contains('is-learning'), true);
  await f.advance(1100);
  assert.equal(f.brand.classList.contains('has-life-update'), false);
  assert.equal(f.root().classList.contains('is-learning'), false);
  f.window.MengliIntro.updateLife(card(9000, 'other-branch'));
  assert.equal(f.brand.classList.contains('has-life-update'), false);
  f.window.MengliIntro.updateLife(card(9001, 'other-branch', 0));
  assert.equal(f.brand.classList.contains('has-life-update'), false);
});

test('zero values remain zero; missing data and names are never interpreted as HTML', () => {
  const value = card(0);
  value.identity.name = '<img src=x onerror=bad()>';
  value.latest.momentum_norms = [0, 0, 0, 0];
  value.latest.updates = 0;
  value.age.byte_events = 0;
  const f = fixture({ life: value });
  assert.equal(f.document.getElementById('mengli-intro-title').textContent, value.identity.name);
  assert(withClass(f.root(), 'intro-life-age').textContent.includes('已历 0 轮 · 0 字节事件'));
  assert(withClass(f.root(), 'intro-life-layers').textContent.includes('0.00 / 0.00 / 0.00 / 0.00'));
  f.window.MengliIntro.updateLife(null);
  assert.equal(withClass(f.root(), 'intro-life').hidden, true);
  assert.equal(f.root().dataset.life, 'unbound');
});

test('reduced-motion users can read true facts without update flashes', () => {
  const f = fixture({ life: card(), reduced: true });
  f.window.MengliIntro.updateLife(card(3510));
  assert.equal(f.brand.classList.contains('has-life-update'), false);
  assert.equal(f.root().classList.contains('is-learning'), false);
  assert(withClass(f.root(), 'intro-life-source').textContent.includes('3,510'));
});
