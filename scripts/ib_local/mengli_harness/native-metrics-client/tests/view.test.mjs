// Lightweight element-tree acceptance; not a browser/layout or visual substitute.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { buildMetricsClient } from '../build.mjs';
import { fixtureCard, fixtureRow } from './fixture-plugin.mjs';

function nativeView(snapshot, { entered = false } = {}) {
  const source = fs.readFileSync(buildMetricsClient().client, 'utf8');
  const states = []; let stateIndex = 0, plugin;
  const registrations = [], slots = [], effects = [], observers = [], opened = [];
  const window = Object.assign(new EventTarget(), { MengliIntro: { entered } });
  const observe = { notifications: 0, reads: 0, lifeStates: [] };
  window.addEventListener('mengli:core-anchor', () => observe.notifications++);
  window.addEventListener('mengli:life-state', event => observe.lifeStates.push(event.detail));
  class ResizeObserver {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(target) { this.target = target; }
    disconnect() { this.disconnected = true; }
  }
  const React = {
    Fragment: 'fragment',
    createElement: (type, props, ...children) => ({ type, props: { ...props, children }, children }),
    useSyncExternalStore: () => snapshot,
    useEffect(fn) { effects.push(fn); },
    useRef(initial) {
      const [ref] = this.useState({ current: initial });
      return ref;
    },
    useState(initial) {
      const index = stateIndex++;
      if (!(index in states)) states[index] = initial;
      return [states[index], value => { states[index] = typeof value === 'function' ? value(states[index]) : value; }];
    },
  };
  const document = { createElement: () => ({ dataset: {}, remove() {} }), head: { appendChild() {} } };
  window.__ModuleLoader__ = { load(value) {
    plugin = value.factory(name => name === 'react' ? React : { Button: 'native-button', Menu: 'native-menu' });
  } };
  vm.runInNewContext(source, { document, window, Event, CustomEvent, ResizeObserver, fetch: async () => {
    observe.reads++;
    return { ok: true, json: async () => ({ source: snapshot.source, life: { card: snapshot.card }, history: { rows: snapshot.rows } }) };
  } });
  plugin.apply({ effect(fn) { fn(); },
    slots: { inject(_name, callback) { callback(); }, register(def, render) { slots.push({ def, render }); } },
    sidebarRightTabs: { register(def) { registrations.push(def); return () => {}; } },
    sidebarRight: { openTab(kind, options) { opened.push({ kind, ...options }); } }, layout: {} });
  function expand(node) {
    if (node === null || node === undefined || node === false) return null;
    if (Array.isArray(node)) return node.flatMap(value => [expand(value)]);
    if (typeof node !== 'object') return String(node);
    if (typeof node.type === 'function') return expand(node.type(node.props));
    if (node.props.ref) node.props.ref.current = node;
    return { ...node, children: node.children.map(expand).flat(Infinity).filter(value => value !== null) };
  }
  function render(kind) {
    stateIndex = 0;
    return expand(slots.find(slot => slot.def.key === '@mengli/native-metrics-client/' + kind).render());
  }
  return { render, registrations, window, opened, observers, observe,
    renderHeader() {
      stateIndex = 0;
      const tree = expand(slots.find(slot => slot.def.id === 'mengli-metrics-open').render());
      // A native Menu renders its anchor, while this element-tree double does not.
      const menu = collect(tree, node => node.type === 'native-menu')[0];
      menu.props.anchor = expand(menu.props.anchor);
      return tree;
    },
    flushEffects() {
      const cleanups = effects.splice(0).map(fn => fn()).filter(fn => typeof fn === 'function');
      return () => cleanups.forEach(fn => fn());
    },
  };
}
const collect = (tree, predicate) => !tree || typeof tree !== 'object' ? [] :
  [...(predicate(tree) ? [tree] : []), ...tree.children.flatMap(child => collect(child, predicate))];
const text = tree => !tree ? '' : typeof tree === 'string' ? tree : tree.children.map(text).join('');
const snapshot = () => ({ loading: false, error: null,
  source: { cycle: 3509, checkpoint: '3509/current.json', branch_id: 'fixture-bound-3509', kind: 'saved_snapshot' },
  card: fixtureCard, rows: [fixtureRow], history: { limited: false, source: '3509/metrics.jsonl' },
  family: { connected: true, state: { cycle: 0, loaded: false, busy: false, mode: 'paused' } } });

test('research observer renders separate completion/save boundaries and a named native progress element', () => {
  const data = snapshot();
  data.source.observation = { latest_completed_cycle: 4506, saved_cycle: 4507, start_cycle: 4208, target_cycle: 4720 };
  data.family.state = { read_only: true, observation: { status: 'running' } };
  const view = nativeView(data), tree = view.render('mengli-learning-metrics');
  assert(text(tree).includes('训练中'));
  assert(text(tree).includes('已完成事件4,506'));
  assert(text(tree).includes('已保存检查点4,507'));
  assert(!text(tree).includes('对话展示记录的世界回应'));
  const progress = collect(tree, node => node.type === 'progress');
  assert.equal(progress.length, 1);
  assert.equal(progress[0].props['aria-label'], '当前研究训练进度');
  assert.equal(progress[0].props.max, 512);
  assert.equal(progress[0].props.value, 298);
  assert.equal(progress[0].props.tabIndex, undefined);
});

test('life native element tree shows measured age, occupancy, write interval and directly-labelled raw norms', () => {
  const view = nativeView(snapshot());
  const tree = view.render('mengli-life-metrics');
  assert(text(tree).includes('3,509'));
  assert(text(tree).includes('456,789'));
  assert(text(tree).includes('213 / 4,096'));
  assert(text(tree).includes('45,632 byte'));
  assert(text(tree).includes('已保存快照'));
  assert.equal(collect(tree, node => node.props.className === 'mm-bar-row').length, 4);
  const bars = collect(tree, node => node.props.className === 'mm-bar');
  assert.equal(bars[0].props.style.width, '0%');
  assert.equal(bars.at(-1).props.style.width, '100%');
  assert.equal(collect(tree, node => ['iframe', 'canvas'].includes(node.type)).length, 0);
});

test('learning single record and data table use concise labels while preserving missing values', () => {
  const view = nativeView(snapshot());
  let tree = view.render('mengli-learning-metrics');
  assert(text(tree).includes('本轮更新4'));
  assert(!text(tree).includes('仅本轮'));
  assert.equal(collect(tree, node => node.type === 'svg').length, 0);
  assert.equal(collect(tree, node => node.props.className === 'mm-point').length, 0);
  assert(!text(tree).includes('暂无趋势'));
  const record = collect(tree, node => node.props['data-metric'] === 'response_nll')[0];
  assert(record.props.title.includes('1.4523'));
  assert(text(record).includes('cycle 3509 · NLL'));
  const button = collect(tree, node => node.type === 'native-button' && text(node) === '数据表')[0];
  assert.equal(button.props['aria-expanded'], false);
  button.props.onClick();
  tree = view.render('mengli-learning-metrics');
  assert.equal(collect(tree, node => node.type === 'table').length, 1);
  assert(text(tree).includes('—'));
  assert(!text(tree).includes('response_nll'));
  assert(!text(tree).includes('完整原始记录'));
});

test('missing values never produce fake points or empty-card grids; zero is a real measurement', () => {
  const data = snapshot();
  data.rows = [{ cycle: 3509, updates: 0, response_nll: null, progress: 'unknown' }];
  let view = nativeView(data);
  let tree = view.render('mengli-learning-metrics');
  assert(text(tree).includes('本轮更新0'));
  assert.equal(collect(tree, node => node.props.className === 'mm-point').length, 0);
  assert.equal(collect(tree, node => node.props.className === 'mm-heroes')[0].children.length, 2);
  data.rows[0].response_nll = 0;
  view = nativeView(data); tree = view.render('mengli-learning-metrics');
  assert.equal(collect(tree, node => node.type === 'svg').length, 0);
  assert.equal(collect(tree, node => node.props['data-metric'] === 'response_nll').length, 1);
});

test('measured samples use an unsmoothed line and accessible keyboard readout', () => {
  const data = snapshot();
  data.rows = [fixtureRow, { ...fixtureRow, cycle: 3510, response_nll: 0, joint_progress: null }];
  const view = nativeView(data);
  let tree = view.render('mengli-learning-metrics');
  assert.equal(collect(tree, node => node.type === 'svg').length, 1);
  assert.equal(collect(tree, node => node.props.className === 'mm-point').length, 1);
  const series = collect(tree, node => node.props.className === 'mm-series')[0];
  assert.match(series.props.d, /^M[\d.,]+ L[\d.,]+$/);
  assert(!text(tree).includes('缺测处断开'));
  assert.equal(collect(tree,node=>node.type==='input'&&node.props.type==='number').length,4);
  const hit = collect(tree, node => node.props.className === 'mm-hit')[0];
  assert.equal(hit.props.tabIndex, 0);
  assert(hit.props['aria-label'].includes('1.4523'));
  hit.props.onFocus();
  tree = view.render('mengli-learning-metrics');
  assert(text(tree).includes('cycle 3509'));
});

test('identity entry opens the native life tab once after header mount, not during plugin apply', () => {
  const view = nativeView(snapshot());
  assert.equal(view.registrations.length, 2);
  assert.equal(view.opened.length, 0);
  view.renderHeader();
  assert.equal(view.opened.length, 0);
  const unmount = view.flushEffects();
  assert.equal(view.opened.length, 0);
  view.window.dispatchEvent(new Event('mengli:enter'));
  assert.deepEqual(view.opened, [{ kind: 'mengli-life-metrics', destination: 'sidebar' }]);
  // Native close is not intercepted. Repeated entry/replay must not reopen it.
  view.window.dispatchEvent(new Event('mengli:enter'));
  assert.equal(view.opened.length, 1);
  unmount();
  view.window.MengliIntro.entered = true;
  view.renderHeader();
  const dispose = view.flushEffects();
  assert.equal(view.opened.length, 1);
  dispose();
});

test('already-entered and no-playback startup opens on the first native header effect', () => {
  const view = nativeView(snapshot(), { entered: true });
  assert.equal(view.opened.length, 0);
  view.renderHeader();
  const unmount = view.flushEffects();
  assert.deepEqual(view.opened, [{ kind: 'mengli-life-metrics', destination: 'sidebar' }]);
  assert.equal(view.observe.reads, 0);
  view.render('mengli-life-metrics');
  const closePanel = view.flushEffects();
  assert.equal(view.observe.reads, 1);
  closePanel(); unmount();
  view.render('mengli-life-metrics');
  const reopenPanel = view.flushEffects();
  assert.equal(view.observe.reads, 1);
  reopenPanel();
});

test('manual life open consumes automatic entry; entry listeners are removed on unmount', () => {
  const view = nativeView(snapshot());
  const tree = view.renderHeader();
  const unmount = view.flushEffects();
  collect(tree, node => node.props['aria-label'] === '打开生命体征')[0].props.onClick();
  view.window.dispatchEvent(new Event('mengli:enter'));
  assert.equal(view.opened.length, 1);
  unmount();
  const unused = nativeView(snapshot());
  unused.renderHeader();
  unused.flushEffects()();
  unused.window.dispatchEvent(new Event('mengli:enter'));
  assert.equal(unused.opened.length, 0);
});

test('life opened through the native guide is not reopened by a later identity entry', () => {
  const view = nativeView(snapshot());
  view.renderHeader();
  const unmountHeader = view.flushEffects();
  view.render('mengli-life-metrics');
  const closePanel = view.flushEffects();
  closePanel();
  view.window.dispatchEvent(new Event('mengli:enter'));
  assert.equal(view.opened.length, 0);
  unmountHeader();
});

test('telemetry forwards changed card references to identity without another GET or a fake commit', async () => {
  const data = snapshot();
  const view = nativeView(data);
  const listeners = new Set();
  let family = { connected: true, state: { events: [] } };
  view.window.__MENGLI_NATIVE_FAMILY__ = { controller: {
    getSnapshot: () => family,
    subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  } };
  view.render('mengli-life-metrics');
  view.flushEffects();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(view.observe.reads, 1);
  assert.equal(view.observe.lifeStates.length, 1);
  assert.equal(view.observe.lifeStates[0].card, data.card);
  assert.deepEqual(Object.keys(view.observe.lifeStates[0]), ['card']);
  // A family state publish and another panel mount do not republish the same card.
  for (const listener of listeners) listener();
  view.render('mengli-learning-metrics');
  view.flushEffects();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(view.observe.reads, 1);
  assert.equal(view.observe.lifeStates.length, 1);
  const nextCard = { ...data.card, source: { cycle: 3510 } };
  family = { connected: true, state: { events: [{ status: 'committed', cursor: 1, cycle: 3510,
    branch_id: data.source.branch_id, life_card_data: nextCard }] } };
  for (const listener of listeners) listener();
  assert.equal(view.observe.lifeStates.length, 2);
  assert.equal(view.observe.lifeStates[1].card, nextCard);
});

test('empty dock is beside life h2; native desktop and mobile buttons keep empty park anchors', () => {
  const view = nativeView(snapshot());
  const life = view.render('mengli-life-metrics');
  const title = collect(life, node => node.props.className === 'mm-title-row')[0];
  assert.equal(title.children[0].type, 'h2');
  assert.equal(title.children[1].props['data-mengli-core-dock'], '');
  assert.equal(title.children[1].props['aria-hidden'], true);
  assert.equal(title.children[1].children.length, 0);
  assert.equal(collect(view.render('mengli-learning-metrics'), node => 'data-mengli-core-dock' in node.props).length, 0);
  // The native header is a separate React root, with its own hook storage.
  const header = nativeView(snapshot()).renderHeader();
  const desktop = collect(header, node => node.props['aria-label'] === '打开生命体征')[0];
  const mobile = collect(header, node => node.type === 'native-menu')[0].props.anchor;
  for (const button of [desktop, mobile]) {
    const park = collect(button, node => 'data-mengli-core-park' in node.props);
    assert.equal(park.length, 1);
    assert.equal(park[0].props['aria-hidden'], true);
    assert.equal(park[0].children.length, 0);
    assert.equal(typeof button.props.onClick, 'function');
  }
  assert.equal(collect(header, node => node.props['aria-label'] === '打开学习训练')[0].children.length, 1);
});

test('anchor events follow mount, observed layout changes and cleanup with no polling', () => {
  const view = nativeView(snapshot());
  view.renderHeader();
  const unmount = view.flushEffects();
  assert.equal(view.observers.length, 2);
  assert.equal(view.observe.notifications, 2);
  for (const observer of view.observers) {
    assert(observer.target);
    observer.callback();
  }
  assert.equal(view.observe.notifications, 4);
  unmount();
  assert(view.observers.every(observer => observer.disconnected));
  assert.equal(view.observe.notifications, 6);
  const source = fs.readFileSync(new URL('../client-source.js', import.meta.url), 'utf8');
  assert(!/setInterval|requestAnimationFrame|MutationObserver/.test(source));
});

test('dynamic layers retain measured zero and missing values without explanatory report text', () => {
  const data = snapshot();
  data.card = { ...fixtureCard, latest: { momentum_norms: [0, null, 2, undefined, 'unknown', 4] } };
  const view = nativeView(data);
  let tree = view.render('mengli-life-metrics');
  assert(text(tree).includes('分层动量'));
  assert(!text(tree).includes('四层动量'));
  assert.equal(collect(tree, node => node.props.className === 'mm-bar-row').length, 6);
  assert.equal(collect(tree, node => node.props.className === 'mm-bar').length, 3);
  assert(!text(tree).includes('各有效层均值'));
  assert(view.registrations[0].guide[0].description().includes('分层动量'));
  assert.equal(collect(tree,node=>node.type==='details'&&text(node).includes('来源、范围与口径')).length,0);
  data.card.latest.momentum_norms = [null, undefined, 'unknown'];
  tree = view.render('mengli-life-metrics');
  assert.equal(collect(tree, node => node.props.className === 'mm-bar-row').length, 3);
  assert.equal(collect(tree, node => node.props.className === 'mm-bar').length, 0);
  assert(!text(tree).includes('层有效'));
});

test('CSS follows the actual official dark-theme marker and remains scoped to metrics', () => {
  const css = fs.readFileSync(new URL('../metrics.css', import.meta.url), 'utf8');
  assert(css.includes('[data-ds-dark-theme] .mengli-metrics'));
  assert(!css.includes('@keyframes'));
  assert(!css.includes('prefers-color-scheme'));
  const selectors = css.split('{').slice(0, -1).map(value => value.slice(value.lastIndexOf('}') + 1).trim());
  assert(selectors.every(value => value.startsWith('@') || value.includes('.mengli-metrics')));
});

test('life panel omits the requested pause action and keeps measured readings', () => {
  const data = snapshot(), view = nativeView(data), changes = [];
  view.window.MengliIntro.setMotionPaused = value => { changes.push(value); return value; };
  let tree = view.render('mengli-life-metrics');
  assert.equal(collect(tree,node=>node.type==='native-button'&&text(node)==='暂停律动').length,0);
  assert(text(tree).includes('分层动量'));
  assert.deepEqual(changes, []);
  assert.equal(view.observe.reads, 0);
});
