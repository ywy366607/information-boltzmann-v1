import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { buildMetricsClient, metricsPatchRows } from '../build.mjs';
import { createMetricsClient, formatMetric, metricPoints, metricLineSegments, metricViewport, lifeStatus, meanMeasured, MAX_METRICS_ROWS } from '../metrics-state.mjs';

test('lines preserve missing-value gaps, true zero and cycle resets', () => {
  const rows = [{cycle: 1, v: 2}, {cycle: 2, v: null}, {cycle: 3, v: 0},
    {cycle: 4, v: 1}, {cycle: 1, v: 3}];
  assert.deepEqual(metricLineSegments(metricPoints(rows, 'v')).map(s => s.map(p => p.x)), [[1], [3, 4], [1]]);
});

test('chart viewport selects cycle bounds, preserves gaps and accepts manual y bounds without changing records',()=>{
  const points=metricPoints([{cycle:10,v:2},{cycle:11,v:null},{cycle:12,v:0},{cycle:13,v:-1}],'v');
  const view=metricViewport(points,{from:'11',to:'13',low:'-2',high:'3'});
  assert.deepEqual(view.points.map(p=>p.x),[12,13]);
  assert.equal(view.low,-2);assert.equal(view.high,3);
  assert.equal(points.length,3);
  assert.deepEqual(metricLineSegments(metricViewport(points,{from:'10',to:'12'}).points).map(s=>s.map(p=>p.x)),[[10],[12]]);
  assert(metricViewport(points,{from:'14',to:'12'}).error);
  assert(metricViewport(points,{low:'2',high:'1'}).error);
  assert(metricViewport(points,{low:'invalid'}).error);
  assert.equal(metricViewport(points,{to:'5'}).points.length,0);
  assert.equal(metricViewport(points,{from:'20'}).points.length,0);
});

test('adjacent array entries never bridge missing cycles from bounded log tails', () => {
  const rows = [{ cycle: 360, v: 1 }, { cycle: 361, v: 2 },
    { cycle: 976, v: 3 }, { cycle: 977, v: 0 }];
  assert.deepEqual(metricLineSegments(metricPoints(rows, 'v')).map(s => s.map(p => p.x)),
    [[360, 361], [976, 977]]);
});

function fixture() {
  let family = { connected: true, state: { mode: 'paused', loaded: false, busy: false, cycle: 0, events: [] } };
  const listeners = new Set();
  const controller = { subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }, getSnapshot: () => family };
  const bridge = { controller };
  let calls = 0;
  const telemetry = { schema: 1, source: { checkpoint: '3509/current.json', cycle: 3509, branch_id: 'bound-3509', kind: 'saved_snapshot' },
    life: { card: { age: { cycles: 3509, byte_events: 67890 }, memory: { occupied: 0, capacity: 4096, episodes_committed: 0, oldest_since_write_events: null }, latest: { momentum_norms: [0, 0.2, null, 0.4] } } },
    history: { rows: [{ cycle: 3509, updates: 4, response_nll: 0, ambient_response_nll: null, progress: 'unknown' }], source: 'bound metrics.jsonl', limited: false }, latest_event: null };
  const client = createMetricsClient({ getBridge: () => bridge, fetcher: async (url, options) => {
    calls++; assert.equal(url, '/api/family/telemetry'); assert.equal(options.method, 'GET'); assert.equal(options.credentials, 'same-origin');
    return { ok: true, json: async () => telemetry };
  } });
  return { client, telemetry, bridge, get calls() { return calls; }, get listeners() { return listeners.size; },
    update(events, rest = {}) { family = { connected: true, state: { ...family.state, ...rest, events } }; for (const fn of listeners) fn(); } };
}

test('one lazy GET shared across open/close/reopen; initial family zero does not replace checkpoint', async () => {
  const f = fixture();
  assert.equal(f.calls, 0);
  await Promise.all([f.client.open(), f.client.open()]);
  await f.client.open();
  assert.equal(f.calls, 1); assert.equal(f.client.getSnapshot().source.cycle, 3509);
  assert.equal(f.client.getSnapshot().rows[0].updates, 4);
  assert.equal(lifeStatus(f.client.getSnapshot()), '已保存快照');
  f.client.dispose(); assert.equal(f.listeners, 0);
});

test('only committed same-branch events advance; repeated events do not duplicate rows', async () => {
  const f = fixture(); await f.client.open();
  const event = { status: 'committed', cursor: 1, branch_id: 'bound-3509', cycle: 3510, checkpoint: '3509/next.json',
    row: { cycle: 3510, updates: 0, response_nll: null, progress: 0 },
    card: { age: { cycles: 3510, byte_events: 68018 }, latest: { momentum_norms: [1, 2, 3, 4] } } };
  f.update([{ ...event, status: 'running' }], { loaded: true, busy: true });
  assert.equal(f.client.getSnapshot().source.cycle, 3509);
  assert.equal(lifeStatus(f.client.getSnapshot()), '进行中');
  f.update([event], { busy: false }); f.update([event]);
  assert.equal(f.client.getSnapshot().rows.length, 2);
  assert.equal(f.client.getSnapshot().source.kind, 'committed');
  assert.equal(f.client.getSnapshot().rows.at(-1).updates, 0);
  assert.equal(f.client.getSnapshot().card.age.cycles, 3510);
  f.update([{ ...event, cursor: 2, branch_id: 'parallel-3510', cycle: 9999 }]);
  assert.equal(f.client.getSnapshot().source.cycle, 3510);
  assert.equal(f.calls, 1);
});

test('unknown/null are not zero; sparse samples remain sparse', () => {
  assert.equal(formatMetric(0), '0'); assert.equal(formatMetric(null), '—');
  assert.equal(formatMetric(undefined), '—'); assert.equal(formatMetric('unknown'), '未知');
  assert.equal(formatMetric(NaN), '—');
  assert.deepEqual(metricPoints([{ cycle: 1, x: 0 }, { cycle: 2, x: null }, { cycle: 4, x: 3 }, { cycle: 5, x: 'unknown' }], 'x'),
    [{ x: 1, y: 0, index: 0 }, { x: 4, y: 3, index: 2 }]);
});

test('layer mean includes measured zero and excludes missing or nonnumeric layers', () => {
  assert.equal(meanMeasured([0, 2, null, undefined, 'unknown', NaN, Infinity, '4']), 1);
  assert.equal(meanMeasured([0, 0, 0]), 0);
  assert.equal(meanMeasured([null, undefined, 'unknown']), null);
  assert.equal(meanMeasured([]), null);
  assert.equal(meanMeasured(), null);
});

test('history and saved event read model are bounded', async () => {
  const f = fixture();
  f.telemetry.history.rows = Array.from({ length: MAX_METRICS_ROWS + 8 }, (_, index) => ({ cycle: index, updates: 1 }));
  await f.client.open();
  assert.equal(f.client.getSnapshot().rows.length, MAX_METRICS_ROWS);
  assert.equal(f.client.getSnapshot().clientLimited, true);
  f.update([{ status: 'committed', cursor: 1, cycle: 4000, row: { cycle: 4000, updates: 0 } }]);
  assert.equal(f.client.getSnapshot().rows.length, MAX_METRICS_ROWS);
});

test('GET error is visible and does not trigger automatic retries', async () => {
  let count = 0;
  const client = createMetricsClient({ fetcher: async () => { count++; return { ok: false, status: 403, json: async () => ({ error: 'Denied' }) }; } });
  await client.open(); await client.open();
  assert.equal(client.getSnapshot().error, 'Denied'); assert.equal(count, 1);
});

test('live_steward observing state recovers from prior telemetry error and reports running status', async () => {
  const f = fixture();
  let failOnce = true;
  const client = createMetricsClient({
    getBridge: () => f.bridge,
    fetcher: async () => failOnce
      ? (failOnce = false, { ok: true, json: async () => ({ error: 'Invalid explicit observation storage roots', history: { rows: [] }, life: { card: null } }) })
      : { ok: true, json: async () => f.telemetry },
  });
  await client.open();
  assert.equal(client.getSnapshot().error, 'Invalid explicit observation storage roots');
  const card = {
    schema: 1,
    identity: { individual_id: 'mengli-1' },
    branch_id: 'v13-clean',
    source: { checkpoint: 'D:\\v13\\body\\last.pt.gz', cycle: 6675 },
    age: { cycles: 6675, byte_events: 12700000 },
  };
  f.update(
    [{ status: 'committed', cursor: 6676, branch_id: 'v13-clean', cycle: 6676, row: { cycle: 6676, updates: 2, response_nll: 1.42 } }],
    {
      mode: 'observing',
      read_only: false,
      live_steward: true,
      loaded: false,
      busy: false,
      error: null,
      individual_id: 'mengli-1',
      branch_id: 'v13-clean',
      checkpoint: 'D:\\v13\\body\\last.pt.gz',
      cycle: 6676,
      observation: { status: 'running', live_steward: true, latest_completed_cycle: 6676, saved_cycle: 6675 },
      life_card_data: card,
    },
  );
  assert.equal(client.getSnapshot().error, null);
  assert.equal(client.getSnapshot().source.branch_id, 'v13-clean');
  assert.equal(client.getSnapshot().source.cycle, 6676);
  assert.equal(client.getSnapshot().rows.at(-1).cycle, 6676);
  assert.equal(lifeStatus(client.getSnapshot()), '训练中');
  client.dispose();
});

test('build synchronously emits independent package, native guide entries, and Cordis insert', () => {
  const built = buildMetricsClient();
  const rows = metricsPatchRows();
  assert.deepEqual(rows, [{ insert: [{ id: 'mengli-native-metrics', name: built.entry }] }]);
  const source = fs.readFileSync(built.client, 'utf8');
  new vm.Script(source);
  const registrations = [], slots = [];
  const React = { createElement: (...args) => args };
  let module;
  const document = { createElement: () => ({ dataset: {}, remove() {} }), head: { appendChild() {} } };
  vm.runInNewContext(source, { document, window: { __ModuleLoader__: { load: value => { module = value.factory(name => name === 'react' ? React : { Button() {} }); } } } });
  module.apply({ effect(fn) { fn(); }, slots: { inject(name, callback) { callback(); }, register(def, render) { slots.push({ def, render }); } },
    sidebarRightTabs: { register(def) { registrations.push(def); return () => {}; } }, sidebarRight: {}, layout: {} });
  assert.deepEqual(registrations.map(tab => tab.kind), ['mengli-life-metrics', 'mengli-learning-metrics']);
  assert.deepEqual(registrations.map(tab => tab.guide[0].title()), ['生命体征', '学习训练']);
  assert.equal(slots.filter(slot => slot.def.name === 'sidebar.right.pane.tab').length, 2);
  assert(!/setInterval|new EventSource|iframe|\.pt\b|torch/.test(source));
});
