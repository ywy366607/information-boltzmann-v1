import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createFamilyClient, projectFamilyEvents, projectFamilyHistory } from '../family-state.mjs';
import { buildNativeFamilyPatch } from '../profile.mjs';
import { DEFAULT_PACKAGE_ROOT } from '../build.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));
const event = (cursor = 1, actor = 'parent') => ({ cursor, actor, status: 'committed', utterance: { text: `行动${cursor}` }, delivered: { text: `摄入${cursor}` }, checkpoint: 'saved.pt' });
const state = (events = []) => ({ loaded: true, mode: 'parent', busy: false, cursor: events.at(-1)?.cursor ?? 0, events, plan: { version: 1, text: '照护' } });
const response = (value, ok = true) => ({ ok, status: ok ? 200 : 409, json: async () => value });
class FakeEvents {
  static current;
  constructor(url, options) { this.url = url; this.options = options; this.handlers = {}; FakeEvents.current = this; }
  addEventListener(name, handler) { this.handlers[name] = handler; }
  emit(value) { this.handlers.family({ data: JSON.stringify(value) }); }
  close() { this.closed = true; }
}

test('committed projection preserves action-before-input, true actors and stable native identities', () => {
  const cache = new Map();
  const rows = projectFamilyEvents([event(), { ...event(2), status: 'queued' }, event(3, 'mother')], cache);
  assert.deepEqual(rows.map(row => row.key), ['family:1:action', 'family:1:input', 'family:3:action', 'family:3:input']);
  assert.equal(rows[0].kind, 'assistant-step');
  assert.equal(rows[1].kind, 'user');
  assert.equal(rows[1].data.source.actor, 'parent');
  assert.equal(rows[3].kind, 'user');
  assert.equal(rows[3].family.actor, 'mother');
  assert.equal(rows[1].data.content[0].text, '摄入1');
  assert.equal(rows[0].family.timing, 'before_caregiver_input');
  assert.equal(rows[0].data.blocks[0].text, '**梦漓**\n\n行动1');
  assert.equal(rows[3].data.content[0].text, '摄入3');
  assert.equal(rows[3].data.source.label, '妈妈');
  assert.equal(rows[1].data.source.label, '你 · 家长');
  assert.deepEqual(rows.filter(row => row.kind === 'user').map(row => row.family.actor), ['parent', 'mother']);
  const same = projectFamilyEvents([event(), event(3, 'mother')], cache);
  rows.forEach((row, index) => assert.equal(row, same[index]));
});

test('saved source history keeps raw roles and stable keys without inventing delivery or cycles', () => {
  const history = { checkpoint_ref: { session_id: 'source', turn_end_seq: 55 }, turns: [{ role: 'user', content: 'old input' }, { role: 'assistant', content: 'old answer' }] };
  const cache = new Map();
  const nodes = projectFamilyHistory(history, cache);
  assert.equal(nodes.length, 2);
  assert.equal(nodes[0].family.source, 'saved-history');
  assert.equal(nodes[0].family.original_role, 'user');
  assert.equal(history.turns[0].role, 'user');
  assert.equal(nodes[0].kind, 'assistant-step');
  assert.equal(nodes[1].kind, 'user');
  assert.equal(nodes[0].family.label, '梦漓');
  assert.ok(nodes.every(node => !JSON.stringify(node.data).includes('已保存历史')));
  assert.equal(nodes[0].family.actor, 'mengli');
  assert.equal(nodes[1].family.actor, 'mother');
  assert.doesNotMatch(nodes[1].data.content[0].text, /实际摄入/);
  assert.equal(nodes[0].family.cursor, undefined);
  assert.equal(nodes[0].family.cycle, undefined);
  assert.ok(nodes.every(node => node.anchorSeq < 0));
  assert.equal(projectFamilyHistory(history, cache)[0], nodes[0]);
  assert.equal(projectFamilyEvents([], cache).length, 0);
});

test('same-origin GET/SSE and parent say publish only saved records with no DS prompt', async () => {
  const calls = [];
  const client = createFamilyClient({ EventSourceImpl: FakeEvents, fetcher: async (url, options) => {
    calls.push({ url, options });
    return response(url.endsWith('/say') ? state([event()]) : state());
  } });
  await client.connect();
  assert.equal(FakeEvents.current.url, '/api/family/events');
  assert.equal(FakeEvents.current.options.withCredentials, true);
  assert.equal(client.getSnapshot().state.events.length, 0);
  await client.say('您好');
  assert.deepEqual(calls.map(call => call.url), ['/api/family', '/api/family/history', '/api/family/say']);
  assert.equal(calls[2].options.credentials, 'same-origin');
  assert.deepEqual(JSON.parse(calls[2].options.body), { text: '您好' });
  FakeEvents.current.emit(state([event()]));
  FakeEvents.current.emit(state([event(2, 'mother')]));
  assert.equal(client.getSnapshot().state.events.length, 2);
  client.dispose();
  assert.equal(FakeEvents.current.closed, true);
});

test('offline, busy and unowned submits fail; slash text never invokes a DS command', async () => {
  const calls = [];
  const client = createFamilyClient({ fetcher: async url => { calls.push(url); return response({ error: { message: '保存失败' } }, false); } });
  await assert.rejects(client.say('a'), /家长接管/);
  client.adopt({ ...state(), mode: 'mother' });
  await assert.rejects(client.say('a'), /家长接管/);
  client.adopt({ ...state(), busy: true });
  await assert.rejects(client.say('a'), /家长接管/);
  client.adopt(state());
  await assert.rejects(client.say('/goal run'), /保存失败/);
  await assert.rejects(client.say('a'), /保存失败/);
  assert.deepEqual(calls, ['/api/family/say', '/api/family/say']);
  assert.equal(client.getSnapshot().state.events.length, 0);
});

test('plan updates send version CAS and surface conflict without discarding state', async () => {
  let payload;
  const client = createFamilyClient({ fetcher: async (_url, options) => { payload = JSON.parse(options.body); return response({ error: '计划版本冲突' }, false); } });
  client.adopt(state());
  await assert.rejects(client.command('plan', { text: 'new', version: 1 }), /计划版本冲突/);
  assert.deepEqual(payload, { text: 'new', version: 1 });
  assert.equal(client.getSnapshot().state.plan.text, '照护');
});

// Execute the real shipped official modules. Only React/host APIs are fixture adapters;
// ChatSnapshotBuilder, ChatFlow, ChatNodeSeat, user/assistant and InputHub are official code.
function officialModule(name, expose) {
  let exported;
  const jsx = (type, props, key) => ({ type, props: { ...props, key } });
  const react = {
    memo: value => value, forwardRef: value => value, createElement: jsx,
    createContext: () => ({}), useMemo: fn => fn(), useCallback: fn => fn,
    useRef: value => ({ current: value }), useState: value => [typeof value === 'function' ? value() : value, () => {}],
    useEffect() {}, useLayoutEffect() {}, useId: () => 'id', useSyncExternalStore: (_subscribe, get) => get(),
  };
  const schema = new Proxy(function () {}, { get: () => schema, apply: () => schema });
  const store = {
    createSnapshotStore(value) { const listeners = new Set(); return { getSnapshot: () => value, set(next) { value = next; listeners.forEach(fn => fn()); }, subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); } }; },
    defineStore: value => value,
  };
  const modules = { react, 'react/jsx-runtime': { jsx, jsxs: jsx }, '@deepseek-ai/dsh-client-store': store, '@deepseek-ai/schemastery': { default: schema, __esModule: true } };
  const noop = new Proxy(function () {}, { get: (_target, key) => key === 'default' ? noop : noop, apply: () => noop });
  const require = Object.assign(id => modules[id] ?? noop, { asset: value => value });
  const window = { __ModuleLoader__: { load({ factory }) { exported = factory(require); } } };
  let source = fs.readFileSync(`${root}official/${name}/lib/client.js`, 'utf8');
  source = source.replace('return module.exports;', `Object.assign(exports, {${expose}}); return module.exports;`);
  vm.runInNewContext(source, { window, console, Date, Map, Set, WeakMap, AbortController, setTimeout, clearTimeout, URL }, { filename: name });
  return { exported, window };
}

buildNativeFamilyPatch();

test('owned bundles retain official identities and native registration; no replacement chat app', () => {
  const rows = buildNativeFamilyPatch();
  assert.equal(rows[0].id, 'ui-layout');
  assert.equal(rows[1].id, 'ui-conversation');
  assert.equal(rows[2].id, 'ui-chat');
  assert.equal(rows[3].id, 'ui-renderer');
  assert.ok(rows.slice(0, 5).every(row => row.disabled));
  assert.match(rows[5].insert[3].name, /lib[\\/]index\.js$/);
  assert.equal(rows[5].insert.length, 6);
  for (const name of ['ui-layout', 'ui-conversation', 'ui-chat', 'ui-renderer', 'ui-settings-account']) {
    const pkg = JSON.parse(fs.readFileSync(`${root}official/${name}/package.json`));
    assert.equal(pkg.name, `@deepseek-ai/dsh-client-${name}`);
    assert.equal(pkg.dsh.client.platform, 'web');
  }
  const plugin = fs.readFileSync(`${root}lib/client.js`, 'utf8');
  assert.doesNotMatch(plugin, /iframe|innerHTML|appendChild|createRoot|setInterval/);
});

test('native root survives a registration handoff without removing its startup admission check', () => {
  const source = fs.readFileSync(`${root}official/ui-renderer/lib/client.js`, 'utf8');
  assert.match(source, /'root' has no registration — a layout entry must register/);
  const start = source.indexOf('function RootOutlet({ ownerProps }) {');
  const indent = source.slice(source.lastIndexOf('\n', start) + 1, start);
  const end = source.indexOf('\n' + indent + '}', start) + indent.length + 2;
  let entries = [], subscriptions = 0;
  const host = { subscribe() { subscriptions++; }, getVersion: () => entries.length,
    entriesOfSlot: () => entries, entriesOf: () => entries };
  const context = { useHost: () => host, useLocaleRevision() {},
    react: { useSyncExternalStore(subscribe) { subscribe(() => {}); } },
    react_jsx_runtime: { jsx: (type, props) => ({ type, props }) },
    ANCHOR_STYLE: {}, SlotErrorBoundary() {}, RootEntry() {}, EMPTY_SLOT_INJECT: {}, entryKeyOf: () => 1 };
  const render = vm.runInNewContext(source.slice(start, end) + '\nRootOutlet', context);
  assert.equal(render({ ownerProps: {} }), null);
  entries = [{}];
  assert.equal(render({ ownerProps: {} }).type, 'div');
  assert.equal(subscriptions, 2);
});

test('real Cordis composition disables originals and inserts importable owned package entries', async () => {
  const { composeEntries } = await import(pathToFileURL(`${DEFAULT_PACKAGE_ROOT}/dsh-app-boot/lib/index.js`).href);
  const warnings = [];
  const composed = composeEntries([[{ insert: [
    { id: 'ui-layout', name: '@deepseek-ai/dsh-client-ui-layout' },
    { id: 'ui-conversation', name: '@deepseek-ai/dsh-client-ui-conversation' },
    { id: 'ui-chat', name: '@deepseek-ai/dsh-client-ui-chat' },
    { id: 'ui-brand-official', name: '@deepseek-ai/dsh-client-ui-brand-official' },
    { id: 'ui-renderer', name: '@deepseek-ai/dsh-client-ui-renderer' },
    { id: 'ui-settings-account', name: '@deepseek-ai/dsh-client-ui-settings-account' },
  ] }], buildNativeFamilyPatch()], warning => warnings.push(warning));
  assert.deepEqual(warnings, []);
  assert.equal(composed[0].disabled, true);
  assert.equal(composed[1].disabled, true);
  assert.equal(composed[2].disabled, true);
  assert.equal(composed[3].disabled, true);
  assert.equal(composed[4].disabled, true);
  assert.equal(composed[5].disabled, true);
  assert.equal(composed.length, 14);
  for (const row of composed.slice(6)) {
    assert.ok(fs.existsSync(row.name));
    assert.equal(typeof (await import(pathToFileURL(row.name).href)).apply, 'function');
  }
});

test('real official ChatSnapshotBuilder and ChatFlow produce native keyed seats from committed body records', () => {
  const { exported } = officialModule('ui-chat', 'ChatSnapshotBuilder, ChatFlow, ChatNodeSeat, UserMessageNodeView, AssistantNodeView');
  assert.ok(exported.ChatSnapshotBuilder, 'test must exercise the actual official builder');
  const builder = new exported.ChatSnapshotBuilder();
  const timeline = { turnOrder: [], turns: new Map() };
  const cache = new Map();
  const nodes = projectFamilyEvents([event()], cache);
  const snapshot = builder.apply({ upserts: nodes, timeline });
  builder.publish();
  assert.deepEqual([...snapshot.order], nodes.map(node => node.key));
  assert.equal(snapshot.nodes.get(nodes[0].key), nodes[0]);
  const next = builder.apply({ upserts: projectFamilyEvents([event(2)], cache), timeline });
  builder.publish();
  assert.equal(next.nodes.get(nodes[0].key), nodes[0]);
  const rows = exported.ChatFlow({ entries: snapshot.order.map(key => ({ kind: 'node', key })), pendingInputs: [],
    useChat: fn => fn(snapshot), useSession: fn => fn({ running: false }), useChatNode: (key, fn) => fn(snapshot.nodes.get(key)),
    useConnectionState: fn => fn('connected'),
  });
  assert.equal(rows.length, 2);
  assert.equal(rows[0].type, exported.ChatNodeSeat);
  assert.equal(rows[1].type, exported.ChatNodeSeat);
  const familyRows = exported.ChatFlow({ familyProjection: {
    useChat: fn => fn(snapshot), useSession: fn => fn({ running: false }),
    useChatNode: (key, fn) => fn(snapshot.nodes.get(key)),
  }, entries: snapshot.order.map(key => ({ kind: 'node', key })), pendingInputs: [],
    useChat: () => { throw new Error('must not use mom Chat target'); },
    useSession: () => { throw new Error('must not use mom running state'); },
    useConnectionState: fn => fn('connected'),
  });
  assert.equal(familyRows[0].type, exported.ChatNodeSeat);
  const user = exported.UserMessageNodeView({ node: nodes[1] });
  assert.equal(typeof user.type, 'function');
  const assistant = exported.AssistantNodeView({ node: nodes[0], useContextMenu: () => ({ getTriggerProps() {} }),
    useTurnData: () => undefined, usePresentation: fn => fn({ foldedReplies: 'all' }) });
  assert.equal(typeof assistant.type, 'function');
});

test('official submit machine preserves failed body draft and commits success for Enter and button', async () => {
  const { exported } = officialModule('ui-conversation', 'SessionInputShell, SubmitMachine');
  for (const source of ['enter', 'click']) {
    let clear = 0, sends = 0, commandCalls = 0;
    let reject = true;
    const shell = Object.create(exported.SessionInputShell.prototype);
    shell.disposed = false;
    shell.attachmentIds = [];
    shell.noticeSeq = 0;
    shell.notices = { set() {} };
    shell.core = new exported.SubmitMachine();
    shell.draftEditor = { projection: { clipboardText: '家长原生草稿', occurrences: [], quotes: [] },
      clearCommittedDraft() { clear++; }, clearHistory() {} };
    shell.state = { getSnapshot: () => ({ phase: shell.core.state.phase, draft: '家长原生草稿' }) };
    shell.publish = () => {};
    shell.deps = {
      familyView: () => true, submissionState: () => ({ familyView: true }),
      inputTriggers: () => ({ track() {}, adjudicate() { commandCalls++; return Promise.resolve(); } }),
      defaultSink: async (_text, _ids, _mode, _signal, _accept, body) => {
        assert.equal(body, true); sends++;
        if (reject) throw new Error('保存失败');
        return { kind: 'success' };
      },
    };
    shell.submit('queue', source);
    assert.equal(shell.core.sendingAttempt.submission.state.familyView, true);
    await new Promise(setImmediate);
    assert.equal(clear, 0);
    assert.equal(shell.core.state.phase, 'plain');
    reject = false;
    shell.submit('queue', source);
    await new Promise(setImmediate);
    assert.equal(clear, 1);
    assert.equal(sends, 2);
    assert.equal(commandCalls, 0);
  }
});

test('plan width uses the official layout action, not a custom panel', () => {
  const { exported } = officialModule('ui-layout', 'LayoutController');
  const widths = [];
  const controller = new exported.LayoutController({ setRightbar: width => widths.push(width) });
  controller.setRightbarWidth(380);
  assert.deepEqual(widths, [380]);
});

test('official InputHub pins body sink, never falls back, and mom retains its exact normal sink', async () => {
  const { exported, window } = officialModule('ui-conversation', 'InputHub, SessionInputShell');
  assert.ok(exported.InputHub);
  const hub = Object.create(exported.InputHub.prototype);
  const sends = [];
  hub.conversation = () => ({ sendSession: (...args) => { sends.push(args); return Promise.resolve({ kind: 'success' }); } });
  const session = { sessionId: 'canonical' };
  await assert.rejects(hub.sink(session, 'body', [], 'queue', undefined, () => {}, true), /阻止默认模型/);
  assert.equal(sends.length, 0);
  const familyCalls = [];
  window.__MENGLI_NATIVE_FAMILY__.controller = { say: async (...args) => { familyCalls.push(args); return { kind: 'success' }; } };
  // Current view changes after submit: the captured body bit remains authoritative.
  window.__MENGLI_NATIVE_FAMILY__.views.set('canonical', 'chat');
  await hub.sink(session, 'body', [], 'queue', undefined, () => {}, true);
  assert.equal(familyCalls[0][0], 'body');
  assert.equal(sends.length, 0);
  await hub.sink(session, 'mom', [], 'queue', undefined, () => {}, false);
  assert.equal(sends[0][1], 'mom');
});
