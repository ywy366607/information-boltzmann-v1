// Run with node --test tests/test_mengli_dashboard.mjs; no server or API calls.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../scripts/ib_local/mengli_harness/web/index.html', import.meta.url), 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];

class Element {
  constructor() {
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.attributes = {};
    this.className = '';
    this._text = '';
    this.scrollHeight = 100;
    this.scrollTop = 0;
    this.clientHeight = 100;
    this.classList = { toggle: (name, enabled) => {
      this.className = enabled ? name : '';
    } };
  }
  set innerHTML(_) { throw new Error('Unsafe HTML insertion'); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  append(...children) {
    for (const child of children) { child.parentElement = this; this.children.push(child); }
  }
  replaceChildren() { this.children = []; this._text = ''; }
  setAttribute(name, value) { this.attributes[name] = value; }
}

async function dashboard(vitals = {}, session = {}, failures = []) {
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(([, id]) => [id, new Element()]));
  for (const [id, element] of elements) {
    if (id.endsWith('-bar') || id.startsWith('bar-')) element.parentElement = new Element();
  }
  const requests = [];
  const context = vm.createContext({
    document: { getElementById: id => {
      assert.ok(elements.has(id), `Missing HTML element: ${id}`);
      return elements.get(id);
    }, createElement: () => new Element() },
    setInterval: () => 0,
    fetch: async (url, options) => {
      requests.push({ url, options });
      if (failures.includes(url)) return { ok: false, status: 503 };
      return { ok: true, json: async () => url === '/api/vitals' ? vitals : session };
    },
  });
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  return { elements, requests, context };
}

const text = (view, id) => view.elements.get(id).textContent;

test('missing metrics stay unmeasured without fake defaults or activity', async () => {
  const view = await dashboard();
  for (const id of ['hdr-cycle', 'card-void', 'card-momentum', 'card-gain', 'card-hitrate',
    'card-latency', 'tok-hit', 'tok-miss', 'tok-out', 'l1-val']) {
    assert.equal(text(view, id), '未测', id);
  }
  assert.equal(text(view, 'service-status'), '服务 未测 / 训练 未测');
  assert.equal(view.elements.get('service-dot').className, '');
  assert.equal(text(view, 'card-relations'), '未测 / 未测');
  assert.equal(view.elements.get('l1-bar').parentElement.dataset.unmeasured, 'true');
  assert.equal(view.elements.get('dialogue-stream').children.length, 0);
});

test('actual zeros and summary version zero remain visible', async () => {
  const view = await dashboard({
    cycle: 0, active_relations: 0, relation_capacity: 64, p_momentum_norm: 0,
    s_state_norm: 0, q_invariant_norm: 0, metal_g: 0, provider_hit_rate: 0,
    latency_seconds: 0, cache_hit_tokens: 0, cache_miss_tokens: 0,
    completion_tokens: 0, layer_gradient_norms: [0, 0, 0, 0],
    reply_usage: { prompt_tokens: 0, completion_tokens: 0, cache_hit_tokens: 0, cache_miss_tokens: 0 },
    summary_usage: { prompt_tokens: 10, completion_tokens: 2, cache_hit_tokens: 4, cache_miss_tokens: 6 },
  }, { summary: null, summary_version: 0, total_turns: 0, turns: [] });
  assert.equal(text(view, 'hdr-cycle'), '#0');
  assert.equal(text(view, 'card-relations'), '0 / 64');
  assert.equal(text(view, 'card-momentum'), '0.0000');
  assert.equal(text(view, 'card-gain'), '0.0000');
  assert.equal(text(view, 'card-hitrate'), '0.0%');
  assert.equal(text(view, 'card-latency'), '0.00 s');
  assert.equal(text(view, 'tok-hit'), '0');
  assert.equal(text(view, 'l1-val'), '0.00e+0');
  assert.equal(view.elements.get('l1-bar').style.width, '0%');
  assert.equal(view.elements.get('l1-bar').parentElement.dataset.unmeasured, 'false');
  assert.match(text(view, 'summary-meta'), /版本 0/);
  assert.match(text(view, 'stream-turn-count'), /已提交 0 轮/);
  assert.match(text(view, 'tok-ledger'), /应答 输入 0/);
  assert.match(text(view, 'tok-ledger'), /摘要 输入 10（命中 4 \/ 未命中 6）\/ 输出 2/);
});

test('gradient norms are not P/Q; missing layer and capacity do not get invented', async () => {
  const view = await dashboard({ question_norm: 10, layer_gradient_norms: [0, 2, null], active_relations: 8 });
  assert.equal(text(view, 'card-momentum'), '未测');
  assert.equal(text(view, 'state-norms'), 'S 未测 / Q 未测');
  assert.equal(text(view, 'card-relations'), '8 / 未测');
  assert.equal(view.elements.get('bar-relations').parentElement.dataset.unmeasured, 'true');
  assert.equal(text(view, 'l3-val'), '未测');
  assert.equal(text(view, 'l4-val'), '未测');
  assert.equal(view.elements.get('l2-bar').style.width, '100%');
});

test('untrusted child, mother, summary and action text never become HTML or class names', async () => {
  const hostile = '<img src=x onerror="globalThis.pwned=true">';
  const view = await dashboard({}, {
    summary: hostile, summary_version: 0, total_turns: 1, session_ref: { session_id: hostile },
    turns: [{ child_text: hostile, mother_reply: hostile, choice: 'generate hacked', cycle: 0 }],
  });
  const stream = view.elements.get('dialogue-stream');
  assert.equal(stream.children.length, 2);
  assert.match(stream.textContent, /<img src=x/);
  assert.equal(stream.children[0].children[0].children[0].children[1].className, 'tag tag-unknown');
  assert.match(stream.textContent, /C0/);
  assert.equal(text(view, 'summary-text'), hostile);
  assert.equal(view.context.pwned, undefined);
  assert.deepEqual(view.requests.map(request => request.url), ['/api/vitals', '/api/session']);
});

test('official role/content messages render without fabricated action or cycle', async () => {
  const view = await dashboard({}, {
    total_turns: 1, turns: [
      { role: 'user', content: 'child raw' },
      { role: 'assistant', content: [{ type: 'text', text: '<b>mother raw</b>' }] },
    ],
  });
  const stream = view.elements.get('dialogue-stream');
  assert.equal(stream.children.length, 2);
  assert.match(stream.textContent, /child raw/);
  assert.match(stream.textContent, /<b>mother raw<\/b>/);
  assert.match(stream.textContent, /未测 \/ C未测/);
  assert.match(text(view, 'stream-turn-count'), /已提交 1 轮 \/ 显示 2 条记录/);
});

test('service and training statuses and provenance are separate returned values', async () => {
  const view = await dashboard({ metadata: {
    service_status: 'ready', training_status: 'offline', run_id: 'run-A', arm: 'weighted',
    source_checkpoint: 'source.pt.gz', metrics_source: 'explicit/metrics.jsonl',
  } });
  assert.equal(text(view, 'service-status'), '服务 ready / 训练 offline');
  assert.equal(text(view, 'hdr-run'), 'run-A / ARM weighted');
  assert.match(text(view, 'stream-source'), /source.pt.gz/);
  assert.match(text(view, 'stream-source'), /explicit\/metrics.jsonl/);
  assert.equal(view.elements.get('service-dot').className, 'is-online');
});

test('vitals failure marks service offline while session can still load', async () => {
  const view = await dashboard({}, { summary: 'real summary', turns: [], total_turns: 0 }, ['/api/vitals']);
  assert.equal(text(view, 'service-status'), '服务 offline / 训练 未测');
  assert.equal(text(view, 'summary-text'), 'real summary');
});

test('session failure clears stale summary and dialogue, but not real vitals', async () => {
  const view = await dashboard({ cycle: 0 }, {}, ['/api/session']);
  assert.equal(text(view, 'hdr-cycle'), '#0');
  assert.equal(text(view, 'summary-text'), '未测');
  assert.equal(text(view, 'stream-turn-count'), '会话读取失败');
  assert.equal(view.elements.get('dialogue-stream').children.length, 0);
});
