import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source = fs.readFileSync(new URL('../account-client.js', import.meta.url), 'utf8');
const create = vm.runInNewContext(source + '\ncreateLocalAccountClient');

test('account UI reads and changes one display name without touching chat or family endpoints', async () => {
  const requests = [];
  const client = create({ fetcher: async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ displayName: options.body ? JSON.parse(options.body).displayName : 'fixture-parent' }) };
  } });
  await client.load();
  assert.equal(client.getSnapshot().displayName, 'fixture-parent');
  await client.saveName('renamed-parent');
  assert.equal(client.getSnapshot().displayName, 'renamed-parent');
  assert.deepEqual(requests.map(request => request.url), ['/api/local-account', '/api/local-account/profile']);
  assert(requests.every(request => request.options.credentials === 'same-origin'));
});

test('key fields are sent only to the key endpoint and never retained in the profile snapshot', async () => {
  const requests = [];
  const client = create({ fetcher: async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ ok: true }) };
  } });
  await client.changeKey('fixture-current-key', 'fixture-next-key');
  assert.equal(requests[0].url, '/api/local-account/key');
  assert.deepEqual(JSON.parse(requests[0].options.body), { currentKey: 'fixture-current-key', newKey: 'fixture-next-key' });
  assert(!JSON.stringify(client.getSnapshot()).includes('fixture-current-key'));
  assert(!JSON.stringify(client.getSnapshot()).includes('fixture-next-key'));
  await client.logout();
  assert.equal(requests[1].url, '/api/local-account/logout');
});

test('only human source labels use the profile name; original actors and labels stay unchanged', () => {
  const client = fs.readFileSync(new URL('../client-source.js', import.meta.url), 'utf8');
  const start = client.indexOf('function SourceLabel({ source }) {');
  const indent = client.slice(client.lastIndexOf('\n', start) + 1, start);
  const end = client.indexOf('\n' + indent + '}', start) + indent.length + 2;
  const render = vm.runInNewContext(client.slice(start, end) + '\nSourceLabel', {
    account: { useAccount: () => ({ displayName: 'fixture-parent' }) }, h: (_tag, _props, text) => text,
  });
  const parent = Object.freeze({ kind: 'family', actor: 'parent', label: '你 · 家长' });
  assert.equal(render({ source: parent }), 'fixture-parent');
  assert.equal(render({ source: Object.freeze({ kind: 'user' }) }), 'fixture-parent');
  assert.equal(render({ source: Object.freeze({ kind: 'family', actor: 'mother', label: '妈妈' }) }), '妈妈');
  assert.equal(parent.actor, 'parent');
  assert.equal(parent.label, '你 · 家长');
});

test('account errors remain visible and do not trigger retries', async () => {
  let calls = 0;
  const client = create({ fetcher: async () => { calls++; return { ok: false, json: async () => ({ error: '请先登录' }) }; } });
  await client.load();
  assert.equal(client.getSnapshot().error, '请先登录');
  assert.equal(calls, 1);
});
