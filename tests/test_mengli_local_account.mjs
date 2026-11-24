import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { openLocalAccount } from '../scripts/ib_local/mengli_harness/local-account.mjs';
import { installLocalLogin } from '../scripts/ib_local/mengli_harness/local-login.mjs';
import { entranceReadings, loginPage } from '../scripts/ib_local/mengli_harness/login-page.mjs';

const KEY = 'fixture-local-key';
const NEXT_KEY = 'fixture-changed-key';
const ORIGIN = 'http://127.0.0.1:8900';
const HOST = '127.0.0.1:8900';
const COOKIE_NAME = 'official_fixture_authority';
const V1 = `${COOKIE_NAME}=fixture-v1-browser`;
const V2 = `${COOKIE_NAME}=fixture-v2-operator`;
const LAUNCH = 'fixture-native-launch';
const testsRoot = path.dirname(fileURLToPath(import.meta.url));
async function fixture(t, options = { initialKey: KEY }) {
  const directory = await mkdtemp(path.join(testsRoot, '.local-account-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = path.join(directory, 'account.json');
  return { directory, file, account: await openLocalAccount(file, options) };
}
const getHeader = (request, name) => request.headers instanceof Headers ? request.headers.get(name) : request.headers[name];

// Minimal official contract double: admission accepts browser + operator, but
// authorizeIndex accepts ONLY v1, or exchanges the native launch credential.
function harness(account, options = {}) {
  const raw = new Map(), api = new Map(), cleanup = [];
  const peer = {};
  const calls = { authorization: [], reads: [], exchanges: 0 };
  const ctx = {
    effect(effect) { const dispose = effect(); cleanup.push(dispose); return dispose; },
    webServer: {
      protocol: 'http:',
      register(route) {
        assert.equal(route.kind, 'exact');
        assert.equal(raw.has(route.path), false);
        raw.set(route.path, route.handler);
        return () => raw.delete(route.path);
      },
      async readResource(request, actualPeer) {
        assert.equal(actualPeer, peer);
        assert.equal(new URL(request.url).pathname, '/index.html');
        calls.reads.push(request);
        return new Response('<html>FIXTURE_NATIVE_BOOT_SESSION_BODY</html>', { headers: { 'content-type': 'text/html' } });
      },
    },
    connection: {
      browserPeer: peer,
      requestRejection(request) {
        if (getHeader(request, 'host') !== HOST || getHeader(request, 'origin') && getHeader(request, 'origin') !== ORIGIN) return 403;
        const cookie = getHeader(request, 'cookie');
        return cookie === V1 || cookie === V2 ? undefined : 401;
      },
      authenticatedUrl(base) { const url = new URL(base); url.searchParams.set('token', LAUNCH); return url.href; },
      authorizeIndex(request, response) {
        calls.authorization.push(request);
        const url = new URL(request.url, ORIGIN);
        if (request.method === 'GET' && url.pathname === '/' && url.searchParams.get('token') === LAUNCH) {
          calls.exchanges++;
          response.writeHead(303, { location: './', 'set-cookie': `${V1}; Max-Age=86400; Path=/; HttpOnly; SameSite=Strict`, 'cache-control': 'no-store' });
          response.end();
          return false;
        }
        if (getHeader(request, 'cookie') === V1 && !url.searchParams.has('token')) return true;
        response.writeHead(401, { 'content-type': 'text/plain' }); response.end('unauthorized'); return false;
      },
      fetch: { register(route) {
        assert.equal(route.requestBody, route.methods[0] === 'GET' ? 'buffered' : 'streaming');
        api.set(route.path, route);
        const dispose = () => api.delete(route.path);
        cleanup.push(dispose); // Connection itself owns this scope in production.
        return dispose;
      } },
    },
  };
  assert.equal(installLocalLogin(ctx, { account, ...options }), account);
  const rawRequest = async (url = '/', { method = 'GET', body, chunks, headers = {} } = {}) => {
    const request = Readable.from(chunks ?? (body === undefined ? [] : [JSON.stringify(body)]));
    request.method = method; request.url = url;
    request.headers = { host: HOST, ...(method === 'POST' ? { origin: ORIGIN, 'sec-fetch-site': 'same-origin', 'content-type': 'application/json' } : {}), ...headers };
    let status, responseHeaders, responseBody;
    const response = {
      writeHead(nextStatus, nextHeaders = {}) { status = nextStatus; responseHeaders = nextHeaders; },
      end(text) { responseBody = text; },
    };
    await raw.get(new URL(url, ORIGIN).pathname)(request, response);
    const result = new Response(responseBody, { status, headers: Object.entries(responseHeaders).flatMap(([key, value]) =>
      (Array.isArray(value) ? value : [value]).map(item => [key, item])) });
    return { response: result, request };
  };
  const fetchApi = (url = '/api/local-account', { body, headers = {}, chunks } = {}) => {
    const pathname = new URL(url, ORIGIN).pathname;
    const route = api.get(pathname), method = route.methods[0];
    const stream = chunks && new ReadableStream({ start(controller) {
      for (const chunk of chunks) controller.enqueue(Buffer.from(chunk)); controller.close();
    } });
    const request = new Request(new URL(url, 'http://dsh.internal'), { method,
      headers: { host: HOST, cookie: V1, ...(method === 'POST' ? { origin: ORIGIN, 'sec-fetch-site': 'same-origin', 'content-type': 'application/json' } : {}), ...headers },
      ...(method === 'POST' ? { body: stream ?? JSON.stringify(body ?? {}), ...(stream ? { duplex: 'half' } : {}) } : {}),
    });
    return route.fetch(request);
  };
  return { raw, api, rawRequest, fetchApi, calls, cleanup: () => cleanup.reverse().forEach(dispose => dispose()) };
}

test('scrypt verification, random salts, default name and durable public view', async t => {
  const { account, file, directory } = await fixture(t);
  assert.deepEqual(account.view(), { displayName: '无月酱酱酱酱' });
  assert.equal(await account.verify(KEY), true);
  for (const value of ['wrong-fixture', '', null, 42, 'x'.repeat(513)]) assert.equal(await account.verify(value), false);
  const record = JSON.parse(await readFile(file, 'utf8'));
  assert.deepEqual(Object.keys(record).sort(), ['displayName', 'hash', 'salt']);
  assert.match(record.salt, /^[a-f0-9]{32}$/); assert.match(record.hash, /^[a-f0-9]{128}$/);
  assert.equal(JSON.stringify(record).includes(KEY), false);
  const second = await openLocalAccount(path.join(directory, 'second.json'), { initialKey: KEY });
  assert.equal(await second.verify(KEY), true);
  const other = JSON.parse(await readFile(path.join(directory, 'second.json'), 'utf8'));
  assert.notEqual(record.salt, other.salt); assert.notEqual(record.hash, other.hash);
  account.view().displayName = 'not-persisted';
  assert.deepEqual(await account.setDisplayName('  fixture家长  '), { displayName: 'fixture家长' });
  const reopened = await openLocalAccount(file, { initialKey: 'ignored-fixture', displayName: 'ignored-name' });
  assert.equal(await reopened.verify(KEY), true); assert.equal(await reopened.verify('ignored-fixture'), false);
  assert.deepEqual(reopened.view(), { displayName: 'fixture家长' });
  assert.deepEqual((await readdir(directory)).sort(), ['account.json', 'second.json']);
});

test('missing initialization, malformed files and invalid account input fail without echoing secrets', async t => {
  const { directory, account } = await fixture(t);
  await assert.rejects(openLocalAccount(path.join(directory, 'missing.json')), /initialKey/);
  await writeFile(path.join(directory, 'bad.json'), KEY);
  await assert.rejects(openLocalAccount(path.join(directory, 'bad.json')), error => !error.message.includes(KEY));
  for (const name of ['', ' ', 'x'.repeat(81), null]) await assert.rejects(account.setDisplayName(name), /昵称/);
  for (const key of ['', null, 'x'.repeat(513)]) await assert.rejects(account.changeKey(KEY, key), /密钥/);
});

test('key changes persist a new salt/hash and affect only subsequent local verification', async t => {
  const { account, file } = await fixture(t);
  const before = JSON.parse(await readFile(file, 'utf8'));
  await assert.rejects(account.changeKey('wrong-fixture', NEXT_KEY), { code: 'INVALID_KEY' });
  assert.equal(await account.verify(KEY), true);
  assert.deepEqual(await account.changeKey(KEY, NEXT_KEY), { ok: true });
  assert.equal(await account.verify(KEY), false); assert.equal(await account.verify(NEXT_KEY), true);
  const after = JSON.parse(await readFile(file, 'utf8'));
  assert.notEqual(before.salt, after.salt); assert.notEqual(before.hash, after.hash);
  assert.equal(await (await openLocalAccount(file)).verify(NEXT_KEY), true);
});

test('anonymous entrance excludes private data; trust rejection precedes auth; true alone reads native index', async t => {
  const { account } = await fixture(t, { initialKey: KEY, displayName: 'PRIVATE_FIXTURE_NAME' });
  const h = harness(account);
  const { response: anonymous, request } = await h.rawRequest('/');
  assert.equal(anonymous.status, 401); assert.match(anonymous.headers.get('content-type'), /text\/html/);
  const html = await anonymous.text();
  assert.match(html, /createNebulaRenderer/); assert.match(html, /type="password"/); assert.match(html, /location.replace\('\/'\)/);
  for (const secret of ['PRIVATE_FIXTURE_NAME', KEY, LAUNCH, 'FIXTURE_NATIVE_BOOT_SESSION_BODY', '__DSH', '/api/family']) assert.equal(html.includes(secret), false, secret);
  assert.equal((html.match(/<input /g) ?? []).length, 1);
  assert.equal(h.calls.authorization[0], request); assert.equal(h.calls.reads.length, 0);
  const count = h.calls.authorization.length;
  assert.equal((await h.rawRequest('/', { headers: { host: 'evil.invalid' } })).response.status, 403);
  assert.equal(h.calls.authorization.length, count);
  assert.equal((await h.rawRequest('/', { headers: { cookie: V2 } })).response.status, 401);
  const signedIn = (await h.rawRequest('/', { headers: { cookie: V1 } })).response;
  assert.equal(signedIn.status, 200); assert.match(await signedIn.text(), /FIXTURE_NATIVE_BOOT_SESSION_BODY/);
  assert.equal(h.calls.reads.length, 1);
  h.cleanup(); assert.equal(h.raw.size, 0); assert.equal(h.api.size, 0);
});

test('anonymous motion bootstrap retains measured zero and layer/head positions while excluding private card fields', async t => {
  const card = { schema: 1, identity: { name: 'PRIVATE_CARD_NAME' }, branch_id: 'PRIVATE_BRANCH',
    source: { cycle: 2612, checkpoint: 'PRIVATE_CHECKPOINT', sha256: 'PRIVATE_HASH' },
    age: { byte_events: 2699511, born_at: '2026-11-15T21:03:31+08:00', private: 'PRIVATE_AGE' },
    latest: { momentum_norms: [0, 25, NaN] },
    dynamics: { frequency: [[0, 1.2], [4, -1], [Infinity, 2]], private: 'PRIVATE_DYNAMICS' },
    history: 'PRIVATE_HISTORY' };
  const expected = { schema: 1, source: { cycle: 2612 }, age: { byte_events: 2699511, born_at: '2026-11-15T21:03:31+08:00' }, latest: { momentum_norms: [0, 25, null] },
    dynamics: { frequency: [[0, 1.2], [4, null], [null, 2]] } };
  assert.deepEqual(entranceReadings(card), expected);
  assert.equal(entranceReadings({ ...card, source: { cycle: -1 } }), null);
  const h = harness((await fixture(t)).account, { getEntranceCard: async () => card });
  const html = await (await h.rawRequest('/')).response.text();
  assert.ok(html.includes(`window.MengliIntro.updateLife(${JSON.stringify(expected)})`));
  for (const secret of ['PRIVATE_CARD_NAME', 'PRIVATE_BRANCH', 'PRIVATE_CHECKPOINT', 'PRIVATE_HASH', 'PRIVATE_DYNAMICS', 'PRIVATE_HISTORY', 'PRIVATE_AGE']) {
    assert.equal(html.includes(secret), false, secret);
  }
  h.cleanup();
});

test('body bridge failure keeps the anonymous key form available', async t => {
  const h = harness((await fixture(t)).account, { getEntranceCard: async () => { throw new Error('offline'); } });
  const response = (await h.rawRequest('/')).response;
  assert.equal(response.status, 401);
  assert.match(await response.text(), /window.MengliIntro.updateLife\(null\)/);
  h.cleanup();
});

test('old native launch URLs preserve their official 303 token exchange', async t => {
  const h = harness((await fixture(t)).account);
  const { response, request } = await h.rawRequest(`/?token=${LAUNCH}`);
  assert.equal(response.status, 303); assert.equal(response.headers.get('location'), './');
  assert.match(response.headers.get('set-cookie'), /fixture-v1-browser/);
  assert.equal(h.calls.authorization[0], request); assert.equal(h.calls.reads.length, 0);
});

test('local login issues the native cookie in-process without forwarding token or redirect', async t => {
  const h = harness((await fixture(t)).account);
  const wrong = (await h.rawRequest('/auth/login', { method: 'POST', body: { key: 'wrong-fixture' } })).response;
  assert.equal(wrong.status, 401); assert.equal(wrong.headers.has('set-cookie'), false); assert.equal(h.calls.exchanges, 0);
  const success = (await h.rawRequest('/auth/login', { method: 'POST', body: { key: KEY } })).response;
  assert.equal(success.status, 200); assert.deepEqual(await success.json(), { ok: true });
  assert.equal(success.headers.has('location'), false); assert.match(success.headers.get('set-cookie'), /HttpOnly; SameSite=Strict/);
  assert.equal(h.calls.exchanges, 1); assert.equal(h.calls.reads.length, 0);
  assert.equal((await h.rawRequest('/auth/login')).response.status, 405);
});

test('login fences Host, Origin, fetch-site, content type and streaming body bytes', async t => {
  const h = harness((await fixture(t)).account);
  for (const headers of [{ host: 'evil.invalid' }, { origin: 'https://evil.invalid' }, { origin: undefined },
    { 'sec-fetch-site': 'cross-site' }, { 'sec-fetch-site': 'same-site' }, { 'sec-fetch-site': 'none' }]) {
    assert.equal((await h.rawRequest('/auth/login', { method: 'POST', body: { key: KEY }, headers })).response.status, 403);
  }
  assert.equal((await h.rawRequest('/auth/login', { method: 'POST', body: { key: KEY }, headers: { 'content-type': 'text/plain' } })).response.status, 415);
  assert.equal((await h.rawRequest('/auth/login', { method: 'POST', chunks: ['{broken'] })).response.status, 400);
  assert.equal((await h.rawRequest('/auth/login', { method: 'POST', body: { key: KEY, extra: true } })).response.status, 400);
  for (const chunks of [[' '.repeat(2049)], [' '.repeat(1024), ' '.repeat(1025)], ['玉'.repeat(700)]]) {
    assert.equal((await h.rawRequest('/auth/login', { method: 'POST', chunks })).response.status, 413);
  }
  assert.equal((await h.rawRequest('/auth/login', { method: 'POST', body: { key: KEY }, headers: { 'content-length': '2049' } })).response.status, 413);
  assert.equal(h.calls.exchanges, 0);
});

test('native password form fallback preserves verification and redirects with the official cookie', async t => {
  const h = harness((await fixture(t)).account);
  const headers = { 'content-type': 'application/x-www-form-urlencoded' };
  const { response } = await h.rawRequest('/auth/login', { method: 'POST', chunks: [new URLSearchParams({ key: KEY }).toString()], headers });
  assert.equal(response.status, 303); assert.equal(response.headers.get('location'), '/');
  assert.ok(response.headers.get('set-cookie').includes(V1));
  assert.equal((await h.rawRequest('/auth/login', { method: 'POST', chunks: ['key=a&key=b'], headers })).response.status, 400);
});

test('every account API enforces v1 even though the outer fence admits v2 operators', async t => {
  const h = harness((await fixture(t)).account);
  assert.deepEqual([...h.api.keys()], ['/api/local-account', '/api/local-account/profile', '/api/local-account/key', '/api/local-account/logout']);
  for (const url of h.api.keys()) {
    for (const cookie of ['', V2, `${COOKIE_NAME}=forged`]) {
      const response = await h.fetchApi(`${url}?token=${LAUNCH}`, { headers: { cookie } });
      assert.equal(response.status, 401, `${url}: ${cookie}`);
      const text = await response.text(); assert.equal(text.includes(KEY), false); assert.equal(text.includes('无月'), false);
    }
  }
  assert.equal(h.calls.exchanges, 0);
  assert.deepEqual(await (await h.fetchApi()).json(), { displayName: '无月酱酱酱酱' });
  const probes = h.calls.authorization.filter(request => request.headers instanceof Headers);
  assert.ok(probes.length >= 5); assert.ok(probes.every(request => request.url === '/' && request.method === 'GET'));
});

test('native GET bridge must select buffered mode; POST retains streaming limits', async t => {
  const h = harness((await fixture(t)).account);
  const get = h.api.get('/api/local-account');
  const nativeRequest = requestBody => new Request(`${ORIGIN}/api/local-account`, {
    method: 'GET', headers: { host: HOST, cookie: V1 },
    ...(requestBody === 'streaming' ? { body: Readable.toWeb(Readable.from([])), duplex: 'half' } : {}),
  });
  assert.throws(() => nativeRequest('streaming'), /GET\/HEAD method cannot have body/);
  assert.equal((await get.fetch(nativeRequest(get.requestBody))).status, 200);
  for (const route of h.api.values()) {
    if (route.methods.includes('POST')) assert.equal(route.requestBody, 'streaming');
  }
});

test('profile/key/logout routes return only public data, use current-authority official cookie name', async t => {
  const { account, file } = await fixture(t);
  const h = harness(account);
  assert.deepEqual(await (await h.fetchApi('/api/local-account/profile', { body: { displayName: 'fixture昵称' } })).json(), { displayName: 'fixture昵称' });
  assert.deepEqual((await openLocalAccount(file)).view(), { displayName: 'fixture昵称' });
  assert.equal((await h.fetchApi('/api/local-account/key', { body: { currentKey: 'wrong-fixture', newKey: NEXT_KEY } })).status, 401);
  assert.deepEqual(await (await h.fetchApi('/api/local-account/key', { body: { currentKey: KEY, newKey: NEXT_KEY } })).json(), { ok: true });
  assert.equal((await h.rawRequest('/auth/login', { method: 'POST', body: { key: KEY } })).response.status, 401);
  assert.equal((await h.rawRequest('/auth/login', { method: 'POST', body: { key: NEXT_KEY } })).response.status, 200);
  // Changing a key does not revoke already-issued native browser cookies.
  assert.equal((await h.fetchApi()).status, 200);
  const logout = await h.fetchApi('/api/local-account/logout');
  assert.deepEqual(await logout.json(), { ok: true });
  assert.equal(logout.headers.get('set-cookie'), `${COOKIE_NAME}=; Max-Age=0; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict`);
  assert.equal(logout.headers.has('location'), false);
  assert.equal((await h.fetchApi('/api/local-account/profile', { body: { displayName: '' } })).status, 400);
  assert.equal((await h.fetchApi('/api/local-account/key', { body: { currentKey: NEXT_KEY, newKey: '' } })).status, 400);
  assert.equal((await h.fetchApi('/api/local-account/logout', { body: { extra: true } })).status, 400);
  assert.equal((await h.fetchApi('/api/local-account/profile', { chunks: [' '.repeat(1000), ' '.repeat(1049)] })).status, 413);
  assert.equal((await h.fetchApi('/api/local-account/profile', { body: { displayName: 'unused' }, headers: { 'sec-fetch-site': 'cross-site' } })).status, 403);
});
