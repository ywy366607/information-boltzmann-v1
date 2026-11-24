import { isIP } from 'node:net';
import { loginPage } from './login-page.mjs';

const BODY_LIMIT = 2048;
const header = (request, name) => request.headers instanceof Headers
  ? request.headers.get(name) : request.headers[name];
const failure = (status, message) => Object.assign(new Error(message), { status });
const json = (value, status = 200, headers = {}) => Response.json(value, {
  status, headers: { 'cache-control': 'no-store', ...headers },
});

function originFor(request, webServer) {
  const host = header(request, 'host');
  if (typeof host !== 'string' || /[\s/@?#\\]/.test(host)) throw failure(403, '请求来源不允许');
  let origin;
  try { origin = new URL(`${webServer.protocol}//${host}`); }
  catch { throw failure(403, '请求来源不允许'); }
  const hostname = origin.hostname;
  if (hostname !== 'localhost' && hostname !== '[::1]' && !(isIP(hostname) === 4 && hostname.startsWith('127.'))) {
    throw failure(403, '仅允许本机访问');
  }
  return origin.origin;
}
function sameOrigin(request, origin, required) {
  const supplied = header(request, 'origin');
  const site = header(request, 'sec-fetch-site');
  if (((required || supplied) && supplied !== origin) || (site && site !== 'same-origin')) {
    throw failure(403, '请求来源不允许');
  }
}
async function readJson(request, { allowForm = false } = {}) {
  const media = header(request, 'content-type')?.split(';')[0].trim().toLowerCase();
  const form = allowForm && media === 'application/x-www-form-urlencoded';
  if (media !== 'application/json' && !form) {
    throw failure(415, '请使用JSON请求');
  }
  const length = header(request, 'content-length');
  if (length !== null && length !== undefined && (!/^\d+$/.test(length) || Number(length) > BODY_LIMIT)) {
    throw failure(413, '请求过长');
  }
  const stream = request instanceof Request ? request.body : request.iterator({ destroyOnReturn: false });
  const chunks = [];
  let size = 0;
  for await (const chunk of stream ?? []) {
    const bytes = Buffer.from(chunk);
    size += bytes.length;
    if (size > BODY_LIMIT) throw failure(413, '请求过长');
    chunks.push(bytes);
  }
  try {
    const text = Buffer.concat(chunks).toString('utf8');
    if (form) {
      const entries = [...new URLSearchParams(text)];
      if (entries.length !== 1 || entries[0][0] !== 'key') throw new Error();
      return Object.fromEntries(entries);
    }
    const body = JSON.parse(text);
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error();
    return body;
  } catch { throw failure(400, 'JSON格式无效'); }
}
function fields(body, names) {
  if (Object.keys(body).some(name => !names.includes(name))) throw failure(400, '请求字段无效');
}
function keyValue(key) {
  if (typeof key !== 'string' || !key.length || Buffer.byteLength(key) > 512) {
    throw failure(400, '密钥须为非空文本，且不超过512字节');
  }
  return key;
}
function captureIndex(connection, request) {
  const capture = { allowed: false, status: 401, headers: new Headers(), body: undefined };
  capture.allowed = connection.authorizeIndex(request, {
    writeHead(status, headers) { capture.status = status; capture.headers = new Headers(headers); },
    end(body) { capture.body = body; },
  }) === true;
  return capture;
}
function browserCookie(connection, origin, headers) {
  // The launch credential never leaves this in-process exchange. Native auth owns signing.
  const capture = captureIndex(connection, {
    method: 'GET', url: connection.authenticatedUrl(`${origin}/`), headers,
  });
  const cookie = capture.headers.get('set-cookie');
  if (capture.status !== 303 || !cookie) throw failure(500, '无法建立本地登录');
  return cookie;
}
async function send(res, response) {
  const headers = Object.fromEntries(response.headers);
  if (response.headers.getSetCookie) {
    const cookies = response.headers.getSetCookie();
    if (cookies.length) headers['set-cookie'] = cookies;
  }
  res.writeHead(response.status, headers);
  res.end(response.body ? Buffer.from(await response.arrayBuffer()) : undefined);
}
function errorResponse(error) {
  return json({ error: error.status ? error.message : '本地账号操作失败' }, error.status ?? 500,
    error.status === 413 ? { connection: 'close' } : {});
}

/** Synchronous installation. Web routes use ctx.effect; Connection scopes its own Fetch registrations. */
export function installLocalLogin(ctx, { account, getEntranceCard = async () => null }) {
  const { connection, webServer } = ctx;
  ctx.effect(() => webServer.register({ kind: 'exact', path: '/', handler: async (req, res) => {
    try {
      // Enforce the official trust fence before any auth exchange or resource read.
      if (connection.requestRejection(req) === 403) throw failure(403, '请求来源不允许');
      if (req.method !== 'GET') return await send(res, new Response(null, { status: 405, headers: { allow: 'GET' } }));
      const captured = captureIndex(connection, req);
      if (captured.allowed) {
        const origin = originFor(req, webServer);
        return await send(res, await webServer.readResource(new Request(`${origin}/index.html`), connection.browserPeer));
      }
      if (captured.status !== 401) return await send(res, new Response(captured.body, {
        status: captured.status, headers: captured.headers,
      }));
      let entranceCard = null;
      try { entranceCard = await getEntranceCard(); } catch { /* Login remains available while the body bridge is offline. */ }
      return await send(res, new Response(loginPage(entranceCard), { status: 401, headers: {
        'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store',
        'referrer-policy': 'no-referrer', 'x-content-type-options': 'nosniff',
        'content-security-policy': "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; media-src data:; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
      } }));
    } catch (error) { await send(res, errorResponse(error)); }
  } }), 'mengli: local login index');

  ctx.effect(() => webServer.register({ kind: 'exact', path: '/auth/login', handler: async (req, res) => {
    try {
      if (connection.requestRejection(req) === 403) throw failure(403, '请求来源不允许');
      if (req.method !== 'POST') return await send(res, new Response(null, { status: 405, headers: { allow: 'POST' } }));
      const origin = originFor(req, webServer);
      sameOrigin(req, origin, true);
      const body = await readJson(req, { allowForm: true });
      fields(body, ['key']);
      if (!await account.verify(keyValue(body.key))) throw failure(401, '密钥不正确');
      const cookie = browserCookie(connection, origin, req.headers);
      if (header(req, 'content-type')?.startsWith('application/x-www-form-urlencoded')) {
        return await send(res, new Response(null, { status: 303, headers: { location: '/', 'set-cookie': cookie, 'cache-control': 'no-store' } }));
      }
      await send(res, json({ ok: true }, 200, { 'set-cookie': cookie }));
    } catch (error) { await send(res, errorResponse(error)); }
  } }), 'mengli: local login exchange');

  const routes = [
    ['/api/local-account', 'GET', async () => json(account.view())],
    ['/api/local-account/profile', 'POST', async request => {
      const body = await readJson(request);
      fields(body, ['displayName']);
      if (typeof body.displayName !== 'string' || !body.displayName.trim() || body.displayName.trim().length > 80) {
        throw failure(400, '昵称须为1至80个字符');
      }
      return json(await account.setDisplayName(body.displayName));
    }],
    ['/api/local-account/key', 'POST', async request => {
      const body = await readJson(request);
      fields(body, ['currentKey', 'newKey']);
      try { return json(await account.changeKey(keyValue(body.currentKey), keyValue(body.newKey))); }
      catch (error) {
        if (error.code === 'INVALID_KEY') throw failure(401, '当前密钥不正确');
        throw error;
      }
    }],
    ['/api/local-account/logout', 'POST', async (request, origin) => {
      fields(await readJson(request), []);
      const issued = browserCookie(connection, origin, request.headers);
      const name = issued.slice(0, issued.indexOf('='));
      const secure = /;\s*Secure(?:;|$)/i.test(issued) ? '; Secure' : '';
      return json({ ok: true }, 200, { 'set-cookie': `${name}=; Max-Age=0; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; SameSite=Strict${secure}` });
    }],
  ];
  for (const [path, method, handler] of routes) connection.fetch.register({
    // Native streaming transport always attaches a body; Fetch forbids one on GET.
    path, methods: [method], requestBody: method === 'GET' ? 'buffered' : 'streaming',
    fetch: async request => {
      try {
        const rejection = connection.requestRejection(request);
        if (rejection) throw failure(rejection, rejection === 403 ? '请求来源不允许' : '请先登录');
        const origin = originFor(request, webServer);
        sameOrigin(request, origin, method === 'POST');
        // requestRejection also admits v2 operator cookies. Only native v1 browser
        // auth can pass this token-free root probe; query credentials cannot help.
        if (!captureIndex(connection, { method: 'GET', url: '/', headers: request.headers }).allowed) {
          throw failure(401, '请先登录');
        }
        return await handler(request, origin);
      } catch (error) { return errorResponse(error); }
    },
  });
  return account;
}
