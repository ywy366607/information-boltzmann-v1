import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { patchAccountMenu, DEFAULT_PACKAGE_ROOT } from '../build.mjs';

const source = patchAccountMenu(fs.readFileSync(`${DEFAULT_PACKAGE_ROOT}/dsh-client-ui-settings-account/lib/client.js`, 'utf8'));
const start = source.indexOf('const requestSignOut = async () => {');
const end = source.indexOf('const beginSignIn =', start);
const handler = source.slice(start, end) + '\nrequestSignOut';

test('footer logout clears the local session before navigating to the entrance', async () => {
  const calls = [];
  const logout = vm.runInNewContext(handler, {
    busy: false, setBusy: value => calls.push(['busy', value]), setLogoutError: value => calls.push(['error', value]),
    fetch: async (url, options) => { calls.push(['request', url, options]); return { ok: true }; },
    window: { location: { replace: url => calls.push(['navigate', url]) } },
  });
  await logout();
  const request = calls.find(row => row[0] === 'request');
  assert.equal(request[1], '/api/local-account/logout');
  assert.equal(request[2].method, 'POST');
  assert.equal(request[2].credentials, 'same-origin');
  assert.deepEqual(calls.at(-1), ['navigate', '/']);
  const menu = source.slice(source.indexOf('items: [', start), source.indexOf('onClose:', start));
  assert.match(menu, /id: "signout", label: "退出登录"/);
  assert.doesNotMatch(menu, /id: "signin"/);
});

test('a rejected logout stays on the page and permits retry', async () => {
  const errors = [], states = [];
  const logout = vm.runInNewContext(handler, {
    busy: false, setBusy: value => states.push(value), setLogoutError: value => errors.push(value),
    fetch: async () => ({ ok: false, json: async () => ({ error: '退出失败' }) }),
    window: { location: { replace: () => assert.fail('failed logout must remain on page') } },
  });
  await logout();
  assert.equal(errors.at(-1), '退出失败');
  assert.deepEqual(states, [true, false]);
});
