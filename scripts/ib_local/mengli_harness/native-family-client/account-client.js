// UI-only local profile. Keys never enter a session, draft, or family command.
function createLocalAccountClient({ fetcher = fetch } = {}) {
  let snapshot = { displayName: '家长', loading: true, error: null };
  const listeners = new Set();
  const publish = next => { snapshot = { ...snapshot, ...next }; listeners.forEach(listener => listener()); };
  async function request(path, body) {
    const response = await fetcher('/api/local-account' + path, { credentials: 'same-origin',
      ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error?.message ?? value.error ?? '账号操作失败');
    return value;
  }
  return {
    getSnapshot: () => snapshot,
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    async load() {
      try { const value = await request(''); publish({ displayName: value.displayName, loading: false, error: null }); }
      catch (error) { publish({ loading: false, error: error.message }); }
    },
    async saveName(displayName) {
      const value = await request('/profile', { displayName });
      publish({ displayName: value.displayName, error: null });
    },
    changeKey: (currentKey, newKey) => request('/key', { currentKey, newKey }),
    logout: () => request('/logout', {}),
    dispose() { listeners.clear(); },
  };
}
function installLocalAccountUI(ctx) {
  const h = React.createElement;
  const account = createLocalAccountClient();
  const useAccount = () => React.useSyncExternalStore(account.subscribe, account.getSnapshot);
  const open = () => { ctx.layout.setRightbarWidth(380); ctx.sidebarRight.openTab('mengli-local-account', { destination: 'sidebar' }); };
  account.load();
  ctx.effect(() => () => account.dispose());
  function AccountLauncher() {
    const value = useAccount();
    return h(Button, { size: 'sm', title: '账号设置', 'aria-label': '账号设置', onClick: open,
      style: { maxWidth: 210, overflow: 'hidden', textOverflow: 'ellipsis' } }, value.loading ? '账号' : value.displayName);
  }
  ctx.slots.inject('sidebar.footer.action', () => ctx.slots.register({
    name: 'sidebar.footer.action', id: 'mengli-local-account', order: -10,
  }, AccountLauncher));
  const id = '@mengli/native-family-client/account';
  ctx.effect(() => ctx.sidebarRightTabs.register({ id, kind: 'mengli-local-account', keepMounted: false, title: () => '账号设置' }));
  function AccountPanel() {
    const value = useAccount();
    const [displayName, setDisplayName] = React.useState(value.displayName);
    const [currentKey, setCurrentKey] = React.useState('');
    const [newKey, setNewKey] = React.useState('');
    const [confirmation, setConfirmation] = React.useState('');
    const [pending, setPending] = React.useState(false);
    const [notice, setNotice] = React.useState('');
    const [error, setError] = React.useState(null);
    const form = OFFICIAL_FORM_CLASSES;
    React.useEffect(() => setDisplayName(value.displayName), [value.displayName]);
    const action = async operation => {
      setPending(true); setNotice(''); setError(null);
      try { await operation(); } catch (failure) { setError(failure.message); }
      finally { setPending(false); }
    };
    const field = (label, type, text, set, autocomplete) => h('label', { className: form.field },
      h('span', { className: form.fieldLabel }, label), h('input', { className: form.input, type,
        'aria-label': label, autoComplete: autocomplete, value: text, maxLength: type === 'password' ? 256 : 64,
        onChange: event => set(event.target.value), style: { width: '100%', minHeight: 42, padding: '8px 10px' } }));
    return h('section', { 'aria-label': '账号设置', className: form.section,
      style: { padding: 24, height: '100%', overflow: 'auto', boxSizing: 'border-box', fontSize: 14, lineHeight: '24px' } },
      h('h2', { className: form.editorTitle }, '账号设置'),
      h('p', { className: form.intro }, '家长的显示名与本机登录密钥。'),
      value.error && h('p', { role: 'alert', className: form.error }, value.error),
      field('显示名', 'text', displayName, setDisplayName, 'nickname'),
      h(Button, { size: 'sm', disabled: pending || value.loading || !displayName.trim(), onClick: () => action(async () => {
        await account.saveName(displayName); setNotice('显示名已保存。');
      }) }, '保存显示名'),
      h('h3', { style: { marginTop: 32, marginBottom: 12 } }, '更改登录密钥'),
      field('当前密钥', 'password', currentKey, setCurrentKey, 'current-password'),
      field('新密钥', 'password', newKey, setNewKey, 'new-password'),
      field('再次输入新密钥', 'password', confirmation, setConfirmation, 'new-password'),
      h(Button, { size: 'sm', disabled: pending || !currentKey || !newKey, onClick: () => action(async () => {
        if (newKey !== confirmation) throw new Error('两次输入的新密钥不一致。');
        await account.changeKey(currentKey, newKey);
        setCurrentKey(''); setNewKey(''); setConfirmation('');
        setNotice('密钥已更改，下次登录请使用新密钥。');
      }) }, '保存新密钥'),
      h('p', { className: form.intro }, '更改密钥不会自动退出已登录的浏览器。'),
      notice && h('p', { role: 'status' }, notice),
      error && h('p', { role: 'alert', className: form.error }, error),
      h('div', { style: { marginTop: 36 } }, h(Button, { size: 'sm', disabled: pending, onClick: () => action(async () => {
        await account.logout(); window.location.replace('/');
      }) }, '退出登录')));
  }
  ctx.slots.inject('sidebar.right.pane.tab', () => ctx.slots.register({ name: 'sidebar.right.pane.tab', key: id }, AccountPanel));
  return { useAccount, open };
}
