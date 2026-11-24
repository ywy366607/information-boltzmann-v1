// Included inside the project-owned OFFICIAL ui-chat apply(). Both views use
// one native factory, so flow/node child slots keep a SINGLE official owner.
exports.registerNativeFamilyView = (owner, family, projectEvents, projectHistory) => {
  const builder = new ChatSnapshotBuilder();
  const chat = _deepseek_ai_dsh_client_store.createSnapshotStore(builder.empty);
  const cache = new Map();
  const timeline = { turnOrder: [], turns: new Map() };
  let previous = [];
  const update = () => {
    const snapshot = family.getSnapshot();
    const nodes = [...projectHistory(snapshot.history, cache), ...projectEvents(snapshot.state?.events ?? [], cache)];
    if (nodes.length === previous.length && nodes.every((node, index) => node === previous[index])) return;
    const upserts = nodes.filter(node => !previous.includes(node));
    previous = nodes;
    chat.set(builder.apply({ upserts, timeline }));
    builder.publish();
  };
  owner.effect(() => family.subscribe(update));
  update();
  const idle = { running: false, openState: 'open', openError: null, hasMore: false, loadingOlder: false, pendingSubmissions: [] };
  const useSource = (source, selector = value => value) => selector(react.useSyncExternalStore(source.subscribe, source.getSnapshot));
  const familyProjection = {
    useChat: selector => useSource(chat, selector),
    useSession: selector => selector(idle),
    useConversation: selector => selector({ views: { grouped: () => undefined } }),
    useProjection: () => undefined,
    useInteractionIds: selector => selector([]),
    dispatchInteraction: () => {},
    useChatNode: (key, selector) => useSource(chat.getSnapshot().nodes.source(key), selector),
    useChatNodeBottom: (key, selector) => useSource(chat.getSnapshot().nodes.bottomSource(key), selector),
    useChatNodeProcess: (key, selector) => useSource(chat.getSnapshot().nodes.processSource(key), selector),
  };
  const scrollPositions = new Map();
  function FamilyChatView(props) {
    const snapshot = react.useSyncExternalStore(family.subscribe, family.getSnapshot);
    const canonical = snapshot.state?.management_session_id;
    if (canonical && canonical !== props.sessionId) return '请在家庭主会话中打开梦漓';
    return props.renderFactorySlot('conversation.chat.native-view', {
      familyProjection: { ...familyProjection,
        chatScroll: { save: position => scrollPositions.set(props.sessionId, position), read: () => scrollPositions.get(props.sessionId) ?? null },
      },
    });
  }
  owner.slots.inject('conversation.view', () => owner.slots.register({
    name: 'conversation.view', id: 'mengli', order: 0, label: () => '梦漓',
  }, FamilyChatView));
};
