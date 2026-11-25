// Committed family records are an independent read model, never Session JSONL.
export const FAMILY_VIEW = 'mengli';

export function createFamilyClient({ fetcher = globalThis.fetch, EventSourceImpl = globalThis.EventSource } = {}) {
  let snapshot = { connected: false, state: null, history: null, historyError: null, error: null };
  const listeners = new Set();
  const records = new Map();
  const publish = patch => {
    snapshot = { ...snapshot, ...patch };
    for (const listener of listeners) listener();
  };
  const adopt = state => {
    if (snapshot.state?.observation?.run_id !== state.observation?.run_id || snapshot.state?.branch_id !== state.branch_id) records.clear();
    for (const event of state.events ?? []) {
      if (event.status === 'committed') records.set(event.cursor, event);
    }
    while (records.size > 128) records.delete(records.keys().next().value);
    publish({ connected: true, error: null, state: { ...state, events: [...records.values()].sort((a, b) => a.cursor - b.cursor) } });
    return state;
  };
  async function request(path = '', body, { signal } = {}) {
    const response = await fetcher(`/api/family${path}`, {
      method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin', signal,
      ...(body === undefined ? {} : { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error?.message ?? value.error ?? value.message ?? `HTTP ${response.status}`);
    return value;
  }
  let stream;
  return {
    getSnapshot: () => snapshot,
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    adopt,
    async connect() {
      try { adopt(await request()); }
      catch (error) { publish({ connected: false, error: error.message }); }
      try { publish({ history: await request('/history'), historyError: null }); }
      catch (error) { publish({ historyError: error.message }); }
      stream = new EventSourceImpl('/api/family/events', { withCredentials: true });
      stream.addEventListener('family', event => { adopt(JSON.parse(event.data)); });
      stream.onerror = () => publish({ connected: false, error: '家庭连接中断；未提交的消息不会显示' });
    },
    dispose() { stream?.close(); listeners.clear(); },
    async command(name, body = {}, options) {
      try { return adopt(await request(`/${name}`, body, options)); }
      catch (error) { publish({ error: error.message }); throw error; }
    },
    async say(text, signal) {
      const state = snapshot.state;
      if (state?.live_steward) {
        if (!text.trim()) throw new Error('请输入家长文字');
        await this.command('say', { text }, { signal });
        return { kind: 'success' };
      }
      if (state?.read_only) throw new Error('研究训练只读观察，当前不能从网页接管或发送输入');
      if (!snapshot.connected || !state?.loaded || state.mode !== 'parent' || state.busy || state.pending_input) {
        throw new Error('请先连接身体并由家长接管，上一轮完整保存后才能输入');
      }
      if (!text.trim()) throw new Error('请输入家长文字');
      await this.command('say', { text }, { signal });
      return { kind: 'success' };
    },
  };
}

// Source-only legacy conversation: no delivered receipt and no invented body cycle.
export function projectFamilyHistory(history, cache = new Map()) {
  const turns = history?.turns ?? [];
  const reference = history?.checkpoint_ref ?? history?.session_ref ?? history?.ref;
  const source = typeof reference === 'string' ? reference : JSON.stringify(reference ?? 'saved-source');
  return turns.flatMap((turn, index) => {
    if (!['user', 'assistant'].includes(turn.role) || typeof turn.content !== 'string' || !turn.content) return [];
    const key = `history:${source}:${index}`;
    let node = cache.get(key);
    if (!node) {
      const mother = turn.role === 'assistant';
      const label = mother ? '妈妈' : '梦漓';
      node = { key, target: 'chat', anchorSeq: index - turns.length - 1,
        location: { kind: 'session' }, visibility: 'visible', kind: mother ? 'user' : 'assistant-step',
        family: { source: 'saved-history', actor: mother ? 'mother' : 'mengli', original_role: turn.role, reference, label },
        data: mother
          ? { kind: 'user', messageId: key, seq: index - turns.length - 1, content: [{ type: 'text', text: turn.content }],
            source: { kind: 'family-history', actor: 'mother', label } }
          : { status: 'settled', blocks: [{ kind: 'text', text: `**${label}**\n\n${turn.content}` }], step: 1 },
      };
      cache.set(key, node);
    }
    return [node];
  });
}

function isNonTrivialEvent(event) {
  const ch = event.umwelt?.channel;
  if (ch) {
    // Keep MengLi's real vocal expressions (including self-talk `inner_monologue` & `read_aloud_solo`)
    // visible in the MengLi stream; only hide silent routine book/evolution pulses with no speech.
    if (ch === 'storybook_spike_fallback' || ch === 'book_navigation' || ch === 'self_evolution') return false;
    return true;
  }
  return event.observation?.kind !== 'natural_text_reply';
}

function mengliActionLabel(event) {
  const ch = event.umwelt?.channel;
  const mode = event.umwelt?.interaction_mode;
  if (ch === 'inner_monologue' || mode === 'inner_monologue') return '梦漓 · 自言自语';
  if (ch === 'read_aloud_solo' || mode === 'read_aloud_solo') return '梦漓 · 自读出声';
  if (mode === 'self_talk_observing_mother') return '梦漓 · 看着妈妈自言自语';
  if (mode === 'call_book_and_pause' || mode === 'call_mother_and_pause') return '梦漓 · 自言自语';
  return '梦漓';
}

function actorDisplayLabel(event, caregiver) {
  const ch = event.umwelt?.channel;
  const sp = event.umwelt?.speaker;
  if (caregiver === 'parent' || ch === 'live_participant') return sp && sp !== 'parent' ? sp : '爸爸';
  if (ch === 'tool' || ch === 'dynamic_object') return sp || '工具';
  if (ch === 'book_addressed') return '书页 · 检索';
  if (ch === 'read_aloud_solo') return '书页';
  if (ch === 'mother_quiet_presence') return '妈妈 · 安静陪伴（书页）';
  return '妈妈';
}

// Stable native Chat identities: show real MengLi utterances (including self-talk) and social replies.
export function projectFamilyEvents(events, cache = new Map()) {
  const nodes = [];
  for (const event of events) {
    if (event.status !== 'committed') continue;
    if (!isNonTrivialEvent(event)) continue;
    const time = typeof event.time === 'number' ? event.time : undefined;
    const caregiver = event.actor ?? event.caregiver;
    const observed = event.observation;
    const actionLabel = mengliActionLabel(event);
    const replyLabel = actorDisplayLabel(event, caregiver);
    const umwelt = event.umwelt ?? null;
    const versionSig = umwelt ? `${umwelt.channel}:${umwelt.interaction_mode ?? ''}:${umwelt.perspective_mode ?? ''}:${(umwelt.mirror_excerpt ?? '').length}:${(umwelt.book_excerpt ?? '').length}` : 'v0';
    for (const [part, text, label, assistant] of [
      ['action', event.utterance?.text, actionLabel, true],
      ['input', event.delivered?.text ?? observed?.text, replyLabel, false],
    ]) {
      if (!text) continue;
      const key = `family:${event.cursor}:${part}:${versionSig}`;
      let node = cache.get(key);
      if (!node) {
        node = {
          key, target: 'chat', anchorSeq: event.cursor * 2 + (part === 'action' ? 0 : 1),
          location: { kind: 'session' }, visibility: 'visible',
          kind: assistant ? 'assistant-step' : 'user',
          family: { cursor: event.cursor, actor: part === 'action' ? 'mengli' : caregiver, label,
            timing: part === 'action' ? 'before_caregiver_input' : observed ? 'recorded_reply' : 'delivered', saved: event.saved ?? event.checkpoint ?? event.checkpoint_ref },
          data: assistant
            ? { status: 'settled', turn: event.cursor, step: 1, blocks: [{ kind: 'text', text: `**${label}**\n\n${text}` }], time }
            : { kind: 'user', messageId: key, seq: event.cursor * 2 + 1, time, content: [{ type: 'text', text }],
                source: { kind: 'family', actor: caregiver, label, cursor: event.cursor, umwelt } },
        };
        cache.set(key, node);
      }
      nodes.push(node);
    }
  }
  return nodes;
}

export function projectUmweltTrajectory(events) {
  const items = [];
  for (const event of events ?? []) {
    if (event.status !== 'committed') continue;
    const u = event.umwelt ?? {};
    const ch = u.channel ?? (event.observation?.kind === 'natural_text_reply' ? 'storybook_spike_fallback' : 'dsh_mother');
    const spoke = Boolean(event.utterance?.spoke ?? event.utterance?.text);
    const mode = u.interaction_mode ?? null;
    const isSocial = ch !== 'storybook_spike_fallback' && ch !== 'book_navigation' && ch !== 'read_aloud_solo';
    let route = '静读停顿 · 书页';
    if (ch === 'inner_monologue' || mode === 'inner_monologue') route = '梦漓自言自语（out ∅ · in ∅）';
    else if (ch === 'self_evolution' || mode === 'quiet_self_evolution') route = '梦漓内部自我演化（out ∅ · in ∅）';
    else if (ch === 'read_aloud_solo' || mode === 'read_aloud_solo') route = '梦漓自读出声 · 书页';
    else if (ch === 'mother_quiet_presence') route = '妈妈安静陪伴（未打断）';
    else if (ch === 'live_participant') route = `${u.speaker || '爸爸'} → 梦漓`;
    else if (ch === 'reply_to_parent') route = `梦漓指向 → ${u.speaker || '爸爸'}`;
    else if (ch === 'call_parent') route = `梦漓呼唤 → ${u.speaker || '爸爸'}`;
    else if (ch === 'tool' || ch === 'dynamic_object') route = `梦漓指向 → ${u.speaker || '工具'}`;
    else if (ch === 'book_addressed') route = '梦漓 @书页 → 书页检索';
    else if (ch === 'book_navigation') route = u.routing_reason === 'gaze_book_search' ? '梦漓指向书页 · 检索' : '梦漓 → 书页（自主检索）';
    else if (ch === 'dsh_mother') {
      if (u.mother_out === 'parent') route = '妈妈对爸爸说（梦漓旁听）';
      else if (mode === 'face_to_face_chat') route = '梦漓与妈妈 · 面对面聊天';
      else if (mode === 'co_reading_talk') route = '梦漓与妈妈 · 共读对话';
      else if (mode === 'self_talk_observing_mother') route = '梦漓看着妈妈 · 自言自语';
      else if (u.routing_reason === 'gaze_mother_speak') route = '梦漓指向妈妈 · 开口';
      else if (u.routing_reason === 'gaze_mother_look') route = '梦漓抬头看妈妈';
      else if (u.routing_reason === 'vocal_call_mother' || u.routing_reason === 'explicit_mother') route = '梦漓呼唤 → 妈妈';
      else if (u.trigger_kind === 'mother_proactive') route = '妈妈共读开口 → 梦漓';
      else if (u.trigger_kind === 'child_replied') route = '梦漓回应 → 妈妈';
      else if (spoke) route = '梦漓开口 → 妈妈';
      else route = '妈妈伴读 → 梦漓';
      if (u.mother_out === 'book') route += ' · 妈妈翻书';
    } else if (spoke) {
      route = '梦漓 → 书页（自主检索）';
    }
    const hit = u.cache_hit_tokens, miss = u.cache_miss_tokens;
    const hitRate = typeof hit === 'number' && typeof miss === 'number' && (hit + miss) > 0 ? hit / (hit + miss) : null;
    items.push({
      cycle: event.cycle ?? event.cursor,
      channel: ch,
      isSocial,
      route,
      perspective: u.perspective_mode ?? null,
      stage: u.care_stage ?? null,
      planVersion: u.care_plan_version ?? null,
      utterance: event.utterance?.text ?? '',
      spoke,
      reply: event.delivered?.text ?? event.observation?.text ?? '',
      mirrorExcerpt: u.mirror_excerpt ?? '',
      bookExcerpt: u.book_excerpt ?? '',
      metalG: typeof u.metal_g === 'number' ? u.metal_g : (typeof event.row?.metal_g === 'number' ? event.row.metal_g : null),
      nll: typeof u.response_nll === 'number' ? u.response_nll : (typeof event.row?.response_nll === 'number' ? event.row.response_nll : null),
      hitRate,
      externalBytes: event.observation?.external_bytes ?? event.row?.external_bytes ?? null,
      motherBytes: event.observation?.mother_bytes ?? null,
      learningTrace: u.learning_trace ?? null,
    });
  }
  return items.reverse();
}
