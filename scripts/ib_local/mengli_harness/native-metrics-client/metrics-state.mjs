// Read model only. Missing/null/unknown never become numeric zero.
export const MAX_METRICS_ROWS = 256;
export const isMeasured = value => typeof value === 'number' && Number.isFinite(value);
export const metricAt = (object, path) => path.split('.').reduce((value, key) => value?.[key], object);
export function meanMeasured(values = []) {
  const measured = values.filter(isMeasured);
  return measured.length ? measured.reduce((sum, value) => sum + value, 0) / measured.length : null;
}
export function formatMetric(value, integer = false) {
  if (!isMeasured(value)) return value === 'unknown' ? '未知' : '—';
  if (integer) return value.toLocaleString('zh-CN', { maximumFractionDigits: 0 });
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toLocaleString('zh-CN', { maximumFractionDigits: 4 });
}
export function metricPoints(rows, path) {
  return rows.flatMap((row, index) => {
    const y = metricAt(row, path);
    return isMeasured(y) ? [{ x: isMeasured(row.cycle) ? row.cycle : index + 1, y, index }] : [];
  });
}
export function metricLineSegments(points) {
  const segments = [];
  for (const point of points) {
    const last = segments.at(-1)?.at(-1);
    // Missing observations and a restarted cycle axis are genuine gaps.
    if (!last || point.index !== last.index + 1 || point.x !== last.x + 1) segments.push([]);
    segments.at(-1).push(point);
  }
  return segments;
}
export function metricViewport(allPoints, range = {}) {
  const parsed = {};
  for (const key of ['from','to','low','high']) {
    const raw=range[key];
    parsed[key]=raw==null||String(raw).trim()===''?null:Number(raw);
    if(parsed[key]!==null&&!Number.isFinite(parsed[key]))return {error:'请输入有效数值'};
  }
  const minX=parsed.from??Math.min(parsed.to??Infinity,...allPoints.map(p=>p.x));
  const maxX=parsed.to??Math.max(parsed.from??-Infinity,...allPoints.map(p=>p.x));
  if(minX>maxX)return {error:'cycle 起点须小于或等于终点'};
  const points=allPoints.filter(p=>p.x>=minX&&p.x<=maxX);
  const measured=points.length?points:allPoints;
  const automaticLow=Math.min(0,...measured.map(p=>p.y));
  const automaticHigh=Math.max(0,...measured.map(p=>p.y));
  const low=parsed.low??automaticLow;
  const high=parsed.high??(automaticHigh===automaticLow?automaticLow+1:automaticHigh);
  if(low>=high)return {error:'数值下限须小于上限'};
  return {points,minX,maxX,low,high,error:null};
}
export function lifeStatus(snapshot) {
  const state = snapshot.family?.state;
  if (state?.read_only) {
    const labels = { running: '训练中', complete: '训练完成', stopped: '已保存停止', interrupted: '训练中断', failed: '训练失败', waiting_resources: '等待资源', unavailable: '观察不可用' };
    return snapshot.family?.connected ? labels[state.observation?.status] ?? '已连接' : '连接中断';
  }
  if (!snapshot.family?.connected || !state?.loaded) return '已保存快照';
  if (state.busy) return '进行中';
  if (state.mode === 'paused') return '已暂停';
  if (state.mode === 'pausing') return '正在交接';
  return '已连接 · ' + (state.mode === 'mother' ? '妈妈照护' : state.mode === 'parent' ? '家长照护' : '待命');
}

export function createMetricsClient({ fetcher = globalThis.fetch, getBridge = () => globalThis.window?.__MENGLI_NATIVE_FAMILY__ } = {}) {
  let snapshot = { loading: false, error: null, source: null, card: null, rows: [], history: null, family: null, latestEvent: null, clientLimited: false };
  let initial, controller, unbindController, unbindBridge;
  let lastCursor = -1;
  let disposed = false;
  const seen = new Set();
  const listeners = new Set();
  const publish = patch => {
    if (disposed) return;
    snapshot = { ...snapshot, ...patch };
    for (const listener of listeners) listener();
  };
  function appendRow(rows, row) {
    if (!row || typeof row !== 'object') return rows;
    const index = isMeasured(row.cycle) ? rows.findIndex(value => value.cycle === row.cycle) : -1;
    const next = [...rows];
    if (index < 0) next.push(row); else next[index] = row;
    return next.slice(-MAX_METRICS_ROWS);
  }
  function adoptEvent(event) {
    const card = event?.card ?? event?.life_card_data;
    if (event?.status !== 'committed' || (!event.row && !card)) return;
    const branch = event.branch_id ?? card?.branch_id ?? event.row?.branch_id;
    const cycle = event.cycle ?? event.row?.cycle ?? card?.source?.cycle;
    if (branch && snapshot.source?.branch_id && branch !== snapshot.source.branch_id) return;
    if (isMeasured(cycle) && isMeasured(snapshot.source?.cycle) && cycle < snapshot.source.cycle) return;
    if (isMeasured(event.cursor) && event.cursor <= lastCursor) return;
    const key = `${branch ?? ''}:${event.cursor ?? cycle}`;
    if (seen.has(key)) return;
    seen.add(key);
    if (isMeasured(event.cursor)) lastCursor = event.cursor;
    if (seen.size > MAX_METRICS_ROWS * 2) seen.delete(seen.values().next().value);
    const rows = appendRow(snapshot.rows, event.row);
    publish({ rows, card: card ?? snapshot.card, latestEvent: event,
      clientLimited: snapshot.clientLimited || (snapshot.rows.length === MAX_METRICS_ROWS && rows !== snapshot.rows),
      source: { ...snapshot.source, checkpoint: event.checkpoint ?? card?.source?.checkpoint ?? snapshot.source?.checkpoint,
        cycle: cycle ?? snapshot.source?.cycle, kind: 'committed' } });
  }
  function syncFamily() {
    const bridge = getBridge();
    if (controller !== bridge?.controller) {
      unbindController?.();
      controller = bridge?.controller;
      unbindController = controller?.subscribe(syncFamily);
    }
    const family = controller?.getSnapshot?.();
    if (family) {
      publish({ family });
      const state = family.state;
      if (state?.read_only && state.error) {
        seen.clear(); lastCursor = -1;
        publish({ error: state.error, card: null, rows: [], history: null, latestEvent: null,
          source: { kind: 'research_observation', observation: state.observation } });
        return;
      }
      if (state?.read_only) {
        const card = state.life_card_data;
        // Only a server-validated saved card can restore the attachment. A
        // previous mismatch deliberately removed source.branch_id: requiring
        // that missing id here made a transient save/binding race permanent.
        const valid = card?.schema === 1 && card.identity?.individual_id === state.individual_id &&
          card.branch_id === state.branch_id && card.source?.checkpoint === state.checkpoint &&
          Number.isSafeInteger(card.source?.cycle) && card.age?.cycles === card.source.cycle;
        if (!valid) return;
        const changed = snapshot.source?.branch_id !== state.branch_id ||
          snapshot.source?.checkpoint !== state.checkpoint;
        if (changed) { seen.clear(); lastCursor = -1; }
        publish({ card, error: null, ...(changed ? { rows: [], history: null, latestEvent: null } : {}),
          source: { ...snapshot.source, kind: 'research_observation', checkpoint: state.checkpoint,
            branch_id: state.branch_id, cycle: changed ? null : snapshot.source?.cycle,
            observation: state.observation } });
      }
      // Initial family cycle is often zero. Only saved events advance the display.
      if (snapshot.source) for (const event of family.state?.events ?? []) adoptEvent(event);
      if (state?.read_only && snapshot.source?.branch_id === state.branch_id) {
        publish({ source: { ...snapshot.source, cycle: state.cycle } });
      }
    }
  }
  return {
    getSnapshot: () => snapshot,
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    open() {
      if (initial) return initial;
      publish({ loading: true });
      unbindBridge = getBridge()?.subscribe?.(syncFamily);
      syncFamily();
      initial = (async () => {
        try {
          const response = await fetcher('/api/family/telemetry', { method: 'GET', credentials: 'same-origin' });
          const value = await response.json();
          if (!response.ok) throw new Error(value.error?.message ?? value.error ?? `HTTP ${response.status}`);
          if (value.error) throw new Error(value.error);
          const allRows = value.history?.rows ?? [];
          publish({ loading: false, error: null, source: value.source, card: value.life?.card ?? null,
            rows: allRows.slice(-MAX_METRICS_ROWS), history: value.history ?? null,
            clientLimited: allRows.length > MAX_METRICS_ROWS, latestEvent: value.latest_event ?? null });
          adoptEvent(value.latest_event);
          syncFamily();
        } catch (error) {
          publish({ loading: false, error: error.message });
          // An earlier HTTP request can finish after a healthy SSE snapshot.
          // Reconcile against the current validated state, not its old error.
          syncFamily();
        }
      })();
      return initial;
    },
    dispose() { disposed = true; unbindController?.(); unbindBridge?.(); listeners.clear(); },
  };
}
