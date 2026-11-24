import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { randomUUID } from 'node:crypto';
import { BridgeError, HARNESS_ROOT, REPO_ROOT } from './dsh_runtime.mjs';
import { TrainingObserver } from './training_observer.mjs';

const read = file => fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
function write(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(`${file}.tmp`, JSON.stringify(value, null, 2));
  fs.renameSync(`${file}.tmp`, file);
}
const requireValue = (condition, code, message) => {
  if (!condition) throw new BridgeError(code, message, 409);
};

export class BodyWorker {
  constructor({ python = process.env.MENGLI_PYTHON, endpoint, onError = () => {} } = {}) {
    this.python = python || (fs.existsSync(path.join(REPO_ROOT, '.venv/Scripts/python.exe'))
      ? path.join(REPO_ROOT, '.venv/Scripts/python.exe') : 'python');
    this.endpoint = endpoint;
    this.onError = onError;
    this.pending = new Map();
  }
  start() {
    if (this.child) return;
    this.child = spawn(this.python, ['-u', path.join(REPO_ROOT, 'scripts/ib_local/mengli_harness/body_worker.py')], {
      cwd: REPO_ROOT, env: { ...process.env, ML_CACHE_ROOT: process.env.ML_CACHE_ROOT || 'D:/ml_cache' },
      stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
    });
    const lines = createInterface({ input: this.child.stdout });
    lines.on('line', line => {
      let message;
      try { message = JSON.parse(line); } catch { return this.onError('Body worker emitted a non-protocol line'); }
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.ok) pending.resolve(message.result);
      else pending.reject(new BridgeError('BODY_FAILED', message.error?.message || 'Body worker failed', 502));
    });
    this.child.stderr.on('data', data => process.stderr.write(data));
    const failed = error => {
      for (const pending of this.pending.values()) pending.reject(new BridgeError('BODY_EXITED', error, 502));
      this.pending.clear();
      this.child = null;
      this.onError(error);
    };
    this.child.on('error', error => failed(error.message));
    this.child.on('exit', code => failed(`Body worker exited (${code})`));
  }
  request(command, payload = {}) {
    this.start();
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(JSON.stringify({ id, command, ...payload }) + '\n', error => {
        if (error) { this.pending.delete(id); reject(error); }
      });
    });
  }
  async close() {
    if (!this.child) return;
    await this.request('close');
    this.child?.stdin.end();
  }
}

export class FamilyRuntime {
  constructor({ dataRoot = path.join(HARNESS_ROOT, 'family'), lifeCardPath, worker, onEvent = () => {}, observer } = {}) {
    this.dataRoot = dataRoot;
    this.file = path.join(dataRoot, 'current.json');
    this.log = path.join(dataRoot, 'care.jsonl');
    this.lifeCardPath = lifeCardPath || path.join(HARNESS_ROOT, 'life/current.json');
    const life = read(this.lifeCardPath);
    this.state = read(this.file) || {
      schema: 1, family_id: randomUUID(), individual_id: life?.identity?.individual_id ?? null,
      branch_id: life?.branch_id ?? null, checkpoint: life?.source?.checkpoint ?? null,
      source_checkpoint: life?.source?.checkpoint ?? null, cycle: life?.source?.cycle ?? null,
      mother_checkpoint_ref: life?.mother?.checkpoint_ref ?? null,
      source_branch_id: life?.branch_id ?? null, source_mother_checkpoint_ref: life?.mother?.checkpoint_ref ?? null,
      output: path.join(dataRoot, `body-${randomUUID()}`),
      plan: { version: 0, text: '', effective_version: 0 }, cursor: 0,
    };
    this.state.pending_input = this.state.pending_input ? { ...this.state.pending_input, status: 'interrupted' } : null;
    this.mode = 'paused';
    this.loaded = false;
    this.busy = false;
    this.events = fs.existsSync(this.log)
      ? fs.readFileSync(this.log, 'utf8').trim().split('\n').filter(Boolean).map(line => JSON.parse(line)).slice(-100) : [];
    this.onEvent = onEvent;
    this.worker = worker || new BodyWorker({ onError: error => { this.loaded = false; this.error = error; this.publish(); } });
    this.workbench = null;
    this.observer = observer ?? new TrainingObserver({ bindingPath: path.join(dataRoot, 'training-observer.json') });
  }
  persist() { write(this.file, this.state); }
  inspect(after = null) {
    const observed = this.observer.snapshot();
    if (observed) return {
      schema: 1, family_id: this.state.family_id, individual_id: observed.binding?.individual_id ?? null,
      branch_id: observed.binding?.branch_id ?? null, checkpoint: observed.checkpoint ?? null,
      source_checkpoint: observed.checkpoint ?? null, cycle: observed.observation.latest_completed_cycle ?? null,
      mode: 'observing', loaded: false, busy: false, read_only: !observed.observation?.live_steward,
      live_steward: Boolean(observed.observation?.live_steward), observation: observed.observation,
      error: observed.error ?? null, cursor: observed.observation.latest_completed_cycle ?? 0,
      events: observed.events.filter(event => after === null || event.cursor > after),
      history_start: observed.events[0]?.cursor ?? null, pending_input: null,
      management_session_id: this.state.management_session_id, workbench: this.workbenchInfo ?? null,
      plan: this.state.plan, event_scope: 'Completed cycles of the explicitly attached research queue; read-only.',
      life_card_data: observed.card,
    };
    return { ...this.state, mode: this.busy && this.targetMode ? 'pausing' : this.mode,
      requested_mode: this.targetMode ?? null, loaded: this.loaded, busy: this.busy,
      error: this.error ?? null, response_bytes: this.responseBytes ?? null,
      events: this.events.filter(event => after === null || event.cursor > after),
      history_start: this.events[0]?.cursor ?? null, workbench: this.workbenchInfo ?? null,
      individual_kind: 'continuously_learning_text_organism',
      event_scope: 'events/cursor count this family workbench\'s newly committed cycles, not the individual\'s lifetime or inherited caregiver conversations',
      checkpoint_ref_meaning: 'session_id identifies a caregiver log; *_seq fields are event positions, not dialogue-round counts' };
  }
  publish() { this.onEvent(this.inspect()); }
  requireControl() {
    requireValue(!this.observer.bound, 'EXTERNAL_BODY_OWNER', '研究队列正在管理同一个身体；网页仅观察，不能从旧档另起身体或接管。');
  }
  setWorkbench(workbench, info) {
    this.workbench = workbench;
    this.workbenchInfo = info;
    this.state.management_session_id = info.session_id ?? null;
    this.persist();
    this.publish();
  }
  async load(endpoint) {
    this.requireControl();
    requireValue(!this.busy, 'BODY_BUSY', '身体正在完成当前轮');
    if (this.loaded) return this.inspect();
    requireValue(this.state.checkpoint, 'BODY_UNBOUND', '尚未绑定完整续接存档');
    this.busy = true;
    this.error = null;
    this.publish();
    try {
      const result = await this.worker.request('load', { checkpoint: this.state.checkpoint, output: this.state.output,
        mother_harness_url: endpoint, life_card: this.lifeCardPath });
      this.accept(result);
      this.loaded = true;
      this.persist();
    } catch (error) { this.error = error.message; throw error; }
    finally { this.finish(); }
    return this.inspect();
  }
  accept(result) {
    for (const key of ['branch_id', 'cycle', 'checkpoint', 'mother_checkpoint_ref']) {
      if (result[key] !== undefined) this.state[key] = result[key];
    }
    this.responseBytes = result.response_bytes ?? this.responseBytes;
  }
  finish() {
    this.busy = false;
    if (this.targetMode) { this.mode = this.targetMode; this.targetMode = null; }
    this.publish();
  }
  async control(mode) {
    this.requireControl();
    requireValue(['paused', 'mother', 'parent'].includes(mode), 'INVALID_MODE', '请选择妈妈照护、暂停或家长接管');
    if (mode !== 'mother') {
      // Admission closes before waiting for the official agent to stop scheduling.
      this.targetMode = mode;
      if (!this.busy) { this.mode = mode; this.targetMode = null; }
      this.publish();
      await this.workbench?.pauseCare?.();
    } else {
      requireValue(!this.busy && !this.targetMode, 'BODY_BUSY', '当前轮保存后才能恢复妈妈');
      this.mode = 'mother';
      this.publish();
      await this.workbench?.resumeCare?.();
    }
    return this.inspect();
  }
  updatePlan({ text, version }) {
    const observed = this.observer.snapshot();
    if (!observed?.inbox_file) this.requireControl();
    requireValue(typeof text === 'string' && text.length <= 16000, 'INVALID_PLAN', '计划须为不超过16000字符的文字');
    requireValue(version === undefined || version === this.state.plan.version, 'PLAN_CONFLICT', '计划已更新，请读取新版本后再保存');
    this.state.plan = { ...this.state.plan, version: this.state.plan.version + 1, text };
    if (observed?.inbox_file) {
      fs.mkdirSync(path.dirname(observed.inbox_file), { recursive: true });
      fs.appendFileSync(observed.inbox_file, JSON.stringify({ kind: 'care_plan', text, version: this.state.plan.version }) + '\n', 'utf8');
      this.state.plan.effective_version = this.state.plan.version;
    }
    this.persist();
    this.publish();
    return this.inspect();
  }
  async step({ cycles = 1 } = {}) {
    this.requireControl();
    requireValue(Number.isSafeInteger(cycles) && cycles >= 1 && cycles <= 8, 'INVALID_CYCLES', '每次执行1至8轮');
    requireValue(this.mode === 'mother' && !this.targetMode, 'CARE_PAUSED', '请先明确恢复妈妈照护');
    requireValue(this.loaded, 'BODY_NOT_LOADED', '请先连接完整续接体');
    requireValue(!this.busy, 'BODY_BUSY', '身体正在完成当前轮');
    this.busy = true;
    this.publish();
    const committed = [];
    try {
      for (let i = 0; i < cycles && !this.targetMode; i++) committed.push(await this.runStep('mother'));
    } finally { this.finish(); }
    return { ...this.inspect(), committed };
  }
  async say({ text }) {
    const observed = this.observer.snapshot();
    if (observed?.inbox_file && observed.observation?.status === 'running') {
      requireValue(typeof text === 'string' && text.trim().length > 0 && Buffer.byteLength(text) <= 16384,
        'INVALID_INPUT', '请输入不超过16384字节的文字');
      fs.mkdirSync(path.dirname(observed.inbox_file), { recursive: true });
      fs.appendFileSync(observed.inbox_file, JSON.stringify({
        kind: 'participant', speaker: '爸爸', role: 'parent', text: text.trim(),
      }) + '\n', 'utf8');
      this.publish();
      return { ...this.inspect(), queued_live_inbox: true };
    }
    this.requireControl();
    requireValue(this.mode === 'parent' && !this.targetMode, 'NOT_PARENT', '暂停妈妈并接管后才能对梦漓说话');
    requireValue(this.loaded && !this.busy, 'BODY_BUSY', '请先连接身体并等待当前轮保存');
    requireValue(typeof text === 'string' && text.trim().length > 0 && Buffer.byteLength(text) <= 16384,
      'INVALID_INPUT', '请输入不超过16384字节的文字');
    requireValue(!this.state.pending_input, 'PENDING_INPUT', '上一条输入尚未确认，请核查完整存档');
    this.state.pending_input = { id: randomUUID(), text, status: 'queued' };
    this.persist();
    this.busy = true;
    this.publish();
    let committed;
    try { committed = await this.runStep('parent', text); }
    finally { this.finish(); }
    return { ...this.inspect(), committed: [committed] };
  }
  async runStep(caregiver, text) {
    const plan = { version: this.state.plan.version, text: this.state.plan.text };
    try {
      const result = await this.worker.request('step', { caregiver, text, plan });
      this.accept(result);
      const event = { ...result, cursor: this.state.cursor + 1, actor: caregiver, recipient: 'mengli',
        channel: 'care', status: 'committed', input_id: this.state.pending_input?.id ?? null };
      fs.mkdirSync(this.dataRoot, { recursive: true });
      fs.appendFileSync(this.log, JSON.stringify(event) + '\n');
      this.state.cursor = event.cursor;
      this.state.plan.effective_version = plan.version;
      this.state.pending_input = null;
      this.persist();
      this.events.push(event);
      this.events = this.events.slice(-100);
      this.error = null;
      this.publish();
      return event;
    } catch (error) {
      this.error = error.message;
      this.mode = 'paused';
      this.loaded = false;
      if (this.state.pending_input) this.state.pending_input.status = 'failed';
      this.persist();
      throw error;
    }
  }
  async close() { await this.worker.close(); await this.workbench?.close?.(); }
}
