// Explicit, bounded, read-only attachment to one research-owned individual.
// Never loads Torch/checkpoints, discovers runs, or takes ownership of training.
import fs from 'node:fs';
import path from 'node:path';
import { HARNESS_ROOT, REPO_ROOT } from './dsh_runtime.mjs';
import { familyLife } from './family_readmodel.mjs';

function inside(file, root) {
  const relative = path.relative(root, file);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}
function boundedPath(file, roots = [REPO_ROOT]) {
  if (typeof file !== 'string') throw new Error('Observation path missing');
  const resolved = path.resolve(REPO_ROOT, file);
  if (!roots.some(root => inside(resolved, root)) ||
      (fs.existsSync(resolved) && !roots.some(root => inside(fs.realpathSync(resolved), root)))) {
    throw new Error('Observation path outside repository');
  }
  return resolved;
}
function json(file, max = 1024 * 1024) {
  const fd = fs.openSync(file, 'r');
  try {
    const size = fs.fstatSync(fd).size;
    if (size > max) throw new Error('Observation metadata exceeds budget');
    const buffer = Buffer.alloc(size + 1), n = fs.readSync(fd, buffer, 0, buffer.length, 0);
    if (n > max) throw new Error('Observation metadata grew beyond budget');
    return JSON.parse(buffer.toString('utf8', 0, n));
  } finally { fs.closeSync(fd); }
}
const alive = pid => { try { if (!Number.isSafeInteger(pid) || pid < 1) return false; process.kill(pid, 0); return true; } catch { return false; } };
const rowKeys = ['cycle', 'updates', 'cumulative_optimizer_updates', 'elapsed_seconds', 'response_nll',
  'since_birth_updates', 'since_birth_external_bytes', 'since_birth_self_bytes',
  'mother_response_nll', 'ambient_response_nll', 'progress', 'joint_progress', 'metal_g', 'witness_g',
  'choice', 'spoke', 'self_bytes', 'external_bytes', 'fixed_validation', 'water_credit_records',
  'wood_policy_tv', 'bfss_dynamics', 'learned_head_coupling', 'shared_unknown',
  'adaptive_dynamics', 'momentum_wood', 'internal_ticks'];
function compact(row) {
  const result = Object.fromEntries(rowKeys.filter(k => k in row).map(k => [k, row[k]]));
  if (row.causal_program) result.causal_program = Object.fromEntries(
    ['active_relations', 'evidence_count', 'mother_prequential', 'question_norm', 'roles']
      .filter(k => k in row.causal_program).map(k => [k, row.causal_program[k]]));
  result.phases = {
    episodes: row.causal_archive?.episodes, relations: row.causal_archive?.relations, references: row.causal_archive?.references,
    candidates: row.shared_unknown?.candidate_count, drafts: row.joint_language_search?.candidates?.length,
    reason: row.expression?.reason, received: row.world?.mother_bytes, fact: row.causal_program?.committed_episode_identity,
    expected: row.causal_use?.expected_use, verified: row.causal_use?.finite_U?.verified,
    clock: row.persistent_rhythm?.clock, quiet: row.persistent_rhythm?.within_learning_quiet,
  };
  return result;
}

export class TrainingObserver {
  constructor({ bindingPath = path.join(HARNESS_ROOT, 'workbench/training-observer.json') } = {}) {
    this.bindingPath = bindingPath;
    this.logs = new Map();
  }
  get bound() { return fs.existsSync(this.bindingPath); }
  readRows(file) {
    if (!fs.existsSync(file)) return { rows: [], limited: false };
    const stat = fs.statSync(file), key = `${stat.size}:${stat.mtimeMs}`;
    if (this.logs.get(file)?.key === key) return this.logs.get(file).value;
    const size = Math.min(stat.size, 8 * 1024 * 1024), offset = stat.size - size;
    const buffer = Buffer.alloc(size), fd = fs.openSync(file, 'r');
    let n;
    try { n = fs.readSync(fd, buffer, 0, size, offset); } finally { fs.closeSync(fd); }
    const lines = buffer.toString('utf8', 0, n).split('\n');
    if (offset) lines.shift();
    lines.pop(); // A final record is visible only after its newline is committed.
    const rows = [];
    for (const line of lines) {
      try { const row = JSON.parse(line); if (Number.isSafeInteger(row.cycle)) rows.push(row); } catch { /* torn/non-record */ }
    }
    const value = { rows: rows.slice(-128), limited: offset > 0 || rows.length > 128, mtime: stat.mtime.toISOString() };
    this.logs.set(file, { key, value });
    if (this.logs.size > 32) this.logs.delete(this.logs.keys().next().value);
    return value;
  }
  snapshot() {
    if (!this.bound) return null;
    try {
      const binding = json(this.bindingPath, 32768);
      if (binding.schema !== 1 || !binding.run_id || !binding.individual_id || !binding.branch_id ||
          !Array.isArray(binding.segments) || !binding.segments.length || binding.segments.length > 32) {
        throw new Error('Invalid explicit observation binding');
      }
      // Storage roots are a local administrator binding, never an HTTP input.
      // Permit an explicit body folder on another drive, not drive-wide discovery.
      const extraRoots = binding.storage_roots ?? [];
      if (!Array.isArray(extraRoots) || extraRoots.length > 4 || extraRoots.some(root =>
          typeof root !== 'string' || !path.isAbsolute(root) ||
          path.resolve(root) === path.parse(path.resolve(root)).root || !fs.existsSync(root))) {
        throw new Error('Invalid explicit observation storage roots');
      }
      const roots = [REPO_ROOT, ...extraRoots.map(root => fs.realpathSync(root))];
      const body = boundedPath(binding.body, roots), checkpoint = path.join(body, 'last.pt.gz');
      const config = json(path.join(body, 'config.json'));
      if (config.life_identity?.individual_id !== binding.individual_id ||
          (config.mother_context_harness?.branch_id ?? config.life_card?.owner_branch_id) !== binding.branch_id) throw new Error('Body identity mismatch');
      const cardFile = boundedPath(binding.life_card);
      let card = json(cardFile, 65536);
      if (card.schema !== 1 || card.identity?.individual_id !== binding.individual_id ||
          card.branch_id !== binding.branch_id || path.resolve(card.source?.checkpoint ?? '') !== checkpoint ||
          !Number.isSafeInteger(card.source?.cycle) || card.age?.cycles !== card.source.cycle ||
          !/^[0-9a-f]{64}$/.test(card.source?.sha256 ?? '')) throw new Error('Life card does not match selected body');
      const merged = new Map(); let limited = false;
      const files = binding.segments.map(segment => path.join(boundedPath(segment, roots), 'metrics.jsonl'));
      for (const file of files) {
        const log = this.readRows(file); limited ||= log.limited;
        for (const row of log.rows) if (row.cycle <= card.source.cycle) merged.set(row.cycle, row);
      }
      const all = [...merged.values()].sort((a, b) => a.cycle - b.cycle);
      limited ||= all.length > 128;
      // Each bounded file tail can retain a different interval. Do not pad a
      // recent window with a retired segment across hundreds of absent cycles.
      // The raw logs stay intact; the default live view is its contiguous tail.
      const recent = all.slice(-128);
      let start = recent.length ? recent.length - 1 : 0;
      while (start > 0 && recent[start].cycle === recent[start - 1].cycle + 1) start--;
      limited ||= start > 0;
      const rows = recent.slice(start), last = rows.at(-1), cycle = last?.cycle ?? null;
      card = familyLife({ card, state: 'saved' }, checkpoint,
        { row: rows.find(row => row.cycle === card.source.cycle) }).card;
      const topoFile = boundedPath(path.join(body, 'reservoir-topology.json'), roots);
      let topology;
      if (fs.existsSync(topoFile) && last?.cycle === card.source.cycle) {
        const topo = json(topoFile, 4096);
        if (topo.cycle === last.cycle && topo.checkpoint_sha256 === card.source.sha256)
          topology = topo;
      }
      const project = row => {
        const view = compact(row);
        if (topology && row === last) view.phases.topology = topology;
        return view;
      };
      const guardFile = boundedPath(binding.guard_status, roots), guard = json(guardFile);
      const inboxFile = binding.inbox_file ? boundedPath(binding.inbox_file, roots) : null;
      const ownerAlive = alive(guard.pid), workerAlive = alive(guard.child_pid);
      const status = guard.status === 'running' && (!ownerAlive || !workerAlive) ? 'interrupted' : guard.status;
      const events = rows.slice(-64).map(row => {
        const w = row.world ?? {};
        const src = w.source ?? 'text';
        const ch = w.channel ?? (src === 'text' ? 'storybook_spike_fallback' : 'dsh_mother');
        const actor = src === 'parent' ? 'parent' : (src === 'tool' || src === 'object') ? 'tool' : src === 'text' ? 'world' : 'mother';
        const bookExcerpt = w.book_excerpt || (src === 'text' ? (w.text_reply ?? '') : (w.text_reply ?? '')) || '';
        const cacheHit = typeof w.caregiver_cache_hit_tokens === 'number' ? w.caregiver_cache_hit_tokens : null;
        const cacheMiss = typeof w.caregiver_cache_miss_tokens === 'number' ? w.caregiver_cache_miss_tokens : null;
        return {
          status: 'committed', cursor: row.cycle, cycle: row.cycle, branch_id: binding.branch_id,
          actor, channel: 'research-observation', choice: row.choice, updates: row.updates,
          utterance: { text: w.child_text ?? '', spoke: Boolean(row.spoke), timing: 'before_caregiver_input' },
          // Full provider text is NOT a receipt of all bytes ingested (ambient fills the remaining input).
          observation: {
            text: src === 'text' ? (w.text_reply ?? '') : (w.mother_reply ?? w.text_reply ?? ''),
            kind: src === 'text' ? 'natural_text_reply' : 'recorded_mother_reply',
            mother_bytes: w.mother_bytes ?? null, external_bytes: row.external_bytes ?? null,
          },
          umwelt: {
            channel: ch,
            source: src,
            speaker: w.speaker ?? (actor === 'world' ? 'book' : actor),
            trigger_kind: w.trigger_kind ?? (row.spoke ? 'child_spoke' : 'quiet_pause'),
            routing_reason: w.routing_reason ?? null,
            active_floor: w.active_floor ?? null,
            explicit_target: w.explicit_target ?? null,
            perspective_mode: w.perspective_mode ?? null,
            care_stage: w.care_stage ?? null,
            care_plan_version: w.care_plan_version ?? null,
            mirror_excerpt: w.mirror_excerpt || '',
            mirror_trigger_cause: w.mirror_trigger_cause || '',
            book_excerpt: typeof bookExcerpt === 'string' ? bookExcerpt.slice(0, 260) : '',
            metal_g: typeof row.metal_g === 'number' ? row.metal_g : null,
            response_nll: typeof row.response_nll === 'number' ? row.response_nll : null,
            cache_hit_tokens: cacheHit,
            cache_miss_tokens: cacheMiss,
          },
          row: project(row),
        };
      });
      const observation = { run_id: binding.run_id, status, read_only: true, owner: 'research_queue',
        live_steward: Boolean(inboxFile && status === 'running'),
        guard_pid: guard.pid, worker_pid: guard.child_pid, owner_alive: ownerAlive, worker_alive: workerAlive,
        latest_completed_cycle: cycle, saved_cycle: card.source.cycle, target_cycle: binding.target_cycle ?? null,
        start_cycle: binding.start_cycle ?? null, checkpoint, checkpoint_sha256: card.source.sha256,
        data_time: last ? this.readRows(files.at(-1)).mtime : null, observed_at: new Date().toISOString(),
        meaning: 'Completed records follow full-body save; saved_cycle may lead the visible metrics by one event.' };
      return { binding, card, events, rows: rows.map(project), observation, checkpoint, inbox_file: inboxFile,
        history: { rows: rows.map(project), source: files.at(-1), sources: files, state: 'research_history',
          limited, from_cycle: rows[0]?.cycle ?? null, through_cycle: cycle, row_limit: 128,
          meaning: 'Most recent contiguous completed cycles from explicit segments; older gaps are omitted, not interpolated. Update totals and elapsed time are segment-scoped.' } };
    } catch (error) {
      // A configured but broken attachment must NEVER fall back to an old controllable body.
      return { error: error.message, events: [], rows: [], card: null,
        observation: { read_only: true, owner: 'research_queue', status: 'unavailable', error: error.message } };
    }
  }
}
