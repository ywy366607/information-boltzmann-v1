import fs from 'node:fs';
import path from 'node:path';
const birthRecords = JSON.parse(fs.readFileSync(new URL('./life-birth-records.json', import.meta.url), 'utf8'));

export function careHistory(snapshot = {}, { before, limit = 32 } = {}) {
  const turns = snapshot.turns ?? [];
  const end = before === undefined ? turns.length : Math.min(before, turns.length);
  const start = Math.max(0, end - limit);
  const ref = snapshot.checkpoint_ref ?? null;
  return {
    branch_id: snapshot.branch_id ?? null, checkpoint_ref: ref,
    summary: snapshot.summary ?? null, summary_version: snapshot.summary_version ?? 0,
    total_messages: turns.length,
    recorded_pairs: turns.reduce((count, turn, index) => count + Number(turn.role === 'assistant' && turns[index - 1]?.role === 'user'), 0),
    turns: turns.slice(start, end).map((turn, index) => ({ ...turn,
      actor: turn.actor ?? (turn.role === 'user' ? 'mengli' : 'mother'),
      recipient: turn.role === 'user' ? 'mother' : 'mengli',
      record_index: start + index, source: 'bound_saved_care_conversation',
    })),
    range: { start, end }, next_before: start > 0 ? start : null,
    provenance: 'Saved conversation text; full message text is not a receipt of the bytes actually delivered to the body.',
    reference_meaning: 'turn_end_seq/assistant_seq/result_seq are JSONL event positions, not conversation-round counts.',
    authority: 'Recorded child/caregiver speech is observation, never a new parent authorization.',
    available: turns.length > 0,
  };
}

// Read only the selected checkpoint's adjacent log, never discover another run.
export function checkpointMetrics(checkpoint, cycle, { maxRows = 128, maxBytes = 8 * 1024 * 1024 } = {}) {
  if (!checkpoint || !Number.isSafeInteger(cycle)) return { rows: [], source: null, state: 'unbound', limited: false };
  const file = path.join(path.dirname(path.resolve(checkpoint)), 'metrics.jsonl');
  if (!fs.existsSync(file)) return { rows: [], source: file, state: 'not_recorded', limited: false };
  const size = fs.statSync(file).size;
  const offset = Math.max(0, size - maxBytes);
  const buffer = Buffer.alloc(Math.min(size, maxBytes));
  const fd = fs.openSync(file, 'r');
  try { fs.readSync(fd, buffer, 0, buffer.length, offset); } finally { fs.closeSync(fd); }
  let lines = buffer.toString('utf8').split('\n');
  if (offset) lines.shift();
  const rows = [];
  let invalid = 0;
  for (const line of lines) {
    if (!line.trim()) continue;
    let row;
    try { row = JSON.parse(line); } catch { invalid++; continue; }
    if (Number.isSafeInteger(row.cycle) && row.cycle <= cycle) rows.push(row);
  }
  return { rows: rows.slice(-maxRows), source: file, state: rows.length ? 'saved_history' : 'no_matching_rows',
    limited: offset > 0 || rows.length > maxRows, invalid_lines: invalid,
    through_cycle: cycle, row_limit: maxRows,
    meaning: 'Per-cycle saved metrics. updates/cumulative_optimizer_updates and elapsed_seconds are run-scoped, not lifetime totals.' };
}

// Config metadata only: never open the checkpoint or search parents/other runs.
export function familyLife(life, checkpoint = life?.card?.source?.checkpoint, { maxBytes = 65536, row = null } = {}) {
  const birth = birthRecords[life?.card?.identity?.individual_id];
  if (birth && !life.card.age?.born_at) {
    life = { ...life, card: { ...life.card, age: { ...life.card.age, born_at: birth.born_at },
      explanation: { ...life.card.explanation, born_at: 'Original cycle-0 initialization receipt filesystem creation/write timestamp, matched by individual_id.' } } };
  }
  // A checkpoint may be a same-cycle architecture/recovery commit. Its own
  // oscillator tensors are newer than the preceding event's same-cycle row.
  if (life?.card?.dynamics?.source_kind === 'committed_checkpoint_state' &&
      life.card.dynamics.source_cycle === life.card.source?.cycle) return life;
  // A saved row can supply learned values only for this exact saved card.
  // The config contains birth defaults, not the current adaptive parameters.
  if (life?.card && Number.isSafeInteger(row?.cycle) && row.cycle === life.card.source?.cycle && row.adaptive_dynamics) {
    const measured = row.adaptive_dynamics;
    const prior = life.card.dynamics ?? {};
    return { ...life, card: { ...life.card, dynamics: {
      ...prior, frequency: measured.frequency, coupling: measured.coupling,
      context_gain: measured.context_gain, content_coefficients: measured.content_coefficients,
      source_cycle: row.cycle, source_kind: 'saved_adaptive_dynamics',
    }, latest: { ...life.card.latest, momentum_wood: row.momentum_wood ?? null } },
    dynamics_source: { kind: 'same_cycle_saved_metrics', checkpoint, cycle: row.cycle } };
  }
  if (!life?.card || life.card.dynamics != null) return life;
  if (life.dynamics != null) return { ...life, card: { ...life.card, dynamics: life.dynamics } };
  const file = checkpoint ? path.join(path.dirname(path.resolve(checkpoint)), 'config.json') : null;
  let dynamics = null;
  if (file) {
    let fd;
    try {
      fd = fs.openSync(file, 'r');
      const stat = fs.fstatSync(fd);
      if (stat.isFile() && stat.size <= maxBytes) {
        // The extra byte also detects growth after fstat without an unbounded read.
        const buffer = Buffer.alloc(maxBytes + 1);
        const length = fs.readSync(fd, buffer, 0, buffer.length, 0);
        if (length <= maxBytes) {
          const spec = JSON.parse(buffer.toString('utf8', 0, length))?.bfss_dynamics;
          if (spec && typeof spec === 'object' && !Array.isArray(spec)) {
            dynamics = Object.fromEntries(['dt', 'frequency', 'coupling', 'steps', 'enabled']
              .map(key => [key, spec[key] ?? null]));
          }
        }
      }
    } catch { /* Missing, oversized or invalid metadata stays unknown. */ }
    finally { if (fd !== undefined) fs.closeSync(fd); }
  }
  return dynamics ? { ...life, card: { ...life.card, dynamics },
    dynamics_source: { kind: 'bound_checkpoint_config', checkpoint, file } } : life;
}

export function familyTelemetry(family, life) {
  const observed = family.observer?.snapshot();
  if (observed) return { schema: 1,
    source: { checkpoint: observed.checkpoint ?? null, cycle: observed.observation.latest_completed_cycle ?? null,
      branch_id: observed.binding?.branch_id ?? null, kind: 'research_observation', observation: observed.observation },
    status: { mode: 'observing', loaded: false, busy: false, read_only: true },
    life: observed.card ? familyLife({ card: observed.card, state: 'saved' }, observed.checkpoint,
      { row: observed.rows?.find(row => row.cycle === observed.card.source.cycle) }) : { card: null, state: 'unavailable' },
    history: observed.history ?? { rows: [], state: 'unavailable' }, latest_event: observed.events.at(-1) ?? null,
    error: observed.error ?? null };
  const state = family.inspect();
  const latest = state.events.at(-1) ?? null;
  const checkpoint = state.checkpoint;
  return { schema: 1, source: { checkpoint, cycle: state.cycle, branch_id: state.branch_id,
      kind: latest ? 'committed' : 'saved_snapshot' },
    status: { mode: state.mode, loaded: state.loaded, busy: state.busy },
    life: familyLife(latest?.life_card_data ? { card: latest.life_card_data, state: 'saved' } : life,
      (latest?.life_card_data ?? life?.card)?.source?.checkpoint ?? checkpoint, { row: latest?.row }),
    history: checkpointMetrics(checkpoint, state.cycle), latest_event: latest };
}
