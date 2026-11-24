"""Read-only life cards bound to one saved body snapshot, never a live clock."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LIFE_ROOT = ROOT / 'data/harness/life'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity_for(config, source_sha, name=None):
    """Return display metadata without changing the source configuration."""
    if config.get('life_identity') is not None:
        return copy.deepcopy(config['life_identity'])
    return dict(name='梦漓' if name is None else name,
                individual_id='mengli-' + source_sha[:16])


def branch_id_for(config, checkpoint):
    context = config.get('mother_context_harness') or {}
    if context.get('kind') == 'dsh':
        return context['branch_id']
    # Non-DSH research outputs also get separate cards, not separate individuals.
    parent = str(Path(checkpoint).resolve().parent)
    return 'research-' + hashlib.sha256(parent.encode('utf-8')).hexdigest()[:24]


def _output_path(output):
    path = Path(output)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(LIFE_ROOT.resolve()):
        raise ValueError('life cards must be written under data/harness/life')
    return path


def card_config_for(config, branch_id, output=None):
    """Inherit a card only for its owner branch; shared paths require opt-in."""
    previous = config.get('life_card') or {}
    if output is None and previous.get('owner_branch_id') == branch_id:
        output = previous['path']
    if output is None:
        key = hashlib.sha256(branch_id.encode('utf-8')).hexdigest()[:24]
        output = LIFE_ROOT / ('branch-' + key + '.json')
    return dict(path=str(_output_path(output)), owner_branch_id=branch_id)


def _scalar(state, key):
    value = state.get(key)
    return None if value is None else int(value)


def from_state_dict(state_dict, *, config, checkpoint, source_sha, cycle,
                    row=None, mother_state=None, branch_id=None, name=None):
    """Extract existing memory buffers; row quantities describe this cycle only."""
    if row is not None and row.get('cycle', cycle) != cycle:
        row = None
    row = row or {}
    dynamics = row.get('bfss_dynamics') or {}
    clock = _scalar(state_dict, 'memory.clock')
    occupied = state_dict.get('memory.occupied')
    count = None if occupied is None else int(occupied.sum())
    last_write = state_dict.get('memory.last_write')
    oldest = None
    if clock is not None and count and last_write is not None:
        oldest = clock - int(last_write[occupied].min())
    harness = (mother_state or {}).get('dsh_harness') or {}
    return dict(
        schema=1,
        identity=identity_for(config, source_sha, name),
        branch_id=branch_id if branch_id is not None else branch_id_for(config, checkpoint),
        source=dict(checkpoint=str(Path(checkpoint).resolve()), sha256=source_sha, cycle=cycle),
        age=dict(cycles=cycle, byte_events=clock, born_at=None, active_seconds=None),
        memory=dict(occupied=count, capacity=None if occupied is None else occupied.numel(),
                    episodes_committed=_scalar(state_dict, 'memory.episodes'),
                    oldest_since_write_events=oldest),
        latest=dict(updates=row.get('updates'), state_change=dynamics.get('state_change'),
                    momentum_norms=copy.deepcopy(dynamics.get('momentum_norms'))),
        mother=dict(checkpoint_ref=copy.deepcopy(harness.get('checkpoint_ref')),
                    turns=harness.get('turn_count'), turn_scope='since_dsh_migration'),
        exported_at=datetime.now(timezone.utc).isoformat(),
        explanation=dict(snapshot='saved snapshot', byte_events='memory.clock; includes self-hearing',
                         episodes_committed='commit_episode calls, including merges',
                         oldest_since_write_events='byte events since oldest occupied last_write; not birth age',
                         updates='optimizer updates in this cycle, not lifetime',
                         dynamics='same-cycle bfss_dynamics; null when unrecorded'))


def write_card(card, output):
    """Atomically replace only a small JSON file inside the project life directory."""
    output = _output_path(output)
    payload = json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                         prefix=output.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return card


def _cycle_row(metrics, cycle):
    if metrics is None:
        return None
    match = None
    with Path(metrics).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('cycle') == cycle:
                    match = row
    return match


def export_checkpoint(checkpoint, *, metrics=None, output, name=None):
    """Load one checkpoint on CPU, without constructing a body or calling APIs."""
    _output_path(output)
    checkpoint = Path(checkpoint)
    source_sha = sha256_file(checkpoint)
    opener = gzip.open if checkpoint.suffix == '.gz' else open
    with opener(checkpoint, 'rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=False)
    cycle = int(saved['cycle'])
    card = from_state_dict(saved['model'], config=saved['config'], checkpoint=checkpoint,
                           source_sha=source_sha, cycle=cycle, row=_cycle_row(metrics, cycle),
                           mother_state=saved.get('world', {}).get('mother_provider_state'), name=name)
    attach_saved_dynamics(card, saved['model'], saved.get('engine', {}).get('bfss_momenta'))
    return write_card(card, output)


def export_live(engine, *, checkpoint, config, cycle, output, row=None, branch_id=None,
                committed_sha=None):
    """Call just after save: read existing buffers, never reload or run the model."""
    mother = getattr(engine.world, 'mother', None)
    mother_state = mother.state_dict() if hasattr(mother, 'state_dict') else None
    state_dict = engine.model.state_dict()
    card = from_state_dict(state_dict, config=config, checkpoint=checkpoint,
                           source_sha=committed_sha or sha256_file(checkpoint), cycle=cycle, row=row,
                           mother_state=mother_state, branch_id=branch_id)
    attach_saved_dynamics(card, state_dict, getattr(engine, 'bfss_momenta', None))
    return write_card(card, output)


@torch.no_grad()
def attach_saved_dynamics(card, state_dict, momenta):
    """Read the committed body's P and oscillator coefficients, without a model forward."""
    if momenta is not None:
        values = [float(p.float().norm()) for p in momenta]
        if values and all(math.isfinite(value) for value in values):
            card['latest']['momentum_norms'] = values
            card['explanation']['momentum_norms'] = 'per-layer P norm from this committed checkpoint'
    key = next((k for k in state_dict if k.endswith('head_coupling.theta')), None)
    if key is not None:
        prefix = key[:-len('theta')]
        theta = state_dict[key]
    else:
        freq_key = next((k for k in state_dict if k.endswith('head_coupling.theta_freq')), None)
        if freq_key is None:
            return card
        prefix = freq_key[:-len('theta_freq')]
        parts = [state_dict.get(prefix + name) for name in
                 ('theta_freq', 'theta_context', 'theta_coupling', 'theta_pair')]
        if any(part is None for part in parts):
            return card
        theta = torch.cat(parts, dim=-1)
    context = state_dict.get(prefix+'context')
    scale = state_dict.get(prefix+'frequency_scale')
    if context is None:
        return card
    from scripts.ib_local.adaptive_hamiltonian import coefficients
    omega, strength = coefficients(theta, context, theta.shape[-1]-4, scale)
    card['dynamics'] = dict(frequency=omega.flatten(1).cpu().tolist(),
        coupling=strength.cpu().tolist(), context_gain=theta[:, theta.shape[-1]-4].cpu().tolist(),
        content_coefficients=theta[:, theta.shape[-1]-2:].cpu().tolist(),
        source_cycle=card['source']['cycle'], source_kind='committed_checkpoint_state')
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, help='JSONL; only the checkpoint cycle is used')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', help='display name for an individual without life_identity')
    args = parser.parse_args()
    card = export_checkpoint(args.checkpoint, metrics=args.metrics, output=args.output, name=args.name)
    print(json.dumps(card, ensure_ascii=False))


if __name__ == '__main__':
    main()
