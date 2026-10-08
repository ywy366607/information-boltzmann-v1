"""No-update task reachability at the actual dynamic-read insertion point."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.optimization import initialize_dynamic_read_branch
from scripts.ib.audit_medium_dynamic_read import digest
from scripts.ib.train_plastic_conductance import unpack_belief


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def collect(model, belief, ids, duration):
    field_measurement, motion_measurement, bases = [], [], []
    handles = [
        model.readout.merge.register_forward_pre_hook(lambda _m, x: field_measurement.append(x[0].clone())),
        model.readout.motion_merge.register_forward_pre_hook(lambda _m, x: motion_measurement.append(x[0].clone())),
        model.readout.correction.register_forward_pre_hook(lambda _m, x: bases.append(x[0].clone())),
    ]
    try:
        with torch.no_grad():
            table = F.normalize(model.source.embedding.weight, dim=-1)
            prepared = model.medium.prepare_evolution()
            for event, token in enumerate(ids):
                belief, _ = model.assimilate(belief, token.reshape(1), token_features=table,
                    diagnostics=False, training_terms=False)
                belief, _ = model.advance(belief, duration, prepared=prepared, diagnostics=False)
                model.read(belief, decode=False, prepared=prepared)
                if (event + 1) % 256 == 0:
                    print(f'Collected actual continuous events {event + 1}/1024', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(bases), torch.cat(field_measurement), torch.cat(motion_measurement), belief


def logits(model, base):
    return model.decode(base + model.readout.correction(base))


def loss_and_partials(model, base, targets):
    derivatives, losses = [], []
    for left in range(0, len(targets), 32):
        local = base[left:left + 32].detach().requires_grad_(True)
        ce = F.cross_entropy(logits(model, local), targets[left:left + 32], reduction='none')
        derivatives.append(torch.autograd.grad(ce.sum(), local)[0].detach())
        losses.append(ce.detach())
    return torch.cat(losses), torch.cat(derivatives)


@torch.no_grad()
def losses_with_offset(model, base, offset, targets, step):
    pieces = []
    for left in range(0, len(targets), 32):
        predicted = logits(model, base[left:left + 32] + step * offset[left:left + 32])
        pieces.append(F.cross_entropy(predicted, targets[left:left + 32], reduction='none'))
    return torch.cat(pieces)


def summarize_direction(model, base, values, partials, targets, mean_free):
    # Use float64 for gradient-statistic contractions/mean projection. Physical
    # model evaluation remains its production FP32.
    x, v = values.double(), partials.double()
    train = slice(0, 512)
    gradient = v[train].T @ x[train] / 512
    mu = x[train].mean(0)
    projected = gradient
    if mean_free and float(mu.square().sum()) > 0:
        projected = gradient - (gradient @ mu)[:, None] * mu[None] / mu.square().sum()
    direction = -projected
    raw_norm = float(direction.norm())
    train_response = x[train] @ direction.T
    response_rms = float(train_response.square().mean().sqrt())
    if response_rms == 0:
        return {'nonzero_measurement_direction': False, 'gradient_norm': float(gradient.norm())}
    direction /= response_rms
    offset = (x @ direction.T).float()
    per_token_derivative = (v * offset.double()).sum(-1)
    train_gradient_after_projection = float((gradient * direction).sum())
    assert train_gradient_after_projection < 0.
    mean_response = float((direction @ mu).norm())
    if mean_free:
        assert mean_response < 1e-7 * max(1., float(direction.norm() * mu.norm()))
    held_gradient = v[512:].T @ x[512:] / 512
    cosine = float((projected * held_gradient).sum() /
                   (projected.norm() * held_gradient.norm()).clamp_min(1e-30))
    record = dict(nonzero_measurement_direction=True, gradient_norm=float(gradient.norm()),
        projected_gradient_norm=raw_norm, train_feature_std_rms=float(x[:512].std(0, unbiased=False).square().mean().sqrt()),
        mean_free=mean_free, mean_base_response_norm=mean_response,
        train_base_response_rms=float(offset[:512].square().mean().sqrt()),
        train_directional_derivative=float(per_token_derivative[:512].mean()),
        heldout_directional_derivative=float(per_token_derivative[512:].mean()),
        heldout_block_derivatives=[float(per_token_derivative[a:a + 256].mean()) for a in (512, 768)],
        train_projected_vs_heldout_raw_gradient_cosine=cosine, finite_differences=[])
    # This is a legitimate weight-space ray for motion and field. A shuffled
    # feature ray is only the offline negative control.
    base_rms = float(base[:512].square().mean().sqrt())
    for fraction in (2**-8, 2**-9):
        epsilon = fraction * base_rms
        positive = losses_with_offset(model, base[512:], offset[512:], targets[512:], epsilon)
        negative = losses_with_offset(model, base[512:], offset[512:], targets[512:], -epsilon)
        central = float((positive.double() - negative.double()).mean() / (2 * epsilon))
        derivative = record['heldout_directional_derivative']
        record['finite_differences'].append(dict(base_rms_fraction=fraction, epsilon=epsilon,
            central_derivative=central, absolute_error=abs(central - derivative),
            relative_error=abs(central - derivative) / max(abs(derivative), 1e-12)))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    constructor = dict(saved['config']['constructor'])
    constructor.update(medium_execution='native', port_execution='native', read_mode='dynamic')
    model = PlasticMediumPorts3D(**constructor).eval()
    initialize_dynamic_read_branch(model, saved['model'])
    initial_hash = digest(model)
    belief = unpack_belief(saved['belief'], 'cpu')
    initial_time = float(belief.medium.elapsed[0])
    data_path = Path('data/ib_owt_gpt2/train.npy')
    data_hash = file_sha256(data_path)
    assert data_hash == saved['config']['data_sha256']['train.npy'], 'Checkpoint corpus mismatch'
    corpus = np.load(data_path, mmap_mode='r')
    targets = torch.tensor(np.array(corpus[saved['cursor']:saved['cursor'] + 1024], dtype=np.int64))
    assert len(targets) == 1024
    ids = torch.cat((targets.new_tensor([saved['learner']['carry_token']]), targets[:-1]))
    base, field, motion, final_belief = collect(model, belief, ids, saved['config']['event_duration'])
    losses, partials = loss_and_partials(model, base, targets)
    generator = torch.Generator().manual_seed(712)
    shuffled = torch.cat((motion[:512][torch.randperm(512, generator=generator)],
                          motion[512:][torch.randperm(512, generator=generator)]))
    records = {}
    for name, features in (('motion', motion), ('same_size_field', field),
                           ('partition_shuffled_motion', shuffled)):
        records[name] = {}
        for mean_free in (False, True):
            print(f'Checking production tangent: {name}, mean-free {mean_free}', flush=True)
            record = summarize_direction(model, base, features, partials, targets, mean_free)
            record['production_parameter_direction'] = name != 'partition_shuffled_motion'
            record['direction_kind'] = ('offline shuffled-feature control' if
                name == 'partition_shuffled_motion' else 'actual production weight subspace')
            records[name]['mean_free' if mean_free else 'raw'] = record
    report = dict(purpose='Local production-subspace task reachability, not capability training',
        checkpoint=str(args.checkpoint.resolve()), corpus_offset=saved['cursor'],
        checkpoint_sha256=file_sha256(args.checkpoint), actual_corpus_sha256=data_hash,
        carry_token=int(saved['learner']['carry_token']),
        carry_matches_train_predecessor=(bool(saved['learner']['carry_token'] ==
            int(corpus[saved['cursor'] - 1])) if saved['cursor'] > 0 else None),
        carry_source='saved live learner; a scored phase bridge may differ from corpus predecessor',
        actual_forward_events=1024, direction_targets=512, heldout_targets=512,
        optimizer_updates=0, parameters_unchanged=initial_hash == digest(model),
        physical_time_before=initial_time, physical_time_after=float(final_belief.medium.elapsed[0]),
        baseline_nll=dict(direction=float(losses[:512].mean()), heldout=float(losses[512:].mean())),
        insertion_point='base += W @ motion before correction, RMSNorm and decoder',
        arms=records, elapsed_seconds=time.perf_counter() - started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preregistration='docs/information_boltzmann/MEDIUM_DYNAMIC_REACHABILITY_20261008.md')
    assert report['parameters_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
