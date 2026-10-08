"""No-update continuous-cadence intervention; no autonomous clock claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.nn import functional as F

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from scripts.ib.audit_medium_dynamic_read import digest
from scripts.ib.audit_medium_dynamic_reachability import file_sha256
from scripts.ib.train_plastic_conductance import unpack_belief


@torch.no_grad()
def trajectory(model, initial, ids, targets, duration, substeps, prepared, table):
    """Each counterfactual retains its own full state across the512 split."""
    belief = initial
    initial_hash = digest(model)
    losses = []
    started = time.perf_counter()
    for event, (observed, target) in enumerate(zip(ids, targets)):
        belief, _ = model.assimilate(belief, observed.reshape(1),
            token_features=table, diagnostics=False, training_terms=False)
        belief, _ = model.advance(belief, duration, substeps=substeps,
            prepared=prepared, diagnostics=False)
        predicted, _ = model.read(belief, prepared=prepared)
        losses.append(float(F.cross_entropy(predicted, target.reshape(1))))
        if (event + 1) % 256 == 0:
            print(f'Continuous cadence arm {duration:.8g}, resolution{substeps}: '
                  f'{event + 1}/{len(ids)} events', flush=True)
    values = np.asarray(losses, dtype=np.float64)
    record = dict(duration=duration, solver_substeps=substeps,
        actual_forward_events=len(ids), solver_steps=len(ids) * substeps,
        elapsed_before=float(initial.medium.elapsed[0]),
        elapsed_after=float(belief.medium.elapsed[0]),
        prefix_nll=float(values[:512].mean()), subsequent_nll=float(values[512:].mean()),
        subsequent_block_nll=[float(values[a:a + 256].mean()) for a in (512, 768)],
        wall_seconds=time.perf_counter() - started,
        parameter_sha256_before=initial_hash, parameter_sha256_after=digest(model))
    assert record['parameter_sha256_before'] == record['parameter_sha256_after']
    return record, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    constructor = dict(saved['config']['constructor'])
    constructor.update(medium_execution='native', port_execution='native')
    model = PlasticMediumPorts3D(**constructor).eval()
    model.load_state_dict(saved['model'])
    initial_hash = digest(model)
    initial = unpack_belief(saved['belief'], 'cpu')
    data_path = Path('data/ib_owt_gpt2/train.npy')
    data_hash = file_sha256(data_path)
    assert data_hash == saved['config']['data_sha256']['train.npy'], 'Checkpoint corpus mismatch'
    corpus = np.load(data_path, mmap_mode='r')
    targets = torch.tensor(np.array(corpus[saved['cursor']:saved['cursor'] + 1024], dtype=np.int64))
    assert len(targets) == 1024
    ids = torch.cat((targets.new_tensor([saved['learner']['carry_token']]), targets[:-1]))
    duration = saved['config']['event_duration']
    with torch.no_grad():
        table = F.normalize(model.source.embedding.weight, dim=-1)
        prepared = model.medium.prepare_evolution()
    baseline, reference = trajectory(model, initial, ids, targets, duration, 1, prepared, table)
    refined, refined_values = trajectory(model, initial, ids, targets, duration, 2, prepared, table)
    resolutions = []
    for epsilon in (2**-8, 2**-9):
        positive, plus = trajectory(model, initial, ids, targets,
            duration * math.exp(epsilon), 1, prepared, table)
        negative, minus = trajectory(model, initial, ids, targets,
            duration * math.exp(-epsilon), 1, prepared, table)
        prefix_slope = float((plus[:512] - minus[:512]).mean() / (2 * epsilon))
        sign = -int(np.sign(prefix_slope))
        selected = plus if sign == 1 else minus if sign == -1 else reference
        resolutions.append(dict(epsilon=epsilon, positive=positive, negative=negative,
            prefix_central_derivative=prefix_slope, selected_sign_from_prefix=sign,
            subsequent_central_derivative=float((plus[512:] - minus[512:]).mean() / (2 * epsilon)),
            selected_prefix_nll_change=float((selected[:512] - reference[:512]).mean()),
            selected_subsequent_nll_change=float((selected[512:] - reference[512:]).mean()),
            selected_subsequent_block_changes=[float((selected[a:a + 256] -
                reference[a:a + 256]).mean()) for a in (512, 768)]))
    report = dict(purpose='Local continuous cadence utility, not autonomous clock or capability validation',
        checkpoint=str(args.checkpoint.resolve()), corpus_offset=saved['cursor'],
        corpus_sha256=data_hash, checkpoint_sha256=file_sha256(args.checkpoint),
        carry_token=int(saved['learner']['carry_token']),
        carry_matches_train_predecessor=(bool(saved['learner']['carry_token'] ==
            int(corpus[saved['cursor'] - 1])) if saved['cursor'] > 0 else None),
        carry_source='saved live learner; a scored phase bridge may differ from corpus predecessor',
        optimizer_updates=0, parameters_unchanged=digest(model) == initial_hash,
        baseline=baseline, fixed_duration_refinement=refined,
        refinement_subsequent_nll_change=float((refined_values[512:] - reference[512:]).mean()),
        interventions=resolutions, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preregistration='docs/information_boltzmann/MEDIUM_CONTINUOUS_TIME_VALUE_20261008.md')
    assert report['parameters_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
