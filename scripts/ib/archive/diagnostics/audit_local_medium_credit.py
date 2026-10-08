"""Real-OWT two-event derivative audit; numerical evidence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.local_credit import LocalPlasticTrainer
from information_boltzmann.runtime.training import quiet_training_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(449)
    model = PlasticMediumPorts3D(bath_type='conductance', activity_adaptation=True,
                                short_term_plasticity=True).to(args.device).double()
    model.medium.short_term_plasticity.fuse_execution = False
    ids = torch.from_numpy(np.array(np.load(args.data_dir/'train.npy', mmap_mode='r')[:3],
                                    dtype=np.int64)).to(args.device)
    _, first, _ = quiet_training_chunk(model, ids[:1].view(1, 1), ids[1:2].view(1, 1),
                                       model.initial_belief(), event_duration=.005)
    loss, _, _ = quiet_training_chunk(model, ids[1:2].view(1, 1), ids[2:3].view(1, 1),
                                      first, event_duration=.005)
    names, parameters = zip(*model.named_parameters())
    exact = torch.autograd.grad(loss, parameters, allow_unused=True)
    direct_loss, _, _ = quiet_training_chunk(model, ids[1:2].view(1, 1), ids[2:3].view(1, 1),
                                              first.detach(), event_duration=.005)
    direct = torch.autograd.grad(direct_loss, parameters, allow_unused=True)
    learner = LocalPlasticTrainer(model, event_duration=.005)
    learner.backward_event(ids[:1], ids[1:2])
    model.zero_grad(set_to_none=True)
    result = learner.backward_event(ids[1:2], ids[2:3])
    # Only closing rows have delayed credit in this first local migration.
    # The full-graph historical derivative also contains voltage/space feedback;
    # measuring its omission prevents confusing conditional exactness with RTRL.
    name = 'medium.conductance_response.log_parameters.bias'
    index = names.index(name)
    exact_history = (exact[index] - direct[index]).reshape(14, model.medium.channels)[6:8]
    local_history = (parameters[index].grad - direct[index]).reshape(14, model.medium.channels)[6:8]
    eps = torch.finfo(exact_history.dtype).eps
    norm = exact_history.norm()
    report = {
        'scope': 'Numerical audit, real OWT events; no capability or convergence claim',
        'shape': list(model.medium.shape), 'channels': model.medium.channels,
        'dtype': 'float64', 'event_duration': .005, 'events': 2,
        'target_parameter_rows': name + '[closing_E, closing_I]',
        'full_history_norm': float(norm), 'local_history_norm': float(local_history.norm()),
        'relative_error_to_full_history': float((local_history - exact_history).norm() / norm.clamp_min(eps)),
        'cosine_to_full_history': float((local_history * exact_history).sum() /
                                      (local_history.norm() * norm).clamp_min(eps)),
        'eligibility_bytes': learner.eligibility_bytes(),
        'local_feedback_gradient_norm': float(result['history_gradient_norm']),
        'omitted_paths': ['past voltage/current feedback into opening rates',
                         'past cross-site/transport/collision/port sensitivity',
                         'past STP, structural conduction, precision sensitivity'],
        'status': 'Conditional receptor derivative verified separately; full-history approximation measured here'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
