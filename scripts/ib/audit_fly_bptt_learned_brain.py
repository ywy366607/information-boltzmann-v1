"""Attribute accumulated brain changes on a continuing checkpoint fork.

This fixed-weight intervention is a mechanism diagnostic, not the primary
active-learning evaluation or a matched frozen-brain training control.
Every arm starts from the same saved physical life; no zero/reset state.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState, advance_fly_input_event
from scripts.ib.inspect_fly_bptt_credit import move_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/q8_fly_bptt32_continuous_100k'))
    parser.add_argument('--burn-in', type=int, default=256)
    parser.add_argument('--score-tokens', type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.85)
    saved = torch.load(args.run / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    settle_ticks = saved['learner'].get('settle_ticks', 0)
    writer_clock = saved['learner'].get('writer_baseline_clock', 'physical')
    origin = torch.load(saved['origin']['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    cfg = saved['config']
    model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257,
        d_model=cfg['d_model'], injection='topographic', read_surface='output',
        synapse_model='coba', use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'])
    current = saved['model']
    loaded = model.load_state_dict({k: v for k, v in current.items()
        if k not in ('edge_weight_e', 'edge_weight_i')}, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError(str(loaded))
    model = model.cuda().requires_grad_(False)
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    internal = [k for k in current if k.startswith('log_') or k == 'logit_u0']
    edges = ['edge_weight_e', 'edge_weight_i']
    initial = FlyPhysicalState(**saved['learner']['physical'])
    val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
    cursor = saved['val_cursor']
    count = args.burn_in + args.score_tokens
    targets = np.array(val[cursor:cursor+count], dtype=np.int64)
    if len(targets) != count:
        raise ValueError('Fresh validation stream exhausted')
    inputs = np.concatenate(([saved['learner']['previous_token']], targets[:-1]))
    report = {'scope': 'fixed-weight paired mechanism intervention; not active evaluation or frozen-brain training comparison',
        'checkpoint': str(args.run / 'last.pt'), 'origin': saved['origin'],
        'train_tokens_added': saved['bptt_train_tokens'], 'updates': saved['learner']['updates'],
        'validation_cursor': cursor, 'burn_in_events': args.burn_in,
        'score_events': args.score_tokens, 'state_policy': 'same saved full physical state per arm; no reset; real unscored burn-in',
        'rollback_physical_names': internal, 'arms': {}}
    reference_scores = None
    reference_features = None
    with torch.no_grad():
        for arm, rollback in [('learned', []), ('rollback_edges', edges),
                              ('rollback_physiology', internal), ('rollback_both', edges + internal)]:
            for name in edges + internal:
                tensors[name].copy_((origin['model'] if name in rollback else current)[name])
            state = move_state(initial, 'cuda')
            features = []
            start = time.perf_counter()
            for index, token in enumerate(inputs):
                state = advance_fly_input_event(model, state,
                    torch.tensor([int(token)], device='cuda'), settle_ticks=settle_ticks,
                    writer_baseline_clock=writer_clock)
                if index >= args.burn_in:
                    features.append(model.output_read(state.h[:, model.read_indices]))
                if (index + 1) % 256 == 0:
                    print(f'{arm}: {index+1}/{count} events', flush=True)
            feature = torch.cat(features)
            # Bound temporary vocabulary logits rather than allocate all tokens.
            scores = torch.cat([F.cross_entropy(model.decoder(model.read_norm(feature[i:i+32])),
                torch.as_tensor(targets[args.burn_in+i:args.burn_in+i+32], device='cuda'), reduction='none')
                for i in range(0, args.score_tokens, 32)]).cpu().numpy()
            feature = feature.cpu().numpy()
            if reference_scores is None:
                reference_scores, reference_features = scores.copy(), feature.copy()
            delta = scores - reference_scores
            report['arms'][arm] = {'nll': float(scores.mean()),
                'nll_increase_on_rollback': float(delta.mean()),
                'paired_128_event_blocks': [float(delta[i:i+128].mean()) for i in range(0,len(delta),128)],
                'read_feature_relative_l2_difference': float(np.linalg.norm(feature-reference_features)/max(np.linalg.norm(reference_features),1e-30)),
                'seconds': time.perf_counter()-start,
                'scores': scores.tolist()}
            print(arm, {k: v for k, v in report['arms'][arm].items() if k != 'scores'}, flush=True)
            (args.run / 'learned_brain_intervention.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    (args.run / 'learned_brain_intervention.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
