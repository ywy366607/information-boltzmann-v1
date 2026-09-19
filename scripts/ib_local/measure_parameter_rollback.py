"""Groupwise parameter rollback, with tied weights handled as one group."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.window import LocalWindow
from scripts.ib_local.sampling import sample_window
from scripts.ib_bpe_window import clock_inputs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sites', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    initial = torch.load(a.initial, map_location='cpu', weights_only=False)
    final = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    assert initial['step'] == 0 and final['step'] == 3000
    before, after = initial['model'], final['model']
    assert before.keys() == after.keys()
    tied = {'core.initial.embedding.weight', 'core.force.embedding.weight', 'core.decoder.weight'}
    groups = {
        'keep': [],
        'shared_embedding_output': sorted(tied),
        'drive_mlp': [k for k in after if k.startswith('core.force.net.')],
        'feature_mlp_readout_bias': [k for k in after if k.startswith('core.features.') or k == 'core.decoder.bias'],
        'collision': [k for k in after if k.startswith('core.collision.')],
        'gamma': [k for k in after if k.startswith('core.force.gamma_field.')],
        'birth_operator': [k for k in after if k.startswith('core.initial.') and k not in tied],
        'all_initial': list(after),
    }
    models = {}
    for arm, keys in groups.items():
        model = LocalWindow(hidden=final['config']['hidden'], particles=final['config']['particles']).cuda().eval()
        state = {k: before[k] if k in keys else v for k, v in after.items()}
        model.load_state_dict(state)
        model.requires_grad_(False)
        model.recompute = False
        assert model.core.decoder.weight is model.core.force.embedding.weight
        assert model.core.initial.embedding.weight is model.core.force.embedding.weight
        actual = model.state_dict()
        assert all(torch.equal(actual[k].cpu(), v) for k, v in state.items())
        models[arm] = model
    data = np.load(a.data / 'validation.npy', mmap_mode='r')
    report = dict(checkpoint_step=3000, trained_tokens=final['events'],
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        initial_sha256=hashlib.sha256(a.initial.read_bytes()).hexdigest(),
        groups=groups, history=256, horizon=128,
        protocol='Rollback before birth and warmup; same OWT positions, initial random draws, bath noise and collision proposals. No optimization. Shared input/output embedding rolled back jointly.',
        checks={'all_group_and_unchanged_weights_verified': True, 'tied_alias_identity_verified': True},
        limitations=['4 text locations, 512 scored tokens per arm', 'Rollback measures dependence on learned parameters under module coadaptation, not additive attribution of training loss reduction', 'Shared embedding rollback changes input, birth conditioning and output vocabulary simultaneously'], sites=[])
    n = final['config']['particles']
    gen = torch.Generator(device='cuda')
    for site in range(a.sites):
        start = 8192 + site * 4096
        assert start + 384 < len(data)
        states = {}
        for arm, model in models.items():
            gen.manual_seed(17100 + site)
            born = model.core.initialize(torch.tensor([50256], device='cuda'), gen)
            states[arm] = (born.x, born.v)
        rng = np.random.default_rng(18100 + site)
        losses = {arm: [] for arm in models}
        for t in range(384):
            tables, _ = sample_window(rng, 1, 4, n, 'cuda')
            noise = torch.randn(16, n, 4, device='cuda', generator=gen)
            ids = torch.tensor([50256 if t == 0 else int(data[start+t-1])], device='cuda')
            target = torch.tensor([int(data[start+t])], device='cuda')
            clocks = clock_inputs(t, 1, 4, 'cuda')
            for arm, model in models.items():
                out = model(*states[arm], ids, target, clocks, noise, tables)
                states[arm] = (out[2], out[3])
                if t >= 256:
                    losses[arm].append(float(out[1]))
            if t == 255:
                print(json.dumps({'site': site, 'warmup_complete': True}), flush=True)
        assert all(np.isfinite(v).all() for v in losses.values())
        report['sites'].append(dict(site=site, start=start, losses=losses))
        report['summary'] = {}
        for arm in models:
            delta = np.array([s['losses'][arm] for s in report['sites']])-np.array([s['losses']['keep'] for s in report['sites']])
            report['summary'][arm] = dict(nll=float(np.mean([s['losses'][arm] for s in report['sites']])),
                delta_nll=float(delta.mean()), per_site_delta=delta.mean(1).tolist())
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({'site':site, 'summary':report['summary']}), flush=True)


if __name__ == '__main__':
    main()
