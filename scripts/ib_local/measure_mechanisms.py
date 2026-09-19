"""Frozen checkpoint mechanism interventions on held-out OWT, not retraining ablations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.window import LocalWindow
from scripts.ib_local.device_collision import collision_device
from scripts.ib_local.sampling import sample_window
from scripts.ib_bpe_window import clock_inputs
from scripts.ib_fused_ou import FusedOU


class InterventionWindow(LocalWindow):
    arm = 'keep'
    fixed_gamma = .3

    def kick(self, x, v, shared, noise, h, gamma=None):
        if self.arm not in ('no_drive', 'fixed_gamma'):
            return super().kick(x, v, shared, noise, h, gamma=gamma)
        force = self.core.force
        if shared.shape[-1] == 4:
            drive = shared
        else:
            drive = .5 * force.net[2](force.net[1](F.linear(x, force.net[0].weight[:, :4]) + shared)).tanh()
        if self.arm == 'no_drive':
            drive = torch.zeros_like(drive)
        if self.arm == 'fixed_gamma':
            gamma_eff = torch.full_like(x[:, :1], self.fixed_gamma)
        else:
            gamma_eff = gamma if gamma is not None else force.damping(x)
        return FusedOU.apply(v, -x + drive, gamma_eff, noise, h, .1)

    def evolve(self, x, v, shared, clocks, noise, layers, gamma=None):
        if self.arm != 'no_transport':
            return super().evolve(x, v, shared, clocks, noise, layers, gamma=gamma)
        dt = 1. / self.steps
        for half in range(2):
            drive_input = shared + clocks[half]
            v = self.kick(x, v, drive_input, noise[half * 2], dt / 4)
            # Intervention: suppress dx/dt=v, retain every velocity kick.
            v = self.kick(x, v, drive_input, noise[half * 2 + 1], dt / 4)
            if half == 0:
                v, lp, accepted = collision_device(x, v, self.core.collision, layers)
        return x, v, lp, accepted


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sites', type=int, default=4)
    p.add_argument('--history', type=int, default=256)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--arms', nargs='+', default=None,
                   help='Restrict interventions, e.g. keep no_collision.')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    coupling = saved['config'].get('coupling_mode', 'none')
    score_scale = saved['config'].get('score_scale', 1.0)
    model = InterventionWindow(hidden=saved['config']['hidden'], particles=saved['config']['particles'], coupling_mode=coupling, score_scale=score_scale).cuda().eval()
    model.load_state_dict(saved['model'])
    model.requires_grad_(False)
    model.recompute = False
    original = LocalWindow(hidden=saved['config']['hidden'], particles=saved['config']['particles'], coupling_mode=coupling, score_scale=score_scale).cuda().eval()
    original.load_state_dict(saved['model'])
    original.requires_grad_(False)
    data = np.load(a.data / 'validation.npy', mmap_mode='r')
    arms = ['keep', 'no_collision', 'reset_each_token', 'no_drive', 'no_transport', 'fixed_gamma', 'zero_birth', 'reset_no_collision']
    if a.arms is not None:
        if 'keep' not in a.arms or not set(a.arms).issubset(arms):
            p.error('--arms must contain keep and valid intervention names')
        arms = a.arms
    if coupling != 'none':
        unsupported = {'no_drive', 'no_transport', 'fixed_gamma'}
        if a.arms is not None and unsupported.intersection(arms):
            p.error('These interventions require coupling-aware implementations: '
                    + ', '.join(sorted(unsupported.intersection(arms))))
        arms = [arm for arm in arms if arm not in unsupported]
    report = dict(checkpoint_step=saved['step'], trained_tokens=saved['events'],
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        config=saved['config'], history=a.history, horizon=a.horizon,
        protocol='Frozen weights; common text/noise/proposals. Zero birth starts at history origin; other interventions start after common history. Fixed gamma matched to branchpoint spatial mean. No-drive removes full learned bounded drive, not harmonic trap. Not a Transformer training comparison.',
        sites=[], evaluation_route='coupling-aware eager forward', arms=arms)
    gen = torch.Generator(device='cuda')
    n = saved['config']['particles']

    def step(state, previous, target, clock, tables, noise, arm):
        model.arm = arm
        if arm in ('reset_each_token', 'reset_no_collision'):
            state = tuple(torch.zeros_like(x) for x in state)
        if arm in ('no_collision', 'reset_no_collision'):
            tables = [[] for _ in tables]
        args = (*state, torch.tensor([previous], device='cuda'), torch.tensor([target], device='cuda'),
                clock_inputs(clock, 1, model.steps, 'cuda'), noise, tables)
        out = model(*args)
        return (out[2], out[3]), float(out[1]), args, out

    for site in range(a.sites):
        start = 8192 + site * 4096
        assert start + a.history + a.horizon < len(data)
        gen.manual_seed(17100 + site)
        rng = np.random.default_rng(18100 + site)
        birth = model.core.initialize(torch.tensor([50256], device='cuda'), gen)
        states = {'keep': (birth.x, birth.v), 'zero_birth': (torch.zeros_like(birth.x), torch.zeros_like(birth.v))}
        for t in range(a.history):
            tables, _ = sample_window(rng, 1, model.steps, n, 'cuda')
            noise = torch.randn(model.steps*4, n, 4, device='cuda', generator=gen)
            for arm in list(states):
                states[arm], _, args, out = step(states[arm], 50256 if t == 0 else int(data[start+t-1]), int(data[start+t]), t, tables, noise, 'keep')
                if site == 0 and t == 0 and arm == 'keep':
                    reference = original(*args)
                    for i in [1, 2, 3]:
                        torch.testing.assert_close(out[i], reference[i], atol=0, rtol=0)
        model.fixed_gamma = float(model.core.force.damping(states['keep'][0]).mean())
        for arm in arms:
            if arm not in states:
                states[arm] = tuple(x.clone() for x in states['keep'])
        losses = {arm: [] for arm in arms}
        for t in range(a.horizon):
            tables, _ = sample_window(rng, 1, model.steps, n, 'cuda')
            noise = torch.randn(model.steps*4, n, 4, device='cuda', generator=gen)
            cursor = start + a.history + t
            for arm in arms:
                states[arm], loss, _, _ = step(states[arm], int(data[cursor-1]), int(data[cursor]), a.history+t, tables, noise, arm)
                losses[arm].append(loss)
        assert all(np.isfinite(v).all() for v in losses.values())
        report['sites'].append(dict(site=site, start=start, fixed_gamma=model.fixed_gamma, losses=losses))
        report['summary'] = {arm: dict(nll=float(np.mean([s['losses'][arm] for s in report['sites']])),
            per_site_delta=[float(np.mean(s['losses'][arm])-np.mean(s['losses']['keep'])) for s in report['sites']]) for arm in arms}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({'site': site, 'summary': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
