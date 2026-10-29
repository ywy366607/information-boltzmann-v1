"""Fixed held-out ablations on ResourceCycle checkpoints; no training or tuning."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import run_material_optimizer_owt as runner  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def score(model, batches):
    total_nll = 0.0
    total_bytes = 0
    for x, y in batches:
        losses = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction='none')
        mask = y.flatten() >= 4
        total_nll += float(losses[mask].sum())
        total_bytes += int(mask.sum())
    return total_nll / total_bytes / math.log(2)


def paired_next_step(record, saved, batch):
    """Same checkpoint and batch, only final resource allocation is toggled."""
    directions = {}
    for label, strength in (('on', 1.0), ('off', 0.0)):
        model = runner.make_model(record)
        model.load_state_dict(saved['model'])
        opt = runner.make_optimizer(model, record, 'latch_control')
        opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        opt.param_groups[0]['competition_strength'] = strength
        before = [p.detach().clone() for p in model.parameters()]
        x, y = batch
        opt.zero_grad(set_to_none=False)
        F.cross_entropy(model(x).flatten(0, 1), y.flatten()).backward()
        opt.step()
        directions[label] = torch.cat([(p.detach() - old).flatten() for p, old in zip(model.parameters(), before)])
    on, off = directions['on'], directions['off']
    return {'on_l2':float(on.norm()), 'off_l2':float(off.norm()),
            'cosine':float(F.cosine_similarity(on, off, dim=0)),
            'difference_l2':float((on - off).norm())}


def paired_fatigue_steps(record, saved, batches):
    """Same checkpoint and two batches; only resource depletion is toggled."""
    trajectories = {}
    for label, fatigue in (('on', True), ('off', False)):
        model = runner.make_model(record)
        model.load_state_dict(saved['model'])
        opt = runner.make_optimizer(model, record, 'fatigue_on')
        opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        opt.param_groups[0]['resource_fatigue'] = fatigue
        vectors, means = [], []
        for x, y in batches:
            before = [p.detach().clone() for p in model.parameters()]
            opt.zero_grad(set_to_none=False)
            F.cross_entropy(model(x).flatten(0, 1), y.flatten()).backward()
            opt.step()
            vectors.append(torch.cat([(p.detach() - old).flatten() for p, old in zip(model.parameters(), before)]))
            resources = [s[k] for s in opt.state.values() for k in ('row_reserve', 'col_reserve') if k in s]
            means.append(float(sum(x.sum() for x in resources) / sum(x.numel() for x in resources)))
        trajectories[label] = {'updates': vectors, 'mean_reserve': means}
    steps = []
    for t in range(len(batches)):
        on, off = trajectories['on']['updates'][t], trajectories['off']['updates'][t]
        steps.append({'step':t + 1,'on_l2':float(on.norm()),'off_l2':float(off.norm()),
            'cosine':float(F.cosine_similarity(on,off,dim=0)), 'difference_l2':float((on-off).norm()),
            'on_mean_reserve':trajectories['on']['mean_reserve'][t],
            'off_mean_reserve':trajectories['off']['mean_reserve'][t]})
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--prereg', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text(encoding='utf-8'))
    record = runner.load_registration(args.registration)
    runner.torch_setup(record['experiment']['seed'])
    if record['model']['ff_width'] != 128 or record['model']['layers'] != 2:
        raise ValueError('registered group layout assumes two 128-wide FF layers')
    batches = {}
    for domain in ('A', 'B'):
        stream = np.load(ROOT / record['data']['streams']['validation_' + domain]['path'], mmap_mode='r')
        batches[domain] = [runner.batch_from(stream, cursor, 8, 128) for cursor in range(0, 4096, 1024)]

    output = {'scope': prereg['scope'], 'registration_sha256': digest(args.registration),
              'probe_prereg_sha256': digest(args.prereg), 'targets_per_domain': 4096,
              'arms': {}, 'limitations': prereg['limitations']}
    for arm in prereg['checkpoints']:
        path = args.checkpoint_root / arm / 'last.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['registration_sha256'] != digest(args.registration) or saved['progress']['update'] != 5000:
            raise ValueError('checkpoint is not the registered completed run: ' + arm)
        model = runner.make_model(record)
        model.load_state_dict(saved['model'])
        model.eval()
        baseline = {d: score(model, batches[d]) for d in ('A', 'B')}
        groups = []
        for layer in range(2):
            for start in range(0, 128, 8):
                def ablate(_module, _args, result, start=start):
                    masked = result.clone()
                    masked[..., start:start + 8] = 0
                    return masked
                hook = model.blocks[layer].ff1.register_forward_hook(ablate)
                try:
                    changed = {d: score(model, batches[d]) for d in ('A', 'B')}
                finally:
                    hook.remove()
                groups.append({'layer': layer, 'start': start, 'end_exclusive': start + 8,
                               'delta_A_bpb': changed['A'] - baseline['A'],
                               'delta_B_bpb': changed['B'] - baseline['B']})
        repeat = {d: score(model, batches[d]) for d in ('A', 'B')}
        if baseline != repeat:
            raise AssertionError('ablation changed the model or hook was not removed')
        summaries = {}
        for domain in ('A', 'B'):
            effects = np.array([abs(g['delta_' + domain + '_bpb']) for g in groups])
            total = float(effects.sum())
            shares = effects / total if total else effects
            summaries[domain] = {'mean_abs_delta_bpb':float(effects.mean()),
                'max_absolute_effect_share':float(shares.max()),
                'effective_groups':float(1 / np.square(shares).sum()) if total else None}
        output['arms'][arm] = {'checkpoint_sha256':digest(path), 'baseline_bpb':baseline,
                               'groups':groups, 'summary':summaries}
        if arm == 'latch_control':
            output['same_checkpoint_next_step'] = {
                'status': 'exploratory; added after the allocation-off trajectory was observed',
                'domains': {d: paired_next_step(record, saved, batches[d][0]) for d in ('A', 'B')}}
        if arm == 'fatigue_on' and prereg.get('fatigue_two_step_probe'):
            output['same_checkpoint_fatigue_two_step'] = {
                d: paired_fatigue_steps(record, saved, batches[d][:2]) for d in ('A', 'B')}
        del model, saved
        torch.cuda.empty_cache()
        print(json.dumps({arm:summaries}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
