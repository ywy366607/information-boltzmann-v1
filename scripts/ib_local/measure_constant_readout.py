"""Text-independent readout controls on the exact previously scored OWT targets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.window import LocalWindow
from scripts.ib_local.sampling import sample_window
from scripts.ib_bpe_window import clock_inputs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2)
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    ref = json.loads(a.reference.read_text())
    assert digest == ref['checkpoint_sha256']
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    coupling = saved['config'].get('coupling_mode', 'none')
    score_scale = saved['config'].get('score_scale', 1.0)
    model = LocalWindow(hidden=saved['config']['hidden'], particles=saved['config']['particles'], coupling_mode=coupling, score_scale=score_scale).cuda().eval()
    model.load_state_dict(saved['model'])
    model.requires_grad_(False)
    data = np.load(a.data/'validation.npy', mmap_mode='r')
    n = saved['config']['particles']
    gen = torch.Generator(device='cuda').manual_seed(29173)
    birth = model.core.initialize(torch.tensor([50256], device='cuda'), gen)
    constants = {'zero_state': model.core.features(torch.zeros(n, 8, device='cuda')).mean(0),
                 'fixed_birth': model.core.features(torch.cat((birth.x, birth.v), -1)).mean(0)}
    x, v = birth.x, birth.v
    rng = np.random.default_rng(39173)
    donor_start = 65536
    assert len(data) > donor_start + 256
    collected = []
    for t in range(256):
        tables, _ = sample_window(rng, 1, 4, n, 'cuda')
        noise = torch.randn(16, n, 4, device='cuda', generator=gen)
        ids = torch.tensor([50256 if t == 0 else int(data[donor_start+t-1])], device='cuda')
        target = torch.tensor([int(data[donor_start+t])], device='cuda')
        _, _, x, v, _ = model(x, v, ids, target, clock_inputs(t, 1, 4, 'cuda'), noise, tables)
        if t >= 128:
            collected.append(model.core.features(torch.cat((x,v), -1)).mean(0))
    constants['fixed_unrelated_state'] = collected[-1]
    constants['unrelated_mean_feature'] = torch.stack(collected).mean(0)
    constants['decoder_bias_only'] = torch.zeros_like(constants['zero_state'])
    logits = {k: F.linear(h, model.core.decoder.weight, model.core.decoder.bias) for k,h in constants.items()}
    report = dict(checkpoint_sha256=digest, checkpoint_step=saved['step'],
        reference=str(a.reference), donor_start=donor_start,
        protocol='All constant features chosen without scored target text; identical feature/logits at every scored token and every site. Normal losses reused from hash-matched prior measurement; exact same targets.',
        sites=[], limitations=['4 locations, 512 scored tokens','Constant controls are not an empirically fitted unigram baseline','Loss difference is not an additive attribution of training gain'])
    for s in ref['sites']:
        cursor = s['start'] + ref['history']
        targets = torch.tensor(np.array(data[cursor:cursor+ref['horizon']]),device='cuda',dtype=torch.long)
        losses = {'normal': s['losses']['keep']}
        for k,l in logits.items():
            losses[k] = F.cross_entropy(l.expand(len(targets),-1),targets,reduction='none').cpu().tolist()
        report['sites'].append(dict(site=s['site'],start=s['start'],losses=losses))
    report['summary'] = {}
    for k in report['sites'][0]['losses']:
        vals=np.array([s['losses'][k] for s in report['sites']])
        normal=np.array([s['losses']['normal'] for s in report['sites']])
        report['summary'][k] = dict(nll=float(vals.mean()),delta_nll=float((vals-normal).mean()),per_site_delta=(vals-normal).mean(1).tolist())
    report['checks'] = {'checkpoint_hash_matches_reference':True,'constants_independent_of_scored_text':True,'all_finite':all(np.isfinite(s['nll']) for s in report['summary'].values())}
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
