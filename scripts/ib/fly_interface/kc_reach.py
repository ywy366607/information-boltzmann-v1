"""Can an odour reach the Kenyon cells?  Fixed weights, text at mean 8 ticks/word (pulses); env NO_CHEMO_WRITER=1
takes words out of the chemosensory cells.  Per checkpoint:
  Kenyon rates over 4,000 ticks: fraction silent (< 0.1%), fraction tonic (> 50%), median;
  the silent ones' distance below threshold, (theta - h) / theta averaged over ticks, and its 90th percentile of the
  per-cell maximum h / theta;
  an odour (6 glomeruli, receptor cells at 16 rheobase for 8 ticks, 8 trials): the fraction of Kenyon cells firing at
  least once in 12 ticks, minus a matched control window, and the rise of the per-cell peak h / theta of the Kenyon
  cells receiving >= 3 synapses-weighted inputs from the matched PNs.
Usage: python kc_reach.py <checkpoint dir> [out .jsonl]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core import fly_odour_world as ow

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/published/fly_consequence_20261012/kc_reach.jsonl')
model, cfg, L = load(fork)
import os
if os.environ.get('NO_CHEMO_WRITER'):
    # words are seen and heard, not smelled or tasted: the writer no longer drives the chemosensory cells
    with torch.no_grad():
        model.topographic_writer.proj_chemo.weight.zero_()
        if getattr(model.topographic_writer.proj_chemo, 'bias', None) is not None:
            model.topographic_writer.proj_chemo.bias.zero_()
graph = ROOT / cfg['graph']; ann_path = graph.parent / 'body-annotations.feather'
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
sup = np.array(list(g['superclass_names']))[g['superclass_id']]
N = model.n_neurons
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
with torch.no_grad():
    rates0 = model.get_decay_rates()[0].expand(1, N)
    rheobase = (model.get_thresholds().expand(1, N) * -torch.log(rates0.clamp(1e-5, 1 - 1e-7)))[0].float()
writer = model.topographic_writer
quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[80_000:95_000], dtype=np.int64)
rng = np.random.default_rng(1)
st = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0, 'left': 0}


def tick(drive=None):
    s = st['s']; model.external_drive = drive
    if st['left'] == 0:
        st['left'] = int(rng.geometric(1 / 8))
        tok = torch.tensor([int(text[st['i'] % len(text)])], device='cuda'); st['i'] += 1
        src, base = fl.write_input(model, s, tok)
        s = fl.step_fly_physical_tick(model, s, tok, src, base, opts, base_rates=opts['base_rates'])
    else:
        base = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, base, opts, base_rates=opts['base_rates'])
    st['left'] -= 1; model.external_drive = None; st['s'] = s
    return s, (s.ring[0][0] > 0).float()


KC_RENORM = bool(os.environ.get('KC_RENORM'))
if KC_RENORM:
    # the birth normalization divided every input of a Kenyon cell by sqrt(K_in) counting its axo-axonal Kenyon
    # partners (median 162), which are not dendritic input and are pinned at 0; with the real dendritic partners
    # (median 13) each remaining input onto a Kenyon cell is sqrt(K_all / K_dendritic) larger (median 3.6)
    kc_mask = cls == 'Kenyon_Cell'
    k_all = np.bincount(g['edge_post'], minlength=N)
    k_den = np.bincount(g['edge_post'][~kc_mask[g['edge_pre']]], minlength=N)
    factor = torch.as_tensor(np.where(kc_mask, np.sqrt(k_all / np.maximum(k_den, 1)), 1.0), dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for sfx in ('e', 'i'):
            w = getattr(model, f'edge_weight_{sfx}'); pre_ = getattr(model, f'edge_pre_{sfx}').long(); post_ = getattr(model, f'edge_post_{sfx}').long()
            keep = ~torch.as_tensor(kc_mask, device='cuda')[pre_]
            w.mul_(torch.where(keep, factor[post_], torch.ones_like(w)))
rep = {'checkpoint': str(fork), 'no_chemo_writer': bool(os.environ.get('NO_CHEMO_WRITER')), 'kc_renorm': KC_RENORM}
alpn = cls == 'ALPN'; kc_idx = torch.as_tensor(np.flatnonzero(cls == 'Kenyon_Cell'), device='cuda')
theta = model.get_thresholds().expand(1, N)[0]
pre, post = g['edge_pre'], g['edge_post']
with torch.no_grad():
    for _ in range(500):
        tick()
    acc = torch.zeros(N, device='cuda'); dist = torch.zeros(N, device='cuda'); peak = torch.full((N,), -1e9, device='cuda')
    T = 4000
    for _ in range(T):
        s, sp = tick(); acc += sp
        dist += (theta - s.h[0]) / theta; peak = torch.maximum(peak, s.h[0] / theta)
    r = (acc / T)[kc_idx]; silent = r < 0.001
    rep['kc_fraction_silent'] = float(silent.float().mean()); rep['kc_fraction_tonic'] = float((r > 0.5).float().mean())
    rep['kc_rate_median'] = float(r.median()); rep['kc_rate_mean'] = float(r.mean())
    rep['silent_distance_mean_theta'] = float((dist / T)[kc_idx][silent].mean())
    rep['silent_peak_h_over_theta_p90'] = float(torch.quantile(peak[kc_idx][silent], 0.9))
    gains, rises = [], []
    for trial in range(8):
        gl = list(rng.choice(ow.POOL, 6, replace=False))
        orn = np.concatenate([np.flatnonzero(typ == f'ORN_{x}') for x in gl])
        pn = np.flatnonzero(alpn & np.array([any(t.startswith(x + '_') for x in gl) for t in typ]))
        m = np.isin(pre, pn) & (cls[post] == 'Kenyon_Cell')
        target = np.flatnonzero(np.bincount(post[m], minlength=N) >= 3)
        tgt = torch.as_tensor(target, device='cuda')
        drive = torch.zeros(1, N, device='cuda'); idx = torch.as_tensor(orn, device='cuda'); drive[0, idx] = 16.0 * rheobase[idx]
        res = {}
        for phase in ('control', 'odour'):
            for _ in range(60):
                tick()
            union = torch.zeros(N, device='cuda'); pk = torch.full((N,), -1e9, device='cuda')
            for t in range(12):
                s, sp = tick(drive if (phase == 'odour' and t < 8) else None)
                union = torch.maximum(union, sp); pk = torch.maximum(pk, s.h[0] / theta)
            res[phase] = (float(union[kc_idx].mean()), float(pk[tgt].mean()) if len(target) else float('nan'))
        gains.append(res['odour'][0] - res['control'][0]); rises.append(res['odour'][1] - res['control'][1])
    rep['odour_kc_extra_fraction'] = float(np.mean(gains)); rep['odour_target_peak_rise_theta'] = float(np.nanmean(rises))
print(json.dumps(rep), flush=True)
open(out, 'a').write(json.dumps(rep) + chr(10))
