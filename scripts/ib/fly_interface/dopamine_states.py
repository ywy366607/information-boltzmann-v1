"""Dopamine as a connectome neuromodulator: causality, surprise tracking, emotion-like states (fixed weights).

1. NLL change in self-paced reading / listening (k=4) / after 16-tick gaps when the sensitivities are zeroed (dopamine
   has no effect on excitability) and when the 392 dopaminergic cells are silenced.
2. Chaos episodes: L = 8 / 16 / 32 words drawn uniformly from the vocabulary inside normal text; per-word z-scores
   (vs. the pre-chaos baseline) of prediction error, written innovation, dopamine-cell spikes, mean phasic dopamine at
   the targets, thinking ticks; persistence after the chaos and grading with L.
(Surprise tracking during normal reading: dopamine_response.py.)
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
start = lambda: move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
dan = model.dan_indices
region, BURN, SCORE = 60_000, 256, 768
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))


def read(tokens, nexts, gap_every=0, gap=0, trace=False):
    torch.manual_seed(0)
    s, out, rows = start(), [], []
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            t = torch.tensor([int(tok)], device='cuda')
            innov = 0.0
            if trace:
                drive, _ = fl.write_input(model, s, t)
                innov = float(drive[0, model.topographic_writer.injection_index].norm())
            g = gap if (gap_every and i % gap_every == 0) else 0
            s, lat, _, _, _, after = fl.advance_fly_token_go(model, s, t, [], max_ticks=max_ticks,
                                                              writer_baseline_clock='physical', silence_after=g, **opts)
            z = after if g else lat[-1:]
            ce = float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), torch.tensor([int(nexts[i])], device='cuda')))
            out.append(ce)
            if trace:
                dop = s.dopamine
                targets_mask = dop[0] > 0
                phasic = float((dop[0] - dop[1])[targets_mask].mean()) if bool(targets_mask.any()) else 0.0
                rows.append((ce, innov, float((s.ring[0][0, dan] > 0).float().sum()), phasic, lat.shape[0]))
    return np.array(out), rows


def listen(k):
    s, out = start(), []
    with torch.no_grad():
        for i, tok in enumerate(inputs):
            s, z = fl.advance_fly_token_stream(model, s, torch.tensor([int(tok)], device='cuda'), ticks=k,
                                               writer_baseline_clock='physical', **opts)
            out.append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), torch.tensor([int(targets[i])], device='cuda'))))
    return np.array(out)


def worlds():
    r, _ = read(inputs, targets)
    gp, _ = read(inputs, targets, gap_every=4, gap=16)
    sl = slice(BURN, None)
    return {'read': float(r[sl].mean()), 'listen_k4': float(listen(4)[sl].mean()),
            'gap16_after': float(gp[sl][np.arange(SCORE) % 4 == 0].mean())}


report = {'fork': str(fork)}
report['worlds_full'] = worlds()
print('worlds:', {k: round(v, 4) for k, v in report['worlds_full'].items()}, flush=True)
sens = model.dopamine_sensitivity.detach().clone()
report['sensitivity'] = {'min': float(sens.min()), 'max': float(sens.max()), 'mean_abs': float(sens.abs().mean())}
model.dopamine_sensitivity.data.zero_()
report['worlds_no_dopamine_effect'] = worlds()
model.dopamine_sensitivity.data.copy_(sens)
th = model.get_thresholds().clone().expand(1, model.n_neurons).clone()
th[0, dan] = 1e6
opts['thresholds'] = th
report['worlds_dopamine_cells_silenced'] = worlds()
opts['thresholds'] = model.get_thresholds()
for k in ('worlds_no_dopamine_effect', 'worlds_dopamine_cells_silenced'):
    print(k, {w: round(report[k][w] - report['worlds_full'][w], 4) for w in report['worlds_full']}, flush=True)
print('sensitivities:', {k: round(v, 4) for k, v in report['sensitivity'].items()}, flush=True)

rng = np.random.default_rng(7)
report['episodes'] = {}
for Lc in (8, 16, 32):
    pre, post = 160, 96
    seq = np.array(val[region + 5000:region + 5000 + pre + Lc + post + 1], dtype=np.int64)
    seq[pre:pre + Lc] = rng.integers(0, 50257, Lc)
    _, rows = read(seq[:-1], seq[1:], trace=True)
    a = np.array(rows)
    ref = a[64:pre]
    z = (a - ref.mean(0)) / (ref.std(0) + 1e-9)
    after = z[pre + Lc:]
    ep = {}
    for j, name in enumerate(('error', 'innovation', 'dopamine_spikes', 'phasic_dopamine', 'ticks')):
        back = next((t for t in range(len(after)) if abs(after[t:t + 4, j].mean()) < 1.0), len(after))
        ep[name] = {'z_during': float(z[pre:pre + Lc, j].mean()), 'z_first_8_after': float(after[:8, j].mean()),
                    'z_8_to_32_after': float(after[8:32, j].mean()), 'words_until_back': int(back)}
    report['episodes'][Lc] = ep
    print(f'chaos L={Lc}:', {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in ep.items()}, flush=True)
json.dump(report, open(fork / 'dopamine_states.json', 'w'), indent=1)
