"""The gate before any training (user, 2026-10-13: errors kept coming because no fix was followed by a look at what it
did to the cells): per functional group, is the brain alive where it should be?  Zero-shot, fixed weights, reading
the validation text (pulse input, 8 ticks per word, 256 words after 32 burn-in), the physical state carried.
Per group: spikes per cell per tick (min | median | mean | max over cells), the fraction of silent cells (no spike),
of saturated cells (> 0.5 spikes per tick), the median time-averaged v / theta, and how strongly the group follows
the text (correlation of its population rate with the word onsets).  Groups: every annotation class, plus PAM, PPL1,
OA, 5-HT, APL, DPM, the read (output) cells, descending neurons.  Flags groups that are dead (> 90% silent) or
saturated (> 20% saturated).  Zero-shot interventions through the loader's env (ZS_DAN_RENORM, INTRINSIC_KAPPA).
Usage: python cell_health_probe.py <checkpoint dir> [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'cell_health.json'
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))     # the world holds a word on the senses for all its ticks
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values.astype(str), ann['class'].astype(str).values.astype(str)
sc = np.array(graph['superclass_names'])[graph['superclass_id']]
dan = np.zeros(len(typ), bool); dan[graph['dan_indices']] = True
read = np.zeros(len(typ), bool); read[model.read_indices.cpu().numpy()] = True
groups = {f'class:{c}': cls == c for c in np.unique(cls) if (cls == c).sum() >= 2}
groups.update({'PAM': dan & np.char.startswith(typ, 'PAM'), 'PPL1': dan & np.char.startswith(typ, 'PPL1'),
               'OA': np.char.startswith(typ, 'OA-'), '5-HT': np.char.startswith(typ, '5-HT'), 'APL': typ == 'APL',
               'DPM': typ == 'DPM', 'read_cells': read, 'descending': sc == 'descending_neuron'})
groups = {k: v for k, v in groups.items() if v.sum() >= 2}
G = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
BURN, WORDS, K = 32, 256, 8
seq = np.array(val[100_000:100_000 + BURN + WORDS], dtype=np.int64)
thr = opts['thresholds'].detach().expand(1, model.n_neurons)[0]
counts = torch.zeros(model.n_neurons, device='cuda'); vsum = torch.zeros(model.n_neurons, device='cuda')
pop = {k: [] for k in G}; onset = []
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
ticks = 0
with torch.no_grad():
    for i, w in enumerate(seq):
        token = torch.tensor([int(w)], device='cuda')
        for t in range(K):
            if t == 0 or HOLD:
                drive, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            if i >= BURN:
                spk = (s.ring[0][0] > 0).float()
                counts += spk; vsum += s.h[0] / thr; ticks += 1
                for k, m in G.items():
                    pop[k].append(float(spk[m].mean()))
                onset.append(1.0 if t == 0 else 0.0)
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
rate = (counts / ticks).cpu().numpy(); v = (vsum / ticks).cpu().numpy()
# saturated = at >= 90% of the rate cap (one spike per 1 + refractory ticks), so a cell pinned at the cap is seen
SAT = 0.9 / (1 + int(cfg.get('refractory_ticks') or 0))
onset = np.array(onset)
q = lambda x: [float(np.min(x)), float(np.median(x)), float(np.mean(x)), float(np.max(x))]
rep = {'fork': str(fork), 'ticks': ticks, 'groups': {}}
flags = []
for k, m in groups.items():
    r = rate[m]
    p = np.array(pop[k])
    follow = float(np.corrcoef(np.roll(onset, 2), p)[0, 1]) if p.std() > 0 else 0.0     # 2-tick lag
    g = {'n': int(m.sum()), 'rate_min_median_mean_max': q(r), 'silent_frac': float((r == 0).mean()),
         'saturated_frac': float((r >= SAT).mean()), 'median_v_over_theta': float(np.median(v[m])), 'follows_text': follow}
    rep['groups'][k] = g
    if g['silent_frac'] > 0.9:
        flags.append(f'DEAD {k} ({g["silent_frac"]:.0%} silent)')
    if g['saturated_frac'] > 0.2:
        flags.append(f'SATURATED {k} ({g["saturated_frac"]:.0%})')
rep['flags'] = flags
rep['whole_brain'] = {'rate_hz_mean': float(rate.mean() * 1000), 'rate_hz_median': float(np.median(rate) * 1000),
                      'silent_frac': float((rate == 0).mean()), 'saturated_frac': float((rate >= SAT).mean()), 'saturation_rate': SAT}
print(f"whole brain: mean {rep['whole_brain']['rate_hz_mean']:.1f} Hz, median {rep['whole_brain']['rate_hz_median']:.1f} Hz, "
      f"silent {rep['whole_brain']['silent_frac']:.2f}, saturated (>= {SAT * 1000:.0f} Hz) {rep['whole_brain']['saturated_frac']:.4f}")
for k in sorted(groups, key=lambda k: (not k.startswith('class:'), k)):
    g = rep['groups'][k]
    print(f'{k:>38} n {g["n"]:>6}  rate mean {g["rate_min_median_mean_max"][2]:.4f}  silent {g["silent_frac"]:.2f}  '
          f'sat {g["saturated_frac"]:.2f}  v/theta {g["median_v_over_theta"]:+.2f}  follows text {g["follows_text"]:+.2f}')
print('FLAGS:', flags)
json.dump(rep, open(out, 'w'), indent=1)
