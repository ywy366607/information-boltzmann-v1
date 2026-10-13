"""Kenyon coding level while reading: per word (8 ticks), the fraction of Kenyon cells with at least one spike, and
the mean over words; also the per-word overlap of the active sets of consecutive words (a dense code overlaps), the
APL / CT1 release (graded) or rate, and the MBON rate.  In the fly ~5-10% of Kenyon cells respond to an odour (the
APL feedback keeps it sparse); a sparse code is the precondition of the mushroom-body memory.  Zero-shot loader
interventions apply (ZS_GRADED, ZS_GRADED_GAIN, ZS_SILENCE_OUTPUT).
Usage: python kenyon_coding_probe.py <checkpoint dir>
"""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))     # the world holds a word on the senses for all its ticks
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
LAYERS = {'olfactory receptors': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'ALLN': cls == 'ALLN', 'Kenyon': cls == 'Kenyon_Cell'}
LT = {k: torch.as_tensor(v, device='cuda') for k, v in LAYERS.items()}
lay = {k: {'level': [], 'overlap': []} for k in LAYERS}; prevs = {}
kc = torch.as_tensor(cls == 'Kenyon_Cell', device='cuda'); mbon = torch.as_tensor(cls == 'MBON', device='cuda')
apl = torch.as_tensor(typ == 'APL', device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)
levels, overlaps, apl_out, mbon_rate = [], [], [], []
with torch.no_grad():
    for place in range(4):
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        prev = None
        for i, w in enumerate(np.array(val[90_000 + place * 3_000:90_000 + place * 3_000 + 64], dtype=np.int64)):
            token = torch.tensor([int(w)], device='cuda'); act = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
            for t in range(TICKS):
                if t == 0 or HOLD:
                    dr, baseline = fl.write_input(model, s, token)
                    s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
                else:
                    baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                    s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
                act |= s.ring[0][0] > 0
                apl_out.append(float(s.ring[0][0][apl].mean())); mbon_rate.append(float((s.ring[0][0][mbon] > 0).float().mean()))
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
            if i >= 16:
                for k, m in LT.items():
                    a_k = act[m]
                    lay[k]['level'].append(float(a_k.float().mean()))
                    if k in prevs and a_k.any():
                        lay[k]['overlap'].append(float((a_k & prevs[k]).sum() / max(1, int(a_k.sum()))))
                    prevs[k] = a_k
            if i >= 16:                                           # after a burn-in
                a = act[kc]
                levels.append(float(a.float().mean()))
                if prev is not None and a.any():
                    overlaps.append(float((a & prev).sum() / max(1, int(a.sum()))))
            prev = act[kc]
for k, d in lay.items():
    print(f'   {k:>20}: active per word {np.mean(d["level"]):.3f}, consecutive-word overlap {np.mean(d["overlap"]):.2f} '
          f'(chance = the level)', flush=True)
lv = np.array(levels)
print(f'Kenyon coding level per word: min {lv.min():.3f} median {np.median(lv):.3f} mean {lv.mean():.3f} max {lv.max():.3f}; '
      f'consecutive-word overlap {np.mean(overlaps):.2f}; APL output per tick {np.mean(apl_out):.3f}; MBON rate {np.mean(mbon_rate) * 1000:.1f} Hz', flush=True)
