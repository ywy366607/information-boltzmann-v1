"""Where does an isolated word's drive die on its way to the Kenyon cells?  With quiet gaps between words a single word
activates ~10% of the olfactory receptors, ~1.7% of the ALPN and no Kenyon cell (m_main c7).  For isolated words
(held for the word's ticks, then `gap` quiet ticks), per layer: the fraction of cells that spike, and the membrane's
best approach to threshold over the word's ticks, as the margin h / theta (1 = threshold), at the layer's top 1%, 3%
and 10% of cells per word -- how far below threshold the would-be code stays.  Distributions over words as
[min | median | mean | max].
Usage: python word_margin_probe.py <checkpoint dir> [words=200] [gap=32] [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
N_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 32
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
theta = opts['thresholds'].detach().float().expand(1, model.n_neurons)[0]
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
cls = ann['class'].astype(str).values
LAYERS = {'olfactory receptors': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell'}
LT = {k: torch.as_tensor(v, device='cuda') for k, v in LAYERS.items()}
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)
words = np.array(val[40_000:40_000 + N_WORDS], dtype=np.int64)
res = {k: {'active': [], 'top1': [], 'top3': [], 'top10': []} for k in LAYERS}
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        act = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
        peak = torch.full((model.n_neurons,), -1e9, device='cuda')
        for t in range(TICKS):
            if t == 0 or HOLD:
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            spk = s.ring[0][0] > 0
            act |= spk
            # a spiking cell's membrane was reset this tick: count it at threshold
            peak = torch.maximum(peak, torch.where(spk, theta, s.h[0].float()))
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        for t in range(GAP):
            baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
            s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
        for k, m in LT.items():
            margin = (peak[m] / theta[m].clamp_min(1e-6)).cpu().numpy()
            res[k]['active'].append(float(act[m].float().mean()))
            for q, name in ((0.99, 'top1'), (0.97, 'top3'), (0.90, 'top10')):
                res[k][name].append(float(np.quantile(margin, q)))


def dist(v):
    v = np.asarray(v[8:])                                     # after a short burn-in
    return [round(float(x), 4) for x in (v.min(), np.median(v), v.mean(), v.max())]


rep = {'words': N_WORDS - 8, 'gap_ticks': GAP, 'ticks_per_word': TICKS}
for k, d in res.items():
    rep[k] = {name: dist(v) for name, v in d.items()}
    print(f'{k:>20}: active {rep[k]["active"]}; margin h/theta at top 1% {rep[k]["top1"]}, 3% {rep[k]["top3"]}, '
          f'10% {rep[k]["top10"]}', flush=True)
print(json.dumps(rep), flush=True)
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
