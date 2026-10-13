"""Zero-training check (diversity first, step a): does waiting to answer have value in a world, for this brain?
Fixed weights, the checkpoint's own state, 1,024 validation words after 256 burn-in, pulse input, every word held for
16 ticks; the read after every tick against the next word gives a CE curve per word.  Then, analytically per world:
  pace: geometric mean mu, or a shifted discrete Pareto P(K >= k) = ((k - 1 + c)/c)^-alpha (heavy tail, falling hazard);
  miss value: uniform (-log V) or the listener's prior (-CE of the next word under the unigram distribution: an answer
  is worth the information it adds beyond what the listener already expects).
For each word the value of answering at tick tau is P(K >= tau)(-CE_tau) + P(K < tau)(miss), tau = 1..16 (a fixed
plan per word, hindsight-optimal per word = an upper bound; the policy optimal per pace = best tau for all words).
Reported: fraction of words whose best tau > 1, mean gain of the per-word best over tau = 1, the best common tau, and
the gain of waiting split by the word's tick-1 CE quartile (does difficulty predict the value of waiting?).
Usage: python wait_oracle.py <checkpoint dir> [out.json]
"""
import sys, json, math, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
quiet = torch.zeros(1, model.n_neurons, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, T = 60_000, 16
seq = np.array(val[region - 1:region + BURN + SCORE], dtype=np.int64)
train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
counts = np.bincount(np.asarray(train[:20_000_000], dtype=np.int64), minlength=50257).astype(np.float64) + 0.5
unigram_ce = -np.log(counts / counts.sum())
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
curves = []
with torch.no_grad():
    for i in range(len(seq) - 1):
        token = torch.tensor([int(seq[i])], device='cuda'); target = int(seq[i + 1])
        ces = []
        for t in range(T):
            if t == 0:
                drive, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                s = fl.step_fly_physical_tick(model, s, None, quiet, writer.lambda_adapt * s.baseline, opts, base_rates=opts['base_rates'])
            lat = fl.read_fly_motor(model, s)
            ces.append(float(F.cross_entropy(model.decoder(model.read_norm(lat)).float(), torch.tensor([target], device='cuda'))))
        s = fl.expect_next(model, s, lat)
        if i >= BURN:
            curves.append(ces)
C = np.array(curves)                                             # [words, T]
nxt = seq[BURN + 1:BURN + 1 + len(C)]
U = unigram_ce[nxt]
q = np.quantile(C[:, 0], [0.25, 0.5, 0.75])
diff = np.digitize(C[:, 0], q)                                   # 0 easiest .. 3 hardest quartile


def survival(kind, a, b):
    t = np.arange(1, T + 1)
    if kind == 'geo':
        return (1 - 1 / a) ** (t - 1)                            # P(K >= t), mean a
    return ((t - 1 + b) / b) ** (-a)                             # shifted Pareto, alpha a, scale c = b


rep = {'checkpoint': str(fork), 'ce_by_tick_mean': [round(float(x), 3) for x in C.mean(0)],
       'unigram_ce_mean': float(U.mean()), 'words': int(len(C)), 'worlds': []}
for kind, a, b in (('geo', 2, 0), ('geo', 8, 0), ('geo', 16, 0), ('pareto', 1.0, 2), ('pareto', 1.0, 4), ('pareto', 1.0, 8),
                   ('pareto', 0.5, 4)):
    S = survival(kind, a, b)
    for miss_name, miss in (('uniform', np.full(len(C), math.log(50257))), ('prior', U)):
        V = S[None, :] * (-C) + (1 - S[None, :]) * (-miss[:, None])          # value of answering at tau
        best = V.argmax(1)
        gain = V.max(1) - V[:, 0]
        common = int(V.mean(0).argmax()) + 1
        w = {'pace': f'{kind}({a},{b})' if kind == 'pareto' else f'geo(mu={a})', 'miss': miss_name,
             'frac_best_after_tick1': float((best > 0).mean()), 'mean_gain_per_word_best': float(gain.mean()),
             'best_common_tau': common, 'gain_common_over_tick1': float(V.mean(0).max() - V.mean(0)[0]),
             'median_best_tau': float(np.median(best + 1)),
             'gain_by_difficulty_quartile': [round(float(gain[diff == k].mean()), 3) for k in range(4)],
             'frac_wait_by_difficulty_quartile': [round(float((best[diff == k] > 0).mean()), 3) for k in range(4)]}
        rep['worlds'].append(w)
        print(json.dumps(w), flush=True)
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'wait_oracle.json'
json.dump(rep, open(out, 'w'), indent=1)
print('ce_by_tick', rep['ce_by_tick_mean'], 'unigram CE', round(rep['unigram_ce_mean'], 3))
