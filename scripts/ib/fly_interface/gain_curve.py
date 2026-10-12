"""What would an opportunity cost of time do?  Fixed weights, each word read for 16 ticks, the read at every tick
predicts the next word (the next word arrives after the 16 ticks, so nothing leaks).

Per word: CE by tick.  The marginal value theorem: leave a word when the gain of one more tick falls below the average
gain rate rho = (CE at tick 1 - CE at leaving) / ticks spent, averaged over words (self-consistent).  Reported: the
mean curve, the rate-optimal common leaving tick and its NLL cost, and whether the per-word benefit of thinking longer
(CE at tick 3 - CE at tick 10, or at the word's best tick) is predictable from the word's rarity or from the output's
entropy at tick 3 (what a precision allocation would need).
Usage: python gain_curve.py <fork dir>
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
T = 16
reads = []
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    reads.append(fl.read_fly_motor(model_, out))
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 1024
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
counts = np.bincount(np.load(ROOT / cfg['data'] / 'train.npy').astype(np.int64), minlength=50257) + 1.0
rarity = -np.log(counts / counts.sum())
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
curves, ent3 = [], []
with torch.no_grad():
    for i, tok in enumerate(inputs):
        reads.clear()
        s, _ = fl.advance_fly_token_stream(model, s, torch.tensor([int(tok)], device='cuda'), ticks=T,
                                           writer_baseline_clock='physical', **opts)
        if i < BURN:
            continue
        z = torch.cat(reads[-T:], 0)
        logits = model.decoder(model.read_norm(z)).float()
        tgt = torch.full((T,), int(targets[i]), device='cuda')
        curves.append(F.cross_entropy(logits, tgt, reduction='none').cpu().numpy())
        lp = torch.log_softmax(logits[2], -1)
        ent3.append(float(-(lp.exp() * lp).sum()))
C = np.array(curves)                         # words x ticks
mean = C.mean(0)
best_fixed = int(np.argmin(mean)) + 1
# self-consistent common leaving tick under the marginal value theorem (one tick to take in the next word)
rates = {t: (mean[0] - mean[t - 1]) / (t + 1) for t in range(1, T + 1)}
mvt_tick = max(rates, key=rates.get)
benefit = C[:, 2] - C[:, best_fixed - 1]
oracle = C.min(1)
rar = rarity[inputs[BURN:]]
report = {'fork': str(fork), 'ticks': T, 'mean_ce_by_tick': mean.tolist(), 'best_fixed_tick': best_fixed,
          'nll_best_fixed': float(mean[best_fixed - 1]), 'rate_optimal_tick': mvt_tick, 'rate_at_optimum_nats_per_tick': float(rates[mvt_tick]),
          'nll_at_rate_optimal_tick': float(mean[mvt_tick - 1]), 'rate_at_best_fixed': float(rates[best_fixed]),
          'oracle_per_word_best_tick_nll': float(oracle.mean()),
          'benefit_tick3_to_best_fixed': {'mean': float(benefit.mean()), 'sd': float(benefit.std()),
                                          'spearman_with_rarity': float(spearmanr(rar, benefit).correlation),
                                          'spearman_with_entropy_at_tick3': float(spearmanr(ent3, benefit).correlation)}}
print(json.dumps(report, indent=1), flush=True)
json.dump(report, open(fork / 'gain_curve.json', 'w'), indent=1)
