"""How much could thinking on demand gain?  The best individual reads with a fixed 12 ticks per token; at every tick its
read predicts the next token.  Compare: the best single fixed tick for all tokens, vs. each token stopping at its own
best tick (an oracle upper bound for a perfect go neuron)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/think_and_speak.py', encoding='utf-8').read().split('report = {')[0])
T = 12
torch.manual_seed(0)
targets = np.array(val[cursor:cursor + 1280], dtype=np.int64)
inputs = np.concatenate(([L['previous_token']], targets[:-1]))
curves = []
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
with torch.no_grad():
    for i, tok in enumerate(inputs):
        s, ticks = fl.advance_fly_input_event(model, s, torch.tensor([int(tok)], device='cuda'), settle_ticks=T - 1,
                                              writer_baseline_clock='physical', return_ticks=True, **opts)
        if i >= 256:
            lat = torch.cat([fl.read_fly_motor(model, st) for st in ticks], 0)
            ce = F.cross_entropy(model.decoder(model.read_norm(lat)).float(), torch.full((T,), int(targets[i]), device='cuda'),
                                 reduction='none')
            curves.append(ce.cpu().numpy())
c = np.array(curves)
mean_by_tick = c.mean(0)
best_fixed = int(mean_by_tick.argmin()) + 1
oracle = c.min(1)
best_tick = c.argmin(1) + 1
print('mean NLL by tick 1..12:', np.round(mean_by_tick, 3).tolist())
print(f'best fixed tick {best_fixed}: {mean_by_tick.min():.4f}   per-token oracle: {oracle.mean():.4f}   headroom {mean_by_tick.min() - oracle.mean():.4f}')
print('per-token best tick histogram:', {k: int((best_tick == k).sum()) for k in range(1, T + 1)})
# a less flattering bound: the oracle among ticks >= 4 only, and noise: how much of the per-token minimum is chance?
half = np.random.default_rng(0).permutation(len(c))
print('oracle restricted to ticks 6..10:', round(float(c[:, 5:10].min(1).mean()), 4))
json.dump({'mean_by_tick': mean_by_tick.tolist(), 'best_fixed_tick': best_fixed, 'oracle': float(oracle.mean()),
           'best_tick_hist': np.bincount(best_tick, minlength=T + 1).tolist()},
          open(fork / 'think_headroom.json', 'w'), indent=1)
