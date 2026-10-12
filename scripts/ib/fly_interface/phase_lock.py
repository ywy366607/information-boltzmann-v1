"""Does the brain take in the next word at a particular phase of the core's fast inhibitory rhythm?  Fixed weights,
self-paced reading at the natural pace.

The rhythm's frequency f0 is the strongest 100-250 Hz peak of the core's inhibitory cells in 2000 ticks of silence.
For every word arrival (= the tick after the go neuron fired) the phase is estimated causally: a sinusoid at f0 is fitted
by least squares to the population rate in the 20 ticks before the arrival and extrapolated to the arrival tick.  Phase
locking |<exp(i phase)>| against the same estimate at random ticks of the same recording; NLL of the word by quarter of
that phase, with a shuffle test.
Usage: python phase_lock.py <fork dir>
"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
N = model.n_neurons
graph = ROOT / cfg['graph']
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
sign = np.asarray(g['nt_sign']).reshape(-1)[:N]
inh = torch.as_tensor((sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan & (sign < 0), device='cuda')
trace = []
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    trace.append((out.ring[0][0][inh] > 0).float().sum())
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 1024
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
arrivals, ce = [], []
with torch.no_grad():
    for i, tok in enumerate(inputs):
        arrivals.append(len(trace))
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=max_ticks,
                                                     writer_baseline_clock='physical', **opts)
        ce.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), torch.tensor([int(targets[i])], device='cuda'))))
    reading = len(trace)
    s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(targets[-1])], device='cuda'), [], max_ticks=max_ticks,
                                    writer_baseline_clock='physical', silence_after=2000, **opts)
x = torch.stack(trace).cpu().numpy()
quiet = x[reading + 200:]
f = np.fft.rfftfreq(len(quiet), 1e-3)
p = np.abs(np.fft.rfft((quiet - quiet.mean()) * np.hanning(len(quiet)))) ** 2
band = (f >= 100) & (f <= 250)
f0 = float(f[band][np.argmax(p[band])])
W = 20
tt = np.arange(-W, 0)
X = np.c_[np.cos(2 * np.pi * f0 * tt / 1000), np.sin(2 * np.pi * f0 * tt / 1000), np.ones(W)]
pinv = np.linalg.pinv(X)


def phase_at(t):
    c, s_, _ = pinv @ x[t - W:t]
    return np.angle(c - 1j * s_)          # phase of cos(2 pi f0 (t' - t)) at t' = t, i.e. extrapolated to t


arr = np.array(arrivals[BURN:])
ph = np.array([phase_at(t) for t in arr])
rng = np.random.default_rng(0)
rand_t = rng.integers(arrivals[BURN], reading, size=(200, len(arr)))
plv = float(np.abs(np.exp(1j * ph).mean()))
null = np.array([np.abs(np.exp(1j * np.array([phase_at(t) for t in row])).mean()) for row in rand_t[:50]])
c = np.array(ce[BURN:])
q = ((ph + np.pi) // (np.pi / 2)).astype(int).clip(0, 3)
by_q = [float(c[q == k].mean()) for k in range(4)]
spread = max(by_q) - min(by_q)
shuf = [max(c[perm == k].mean() for k in range(4)) - min(c[perm == k].mean() for k in range(4))
        for perm in (rng.permutation(q) for _ in range(1000))]
iv = np.diff(np.array(arrivals[BURN:]))
report = {'fork': str(fork), 'f0_hz_silence': f0, 'peak_power_share': float(p[band].max() / p[(f >= 2) & (f <= 250)].sum()),
          'ms_per_word': {'min': int(iv.min()), 'median': float(np.median(iv)), 'mean': float(iv.mean()), 'max': int(iv.max())},
          'phase_locking': plv, 'phase_locking_random_ticks': {'mean': float(null.mean()), 'max': float(null.max())},
          'arrival_count_by_phase_quarter': [int((q == k).sum()) for k in range(4)],
          'nll_by_phase_quarter': by_q, 'nll_spread': float(spread), 'nll_spread_shuffle_p': float((np.array(shuf) >= spread).mean()),
          'interval_mod_period_counts': np.bincount(iv % max(1, round(1000 / f0)), minlength=max(1, round(1000 / f0))).tolist()}
print(json.dumps(report, indent=1), flush=True)
json.dump(report, open(fork / 'phase_lock.json', 'w'), indent=1)
