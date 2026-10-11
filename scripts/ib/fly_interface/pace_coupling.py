"""Does the core's ~8 Hz rhythm couple to the pace of the world when words come slower?  Fixed weights.

The world of 'reading with gaps' the individual knows from training, with a silence after every word: the brain reads
the word at its own pace (go neuron), then G silent ticks, then the next word; the word is predicted by the read at
the end of the silence (as in training).  G = 0, 8, 23, 53, 113 -> about 7, 15, 30, 60, 120 ms per word (1 tick = 1 ms).
Per pace: NLL (read at the end of the silence, and at the go neuron's firing); the core's dominant frequency; the
phase of the core rhythm (band-passed around its own natural frequency, measured in silence) at every word arrival:
phase locking |<exp(i phase)>| (0 = arrivals at random phases, 1 = always at the same phase); and the NLL by quarter of
that phase (does it matter at which phase of the rhythm a word arrives?) with a shuffle test.
Usage: python pace_coupling.py <fork dir> [tag]
"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
from scipy.signal import hilbert
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else ''
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
core = torch.as_tensor((sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan, device='cuda')
n_core = float(core.sum())
trace = []
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    trace.append((out.ring[0][0][core] > 0).float().sum())
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 128, 640
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))


def spectrum(x):
    x = x - x.mean()
    f = np.fft.rfftfreq(len(x), d=1e-3)
    return f, np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2


def bandpass(x, f0, half=2.5):
    f = np.fft.rfftfreq(len(x), d=1e-3)
    X = np.fft.rfft(x - x.mean())
    X[(f < f0 - half) | (f > f0 + half)] = 0
    return np.fft.irfft(X, len(x))


# the core's natural frequency: 3000 ticks of silence after a short warm-up
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
with torch.no_grad():
    for i in range(64):
        s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(inputs[i])], device='cuda'), [], max_ticks=max_ticks,
                                        writer_baseline_clock='physical', **opts)
    trace.clear()
    s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(inputs[64])], device='cuda'), [], max_ticks=max_ticks,
                                    writer_baseline_clock='physical', silence_after=3000, **opts)
quiet = torch.stack(trace).cpu().numpy()[200:] / n_core
f, p = spectrum(quiet)
band = (f >= 2) & (f <= 60)
f0 = float(f[band][np.argmax(p[band])])
report = {'fork': str(fork), 'natural_frequency_hz_in_silence': f0, 'paces': {}}
print(f'core natural frequency in silence: {f0:.2f} Hz', flush=True)

rng = np.random.default_rng(0)
for G in (0, 8, 23, 53, 113):
    torch.manual_seed(0)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    trace.clear()
    arrivals, ce_end, ce_go, ticks = [], [], [], []
    with torch.no_grad():
        for i, tok in enumerate(inputs):
            arrivals.append(len(trace))
            s, lat, _, _, _, after = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [],
                                                             max_ticks=max_ticks, writer_baseline_clock='physical',
                                                             silence_after=G, **opts)
            tgt = torch.tensor([int(targets[i])], device='cuda')
            ce_go.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), tgt)))
            ce_end.append(float(F.cross_entropy(model.decoder(model.read_norm(after if G else lat[-1:])).float(), tgt)))
            ticks.append(lat.shape[0])
    x = torch.stack(trace).cpu().numpy() / n_core
    arr = np.array(arrivals)
    keep = np.arange(BURN, len(inputs))
    start = arr[BURN]
    xs = x[start:]
    fs, ps = spectrum(xs)
    b = (fs >= 2) & (fs <= 150)
    phase = np.angle(hilbert(bandpass(x, f0)))
    ph = phase[arr[keep]]
    plv = float(np.abs(np.exp(1j * ph).mean()))
    ce = np.array(ce_end)[keep]
    quart = ((ph + np.pi) // (np.pi / 2)).astype(int).clip(0, 3)
    by_q = [float(ce[quart == q].mean()) if (quart == q).any() else float('nan') for q in range(4)]
    spread = np.nanmax(by_q) - np.nanmin(by_q)
    null = []
    for _ in range(500):
        sh = rng.permutation(quart)
        m = [ce[sh == q].mean() for q in range(4) if (sh == q).any()]
        null.append(max(m) - min(m))
    interval = float(np.diff(arr[keep]).mean())
    out = {'gap_ticks': G, 'ms_per_word': interval, 'word_rate_hz': 1000.0 / interval, 'go_ticks': float(np.mean(np.array(ticks)[keep])),
           'nll_read_at_end': float(ce.mean()), 'nll_read_at_go': float(np.array(ce_go)[keep].mean()),
           'core_rate_hz': float(xs.mean() * 1000), 'core_peak_hz': float(fs[b][np.argmax(ps[b])]),
           'core_power_at_natural_frequency_share': float(ps[b][np.argmin(np.abs(fs[b] - f0))] / ps[b].sum()),
           'phase_locking_of_arrivals': plv, 'phase_locking_null_mean': float(1 / np.sqrt(len(ph))),
           'nll_by_arrival_phase_quarter': by_q, 'nll_phase_spread': float(spread),
           'nll_phase_spread_shuffle_p': float((np.array(null) >= spread).mean())}
    report['paces'][G] = out
    print(f'G={G:3d}', {k: (round(v, 4) if isinstance(v, float) else ([round(z, 3) for z in v] if isinstance(v, list) else v)) for k, v in out.items()}, flush=True)
json.dump(report, open(fork / f'pace_coupling{tag}.json', 'w'), indent=1)
