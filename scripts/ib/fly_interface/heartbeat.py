"""The core's 'heartbeat': population activity tick by tick (1 tick = 1 ms: delays from axon length at 0.3 m/s plus
0.8 ms synaptic delay, membrane time constants in ms).  Fixed weights.

Records 2,000 ticks of self-paced reading (after 256 words of warm-up), then 2,000 ticks of silence, per tick:
spikes per neuron of the core (excitatory / inhibitory cells apart), dopamine cells, Kenyon cells, central complex,
motor read surface, sensory (injected) neurons, rest of the brain; the membrane potential (relative to threshold) of
four individual core cells and two read cells; the ticks at which words arrive.  Then the power spectrum of each
population (Hz) and an ECG-style figure.
Usage: python heartbeat.py <fork dir> <out png>
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fork, png = Path(sys.argv[1]), Path(sys.argv[2])
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
is_read = np.zeros(N, bool); is_read[model.read_indices.cpu().numpy()] = True
is_inj = np.zeros(N, bool); is_inj[model.topographic_writer.injection_index.cpu().numpy()] = True
sign = np.asarray(g['nt_sign']).reshape(-1)[:N]
core = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan
groups = {'core (excitatory cells)': core & (sign > 0), 'core (inhibitory cells)': core & (sign < 0),
          'dopamine cells': is_dan, 'Kenyon cells': (sup == 'cb_intrinsic') & (cls == 'Kenyon_Cell'),
          'central complex': (sup == 'cb_intrinsic') & (cls == 'CX'), 'motor read surface': is_read,
          'sensory (injected)': is_inj & ~core, 'rest of the brain': ~core & ~is_dan & ~is_read & ~is_inj & (sup != 'cb_intrinsic')}
masks = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
sizes = {k: float(v.sum()) for k, v in groups.items()}
thr = model.get_thresholds().reshape(-1)
rng = np.random.default_rng(3)
log = {'rows': [], 'volt': []}
original_step = fl.step_fly_physical_tick
watch = None


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    if watch is not None:
        s = (out.ring[0][0] > 0).float()
        log['rows'].append(torch.stack([s[m].sum() for m in masks.values()]))
        log['volt'].append((out.h[0, watch] / thr[watch]).clone())
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, TICKS = 60_000, 256, 2000
text = np.array(val[region - 1:region + 4000], dtype=np.int64)
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
arrivals = []
with torch.no_grad():
    i = 0
    for i in range(BURN):
        s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(text[i])], device='cuda'), [], max_ticks=max_ticks,
                                        writer_baseline_clock='physical', **opts)
    # cells whose membrane is recorded (the most variable are shown)
    watch_core = rng.choice(np.flatnonzero(core), 4000, replace=False)
    watch_read = rng.choice(np.flatnonzero(is_read), 400, replace=False)
    watch = torch.as_tensor(np.r_[watch_core, watch_read], device='cuda')
    while len(log['rows']) < TICKS:
        arrivals.append(len(log['rows']))
        i += 1
        s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(text[i])], device='cuda'), [], max_ticks=max_ticks,
                                        writer_baseline_clock='physical', **opts)
    reading = len(log['rows'])
    # silence: the last word, then nothing
    s, *_ = fl.advance_fly_token_go(model, s, torch.tensor([int(text[i + 1])], device='cuda'), [], max_ticks=max_ticks,
                                    writer_baseline_clock='physical', silence_after=TICKS, **opts)
    arrivals.append(reading)
R = torch.stack(log['rows']).cpu().numpy() / np.array(list(sizes.values()))[None]       # spikes per neuron per tick
V = torch.stack(log['volt']).cpu().numpy()
names = list(groups)
# choose the 4 core cells and 2 read cells that fire most (for the ECG strip)
show_core = np.argsort(-V[:, :4000].std(0))[:4]
show_read = 4000 + np.argsort(-V[:, 4000:].std(0))[:2]


def spectrum(x):
    x = x - x.mean()
    f = np.fft.rfftfreq(len(x), d=1e-3)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return f, p


report = {'fork': str(fork), 'ms_per_tick': 1.0, 'reading_ticks': int(reading), 'silence_ticks': int(len(R) - reading),
          'words_read': len(arrivals) - 1, 'mean_ms_per_word': float(reading / max(len(arrivals) - 1, 1)), 'populations': {}}
for j, n in enumerate(names):
    out = {'rate_reading_hz': float(R[:reading, j].mean() * 1000), 'rate_silence_hz': float(R[reading:, j].mean() * 1000)}
    for part, sl in (('reading', slice(0, reading)), ('silence', slice(reading + 200, None))):
        f, p = spectrum(R[sl, j])
        band = (f >= 1) & (f <= 200)
        k = np.argmax(p[band])
        out[f'peak_hz_{part}'] = float(f[band][k])
        out[f'peak_power_share_{part}'] = float(p[band][k] / p[band].sum())
    report['populations'][n] = out
    print(f'{n:26s}', {k: round(v, 3) for k, v in out.items()}, flush=True)
json.dump(report, open(png.with_suffix('.json'), 'w'), indent=1)
np.savez_compressed(png.with_suffix('.npz'), rates=R, volt=V[:, np.r_[show_core, show_read]], arrivals=np.array(arrivals),
                    reading=reading, names=np.array(names))

# figure: ECG-style strips
t = np.arange(len(R))
fig, axes = plt.subplots(6, 1, figsize=(15, 13), sharex=False, gridspec_kw={'height_ratios': [1.3, 1.3, 1, 1, 1.4, 1.4]})
win = slice(reading - 600, reading + 600)          # 0.6 s of reading, then 0.6 s of silence
ax = axes[0]
for j in (0, 1):
    ax.plot(t[win], R[win, j] * 1000, lw=0.9, label=names[j])
for a in arrivals:
    if win.start <= a < win.stop and a < reading:
        ax.axvline(a, color='0.75', lw=0.5, zorder=0)
ax.axvline(reading, color='k', lw=1.2)
ax.set_ylabel('Hz per cell'); ax.legend(loc='upper left', fontsize=8)
ax.set_title('Core population "heartbeat": 0.6 s reading (grey lines = a word arrives) | 0.6 s of silence (after the black line)')
ax = axes[1]
for j in (2, 3, 5, 6):
    ax.plot(t[win], R[win, j] * 1000, lw=0.8, label=names[j])
ax.axvline(reading, color='k', lw=1.2); ax.set_ylabel('Hz per cell'); ax.legend(loc='upper left', fontsize=8)
for k, (idx, lab) in enumerate(((show_core[:2], 'core cell'), (show_read[:2], 'read cell'))):
    ax = axes[2 + k]
    for c in idx:
        ax.plot(t[win], V[win, c], lw=0.7)
    ax.axhline(1.0, color='r', lw=0.6, ls='--'); ax.axvline(reading, color='k', lw=1.2)
    ax.set_ylabel(f'{lab}\nV / threshold')
for k, (part, sl) in enumerate((('reading', slice(0, reading)), ('silence', slice(reading + 200, None)))):
    ax = axes[4 + k]
    for j in (0, 1, 2, 5):
        f, p = spectrum(R[sl, j])
        band = (f >= 1) & (f <= 150)
        ax.semilogy(f[band], p[band] / p[band].sum(), lw=0.9, label=names[j])
    ax.set_ylabel(f'power share\n({part})'); ax.legend(fontsize=8)
axes[-1].set_xlabel('frequency (Hz)')
for ax in axes[:4]:
    ax.set_xlabel('time (ms; 1 tick = 1 ms)')
fig.tight_layout()
fig.savefig(png, dpi=110)
print('saved', png, flush=True)
