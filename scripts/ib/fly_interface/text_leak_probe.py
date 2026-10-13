"""Does text drive the valence channel?  (Audit 2026-10-13: the old writer drove every receptor, gustatory included,
so every word 'tasted' sugar and bitter.)  From the reading state (4 places), 64 words read as the world does vs a
fork with the same state where the words are withheld (quiet ticks); rates (Hz per cell) of MN9, PPL1, PAM, the
gustatory receptors, the olfactory receptors, and the fraction of sensory cells whose membrane falls below E_I at any
tick (non-physical).  Text -> valence leak = reading - withheld for MN9 / PPL1 / PAM / gustatory.
Usage: python text_leak_probe.py <checkpoint dir>
"""
import os, sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values.astype(str), ann['class'].astype(str).values.astype(str)
dan = np.zeros(len(typ), bool); dan[graph['dan_indices']] = True
sc = np.array(graph['superclass_names'])[graph['superclass_id']].astype(str)
G = {'MN9': typ == 'MN9', 'PPL1': dan & np.char.startswith(typ, 'PPL1'), 'PAM': dan & np.char.startswith(typ, 'PAM'),
     'gustatory': cls == 'gustatory', 'olfactory': cls == 'olfactory', 'MBON': cls == 'MBON'}
G = {k: torch.as_tensor(v, device='cuda') for k, v in G.items()}
sens = torch.as_tensor(np.char.find(sc, 'sensory') >= 0, device='cuda')
E_I = float(getattr(model, 'E_I', -0.2)); thr = opts['thresholds'].detach().expand(1, model.n_neurons)[0]
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)


def read(s, words, text=True):
    c = {k: 0.0 for k in G}; below = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda'); n = 0
    allc = torch.zeros(model.n_neurons, device='cuda')
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        for t in range(TICKS):
            if text and (t == 0 or HOLD):
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            spk = s.ring[0][0] > 0
            allc += spk.float()
            for k, m in G.items():
                c[k] += float(spk[m].float().mean())
            below |= s.h[0] < E_I
            n += 1
        if text:
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
    rate = allc / n * 1000
    c['cells >200 Hz'] = float((rate > 200).sum()) * n / 1000; c['brain mean'] = float(rate.mean()) * n / 1000
    read.last_rate = rate.cpu().numpy()
    return s, {k: v / n * 1000 for k, v in c.items()}, float(below[sens].float().mean())


rows = []
with torch.no_grad():
    for place in range(4):
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        s, _, _ = read(s, np.array(val[70_000 + place * 3_000:70_000 + place * 3_000 + 32], dtype=np.int64))
        words = np.array(val[70_000 + place * 3_000 + 32:70_000 + place * 3_000 + 96], dtype=np.int64)
        _, on, below = read(s, words, True)
        _, off, _ = read(s, words, False)
        rows.append((on, off, below))
        latched = latched + [read.last_rate > 200] if 'latched' in dir() else [read.last_rate > 200]
for k in list(G) + ['cells >200 Hz', 'brain mean']:
    on = np.mean([r[0][k] for r in rows]); off = np.mean([r[1][k] for r in rows])
    print(f'{k:>10}: reading {on:7.2f} Hz, words withheld {off:7.2f} Hz, text-driven {on - off:+7.2f} Hz', flush=True)
print(f'sensory cells below E_I at some tick while reading: {np.mean([r[2] for r in rows]):.3f}', flush=True)
# which cells stay above 200 Hz with the words withheld (in at least 2 of the 4 places), and do they feed the output?
L2 = np.sum(latched, 0) >= 2
if os.environ.get('LATCHED_OUT'):
    np.save(os.environ['LATCHED_OUT'], L2)                    # the latched cells, for attribution
read_cells = np.zeros(len(typ), bool); read_cells[model.read_indices.cpu().numpy()] = True
print(f'latched (> 200 Hz without words, >= 2 of 4 places): {int(L2.sum())} cells; read cells among them {int((L2 & read_cells).sum())}; '
      f'motor / descending {int((L2 & np.isin(sc, ["cb_motor", "vnc_motor", "descending_neuron"])).sum())}', flush=True)
print('   by superclass:', pd.Series(sc[L2]).value_counts().head(8).to_dict(), flush=True)
print('   by type:', pd.Series(typ[L2]).value_counts().head(15).to_dict(), flush=True)
