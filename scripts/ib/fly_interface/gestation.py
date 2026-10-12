"""Gestation: a newborn with homeostatic intrinsic plasticity lives with fixed weights (no learning) while reading text,
so that every neuron's excitability settles toward its own target rate before experience starts to change synapses.

Every REPORT ticks: per group (dopamine cells, Kenyon cells, core excitatory / inhibitory, MBONs, motor read surface,
sensory, rest) the spike rate, the mean intrinsic drive b; over all neurons with a measurable rate the median |log
(rate / target)| and the fraction within a factor of 2 of their target (rates as exponential averages over 5,000
ticks); the core's regularity (spikes per neuron per tick vs its own tau_m: duty cycle); NLL of the words read.
At the end the gestated physical state is written into <out>/last.pt (a copy of the newborn checkpoint with the state
replaced), ready for --extend-budget.
Usage: python gestation.py <newborn dir> <out dir> <ticks>
"""
import sys, json, shutil, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

src, out, TOTAL = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
REPORT = 5000
model, cfg, L = load(src)
assert getattr(model, 'homeostasis_target', None) is not None, 'newborn needs --homeostasis'
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
groups = {'dopamine cells': is_dan, 'Kenyon cells': cls == 'Kenyon_Cell', 'core excitatory': core & (sign > 0),
          'core inhibitory': core & (sign < 0), 'MBONs': cls == 'MBON', 'motor read surface': is_read,
          'sensory': is_inj, 'rest': ~core & ~is_dan & ~is_read & ~is_inj & (cls != 'Kenyon_Cell') & (cls != 'MBON')}
masks = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
target = model.homeostasis_target[0]
tau_m = torch.as_tensor(g['tau_m'], dtype=torch.float32, device='cuda').clamp_min(1.0)
ema = torch.zeros(N, device='cuda'); lam = 1 - 1 / 5000
acc = {'spikes': torch.zeros(len(groups), device='cuda'), 'ticks': 0}
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    global ema
    out_ = original_step(model_, current, *a, **k)
    s = (out_.ring[0][0] > 0).float()
    ema = lam * ema + (1 - lam) * s
    acc['spikes'] += torch.stack([s[m].mean() for m in masks.values()])
    acc['ticks'] += 1
    return out_


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val, dtype=np.int64)
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
ticks, i, ce, rows = 0, 0, [], []
next_report = REPORT
log = open(out.with_suffix('.jsonl'), 'w') if False else None
out.mkdir(parents=True, exist_ok=True)
logf = open(out / 'gestation.jsonl', 'w')
with torch.no_grad():
    while ticks < TOTAL:
        tok, nxt = int(text[i % (len(text) - 1)]), int(text[(i + 1) % (len(text) - 1)])
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([tok], device='cuda'), [], max_ticks=max_ticks,
                                                     writer_baseline_clock='physical', **opts)
        ce.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), torch.tensor([nxt], device='cuda'))))
        i += 1
        ticks = ticks + lat.shape[0]
        if acc['ticks'] >= REPORT:
            b = s.homeostasis[0]
            rate = ema
            ok = rate > 0
            logr = torch.log(rate[ok] / target[ok])
            row = {'tick': ticks, 'words': i, 'nll': float(np.mean(ce[-200:])),
                   'rate_per_group': {k: float(v / acc['ticks']) for k, v in zip(groups, acc['spikes'])},
                   'drive_b_per_group': {k: float(b[m].mean()) for k, m in masks.items()},
                   'fraction_silent_5k': float((rate < 1e-5).float().mean()),
                   'median_abs_log_rate_over_target': float(logr.abs().median()) if ok.any() else None,
                   'fraction_within_2x_of_target': float(((rate / target) > 0.5).logical_and((rate / target) < 2).float().mean()),
                   'core_duty_cycle_median': float((rate * tau_m)[masks['core excitatory'] | masks['core inhibitory']].median()),
                   'b_min': float(b.min()), 'b_max': float(b.max())}
            rows.append(row)
            logf.write(json.dumps(row) + '\n'); logf.flush()
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else ({kk: round(vv, 5) for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in row.items()}), flush=True)
            acc['spikes'].zero_(); acc['ticks'] = 0
# write the gestated state into a copy of the newborn checkpoint
saved = torch.load(src / 'last.pt', map_location='cpu', weights_only=False)
saved['learner']['physical'] = {k: (tuple(t.detach().cpu() for t in v) if isinstance(v, tuple) else v.detach().cpu())
                                for k, v in s.state_dict().items()}
torch.save(saved, out / 'last.pt')
shutil.copy(src / 'config.json', out / 'config.json')
np.save(out / 'gestation_rate_ema.npy', ema.cpu().numpy())
print('gestated state written to', out / 'last.pt', flush=True)
