"""Gestation with regime homeostasis (attach_regime_homeostasis): the newborn's birth weights stay fixed (no learning)
while it reads text, so that every neuron's threshold settles to its place inside its own free-membrane distribution
(location b, width = input gain s) before experience changes synapses.  The newborn's earlier rate-homeostasis state is
discarded; the weights are the birth weights (the earlier gestation did not learn).

Every REPORT ticks, per group: spike rate (Hz at 1 tick = 1 ms) and duty (rate x tau_m), mean drive b (rheobase
units), median log gain, median z = (theta - mean u) / sd u of the free membrane (target ~1), the fraction of ticks
with u above threshold (target q* ~ 0.16), and the median distance of the membrane to threshold in thresholds;
NLL of the words read.  At the end the state goes into <out>/last.pt with config regime_homeostasis = true.
With 'tie' as the fifth argument the writer is first tied by sensory cell type (attach_tied_writer, convert from
the newborn's per-cell rows); a newborn that already has regime homeostasis keeps its regime state.
Usage: python gestation_regime.py <newborn dir> <out dir> <ticks> [seed] [tie]
"""
import sys, json, shutil, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

src, out, TOTAL = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
TIE = len(sys.argv) > 5 and sys.argv[5] == 'tie'
REPORT = 5000
model, cfg, L = load(src)
for name in ('homeostasis_target',):
    model._buffers.pop(name, None)
graph = ROOT / cfg['graph']
KEEP = bool(cfg.get('regime_homeostasis'))
if not KEEP:
    fl.attach_regime_homeostasis(model, graph, seed=SEED)
if TIE and not cfg.get('tied_writer'):
    print('tied writer', fl.attach_tied_writer(model, graph, graph.parent / 'body-annotations.feather', convert=True), flush=True)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
N = model.n_neurons
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
is_read = np.zeros(N, bool); is_read[model.read_indices.cpu().numpy()] = True
is_inj = np.zeros(N, bool); is_inj[model.topographic_writer.injection_index.cpu().numpy()] = True
core = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan
groups = {'ORN': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON',
          'dopamine': is_dan, 'core': core, 'read surface': is_read, 'sensory (written)': is_inj, 'all': np.ones(N, bool)}
masks = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
tau_m = torch.as_tensor(g['tau_m'], dtype=torch.float32, device='cuda').clamp_min(1.0)
theta = model.get_thresholds().expand(1, N)[0].detach()
acc = {'spikes': torch.zeros(N, device='cuda'), 'above': torch.zeros(N, device='cuda'), 'ticks': 0}
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out_ = original_step(model_, current, *a, **k)
    acc['spikes'] += (out_.ring[0][0] > 0).float()
    acc['above'] += (out_.regime[0] > theta).float()
    acc['ticks'] += 1
    return out_


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val, dtype=np.int64)
torch.manual_seed(0)
physical = dict(L['physical'])
if not KEEP:
    physical['homeostasis'] = torch.empty(0); physical['regime'] = torch.empty(0)
s = move_state(fl.FlyPhysicalState(**physical), 'cuda')
ticks, i, ce = 0, 0, []
out.mkdir(parents=True, exist_ok=True)
logf = open(out / 'gestation.jsonl', 'w')
med = lambda x: float(x.median())
with torch.no_grad():
    while ticks < TOTAL:
        tok, nxt = int(text[i % (len(text) - 1)]), int(text[(i + 1) % (len(text) - 1)])
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([tok], device='cuda'), [], max_ticks=max_ticks,
                                                     writer_baseline_clock='physical', **opts)
        ce.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), torch.tensor([nxt], device='cuda'))))
        i += 1
        ticks += lat.shape[0]
        if acc['ticks'] >= REPORT:
            n = acc['ticks']
            rate = acc['spikes'] / n
            above = acc['above'] / n
            b, R = s.homeostasis[0], s.regime
            sd = (R[2] - R[1] ** 2).clamp_min(1e-12).sqrt()
            z = (theta - R[1]) / sd
            dist = (theta - s.h[0]) / theta.abs()
            row = {'tick': ticks, 'words': i, 'ticks_per_word': ticks / i, 'nll': float(np.mean(ce[-200:])),
                   'silent_fraction': float((rate == 0).float().mean())}
            for k, m in masks.items():
                row[k] = {'Hz': float(rate[m].mean()) * 1000, 'duty_median': med((rate * tau_m)[m]), 'b_mean': float(b[m].mean()),
                          'log_gain_median': med(R[3][m]), 'z_median': med(z[m]), 'above_fraction_median': med(above[m]),
                          'distance_median': med(dist[m])}
            logf.write(json.dumps(row) + '\n'); logf.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v))
                              for k, v in row.items()}), flush=True)
            acc['spikes'].zero_(); acc['above'].zero_(); acc['ticks'] = 0
saved = torch.load(src / 'last.pt', map_location='cpu', weights_only=False)
saved['learner']['physical'] = {k: (tuple(t.detach().cpu() for t in v) if isinstance(v, tuple) else v.detach().cpu())
                                for k, v in s.state_dict().items()}
saved['config'] = {**saved['config'], 'homeostasis': False, 'regime_homeostasis': True, 'regime_seed': SEED}
if TIE:
    saved['model'] = {k: v for k, v in saved['model'].items() if not k.startswith('topographic_writer.proj_')}
    saved['model'].update({k: v.detach().cpu() for k, v in model.named_parameters() if k.startswith('topographic_writer.proj_')})
    saved['config']['tied_writer'] = True
torch.save(saved, out / 'last.pt')
config = json.load(open(src / 'config.json'))
config.update(homeostasis=False, regime_homeostasis=True, regime_seed=SEED, **({'tied_writer': True} if TIE else {}))
json.dump(config, open(out / 'config.json', 'w'))
print('gestated state written to', out / 'last.pt', flush=True)
