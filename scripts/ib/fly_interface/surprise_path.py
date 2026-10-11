"""Where does the surprise signal die on its way to the dopamine cells?  Fixed weights, predictive-coding individual,
intrinsic current removed (kappa = 0).  Per word: written innovation norm (surprise) and the spike counts, summed over
the word's ticks, of: sensory (injection) neurons; their direct targets; the dopamine cells' presynaptic partners
(split: those that are direct sensory targets / those that are not); the dopamine cells.  Spearman of each layer's
activity with surprise, and its composition (which superclasses feed the dopamine cells).
"""
import sys, json, collections, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
if getattr(model, 'intrinsic_kappa', None) is not None:
    model.intrinsic_kappa.data.zero_()
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
g = np.load(ROOT / cfg['graph'])
N = model.n_neurons
names = [str(x) for x in g['superclass_names']]
sc = g['superclass_id']
pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]); post = np.concatenate([g['edge_post_e'], g['edge_post_i']])
inj = model.topographic_writer.injection_index.cpu().numpy()
dan = model.dan_indices.cpu().numpy()
inj_mask = np.zeros(N, bool); inj_mask[inj] = True
dan_mask = np.zeros(N, bool); dan_mask[dan] = True
sensory_targets = np.zeros(N, bool); sensory_targets[np.unique(post[inj_mask[pre]])] = True
sensory_targets &= ~inj_mask
dan_inputs = np.zeros(N, bool); dan_inputs[np.unique(pre[dan_mask[post]])] = True
dan_inputs &= ~dan_mask
groups = {'sensory': inj_mask, 'sensory targets': sensory_targets,
          'DAN inputs that are sensory targets': dan_inputs & sensory_targets,
          'DAN inputs that are not': dan_inputs & ~sensory_targets & ~inj_mask,
          'dopamine cells': dan_mask}
print({k: int(v.sum()) for k, v in groups.items()})
print('superclasses feeding the dopamine cells (synapse counts):',
      collections.Counter(names[sc[p]] for p in pre[dan_mask[post]]).most_common(8))
gm = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
counts = []
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    s = (out.ring[0][0] > 0)
    counts.append([float(s[m].float().sum()) for m in gm.values()])
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 1024
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
torch.manual_seed(0)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
rows = []
with torch.no_grad():
    for i, tok in enumerate(inputs):
        t = torch.tensor([int(tok)], device='cuda')
        drive, _ = fl.write_input(model, s, t)
        innov = float(drive[0, model.topographic_writer.injection_index].norm())
        pos_part = float(drive[0, model.topographic_writer.injection_index].clamp_min(0).sum())
        counts.clear()
        s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, t, [], max_ticks=max_ticks, writer_baseline_clock='physical', **opts)
        c = np.array(counts)
        rows.append([innov, pos_part] + c.sum(0).tolist() + [c[0, 0]])
a = np.array(rows[BURN:])
res = {}
for j, name in enumerate(groups):
    res[name] = {'spikes_per_word': float(a[:, 2 + j].mean()),
                 'spearman_with_innovation_norm': float(spearmanr(a[:, 0], a[:, 2 + j]).correlation),
                 'spearman_with_positive_drive': float(spearmanr(a[:, 1], a[:, 2 + j]).correlation)}
res['sensory spikes on the input tick'] = {'spearman_with_innovation_norm': float(spearmanr(a[:, 0], a[:, -1]).correlation),
                                           'spearman_with_positive_drive': float(spearmanr(a[:, 1], a[:, -1]).correlation)}
for k, v in res.items():
    print(f'{k:40s}', {kk: round(vv, 3) for kk, vv in v.items()})
json.dump(res, open(fork / 'surprise_path.json', 'w'), indent=1)
