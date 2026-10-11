"""What does the content-free core do for the motor read surface?  Fixed weights, self-paced reading, same text.

The core ('core: other', 24,687 central-brain intrinsic cells) sends ~82% of the excitatory and ~84% of the inhibitory
synaptic drive the motor read surface receives, one hop away.  Arms (what the rest of the brain receives from the core):
  intact            as learned
  silenced          no core spikes
  poisson           each core cell fires at its own measured rate, independently and at random times: same mean drive,
                    no rhythm, no timing, no correlations
  excitatory off    only the core's excitatory cells silenced
  inhibitory off    only the core's inhibitory cells silenced
Per arm: NLL, and the read surface's operating point at the read: mean distance to threshold (in thresholds), mean
excitatory / inhibitory conductance, and the spread of the read across words (how much the read varies with the word).
Usage: python core_role.py <fork dir>
"""
import sys, json, dataclasses, numpy as np, pandas as pd, torch, torch.nn.functional as F
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
core_np = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan
sign = np.asarray(g['nt_sign']).reshape(-1)[:N]
core = torch.as_tensor(core_np, device='cuda')
core_exc = torch.as_tensor(core_np & (sign > 0), device='cuda')
core_inh = torch.as_tensor(core_np & (sign < 0), device='cuda')
print('core cells:', int(core.sum()), 'excitatory', int(core_exc.sum()), 'inhibitory', int(core_inh.sum()), flush=True)
read = model.read_indices
thr_read = model.get_thresholds().reshape(-1)[read]
mode = {'arm': 'intact', 'rate': None, 'count': None, 'ticks': 0}
gen = torch.Generator(device='cuda')
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    s0 = out.ring[0]
    if mode['count'] is not None:
        mode['count'] += (s0[0] > 0).float(); mode['ticks'] += 1
    arm = mode['arm']
    if arm == 'intact':
        return out
    one = s0.new_ones(())
    if arm == 'silenced':
        new = torch.where(core.unsqueeze(0), torch.zeros_like(s0), s0)
    elif arm == 'excitatory off':
        new = torch.where(core_exc.unsqueeze(0), torch.zeros_like(s0), s0)
    elif arm == 'inhibitory off':
        new = torch.where(core_inh.unsqueeze(0), torch.zeros_like(s0), s0)
    elif arm == 'poisson':
        fire = (torch.rand(N, device='cuda', generator=gen) < mode['rate']).unsqueeze(0)
        new = torch.where(core.unsqueeze(0), torch.where(fire, one * 1.0, torch.zeros_like(s0)), s0)
    return dataclasses.replace(out, ring=(new.to(s0.dtype), *out.ring[1:]))


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 768
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))


def run(arm, count=False):
    torch.manual_seed(0); gen.manual_seed(1)
    mode.update(arm=arm, count=torch.zeros(N, device='cuda') if count else None, ticks=0)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    ce, dist, ge, gi, reads, ticks = [], [], [], [], [], []
    with torch.no_grad():
        for i, tok in enumerate(inputs):
            s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=max_ticks,
                                                         writer_baseline_clock='physical', **opts)
            if i < BURN:
                continue
            ce.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), torch.tensor([int(targets[i])], device='cuda'))))
            h = s.h[0, read]
            dist.append(float(((thr_read - h) / thr_read.abs().clamp_min(1e-9)).mean()))
            ge.append(float(s.ge[0, read].mean()) if s.ge.numel() else np.nan)
            gi.append(float(s.gi[0, read].mean()) if s.gi.numel() else np.nan)
            reads.append(lat[-1].float().cpu()); ticks.append(lat.shape[0])
    R = torch.stack(reads)
    out = {'nll': float(np.mean(ce)), 'ticks': float(np.mean(ticks)),
           'read_distance_to_threshold': float(np.mean(dist)), 'read_ge': float(np.nanmean(ge)), 'read_gi': float(np.nanmean(gi)),
           'read_spread_across_words': float(R.std(0).mean()), 'read_mean_abs': float(R.abs().mean())}
    if count:
        mode['rate_measured'] = (mode['count'] / max(mode['ticks'], 1)).clamp(0, 1)
    return out


report = {'fork': str(fork)}
report['intact'] = run('intact', count=True)
mode['rate'] = mode['rate_measured']
print('core mean rate', float(mode['rate'][core].mean()), flush=True)
for arm in ('intact', 'silenced', 'poisson', 'excitatory off', 'inhibitory off'):
    if arm != 'intact':
        report[arm] = run(arm)
    r = report[arm]
    print(f'{arm:15s}', {k: round(v, 4) for k, v in r.items()}, f' dNLL={r["nll"] - report["intact"]["nll"]:+.3f}', flush=True)
json.dump(report, open(fork / 'core_role.json', 'w'), indent=1)
