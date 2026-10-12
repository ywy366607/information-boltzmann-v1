"""Population bursts in any individual, on a fixed clock.  Fixed weights.  Reads text with TPT ticks per token (the
token written at the first tick), records per tick the fraction of cells spiking in Kenyon cells, ALPNs, MBONs, core
and the whole brain; reports per group the median, p99, p99/median, peak, and the number of burst onsets (Kenyon
fraction > 1% after >= 50 quiet ticks) with their intervals.  A weights-only checkpoint (best.pt) borrows the physical
state of a full checkpoint (state_from) and burns in for BURN ticks first.
Usage: python burst_simple.py <checkpoint .pt> <ticks per token> [state_from .pt] [ticks]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
ckpt, TPT = Path(sys.argv[1]), int(sys.argv[2])
STATE_FROM = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '-' else None
TOTAL = int(sys.argv[4]) if len(sys.argv) > 4 else 6144
BURN = 2000
if ckpt.name == 'last.pt':
    model, cfg, L = load(ckpt.parent)
else:                                       # weights only: borrow a full checkpoint's state, written to a temporary copy
    tmp = Path('scratch/fly_stream/_burst_tmp'); tmp.mkdir(exist_ok=True)
    saved = torch.load(ckpt, map_location='cpu', mmap=True, weights_only=False)
    full = torch.load(STATE_FROM, map_location='cpu', mmap=True, weights_only=False)
    torch.save({**full, 'model': saved['model']}, tmp / 'last.pt')
    del saved, full
    model, cfg, L = load(tmp)
import os
if os.environ.get('NO_HOMEOSTASIS'):
    # the birth without any homeostasis (the adult lineage's own birth): drop the rule and its state
    for name in ('homeostasis_target', 'regime_quantile', 'regime_eta_b', 'regime_eta_s', 'regime_window_rate'):
        model._buffers.pop(name, None)
    L = {**L, 'physical': {**L['physical'], 'homeostasis': torch.empty(0), 'regime': torch.empty(0)}}
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons; graph = ROOT / cfg['graph']; g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
core = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan
groups = {'Kenyon': cls == 'Kenyon_Cell', 'ALPN': cls == 'ALPN', 'MBON': cls == 'MBON', 'core': core, 'all': np.ones(N, bool)}
gm = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
writer = model.topographic_writer; quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); text = np.array(val[90_000:110_000], dtype=np.int64)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
rows, t, i = [], 0, 0
with torch.no_grad():
    while t < BURN + TOTAL:
        tok = torch.tensor([int(text[i])], device='cuda'); i += 1
        for k in range(TPT):
            if k == 0:
                drive, baseline = fl.write_input(model, s, tok)
                s = fl.step_fly_physical_tick(model, s, tok, drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            if t >= BURN:
                s0 = s.ring[0][0] > 0
                rows.append([float(s0[m].float().mean()) for m in gm.values()])
            t += 1
X = np.array(rows)
rep = {'checkpoint': str(ckpt), 'ticks_per_token': TPT, 'ticks': len(X), 'no_homeostasis': bool(os.environ.get('NO_HOMEOSTASIS'))}
for j, k in enumerate(groups):
    x = X[:, j]; p50, p99 = float(np.median(x)), float(np.percentile(x, 99))
    rep[k] = {'mean': float(x.mean()), 'median': p50, 'p99': p99, 'p99_over_median': p99 / max(p50, 1e-6), 'peak': float(x.max())}
kc = X[:, 0]
on = [u for u in range(50, len(kc)) if kc[u] > 0.01 and kc[u - 50:u].max() <= 0.01]
rep['kc_burst_onsets'] = len(on); rep['kc_burst_intervals'] = np.diff(on).tolist()
print(json.dumps({k: (({a: round(b, 5) for a, b in v.items()}) if isinstance(v, dict) else v) for k, v in rep.items()}), flush=True)
out = Path('scratch/fly_stream/burst_simple.jsonl'); open(out, 'a').write(json.dumps(rep) + '\n')
