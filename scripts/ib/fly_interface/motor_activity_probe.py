"""How active are the output (motor / descending / efferent) cells the acts would be read from?  Fixed weights, the
checkpoint's own state, 256 validation words, pulse input, 8 ticks per word.  Reports: number of output cells, the
fraction of them spiking per tick, per-cell firing probability [min | median | mean | max], cells that never fire,
and how much of the across-word variance of the spike pattern vs the membranes is word-locked (the signal an act could
use).  Usage: python motor_activity_probe.py <checkpoint dir>
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
quiet = torch.zeros(1, model.n_neurons, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
words = np.array(val[60_000:60_256], dtype=np.int64)
idx = model.read_indices
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
spikes, membranes = [], []
with torch.no_grad():
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        for t in range(8):
            if t == 0:
                drive, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                s = fl.step_fly_physical_tick(model, s, None, quiet, writer.lambda_adapt * s.baseline, opts, base_rates=opts['base_rates'])
            spikes.append((s.ring[0][0, idx] > 0).float().cpu().numpy())
            membranes.append(s.h[0, idx].float().cpu().numpy())
S, H = np.array(spikes), np.array(membranes)                    # [ticks, cells]
p = S.mean(0)
d = lambda x: [float(np.min(x)), float(np.median(x)), float(np.mean(x)), float(np.max(x))]
per_tick = S.mean(1)
# word-locked share: variance of the per-word mean pattern over the total variance (summed over cells)
Sw, Hw = S.reshape(len(words), 8, -1).mean(1), H.reshape(len(words), 8, -1).mean(1)
share = lambda X, Xw: float(Xw.var(0).sum() / max(X.var(0).sum(), 1e-12))
rep = {'checkpoint': str(fork), 'output_cells': int(len(idx)), 'fraction_spiking_per_tick': d(per_tick),
       'cell_fire_prob': d(p), 'cells_never_firing': int((p == 0).sum()), 'cells_firing_over_10pct': int((p > 0.1).sum()),
       'word_locked_share_spikes': share(S, Sw), 'word_locked_share_membranes': share(H, Hw)}
print(json.dumps(rep, indent=1))
