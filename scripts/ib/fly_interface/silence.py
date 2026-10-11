"""Does the champion's activity sustain itself without any input?  From its saved never-reset state, run quiet ticks
(zero sensory drive, writer baseline decaying on the physical clock) and record firing, membrane energy, conductances
and how much the motor read is still changing."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state
ROOT = Path('D:/information-boltzmann-v1')
saved = torch.load('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                       read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                       decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                       surrogate_mode=cfg.get('surrogate_mode', 'absolute'), transmission_mode=cfg.get('transmission_mode', 'atomic'))
edges = ('edge_weight_e', 'edge_weight_i')
model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
model = model.cuda().eval().requires_grad_(False)
bufs = dict(model.named_buffers())
for k in edges: bufs[k].copy_(saved['model'][k])
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
sc = model.superclass_id if hasattr(model, 'superclass_id') else None
rows, prev_z = [], fl.read_fly_motor(model, s)
quiet = torch.zeros_like(s.h)
with torch.no_grad():
    for t in range(1, 5001):
        s = fl.step_fly_physical_tick(model, s, None, quiet, model.topographic_writer.lambda_adapt * s.baseline, opts)
        if t in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000):
            z = fl.read_fly_motor(model, s)
            spikes = (s.ring[0] > 0).float()
            rows.append({'tick': t, 'firing_rate': float(spikes.mean()), 'spiking_neurons': int(spikes.sum()),
                         'membrane_energy': float(s.h.square().mean()), 'ge_mean': float(s.ge.mean()), 'gi_mean': float(s.gi.mean()),
                         'motor_read_change': float((z - prev_z).norm() / (prev_z.norm() + 1e-12)),
                         'sensory_baseline_norm': float(s.baseline.norm())})
            prev_z = z
            print(rows[-1], flush=True)
json.dump(rows, open(ROOT / 'scratch/fly_stream/silence.json', 'w'), indent=1)
# --- what the self-sustained activity looks like (continuing the same silent life) ---
import collections
names = [str(x) for x in np.load(ROOT / cfg['graph'])['superclass_names']]
sc = torch.as_tensor(np.load(ROOT / cfg['graph'])['superclass_id'], device='cuda').long()
sets, counts = [], torch.zeros(model.n_neurons, device='cuda')
with torch.no_grad():
    for t in range(2000):
        s = fl.step_fly_physical_tick(model, s, None, quiet, model.topographic_writer.lambda_adapt * s.baseline, opts)
        spk = (s.ring[0] > 0).flatten()
        counts += spk.float()
        if t % 100 == 0 or t in (1, 2, 3):
            sets.append((t, spk.clone()))
def jac(a, b): return float((a & b).sum() / (a | b).sum().clamp_min(1))
base = sets[0][1]
print('Jaccard of spiking set vs first sample:', [(t, round(jac(base, x), 3)) for t, x in sets[1:8]])
print('Jaccard tick-to-tick (t, t+1):', round(jac(sets[1][1], sets[2][1]), 3))
ever = counts > 0
rate = counts / 2000
print('neurons that fired at least once in 2000 silent ticks:', int(ever.sum()), f'({100 * float(ever.float().mean()):.1f}%)')
print('neurons firing >50% of ticks (tonic):', int((rate > .5).sum()), ' 1-50%:', int(((rate > .01) & (rate <= .5)).sum()),
      ' sparse <=1%:', int(((rate > 0) & (rate <= .01)).sum()))
by = collections.Counter()
for k in torch.unique(sc[ever]).tolist():
    by[names[k]] = (int((ever & (sc == k)).sum()), int((sc == k).sum()))
print('active / total by superclass:', {k: v for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:12]})
