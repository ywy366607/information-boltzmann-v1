"""Giant fiber (DNp01 L/R) in the champion: firing, membrane relative to threshold, timing inside a token."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state
import pandas as pd
ROOT = Path('D:/information-boltzmann-v1')
saved = torch.load('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
print('surrogate mode', cfg.get('surrogate_mode'), 'detach_reset', cfg.get('detach_reset'))
model = FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                       read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                       decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                       surrogate_mode=cfg.get('surrogate_mode', 'absolute'), transmission_mode=cfg.get('transmission_mode', 'atomic'))
edges = ('edge_weight_e', 'edge_weight_i')
model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
model = model.cuda().eval().requires_grad_(False)
for k in edges: dict(model.named_buffers())[k].copy_(saved['model'][k])
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
g = np.load(ROOT / cfg['graph']); ids = g['neuron_body_ids']
gf = [int(np.flatnonzero(ids == b)[0]) for b in (10001, 10010)]
thr = opts['thresholds'][0, gf]
print('GF indices', gf, 'thresholds', thr.tolist())
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); cur = int(saved['val_cursor'])
toks = np.array(val[cur:cur + 600], dtype=np.int64)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
spk = {k: [] for k in range(1, 15)}; hs = []
with torch.no_grad():
    for i, tok in enumerate(toks):
        s, ticks = fl.advance_fly_input_event(model, s, torch.tensor([int(tok)], device='cuda'), settle_ticks=13,
                                              writer_baseline_clock='physical', return_ticks=True, **opts)
        if i >= 100:
            for k, st in enumerate(ticks, start=1):
                spk[k].append((st.ring[0][0, gf] > 0).float().cpu().numpy())
                hs.append((st.h[0, gf] / thr).cpu().numpy())
hs = np.array(hs)
print('GF firing probability per tick of a 14-tick token window (L, R):')
print({k: np.round(np.mean(v, 0), 3).tolist() for k, v in spk.items()})
print('membrane / threshold percentiles (L, R):', np.round(np.percentile(hs, [1, 25, 50, 75, 99, 100], axis=0), 3).tolist())
