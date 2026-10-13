"""Where does a group's synaptic drive come from while reading?  Zero-shot, fixed weights (loader env interventions
apply).  For the target class or type (argv[2], e.g. ALPN or type:APL), every excitatory and inhibitory edge onto it
delivers w x transmitted pulse of its presynaptic cell per tick (an accounting of input, delays ignored); summed per
presynaptic class, per target cell and tick, relative to the target's threshold.  Reading: 4 places x 64 words,
the world's ticks per word.  Prints the top excitatory and inhibitory source classes.
Usage: python drive_by_class_probe.py <checkpoint dir> <class | type:TYPE>
"""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1]); target = sys.argv[2]
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))     # the world holds a word on the senses for all its ticks
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
sc = np.array(graph['superclass_names'])[graph['superclass_id']].astype(str)
label = np.where(cls != 'None', cls, np.char.add('sc:', sc))
import os
if os.environ.get('BY_TYPE'):
    label = np.char.add(np.char.add(typ.astype(str), '|'), label.astype(str))
tgt = (typ == target[5:]) if target.startswith('type:') else (cls == target)
T = torch.as_tensor(tgt, device='cuda')
names = np.unique(label); lab_id = torch.as_tensor(np.searchsorted(names, label), device='cuda')
thr = opts['thresholds'].detach().expand(1, model.n_neurons)[0]
acc = {}
for sfx in ('e', 'i'):
    pre, post = getattr(model, f'edge_pre_{sfx}').long(), getattr(model, f'edge_post_{sfx}').long()
    on = T[post]
    acc[sfx] = dict(pre=pre[on], w=getattr(model, f'edge_weight_{sfx}').detach()[on].abs() / thr[post[on]],
                    lab=lab_id[pre[on]], total=torch.zeros(len(names), device='cuda'))
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8); ticks = 0
with torch.no_grad():
    for place in range(4):
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        for i, w in enumerate(np.array(val[90_000 + place * 3_000:90_000 + place * 3_000 + 64], dtype=np.int64)):
            token = torch.tensor([int(w)], device='cuda')
            for t in range(TICKS):
                if t == 0 or HOLD:
                    dr, baseline = fl.write_input(model, s, token)
                    s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
                else:
                    baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                    s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
                if i >= 16:
                    p = s.ring[0][0]
                    for a in acc.values():
                        a['total'].index_add_(0, a['lab'], p[a['pre']] * a['w'])
                    ticks += 1
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
n = int(tgt.sum())
for sfx, nm in (('e', 'excitatory'), ('i', 'inhibitory')):
    tot = (acc[sfx]['total'] / n / ticks).cpu().numpy()
    order = np.argsort(-tot)[:8]
    print(f'{target} ({n} cells) {nm} drive per cell per tick / threshold: total {tot.sum():.3f} | ' +
          ', '.join(f'{names[j]} {tot[j]:.3f}' for j in order if tot[j] > 0), flush=True)
