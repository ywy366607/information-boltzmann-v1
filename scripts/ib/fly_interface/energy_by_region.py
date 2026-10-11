"""Same global firing (~2.5%) at work and at rest: same neurons or a redistribution?  Also: the firing time course
inside a token, and whether the BIRTH weights sustain activity without input."""
import sys, json, collections, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state
ROOT = Path('D:/information-boltzmann-v1')
saved = torch.load('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt', map_location='cpu', mmap=True, weights_only=False)
cfg, L = saved['config'], saved['learner']
def build():
    return FlyReservoirLM(ROOT / cfg['graph'], vocab_size=50257, d_model=cfg['d_model'], injection='topographic',
                          read_surface='output', synapse_model='coba', use_alif=True, use_stp=True,
                          decoder_bias=cfg['decoder_bias'], detach_reset=cfg.get('detach_reset', False),
                          surrogate_mode=cfg.get('surrogate_mode', 'absolute'), transmission_mode=cfg.get('transmission_mode', 'atomic'))
edges = ('edge_weight_e', 'edge_weight_i')
model = build()
model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False)
model = model.cuda().eval().requires_grad_(False)
bufs = dict(model.named_buffers())
for k in edges: bufs[k].copy_(saved['model'][k])
g = np.load(ROOT / cfg['graph'])
names = [str(x) for x in g['superclass_names']]
sc = torch.as_tensor(g['superclass_id'], device='cuda').long()
inj = model.topographic_writer.injection_index
groups = {'sensory (injection)': torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda').index_fill_(0, inj, True),
          'motor/descending (read)': torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda').index_fill_(0, model.read_indices, True)}
for n in ('cb_intrinsic', 'ol_intrinsic', 'vnc_intrinsic', 'visual_projection', 'ascending_neuron'):
    groups[n] = sc == names.index(n)
def opts(m):
    return dict(base_rates=m.get_decay_rates(), thresholds=m.get_thresholds(), conductance_gains=m.get_conductance_gains(),
                alif_params=m.get_alif_params(), stp_params=m.get_stp_params())
o = opts(model)
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
toks = np.array(val[int(saved['val_cursor']):int(saved['val_cursor']) + 400], dtype=np.int64)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
work = torch.zeros(model.n_neurons, device='cuda'); work_ticks = 0
course = collections.defaultdict(list)
with torch.no_grad():
    for i, tok in enumerate(toks):
        s, ticks = fl.advance_fly_input_event(model, s, torch.tensor([int(tok)], device='cuda'), settle_ticks=6,
                                              writer_baseline_clock='physical', return_ticks=True, **o)
        if i >= 100:
            for j, st in enumerate(ticks):
                spk = (st.ring[0] > 0).flatten().float()
                work += spk; work_ticks += 1
                course[j].append(float(spk.mean()))
    for _ in range(300):                                   # wash out the input
        s = fl.step_fly_physical_tick(model, s, None, torch.zeros_like(s.h), model.topographic_writer.lambda_adapt * s.baseline, o)
    rest = torch.zeros(model.n_neurons, device='cuda'); rest_ticks = 0
    for _ in range(2100):
        s = fl.step_fly_physical_tick(model, s, None, torch.zeros_like(s.h), model.topographic_writer.lambda_adapt * s.baseline, o)
        rest += (s.ring[0] > 0).flatten().float(); rest_ticks += 1
wr, rr = work / work_ticks, rest / rest_ticks
pre = torch.cat([model.edge_pre_e.long(), model.edge_pre_i.long()])
outdeg = torch.bincount(pre, minlength=model.n_neurons).float()
E = outdeg.sum()
ew, er = (wr * outdeg).sum(), (rr * outdeg).sum()           # synaptic events per tick
print(f'energy = spikes x outgoing synapses: work {float(ew):.0f} events/tick ({100 * float(ew / E):.2f}% of all synapses), '
      f'rest {float(er):.0f} ({100 * float(er / E):.2f}%), work/rest {float(ew / er):.3f}')
print(f'{"group":26s} {"synapses":>9s} {"share%":>7s} {"work ev/tick":>12s} {"rest ev/tick":>12s} {"ratio":>6s} {"share of energy w/r":>20s}')
for name, mask in groups.items():
    a, b = float((wr * outdeg)[mask].sum()), float((rr * outdeg)[mask].sum())
    print(f'{name:26s} {int(outdeg[mask].sum()):9d} {100 * float(outdeg[mask].sum() / E):7.2f} {a:12.0f} {b:12.0f} '
          f'{a / max(b, 1e-9):6.2f} {100 * a / float(ew):9.1f}% / {100 * b / float(er):5.1f}%')
rest_mask = ~torch.stack(list(groups.values())).any(0)
a, b = float((wr * outdeg)[rest_mask].sum()), float((rr * outdeg)[rest_mask].sum())
print(f'{"all other neurons":26s} {int(outdeg[rest_mask].sum()):9d} {100 * float(outdeg[rest_mask].sum() / E):7.2f} {a:12.0f} {b:12.0f} {a / max(b, 1e-9):6.2f} {100 * a / float(ew):9.1f}% / {100 * b / float(er):5.1f}%')
print(f'global firing: work {100 * float(wr.mean()):.2f}%  rest {100 * float(rr.mean()):.2f}%')
print('firing inside a token (tick 1 = input):', {j + 1: round(100 * np.mean(v), 2) for j, v in sorted(course.items())})
print(f'{"group":26s} {"n":>6s} {"work %":>7s} {"rest %":>7s} {"ratio":>6s}')
for name, mask in groups.items():
    w, r = 100 * float(wr[mask].mean()), 100 * float(rr[mask].mean())
    print(f'{name:26s} {int(mask.sum()):6d} {w:7.2f} {r:7.2f} {w / max(r, 1e-9):6.2f}')
aw, ar = wr > 0, rr > 0
print(f'active sets: work {int(aw.sum())}, rest {int(ar.sum())}, both {int((aw & ar).sum())}, Jaccard {float((aw & ar).sum() / (aw | ar).sum()):.3f}')
both = aw | ar
print(f'per-neuron rate correlation work vs rest (neurons active in either): {float(torch.corrcoef(torch.stack([wr[both], rr[both]]))[0, 1]):.3f}')
top_w = set(torch.topk(wr, 1000).indices.tolist()); top_r = set(torch.topk(rr, 1000).indices.tolist())
print(f'overlap of the 1000 most active neurons at work and at rest: {len(top_w & top_r)}')

