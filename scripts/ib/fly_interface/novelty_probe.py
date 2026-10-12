"""Does novelty reach dopamine, and does it habituate?  (The fly's alpha'3 compartment: MBON-alpha'3 responds to novel
odours and habituates with repetition through its own dopamine; that is a biological form of the epistemic value.)
Fixed weights except the online compartment memory (it runs, as in life).  Text streams at mean 8 ticks/word.
For 3 new odours (8 receptor types, whiffs p = 0.5 at 32 rheobase for 24 ticks, no outcome), each presented 8 times
with 100-tick gaps: per presentation, the extra firing (odour window minus the matched window before it) of PPL104
(alpha'3 dopamine), all PAM, all PPL1, MBON16/17 (alpha'3), all MBONs and the Kenyon cells, and the change of phi on
the alpha'3 edges.  Habituation = slope over presentations.
Usage: python novelty_probe.py <checkpoint dir> [out .jsonl]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core import fly_odour_world as ow

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/published/fly_consequence_20261012/novelty_probe.jsonl')
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']; ann_path = graph.parent / 'body-annotations.feather'
if getattr(model, 'dopamine_sensitivity', None) is None:
    fl.attach_dopamine_modulation(model, site='pre')
if getattr(model, 'comp_pre', None) is None:
    fl.attach_compartment_memory(model, graph, ann_path, eta=-2.0, forget=0.999)
model.eval().requires_grad_(False)
world = ow.OdourWorld(model, graph, ann_path, seed=11)
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
dan = np.zeros(len(typ), bool); dan[g['dan_indices']] = True
groups = {'PPL104': typ == 'PPL104', 'PAM': np.char.startswith(typ.astype(str), 'PAM') & dan,
          'PPL1': np.char.startswith(typ.astype(str), 'PPL1') & dan,
          'MBON_a3': np.isin(typ, ['MBON16', 'MBON17', 'MBON17-like']), 'MBON': cls == 'MBON', 'KC': cls == 'Kenyon_Cell'}
gm = {k: torch.as_tensor(v, device='cuda') for k, v in groups.items()}
a3_edges = torch.as_tensor(groups['MBON_a3'], device='cuda')[model.comp_post.long()]
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons; writer = model.topographic_writer; quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); text = np.array(val[50_000:70_000], dtype=np.int64)
rng = np.random.default_rng(5)
st = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0, 'left': 0}


def tick(drive=None):
    s = st['s']; model.external_drive = drive
    if st['left'] == 0:
        st['left'] = int(rng.geometric(1 / 8))
        tok = torch.tensor([int(text[st['i'] % len(text)])], device='cuda'); st['i'] += 1
        src, base = fl.write_input(model, s, tok)
        s = fl.step_fly_physical_tick(model, s, tok, src, base, opts, base_rates=opts['base_rates'])
    else:
        base = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, base, opts, base_rates=opts['base_rates'])
    st['left'] -= 1; model.external_drive = None; st['s'] = s
    return s, (s.ring[0][0] > 0).float()


def window(odour_cells=None, ticks=24):
    acc = torch.zeros(N, device='cuda')
    for t in range(ticks):
        d = None
        if odour_cells is not None and rng.random() < 0.5:
            d = torch.zeros(1, N, device='cuda'); d[0, odour_cells] = 32.0 * world.rheobase[odour_cells]
        acc += tick(d)[1]
    return {k: float(acc[m].mean() / ticks) for k, m in gm.items()}


rep = {'checkpoint': str(fork), 'odours': []}
with torch.no_grad():
    for _ in range(500):
        tick()
    for o in range(3):
        types = list(rng.choice(world.pool, 8, replace=False))
        cells = torch.as_tensor(np.concatenate([world.receptor[x] for x in types]), device='cuda')
        rows = []
        for k in range(8):
            for _ in range(76):
                tick()
            before = window()
            phi0 = st['s'].comp_memory.clone() if st['s'].comp_memory.numel() else None
            during = window(cells)
            row = {k2: during[k2] - before[k2] for k2 in during}
            if phi0 is not None:
                row['dphi_a3'] = float((st['s'].comp_memory - phi0)[0, a3_edges].mean())
            rows.append(row)
        slope = {k2: float(np.polyfit(np.arange(8), [r[k2] for r in rows], 1)[0]) for k2 in rows[0]}
        rep['odours'].append({'types': types, 'per_presentation': rows, 'slope': slope})
        print('odour', o, 'first', {k2: round(v, 4) for k2, v in rows[0].items()}, 'slope', {k2: round(v, 5) for k2, v in slope.items()}, flush=True)
open(out, 'a').write(json.dumps(rep) + chr(10))
print('saved', out)
