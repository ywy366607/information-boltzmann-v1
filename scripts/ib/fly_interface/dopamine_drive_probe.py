"""Where does the dopamine cells' drive come from, in activity?  Zero-shot, fixed weights (loader env interventions
apply).  For every excitatory edge onto a dopamine cell (pinned edges carry 0), the drive it delivers per tick is
w * transmitted pulse of its presynaptic cell (delays and conductance dynamics ignored: an accounting of input, not a
simulation).  Summed per source group: the taste relays (every first-order target of any GRN), MBONs, other
dopamine cells, the read cells, and the rest; and per dopamine group (PAM, PPL1).  Conditions from the same reading
state (6 places): reading on (8 ticks per word, 64 words), quiet, and each taste set driven (1.5 x threshold) for 200
ticks.  Reports the mean drive per cell per tick by source, relative to the cell's threshold, and the taste-specific
change.  Usage: python dopamine_drive_probe.py <checkpoint dir> [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'dopamine_drive.json'
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values.astype(str), ann['class'].astype(str).values.astype(str)
N = len(typ)
dan = np.zeros(N, bool); dan[graph['dan_indices']] = True
pam, ppl1 = dan & np.char.startswith(typ, 'PAM'), dan & np.char.startswith(typ, 'PPL1')
pre_e, post_e = model.edge_pre_e.long(), model.edge_post_e.long()
grn = cls == 'gustatory'
gp = graph['edge_pre_e']; gq = graph['edge_post_e']
relay = np.zeros(N, bool); relay[np.unique(gq[grn[gp]])] = True
relay &= ~dan
read = np.zeros(N, bool); read[model.read_indices.cpu().numpy()] = True
sources = {'taste_relays': relay, 'MBON': cls == 'MBON', 'dopamine': dan, 'read_cells': read & ~relay}
sources['rest'] = ~(sources['taste_relays'] | sources['MBON'] | sources['dopamine'] | sources['read_cells'])
onto = torch.as_tensor(dan, device='cuda')[post_e]
E_pre, E_post = pre_e[onto], post_e[onto]
ge = float(model.get_conductance_gains()[0].flatten()[0]) if hasattr(model, 'get_conductance_gains') else 1.0
W = model.edge_weight_e.detach()[onto].abs() * ge
src_of = {k: torch.as_tensor(v, device='cuda')[E_pre] for k, v in sources.items()}
thr = opts['thresholds'].detach().expand(1, N)[0]
tgt = {'PAM': torch.as_tensor(pam, device='cuda'), 'PPL1': torch.as_tensor(ppl1, device='cuda')}
sets = {'sugar': np.isin(typ, ['LB3b', 'LB3c', 'dorsal_tpGRN']), 'bitter': np.isin(typ, ['LB1a', 'LB1b', 'LB1c', 'LB1d'])}
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')


def drive_now(s):
    pulse = s.ring[0][0][E_pre] * W                                    # per edge onto dopamine cells
    res = {}
    for g, tm in tgt.items():
        n = int(tm.sum())
        for k, sm in src_of.items():
            d = torch.zeros(N, device='cuda').index_add_(0, E_post[sm], pulse[sm])
            res[f'{g}<-{k}'] = float((d[tm] / thr[tm]).sum() / n)
    return res


def tick(s, drive=None, token=None):
    model.external_drive = drive
    if token is not None:
        dr, baseline = fl.write_input(model, s, token)
        s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
    else:
        baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
    model.external_drive = None
    return s


def avg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


res = {c: [] for c in ('reading', 'quiet', 'sugar', 'bitter')}
with torch.no_grad():
    for place in range(6):
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        seq = np.array(val[80_000 + place * 3_000:80_000 + place * 3_000 + 64], dtype=np.int64)
        rows = []
        for w in seq:
            token = torch.tensor([int(w)], device='cuda')
            for t in range(8):
                s = tick(s, token=token if t == 0 else None)
                rows.append(drive_now(s))
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        res['reading'].append(avg(rows[-256:]))
        for cond in ('quiet', 'sugar', 'bitter'):
            drive = None
            if cond != 'quiet':
                drive = torch.zeros(1, N, device='cuda')
                idx = torch.as_tensor(np.nonzero(sets[cond])[0], device='cuda')
                drive[0, idx] = 1.5 * thr[idx]
            x, rows = s, []
            for k in range(200):
                x = tick(x, drive)
                rows.append(drive_now(x))
            res[cond].append(avg(rows[50:]))                        # after the first 50 ticks
rep = {'fork': str(fork), 'note': 'mean drive per cell per tick / threshold, by target group <- source group'}
for c, rows in res.items():
    rep[c] = avg(rows)
for k in rep['quiet']:
    rep.setdefault('taste_specific', {})[k] = {t: rep[t][k] - rep['quiet'][k] for t in ('sugar', 'bitter')}
for c in ('reading', 'quiet', 'sugar', 'bitter'):
    print(c, {k: round(v, 4) for k, v in rep[c].items()}, flush=True)
print('taste - quiet', {k: {t: round(v, 4) for t, v in d.items()} for k, d in rep['taste_specific'].items()})
json.dump(rep, open(out, 'w'), indent=1)
