"""Are the mushroom body's ports deaf, or only silenced by their homeostatic offset?  Fixed weights, 12 positions.
Each group is driven for 15 quiet ticks with a current that first cancels its own homeostatic drive b (when b < 1)
and then adds k rheobase on top: extra = (max(0, 1 - b) + k) * rheobase, k in (1, 2, 5).  Per group and dose: the
driven cells' own rate, how many other neurons change their spike count, the change of logit(" yes") - logit(" no"),
and the change in dopamine-cell, Kenyon and MBON firing.  Groups: PAM-type MBONs, PPL1-type MBONs, PAM dopamine
cells, PPL1 dopamine cells, 5% of Kenyon cells (random), and 97 random core cells as a reference.
Read surface: change of its total spike count, against the spread of that count across positions.
Usage: python port_check.py <fork dir>
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

KS, TICKS, POSITIONS = (1.0, 2.0, 5.0), 150, 12     # 150 ticks >= 3 tau_m for nearly every cell
YES, NO = 3763, 645
fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
graph = ROOT / cfg['graph']
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls, sup = ann['type'].astype(str).values, ann['class'].astype(str).values, ann['superclass'].astype(str).values
pam, ppl1 = np.char.startswith(typ.astype(str), 'PAM'), np.char.startswith(typ.astype(str), 'PPL1')
is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
kc, mbon = cls == 'Kenyon_Cell', cls == 'MBON'
dpre, dpost, dw = g['dan_edge_pre'], g['dan_edge_post'], np.abs(g['dan_edge_weight']).astype(np.float64)
def da_in(src):
    e = src[dpre]
    return np.bincount(dpost[e], weights=dw[e], minlength=N)
fp, fq = da_in(pam), da_in(ppl1); tot = fp + fq
share = np.where(tot > 0, fp / np.maximum(tot, 1e-12), np.nan)
rng = np.random.default_rng(0)
core_np = (sup == 'cb_intrinsic') & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan
core_rand = np.zeros(N, bool); core_rand[rng.choice(np.flatnonzero(core_np), int(mbon.sum()), replace=False)] = True
kc_rand = np.zeros(N, bool); kc_rand[rng.choice(np.flatnonzero(kc), int(0.05 * kc.sum()), replace=False)] = True
groups = {'PAM-type MBONs': mbon & (share > 0.8), 'PPL1-type MBONs': mbon & (share < 0.2), 'PAM dopamine': pam & is_dan,
          'PPL1 dopamine': ppl1 & is_dan, 'Kenyon 5%': kc_rand, 'random core': core_rand}
T = lambda m: torch.as_tensor(m, device='cuda')
read = np.zeros(N, bool); read[model.read_indices.cpu().numpy()] = True
watch = {'dopamine': T(is_dan), 'Kenyon': T(kc), 'MBON': T(mbon), 'read surface': T(read)}

saved_b = L['physical'].get('homeostasis')
b = (saved_b.reshape(-1).float().cuda() if saved_b is not None and saved_b.numel() == N else torch.zeros(N, device='cuda'))
rates0 = model.get_decay_rates()[0].expand(1, N).detach()
rheobase = (model.get_thresholds().expand(1, N).detach() * -torch.log(rates0.clamp(1e-5, 1 - 1e-7)))[0]
offset = (1.0 - b).clamp_min(0.0)
box = {'extra': None, 'count': None}
original_step = fl.step_fly_physical_tick


def step(model_, current, token, source, *a, **k):
    if box['extra'] is not None:
        source = box['extra'] if source is None else source + box['extra']
    out = original_step(model_, current, token, source, *a, **k)
    if box['count'] is not None:
        box['count'] += (out.ring[0][0] > 0).float()
    return out


fl.step_fly_physical_tick = step
writer = model.topographic_writer
quiet = torch.zeros(1, N, device='cuda')


def ticks(s, n, mask=None, k=0.0):
    hb = getattr(s, 'homeostasis', torch.empty(0))
    off = (1.0 - hb.reshape(-1)).clamp_min(0.0) if hb.numel() == N else offset     # cancel the living state's own b
    box['extra'] = None if mask is None else (((off + k) * rheobase) * mask.float()).unsqueeze(0)
    box['count'] = torch.zeros(N, device='cuda')
    for _ in range(n):
        baseline = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
    c = box['count']; box.update(extra=None, count=None)
    return s, c


def yes_no(s):
    lg = model.decoder(model.read_norm(fl.read_fly_motor(model, s))).float().reshape(-1)
    return float(lg[YES] - lg[NO])


val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[70_000:71_000], dtype=np.int64)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
rows = []
torch.manual_seed(0)
with torch.no_grad():
    for tok in text[:128]:
        s, _, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=32,
                                                   writer_baseline_clock='physical', **opts)
    pos = 128
    for p in range(POSITIONS):
        for tok in text[pos:pos + 24]:
            s, _, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [], max_ticks=32,
                                                       writer_baseline_clock='physical', **opts)
        pos += 24
        s0, c0 = ticks(s, TICKS)
        y0 = yes_no(s0)
        row = {'yes_minus_no': y0, 'read surface spikes (none)': float(c0[watch['read surface']].sum())}
        for name, m in groups.items():
            mt = T(m)
            for k in KS:
                s1, c1 = ticks(s, TICKS, mt, k)
                key = f'{name}|k{k:g}'
                row[key + '|driven Hz'] = float(c1[mt].mean()) / TICKS * 1000
                row[key + '|neurons changed'] = float(((c1 - c0).abs() > 0).sum() - mt.sum())
                row[key + '|d yes-no'] = yes_no(s1) - y0
                row[key + '|d read surface spikes'] = float(c1[watch['read surface']].sum() - c0[watch['read surface']].sum())
                for wname, wm in watch.items():
                    rest = wm & ~mt
                    row[key + f'|{wname} dHz'] = float((c1[rest] - c0[rest]).mean()) / TICKS * 1000
        rows.append(row)
        print('position', p, {k: round(v, 3) for k, v in row.items() if 'k5' in k and ('Hz' in k or 'yes' in k)}, flush=True)
R = pd.DataFrame(rows)
nat = float(R['yes_minus_no'].std())
out = {'fork': str(fork), 'natural_sd_yes_minus_no': nat, 'natural_sd_read_surface_spikes': float(R['read surface spikes (none)'].std()),
       'b_median_by_group': {n: float(b[T(m)].median()) for n, m in groups.items()},
       'stats_min_med_mean_max': {c: [float(R[c].min()), float(R[c].median()), float(R[c].mean()), float(R[c].max())] for c in R.columns}}
json.dump(out, open(fork / 'port_check.json', 'w'), indent=1)
print('natural sd', nat, 'b median', out['b_median_by_group'], flush=True)
print('done', flush=True)
