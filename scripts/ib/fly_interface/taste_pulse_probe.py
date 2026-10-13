"""Before closing the consequence loop: does tasting reach the dopamine cells in ACTIVITY, and with valence?  (The
static check only says a weighted path exists.)  Receptor sets from Vicente et al. 2025 (maleCNS typing): sugar =
LB3b, LB3c, dorsal_tpGRN; bitter = LB1a-d; water = LB3a; high salt = LB3d.  Fixed weights.  From the physical state
after reading (64 words, 8 ticks each, at 6 places in the validation text), fork: control (quiet ticks) vs stimulus
(the set driven with a tonic current of `gain` x its own thresholds for 20 ticks, then 20 quiet ticks).  Per fork the
extra spikes (stimulus - control) of: the driven receptors themselves, PAM, PPL1, every dopamine type (top 6), the
MBONs, Kenyon cells; and the change of the compartment memory phi.  Reports [min | median | mean | max] over places.
Usage: python taste_pulse_probe.py <checkpoint dir> [out.json] [gain]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'taste_pulse.json'
GAIN = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
model, cfg, L = load(fork)
import os
if os.environ.get('INTRINSIC_KAPPA') and getattr(model, 'intrinsic_kappa', None) is None:
    # zero-shot: release the dopamine cells' own tonic activity (attach_intrinsic_drive, the fly's spontaneously
    # active types; here the dopaminergic PAM / PPL only), kappa in units of their rheobase
    fl.attach_intrinsic_drive(model, ROOT / cfg['graph'], ROOT / 'data/malecns_v1/body-annotations.feather',
                              kappa=float(os.environ['INTRINSIC_KAPPA']), pattern=r'^(PAM\d|PPL\d)')
    print('intrinsic drive on PAM / PPL, kappa', os.environ['INTRINSIC_KAPPA'], flush=True)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
dan = np.zeros(len(typ), bool); dan[graph['dan_indices']] = True
sets = {'sugar': np.isin(typ, ['LB3b', 'LB3c', 'dorsal_tpGRN']), 'bitter': np.isin(typ, ['LB1a', 'LB1b', 'LB1c', 'LB1d']),
        'water': np.isin(typ, ['LB3a']), 'high_salt': np.isin(typ, ['LB3d'])}
groups = {'PAM': np.char.startswith(typ.astype(str), 'PAM') & dan, 'PPL1': np.char.startswith(typ.astype(str), 'PPL1') & dan,
          'DAN_all': dan, 'MBON': cls == 'MBON', 'KC': cls == 'Kenyon_Cell'}
dan_types = [t for t in np.unique(typ[dan]) if t not in ('None', 'nan')]
G = {k: torch.as_tensor(v, device='cuda') for k, v in {**groups, **{f't:{t}': dan & (typ == t) for t in dan_types}}.items()}
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
thr = opts['thresholds'].detach()
if thr.dim() == 1:
    thr = thr.unsqueeze(0)


def tick(s, drive):
    model.external_drive = drive
    baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
    s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
    model.external_drive = None
    return s


# where does it stop?  the hop-1 relays of each receptor set (its 30 strongest excitatory targets, birth weights) and
# the dopamine cells' membranes (sub-threshold depolarisation, relative to their thresholds)
pre_e, post_e, w_e = graph['edge_pre_e'], graph['edge_post_e'], graph['edge_weight_e']
relays = {}
for name, m in sets.items():
    sel = m[pre_e]
    tot = np.bincount(post_e[sel], weights=w_e[sel], minlength=len(typ))
    relays[name] = torch.as_tensor(np.argsort(-tot)[:30], device='cuda')
DANm = torch.as_tensor(dan, device='cuda')


def run(s0, drive, relay=None):
    s, counts = s0, {k: 0.0 for k in G}
    own = 0.0
    counts['relay'] = 0.0; counts['dan_v_over_theta'] = 0.0
    phi0 = s0.comp_memory.clone() if s0.comp_memory.numel() else None
    for k in range(40):
        s = tick(s, drive if (drive is not None and k < 20) else None)
        spk = (s.ring[0][0] > 0).float()
        for name, m in G.items():
            counts[name] += float(spk[m].sum())
        if relay is not None:
            counts['relay'] += float(spk[relay].sum())
        counts['dan_v_over_theta'] += float((s.h[0, DANm] / thr.expand(1, -1)[0, DANm]).mean()) / 40
        if drive is not None:
            own += float(spk[drive[0] > 0].sum())
    dphi = float((s.comp_memory - phi0).abs().mean()) if phi0 is not None else None
    return counts, own, dphi


rows = {name: [] for name in sets}
with torch.no_grad():
    for place in range(6):
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        seq = np.array(val[80_000 + place * 3_000:80_000 + place * 3_000 + 64], dtype=np.int64)
        for w in seq:
            token = torch.tensor([int(w)], device='cuda')
            for t in range(8):
                if t == 0:
                    drive, baseline = fl.write_input(model, s, token)
                    s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
                else:
                    s = tick(s, None)
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        for name, m in sets.items():
            base, _, dphi0 = run(s, None, relays[name])
            drive = torch.zeros(1, model.n_neurons, device='cuda')
            idx = torch.as_tensor(np.nonzero(m)[0], device='cuda')
            drive[0, idx] = GAIN * thr.expand(1, -1)[0, idx]
            stim, own, dphi = run(s, drive, relays[name])
            rows[name].append({'receptor_spikes_per_cell': own / max(1, int(m.sum())),
                               **{f'extra_{k}': stim[k] - base[k] for k in G},
                               'extra_relay_spikes': stim['relay'] - base['relay'],
                               'dan_v_over_theta_control': base['dan_v_over_theta'],
                               'extra_dan_v_over_theta': stim['dan_v_over_theta'] - base['dan_v_over_theta'],
                               'dphi_stim': dphi, 'dphi_control': dphi0})
q = lambda v: [float(np.min(v)), float(np.median(v)), float(np.mean(v)), float(np.max(v))]
rep = {'fork': str(fork), 'gain_x_threshold': GAIN, 'sizes': {k: int(v.sum()) for k, v in {**sets, **groups}.items()}}
for name, rs in rows.items():
    summ = {k: q([r[k] for r in rs]) for k in rs[0] if rs[0][k] is not None}
    top = sorted(((k[8:], summ[k][2]) for k in summ if k.startswith('extra_t:')), key=lambda z: -abs(z[1]))[:6]
    rep[name] = {'summary': {k: v for k, v in summ.items() if not k.startswith('extra_t:')}, 'top_dopamine_types_mean_extra': top}
    print(name, {k: [round(x, 2) for x in v] for k, v in rep[name]['summary'].items()}, 'top', [(t, round(v, 2)) for t, v in top], flush=True)
json.dump(rep, open(out, 'w'), indent=1)
