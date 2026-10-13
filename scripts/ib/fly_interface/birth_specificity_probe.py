import os, sys, numpy as np, torch, pandas as pd
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core import fly_bptt_learning as fl
EE, EI = float(os.environ.get('EE', 0.55)), float(os.environ.get('EI', -0.15))
G = 'D:/information-boltzmann-v1/data/malecns_v1/fly_reservoir_coba_syn.npz'; W0 = 0.000979
g = np.load(G, allow_pickle=True); N = len(g['neuron_body_ids'])
ann = pd.read_feather('D:/information-boltzmann-v1/data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values.astype(str), ann['class'].astype(str).values.astype(str)
sc = np.array(g['superclass_names'])[g['superclass_id']]
dan = np.zeros(N, bool); dan[g['dan_indices']] = True
model = FlyReservoirLM(G, d_model=768, read_surface='output', injection='topographic', synapse_model='coba', use_alif=True, use_stp=True,
                       detach_reset=True, surrogate_mode='threshold', transmission_mode='incoming', E_E=EE, E_I=EI).cuda().eval().requires_grad_(False)
import math
if os.environ.get('STP_U0'):
    with torch.no_grad():
        model.logit_u0.fill_(math.log(float(os.environ['STP_U0']) / (1 - float(os.environ['STP_U0']))))
if os.environ.get('STP_TAU_REC'):
    with torch.no_grad():
        model.log_tau_rec.fill_(math.log(float(os.environ['STP_TAU_REC'])))
if os.environ.get('ALIF_OFF'):
    with torch.no_grad():
        model.log_beta.fill_(-30.0)          # adaptation strength ~0
BGS = tuple(float(x) for x in os.environ.get('BGS', '0.0,0.01,0.05').split(','))
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
thr = model.get_thresholds().detach().expand(1, N)[0]; z = torch.zeros(1, N, device='cuda')
fresh = lambda: fl.FlyPhysicalState(h=z.clone(), ring=tuple(z.clone() for _ in range(4)), ge=z.clone(), gi=z.clone(), b=z.clone(),
                                    x=torch.ones_like(z), u=opts['stp_params'][0].detach().expand(1, N).clone(), baseline=z.clone())
# calibrate: peak PSP per synapse -> 0.617% of threshold
pre_e, post_e, w_e = g['edge_pre_e'], g['edge_post_e'], np.abs(g['edge_weight_e'])
rat = []
with torch.no_grad():
    for j in np.random.default_rng(1).choice(np.unique(pre_e), 20, replace=False):
        s = fresh(); kick = z.clone(); kick[0, j] = 10 * float(thr[j]); pk = torch.full((N,), -1e9, device='cuda')
        for t in range(40):
            model.external_drive = kick if t == 0 else None
            s = fl.step_fly_physical_tick(model, s, None, z, s.baseline, opts, base_rates=opts['base_rates']); pk = torch.maximum(pk, s.h[0])
        model.external_drive = None
        m = pre_e == j; tg = torch.as_tensor(post_e[m], device='cuda'); p = (pk[tg] / thr[tg]).cpu().numpy(); c = w_e[m] / W0
        ok = (p > 0) & (p < 0.9); rat += list(p[ok] / c[ok])
    f = 0.00617 / np.median(rat)
    for sfx in ('e', 'i'):
        getattr(model, f'edge_weight_{sfx}').mul_(f)
print(f'E_E {EE} E_I {EI}: weights x{f:.3f} to keep the PSP at 0.617% of threshold', flush=True)
quiet = z
sens = torch.as_tensor(np.nonzero(np.char.find(sc.astype(str), 'sensory') >= 0)[0], device='cuda')
S = {'sugar': np.isin(typ, ['LB3b', 'LB3c', 'dorsal_tpGRN']), 'bitter': np.isin(typ, ['LB1a', 'LB1b', 'LB1c', 'LB1d'])}
Si = {k: torch.as_tensor(np.nonzero(v)[0], device='cuda') for k, v in S.items()}
groups = {'PAM': dan & np.char.startswith(typ, 'PAM'), 'PPL1': dan & np.char.startswith(typ, 'PPL1'), 'MBON': cls == 'MBON', 'KC': cls == 'Kenyon_Cell',
          'motor': np.isin(sc, ['cb_motor', 'vnc_motor']), 'OA-VPM3': typ == 'OA-VPM3', 'OA-VPM4': typ == 'OA-VPM4', 'OA_all': np.char.startswith(typ, 'OA-'), 'PAM01': typ == 'PAM01', 'PPL101': typ == 'PPL101', 'PPL102': typ == 'PPL102', 'PPL106': typ == 'PPL106'}
def hops(src, k=3):
    seen = src.copy(); fr = src.copy(); out = []
    for _ in range(k):
        nx = np.zeros(N, bool); nx[post_e[fr[pre_e]]] = True; nx &= ~seen; out.append(nx); seen |= nx; fr = nx
    return out
H = {k: hops(v) for k, v in S.items()}
def run(s, T, bg, stim=None, rate=0.15, seed=1):
    gen = torch.Generator(device='cuda').manual_seed(seed); c = torch.zeros(N, device='cuda')
    for _ in range(T):
        d = torch.zeros(1, N, device='cuda'); on = sens[torch.rand(sens.numel(), device='cuda', generator=gen) < bg]; d[0, on] = 1.5 * thr[on]
        r = torch.rand(N, device='cuda', generator=gen)
        if stim is not None:
            hit = Si[stim][r[Si[stim]] < rate]; d[0, hit] = 1.5 * thr[hit]
        model.external_drive = d
        s = fl.step_fly_physical_tick(model, s, None, quiet, s.baseline, opts, base_rates=opts['base_rates']); c += (s.ring[0][0] > 0).float()
    model.external_drive = None
    return s, c.cpu().numpy()
with torch.no_grad():
    print('ALIF off' if os.environ.get('ALIF_OFF') else 'ALIF on', 'STP u0', os.environ.get('STP_U0', 'default'), 'tau_rec', os.environ.get('STP_TAU_REC', 'default'), flush=True)
    for bg in BGS:
        s0, c0 = run(fresh(), 300, bg, seed=0)
        print(f'bg {bg:.2f}: background rate {c0.mean() / 300:.4f}, saturated {(c0 / 300 > 0.5).mean():.4f}', flush=True)
        for name in ('sugar', 'bitter'):
            ex = {k: [] for k in groups}; hop = [[] for _ in range(4)]
            for rep in range(3):
                base = run(s0, 300, bg, None, seed=30 + rep)[1]; extra = run(s0, 300, bg, name, seed=30 + rep)[1] - base
                for k, m in groups.items(): ex[k].append(extra[m].mean())
                for k, h in enumerate([S[name]] + H[name]): hop[k].append(extra[h].mean())
            print(f'   {name:>6}: hops', ' '.join(f'{np.mean(v):+.2f}' for v in hop), '|', ' '.join(f'{k}:{np.mean(v):+.2f}' for k, v in ex.items()), flush=True)

# strong-line targets: rank cells by the signed 2-hop / 3-hop synapse mass from the stimulus (birth graph, E +, I -)
import scipy.sparse as sp
we = np.abs(g['edge_weight_e']) / W0; wi = np.abs(g['edge_weight_i']) / W0
A = sp.coo_matrix((np.concatenate([we, -wi]), (np.concatenate([post_e, g['edge_post_i']]), np.concatenate([pre_e, g['edge_pre_i']]))), shape=(N, N)).tocsr()
with torch.no_grad():
    bg = BGS[-1]
    s0 = run(fresh(), 300, bg, seed=0)[0]
    for name in ('sugar', 'bitter'):
        x1 = A @ S[name].astype(float); x2 = A @ np.maximum(x1, 0); x3 = A @ np.maximum(x2, 0)
        sets = {'hop1 top100': np.argsort(-x1)[:100], 'hop2 top100': np.argsort(-np.where(x1 > 0, -1e9, x2))[:100],
                'hop3 top100': np.argsort(-np.where((x1 > 0) | (x2 > 0), -1e9, x3))[:100],
                'PAM by hop3': np.nonzero(groups['PAM'])[0][np.argsort(-x3[groups['PAM']])[:20]]}
        exs = {k: [] for k in sets}
        for rep in range(3):
            base = run(s0, 300, bg, None, seed=40 + rep)[1]; extra = run(s0, 300, bg, name, seed=40 + rep)[1] - base
            for k, ix in sets.items():
                exs[k].append(extra[ix].mean())
        print(f'   {name:>6} strong lines (synapse-weighted):', ' | '.join(f'{k}: {np.mean(v):+.2f} (syn mass median {np.median({"hop1 top100": x1, "hop2 top100": x2, "hop3 top100": x3, "PAM by hop3": x3}[k][sets[k]]):.0f})' for k, v in exs.items()), flush=True)
