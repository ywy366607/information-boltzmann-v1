"""Does this brain reproduce the validated whole-brain LIF (Shiu et al. 2024) before our own mechanisms are added?
Zero-shot on the corrected birth graph (uniform per-synapse efficacy, PSP 0.617% of threshold, tau_m 20, biophysical
reversals E_E 0.55 / E_I -0.15).  Sugar or bitter GRNs fire Poisson at RATE Hz (1 tick = 1 ms) for T ticks from rest;
read: firing rate (Hz) of MN9 (proboscis motor, Shiu's sugar readout), PAM01, all PAM, PPL1, and mean extra spikes
by hop.  Arms switch our additions on one at a time: shiu (STP off, ALIF off, no background), +stp, +alif,
+background (1% of sensory cells per tick), all.  If 'shiu' fails on MN9 the propagation itself is wrong; if it passes,
the loss of a sense is caused by the arm that kills it.
Usage: python shiu_replication_probe.py   (env RATE=100 T=1000 ARMS=shiu,stp,alif,bg,all)
"""
import os, sys, math, numpy as np, torch, pandas as pd
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core import fly_bptt_learning as fl
ROOT = 'D:/information-boltzmann-v1'
G = f'{ROOT}/data/malecns_v1/fly_reservoir_coba_syn.npz'; W0 = 0.000979; WSCALE = float(os.environ.get('WSCALE', 1.82))
SEED = int(os.environ.get('SEED', 7))
RATE, T = float(os.environ.get('RATE', 100)), int(os.environ.get('T', 1000))
g = np.load(G, allow_pickle=True); N = len(g['neuron_body_ids'])
ann = pd.read_feather(f'{ROOT}/data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
typ = ann['type'].astype(str).values.astype(str)
sc = np.array(g['superclass_names'])[g['superclass_id']]
dan = np.zeros(N, bool); dan[g['dan_indices']] = True
pre_e, post_e = g['edge_pre_e'], g['edge_post_e']
S = {'sugar': np.isin(typ, ['LB3b', 'LB3c', 'dorsal_tpGRN']), 'bitter': np.isin(typ, ['LB1a', 'LB1b', 'LB1c', 'LB1d'])}
read = {'MN9': typ == 'MN9', 'MN6': typ == 'MN6', 'MN8': typ == 'MN8', 'PAM01': typ == 'PAM01', 'PAM': dan & np.char.startswith(typ, 'PAM'),
        'PPL1': dan & np.char.startswith(typ, 'PPL1'), 'SMP744': typ == 'SMP744'}


def hops(src, k=4):
    seen = src.copy(); fr = src.copy(); out = []
    for _ in range(k):
        nx = np.zeros(N, bool); nx[post_e[fr[pre_e]]] = True; nx &= ~seen; out.append(nx); seen |= nx; fr = nx
    return out


H = {k: [v] + hops(v) for k, v in S.items()}
model = FlyReservoirLM(G, d_model=768, read_surface='output', injection='topographic', synapse_model='coba', use_alif=True, use_stp=True,
                       detach_reset=True, surrogate_mode='threshold', transmission_mode='incoming',
                       E_E=float(os.environ.get('EE', 0.55)), E_I=float(os.environ.get('EI', -0.15))).cuda().eval().requires_grad_(False)
with torch.no_grad():
    for sfx in ('e', 'i'):
        getattr(model, f'edge_weight_{sfx}').mul_(WSCALE)            # x1.82 keeps the PSP at 0.617% under E_E 0.55 (calibrated)
    # WI_SCALE: inhibitory weights x this; 0.55 / 0.15 = 3.67 makes an IPSP at rest as large as an EPSP (Shiu: equal |PSP|)
    model.edge_weight_i.mul_(float(os.environ.get('WI_SCALE', 1.0)))
    # MIN_SYN: drop connections with fewer synapses (FlyWire's public connectivity keeps >= 5)
    MIN_SYN = int(os.environ.get('MIN_SYN', 0))
    for sfx in ('e', 'i'):
        w = getattr(model, f'edge_weight_{sfx}')
        w.mul_((w.abs() / (W0 * WSCALE * (float(os.environ.get('WI_SCALE', 1.0)) if sfx == 'i' else 1.0)) >= MIN_SYN - 0.5).float())
# NO_SENSORY_INPUT: sensory neurons take no synaptic input (their inputs sit on axon terminals: presynaptic modulation,
# not somatic drive -- in a point model they make the receptors excite each other)
if int(os.environ.get('NO_SENSORY_INPUT', 0)):
    with torch.no_grad():
        is_sens = torch.as_tensor(np.char.find(sc.astype(str), 'sensory') >= 0, device='cuda')
        for sfx in ('e', 'i'):
            w = getattr(model, f'edge_weight_{sfx}'); w.mul_((~is_sens[getattr(model, f'edge_post_{sfx}').long()]).float())
# SHIU_RESET: on a spike the synaptic drive is consumed (g = 0) and the cell is refractory for RFC ticks (Shiu: 2.2 ms;
# v and g held at 0) -- our engine only resets v and keeps the conductances, so a strongly driven cell refires every tick
SHIU_RESET, RFC = bool(int(os.environ.get('SHIU_RESET', 0))), int(os.environ.get('RFC', 2))
import dataclasses
beta0 = model.log_beta.detach().clone()
thr = model.get_thresholds().detach().expand(1, N)[0]; z = torch.zeros(1, N, device='cuda')
sens = torch.as_tensor(np.nonzero(np.char.find(sc.astype(str), 'sensory') >= 0)[0], device='cuda')
Si = {k: torch.as_tensor(np.nonzero(v)[0], device='cuda') for k, v in S.items()}


def configure(stp, alif):
    model.use_stp = stp
    with torch.no_grad():
        model.log_beta.copy_(beta0 if alif else torch.full_like(beta0, -30.0))
    return dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
                alif_params=model.get_alif_params(), stp_params=model.get_stp_params())


def fresh(opts):
    return fl.FlyPhysicalState(h=z.clone(), ring=tuple(z.clone() for _ in range(4)), ge=z.clone(), gi=z.clone(), b=z.clone(),
                               x=torch.ones_like(z), u=opts['stp_params'][0].detach().expand(1, N).clone(), baseline=z.clone())


def run(opts, stim, bg, seed, T=T, warm=300):
    gen = torch.Generator(device='cuda').manual_seed(seed); s = fresh(opts); c = torch.zeros(N, device='cuda')
    refr = torch.zeros(1, N, device='cuda')
    for t in range(warm + T):
        d = torch.zeros(1, N, device='cuda')
        if bg > 0:
            on = sens[torch.rand(sens.numel(), device='cuda', generator=gen) < bg]; d[0, on] = 1.5 * thr[on]
        r = torch.rand(N, device='cuda', generator=gen)
        if stim is not None and t >= warm:
            hit = Si[stim][r[Si[stim]] < RATE / 1000]; d[0, hit] = 1.5 * thr[hit]
        model.external_drive = d
        s = fl.step_fly_physical_tick(model, s, None, z, s.baseline, opts, base_rates=opts['base_rates'])
        if SHIU_RESET:
            refr = torch.where(s.ring[0] > 0, torch.full_like(refr, RFC), (refr - 1).clamp_min(0))
            keep = (refr == 0).float()
            s = dataclasses.replace(s, h=s.h * keep, ge=s.ge * keep, gi=s.gi * keep)
        if t >= warm:
            c += (s.ring[0][0] > 0).float()
    model.external_drive = None
    return c.cpu().numpy() * 1000 / T                              # Hz


ARMS = {'shiu': (False, False, 0.0), 'stp': (True, False, 0.0), 'alif': (False, True, 0.0), 'bg': (False, False, 0.01), 'all': (True, True, 0.01), 'bgalif': (False, True, 0.01), 'bgalif5': (False, True, 0.002)}
print(f'input {RATE:.0f} Hz Poisson, {T} ms; rates in Hz (stimulus - control)', flush=True)
with torch.no_grad():
    for arm in os.environ.get('ARMS', 'shiu,stp,alif,bg,all').split(','):
        stp, alif, bg = ARMS[arm]; opts = configure(stp, alif)
        base = run(opts, None, bg, seed=SEED)
        print(f'{arm:>5} (STP {"on" if stp else "off"}, ALIF {"on" if alif else "off"}, bg {bg}): control mean {base.mean():.2f} Hz, '
              f'saturated {(base > 500).mean():.4f}', flush=True)
        print(f'      settings: WI_SCALE {os.environ.get("WI_SCALE", 1.0)} MIN_SYN {MIN_SYN} SHIU_RESET {SHIU_RESET}', flush=True)
        for name in S:
            r = run(opts, name, bg, seed=SEED)
            ex = r - base
            print(f'      {name}: cells > 10 Hz above control {int((ex > 10).sum())}, > 100 Hz {int((ex > 100).sum())} (of {N})', flush=True)
            print(f'   {name:>6}: hop Hz ' + ' '.join(f'{ex[h].mean():+.2f}' for h in H[name]) + ' | ' +
                  ' '.join(f'{k} {r[m].mean():.1f}({ex[m].mean():+.1f})' for k, m in read.items()), flush=True)

# DECOMP=<type>: in the last arm, which presynaptic cells carry each stimulus into that type, in activity:
# extra spikes of each presynaptic cell x its signed synapses onto the type (E +, I -), top contributors
if os.environ.get('DECOMP'):
    tgt = np.nonzero(typ == os.environ['DECOMP'])[0]
    import scipy.sparse as sp
    A = sp.coo_matrix((np.concatenate([np.abs(g['edge_weight_e']), -np.abs(g['edge_weight_i'])]) / W0,
                       (np.concatenate([post_e, g['edge_post_i']]), np.concatenate([pre_e, g['edge_pre_i']]))), shape=(N, N)).tocsr()
    Ap = A[tgt].toarray().sum(0)
    with torch.no_grad():
        base = run(opts, None, bg, seed=SEED)
        for name in S:
            ex = run(opts, name, bg, seed=SEED) - base
            c = ex * Ap
            print(f'   {name} -> {os.environ["DECOMP"]}: signed drive (Hz x synapses) total {c.sum():+.0f} = E {c[Ap > 0].sum():+.0f} + I {c[Ap < 0].sum():+.0f}', flush=True)
            for j in np.argsort(-np.abs(c))[:10]:
                print(f'      {typ[j]:>16} syn {Ap[j]:+5.0f} extra {ex[j]:+7.1f} Hz  drive {c[j]:+8.0f}  {"GRN" if S["sugar"][j] or S["bitter"][j] else ""}', flush=True)
