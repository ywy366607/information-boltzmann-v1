"""What drives the MBONs while reading?  The Kenyon code is a context code (m_main c7: same last 3 words overlap 0.47 vs
0.08), but the MBONs do not read it (same vs different contexts 0.53-0.60 vs 0.50-0.53; 14% of them fire for every
word).  This splits the synaptic drive reaching the MBONs per tick by presynaptic class (transmitted output x edge
weight; graded cells transmit their graded output), with the Kenyon -> MBON compartment synapses counted separately at
their current efficacy comp_weight x (1 + phi) (they leave the static edge list when the compartment memory is on);
and asks whether the Kenyon -> MBON drive itself is context-specific: the correlation of each word's Kenyon -> MBON
drive vector (per MBON, summed over the word's ticks) between successive occurrences of the same last-3-word context
vs random pairs.  Also the MBONs' firing per word (fraction of ticks each fires), and phi's distribution.
Usage: python mbon_input_probe.py <checkpoint dir> [words=3000] [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
N_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
cls = ann['class'].astype(str).values
names = sorted(set(cls.tolist())) + ['Kenyon_Cell (compartment)']
COMP = len(names) - 1
cid = torch.as_tensor(np.array([names.index(c) for c in cls]), device='cuda')
mbon = torch.as_tensor(cls == 'MBON', device='cuda')
n_mbon = int(mbon.sum()); mbon_index = torch.cumsum(mbon.long(), 0) - 1
sel = {}
for sfx, sign in (('e', 1.0), ('i', -1.0)):
    pre = getattr(model, f'edge_pre_{sfx}').long(); post = getattr(model, f'edge_post_{sfx}').long()
    m = mbon[post]
    sel[sfx] = (pre[m], post[m], getattr(model, f'edge_weight_{sfx}')[m].detach().float() * sign)
has_comp = getattr(model, 'comp_weight', None) is not None
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)
words = np.array(val[40_000:40_000 + N_WORDS], dtype=np.int64)
n_cls = len(names)
drive_all = torch.zeros(n_cls, device='cuda')
kc_word, fire_word = [], []
phi0 = None
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        kd = torch.zeros(n_mbon, device='cuda'); fires = torch.zeros(n_mbon, device='cuda')
        for t in range(TICKS):
            out = s.ring[0][0].float()
            for sfx, (pre, post, wt) in sel.items():
                drive_all.index_add_(0, cid[pre], out[pre] * wt)
            if has_comp:
                phi = s.comp_memory[0] if s.comp_memory.numel() == model.comp_pre.numel() else torch.zeros_like(model.comp_weight)
                if phi0 is None:
                    phi0 = phi.detach().cpu().numpy()
                c = out[model.comp_pre.long()] * model.comp_weight.detach().float() * (1.0 + phi.float())
                drive_all[COMP] += c.sum()
                kd.index_add_(0, mbon_index[model.comp_post.long()], c)
            if t == 0 or HOLD:
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            fires += (s.ring[0][0][mbon] > 0).float()
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        kc_word.append(kd.cpu().numpy()); fire_word.append((fires / TICKS).cpu().numpy())


def share(v):
    v = v.cpu().numpy(); tot = np.abs(v).sum()
    order = np.argsort(-np.abs(v))[:10]
    return {names[i]: round(float(v[i] / tot), 4) for i in order if v[i] != 0}


q = lambda v: [round(float(x), 4) for x in (np.min(v), np.median(v), np.mean(v), np.max(v))] if len(v) else None
BURN = 32
K = np.array(kc_word); Fw = np.array(fire_word)
last, same, diff, same_f, diff_f = {}, [], [], [], []
rng = np.random.default_rng(0)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


for i in range(BURN, len(words)):
    key = tuple(int(x) for x in words[i - 2:i + 1])
    if key in last:
        a = last[key]; j = int(rng.integers(BURN, len(words)))
        same.append(corr(K[a], K[i])); diff.append(corr(K[j], K[i]))
        same_f.append(corr(Fw[a], Fw[i])); diff_f.append(corr(Fw[j], Fw[i]))
    last[key] = i
fin = lambda v: [x for x in v if np.isfinite(x)]
rate = Fw[BURN:]
rep = {'words': len(words) - BURN, 'compartment_memory': has_comp,
       'drive_share_all_ticks': share(drive_all),
       'mbon_fire_fraction_of_ticks_per_cell': q(rate.mean(0)),
       'mbon_cells_firing_on_over_90pct_of_words': round(float(((rate > 0).mean(0) > 0.9).mean()), 4),
       'kc_to_mbon_drive_corr_same_last3': q(fin(same)), 'kc_to_mbon_drive_corr_random': q(fin(diff)),
       'mbon_firing_corr_same_last3': q(fin(same_f)), 'mbon_firing_corr_random': q(fin(diff_f)),
       'pairs': len(same)}
if phi0 is not None:
    rep['phi_at_start'] = q(phi0)
    rep['phi_fraction_below_minus_0.5'] = round(float((phi0 < -0.5).mean()), 4)
print(json.dumps(rep), flush=True)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
