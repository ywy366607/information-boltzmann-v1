"""What drives the Kenyon cells while reading?  A word's Kenyon code repeats only weakly when the word repeats (m_main
c7: same word 0.15 vs different words 0.10, level 0.03; the ALPN layer above it repeats, 0.48 vs 0.25), so the Kenyon
cells fire mostly from something other than the word's projection-neuron input.  This splits the synaptic drive that
reaches the Kenyon cells, per tick, by the presynaptic cell class (transmitted spikes x edge weight, excitatory and
inhibitory lists), over the ticks when a Kenyon cell fires and over all ticks; and the per-word reliability of the
ALPN -> Kenyon drive alone (does the word's PN input to the Kenyon cells repeat?).
Usage: python kenyon_input_probe.py <checkpoint dir> [words=400] [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
N_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
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
kc_np = cls == 'Kenyon_Cell'
names = sorted(set(cls.tolist()))
cid = torch.as_tensor(np.array([names.index(c) for c in cls]), device='cuda')
kc = torch.as_tensor(kc_np, device='cuda')
sel = {}
for sfx, sign in (('e', 1.0), ('i', -1.0)):
    pre = getattr(model, f'edge_pre_{sfx}').long(); post = getattr(model, f'edge_post_{sfx}').long()
    m = kc[post]
    sel[sfx] = (pre[m], post[m], getattr(model, f'edge_weight_{sfx}')[m].detach().float() * sign)
alpn = torch.as_tensor(cls == 'ALPN', device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)
words = np.array(val[40_000:40_000 + N_WORDS], dtype=np.int64)
n_cls = len(names)
drive_all = torch.zeros(n_cls, device='cuda'); drive_fire = torch.zeros(n_cls, device='cuda')
pn_word = []                                     # per word: ALPN -> Kenyon drive per Kenyon cell (summed over ticks)
n_kc = int(kc.sum()); kc_index = torch.cumsum(kc.long(), 0) - 1
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        pn_drive = torch.zeros(n_kc, device='cuda')
        for t in range(TICKS):
            spikes = (s.ring[0][0] > 0).float()
            for sfx, (pre, post, wt) in sel.items():
                c = spikes[pre] * wt                                # this tick's transmitted drive per edge
                drive_all.index_add_(0, cid[pre], c)
                fired_post = (s.ring[0][0][post] > 0).float()       # into a Kenyon cell that fires now
                drive_fire.index_add_(0, cid[pre], c * fired_post)
                if sfx == 'e':
                    is_pn = alpn[pre]
                    pn_drive.index_add_(0, kc_index[post[is_pn]], c[is_pn])
            if t == 0 or HOLD:
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        pn_word.append(pn_drive.cpu().numpy())


def share(v):
    v = v.cpu().numpy(); tot = np.abs(v).sum()
    order = np.argsort(-np.abs(v))[:8]
    return {names[i]: round(float(v[i] / tot), 4) for i in order if v[i] != 0}


# reliability of the word's PN input: correlation of the ALPN -> Kenyon drive vectors, same word vs different words
P = np.array(pn_word[32:]); W = words[32:]
last, same, diff = {}, [], []
rng = np.random.default_rng(0)
for i, w in enumerate(W):
    if w in last:
        a, b = P[last[w]], P[i]
        same.append(float(np.corrcoef(a, b)[0, 1]))
        j = int(rng.integers(0, len(W)))
        if W[j] != w:
            diff.append(float(np.corrcoef(P[j], b)[0, 1]))
    last[w] = i
q = lambda v: [round(float(x), 4) for x in (np.min(v), np.median(v), np.mean(v), np.max(v))] if len(v) else None
rep = {'words': len(W), 'drive_share_all_ticks': share(drive_all), 'drive_share_when_kenyon_fires': share(drive_fire),
       'pn_drive_corr_same_word': q(same), 'pn_drive_corr_different_word': q(diff)}
print(json.dumps(rep), flush=True)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
