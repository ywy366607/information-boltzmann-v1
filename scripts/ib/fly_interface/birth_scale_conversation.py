"""The rebirth question (2026-10-15): at which birth synapse scale does the conversation reach the deep brain, without
the brain running away?  Zero-shot on a BIRTH checkpoint (no training), with the general compartment rule applied
(ZS_COMPARTMENT), every edge rescaled to a uniform efficacy per synapse e x theta (count = w sqrt(K_in) / g, g = 0.0058,
as birth_scale_probe.py; the compartment shares stay as fractions of the count).  The stream is Mengli's own
conversation (mother.jsonl, last N tokens), each word held TICKS ticks, the mother's words with her voice on
Johnston's organ.  Paired branches from the same state:
  V  voice on - off, per mother word;  H  word written - quiet, per word (every 4th);
by group: JO, auditory first / second order, hop 1..4 from the written cells, Kenyon, MBON, PAM, PPL1, output cells.
S  stability along the main trajectory: mean rate of all cells and per group, fraction of cells firing on > 50% of
   ticks (saturation), fraction never firing.
Usage: python birth_scale_conversation.py <birth .pt> <run dir> <e or 'born'> [tokens=400] [out.json]
"""
import sys, json, math, os, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
os.environ.setdefault('ZS_COMPARTMENT', '1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

src, RUN, SCALE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
N_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 400
OUT = Path(sys.argv[5]) if len(sys.argv) > 5 else None
TICKS = 6
WRITE_GAIN = float(os.environ.get('WRITE_GAIN', 1.0))
G_CONV = 0.0058


class _Fork:
    def __truediv__(self, name):
        return src


model, cfg, L = load(_Fork())
graph_path = ROOT / cfg['graph']
ann_path = graph_path.parent / 'body-annotations.feather'
fl.attach_mother_voice(model, graph_path, ann_path, level=1.5)
dev = 'cuda'
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
theta = opts['thresholds'].detach().float().expand(1, N)[0]
g = np.load(graph_path, allow_pickle=True)
K = torch.as_tensor(np.maximum(np.bincount(np.asarray(g['edge_post']), minlength=N), 1), dtype=torch.float32, device=dev)
if SCALE != 'born':
    e = float(SCALE)
    with torch.no_grad():
        for sfx in ('e', 'i'):
            post = getattr(model, f'edge_post_{sfx}').long()
            getattr(model, f'edge_weight_{sfx}').mul_(torch.sqrt(K[post]) / G_CONV * e * theta[post])
        if getattr(model, 'comp_weight', None) is not None:
            post = model.comp_post.long()
            model.comp_weight.mul_(torch.sqrt(K[post]) / G_CONV * e * theta[post])
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, N, device=dev)
s0 = move_state(fl.FlyPhysicalState(**L['physical']), dev)
if getattr(model, 'dopamine_sensitivity', None) is not None:
    opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, s0.h)

ann = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls, sup = (ann[c].astype(str).values for c in ('type', 'class', 'superclass'))
import scipy.sparse as sp
pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]).astype(np.int64)
post_all = np.concatenate([g['edge_post_e'], g['edge_post_i']]).astype(np.int64)
A = sp.csr_matrix((np.ones(pre.size, dtype=np.int8), (pre, post_all)), shape=(N, N))


def hops_from(seed, max_hop=4):
    d = np.full(N, 99); d[seed] = 0
    front = np.zeros(N, bool); front[seed] = True
    for h in range(1, max_hop + 1):
        new = (np.asarray(A.T @ front.astype(np.int8)).ravel() > 0) & (d == 99)
        d[new] = h; front = new
    return d


jo = model.voice_cells.cpu().numpy()
We = sp.csr_matrix((np.abs(g['edge_weight_e']).astype(np.float64), (g['edge_pre_e'], g['edge_post_e'])), shape=(N, N))
exc_in = np.maximum(np.asarray(We.sum(0)).ravel(), 1e-30)
first = np.setdiff1d(np.flatnonzero(np.asarray(We[jo].sum(0)).ravel() >= 0.05 * exc_in), jo)
second = np.setdiff1d(np.flatnonzero(np.asarray(We[first].sum(0)).ravel() >= 0.05 * exc_in), np.concatenate([jo, first]))
dan = g['dan_indices'].astype(np.int64)
deep = {'KC': np.flatnonzero(cls == 'Kenyon_Cell'), 'MBON': np.flatnonzero(cls == 'MBON'),
        'PAM': dan[np.char.startswith(typ[dan].astype(str), 'PAM')], 'PPL1': dan[np.char.startswith(typ[dan].astype(str), 'PPL1')],
        'output': np.flatnonzero(np.isin(sup, ('descending_neuron', 'vnc_motor', 'cb_motor', 'vnc_efferent')))}


def masks(groups):
    return {k: torch.as_tensor(np.isin(np.arange(N), v), device=dev) for k, v in groups.items() if len(v)}


MV = masks({'JO': jo, 'aud1': first, 'aud2': second, **deep})

import tiktoken
tok = tiktoken.get_encoding('gpt2')
lines = [json.loads(l) for l in (RUN / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
stream, who = [], []
for r in reversed(lines):
    ids = tok.encode(' ' + r['text'].strip() + ('\n' if r['who'] == 'mother' else ''))
    stream = ids + stream; who = [r['who']] * len(ids) + who
    if len(stream) >= N_TOK + 1:
        break
stream, who = stream[-(N_TOK + 1):], who[-(N_TOK + 1):]


def word(s, w, voice=False, write=True, groups=None, count=None):
    token = torch.tensor([int(w)], device=dev)
    vd = fl.mother_voice_drive(model, s.h) if voice else None
    acc = {k: 0.0 for k in (groups or {})}
    for t in range(TICKS):
        model.external_drive = vd
        if write and (t == 0 or HOLD):
            dr, baseline = fl.write_input(model, s, token)
            dr = dr * WRITE_GAIN                     # dose-response of the text writer (env WRITE_GAIN, default 1)
            s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
        else:
            baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
            s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
        model.external_drive = None
        spk = (s.ring[0][0] > 0).float()
        if count is not None:
            count += spk
        for k, m in (groups or {}).items():
            acc[k] += float(spk[m].mean()) / TICKS
    return s, acc


def dist(x):
    x = np.asarray(x, dtype=np.float64)
    return {'mean': float(x.mean()), 'se': float(x.std(ddof=1) / math.sqrt(x.size)), 'min': float(x.min()),
            'median': float(np.median(x)), 'max': float(x.max())} if x.size > 1 else None


with torch.no_grad():
    drv = torch.zeros(N, dtype=torch.bool, device=dev)
    for w in stream[:64]:
        dr, _ = fl.write_input(model, s0, torch.tensor([int(w)], device=dev))
        drv |= dr[0].abs() > 0
    d_in = hops_from(np.flatnonzero(drv.cpu().numpy()))
    MH = masks({**{f'hop{h}': np.flatnonzero(d_in == h) for h in range(0, 5)}, **deep})
    V, H = [], []
    count = torch.zeros(N, device=dev)
    s, ticks_run = s0, 0
    MS = masks({'all': np.arange(N), **deep})
    stab = []
    for i in range(len(stream) - 1):
        w, mom = stream[i], who[i] == 'mother'
        if mom:
            _, on = word(s, w, voice=True, groups=MV)
            _, off = word(s, w, voice=False, groups=MV)
            V.append({k: on[k] - off[k] for k in on})
        if i % 4 == 0:
            _, wr = word(s, w, voice=mom, groups=MH)
            _, qu = word(s, w, voice=mom, write=False, groups=MH)
            H.append({k: wr[k] - qu[k] for k in wr} | {f'{k}_q': qu[k] for k in qu})
        s, r = word(s, w, voice=mom, groups=MS, count=count)
        ticks_run += TICKS
        stab.append(r)
        if not torch.isfinite(s.h).all():
            break
    frac = (count / max(ticks_run, 1)).cpu().numpy()
rep = {'checkpoint': str(src), 'scale': SCALE, 'write_gain': WRITE_GAIN, 'tokens': len(stream) - 1, 'ticks': ticks_run,
       'V': {k: dist([r[k] for r in V]) for k in MV}, 'H': {k: dist([r[k] for r in H]) for k in MH},
       'H_quiet_rate': {k: float(np.mean([r[f'{k}_q'] for r in H])) for k in MH},
       'S_rate': {k: {'first_quarter': float(np.mean([r[k] for r in stab[:len(stab) // 4]])),
                      'last_quarter': float(np.mean([r[k] for r in stab[-len(stab) // 4:]]))} for k in MS},
       'S_saturated_frac': float((frac > 0.5).mean()), 'S_silent_frac': float((frac == 0).mean()),
       'finite': bool(torch.isfinite(s.h).all())}
print(json.dumps(rep, indent=1), flush=True)
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
