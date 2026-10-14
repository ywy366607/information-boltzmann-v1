"""P3 with GPT-2 as the measure of meaning: do contexts that mean the same with different words get closer Kenyon codes?
Reads a stretch of validation text as kenyon_repeat_probe.py does (gap 0) and keeps every word's active set per layer.
Meaning of the context at word t, from GPT-2 small (layer 6 of 12, centred over the stretch, cosine):
- short: GPT-2 fed only the last 3 tokens (the Kenyon window, ~3 words), hidden state at the last token;
- long:  GPT-2 fed up to 512 tokens of preceding text, hidden state at t.
For anchor words, the nearest context sharing NO token among the last 3 words vs a random context, both >= 16 words
away: overlap |a & b| / sqrt(|a||b|) per layer.  And over 20,000 random pairs (no shared token): the rank correlation of
GPT-2 cosine with the Kenyon overlap, raw and after regressing out the receptor and ALPN overlaps (what the senses
already share).  If the Kenyon code carries meaning, neighbour > random and the partial correlation > 0.
Distributions as [min | median | mean | max].
Usage: python kenyon_gpt2_probe.py <checkpoint dir> [words=3000] [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

GPT2 = 'F:/ml_cache/huggingface'
LAYER, LONG = 6, 512
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
LAYERS = {'olfactory receptors': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON'}
LT = {k: torch.as_tensor(v, device='cuda') for k, v in LAYERS.items()}
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)
START = 40_000
words = np.array(val[START:START + N_WORDS], dtype=np.int64)
sets = {k: [] for k in LAYERS}
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        act = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
        for t in range(TICKS):
            if t == 0 or HOLD:
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            act |= s.ring[0][0] > 0
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        for k, m in LT.items():
            sets[k].append(act[m].cpu().numpy())
del s, model
torch.cuda.empty_cache()
A = {k: np.stack(v).astype(np.float32) for k, v in sets.items()}
T, BURN = len(words), 32

from transformers import GPT2Model
gpt = GPT2Model.from_pretrained('openai-community/gpt2', cache_dir=GPT2, local_files_only=True).cuda().eval()


def centred_unit(h):
    h = h - h.mean(0, keepdims=True)
    return h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)


with torch.no_grad():
    ids = np.stack([np.array(val[START + t - 2:START + t + 1], dtype=np.int64) for t in range(T)])
    short = []
    for b in range(0, T, 500):
        out = gpt(torch.tensor(ids[b:b + 500], device='cuda'), output_hidden_states=True)
        short.append(out.hidden_states[LAYER][:, -1].float().cpu().numpy())
    short = centred_unit(np.concatenate(short))
    text = np.array(val[START - LONG:START + T], dtype=np.int64)
    long = np.zeros((T, short.shape[1]), dtype=np.float32)
    for b in range(0, T, LONG):                       # positions b..b+LONG-1 each see LONG tokens before the window
        seg = torch.tensor(text[b:b + 2 * LONG][None], device='cuda')
        h = gpt(seg, output_hidden_states=True).hidden_states[LAYER][0].float().cpu().numpy()
        n = min(LONG, T - b)
        long[b:b + n] = h[LONG:LONG + n]
    long = centred_unit(long)
del gpt
torch.cuda.empty_cache()


def q(v):
    v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
    return [round(float(x), 4) for x in (v.min(), np.median(v), v.mean(), v.max())] if v.size else None


def ov(k, a, b):
    x, y = A[k][a], A[k][b]
    return float((x * y).sum() / np.sqrt(x.sum() * y.sum())) if x.sum() > 0 and y.sum() > 0 else np.nan


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())), 4)


last3 = [set(int(x) for x in words[max(0, t - 2):t + 1]) for t in range(T)]
idx_all = np.arange(BURN, T)
rng = np.random.default_rng(0)
anchors = rng.choice(idx_all, size=min(600, idx_all.size), replace=False)
rep = {'words': T - BURN, 'gpt2_layer': LAYER}
for name, H in (('short_3_tokens', short), ('long_512_tokens', long)):
    nb, rd, nbc, rdc = [], [], [], []
    for a in anchors:
        far = idx_all[np.abs(idx_all - a) >= 16]
        far = far[[not (last3[a] & last3[j]) for j in far]]
        if far.size == 0:
            continue
        c = H[far] @ H[a]
        j, jr = int(far[np.argmax(c)]), int(rng.choice(far))
        nb.append((int(a), j)); rd.append((int(a), jr)); nbc.append(float(c.max())); rdc.append(float(H[jr] @ H[a]))
    r = {'pairs': len(nb), 'cosine_neighbour': q(nbc), 'cosine_random': q(rdc)}
    for k in LAYERS:
        x = np.array([ov(k, a, b) for a, b in nb]); y = np.array([ov(k, a, b) for a, b in rd])
        r[k] = {'neighbour': q(x), 'random': q(y), 'ratio': round(float(np.nanmean(x) / max(np.nanmean(y), 1e-9)), 3)}
    # random pairs: does GPT-2 similarity predict Kenyon overlap beyond what the senses share?
    P = []
    while len(P) < 20_000:
        a, b = rng.choice(idx_all, 2)
        if abs(a - b) >= 16 and not (last3[a] & last3[b]):
            P.append((int(a), int(b)))
    cos = np.array([float(H[a] @ H[b]) for a, b in P])
    o = {k: np.array([ov(k, a, b) for a, b in P]) for k in LAYERS}
    m = np.isfinite(o['Kenyon']) & np.isfinite(o['olfactory receptors']) & np.isfinite(o['ALPN'])
    X = np.stack([np.ones(m.sum()), o['olfactory receptors'][m], o['ALPN'][m]], 1)
    resid = o['Kenyon'][m] - X @ np.linalg.lstsq(X, o['Kenyon'][m], rcond=None)[0]
    r['rho_cosine_overlap'] = {k: spearman(cos, o[k]) for k in LAYERS}
    r['rho_cosine_kenyon_beyond_senses'] = spearman(cos[m], resid)
    r['random_pairs_kenyon_defined'] = int(m.sum())
    rep[name] = r
    print(name, json.dumps(r), flush=True)
print(json.dumps(rep), flush=True)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
