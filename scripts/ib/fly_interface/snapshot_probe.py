"""One load, one report (user, 2026-10-14: speed decides how soon we are back on the mainline): the Kenyon repeat probe
(P1), the GPT-2 meaning probe (P3) and the word margin probe in a single pass, ~5 min instead of ~15.
Stream A (gap 0, the trainer's way, N words): every word's active set per layer -> coding level, same word / same
last-2 / last-3 words vs controls; GPT-2 (layer 6, last 3 tokens) nearest context with no shared token vs random ->
overlap ratio per layer and the Kenyon partial rho beyond the receptor + ALPN overlap.
Stream B (isolated words: the word's ticks, then 32 quiet ticks; 200 words): spiking fraction and the top-1% h / theta
per layer.
Distributions as [min | median | mean | max].  Zero-shot interventions through world_eval's env (ZS_COMPARTMENT ...).
Usage: python snapshot_probe.py <checkpoint dir> [out.json] [words=3000]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
N_WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
theta = opts['thresholds'].detach().float().expand(1, model.n_neurons)[0]
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
cls = ann['class'].astype(str).values
LAYERS = {'receptors': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON'}
LT = {k: torch.as_tensor(v, device='cuda') for k, v in LAYERS.items()}
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)
START = 40_000
words = np.array(val[START:START + N_WORDS], dtype=np.int64)


def quiet_tick(s):
    baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
    return fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])


def word_ticks(s, token, track_peak=False):
    act = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
    peak = torch.full((model.n_neurons,), -1e9, device='cuda') if track_peak else None
    for t in range(TICKS):
        if t == 0 or HOLD:
            dr, baseline = fl.write_input(model, s, token)
            s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
        else:
            s = quiet_tick(s)
        spk = s.ring[0][0] > 0
        act |= spk
        if track_peak:
            peak = torch.maximum(peak, torch.where(spk, theta, s.h[0].float()))
    s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
    return s, act, peak


sets = {k: [] for k in LAYERS}
margin = {k: {'active': [], 'top1': []} for k in LAYERS}
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for w in words:                                                   # stream A
        s, act, _ = word_ticks(s, torch.tensor([int(w)], device='cuda'))
        for k, m in LT.items():
            sets[k].append(act[m].cpu().numpy())
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for w in np.array(val[START + 5_000:START + 5_200], dtype=np.int64):   # stream B: isolated words
        s, act, peak = word_ticks(s, torch.tensor([int(w)], device='cuda'), track_peak=True)
        for _ in range(32):
            s = quiet_tick(s)
        for k, m in LT.items():
            margin[k]['active'].append(float(act[m].float().mean()))
            margin[k]['top1'].append(float(torch.quantile((peak[m] / theta[m].clamp_min(1e-6)).float(), 0.99)))
del s, model
torch.cuda.empty_cache()
A = {k: np.stack(v).astype(np.float32) for k, v in sets.items()}
T, BURN = len(words), 32


def q(v):
    v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
    return [round(float(x), 4) for x in (v.min(), np.median(v), v.mean(), v.max())] if v.size else None


def ov(k, a, b):
    x, y = A[k][a], A[k][b]
    return float((x * y).sum() / np.sqrt(x.sum() * y.sum())) if x.sum() > 0 and y.sum() > 0 else np.nan


rng = np.random.default_rng(0)
idx = np.arange(BURN, T)


def pairs_for(n):
    last, same, diff = {}, [], []
    for i in idx:
        key = tuple(int(x) for x in words[i - n + 1:i + 1])
        if key in last:
            same.append((last[key], i))
        last[key] = i
    for a, b in same:
        d = b - a
        for _ in range(20):
            j = int(rng.integers(BURN, T - d))
            if tuple(words[j - n + 1:j + 1]) != tuple(words[j + d - n + 1:j + d + 1]):
                diff.append((j, j + d)); break
    return same, diff


rep = {'words': T - BURN, 'compartment_rule_zero_shot': bool(__import__('os').environ.get('ZS_COMPARTMENT'))}
P = {n: pairs_for(n) for n in (1, 2, 3)}
for k in LAYERS:
    r = {'level': round(float(A[k][BURN:].mean()), 4)}
    for n, (sp, dp) in P.items():
        a = np.array([ov(k, *p) for p in sp]); b = np.array([ov(k, *p) for p in dp])
        r[f'same{n}_vs_diff_median'] = [q(a)[1] if q(a) else None, q(b)[1] if q(b) else None]
        r[f'same{n}_over_diff_mean'] = round(float(np.nanmean(a) / max(np.nanmean(b), 1e-9)), 3)
    r['isolated_active'] = q(margin[k]['active']); r['isolated_top1_h_over_theta'] = q(margin[k]['top1'])
    rep[k] = r

# P3 with GPT-2 (last 3 tokens, layer 6)
from transformers import GPT2Model
gpt = GPT2Model.from_pretrained('openai-community/gpt2', cache_dir='F:/ml_cache/huggingface', local_files_only=True).cuda().eval()
with torch.no_grad():
    ids = np.stack([np.array(val[START + t - 2:START + t + 1], dtype=np.int64) for t in range(T)])
    H = np.concatenate([gpt(torch.tensor(ids[b:b + 500], device='cuda'), output_hidden_states=True).hidden_states[6][:, -1].float().cpu().numpy()
                        for b in range(0, T, 500)])
del gpt
torch.cuda.empty_cache()
H = H - H.mean(0, keepdims=True); H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
last3 = [set(int(x) for x in words[max(0, t - 2):t + 1]) for t in range(T)]
anchors = rng.choice(idx, size=min(600, idx.size), replace=False)
nb, rd = [], []
for a in anchors:
    far = idx[np.abs(idx - a) >= 16]
    far = far[[not (last3[a] & last3[j]) for j in far]]
    if far.size:
        nb.append((int(a), int(far[np.argmax(H[far] @ H[a])]))); rd.append((int(a), int(rng.choice(far))))
p3 = {k: round(float(np.nanmean([ov(k, *p) for p in nb]) / max(np.nanmean([ov(k, *p) for p in rd]), 1e-9)), 3) for k in LAYERS}
pairs = []
while len(pairs) < 20_000:
    a, b = rng.choice(idx, 2)
    if abs(a - b) >= 16 and not (last3[a] & last3[b]):
        pairs.append((int(a), int(b)))
cos = np.array([float(H[a] @ H[b]) for a, b in pairs])
o = {k: np.array([ov(k, *p) for p in pairs]) for k in LAYERS}
m = np.isfinite(o['Kenyon']) & np.isfinite(o['receptors']) & np.isfinite(o['ALPN'])
X = np.stack([np.ones(m.sum()), o['receptors'][m], o['ALPN'][m]], 1)
resid = o['Kenyon'][m] - X @ np.linalg.lstsq(X, o['Kenyon'][m], rcond=None)[0]
rep['P3_neighbour_over_random'] = p3
rep['P3_kenyon_rho_beyond_senses'] = round(float(pd.Series(cos[m]).rank().corr(pd.Series(resid).rank())), 4)
print(json.dumps(rep), flush=True)
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
