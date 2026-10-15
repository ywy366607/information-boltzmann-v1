"""Is the Kenyon code word-selective?  Familiarity (the MBON-alpha'3 habituation) needs the repeated word to fire its
own Kenyon cells.  From the checkpoint's state, reading the lineage's own conversation (mother.jsonl, the last turns,
`ticks` per word, held input, the mother's voice on her words when attached), fixed weights except the online rules.
Per word: the set of Kenyon cells that spiked during its ticks, by Kenyon subtype (ab, a'b', g).
Reports [min | median | mean | max]:
  active fraction per word; cells ever active; Jaccard overlap of the sets for two occurrences of the same token vs two
  different tokens (selective code: same >> different); the same for 'Mengli'; the subtypes' share of the active cells.
Usage: python kc_code_probe.py <checkpoint dir> <run dir> [n_tokens=600] [ticks=8] [out.json]"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

src, RUN = Path(sys.argv[1]), Path(sys.argv[2])
N_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 600
TICKS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
OUT = Path(sys.argv[5]) if len(sys.argv) > 5 else None
fork = src if src.is_dir() else src.parent      # load() reads <dir>/last.pt
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
if getattr(model, 'dopamine_sensitivity', None) is not None:
    opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, s.h)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId')
ids = np.load(graph, allow_pickle=True)['neuron_body_ids']
cls = ann.reindex(ids)['class'].astype(str).values
typ = ann.reindex(ids)['type'].astype(str).values
kc = np.nonzero(cls == 'Kenyon_Cell')[0]
sub = np.array(['a\'b\'' if ("a'b'" in t or 'apbp' in t.lower() or "α'β'" in t) else 'g' if t.startswith('KCg') else 'ab'
                for t in typ[kc]])
kc_t = torch.as_tensor(kc, device='cuda')
import tiktoken
tok = tiktoken.get_encoding('gpt2')
lines = [json.loads(l) for l in (RUN / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
stream, who = [], []
for r in lines[::-1]:
    t = r['text'].strip()
    if not t or t == '...':
        continue
    ids_ = tok.encode(' ' + t)
    stream = ids_ + stream
    who = [r['who']] * len(ids_) + who
    if len(stream) >= N_TOK:
        break
stream, who = stream[-N_TOK:], who[-N_TOK:]
meng = set(tok.encode(' Mengli')) | set(tok.encode('Mengli'))
voice_on = getattr(model, 'voice_cells', None) is not None
sets, resid = [], []
counts = torch.zeros(model.n_neurons, device='cuda')
vsum = torch.zeros(model.n_neurons, device='cuda')
nt = 0
with torch.no_grad():
    for w, wh in zip(stream, who):
        fired = torch.zeros(kc.size, dtype=torch.bool, device='cuda')
        for t in range(TICKS):
            kw = dict(voice=(wh == 'mother')) if voice_on else {}
            out = fl.advance_fly_token_stream(model, s, torch.tensor([int(w)], device='cuda'), [1], ticks=1,
                                              hold_input=bool(cfg.get('hold_input')), fresh=t == 0, **kw, **opts)
            s = out[0]
            fired |= s.ring[0][0].index_select(0, kc_t) > 0
            counts += (s.ring[0][0] > 0).float()
            vsum += s.h[0] / model.get_thresholds().reshape(-1)[:model.n_neurons]
            nt += 1
        sets.append(fired.cpu().numpy())
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))      # the brain's prediction of the next word
        resid.append(float(s.expected_input.float().norm()) if s.expected_input.numel() else None)
S = np.array(sets)
dist = lambda v: [float(np.min(v)), float(np.median(v)), float(np.mean(v)), float(np.max(v))] if len(v) else None


def jacc(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else None


rng = np.random.default_rng(0)
by_tok = {}
for i, w in enumerate(stream):
    by_tok.setdefault(w, []).append(i)
same, diff, same_m = [], [], []
for w, idx in by_tok.items():
    for a in range(len(idx)):
        for b in range(a + 1, min(len(idx), a + 6)):
            j = jacc(S[idx[a]], S[idx[b]])
            if j is not None:
                (same_m if w in meng else same).append(j)
for _ in range(4000):
    i, k = rng.integers(0, len(stream), 2)
    if stream[i] != stream[k]:
        j = jacc(S[i], S[k])
        if j is not None:
            diff.append(j)
ever = S.any(0)
rep = {'checkpoint': str(src), 'tokens': len(stream), 'ticks': TICKS, 'kenyon': int(kc.size),
       'subtypes': {k: int((sub == k).sum()) for k in ("ab", "a'b'", 'g')},
       'active_fraction_per_word': dist(S.mean(1)),
       'silent_words_fraction': float((S.sum(1) == 0).mean()),
       'ever_active_fraction': float(ever.mean()),
       'ever_active_by_subtype': {k: float(ever[sub == k].mean()) if (sub == k).any() else None for k in ("ab", "a'b'", 'g')},
       'jaccard_same_token': dist(same), 'jaccard_same_mengli': dist(same_m), 'jaccard_different_tokens': dist(diff),
       'n_pairs': {'same': len(same), 'mengli': len(same_m), 'different': len(diff)},
       'active_fraction_mengli': dist([S[i].mean() for i, w in enumerate(stream) if w in meng]),
       'active_fraction_other': dist([S[i].mean() for i, w in enumerate(stream) if w not in meng])}
print(json.dumps(rep, indent=1))
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    np.savez_compressed(OUT.with_suffix('.cells.npz'), rate=(counts / nt).cpu().numpy(), v_over_theta=(vsum / nt).cpu().numpy())
