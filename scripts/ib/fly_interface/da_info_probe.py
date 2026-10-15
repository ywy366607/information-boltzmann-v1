"""Gate 2 at the compartments: does the ordinary words' dopamine carry anything?  Zero-shot on the lineage's
mother text (fixed weights; the online rules run as in life).  Per word: the compartment write drive sum over edges and
ticks of gate * eligibility * suprathreshold phasic (the term that writes phi), and the share of edges written.
Reported: Spearman correlation with the word's unigram surprisal (first 20M training tokens), the share of variance
explained by word identity (words seen >= 3 times), and the trend over repeats of the same word (MBON -> DAN
self-limiting predicts a decline).  Usage: python da_info_probe.py <checkpoint dir> <run dir> [words=512] [out.json]
(env: WORD_TICKS, and the zero-shot ZS_* switches of world_eval)."""
import sys, json, os, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
fork, RUN = Path(sys.argv[1]), Path(sys.argv[2])
N_WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 512
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else None
model, cfg, L = load(fork)
WT = int(os.environ.get('WORD_TICKS') or round(float(cfg.get('stream_pace_mean') or 16)))   # the individual's own word interval
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
if getattr(model, 'dopamine_sensitivity', None) is not None:
    opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, s.h)
import tiktoken
tok = tiktoken.get_encoding('gpt2')
text = []
for r in (json.loads(l) for l in (RUN / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()):
    t = r['text'].strip()
    if t and t != '...' and not t.startswith('Mengli'):
        text += tok.encode(' ' + t)
    if len(text) >= N_WORDS + 64:
        break
train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
counts = np.bincount(np.asarray(train[:20_000_000], dtype=np.int64), minlength=50257).astype(np.float64) + 0.5
surprisal = -np.log(counts / counts.sum())
E = model.comp_pre.numel()
comp_pre_l = model.comp_pre.long()
mbon_idx = torch.unique(model.comp_post.long())
drive, share, words = [], [], []
mbon_resp, phi_word = [], []      # familiarity: the MBONs' response to the word, phi on the edges of its Kenyon cells
with torch.no_grad():
    for i, w in enumerate(text[:64 + N_WORDS]):
        model._comp_rate_probe = {'dep': torch.zeros(E, device='cuda'), 'pot': torch.zeros(E, device='cuda'),
                                  'gate': torch.zeros(E, device='cuda'), 'ticks': 0}
        fired = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
        mb = 0.0
        for t in range(WT):
            s = fl.advance_fly_token_stream(model, s, torch.tensor([int(w)], device='cuda'), [1], ticks=1, hold_input=True,
                                            fresh=False, **opts)[0]
            spk = s.ring[0][0] > 0
            fired |= spk
            mb += float(spk.index_select(0, mbon_idx).float().mean())
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        pr = model._comp_rate_probe
        if i >= 64:                       # the first 64 words settle the threshold's background
            d = pr['dep'] + pr['pot']
            drive.append(float(d.sum())); share.append(float((d > 0).float().mean())); words.append(int(w))
            sel = fired.index_select(0, comp_pre_l)
            phi_word.append(float(s.comp_memory[0][sel].mean()) if bool(sel.any()) else float('nan'))
            mbon_resp.append(mb / WT)
model._comp_rate_probe = None
drive, share, words = np.array(drive), np.array(share), np.array(words)
from scipy.stats import spearmanr
rho, pval = spearmanr(drive, surprisal[words])
uniq, inv, cnt = np.unique(words, return_inverse=True, return_counts=True)
rep = cnt[inv] >= 3
eta2 = None
if rep.sum() > 10:
    dv, iv = drive[rep], inv[rep]
    grand = dv.mean()
    between = sum(((dv[iv == u].mean() - grand) ** 2) * (iv == u).sum() for u in np.unique(iv))
    eta2 = float(between / max(((dv - grand) ** 2).sum(), 1e-12))
occ_idx = np.zeros(len(words), int)
seen = {}
for j, w in enumerate(words):
    occ_idx[j] = seen.get(w, 0); seen[w] = occ_idx[j] + 1
trend = None
mask = cnt[inv] >= 3
if mask.sum() > 10:
    # within-word: drive relative to that word's mean, against its occurrence index
    rel = drive[mask] / np.maximum(np.array([drive[inv == inv[j]].mean() for j in np.flatnonzero(mask)]), 1e-12)
    trend = float(spearmanr(occ_idx[mask], rel)[0])
q = lambda v: [round(float(np.min(v)), 4), round(float(np.median(v)), 4), round(float(np.mean(v)), 4), round(float(np.max(v)), 4)]
# familiarity: within each word seen >= 3 times, the MBONs' response and phi against its occurrence index (a
# repetition-suppression / familiarity code falls with repeats), and phi against rarity across all words
mbon_resp, phi_word = np.array(mbon_resp), np.array(phi_word)
fam_mbon = fam_phi = None
if mask.sum() > 10:
    def rel_to_word(v):
        return np.array([v[j] - np.nanmean(v[inv == inv[j]]) for j in np.flatnonzero(mask)])
    fam_mbon = float(spearmanr(occ_idx[mask], rel_to_word(mbon_resp))[0])
    pv = rel_to_word(phi_word)
    ok = np.isfinite(pv)
    fam_phi = float(spearmanr(occ_idx[mask][ok], pv[ok])[0]) if ok.sum() > 10 else None
okp = np.isfinite(phi_word)
phi_rarity = float(spearmanr(phi_word[okp], surprisal[words][okp])[0]) if okp.sum() > 10 else None
rep_ = {'words': int(len(words)), 'drive [min,med,mean,max]': q(drive), 'edges written share': q(share),
        'spearman drive~surprisal': round(float(rho), 3), 'p': float(pval), 'identity eta2 (>=3 occurrences)': eta2,
        'repeat trend (spearman occurrence~relative drive)': trend, 'distinct words': int(len(uniq)),
        'words seen >=3': int((cnt >= 3).sum()),
        'familiarity: MBON response over repeats (spearman)': fam_mbon,
        'familiarity: phi over repeats (spearman)': fam_phi,
        'phi ~ rarity (spearman)': phi_rarity}
print(json.dumps(rep_, indent=1), flush=True)
if OUT:
    OUT.write_text(json.dumps(rep_, indent=1), encoding='utf-8')
