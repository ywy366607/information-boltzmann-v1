"""Living causal gate (2): does surprise reach dopamine in the learning individual?  From a --score-log written with
--log-word-dopamine: per window, scores[j] is the surprise of word j+1 (input j+1 = target j) and word_da[j+1] the
phasic dopamine (fast - slow) / slow and the dopamine-cell firing at the end of word j+1's ticks.
Spearman over all pairs, and per 1024-word block [min | median | mean | max].
Surprise in the stream is mostly word rarity (m_main c6-c7: Spearman(surprise, unigram CE) 0.97), so also: dopamine
against rarity alone (unigram CE) and against the brain's own part, the residual surprise - unigram CE (what the
context made more or less expected than the word's frequency).  Usage: python live_da_report.py <scores.jsonl>"""
import sys, json, numpy as np


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')


rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
rows = [r for r in rows if r.get('word_da')]
pairs = {'same': [], 'next': [], 'cells': [], 'rarity_same': [], 'residual_same': []}
blocks = []
for i, r in enumerate(rows):
    s, da = np.array(r['scores']), np.array(r['word_da'])
    same = [(s[j], da[j + 1, 0]) for j in range(len(s) - 1)]
    nxt = [(s[j], da[j + 2, 0]) for j in range(len(s) - 2)]
    cells = [(s[j], da[j + 1, 1]) for j in range(len(s) - 1)]
    pairs['same'] += same; pairs['next'] += nxt; pairs['cells'] += cells
    if r.get('unigram') is not None:
        u = np.array(r['unigram'])
        pairs['rarity_same'] += [(u[j], da[j + 1, 0]) for j in range(len(s) - 1)]
        pairs['residual_same'] += [(s[j] - u[j], da[j + 1, 0]) for j in range(len(s) - 1)]
    if (i + 1) % 32 == 0:
        blocks.append(spearman(*zip(*pairs['same'][-31 * 32:])))
rep = {'words': sum(len(r['scores']) + 1 for r in rows)}
for k, v in pairs.items():
    rep[f'rho_surprise_{k}'] = round(spearman(*zip(*v)), 4) if v else None
if blocks:
    b = np.array(blocks)
    rep['rho_same_per_1024_min_median_mean_max'] = [round(float(x), 4) for x in (np.nanmin(b), np.nanmedian(b), np.nanmean(b), np.nanmax(b))]
ph = np.concatenate([np.array(r['word_da'])[:, 0] for r in rows]) if rows else np.zeros(0)
rep['phasic_sd'] = round(float(ph.std()), 4) if ph.size else None
# B (her answer's dopamine pulse): phasic dopamine on the words that carried it against all other words
# (reply_onsets per word, aligned with word_da; written since 2026-10-16)
on, off = [], []
for r in rows:
    ro = r.get('reply_onsets')
    if ro and len(ro) == len(r['word_da']):
        da = np.array(r['word_da'])[:, 0]
        on += list(da[np.array(ro, bool)]); off += list(da[~np.array(ro, bool)])
if on and off:
    rep['B_onset_words'] = len(on)
    rep['B_phasic_onset_over_other'] = round(float(np.mean(on) / max(abs(np.mean(off)), 1e-9)), 3)
print(json.dumps(rep))
