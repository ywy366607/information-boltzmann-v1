"""Summary of the odour world's encounters (odour.jsonl), per chunk of training words.
Per chunk: encounters; act shares (approach / avoid / none / drift); first encounter of an odour in its segment vs
revisits after >= 1 outcome: P(approach | sweet) - P(approach | bitter) and P(avoid | bitter); mean fitness; moves
made in the gaps per encounter; latency of acts.  Reversed encounters are reported apart.
Usage: python odour_summary.py <odour.jsonl> [chunk words]
"""
import sys, json, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
if not rows:
    sys.exit('no encounters')
t0 = min(r['bptt_train_tokens'] for r in rows)


def p(sel, act):
    sel = list(sel)
    return (sum(r['act'] == act for r in sel) / len(sel), len(sel)) if sel else (float('nan'), 0)


out = []
for c in sorted({(r['bptt_train_tokens'] - t0) // chunk for r in rows}):
    rs = [r for r in rows if (r['bptt_train_tokens'] - t0) // chunk == c]
    rep = {'chunk': int(c), 'encounters': len(rs), 'fitness_mean': float(np.mean([r['fitness'] for r in rs])),
           'moves_per_encounter': float(np.mean([r['moves_before'] for r in rs]))}
    for a in ('approach', 'avoid', 'none', 'drift'):
        rep[f'share_{a}'] = float(np.mean([r['act'] == a for r in rs]))
    for name, cond in (('first', lambda r: r['prior_outcomes'] == 0), ('revisit', lambda r: r['prior_outcomes'] >= 1)):
        base = [r for r in rs if cond(r) and not r['reversed']]
        ps, ns = p((r for r in base if r['valence'] == 'sweet'), 'approach')
        pb, nb = p((r for r in base if r['valence'] == 'bitter'), 'approach')
        vb, _ = p((r for r in base if r['valence'] == 'bitter'), 'avoid')
        rep[f'{name}_approach_sweet_minus_bitter'] = ps - pb
        rep[f'{name}_avoid_given_bitter'] = vb
        rep[f'{name}_n_sweet_bitter'] = [ns, nb]
    lat = [r['latency'] for r in rs if r['act'] in ('approach', 'avoid')]
    rep['act_latency_median'] = float(np.median(lat)) if lat else None
    out.append(rep)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()}))
