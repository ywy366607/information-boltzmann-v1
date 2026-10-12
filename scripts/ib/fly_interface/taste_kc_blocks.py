"""Why do the cues barely reach the Kenyon cells inside a taste episode?  Fixed weights.  One episode layout as in
taste_episodes.py, run from the same state four ways: {pulsed, held cues} x {with, without the dopamine-cell drive};
per 32-tick block, the fraction of Kenyon cells that spike, their spikes per cell, and the same for plain text read on
the same clock (reference).  Also the dopamine-driven threshold factor at the Kenyon cells (exp(-s phasic)).
Usage: python taste_kc_blocks.py <fork dir>
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
src = open('D:/information-boltzmann-v1/scripts/ib/fly_interface/taste_episodes.py', encoding='utf-8').read()
src = src.split('rng = np.random.default_rng(0)')[0]
sys.argv = [sys.argv[0], sys.argv[1], '1']
exec(src)
rng = np.random.default_rng(5)
res = {}
with torch.no_grad():
    s, _ = run_tokens(s, text[:256])
    base_state = s
    cues = tuple(int(c) for c in rng.choice(pool, 2, replace=False))
    dist = text[1000:1008]
    toks = [cues[0]] * 4 + list(dist[:4]) + [cues[1]] * 4 + list(dist[4:]) + [cues[0]] * 4 + [QUESTION] * 4 + [cues[1]] * 4 + [QUESTION] * 4
    drive_at = {t: drives['sweet'] for t in range(16, 41)}; drive_at.update({64 + t: drives['bitter'] for t in range(16, 41)})
    for hold in (False, True):
        for dname, dmap in (('drive', drive_at), ('no drive', None)):
            HOLD = hold
            _, rec = run_tokens(base_state, toks, dmap, record=True, held=cues)
            KC = rec[1].float()
            blocks = [{'active_fraction': float(KC[i:i + 32].any(0).float().mean()), 'spikes_per_cell': float(KC[i:i + 32].sum(0).mean())}
                      for i in range(0, 256, 32)]
            res[f"{'held' if hold else 'pulsed'} / {dname}"] = blocks
            print(f"{'held' if hold else 'pulsed'} / {dname}", [round(b['active_fraction'], 4) for b in blocks], flush=True)
    HOLD = False
    _, rec = run_tokens(base_state, text[2000:2032], None, record=True)
    KC = rec[1].float()
    res['text'] = [{'active_fraction': float(KC[i:i + 32].any(0).float().mean()), 'spikes_per_cell': float(KC[i:i + 32].sum(0).mean())}
                   for i in range(0, 256, 32)]
    print('text', [round(b['active_fraction'], 4) for b in res['text']], flush=True)
    ph = None
    if base_state.dopamine.numel() == 2 * N and getattr(model, 'dopamine_sensitivity', None) is not None:
        res['dopamine_sensitivity_kc'] = float(model.dopamine_sensitivity[model.superclass_id[kc_idx]].mean())
    res['gestation_kc_rate_note'] = 'compare with gestation.jsonl Kenyon Hz'
json.dump(res, open(fork / 'taste_kc_blocks.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)
