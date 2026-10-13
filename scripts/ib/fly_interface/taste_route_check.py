"""Static check (weights only, CPU) before closing the loop: do the innate taste lines reach dopamine?  Receptor
identities from Vicente et al. 2025 (bioRxiv 10.1101/2025.08.25.671814; typed in this very maleCNS volume): sugar =
LB3b, LB3c (Gr64f) and dorsal_tpGRN (Gr5a); bitter = LB1a-d (Gr33a); water = LB3a (ppk28); high salt = LB3d (Ir7c,
ppk23).  (The first version used guessed types -- dorsal_tpGRN, a Gr5a sugar cell, was counted as bitter.)
Unit activity on each receptor set spread along A = g_e |w_e| + g_i |w_i| for hops 1..6, in ABSOLUTE mass (shares
mislead across birth / trained: every excitatory weight grew 4.4x): mass landing on PAM, PPL1, all dopamine cells and
the 15 largest dopamine types, compared with the same mass from olfactory receptors and from the text-driven cells.
Usage: python taste_route_check.py <checkpoint dir>
"""
import sys, json, numpy as np
from pathlib import Path
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]) / 'taste_route.json'
sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
src = open('D:/information-boltzmann-v1/scripts/ib/fly_interface/static_route_analysis.py', encoding='utf-8').read()
exec(src.split('rep = {')[0])                      # groups, matrix(), drive, ann, cls, typ
types = lambda *t: np.isin(typ, list(t))
dan = cls == 'DAN'
pam = np.char.startswith(typ.astype(str), 'PAM') & dan
ppl1 = np.char.startswith(typ.astype(str), 'PPL1') & dan
S = {'sugar': types('LB3b', 'LB3c', 'dorsal_tpGRN'), 'bitter': types('LB1a', 'LB1b', 'LB1c', 'LB1d'),
     'water': types('LB3a'), 'high_salt': types('LB3d'), 'ORN': np.char.startswith(typ.astype(str), 'ORN_'),
     'text': drive > 0}
print({k: int(v.sum()) for k, v in S.items()}, '| DAN', int(dan.sum()), 'PAM', int(pam.sum()), 'PPL1', int(ppl1.sum()))
rep = {'sizes': {k: int(v.sum()) for k, v in S.items()}}
for label, trained in (('birth', False), ('trained', True)):
    A = matrix(trained)
    rep[label] = {}
    print(f'== {label}')
    for s, m in S.items():
        x = m.astype(float) / m.sum()                # one unit of activity spread over the set
        rows = []
        for k in range(1, 7):
            x = A @ x
            rows.append({'total': float(x.sum()), 'DAN': float(x[dan].sum()), 'PAM': float(x[pam].sum()),
                         'PPL1': float(x[ppl1].sum()),
                         'DAN_share': float(x[dan].sum() / x.sum()),
                         'top_types': sorted(((t, float(x[dan & (typ == t)].sum())) for t in np.unique(typ[dan])),
                                             key=lambda z: -z[1])[:5]})
        rep[label][s] = rows
        print(f'  {s:>9}: DAN mass by hop', ' '.join(f'{r["DAN"]:.2e}' for r in rows),
              '| PAM/PPL1 at hop 3', f'{rows[2]["PAM"]:.2e}/{rows[2]["PPL1"]:.2e}', '| hop-3 top', rows[2]['top_types'][:3])
json.dump(rep, open(out, 'w'), indent=1)
