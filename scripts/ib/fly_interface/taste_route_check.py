"""Static check (weights only, CPU) before closing the loop: do the innate taste lines reach dopamine?  From the sweet
taste receptor types (LgAG3, LB1b, PhG13) and the bitter ones (LgAG1, PhG8, dorsal_tpGRN), unit mass spread along
A = g_e |w_e| + g_i |w_i| for hops 1..6: the share of each hop's mass on PAM, PPL1, all dopamine cells, the sweet /
bitter dopamine types of the odour world, Kenyon cells, MBONs; compared with olfactory receptors (the odour line) and
the text-driven sensory cells; birth vs trained weights.  Also the dopamine cells' own inputs: which share comes from
gustatory receptors within 3 hops.  Usage: python taste_route_check.py <checkpoint dir>
"""
import sys, numpy as np
from pathlib import Path
sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
src = open('D:/information-boltzmann-v1/scripts/ib/fly_interface/static_route_analysis.py', encoding='utf-8').read()
exec(src.split('rep = {')[0])                      # groups, matrix(), drive, ann
pam = np.char.startswith(typ.astype(str), 'PAM') & (cls == 'DAN')
ppl1 = np.char.startswith(typ.astype(str), 'PPL1') & (cls == 'DAN')
types = lambda *t: np.isin(typ, list(t))
G = {'sweet_GRN': types('LgAG3', 'LB1b', 'PhG13'), 'bitter_GRN': types('LgAG1', 'PhG8', 'dorsal_tpGRN'),
     'ORN': np.char.startswith(typ.astype(str), 'ORN_'), 'text_sensory': drive > 0}
T = {'PAM': pam, 'PPL1': ppl1, 'DAN_all': cls == 'DAN', 'sweet_DAN': types('PAM01', 'PAM02', 'PAM11'),
     'bitter_DAN': types('PPL101', 'PPL103', 'PPL106'), 'KC': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON'}
print({k: int(v.sum()) for k, v in {**G, **T}.items()})
for label, trained in (('birth', False), ('trained', True)):
    A = matrix(trained)
    print(f'== {label}')
    for s, m in G.items():
        x = m.astype(float) / m.sum()
        rows = []
        for k in range(1, 7):
            x = A @ x
            tot = x.sum()
            rows.append({t: x[tm].sum() / tot for t, tm in T.items()})
        print(f'  {s:>12}:', ' | '.join(f'{t} ' + ' '.join(f'{r[t]:.4f}' for r in rows[:5]) for t in ('sweet_DAN', 'bitter_DAN', 'PAM', 'PPL1', 'KC')))
