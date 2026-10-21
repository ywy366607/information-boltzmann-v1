"""V2 / T4b - pre-registered after T4 collapsed: find the scarcity band, then compare the release rules there.

T4: with T* at the blind pilot's mean support every terminal counted as starved; structure peaked at 10 s and decayed
at lambda0 in all four conditions (P4 failed). The framework predicts adaptation only between too scarce (the loop
falls apart) and too rich (no competition). Here each rule's own blind pilot support level T_pilot is multiplied by a
scarcity factor f (T* = f T_pilot; small f = rich).
Predictions written before running:
 P4' some f gives structure that neither vanishes nor is all kept (final total 10-90% of its peak)
 P1' at the f where both rules keep a similar total, residual: predictors / random >= 2; flux: <= 1
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import TrophicParams
src = open(Path(__file__).with_name('v2_t4_trophic.py'), encoding='utf-8').read().split('out = {}')[0]
ns = {'__file__': str(Path(__file__).with_name('v2_t4_trophic.py'))}; exec(src, ns)
run, summary = ns['run'], ns['summary']

out = {}
for rule in ('residual', 'flux'):
    pilot = TrophicParams(rule=rule, lambda0=0.0, T_star=1.0, q_flux=1.0)
    m, *_ = run(rule, 5.4, 20000.0, pilot)
    T_pilot = float(m.T[m.trophic_edges].mean())
    out[f'{rule}_T_pilot'] = T_pilot
    for f in (0.3, 0.1, 0.03, 0.01):
        tp = TrophicParams(rule=rule, T_star=f * T_pilot, q_flux=1.0)
        s = summary(*run(rule, 5.4, 120000.0, tp))
        traj = s['total_structure_every_10s']
        s['final_over_peak'] = traj[-1] / max(traj)
        out[f'{rule}_f{f}'] = s
        print(rule, f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != 'total_structure_every_10s'},
              'traj', traj, flush=True)
Path('F:/fly_stream_overflow/cgb_v2/t4b_scarcity.json').write_text(json.dumps(out, indent=1))
