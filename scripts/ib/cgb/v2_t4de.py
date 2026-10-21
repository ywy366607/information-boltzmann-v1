"""V2 / T4d + T4e - after T4c: a sliding threshold (BCM) for the tags, then growth balanced by turnover.

T4c: tag capture favoured the correlated group at equal rates (1.8x) but not at 3x the random rate (0.83x): the
template's tags are BCM-like (more spikes, more potentiation). Nature's fix is the sliding threshold: a slow pool of
the cell's own mean activity raises the potentiation threshold, so only coincidences well above the cell's usual
calcium potentiate. Implemented as thK x (1 + A/A0), A = slow mean rate (tau 5 s), A0 = 5 Hz.
Predictions written before running:
 T4d (tags only, E6 off, tag capture, f 0.3): P3d correlated / random >= 2 at random 4 Hz AND at 12 Hz
 T4e (E6 on: tags grow structure, 180 s, tag capture, f 0.3, random 4 Hz):
     P4e total structure settles (last three 10 s samples within 10%); halving q0 lowers the plateau by >= 20%;
     correlated / random >= 2
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from information_boltzmann.cgb.medium import Medium, CellClass, SynapseParams, TrophicParams
src = open(Path(__file__).with_name('v2_t4c_capture.py'), encoding='utf-8').read().split('out = {}')[0]
ns = {'__file__': str(Path(__file__).with_name('v2_t4c_capture.py'))}; exec(src, ns)
import information_boltzmann.cgb.medium as med

SLIDE = dict(slide_A0=5.0, slide_tau=5000.0)
_orig = med.SynapseParams


def patched(eta):
    def make(**kw):
        kw.update(SLIDE); kw['eta'] = eta
        return _orig(**kw)
    return make


out = {}
# T4d: tags only, with the sliding threshold
ns['SynapseParams'] = patched(0.0)
m, _ = ns['run']('tag', 4.0, 20000.0)
T_pilot_d = float(m.T.mean()); out['T4d_T_pilot'] = T_pilot_d
for rand_hz in (4.0, 12.0):
    _, s = ns['run']('tag', rand_hz, 120000.0, 0.3, T_pilot_d)
    out[f'T4d_r{rand_hz:g}'] = s
    print('T4d', rand_hz, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
# T4e: growth on (E6), same calibration rule
ns['SynapseParams'] = patched(4e-5)
m, _ = ns['run']('tag', 4.0, 20000.0)
T_pilot_e = float(m.T.mean()); out['T4e_T_pilot'] = T_pilot_e
for q0 in (1.0, 0.5):
    _, s = ns['run']('tag', 4.0, 180000.0, 0.3, T_pilot_e, q0=q0)
    tr = s['total_every_10s']; last = tr[-3:]
    s['plateau_spread_last3'] = (max(last) - min(last)) / max(np.mean(last), 1e-12)
    out[f'T4e_q{q0:g}'] = s
    print('T4e q0', q0, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
out['T4e_plateau_ratio_half_over_full'] = out['T4e_q0.5']['total_every_10s'][-1] / max(out['T4e_q1']['total_every_10s'][-1], 1e-12)
print('half/full plateau', out['T4e_plateau_ratio_half_over_full'])
Path('F:/fly_stream_overflow/cgb_v2/t4de.json').write_text(json.dumps(out, indent=1))
