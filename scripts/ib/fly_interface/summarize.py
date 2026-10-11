"""Four-pillar summary of the streaming-clock forks: [min | median | mean | max] over the active evaluations."""
import json, sys
from pathlib import Path
import numpy as np

root = Path(__file__).parent
arms = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ctrl_adaptive', 'stream4', 'stream3', 'stream3_gamma', 'stream3_hold_gamma']


def stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    return '—' if not len(v) else f'{v.min():.3f} | {np.median(v):.3f} | {v.mean():.3f} | {v.max():.3f}'


rows = {}
for arm in arms:
    d = root / arm
    if not (d / 'lifelong_evaluation.jsonl').exists():
        continue
    ev = [json.loads(l) for l in open(d / 'lifelong_evaluation.jsonl', encoding='utf-8') if l.strip()]
    met = [json.loads(l) for l in open(d / 'metrics.jsonl', encoding='utf-8') if l.strip()]
    prog = json.load(open(d / 'progress.json', encoding='utf-8'))
    windows = [m['window_prequential_nll'] for m in met if 'window_prequential_nll' in m]
    ms = [m['milliseconds_per_token'] for m in met if 'milliseconds_per_token' in m][1:]   # first window compiles
    rows[arm] = {
        'evals': len(ev),
        'first_pass_phase_nll': prog.get('phase_prequential_nll'),
        'first_pass_first_quarter': float(np.mean(windows[:len(windows) // 4])) if windows else None,
        'first_pass_last_quarter': float(np.mean(windows[-len(windows) // 4:])) if windows else None,
        'B_live_nll': stats([e['live_prequential_nll'] for e in ev]),
        'AG': stats([e['recovery'].get('ag') for e in ev]),
        'half_recovery_tokens': stats([e['recovery'].get('half_recovery_tokens') for e in ev]),
        'A_revisit_saving': stats([e['savings'].get('nll_saving') for e in ev]),
        'matched_revisit_change': stats([e['matched_revisit_nll_change'] for e in ev]),
        'firing_rate': stats([e['firing_rate'] for e in ev]),
        'field_energy': stats([e['field_energy'] for e in ev]),
        'effective_rank': stats([e['centered_effective_rank'] for e in ev]),
        'ms_per_training_token': float(np.median(ms)) if ms else None,
        'ticks_per_token': prog.get('physical_ticks_per_input') if prog.get('stream_ticks') else
                           1 + (prog.get('adaptive_ticks_mean') or 0),
        'status': prog.get('status')}
# paired deltas against the first arm on the same fresh B blocks and the same training windows
base = arms[0]
def curve(arm, key):
    d = root / arm
    if key == 'B':
        return [json.loads(l)['live_prequential_nll'] for l in open(d / 'lifelong_evaluation.jsonl', encoding='utf-8') if l.strip()]
    return [json.loads(l)['window_prequential_nll'] for l in open(d / 'metrics.jsonl', encoding='utf-8')
            if l.strip() and 'window_prequential_nll' in json.loads(l)]
for arm in arms[1:]:
    if arm not in rows or base not in rows:
        continue
    b0, b1 = curve(base, 'B'), curve(arm, 'B')
    n = min(len(b0), len(b1))
    rows[arm]['paired_B_delta_per_eval'] = [round(b1[i] - b0[i], 3) for i in range(n)]
    w0, w1 = curve(base, 'w'), curve(arm, 'w')
    n = min(len(w0), len(w1)); q = max(1, n // 4)
    rows[arm]['paired_first_pass_delta_by_quarter'] = [round(float(np.mean(np.array(w1[j:j + q]) - np.array(w0[j:j + q]))), 3)
                                                       for j in range(0, q * 4, q)]
out_name = sys.argv[2] if len(sys.argv) > 2 else 'summary.json'
json.dump(rows, open(root / out_name, 'w'), indent=1)
for arm, r in rows.items():
    print(f'== {arm}')
    for k, v in r.items():
        print(f'  {k:28s} {v if not isinstance(v, float) else round(v, 4)}')
