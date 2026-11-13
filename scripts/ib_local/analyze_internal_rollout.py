"""Read committed events; compare predictions made before observed outcomes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean


def average(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return mean(values) if values else None


def summarize(rows):
    packets = [r['internal_rollout'] for r in rows]
    settled = [p['settled_two_turn'] for p in packets if p.get('settled_two_turn')]
    paired = [p for p in packets if p.get('unchanged_state_prequential_mse') is not None]
    return dict(
        cycles=len(rows), last_cycle=rows[-1]['cycle'] if rows else None,
        choices=dict(Counter(r['choice'] for r in rows)),
        external_bytes=sum(r['external_bytes'] for r in rows),
        optimizer_updates=sum(r.get('updates',0) for r in rows),
        response_nll=average([r.get('response_nll') for r in rows]),
        observed_joint_progress=average([r['metal_g'] for r in rows]),
        boundary_mse=average([p['next_boundary_prequential_mse'] for p in packets]),
        paired_boundary_mse=average([p['next_boundary_prequential_mse'] for p in paired]),
        unchanged_state_mse=average([p['unchanged_state_prequential_mse'] for p in paired]),
        other_action_mse=average([p['other_action_prequential_mse'] for p in paired]),
        paired_boundary_count=len(paired),
        settled_two_turn_count=len(settled),
        settled_mse={name:average([(s[name]-s['target'])**2 for s in settled])
                     for name in ('baseline','prediction','uncalibrated')},
        delivered_events=sum(any(v != 0 for v in p['delivered_offset']) for p in packets),
        max_delivered_logit=max((abs(v) for p in packets for v in p['delivered_offset']), default=0.),
        mean_choice_probability_tv=average([p.get('choice_probability_tv') for p in packets]),
        final_valid_boundary_pairs=packets[-1]['valid_boundary_pairs'] if packets else 0,
        mother_bytes=sum(r['world'].get('mother_bytes',0) for r in rows))


def analyze(root):
    state=json.loads((root/'status.json').read_text(encoding='utf-8'))
    rows={mode:[] for mode in ('real','observe')}
    for record in state['completed']:
        path=Path(record['output'])/'metrics.jsonl'
        rows[record['mode']].extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    report=dict(status=state['status'], arms={},
        scope='single continuous trajectory per arm; action divergence changes visited data',
        settlement='inherited joint OWT continuation; not mother-domain reward',
        boundary_scope='fixed decision-time encoder per pair; raw body itself keeps learning',
        other_action_scope='alternative predicted action on actual outcome; not a separate world intervention')
    for mode, data in rows.items():
        report['arms'][mode]=dict(all=summarize(data),
            blocks=[dict(begin=i+1,end=min(i+64,len(data)),**summarize(data[i:i+64]))
                    for i in range(0,len(data),64)])
    pairs=list(zip(rows['real'],rows['observe']))
    report['matched_cycles']=len(pairs)
    report['first_different_utterance_cycle']=next((a['cycle'] for a,b in pairs
        if a['utterance_hex']!=b['utterance_hex']),None)
    report['validation']={p.stem:json.loads(p.read_text(encoding='utf-8'))
                          for p in sorted(root.glob('eval_*.json'))}
    (root/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.root)
    print(json.dumps({k:v['all'] for k,v in report['arms'].items()},ensure_ascii=False))


if __name__=='__main__':
    main()
