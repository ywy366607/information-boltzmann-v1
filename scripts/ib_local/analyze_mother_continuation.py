"""Summarize actual paired mother-world trajectories, not just NLL."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def load(paths):
    rows=[]
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line)
    cycles=[r['cycle'] for r in rows]
    if not rows or cycles != list(range(cycles[0],cycles[0]+len(cycles))):
        raise ValueError('each arm must contain exactly one contiguous event history')
    return rows


def mean(values):
    return statistics.fmean(values) if values else None


def describe(rows):
    supported=[r for r in rows if r.get('causal_read_support',[0,0,0,0])[3]>0]
    return dict(cycles=len(rows),from_cycle=rows[0]['cycle'],to_cycle=rows[-1]['cycle'],
        choices=dict(Counter(r['choice'] for r in rows)),
        mother_bytes=sum(r['world']['mother_bytes'] for r in rows),
        ambient_navigation_matches=sum(r['world'].get('ambient_match',0)>0 for r in rows),
        eligible_gain_reads=len(supported),
        mean_selected_direct_gain_error=mean([r['episodic_used_real_error'][3] for r in supported]),
        mean_selected_shuffled_gain_error=mean([r['episodic_used_shuffled_error'][3] for r in supported]),
        wood_replay_examples=sum(r.get('wood_replay_samples',0) for r in rows),
        wood_reward_replay_examples=sum(r.get('wood_reward_replay_samples',0) for r in rows),
        mean_short_latent_loss=mean([r['short_loss'] for r in rows]),
        mean_long_latent_loss=mean([r['long_loss'] for r in rows]),
        mean_response_nll=mean([r['response_nll'] for r in rows]),
        note='latent losses and same-trajectory read errors are diagnostics, not proof of causal understanding')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real',type=Path,nargs='+',required=True)
    p.add_argument('--blocked',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    real,blocked=load(args.real),load(args.blocked)
    if [r['cycle'] for r in real] != [r['cycle'] for r in blocked]:
        raise ValueError('paired arms must cover the same completed cycle numbers')
    def first(predicate):
        return next((a['cycle'] for a,b in zip(real,blocked) if predicate(a,b)),None)
    report=dict(status='complete',real=describe(real),blocked=describe(blocked),
        first_choice_divergence=first(lambda a,b:a['choice']!=b['choice']),
        first_utterance_divergence=first(lambda a,b:a['utterance_hex']!=b['utterance_hex']),
        first_mother_reply_divergence=first(lambda a,b:a['world']['mother_reply']!=b['world']['mother_reply']),
        first_ambient_anchor_divergence=first(lambda a,b:a['world']['anchor']!=b['world']['anchor']),
        same_utterance=sum(a['utterance_hex']==b['utterance_hex'] for a,b in zip(real,blocked)),
        same_mother_reply=sum(a['world']['mother_reply']==b['world']['mother_reply'] for a,b in zip(real,blocked)),
        control='blocks qualified experience-to-choice increment; other loops remain active',
        context='ambient policy explicitly changed at the same cycle in both arms',
        evidence='single continuing source, one seed, short run; not sufficient training')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
