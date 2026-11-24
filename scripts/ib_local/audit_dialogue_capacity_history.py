"""Read completed real events without loading or changing the running body."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def mean(values):
    return statistics.mean(values) if values else None


def describe(rows):
    generated = [r for r in rows if r['choice'] == 'generate']
    settled = [r['shadow_preview']['settled_two_turn'] for r in rows
               if (r.get('shadow_preview', {}).get('settled_two_turn') or {}).get('eligible')]
    options = [r['self_action_observer']['options'] for r in rows
               if r.get('self_action_observer', {}).get('available')]
    drafts = [r['self_action_observer']['generated_drafts'] for r in rows
              if r.get('self_action_observer', {}).get('available')
              and r['self_action_observer'].get('generated_drafts') is not None]
    speech = [bytes.fromhex(r['utterance_hex']) for r in generated]
    return dict(events=len(rows), cycles=[rows[0]['cycle'], rows[-1]['cycle']] if rows else [],
        updates=sum(r.get('updates', 0) for r in rows),
        choices=dict(Counter(r['choice'] for r in rows)), generated=len(speech),
        generated_lengths=dict(Counter(len(s) for s in speech)),
        sentence_punctuation_at_end=sum(s.rstrip().endswith((b'.', b'?', b'!')) for s in speech),
        punctuation_anywhere=sum(any(c in s for c in (b'.', b'?', b'!')) for s in speech),
        reported_truncated=sum(bool(r['truncated']) for r in generated),
        mother_bytes_mean=mean([r['world']['mother_bytes'] for r in rows]),
        external_bytes_mean=mean([r['external_bytes'] for r in rows]),
        mother_source_progress_mean=mean([x['pre']-x['post'] for r in rows
            for x in r.get('world_learning', {}).get('readings', []) if x['source']=='mother']),
        ambient_source_progress_mean=mean([x['pre']-x['post'] for r in rows
            for x in r.get('world_learning', {}).get('readings', []) if x['source']=='ambient']),
        recalled=sum(r.get('shared_unknown', {}).get('recalled', False) for r in rows),
        relation_chosen_mse=mean([s['relation_mse'][s['chosen']] for s in options]),
        original_chosen_mse=mean([s['original_mse'][s['chosen']] for s in options]),
        draft_relation_wins=sum(s.get('relation_wins', 0) for s in drafts),
        draft_original_wins=sum(s.get('original_wins', 0) for s in drafts),
        draft_comparisons=sum(len(s['different_alternatives']) for s in drafts),
        shadow_settlements=len(settled),
        shadow_mse=mean([(s['prediction']-s['target'])**2 for s in settled]),
        shadow_baseline_mse=mean([(s['baseline']-s['target'])**2 for s in settled]),
        shadow_abs_offset=mean([abs(v) for r in rows
            for v in r.get('shadow_preview', {}).get('delivered_offset', [])]),
        draft_uniform_tv=mean([sum(abs(p-1/len(r['joint_language_search']['distribution']))
            for p in r['joint_language_search']['distribution'])/2 for r in rows
            if 'joint_language_search' in r]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding', type=Path, default=Path('data/harness/workbench/training-observer.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--current-start', type=int, default=5165)
    args = p.parse_args()
    binding = json.loads(args.binding.read_text(encoding='utf-8'))
    retained, sources, validation = {}, [], []
    for directory in binding['segments']:
        path=Path(directory)/'metrics.jsonl'
        if not path.exists():
            continue
        accepted=0
        with path.open('rb') as stream:
            for raw in stream:
                if not raw.endswith(b'\n'):
                    break  # A concurrent append is not a committed event.
                row=json.loads(raw);cycle=row['cycle']
                if cycle in retained:
                    old=retained[cycle]
                    if any(old.get(k)!=row.get(k) for k in ('utterance_hex','metal_g','shared_unknown')):
                        raise ValueError(f'conflicting histories at {cycle}')
                    continue
                retained[cycle]=row;accepted+=1
                if 'fixed_validation' in row:
                    validation.append(dict(cycle=cycle,**row['fixed_validation']))
        sources.append(dict(path=str(path),events=accepted))
    rows=[retained[k] for k in sorted(retained)]
    questions=defaultdict(list)
    for r in rows:
        q=r.get('shared_unknown')
        if not q:
            continue
        score=r['causal_program'].get('mother_prequential')
        questions[q['identity']].append(dict(cycle=r['cycle'],version=q['version'],
            next_version=q['revision'].get('version_after'),recalled=q['recalled'],
            role=q['role'],choice=r['choice'],words=bytes.fromhex(r['utterance_hex']).decode('utf-8','replace'),
            reply=r['world']['mother_reply'],mse=None if score is None else score['mse'],
            log_score=None if score is None else score['log_score'],
            observed_episode=q['revision'].get('observed_episode')))
    repeated={k:v for k,v in questions.items() if len(v)>1}
    ordered=sorted(repeated.items(),key=lambda x:len(x[1]),reverse=True)
    current=[r for r in rows if r['cycle']>=args.current_start]
    mid=len(current)//2
    config=json.loads((Path(binding['body'])/'config.json').read_text(encoding='utf-8'))
    report=dict(schema=1,scope='read-only factual history; no new events, inference or optimizer updates',
        individual_id=binding['individual_id'],branch_id=binding['branch_id'],
        parameter_audit=config.get('mother_parameter_audit'),
        sources=sources,all_events=len(rows),current=describe(current),
        current_first_half=describe(current[:mid]),current_second_half=describe(current[mid:]),
        fixed_validation=validation,unique_unknowns=len(questions),revisited_unknowns=len(repeated),
        max_visits=max(map(len,questions.values()),default=0),
        examples=[dict(identity=k,events=v) for k,v in ordered[:8]],
        interpretation=dict(sentence_end='32-byte budget is not a learned sentence boundary',
            repeated_questions='versioned use and factual scores; changing actions/replies are not matched tests',
            size='no controlled parameter/data scaling comparison in this audit; do not assign sole cause to size'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('all_events','current','current_first_half',
        'current_second_half','unique_unknowns','revisited_unknowns','max_visits')},ensure_ascii=False,indent=2))
    print('validation_tail',json.dumps(validation[-4:],ensure_ascii=False))


if __name__=='__main__':
    main()
