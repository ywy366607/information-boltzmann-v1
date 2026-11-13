"""Locate labelled fact tokens in the actually selected local capsules.

This CPU provenance check does not infer contents of a held recurrent boundary.
It distinguishes explicit local input participation from possible boundary-only
memory. It never supplies these researcher labels to the learner.
"""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def fact_span(row, tokenizer):
    text = row['input']+row.get('answer_prefix','')
    answer = row['outputs'][0]
    start = text.find(answer)
    if start<0 or text.find(answer,start+len(answer))>=0:
        raise ValueError('one observed annotated fact occurrence required')
    encoded = tokenizer.encode(text,add_special_tokens=False)
    indices=[i for i,(a,b) in enumerate(encoded.offsets) if a<start+len(answer) and b>start]
    if not indices or row.get('token_position_answer',indices[0])!=indices[0]:
        raise ValueError('official fact position and local tokenization disagree')
    return indices[0],indices[-1]


def classify_capsule(source_time, fact_start, fact_end, length=65):
    first=source_time-length+1
    if first<=fact_start and fact_end<=source_time:
        return 'fact_in_reexecuted_input'
    if fact_end<first:
        return 'fact_precedes_held_boundary'
    if source_time<fact_start:
        return 'capsule_precedes_fact'
    return 'partial_fact_in_input'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','manifest','report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-life',type=Path,action='append',default=[])
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    data=next(x for x in manifest['data'] if x['task']=='niah_single_1' and x['length']==2048)
    tasks=[json.loads(line) for line in Path(data['path']).read_text().splitlines()]
    tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
    raw=json.loads(args.report.read_text())
    prior_answers=set()
    for path in args.prior_life:
        previous=json.loads(path.read_text())
        for case in previous['arms']['full']['cases']:
            prior_answers.update(case['references'])
    results={}
    for arm,body in raw['arms'].items():
        cases=[]
        seen=set(prior_answers)
        for case in body['cases']:
            task=tasks[case['source_index']]
            if task['outputs']!=case['references']:
                raise ValueError('task identity mismatch')
            lo,hi=fact_span(task,tokenizer)
            fact_start,fact_end=case['incoming_seen']+lo,case['incoming_seen']+hi
            layers={}
            for event in case['settlement']['rows']:
                for layer,read in event['reads'].items():
                    if read['queries']!=len(read['slots']):
                        raise ValueError('missing actual per-position reads')
                    ages=iter(read['source_ages'])
                    query_start=event['decision_seen']-read['queries']
                    counts=layers.setdefault(layer,{})
                    for position,slots in enumerate(read['slots']):
                        for _ in slots:
                            age=next(ages)
                            source=query_start+position-age
                            label=classify_capsule(source,fact_start,fact_end)
                            counts[label]=counts.get(label,0)+1
                    if next(ages,None) is not None:
                        raise ValueError('age identities do not align with occupied slots')
            cases.append(dict(case=case['case'],references=case['references'],
                new_answer_in_reported_life=not bool(set(case['references'])&seen),
                actual_fact_token_range=[fact_start,fact_end],
                fact_distance_to_answer=case['settlement']['rows'][0]['decision_seen']-fact_end,
                actual_selected_participations=layers,correct=case['first_number_exact']))
            seen.update(case['references'])
        results[arm]=cases
    result=dict(source_report=str(args.report),source_status=raw['status'],arms=results,
        interpretation='Explicit local input coverage only; boundary-only memory can still contain the fact.',
        protocol='Read cadence1; source index64; 65-token capsules; no GPU or learner update.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
