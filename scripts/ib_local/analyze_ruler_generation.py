"""Read-only rollout diagnostics alongside the unmodified official score."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def numeric_scores(prediction, references):
    if len(references)!=1 or not references[0].isdigit():
        raise ValueError('this numeric diagnosis applies to single numeric targets')
    gold=references[0]
    first=re.search(r'\d+',prediction)
    number=None if first is None else first.group()
    return dict(official_contains=gold in prediction,
        standalone_number=bool(re.search(r'(?<!\d)'+re.escape(gold)+r'(?!\d)',prediction)),
        first_number_exact=number==gold,
        first_number_prefix=number is not None and number[:len(gold)]==gold,
        first_number=number)


def summarize(cell):
    cases=[]
    for s in cell['samples']:
        cases.append(dict(index=s['index'],gold=s['references'],
            prediction_prefix=s['prediction'][:100],generated_tokens=s['generated_tokens'],
            hit_generation_cap=(not s.get('stopped_on_eos',False) and
                s['generated_tokens']==s.get('max_new_tokens',128)),
            answer_distance_tokens=s.get('answer_distance_tokens'),
            **numeric_scores(s['prediction'],s['references'])))
    return dict(length=cell['length'],samples=len(cases),official_score=cell['ruler_score'],
        answer_nll=cell['answer_nll'],
        counts={k:sum(bool(c[k]) for c in cases) for k in (
            'official_contains','standalone_number','first_number_exact','first_number_prefix','hit_generation_cap')},
        cases=cases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raw=args.report.read_bytes()
    report=json.loads(raw)
    result=dict(scope='derived rollout diagnosis, not a new official RULER metric or credit success claim',
        source_status=report['status'],source_report=str(args.report),
        source_report_sha256=hashlib.sha256(raw).hexdigest(),
        initial=[summarize(c) for c in report.get('shared_initial_evaluation',[])],arms={})
    for name,arm in report['arms'].items():
        result['arms'][name]=dict(training_examples=len(arm['steps']),
            evaluations=[dict(example=e['example'],results=[summarize(c) for c in e['results']])
                         for e in arm['evaluations']])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for tag,cells in [('initial',result['initial']),*[(f'{a}_{e["example"]}',e['results'])
            for a,v in result['arms'].items() for e in v['evaluations']]]:
        print(tag,[(c['length'],c['counts']) for c in cells])


if __name__=='__main__':
    main()
