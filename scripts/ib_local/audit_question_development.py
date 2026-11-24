"""Read committed logs only: separate question reuse from demonstrated inquiry.

No Torch imports, checkpoint loads, world calls, or learner mutations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean


def records(path):
    data = path.read_bytes()
    # An in-flight partial final record is not a committed event.
    lines = data.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith(b'\n') and line.strip()]
    return rows, hashlib.sha256(data).hexdigest()


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        identity = row.get('shared_unknown', {}).get('identity')
        if identity:
            grouped[identity].append(row)
    tv = []
    for row in rows:
        p = row.get('joint_language_search', {}).get('distribution', [])
        if p:
            tv.append(sum(abs(x-1/len(p)) for x in p)/2)
    examples = []
    for identity, group in sorted(grouped.items(), key=lambda pair: -len(pair[1])):
        if len(group) < 2:
            continue
        examples.append(dict(identity=identity, turns=[dict(
            cycle=r['cycle'], choice=r['choice'],
            utterance=bytes.fromhex(r['utterance_hex']).decode('utf8', 'replace'),
            reply=r['world'].get('mother_reply', r['world'].get('text_reply', ''))[:240],
            unknown_version=r['shared_unknown'].get('version'),
            version_after=r['shared_unknown'].get('revision', {}).get('version_after'),
            global_language_gain=r.get('metal_g')) for r in group]))
        if len(examples) == 3:
            break
    return dict(first_cycle=rows[0]['cycle'], last_cycle=rows[-1]['cycle'], events=len(rows),
        updates=sum(r['updates'] for r in rows), spoken_events=sum(bool(r['spoke']) for r in rows),
        selected_roles=dict(Counter(r.get('shared_unknown', {}).get('role') for r in rows)),
        unknown_identities=len(grouped),
        identities_selected_multiple_times_within_segment=sum(len(g)>1 for g in grouped.values()),
        recalled_selections=sum(bool(r.get('shared_unknown', {}).get('recalled')) for r in rows),
        same_fact_contract_passes=sum(bool(r.get('joint_relation_update', {}).get(
            'evidence_contract', {}).get('same_facts')) for r in rows),
        mean_draft_choice_tv_from_uniform=mean(tv) if tv else None,
        examples=examples,
        functional_inquiry='not established by identity/version or global language gain alone')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    small_path=args.root/'results/cgb_content_question_20261114/learning/body/metrics.jsonl'
    current=args.root/'results/cgb_grown_body_20261114/learning_v2'
    old, old_sha = records(small_path)
    live, live_sha = records(current/'metrics.jsonl')
    live=[r for r in live if r['cycle']>5543]
    config=json.loads((current/'last.metadata.json').read_text(encoding='utf8'))['config']
    prefix=config['natural_text_corpus']['prefix_bytes']
    total=prefix+config['natural_text_corpus']['train_bytes']
    fineweb=0
    for r in live:
        start=r['world']['start']
        assert r['world']['source']=='text' and start+r['external_bytes'] <= total
        fineweb+=max(0, min(start+r['external_bytes'], total)-max(start,prefix))
    end=live[-1]['world']['start']+live[-1]['external_bytes']
    report=dict(schema=1, scope='read-only committed history; no additional training or intervention',
        small_body=dict(path=str(small_path), file_sha256=old_sha, **summarize(old)),
        text_body=dict(path=str(current/'metrics.jsonl'), file_sha256_at_read=live_sha,
            **summarize(live), external_bytes=sum(r['external_bytes'] for r in live),
            fineweb_actual_bytes=fineweb, provider_calls=sum(r['world'].get('provider_calls',0) for r in live),
            successful_suffix_moves=sum(r['world'].get('match',0)>0 for r in live),
            fixed_validation_cycles=[r['cycle'] for r in live if r.get('fixed_validation')],
            cycles_to_appended_source_if_no_new_jump=max(0,(prefix-end+959)//960)),
        decision=dict(keep_current_text_growth=True, restore_mother=False,
            missing_acceptance='same unknown: pre-response prediction -> evidence relevance -> held-out later use -> changed seeking',
            implementation_boundary='typed latent condition/action/outcome variables are not identified natural-language questions',
            supply_boundary='appended text availability is distinct from bytes actually consumed'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='examples'} for k,v in report.items()
        if k in ('small_body','text_body')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
