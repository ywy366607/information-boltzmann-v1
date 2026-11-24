"""Read prequential future scores from the completed native pending arms."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    names = ('baseline','pending_mode_cut','pending_matched')
    paths = {n:args.work/n/'result.json' for n in names}
    data = {n:json.loads(p.read_text(encoding='utf8')) for n,p in paths.items()}
    assert all(r['status']=='completed' for r in data.values())
    assert all(r['before']==data['baseline']['before'] for r in data.values())
    count = min(sum(r['autonomous_event']['kind']=='external' for r in d['receipts'])
                for d in data.values())
    rows = {n:[r for r in d['receipts'] if r['autonomous_event']['kind']=='external'][:count]
            for n,d in data.items()}
    fields = ('heldout_pre_nll','heldout_post_nll','joint_progress','witness_pre',
              'witness_post','witness_current_pre','witness_current_post')
    result = dict(status='completed',external_events=count,
        source_sha256={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()},
        scores={},paired_post_intervention_deltas={},same_realized_utterances={})
    for name, arm in rows.items():
        result['same_realized_utterances'][name] = all(a['utterance_hex']==b['utterance_hex']
                                                     for a,b in zip(rows['baseline'],arm))
        result['scores'][name] = [dict(round=i+1,**{k:r.get(k) for k in fields},
            mother=r['causal_program']['mother_prequential'],
            decision=r['causal_program']['decision_prequential']) for i,r in enumerate(arm)]
        if name=='baseline':
            continue
        result['paired_post_intervention_deltas'][name] = [dict(round=i+1,
            heldout_pre_nll=arm[i]['heldout_pre_nll']-rows['baseline'][i]['heldout_pre_nll'],
            mother_log_score=arm[i]['causal_program']['mother_prequential']['log_score']-
                rows['baseline'][i]['causal_program']['mother_prequential']['log_score'],
            decision_log_score=arm[i]['causal_program']['decision_prequential']['log_score']-
                rows['baseline'][i]['causal_program']['decision_prequential']['log_score']) for i in range(1,count)]
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['paired_post_intervention_deltas']))


if __name__=='__main__':
    main()
