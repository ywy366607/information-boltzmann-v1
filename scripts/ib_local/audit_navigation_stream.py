"""Offline action-world coverage and age summary from append-only run logs."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();raw=(args.run/'metrics.jsonl').read_bytes()
    # Concurrent training may have appended only part of its final JSON line.
    complete = raw if raw.endswith(b'\n') else raw.rsplit(b'\n', 1)[0]
    rows=[json.loads(line) for line in complete.splitlines() if line.strip()]
    if not rows:raise ValueError('empty run')
    intervals=sorted((r['world']['start'],r['world']['end']) for r in rows)
    coverage=0;end=-1
    for lo,hi in intervals:
        if hi>end:
            coverage+=hi-max(lo,end);end=hi
    starts=collections.Counter(r['world']['start'] for r in rows)
    total=sum(r['external_bytes'] for r in rows)
    result=dict(scope='observed logged response-plus-heldout source intervals, not semantic novelty',
        cycles=len(rows),metrics_sha256=hashlib.sha256(raw).hexdigest(),
        external_bytes=total,unique_world_positions=coverage,repeated_position_fraction=1-coverage/total,
        distinct_starts=len(starts),max_start_repeats=max(starts.values()),
        mean_progress=sum(r['progress'] for r in rows)/len(rows),
        utf8_valid_fraction=sum(r['utf8_valid'] for r in rows)/len(rows),
        matched_query_fraction=sum(r['world']['source']=='matched' for r in rows)/len(rows),
        validation=[dict(cycle=r['cycle'],nll=r['validation_nll'],oldest_mean=r['read_age_oldest_mean'],
                         age_component_ablation=r.get('age_component_ablation'))
                    for r in rows if 'validation_nll' in r])
    if 'holdout_position_fresh' in rows[0]['world']:
        fresh=[r for r in rows if r['world']['holdout_position_fresh']]
        revisit=[r for r in rows if not r['world']['holdout_position_fresh']]
        result['position_freshness']=dict(
            completely_fresh_segments=len(fresh),completely_fresh_fraction=len(fresh)/len(rows),
            mean_previously_seen_holdout_bytes=sum(r['world']['holdout_previously_seen_bytes'] for r in rows)/len(rows),
            fresh_mean_progress=sum(r['progress'] for r in fresh)/len(fresh) if fresh else None,
            revisit_mean_progress=sum(r['progress'] for r in revisit)/len(revisit) if revisit else None,
            definition='source positions only; not semantic or corpus-wide novelty')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
