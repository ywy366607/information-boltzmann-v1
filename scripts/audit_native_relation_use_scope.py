"""Inspect actual scope recurrence before adding delayed retention evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    metadata = json.loads((args.work/'body/last.metadata.json').read_text(encoding='utf8'))
    events = [json.loads(x) for x in (args.work/'events.jsonl').read_text(encoding='utf8').splitlines()]
    rows = [r for r in events if r['autonomous_event']['kind']=='external']
    chain = []
    for r in rows:
        q = r['shared_unknown']
        chain.append(dict(cycle=r['cycle'], relation=q['relation_identity'], unknown=q['identity'],
            recalled=q['recalled'], version=q['revision'].get('version_before'),
            fact=r['causal_program']['committed_episode_identity'],
            supports=r['causal_use']['finite_U'].get('supports',[]),
            read_targets=[x['target_fact_id'] for x in r.get('read_use',[])]))
    path = Path(metadata['config']['causal_archive']['snapshot'])
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        counts = {name:db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0]
                  for name in ('relations','unknown_bindings','unknown_turns','relation_variables')}
        repeats = db.execute('SELECT unknown_id,COUNT(*) FROM unknown_turns GROUP BY unknown_id '
                             'HAVING COUNT(*)>1 ORDER BY COUNT(*) DESC LIMIT 8').fetchall()
        dependent = db.execute('SELECT relation_id,COUNT(*) FROM relation_variables GROUP BY relation_id '
                              'ORDER BY COUNT(*) DESC LIMIT 8').fetchall()
    report = dict(cycle=metadata['cycle'], chain=chain, archive_counts=counts,
        relation_uses=dict(Counter(str(r['relation']) for r in chain)),
        unknown_uses=dict(Counter(r['unknown'] for r in chain)),
        historical_repeated_unknowns=repeats, largest_relation_dependencies=dependent,
        optimizer_updates=0,world_calls=0)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
