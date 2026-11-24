"""Audit the complete normal learning-event pair and its actual signal map.

Read-only recorded data analysis. Support-membership rank is a property of
this message attribution map, not a model-capacity or semantic-rank claim.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_native_dowker_measurement import rank_integer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    if isinstance(value, dict):
        if 'tensor_ref' in value:
            r = value['tensor_ref']
            return r['sha256'], r['dtype'], r['shape']
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def signature(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def support_rank(groups, facts):
    return rank_integer([{i: 1 for i, fact in enumerate(facts) if fact in group}
                         for group in groups])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--observations', type=Path,
                        help='relocated immutable observation archive; original status stays unchanged')
    args = parser.parse_args()
    plain_path, observed_path = [args.work/(p+'_status.json') for p in ('pair_plain', 'pair_observed')]
    plain, observed = [json.loads(p.read_text(encoding='utf8')) for p in (plain_path, observed_path)]
    equal = {name: plain[name] == observed[name] for name in ('before', 'after', 'receipts')}
    rng_equal = all(observed['observer_rng_checks'])
    if not (all(equal.values()) and rng_equal and plain['status'] == observed['status'] == 'completed'):
        raise AssertionError('full normal event observer admission failed')
    directory = args.observations or Path(observed['observations_output'])
    source = directory/'operations.jsonl.gz'
    with gzip.open(source, 'rt', encoding='utf8') as stream:
        rows = [json.loads(line) for line in stream]
    counts, operations = collections.Counter(), collections.Counter()
    old_groups, full_groups = collections.defaultdict(set), collections.defaultdict(set)
    signal_rows = [r for r in rows if r['kind'] == 'relation_signal']
    children = collections.defaultdict(list)
    for row in signal_rows:
        data = row['data']
        operations[data['operation']] += 1
        children[data['parent_id']].append(data)
        if data['operation'] != 'mechanism':
            continue
        state = data['state']
        key = (row['event'], row['update_clock'], state['identity'], state['generation'],
               state['version'], signature(state['variable_uses']), data['kind'],
               *(signature(data.get(name)) for name in ('source', 'action', 'mother', 'momentum')))
        output = signature(data['prediction']['mean'])
        old_groups[key].add(output)
        full_groups[key+(signature(state), data['ridge'], data['exclude'], data['exclude_fact'])].add(output)
    for row in rows:
        counts[row['kind']] += 1
    support_maps = []
    for row in signal_rows:
        data = row['data']
        if data['operation'] != 'relation_variables':
            continue
        uses = [c for c in children[data['call_id']] if c['operation'] == '_supports']
        groups = [{e['fact_id'] for e in c['facts']} for c in uses]
        union = sorted(set.union(*groups)) if groups else []
        retained = sorted({e['fact_id'] for e in data['loaded_state']['constraints']})
        restricted = [group & set(retained) for group in groups]
        equivalent = collections.defaultdict(list)
        for use, group in zip(uses, restricted):
            equivalent[tuple(sorted(group))].append(use['unknown_identity'])
        support_maps.append(dict(event=row['event'], call_id=data['call_id'],
            relation=[data['state']['identity'], data['state']['generation']],
            unknowns=[c['unknown_identity'] for c in uses], union_fact_ids=union,
            retained_fact_ids=retained, support_columns=[sorted(g) for g in groups],
            retained_support_columns=[sorted(g) for g in restricted],
            exact_membership_rank_before=support_rank(groups, union),
            exact_membership_rank_after=support_rank(restricted, retained),
            distinct_support_columns_before=len({tuple(sorted(g)) for g in groups}),
            distinct_support_columns_after=len(equivalent),
            identical_retained_membership_groups=list(equivalent.values())))
    receipts = observed['receipts']
    result = dict(full_event_equivalence=equal, observer_rng_equal=rng_equal,
        start_cycle=observed['start_cycle'], private_final_cycle=observed['final_cycle'],
        source_sha256=observed['source_sha'],
        event_kinds=[r['autonomous_event']['kind'] for r in receipts],
        real_optimizer_calls=counts['optimizer'], record_counts=dict(counts), signal_counts=dict(operations),
        seconds=dict(plain=plain['seconds'], observed=observed['seconds']),
        branch_identity=dict(old_context_groups=len(old_groups), complete_context_groups=len(full_groups),
            old_ambiguous_groups=sum(len(v) > 1 for v in old_groups.values()),
            complete_ambiguous_groups=sum(len(v) > 1 for v in full_groups.values())),
        support_maps=support_maps,
        map_interpretation='Exact ranks of observed fact-to-unknown membership before/after the existing finite retention. Not the rank of the whole model, not an H1 activity measurement.',
        main_body_continued=False, paid_world_calls=0,
        sources=[dict(path=str(p), sha256=sha(p)) for p in (plain_path, observed_path, source, directory/'tensors.bin')])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('support_maps', 'sources')}, ensure_ascii=False))
    print(json.dumps(dict(support_map_ranks=[(m['exact_membership_rank_before'], m['exact_membership_rank_after'])
                                             for m in support_maps]), ensure_ascii=False))


if __name__ == '__main__':
    main()
