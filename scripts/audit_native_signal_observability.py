"""Check which existing normal-life signals support Hodge measurements.

No imputation, context binning, frozen replay or training is performed.
Input equality is a necessary matching condition, not certification that
unrecorded functional revisions were equal.
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path


def tensor_signature(value):
    if value is None:
        return None
    ref = value['tensor_ref']
    return ref['sha256'], ref['dtype'], tuple(ref['shape'])


def state_signature(data):
    return (data['identity'], data['generation'], data['version'],
            tuple((use[0]['identity'], use[0]['port'], use[1], use[2])
                  for use in data['variable_uses']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    counts, groups, sources = collections.Counter(), collections.defaultdict(list), []
    for directory in sorted(args.work.glob('collect_observations*')):
        path = directory/'operations.jsonl.gz'
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            for line_index, line in enumerate(stream):
                row = json.loads(line)
                counts[row['kind']] += 1
                if row['kind'] != 'mechanism':
                    continue
                data = row['data']
                key = (directory.name, row['event'], row['update_clock'], data['kind'],
                       *(tensor_signature(data.get(name)) for name in
                         ('source', 'action', 'mother', 'momentum')))
                groups[key].append((line_index, data))
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    ambiguous, comparable, vertices = [], [], set()
    for key, values in groups.items():
        by_state = collections.defaultdict(list)
        by_relation = {}
        for line, data in values:
            by_state[state_signature(data)].append((line, data))
            identity = data['identity'], data['generation']
            by_relation[identity] = data
            vertices.add(identity)
        for state, duplicates in by_state.items():
            means = {tensor_signature(data['mean']) for _, data in duplicates}
            if len(means) > 1:
                ambiguous.append(dict(segment=key[0], event=key[1], update_clock=key[2],
                    kind=key[3], relation=state[:3], calls=len(duplicates),
                    distinct_outputs=len(means), lines=[line+1 for line, _ in duplicates]))
        if len(by_relation) > 1:
            edges = []
            for (a, da), (b, db) in itertools.combinations(sorted(by_relation.items()), 2):
                sa = {(u[0]['identity'], u[0]['port']) for u in da['variable_uses']}
                sb = {(u[0]['identity'], u[0]['port']) for u in db['variable_uses']}
                if sa & sb:
                    edges.append(dict(endpoints=[a, b], shared_typed_unknowns=sorted(sa & sb)))
            comparable.append(dict(segment=key[0], event=key[1], update_clock=key[2],
                kind=key[3], relations=sorted(by_relation), witnessed_pairs=edges,
                strict_branch_state_certified=False))
    result = dict(sources=sources, record_counts=dict(counts),
        raw_input_context_groups=len(groups), actual_relations_called=len(vertices),
        same_input_and_state_metadata_with_different_output_groups=len(ambiguous),
        ambiguous_groups=ambiguous,
        multiple_relation_context_groups=len(comparable), comparable_groups=comparable,
        groups_with_witnessed_pair=sum(bool(g['witnessed_pairs']) for g in comparable),
        largest_matched_relation_set=max((len(g['relations']) for g in comparable), default=0),
        independently_recorded_dowker_edge_vectors=0,
        independently_recorded_dowker_face_vectors=0,
        measured_harmonic_activity=None,
        missing_is_not_zero=True,
        interpretation=(
            'Old mean/base are node-valued relation predictions. Their differences '
            'are exact cochains and have zero harmonic projection by construction. '
            'Functional evidence/branch provenance is missing, so raw input matches '
            'are candidate matches only; no Hodge activity or synchronization result '
            'can be recovered by zero-filling these records.'),
        model_updates=0, world_calls=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('sources', 'ambiguous_groups', 'comparable_groups')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
