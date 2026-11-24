"""Apply the same supported-complex measurement to normal-life snapshots.

Uses completed recorded learning events, without replay, freezing or training.
No sliding coactivity window or threshold is used to create cells.
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_native_dowker_measurement import make_boundary, rank_integer


def measure(raw):
    support = collections.defaultdict(set)
    for relation, unknown, port, generation, version in raw:
        support[unknown, port].add((relation, generation))
    cells = [sorted(set(cell for group in support.values()
                        for cell in itertools.combinations(sorted(group), size)))
             for size in range(1, 5)]
    rank = [rank_integer(make_boundary(cells[n-1], cells[n])) for n in range(1, 4)]
    return dict(simplex_counts=list(map(len, cells)), boundary_ranks_over_Q=rank,
                betti_over_Q=[len(cells[0])-rank[0], len(cells[1])-rank[0]-rank[1],
                              len(cells[2])-rank[1]-rank[2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    measurements, sources = [], []
    for directory in sorted(args.work.glob('collect_observations*')):
        source = directory/'operations.jsonl.gz'
        with gzip.open(source, 'rt', encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'structural_relation_dependencies':
                    measurements.append(dict(cycle=row['external_cycle'], segment=directory.name,
                        local_event=row['event'], **measure(row['data']['edges'])))
        sources.append(dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    result = dict(sources=sources, snapshots=len(measurements),
        first=measurements[0], last=measurements[-1],
        betti_min=[min(r['betti_over_Q'][k] for r in measurements) for k in range(3)],
        betti_max=[max(r['betti_over_Q'][k] for r in measurements) for k in range(3)],
        betti_changes=sum(a['betti_over_Q'] != b['betti_over_Q'] for a,b in zip(measurements, measurements[1:])),
        measurements=measurements, model_updates=0, world_calls=0,
        meaning='topology of shared-variable dependence during recorded full learning; not synchronization or memory utility')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources', 'measurements')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
