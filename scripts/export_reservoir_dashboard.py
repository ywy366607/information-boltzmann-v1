"""Export structural reservoir measures from one committed causal archive."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sqlite3

from scripts.audit_native_dowker_measurement import make_boundary, rank_integer
from scripts.measure_native_relation_percolation import build


def snapshot(body):
    meta = json.loads((body/'last.metadata.json').read_text(encoding='utf8'))
    archive = meta['config']['causal_archive']
    path = Path(archive['snapshot']).resolve()
    if sha256(path.read_bytes()).hexdigest() != archive['sha256']:
        raise ValueError('archive digest mismatch')
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    try:
        n = db.execute('SELECT count(*) FROM relations').fetchone()[0]
        rows = db.execute('SELECT relation_id,unknown_id,port,generation,seen_version FROM relation_variables').fetchall()
        unknowns = db.execute('SELECT count(*) FROM unknown_bindings').fetchone()[0]
    finally:
        db.close()
    graph = build(dict(rows=rows,receipt=dict(causal_archive=dict(relations=n))))
    edges = sorted(graph['witnesses'])
    faces = sorted({face for group in graph['supports'].values() for face in combinations(sorted(group),3)})
    b1_rank = len(graph['vertices'])-len(graph['components'])
    beta1 = len(edges)-b1_rank-rank_integer(make_boundary(edges,faces))
    return dict(cycle=meta['cycle'], checkpoint_sha256=meta['checkpoint_sha256'], archive_sha256=archive['sha256'],
                relations=n, unknowns=unknowns, giant=len(graph['largest']), beta1=beta1, edges=len(edges), faces=len(faces))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('body',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=snapshot(args.body)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))
