"""Export a real archived relation/unknown snapshot as an executable Lean instance.

Identity is relation id + generation, and unknown id. Version remains edge
metadata. Export does not reinterpret these values as angular variables.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('observations', type=Path)
    p.add_argument('lean_output', type=Path)
    p.add_argument('manifest', type=Path)
    a = p.parse_args()
    archive = a.observations / 'operations.jsonl.gz'
    chosen = None
    with gzip.open(archive, 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'structural_relation_dependencies':
                chosen = row
    if chosen is None:
        raise ValueError('No real dependency snapshot')
    edges = chosen['data']['edges']
    relations = sorted({(x[0], x[3]) for x in edges})
    unknowns = sorted({x[1] for x in edges})
    rid = {key: i for i, key in enumerate(relations)}
    uid = {key: i for i, key in enumerate(unknowns)}
    if len({(x[0], x[1]) for x in edges}) != len(edges):
        raise ValueError('Primary dependency identity is not unique')
    if any(x[2] not in ('condition', 'action', 'mother') for x in edges):
        raise ValueError('Unknown actual port type')
    raw_rows = []
    dense_rows = []
    for r, u, port, generation, version in edges:
        raw_rows.append(f'  ⟨{r}, {generation}, {json.dumps(u)}, .{port}, {version}⟩')
        dense_rows.append(f'({rid[(r,generation)]}, {uid[u]})')
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    reference = Path(__file__).resolve().parents[1] / 'lean/CGBAxioms/reviews/NativeMeasurement.lean'
    text = reference.read_text(encoding='utf8') + '''

/- Generated from an actual completed whole-loop archive; not a made-up graph. -/
namespace CGB.NativeMeasurement.Actual2401
set_option linter.style.longLine false
set_option linter.style.setOption false
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
'''
    text += f'def sourceSHA : String := "{sha}"\n'
    text += f'def observedExternalCycle : Nat := {chosen["external_cycle"]}\n'
    text += 'def dependencies : Array Dependency := #[\n' + ',\n'.join(raw_rows) + '\n]\n'
    text += 'def relationIDs : Array (Nat × Nat) := #[' + ','.join(f'({r},{g})' for r,g in relations) + ']\n'
    text += 'def unknownIDs : Array String := #[' + ','.join(json.dumps(u) for u in unknowns) + ']\n'
    text += 'def denseEndpoints : Array (Nat × Nat) := #[' + ','.join(dense_rows) + ']\n'
    text += f'''def faithful : Bool :=
  dependencies.size == {len(edges)} && relationIDs.size == {len(relations)} &&
  unknownIDs.size == {len(unknowns)} && denseEndpoints.size == dependencies.size &&
  (List.range dependencies.size).all (fun i =>
    let d := dependencies[i]!
    let endpoints := denseEndpoints[i]!
    endpoints.1 < relationIDs.size && endpoints.2 < unknownIDs.size &&
    relationIDs[endpoints.1]! == (d.relation, d.generation) &&
    unknownIDs[endpoints.2]! == d.unknown)

theorem actual_export_faithful : faithful = true := by native_decide

def actualGraph : Graph {len(relations)} {len(unknowns)} {len(edges)} where
  relationEnd k := Fin.ofNat {len(relations)} (denseEndpoints[k.val]!).1
  unknownEnd k := Fin.ofNat {len(unknowns)} (denseEndpoints[k.val]!).2

theorem actual_dirac_squared
    (x : (Nodes {len(relations)} {len(unknowns)} → ℚ) × (Fin {len(edges)} → ℚ)) :
    dirac actualGraph (dirac actualGraph x) =
      (nodeLaplacian actualGraph x.1, edgeLaplacian actualGraph x.2) :=
  dirac_squared actualGraph x

end CGB.NativeMeasurement.Actual2401
'''
    a.lean_output.write_text(text, encoding='utf8')
    manifest = dict(archive=str(archive), sha256=sha, external_cycle=chosen['external_cycle'],
                    reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    segment_event=chosen['event'], relations=len(relations), unknowns=len(unknowns), edges=len(edges),
                    output=str(a.lean_output), meaning='real persistent relationship incidence, not full-body topology or phases')
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
