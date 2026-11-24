"""Export actual typed calls and closed-loop dispatch receipts for Lean checking.

No model import, GPU allocation, training, or synthetic observation is involved.
Every invocation retains its own identity, including repeated local programs.
Tensor contents are opaque in this abstraction. Their real recorded dimensions
are checked, without claiming proof of floating-point kernel equivalence.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


OPERATIONS = {
    'mother_response': 'motherResponse',
    'ambient_transport': 'ambientTransport',
    'body_transition': 'bodyTransition',
    'existing_continuation': 'continuation',
}


def lean_list(items):
    return '[' + ', '.join(items) + ']'


def variable(identity, role):
    return f'⟨{json.dumps(identity)}, .{role}⟩'


def value_shape(value):
    shape = value['tensor']['tensor_ref']['shape']
    return f'⟨{variable(value["identity"], value["port"])}, {lean_list(map(str, shape))}⟩'


def execution(row, ordinal):
    data = row['data']
    calls = []
    for call in data['calls']:
        args = lean_list(variable(*v) for v in call['inputs'])
        calls.append(f'⟨.{OPERATIONS[call["operation"]]}, {args}, {variable(*call["output"])}⟩')
    batch = data['actual_values'][0]['tensor']['tensor_ref']['shape'][0]
    return (f'⟨{row["event"]}, {ordinal}, {batch}, {lean_list(calls)}, '
            f'{lean_list(value_shape(v) for v in data["actual_values"])}, '
            f'{lean_list(value_shape(v) for v in data["alternative_outputs"])}⟩')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', type=Path)
    parser.add_argument('lean_output', type=Path)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    archive = args.observations / 'operations.jsonl.gz'
    with gzip.open(archive, 'rt', encoding='utf8') as stream:
        rows = [json.loads(line) for line in stream]
    counts = collections.Counter(row['kind'] for row in rows)
    world = collections.Counter(row['event'] for row in rows if row['kind'] == 'world')
    optim = collections.Counter(row['event'] for row in rows if row['kind'] == 'optimizer')
    executions, receipts, revisions = [], [], collections.Counter()
    for row in rows:
        data = row['data']
        if row['kind'] == 'typed_execution':
            executions.append(execution(row, len(executions)))
        elif row['kind'] == 'relation_revision':
            revisions['observed' if data['observed'] else 'hypothetical'] += 1
        elif row['kind'] == 'event':
            kind = data['autonomous_event']['kind']
            evidence = str(bool(data['autonomous_event']['new_world_evidence'])).lower()
            receipts.append(f'⟨{row["event"]}, .{kind}, {data["updates"]}, '
                            f'{optim[row["event"]]}, {world[row["event"]]}, '
                            f'{data["external_bytes"]}, {evidence}⟩')
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    reference = Path(__file__).resolve().parents[1] / 'lean/CGBAxioms/reviews/NativeExecution.lean'
    text = reference.read_text(encoding='utf8') + '''

/-! Generated from the real completed 2383--2401 archive. No simulated calls. -/
namespace CGB.NativeExecution.Actual2401
set_option linter.style.longLine false
set_option linter.style.setOption false
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
'''
    text += f'def sourceSHA : String := "{sha}"\n'
    text += 'def executions : Array Execution := #[\n' + ',\n'.join(executions) + '\n]\n'
    text += 'def receipts : List EventReceipt := [\n' + ',\n'.join(receipts) + '\n]\n'
    text += f'''
theorem actual_execution_count : executions.size = {len(executions)} := by native_decide
theorem actual_calls_typed : executions.all executionValid = true := by native_decide
theorem actual_dispatch_valid : dispatchValid false receipts = true := by native_decide
end CGB.NativeExecution.Actual2401
'''
    args.lean_output.write_text(text, encoding='utf8')
    manifest = dict(archive=str(archive), sha256=sha, counts=dict(counts),
                    reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    executions=len(executions), receipts=len(receipts), revisions=dict(revisions),
                    abstraction='actual call identities, port dimensions, event dispatch and boundary counts',
                    opaque='tensor content; numerical transition kernels are not proved by this certificate',
                    lean=str(args.lean_output))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
