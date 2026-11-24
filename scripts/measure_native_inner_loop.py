"""Join paper phase observables to actual preview, choice and feedback clocks.

Uses the existing complete body recordings, the certified physical carrier,
and the actual call-parent relation. All time indices refer to native events.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zlib

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from scripts.measure_native_paper_phase_dirac import phase_measurement

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(stream, ref):
    if ref is None:
        return None
    r = ref['tensor_ref']
    stream.seek(r['offset'])
    raw = zlib.decompress(stream.read(r['compressed_bytes']))
    return np.frombuffer(raw, dtype=r['dtype']).reshape(r['stored_shape']).astype(float)


def quantiles(values):
    return np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else []


def difference(a, b):
    d = b-a
    return dict(norm=float(np.linalg.norm(d)), max_abs=float(abs(d).max()),
                reference_norm=float(np.linalg.norm(a)))


def analyze(directory):
    obs = directory/'observations'
    phase, arrays = phase_measurement(obs)
    rows = [json.loads(line) for line in gzip.open(obs/'operations.jsonl.gz', 'rt', encoding='utf8')]
    calls = {r['data']['call_id']: r for r in rows if r['kind'] == 'relation_signal'}
    events = {r['event']: r for r in rows if r['kind'] == 'event'}
    pairs, updates, witnesses = [], [], []
    grouped = defaultdict(list)
    for r in calls.values():
        d = r['data']
        if d['operation'] == 'compose' and d['caller']['function'] == '_sample_internal_choice':
            grouped[r['event']].append(r)
    with (obs/'tensors.bin').open('rb') as stream:
        for event, roots in grouped.items():
            roots.sort(key=lambda r: r['data']['call_id'])
            assert len(roots) == 2
            root_ids = [r['data']['call_id'] for r in roots]
            children = [[r for r in calls.values() if r['data']['operation'] == 'mechanism'
                         and r['data']['parent_id'] == cid] for cid in root_ids]
            assert [len(x) for x in children] == [3, 3]
            def child_port(row):
                d = row['data']
                if d['kind'] == 'mother':
                    return 'predict_mother'
                assert d['kind'] == 'decision'
                return 'transport' if d['mother'] is not None else 'direct_prediction'
            child_maps = [{child_port(r): r for r in group} for group in children]
            assert all(len(m) == 3 for m in child_maps)
            assert child_maps[0].keys() == child_maps[1].keys()
            values = {}
            for name in child_maps[0]:
                a, b = child_maps[0][name], child_maps[1][name]
                da, db = a['data'], b['data']
                assert a['event'] == b['event'] == event
                assert a['update_clock'] == b['update_clock']
                assert (da['state']['identity'], da['state']['generation']) == (
                    db['state']['identity'], db['state']['generation'])
                values[name] = difference(load(stream, da['prediction']['mean']),
                                          load(stream, db['prediction']['mean']))
                witnesses.append([event, a['event'], b['event'], a['update_clock'], b['update_clock'],
                                  root_ids[0], root_ids[1], da['parent_id'], db['parent_id'],
                                  da['call_id'], db['call_id']])
            a, b = [m['predict_mother']['data'] for m in child_maps]
            momentum_a, momentum_b = load(stream, a['momentum']), load(stream, b['momentum'])
            assert momentum_a.shape == momentum_b.shape == (1, 8192)
            row = dict(event=event, update_clock=roots[0]['update_clock'], compose_calls=root_ids,
                       relation=[a['state']['identity'], a['state']['generation']],
                       source=difference(load(stream, a['source']), load(stream, b['source'])),
                       momentum=difference(momentum_a, momentum_b), predictions=values,
                       momentum_change_by_layer=np.linalg.norm((momentum_b-momentum_a).reshape(32, 256),axis=1).tolist(),
                       admission=events[event]['data']['autonomous_event']['admission'])
            # Exact pair of the existing physical preview, same event/clock.
            ps = [s for s in phase['samples'] if s['event']==event and s['kind'].startswith('preview_quiet_')]
            assert len(ps)==2 and ps[0]['kind']=='preview_quiet_before' and ps[1]['kind']=='preview_quiet_after'
            xo0, xo1 = [arrays[f'X_o_{s["sample"]}'] for s in ps]
            row['X_o_change_by_layer'] = np.median(abs(xo1-xo0), axis=1).tolist()
            row['X_o_change_quantiles'] = quantiles(abs(xo1-xo0).flatten())
            pairs.append(row)
        for r in rows:
            if r['kind'] != 'optimizer':
                continue
            d = r['data']
            gradient, delta = load(stream,d['actual_gradient']), load(stream,d['applied_delta'])
            assert delta.shape == (32,12)
            updates.append(dict(event=r['event'], update_clock=r['update_clock'],
                gradient_norm=None if gradient is None else float(np.linalg.norm(gradient)),
                applied_norm=float(np.linalg.norm(delta)),
                changed_coordinates=int(np.count_nonzero(delta)),
                changed_layers=int(np.count_nonzero(np.linalg.norm(delta,axis=1))),
                applied_by_layer=np.linalg.norm(delta,axis=1).tolist(),
                gradient_by_layer=None if gradient is None else np.linalg.norm(gradient,axis=1).tolist(),
                budget_gate=None if d['budget_gate'] is None else load(stream,d['budget_gate']).tolist()))
    operations = defaultdict(list)
    for row in phase['operation_changes']:
        operations[row['operation']].append(row['median_abs_complex_X_o_change'])
    # Real chronological persistence uses committed event-after states only.
    committed = [s for s in phase['samples'] if s['kind']=='event_after']
    committed_changes = []
    for a,b in zip(committed,committed[1:]):
        x,y = arrays[f'X_o_{a["sample"]}'], arrays[f'X_o_{b["sample"]}']
        committed_changes.append(dict(from_event=a['event'],to_event=b['event'],
            event_kind=events[b['event']]['data']['autonomous_event']['kind'],
            delta_by_layer=np.median(abs(y-x),axis=1).tolist(),
            delta_quantiles=quantiles(abs(y-x).flatten())))
    action_rows = []
    for event,r in events.items():
        d=r['data']
        if d['autonomous_event']['kind']=='external':
            action_rows.append(dict(event=event,update_clock=r['update_clock'],
                choice=d['choice'],probability=d['choice_probability'],world_anchor=d['world']['anchor'],
                utterance_hex=d['utterance_hex'],external_bytes=d['external_bytes'],
                settlement=d['settlement'],heldout_pre_nll=d['heldout_pre_nll'],
                causal_program_loss=d['causal_program_loss'],
                current_credit_samples=d.get('momentum_current_credit_samples'),
                delayed_credit_samples=d.get('momentum_delayed_credit_samples'),
                adaptive=d['adaptive_dynamics']))
    return dict(status='measured',directory=str(directory),
        sources=[dict(path=str(obs/n),sha256=sha(obs/n)) for n in ('operations.jsonl.gz','tensors.bin')],
        phase=phase,preview_pairs=pairs,learning_updates=updates,
        committed_event_changes=committed_changes,actions=action_rows,
        summary=dict(events=len(events),event_kinds=dict(Counter(r['data']['autonomous_event']['kind'] for r in events.values())),
            physical_samples=len(phase['samples']),preview_pairs=len(pairs),optimizer_updates=len(updates),
            theta_gradient_updates=sum(u['gradient_norm'] is not None for u in updates),
            theta_applied_updates=sum(u['applied_norm']>0 for u in updates),
            operation_X_o_change_quantiles={k:quantiles(v) for k,v in operations.items()},
            preview_mother_change_quantiles=quantiles([p['predictions']['predict_mother']['norm'] for p in pairs]),
            preview_momentum_change_quantiles=quantiles([p['momentum']['norm'] for p in pairs]),
            external_bytes=sum(r['external_bytes'] for r in action_rows)),
        call_mapping_certificate=witnesses), arrays


def export_lean(path, report):
    # Concrete event/clock/parent witnesses from the recorded runtime, together
    # with the actual read and local feedback contraction dimensions.
    data=report['call_mapping_certificate']
    text='''import Mathlib.Tactic
namespace CGB.NativeInnerLoopActual2401
def sourceWidth : Nat := 256
def layerCount : Nat := 32
def readWidth : Nat := layerCount * sourceWidth
example : readWidth = 8192 := by decide
def actualCalls : List (List Nat) := DATA
def valid (r : List Nat) : Bool :=
  r.length == 11 && r[0]! == r[1]! && r[0]! == r[2]! &&
  r[3]! == r[4]! && r[5]! == r[7]! && r[6]! == r[8]! &&
  r[5]! < r[9]! && r[9]! < r[6]! && r[6]! < r[10]!
example : actualCalls.all valid = true := by native_decide
example : actualCalls.length = COUNT := by native_decide

/-- Actual P-read credit: one 256-coordinate read per each of 32 layers,
    twelve local oscillator parameters per layer. Same contraction as
    MomentumReadBridge.backward: einsum('ld,pld->lp', cotangent, sensitivity). -/
def readCredit (g : Fin 32 → Fin 256 → ℝ)
    (elig : Fin 12 → Fin 32 → Fin 256 → ℝ) (l : Fin 32) (p : Fin 12) : ℝ :=
  ∑ d, g l d * elig p l d
theorem credit_add (g h : Fin 32 → Fin 256 → ℝ)
    (elig : Fin 12 → Fin 32 → Fin 256 → ℝ) (l : Fin 32) (p : Fin 12) :
    readCredit (fun l d => g l d + h l d) elig l p =
    readCredit g elig l p + readCredit h elig l p := by
  simp [readCredit, add_mul, Finset.sum_add_distrib]

/-- Finite feedback intervention retains the exact primal and optimizer
    proposal, selecting whether its parameter displacement is committed. -/
def commit (keep : Bool) (old proposal : Fin 32 → Fin 12 → ℝ) :=
  if keep then proposal else old
theorem commit_live (old proposal : Fin 32 → Fin 12 → ℝ) :
    commit true old proposal = proposal := rfl
theorem commit_held (old proposal : Fin 32 → Fin 12 → ℝ) :
    commit false old proposal = old := rfl
end CGB.NativeInnerLoopActual2401
'''.replace('DATA',str(data)).replace('COUNT',str(len(data)))
    path.write_text(text,encoding='utf8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--lean',action='store_true')
    args=p.parse_args()
    report,arrays=analyze(args.directory)
    array_path=args.output.with_suffix('.npz')
    np.savez_compressed(array_path,**arrays)
    report['arrays']=dict(path=str(array_path),sha256=sha(array_path))
    if args.lean:
        proof=ROOT/'lean/CGBAxioms/reviews/NativeInnerLoopActual2401.lean'
        export_lean(proof,report)
        done=subprocess.run(['lake','env','lean',str(proof.relative_to(ROOT/'lean/CGBAxioms'))],
            cwd=ROOT/'lean/CGBAxioms',capture_output=True,text=True,encoding='utf8')
        args.output.with_suffix('.lean.log').write_text(done.stdout+done.stderr,encoding='utf8')
        if done.returncode:raise RuntimeError(done.stdout+done.stderr)
        report['lean']=dict(path=str(proof),sha256=sha(proof),exit_code=done.returncode)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report['summary'],ensure_ascii=False))


if __name__=='__main__':
    main()
