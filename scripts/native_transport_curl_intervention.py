"""Lift the observed internal-preview curl change to its actual fact inputs.

The carrier, support and clocks come from the admitted relation 213/4 calls
14 and 26. A fixed residual offset keeps the original differentiation path.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import eigh

from scripts.measure_native_inner_loop import ROOT, load, sha
from scripts.measure_native_inner_hodge_transport import first_calls
from scripts.measure_native_paper_phase_dirac import decode_cells
from scripts.audit_native_dowker_measurement import make_boundary, to_sparse


def prepare(work: Path, output: Path):
    source_path = ROOT/'results/published/cgb_native_shared_response_scoped_2401_20261117.json'
    source = json.loads(source_path.read_text(encoding='utf8'))
    refs = {r['call_id']: r for r in source['calls']}
    proof = Path(source['scope_lean']['path'])
    assert sha(proof) == source['scope_lean']['sha256']
    cells = [decode_cells(proof.read_text(encoding='utf8'), f'expanded{n}') for n in range(3)]
    b1 = to_sparse(make_boundary(cells[0], cells[1]), len(cells[0])).astype(float)
    b2 = to_sparse(make_boundary(cells[1], cells[2]), len(cells[1])).astype(float)
    eig, vec = eigh((b1@b1.T).toarray())
    live = eig > 1e-10
    inv = (vec[:, live]/eig[live])@vec[:, live].T
    with np.load(source['arrays']['path']) as z:
        h = z['harmonic_basis']
    ids = json.loads((ROOT/'results/published/cgb_native_harmonic_observability_2401_20261117.json').read_text(encoding='utf8'))['identities']['relations']
    nodes = sorted([tuple(x) for x in ids]+[tuple(x) for x in source['retained_saved_scope_vertices']])
    obs = work/'baseline/observations'
    calls = first_calls(obs/'operations.jsonl.gz')
    data = {}
    with (obs/'tensors.bin').open('rb') as stream:
        for cid in (14, 26):
            row = calls[cid]; d = row['data']; ref = refs[cid]
            assert row['event'] == 1 and row['update_clock'] == 0
            evidence = next(r['data'] for r in calls.values() if r['data']['parent_id'] == cid and r['data']['operation'] == '_bound_evidence_design')
            keys = [(f['fact_id'], f['kind']) for f in evidence['facts']]
            old_keys = [tuple(k) for k in ref['fact_keys']]
            assert len(keys) == len(set(keys)) == 8 and set(keys) == set(old_keys)
            mask = np.asarray(ref['mask'])[:, [old_keys.index(k) for k in keys]]
            pred = d['prediction']
            x, y, q = [load(stream, pred[k]).astype(float) for k in ('x', 'y', 'query')]
            ridge = load(stream, pred['ridge']) if isinstance(pred['ridge'], dict) else pred['ridge']
            alpha = np.linalg.solve(x@x.T+float(np.asarray(ridge))*np.eye(len(x)), (q@x.T).T).T
            embed = np.zeros((len(cells[1]), 4))
            for col, edge in enumerate(ref['edge_ids']):
                embed[edge, col] = 1 if nodes[cells[1][edge][1]] == tuple(ref['relation']) else -1
            a = embed@(mask*alpha)
            c = a-b1.T@(inv@(b1@a))-h@(h.T@a)
            data[cid] = dict(x=x, y=y, query=q, alpha=alpha, a=a, c=c, mask=mask,
                             fact_keys=keys, state=d['state'])
    before, after = data[14], data[26]
    target = after['c']@after['y']-before['c']@before['y']
    c = after['c']
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    rank = int((s > s[0]*1e-10).sum())
    pinv = (vt[:rank].T/s[:rank])@u[:, :rank].T
    delta = -pinv@target
    complement = np.eye(8)-vt[:rank].T@vt[:rank]
    control = complement@after['y']
    control *= np.linalg.norm(delta)/np.linalg.norm(control)
    # Strengthen matching at the actual downstream mean, too. The minimum
    # mean-matched kernel control needs 11.5% more input norm in this body.
    # Add only a joint-null direction to the target to equalize both norms.
    mean_delta = after['alpha']@delta
    direction = complement@after['alpha'].T
    paired_control = direction@(mean_delta/(after['alpha']@direction))
    joint = np.vstack((c, after['alpha']))
    _, js, jv = np.linalg.svd(joint, full_matrices=False)
    jr = int((js > js[0]*1e-10).sum())
    null = np.eye(8)-jv[:jr].T@jv[:jr]
    filler = null@after['y']
    extra = np.linalg.norm(paired_control)**2-np.linalg.norm(delta)**2
    assert extra > 0
    paired_target = delta+filler*(np.sqrt(extra)/np.linalg.norm(filler))
    reach_error = float(np.linalg.norm(c@delta+target)/np.linalg.norm(target))
    assert reach_error < 1e-8
    assert np.linalg.norm(c@control) < 1e-9
    assert abs(np.linalg.norm(control)-np.linalg.norm(delta)) < 1e-9
    arrays = dict(delta=delta, control=control, coefficient=c, target=target,
                  paired_target=paired_target, paired_control=paired_control,
                  x=after['x'], y=after['y'], query=after['query'], alpha=after['alpha'],
                  edge_map=after['a'], complement=complement)
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    def effects(change):
        field = after['a']@change
        grad = b1.T@(inv@(b1@field)); harm = h@(h.T@field)
        return dict(input_norm=float(np.linalg.norm(change)),
                    relative_input_norm=float(np.linalg.norm(change)/np.linalg.norm(after['y'])),
                    mean_change_norm=float(np.linalg.norm(after['alpha']@change)),
                    gradient_energy=float(np.square(grad).sum()),
                    curl_energy=float(np.square(c@change).sum()),
                    harmonic_energy=float(np.square(harm).sum()),
                    dirac_node_energy=float(np.square(b1@field).sum()),
                    dirac_face_energy=float(np.square(b2.T@field).sum()))
    result = dict(status='reachable_input_lift', relation=[213, 4], event=1, update_clock=0,
                  before_call=14, after_call=26, rank=rank, singular_values=s.tolist(),
                  relative_reachability_residual=reach_error, fact_keys=after['fact_keys'],
                  variable_uses=after['state']['variable_uses'],
                  target=effects(delta), matched=effects(control),
                  paired_target=effects(paired_target), paired_control=effects(paired_control),
                  paired_input_norm_error=float(abs(np.linalg.norm(paired_target)-np.linalg.norm(paired_control))),
                  paired_mean_error=float(np.linalg.norm(after['alpha']@(paired_target-paired_control))),
                  source=dict(path=str(source_path), sha256=sha(source_path)),
                  operations=dict(path=str(obs/'operations.jsonl.gz'), sha256=sha(obs/'operations.jsonl.gz')),
                  tensors=dict(path=str(obs/'tensors.bin'), sha256=sha(obs/'tensors.bin')),
                  arrays=dict(path=str(output.with_suffix('.npz')), sha256=sha(output.with_suffix('.npz'))))
    assert result['paired_input_norm_error'] < 1e-10 and result['paired_mean_error'] < 1e-10
    assert np.linalg.norm(c@paired_target+target) < 1e-10
    assert np.linalg.norm(c@paired_control) < 1e-10
    proof_path = ROOT/'lean/CGBAxioms/reviews/NativeTransportCurlUse.lean'
    proof_path.write_text(LEAN, encoding='utf8')
    checked = subprocess.run(['lake', 'env', 'lean', str(proof_path.relative_to(ROOT/'lean/CGBAxioms'))],
                             cwd=ROOT/'lean/CGBAxioms', capture_output=True, text=True, encoding='utf8')
    output.with_suffix('.lean.log').write_text(checked.stdout+checked.stderr, encoding='utf8')
    if checked.returncode:
        raise RuntimeError(checked.stdout+checked.stderr)
    result['proof'] = dict(path=str(proof_path), sha256=sha(proof_path), exit_code=0,
                           binding='981 actual edges, eight stable fact rows, 128 response coordinates; numerical lift identities in arrays')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    return result


LEAN = '''import Mathlib.Tactic
import Mathlib.Analysis.Calculus.FDeriv.Add
namespace CGB.NativeTransportCurlUse
open Matrix
/-- C = P_c E signed-M diag(alpha) on the admitted 981-edge body carrier.
The before/after curl difference T is lifted through eight actual fact rows.
N is joint-null for C and the original posterior mean map A. -/
theorem paired_curl (C : Matrix (Fin 981) (Fin 8) ℝ)
    (Y L N K : Matrix (Fin 8) (Fin 128) ℝ)
    (T : Matrix (Fin 981) (Fin 128) ℝ)
    (lift : C * L = -T) (null : C * N = 0) (control : C * K = 0) :
    C * (Y + (L + N)) = C * Y - T ∧ C * (Y + K) = C * Y := by
  constructor <;> simp [Matrix.mul_add, lift, null, control, sub_eq_add_neg]
theorem paired_mean (A : Matrix (Fin 1) (Fin 8) ℝ)
    (Y L N K : Matrix (Fin 8) (Fin 128) ℝ)
    (null : A * N = 0) (matched : A * L = A * K) :
    A * (Y + (L + N)) = A * (Y + K) := by
  simp [Matrix.mul_add, null, matched]
theorem fixed_offset_derivative (delta x : (Fin 8 → Fin 128 → ℝ)) :
    HasFDerivAt (fun y => y + delta) (ContinuousLinearMap.id ℝ _) x := by
  simpa using (hasFDerivAt_id x).add_const delta
end CGB.NativeTransportCurlUse
'''


def install(observer, family, specification):
    import functools
    import sys
    import torch
    from scripts.ib_local import causal_relation_program as cp

    assert sha(specification['arrays']['path']) == specification['arrays']['sha256']
    arrays = np.load(specification['arrays']['path'])
    old_choice = observer.engine._sample_internal_choice
    old_mechanism, old_posterior = cp.mechanism, cp.posterior
    active = False
    completed = False
    count = 0
    contexts, receipts, bindings = [], [], []

    @functools.wraps(old_choice)
    def choice(*args, **kwargs):
        nonlocal active
        active = not completed
        try:
            return old_choice(*args, **kwargs)
        finally:
            active = False

    @functools.wraps(old_mechanism)
    def mechanism(*args, **kwargs):
        nonlocal count
        is_transport = active and kwargs.get('mother') is not None
        if is_transport:
            count += 1
        selected = is_transport and count == 2
        if selected:
            state = args[1]
            assert (state.identity, state.generation) == (213, 4)
            assert observer.clock == 1 and observer.update_clock == 0
            actual = [(v.identity, v.port.value, version, seen) for v, version, seen in state.variable_uses]
            expected = [(v['identity'], v['port'], version, seen) for v, version, seen in specification['variable_uses']]
            assert actual == expected
        contexts.append(selected)
        try:
            result = old_mechanism(*args, **kwargs)
            return result
        finally:
            contexts.pop()

    @functools.wraps(old_posterior)
    def posterior(x, y, query, ridge, **kwargs):
        nonlocal completed
        if not contexts or not contexts[-1]:
            return old_posterior(x, y, query, ridge, **kwargs)
        actual = [t.detach().cpu().double().numpy() for t in (x, y, query)]
        for key, value in zip(('x', 'y', 'query'), actual):
            assert np.array_equal(value, arrays[key]), f'actual {key} differs from bound first-preview data'
        delta = (np.zeros_like(arrays['y']) if family == 'baseline' else
                 arrays['paired_target' if family == 'target' else 'paired_control'])
        revised = y if family == 'baseline' else y+torch.as_tensor(delta, device=y.device, dtype=y.dtype)
        result = old_posterior(x, revised, query, ridge, **kwargs)
        actual_delta = revised.detach().cpu().double().numpy()-actual[1]
        c, target = arrays['coefficient'], arrays['target']
        measured = c@actual_delta
        receipt = dict(family=family, event=observer.clock, update_clock=observer.update_clock,
                       relation=[213, 4], original_call=26,
                       input_delta_norm=float(np.linalg.norm(actual_delta)),
                       relative_motion_curl_remaining=float(np.linalg.norm(target+measured)/np.linalg.norm(target)),
                       relative_motion_curl_change=float(np.linalg.norm(measured)/np.linalg.norm(target)),
                       mean_delta=(arrays['alpha']@actual_delta).tolist(),
                       mean_delta_norm=float(np.linalg.norm(arrays['alpha']@actual_delta)),
                       offset_rounding_norm=float(np.linalg.norm(actual_delta-delta)),
                       original_gradient_port_retained=revised.requires_grad == y.requires_grad)
        observer.record('transport_curl_input_intervention', dict(receipt,
                        residual_before=y, residual_after=revised, distribution=result))
        receipts.append(receipt)
        completed = True
        return result

    observer.engine._sample_internal_choice = choice
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, '__name__', '').startswith('scripts.ib_local'):
            continue
        for name, old, new in [('mechanism', old_mechanism, mechanism), ('posterior', old_posterior, posterior)]:
            if getattr(module, name, None) is old:
                bindings.append((module, name, old))
                setattr(module, name, new)

    def undo():
        observer.engine._sample_internal_choice = old_choice
        for module, name, old in reversed(bindings):
            setattr(module, name, old)
    return receipts, undo


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.work, args.output), ensure_ascii=False))
