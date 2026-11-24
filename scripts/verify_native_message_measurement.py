"""Verify native-message candidates on the existing actual Dowker complex.

This checks measurement definitions, not neural capability. Redistribution
witnesses are alternative readings of ONE recorded native fact message, never
new experience or a modified body. No Torch, training, or alternate graph.
"""
import argparse
import collections
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
from scipy.linalg import eigh
from sympy import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_native_path_composition import load
from scripts.audit_native_dowker_measurement import make_boundary, to_sparse


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def components(vertices, edges):
    graph = collections.defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    unseen, result = set(vertices), []
    while unseen:
        reached = {min(unseen)}
        todo = list(reached)
        while todo:
            for other in graph[todo.pop()] - reached:
                reached.add(other)
                todo.append(other)
        unseen -= reached
        result.append(reached)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('lean_output', type=Path)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    obs = a.task/'work/signal_admission_2401_20261117/pair_observed_observations'
    source = obs/'operations.jsonl.gz'
    with gzip.open(source, 'rt', encoding='utf8') as stream:
        records = [json.loads(line) for line in stream]
    signals = {r['data']['call_id']: r for r in records if r['kind'] == 'relation_signal'}
    children = collections.defaultdict(list)
    for r in signals.values():
        children[r['data']['parent_id']].append(r['data'])
    db_path = next((a.task/'body').glob('*.sqlite'))
    db = sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True)
    raw = db.execute('SELECT relation_id,unknown_id,port,generation,seen_version FROM relation_variables').fetchall()
    uses, supports = collections.defaultdict(set), collections.defaultdict(set)
    for rid, unknown, port, generation, _ in raw:
        uses[rid, generation].add((unknown, port))
        supports[unknown, port].add((rid, generation))
    cells = [sorted({cell for group in supports.values()
                     for cell in itertools.combinations(sorted(group), n)}) for n in (1, 2, 3)]
    vertices = [c[0] for c in cells[0]]
    vi = {v: i for i, v in enumerate(vertices)}
    edges = cells[1]
    b1, b2 = [to_sparse(make_boundary(cells[n-1], cells[n]), len(cells[n-1])) for n in (1, 2)]
    old = json.loads((project/'results/published/cgb_native_harmonic_observability_2401_20261117.json').read_text(encoding='utf8'))
    if cells[1] != [tuple(map(tuple, e)) for e in old['identities']['edges']]:
        raise AssertionError('live archive is not the certified reference complex')
    h = np.load(project/'results/published/cgb_native_harmonic_observability_2401_20261117.npz')['harmonic_basis']
    kernel_residuals = dict(boundary=float(np.abs(b1 @ h).max()),
        coboundary=float(np.abs(b2.T @ h).max()),
        orthonormality=float(np.abs(h.T @ h-np.eye(h.shape[1])).max()))
    if max(kernel_residuals.values()) > 1e-10:
        raise AssertionError('cached harmonic basis does not match actual boundaries')
    receiver = (213, 4)
    # Only a complete, already recorded original invocation; no replay.
    call = next(r for r in signals.values() if r['data']['operation'] == 'mechanism'
                and (r['data']['state']['identity'], r['data']['state']['generation']) == receiver
                and r['data']['kind'] == 'mother' and r['data']['mother'] is None)
    d = call['data']
    design = next(c for c in children[d['call_id']] if c['operation'] == '_bound_evidence_design')
    posterior = next(c for c in children[d['call_id']] if c['operation'] == 'posterior')
    loaded = next(c['data'] for c in signals.values() if c['data']['operation'] == 'relation_variables'
                  and c['data']['loaded_state']['identity'] == receiver[0])
    read_supports = {c['unknown_identity']: {e['fact_id'] for e in c['facts']}
                     for c in children[loaded['call_id']] if c['operation'] == '_supports'}
    first, second, fact_id = ('unknown:1293', 'mother'), ('unknown:1643', 'condition'), 'episode:203'
    assert first in uses[receiver] and second in uses[receiver]
    assert fact_id in read_supports[first[0]] & read_supports[second[0]]
    state_uses = {(x[0]['identity'], x[0]['port']) for x in d['state']['variable_uses']}
    assert state_uses == uses[receiver]
    row_start = 0
    for fact, count in zip(design['facts'], design['evidence_row_counts']):
        if fact['fact_id'] == fact_id:
            if count != 1:
                raise AssertionError('this finite witness requires one recorded row')
            break
        row_start += count
    else:
        raise AssertionError('fact is not used by the native posterior')
    with (obs/'tensors.bin').open('rb') as stream:
        x = load(stream, posterior['design'])
        q = load(stream, posterior['query'])
        coefficient = load(stream, posterior['prediction']['coefficient'])
        correction = load(stream, posterior['prediction']['mean'])
    assert len(x) == sum(design['evidence_row_counts'])
    messages = (q @ x.T)[:, :, None] * coefficient[None]
    message = messages[0, row_start]
    channel = int(np.abs(message).argmax())
    # Exact rational embedding of the captured FP32 operands.
    scalar = sum(Fraction(float(q[0, k]))*Fraction(float(x[row_start, k]))
                 for k in range(x.shape[1])) * Fraction(float(coefficient[row_start, channel]))
    assert scalar != 0

    def lift(unknown):
        out = []
        for s, t in edges:
            out.append(-int(unknown in uses[t]) if s == receiver else
                       int(unknown in uses[s]) if t == receiver else 0)
        return np.array(out, dtype=np.int64)

    delta = lift(first) - lift(second)
    curl = np.asarray(b2.T @ delta).ravel()
    triangle_index = int(np.flatnonzero(curl)[0])
    triangle = cells[2][triangle_index]
    triangle_boundary = make_boundary(cells[1], [triangle])[0]
    # Measure the ambiguity, not a native Hodge activity field.
    ev, vectors = eigh((b1 @ b1.T).toarray().astype(float))
    nonzero = ev > 1e-10*max(1., ev.max())
    potential = vectors[:, nonzero] @ ((vectors[:, nonzero].T @ (b1 @ delta))/ev[nonzero])
    grad = np.asarray(b1.T @ potential).ravel()
    harmonic = h @ (h.T @ delta)
    rotational = delta - grad - harmonic
    fractions = {name: float(value @ value / (delta @ delta)) for name, value in
                 [('gradient', grad), ('curl', rotational), ('harmonic', harmonic)]}

    # Exact full-complex certificates for all edges incident to receiver.
    # Cut gradients and actual supported triangle boundaries span these units.
    star = [i for i, e in enumerate(edges) if receiver in e]
    clique = sorted({receiver} | {v for i in star for v in edges[i]})
    clique_edges = [i for i, e in enumerate(edges) if set(e) <= set(clique)]
    remaining = [e for i, e in enumerate(edges) if i not in clique_edges]
    pieces = components(vertices, remaining)
    cuts = [next(piece for piece in pieces if v in piece) for v in clique]
    if len({tuple(sorted(piece)) for piece in cuts}) != len(clique):
        raise AssertionError('the actual block is not separated by these cuts')
    cut_gradients = np.array([[int(t in cut)-int(s in cut) for s, t in edges] for cut in cuts]).T
    face_ids = [i for i, f in enumerate(cells[2]) if set(f) <= set(clique)]
    face_boundaries = b2[:, face_ids].toarray()
    span = np.concatenate((cut_gradients, face_boundaries), axis=1).astype(np.int64)
    certificates = []
    for edge in star:
        unit = np.zeros(len(edges), dtype=np.int64)
        unit[edge] = 1
        small = Matrix(span[clique_edges].tolist())
        solution, params = small.gauss_jordan_solve(Matrix(unit[clique_edges].tolist()))
        values = [v.subs({symbol: 0 for symbol in params}) for v in solution]
        scale = math.lcm(*(int(v.q) for v in values))
        coefficients = [int(v*scale) for v in values]
        if not np.array_equal(span @ np.array(coefficients, dtype=np.int64), scale*unit):
            raise AssertionError('certificate fails on full existing 976-edge complex')
        certificates.append(dict(edge=edge, scale=scale, coefficients=coefficients))

    # Capture data/identity scope, including genuinely old pending snapshots.
    scope_counts = collections.Counter()
    scope_mismatches = []
    for r in signals.values():
        data = r['data']
        if data['operation'] != 'mechanism':
            continue
        state = data['state']; identity = (state['identity'], state['generation'])
        actual = {(x[0]['identity'], x[0]['port']) for x in state['variable_uses']}
        matches = actual == uses[identity]
        scope_counts['matches' if matches else 'pending_or_different_scope'] += 1
        if not matches:
            scope_mismatches.append(dict(call_id=data['call_id'], relation=identity,
                stored_uses=sorted(actual), reference_uses=sorted(uses[identity])))
    origins = {}
    with (a.task/'body/metrics.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            revision = (row.get('shared_unknown') or {}).get('revision') or {}
            roles = (row.get('causal_program') or {}).get('roles') or {}
            if revision.get('committed') and revision.get('observed_episode') is not None and 'identity' in roles:
                key = 'episode:'+str(revision['observed_episode'])
                value = (roles['identity'], roles['generation'])
                if key in origins and origins[key] != value:
                    raise AssertionError('ambiguous real fact origin')
                origins[key] = value
    origin_candidates = []
    for fact in design['facts']:
        origin = origins.get(fact['fact_id'])
        edge = tuple(sorted((origin, receiver))) if origin and origin != receiver else None
        status = ('origin_not_in_retained_history' if origin is None else 'same_relation'
                  if origin == receiver else 'reference_edge' if edge in set(edges) else 'outside_reference_complex')
        origin_candidates.append(dict(fact_id=fact['fact_id'], origin=origin, status=status))

    lean = emit_lean(vertices, edges, clique, cuts, cells[2], face_ids, span,
                     certificates, delta, triangle_index, triangle_boundary, scalar,
                     receiver, first, second, fact_id, source, sha(source),
                     supports[first], supports[second], read_supports[first[0]], read_supports[second[0]])
    a.lean_output.write_text(lean, encoding='utf8')
    result = dict(status='numeric_checks_passed_lean_pending',
        reference_complex=dict(vertices=len(vertices), edges=len(edges), faces=len(cells[2]),
            harmonic_dimension=h.shape[1], existing_dirac_recomputed=False,
            reused_harmonic_basis_residuals=kernel_residuals),
        native_call=dict(call_id=d['call_id'], event=call['event'], update_clock=call['update_clock'],
            relation=receiver, port=d['kind'], fact_id=fact_id, evidence_row=row_start,
            original_query_shape=q.shape, original_design_shape=x.shape,
            exact_message_coordinate=dict(index=channel, numerator=scalar.numerator, denominator=scalar.denominator),
            native_message_l2=float(np.linalg.norm(message)),
            fp64_row_sum_vs_original_fp32_max_abs=float(np.abs(messages.sum(1)-correction).max())),
        candidate=dict(name='allocate one pooled fact among its recorded unknown owners, then shared-witness edge restriction',
            owners=[first, second], allocations=[[1, 0], [0, 1]],
            native_pooled_message_same_exact=True, delta_edges=np.flatnonzero(delta).tolist(),
            delta_coefficients=delta[np.flatnonzero(delta)].tolist(),
            curl_witness=dict(triangle=triangle, triangle_index=triangle_index,
                boundary=triangle_boundary, integer_difference=int(curl[triangle_index])),
            hodge_of_attribution_difference=fractions,
            harmonic_projection_coefficient_norm=float(np.linalg.norm(h.T @ delta)),
            verdict='not identifiable for full Hodge decomposition; these are representation witnesses, NOT measured native activity'),
        exact_star_annihilation=dict(receiver=receiver, incident_edges=star,
            clique=clique, cut_sizes=[len(c) for c in cuts], face_ids=face_ids,
            certificates=certificates,
            statement='Every unit on these four edges lies exactly in im(B1.T)+im(B2), hence no harmonic covector reads it. Applies to this reference complex, not all body activity.'),
        branch_scope=dict(counts=dict(scope_counts), mismatches=scope_mismatches),
        historical_origin_candidate=origin_candidates,
        actual_hodge_activity_measured=False, body_updates=0, world_calls=0,
        next_step='Step 1/2 incomplete for a physical edge field: retain the certified complex, reject attribution as full-Hodge measurement, and use exact star certificates to constrain any successor mapping.',
        lean=dict(path=str(a.lean_output), sha256=sha(a.lean_output), status='pending'),
        sources=[dict(path=str(p), sha256=sha(p)) for p in (source, obs/'tensors.bin', db_path,
            a.task/'body/metrics.jsonl', project/'results/published/cgb_native_harmonic_observability_2401_20261117.npz',
            project/'results/published/cgb_native_dowker_measurement_2401_20261117.json')])
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status=result['status'], native_call=d['call_id'],
        same_pooled_message=True, curl_difference=int(curl[triangle_index]),
        attribution_difference_hodge_fractions=fractions,
        full_complex_integer_certificates=len(certificates),
        matched_prediction_scopes=scope_counts['matches'],
        old_pending_scopes=scope_counts['pending_or_different_scope'],
        actual_hodge_activity_measured=False), ensure_ascii=False))


def emit_lean(vertices, edges, clique, cuts, faces, face_ids, span, certificates,
              delta, triangle_index, triangle_boundary, scalar, receiver,
              first, second, fact_id, source, source_hash, first_support,
              second_support, first_facts, second_facts):
    """Finite certificate uses all actual edges, not a replacement small model."""
    vi = {v: i for i, v in enumerate(vertices)}
    arr = lambda xs: '#['+', '.join(map(str, xs))+']'
    nested = lambda xs: '#['+', '.join(arr(x) for x in xs)+']'
    lines = ['import Mathlib.Tactic', '', '/-!',
        'Measurement certificate on all 280 vertices / 976 edges of the existing actual complex.',
        f'Source: {source.as_posix()}', f'SHA256: {source_hash}',
        f'Recorded fact {fact_id}, receiver {receiver}, owners {first} and {second}.',
        'Alternative allocations are a measurement counterexample, not a new body or training task.',
        'Integer data checks use native_decide; the annihilation theorem below is ordinary Lean.',
        '-/', 'namespace CGB.NativeMessageMeasurement', '',
        'def edges : Array (Nat × Nat) := '+arr(f'({vi[s]}, {vi[t]})' for s,t in edges),
        'def cutNodes : Array (Array Nat) := '+nested([[vi[v] for v in sorted(c)] for c in cuts]),
        'def faceBoundaries : Array (Array (Nat × Int)) := #[' + ', '.join(
            arr(f'({e}, {sign})' for e,sign in make_boundary([tuple(e) for e in edges], [faces[i]])[0])
            for i in face_ids)+']',
        'def spanValue (j e : Nat) : Int :=',
        '  if j < cutNodes.size then',
        '    (if cutNodes[j]!.contains edges[e]!.2 then 1 else 0) -',
        '    (if cutNodes[j]!.contains edges[e]!.1 then 1 else 0)',
        '  else (faceBoundaries[j-cutNodes.size]!).foldl',
        '    (fun n term => if term.1 == e then n + term.2 else n) 0',
        'def scales : Array Int := '+arr(c['scale'] for c in certificates),
        'def targets : Array Nat := '+arr(c['edge'] for c in certificates),
        'def coefficients : Array (Array Int) := '+nested([c['coefficients'] for c in certificates]),
        'def certificate (k e : Nat) : Prop :=',
        f'  (∑ j : Fin {span.shape[1]}, (coefficients[k]!)[j.val]! * spanValue j.val e) =',
        '    if e == targets[k]! then scales[k]! else 0',
        'instance (k e : Nat) : Decidable (certificate k e) := inferInstanceAs (Decidable (_ = _))',
        'def certificatesChecked : Bool :=',
        '  (List.range targets.size).all (fun k => scales[k]! != 0 &&',
        '    (List.range edges.size).all (fun e => decide (certificate k e)))',
        'theorem actual_full_complex_certificates : certificatesChecked = true := by native_decide',
        f'theorem actual_certificates_pointwise : ∀ k : Fin {len(certificates)}, ∀ e : Fin {len(edges)},',
        '    certificate k.val e.val := by native_decide',
        f'theorem actual_scales_nonzero : ∀ k : Fin {len(certificates)}, scales[k.val]! ≠ 0 := by native_decide',
        'def firstSupport : Array Nat := '+arr(vi[v] for v in sorted(first_support)),
        'def secondSupport : Array Nat := '+arr(vi[v] for v in sorted(second_support)),
        f'def receiverIndex : Nat := {vi[receiver]}',
        'def candidateLift (support : Array Nat) (e : Nat) : Int :=',
        '  if edges[e]!.1 == receiverIndex then -(if support.contains edges[e]!.2 then 1 else 0)',
        '  else if edges[e]!.2 == receiverIndex then (if support.contains edges[e]!.1 then 1 else 0)',
        '  else 0',
        'def delta : Array Int := '+arr(delta.tolist()),
        f'theorem actual_delta_from_supports : ∀ e : Fin {len(edges)},',
        '    delta[e.val]! = candidateLift firstSupport e.val - candidateLift secondSupport e.val := by native_decide',
        'def firstFacts : Array String := '+arr(json.dumps(x) for x in sorted(first_facts)),
        'def secondFacts : Array String := '+arr(json.dumps(x) for x in sorted(second_facts)),
        f'theorem actual_fact_has_both_supports : firstFacts.contains {json.dumps(fact_id)} = true ∧',
        f'    secondFacts.contains {json.dumps(fact_id)} = true := by native_decide',
        'def curlWitness : Int := '+ ' + '.join(f'({s}) * delta[{e}]!' for e,s in triangle_boundary),
        f'theorem actual_curl_difference : curlWitness = {int(sum(s*delta[e] for e,s in triangle_boundary))} := by native_decide',
        f'def originalMessageCoordinate : ℚ := ({scalar.numerator} : ℚ) / {scalar.denominator}',
        'theorem original_message_coordinate_nonzero : originalMessageCoordinate ≠ 0 := by',
        '  norm_num [originalMessageCoordinate]',
        'theorem actual_same_native_message_different_curl :',
        '    1*originalMessageCoordinate+0*originalMessageCoordinate =',
        '      0*originalMessageCoordinate+1*originalMessageCoordinate ∧',
        '    (curlWitness : ℚ)*originalMessageCoordinate ≠ 0 := by',
        '  constructor',
        '  · ring',
        '  · rw [actual_curl_difference]',
        '    simpa using original_message_coordinate_nonzero',
        '/-- Native pooled calculation sees the same fact once under either proposed owner. -/',
        'theorem pooled_message_preserved (m : ℝ) : 1*m+0*m = 0*m+1*m := by ring',
        '',
        '/-- A verified full-edge integer span certificate annihilates every covector',
        'orthogonal to the actual cut gradients and triangle boundaries. Instantiate g',
        'with each certified unit; harmonic covectors obey these orthogonality conditions. -/',
        'theorem certified_span_annihilates',
        '    {E J : Type*} [Fintype E] [Fintype J]',
        '    (g h : E → ℝ) (columns : J → E → ℝ) (c : J → ℝ) (scale : ℝ)',
        '    (nonzero : scale ≠ 0)',
        '    (identity : ∀ e, scale * g e = ∑ j, c j * columns j e)',
        '    (annihilates : ∀ j, ∑ e, columns j e * h e = 0) :',
        '    ∑ e, g e * h e = 0 := by',
        '  have zero : scale * (∑ e, g e * h e) = 0 := by',
        '    calc',
        '      _ = ∑ e, (scale * g e) * h e := by',
        '        rw [Finset.mul_sum]',
        '        apply Finset.sum_congr rfl',
        '        intro e _',
        '        ring',
        '      _ = ∑ e, (∑ j, c j * columns j e) * h e := by simp_rw [identity]',
        '      _ = ∑ j, c j * (∑ e, columns j e * h e) := by',
        '        simp_rw [Finset.sum_mul, mul_assoc]',
        '        rw [Finset.sum_comm]',
        '        simp_rw [Finset.mul_sum]',
        '      _ = 0 := by simp [annihilates]',
        '  exact (mul_eq_zero.mp zero).resolve_left nonzero',
        '',
        f'def actualColumn (j : Fin {span.shape[1]}) (e : Fin {len(edges)}) : ℝ :=',
        '  (spanValue j.val e.val : ℝ)',
        f'def actualUnit (k : Fin {len(certificates)}) (e : Fin {len(edges)}) : ℝ :=',
        '  if e.val == targets[k.val]! then 1 else 0',
        f'theorem actual_star_annihilates (k : Fin {len(certificates)}) (h : Fin {len(edges)} → ℝ)',
        f'    (orthogonal : ∀ j : Fin {span.shape[1]}, ∑ e, actualColumn j e * h e = 0) :',
        '    ∑ e, actualUnit k e * h e = 0 := by',
        '  apply certified_span_annihilates (actualUnit k) h actualColumn',
        '    (fun j => ((coefficients[k.val]!)[j.val]! : ℝ)) (scales[k.val]! : ℝ)',
        '  · exact_mod_cast actual_scales_nonzero k',
        '  · intro e',
        '    have raw := actual_certificates_pointwise k e',
        '    unfold certificate at raw',
        '    have casted := congrArg (fun n : Int => (n : ℝ)) raw',
        '    push_cast at casted',
        '    simpa [actualUnit, actualColumn, mul_ite] using casted.symm',
        '  · exact orthogonal',
        '', 'end CGB.NativeMessageMeasurement', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
