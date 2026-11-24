"""Hodge measurement of edge-restricted native residual response.

The signal is an analytic response of an actually executed posterior, not an
allocation of its correction to putative owners. For an edge, select the UNION
of facts in its common typed unknowns. Perturb their encoded residuals together.
No extra prediction, learning, world call, or fact-owner allocation is run.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_native_path_composition import load
from scripts.audit_native_dowker_measurement import make_boundary, to_sparse


def sql_fact_support(db, identity):
    """Mirror _read_evidence's persisted validity rules without importing Torch."""
    schema = json.loads(db.execute('SELECT value FROM meta WHERE key="schema"').fetchone()[0])
    offsets, offset = {}, 0
    for name in sorted(schema):
        spec = schema[name]
        count = int(np.prod(spec['shape'])) if spec['shape'] else 1
        offsets[name] = offset, spec['dtype'], count
        offset += count*np.dtype(spec['dtype']).itemsize
    result = set()
    for eid, payload in db.execute('SELECT e.id,e.payload FROM unknown_evidence u '
            'JOIN episodes e ON e.id=u.episode_id WHERE u.unknown_id=? ORDER BY u.ordinal',
            (identity,)):
        assert len(payload) == offset
        def scalar(name):
            start, dtype, count = offsets[name]
            assert count == 1
            return np.frombuffer(payload, dtype=dtype, count=1, offset=start)[0].item()
        if scalar('count') != 1 or not scalar('mother_valid') or scalar('mother_stamp') != scalar('last_write'):
            raise ValueError('invalid observed support in actual persistent archive')
        result.add((f'episode:{eid}', 'mother'))
        if scalar('internal_valid') and scalar('internal_stamp') == scalar('last_write'):
            result.add((f'episode:{eid}', 'decision'))
    return frozenset(result)


def scope_certificate(root, output, base_cells, cells, owners):
    """Finite actual-incidence certificate for inclusion, not a substitute graph."""
    # All old cells/vertices survive, with their original oriented order.
    vertex_ids = {c[0]: i for i, c in enumerate(cells[0])}
    vertices = [[vertex_ids[v] for v in c] for layer in cells for c in layer]
    assert all(v == sorted(v) for v in vertices)
    support_keys = sorted(owners)
    support_ids = {u: i for i, u in enumerate(support_keys)}
    supports = [sorted(vertex_ids[r] for r in owners[u]) for u in support_keys]
    encoded = []
    for layer in cells:
        groups = []
        for cell in layer:
            witness = next(u for u in support_keys if set(cell) <= owners[u])
            groups.append(str(support_ids[witness])+':'+','.join(str(vertex_ids[v]) for v in cell))
        encoded.append('|'.join(groups))
    generic = (root/'lean/CGBAxioms/reviews/NativeDowker.lean').read_text(encoding='utf8')
    text = generic + '\nnamespace CGB.NativePendingScopeActual2401\nopen CGB.NativeDowker\n'
    text += 'def actualSupports : Array (List Nat) := #'+str(supports)+'\n'
    for vertex in vertex_ids:
        if len(vertex) != 3:
            continue
        current = [support_ids[u] for u in support_keys if vertex[:2] in owners[u]]
        pending = [support_ids[u] for u in support_keys if vertex in owners[u]]
        text += 'def currentDependencies : List Nat := '+str(current)+'\n'
        text += 'def pendingDependencies : List Nat := '+str(pending)+'\n'
        text += 'example : currentDependencies ≠ pendingDependencies := by native_decide\n'
    for n, encoded_layer in enumerate(encoded):
        text += f'def expanded{n} : Array Cell := decodeCells "{encoded_layer}"\n'
        old = [[vertex_ids[v] for v in cell] for cell in base_cells[n]]
        text += f'def inherited{n} : Array (List Nat) := #'+str(old)+'\n'
        text += f'example : expanded{n}.size = {len(cells[n])} := by native_decide\n'
        text += f'example : expanded{n}.all (validCell actualSupports {len(cells[0])} {n+1}) = true := by native_decide\n'
        text += f'example : completeCells actualSupports {n+1} expanded{n} = true := by native_decide\n'
        text += f'example : inherited{n}.all (fun old => expanded{n}.any (fun c => c.vertices == old)) = true := by native_decide\n'
    # Oriented boundary inclusions follow by matching each inherited ordered
    # cell and each of its ordered faces. Record every one, not just counts.
    text += '''
def inheritedBoundaryIncluded (low high : Array (List Nat))
    (expandedLow expandedHigh : Array Cell) : Bool :=
  high.all (fun c => expandedHigh.any (fun e => e.vertices == c) &&
    (List.range c.length).all (fun j =>
      low.contains (c.eraseIdx j) &&
      expandedLow.any (fun e => e.vertices == c.eraseIdx j)))

def inclusionCommutes (low high : Array (List Nat))
    (expandedLow expandedHigh : Array Cell) : Bool :=
  let oldLow : Array Cell := low.map (fun vertices => ⟨vertices, 0⟩)
  let oldHigh : Array Cell := high.map (fun vertices => ⟨vertices, 0⟩)
  let lowMap := low.map (fun c =>
    (expandedLow.toList.findIdx? (fun e => e.vertices == c)).getD expandedLow.size)
  let oldBoundary := deriveBoundary oldLow oldHigh
  let newBoundary := deriveBoundary expandedLow expandedHigh
  (List.range high.size).all (fun i =>
    let j := (expandedHigh.toList.findIdx? (fun e => e.vertices == high[i]!)).getD expandedHigh.size
    j < expandedHigh.size &&
    oldBoundary[i]!.map (fun term => (lowMap[term.1]!, term.2)) == newBoundary[j]!)
'''
    for n in (1, 2, 3):
        text += f'example : inheritedBoundaryIncluded inherited{n-1} inherited{n} expanded{n-1} expanded{n} = true := by native_decide\n'
        text += f'example : inclusionCommutes inherited{n-1} inherited{n} expanded{n-1} expanded{n} = true := by native_decide\n'
    text += 'end CGB.NativePendingScopeActual2401\n'
    output.write_text(text, encoding='utf8')
    return dict(path=str(output), sha256=digest(output), status='pending',
                inherited_cells=list(map(len, base_cells)), expanded_cells=list(map(len, cells)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lean-output', type=Path)
    p.add_argument('--include-pending-scope', action='store_true',
                   help='Retain actual saved dependency scopes as distinct vertices')
    p.add_argument('--scope-lean-output', type=Path)
    p.add_argument('--check-lean', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    obs = a.task/'work/signal_admission_2401_20261117/pair_observed_observations'
    source = obs/'operations.jsonl.gz'
    with gzip.open(source, 'rt', encoding='utf8') as stream:
        records = [json.loads(line) for line in stream]
    calls = {r['data']['call_id']: r for r in records if r['kind'] == 'relation_signal'}
    children = defaultdict(list)
    for row in calls.values():
        children[row['data']['parent_id']].append(row['data'])
    db_path = next((a.task/'body').glob('*.sqlite'))
    db = sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True)
    uses = defaultdict(set)
    owners = defaultdict(set)
    for rid, uid, port, gen in db.execute(
            'SELECT relation_id,unknown_id,port,generation FROM relation_variables'):
        uses[rid, gen].add((uid, port))
        owners[uid, port].add((rid, gen))
    cells = [sorted({c for group in owners.values()
                     for c in itertools.combinations(sorted(group), n)})
             for n in (1, 2, 3)]
    base_cells = cells + [sorted({c for group in owners.values()
                                 for c in itertools.combinations(sorted(group), 4)})]
    base_uses = {r: frozenset(u) for r, u in uses.items()}
    identities_path = root/'results/published/cgb_native_harmonic_observability_2401_20261117.json'
    identities = json.loads(identities_path.read_text(encoding='utf8'))
    assert identities['identities']['edges'] == [[list(v) for v in e] for e in cells[1]]
    basis_path = identities_path.with_suffix('.npz')
    scope_vertices = {}
    support_sql_versions = 0
    if a.include_pending_scope:
        for row in calls.values():
            d = row['data']
            if d['operation'] != 'mechanism':
                continue
            rid = d['state']['identity'], d['state']['generation']
            scope = frozenset((v['identity'], v['port'])
                              for v, _, _ in d['state']['variable_uses'])
            if scope != base_uses[rid]:
                key = (rid, tuple(sorted(scope)))
                if key not in scope_vertices:
                    # The third coordinate names the captured saved scope; it
                    # never replaces or mutates the extant two-coordinate node.
                    vertex = (*rid, len(scope_vertices)+1)
                    scope_vertices[key] = vertex
                    uses[vertex] = set(scope)
                    for unknown in scope:
                        owners[unknown].add(vertex)
        cells = [sorted({c for group in owners.values()
                         for c in itertools.combinations(sorted(group), n)})
                 for n in (1, 2, 3)]
        assert all(set(base) <= set(expanded) for base, expanded in zip(base_cells, cells))
    b1 = to_sparse(make_boundary(cells[0], cells[1]), len(cells[0])).astype(float)
    b2 = to_sparse(make_boundary(cells[1], cells[2]), len(cells[1])).astype(float)
    if scope_vertices:
        ev, evec = eigh((b1.T@b1+b2@b2.T).toarray())
        harmonic = evec[:, abs(ev) < 1e-10*max(1., abs(ev).max())]
    else:
        harmonic = np.load(basis_path)['harmonic_basis']
    values, vectors = eigh((b1@b1.T).toarray())
    active = values > 1e-10
    node_inverse = (vectors[:, active]/values[active])@vectors[:, active].T
    # Captured support reads, indexed by the exact dependency version they loaded.
    supports = {}
    for row in calls.values():
        d = row['data']
        if d['operation'] != 'relation_variables':
            continue
        by_id = {c['unknown_identity']: c for c in children[d['call_id']]
                 if c['operation'] == '_supports'}
        for variable, version, _ in d['loaded_state']['variable_uses']:
            uid, port = variable['identity'], variable['port']
            if uid in by_id:
                facts = frozenset((f['fact_id'], f['kind']) for f in by_id[uid]['facts'])
                key = uid, port, version
                if key in supports:
                    assert supports[key] == facts, 'same version has different supports'
                supports[key] = facts
    if a.include_pending_scope:
        # These are reads of the same version retained in the pending scope,
        # not substitutions of current evidence for a stale dependency version.
        for row in calls.values():
            d = row['data']
            if d['operation'] != 'mechanism':
                continue
            for variable, version, _ in d['state']['variable_uses']:
                uid, port = variable['identity'], variable['port']
                key = uid, port, version
                if key in supports:
                    continue
                actual_version = db.execute('SELECT version FROM unknown_bindings WHERE id=?',
                                            (uid,)).fetchone()
                if actual_version != (version,):
                    raise ValueError(f'uncaptured historical support version: {key}')
                supports[key] = sql_fact_support(db, uid)
                support_sql_versions += 1
    all_results, measured, omitted = [], [], []
    certificate_cases = []
    arrays = {}
    maximum = dict(native_mean_error=0., native_coefficient_error=0.,
                   affine_identity_error=0., reconstruction_error=0.,
                   divergence_curl_error=0., curl_gradient_error=0.,
                   orthogonality_error=0.)
    with (obs/'tensors.bin').open('rb') as handle:
        for call_id, row in sorted(calls.items()):
            d = row['data']
            if d['operation'] != 'mechanism':
                continue
            native = next((c for c in children[call_id] if c['operation'] == 'posterior'), None)
            design = next((c for c in children[call_id]
                           if c['operation'] == '_bound_evidence_design'), None)
            if native is None or design is None:
                raise ValueError('executed mechanism lacks its native row design')
            x, y, q = (load(handle, native[k]) for k in ('design', 'residual', 'query'))
            kernel = x@x.T + native['ridge']*np.eye(len(x))
            # d mean / d Y = q X^T K^-1. Coefficient rows K^-1 Y are
            # correlated mixtures, not the causal effect of one fact residual.
            alpha = np.linalg.solve(kernel, (q@x.T).T).T
            mean = alpha@y
            native_mean = load(handle, native['prediction']['mean'])
            native_coefficient = load(handle, native['prediction']['coefficient'])
            maximum['native_mean_error'] = max(maximum['native_mean_error'],
                                               float(np.max(abs(mean-native_mean))))
            maximum['native_coefficient_error'] = max(maximum['native_coefficient_error'],
                float(np.max(abs(np.linalg.solve(kernel, y)-native_coefficient))))
            fact_keys = []
            for fact_index, (fact, count) in enumerate(zip(
                    design['facts'], design['evidence_row_counts'])):
                identity = fact['fact_id'] or f'local:{call_id}:{fact_index}'
                fact_keys.extend([(identity, fact['kind'])]*count)
            assert len(fact_keys) == len(x)
            rid = d['state']['identity'], d['state']['generation']
            scope = {(v['identity'], v['port']): version
                     for v, version, _ in d['state']['variable_uses']}
            result = dict(call_id=call_id, event=row['event'], update_clock=row['update_clock'],
                          relation=list(rid), output_port=d['kind'],
                          transport=design['transport'], query_rows=len(q),
                          fact_rows=len(x), fact_keys=fact_keys,
                          caller=d['caller'])
            mixed_rows = (q@x.T)[:, :, None]*native_coefficient[None]
            physical_rows = alpha[:, :, None]*y[None]
            result['mixed_coefficient_rows_vs_residual_response_relative_l2'] = float(
                np.linalg.norm(mixed_rows-physical_rows)/max(np.linalg.norm(physical_rows), 1e-300))
            if a.include_pending_scope and set(scope) != base_uses[rid]:
                rid = scope_vertices[rid, tuple(sorted(scope))]
                result['captured_scope_vertex'] = list(rid)
            if set(scope) != uses[rid] or any((u, port, version) not in supports
                                             for (u, port), version in scope.items()):
                result['hodge_scope'] = 'saved pending dependencies, separate from current carrier'
                omitted.append(call_id)
                all_results.append(result)
                arrays[f'fact_response_{call_id}'] = alpha[:, :, None]*y[None]
                continue
            edge_ids, masks = [], []
            for e, (left, right) in enumerate(cells[1]):
                if rid not in (left, right):
                    continue
                common = uses[left] & uses[right]
                fact_sets = [supports[u, port, scope[u, port]] for u, port in sorted(common)]
                union = set().union(*fact_sets)
                edge_ids.append(e)
                masks.append([int(key in union) for key in fact_keys])
                certificate_cases.append((tuple(tuple(sorted(
                    {int(f.split(':')[1]) * 2 + int(k == 'decision') for f, k in group}))
                    for group in fact_sets),
                    [int(f.split(':')[1]) * 2 + int(k == 'decision')
                     if f.startswith('episode:') else 0 for f, k in fact_keys],
                    masks[-1]))
            masks = np.asarray(masks, dtype=float)
            # phi is the response contributed by THIS invocation. Other
            # invocations are distinct measurements, not missing node predictions.
            response = np.einsum('bf,ef,fd->ebd', alpha, masks, y)
            signs = np.array([1 if cells[1][e][1] == rid else -1 for e in edge_ids])
            phi = np.zeros((len(cells[1]), len(q)*y.shape[1]))
            phi[edge_ids] = (response*signs[:, None, None]).reshape(len(edge_ids), -1)
            potential = node_inverse@(b1@phi)
            gradient = b1.T@potential
            harmonic_coefficients = harmonic.T@phi
            harm = harmonic@harmonic_coefficients
            curl = phi-gradient-harm
            rebuilt = gradient+curl+harm
            energies = np.array([np.square(z).sum() for z in (gradient, curl, harm)])
            total = float(np.square(phi).sum())
            # Verify the exact affine identity on every selected physical row set.
            # 1/8 is a check increment, not a fitted hyperparameter or body change.
            for mask, observed in zip(masks, response):
                actual = alpha@(y*(1.+.125*mask[:, None]))
                error = float(np.max(abs(actual-mean-.125*observed)))
                maximum['affine_identity_error'] = max(maximum['affine_identity_error'], error)
            maximum['reconstruction_error'] = max(maximum['reconstruction_error'],
                                                    float(np.max(abs(phi-rebuilt))))
            maximum['divergence_curl_error'] = max(maximum['divergence_curl_error'],
                                                    float(np.max(abs(b1@curl))))
            maximum['curl_gradient_error'] = max(maximum['curl_gradient_error'],
                                                  float(np.max(abs(b2.T@gradient))))
            cross = [abs(float(np.vdot(a_, b_))) for a_, b_ in
                     ((gradient, curl), (gradient, harm), (curl, harm))]
            maximum['orthogonality_error'] = max(maximum['orthogonality_error'], max(cross))
            result.update(hodge_scope='scope-expanded carrier' if scope_vertices else 'current certified carrier', edge_ids=edge_ids,
                mask=masks.astype(int).tolist(), total_energy=total,
                energies=dict(zip(('gradient', 'curl', 'harmonic'), map(float, energies))),
                energy_fractions=dict(zip(('gradient', 'curl', 'harmonic'),
                                          map(float, energies/total if total else energies))),
                response_norm=float(np.sqrt(total)),
                triangle_circulation_norm=float(np.linalg.norm(b2.T@phi)))
            arrays[f'edge_response_{call_id}'] = phi[edge_ids]
            # All posterior row responses, including those outside shared edge probes.
            arrays[f'fact_response_{call_id}'] = alpha[:, :, None]*y[None]
            measured.append(result)
            all_results.append(result)
    db.close()
    numerical = a.output.with_suffix('.npz')
    # The sparse inputs plus the inherited operators reconstruct every
    # projection, without a dense derived node field for every invocation.
    arrays['harmonic_basis'] = harmonic
    np.savez_compressed(numerical, **arrays)
    fractions = np.array([list(r['energy_fractions'].values()) for r in measured])
    report = dict(status='computed_pending_lean_binding',
        observable='edge-restricted encoded-residual linear response of an executed native posterior',
        formula='alpha=q X^T (X X^T+ridge I)^-1; phi_e=orientation * sum_f 1[f in UNION(shared supports of e)] alpha_f Y_f',
        meaning='How the actually executed forecast changes if the encoded residuals of the shared fact set are scaled together; not an ownership split or a claim of physical edge transport.',
        units='native encoded prediction; derivative with respect to dimensionless residual scaling',
        source=dict(operations=str(source), operations_sha256=digest(source),
            tensors_sha256=digest(obs/'tensors.bin'), sqlite_sha256=digest(db_path),
            inherited_basis_sha256=digest(basis_path), measurement_script_sha256=digest(__file__),
            native_posterior_source_sha256=digest(root/'scripts/ib_local/causal_relation_program.py'),
            support_reader_source_sha256=digest(root/'scripts/ib_local/shared_unknown_bindings.py')),
        carrier_dimensions=list(map(len, cells)), latent_output_width=128,
        inherited_carrier_dimensions=list(map(len, base_cells)),
        retained_saved_scope_vertices=[list(v) for v in scope_vertices.values()],
        added_edges=[[list(v) for v in e] for e in cells[1] if e not in set(base_cells[1])],
        exact_version_sql_support_reads=support_sql_versions,
        harmonic_dimension=harmonic.shape[1],
        native_calls=len(all_results), mapped_calls=len(measured),
        separately_preserved_pending_calls=omitted,
        maximum_numerical_errors=maximum,
        hodge_energy_fraction_summary=dict(
            order=['gradient', 'curl', 'harmonic'],
            minimum=fractions.min(0).tolist(), median=np.median(fractions, axis=0).tolist(),
            maximum=fractions.max(0).tolist(),
            aggregation='descriptive per invocation, not independent trials or a time integral'),
        calls=all_results, arrays=dict(path=str(numerical), sha256=digest(numerical)),
        actual_body_events_observed=3, actual_original_optimizer_updates_observed=4,
        new_model_updates=0, new_world_calls=0, main_body_cycle=2401,
        next='Bind the set-union response map in Lean; then carry this measured observable into the remaining paper quantities, with separate actual angle mappings for synchronization.')
    if a.lean_output:
        unique = sorted(set((tuple(s), tuple(f), tuple(m)) for s, f, m in certificate_cases))
        cases = ',\n  '.join('(' + str([list(v) for v in c[0]]) + ', ' + str(list(c[1]))
                             + ', ' + str(list(c[2])) + ')' for c in unique)
        actual_calls = ', '.join(str(r['call_id']) for r in measured)
        text = '''import Mathlib.Tactic

/-! Actual native posterior response measurement. No fitted proxy and no
assignment of a pooled correction to owners. The native inverse-kernel row
weights are held fixed while the selected encoded residual rows are scaled.
Finite masks below bind this definition to the captured actual event. -/
namespace CGB.NativeSharedResponseActual2401

variable {I : Type*} [Fintype I]

def forecast (weight residual : I → ℝ) : ℝ :=
  ∑ i, weight i * residual i

def response (weight residual : I → ℝ) (selected : I → Bool) : ℝ :=
  ∑ i, if selected i then weight i * residual i else 0

/-- Exact affine change of the original posterior, holding its actual
query/design/kernel/base fixed. It applies to each output and query row. -/
theorem native_residual_scaling (weight residual : I → ℝ)
    (selected : I → Bool) (step : ℝ) :
    forecast weight (fun i => residual i *
      (1 + if selected i then step else 0)) =
      forecast weight residual + step * response weight residual selected := by
  unfold forecast response
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  cases h : selected i
  · simp [h]
  · simp [h]
    ring

/-- Coordinate weights are columns of the inverse kernel, rather than the
mixtures already solved into the native coefficient rows. -/
theorem inverse_kernel_reassociation (queryDot : I → ℝ)
    (inverseKernel : I → I → ℝ) (residual : I → ℝ) :
    (∑ j, queryDot j * ∑ i, inverseKernel j i * residual i) =
      forecast (fun i => ∑ j, queryDot j * inverseKernel j i) residual := by
  unfold forecast
  simp only [Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Any two reference representations selecting the same distinct facts
give the same response, regardless of which unknown first supplied them. -/
theorem identical_fact_set_same_response (weight residual : I → ℝ)
    (a b : I → Bool) (same : ∀ i, a i = b i) :
    response weight residual a = response weight residual b := by
  unfold response
  apply Finset.sum_congr rfl
  intro i _
  rw [same i]

theorem reverse_edge_orientation (r : ℝ) : -(-r) = r := neg_neg r

def supportMask (supports : List (List Nat)) (fact : Nat) : Bool :=
  supports.flatten.contains fact

/-- Repeated references to the same observed supports never duplicate a fact. -/
theorem duplicate_supports_same_mask (supports : List (List Nat)) (fact : Nat) :
    supportMask (supports ++ supports) fact = supportMask supports fact := by
  simp [supportMask]

def actualMasks : List (List (List Nat) × List Nat × List Nat) := [
  CASES]

def maskMatches (c : List (List Nat) × List Nat × List Nat) : Bool :=
  c.2.2 == c.2.1.map (fun fact => if supportMask c.1 fact then 1 else 0)

-- Fact tokens are 2*episodeId + outputKind. Zero marks a call-local imagined
-- row, never present in a shared observed support. Each set is the union of the
-- actual captured supports common to that certified edge and call scope.
example : actualMasks.all maskMatches = true := by native_decide
example : actualMasks.all (fun c => c.1.all (fun s => s.eraseDups == s)) = true := by native_decide

def actualMappedCalls : List Nat := [CALLS]
example : actualMappedCalls.length = COUNT := by native_decide
example : actualMappedCalls.eraseDups.length = COUNT := by native_decide

end CGB.NativeSharedResponseActual2401
'''.replace('CASES', cases).replace('CALLS', actual_calls).replace('COUNT', str(len(measured)))
        a.lean_output.write_text(text, encoding='utf8')
        report['lean'] = dict(path=str(a.lean_output), sha256=digest(a.lean_output),
                              unique_actual_masks=len(unique), status='pending')
    if a.scope_lean_output:
        cells3 = sorted({c for group in owners.values()
                         for c in itertools.combinations(sorted(group), 4)})
        report['scope_lean'] = scope_certificate(root, a.scope_lean_output,
                                               base_cells, cells+[cells3], owners)
    if a.check_lean:
        if not a.lean_output:
            raise ValueError('--check-lean requires --lean-output')
        for name in ('lean', 'scope_lean'):
            if name not in report:
                continue
            entry = report[name]
            proof = (root/entry['path']).resolve()
            checked = subprocess.run(['lake', 'env', 'lean',
                str(proof.relative_to(root/'lean/CGBAxioms'))],
                cwd=root/'lean/CGBAxioms', capture_output=True, text=True,
                encoding='utf8', errors='replace')
            log = a.output.with_suffix('.'+name+'.log')
            log.write_text(checked.stdout+checked.stderr, encoding='utf8')
            entry.update(exit_code=checked.returncode, log=str(log))
            assert digest(proof) == entry['sha256'], 'proof changed during checking'
            if checked.returncode:
                a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
                raise RuntimeError(f'{name} failed; see {log}')
            entry['status'] = 'passed'
        report['status'] = 'response_mapping_and_hodge_checks_passed'
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: report[k] for k in ('native_calls','mapped_calls',
        'separately_preserved_pending_calls','maximum_numerical_errors','hodge_energy_fraction_summary')}, indent=2))


if __name__ == '__main__':
    main()
