"""Paper order parameters on native S/P angles, and actual relation D readouts.

The two typed carriers are kept distinct and aligned by native event IDs.
All inputs are already-recorded normal learning events. No Torch or new run.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_native_dowker_measurement import make_boundary, to_sparse


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complex_list(value):
    z = np.asarray(value)
    return dict(real=z.real.tolist(), imag=z.imag.tolist(), modulus=abs(z).tolist())


def decode_cells(proof, name):
    raw = re.search(r'def '+name+r' : Array Cell := decodeCells "([^"]*)"', proof).group(1)
    return [tuple(map(int, x.split(':')[1].split(','))) for x in raw.split('|')]


def phase_measurement(obs):
    edges = list(itertools.combinations(range(8), 2))
    b = np.zeros((8, 28), dtype=int)
    lookup = {e: j for j, e in enumerate(edges)}
    for e, (i, j) in enumerate(edges):
        b[i, e], b[j, e] = -1, 1
    loops, oriented = [], []
    for i, j, k in itertools.combinations(range(8), 3):
        loop = [(lookup[i, j], 1), (lookup[j, k], 1), (lookup[i, k], -1)]
        loops.append([i, j, k])
        oriented.append(loop)
    results, sources, arrays = [], [], {}
    projection_error, overlap_min, overlap_max, wrapped_error = 0., 1., 0., 0.
    for path in sorted(obs.glob('physical_*.npz')):
        sources.append(dict(path=str(path), sha256=sha(path)))
        with np.load(path, allow_pickle=False) as z:
            c = z['complex_overlap_real'].astype(float)+1j*z['complex_overlap_imag'].astype(float)
            edge_c = c[:, :, tuple(i for i, _ in edges), tuple(j for _, j in edges)]
            mag = abs(edge_c)
            if np.any(mag == 0):
                raise ValueError('zero complex overlap has no phase; report its support separately')
            unit = edge_c/mag
            phase = np.angle(edge_c)
            # Box 2 X^- = mean exp(i B1 phi). Integer powers avoid phase unwrapping.
            projected = np.exp(1j*np.einsum('ve,tle->tlv', b, phase))
            by_products = np.prod(unit[:, :, None, :]**b[None, None, :, :], axis=-1)
            projection_error = max(projection_error, float(abs(projected-by_products).max()))
            # The principal lift is not part of the observable. Adding 2pi on
            # alternating edge coordinates must give the same paper quantity.
            lifted = phase+2*np.pi*(np.arange(28)%2)[None, None, :]
            wrapped_error = max(wrapped_error, float(abs(
                np.exp(1j*np.einsum('ve,tle->tlv', b, lifted))-projected).max()))
            xm = projected.mean(-1)
            xo = np.stack([sum(unit[:, :, e] if sign == 1 else unit[:, :, e].conj()
                               for e, sign in loop)/3 for loop in oriented], axis=-1)
            overlap_min, overlap_max = min(overlap_min, float(mag.min())), max(overlap_max, float(mag.max()))
            for t, kind in enumerate(z['kind']):
                idx = len(results)
                event = int(z['event'][t])
                results.append(dict(sample=idx, event=event, kind=str(kind),
                    quiet_time=float(z['quiet_time'][t]), update_clock=int(z['updates'][t]),
                    X_minus_by_layer=complex_list(xm[t]),
                    X_minus_all_256_nodes=complex_list(projected[t].mean()),
                    X_o_modulus_quantiles=np.quantile(abs(xo[t]), [0,.25,.5,.75,1]).tolist(),
                    overlap_magnitude_quantiles=np.quantile(mag[t],[0,.25,.5,.75,1]).tolist()))
                arrays[f'X_o_{idx}'] = xo[t]
                arrays[f'edge_unit_phasor_{idx}'] = unit[t]
    before_after = []
    for left, right in zip(results, results[1:]):
        if left['event'] != right['event']:
            continue
        pair = left['kind'], right['kind']
        if pair not in [('quiet_before', 'quiet_after'), ('update_before', 'update_after'),
                        ('preview_quiet_before', 'preview_quiet_after')]:
            continue
        a, d = arrays[f'X_o_{left["sample"]}'], arrays[f'X_o_{right["sample"]}']
        before_after.append(dict(event=left['event'], operation=pair[0].removesuffix('_before'),
            update_clock_before=left['update_clock'], update_clock_after=right['update_clock'],
            median_abs_complex_X_o_change=float(np.median(abs(d-a))),
            max_abs_complex_X_o_change=float(abs(d-a).max())))
    return dict(carrier='32 actual pair-potential graphs, 8 heads and 28 physical pair terms per layer',
        native_complex_coordinate='sqrt(Z)*S + i*P/sqrt(Z)',
        angle='arg <z_i,z_j> with retained overlap magnitude; matrix relative phase',
        edges=edges, closed_oriented_loops=oriented, loop_vertices=loops,
        C2_cells=0, X_plus=None,
        X_plus_reason='This physical carrier contains native pair terms; these closed walks are not filled faces.',
        branch_lift_invariance_max_error=wrapped_error,
        direct_exp_vs_circle_product_max_error=projection_error,
        overlap_magnitude_range=[overlap_min,overlap_max],
        native_sample_kinds=dict(Counter(r['kind'] for r in results)),
        samples=results, operation_changes=before_after, sources=sources,
        interpretation='Paper Box 2 observables on the physical phase carrier. Relation response fields retain their real encoded units; no relation phase is synthesized.'), arrays


def dirac_measurement(project, report):
    source = project/'results/published/cgb_native_shared_response_scoped_2401_20261117.json'
    r = json.loads(source.read_text(encoding='utf8'))
    assert r['status'] == 'response_mapping_and_hodge_checks_passed'
    proof_path = project/r['scope_lean']['path']
    assert sha(proof_path) == r['scope_lean']['sha256']
    text = proof_path.read_text(encoding='utf8')
    cells = [decode_cells(text, f'expanded{n}') for n in range(4)]
    b1 = to_sparse(make_boundary(cells[0], cells[1]), len(cells[0])).astype(float)
    b2 = to_sparse(make_boundary(cells[1], cells[2]), len(cells[1])).astype(float)
    node_ids = json.loads((project/'results/published/cgb_native_harmonic_observability_2401_20261117.json').read_text(encoding='utf8'))['identities']['relations']
    base_nodes = [tuple(v) for v in node_ids]
    expanded_nodes = sorted(base_nodes+[tuple(v) for v in r['retained_saved_scope_vertices']])
    old_text = (project/'lean/CGBAxioms/reviews/NativeDowkerActual2401.lean').read_text(encoding='utf8')
    old_cells = [decode_cells(old_text, f'cells{n}') for n in range(4)]
    old_b1 = to_sparse(make_boundary(old_cells[0],old_cells[1]),len(old_cells[0])).astype(float)
    old_b2 = to_sparse(make_boundary(old_cells[1],old_cells[2]),len(old_cells[1])).astype(float)
    # The previously certified inclusion supplies index maps, with old order intact.
    maps = []
    for n in range(3):
        lookup = {c:i for i,c in enumerate(cells[n])}
        maps.append([lookup[tuple(expanded_nodes.index(base_nodes[v]) for v in c)] for c in old_cells[n]])
    restriction_lookup = {v:k for k,v in enumerate(maps[1])}
    arrays_path = project/r['arrays']['path']
    assert sha(arrays_path) == r['arrays']['sha256']
    residual, inherited_error, harmonic_error = 0., 0., 0.
    results = []
    with np.load(arrays_path) as stored:
        h = stored['harmonic_basis']
        for call in r['calls']:
            phi = np.zeros((len(cells[1]),call['query_rows']*128))
            phi[call['edge_ids']] = stored[f'edge_response_{call["call_id"]}']
            down, up = b1@phi, b2.T@phi
            lap = b1.T@down+b2@up
            energy = float(np.square(down).sum()+np.square(up).sum())
            residual = max(residual, abs(energy-float(np.vdot(phi,lap))))
            harmonic = h@(h.T@phi)
            harmonic_error = max(harmonic_error, float(np.linalg.norm(b1@harmonic)),
                                 float(np.linalg.norm(b2.T@harmonic)))
            out = dict(call_id=call['call_id'], event=call['event'],
                update_clock=call['update_clock'], relation=call['relation'],
                output_port=call['output_port'], transport=call['transport'],
                edge_response_energy=call['total_energy'],
                node_D_readout_energy=float(np.square(down).sum()),
                face_D_readout_energy=float(np.square(up).sum()),
                total_D_readout_energy=energy,
                harmonic_response_energy=call['energies']['harmonic'])
            if 'captured_scope_vertex' not in call:
                old_phi = np.zeros((len(old_cells[1]),phi.shape[1]))
                for edge in call['edge_ids']:
                    old_phi[restriction_lookup[edge]] = phi[edge]
                old_down, old_up = old_b1@old_phi, old_b2.T@old_phi
                inherited_error = max(inherited_error,
                    float(np.max(abs(down[maps[0]]-old_down))),
                    float(np.max(abs(up[maps[2]]-old_up))))
                assert np.all(down[list(set(range(len(cells[0])))-set(maps[0]))] == 0)
                assert np.all(up[list(set(range(len(cells[2])))-set(maps[2]))] == 0)
                out['original_5082_D_channel_readout_preserved'] = True
            results.append(out)
    report.update(source_report=str(source), source_report_sha256=sha(source),
        response_arrays_sha256=sha(arrays_path), actual_scope_proof_sha256=sha(proof_path),
        signal='known grade-1 native shared-residual response contribution; 128 output coordinates per query',
        operator_action='D on the edge contribution gives B1 phi at grade 0 and B2^T phi at grade 2',
        inherited_D_dimension=5082, saved_scope_D_dimension=sum(map(len,cells)),
        energy_identity_max_absolute_error=residual,
        harmonic_D_readout_max_norm=harmonic_error,
        inherited_174_call_readout_max_error=inherited_error,
        calls=results,
        interpretation='The edge-channel contribution to D is identified. Other native grade signals are not imputed by setting them to zero.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--lean-output',type=Path)
    p.add_argument('--check-lean',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    obs=a.task/'work/signal_admission_2401_20261117/pair_observed_observations'
    phase,arrays=phase_measurement(obs)
    d={}
    dirac_measurement(root,d)
    array_path=a.output.with_suffix('.npz')
    np.savez_compressed(array_path,**arrays)
    report=dict(status='numerical_checks_complete_pending_lean',physical_phase=phase,
        relation_dirac=d,arrays=dict(path=str(array_path),sha256=sha(array_path)),
        main_body_cycle=2401,new_model_updates=0,new_world_calls=0,
        next='Measure actual relation-use connection changes and paper R(t), then test identified modes in the preserved normal loop. Keep the physical phase and relation-response carriers distinct.')
    report['sources'] = {name:sha(root/name) for name in (
        'scripts/ib_local/adaptive_hamiltonian.py','scripts/native_topology_observer.py',
        'scripts/measure_native_paper_phase_dirac.py')}
    if a.lean_output:
        scope=root/'lean/CGBAxioms/reviews/NativePendingScopeActual2401.lean'
        formula=root/'lean/CGBAxioms/reviews/NativePaperPhaseDirac.lean'
        text='import Mathlib.Analysis.SpecialFunctions.Complex.Circle\n'+scope.read_text(encoding='utf8')+'\n'
        text+='\n'.join(l for l in formula.read_text(encoding='utf8').splitlines()
                        if not l.startswith('import '))+'\n'
        text+='''
namespace CGB.NativePaperReadoutActual2401
open CGB.NativeDowker CGB.NativePendingScopeActual2401 CGB.NativePaperPhaseDirac

def entry (boundary : Array Boundary) (row column : Nat) : ℝ :=
  ((boundary[column]!).foldl (fun acc term =>
    if term.1 == row then acc + term.2 else acc) (0 : Int) : ℝ)

def actualDown (v : Fin 281) (e : Fin 981) : ℝ :=
  entry (deriveBoundary expanded0 expanded1) v.val e.val

def actualUp (t : Fin 1624) (e : Fin 981) : ℝ :=
  entry (deriveBoundary expanded1 expanded2) e.val t.val

theorem actual_relation_dirac_energy (signal : Fin 981 → ℝ) :
    (∑ v, multiply actualDown signal v ^ 2) + (∑ t, multiply actualUp signal t ^ 2) =
      ∑ e, signal e * (adjoint actualDown (multiply actualDown signal) e +
        adjoint actualUp (multiply actualUp signal) e) :=
  edge_dirac_energy actualDown actualUp signal

def physicalBoundary (v : Fin 256) (e : Fin 896) : ℤ :=
  if v.val / 8 == e.val / 28 then pairBoundary (v.val % 8) (e.val % 28) else 0

theorem actual_physical_phase_lift (phase : Fin 896 → ℝ) (turns : Fin 896 → ℤ)
    (v : Fin 256) :
    Circle.exp (∑ e, (physicalBoundary v e : ℝ)*(phase e + turns e*(2*Real.pi))) =
      Circle.exp (∑ e, (physicalBoundary v e : ℝ)*phase e) :=
  integer_boundary_lift (physicalBoundary v) phase turns

end CGB.NativePaperReadoutActual2401
'''
        a.lean_output.write_text(text,encoding='utf8')
        report['lean']=dict(path=str(a.lean_output),sha256=sha(a.lean_output),status='pending',
                            formula_sha256=sha(formula),scope_sha256=sha(scope))
        if a.check_lean:
            checked=subprocess.run(['lake','env','lean',str(a.lean_output.resolve().relative_to(root/'lean/CGBAxioms'))],
                cwd=root/'lean/CGBAxioms',capture_output=True,text=True,encoding='utf8',errors='replace')
            log=a.output.with_suffix('.lean.log')
            log.write_text(checked.stdout+checked.stderr,encoding='utf8')
            report['lean'].update(exit_code=checked.returncode,log=str(log))
            if checked.returncode:
                a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
                raise RuntimeError(f'Lean check failed: {log}')
            assert sha(a.lean_output)==report['lean']['sha256']
            report['lean']['status']='passed'
            report['status']='native_phase_and_relation_dirac_measurements_checked'
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(samples=len(phase['samples']),phase_operation_changes=phase['operation_changes'],
        dirac_calls=len(d['calls']),energy_identity_error=d['energy_identity_max_absolute_error'],
        inheritance_error=d['inherited_174_call_readout_max_error'],
        examples=[r for r in d['calls'] if r['call_id'] in (11,947,950)]),indent=2))


if __name__=='__main__':
    main()
