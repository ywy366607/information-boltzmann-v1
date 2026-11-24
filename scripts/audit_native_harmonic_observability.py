"""Build a sufficient readout for the actual certified H1 space.

This is measurement design on the actual recorded complex, not an activity
measurement. In particular, 62 raw edge sensors are not automatically enough
to recover 62 harmonic coefficients of a signal with nonharmonic components.
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

import numpy as np
from scipy.linalg import eigh, qr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_native_dowker_measurement import make_boundary, to_sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    path = args.observations/'operations.jsonl.gz'
    last = None
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'structural_relation_dependencies':
                last = row
    supports = collections.defaultdict(set)
    for rid, unknown, port, generation, _ in last['data']['edges']:
        supports[unknown, port].add((rid, generation))
    cells = [sorted(set(c for group in supports.values()
                        for c in itertools.combinations(sorted(group), n))) for n in (1, 2, 3)]
    b1, b2 = [to_sparse(make_boundary(cells[n-1], cells[n]), len(cells[n-1]))
              for n in (1, 2)]
    l1 = (b1.T@b1+b2@b2.T).toarray().astype(float)
    eigenvalues, vectors = eigh(l1)
    threshold = 1e-10*max(1., abs(eigenvalues).max())
    harmonic = vectors[:, abs(eigenvalues) < threshold]
    rank = harmonic.shape[1]
    if rank != 62:
        raise AssertionError('input is not the previously certified final 62-mode complex')
    # Diagonal of the orthogonal projector is basis-independent.
    leverage = np.square(harmonic).sum(1)
    visible = leverage > 1e-12
    # Pivot selection is enough only AFTER assuming the signal is harmonic.
    _, _, pivots = qr(harmonic.T, pivoting=True, mode='economic')
    selected = pivots[:rank]
    missing = np.setdiff1d(np.arange(len(cells[1])), selected)
    unobserved = int(missing[np.argmax(leverage[missing])])
    # e_unobserved has zero samples on selected edges but nonzero Hodge readout.
    witness = harmonic[unobserved]
    full_readout_gram = harmonic.T@harmonic
    result = dict(source=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        cycle=last['external_cycle'], edge_count=len(cells[1]), harmonic_dimension=rank,
        first_positive_eigenvalue=float(eigenvalues[rank]),
        identities={'relations': [c[0] for c in cells[0]], 'edges': cells[1]},
        boundary_residual_max=float(np.abs(b1@harmonic).max()),
        coboundary_residual_max=float(np.abs(b2.T@harmonic).max()),
        modal_readout_gram_error=float(np.abs(full_readout_gram-np.eye(rank)).max()),
        basis_invariant_edge_leverage=leverage.tolist(),
        numerically_nonzero_harmonic_support_edges=int(visible.sum()),
        support_threshold=1e-12,
        support_meaning='floating-point projector support, not an exact integer sparsity certificate',
        raw_sensor_candidate=dict(edges=selected.tolist(),
            restricted_harmonic_rank=int(np.linalg.matrix_rank(harmonic[selected])),
            adequate_for_arbitrary_edge_signal=False,
            indistinguishable_signal_pair=['zero', f'unit cochain on unobserved edge {unobserved}'],
            witness_edge=unobserved, witness_endpoints=cells[1][unobserved],
            recorded_sample_difference_exact=0,
            harmonic_readout_difference_l2=float(np.linalg.norm(witness))),
        sufficient_readout=dict(formula='H.T @ phi', output_coordinates=rank,
            admits_nonharmonic_signal=True,
            note='Columns of H span ker B1 intersect ker B2.T. This records modal inner products, not 62 chosen raw edges. Independent actual phi is still required.'),
        actual_activity_measured=False, model_updates=0, world_calls=0)
    basis_path = args.output.with_suffix('.npz')
    np.savez_compressed(basis_path, harmonic_basis=harmonic, raw_selected_edges=selected)
    result['basis'] = dict(path=str(basis_path), sha256=hashlib.sha256(basis_path.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('identities', 'basis_invariant_edge_leverage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
