"""Actual relation/shared-unknown Dowker complex, Lean certificate and Hodge.

Faces come from common typed unknown dependencies, not phase labels, guessed
triangles or activity thresholds. This is a measurement of that actual
dependency structure. It neither adds cells to the runtime nor claims native
oscillator phases, emergent memory usefulness or a full-body topological model.
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]


def rank_integer(columns):
    """Exact rank over Q by sparse fraction-free column elimination.

    Each step scales by a nonzero integer and subtracts an existing pivot
    column; gcd normalization only removes a nonzero scalar. No tolerance or
    modular-field assumption is involved. The Lean certificate checks the
    complex, independently of this Python exact-rank implementation.
    """
    basis = {}
    for source in columns:
        column = dict(source)
        while column:
            pivot = max(column)
            if pivot not in basis:
                basis[pivot] = column
                break
            old = basis[pivot]
            a, b = old[pivot], column[pivot]
            if a != 1:
                column = {i: a*v for i, v in column.items()}
            for i, value in old.items():
                new = column.get(i, 0)-b*value
                if new:
                    column[i] = new
                else:
                    column.pop(i, None)
            if column:
                divisor = math.gcd(*column.values())
                if divisor > 1:
                    column = {i: v//divisor for i, v in column.items()}
    return len(basis)


def make_boundary(low, high):
    lookup = {cell: i for i, cell in enumerate(low)}
    return [[(lookup[cell[:j]+cell[j+1:]], (-1)**j) for j in range(len(cell))]
            for cell in high]


def to_sparse(columns, rows):
    rr, cc, vv = [], [], []
    for j, column in enumerate(columns):
        for i, value in column:
            rr.append(i); cc.append(j); vv.append(value)
    return sparse.csc_matrix((np.array(vv, dtype=np.int64), (rr, cc)), shape=(rows, len(columns)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', type=Path)
    parser.add_argument('lean_output', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    source = args.observations/'operations.jsonl.gz'
    last = None
    with gzip.open(source, 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'structural_relation_dependencies':
                last = row
    raw = last['data']['edges']
    relations = sorted({(r, g) for r, u, p, g, v in raw})
    unknowns = sorted({(u, p) for r, u, p, g, v in raw})
    ri, ui = {x: i for i, x in enumerate(relations)}, {x: i for i, x in enumerate(unknowns)}
    supports = [set() for _ in unknowns]
    dependencies = []
    for r, u, p, g, v in raw:
        i, j = ri[r, g], ui[u, p]
        supports[j].add(i)
        dependencies.append((i, j))
    supports = [sorted(s) for s in supports]
    cells, witnesses = [], []
    for size in range(1, 5):
        witness = {}
        for u, support in enumerate(supports):
            for cell in itertools.combinations(support, size):
                witness.setdefault(cell, u)
        cells.append(sorted(witness))
        witnesses.append(witness)
    boundaries = [make_boundary(cells[n-1], cells[n]) for n in range(1, 4)]
    matrices = [to_sparse(boundaries[n], len(cells[n])) for n in range(3)]
    for left, right in zip(matrices, matrices[1:]):
        product = left@right
        product.eliminate_zeros()
        if product.nnz:
            raise AssertionError('actual derived complex has nonzero consecutive boundary product')
    ranks = [rank_integer(b) for b in boundaries]
    betti = [len(cells[0])-ranks[0], len(cells[1])-ranks[0]-ranks[1],
             len(cells[2])-ranks[1]-ranks[2]]
    # All lower-order Hodge terms are included: B3 is essential for L2.
    laplacians = [matrices[0]@matrices[0].T,
                  matrices[0].T@matrices[0]+matrices[1]@matrices[1].T,
                  matrices[1].T@matrices[1]+matrices[2]@matrices[2].T]
    spectra = []
    for order, laplacian in enumerate(laplacians):
        eigenvalues = np.linalg.eigvalsh(laplacian.toarray().astype(np.float64))
        tolerance = 1e-10*max(1., float(abs(eigenvalues).max()))
        zeros = int((abs(eigenvalues) < tolerance).sum())
        if zeros != betti[order]:
            raise AssertionError('real Hodge nullity disagrees with exact rational boundary ranks')
        spectra.append(dict(order=order, dimension=len(eigenvalues), nullity=zeros,
                            smallest_positive=float(eigenvalues[zeros]) if zeros < len(eigenvalues) else None,
                            largest=float(eigenvalues[-1]), eigenvalues=eigenvalues.tolist()))
    # Matrix-free sparse Dirac check includes C3, making its first three square
    # blocks precisely the Hodge operators above. Top block is a 3-skeleton.
    dims = list(map(len, cells))
    blocks = [[None]*4 for _ in range(4)]
    for i, b in enumerate(matrices):
        blocks[i][i+1], blocks[i+1][i] = b, b.T
    for i, n in enumerate(dims):
        blocks[i][i] = sparse.csc_matrix((n, n), dtype=np.int64)
    dirac = sparse.bmat(blocks, format='csc')
    expected = sparse.block_diag(laplacians+[matrices[2].T@matrices[2]], format='csc')
    difference = dirac@dirac-expected
    difference.eliminate_zeros()
    if difference.nnz:
        raise AssertionError('actual sparse Dirac square has spurious cross-order blocks')
    reference = ROOT/'lean/CGBAxioms/reviews/NativeDowker.lean'
    text = reference.read_text(encoding='utf8') + '''

namespace CGB.NativeDowker.Actual2401
set_option linter.style.longLine false
set_option linter.style.setOption false
set_option maxRecDepth 20000
set_option maxHeartbeats 4000000
'''
    text += f'def sourceSHA : String := "{hashlib.sha256(source.read_bytes()).hexdigest()}"\n'
    text += 'def dependencies : Array (Nat × Nat) := #[' + ','.join(f'({r},{u})' for r,u in dependencies) + ']\n'
    text += 'def supports : Array (List Nat) := #[' + ','.join(str(s) for s in supports) + ']\n'
    for n in range(4):
        encoded = '|'.join(f'{witnesses[n][cell]}:'+','.join(map(str,cell)) for cell in cells[n])
        text += f'def cells{n} : Array Cell := decodeCells {json.dumps(encoded)}\n'
    for n in range(1, 4):
        text += f'def boundary{n} : Array Boundary := deriveBoundary cells{n-1} cells{n}\n'
    text += '''def supportFaithful : Bool :=
  dependencies.all (fun (r,u) => u < supports.size && supports[u]!.contains r) &&
  (List.range supports.size).all (fun u =>
    supports[u]!.all (fun r => dependencies.contains (r,u)))
theorem actual_support_faithful : supportFaithful = true := by native_decide
'''
    for n in range(4):
        text += f'theorem actual_cells{n}_count : cells{n}.size = {dims[n]} := by native_decide\n'
        text += f'theorem actual_cells{n}_unique : uniqueCells cells{n} = true := by native_decide\n'
        text += f'theorem actual_cells{n}_supported : cells{n}.all (validCell supports {dims[0]} {n+1}) = true := by native_decide\n'
        text += f'theorem actual_cells{n}_complete : completeCells supports {n+1} cells{n} = true := by native_decide\n'
    for n in range(1, 4):
        text += f'theorem actual_boundary{n}_correct : boundaryCorrect cells{n-1} cells{n} boundary{n} = true := by native_decide\n'
    for n in range(1, 3):
        text += f'theorem actual_boundary{n}_{n+1}_zero : boundarySquareZero boundary{n} boundary{n+1} = true := by native_decide\n'
    text += 'end CGB.NativeDowker.Actual2401\n'
    args.lean_output.write_text(text, encoding='utf8')
    result = dict(status='numeric_checks_passed_lean_not_yet_checked',
        source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        observed_cycle=last['external_cycle'], relation_identities=relations, typed_unknowns=unknowns,
        dependency_count=len(raw), simplex_counts=dims, boundary_ranks_over_Q=ranks,
        betti_over_Q_orders_0_1_2=betti, hodge=spectra,
        dirac=dict(dimensions=dims, total_dimension=sum(dims), square_exact=True,
                   meaning='associated dependency 3-skeleton, not claimed native Dirac dynamics'),
        lean_output=str(args.lean_output), reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-start, model_updates=0, world_calls=0)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in
                     ('relation_identities', 'typed_unknowns', 'hodge')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
