# Local anisotropic material transport

Implementation acceptance, 2026-10-08. No language capability training was run.

## Constitutive law

Enable `--anisotropic-transport` in `scripts/ib/train_medium_active_stream.py`.
It composes with `--temporal-read`, existing compact write/read ports, adaptive
conduction, receptors, STP, BPTT and continuous state restoration.

At each location the material predicts three shear coefficients. Together with
existing positive speeds they form

```
T = [[1, 0, 0], [s0, 1, 0], [s1, s2, 1]]
B = diag(c0, c1, c2) T
A = B B^T
```

Any symmetric positive-definite 3x3 tensor has a lower triangular Cholesky
factor with positive diagonal, so this local factorization spans the SPD family.
The finite material basis still limits spatial complexity. At finite positive
speeds A is positive definite; STP may close a row, giving a semidefinite effective
tensor. Zero shear reproduces the previous diagonal transport, including its
heterogeneous coefficients and edge-color schedule. Shear starts at zero.
Default material width 8 adds 27 parameters and no persistent state tensors.

For frozen coefficients the continuum generator is

```
df/dt = -div(B j)
dj/dt = -B^T grad(f)
H = 1/2 integral (|f|^2 + |j|^2)
```

On the periodic domain the two power terms cancel by integration by parts.
B occurs in mutually adjoint couplings rather than in the storage metric. Thus
coefficient changes do not change H instantaneously. This is a modeling choice;
metabolic work to physically construct material is outside this energy ledger.

For an edge i->j, row b of B couples the field difference to the projection of
the three origin flux stores along b. An exact rotation updates that projection
and the field difference; transverse flux and the pair mean are retained. Each
edge flow preserves L2 energy and the field integral. Six edge colors approximate
the combined generator by first-order splitting. Origin flux components converge
to a collocated vector under spatial refinement. Spatial locality is retained.

This is reciprocal directional propagation. A functional one-way axon and useful
task-specific topology are separate claims requiring evidence. Changing grid
resolution evaluates the same material law; the represented spatial bandwidth
remains controlled by the continuous material basis.

## Integration and continuation

`field_rhs` uses the same cross-flux divergence, so dynamic and temporal probes
measure the upgraded operator. Material and speed gradients remain live through
joint likelihood training. Continuous runtime payloads declare the tensor law;
restoration rejects a mismatched model. Existing checkpoints lacking the flag
retain the diagonal interpretation. New tensor training is an explicit branch,
not an implicit reinterpretation of old optimizer or physical state.

CUDA Graph was verified through the production `CapturedPlasticChunk` interface,
including backward and replay versus eager. The ordinary compiled-medium entry
still dispatches to the same native equations; a production fused performance
comparison has not been measured.

## Acceptance

- 56 affected CPU tests passed; 4 optional CUDA tests skipped in that invocation.
- All 10 tests in the new suite passed with its CUDA check explicitly enabled.
- Standalone Deslice/scatter/gate checks passed.
- Zero-shear recovery, linearity at frozen coefficients, positive eigenvalues,
  mass/energy conservation, finite-difference parameter gradients and the
  analytic probe derivative passed.
- Joint likelihood reaches new shear parameters and the writer; temporal history
  and complete physical state resume consistently.
- Measured FP64 energy and field-integral errors: 1.78e-15.
- A constant oblique material plane-wave check at fixed duration 0.02 gave RMS
  errors 0.002827, 0.0007511, 0.0001924 on grids 4^3, 8^3, 16^3, respectively.
  Solver steps were refined with the grid; this checks numerical convergence,
  not task capability.
- The full repository attempt stopped at three external failures: one missing
  historical checkpoint and two fly-graph host-memory allocation failures;
  53 tests passed and 2 skipped before stopping.

Machine-readable measurements and source hashes:
`results/published/medium_anisotropic_transport_20261008.json`.

Next capability test should use sufficient joint real-stream training with
matched data/optimizer budgets and pre-update paired NLL. No such gain is claimed
by this numerical delivery, and no long training was launched.
