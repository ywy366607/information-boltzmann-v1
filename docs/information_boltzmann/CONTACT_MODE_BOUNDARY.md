# Contact-mode boundary for the learned 3D medium

Date: 2026-10-04. Replaces the full-field W4 exchange in the plastic medium,
while retaining the categorical prediction, innovation chart, action prior and
posterior, precision, electrical response and learned material.

## The exact repair

The historical boundary multiplies every spatial location by a channel-wise
`cos(theta)`. A new observation therefore attenuates old modes unrelated to
its incident packet. The corrected boundary couples only one complete
spatial/content packet to one environmental amplitude per observation.

The existing packet delta(x,a) can be full rank as a spatial-by-content matrix.
The learned admittance and precision define the contacted direction:

```
w(x,a) = admittance(a) * sqrt(precision(a)) * delta(x,a)
r2 = <w,w>; h = <w,f>; d = sqrt(1+r2)
e = ||delta / ||delta||_precision||
f_plus = f - w*h/[d*(d+1)] + w*e/d
e_out = (e-h)/d
```

All inner products use cell volume 1/(Nx*Ny*Nz). The incident amplitude e
preserves the physical energy of the historical precision-normalized incident
packet. The event angle is atan(||w||); at uniform channel admittance it agrees
with the historical event-total angle. Channel actions now shape both the
coupling direction and its strength. There are no additional trainable
parameters, persistent tensors, event loops or physical thresholds.

For w != 0, let psi=w/||w|| and a=<psi,f>. These expressions are exactly

```
a_plus = cos(theta)*a + sin(theta)*e
e_out = -sin(theta)*a + cos(theta)*e
f_plus = f - psi*a + psi*a_plus
```

Consequently the field orthogonal to psi is preserved, and
`||f_plus||^2 + e_out^2 = ||f||^2 + e^2`. The rational implementation has no
division by ||w|| and has finite derivatives when w=0. At fixed w/e, the field
Jacobian is identity on the orthogonal complement; this statement concerns
the direct path, not the full state-dependent writer Jacobian.

The contacted direction is an entire current packet, rather than a rank-one
spatial-envelope/content factorization. It can have broad spatial support:
this repair selects a mode and does not introduce a fixed anatomical region
or claim geographically compact injection. Later observations can contact
different modes. The material and activity-dependent pathways remain learned.
Long-run boundedness continues to depend on the existing open-system source
and loss assumptions; conserving a boundary ledger alone is not a global
stability proof.

## Integration and reproducibility

`PlasticMediumPorts3D` now defaults to `write_exchange='contact_mode'` and
marks its architecture accordingly. The shared historical W4 agent defaults
to `global`, preserving existing torus experiments. Weight dictionaries are
unchanged and load with strict=True. Reflection is a scalar environmental
amplitude in contact mode and stays outside readout.

Training configuration records the exchange law. Resume interprets old
configurations without the field as `global` and rejects changing it as exact
continuation. Continuous-stream continuation similarly stores and checks the
law. The checkpoint audit reconstructs historical runs as global unless an
explicit `--write-exchange contact_mode` diagnostic is requested.

An existing null-innovation derivative defect was also fixed: the mathematically
equivalent `vector_norm / sqrt(volume_count)` replaces `sqrt(mean(square))` for
the posterior innovation summary. It defines a finite zero subgradient.

## Mathematical and implementation checks

75 affected CPU tests pass; six targeted training/evaluation tests pass with
CUDA enabled, including both full CUDA Graph tests. New checks establish:

- Energy closure and exact preservation of the untouched complement.
- Null-coupling identity and finite null-innovation gradients.
- Analytical versus finite-difference derivatives.
- Preservation of gradients along untouched modes at frozen boundary action.
- Identical incident energy budget with identical loaded weights.
- Resolution-independent event norm and angle.
- Explicit exchange-law continuation and rejection of silent switching.

## Trained-weight diagnostic, not a new trained result

The old jointly trained update703 weights were loaded without optimization.
Four real OWT contexts use full mature-state continuation, warm256/score128;
128 scored tokens are traced. In addition, paired single boundary events
start from the identical saved state and identical observations.

| Measurement | Historical boundary | Contact-mode diagnostic |
| --- | --- | --- |
| Direct old-field energy retention, paired identical state | 32.36%–33.52% | 99.971%–99.999% |
| Incident energy, paired identical state | 0.08848–0.09238 | Exactly the same |
| Direct old-field energy retention, continued trace mean | 34.13% | 99.60% |
| Spatial power participation rank, four contexts | 4.54–5.30 | 6.24–8.22 |
| Transport removal mean delta NLL | -0.00233 | +0.08297 |
| Full NLL | 7.51761 | 7.99781 |

The preservation repair works without suppressing input. Existing weights are
not adapted to the new state distribution: full NLL is worse. Increased
transport reliance in this diagnostic does not establish improved capability.
Collision remains weak (mean removal delta -0.01068); the previously measured
small collision rate versus event cadence remains a separate next design issue.
The new field retains spatial structure (trace mean86.82%) and differentiated
read heads (mean pairwise TV0.348). All inspected direct CE parameter gradients
remain finite and nonzero.

Artifacts: `results/published/contact_mode_port_703_diagnostic.json` and
`scripts/ib/audit_trained_plastic_medium.py`. The original checkpoint and
update703 report are retained. No long training was started by this repair.

## Matched execution check

At the same 8x8x4/D128, real OWT128-token update, BPTT8, FP32 CUDA Graph,
duration0.005/substeps1, one warmed measured update per arm took3.095s for
global exchange and3.111s for contact exchange (about0.53% difference).
Peak PyTorch allocated memory was405.0 versus406.8MiB. This is a short
execution sample under GPU contention, not a steady-state throughput estimate.
Both production-size graph checks matched eager losses, state and accumulated
gradients. The independent fly training was left running. Reports:
`results/published/contact_port_execution_global.json` and
`results/published/contact_port_execution_new.json`.

```powershell
python scripts/ib/audit_trained_plastic_medium.py --checkpoint results/plastic_conductance_d128_3000/last.pt --output results/published/contact_mode_port_703_diagnostic.json --write-exchange contact_mode --trace-tokens 32
```
