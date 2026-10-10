# Material tensor, memory and structural energy audit

## Actual propagation

The active trainer enables `anisotropic_transport`. In `PlasticMedium3D`,
`B=diag(c)T`, where T is unit lower triangular with three learned shears,
and `A=B B^T`. Positive finite c makes B invertible and A positive definite.
This is a Cholesky parameterization; a separate epsilon is not required for
strict positivity. Conditioning still depends on learned coefficients.

The spatial stencil uses three coordinate directions. Each edge rotation
projects all three flux stores onto a row of B and updates all of them.
Nonzero shear therefore supports oblique principal directions. Mesh-direction
splitting remains a finite-resolution approximation; refinement tests check
convergence. Shear weights start at zero and subsequently learn. Randomizing
the material field alone does not randomize the initial shear.

## Material time

Persistent local state includes E/I receptor opening fractions, STP resource
and utilization, conduction adaptation, field and flux. Their learned kinetics
provide response times independent of the numerical integration subdivision.
Temporal probes additionally retain sensor history. The trainer currently
advances the medium by 0.005 nondimensional time per token event; the runtime
can advance without input. A free-running asynchronous deployment scheduler
is a separate integration task.

## Energy scope

The activity storage is `H=(||field||^2+sum ||flux_a||^2)/(2N)`.
For frozen coefficients the transport generator is skew-adjoint, so
`dH/dt=s^T K(m)s=0`. This identity also holds for time-varying coefficients
when the paired generator retains its skew-adjoint form. Discrete edge flows
are orthogonal; their product preserves H, including with heterogeneous B.

Conductance response uses energy coordinates `z=sqrt(C)V` and `q=sqrt(L)I`.
Parameter updates hold these stored coordinates fixed. Thus the chosen H
has no explicit material derivative; voltage/current interpretations change
when C/L change. This is a defined numerical convention, not an accounting
of metabolic cost, receptor chemical storage, or structural growth energy.
If future updates instead hold physical V/I fixed while changing C/L, they
must account for `0.5 V^2 dC + 0.5 I^2 dL` as structural work.

## Dashboard

Telemetry now exports the actual effective factor B including conduction,
STP, reference speed and optional structural allocation. Tensor view follows
its unoriented principal axes with continuous lines; isotropic axes are hidden.
Energy-flow view follows the signed transport current
`J_a(i)=sum_D f_D(i+e_a) sum_k B_ak(i) q_kD(i)`.
Its discrete divergence exactly matches the local transport energy derivative
under the documented edge-origin flux storage convention. Particle animation
uses normalized display speed; it visualizes energy direction rather than
semantic information or fluid mass velocity. Grid lines remain the numerical
stencil, not anatomical cables.

Port coordinates are learned parameters initialized on balanced grids. Ghost
markers show initial coordinates, current markers show actual coordinates, and
periodic displacement is reported. Token dependence changes port gates, while
coordinates change through learning. Read attention remains separated by head.

## Restart

Old individual saved and stopped at 30,016 fresh tokens. The new individual
uses D768, 8x8x8 sites, spectral-Xavier material initialization, tensor transport
and temporal readout. Three eager BPTT32 AdamW updates passed on this grid;
12x8x8 exhausted memory on update two. Production compilation and its full
optimizer memory are checked separately; these numerical checks establish
implementation contracts, while learning gains require the real OWT run.
