# Finite-workspace computation for the persistent medium

Status: source and arithmetic assessment, plus CPU numerical identity checks.
The active D768 8x8x8 individual continues unchanged. No GPU speed or language
improvement is claimed by this assessment.

## Goal and evidence

Reduce complete 32-event training time and peak dedicated memory while keeping
finite movable ports, full persistent state, physical elapsed time, linear
transport/nonlinear collision, and all declared temporal credit intact.

The current model has 512 spatial sites, D=768 and no N-by-N mesh attention.
Recent logged warm updates are about 9--10 seconds/32 events. The accepted
two-window calibration measured 14.846 seconds for its warm second update;
learned scheduling and runtime variation prevent treating either as fixed cost.

Static FP32 storage, excluding allocator/compiler/intermediate buffers:

| Object | MiB |
| --- | ---: |
| All 119,206,751 parameters | 454.738 |
| Gradients of 75,289,409 active parameters | 287.206 |
| Two Adam moments for active parameters | 574.413 |
| Field + three fluxes + two receptor fields, one time slice | 9.000 |
| The preceding six fields at 32 time slices | 288.000 |
| Complex temporal probe history | 0.750 |

Adam moments are staged to host during graph execution; the table is a storage
account, not a sum of simultaneously observed GPU allocations. Frozen embedding
147.237 MiB and decoder 147.429 MiB are already included in parameter storage.
The nine-MiB state excludes material coefficients, STP, precision and workspaces.

## Independent problem formulations and hypotheses

1. Spatial interaction complexity: a global N-squared attention mechanism could
   benefit from Transolver's N-to-M learned slicing. This mechanism is absent
   from the current medium; adding global slice attention would add an instant
   communication path rather than remove an existing quadratic interaction.
2. Observation work: wide pointwise projections are computed at locations that
   have exactly zero read weight. Restricting evaluation to finite support can
   preserve the represented function and its first-order derivatives.
3. Learning execution/storage: a full auxiliary replay, checkpoint recomputation,
   vocabulary parameters and Adam staging can dominate despite a small mesh.
   Changing mesh representation alone cannot remove these costs.

Competing candidates: exact support-restricted evaluation; exact elimination of
unused auxiliary outputs; approximate local Slice control; full latent-state
compression; vocabulary/optimizer storage changes. Select the first two as one
finite-observation-workspace implementation. Keep approximate candidates separate.

## Selected exact candidate

For head h let S_h be the union of its four nonzero query supports. Compute only
the output rows W_h needed by that head, at sites in S_h. Preserve the original
Parameters, normalization, query-specific masks, attention and footprint weights.

Current two key projections cost 2 N D^2 = 603,979,776 MAC per reader call.
Restricted evaluation costs 2 sum_h |S_h| D (D/4). At birth, |S_h|=32 and the
total is 37,748,736 MAC: sixteen-fold less arithmetic for this component.
With saved radii, each query has at most 2x3x3=18 supported grid sites. Even
without overlap reuse, the bound is 84,934,656 MAC, approximately 7.11-fold less.
This is not an estimate of whole-update speedup. Full-union projection alone
would give only a four-fold birth reduction.

The existing compact C1 kernel has zero value and derivative outside support.
Indices must be rebuilt as positions change; retain the live differentiable
footprint inside the selected supports. Fixed padded local boxes are a possible
GPU implementation, with periodic indices and invalid entries masked exactly.
The reader also uses log(clamp_min(eps)) and an exact-zero hard attention mask;
the complete reader is not thereby proven globally C1. Reproduce its original
PyTorch support-branch and zero-point gradient conventions, including tiny
positive footprint values, rather than inventing a new masking threshold.

The writer-auxiliary replay discards the final nonlinear read feature of every
event. Its future writer/clock do not use that feature. Omit this unused read,
while retaining advancement, endpoint RHS, temporal bank and outgoing belief.
Do not remove the physical trajectory needed by subsequent auxiliary losses.

CPU check uses the actual PredictivePhysicalReadAgent at D32, an 8-cubed grid,
4 heads/4 queries, and unchanged parameters. Birth, overlap/periodic wrap and
support-boundary placements pass in FP32/FP64. Every case compares the output
and 28 input/parameter gradients. Output error is zero; maximum gradient error
is 1.1921e-7 / 3.3307e-16 respectively. D768 MACs above are static arithmetic,
not a D768 GPU benchmark or a capability test.

## What to borrow from Transolver

Transolver compresses spatial tokens into a small physical-state workspace. Its
original full-mesh projections still cost O(N C^2); Transolver-3 explicitly moves
linear projections through Slice/Deslice by associativity and tiles workspaces.
The current temporal sampler already applies the exact identity P(FQ)=(PF)Q.

Approximate local Slice is a future option for expensive controllers at much
larger spatial resolutions. However, the collision and receptor-opening MLPs
already have hidden width 64: their costs are 251,789,312 and 151,257,088 MAC per
physical substep, respectively. They are not uncompressed D-by-D mesh attention.
Pooling before their nonlinearity generally changes the function. Retaining the
original skew rotations would preserve norm, but would not certify task fidelity.

Global pooling/attention/deslicing usually gives dc_i/dstate_j nonzero for distant
j. For this model, use only declared local patches if such an approximation is
explored; record its added causal radius. A fully resolved independent N-by-D
persistent state retains an O(ND) storage lower bound. Lossless finite-latent
closure needs an invariant subspace or sufficient-statistic proof; retaining a
fine residual preserves information but also preserves its evolution cost.

## Acceptance and decision

Before production integration: test dense/restricted forward, all physical and
temporal states, relevant parameter/input gradients, port movement and boundary
cases; then one complete D768 BPTT32 update and warm complete-update timing under
the existing 3072-MiB dedicated cap. Check optimizer/pending-gradient continuity.
Do not infer end-to-end gains from isolated GEMM timing or the MAC ratio.

If equivalent and faster, adopt this execution optimization with unchanged model
capacity. If gathering/launch overhead consumes the saving, retain the faster
dense kernel and stop this item. Revisit approximate local Slice only after the
remaining complete-update profile justifies changing model expressivity.

Second-priority exact candidate: evaluate observer-only local collision/response
RHS on read-support plus the full incident transport stencil, keeping public full
field_rhs semantics. This needs its own consumer/halo/gradient audit and is not
included in the first patch.

Sources: https://proceedings.mlr.press/v235/wu24r.html ;
https://github.com/thuml/Transolver/blob/main/Physics_Attention.py ;
https://arxiv.org/abs/2602.04940 .

Numerical artifact: results/published/medium_sparse_read_identity_20261009.json.
Independent source review: scratch/medium_transolver_complexity_review_20261009.md.
The source reviewer accepted the minimal numerical/performance acceptance plan;
GPU integration and end-to-end acceptance remain pending.
