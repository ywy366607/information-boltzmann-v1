# Learned periodic read priors

The intervention changes the reader only. Writing, transport, collision,
quadratic bath, event duration T=3, K=16, BPTT32, data and validation remain
those of `q8_w4_quadratic_t3_k16_3000`. This control continues with its original
atlas reader; its existing trajectory is not relabeled as the new model.

## Posterior aperture

Each head/query has a trainable continuous torus coordinate p[h,q]. Let A(x)
be the existing field-conditioned atlas mixture and c(x)=q_hat dot k_hat(x).
The new normalized physical measurement weights are

    w[h,q](x) = softmax_x(g[h] c[h,q](x) + log A[h,q](x)
                         + b[h]/3 sum_j cos(2 pi (x[j]-p[h,q,j]))).

The periodic factor is a spatial prior, updated by the atlas and semantic
evidence. The original categorical action prior/posterior and their KL term
remain intact. Every finite-weight site has positive probability. Continuous
coordinates can move across periodic boundaries without a discontinuous clamp.
The measurement still contains the local invariant/nullspace mean and variance.

For isotropic unit semantic Q/K, Var(q_hat dot k_hat)=1/head_dim, so g starts
at sqrt(head_dim). For the normalized six-component sin/cos position inner
product on a uniform 3-torus, its variance is 1/6, so b starts at sqrt(6).
Both are independent per-head log parameters with freely learned positive
scales. These calibrate initial score variance; they do not prescribe final
entropy, head diversity or a universal optimal temperature.

The 16 initial positions cover balanced 4x2x2 cell centers. Each head's four
query measurements contribute equally at initialization; this keeps feature
amplitude unchanged on a uniform field and gives every query a direct CE
gradient from the first update. Subsequent merge weights are learned.

## Reproduction and compatibility

New belief-reader runs default to `--readout-aperture learned_probes`.
Use `--readout-aperture atlas` for an explicit historical control. Resume
inherits the stored aperture; old configurations without the key mean atlas.
Checkpoint loading and evaluation reconstruct the exact selected variant.
Cross-aperture strict loading fails rather than silently changing a model.

    python scripts/ib/train_q8_port_agents.py \
      --output results/q8_w4_quadratic_t3_learned_read_3000 \
      --readout-aperture learned_probes --steps 3000 --tokens 128 \
      --chunk-tokens 32 --micro-steps 16 --event-duration 3 \
      --shape 8 8 4 --content-dim 16 --dissipation-type quadratic \
      --match-unified-initialization --compile-operators --validate-every 500

The live monitor exposes each Query separately, probe coordinates and per-head
QK scales. It retains a head-average view as an explicit selection. Probe
locations are shown only when supplied by the selected architecture.

## Acceptance

Numerical checks cover positive normalized attention, nonzero finite CE
gradients to every coordinate/query and scale, periodicity, position
sensitivity, checkpoint compatibility and cross-grid loading. Compiled CUDA
Graph forward/backward must agree with eager execution. Capability acceptance
uses at least 3000 joint OWT updates with matched warm-local NLL; sharpness
and transport dependence are diagnostics, not substitutes for language benefit.
