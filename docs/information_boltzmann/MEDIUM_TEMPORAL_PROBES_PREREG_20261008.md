# Local temporal probes: numerical and interface acceptance

Goal: expose local wave histories, signed phase and multiple timescales to the existing 3D reader. No capability ranking, training, physical clock change or waiting policy is proposed by this acceptance run.

Competing explanations: instantaneous observations alias histories; current field/motion may already be task-sufficient; added temporal state may be numerically or causally unusable. This test can resolve expression and interface feasibility, not choose the best language architecture.

Candidate: explicit complex filter state per compact probe, channel and mode, driven by local field and optional analytic local motion. Exact held-input update for dz/dt=-(alpha+i omega)z+alpha*r with alpha>0. Store state/time separately from the physical belief; never advance it merely by reading. Preserve signed real/imaginary features.

Registered acceptance before execution:

- Double-precision held-input formula and matrix-exponential reference agree; same held input with split intervals matches one interval; dt=0 is identity.
- A complex exponentially sampled input with ZOH holding has the known discrete transfer function and phase; compare that exact sampled-input reference, not a continuous sinusoid.
- Histories with equal current measurement can have different bank states. Equal future input contracts their difference by exp(-alpha*t), for fixed finite positive rates.
- Chunk continuation and explicit serialization preserve state and time exactly. Full vs split gradient tapes match without detach. Gradient checks cover alpha, frequency, initial state, input and duration.
- Each immediate probe input has exact compact support. Historical support is the union of earlier supports. No coordinate/rate updates during this acceptance. Geometry gradients express sensitivity at the fixed current trajectory, not moving-history correctness.
- Real OWT interface: freshly seeded existing 8x8x4 D128 model, 64 burn-in tokens and 32 scored next-token targets, actual persistent physical/bank state. Mature checkpoint unavailable; no replacement maturity or NLL-benefit claim. Burn-in is no-grad, score32 has full connected gradients. No optimizer, GPU, PT output, label-driven input, extra physical evolution or stopping search.
- Diagnostic bank uses four dyadic half-lives derived from event interval and three nonzero Nyquist-relative frequencies plus one zero frequency. These are explicit numerical coverage settings, not justified production optima. Rates/frequencies are learnable. Report state cost and bank vs baseline read costs separately from full training.

Decision: failed identities/causality/gradient/locality require repair and block task claims. Passing contracts establishes a viable local historical observation path; sufficient joint learning with a size-matched nonoscillatory control would be the next task-performance test. Stop after this acceptance.

Independent plan review: /root/rtc_contract_review approved with the exact ZOH coefficient, held-input refinement scope, history-difference contraction, geometry/rate update provenance and fresh-weight limitations.
