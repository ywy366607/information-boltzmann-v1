# One prediction loop; temporal-relation readout as a separate candidate

Status: design proposal after source audit and independent theory review.
No model implementation or new training is authorized by this document.

## Target and evidence

Improve same-target fresh-stream prequential NLL while retaining one continuous
individual, local write/read apertures and the existing heterogeneous 3D medium.
The completed96k run has +0.081708 pooled fresh-B gain over its locked training
prior. The auxiliary writer gradient is larger than next-token credit in the
writer and input interface; directions are mixed. This motivates simplifying
prediction ownership but does not establish that auxiliary loss caused failure.

Competing explanations remain: incompatible prediction ownership; readout loses
temporal relations; credit truncation versus slow persistent state; and sparse
random-word-interface exposure. A shared head addresses the first explanation;
CTM supplies an independently testable direction for the second. Do not combine
both in an initial experiment and then attribute their aggregate result to one.

## Current internal time, verified in source

The state contains field, three fluxes, receptors, conduction, STP transmission,
precision and an elapsed clock. The model has separate assimilate, advance and
read APIs. Material/electrical/structural rates set local dynamics and memory
times. In the registered training, each token event advances duration .005 with
one solver step; event time is thus tied to corpus arrivals by this scheduler.
The ContinuousStream/LiveStream runtime already supports timestamped writes,
input-free evolution and reads during ongoing state evolution. That runtime's
independent event schedule has not been trained by the current fixed-cadence run.

Physical duration, solver resolution and credit window are distinct:

- duration is model elapsed time;
- substeps approximates that duration numerically;
- BPTT32 controls the trainable tape, retaining all forward physical state.

## Minimal proposed repair: share the categorical prediction

Before observing x_t, issue one causal distribution from the current read port:

    p_t = softmax(D(R(s_t^-)))
    delta_t = E(x_t) - sum_v p_t(v) E(v)
    s_t^+ = Write(s_t^-, delta_t)
    s_(t+1)^- = Advance(s_t^+, elapsed interval)

The prediction of x_t is scored exactly once before it can update parameters;
after evolution the next prediction supplies the next write's categorical prior.
Remove the separate port vocabulary classifier and its observed-token CE in the
new branch. Preserve contact-mode scattering, local geometry, conductance,
transport, collision, physical state and learned read. Keep action KL explicitly
as an action-complexity regularizer, rather than labeling this a full latent ELBO.

The exact structural predictions are one categorical likelihood per observed
target and removal of gradients from the duplicate port classifier. Predictive
quality is an empirical prediction with unknown effect size. A shared forecast
can itself be poorly calibrated; unifying it alone is not a correctness theorem.

### Causality and persistence contract

Cache issued probabilities with target/event timestamp, parameter revision and
scoring status. Translate probabilities into features using the current embedding
table; caching an old expected feature across an embedding update mixes charts.
Preserve probability gradients within the actual BPTT tape; detach at an explicit
tape boundary and retain the cache in checkpoints. Do not rescore an already
accounted forecast when its token becomes the next writer observation.

The existing predicted_feature override still computes the old classifier/loss.
A real production branch must bypass those operations, not merely pass that
override. A cache is a causal forecast ledger, never future-token information.

### Compute contract

This removes a duplicate vocabulary classifier, but shared decode probabilities
are needed before the next write. Today's batched32-feature decoder GEMM may
become sequential decode/softmax/expectation operations. Throughput improvement
must be measured under matched32-token CUDA capture. Do not promise speedup from
parameter removal alone. A50257-element FP32 probability cache is about0.192MiB
per individual; production memory includes its tape and optimizer as well.

## CTM relevance, primary sources

Paper: https://arxiv.org/html/2505.05522v4
Official implementation: https://github.com/SakanaAI/continuous-thought-machines
Source files reviewed: models/ctm.py, models/ctm_rl.py, utils/losses.py.

CTM combines an internal discrete tick sequence, neuron-specific processing of
activation histories, and a read/action representation formed from temporal
activity inner products. Its appendixH gives recursive selected-pair statistics.
The standard classification forward initializes learned traces per input and
runs configured iterations. The RL variant carries history across environment
steps and initializes episodes. These are concrete continuous-processing
examples, not evidence of indefinite never-reset online learning or unlimited
credit. Their optimization still uses differentiable unrolled models.

Our existing material dynamics already supplies a different neuron/medium
history mechanism. Importing the entire CTM core would replace that mechanism.
The distinct borrowable idea is to expose temporal relations at the finite read
surface, while retaining spatial locality and current dynamics.

## Separate later candidate: selected local temporal relations

Use bounded local activity features u collected at the existing read apertures.
For a limited set of channel/port pairs, maintain a recursive second moment:

    rho_ij = exp(-Delta_tau / tau_ij), tau_ij > 0 learned
    C_ij_next = rho_ij C_ij + (1-rho_ij) u_i u_j
    output = D(R_instant(s), C)

This is an adaptation inspired by CTM, not its exact alpha/sqrt(beta) formula.
For |u_i|<=1 and |C_ij_initial|<=1, the convex update proves |C_ij|<=1 for every
positive duration, supporting bounded never-reset statistics. Storage/work are
O(P) for P selected pairs and independent of lifetime. This is forward-state
efficiency; it does not give O(1) exact lifelong parameter credit.

It can distinguish histories whose present field is equal but whose coactivity
differs. If the present field already carries all task-relevant history or the
traces only track background means, gains may vanish. Pair products encode
coactivity, not an automatic proof of phase locking or biological oscillations.

## Proposed validation and decision, before launch

First shared-prediction branch: causal unique scoring, normalization-chart
consistency, full-state/cache continuation, within-tape gradient equivalence and
captured speed/memory calibration. Use the existing complete last checkpoint as
a continuous starting individual for matched branches; record changed objective
and optimizer policy rather than calling it exact old-run continuation.

For a formal task comparison, both arms use identical actual corpus exposure,
initialization, optimizer cadence and live A->B->A policy. At least3000 joint
updates, record curves and budget/convergence separately. Report same-target
locked-prior gain and current bias-only decomposition in the same forward.
Stop/reject on causal leakage, duplicate scores, cache/reset faults, nonfinite
behavior or GPU limit; task losses with valid execution reject benefit at that
budget. Equal outcomes preserve the smaller graph for engineering reasons only.

CTM-style readout is a later separate candidate, assessed with identical pair
budget and a matched instantaneous readout. Neither candidate is automatically
approved for expensive training by a literature analogy.

Independent theory reviewer /root/rtc_contract_review accepts shared prediction
as feasible, conditional on the forecast ledger, real classifier removal and
matched compute calibration. CTM-inspired temporal readout stays a hypothesis.
