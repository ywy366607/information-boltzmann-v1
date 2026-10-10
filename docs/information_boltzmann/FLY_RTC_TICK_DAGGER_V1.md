# Fly RTC tick-response student v1

Implementation candidate, 2026-10-07. User authorized an independent version;
this document does not declare language improvement or biological equivalence.

## Problem and one closed loop

The real connectome continues one physical input tick per observed token.
A cheaper regional student anticipates delayed motor responses. Timestamped
drafts can cover a response deadline while physical verification proceeds.
When a matching response arrives it is scored against the originally issued
draft before correction. New observations replace only the uncommitted suffix.

RTC supplies scheduling, tick rollout supplies anticipation, and DAgger-inspired
queries supply recovery supervision on student-selected deviations. No Gamma,
extra brain reset, broadcast injection or extra vocabulary decoder is added.

The implementation is independent of HX-0/HX-1 and the one-step streaming arm:

- `core/fly_rtc_student.py`: fixed full-local-state sampled codec, actual initial
  E/I-by-delay regional graph, shared tick transition, motor-only horizon read.
- `core/fly_rtc_learning.py`: joint CE, input-conditioned arrival supervision,
  endpoint quiet-response supervision, full physical query replay.
- `runtime/fly_rtc.py`: timestamp/revision handoff and asynchronous physical
  response verification; preserves committed outputs.
- `runtime/fly_rtc_flow.py`: continuous sensory inbox with one physical worker,
  speculative input responses and arrival-driven suffix correction.
- `scripts/ib/train_fly_rtc_dagger.py`: independent real OWT active-learning
  entry point, full continuation, data/source hashes, best/last checkpointing.

## Mathematical and information contract

For a complete physical state S and explicit driving input I:

    S_next = F_theta(S, I)
    z = E(S)
    z_hat_next = G_phi(latent_delay_history, E_drive(I))

E samples h, g_E, g_I, ALIF adaptation, STP x/u and all four pulse slots.
Its seeded projection is fixed; a learnable encoder cannot lower target loss
by collapsing the projection. This is still a lossy approximation, not an
exact Markov quotient. Initial connectome aggregates remain fixed during this
candidate; plastic physical synapses keep learning and teacher labels use
their current weights. This approximation is explicitly recorded in config.

Every real input transition has its own observed sensory drive. The teacher
response is the actual following physical state; no future token is supplied
to the predictor of that transition. Separately, the 14-horizon forecast is
conditional on **zero future external drive**. It predicts the response to
currently in-flight pulses, not the actual unknown future text trajectory.

The CE read feature is:

    r_t = native_motor(S_t) + sum_k attention_k * adapter(z_hat_motor[k] - z_motor[0])

The same adapter supplies per-tick future motor-response features for the
execution cache, and is supervised against real motor-feature changes on a
quiet physical copy. CE's adaptive horizon mixture is a present decision
feature; k is a physical forecast offset, not k future language tokens.

## Learning and query aggregation

The loss is next-token CE plus an explicit auxiliary coefficient times:

1. Actual input-conditioned one-tick prediction MSE.
2. Matched endpoint zero-drive multi-tick physical-state MSE.
3. Matched future motor-response feature MSE.
4. One re-encoded physical-query transition MSE.

All task-bearing physical parameters, sensory projections, edges, readout,
decoder and student are jointly trained using the existing BPTT window. The
input embedding retains the inherited pretrained/frozen policy. BPTT32 limits
gradient history while full physical state persists across windows. This is
not a claim of unlimited credit assignment.

At each training-window endpoint a detached physical copy runs H quiet ticks.
The student's final discrepancy chooses an adjoint membrane perturbation.
The perturbation is external work, not an inverse encoder or closed-system
energy-conserving operation. The actual perturbed source is then re-encoded.

    S_query = perturb(copy(S_reference), student_error)
    history_query[0] = E(S_query)
    label_query = E(F_current_physical(S_query, zero_drive))
    train G(history_query, zero_drive) against label_query

The preceding actual latent history is saved with the full physical source.
The bounded CPU replay queue holds complete source states and controls, not
stale labels. One entry is relabeled with current teacher weights per window.
This is DAgger-inspired physical query aggregation; the reduced state has not
been proved sufficient for strict DAgger guarantees.

H=14, latent_dim=128, sample_per_region=64, replay_capacity=2 and query_radius=0.1
are exposed computational/query budgets, not biological constants. Radius
scales actual regional membrane RMS. Each 32-token window advances the real
individual by 32 ticks, the quiet teacher by H+1 separate query ticks and the
student by 32H forecast ticks. These counts are logged separately.

## Runtime and execution

`predict_rtc_next` is the label-free streaming path shared with training.
`FlyRTCExecutor` attaches to a complete physical snapshot and its observed
latent history, stages future motor-response features and launches background
quiet physical verification. Outputs can be consumed while that worker runs.

    model.eval()
    executor = FlyRTCExecutor(model, origin_tick=state_tick)
    audit = executor.stage(physical, student_history, state_tick,
                           input_cutoff=latest_observation_tick)
    logits, issued_key = executor.output(state_tick + 1)

The caller supplies new physical snapshots after each real input. New input
epochs invalidate speculative suffixes. The version key contains origin tick,
absolute target tick, input cutoff, plan revision and model revision. Old
results cannot overwrite fresh plans or committed outputs. Matching quiet
teacher results may score the original draft even after that output was used.
Actual future-input branches must never be counted as matching quiet teachers.

`FlyRTCFlow` accepts new input while real physical work is pending:

    model.eval()
    flow = FlyRTCFlow(model, physical, student_state=continuing_student)
    tick = flow.submit(token)       # fast prediction plus queued real work
    logits, key = flow.output(tick) # arrived response or available draft
    saved_runtime = flow.state_dict()

The writer chooses each source once using the latest **arrived** physical
membrane and continuing writer baseline. Both teacher and student consume that
exact selected source. The real physical worker progresses in order. Arrival
reanchors the student, replays outstanding known drives and regenerates the
uncommitted suffix. A new input keeps valid earlier outputs at or before its
origin, including unconsumed slots, while replacing subsequent predictions.

The H-tick bounded inbox provides backpressure if physics falls too far behind.
During quiet intervals `submit(None)` represents a real elapsed zero-drive
tick. Parameter updates happen after elapsed outputs are committed, and wait
for queued real work; runtime checkpoints retain the complete continuing life.

Synchronous training conditions the writer on actual old physical state; under
runtime lag, it uses the latest arrived physical state. This causal deployment
difference requires task validation under lag. CE's horizon-weighted feature
and runtime's per-horizon decoded feature also require separate task evaluation,
even though they share the supervised motor-response adapter. CPU interface
tests establish neither one's real-time language quality.

Inference jobs and optimizer writes must be serialized through
`update_parameters`. Copying a physical state alone does not freeze the
transition weights. Runtime checkpointing quiesces the teacher worker and saves
pending drafts, issued originals, revisions and executed tick; caller also
saves the physical/student state and model. Trainer checkpoints already save
full learning life including replay, optimizer states, ticks, RNG and cursors.

Trainer physics/query work is sequential. The execution API is asynchronous,
but shared-GPU kernel overlap and throughput gains have not been demonstrated.
This is not the exact flow-policy inpainting algorithm or distribution-exact
LLM speculative decoding. It is a predictive response/execution implementation.

## Competing explanations and prospective acceptance

The target remains improved first-pass and fresh-context NLL under never-reset
active learning, with dedicated memory below 3900 MiB. Three explanations are
kept distinct: delayed response anticipation is useful; matching the physical
trajectory is useful only for its own MSE; the apparent benefit comes from
extra read capacity/optimization rather than query aggregation.

Numerical tests establish causality, same-source labels, gradients and complete
continuation. One real 32-token calibration establishes executable gradients,
resource demand and measured eager speed. It cannot establish NLL improvement.
Capability evaluation needs at least 3000 joint optimizer windows with matched
real OWT targets and live evaluation, retaining the strongest baseline. A
budget-limited result reports uncertainty rather than claiming convergence.
Improved auxiliary MSE without NLL benefit triggers removing the auxiliary
from the task route or rejecting this candidate; it does not trigger another
unregistered corrective module.

References: Ross et al. (2011), https://proceedings.mlr.press/v15/ross11a;
Black et al. RTC, https://arxiv.org/abs/2506.07339;
Leviathan et al. (2023), https://proceedings.mlr.press/v202/leviathan23a.

## Authorized bounded CPU response check (2026-10-07)

The user explicitly requested a small task/short CPU test. The independent
reviewer approved numerical waiting coverage and a fixed-oracle response-fit
assay, with complete stimulus lanes held out and a persistence baseline.
The entry point is `scripts/ib/check_fly_rtc_cpu.py`; it allocates no CUDA
model and writes no checkpoints. This is separate from joint OWT capability
training and has no Sudoku or vocabulary rule prior.

An Event-blocked physical worker admitted three consecutive inputs, emitted
finite drafts before teacher completion, retained committed outputs, and
refreshed the suffix after arrival. The focused regression suite passed
25 tests. This establishes logical waiting coverage, not GPU overlap/speedup.

An anatomical selection, made before seeing responses, retained 256 real
MaleCNS neurons and 5,901 edges (3,303 E / 2,598 I), keeping delays1–4 and
production COBA/ALIF/STP physics. The 128,055 dropped boundary edges make
this an induced boundary-value system, not a model of the full brain.
Four independent stimulus lanes train the student; four entire other lanes
with different currents and timings are held out. Each lane carries its state
continuously from its own birth. Quiet future labels refer to the declared
zero-drive scenario and include 14 physical ticks.

At 120 AdamW student updates, seed11, the latest bounded run took9.21seconds
after Python imports. Held-out latent forecast MSE/persistence was0.8815,
but motor-response-delta MSE/persistence was3.3620. The raw motor response
was nonzero; tiny decoded deltas (~3e-8 baseline MSE) and an unsettled train
loss prohibit a capacity or convergence verdict. Instrumentation-only reruns
repeated the same seed/budget/result; they are not independent experiments.

A no-training least-squares read of *true* future codec differences also had
held-out error/persistence7.7410. An ambient row-space calculation found
81.87% of the actual linear motor observable's squared coefficient norm
outside the codec's observable row space. Exact reconstruction for arbitrary
full states is therefore not guaranteed. This does not prove impossibility
on the nonlinear reachable manifold or identify a unique language bottleneck.

The next bounded candidate is to preserve the actual motor observable in the
student state, then repeat this same response assay. Waiting coverage is
already established; long language training should wait for useful drafts.
The short check does not evaluate DAgger's performance contribution.

Results: `results/published/fly_rtc_cpu_response_20261007.json`.
Graph extraction now streams edge arrays in bounded blocks; its256-node graph
matches the earlier full-array extraction hash exactly and occupies151,046bytes.
An intermediate rerun hit Windows commit exhaustion, before any teacher fit;
the completed result and separate row-space calculation remain recorded.
