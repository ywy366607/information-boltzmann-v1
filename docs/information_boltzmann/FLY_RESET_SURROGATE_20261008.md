# Reset-surrogate gain: bounded numerical repair candidate

## Goal and evidence

Maintain the continuous individual and sensory-write/motor-read surfaces,
S=14 and W=32. The previous real 480-tick window had finite gradient entries
but FP64 total norm 2.4471433607872168e20. Global clipping reduced current
read-projection credit to approximately 2.52e-21. Forward parameters/state
were finite. Numerical stability, task learning and lifetime-memory cost
are distinct requirements.

## Competing explanations

1. Reset surrogate adds a non-physical local gain even without a spike.
2. Delayed synaptic feedback and augmented ALIF/STP state Jacobians amplify
   credit independently of the reset term.
3. Shared global clipping transmits a dynamics problem into the read group;
   this is downstream optimizer coupling rather than the initiating gain.

## Analytic closure

At fixed incoming conductance/current and threshold, write v=alpha*h+I,
s=H(v-theta), h_next=v*(1-s), psi=1/(1+(pi*(v-theta))^2).
The existing surrogate direct membrane derivative is
alpha*((1-s)-v*psi). Negative subthreshold voltage can make it exceed one.
An illustrative local case alpha=.987, v=-.3, theta=.1 gives 1.101805857.
These illustrative values are not fitted model constants or a measured
480-tick real trajectory.

Using the same spike with only its RESET use detached changes that direct
derivative to alpha*(1-s). The passive local branch is bounded by one when
0<=alpha<=1. Spike transmission, ALIF and STP still use the attached spike.
Forward numerical equations are identical. This explicitly changes the
surrogate derivative, including its reset-mediated threshold term; it is
not an exact hybrid event-time/saltation derivative or a contraction proof
for the entire delayed recurrent Jacobian.

## Pre-registered predictions and decision

Same complete saved individual, weights, physical state, optimizer moments,
previous token and actual 32 OWT targets. Compare baseline and detach-reset
sequentially. No new weights saved; suppress both optimizer steps.

- Forward scores, training objective and complete terminal physical-state
  hashes must match exactly. Any mismatch invalidates the comparison.
- The analytic local gain term must disappear in deterministic derivative
  tests. Axonal-ring, ALIF and STP gradients must remain available.
- If the real whole-window norm falls and current clipped read gradients
  recover, this supports reset-term amplification as a contributing cause.
- If the whole-window norm stays extreme, prioritize feedback Jacobians;
  do not shorten W or claim the reset convention fixes overall stability.
- No NLL benefit can be inferred from identical pre-update predictions.

CPU interface fixtures check both COBA and CUBA, checkpoint equivalence and
attached delayed-state routes. They are numerical checks, not capability
studies. Resource ceiling: 3900 MiB board snapshots and PyTorch counters.

## Literature and biological scope

SpikingJelly provides detach_reset as an explicit surrogate-gradient option:
https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.10/spikingjelly.clock_driven.neuron.html
Surrogate gradients are a learning convention:
https://arxiv.org/abs/1901.09948
Actual fly memory uses interacting short/long-term memory units and
dopamine-regulated plasticity:
https://www.nature.com/articles/s41586-024-07819-w
These observations support distinguishing local biological plasticity
from storage and multiplication of an unrolled reverse-time graph. They
do not establish that this candidate replicates biological learning or
provides unlimited exact credit in fixed memory.

## Review

Independent reviewer /root/rtc_contract_review accepted the analytic
candidate and bounded paired diagnostic. Keep default False for old
individuals; serialize the explicit option for continuation. Results and
decision will be appended after the registered numerical gate.

## Numerical gate results

91 affected numerical/interface tests pass, including complete W32
checkpoint equality under both conventions and retained ring/ALIF/STP spike
gradients. The continuing CLI saves/inherits the explicit flag; older
individuals default to False. No production default was switched.

Three real backward-only runs restored the same full source and scored the
same targets. Both optimizers were suppressed; every parameter remained
unchanged, 480 ticks ran and no PT was created.

| Quantity | Baseline | Reset detached |
| --- | ---: | ---: |
| FP64 gradient norm | 2.4471440e20 | 1.2523405e14 |
| Current clipped read-projection norm | 2.51994e-21 | 4.92411e-15 |
| Allocated peak MiB | 2158.67 | 2151.74 |
| Board occupation snapshots MiB | 2717 | 2711 |

The observed norm ratio is 1.954056e6; the residual remains extreme and
shared clipping still suppresses current read credit. This is exploratory
numerical evidence, not a complete stability repair or task improvement.

The preregistered GPU bitwise gate FAILED. Pair max score difference was
9.54e-7; terminal floating-channel hashes differed. An unchanged-baseline
repeat also failed bitwise equality, max score difference 5.25e-6 and norm
relative difference 2.67e-7. The forward Triton kernel documents float32
atomic reductions with nondeterministic summation order. CPU exact contracts
and the unchanged equation establish the local forward equivalence, but
the full GPU paired result does not pass the originally registered exact
criterion. It must not be silently upgraded to that criterion.

Next audit: checkpoint recomputation must follow the original hard-spike
trajectory, and the delayed augmented surrogate Jacobian must be separated
from this reset branch. Preserve the whole S14/W32 window. Do not start a
long training run or declare biological constant-memory learning solved.

Reports: results/published/fly_w32_reset_comparison_20261008.json and the
three arm reports it references. These tests measure credit execution;
their unchanged pre-update NLL cannot demonstrate a training benefit.

## Same-backward checkpoint gate (registered before execution)

Restore the same complete source and32 fresh targets. Use reset-detached
candidate and full S14/W32. Compare each of480 original SpikeFn inputs with
its corresponding input in the same backward recomputation, using unique
token context IDs and physical-tick offsets. Save voltage margins on CPU;
leave all outputs and derivatives unchanged. Record before the custom
forward save, since non-reentrant checkpoint can stop early.

Decision: actual sign flips first redirect work to faithful recomputation;
zero flips with full coverage permits augmented-feedback Jacobian analysis.
Partial coverage remains explicitly partial. Hook overhead is diagnostic
cost, not production speed. Source weights/optimizer moments remain unchanged,
no PT written, GPU ceiling3900MiB. Four deterministic instrumentation tests
pass, including injected flip detection and unchanged gradients. Independent
reviewer /root/rtc_contract_review approved this gate with early-stop and
hook-restoration requirements.

The same-backward gate completed:480 original and480 recomputed spike calls,
79,258,560 decisions, zero sign flips. Max voltage-margin difference9.54e-7;
smallest original margin7.45e-9. All values finite. Source parameters stayed
unchanged, norm remained1.2523404e14 and board snapshots peaked2711MiB.
First instrumented attempt hit a CUDA error; mapped source loading and a
fresh blocking diagnostic retry succeeded. No failed-attempt mechanism
claim is retained. This one window supports inspecting augmented feedback;
it does not certify all future checkpoint trajectories.

Next registered observer: retain full480-tick backward and record original
physical-state output adjoints h/ge/gi/b/x/u/spike/pulse/v_pre. Summarize
norm, nonzero support and power on currently silent cells, plus passive
alpha and surrogate derivative means. These are observed credit signals,
not singular-value estimates or a causal ablation. Five observer contracts
pass, including unchanged full objective and parameter gradients. A dominant
silent-spike feedback signal would motivate one explicit backward-path
contrast, not an immediate production mask or a biological conclusion.

## Observed full-window adjoints and registered pulse-path contrast

The unchanged observer covered480original ticks, norm1.252340468e14.
Physical spiking is about2.6–4.8%; mean ATan derivative on silent cells
is.87–.88. At tick0,99.527%of spike-adjoint power and98.246%of transmitted
pulse-adjoint power is on currently silent cells. Membrane adjoint norm
is.0746at tick479 and6.214e13at tick0. Channel units differ; adjoint norms
are not comparable singular values or a causal attribution. Board snapshots
2711MiB, unchanged source parameters and zero updates.

Competing explanations: dense silent surrogate recurrent feedback; actual
firing-path/slow-state feedback; mixed active/silent loops. Recompute sign
flips were ruled out for this same source window, not all possible windows.

Next gate, registered before GPU execution: keep the same immutable source,
32real OWT targets, full480physical ticks, detached-reset candidate. Hook
only original transmitted pulses during backward: active_only passes
gradient where the original transmitted pulse is nonzero; silent_only
passes its complement. Physical forward and target/loss unchanged; no
optimizer step or PT output. Report spike/pulse mask disagreement and hook
coverage (last unused pulses may have no adjoint); inherited four ring
slots stay as saved. Direct read/decoder gradients should remain equal up
to baseline GPU reduction variation. CPU exact forward/gradient contracts
precede the two GPU backwards. These interventions are not production flags.

Predictions/decisions: active_only strongly reducing amplification while
silent_only retains it supports silent-route amplification and motivates
a surrogate geometry/event calibration. Both strongly reducing it supports
mixed feedback loops; inspect their coupling instead of blaming only silent
neurons. Neither reducing it redirects analysis to membrane/ALIF/STP
non-transmission paths. Inconsistent forward or coverage blocks conclusions.
The masks remove mixed recurrent products in both arms; gradients are not
an additive decomposition. Silent neurons still need credit to learn first
firing, so this diagnostic mask cannot itself be promoted as the repair.

Independent /root/rtc_contract_review approved the gate with these scope,
support-definition, nonadditivity and source-state requirements.

## Pulse-path decision results

All source parameters were restored exactly; no optimizer updates, no PT.
Both physical forwards cover480ticks. Pulse hooks fired479times, with the
sole unused final pulse outside any subsequent window loss. Spike/pulse
support disagreement is zero. Nine instrumentation contracts pass.

| Diagnostic derivative | Total FP64 norm | Current read norm after global clip |
| --- | ---: | ---: |
| All pulse routes, reset detached | 1.252340468e14 | 4.924e-15 |
| Actual nonzero pulses only | 2.142329148 | .287849 |
| Zero-pulse routes only | 2.791340251e13 | 2.209e-14 |

Direct raw read norm is.61666665in all three arms; raw decoder norm is
.68914993. Relative raw read/decoder differences are below2e-8. Scores
differ by at most9.54e-7, within prior unchanged-baseline variation5.25e-6;
true training objective remains6.483838within5e-7. GPU bitwise equality
is not claimed. Board stage snapshots2711MiB in both interventions.

Decision: this window's extreme amplification requires the silent-pulse
surrogate feedback pathway. Removing it reduces the norm by about5.85e13.
This localizes the execution problem; it does not establish that silent
credit should be zero or that the network's physical forward is unstable.
The unchanged raw read credit confirms shared clipping is a downstream
effect. Independent reviewer accepted this scope and support/coverage checks.

## Mathematical closure for the next single repair direction

Let z=(h,ge,gi,b,x,u,p0,p1,p2,p3), margin y(z)=v_pre-theta_eff,
spike s=H(y). Hold current external input and parameters fixed. In the
detached-reset convention define:

    A = partial F(z,s)/partial z holding s fixed
    B = partial F(z,s)/partial s with the reset use detached
    C = partial y(z)/partial z
    J_sg = A + B diag(psi) C

This includes all four delay slots and ALIF/STP feedback; B's h/reset rows
are zero but adaptation, resources and new pulse rows remain. A numerical
full10N Jacobian identity test using the actual COBA implementation passes
in double precision, including an entirely silent forward with nonzero
surrogate feedback. This is a tensor algebra test, not a capability task.

A units-only coordinate change gives J'=S J S^-1 and leaves eigenvalues
unchanged. Products transform as S(J_last...J_first)S^-1 for common S.
Changing proxy width without the coordinate chain factor instead changes
the learner; it cannot be presented as pure nondimensionalization.

For a common positive diagonal weighting S, define the nonnegative envelope:

    M0 = abs(S A S^-1)
    M1 = abs(S B) diag(psi) abs(C S^-1)
    a_i = row_sum(M0)_i; k_i = row_sum(M1)_i

If a_i<=1 and every row with k_i>0 has a_i<1, a positive event-credit
attenuation has the sufficient bound:

    chi = min(1, min_{i:k_i>0} (1-a_i)/k_i)
    psi_new = chi * psi
    norm_inf(S J_new S^-1) <= max_i(a_i+chi*k_i) <= 1

Here1is the unit-gain boundary, not a fitted damping constant. Positive
chi preserves nonzero surrogate credit on silent cells. Pure shift rows
may have a_i=1,k_i=0 and retain neutral memory. If any a_i>1, or a_i=1
with k_i>0, this scalar sufficient condition may be infeasible; do not
silently set credit to zero. Choose/test a physically meaningful weighting
or a more precise block condition. This conservative sufficient bound
does not characterize all stable systems or guarantee useful learning.

To bound an entire window, every tick must satisfy the same weighted bound
with the same S; separate per-tick spectral radii or changing unchecked
metrics are insufficient. No real-brain common-S feasibility or chi values
were measured here. Sparse block actions are required; do not materialize
a dense165122-neuron Jacobian. The next action is to establish that
feasibility and then implement a derived proxy calibration. Preserve S14,
W32, all physical forward equations and silent first-firing credit. Neither
active-only production masking nor arbitrary new gamma is approved.

Additional closure requirements from independent review: all k_i=0 gives
chi=1; k_i=0 rows are excluded from the ratio minimum but still require
a_i<=1. Treat chi as a fixed coefficient of the surrogate backward rule,
not a new forward-dependent differentiable state. The Euclidean product
bound is then at most condition_number(S), not automatically1. The10N
identity fixes the drive; actual token dynamics also carry writer baseline
and optional h_mean/Gamma/observer states. Include those in a full training
state bound or separately establish their bounded forcing/observation
contracts. Parameter-gradient magnitude also includes loss forcing and
parameter-source terms; a state-transition bound alone does not bound it
to1. A continuous-life guarantee would require the metric/bounds to remain
valid across physical parameter updates. Absolute-value envelopes lose E/I
cancellation: failure of this sufficient certificate redirects the proof,
not automatically the model. Reviewer accepted the conditional algebra
and withheld real-brain feasibility, deployment and task-benefit claims.

The literature supports treating surrogate derivatives as a distinct
learning convention, and links their probabilistic interpretation to a
specified neuronal escape-noise function. It does not establish this
new gain-bound calibration as a published algorithm:
https://arxiv.org/abs/2404.14964

Sixty affected existing tests pass; three additional augmented-geometry
tests pass. No long training or NLL improvement was claimed this turn.

## Registered common-block feasibility gate (continuation)

Before execution: collect actual fixed-drive COBA blocks throughout the same
480tick saved-individual forward. Absolute incoming weight row sums are
computed sparsely for E/I and all four delays. Actual clamp derivatives,
STP cross terms and ALIF feedback are included. Take a common10x10passive
block envelope U_A,10x1event U_B,1x10margin U_C and max(psi) over the whole
trajectory. If rho(U_A)<1, solve p=(I-U_A)^-1*ones; scale-weighted passive
row sums and feedback U_B*max(psi)*(U_Cp)/p determine maximum positivechi.
This is a conservative certificate construction, not biological parameters.

Seven numerical geometry checks pass, including actual full-Jacobian block
coverage on silent/firing/clipped trajectories and clamp endpoint behavior.
Independent reviewer approved only this forward-only feasibility gate.
No backward, update or PT; original source/targets/S14W32 unchanged. Report
condition number andchi. Infeasible or extremeconditioning/credit attenuation
blocks promotion: tighten the state/block certificate rather than silently
kill learning. A feasible finite bound motivates a separately reviewed full
backward calibration with group-credit checks. Input/writer/observation and
parameter-source scope remains as above. No new NLL claim.

## Common-envelope result and threshold-width candidate

Actual saved-individual 480-tick forward: rho(U_A)=0.99501252,
metric condition number=85,866,348; derived chi=1.521776e-10.
Source parameters and optimizer are unchanged; zero updates, board peak2255MiB.
The sufficient certificate is feasible but unusably conservative. We reject
this chi as a production learning rule: it would again suppress useful credit.
This result concerns the absolute block envelope, not a necessary bound on
the actual signed brain. Report: results/published/fly_w32_feedback_feasibility_20261008.json.

Minimal new candidate, preregistered before its real backward:

    psi = 1 / (1 + (pi * (v - theta_effective) / theta_base)^2)

Width uses existing positive learned base thresholds, detached in backward;
effective threshold remains attached in the margin. Peak is1, silent finite
ordinary voltages retain credit, and there is no new gamma or parameter.
This is a deliberate surrogate-learning convention, NOT a pure change of
units or the exact derivative of a threshold-normalized smooth CDF (which
would include1/theta). The hard forward is unchanged. Legacy absolute-width
mode stays default; continuation explicitly records and inherits the mode.

Literature supports threshold-normalized surrogate margins, not proof that
this exact unit-peak ATan/base-threshold candidate is optimal:
Bellec etal.2018 (https://arxiv.org/abs/1803.09574) uses adaptive-threshold
normalization and a damped triangular proxy; Frontiers2020 eq4
(https://pmc.ncbi.nlm.nih.gov/articles/PMC7339963/) similarly normalizes by
threshold with gamma=.3. We introduce neither that triangle nor gamma=.3.

Prediction: far-subthreshold feedback should fall while near-threshold
first-firing credit remains. Gate: same source, next32realOWTtargets, full
480ticks, backward only; source weights/optimizer immutable, no PT. Report
raw and clipped read/writer/threshold/synapse norms and silent-state adjoints.
CPU full-window physical values must remain bitexact; GPU uses the already
measured atomic self-repeat variation rather than claiming bitexact execution.
Reject if total norm remains huge or learning groups remain starved, even
when technically nonzero. Numerical health is not evidence of improved NLL.
Independent reviewer approved this diagnostic gate;62tests pass,1skip.

The gate passed: norm1.64020811; rawread.616666639 matches the old.616666655,
clippedread.375968322. Clippedthreshold.165516712, synE.442814781,
synI.191371763; writer gate.027190199 and mech.099562144. Mean silent proxy
falls.876039 to.173066. First-tick h adjoint falls6.21383e13 to.127726;
silent first-tick h credit retains87.14% of its energy. All480state hooks
are preserved; no pulse mask is used. Pre-update objective6.483838558 and
NLL6.661264665 agree with the unmodified forward within atomic variation.
Zero updates, all source parameters exact, whole-board2711MiB. These
numbers establish this-window credit recovery, not a task-benefit claim.

Independent reviewer approved a single restored AdamW update (no PT/no long
training). Preregistered gate: preserveS14W32 and original optimizer moments;
compare all480 original/recomputed physical decisions AND saved normalized
proxy margins, verify complete coverage/no flips/finite values before either
optimizer step; then check actual parameter deltas, optimizer states finite,
and3900MiB memory cap. CPU checkpoint/eager gradients match in both modes;
114affectedtests passed,5skipped. Source checkpoint remains immutable.

The guarded single update was blocked before either optimizer ran:480/480
calls compared, but6spike decisions flipped within segment27. Max margin
difference.116577, normalized-margin difference1.15539, proxy difference.928297.
The total gradient norm stayed1.64020822. Thus gradient conditioning improves
but atomic-sum replay is unsafe on this trajectory; do not promote the
candidate as an accepted complete checkpoint repair or erase this failure.

New numerical execution repair, independently approved before its real gate:
stable sort incoming edges by post within each of the existing fourE/I delay
tiers; cache topology only (order, source, row offsets). Each forward gathers
CURRENT weights and uses fixed-order[E,B] segment sums. Original trainable
edge IDs and complete original VJP remain; even zero pulses keep backward
credit. The old CUDA atomic backward remains nondeterministic at rounding
level; only the physical forward is made repeatable. New execution is
optional `transmission_mode=incoming`; old checkpoints inherit `atomic`.

CPUdouble/CUDA repeat and VJP checks cover duplicate edges, emptyrows/tiers,
four delays, silent pulse credit, batch1/2 and changed current weights.
36affected checks pass. Reference implementation follows PyTorch2.9.1
fixed-row two-dimensional segment-reduction source:
https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/SegmentReduce.cu
It does not equate different summation orders bitwise. No new biological
module, derivative mask, event-duration change or NLL claim is introduced.

Next gate: same immutable individual/32targets/full480ticks, no update,
threshold proxy plus incoming summation; exact internal original/recompute
physical margins and normalized margins, zero flips, finite group credit,
whole board<3900MiB. Only after this passes, one restored AdamW calibration
may run. Source weights/optimizer files stay unchanged; no PT/no long train.

Incoming full-window gate result is recorded in
results/published/fly_w32_incoming_backward_20261008.json. The first process
exited without a report; it is inconclusive, not counted as a pass. A rerun
with flushed stage logging completed; its actual coverage/values determine
acceptance. The next single update retains exactly that incoming convention.

The full incoming backward-only gate passed:480/480calls and79,258,560
neuron decisions compared; zero flips; physical margin, normalized margin
and proxy derivative differences all0. Norm1.64020782; rawread.616666639,
clippedread.375968. Wholeboard2999MiB, peakallocated2401.38MiB. No update.

Single restored AdamW calibration also passed with the same exact replay,
32targets/480ticks/1optimizer update. All parameters and optimizer tensors
finite; wholeboard2999MiB. ActualdeltaL2: read.280527 (originalnorm27.8856),
threshold.000210033 (11.8441), synE.206687(59.7033), synI.298272(59.8723),
writermech.100393(576.480). These deltas include preserved optimizer history;
they are not isolated contributions of the current task gradient. Source
checkpoint is unchanged and no PT is written. Results:
results/published/fly_w32_incoming_update_20261008.json.

Validated repair settings on the production continuation CLI:

    --detach-reset --surrogate-mode threshold --transmission-mode incoming

All physical equations, sensory/output restrictions and fullS14/W32 are
preserved. No active/silent backward mask is enabled. Old defaults remain
reproducible; chosen conventions are persisted and inherited on subsequent
continuations. Fixed topology is required for cached layouts; a topology or
delay-partition change must rebuild them. The fixed-row implementation is a
correctness reference with extra topology storage and runtime work. These
instrumented40second checks are not production step-speed measurements.

Numerical credit and replay repair is accepted for this real window. Broader
stability across changing weights and NLL benefit require subsequent joint
training on the registered continuous stream, not inferred from this gate.

Final affected test run:144passed with opt-in CUDA capture/eager parity,
both summation/proxy modes and full-window state/parameter gradients.
Research-tree validator:0errors/0warnings. Independent plan/code/result
review uses `/root/rtc_contract_review`; medium source review uses
`/root/medium_operator_review`. No long training was launched in this repair.
