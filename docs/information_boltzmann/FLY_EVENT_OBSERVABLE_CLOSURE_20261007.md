# Event credit and motor measurement: offline harness reconstruction

Date: 2026-10-07. Stage: EXPLORE / offline derivation and design review.
Parent: `IB_FLY_REPAIR_20261007`.

This is a follow-up interpretation of already inspected results. Its predictions
apply only to a future, separately approved measurement. They are not blind
predictions of the preceding diagnostic. No production change, new GPU work or
training is performed here. The rejected reset-detach candidate stays rejected.
The unavailable Claude Workflow is not claimed as executed: independent roles
and the strict research-tree validator are the available harness enforcement.

## 1. Evidence and the question to solve

O1. The original saved-Adam body displacement decreases fixed-event fit risk by
0.000282541; its fixed-event Taylor prediction differs by only 3.43e-7. The hard
fit change is +0.000135057. Their finite event-path contrast is +0.000417598.

O2. The same original body displacement decreases following-window risk by
0.000551254. A local fit reversal therefore coexists with a following benefit.

O3. Normalized projected-feature variation fractions on fit/following are
0.0001106565 / 0.0000643681. Full-minus-retrospective-common head risk is
-0.000861928 / -0.000299640. These concern the current head on two windows;
the retrospective common predictor is not an online competitor.

O4. A different real prefix changes following risk by -0.0005932003. The
head responds to past experience; this does not certify general semantic utility.

O5. Reset-detach fails its registered body-only and matched-body usefulness
gates. Its joint benefit has an actual-head-update confound from global clipping.

K. Sensory-only input, motor-only output, pre-observation scoring, complete
never-reset physical state, actual saved optimization, real OWT, 4 GB device.

I. These observations focus the question on **which task-relevant contrasts
survive the motor measurement and which finite event changes the learning
operator actually rewards**. Energy and spike-flip counts cannot substitute for
that object. Existing records leave the global attribution unresolved.

Three structurally different formulations:

1. Hybrid dynamics: does the actual parameter displacement change discrete
   events in a direction that helps the issued probabilities?
2. Measurement geometry: where does a history/event contrast lose visibility
   between motor activity and the vocabulary probabilities?
3. Statistical adaptation: is the visible contrast task-aligned and transferable,
   or is the current head mostly adapting shared marginal probabilities?

## 2. Necessary mathematical closure

Let Z denote the complete persistent state. At the issue deadline, old-ring
arrivals produce motor pre-reset potential v, effective threshold theta, spike
s=H(v-theta), and physical transmitted pulse p (including STP). Current target
has not yet been assimilated. The implemented measurement is

    h = v * (1-s)
    r = W_read h
    z = D_gamma r / rho,    rho = sqrt(r^T r / d + epsilon)
    a = W_dec z + b.

The pulse p goes into physical transmission; this read head uses h. This
distinction is a code fact, not a conclusion that a pulse decoder is superior.
Read-centering is off for the locked source, and read_norm is RMSNorm.

### Finite event contribution, rather than a surrogate-gradient identity

For the same initial state, targets and updated body parameters, let H be the
hard-event trajectory and F the trajectory with every original event held fixed.
For one issued target y, define delta = a_H - a_F and p_F=softmax(a_F). Exactly:

    ell_H - ell_F = log E_{p_F}[exp(delta)] - delta_y.

This equation is invariant to adding any common scalar to delta. It measures
how the finite event-path contrast reaches the actual task. F is an artificial
counterfactual used for accounting, not a deployable alternative brain.

Equivalently, with A=delta_y-E_{p_F}[delta] and p_H=softmax(a_H),

    ell_H - ell_F = KL(p_F || p_H) - A.

A is the correctly labeled logit alignment relative to the reference predictive
distribution. KL is the nonnegative change in that distribution. A<=0 makes
this contrast nonbeneficial; A>0 can still be outweighed by KL. That second case
describes the local finite contrast, not a proof that temperature calibration
is the globally correct repair. Computing A and KL needs probabilities/logits;
the saved NLL scalars alone do not identify the two terms.

For the original parameter displacement, the recorded fit identity is

    Delta ell_hard = Delta ell_fixed + (ell_H - ell_F).

The last term includes all downstream continuous consequences of changing the
event path. It is not the ordinary derivative of H, an isolated reset effect,
the contribution of each flipped spike, or proof of global bad learning.
Different target/history pairs can legitimately give opposite signs.

There is a second exact closure assuming finitely many event boundaries, finite
one-sided loss limits, and a piecewise continuously differentiable or absolutely
continuous loss on the intervening intervals. Along a straight parameter path
theta(lambda)=theta_0+lambda*Delta, partition the finite unroll into intervals
with fixed event decisions. Transverse isolated crossings suffice for the event
part of this partition. Then

    Delta L = sum_intervals integral grad L_event(theta(lambda)) dot Delta d lambda
              + sum_boundaries [L(theta(lambda)^+) - L(theta(lambda)^-)].

The first term accounts for smooth motion within event regions; the second
accounts for finite jumps between regions. Simultaneous crossings are treated
as one aggregate boundary. This formula is a conditional mathematical identity,
not a practical instruction to enumerate millions of events. Smooth clamp/gate
breakpoints can be included in the partition. It explains why a correct local
fixed-event derivative alone does not ensure a finite hard step descends.
The recorded H/F contrast is not this sum of boundary jumps: it compares two
complete trajectories at the final parameter point and includes downstream
continuous responses. Both objects must remain distinct.

### The measurement chain has distinct null directions

Holding the normalization gain fixed, its ordinary Jacobian is

    J_norm(r) = D_gamma [I/rho - rr^T/(d rho^3)].

Before D_gamma, a direction orthogonal to r has gain 1/rho. The radial
direction has gain epsilon/rho^3, approaching zero when epsilon is negligible
relative to mean squared feature amplitude. Thus large radial changes may have
little logit effect. Likewise, W_read has null directions; post-reset h maps
every above-threshold v for a neuron to zero. Each stage can reduce contrasts.
The equations identify possible losses, not their measured location or utility.

For an infinitesimal smooth motor perturbation,

    delta ell = (p-onehot(y))^T W_dec J_norm(r) W_read delta h.

Task relevance is this alignment, not ||delta h||, event count or rank alone.
For finite changes, use the exact cross-entropy equation above.

### What is necessary, useful, and merely possible

Necessary: the causal issue information set; actual finite displacement; matched
initial full state/targets; separate H/F accounting; task-weighted effects after
the implemented measurement. Claims about missing information require an
explicit observable and decoder class.

Useful: trace the existing motor v,s,p,h,r,z and logits on the three matched
trajectories. This can locate loss of a particular contrast along the head.

Possible: alternative motor read variables, event-compatible learning,
normalization/centering changes, longer historical credit, or coadaptation changes.
None is forced by the available numbers. RMSNorm itself is compatible with
informative directional codes; its presence does not establish a defect.

## 3. Orthogonal explanations and future discriminators

| Explanation | Distinguishable future observation | Consequence if absent |
| --- | --- | --- |
| E1: finite event displacement has adverse task alignment | Same-parameter H/F contrast raises task risk while F decreases; separate target-logit and partition terms explain the sign | A beneficial H/F term removes this local adverse-event explanation |
| E2: actual Adam/clamp direction, not SG alone | A changed optimizer direction changes H/F usefulness at a controlled functional comparison | This requires another approved intervention; a single direction cannot settle it |
| E3: upstream motor activity is nearly tonic | All native motor observables already have weak resolved history contrast before W_read | Surviving motor contrast shifts attention downstream |
| E4: projection/normalization suppresses a visible motor contrast | Resolved h differences are lost at W_read, or mainly radial r differences vanish in normalized z | Resolved directional z differences shift attention to decoder/task alignment |
| E5: post-reset measurement removes a contrast | Motor v/s/p contrast remains resolved while the corresponding h contrast vanishes; task utility needs an independently trained matched read comparison | Surviving h contrast shifts attention downstream |
| E6: visible contrasts mainly change vocabulary calibration | Update-induced logits are nearly common across events and shared shift accounts for fit/following signs; A and KL specify their local risk | Resolved time-varying task benefit weakens this common-shift explanation |
| E7: inherited state/omitted historical credit | Effects depend on full entering state or missing entry-state parameter tangent | The three-trajectory local trace leaves this hypothesis open |

E1-E7 can coexist. E4/E5 are geometric contrast-loss claims; E6 concerns
usefulness of a contrast that actually reaches logits. E2/E7 require separately
controlled comparisons and are not silently ruled out by this proposed trace.
Finite target/sample heterogeneity is an auxiliary alternative for every risk
claim. It requires replication and remains open under this two-window design.

## 4. One next action: a design-reviewed measurement, not a new mechanism

Proposed discriminator: record the existing read chain on exactly three
trajectories, each on locked fit32 and following32:

    A: old parameters, original hard events
    B: original saved-Adam actual body displacement, hard events
    C: same updated body parameters, all A events fixed.

Keep head/bias identical. For following32, all trajectories start from the old
fit-terminal complete state, as in the preceding diagnostic. This fixes the
conditional continuation question; it does not test deployment where a changed
body has already produced a changed fit-terminal state.

Record only pre-observation motor variables (v,s,p,h), r,z, and centered-logit
contrasts plus A, KL and their exact risk decomposition. Here logit centering
means removing one all-vocabulary scalar per event (a probability gauge), not
removing a vocabulary-vector time mean. Any separate retrospective time-mean
analysis must keep the common vector and reconstruct the complete logits for
CE and KL. Retain exact finite r/z endpoints alongside local radial/tangential
components: the Jacobian alone is a local approximation. Norms in different
coordinate systems or physical units are not directly comparable measures of
information; use each field's replay floor and verify its implemented map.
Derive the motor pulse from the
same old-state STP variables; never re-run full integration to obtain a read.
Do not feed alternative variables to the trained head and call their NLL an
architecture comparison. No free coordinate scales, extra decoder or Gamma.

The old diagnostic removed its staged actual displacement. Reconstructing that
point would require one disposable saved-Adam update, including its original
global clip and post-clamp, rather than pretending the old update still exists.
Any execution proposal must count this backward/update, verify reconstruction
against old norms/dots/clamp/risk and lock every implementation hash first.

Prospective decision: identify only the attenuation location and task sign of
these contrasts. A resolved geometric loss makes the corresponding readout
change a candidate for matched joint-training review. An adverse event term
with surviving readout contrasts makes the learning/event interface the
candidate. Mixed or unresolved effects keep the attribution open.

Stop before execution if reconstruction, causal timing, identity, numerical
floors, full-state continuity or resource bounds cannot be established. The
current offline work grants no execution approval; a measurement review and
source-locked approval must be completed first. Capability decisions still need
sufficient real-stream joint training and active evaluation.

## 5. Literature boundary and audit trail

Gygax & Zenke (2025), *Elucidating the theoretical underpinnings of surrogate
gradient learning in spiking neural networks*, studies the relation of SG to
stochastic derivatives and explains why SG is not automatically the gradient
of a deterministic spiking objective. Its scaled-reset comparisons also prevent
treating reset exclusion as a universal cure. These are theoretical context,
not evidence for the root cause of this fly checkpoint:
https://arxiv.org/html/2404.14964v3

Evidence remains the immutable previous diagnostic and its independent result
review. New independent hypothesis, necessity and design reviews are stored as
separate artifacts; the previous registration and result are not revised.
