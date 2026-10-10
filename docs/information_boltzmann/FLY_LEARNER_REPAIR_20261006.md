# Shared learner repairs and the remaining prediction regression

The sensory-only input and motor-only output surfaces, fixed topology, existing
physical tick/delay units, and continuing membrane/current/ALIF/STP states are
retained. This work repairs implementation contracts; it does not certify that
the upgraded individual has recovered historical predictive quality.

## Repairs

- Optimize only actual causal transmission weights: COBA E/I magnitudes or the
  CUBA signed tensor. DAN modulation wiring stays frozen and outside Adam.
- COBA inhibitory weights are nonnegative conductance magnitudes. Both E/I
  tensors retain that constraint. Signed tensors retain their original sign
  masks across zero crossings and resumes; initially zero edges are nonnegative.
  Incomplete mask metadata is rejected; an old signed continuation without
  masks requires an explicit migration from original graph signs.
- The four-slot delayed kernel returns four ring gradients even when a split
  is empty. Short valid tier metadata is padded; invalid partitions are rejected.
- Optional ALIF/STP states are unpacked by flags; CUBA synaptic current is
  preserved in `ge`. Non-topographic sensory projections remain trainable,
  quiet ticks inject zero current, and writer-specific diagnostics are optional.
- Omitted CLI centering/DAN flags inherit the complete checkpoint. Explicit mode
  changes start a separate measurement phase and archive an in-place old best
  on its existing storage volume. Weights, moments, cursors and physical life
  remain intact.
- Recomputing diagnostics preserve running read means and DAN gate state;
  mapped checkpoint models inherit the read mode rather than silently using
  an uncentered head.

## Optional DAN rule: explicit revision, disabled in the mature audit

All actual delayed DAN pulses consumed by a physical tick contribute to arrival:

\[
a_i(t)=\frac{1}{s_{DAN}}\sum_{j\to i}w_{ji}^{DAN}p_j(t-d_{ji}),\qquad
g_i(t+1)=\rho_{s,i}g_i(t)+(1-\rho_{s,i})a_i(t).
\]

`s_DAN` is the existing graph sensitivity scale. Sharing the target's excitatory
synaptic relaxation (COBA), or current relaxation (CUBA), is an explicit model
assumption, not a measured dopamine time constant. Gate history is detached
from task BPTT, advances during quiet ticks, and is stored in physical state.
Legacy activation starts this previously nonexistent history at zero and records
the migration. DAN disabled adds no gate computation to the physical loop.

The local rule remains once per optimizer window:

\[
\Delta w_{ji}=\eta_{DAN}g_i h_j h_i,
\]

followed by retained sign/conductance constraints. It is **terminal coactivity
modulated by a delayed-pulse EMA**. It has no surprise/reward factor or
cross-window eligibility credit and is not represented as a validated model of
fly learning. The old implementation's scalar indexing and negative inhibitory
clamp are corrected. Gate history, rule version and rate are checkpointed.

## Validation and actual finite update

80 affected numerical/interface tests passed, including CUDA/Triton delayed
transmission, whole-window captured backward, two continuing Graph updates,
optional states, source locality, optimizer history, and DAN gate/sign contracts.
The previously skipped DAN test now runs successfully. An independent reviewer
checked the diff and requested sign-coverage, diagnostic-update and old-best
archive repairs; these were incorporated.

Real calibration uses the complete latest centered checkpoint at train cursor
1,300,000 and its saved Adam moments. It is a fork: production checkpoints are
untouched, no checkpoint files are created, and all controls share the same
old-parameter physical state. Train targets are indices [1300001,1300033),
follow targets [1300033,1300065). A fitted replay is not validation.

| Following-window control | NLL |
| --- | ---: |
| Original weights | 7.198421 |
| Full one-window update | 7.271419 |
| Internal/writer update only | 7.273256 |
| Read/decoder update only | 7.196492 |
| Transmission-edge update only | 7.302045 |
| Sensory-writer update only | 7.255010 |
| Cell-biophysics update only | 7.251920 |

Fitted-window NLL improves 8.161176 → 8.113891. Each isolated body-side update
interferes with this following window; their combined effects are nonlinear.
Relative parameter update norms are edges 0.000446, writer 0.000187 and cell
parameters 0.0000268. Small parameter motion alone does not bound predictive
interference in this thresholded recurrent individual.

Captured 32-event update: 0.844 s (~37.9 token/s); peak allocation 2774 MiB,
reservation 3012 MiB; eager/Graph score difference below 2e-6. Full physical
state remains finite. DAN is disabled and existing STP activation remains false,
matching this source arm; the audit does not silently activate another upgrade.

**Accepted:** repaired numerical, optimizer-coverage, persistence and execution
contracts. **Open:** long-run predictive retention after internal plasticity.
This single finite update narrows the next diagnosis to body-side update effects;
it does not prove their long-run average is harmful or identify a biological
mechanism. The optional DAN defects did not cause this already-DAN-disabled
arm's historical plateau. No long training was launched on the strength of
implementation tests.

Independent review verified the crossed controls: body-side interference
0.074836, read-side benefit 0.001928, joint interference 0.072998; their
interaction in this window is about 0.000091. The mature numerical audit
exercises DAN disabled; enabled DAN execution is covered by numerical/Graph
contracts and its predictive benefit remains unverified.

Reproduction: `scripts/ib/audit_fly_repaired_update.py`; compact result:
`results/published/fly_repaired_update_20261006.json`.
