# Representable local plasticity and first-revisit curve

The sensory/motor surfaces and MaleCNS topology remain unchanged. This revision
resumes the complete paused individual at 34,190 training predictions.

## Numerical and local credit repair

Float32 writer/edge updates use a persisted rounding residual:

    total = proposed_update + residual
    new_weight = round_float32(old_weight + total)
    residual = total - (new_weight - old_weight)

The residual is discarded for updates projected outside existing synaptic
bounds. Learning rates and the dopamine floor are unchanged. This retains
small updates instead of making an arbitrary increase in their amplitude.

Each edge maintains local conductance, voltage and adaptation sensitivities.
With incoming delayed/STP pulses conditioned on, the recurrence is

    e_g(t) = lambda_syn,j e_g(t-1) + (1-lambda_syn,j) pulse_i(t-delay)
    e_vpre = alpha_j e_h(t-1) + dVpre/dg_j e_g(t)
    e_spike = psi_j (e_vpre - beta_j e_b(t-1))
    e_h(t) = (1-spike_j)e_vpre - Vpre_j e_spike
    e_b(t) = rho_j e_b(t-1) + (1-rho_j)e_spike

`dVpre/dg_j` differentiates both the exponential alpha and integration factor
of the actual COBA equation. The partial derivatives match multistep autograd
under this local conditioning. Low-rank directional feedback remains an
approximation to global credit. This is not a claim of exact network gradients.

Per-edge storage is O(edges), O(1) in lifetime. Triton packs destination
coefficients into cache-aligned rows. Runtime calibration measured about
49 ms/token after packing, compared with 162 ms before the kernel access
optimization. The added local histories cost approximately 0.43 GiB.

Old physical state, weights, pending updates, optimizer moments, feedback and
other local traces resume unchanged. Newly defined per-edge derivative states
start from zero conditional sensitivity at the revision boundary; the old
factorized traces cannot reconstruct those histories. This migration is
explicitly stored in config and complete checkpoints. Subsequent checkpoints
include all new histories and rounding residuals for exact continuation.

## First-revisit measurement

Each cohort takes five independent 128-target real-OWT episodes from the fresh
training stream. Assignments rotate across cohorts to avoid always assigning
the earliest text to the shortest lag. Episode measurement gaps are logarithmic
samples: 64, 256, 1024, 4096 and 8192 intervening events. They are reporting
choices, not biological parameters or model memory cutoffs.

Each episode is revisited exactly once, at its assigned gap. Actual intervening
events include all training, other revisits and active validation; requested
and realized gaps are recorded separately. A source cohort starts after a
regular validation boundary to avoid immediate overlap with its old A replay.
Initial source episodes interrupted by another exposure are retained in the
raw log but excluded from paired summaries. Actual source start events are
recorded at the first observed target, including after boundary replays.

All predictions are scored before their own target updates the continuing
individual. Physical state, optimizer cadence and eligibility continue. The
different-context first-target bridge is logged separately and excluded from
the matched-transition average. Both opening-16 risk and whole-episode risk
are recorded. Later predictions within a revisit can benefit from learning its
earlier tokens, so this is an active first-revisit retention/relearning curve,
not a frozen pure-retrieval assay.

Artifacts:

- `first_revisit_episodes.jsonl`: raw paired exposures, all scores and actual gaps.
- `forgetting_curve.json`: per-gap mean paired NLL change, episode counts and
  episode standard error once repetitions exist. Positive means worsening.
- `last.pt`: all pending episode queues, partially collected source episodes,
  running summaries and ordinary continuing-individual state.

In-memory measurement state is bounded by configured maximum gap and cohort
cadence. Raw observations append to disk. There is no prescribed exponential
fit or monotonicity constraint; the empirical curve determines whether a
spaced consolidation policy is warranted. Four-pillar active evaluation runs
alongside these measurements with its original protocol.

Validation: the original 31 affected numerical/interface tests passed, including sub-ULP
accumulation, heterogeneous destination derivatives, CUDA/reference agreement,
complete learner continuation and partial measurement-queue continuation.
Three additional measurement tests cover interrupted exposures, actual source
starts after a boundary revisit, and recovery of historical anticipated starts
from executed revisit intervals.

## Initial continuing-run observations

At training cursor 43,541, compared with the complete paused cursor 34,190,
deterministic strided samples of at most 65,536 entries found representable
changes in 14.78% of excitatory and 10.74% of inhibitory edges. Their relative
changes were 1.77e-5 and 2.59e-6. Recurrent edge updates have no weight-decay
term. Writer changes also include its existing weight decay, so total writer
weight movement alone is not evidence of task-directed learning.

At the first five-gap snapshot, paired revisit-minus-initial NLL was:

| Actual intervening events | Episodes | Mean NLL change | Episode SEM |
| --- | --- | --- | --- |
| 64 | 3 | -0.3326 | 0.0574 |
| 256 | 1 | -0.4091 | unavailable |
| 1024 | 3 | -0.9413 | 0.3153 |
| 4096 | 2 | -0.1774 | 0.7151 |
| 8192 | 2 | +1.1054 | 0.0888 |

All actual intervals equal their requested gaps in this snapshot. Negative
means improvement; positive means degradation. Samples remain few and text
content, context re-entry, general learning drift and within-exposure relearning
all contribute. These observations establish the measurement, not an
exponential forgetting law or a validated periodic-review schedule.
The continuing run keeps accumulating episodes up to the authorized 100,000
fresh training predictions. Current artifacts supersede this dated snapshot.

## Execution optimization at cursor 53,712

For an edge with zero conductance/voltage/adaptation sensitivities and zero
arriving pulse, the next local sensitivities and gradient are exactly zero for
finite coefficients. The Triton kernel now masks only its destination-coefficient
gather. Pending updates, rounding residuals and the bounded weight projection
still execute. Float32 histories, learning cadence and equations remain intact.

A complete-state fork using the same checkpoint and real next OWT targets
measured 49.79 ms/token before masking and 40.73 ms/token after masking, about
20.1 versus 24.6 tokens/s. Edge credit itself fell from 19.65 to 9.59 ms/token.
The 512-edge/eight-warp layout gave 9.69 ms edge credit, so the existing
256-edge/four-warp layout is retained. These are actual eager timings with
nested CUDA events, not CUDA Graph extrapolations. All 17 affected credit,
learner and first-revisit tests pass, including dormant histories with pending
updates, compensation, heterogeneous local autograd and full-state continuation.

Predictive benefit remains unestablished: fresh active NLL at cursors 45,000
and 50,000 was 7.3920 and 7.4234, versus the previous historical best 7.1352.
These evaluations use different fresh text segments and do not constitute a
matched causal comparison. Representable circuit updates establish numerical
plasticity, while NLL must establish its usefulness.
