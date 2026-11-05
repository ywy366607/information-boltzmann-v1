# Continuous learning evaluation for a persistent individual

## CGB joint-body evaluation contract (2026-11-03)

The organism definition and applicable joint premises take priority. Standard
delta's no-write-on-agreement and orthogonal-read preservation are candidate
properties, not universal organism requirements. Losing either proves a change
of recurrence, **not harm**. The three 5000-update runs establish finite-budget
scores, not convergence, long-memory failure, or an architecture ranking.
Pause body repair until the following measurements identify a useful change.

Qwen's official architecture report evaluates downstream ability alongside
loss, cost and stability; its long-context evaluation includes length-stratified
RULER and MRCR. Its base-model knowledge/math/code evaluations and post-trained
agent evaluations are different stages with different protocols. Sources:
[architecture report](https://arxiv.org/html/2608.30320v1),
[official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
We reuse the multi-axis method, not its numerical thresholds or synthetic
retrieval tasks (this repository requires real-world capability datasets).

| Question | Measurements | Controlled comparison and status |
|---|---|---|
| Prediction on unseen language | Nats/byte, BPB, next-byte accuracy; region distribution | Same real OWT targets across models; disjoint regions beyond the validation prefix used for selection. BPB/PPL are transforms of NLL, not extra independent wins. Executable now. |
| Does more history help? | Same-target score change for 32/128/512/2048/8192/32768 known bytes; first 32/128/512 target bytes and full 4096-byte segment | All state persists through each prefix. Execution chunk stays 32. Eight disjoint maximal windows; frozen weights. Concatenated streams may cross documents, so this is not semantic document retrieval. Executable now. |
| Actual comprehension and completion | Exact next-word continuation, target likelihood; later real-document QA EM/F1, code tests/pass@1 | LAMBADA development v1 below is registered and executed on all 4869 passages. Real-document QA and code remain unmeasured. Byte accuracy cannot fill this row. |
| Lifelong adaptation and retention | Pre-update A→B curve, common-horizon loss integral, confirmed recovery time, exact-target A1/A2 savings and relearning curve | Preserve actual state, weights, optimizer and update cadence. Reuse `recovery_summary`/`savings_summary` below. Nine matched 6400-byte log blocks were recovered; full per-byte curves/recovery remain unavailable. Do not fabricate them from block means. |
| Training budget dependence | Same-data-budget learning/downstream curves; separately same wall time or compute | Current 640k-byte exposure is one budget point. Do not extrapolate a winner or tune one arm more. Later budgets and stopping policy are fixed before continuing, one seed initially. Record convergence evidence; missing historical checkpoints cannot be recreated from logs. |
| Resource and stability | Full persistent/optimizer/workspace bytes, throughput, latency, nonfinite events, loss spikes, gradient/activation distribution, memory drift with stream duration | Frozen-evaluator throughput is not training throughput. Matched data and optimizer budget answer recipe performance; architecture-level comparison additionally needs a comparable tuning budget. |
| Organism contract | Same-state boundedness, conservation, causal boundaries, recoverability and applicable theorem premises | Separate formal/implementation coverage from learned usefulness. Unmeasured action/ecological closure remains unmeasured; high benchmark scores do not certify life. |

**Decision policy.** Report tradeoffs by axis, without a post-hoc weighted total.
“This checkpoint worsens metric X on these targets” is permitted; “the mechanism
is harmful” requires evidence across the relevant budgets/tasks and an isolated
intervention. Matched seeds/data do not isolate multiple simultaneous recurrence
changes. A specific contract violation still requires repair, independent of
scores; an optional delta identity does not acquire that status through testing.
More context is evaluated at identical target positions, not by comparing easier
and harder texts. Report region-paired differences and uncertainty conditional on
the trained seed; these intervals do not cover seed or corpus uncertainty.

**Practical order.** First freeze this protocol and evaluate the existing three
checkpoints. Then register natural-language completion/QA data and the scoring
adapter before any new model run. Only after that choose a sufficient extended
training budget, with periodic snapshots and aligned online scores. Both phases
are needed: extending training without downstream evaluation, or testing a very
small undertrained model on advanced tasks at the score floor, cannot settle the
architecture. Test data stay untouched until the recipe/protocol is frozen.

`scripts/ib_local/evaluate_gated_deltanet.py` is the first executable portion,
not the completed suite. It loads the actual checkpoint constructor and weights,
uses CUDA Graph with batched independent regions, verifies source hashes, records
target positions and source/code/checkpoint provenance, and never updates the live
individual. Its output marks online learning and semantic downstream as unmeasured.
This frozen sidecar is separate from the live-continuity protocol below.

### Registered natural-language task: LAMBADA development v1 (2026-11-03)

Before scoring: use all 4869 development passages of `cimec/lambada`, revision
`5953bd97664b64b95754f299b2309ecfbfbe81b9`, validation parquet SHA256
`dc06d2ead0fc40bd0845936d6b2375f00396d7baac92d8335707d6a68b3d25a9`.
Only this split is downloaded. No fitting, prompt selection or training occurs.
Data and manifest are under `data/cgb_eval_lambada`; body/checkpoints unchanged.

Follow [lm-evaluation-harness lambada_standard](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/lambada/lambada_standard.yaml):
split at the last ASCII space, context is everything before it and target is a
leading space plus the last word. Preserve source text and UTF-8 bytes. Compare
last 32 bytes, last 128 bytes, and full context for the identical target, without
instructions or few-shot examples. No passage concatenation. Reset the frozen
sidecar per passage, not within a passage. Length bucketing/right padding affect
execution only; padded scores never contribute and no padding precedes a prompt.

Primary: fraction of passages with every target byte greedy-correct, equivalent
to greedy-prefix agreement at the gold target length. This does **not** test a
following word delimiter. Secondary: mean summed target NLL and its exponential
(target perplexity), target BPB, paired full-versus-short gain, and exact accuracy
flips. These are not OWT whole-stream perplexity or closed-loop learning scores.
Keep per-passage scores/IDs. Report passage-bootstrap uncertainty as conditional
on this single trained seed; book clustering and training-data contamination are
not certified absent. An accuracy floor means the current checkpoints do not yet
demonstrate this ability; it does not distinguish ultimate architecture capacity.

Entry: `scripts/ib_local/evaluate_lambada.py --checkpoint <last.pt> --output <new.json>`.
Run the three existing 5000-update checkpoints at fixed FP32, batch32/chunk32,
CUDA Graph. Record runtime/peak evaluation workspace separately from training.
No ranking on other Qwen benchmarks, live adaptation or organism admission is
implied. The next missing axes remain real-document QA, aligned live relearning,
and sufficiently trained budget curves.

**Executed results.** All three checkpoints scored 0/4869 greedy target-prefix
matches, for all three context lengths. Full-context mean target NLL:
delta 16.203486, old exchange 16.218333, available contact 16.401794.
These are nats per target, not nats/byte. Full versus last32 mean NLL gains
are −0.000158, −0.008463, −0.024821 respectively. The current checkpoints do
not show a full-context improvement on this task. An accuracy floor and one
undertrained budget cannot settle architecture capacity. Results are in
`results/published/cgb_joint_evaluation_20261103.json`; raw per-passage arrays
remain with their run directories. Ten numerical/scoring tests passed on CPU/GPU.
Two runs briefly overlapped on GPU: elapsed times are execution records, not a
matched speed benchmark.

**Recovered live evidence without training.** `aligned_revisit_blocks` maps
recorded window means back to exact source target intervals. It excludes blocks
ending at phase boundaries because their final successor may jump elsewhere.
For each run, 9 × 6400 = 57600 identical targets survive (90% of the revisit).
First/revisit NLL: delta 1.801310/1.625619, old exchange 1.826006/1.676388,
available contact 1.872657/1.728070. These establish improvement on return after
further learning, combining memory, general learning and replay adaptation.
They do not isolate useful latching. Full 64000-byte savings and fine recovery
times remain unknown; no synthetic per-byte curve is reconstructed.

```powershell
python scripts/ib_local/evaluate_gated_deltanet.py `
  --checkpoint results/cgb_joint_gpu_20261103/available_contact_5000/last.pt `
  --data data/ib_owt_10000 --output results/<new-evaluation>.json
```

The accompanying fourth pillar records energy balance, spatial/representation
structure, actual optimizer displacement, predictive risk drift and conditional
full-state response. It adds measurements rather than an objective or critical
controller. See [PERSISTENT_MEDIUM_HEALTH.md](PERSISTENT_MEDIUM_HEALTH.md).

This is the primary evaluation contract for the continuously learning 3D
medium. Its three questions also apply to other never-reset agents: predictive
quality on new experience, adaptation after context change, and availability
of past experience on return. Fly implementation remains maintained separately.

## One running individual

Evaluation continues the actual learner and optimizer. It preserves the field,
flux, receptors, structural conduction, STP, precision, eligibility, parameter
weights, optimizer moments, pending gradients and physical clock. There is no
evaluation-only update rule, mode switch, zero-state initialization, independent
copy per site, or restoration of old weights between the three phases.

The causal order is: use the observed prefix to predict, record predictive NLL,
then let the newly arrived target support learning for future predictions. The
existing learner computes targets only in its loss; its returned `token_nll`
is the predictive score before the optimizer update. That score is recorded
once, even when online learning subsequently makes the token easier.

`tokens_per_update` controls the existing optimizer cadence. It continues across
every context/measurement boundary. Boundaries never flush a partial gradient
group. A checkpoint saves pending gradients/count as well as physical and
eligibility state, so replay after resume follows the same learner trajectory.

## The three measurements

### Second-pillar standard: adaptation speed and predictive quality (2026-10-08)

For a declared horizon H of real, pre-update B observations, report
`APPL_H = exp(sum(B_curve)/H)`. Lower is better. When a fixed training-only
reference scores the identical targets, also report
`G_H = exp(mean(reference_nll) - mean(B_curve))`; values above one favor the
individual. Preserve the log scores as the numerically robust representation.
Every scored transition, early shock and later relapse contributes. No token is
removed because recovery has not occurred. H must be shared in comparisons.

APPL is a monotone transformation of mean prequential NLL, not new information
independent of that NLL. Its purpose is a unified speed/quality reporting scale.
A faster return to the same low-loss regime improves it; a lower adapted loss
also improves it. AUC alone does not identify which of these caused an advantage,
so recovery time and terminal quality remain explanatory measurements. A fixed
unigram reference adjusts lexical frequency difficulty, while exact stream order
and exposure/update budgets remain necessary controls for contextual difficulty.

The shared `recovery_summary(...)["generalization"]` contains the standard.
`plateau_estimate_nll` is the last fixed complete-block window's mean.
`plateau_nll` is populated only if both the fitted end-to-end trend and the
difference between window-half means are within `plateau_tolerance_nll`.
Defaults are max(4, 2*hold_blocks) blocks and 0.1 nats/token tolerance. These are
explicit finite-measurement policies, not dynamics parameters or proof of an
infinite-time asymptote. Report the policy, window interval, dispersion and drift.
The stable-tail check is descriptive, not a statistical stationarity certificate.

`recovery_tokens` is the confirmation time for entering the platform tolerance
band without a later block relapse. `half_recovery_tokens` uses half of the
opening-to-platform drop under the same confirmation condition. Constant good
prediction can have zero recovery time while retaining its absolute quality.
When the platform is unconfirmed, times remain null and APPL still reports the
entire actual trajectory. An unconfirmed platform means more observation is
needed to characterize the endpoint; it does not invalidate the primary score.

Old half-recovery fields remain explicitly labeled legacy for file compatibility.
Readers should use the nested v2 generalization fields for the combined claim.
Running Python processes already holding old functions load this upgrade on
their next safe resume; historic raw logs are kept unchanged. Existing curves
can be backfilled into separate published analysis artifacts without replaying
experience or updating the individual.

1. **First-pass prequential prediction.** Record NLL for every prediction on
   the next unseen stream position; report cumulative and recent scores plus
   actual event/update budgets. First-pass means indexed observations first
   encountered under this protocol, rather than claiming corpus-wide text
   deduplication. Repeated A traffic is separately labeled and excluded from
   the first-pass primary mean.

2. **Context-change adaptation.** Continue into fresh B observations with
   learning active. B's first observation is scored from A's last observation;
   it is a real transition rather than an unscored change of input context.
   Keep the complete B score curve. For a descriptive recovery estimate, use
   complete measurement-block means, the opening block and observed late
   blocks, and confirmation by consecutive blocks. An isolated easy token
   cannot count as recovery. With no observed improvement, insufficient blocks
   or no confirmed crossing, return an explicit status and `null` time.
   These are observed finite-trajectory recovery statistics, not a fitted
   asymptotic biological time constant. Text difficulty is controlled in model
   comparisons by sharing the exact stream order and exposure/update budgets.

3. **Actual A1 -> B -> A2 return.** Store A1's exact observations and predictive
   scores. Process B for the actual recorded number of events, then return to
   A by scoring the B-to-A[0] bridge and replaying the identical A target pairs.
   Report first-block and full-curve NLL savings. Relearning speed uses the
   **same absolute NLL criterion** for both encounters; absent crossings remain
   censored instead of inventing a speedup. The intervening count is B's actual
   events plus the scored return bridge. B observations, replay learning and
   optimizer moments all remain part of subsequent life.

Measurement block/confirmation lengths are explicit reporting resolution,
saved in the run configuration. They introduce no dynamical threshold, clock,
forgetting gate or extra trainable module.

## 3D entry point

`scripts/ib/train_online_plastic.py` now uses this protocol for both existing
online learning modes. Every ordinary training group contributes prequential
scores. At `--validate-every`, the just-completed group is A1, fresh observations
from `validation.npy` provide B, and the exact group is replayed as A2.

```powershell
python scripts/ib/train_online_plastic.py --credit local-receptors `
  --output results/<new-life> --event-duration 0.005 --execution graph `
  --validate-every 500 --change-tokens 384 `
  --measurement-block 32 --recovery-hold-blocks 2
```

`validation.npy` now serves as **fresh online adaptation experience** for this
entry point. It is not described as held-out frozen validation after it teaches
the individual. Its cursor advances, is checkpointed, and fails on exhaustion
instead of silently looping the same B. OWT split changes establish a context
change; claims about specific new semantic domains require labeled domain data.

Outputs:

- `metrics.jsonl`: train-group scores, first-pass/lifetime prequential scores,
  actual optimizer updates, phase scores and physical diagnostics.
- `lifelong_evaluation.jsonl`: complete A1/B/A2 curves, real bridge score,
  recovery/savings measurements, experience intervals and stream cursors.
- `last.pt`: complete ongoing individual and pending optimizer accumulation.
- `best_prequential.pt`: snapshot selected by lifetime first-pass predictive
  NLL, explicitly distinct from a historical frozen-validation best checkpoint.

`--steps` counts primary training groups. B and A2 add real experience and
optimizer updates, reported separately in the live ledger. WSD learning rate
continues at its current value during each inserted protocol; the evaluation
never resets the optimizer or starts its own learning-rate schedule.

`--resume` restores the full live ledger and pending gradients. Older complete
online checkpoints can migrate to this evaluation policy without resetting
physical/eligibility state; their old frozen score is not reused as the new
best metric. Explicit `--initialize-from` starts a declared new eligibility
branch while preserving the saved physical state, as before.

Historical frozen NLL reports retain their original meaning and provenance.
The new primary scores are labeled by the active-learning protocol. A GDN
comparison must use the same experience order, replay exposure, learning policy
and reported training budget to answer which continuously learning system
predicts better.

## Interface verification

Tests cover prediction-before-update, target-independent forward physics,
exact equivalence to an unsegmented continuous trajectory, real A/B/bridge
counts, sustained recovery, a shared savings criterion, and resume including
pending gradients and optimizer moments. A real-OWT small-model CLI fixture
also verifies advancing B cursors and continued checkpoints. These tests
verify implementation; they are not capability or long-memory experiments.

Implementation: `runtime/lifelong_evaluation.py`. No new long training run is
started by adopting this evaluation contract.

Verification on 2026-10-04: 16 local/live checks pass with CUDA enabled,
including two distinct GPU checks. The broader run passes 46 checks with seven
optional skips. Production 8x8x4/d128, real OWT, CUDA Graph first-pass execution
averages 2.753 seconds per 128-token group over two actual updates / 256 events;
peak tensors 315.17 MiB, total dedicated GPU use 657 MiB. Additional B and replay
time is accounted separately rather than hidden in this ordinary-group timing.
Report: `results/published/lifelong_medium_evaluation_interface.json`.
