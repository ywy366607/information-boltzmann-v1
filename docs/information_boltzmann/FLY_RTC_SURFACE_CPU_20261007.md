# Complete motor-observation CPU assay

## Contract

Test the proposed fixed-coordinate motor channel before installing it in the
production RTC route. A student retains the complete motor membrane m and the
existing running mean, predicts future m autonomously, and decodes through the
same original motor-only output_read. Regional compressed dynamics remain
approximate. The new observation channel closes the read reconstruction
interface; it does not establish full reduced-state dynamical closure.

The user authorized this CPU numerical test. rtc_contract_review approved the
paired plan before fitting. No CUDA model, language labels, checkpoints or
production learner modifications were used. The previous failed report remains
unchanged.

## Matched exploration

Use the same anatomically selected256-neuron/5901-edge induced MaleCNS graph,
fresh seed11 COBA/ALIF/STP oracle, eight independent continuing stimulus lanes,
and14 quiet future ticks. Four entire lanes train; four other lanes are held
out, including different current amplitudes and timing. Each oracle lane starts
once and carries physical values continuously.

The complete-input and zero-input arms are cloned from one fresh student.
They have18,310 trainable parameters each, identical initialization, AdamW
lr0.002, no weight decay, norm clip1 and120 optimizer updates. The control zeros
the raw motor input to the motor prediction head at every rollout step. Both
retain the same actual origin motor for the additive response baseline and the
same regional latent history. Later steps use predicted m, never future teacher
m. Future teacher states enter losses only.

Both arms fit train-normalized latent and raw-motor MSE. The previous student
fit latent and decoded-response MSE, so its3.362 result is a descriptive
historical comparison rather than a pure ablation. The old student's15,695
parameters versus the new18,310 are disclosed; the paired comparison itself
matches capacity.

## Results

The paired run took9.91seconds after imports,240 total optimizer updates.
27 focused regression tests passed, including exact zero-tick centered and
uncentered reads, current-read-weight revision, fixed raw history semantics,
and autonomous versus masked motor feedback.

| Held-out metric relative to persistence | Complete motor input | Zero motor input |
| --- | ---: | ---: |
| Native decoded-response MSE | 1.30937 | 2.11370 |
| Raw motor membrane MSE | 1.15155 | 1.77524 |
| Regional latent MSE | 0.66289 | 0.66521 |
| Zero-tick maximum read error | 0 | 0 |

Complete input reduces decoded-response error by38.05% relative to the matched
control. Both full-horizon decoded-response errors remain above persistence.
Complete input's3rd–5th tick errors are0.920,0.922,0.941 times persistence;
from6th tick onward the ratios exceed1 and reach1.740 at14th. This is an observed
horizon-dependent pattern, not a demonstrated unique cause or proof that five
ticks is an optimal compute budget. Training response error is0.341 times
persistence while held-out error is1.309, so trajectory generalization is an
additional unresolved part of this fixed-budget fit.

## Decision

The lossless observation interface passes; short future prediction gains are
partial and long-horizon reliability remains unestablished. Retain this as a
candidate observation repair. Do not advertise language gain, full-brain
predictability, a converged failure, or DAgger performance from this result.
Use already-planned matching physical arrival/re-anchoring and independently
reviewed query learning in the next bounded prediction test before committing
to long task training. Preserve14-tick coverage; do not retrospectively select
only the favorable3rd–5th ticks as the primary score.

Entry: scripts/ib/check_fly_rtc_surface_cpu.py

Results: results/published/fly_rtc_surface_cpu_20261007.json
