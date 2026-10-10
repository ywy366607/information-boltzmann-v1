# Same-individual speed optimization, 2026-10-09

Production is paused at step305 /9760 fresh OWT targets /317 optimizer updates.
The complete `last.pt` and STOP file remain intact. All audits restore that
snapshot independently; they commit no production state or data cursor.

## Actual bottleneck

Warm inclusive production spans before this change:

| Execution phase | Seconds /32 targets | Fraction |
| --- | ---: | ---: |
| Writer auxiliary: complete physical replay and reverse credit |3.75|35.9%|
| Main-task reverse credit, including writer/readout/physics |4.51|43.2%|
| Main forward, decoding and remaining orchestration |1.87|17.9%|
| CPU/GPU Adam moment staging |0.315|3.0%|

These are measured execution phases, including scheduling/stream waiting.
Fused physics is not split into speculative exclusive collision/transport/bath
times. Nested read/write/physics spans must not be added to the outer phases.

## Selected execution

`--shared-credit-execution --fused-optimizer`, with the existing bounded physical
CUDA graphs, compiled ports, event checkpoints and Adam moment CPU staging.
The main and writer objectives share one primal trajectory and one replay per
event. Two independent adjoint lanes retain complete32-event state credit.
Writer auxiliary credit still reaches only its declared writer/embedding leaves;
it does not become additional structural likelihood or medium-parameter credit.

The first retained compiled VJP receives owned, stride-preserving workspace.
The final VJP uses the original saved buffers. Immutable parameter/table storage
is reused. Pack hooks retain detached storage rather than creating autograd
reference cycles. Event loop counts are memoized from the forward schedule;
duration and clock derivatives remain differentiable during replay.

Fused AdamW preserves parameter groups, hyperparameters, saved moments and
None-gradient semantics. A post-step version increment is essential on this
PyTorch build: its fused kernel changes storage without updating Tensor versions.
This keeps replay guards and parameter-norm caches valid. Update monitoring uses
bounded4MiB differences instead of another148MiB vocabulary-sized temporary.
Representation monitoring uses the smaller Gram spectrum instead of rectangular
SVD, with numerical parity checks. State dimensions and objectives are unchanged.

## Matched real-data acceptance

Independent processes restore identical checkpoint/RNG and score the next160
real OWT targets with the actual learner and fourth-pillar observations. The
compile/capture-containing first update is excluded; four warm updates follow.

| Execution | Warm mean seconds /32 | Sampled dedicated peak |
| --- | ---: | ---: |
| Original separate auxiliary replay |10.4059|3344.65MiB|
| Selected shared primal + fused AdamW |9.4029|3356.65MiB|
| Shared + compiled deferred linear-weight GEMMs |9.3931|3256.65MiB|

Selected throughput gain: **1.1067x**; wall-time reduction **9.64%**.
Same-target update NLL differs by at most **2.7381e-7 nats**. This is execution
equivalence evidence; it establishes no language-capability improvement.

Full32 native-reference versus selected compiled execution:11 outgoing state
tensors max difference2.8611e-6, incoming VJP9.3132e-8,101 parameter tensors
1.2346e-5. All pass `atol=5e-6, rtol=5e-4` and matching None semantics.
The parameter maximum is not an absolute5e-6 bound: the combined relative and
absolute acceptance tolerance is used. Complete-state forward AD also succeeds.
The identity process's sampled dedicated peak is3458.65MiB; both acceptance
and warm updates remain below3900MiB and the4096MiB hardware capacity.

Deferred weight credit is mathematically exact after its required pre-clip flush
and passes CPU/AOT/GPU, partial cadence and resume tests. Its observed0.1% speed
gain is unresolved, so it remains optional and disabled in the selected runner.
Exact history CPU offload saves roughly300MiB but adds transfer time; disabled.
The native-port deferred variant loses compiler fusion and is slower; rejected.
Unsafe repeated use of compiled backward workspace produced a boundary-gradient
NaN in an earlier candidate; that execution is rejected and was never deployed.

## Continuation

65 broad CPU checks pass (5 optional CUDA skipped), the final affected suite has
33 passes /3 optional skips, fused AdamW CUDA parity passes, and standalone
Deslice checks pass. Complete CLI continuation validation restores step305,
cursor9761, validation cursor256, next evaluation10000, update317, pending0,
consumes zero targets and writes no production files. Physical dimensions,
intrinsic time, all integration steps, full BPTT32, structural sample/evidence,
loss, learning rate, evaluation and persistent state remain the same.

Curated accounting: `results/published/medium_speed_optimization_20261009.json`.
Continuation arguments: `results/published/medium_speed_resume_arguments_20261009.json`.
The same individual resumes with these validated arguments; live continuation evidence is stored in the curated accounting JSON. The HTML server retains the same run.

## Retained-workspace batching follow-up

A save-slot prefetch hint preserves private first-VJP ownership and exact
padded strides. A missing hint always falls back to an exact private copy;
no value or gradient survives an event. Full GPU11-state/101-parameter credit
identity passed. Unbounded prefetch failed the complete update at Adam moment
restoration; it was rejected. A32MiB bounded variant passed full identity,
five complete real OWT updates and13 affected numerical tests.

Matched uninstrumented timing-only processes then showed8.83185s/32 for lazy
copies versus8.99225s/32 for batched copies, excluding the first capture/compile
update. Sampled dedicated peaks3356.65MiB and3366.65MiB. Batched prefetch has
no throughput benefit, so its experimental helper and tests were removed from
the production code. The original detached pack hook and private-buffer
ownership remain. This avoids carrying an unused execution mechanism.

These later absolute times vary from the earlier9.40294s selected mean;
retain the original matched10.40592-to9.40294 comparison for the claimed
1.1067x acceleration. No language-capability claim follows from these timings.

Live continuation verified past step315 /10080fresh tokens. At10016fresh tokens, full active B and A revisit complete without reset; B NLL6.93109 versus same-text prior7.56036. Full last checkpoint saved after the evaluation. Subsequent training continues, HTML API exposes current progress and spatial state. This one live event is not a matched architecture superiority claim.
