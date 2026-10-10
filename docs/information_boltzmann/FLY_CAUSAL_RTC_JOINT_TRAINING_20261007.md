# Exact physical-response joint training

## Task and falsifiable claim

Use the current sensory token and full continuing physical state to forecast
motor responses before delayed pulses arrive in the actual body. Test whether
those responses improve real next-token predictive NLL at matched fresh data,
initialization and optimizer updates. Numerical forecast accuracy by itself is
not a task improvement.

## Training interface

Each real input advances the COBA/ALIF/STP body once. A separate differentiable
zero-drive branch produces native motor read features at horizons 0 through 14.
A content-dependent attention mixes those features, followed by the existing
RMSNorm and shared vocabulary decoder. Only next-token CE supervises this arm;
there is no detached approximate teacher, future-token input or Gamma trace.
Sensory writer, live E/I weights, physiological parameters, horizon readout and
decoder train together. The branch uses existing Triton transmission and
activation checkpointing. Actual BPTT is 32 events; branch differentiation is
finite, not unlimited online credit.

The branch specifies quiet future boundary conditions, so it predicts a material
response rather than unknowable future sensory input. Its terminal state is
discarded. Actual physical state, optimizer and exposure counters continue.
The independent CPU causal-domain forecaster verifies the physical equations;
its detached sparse operator is not used for joint training.

## Competing explanations and registered predictions

1. Delayed access: a current token affects future motor responses, whereas the
   instantaneous motor state may omit it. Useful look-ahead predicts lower NLL
   than the matched horizon-0 arm.
2. Missing upstream state: approximate observers failed because their compressed
   state lacked future incoming flux. Complete physical response removes that
   numerical defect, but its task usefulness still needs the matched comparison.
3. Optimization/task mismatch: exact response can remain poor for language. If
   numerical checks pass and the adequate joint budget has no predictive gain,
   exact physical forecast alone is insufficient for this task at this budget.

No effect-size or convergence guarantee is asserted in advance.

## Resource gate and execution

First compare checkpointed and plain CUDA gradients at two real tokens and
horizon 2. Threshold-adjacent atomic-reduction differences must remain within
the declared numerical check. Then run one complete 32-token update at H14.
Dedicated CUDA peak allocation/reservation must stay below 3900 MiB and losses,
gradients and updated parameters must remain finite.

If the resource gate passes, continue the calibrated individual for 100,000
additional fresh OWT tokens (3125 fresh optimizer updates). Run the same budget
on H0 sequentially on the single GPU. Both initialize GPT-2 input/decoder weights,
train-only add-one unigram bias, read gain 0.1, identical horizon-head parameters
and seed 11. Shared brain/writer LR is 2e-4; pretrained decoder LR is 2e-5.
Both explicitly train STP coefficients. These initialization/learning choices
are fixed across arms and cannot be credited to the response horizon.

Fresh active validation every approximately 5000 training tokens uses the
actual learner, scoring before update. A-B-A replay is separate from first-pass
traffic. State, optimizer, clocks and dataset cursors never reset or wrap.
Best/last are the only checkpoints; provenance and full-life continuation are
validated. Report actual events and speculative ticks separately.

H14 entails 15 physical steps per event plus branch recomputation in backward.
This establishes a faithful trainable reference, not a cheap learned apprentice
or nonblocking wall-clock runtime. Measure throughput before paying the full
budget. Halt on nonfinite results or memory-budget failure, preserve diagnostics,
and fix the execution issue without reinterpreting it as an architecture result.

## Review

Independent reviewer `/root/rtc_contract_review` approved the conditional GPU
calibration and sequential adequate-budget comparison after CPU gradient,
clock and resume checks. It requested the additional CUDA parity check and
explicit STP coefficient coverage. The runner's backend/metadata resume bug
was fixed and covered by five CLI configuration tests.
