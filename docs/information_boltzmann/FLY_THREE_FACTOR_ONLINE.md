# One continuous fly learner

`train_fly_infinite_stream.py` uses `core/fly_online_learning.py` for training,
fresh active validation and A-to-B-to-A replay. The retired asynchronous entry
point is retained under `scripts/ib/legacy/` for provenance, not as the current run.

The implementation preserves COBA, ALIF, STP and the connectome. Sensory current
eligibility differentiates the old adaptation baseline, post-spike reset and
slow threshold state. Gate weights and biases have historical eligibility too.
The instantaneous readout/RMSNorm/decoder derivative matches the forward epsilon.
Recurrent plasticity remains **factorized e-prop with low-rank direct feedback**;
it is an approximation of long-range learning signals. Conditional local
sensitivity holds arriving recurrent pulses fixed. It is not whole-brain BPTT.

Dopamine occupancy integrates dq/dt = k_on D (1-q) - k_off q analytically,
using delayed STP-transmitted DAN pulses. Its rate defaults are engineering
initial values in model-event units. There is no unmeasured graded-release term.
Every token contributes to accumulated synaptic updates. Signs are encoded by
separate excitatory/inhibitory edges; plastic magnitudes stay nonnegative.

Eight existing physical parameters receive conditional local sensitivity:
threshold, membrane/synaptic timescales, ALIF strength/timescale, and E/I gain.
STP parameters remain fixed in this run, while STP state continues evolving.
This scope is recorded explicitly in config.json.

## Continuing evaluation and checkpoints

Predictions are scored before targets update parameters. All model, optimizer,
physical and eligibility states continue across fresh B and replay A. Every
stream bridge is scored. Cursor exhaustion fails rather than looping held-out
traffic. Fresh B and replay curves are separate in lifelong_evaluation.jsonl.

The fourth pillar records an algebraically closed squared-voltage energy
ledger, centered latent rank/temporal roughness, and a finite-difference
conditional full-physical-state gain (including the writer baseline). These
are model-unit diagnostics, not ATP energy, Shannon entropy export or proof
of biological criticality. FTLE conditions on observed inputs and the actual
current parameter trajectory; it excludes derivatives of the learning update.

`last.pt` restores the complete individual: learned edge magnitudes, random
feedback buffers, all physical/eligibility states, pending gradient groups,
Adam moments, RNGs and cursors. `best.pt` is explicitly a **weights-only branch
asset**. Use --resume only with last.pt; --initialize-from starts a declared
new individual from a weights asset. Static graph arrays reload from the
configured graph. Saves use a temporary file then atomic replacement.

Runtime calibration on real OWT verifies interface, finite updates, memory
and speed only. The 100,000-token run supplies the learning budget; short
calibration scores provide no architecture/capacity conclusion.

## Runtime repair, 2026-10-04

Float32 CUDA kernels fuse delayed-edge credit, projection and gate traces,
projection updates, and the eight physical-parameter sensitivity recurrences.
Node feedback factors are formed once before edge traversal. AdamW uses its
fused CUDA implementation; legacy CPU step counters migrate with saved values
when resuming. CUDA/reference and local-autograd checks are included in the
25 passing affected tests.

An eight-token production-eager checkpoint fork measured 34.08 ms/token,
compared with approximately 51 ms/token before this repair. Component times
are nested inclusive CUDA events, saved in `component_profile.json`. The
100,000-token run resumed its complete state at token 4392 and passed the
token-5000 active evaluation and atomic checkpoint save. Its initial resumed
throughput was 28.4 tokens/s, with approximately 2.1 GiB total GPU usage.

Unused smoke and explicitly retired weights were batch-deleted while keeping
configuration and metric/report files. The deletion inventory is retained in
`results/checkpoint_cleanup_20261004.json`.
