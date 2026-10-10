# HX-1.1 quiet-rollout proposal: source review

Status: code inspection, provisional mechanism interpretation, independent
review pending. No new training or model edits performed for this review.

## Useful structural change

Supervising fourteen counterfactual quiet physical horizons creates a concrete
multi-horizon surrogate-learning problem. Unlike the streaming one-step arm,
the decoder can use multiple inexpensive latent transition steps immediately.
Shared decoder normalization and explicit continuing prior are useful contracts.
Learning both encoding and prediction without stop-gradient is a viable design
family; its non-collapse requirements still need to match the actual objective.

## What the current source implements

- MacroRegionGraph ignores graph_npz_path and registers a literal four-by-four
  adjacency. It neither reconstructs actual E/I connectivity nor delay tiers.
- Regional encoders observe sampled membrane h, while the physical teacher also
  depends on conductances, adaptation, STP and in-flight pulses. Exact reduced
  transition closure is not established.
- forward_window collects fourteen quiet ticks from the window-final physical
  state once per learning window. OPD compares only hop[-1] to this teacher.
  All-token latent rollouts are computed, but all-token fourteen-horizon teacher
  supervision is not implemented.
- Both teacher encoding/physical dynamics and student receive consistency-loss
  gradients. SIGReg is applied to attended z_readout, not to Z_star or the
  encoded quiet targets whose agreement is optimized.
- Quiet teacher frames are counterfactual future states under zero external
  input, not future states of the continuing real text stream. Their useful
  predictive interpretation must be made explicit.

## Evidence scope

scripts/ib/test_opd_delay_alignment.py constructs a synthetic100-neuron graph,
with specified edge delays1+1+2. It measures a motor-norm response peak, not
mutual information, and not a MaleCNS latency. Response peak and earliest
arrival are distinct quantities.

Its fifteen optimization iterations repeat four token targets with Adam0.01.
The returned physical next_state is not assigned to learner.state in that loop.
The terminal ALL CHECKS PASSED message is unconditional: no assert establishes
the listed convergence, delay-attention or continuity claims. This can be an
illustrative numerical exercise after adding explicit checks; it is not an
approved real-data capability study or evidence of a continuing individual's
language improvement. Existing separate unit tests retain their actual scope.

## Mathematical risks and useful corrections

A jointly optimized consistency loss ||P(E(s))-E(F(s))||^2 can decrease through
improved prediction, changed encoding or changed physical target. Regularizing
only a downstream attended output does not directly establish non-collapse of
every regional target embedding. If borrowing LeJEPA's no-stop-gradient design,
place the non-collapse constraint on the embeddings entering that predictive
objective and verify its sample/distribution assumptions. CE is an additional
task constraint, not a proof that every regional embedding retains information.

For uniform RMSNorm gain0.1 and input RMS well above epsilon, the output norm
is approximately0.1*sqrt(d):2.77 at768dimensions and0.8 at64dimensions. This is
an initialization/calibration choice, not a universal pretrained GPT-2 feature
contract, and not a general norm guarantee0.14..0.20.

Standard language-model on-policy distillation supervises student-generated
sequences using teacher feedback. This code instead implements joint latent
consistency on quiet trajectories rooted in the physical individual's current
window endpoint. The latter can be useful under its own declared objective;
its name should not imply the former's established training guarantees.

## Decision

Retain the multi-horizon surrogate direction as a candidate. Before a formal
capability claim, make actual graph provenance, quiet-vs-stream semantics,
state sufficiency and encoder-level anti-collapse explicit. Verify timestamped
causal forecasts and complete train/inference continuation, then use matched
real OWT joint training to decide task benefit. No toy CE claim substitutes
for that decision. No experiment is launched by this review.

Primary external sources:
- https://arxiv.org/abs/2511.08544
- https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
- https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf
