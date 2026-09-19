# Collision evaluation repair — 4250

The previous message-coupling mechanism evaluation used an eager forward path
that skipped the coupling module. Its mechanism conclusions are superseded.

Corrected frozen checkpoint evaluation: four held-out OWT sites, 256-token
warmup each, 128 scored tokens each, common text, proposals and thermal noise.
Full NLL 7.600202729; no-collision NLL 7.599488467; difference -0.000714262.
Site differences: -0.000997102, +0.000464955, -0.000413919, -0.001910985.
There is no net collision benefit on these 512 scored tokens. This does not
establish convergence or invalidate the collision architecture.

## Implemented repairs

- Eager forward now calls adaptive force / message coupling and passes gamma.
- Decoupled clipping applies with block graphs too; score scale defaults to 1,
  without the hidden 0.008 override. Explicit scale applies after score clipping.
- Path and score graph calls use the current token_ids interface.
- Coupling-aware evaluation rejects legacy no-drive/no-transport/fixed-gamma
  interventions until their semantics are implemented correctly.
- Two deterministic route equivalence tests passed. CUDA graph path and score
  backward smoke test passed, with no optimizer update or training run.

## Next architecture specification (not implemented)

For equal-mass pairs let u=(v_i+v_j)/2 and r=(v_i-v_j)/2. Learn an orthogonal
rotation R conditioned on both particle encodings, relative position/velocity,
and local environment. Set v_i'=u+Rr and v_j'=u-Rr. This preserves pair momentum
and kinetic energy. Learn rotation planes and angles; impulse magnitude follows
from the angle and relative speed, rather than an unconstrained extra multiplier.
Use identity-near, non-degenerate initialization and continuous CE gradients.

Replace uniform feature averaging in a separately trainable architecture version
with a small set of learned read queries attending to particle keys and values.
Concatenate query outputs before the vocabulary head. Initially queries are
learned constants so the readout has no direct token-to-logit bypass. Preserve
particle permutation invariance while including spatial features in keys/values.

Collision queries are local and pair-conditioned; prediction queries retrieve
from the whole state. Share particle feature encoding if useful, but keep query
and output heads distinct. Do not swap this readout into the frozen checkpoint
and interpret the resulting loss as an ablation.
