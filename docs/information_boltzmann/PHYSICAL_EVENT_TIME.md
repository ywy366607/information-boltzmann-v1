# Physical event time and recurrent gradient repair

## Diagnosis, 2026-10-02

The W4 quadratic run at K=16, dt=4 advanced the interior by T=64 per
external token. Its field energy remained near 0.12, while logged raw
gradients reached 3.93e10. The update680 run was stopped; checkpoint500 and
all metrics were retained.

At checkpoint500, a 32-token real OWT backward audit gave gradient norm
1.45249e8. Backward-only isolation of the circular-address feedback gave
3206.80; isolation of the state-to-collision-angle feedback gave 229.62;
isolating both gave 33.25. These diagnostic interventions leave the forward
unchanged. They are not used as a training rule.

Native collision also produced amplification (4.41298e7). The unstable
32-token rollout amplified FP32 execution differences into a 0.01089 loss
difference; kernel equivalence must be checked in a well-conditioned regime.

## Why conserved energy does not bound the backward flow

Write the collision, on its nullspace, as C(f)=R(theta(f))f. Its differential
is

    dC = R(theta) df + [dR/dtheta * Dtheta(f) df] f.

Orthogonality controls the first term. The second term is a state-dependent
feedback shear. The temporal adjoint multiplies these Jacobians throughout
each BPTT window. A norm-preserving state map can therefore have expanding
perturbations. Address feedback adds another state-dependent shear.

The quadratic bath is locally B(f)=exp(-a ||f||²)f, with a>=0. Its tangential
Jacobian eigenvalue is exp(-z), and its radial eigenvalue is
exp(-z)(1-2z), z=a||f||². Both have magnitude at most one. This bath controls
amplitude smoothly; replacing it with strong spectral viscosity would also
erase the spatial structure needed by the model.

## Repair

Physical duration T and numerical resolution K are declared separately:

    dt = T/K,
    f(t+T) approximately equals the K-step split evolution of f(t).

The new W4 trial uses T=3 reference units, matching the historical K3
champion's baseline scale, and K=16 (dt=0.1875). Three is a reproducible
reference duration, not a universal stability or criticality constant. All
write, transport, collision, read and bath gradients remain active.

The model's fixed-duration API also rescales dt when evaluation overrides K.
Increasing integration resolution therefore preserves the event duration.
Explicit legacy per-step dt remains available for archived-run reproduction.

With the same checkpoint500 and the same first 32 tokens, T=3 gives raw
gradient norm25.4163. Four consecutive chunks, with the field and precision
carried forward, give 25.42, 45.00, 184.39 and 2777.08. This isolates the
large duration as a major amplifier and also exposes remaining variation.
The fresh joint trial records raw and measured post-clipping gradient norms
throughout training; these diagnostic measurements are not language results.

## Reproduction and evidence

Run configuration: `configs/information_boltzmann/q8_w4_quadratic_t3_k16.json`.
Training output: `results/q8_w4_quadratic_t3_k16_3000/`.
3000 optimizer updates, 128 tokens/update, BPTT32, 8x8x4/d128, seed11;
periodic warm-local independent validation every500 updates. One field and
posterior precision persist across all training chunks.

Numerical audit artifacts:

- `results/published/quad_bath_gradient_feedback_500.json`
- `results/published/quad_bath_gradient_feedback_native_500.json`
- `results/published/quad_bath_gradient_duration3_multiwindow_500.json`
- `results/published/quad_bath_gradient_history.json`

Tests cover duration/resolution separation, legacy timing, full gradients and
Cayley refinement at fixed physical duration. The complete suite reports
668 passes, one skip, and one existing failure from an archived checkpoint
fixture; the Deslice/gate script also passes.
