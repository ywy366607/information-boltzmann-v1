# Local persistent credit in the 3D medium

The forward model and trainable parameter set are unchanged. This first
operator-local migration replaces random global compression with deterministic
receptor eligibility. It adds long-history credit to closing kinetics and their
continuous spatial material map, while all operators retain exact current-event
CE + W4 gradients. Fly code and fly training are outside this change.

## Derivation and scope

At each site, content channel and E/I branch, the implemented receptor flow is

\[
r=a+b,\quad p=a/r,\quad g^+=e^{-hr}g+(1-e^{-hr})p.
\]

The eligibility decay is therefore the existing physical decay `exp(-h*r)`.
There is no chosen credit-window length or separate trace time constant.
Writing `u=1-exp(-h*r)` and `c=-h*exp(-h*r)*(g-p)`, the partials are

\[
\partial_a g^+=c+ub/r^2,\quad
\partial_b g^+=c-ua/r^2.
\]

The native activity feedback also changes opening_E through gate_I and through
its closing-dependent inhibitory baseline. Both derivatives are included.
For a two-branch receptor subsystem, let `J` and `B` be those local 2x2
Jacobians. The persistent sensitivity to physical closing rates is

\[
E^+=JE+B.
\]

Both kinetic halves of each actual conductance step update this same trace.
Independent locations/channels remain separate; no random signs, feedback
matrix, Gaussian perturbation, normalization threshold or added biological
parameter enters this update. The FP32 trace uses 0.5 MiB at 8x8x4/d128.

Voltage and flux trajectories are external inputs **for this local eligibility
recursion**. Their influence on the current event remains fully differentiable.
The trace consequently omits historical voltage/flux feedback into receptor
opening, cross-site recurrent sensitivities, and history in the write,
collision, transport, STP, structural conduction and precision states. Those
operators still receive their exact current-event gradient. This is a local
e-prop approximation with an explicit scope, rather than exact full RTRL.

For incoming gate state `g_t`, the current event computes the real learning
signal `L_t=partial(loss_t)/partial(g_t)`. Historical closing-rate feedback is
`L_t E_t`, pulled through the existing closing-rate/material network. The direct
event gradient already includes this event's transition and both kinetics
halves, so history uses the **pre-event** trace. No double-counting occurs.
Updating shared network weights uses the current closing-field parameter map;
it does not differentiate through previous optimizer steps or past material
maps. This is the usual online local-coefficient interpretation.

The physical belief, trace and counters persist through optimizer updates.
Continuation checkpoints include all three as well as the optimizer and stream
offset. Primary evaluation now continues active learning through actual A/B/A
experience; see `LIFELONG_EVALUATION.md`. Its ledger also saves pending gradients.
Capture warmup restores physical/learning state; it adds no experienced events.

## Verification on 2026-10-04

- 39 affected CPU checks passed; 6 optional CUDA checks skipped in that run.
  The local test set separately passed all 7 checks with CUDA enabled.
- Actual gate and closing partials, with and without inhibition adaptation,
  match autodiff. A 137-step conditional receptor trajectory matches its full
  closing derivative to FP64 tolerances. This is a derivative test.
- First-event physical state, loss and every parameter gradient match the
  original model. Optimizer-boundary persistence, deterministic continuation,
  error rollback and CUDA replay after weight updates pass.
- Real OWT, actual 8x8x4/d128 model, FP64 two-event audit: closing-bias historical
  gradient norm `2.9132e-6` versus local `2.3000e-6`; cosine `0.9999306`, relative
  error `0.2107601`. This quantifies the omitted paths in that trajectory;
  conditional exactness is distinct from full-system gradient accuracy.
- Three joint execution updates / 384 OWT events, CUDA Graph: update times
  `2.8113, 2.7945, 2.8122` seconds per 128 tokens (mean `2.8060`). Trace 0.5 MiB,
  retained allocated memory 261.37 MiB, peak allocated memory 315.74 MiB, total
  dedicated GPU use 661 MiB. All gradients and traces remain finite.

This establishes a usable local online-learning implementation. Language
improvement and lifelong global credit remain open; 384 events are execution
calibration, not a capability study. UORO's earlier 14.78 s measurement covers
all state sensitivities and had different concurrent GPU load, so the two
timings compare different credit scopes as well as implementations.

## Run

```powershell
python scripts/ib/train_online_plastic.py --credit local-receptors `
  --output results/<new-run> --event-duration 0.005 --execution graph
```

The duration is a saved physical cadence, not an eligibility horizon. Preserve
the source checkpoint's duration and solver resolution when using
`--initialize-from`; use `--resume` for exact continuation. `--tokens` controls
optimizer cadence. The reference UORO path remains selectable with
`--credit uoro`. No long local-credit run has been started by this change.

Numerical audit:

```powershell
python scripts/ib/audit_local_medium_credit.py `
  --output results/published/plastic_local_credit_derivative.json
```

Reference: [Bellec et al., e-prop](https://www.nature.com/articles/s41467-020-17236-y)
motivates the eligibility/learning-signal factorization. The specific partials
above are derived from this repository's receptor equations.
