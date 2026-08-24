# Active-Inference GDN-2 for Causal Slice Evolution

Status: **implemented opt-in candidate; v29 passes sign-level causal gates but
is not yet a mature world model**. It extends the North-Star graph and does not
add a second generator, video backbone, or task-private head.

## State and Variational Update

The persistent physical belief is a factorized Gaussian fast-weight field on
one fixed Eulerian atlas:

\[
q(M_t)=\mathcal N(\eta_t/\Lambda_t,\Lambda_t^{-1}).
\]

Process variance releases obsolete precision,
\(\Lambda^-=(\Lambda^{-1}+Q)^{-1}\), while observed Slice likelihoods add
precision and natural parameters. This gives the VFE/Bayes interpretation of
GDN-2 erase/write: erase is uncertainty growth; write is precision-weighted
evidence accumulation. Suppressing normalization and learning fixed gates
recovers the gated delta-rule degeneration.

## Closed Physical Coordinates

Transient content-dependent Slice slots remain the layer workspace and are
never treated as persistent identities. Causal memory instead uses a fixed
physical Slice/Deslice pair over the full-resolution content field:

1. Remove the immobile coordinate basis with
   `content = encode(image) - encode(black_at_same_xy)`.
2. Pool content into the fixed atlas, update Gaussian natural parameters, and
   infer velocity from consecutive physical observations.
3. Query source addresses at `destination - horizon * (velocity + action)`.
4. Scatter the prior-minus-posterior content back to full-resolution `X`.

The atlas already lives in `X` content coordinates, so no dense learned basis
change is allowed. A per-channel trust gate controls the residual. Library
default `initial_prior_trust=0` preserves old checkpoints exactly; the audited
physical warm start is `0.1`. Experimental history predict-before-correct
transport remains off because v21 regressed.

## Separation of Prior Factors

Semantic generation and physical transition share the same `X–Slice–H` graph
but are different prior factors. Text-to-image and editing continue through
the proven F2 language prior and ordinary Slice/Deslice write. GDN-2 acts only
when `target_time > 0`; it cannot alter generation, reconstruction, or editing
at zero horizon. A vacuous `"Predict next frame"` label is not world evidence:
future prediction is defined by history, horizon, and action unless a prompt
contains genuine semantic information.

## Evidence and Admission

Run the registered candidate with:

```powershell
python scripts/train_northstar_capabilities.py --device cpu `
  --temporal-only --active-gdn2 --active-gdn2-initial-trust 0.1 `
  --no-future-language-evidence --steps 50 --eval-every 50
```

v29 preserves generation/current/edit scores `0.891/0.764/0.901`. On the
90-sample future bank, removing causal memory reduces paired IoU by `0.00040`
and segmentation IoU by `0.00073`; history gains are `0.02172/0.03085`, and
action gains are `0.01941/0.00134`. These are positive mechanism results, not
an accuracy-complete claim: the action segmentation effect is still below the
registered `0.005` robustness target and only one seed has run.

Implementation: `fine_grain/active_gdn2.py`, integration:
`fine_grain/native_mot.py`, VFE: `fine_grain/omni_model.py`, tests:
`tests/test_active_gdn2.py`, result:
`results/published/northstar_capability_active_gdn2_v29_trust01.json`.
