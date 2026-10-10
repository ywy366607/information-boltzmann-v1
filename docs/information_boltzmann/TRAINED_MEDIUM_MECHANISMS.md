# Trained conductance medium: mechanism audit at update 703

Date: 2026-10-04. The user requested a mechanism check rather than completion
of the planned training. Training stopped cleanly at update 703, with the full
optimizer and persistent belief saved in `last.pt` (89,984 OWT tokens).
The independent fly job was left running.

## Evidence and protocol

The audited model is the 8x8x4/D128 learned conductance medium with 14,357,682
parameters, event duration 0.005 and one solver substep. The audit uses real
held-out OWT at offsets 8192, 12288, 16384 and 20480. Each window starts from
a clone of the complete mature training belief, warms on 256 observed tokens,
and scores 128 tokens. No zero-state reset or additional optimization occurs.
128 scored tokens are traced through the actual one-substep composition.
The published JSON records checkpoint SHA256 and all measurements.

## What has differentiated

| Quantity | Measurement | Interpretation |
| --- | --- | --- |
| Field spatial energy share | 94.91% at checkpoint; trace mean 93.83% | Spatial structure survives to readout |
| Spatial power participation rank | 4.52 at checkpoint; 4.54–5.30 across windows | Structure is concentrated in a small set of patterns |
| Top eight spatial singular modes | 96.08% of checkpoint spatial power | Rich field variation does not yet imply broad use of capacity |
| Local C, leak, maximum E/I conductance, closing rate | Spatial relative standard deviation 2.93%–3.26% | Learned material coefficients differ by location |
| Effective propagation speed | Spatial variation 5.72% | Functional transmission is spatially heterogeneous |
| Effective membrane time | Spatial variation 3.60% | Local persistence has differentiated |
| Content metric | Spatial variation 3.15% | Position-dependent content preferences are represented |
| Read heads | Mean pairwise attention TV 0.379 | Heads sample different spatial distributions |
| Read attention entropy | 4.86–4.91 versus uniform 5.55 | Sampling is differentiated but remains broad |
| Read coordinates | Mean periodic displacement 0.00392 | Most differentiation occurs through weights on near-initial locations |

Spatial relative standard deviation is standard deviation over locations divided
by spatial RMS, averaged over channels. Participation rank is
`(sum sigma^2)^2 / sum sigma^4`, after spatial-mean removal; it is not algebraic
rank. These establish learned heterogeneity, not a claim of semantic brain
regions or sustained phase locking.

## Operator activity and energy accounting

| Stage | Mean field change / incoming field norm | Additional evidence |
| --- | --- | --- |
| Boundary write | Direct old-field amplitude retention 55.53% | Incident transmitted fraction 64.99%; mean angle 0.944 rad |
| Transport | 6.92% | Mean edge angle 0.10098 rad |
| Collision | 0.288% | Mean angle 0.001458 rad |
| Electrical response | 0.993% change in field plus edge responses | Persistent receptors stay within [0.444, 0.561] |

Maximum traced energy-ledger residuals in FP32 are 2.24e-8 for boundary
exchange, 2.98e-8 for transport/collision, and 1.19e-7 for electrical response.
Mean reversal source work is 0.64261 and Joule heat 0.64481 per event; their
net removes 0.00219 energy. Large opposing gross terms alone do not establish
noise or instability. This verifies the observed trajectory's energy account;
the separate feasibility documents state the conditional long-run bounds.

Direct next-token CE gradients reach material, transport, collision, electrical
response, boundary and readout. The local pathway metric's group gradient norm
is only 1.71e-7 on the inspected eight-token context. This is a weak local
sensitivity, not a disconnected computation graph. Raw group norms have
different parameter counts and units and are not normalized learning-efficiency
comparisons.

## Current inference reliance

Mean full validation NLL is 7.51761. Removing transport changes NLL by
-0.00233; removing collision by -0.00467; removing electrical response by
-0.00587. Each intervention has mixed signs across sites. The positive
language benefit of these operators is therefore weak at this checkpoint,
even though transport really moves state and all learning paths exist.
Removing response disables receptor evolution, reversal work and heat together;
it is not an isolated dissipation intervention. This early audit diagnoses
the checkpoint and does not rank separately trained architectures.

## The concrete scale mismatch to resolve next

The present W4 boundary is

```
theta_a = atan(admittance_a * ||innovation||_precision)
f_plus(x,a) = cos(theta_a) * f(x,a) + injected_innovation(x,a)
```

Its old-field multiplier is channel-specific but shared across every location.
Measured direct amplitude retention is 0.5553 and old-field-weighted energy
retention 0.3413 per event. This freezes the boundary action and separates the
old-field term from injection; it is not the full state-dependent Jacobian or
a memory-lifetime bound. Edge responses retain additional history, while the
current readout observes only the field.

Meanwhile conservative collision uses `angle = duration * learned_rate`; at
duration 0.005 it changes field by only 0.288% per event. This is the clearest
measured mismatch: strong global field refresh precedes weak internal
reorganization. It supports a specific next design question: align selective
boundary retention with the internal interaction and storage timescales.

Express those timescales dimensionlessly before selecting a new cadence:
boundary retention hazard `-log(cos(theta))`, transport angle, collision
`duration * rate`, and electrical `duration / tau`. Check selective preservation
of old spatial modes and their path into readout. Preserve separate write,
linear transport, nonlinear collision and release roles. Avoid choosing an
arbitrary larger timestep solely to force an ablation effect. Stronger internal
processing should improve observed prediction and useful retained information.

## Reproduction

```powershell
python scripts/ib/audit_trained_plastic_medium.py --checkpoint results/plastic_conductance_d128_3000/last.pt --output results/published/plastic_conductance_mechanism_703.json --trace-tokens 32
```

The audit runs on CPU and leaves model parameters and the saved training state
unchanged. Training remains stopped pending the next requested design change.
