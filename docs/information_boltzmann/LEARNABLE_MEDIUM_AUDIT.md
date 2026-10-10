# Learnable medium: fixed priors versus adaptable structure

Date: 2026-10-03. Scope: the independent conductance-based plastic-medium
candidate, not the running fly model or the historical quadratic checkpoint.

## Design objective

Keep equations for storage, reciprocal transmission, local interaction and
release as the admissible family. Learn the material coefficients and content
preferences for the task, rather than transplanting fly-specific physiological
fits. A unit choice, an initialization prior, a mathematical constraint and a
fixed representational budget have different roles and must remain explicit.

| Quantity | Current implementation | Role |
| --- | --- | --- |
| C, L, leak, E/I conductances, resistance | Positive material-conditioned learned fields | Local persistence, gain and loss |
| Receptor opening/closing | Opening reads activity/current/material; closing learns independently | Persistent response with learned relaxation |
| Reversal gaps | Learned positive/negative branches relative to resting gauge | Source work and opposing responses; no fixed 5:1 ratio |
| Propagation | Learned baseline, gain, rate and persistent edge log-speed | Activity-dependent fast pathways |
| Conduction content preference | Now conditioned on local material as well as axis | Different places can favor different content |
| Collision | Shared nonlinear actor reads local activity, material and receptor state | Local specialization and conservative content mixing |
| Readout | Learned coordinates, probes and score scales | Adaptable sampling locations and processing |
| T0, V0, C0 | Declared reference units, defaults one | Nondimensionalization; no implied milliseconds |
| Grid, basis bandwidth, local adjacency, widths/layers | Explicit fixed capacity budgets | Defines which spatial structures the current model can represent |
| Boundary packet channel scale | Learned per-channel parameter, initialized as packet_radius*sqrt(D/64), default radius1.25 | Input-amplitude initialization prior; admittance and precision also learn |
| Solver step/substeps | Caller-selected numerical resolution | Refines physical time, independent of cognitive policy |
| BPTT chunk | Training memory/credit window | Detach retains all forward persistent state |

There is no degree-to-3--78ms mapping, prescribed receptor/membrane time ratio,
global spike threshold0.1, Heaviside/reset surrogate slope4, forced spectral
radius1 or four-slot transmission queue in this candidate. Its physical times
come from tau_mem=C/(gL+gE+gI), tau_receptor=1/(alpha+beta), tau_edge=L/R.

## Two concrete parameterization repairs

### Direct material gradient from electrical coefficients

Previously a(x)=0 and W=0 in log(c(x))=W*a(x)+b. Therefore the isolated
electrical coefficient path had dLoss/da=W^T*dLoss/dlog(c)=0 initially.
Collision and other actors still had alternate material gradients; the defect
was specifically this direct coefficient path.

Retain standard Linear fan-in initialization for W and zero b. Because a=0,
initial coefficients stay exactly at the same uniform reference values, while
their material derivative is immediately available. No extra operator or
parameter is introduced. A deterministic spatial-gradient test isolates this
path from collision, opening actor and routing.

### Local content preference for structural adaptation

The old axis-shared metric is retained for the quadratic control. The
conductance candidate uses, on edge e with axis alpha,

```
M_e = diag(softmax(l_alpha + W_alpha*a(x_e)))
s_e = (2<f_i,f_j>_M + ||j_e||_M^2)
      / (||f_i||_M^2 + ||f_j||_M^2 + ||j_e||_M^2 + eps)
tau_e*dp_e/dt = A_e*s_e-p_e
c_e = c0_e*exp(p_e)
```

The same positive metric weights both endpoints, so Cauchy-Schwarz still gives
|s_e|<=1. For fixed finite learned parameters, p stays in its analytic invariant
interval; propagation remains positive and bounded. At zero initial material
the preference stays uniform. Different learned local material can subsequently
prefer opposing content at different edges.

At D128/material width8 this adds3072 parameters, no persistent state and no
extra time loop. Total candidate parameters:14357682. The previous execution
benchmark measured14354610 parameters; preserve that timing provenance.

## What spatial evolution means here

The periodic grid provides a local substrate. Learning material changes effective
impedance, response, propagation and content preference; runtime conduction state
also adapts without an optimizer. This permits functional fast routes and weakly
coupled regions on that substrate. The current six-neighbor adjacency does not
add arbitrary direct long-range edges. A larger Fourier material basis/grid can
resolve thinner routes. Changing grid resolution with fixed weights samples the
same material bandwidth rather than automatically creating new degrees of freedom.

Task learning changes shared rule parameters and material. Runtime structural
adaptation changes persistent conduction state; it is distinct from autonomous
optimizer-based permanent learning. The implementation keeps this distinction.

## Verification

66 affected CPU checks pass (one separate CUDA-opt-in check skipped); five
explicit CUDA checks pass. They cover material gradients, local metric preference,
likelihood gradients, conservation, analytic bounds, causality, complete
continuation and captured training. These certify admissibility and implementation;
task improvement remains a real-data joint-training question.

Key new tests:

- test_uniform_initial_coefficients_have_an_active_spatial_material_gradient
- test_local_material_can_choose_different_content_on_each_edge
- test_local_metric_receives_likelihood_gradients_and_shared_control_stays_available

See core/conductance_response.py, core/conduction_plasticity.py and
core/plastic_medium.py. No capability training is launched by this audit.
