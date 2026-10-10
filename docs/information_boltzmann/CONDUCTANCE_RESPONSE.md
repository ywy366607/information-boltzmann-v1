# Local conductance response: equation provenance and implementation

Activate with `PlasticMediumPorts3D(bath_type="conductance")`. This candidate
replaces the instantaneous quadratic release network with persistent E/I receptor
fractions, capacitive response, reversal-source work and resistive heat. Transport,
conservative collision, conduction plasticity, W4 writing and learned reading
remain on one persistent individual. The identifier is
`PlasticMedium3D-W4-learned-read-conductance-v3`.

The default `quadratic` remains the historical reproducible control. Existing
weights and the running fly job are unchanged; no new training is launched.

## Equation provenance and explicit modeling choices

* Hodgkin & Huxley (1952), *A quantitative description of membrane current and
  its application to conduction and excitation in nerve*,
  [doi:10.1113/jphysiol.1952.sp004764](https://doi.org/10.1113/jphysiol.1952.sp004764):
  capacitive current balance and conductance times driving potential.
* Destexhe, Mainen & Sejnowski (1994), *An Efficient Method for Computing Synaptic
  Conductances Based on a Kinetic Model of Receptor Binding*,
  [doi:10.1162/neco.1994.6.1.14](https://doi.org/10.1162/neco.1994.6.1.14):
  two-state receptor kinetics and analytic frozen-drive integration.
* Gernandt et al., *Port-Hamiltonian formulation of nonlinear electrical circuits*,
  [arXiv:2004.10821](https://arxiv.org/abs/2004.10821): separate storage,
  conservative interconnection, resistive loss and source work.

This is a conductance-based abstraction for information processing, not a literal
fly-voltage model or a reproduction of fitted squid Na/K channels. The two-state
approximation assumes one effective open fraction per branch and omits detailed
multi-state binding chemistry. The learned opening-rate actor replaces observed
transmitter drive with a content/current/material function. Its neural architecture
and the continuous coefficient basis are engineering choices; they are not
attributed to these papers. The energy identities below are derived for our code.

The fly builder's degree-to-3--78ms mapping, fixed `tau_syn=tau_m/4`,
transmitter-only current signs and soma-distance delay approximation are excluded
from this migration. Independent positive electrical and reaction parameters are
learned here.

## Continuous dynamics

Take resting potential as the voltage gauge, `E_L=0`. At every site/channel:

```
ds_r/dt = alpha_r(V,I,a)*(1-s_r) - beta_r(a)*s_r,   r in {E,I}
g_r = gbar_r(a)*s_r
C(a)*dV/dt = -g_L(a)*V + sum_r g_r*(E_r(a)-V) + I_conservative + I_boundary
L_e(a)*dI_e/dt = conservative_edge_drive - R_e(a)*I_e
```

C,L,gbar,g_L,R,beta are positive learned fields. E_E>0 and E_I<0 are independently
learned relative to the resting gauge. These are depolarizing/hyperpolarizing
reversal branches, not assigned semantic cell types. A branch's current reverses
when V crosses its reversal potential.

```
tau_receptor = 1/(alpha+beta)
tau_membrane = C/(g_L+g_E+g_I)
tau_edge = L/R
```

These scales follow material and activity. There is no imposed ratio between
receptor and membrane times. Opening reads local voltage, three current features
and material. Collision additionally reads the persisted s_E,s_I, so recent
response history can modulate conservative reorganization. Transport stays linear
for fixed material/conduction state.

## Units, initialization and parameter domain

`response_time_reference`, `voltage_reference`, `capacitance_reference` declare
T0,V0,C0. Dimensional units follow: g0=C0/T0, L0=T0^2/C0, R0=T0/C0, reaction
rate unit1/T0. Defaults of one are nondimensional reference units, not millisecond
calibration. Positive material coefficients use exponential log-parameterization.

Spatially uniform unit coefficients and matched initial E/I branches are an
exposed **trainable initialization prior**, not optimal physiological constants.
Opening/closing rates, C, conductances, L,R and reversal gaps all learn independently.
The material-to-log-coefficient map retains Linear's standard fan-in weight
initialization and zero bias. With zero initial material, actual coefficients
remain exactly uniform. Unlike zeroing BOTH material and this weight map,
this supplies a direct coefficient-path gradient to material from the first
backward pass. Other operators had separate material gradients already.
Changing electrical/time units at fixed energy-coordinate unit produces identical
normalized dynamics in the tests. Finite strictly positive coefficients are checked;
invalid numerical parameters fail instead of being silently clipped.

The conduction content metric is also material-conditioned for this candidate;
the historical quadratic control retains its shared-axis metric. See
`LEARNABLE_MEDIUM_AUDIT.md` for the complete learned/fixed distinction and tests.

## Storage and source accounting

The existing field and flux arrays store z=sqrt(C)*V and q_e=sqrt(L_e)*I_e:

```
H = 1/2 integral (|z|^2 + sum_e |q_e|^2) dx
P_reversal = sum_r g_r*E_r*(E_r-V)
Q_Joule = g_L*V^2 + sum_r g_r*(V-E_r)^2 + sum_e R_e*I_e^2 >= 0
dH/dt = P_reversal - Q_Joule
```

Existing transport and collision conserve H. In V/I coordinates they represent
reciprocal coupling with C/L normalization, not measured axonal morphology.
Learning updates change model parameters; a fixed-parameter physical rollout
does not hide a dC/dt term.

Activation can increase stored energy using counted reversal sources. Diagnostics
separate `response_source_work`, `response_joule_heat` and the integrated
`response_energy_residual`. This ledger covers electrical storage/work/heat;
receptor-binding chemistry and ATP regeneration are outside the modeled storage.
Reversal potentials are maintained environmental source ports.

For fixed finite learned material and s in[0,1], let

```
m = min_x,d {g_L/C, R_e/L_e} > 0
u_max = sum_r gbar_r*|E_r|/sqrt(C)
U2 = integral |u_max|^2 dx
```

Young's inequality yields `dH/dt <= -m*H + U2/(2*m) + P_boundary`.
If additional boundary power is bounded by P, the absorbing bound is
`P/m + U2/(2*m^2)`. This controls energy even with nonlinear content-driven
gates. It is a conditional long-time energy result, not a useful-memory or
criticality theorem. The earlier quadratic-source locking witness is not
automatically a certificate for this new active response.

The implemented frozen electrical step obeys the same integrated inequality:
`H_next <= exp(-m*dt)*H + B*(1-exp(-m*dt))`, where `B=U2/(2*m^2)`.
Gate updates change no electrical storage, and the other internal steps conserve
it. Therefore composing this actual splitting step preserves the energy bound
for every dt>=0, even though accuracy of the coupled trajectory still requires
refinement. This is an infinite-step argument, not an extrapolation from a short run.

## Integration and persistent state

For frozen reaction rates, s_inf=alpha/(alpha+beta) and

```
s_next = s + (1-exp(-(alpha+beta)*dt))*(s_inf-s)
```

is an exact convex update preserving[0,1] for every dt>=0. For frozen conductance,

```
V_inf = sum_r g_r*E_r/(g_L+sum_r g_r)
V_next = exp(-dt/tau_membrane)*V + (1-exp(-dt/tau_membrane))*V_inf
I_next = exp(-dt*R/L)*I
```

is exact. Kinetics/electrical/kinetics uses symmetric sub-splitting. The complete
transport/collision/response step still has splitting error: tests compare its
actual generator and refine the same elapsed duration. No spike/reset surrogate
or fixed token pondering duration is introduced.

Receptors survive writes, propagation, collision, explicit detach and idle flow.
Runtime checkpoint schema2 includes them. Schema1 remains valid for the quadratic
model; a conductance continuation missing receptors fails rather than cold-starts.
The timestamped likelihood gives gradients to electrical parameters and gate rates.

## Constructive verification

A member of the actual local actor realizes the exact equilibrium Jacobian
`[[-2,1,-1],[15,-6,0],[20,0,-2]]`. Without its inhibitory feedback the excitation
subsystem, with the inhibitory state held at its equilibrium rather than providing
feedback, has eigenvalue+0.358899. The closed loop has eigenvalues
`-1.183330 +/- 3.678888i`, `-7.633341`: a stable oscillatory response is
representable before training. These numbers are a feasibility witness, absent
from production initialization. They impose no biological frequency. A damped
focus verifies response/recovery; persistent limit cycles and semantic
specialization remain separate properties to establish.

The tests cover kinetics, source/heat identity, inhibition reversal, effective
time, balanced rest, learnable independent time scales, collision modulation,
generator/refinement, likelihood gradients, continuation and strict weight loading
at finer resolution. The full extended continuous RHS is included in
`plastic_feasibility.py`.

At8x8x4/D128, receptors add256KiB FP32 per individual. Response has66,112
parameters versus66,624 in the removed quadratic-rate network. Receptor input
adds16,384 collision parameters at hidden64: net core increase15,872 parameters.
Actual CPU execution is recorded in `conductance_execution_cpu.json`; matched
quiet/cached and monitored paths have identical evolved state. GPU validation
remains subject to the previously established zero-additional-shared-memory guard.

```powershell
python -m pytest tests/test_conductance_response.py -q
python scripts/ib/audit_conductance_response.py --output results/published/conductance_response_feasibility.json
python scripts/ib/benchmark_continuous_execution.py --device cpu --bath-type conductance --step-duration 0.005 --ticks 32 --output results/published/conductance_execution_cpu.json
```
