# Runtime-plastic 3D conduction medium

## Purpose and current scope

The medium now contains persistent structural state in addition to its wave
field and stored edge responses. Inputs change activity; activity changes local
propagation coefficients; later signals therefore travel through a modified
medium. No labels, optimizer call, token counter, or reset are required for this
runtime adaptation. The update rule and its content preferences train jointly
with the existing W4 observation and learned-read agents.

This implements functional fast pathways on the existing periodic local graph.
It changes the strength and speed of functional connections, not the grid's
combinatorial adjacency. It is an engineered bio-inspired law. A biological
analogy is activity-dependent modulation of conduction, not literal growth of
axonal membranes or a claim that fly neurons are myelinated.

Relevant primary evidence: Mitew et al., *Pharmacogenetic stimulation of neuronal
activity increases myelination in an axon-specific manner*, Nature Communications
9, 306 (2018), https://www.nature.com/articles/s41467-017-02719-2.
This supports the motivation for activity-dependent structural conduction;
the specific normalized law below is our design, not an equation from that paper.

## One local structural equation

For each stored positive-axis edge e=(i,j), add dimensionless log-speed state p_e.
The stationary continuous Fourier material still specifies baseline c0_e, gain
A_e>0, rate r_e=1/tau_e>0 and processing preferences. A learned positive diagonal
content metric selects which content reinforces each edge. The historical
quadratic control uses M_alpha=diag(softmax(metric_logits_alpha)), shared by axis.
The conductance candidate now uses
M_e=diag(softmax(metric_logits_alpha+W_alpha*a(x_e))), conditioned on continuous
local material. Thus different places can prefer different content while using
one shared parameterized law. The SAME edge metric weights both endpoints,
preserving the Cauchy-Schwarz bound below. Coefficients are shared functions
rather than per-token trainable parameters.

Let j_e denote the existing stored transmission response, not graph adjacency.

```
s_e = (2 <f_i,f_j>_M + ||j_e||_M^2)
      / (||f_i||_M^2 + ||f_j||_M^2 + ||j_e||_M^2 + eps)

tau_e * dp_e/dt = A_e*s_e - p_e
c_e(t) = c0_e * exp(p_e(t))
```

Coherent endpoint content and active transmission reinforce a pathway;
opposed endpoint content can weaken it. Quiet activity returns p toward its
learned baseline. This is local routing evidence, not a proof of semantic value:
the task objective trains the metric, gain, baseline and time scale to make this
activity-dependent memory useful. Uniform activity can reinforce all directions;
the law permits specialization and does not force arbitrary brain partitions.

The gain initializes to 1 (one unit of log-speed contrast), and the caller
specifies a physical plasticity time reference, default 1 model-time unit.
Both gain and relative rate are learned and spatially conditioned. These are
declared units/initial conditions, not a token-period schedule or a speed cap.
Conduction can adapt when gradients are disabled. Gradients train the rule;
the persistent state carries online experience through detach boundaries.

## Training-before-trial mathematical closure

Cauchy-Schwarz gives |2<f_i,f_j>_M| <= ||f_i||_M^2+||f_j||_M^2,
therefore -1<=s_e<=1 independently of activity magnitude.

For fixed finite trained parameters and initial state, define
L_e=max(|p_e(0)|,A_e). The interval [-L_e,L_e] is forward invariant:
at p=L the derivative is nonpositive, and at p=-L it is nonnegative.

Consequently c0_e*exp(-L_e) <= c_e(t) <= c0_e*exp(L_e) for all t.
This is an analytic bound, not a fitted threshold or a post-hoc clipping rule.
Unrestricted changing optimizer weights need a separate parameter-bound audit;
the runtime theorem applies to fixed weights, or uniformly bounded weights.

For frozen local evidence the exact structural update is

```
p_next = exp(-dt/tau)*p + (1-exp(-dt/tau))*A*s.
```

It is a convex combination for every nonnegative dt. The implementation uses
`-expm1(-dt/tau)` for accuracy at small dt. Repeated updates at fixed activity
obey the exact semigroup. Coupled activity still requires numerical refinement.

Transport uses c_e(t) with opposite signs at the two endpoints:

```
df_i/dt = -c_e*j_e/h
df_j/dt =  c_e*j_e/h
dj_e/dt =  c_e*(f_i-f_j)/h.
```

Thus dE_wave/dt=0 even when c changes in time. Every discrete edge rotation
also preserves wave energy. There is no claim that structural metabolism has
zero physical cost: E_wave is the model's representation-energy ledger, not ATP.
The existing passive bath and declared bounded boundary work retain their
previous wave-energy inequality. Structural adaptation is not bulk damping of f.

At the previous synchronized rotating-orbit witness, f is spatially uniform,
j=0, the metric is isotropic, and p*=A*s is constant. In its full Jacobian,
wave derivatives with respect to p vanish because both spatial differences and
edge responses vanish. The new structural diagonal block is -diag(1/tau).
The Jacobian is block triangular: the original stable wave transverse spectrum
is retained and structural modes add strictly negative eigenvalues. This gives
a stable feasible neighborhood for the extended system, conditional on the same
declared boundary-action witness as before. It is not a certificate for W4's
unclosed innovation-only energy-supply policy.

## Actual implementation and interfaces

- `core/conduction_plasticity.py`: normalized local evidence, learned metric,
  spatial gain/time scale and exact frozen-evidence structural relaxation.
- `MediumState.conduction`: [B,X,Y,Z,3]. Every individual has its own medium.
- `with_field`, transport, collision, bath, reads and explicit detach retain it.
- `advance` performs a half structural step, existing transport/collision/bath,
  then another half structural step for each numerical substep.
- Transport remains linear in wave variables when structural state is frozen;
  the coupled adaptive system is nonlinear. Collision remains its own nonlinear
  local content-scattering operator.
- `PlasticMediumPorts3D` enables this path by default; the base core defaults to
  static conductance for reproducible previous certificates. Set
  `adaptive_conduction=True` on `PlasticMedium3D` for the new core.
- Continuation checkpoints must save field, three fluxes, elapsed, conduction
  and port precision. A missing adaptive state raises an explicit error rather
  than silently resetting learned pathways.
- The material rule/weights remain grid-size independent. A runtime structural
  field lives on the evaluation grid; grid-changing continuation additionally
  requires an explicit periodic structural-state resampling policy.

No new M or T search is introduced. `duration` is actual time; `substeps` only
refines it. Fast c requires tighter integration accuracy, even though edge
rotations remain norm-stable. Full GPU latency is still to be measured.

The historical shared-metric variant adds **438 parameters** and **3072 bytes
of FP32 persistent state per individual** at 8x8x4 / D128. The conductance
candidate's material-conditioned metric adds a further **3072 parameters** at
material width8 and no further persistent state. The new work is local reductions, shifts and elementwise
updates; coefficient networks evaluate once per `advance`, not once per substep.
The benchmark exposes `--adaptive-conduction` / `--no-adaptive-conduction` for
matched execution comparisons. No additional physical-time loops are introduced.

## Constructive audit results

`results/published/conduction_plasticity_feasibility.json` records:

- Prescribed coherent path with gain4: along-path speed54.588 vs transverse1.
- After moving the activity: new-path speed54.588, old-path speed1.000182.
  These are analytic rule-response examples, not discovered semantic wiring.
- Adaptive energy-ledger residual1.69e-14 in CPU FP64.
- Full120-dimensional continuous locking Jacobian: one neutral global phase,
  transverse gap0.125, orbit residual4.44e-16; new structural block exact -I.
- Locality, all-step-size structural bounds, refinement, actual CE gradients,
  per-batch independence, full-state checkpoint continuation and compile tracing
  are covered by deterministic tests. The 41-test affected suite passes.

Reproduce:

```powershell
python scripts/ib/audit_conduction_plasticity.py --output results/published/conduction_plasticity_feasibility.json
pytest tests/test_conduction_plasticity.py tests/test_plastic_medium.py tests/test_plastic_ports.py tests/test_plastic_feasibility.py tests/test_plastic_spatial_feasibility.py tests/test_port_event_timing.py -q
```

Current training gate: close W4 boundary support and available energy budget,
then benchmark actual joint updates in a free GPU window before launching the
matched corpus run. This update does not modify or restart other training jobs.
