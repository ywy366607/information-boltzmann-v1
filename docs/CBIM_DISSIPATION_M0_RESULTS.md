# M0：真实 OWT 前史信号与背景干扰诊断

## Protocol

Four independent validation-stream pairs were measured for each frozen
checkpoint.  Each pair uses two unrelated 1024-token OWT prefixes followed by
the same 512-token real OWT suffix.  At the common suffix boundary, the model
starts from:

1. the correct semantic field;
2. a zero semantic field with controller variables copied from the correct arm;
3. an alien semantic field produced by the unrelated prefix, again with matched
   controller variables;
4. the complete alien state.

Collision and transport bypasses start from the correct state.  All weights are
frozen.  Reported values are means across the four prefix pairs.

Definitions:

\[
S(H)=\operatorname{NLL}_{zero}(H)-\operatorname{NLL}_{true}(H),
\]

\[
N(H)=\operatorname{NLL}_{alien}(H)-\operatorname{NLL}_{zero}(H),
\]

\[
P(H)=\operatorname{NLL}_{alien}(H)-\operatorname{NLL}_{true}(H)
=S(H)+N(H).
\]

`S` is the benefit of having a pre-existing field rather than rebuilding from
zero.  `N` is the extra cost of an unrelated pre-existing field.  `P` measures
whether the identity of the prefix, rather than merely the existence of a
nonzero field, matters to prediction.

## 512-token cumulative results

| model | NLL | `S`: correct vs zero | `N`: alien vs zero | `P`: correct specificity | alien-field distance | no collision | no transport |
|---|---:|---:|---:|---:|---:|---:|---:|
| MaleCNS v1 | 7.3944 | -0.0001 | -0.0005 | -0.0006 | 0.00000047 | +0.0208 | +0.7695 |
| v3 | 7.6809 | +0.0199 | -0.0160 | +0.0039 | 1.1474 | +11.1065 | +0.0191 |
| critical v1 | 7.5808 | +0.0252 | -0.0265 | -0.0013 | 0.9971 | +12.2308 | +0.0077 |
| critical-port | 8.1441 | -0.0464 | +0.0449 | -0.0015 | 1.1552 | +6.3084 | +0.0345 |

`alien-field distance` is the final Frobenius distance from the correct field,
divided by the correct field norm.

## What the experiment establishes

### V1 is a fast-turnover predictive medium

The v1 correct and alien prefix fields become numerically indistinguishable
within roughly 128 suffix tokens.  This does not contradict the earlier
`reset_each_token` cost of +0.9769 NLL: v1 uses recent state strongly, but it
replaces old prefix information quickly.  Its spectral transport is strongly
causal (+0.7695 NLL across these four sites), while its edge scattering remains
small (+0.0208).

The trained transport angles explain part of this result.  V1 has mean absolute
angle 0.8059 and maximum 2.5177 radians.  The corresponding v3 values are only
0.2085 and 0.4868 radians.

### V3 stores state but barely stores prefix-specific predictive information

After 512 shared tokens, the alien v3 field remains 1.15 field norms away from
the correct field.  A correct old field improves NLL over zero by 0.0199, but an
alien old field also improves over zero by 0.0160.  Only 0.0039 NLL remains
specific to the correct prefix.  Most of the persistent state therefore acts
as a generic learned operating point or language prior rather than retrievable
history.

V3 collision is indispensable to its trained recurrence, while transport is
nearly bypassed.  The global multi-query readout can inspect every parcel
directly, so the model has no structural need to transport information to an
observation boundary.  Local collision plus global pooling is the cheaper
optimization route.  This is a more direct diagnosis than "the field is too
cold."

### Critical v1 preserves the same wrong object and its controller cannot fix it

Critical v1 also preserves an alien field for 512 tokens, with no positive
prefix specificity.  Its collision/transport division is even more extreme.
The previous conductance sweep already showed that increasing conductance from
0 to 1000 cannot make its trained conditional Lyapunov exponent cross zero.
The wind-up is therefore a failed actuator, while this experiment shows that
the preserved state was not becoming more history-specific anyway.

### Critical-port turns persistent state into measurable background interference

Critical-port is the cleanest confirmation of the signal-to-interference
hypothesis.  At 512 tokens, the correct old field is 0.0464 NLL worse than
starting from zero.  The alien field is 0.0449 NLL worse than zero.  Correct and
alien prefixes are almost indistinguishable in predictive value, while their
internal fields remain far apart.

The two-port energy identity is correct; the learned semantic result is not.
The same accommodation angle couples token absorption, state reflection and
state loss.  In addition, the controller's angle acts on every parcel while the
incident packet is localized.  It can therefore behave as state-dependent bulk
attenuation even though it is named a boundary port.  Exact energy closure does
not make that attenuation information-selective.

## Refined unified diagnosis

The measured failure is not simply failure to retain information.  V3 and both
critical models retain large trajectory differences for hundreds of tokens.
They fail to transform those differences into future-specific, readable
information.  The central quantity is therefore:

\[
\text{useful memory mode}
=\text{input reachable}
\cap\text{future observable}
\cap\text{stably retained}.
\]

V1 has short retention but strong current input-to-output transport.  V3 has
long state retention but weak transport-to-readout alignment.  Critical-port
adds a well-balanced energy port around a field whose old state is already
mostly background, so it preserves the wrong invariant and worsens NLL.

## Selected next model: v3 core with observable transport and a separate bath

The next model should be derived from v3 because its collision compute pattern,
speed and persistent field are already implemented.  It should make four linked
changes:

1. **Delete causally inactive controller state.** Remove resource, fatigue,
   separate outflow and the online global-Lyapunov controller.
2. **Replace global pooling with probe-port readout.** Several geometrically
   distributed probes have compact support and logits are decoded only from
   their outgoing response.  Probe supports are selected by graph coverage,
   rather than a hand-labelled central decision region.  The token query may
   combine probe responses but cannot move a probe onto every state location.
   The freely reflected token component is excluded.  State stored outside the
   probe supports must arrive through transport before it can affect logits.
3. **Replace scalar-Laplacian rotation with geometric D3Q8 advection.** For
   every anatomical edge, the flux of channel `q` is weighted by
   `max(0, c_q dot r_ij)`.  Occupancy and carried content use the same
   conservative finite-volume flux.  Collision changes the velocity channel,
   and velocity then determines the spatial destination.  Physical axis scales
   must be restored before computing directions; independently normalized
   coordinates are not a valid anatomical Euclidean metric.
4. **Use one three-port scatterer with independent low-rank write and bath
   couplings.** The token coupling controls input reachability; the bath coupling
   provides a structural escape route.  Both live in one orthogonal
   extended-system map, but they are independently parameterized.  Every
   input-reachable mode must have nonzero finite-window boundary observability
   by construction.  CE may learn how semantic content is routed into the
   conservative modes; it must not be responsible for learning whether an
   unstable mode is connected to the bath.

The first bath is a fixed cold absorber: `b_in = 0`.  It has no stochastic
feedback and CE cannot turn off its structural escape floor.

Collision remains local and invariant-preserving.  Transport becomes local,
directional and conservative for nonnegative occupancy and carried content.
Dissipation occurs only through the explicit bath output.
No energy target or global lambda target is added.

Training uses ordinary next-token CE plus low-weight training-only predictions
at 8, 32 and 128-token horizons.  These future targets train semantic routing
and readout.  A windowed input-to-state dissipation inequality must follow from
the parameterization and the fixed open boundary, rather than being learned by
an augmented Lagrangian or maintained by an inference-time controller.

This revision exposes a deeper issue in the current core: the Givens collision
is orthogonal and therefore reversible.  It preserves selected moments and
quadratic energy, but it has no demonstrated discrete H theorem.  The next
implementation must either add a conservative entropy-producing collision
step, or explicitly describe the Givens block as reversible scattering and
place all irreversible entropy export at a structurally passive boundary.

## Before a full run

The implementation must pass two gates:

1. Independent changes of write and bath couplings must give a rank-two local
   response in `(input gain, energy flux, predictive field contraction)`.
2. With readout probes frozen, disabling transport must change the outgoing
   response on a nonzero-state trajectory; a direct global field-to-logit path
   is forbidden.

Only then should a matched OWT run start.  At 750 and 1500 updates it must show
both lower NLL and a positive correct-prefix specificity `P(H)` relative to v3.
Failure rejects the input-output alignment theory or its realization; it does
not authorize another auxiliary state variable.
