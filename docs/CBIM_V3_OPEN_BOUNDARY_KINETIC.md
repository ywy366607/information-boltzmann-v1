# CBIM v3: Open-Boundary Self-Organizing Kinetic Field

## 1. Empirical reason for the redesign

The MaleCNS variants have nearly equal parameter counts (about 3.34M), yet
the role-separated kinetic variants trail MaleCNS v1.  At update 1250 the
held-out OWT NLL values are:

| model | validation NLL | mean field energy |
|---|---:|---:|
| CBIM-1D | 7.6285 | not comparable (global budget) |
| MaleCNS v1 | 7.9248 | 0.6127 |
| kinetic v2 | 7.9638 | 3.8842 |
| kinetic energy v2.1 | 8.0258 | 0.0533 |
| kinetic local-gamma v2.2 | 8.0362 | 0.1759 |

Energy differs by more than 70x across the graph kinetic variants while NLL
changes by only about 0.11.  Gamma calibration is therefore not the missing
language computation.

The kinetic rewrite removed three useful degrees of freedom from v1:

1. The v1 boundary operator reads the token, local state and neighboring
   state and proposes a full 64-dimensional update.  The kinetic source reads
   only the token and injects a separable rank-one velocity-content packet.
2. V1 has nonlinear state-dependent exchange across graph edges.  Kinetic v2
   has only a state-independent spectral transport between nodes.
3. The kinetic collision predicts only six angles and applies each angle to
   every content component.  It also conserves four moments separately for
   every content component, leaving too little room for learned content
   interaction.

V3 restores these computations without returning to bulk decay or a hard-coded
slow/workspace split.

### 1.1 MaleCNS v1 extraction boundary

V1 is a source of useful operator components, not the architecture to restore.

Keep and generalize:

- the full-dimensional state- and neighborhood-conditioned input proposal;
- per-channel continuous graph spectral symbols;
- the state-only multi-query readout and tied token embedding/decoder.

Discard rather than copy:

- the hard local norm projection, which activated on 48.2% of the measured
  post-warmup updates;
- the combined write/erase/outflow implementation without an energy ledger;
- interpreting cross-node edge scattering as a Boltzmann collision;
- the costly state-dependent edge rotations: spatial redistribution belongs
  to velocity transport rather than a separate fifth operator;
- the undifferentiated 64-channel state with no velocity/content semantics;
- topology-specific biological claims before a matched randomized-graph
  control and resolution-transfer test.

V1's spatial scattering changed the state by about 18% per token after warmup,
but its frozen bypass cost only 0.0189 NLL.  V3 therefore does not copy it.
Spatial redistribution is tested through velocity-resolved graph transport.

V1 also had a channel stable rank of only 1.324/64 despite using almost all
spatial parcels.  V3 must log spectral concentration and reject a checkpoint
that attains low NLL by reproducing this channel collapse.

## 2. State and operator split

The persistent individual state is

\[
f_t(x,q,a),\qquad r_t(x)\in[0,1],\qquad z_t(x)\ge 0,
\]

where `f` is the information distribution, `r` is locally available write
resource, and `z` is a slow fatigue/load estimate.  A graph is only a
Galerkin discretization of the continuous coordinate `x`; node-specific
parameter tables are forbidden so weights transfer across resolutions.

One token applies

\[
f_{t+1}=
\mathcal B_{\theta,x_t}
\;\mathcal C_\theta
\;\mathcal T_{\theta,f,z}
\;\mathcal S_{\theta,x_t,f,r}(f_t).
\]

The four operators have non-overlapping responsibilities:

- `S`: state-aware exchange with the token stream at a localized boundary;
- `T`: conservative spatial transport of discrete velocity channels;
- `C`: conservative local velocity/content collision;
- `B`: true outflow only through the active boundary, never uniform bulk
  damping.

## 3. Full-rank, resource-limited boundary exchange

For token embedding `e_t`, local neighborhood aggregate `n_i`, anatomical
features `p_i`, and current state `f_i`, a shared operator produces a full
velocity-content proposal and a write rate:

\[
u_i=U_\theta(e_t,f_i,n_i,p_i)\in\mathbb R^{Q\times A},
\]

\[
\eta_i=w_i(e_t,p_i)\,r_i\,\sigma(g_\theta(e_t,f_i,n_i,p_i)).
\]

The environment exchange is

\[
f_i^+=(1-\eta_i)f_i+\eta_i\,\bar u_i,
\qquad \|\bar u_i\|\le R_{\rm packet}.
\]

This is an open-boundary replacement law, not bulk forgetting.  Repeated or
already represented input may be made cheap by predicting a residual
`u_t - read_i(f_i)` inside `U_theta`.

The resource is persistent:

\[
r_i^+=\operatorname{clamp}_{[0,1]}
\left[r_i+\rho(1-r_i)-c\eta_i\|\bar u_i-f_i\|^2\right].
\]

## 4. Conservative velocity-resolved spatial transport

Use the resolution-independent graph spectral orthogonal transport and give
every `(q,a)` channel its own continuous dispersion symbol. The velocity index
is part of that symbol, so local collision can redirect content into a
different velocity channel and transport then moves it differently through
the graph.

No separate spillover operator is present. In the Boltzmann equation, pressure
and heat flux arise in moment equations from velocity-resolved transport and
collision; free transport does not explicitly read a pressure scalar. Adding
a second cross-node rotation would duplicate transport, obscure causal
attribution, and exceeded the speed budget in the first v3 smoke run.

Fatigue remains a slow boundary-control variable:

\[
z_i^+=\rho_z z_i+(1-\rho_z)\tfrac12\|f_i\|^2.
\]
## 5. Expressive local kinetic collision

Keep D3Q8, but conserve total mass-like and three total momentum-like moments,
not four moments independently for every content feature.  Let `N` span the
nullspace of those four linear constraints in the full `Q*A` state:

\[
y=N^T f_i.
\]

Apply `K` state-dependent Givens rotations to `y`, with pair schedules cycled
across steps so all nullspace directions can interact.  Angles read the full
local state and node features.  Reconstruct

\[
f_i'=f_{i,\rm conserved}+Ny'.
\]

The collision therefore preserves the declared moments and quadratic energy
while permitting velocity-content exchange.  Start first implementation:
`K=16`, two batched rotation layers, no Python loop over nodes.

## 6. Boundary dissipation and infinite-stream condition

Remove `exp(-gamma) * f` from the bulk.  True outflow is restricted to the
currently addressed boundary and is coupled to local load/resource:

\[
f_i^{\rm out}=(1-o_iw_i)f_i,
\qquad
o_i=o_{\max}\sigma(O_\theta(f_i,z_i,r_i,p_i)).
\]

Transport and collision are conservative; only measured boundary exchange
changes total energy.  Slow and fast memory modes are not declared.  They are
the eigenmodes of the learned open evolution operator with weak or strong
overlap with its moving boundary.

For an analytic infinite-stream claim, training measurements are insufficient.
The required condition is a uniform boundary-observability inequality: over
some finite window `K`, every nonzero state direction must couple to the moving
boundary by at least `c > 0`.  Together with bounded packets and nonzero
boundary outflow, this yields an input-to-state energy bound.  A numerical
observability Gramian can test the finite discretization, but it is not the
proof for the continuous operator.

## 7. Implementation order and gates

1. Fork MaleCNS v1, retain its full-rank state-aware writer, readout, and
   content-dependent spectral transport.
2. Replace its edge scattering with velocity-resolved orthogonal graph
   transport; do not add a separate cross-node spillover operator.
3. Replace the rank-one kinetic source with the full-rank boundary exchange and
   add persistent resource/fatigue states.
4. Add the full-state nullspace collision so learned local scattering changes
   which spatial transport mode carries each content component.
5. Remove bulk gamma and hard projection.  Log incoming/outgoing boundary work,
   total energy, spatial peak ratio, resource, fatigue, collision angles, and
   the energy-ledger residual.

Performance gate: 128 BPE tokens/update, less than 1.5 s/update, less than 2 GiB
reserved memory.  Capability gate: after 3000 OWT updates, match or improve the
MaleCNS v1 validation curve at the same updates.  Mechanism gate: frozen-weight
collision and transport bypasses must each worsen held-out NLL, and persistent
state must outperform per-token reset.  Failure of any gate keeps the result as
an ablation and blocks larger graphs.

The no-spillover CUDA Graph smoke test measured 0.61--0.63 seconds per
128-token update with 326 MiB reserved memory.  This is an engineering result,
not a capability result; the 3000-update OWT curve remains the next gate.

