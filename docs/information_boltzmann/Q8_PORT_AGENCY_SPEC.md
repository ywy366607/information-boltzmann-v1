# Q8 port agency: active inference without a decoder bypass

**Status:** proposed mathematical specification, 2026-09-29. This is a
design closure for the predictive-innovation branch. It is neither a
capability result nor authorization to judge the design from a short run.

## Purpose

The Q8 field already supplies the physical state law: boundary writing,
unitary transport, invariant collision, and selective dissipation. The missing
question is *which port configuration should act at an event*. A port agent
answers that question while leaving the physical responsibilities unchanged:

\[
\boxed{\text{write agent chooses admission; transport moves; collision
reorganizes; read agent chooses observation.}}
\]

The agent is internal active inference, not an external token-generation
policy. It never chooses the observed token and it never writes an answer
directly into the decoder.

## Explicit predictive belief and posterior belief

Full active inference requires a belief state, rather than only a recurrent
field and a loss.  At every event Q8 therefore maintains

\[
b_t^-=(\mu_t^-,\Sigma_t^-,\pi_t^-),
\qquad
b_t^+=(\mu_t^+,\Sigma_t^+,\pi_t^+).
\]

\(b_t^-\) is the **prior belief** before observing \(x_t\): a predictive
mean field, a structured uncertainty/precision, and a prior over port actions.
\(b_t^+\) is the **posterior belief** after incorporating \(x_t\).  The
persistent kinetic field used by the rest of the model is the posterior mean
\(\mu_t^+\); the uncertainty state is retained alongside it rather than
discarded after training.

The dynamics generate the next prior by propagating the preceding posterior:

\[
\mu_t^-=\Phi_{a_{t-1},\Delta\tau_{t-1}}(\mu_{t-1}^+),
\qquad
\Sigma_t^-=J_\Phi\Sigma_{t-1}^+J_\Phi^\top+Q_\theta.
\]

The actual implementation need not store a dense
\(32768\times32768\) covariance.  A local block precision for conserved,
flux and non-equilibrium channel coordinates is sufficient, provided it is a
real persistent state with an explicit update law.  A point field with no
precision state is predictive coding; it is a useful limiting case, but it
is not the complete active-inference model proposed here.

Let \(\varphi(x)=P_\eta(x)\) be the common observed-port statistic.  The
prior's categorical observation model and predicted packet must come from one
generative chart:

\[
p_\theta(x_t\mid b_t^-)
=\exp\!\left(
  \langle\zeta_\theta(b_t^-),\varphi(x_t)\rangle
  -A(\zeta_\theta(b_t^-))
\right),
\qquad
\widehat P_t
=\mathbb E_{p_\theta(x\mid b_t^-)}[\varphi(x)]
=\nabla_\zeta A.
\]

This identity prevents an independent token head from making a prediction
that the physical input port cannot express.  A practical BPE implementation
may factorize this expectation, but the factorization must preserve the same
packet statistic.

After observing \(x_t\), the posterior is

\[
q_\phi(F_t^+,a_t^\mathrm w\mid b_t^-,x_t),
\]

and minimizes the realized variational free energy

\[
\mathcal F_t=
-\mathbb E_q\log p_\theta\!\left(\varphi(x_t)\mid b_t^-\right)
+\operatorname{KL}\!\left[
q_\phi(F_t^+,a_t^\mathrm w\mid b_t^-,x_t)
\middle\|
p_\theta(F_t^+,a_t^\mathrm w\mid b_t^-)
\right].
\]

The first term is prediction error at the common physical port.  The second
term is the price of changing state and policy away from the prior.  This
gives the write action a posterior-update meaning rather than treating it as
an unconstrained learned gate.

## One port policy, two causal actions

Let \(F_t^-\in\mathcal H\) be the pre-observation persistent field and
let \(P_\eta(x_t)\in\mathcal H\) be the observed packet. The policy is
causal only with the following factorization:

\[
\begin{aligned}
q_\xi(a_t^\mathrm w,\Delta\tau_t,a_t^\mathrm r\mid F_t^-,x_t)
={}&q_\xi^\mathrm w(a_t^\mathrm w\mid F_t^-,\delta P_t)\\
&q_\xi^\tau(\Delta\tau_t\mid F_t^+,a_t^\mathrm w)\\
&q_\xi^\mathrm r(a_t^\mathrm r\mid F_{t+1}).
\end{aligned}
\]

The pre-event port prediction is a separate map of \(b_t^-\) alone.  No
factor may inspect \(x_{t+1}\).  The write factor is posterior with respect
to the newly observed \(x_t\); duration and read factors are priors with
respect to the still-unseen \(x_{t+1}\).  Their posterior update occurs when
that next observation arrives.

The factors have sharply limited action spaces.

| factor | may choose | may not choose |
| --- | --- | --- |
| write action \(a^\mathrm{w}\) | packet chart, local spatial aperture, velocity mixture and positive local admittance | observed content \(x_t\), a decoder logit, a direct state replacement |
| read action \(a^\mathrm{r}\) | field probe locations, aperture and per-probe precision | the current observed token, reflected port, a token-to-logit residual |
| duration \(\Delta\tau\) | physical evolution duration | numerical solver accuracy |

Thus the two agents are two causal faces of **one port policy**, rather than
two unrelated attention modules.

## Predictive impedance write action

Before the event, the field generates an input-port prediction:

\[
\widehat P_t=\Pi_\theta(F_t^-),\qquad
\delta P_t=P_\eta(x_t)-\widehat P_t.
\]

After observing \(x_t\), the write posterior selects a local passive
admittance \(Y_t\succeq0\) and packet chart from \((F_t^-,\delta P_t)\).
It then performs one port exchange

\[
(F_t^+,R_t)=\mathcal S_{Y_t,a_t^\mathrm{w}}
             (F_t^-,\delta P_t).
\]

\(\mathcal S\) is parameterized as a Cayley/orthogonal scattering map. The
action-dependent angle is a function of the dimensionless innovation energy
\(\delta P_t^\top Y_t\delta P_t\), so it has the required limit

\[
\delta P_t=0\quad\Longrightarrow\quad F_t^+=F_t^-,\quad R_t=0.
\]

This is the precise sense in which the write agent *is* the impedance. The
predictor \(\Pi\) says what the field already accounts for; the admittance
says how much remaining discrepancy the local medium can incorporate. They
must be implemented as one `PredictiveImpedancePort`, not as a predictor plus
a separate amplitude patch.

The reflected port remains an external accounting variable:

\[
\|F_t^+\|^2+\|R_t\|^2=\|F_t^-\|^2+\|\delta P_t\|^2.
\]

It is deliberately unavailable to the decoder. Giving the decoder \(R_t\)
would create a current-observation shortcut and remove the field's causal
responsibility.

## Read action

After the field evolves, the read action chooses a family of normalized
probes \(\rho_h(a_t^\mathrm r)\) and precisions \(\Lambda_h\succ0\):

\[
z_{t,h}=\langle\rho_h(a_t^\mathrm r),F_{t+1}\rangle_{\mathcal H},
\qquad
p_\theta(x_{t+1}\mid F_{t+1},a_t^\mathrm r)
=D_\theta(z_{t,1},\ldots,z_{t,H}).
\]

It chooses *where and at what resolution to observe the field*. It cannot
choose the answer. Read locality therefore gives transport, collision and
distributed storage an actual route to influence language likelihood, while
preserving a field-only decoding path.

## Interior and physical time

For the selected duration, the field follows

\[
F_{t+1}=\exp\!\left[\Delta\tau_t
  (\mathcal L_c+\mathcal C_\theta-\mathcal D_\theta)\right]F_t^+,
\]

where \(\mathcal L_c^*=-\mathcal L_c\) is velocity-aligned transport,
\(\mathcal C_\theta\) is collision on the non-equilibrium tangent space,
and \(\mathcal D_\theta\succeq0\) is selective Onsager dissipation. The
port is the only observation-dependent exchange.

\(K=64\) is the fixed numerical quadrature of this flow. It is not an agent
action. \(\Delta\tau\) is physical internal duration selected by the port
posterior; its computational price is an explicit resource term, rather than
a hidden fixed block length. The dissipative semigroup supplies well-posedness
for every finite \(\Delta\tau\); the available inference budget supplies the
practical horizon.

## Active-inference objective

For a candidate action triple \(a=(a^\mathrm w,a^\mathrm r,\Delta\tau)\),
the common generative model defines expected free energy

\[
G_t(a)=\mathbb E_{q_\theta(o',F'\mid F_t^+,a)}
  \left[\log q_\theta(F'\mid F_t^+,a)
        -\log p_\theta(o',F'\mid F_t^+,a)\right]
  +c_\mathrm{time}\,\Delta\tau.
\]

The likelihood term supplies task-general predictive accuracy; the posterior
term prices unexplained state complexity; their difference contains the usual
epistemic value of actions that reduce uncertainty. For language, \(o'\) is
the next token. For vision, control, or reconstruction, it is the next
observation under the same port interface. No Sudoku constraint occurs in
this definition.

The realized event also has variational free energy

\[
\mathcal F_t(a^\mathrm w)=
 -\log p_\theta(x_t\mid F_t^-)
 +\operatorname{KL}\!\left[
 q_\theta(F_t^+\mid F_t^-,x_t,a^\mathrm w)
 \middle\|p_\theta(F_t^+\mid F_t^-,a^\mathrm w)\right].
\]

This makes the innovation port a posterior state correction rather than a
raw-token gate. Cross-entropy remains the observation likelihood, but it is
no longer the only signal determining port actions.

## M, T and K

An \(M\)-branch run samples full, separate field trajectories through this
causal factorization.  It samples \(a_{t,m}^\mathrm w\), produces
\(F_{t,m}^+\), samples \(\Delta\tau_{t,m}\), evolves to \(F_{t+1,m}\), and
only then samples \(a_{t,m}^\mathrm r\), for \(m=1,\ldots,M\).

Each branch owns a complete Q8 grid; it is not eight probes acting on a few
cells of one shared board. Branch diversity arises from posterior sampling
over port actions and duration, not from an arbitrary anti-collapse penalty.
When the posterior becomes unimodal, convergence of branches is the correct
Bayesian result. When genuine ambiguity remains, distinct actions and full
field trajectories remain available for selection or mixture.

\[
\boxed{M=\text{parallel posterior trajectories};\quad
       T=\Delta\tau=\text{physical duration};\quad
       K=64=\text{integrator resolution}.}
\]

MCTS or GRPO are later optimization methods for discrete or non-reparameterized
port-action distributions. They are not part of the core dynamics and may not
be used to conceal a missing field-to-port predictive model.

## Mathematical obligations and stop conditions

Before code or a capability run, this branch must satisfy:

1. \(\Pi_\theta(F)\) and \(P_\eta(x)\) inhabit exactly the same packet
   space and chart.
2. The entire field-to-logit path excludes \(x_t\) and \(R_t\) after their
   permitted boundary roles.
3. The write map is identity at zero innovation and preserves the declared
   port storage metric.
4. The read probes have a nonzero Jacobian on both conserved and
   non-equilibrium field coordinates; otherwise H2 in the diagnosis remains
   the binding bottleneck.
5. The port policy is causal: its pre-event prediction cannot inspect
   \(x_t\) or \(x_{t+1}\).

If these conditions force an output shortcut, a hidden reset, or an
unbounded controller, port agency is rejected as the Q8 closure and the
research returns to the readout-observability or contextual-routing branches.

## First implementation scope

`information_boltzmann/core/torus3d.py` now contains
`PredictiveImpedanceWriteAgent` and `KineticBeliefState`.  The implementation
maintains a posterior field mean plus channelwise precision; it propagates
precision through learned process variance, forms a posterior observation
precision, and retains that tensor across `belief_step` events.  The
observation packet and predicted packet are synthesized by exactly the same
nonlinear `FullRankTorusWrite.synthesize_packet` chart.  Its structural tests
establish zero-innovation identity, port-metric balance, persistent precision,
and gradients to the port prior and posterior action law.

This is deliberately the first closure, not the final categorical language
likelihood.  The present field-generated port coordinate is a continuous
prediction in the shared chart.  Because the current packet chart has
nonlinear token-dependent address, width and content maps,
\(P(\mathbb E[e])\) is not generally
\(\mathbb E[P(e)]\).  An exact categorical identity
\(\widehat P=\mathbb E_{p(x\mid b^-)}P(x)\) requires a factorized
exponential-family packet chart and remains an explicit next obligation.
Until then, the port density is a physical observation factor and the
field-only token decoder remains a separate likelihood factor.  This prevents
the first implementation from claiming a stronger generative equivalence than
its current geometry provides.

M sampling, the read agent, adaptive physical duration, and the CUDA-graph
trainer intentionally remain unchanged in this commit.  They require the
factorized port chart so that one common prior/posterior model, rather than a
collection of auxiliary losses, governs all port actions.

## Review status

This is a first-principles specification reconstructed by the primary agent.
It has not received an independent theory review.
