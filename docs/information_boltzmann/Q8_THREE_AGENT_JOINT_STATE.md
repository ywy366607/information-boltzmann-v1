# Q8 three-agent persistent field

This language closure contains three causal agents around one persistent
posterior field.  They are distinct responsibilities, not separate hidden
backbones.

\[
(F_{t+1},\Pi_{t+1})=
\mathcal S_\theta\bigl(\mathcal W_\theta(F_t,\Pi_t,x_t)\bigr),
\qquad
\widehat x_{t+1}=\mathcal R_\theta(F_{t+1},\Pi_{t+1}).
\]

`W` is the predictive write agent.  It makes a categorical prediction from
the pre-event field, forms an innovation in the shared packet chart, chooses
its passive admittance, and updates the channelwise posterior precision.
The observed token has this one field-entry route.

`S` is the state agent.  Its continuous interior law is

\[
\frac{dF}{d\tau}=J_{\rm transport}F+
J_{\rm collision}(F)F-R_{\rm bath}(F,\Pi)F.
\]

Transport is the linear Cayley field flow and collision is the local
nonlinear invariant scattering map.  The bath is a field-and-posterior
conditioned outflow, never a second token writer.  At a site and content
coordinate it uses

\[
r_{ia}=\operatorname{softplus}g_\theta
  (\operatorname{LN}F_i,\log\Pi_{ia},
   \log(1+\langle\Pi_iF_i^2\rangle)),
\]

\[
F'_{ia}=e^{-r_{ia}\Delta\tau}F_{ia},\qquad
b^{\rm out}_{ia}=\sqrt{1-e^{-2r_{ia}\Delta\tau}}F_{ia}.
\]

Therefore every selective outflow has an exact coordinatewise energy ledger:
\(\lVert F'\rVert_2^2+\lVert b^{\rm out}\rVert_2^2=\lVert F\rVert_2^2\).
It can discharge low-value or over-occupied posterior coordinates without
seeing the raw token.  The first implementation initializes as gentle nearly
uniform leakage; learned selectivity must emerge from the joint objective.

The exact generator is simultaneous.  K=64 is only its numerical quadrature.
One pair of microsteps uses the palindromic map

\[
T_h\,C_h\,B_h\,B_h\,C_h\,T_h,
\]

so over the pair no term permanently owns the first or last position.  This
is a second-order, structure-preserving approximation to the joint flow at
the same count of transport, collision, and bath applications as the old
ordered loop.  An exact implicit joint solve would require a large nonlinear
matrix-free solve and is deliberately deferred until it can meet the GPU
latency budget.

`R` is the field-only read agent.  It selects a posterior-conditioned spatial
aperture and reads kinetic coordinates from `(F, Pi)`; it receives neither the
observed token nor the reflected boundary packet.  Reflection remains boundary
accounting only.

The immediate run is a real OpenWebText learning smoke test.  A falling train
objective and finite independent warm-local validation show that the three
agents can be trained end-to-end.  They do not establish convergence, an
infinite-stream theorem, or a comparison with GDN; those require the
registered longer budget and evaluation protocol.
