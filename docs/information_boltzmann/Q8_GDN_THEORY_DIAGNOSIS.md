# Q8 continuous field versus GDN-2: theory diagnosis

**Status:** active theory branch, 2026-09-29.  This document is a
mathematical diagnosis and architecture specification.  It makes no new
capability claim and authorizes no short-run architecture judgement.

## 1. Evidence ledger

### Observed, source-locked facts

The registered OpenWebText local-window comparison is recorded in
`results/published/ib_language_registry_20260929.json`:

| model | updates | BPE tokens | mean NLL |
| --- | ---: | ---: | ---: |
| Q8 continuous field | 3,000 | 384,000 | 7.21536 |
| GDN-2 d128 | 3,000 | 384,000 | 6.49815 |

Thus the current Q8 control is behind the current GDN-2 reference by
0.71721 nats under the registered four-site protocol.  A historical Q8
report has no source-locked checkpoint and carries no ranking authority.

The current Q8 implementation facts are:

1. The persistent state is a field
   \(F\in L^2(\mathbb T^3;\mathbb R^{8\times16})\), discretized on
   \(8\times8\times4\) nodes.
2. Its writer forms a full-rank packet \(P(x_t,F_t)\) and applies a local
   orthogonal boundary rotation.  The packet is built from the observed token
   embedding and state/neighbor corrections; it is not explicitly an
   observation residual.
3. Cayley transport and nullspace Givens collision preserve field norm.
   The current site audit reports realized transport phase 1.894 rad and zero
   numerical transport-norm residual, so "transport is identity" is ruled out
   for that checkpoint.
4. The active bath is radial quadratic damping:
   \(F_i\mapsto\exp[-\kappa\Delta\tau\|F_i\|^2/r^2]F_i\).  It receives
   no token surprise or semantic mode label.
5. The reflected port is produced by the writer.  It is an external boundary
   flux and is deliberately excluded from the language decoder.

The current GDN-2 update instead contains an explicit retrieval correction.
With
\(r_t=((b_t\odot k_t)^\top D_tS_{t-1})\), its event increment is

\[
\Delta S_t=k_t\big[(w_t\odot v_t)-r_t\big]^\top.
\]

Decay, retrieval-side erasure and value-side writing are separately gated.

### Deductions from those facts

The Q8 field is a genuine kinetic state rather than an accidentally idle
transport buffer.  Its deficit therefore cannot be attributed to a zero
transport angle or to the unread reflected wave.  The remaining question is
how a bounded kinetic field decides which part of an incessant observation
stream is informative enough to enter its finite state.

### Assumptions for this branch

* A successful infinite-stream system must keep one state field alive, use no
  reset or hidden context cache, and remain input-to-state bounded.
* Text prediction must be read from the persistent field.  A direct observed
  token-to-logit residual would defeat the purpose of this comparison.
* The intended system stays general: an observed token is one instance of an
  observation port, not a Sudoku- or language-specific rule.

## 2. Three reconstructions of the problem

### A. Capacity-flow problem

An infinite sequence continuously brings nonzero raw observations.  A finite
state must decide how much of each observation changes the state and how much
leaves through the boundary.  The central variable is the *innovation* of the
observation relative to the pre-event state, rather than raw input energy.

### B. Kinetic observability problem

Q8 distinguishes conserved moments, velocity fluxes and non-equilibrium
collision coordinates.  A language observable must retain the components
whose propagation and collision influence the future token distribution.

### C. Contextual routing problem

The current Q8 address and read query are generated from a raw token embedding
before the event.  A persistent field needs a state-conditioned way to bind an
occurrence to its current role, rather than only to its vocabulary identity.

## 3. Competing hypotheses

| id | hypothesis | status |
| --- | --- | --- |
| H1 | **Missing predictive boundary innovation.** Q8 writes a raw packet whereas GDN writes a retrieval residual.  Raw writing forces the same boundary to carry predictable traffic and genuine novelty. | leading candidate |
| H2 | **Kinetic state is weakly observable.** The generic probe readout may project away flux and non-equilibrium collision coordinates. | retained competitor |
| H3 | **Static semantic coordinates prevent contextual binding.** A raw token determines its spatial address before the field can route the event by context. | retained competitor |
| H4 | **Optimization or budget alone.** Q8 needs more updates or a different optimizer but its present operator is structurally sufficient. | possible, but it explains neither the raw-write/overheat trade-off nor the recurrent-update difference |

H1 and H2 are compatible: H1 governs what reaches the field; H2 governs what
can be recovered from it.  H3 concerns the geometry of the same boundary
port.  They must not be blurred into a single generic "capacity" claim.

## 4. Necessary mathematical closure

Let \(\mathcal H=L^2(\mathbb T^3;\mathbb R^{Q\times A})\) and let
\(P_t\in\mathcal H\) be the packet synthesized from the observed event.
The current Q8 evolution has the schematic form

\[
F_{t+1}=\mathcal B_t\,\mathcal C_t\,\mathcal T_t\,
\mathcal W_t(F_t,P_t).
\]

Here \(\mathcal T_t\) and \(\mathcal C_t\) are isometries.  The active
bath contracts radial amplitude, and the writer is the only token-dependent
exchange with the field.  Hence the raw-input storage balance is

\[
H(F_{t+1})-H(F_t)
=W_{\rm raw}(F_t,P_t)-D(F_t),
\qquad H(F)=\tfrac12\|F\|_{\mathcal H}^2.
\]

Every ordinary token has nonzero \(\|P_t\|\), including a token the field
already predicts nearly perfectly.  A raw-input law consequently asks the
same amplitude-control mechanism to solve two incompatible jobs:

1. admit a surprising event strongly enough to change memory;
2. reject a predictable event strongly enough to avoid filling capacity with
   repeated information.

W2 impedance can approximate this distinction through its learned angle, but
the distinction itself is absent as a state variable and as an internal
generative consistency equation.  A scalar or radial bath then removes signal
and noise by the same local amplitude law.  This is the common source of the
observed historical failure pattern: additive writing overheats, while strong
impedance or damping underwrites the field.

GDN-2 resolves the same conflict algebraically.  Its update includes

\[
r_t=(b_t\odot k_t)^\top D_tS_{t-1},
\qquad
\Delta S_t=k_t\big[(w_t\odot v_t)-r_t\big]^\top.
\]

When the gated incoming value agrees with retrieved state, the event increment
tends to zero in that key coordinate.  When it disagrees, the residual enters
memory.  This is a predictive-coding boundary law, not merely a decay
schedule.

### Proposed kinetic closure: innovation port

Q8 should carry the same principle in field space:

\[
\widehat P_t=\Pi_\theta(F_t^-),
\qquad
\delta P_t=P(x_t)-\widehat P_t,
\]

\[
F_t^+=\mathcal W_\theta(F_t^-,\delta P_t),
\qquad
F_{t+1}=\mathcal B_\theta\mathcal C_\theta\mathcal T_\theta F_t^+.
\]

\(\Pi_\theta\) is a *field-to-input-port* generative model.  It predicts a
packet in exactly the same Hilbert space, with the same spatial and velocity
coordinates as the observed port.  It is not a token-to-logit bypass.  The
ordinary language likelihood remains

\[
p_\theta(x_{t+1}\mid F_t^+).
\]

The boundary therefore accepts prediction error, while transport and
collision reorganize it.  The reflected wave remains an unobserved external
flux used only for port accounting:

\[
\|F_t^+\|^2+\|R_t\|^2
=\|F_t^-\|^2+\|\delta P_t\|^2.
\]

This form gives the correct infinite-horizon supply rate:

\[
H(F_{t+1})-H(F_t)
\le -D_\theta(F_t^+)+\tfrac12\|\delta P_t\|^2.
\]

If the dissipative operator is strictly positive only on the non-conserved
kinetic subspace, \(D_\theta\ge\alpha\|g\|^2\), then the system is
input-to-state stable in **unpredicted information**, while transport and
collision continue to preserve their stated invariants.  This is the relevant
mathematical substitute for global \(\gamma\): a selective Onsager mobility
on non-equilibrium coordinates, driven by innovation rather than raw energy.

The result is conditional.  It proves boundedness if the stated coercivity,
bounded-port and causal-generator conditions hold.  It does not prove that a
particular learned \(\Pi_\theta\) will attain low language NLL.

## 5. Theory-bound predictions

The innovation-port hypothesis commits the architecture to the following
properties before any capability training is considered:

1. **Equilibrium no-write.** If \(P_t=\widehat P_t\), then
   \(\delta P_t=0\) and the boundary write is the identity.  A predictable
   observation cannot heat the field merely because it occurred.
2. **Innovation, not token norm, controls storage.** Two token events with
   equal packet norm and unequal \(\|\delta P_t\|\) have unequal state
   increments.  Raw Q8 has no such identity.
3. **Invariant-respecting dissipation.** The dissipative flow annihilates
   neither collision invariants nor transport norm directly; it contracts the
   non-equilibrium coordinates selected by a positive-semidefinite mobility.
4. **Bounded infinite trajectory.** For bounded innovation energy, the
   storage inequality bounds time-averaged non-equilibrium energy without a
   reset or a global scalar decay schedule.
5. **Failure condition.** If \(\Pi_\theta\) predicts only a static token
   prior, or if the mobility contracts all coordinates equally, this branch
   reduces to a dressed-up raw writer and loses its theoretical advantage.

H2 instead predicts a nontrivial nullspace of the readout Jacobian on the
collision tangent space.  H3 predicts that two identical tokens in distinct
field contexts cannot be routed to distinct port geometries.  These are
analytic properties of the implemented maps, not short-training outcomes.

## 6. What is deliberately excluded

* Reading the reflected wave into the decoder.  That creates an observed-token
  shortcut and does not improve the field's causal responsibility.
* A global \(\gamma\), hard clip, resource counter or fatigue variable.
  Each controls amplitude without representing predictive surprise.
* Treating numerical resolution \(K\) as the cause.  In the converged
  integrator limit, changing \(K\) cannot repair a missing supply variable.
* Declaring a finite experiment a proof of infinite-life behavior.  A future
  long OWT run can only test the implementation of this theory package.

## 7. Decision and stopping condition

The next action is a formal operator specification, before an architecture
change:

1. Define the common packet Hilbert space for observed and predicted ports.
2. Define a causal \(\Pi_\theta:F_t^-\mapsto\widehat P_t\) with no access to
   \(x_t\) or \(x_{t+1}\).
3. Define a positive-semidefinite local mobility on the collision
   non-equilibrium subspace and prove the stated storage inequality.
4. Derive the readout observability condition for \((m,J,g)\) separately.

Stop this branch if a causal predicted port cannot share the observed port's
geometry, or if the storage proof requires an unbounded controller, a hidden
reset, or an output shortcut.  In that case H1 is rejected and H2/H3 become
the next theory target.

## 8. Minimal architecture implied by H1

The candidate is a *predictive kinetic port*, not a larger decoder and not a
copy of GDN's matrix state.

At event \(t\), first form the pre-event prediction in the physical input
space:

\[
\widehat P_t=\Pi_\theta(F_t^-),
\qquad \Pi_\theta:\mathcal H\to\mathcal H.
\]

The external observation supplies \(P_t=P_\eta(x_t)\) in that same space.
The only new excitation is \(\delta P_t=P_t-\widehat P_t\).  A valid
boundary exchange has to satisfy the two exact limits

\[
\mathcal W(F,0)=F,
\qquad
\|\mathcal W(F,\delta P)\|^2+\|R\|^2
=\|F\|^2+\|\delta P\|^2.
\]

The first condition prohibits predictable tokens from erasing state merely
through a nonzero rotation angle.  The second retains the existing
port-Hamiltonian accounting.  It implies that the scattering angle itself
must vanish continuously with innovation; it cannot be a free scalar gate
whose value is independent of \(\delta P\).

Write \(F=A^\top m+Ng\), where the columns of \(A\) span collision
invariants and the columns of \(N\) span the local non-equilibrium kinetic
coordinates.  The autonomous interior has the split

\[
\dot F=\underbrace{\mathcal L_c F}_{\text{streaming}}
+\underbrace{N\,\Omega_\theta(F)\,N^\top F}_{\text{collision}}
-\underbrace{N\,M_\theta(F)\,N^\top\nabla_g\Psi_\theta(g)}_{
\text{selective dissipation}},
\]

where \(\mathcal L_c^*=-\mathcal L_c\),
\(\Omega_\theta^*=-\Omega_\theta\), and \(M_\theta\succeq0\).  Transport
and collision therefore reorganize information without changing the selected
storage metric, while dissipation can remove only non-equilibrium excess.
The mobility may depend causally on local field state and predicted-port
mismatch; it may not receive the future token or the target token.

The language observation remains a field-only map

\[
\ell_t=D_\theta(F_t^+),\qquad p(x_{t+1}\mid F_t^+)=\operatorname{softmax}\ell_t.
\]

This is the smallest architecture that contains the principle presently
missing from Q8: **prediction creates an equilibrium port; observation writes
only its discrepancy; collision redistributes that discrepancy; selective
dissipation removes obsolete non-equilibrium structure.**

## 9. Review status

This diagnosis was reconstructed from the registered checkpoints and current
code by the primary agent.  It has not yet received an independent
theory-review pass.  Its equations are a proposed closure, not an established
theorem about language capability.
