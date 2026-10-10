# State-continuous branching and Connes-Kreimer bookkeeping

Date: 2026-10-09. Status: independently reviewed theoretical interface proposal.
This note introduces no runtime implementation, experiment or training change.

## Target and competing explanations

The target is a finite persistent medium whose learned routes can branch,
recombine and be reused while input, evolution and output continue. The useful
endpoints remain real-stream prediction, recovery speed/platform and retention,
under the existing dedicated-memory guard and continuous evaluation contract.

Three distinct possibilities must remain distinguishable:

1. Existing continuous capacity/direction learning already supplies useful routes;
   an explicit graph description adds only a reusable structural representation.
2. Apparent branching is a coordinate decomposition of the same calculation;
   fully conjugating dynamics and ports must leave input/output unchanged.
3. New task-valued branch couplings change propagation or retrieval and provide
   useful independent computations. This is the capability hypothesis.

The algebra addresses representation and consistent decomposition. A physical
state-transfer contract addresses continuity. Task/resource optimization decides
which actual couplings to maintain. These are separate mathematical roles.

## Exact CK meanings

Use the commutative rooted-tree Connes-Kreimer Hopf algebra. Its elements are
formal linear combinations of forests; multiplication juxtaposes forests.
For proper admissible cuts c (at most one cut along each root-to-leaf path),

\[
\Delta T=T\otimes1+1\otimes T+
\sum_c P_c(T)\otimes R_c(T).
\]

This is a sum of decompositions, rather than a physical wave-copy operation.
Grafting a forest onto a new root is B_+, which satisfies the cocycle identity

\[
\Delta B_+(F)=B_+(F)\otimes1+(id\otimes B_+)\Delta F.
\]

Coassociativity makes iterated formal decompositions agree, with multiplicities.
It does not imply that arbitrary physical edit schedules commute. The antipode
is a convolution inverse; the counit sends the empty forest to 1 and nonempty
forests to 0. Neither supplies dissipative time reversal or a physical endpoint.
Independent forest juxtaposition may be commutative; overlapping or ordered
physical edits require explicit port/order information.

Sources: [Connes and Kreimer, 1998](https://arxiv.org/abs/hep-th/9808042),
especially the rooted-tree coproduct and grafting cocycle;
[Manchon, 2006](https://arxiv.org/abs/math/0408405), decorated rooted trees and
graph Hopf algebras. Cycles require a graph class closed under its allowed
subgraph extraction and contraction. Alternatively, a rooted operation-history
tree can contain a recurrent physical module as a decoration; that is a history
representation, not a claim that the recurrent graph is itself a rooted tree.

## The continuous-material interface

Keep the full continuous material and activity state as the physical substrate.
A proposed module decoration records spatial support, compatible boundary ports,
installed capacity, propagation/delay contract and a complete state-transfer map.
The module describes a learned route, rather than prescribing an axon template.
Principal-axis display streamlines alone do not define physical branches.

The current PlasticMedium3D uses paired field/flux generators with B and its
negative adjoint. B occurs in the generator, while stored wave energy uses a
fixed Euclidean metric. Changing material coefficients in this fixed state basis
can retain field and all three edge flux stores exactly; no dimension expansion
or moving-coordinate correction is required for such a coefficient update.

The full persistent MediumState additionally includes elapsed time, conduction,
receptors and transmission. Structural edits must retain these or specify valid
constitutive remaps. Receptor fractions and STP resources are bounded physical
variables: applying a wave-mode rotation to them without a separate derivation
would violate their semantics. Learner/RNG/optimizer continuity is also retained.

## A constructive split/merge contract

This construction is our proposed interface, not a consequence of CK axioms.
Let s denote energy-normalized wave coordinates including field and flux, and
reserve an idle orthogonal mode within a fixed state budget. At a discrete split,

\[
c_1=pc,\quad c_2=(1-p)c,\qquad
(s,0)\mapsto(s_1,s_2)=(\sqrt p\,s,\sqrt{1-p}\,s),
\quad 0\le p\le1.
\]

Installed resource and wave energy are conserved. Choosing amplitude shares from
capacity shares is a design choice. The map uses an existing vacant mode; it
does not obtain extra independent storage for free. For differentiable split
parameters, cos(theta), sin(theta) avoid square-root endpoint singularities.

For independently evolved children use the full orthogonal merge:

\[
a=\sqrt p\,s_1+\sqrt{1-p}\,s_2,\qquad
r=\sqrt{1-p}\,s_1-\sqrt p\,s_2,
\]

\[
\|a\|^2+\|r\|^2=\|s_1\|^2+\|s_2\|^2.
\]

Retaining the residual r preserves invertibility. It is zero immediately after
the stated split, but can carry information after independent branch evolution.
Removing it must be an explicit compression/dissipation decision, with its
energy and predictive consequences accounted for. This contract preserves state
information, not automatically future task performance after changing dynamics.
For a different energy metric the required identity is U^T G_new U = G_old.

For three fixed leaf shares w_i, successive splits with conditional resource
ratios give the same injection amplitudes sqrt(w_i)s in either grouping, provided
there is no intermediate evolution or phase shift. Full residual coordinates
still need an explicit associator. Delays, phases and nonlinear interactions may
make two graftings genuinely different physical systems. A continuously moving
orthogonal basis Q additionally needs the connection term dot(Q)Q^T in its
generator; the discrete state transfer above does not claim to supply it.

## Predictions, independent review and next action

Pure coordinate regrouping with conjugated generators and ports must preserve
input/output. State loss on merging independently evolved branches must equal
the omitted residual contribution in the quadratic energy accounting. Genuine
benefit requires a task-valued change to accessible couplings, directions or
time scales; algebraic relabeling alone cannot yield predictive improvement.

Independent reviewer /root/review_flow_plasticity accepted the conditional
split/merge identities and CK interpretation. The review required full f/q and
volume normalization, separate resource/storage accounting, residual retention,
and explicit treatment of ordered overlaps and moving bases. No capability or
biological-equivalence claim was reviewed as established.

Next action: close one bounded local two-mode module on paper, with state and
port maps, residual location and changed generator, then assign a CK decoration
and allowed-cut rule. First separate a pure reparameterization from the actual
task-valued coupling change. Numerical/interface acceptance can verify those
identities; later capability acceptance requires matched real OWT exposure,
adequate joint updates and the active four-pillar endpoints. Reject an edit on
lost persistent state, invalid ports, unaccounted energy/resource/storage,
or changed predictions in the claimed pure-coordinate case. The current running
individual and stage-one capacity implementation remain unchanged.
