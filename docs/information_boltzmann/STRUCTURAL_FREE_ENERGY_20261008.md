# Structural retention by variational model comparison

Status: mathematical candidate, independently reviewed; no production change,
training launch, task-performance claim or new metabolic constitutive law.

## Target and alternatives

Target: useful, differentiated learned transmission with better real OWT
prequential prediction, active recovery and revisit retention. Existing birth
capacity projection remains an allocation ceiling, not growth/metabolism.
Three distinct formulations are: capacity feasibility, predictive structural
model selection, and thermodynamic material formation. This proposal selects
the second; the third requires a separate physical maintenance/supply model.

## Generative model

Let z_e in {0,1} denote one candidate coupling with other structure fixed, w the
uncertain parameters, and s_* the SAME complete persistent initial state.
For observed data D, use a normalized causal likelihood

    p(D|w,z,s_*) = product_t p(x_{t+1}|s_t,w,z),
    s_{t+1} = Phi_{w,z}(s_t,x_t).

Each structure evolves its own trajectory under the same observed inputs.
Conditioning on s_* compares interventions from the current individual; it
does not establish that the pruned architecture could have built that state
from birth. Include the earlier generative history if that is the question.

## Free-energy identity

Define Z_z=integral p(D|w,z,s_*) p(w|z) dw and

    A_z[q_z] = E_q[-log p(D|w,z,s_*)] + KL(q_z(w)||p(w|z))
             = -log Z_z + KL(q_z(w)||p(w|D,z,s_*)).

This is variational free energy in nats, not thermodynamic energy in joules.
Its minimum equals -log evidence only if the posterior is represented exactly.
Unequal approximation gaps can reverse the estimated ranking.

For q(z_e)=Bernoulli(r) and conditional prior Bernoulli(pi), holding the two
conditional A_z fixed,

    F(r) = r A_1 + (1-r) A_0
         + r log(r/pi) + (1-r) log((1-r)/(1-pi)).

Differentiation gives

    dF/dr = A_1-A_0 + logit(r)-logit(pi),
    r* = sigmoid(A_0-A_1+logit(pi)).

The hard posterior-MAP rule retains the path iff

    A_0-A_1 > log((1-pi)/pi).

At exact posteriors A_0-A_1=log(Z_1/Z_0), the log Bayes factor. The soft optimum
does not automatically become a binary pruning decision. Shared parameters
give a coordinate update conditional on them, not an exact global evidence.

## Resource interpretation

A declared maintenance cost C(z) can define a normalized structure prior
p(z) proportional to exp(-lambda C(z)) (possibly times a reference prior).
Conditional on other edges this gives logit(pi)=-lambda Delta C_e. Then

    retain iff predictive evidence gain > lambda Delta C_e.

Alternatively minimize E_q[A_z]+KL(q||p_0) subject to E_q[C]<=C_available.
The Lagrangian yields the same Gibbs tilt. Its multiplier is a shadow price,
with dual ascent lambda <- max(0,lambda+eta(E_q[C]-C_available)). Budget and
cost law still require an explicit source; the dual does not manufacture one.
Capacity trace(B B^T) is currently a proxy, not measured maintenance power.
Likelihood alone cannot uniquely determine a material-maintenance law or the
conversion between nats and physical energy. All structural candidates must
use the same reference measure and quadrature when grid resolution changes.

## Link to conservative medium

For the conservative block, write dot f=-D_m^*q and dot q=D_m f.
H=(||f||^2+||q||^2)/2 gives dot H=0 for any paired D_m and its adjoint,
including time-varying structural coefficients with this fixed storage metric.
Gate the coupling and its paired adjoint together. Preserve f, q, receptor,
STP and probe states across structural updates; disconnected flux storage
remains accounted for and may dissipate through existing resistance.
State preservation does not guarantee preservation of semantic accessibility.
This identity concerns activity energy only; physical structural maintenance
would need its own resource/energy ledger.

## Infinite-stream qualification

Static accumulated evidence favors historical winners. An adaptive structure
requires a stated transition prior p(z_t|z_{t-1},resources_t). For a new block
the same derivation uses predictive prior odds and posterior predictive
likelihoods. Arbitrary resets or unexplained forgetting of evidence are not a
substitute for that transition law. No exact cheap O(1) evidence algorithm is
claimed for the full nonlinear medium.

## Predictions and decision

1. Equal predictive evidence with higher maintenance cost lowers retention.
2. Predictive gain above the SAME marginal cost raises retention; high activity
   alone has no prescribed retention benefit.
3. Redundant paths require joint/conditional comparisons: marginal scores may
   undervalue both or keep duplicates. Cost alone does not guarantee branching.
4. Higher supply relaxes resource price under the regular constrained optimum;
   no claimed numerical effect size or guaranteed language improvement.

Next: choose and justify the maintenance/supply and structure-transition laws,
then specify an affordable evidence approximation. Numerical validation should
cover the variational derivative, posterior odds, state continuity and paired
energy identity. Any capability experiment is a separately registered matched
real OWT joint-training comparison; stop the mechanism claim if ranking depends
on unequal approximation error or if improved pictures lack predictive benefit.

Independent mathematical review confirmed the identities and distinguished
soft posterior from hard MAP selection, intervention from birth architecture,
and activity-energy conservation from metabolic accounting. Engineering and
capability efficacy remain open.

## Primary references

- Friston, Parr, Zeidman, Bayesian model reduction:
  https://arxiv.org/abs/1805.07092 . Its shortcut requires matching likelihoods
  with changed priors and appropriate posterior representations; it is not an
  automatic exact solution for this nonlinear medium.
- Kappel et al., Synaptic Sampling: A Bayesian Approach to Neural Network
  Plasticity and Rewiring, NeurIPS 2015:
  https://papers.nips.cc/paper_files/paper/2015/hash/b1a59b315fc9a3002ce38bbe070ec3f5-Abstract.html .
  Supports structural posterior inference as a research direction, not a proof
  of this medium's biological equivalence or performance.
