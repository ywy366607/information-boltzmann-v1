# Value-guided riverbed plasticity: exploratory theory package

Date: 2026-10-09. Status: independently reviewed proposal; no production learning change, numerical experiment, or new training run.

Stage-one follow-up: [continuous capacity-growth implementation](MEDIUM_CAPACITY_GROWTH_20261009.md)
now closes the coefficient-chart interface and passes numerical/continuation
acceptance. This document retains the original proposal and its unconstrained
local mirror equations; the implemented constrained natural step is specified in
the follow-up. The existing training individual remains the unchanged baseline.

## Goal and observed implementation

The goal is useful persistent prediction and adaptation in a finite 3D medium on real continuous OWT, including first-pass prediction, recovery speed and recovery plateau, and A-to-B-to-A retention. The strongest comparator is the current complete 3D individual under the same continuous evaluation and exposure budget; the fly line is an additional matched comparator. Production remains below the existing 3900 MiB dedicated-memory guard.

The HTML currently starts 64 uniformly seeded display tracers. Its RK2 playback uses `.14 / max(p90_current_norm, local_current_norm)`, with frame duration capped at .05 seconds. Neither tracer count nor playback rate is a physical particle count or physical integration time. Their direction comes from the sampled discrete energy current. Storage hotspots and installed capacity are separate real observables.

Current slow structure is a continuous material field and Gaussian structural posterior. Three active capacities plus idle share a local finite-resource simplex. The slow posterior receives task gradients, Gaussian KL, a maintenance constraint and an inherited OU prior. Effective propagation also has state-dependent conduction and short-term plasticity: the system already has fast runtime adaptation. Visual motion cannot establish that Adam is the structural bottleneck.

## Three framings and competing hypotheses

1. Optimization: useful structural credit exists but is poorly conditioned or cancelled by inherited-prior/maintenance pressure. Prediction: a geometry-aware update improves equal-budget task progress while preserving the same representable structures. This alone would not establish a new biological mechanism.
2. Constitutive adaptation: local history of useful delivery is absent from the installed material state, although fast utilization is present. Prediction: value-guided persistent resource redistribution retains useful routes across context changes. This opens a different state space, rather than only changing an optimizer.
3. Representation: a finite global Fourier coefficient chart couples distant material changes. Prediction: a local constitutive state can change one route without globally changing unrelated routes. Any new local state must explicitly preserve the continuous-field/refinement contract.
4. Objective mismatch: increasing raw current favors irrelevant background, standing/recurrent waves or premature delivery. Prediction: an activity-only growth rule can strengthen high-flow routes while predictive quality deteriorates; signed task value separates useful and harmful use.

These are competing hypotheses. Existing observations establish heterogeneity, not which hypothesis limits capability.

## What the literature supports

- Bejan's constructal principle supplies a design motivation: finite flow systems evolve toward easier access. It does not specify a predictive-learning objective or prove a universal performance theorem. [Bejan, 1996](https://onlinelibrary.wiley.com/doi/abs/10.1002/atr.5670300207).
- A tube-thickening feedback model has explicit precedents in adaptive Physarum transport. Its feedback strength affects its behavior. [Tero, Kobayashi and Nakagaki, 2007](https://pubmed.ncbi.nlm.nih.gov/17069858/).
- Local transport adaptation can optimize a specified energetic cost; fluctuating demand can yield hierarchical loops as well as trees. [Hu and Cai, 2013](https://pubmed.ncbi.nlm.nih.gov/24116821/).
- The cited erosion paper couples landscape evolution with surface-water conservation. It supports mutual flow/structure evolution, rather than a direct identity with predictive credit. [Somfai and Sander, 1997](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.56.R5).

## Preserve the wave medium

The current transport pair is schematically

\[
\dot f=-\nabla\cdot(Bq),\qquad \dot q=-B^T\nabla f,
\qquad E=\tfrac12\int(|f|^2+|q|^2)\,dx.
\]

On the periodic domain the two transport terms cancel in the energy derivative, including when the same instantaneous B is state-dependent. The existing discretization uses paired energy-preserving rotations. This says nothing by itself about the full tangent dynamics or task quality.

For one scalar feature the energy current is f Bq, summed over features for the implemented multi-channel field. q has its own persistent dynamics and phase. The resistive relation J=G delta_mu is a different constitutive closure and must not replace the wave pair merely by analogy.

Changing installed B in fixed spatial/state coordinates can preserve this pairing without erasing f or q. Reinterpreting a direction change as a change of state basis would require the corresponding state transformation and connection terms. Direction independence and positive definiteness need their own constraints; unit row norms alone do not prevent coincident directions.

## Minimal candidate: capacity evolution before directional rewiring

Keep current directions initially. Let p(x) be three active shares plus idle, with p_a>0 and sum_a p_a=1; c_a=R(x)p_a retains the existing local capacity bound. At a fixed event/history, define a structural objective

\[
F_t(p)=L_t(p)+\lambda m^T p+\kappa D_{KL}(p\Vert\pi_t),
\]

where L_t is causal predictive loss at the actual read time, m is explicitly justified maintenance (idle has zero installed maintenance), and pi_t is an inherited full-support structural reference. This is a proposed task-and-resource objective. Calling it a variational evidence bound additionally requires a normalized generative model; simplex shares alone are resource fractions, not a Bayesian structural posterior.

With g_t=grad_p L_t at the current p, a mirror/proximal step is

\[
p^+=\arg\min_{p\in\Delta_4}\{(g_t+\lambda m)^Tp+
\eta^{-1}D_{KL}(p\Vert p_t)+\kappa D_{KL}(p\Vert\pi_t)\}.
\]

Its unique interior solution is

\[
p_a^+=\frac{p_{t,a}^{1/(1+\eta\kappa)}\pi_{t,a}^{\eta\kappa/(1+\eta\kappa)}
\exp[-\eta(g_{t,a}+\lambda m_a)/(1+\eta\kappa)]}{Z}.
\]

This gives finite-resource competition, positive shares from positive references and finite scores, and a bounded change preference. It does not guarantee sparse trees, a minimum useful capacity, or lower nonlinear task loss for arbitrary step sizes. Numerical underflow needs log-space evaluation.

For the continuous exact-gradient limit with r_a=partial F/partial p_a,

\[
\tau\dot p_a=-p_a(r_a-\bar r),\quad \bar r=\sum_b p_b r_b,
\quad \sum_a\dot p_a=0,
\quad \dot F=-\tau^{-1}\sum_a p_a(r_a-\bar r)^2\le0.
\]

The last identity assumes a fixed differentiable F. A changing stream adds partial_t F. An approximate eligibility-trace score does not inherit the exact-gradient proof. A finite-window gradient is exact only for the graph actually retained. These are separate algorithm choices, not interchangeable guarantees.

Local wave-use statistics can be persistent bounded-size traces, but they identify use, not usefulness. The task-value signal must distinguish useful prediction delivery from high-amplitude irrelevant/recurrent activity. Initially reuse actual task derivatives to establish the capacity interface, rather than hiding a new credit algorithm inside growth. Assign one writer to each structural variable: Adam and the constitutive rule must not both independently update the same capacity.

The closed form applies to unconstrained local shares. A field obtained this way generally leaves the current finite Fourier chart. Before implementation choose and justify a continuous local-state representation or an explicit chart-constrained projection; such a projection does not inherit the unconstrained solution or descent proof automatically. This is the main remaining mathematical interface, together with maintenance/prior/time-scale calibration.

## Hopf algebra: composition of valid structural operations

For a rooted branching module T, the Connes-Kreimer coproduct is

\[
\Delta T=T\otimes1+1\otimes T+\sum_{C\text{ proper admissible}}P_C(T)\otimes R_C(T).
\]

P is a pruned forest and R is the retained rooted part; admissible cuts meet each root-to-leaf path at most once. Forest multiplication, cuts and grafting organize hierarchical decomposition and composition. [Connes and Kreimer, 1998](https://arxiv.org/abs/hep-th/9808042).

To describe this medium, decorate operations with physical port signatures, delay/phase response, capacity and state-transfer maps. First define valid split, merge and replacement operations with compatible interfaces. Recurrent routes need a graph representation beyond rooted trees. The antipode is a convolution inverse, not automatic physical rollback, energy conservation or memory protection. Algebra can then manage compositions of already valid operations; it is not an additional predictive-learning force.

## Predictions, review and decision

The independent reviewer checked the actual capacity and wave contracts. The review rejected raw-flow-as-value, assumed direction independence, diffusion-to-wave transfer, and treating Hopf antipodes as physical inverses. It recommended capacity-only value-guided competition before changing directions.

Analytic predictions: total installed-plus-idle capacity is conserved; lower objective sensitivity receives resources relative to higher sensitivity at fixed priors/cost; uniform sensitivities do not create task-driven differentiation; prior/cost can reverse a raw-flow preference. The same persistent f,q survive structural updates and paired transport retains its instantaneous energy cancellation.

Capability prediction remains exploratory: matched real-stream evaluation should improve predictive delivery, recovery or retention without an unacceptable tradeoff in the other pillars. Shape alone, higher anisotropy, higher traffic and lower internal physical MSE are not capability endpoints.

Next action: close the continuous representation and task-credit-to-capacity interface, including a single update owner, before proposing a production patch. No new experiment is authorized or started by this note. A subsequent plan must use matched birth/continuation histories, identical real OWT exposure and optimizer budgets, at least the user's 3000 joint-update credibility floor and explicit convergence evidence. Stop on broken resource/energy/continuation invariants or the memory limit. At adequate budget, absent task benefit with higher cost rejects the candidate; a budget-limited nonconverged result is unresolved. Keep the present working individual as the strongest baseline.
