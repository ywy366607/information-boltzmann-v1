# Adaptive admission source review — 2026-10-08

## Outcome

The candidate makes the number of physical ticks between token injections state-dependent. It currently runs in standalone probes, while `FlyBPTTLearner` and the production entry point still use fixed `settle_ticks`. The source preserves the brain state between tokens and uses no next-target label in the admission decision. This is a useful admission-control candidate; its present execution is serial pulse/quiet/read/admit scheduling.

No GPU training or checkpoint update was performed for this review. Five existing CPU operator tests passed. Additional CPU numerical checks below are implementation counterexamples, not capability experiments.

## Findings and exact scope

1. **Joint-learning connection is absent from the smoke test.** In `scripts/ib/smoke_test_mcr2_dual_engine.py:173`, `Z_emitted.clone().detach().requires_grad_(True)` replaces the brain-connected latent with a new leaf. At lines 205–206 both decoder inputs and MCR input are detached. The first backward verifies latent/decoder derivatives; the second verifies CE decoder derivatives. Neither verifies MCR credit to writer, transmission, membrane or read projection. There is no optimizer step in this script. A joint-learning acceptance test must use connected physical latents and inspect these parameter groups.

2. **Previous numerical repairs are not selected.** The probe constructs `FlyReservoirLM` with legacy defaults at line 48, rather than selecting `detach_reset=True`, `surrogate_mode='threshold'`, `transmission_mode='incoming'`. The new full-window numerical acceptance therefore does not transfer to this probe or an eventual training loop automatically. Continuation should use the production configuration restoration contract and report missing/unexpected checkpoint keys rather than silently relying on `strict=False`.

3. **Low activity is not evidence of input arrival.** The production candidate admits when whole-brain membrane flux is below 0.036, or when a weighted scalar stops decreasing. In a CPU check with unchanged membrane, constant latent and uniform vocabulary logits, it admits on its second physical tick. The delayed ring and conductance states do not enter this flux. Thus an in-flight token can be waiting in the delay ring while the selected criterion considers the membrane settled. A readiness claim needs a causal information-arrival or residual-pending-response contract. The numerical guard of 14 total ticks is a budget cap, not proof that all relevant information arrived.

4. **The criterion is a heuristic score with explicit scales.** `8*flux + normalized_entropy - 0.05*coding_rate` supplies no joint generative density or variational posterior from which an ELBO/free energy can be derived. It is currently an admission score. There are explicit values 8, 0.05, 0.036, 0.5 and the 14-tick cap, as well as a three-history-sample requirement for the turning-point branch. They can be engineering parameters, but comments describing zero artificial constants or physically impossible premature exit exceed the implementation.

5. **Current MCR objective has a scale escape and omits compression.** With no trajectory argument, `R_cluster=0`; a nonzero singleton actually has positive coding rate. The original MCR2 objective includes the weighted class/cluster coding rate. In the trajectory overload, total coding rate uses final emitted points while cluster terms use all trajectory points: a different empirical population, requiring an explicit alternative objective rather than treating it as the standard partition formula. The smoke test supplies no trajectory clusters. Its RMSNorm latent has trainable gains, so amplitude is not fixed by merely calling RMSNorm. For an unchanged rank-8 identity matrix, scales 1/10/100 produce coding rates 6.43775/23.97585/42.38664 and losses -0.32189/-1.19879/-2.11933. A lower regularizer value alone therefore cannot certify richer coding. Standardize the representations on which this term is measured, and define the intended clustering before attaching it to joint learning.

6. **Reported gains use mismatched probes and units.** The archived 32-token aliasing probe saves 57.8% of ticks (14 to mean 5.90625), but reported NLL is 8.78733 fixed versus 9.12687 adaptive. These short frozen probes answer local response/cost questions only. This older script also enforces `t>=3`; the newer gatekeeper does not enforce that same branch uniformly. The fifth benchmark criterion is titled coding-volume growth but computes `R_vol` without using it in the decision: it actually uses `t>=3 and flux<=0.038`. The smoke report's 1.75x is a tick-count ratio (128 versus 224 ticks), not measured wall-time speedup. Per-tick vocabulary decoding, scalar GPU-to-CPU reads and Cholesky add real costs. Production S=14 performs 15 total ticks per input; the probes' `range(14)` performs 14 total ticks.

7. **Physical continuation semantics differ.** The saved source run uses `writer_baseline_clock='physical'`; the new probe leaves the sensory baseline unchanged on quiet steps (`base=st.baseline`). The production event helper decays it on every quiet physical tick. A same-individual comparison must preserve this state transition too.

## Single recommended next action

Keep adaptive admission as the sole candidate change and first wire it into the existing repaired event/BPTT/evaluation contract: selected numerical conventions, unchanged physical-state updates, connected gradients, explicit accepted-token/tick counters, actual elapsed-time measurements, and deterministic schedule replay under checkpointing. Cache the original stopping schedule for replay; recomputation should execute that schedule, not mutate a gatekeeper or make a fresh floating-point stopping decision. Resume must carry the required scheduling state if an input is still in flight.

Treat MCR as a separate optional learning-objective proposal until its population, normalization and gradient connection are repaired. This isolates whether adaptive admission improves the user's real goal: paired pre-update language NLL per resource budget, with persistent physical state and active learning evaluation. No long run is authorized or started by this review.

## Sources

- Local: `information_boltzmann/core/mcr2_rate_distortion.py`, `scripts/ib/smoke_test_mcr2_dual_engine.py`, `scripts/ib/benchmark_admission_criteria.py`, `scripts/ib/test_admission_agent_aliasing.py`, `information_boltzmann/core/fly_bptt_learning.py`.
- Existing reports: `present/smoke_test_mcr2_dual_engine_report.json`, `present/admission_agent_aliasing_report.json`, `present/admission_criteria_benchmark_report.json`.
- MCR2 original paper: https://arxiv.org/abs/2006.08558
- Official loss implementation, including cluster compression: https://raw.githubusercontent.com/ryanchankh/mcr2/master/loss.py

Independent source review requested from `/root/rtc_contract_review`; final review status is recorded in the research tree.
