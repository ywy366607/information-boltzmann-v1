import CGBAxioms.LanguageLifeKernel
import CGBAxioms.CausalLearning
import CGBAxioms.GenerationPredictiveHarvest
import CGBAxioms.GenerationActiveInference

/-!
# Information economics: a small reusable basis

The project has many mechanism-level theorems, but their common economic core
is small.  This module records three contracts only:

1. a compressed causal carrier must preserve both readout and recursive
   transitions;
2. in the historical finite-candidate controller, an observation consumes
   write capacity exactly when it removes a live alternative, while agreement
   costs nothing;
3. predictive growth is bounded by the stream's cross-entropy/Gibbs bound.

The detailed memory, learning, active-inference and capacity results remain
corollaries or model-specific instances.  Its `Memory.reserve` is a bounded
write-credit convention for that one monotone candidate construction, not a
food store or a definition of an organism's information reserve.  No physical
entropy or new organism state is introduced here.
-/

namespace CGB.InformationEconomics

open Finset

/-- **Causal carrier basis.**  A finite state is a valid compressed carrier
exactly when it supports both the required readout and the encoded successor
for every future input. -/
theorem causal_carrier_contract_iff {H S C U : Type*}
    (encode : H → S) (required : H → C) (advance : H → U → H)
    (fallbackC : C) (fallbackS : S) :
    (∃ read : S → C, ∃ next : S → U → S,
      (∀ h, read (encode h) = required h) ∧
      (∀ h u, next (encode h) u = encode (advance h u))) ↔
      ((∀ h h', encode h = encode h' → required h = required h') ∧
       (∀ h h', encode h = encode h' → ∀ u,
          encode (advance h u) = encode (advance h' u))) := by
  constructor
  · rintro ⟨read, next, hread, hnext⟩
    refine ⟨(LanguageLifeKernel.history_readout_exists_iff
      encode required fallbackC).1 ⟨read, hread⟩, ?_⟩
    exact (LanguageLifeKernel.history_transition_exists_iff
      encode advance fallbackS).1 ⟨next, hnext⟩
  · rintro ⟨hread, hnext⟩
    obtain ⟨read, hread'⟩ :=
      (LanguageLifeKernel.history_readout_exists_iff
        encode required fallbackC).2 hread
    obtain ⟨next, hnext'⟩ :=
      (LanguageLifeKernel.history_transition_exists_iff
        encode advance fallbackS).2 hnext
    exact ⟨read, next, hread', hnext'⟩

/-- **Selective write basis.**  Agreement over all live candidates consumes no
write capacity; a funded observation that rules out a live candidate spends
one unit and removes that candidate. -/
theorem selective_write_contract
    {H C A O : Type*} [Fintype H] [DecidableEq H]
    (possible : CausalLearning.Support H C A O) {B : ℕ}
    (m : CausalLearning.Memory H B) (c : C) (a : A) (o : O) :
    ((∀ h ∈ m.candidates, possible h c a o = true) →
      CausalLearning.advance possible m c a o = m ∧
        CausalLearning.fee possible m c a o = 0) ∧
    (0 < m.reserve.val →
      ∀ h, h ∈ m.candidates → possible h c a o = false →
        h ∉ (CausalLearning.advance possible m c a o).candidates ∧
          CausalLearning.fee possible m c a o = 1) := by
  constructor
  · intro h
    exact CausalLearning.agreement_no_write possible m c a o h
  · intro hr h hm ho
    exact CausalLearning.paid_elimination possible m c a o h hr hm ho

/-- **Predictive-growth basis.**  Under the declared reinvestment contract,
growth decomposes into a fixed alphabet term minus cross-entropy and is bounded
by the stream entropy.  The contract is a niche assumption, not a general
physical axiom. -/
theorem predictive_growth_contract [Fintype B] [Nonempty B]
    (p c : B → ℝ) (hp0 : ∀ b, 0 ≤ p b) (hp : ∑ b, p b = 1)
    (hc : ∀ b, 0 < c b) (hcs : ∑ b, c b = 1) :
    PredictiveHarvest.growth p c =
        Real.log (Fintype.card B) - PredictiveHarvest.crossEntropy p c ∧
      PredictiveHarvest.growth p c ≤
        Real.log (Fintype.card B) - PredictiveHarvest.entropy p := by
  constructor
  · exact PredictiveHarvest.growth_eq_log_card_sub_crossEntropy p c hp hc
  · exact PredictiveHarvest.growth_le p c hp0 hp hc hcs

end CGB.InformationEconomics
