import CGBAxioms.PortContractInstances

open CGB.PortContracts CGB.PortContractInstances

namespace PortContractInstancesReview20261024

example : ∃ P : PortSplit CGB.PortContractInstances.Input CGB.SensoryMaintenance.Stim Bool, True :=
  ⟨sensoryResourceSplit, True.intro⟩

example (s : CGB.SensoryMaintenance.Stim) (r : Bool) := split_factorizes s r

example :
    (∀ x, reserveRank.empty x → reserveRank.rank x = 0) ∧
    (∀ x, reserveRank.empty x → reserveRank.empty (reserveRank.step x)) ∧
    (∀ x, ¬reserveRank.empty x → reserveRank.rank (reserveRank.step x) < reserveRank.rank x) :=
  reserve_contract

example :
    (∀ x, admission.reserve.empty x → admission.reserve.rank x = 0) ∧
    (∀ x, admission.reserve.empty x → admission.reserve.empty (admission.reserve.step x)) ∧
    (∀ x, ¬admission.reserve.empty x →
      admission.reserve.rank (admission.reserve.step x) < admission.reserve.rank x) :=
  admission_contract

example (x : CGB.PortContractInstances.State) (hx : ¬reserveRank.empty x) :
    reserveRank.rank (reserveRank.step x) < reserveRank.rank x :=
  reserveRank.strict_loss x hx

example (x : CGB.PortContractInstances.State) (hx : reserveRank.empty x) :
    reserveRank.empty (reserveRank.step x) := reserveRank.step_empty x hx

end PortContractInstancesReview20261024
