import CGBAxioms.SpongeContractInstance

open CGB.PortContracts CGB.RenewableLearning CGB.SpongeContractInstance

namespace SpongeContractInstanceReview20261024

example : ∃ A : CoarseAdmission Input CGB.RenewableLearning.Signal Bool State, True :=
  ⟨coarseAdmission, True.intro⟩

example (s : CGB.RenewableLearning.Signal) (r : Bool) := port_factorizes s r

example :
    (∀ x, coarseAdmission.reserve.empty x → coarseAdmission.reserve.rank x = 0) ∧
    (∀ x, coarseAdmission.reserve.empty x →
      coarseAdmission.reserve.empty (coarseAdmission.reserve.step x)) ∧
    (∀ x, ¬coarseAdmission.reserve.empty x →
      coarseAdmission.reserve.rank (coarseAdmission.reserve.step x) <
        coarseAdmission.reserve.rank x) :=
  reserve_contract

example (x : State) (hx : ¬coarseAdmission.reserve.empty x) :
    coarseAdmission.reserve.rank (coarseAdmission.reserve.step x) <
      coarseAdmission.reserve.rank x :=
  coarseAdmission.reserve.strict_loss x hx

end SpongeContractInstanceReview20261024
