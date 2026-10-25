import CGBAxioms.PortContracts
import CGBAxioms.PortSeparation

/-! The current 24-state maintenance witness is admitted through the coarse
contracts. This file deliberately does not construct another carrier identity
table. -/

namespace CGB.PortContractInstances
open PortContracts PortSeparation

abbrev State := PortSeparation.State
abbrev Input := PortSeparation.Input

def sensoryResourceSplit : PortSplit Input SensoryMaintenance.Stim Bool where
  encode := Prod.mk
  sensor := Prod.fst
  resource := Prod.snd
  sensor_encode _ _ := rfl
  resource_encode _ _ := rfl

def reserveRank : RankedReserve State where
  empty := fun x => x = PortSeparation.empty
  rank := PortSeparation.rank
  step := fun x => PortSeparation.next (none, false) x
  rank_empty := by
    intro x hx
    exact (PortSeparation.starvation_progress none x).2.1.mpr hx
  step_empty := by
    intro x hx
    subst hx
    exact PortSeparation.next_empty (none, false)
  strict_loss := by
    intro x hx
    exact (PortSeparation.starvation_progress none x).2.2.1 hx

def admission : CoarseAdmission Input SensoryMaintenance.Stim Bool State := {
  ports := sensoryResourceSplit
  reserve := reserveRank
}

theorem split_factorizes (s : SensoryMaintenance.Stim) (r : Bool) :
    sensoryResourceSplit.sensor (protocol sensoryResourceSplit (fun _ => s) (fun _ => r) 0) = s ∧
    sensoryResourceSplit.resource (protocol sensoryResourceSplit (fun _ => s) (fun _ => r) 0) = r := by
  exact ⟨protocol_sensor _ _ _ _, protocol_resource _ _ _ _⟩

theorem reserve_contract :
    (∀ x, reserveRank.empty x → reserveRank.rank x = 0) ∧
    (∀ x, reserveRank.empty x → reserveRank.empty (reserveRank.step x)) ∧
    (∀ x, ¬reserveRank.empty x → reserveRank.rank (reserveRank.step x) < reserveRank.rank x) := by
  exact ⟨reserveRank.rank_empty, reserveRank.step_empty, reserveRank.strict_loss⟩

theorem admission_contract :
    (∀ x, admission.reserve.empty x → admission.reserve.rank x = 0) ∧
    (∀ x, admission.reserve.empty x → admission.reserve.empty (admission.reserve.step x)) ∧
    (∀ x, ¬admission.reserve.empty x →
      admission.reserve.rank (admission.reserve.step x) < admission.reserve.rank x) := by
  exact admission_reserve_contract admission

end CGB.PortContractInstances
