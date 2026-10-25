import CGBAxioms.PortContracts
import CGBAxioms.RenewableLearning

/-!
# Coarse admission candidate for a sponge-like continuous body

This is a structural interface witness only. It does not claim that a real
sponge has this two-level state space, nor does it provide a carrier, energy,
or causal-delivery realization.
-/

namespace CGB.SpongeContractInstance
open CGB.PortContracts CGB.RenewableLearning

abbrev State := Fin 2
abbrev Input := Signal × Bool

def port : PortSplit Input Signal Bool where
  encode := Prod.mk
  sensor := Prod.fst
  resource := Prod.snd
  sensor_encode _ _ := rfl
  resource_encode _ _ := rfl

def empty : State := 0

def rank (x : State) : ℕ := x.val

def step (_ : State) : State := empty

def reserve : RankedReserve State where
  empty := fun x => x = empty
  rank := rank
  step := step
  rank_empty := by
    intro x hx
    subst x
    rfl
  step_empty := by
    intro x hx
    subst x
    rfl
  strict_loss := by
    intro x hx
    fin_cases x
    · exact False.elim (hx rfl)
    · decide

def coarseAdmission : CoarseAdmission Input Signal Bool State := {
  ports := port
  reserve := reserve
}

theorem port_factorizes (s : Signal) (r : Bool) :
    coarseAdmission.ports.sensor
        (protocol coarseAdmission.ports (fun _ => s) (fun _ => r) 0) = s ∧
    coarseAdmission.ports.resource
        (protocol coarseAdmission.ports (fun _ => s) (fun _ => r) 0) = r := by
  exact admission_port_factorizes coarseAdmission s r

theorem reserve_contract :
    (∀ x, coarseAdmission.reserve.empty x → coarseAdmission.reserve.rank x = 0) ∧
    (∀ x, coarseAdmission.reserve.empty x →
      coarseAdmission.reserve.empty (coarseAdmission.reserve.step x)) ∧
    (∀ x, ¬coarseAdmission.reserve.empty x →
      coarseAdmission.reserve.rank (coarseAdmission.reserve.step x) <
        coarseAdmission.reserve.rank x) := by
  exact admission_reserve_contract coarseAdmission

end CGB.SpongeContractInstance
