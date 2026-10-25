import CGBAxioms.PortContracts

open CGB.PortContracts

namespace PortContractsReview20261024

def splitBB : PortSplit (Bool × Bool) Bool Bool := {
  encode := fun s r => (s,r), sensor := Prod.fst, resource := Prod.snd,
  sensor_encode := by intros; rfl, resource_encode := by intros; rfl }

example : True := by
  let P : PortSplit (Bool × Bool) Bool Bool := splitBB
  exact True.intro

example (s : Bool) (r : Bool) :
    protocol splitBB (fun _ => s) (fun _ => r) 0 = (s,r) := by rfl

example : ∃ P : PortSplit (Bool × Bool) Bool Bool, True := by
  let P : PortSplit (Bool × Bool) Bool Bool := {
    encode := fun s r => (s,r), sensor := Prod.fst, resource := Prod.snd,
    sensor_encode := by intros; rfl, resource_encode := by intros; rfl }
  exact ⟨P, True.intro⟩

example : ∀ (n : ℕ), n ≤ n := by intro n; omega

end PortContractsReview20261024
