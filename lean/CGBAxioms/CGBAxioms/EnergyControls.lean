import CGBAxioms.EnergyBudget
import CGBAxioms.MaterialSourceControls

/-!
# Energy-interpretation controls on the unchanged material process

The same material transport admits different energy calibrations. The positive
scale sends two units in with the exterior token, one into a work receiver and
one into a heat receiver. These are accounting interpretations, not measured
chemical energies or a complete microscopic thermal realization.
-/

open Finset

namespace CGB.EnergyControls

open Energetics MaterialSourceControls

def internal (a : ℝ) (p : Phase) : ℝ := if p = 1 then 2 * a else 0
def exterior (a : ℝ) (p : Phase) : ℝ := if p = 0 then 2 * a else 0
def workReceiver (a : ℝ) (p : Phase) : ℝ := if p = 2 then a else 0
def heatReceiver (a : ℝ) (p : Phase) : ℝ := if p = 2 then a else 0

theorem total_partition (a : ℝ) (p : Phase) :
    internal a p + exterior a p + workReceiver a p + heatReceiver a p = 2 * a := by
  fin_cases p <;> simp [internal, exterior, workReceiver, heatReceiver, two_mul]

/-- Ports are differences of explicit receivers/reservoirs. Stored energy is
counted once; the outgoing token's chemical energy is not added again. -/
noncomputable def account (a : ℝ) : Account physical where
  stored _ := internal a
  drive _ _ _ _ := 0
  load _ p _ p' := workReceiver a p' - workReceiver a p
  matter _ p _ p' := exterior a p - exterior a p'
  heat _ p _ p' := heatReceiver a p - heatReceiver a p'
  balance _ p _ p' _ := by
    have h0 := total_partition a p
    have h1 := total_partition a p'
    linarith

/-- On every supported event, the material-energy port is carried by the same
exterior token identified in the material-source certificate. -/
theorem material_energy_binding (a : ℝ) (t : ℕ) (p p' : Phase) (e : Option Unit)
    (hp : 0 < physical.step t p e p') :
    (account a).matter t p e p' =
      2 * a * (MaterialSource.moved (owner p) (owner p') .outside .source : ℝ) := by
  fin_cases p <;> fin_cases p' <;> cases e <;>
    norm_num [physical, Interaction.Machine.deterministic, event, next] at hp
  all_goals simp [account, exterior, MaterialSource.moved, owner]

/-- Nonnegative mean production for the declared T=1, zero-system-entropy
accounting interpretation along the actual process; not a derivation of the
microscopic entropy function or of local detailed balance. -/
theorem isothermal_account (a : ℝ) (ha : 0 ≤ a) :
    Isothermal physical (account a) (0 : Phase) 1
      (fun _ _ => 0) (fun _ _ _ _ => 0) := by
  refine ⟨by norm_num, ?_⟩
  intro t
  have hq : flux physical (0 : Phase) (account a).heat t ≤ 0 := by
    apply Finset.sum_nonpos
    intro p _
    apply Finset.sum_nonpos
    intro e _
    apply Finset.sum_nonpos
    intro p' _
    apply mul_nonpos_of_nonneg_of_nonpos ((law_probability physical (0 : Phase) t).1 p)
    fin_cases p <;> fin_cases p' <;> cases e <;>
      simp [physical, Interaction.Machine.deterministic, event, next, account, heatReceiver]
    all_goals linarith
  have hz : flux physical (0 : Phase) (fun _ _ _ _ => 0) t = 0 := by simp [flux]
  simpa only [production, zero_sub, div_one, hz, sub_zero] using neg_nonneg.mpr hq

/-- Fixed identities and identical counts do not select chemical energy or
work. The positive interpretation has a work receiver AND a heat receiver. -/
theorem same_material_different_energy (t : ℕ) :
    MaterialSource.moved (owner 1) (owner 2) .source .body = 1 ∧
    (account 0).load t 1 (some ()) 2 = 0 ∧
    (account 1).load t 1 (some ()) 2 = 1 ∧
    (account 1).heat t 1 (some ()) 2 = -1 ∧
    (account 1).matter t 0 none 1 = 2 := by
  norm_num [MaterialSource.moved, owner, account, internal, exterior, workReceiver, heatReceiver]

end CGB.EnergyControls
