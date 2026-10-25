import CGBAxioms.EnergyBudget

/-!
# Information acquisition as a conditional thermodynamic contract

The general medium axioms do not assign physical energy to a recorded symbol.
This file therefore separates useful predictive information from logically
irreversible erasure and makes the thermodynamic lower bound an explicit model
contract.  It does not turn a design objective into a new universal axiom.
-/

namespace CGB.InformationThermodynamics

structure RecorderContract where
  usefulBits : ℝ
  erasedBits : ℝ
  dissipation : ℝ
  bathTemperature : ℝ
  erasureCost : ℝ
  feasible : Prop
  useful_nonneg : 0 ≤ usefulBits
  erased_nonneg : 0 ≤ erasedBits
  dissipation_nonneg : 0 ≤ dissipation
  temperature_pos : 0 < bathTemperature
  erasure_cost_nonneg : 0 ≤ erasureCost
  landauer_floor : bathTemperature * erasureCost * erasedBits ≤ dissipation

/-- The quantity to minimize after a capability constraint has been fixed. -/
noncomputable def energyPerUsefulBit (r : RecorderContract) : ℝ :=
  if 0 < r.usefulBits then r.dissipation / r.usefulBits else 0

/-- Pareto efficiency is a mechanism-level design property, not a law of all
living systems.  Alternatives must meet the same useful-information target. -/
def ParetoEfficient (r : RecorderContract) : Prop :=
  r.feasible ∧ ∀ s : RecorderContract,
    s.feasible → r.usefulBits ≤ s.usefulBits → r.dissipation ≤ s.dissipation

theorem erased_information_floor (r : RecorderContract) :
    r.bathTemperature * r.erasureCost * r.erasedBits ≤ r.dissipation :=
  r.landauer_floor

theorem positive_erasure_requires_positive_dissipation
    (r : RecorderContract) (hcost : 0 < r.erasureCost)
    (hbits : 0 < r.erasedBits) :
    0 < r.dissipation := by
  have hfloor := r.landauer_floor
  have hprod : 0 < r.bathTemperature * r.erasureCost * r.erasedBits := by
    exact mul_pos (mul_pos r.temperature_pos hcost) hbits
  linarith

theorem efficient_recorder_is_not_free
    (r : RecorderContract) (he : 0 < r.erasedBits)
    (hc : 0 < r.erasureCost) (_hp : ParetoEfficient r) :
    0 < r.dissipation := by
  exact positive_erasure_requires_positive_dissipation r hc he

end CGB.InformationThermodynamics
