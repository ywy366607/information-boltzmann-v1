import CGBAxioms.LearningMaintenanceEnergy
import CGBAxioms.LearningMaintenanceControls

/- Exact mathematical review of the delivered contracts. No training, fitted
parameters, sampled trajectories or learning-capability estimates are used. -/

open CGB CGB.LearningMaintenance

namespace LearningMaintenanceReview20261024

-- Both members of the paired intervention satisfy the same core and clock.
example : body.Axioms ∧ frozen.Axioms ∧
    body.EventClocked 1 ∧ frozen.EventClocked 1 :=
  ⟨admission.1, (plant_admission frozenNext).1,
    admission.2, (plant_admission frozenNext).2⟩

example (q : State) (h : H) :
    memory (next (if h = 0 then 1 else 2) q) =
      CausalLearning.advance possible (memory q) () (policy q) (if h = 0 then 1 else 2) :=
  (paid_learning q h).1

-- Equal feeding stimuli cannot externally echo the retained branch identity.
example : sense 3 = sense 4 ∧ next 3 2 = 2 ∧ next 3 3 = 3 := by decide

-- The actual world reads the acquired action; swapping it loses resource access.
example : moves 3 (fun _ => 2) 3 = 1 ∧ moves 3 (fun _ => 3) 5 = 1 ∧
    moves 4 (fun _ => 3) 4 = 1 ∧ moves 4 (fun _ => 2) 5 = 1 := by
  have ha : policy 2 = 0 := by decide
  have hb : policy 3 = 1 := by decide
  norm_num [moves, ha, hb]

-- The write-disabled body retains MORE fuel at this point, yet fails later.
example : body.init = frozen.init ∧ cost (next 1 1) = 1 ∧
    cost (frozenNext 1 1) = 2 := by decide

example (n : ℕ) :
    Exchange.survival body world he ledger false (fun _ => world.sense) n = 1 :=
  natural_survival n

example (n : ℕ) (hn : 3 ≤ n) :
    Exchange.survival body world he ledger true (fun _ => world.sense) n ≤ 1 / 2 :=
  cut_survival n hn

example (n : ℕ) (hn : 3 ≤ n) :
    Exchange.survival frozen world LearningMaintenanceControls.hf
      LearningMaintenanceControls.ledger false (fun _ => world.sense) n = 0 :=
  LearningMaintenanceControls.frozen_survival n hn

example : Exchange.Persists body world he ledger 0 ∧
    Exchange.SelfMaintains body world he ledger := maintenance_witness

example (cut : Bool) (n : ℕ) (z : LearningMaintenanceEnergy.S) :
    Energetics.law (LearningMaintenanceEnergy.process cut)
      LearningMaintenanceEnergy.initial n z = if cut then
        body.lawCut world (fun _ => world.sense) n z.1 z.2 else
        body.law world (fun _ => world.sense) n z.1 z.2 :=
  LearningMaintenanceEnergy.actual_law cut n z

example (cut : Bool) (n : ℕ) :
    Energetics.mean (Energetics.law (LearningMaintenanceEnergy.process cut)
      LearningMaintenanceEnergy.initial n) ((LearningMaintenanceEnergy.account cut).stored n) +
        ∑ t ∈ Finset.range n, Energetics.flux (LearningMaintenanceEnergy.process cut)
          LearningMaintenanceEnergy.initial (LearningMaintenanceEnergy.port outgoing) t =
      2 + ∑ t ∈ Finset.range n, Energetics.flux (LearningMaintenanceEnergy.process cut)
        LearningMaintenanceEnergy.initial (LearningMaintenanceEnergy.port incoming) t :=
  LearningMaintenanceEnergy.availability_budget cut n

-- Spent plasticity stays spent even if a contradictory feedback arrives later.
example : next 2 2 = 2 ∧ next 1 3 = 3 ∧
    (memory 2).reserve.val = 0 ∧ (memory 3).reserve.val = 0 := by decide

end LearningMaintenanceReview20261024
