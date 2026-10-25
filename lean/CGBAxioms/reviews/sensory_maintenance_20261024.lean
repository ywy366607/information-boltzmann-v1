import CGBAxioms.SensoryMaintenanceClosure
import CGBAxioms.SensoryMaintenanceCarrier

open CGB CGB.SensoryMaintenance
open CGB.RenewableLearning (Cell Signal Env)

namespace SensoryMaintenanceReview20261024

example : body.Axioms ∧ body.EventClocked 1 := admission

example : Fintype.card State = 24 ∧ Fintype.card Stim = 8 ∧ world.clock = 1 :=
  ⟨by decide, by decide, rfl⟩

example : body.calm () ≠ world.quiet () := equilibrium_distinct_from_loss

example (s : Stim) (x : State) := local_budgets s x

example (s : Stim) (x : State) (hs : s ≠ some .calm) (hx : 0 < cost x) :
    0 < outgoing s x := active_loss s x hs hx

example (x : State) : incoming none x = 0 := missing_no_intake x

example (q : Cell) (h : q ≠ .vacuum) := funded_two_tick_bridge q h

example (x : State) : burn x 3 = empty := burn_three x

example (n₀ n : ℕ) : Exchange.survival body world ec material false
    (sensoryProtocol n₀ 1) n = 1 := SensoryMaintenanceRecovery.one_tick_survival n₀ n

example : SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1 :=
  SensoryMaintenanceRecovery.one_tick_tolerance

example (n : ℕ) : Exchange.survival body world ec material false (fun _ => world.sense) n = 1 :=
  SensoryMaintenanceRecovery.natural_survival n

example (n : ℕ) (h : 7 ≤ n) : Exchange.survival body world ec material true
    (fun _ => world.sense) n ≤ 3 / 4 := SensoryMaintenanceClosure.cut_survival n h

example (n₀ T n : ℕ) (hT : 3 ≤ T) (hn : n₀ + 3 ≤ n) :
    Exchange.survival body world ec material false (sensoryProtocol n₀ T) n = 0 :=
  SensoryMaintenanceLimits.blackout_survival_zero n₀ T n hT hn

example (T : ℕ) (h : 3 ≤ T) : ¬ SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 T :=
  SensoryMaintenanceLimits.long_loss_not_tolerated T h

example (stim : ℕ → Stim) (p : ℕ → State) (hp : ∀ t, p (t+1) = next (stim t) (p t))
    (n : ℕ) (ha : ∀ t < n, stim t ≠ some .calm ∧ 0 < cost (p t))
    (hi : ∀ t < n, incoming (stim t) (p t) = 0) : n ≤ cost (p 0) :=
  SensoryMaintenanceLimits.no_input_active_duration stim p hp n ha hi

example (cut : Bool) (sched : ℕ → Env → Unit → Stim) (n : ℕ) :=
  SensoryMaintenanceLimits.fuel_budget cut sched n

example (stim : ℕ → Stim) (p : ℕ → State) (hp : ∀ t, p (t+1) = next (stim t) (p t))
    (n : ℕ) := path_credit_budget stim p hp n

example (q : Cell) (h : q ∈ ({Cell.aReady, Cell.bReady} : Finset Cell)) :
    RenewableLearning.memory (next (some .miss) (q,2)).1 =
      RenewableLearning.reconsider (RenewableLearning.memory q) (RenewableLearning.policy q) 0 ∧
    writes (some .miss) (q,2) = 1 := by
  cases q <;> simp at h <;> decide

example (h : RenewableLearning.H) (r : Fin 3) :
    RenewableLearning.memory (next (some (if h = 0 then .answerA else .answerB)) (.search,r)).1 =
      CausalLearning.advance RenewableLearning.possible (RenewableLearning.memory .search) ()
        (RenewableLearning.policy .search) (if h = 0 then 1 else 2) ∧
    writes (some (if h = 0 then .answerA else .answerB)) (.search,r) = 1 := by
  fin_cases h <;> fin_cases r <;> decide

example (cut : Bool) (sched : ℕ → Env → Unit → Stim) :=
  SensoryMaintenanceCarrier.realized cut sched

example : Fintype.card SensoryMaintenanceCarrier.Token = 24 := by decide

example (cut : Bool) (sched : ℕ → Env → Unit → Stim)
    (r : SensoryMaintenanceCarrier.Carrier cut sched) :
    MaterialSource.count (SensoryMaintenanceCarrier.owner cut sched r) .source = 8 :=
  SensoryMaintenanceCarrier.source_stock cut sched r

example (cut : Bool) (sched : ℕ → Env → Unit → Stim) (n : ℕ) :=
  SensoryMaintenanceCarrier.source_energy_budget cut sched n

example (cut : Bool) (sched : ℕ → Env → Unit → Stim) (t k : ℕ)
    (r : SensoryMaintenanceCarrier.Carrier cut sched) :=
  SensoryMaintenanceCarrier.recovery_preserved cut sched t k r

example (n₀ n : ℕ) : SensoryMaintenanceCarrier.score false (sensoryProtocol n₀ 1) n = 1 := by
  rw [SensoryMaintenanceCarrier.score_preserved, SensoryMaintenanceRecovery.one_tick_survival]

example (n₀ T n : ℕ) (hT : 3 ≤ T) (hn : n₀+3 ≤ n) :
    SensoryMaintenanceCarrier.score false (sensoryProtocol n₀ T) n = 0 := by
  rw [SensoryMaintenanceCarrier.score_preserved,
    SensoryMaintenanceLimits.blackout_survival_zero n₀ T n hT hn]

example (n : ℕ) (hn : 7 ≤ n) :
    SensoryMaintenanceCarrier.score true (fun _ => world.sense) n ≤ 3 / 4 := by
  rw [SensoryMaintenanceCarrier.score_preserved]
  exact SensoryMaintenanceClosure.cut_survival n hn

end SensoryMaintenanceReview20261024
