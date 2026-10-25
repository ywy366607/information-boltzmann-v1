import CGBAxioms.LearningMaintenance

/-! A paired write-disabled control: same initial fuel, sensing, world and clock.
Only the feedback write is suppressed. This is a mathematical intervention. -/

open Finset

namespace CGB.LearningMaintenanceControls
open LearningMaintenance

abbrev hf : frozen.EventClocked world.clock := (plant_admission frozenNext).2

noncomputable def ledger : frozen.MaterialLedger where
  incoming _ s x := incoming (s ()) (x ())
  outgoing _ s x := if s () = 0 ∨ x () = 0 then 0 else if s () = 4 then cost (x ()) else 1
  balance _ s x _ := by
    change cost (frozenNext (s ()) (x ())) +
      (if s () = 0 ∨ x () = 0 then 0 else if s () = 4 then cost (x ()) else 1) =
      cost (x ()) + incoming (s ()) (x ())
    generalize hs : s () = a
    generalize hx : x () = b
    fin_cases a <;> fin_cases b <;> norm_num [frozenNext, incoming, cost]
  closed_zero a s x h _ := by
    have hn : ¬ frozen.Closed a := by simp [Medium.Closed, frozen, plant]
    exact False.elim (hn h)

private theorem sum_configs (f : (Unit → State) → ℝ) :
    ∑ x, f x = f (fun _ => 0) + f (fun _ => 1) + f (fun _ => 2) + f (fun _ => 3) := by
  calc
    _ = ∑ a : State, f (fun _ => a) :=
      Fintype.sum_equiv (Equiv.funUnique Unit State) _ _ (by intro x; congr 1)
    _ = _ := by
      have h : (univ : Finset State) = {0, 1, 2, 3} := by decide
      rw [h]; simp; ring

private theorem kernel_table (s : Unit → Stim) (x y : State) :
    frozen.kernel 1 s (fun _ => x) (fun _ => y) =
      if y = frozenNext (s ()) x then 1 else 0 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs]
  rw [← frozen.eventStep_forget, Fintype.sum_option]
  generalize h : s () = a
  fin_cases a <;> fin_cases x <;> fin_cases y <;>
    norm_num [Medium.eventStep, frozen, plant, frozenNext, funext_iff, h]

private noncomputable def density : ℕ → State → W → ℝ
  | 0, a, w => if a = 1 ∧ w = 0 then 1 else 0
  | 1, a, w => if a = 1 ∧ (w = 1 ∨ w = 2) then 1 / 2 else 0
  | 2, a, w => if a = 1 ∧ (w = 3 ∨ w = 4) then 1 / 2 else 0
  | 3, a, w => if a = 1 ∧ w = 5 then 1 else 0
  | _ + 4, a, w => if a = 0 ∧ w = 5 then 1 else 0

private theorem policy_value (q : State) :
    policy q = if q = 2 then 0 else if q = 3 then 1 else 2 := by
  fin_cases q <;> decide

set_option maxHeartbeats 2000000 in
-- Expand the finite paired-control rows before the all-time induction.
private theorem frozen_law (n : ℕ) (a : State) (w : W) :
    frozen.law world (fun _ => world.sense) n (fun _ => a) w = density n a w := by
  induction n generalizing a w with
  | zero => simp [Medium.law, density, frozen, plant, world, funext_iff]
  | succ n ih =>
    change (∑ x : Unit → State, ∑ v : W,
      frozen.law world (fun _ => world.sense) n x v * moves v x w *
        frozen.kernel 1 (sense v) x (fun _ => a)) = _
    simp only [sum_configs, ActiveLoop.fin6_sum, ih]
    rcases n with _ | n
    · fin_cases a <;> fin_cases w <;>
        norm_num [density, sense, moves, policy_value, kernel_table, frozenNext]
    · rcases n with _ | n
      · fin_cases a <;> fin_cases w <;>
          norm_num [density, sense, moves, policy_value, kernel_table, frozenNext]
      · rcases n with _ | n
        · fin_cases a <;> fin_cases w <;>
            norm_num [density, sense, moves, policy_value, kernel_table, frozenNext]
        · rcases n with _ | n
          · fin_cases a <;> fin_cases w <;>
              norm_num [density, sense, moves, policy_value, kernel_table, frozenNext]
          · fin_cases a <;> fin_cases w <;>
              norm_num [density, sense, moves, policy_value, kernel_table, frozenNext]

set_option maxHeartbeats 1600000 in
-- Resolve the event labels in the absorbing resource-loss state.
private theorem starve_recovery (t k : ℕ) (a : State) :
    Exchange.recovery frozen world hf ledger false (fun _ => world.sense)
      t k ((fun _ => a), (5 : W)) = 0 := by
  induction k generalizing t a with
  | zero => rfl
  | succ k ih =>
    simp only [Exchange.recovery, EventChannels.Kernel.firstHit, Fintype.sum_prod_type]
    simp only [Fintype.sum_option, Fintype.sum_unique, sum_configs, ActiveLoop.fin6_sum]
    fin_cases a <;> norm_num [Medium.jointEventKernel, Medium.eventWorldStep,
      Medium.eventStep, frozen, plant, frozenNext, world, sense, moves, policy_value,
      Exchange.intake, ledger, incoming, cost, funext_iff]
    all_goals exact ih (t + 1) 0

private theorem starve_alive (n : ℕ) (a : State) :
    Exchange.alive frozen world hf ledger false (fun _ => world.sense)
      n ((fun _ => a), (5 : W)) = 0 := by
  classical
  unfold Exchange.alive Viability.alive
  rw [show world.horizon = 3 from rfl, starve_recovery]
  norm_num [world]

/-- With writing disabled the initial two fuel units cannot preserve access to
food. Viability is lost at tick three; the remaining body empties one tick later. -/
theorem frozen_survival (n : ℕ) (hn : 3 ≤ n) :
    Exchange.survival frozen world hf ledger false (fun _ => world.sense) n = 0 := by
  change (∑ x : Unit → State, ∑ w : W,
    Exchange.alive frozen world hf ledger false (fun _ => world.sense) n (x, w) *
      frozen.law world (fun _ => world.sense) n x w) = 0
  simp only [sum_configs, ActiveLoop.fin6_sum, frozen_law]
  rcases n with _ | n
  · omega
  · rcases n with _ | n
    · omega
    · rcases n with _ | n
      · omega
      · rcases n with _ | n
        · norm_num [density]
          exact starve_alive _ _
        · norm_num [density]
          exact starve_alive _ _

theorem frozen_not_persists (δ : ℝ) : ¬ Exchange.Persists frozen world hf ledger δ := by
  rintro ⟨hd, hp⟩
  have h := hp 3
  rw [frozen_survival 3 (by omega)] at h
  linarith

end CGB.LearningMaintenanceControls
