import CGBAxioms.RenewableLearningLife
import CGBAxioms.RecoveryCriteria

/-! Close the ORIGINAL distribution-dependent marginal-ghost comparison.
No learning/storage intervention is substituted for that registered cut. -/

open Finset

namespace CGB.RenewableLearningClosure
open RenewableLearning

noncomputable abbrev body := plant true true
abbrev ec := he true true
noncomputable abbrev material := ledger true true

private theorem policy_table (q : Cell) : policy q =
    match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

private theorem sum_cells (f : Cell → ℝ) : ∑ q, f q =
    f .vacuum + f .search + f .aSpent + f .bSpent +
      f .aReady + f .bReady + f .aHeld + f .bHeld := by
  have h : (univ : Finset Cell) =
      {.vacuum, .search, .aSpent, .bSpent, .aReady, .bReady, .aHeld, .bHeld} := by decide
  rw [h]; simp; ring

private theorem sum_configs (f : (Unit → Cell) → ℝ) :
    ∑ x, f x = f (fun _ => .vacuum) + f (fun _ => .search) +
      f (fun _ => .aSpent) + f (fun _ => .bSpent) + f (fun _ => .aReady) +
      f (fun _ => .bReady) + f (fun _ => .aHeld) + f (fun _ => .bHeld) := by
  rw [Fintype.sum_equiv (Equiv.funUnique Unit Cell) _ (fun q => f (fun _ => q))
    (by intro x; congr 1)]
  exact sum_cells _

private theorem sum_env (f : Env → ℝ) : ∑ w, f w =
    f .feedA + f .feedB + f .missA + f .missB + f .probeA + f .probeB +
      f .answerA + f .answerB + f .trap := by
  have h : (univ : Finset Env) =
      {.feedA, .feedB, .missA, .missB, .probeA, .probeB, .answerA, .answerB, .trap} := by decide
  rw [h]; simp; ring

set_option maxHeartbeats 1800000 in
-- Reduce every finite one-cell input/state row before the cut induction.
private theorem kernel_table (s : Unit → Signal) (q r : Cell) :
    body.kernel 1 s (fun _ => q) (fun _ => r) =
      if r = next true true (s ()) q then 1 else 0 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs, ← body.eventStep_forget, Fintype.sum_option]
  generalize h : s () = a
  cases a <;> cases q <;> cases r <;>
    norm_num [Medium.eventStep, body, plant, next, funext_iff, h]

private noncomputable def early : ℕ → Cell → Env → ℝ
  | 0, q, w => if q = .search ∧ w = .probeA then 1 else 0
  | 1, q, w => if q = .search ∧ w = .answerA then 1 else 0
  | 2, q, w => if q = .aSpent ∧ w = .feedA then 1 else 0
  | 3, q, w => if q = .aReady ∧ (w = .feedA ∨ w = .feedB) then 1 / 2 else 0
  | 4, q, w => if q = .aReady then
      (if w = .missB then 1 / 2 else if w = .feedA ∨ w = .feedB then 1 / 4 else 0) else 0
  | 5, q, w => if q = .search ∧ w = .probeB then 1 / 2 else if q = .aReady then
      (if w = .missB then 1 / 4 else if w = .feedA ∨ w = .feedB then 1 / 8 else 0) else 0
  | _, _, _ => 0

private def ghost (n : ℕ) : Cell := if n < 2 then .search else if n = 2 then .aSpent else .aReady

/-- Until tick four the body marginal is a point mass, so the ORIGINAL cut
transition is derived from that marginal, not assigned a convenient law. -/
private theorem early_move (n : ℕ) (hn : n ≤ 4)
    (hi : ∀ q w, body.lawCut world (fun _ => world.sense) n (fun _ => q) w = early n q w)
    (x : Unit → Cell) (w v : Env) :
    Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x w v =
      moves w (fun _ => ghost n) v := by
  change (∑ g : Unit → Cell, (∑ u : Env,
    body.lawCut world (fun _ => world.sense) n g u) * moves w g v) = _
  simp only [sum_configs, sum_env, hi]
  have hc : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by omega
  rcases hc with rfl | rfl | rfl | rfl | rfl <;> norm_num [early, ghost]

set_option maxHeartbeats 2600000 in
-- Induct through the five exact sparse cut rows, then stop enumerating time.
private theorem cut_early (n : ℕ) (hn : n ≤ 5) (q : Cell) (w : Env) :
    body.lawCut world (fun _ => world.sense) n (fun _ => q) w = early n q w := by
  induction n generalizing q w with
  | zero => simp [Medium.lawCut, early, body, plant, world, funext_iff]
  | succ n ih =>
    have hi := ih (by omega)
    have hm := early_move n (by omega) hi
    change (∑ x : Unit → Cell, ∑ v : Env,
      body.lawCut world (fun _ => world.sense) n x v *
        Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x v w *
          body.kernel 1 (sense v) x (fun _ => q)) = _
    simp only [sum_configs, sum_env, hi, hm]
    have hc : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by omega
    rcases hc with rfl | rfl | rfl | rfl | rfl <;>
      cases q <;> cases w <;>
        norm_num [early, ghost, moves, policy_table, kernel_table, sense, next]

private theorem move_five (x : Unit → Cell) (w v : Env) :
    Medium.cutMove world (body.lawCut world (fun _ => world.sense) 5) x w v =
      (1 : ℝ) / 2 * moves w (fun _ => .search) v +
      (1 : ℝ) / 2 * moves w (fun _ => .aReady) v := by
  change (∑ g : Unit → Cell, (∑ u : Env,
    body.lawCut world (fun _ => world.sense) 5 g u) * moves w g v) = _
  have hi (q : Cell) (w : Env) := cut_early 5 (by omega) q w
  simp only [sum_configs, sum_env, hi]
  norm_num [early]

private noncomputable def trapped (n : ℕ) : ℝ :=
  ∑ x : Unit → Cell, body.lawCut world (fun _ => world.sense) n x .trap

private theorem trapped_next (n : ℕ) : trapped (n + 1) =
    ∑ x : Unit → Cell, ∑ w : Env, body.lawCut world (fun _ => world.sense) n x w *
      Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x w .trap := by
  unfold trapped
  simp only [Medium.lawCut]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro x _
  rw [Finset.sum_comm]
  simp only [← Finset.mul_sum, Medium.kernel_sum, mul_one]

private theorem trapped_six : trapped 6 = 1 / 4 := by
  rw [show 6 = 5 + 1 from rfl, trapped_next]
  have hi (q : Cell) (w : Env) := cut_early 5 (by omega) q w
  simp only [sum_configs, sum_env, hi, move_five]
  norm_num [early, moves, policy_table]

private theorem trap_moves (x : Unit → Cell) (v : Env) :
    moves .trap x v = if v = .trap then 1 else 0 := by
  by_cases hq : x () = .vacuum <;> simp [moves, hq]

private theorem trap_world_step (cut : Bool) (t : ℕ) (x : Unit → Cell) (v : Env) :
    body.eventWorldStep world cut (fun _ => world.sense) t (x, Env.trap) v =
      if v = .trap then 1 else 0 := by
  cases cut with
  | false => simp [Medium.eventWorldStep, trap_moves]
  | true =>
    change (∑ g : Unit → Cell, (∑ u : Env,
      body.lawCut world (fun _ => world.sense) t g u) * moves .trap g v) = _
    simp only [trap_moves]
    have hs : (∑ g : Unit → Cell, ∑ u : Env,
        body.lawCut world (fun _ => world.sense) t g u) = 1 :=
      body.lawCut_sum world (fun _ => world.sense) t
    rw [← Finset.sum_mul, hs, one_mul]

private theorem trapped_monotone (n : ℕ) : trapped n ≤ trapped (n + 1) := by
  rw [trapped_next]
  unfold trapped
  apply Finset.sum_le_sum
  intro x _
  have hw : Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x .trap .trap = 1 :=
    trap_world_step true n x .trap
  calc
    _ = body.lawCut world (fun _ => world.sense) n x .trap *
        Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x .trap .trap := by
      rw [hw, mul_one]
    _ ≤ _ := Finset.single_le_sum (fun w _ => mul_nonneg
      (body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n x w)
      (body.eventWorldStep_nonneg world ec true _ n (x, w) .trap)) (Finset.mem_univ _)

private theorem trap_recovery (t k : ℕ) (x : Unit → Cell) :
    Exchange.recovery body world ec material true (fun _ => world.sense) t k (x, Env.trap) = 0 := by
  apply RecoveryCriteria.firstHit_zero_of_closed
    (body.jointEventKernel world ec true (fun _ => world.sense))
    (Exchange.intake body world material (fun _ => world.sense)) (fun z => z.2 = Env.trap)
  · intro t z e z' hg hp
    have hw := trap_world_step true t z.1 z'.2
    change 0 < body.eventWorldStep world true (fun _ => world.sense) t
      (z.1, z.2) z'.2 * body.eventStep 1 (sense z.2) z.1 e z'.1 at hp
    rw [hg] at hp
    rw [hw] at hp
    by_contra hn
    simp [hn] at hp
  · intro t z e z' hg _
    cases e <;> simp [Exchange.intake, material, ledger, hg, sense, incoming]
  · rfl

private theorem cut_alive (n : ℕ) (x : Unit → Cell) (w : Env) :
    Exchange.alive body world ec material true (fun _ => world.sense) n (x, w) ≤
      if w = .trap then 0 else 1 := by
  classical
  by_cases hw : w = .trap
  · subst w
    have hr := trap_recovery n 2 x
    unfold Exchange.alive Viability.alive
    rw [show world.horizon = 2 from rfl, hr]
    norm_num [world]
  · simp only [hw, ↓reduceIte]
    exact (Viability.alive_bounds _ _ _).2

private theorem cut_mass_bound (n : ℕ) :
    Exchange.survival body world ec material true (fun _ => world.sense) n ≤ 1 - trapped n := by
  classical
  calc
    _ ≤ ∑ x : Unit → Cell, ∑ w : Env, (if w = .trap then 0 else 1) *
        body.lawCut world (fun _ => world.sense) n x w := by
      apply Finset.sum_le_sum
      intro x _
      apply Finset.sum_le_sum
      intro w _
      exact mul_le_mul_of_nonneg_right (cut_alive n x w)
        (body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n x w)
    _ = _ := by
      have hs := body.lawCut_sum world (fun _ => world.sense) n
      change (∑ x : Unit → Cell, ∑ w : Env,
        body.lawCut world (fun _ => world.sense) n x w) = 1 at hs
      unfold trapped
      simp only [sum_configs, sum_env] at hs ⊢
      norm_num at hs ⊢
      linarith

/-- The original cut loses at least a quarter of viable mass by tick six;
absorption extends the bound to ALL later times, including future cut marginals. -/
theorem cut_survival (n : ℕ) (hn : 6 ≤ n) :
    Exchange.survival body world ec material true (fun _ => world.sense) n ≤ 3 / 4 := by
  have hm : (1 : ℝ) / 4 ≤ trapped n := by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
    induction k with
    | zero => simp [trapped_six]
    | succ k ih => exact le_trans (ih (by omega)) (trapped_monotone (6 + k))
  linarith [cut_mass_bound n]

theorem maintenance : Exchange.Persists body world ec material 0 ∧
    Exchange.SelfMaintains body world ec material := by
  refine ⟨RenewableLearningLife.persistence, 6, (1 : ℝ) / 4, by norm_num, ?_⟩
  intro n hn
  rw [RenewableLearningLife.natural_survival]
  linarith [cut_survival n hn]

/-- Exact seed mass for the absorbing failure region under the original cut. -/
theorem cut_seed_mass :
    (∑ x : Unit → Cell,
      body.lawCut world (fun _ => world.sense) 6 x Env.trap) = 1 / 4 :=
  trapped_six

end CGB.RenewableLearningClosure
