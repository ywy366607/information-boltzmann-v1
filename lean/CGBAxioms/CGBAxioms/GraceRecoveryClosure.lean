import CGBAxioms.GraceRecovery

/-! Original marginal-ghost cut after diagnostic grace. The new failure
route exhausts actual body reserve one tick later; no new cut is defined. -/

open Finset

namespace CGB.GraceRecoveryClosure
open RenewableLearning

noncomputable abbrev body := plant true true
abbrev ec := GraceRecovery.ec
noncomputable abbrev world := GraceRecovery.world
noncomputable abbrev moves := GraceRecovery.moves
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

private theorem kernel_table (s : Unit → Signal) (q r : Cell) :
    body.kernel 1 s (fun _ => q) (fun _ => r) =
      if r = next true true (s ()) q then 1 else 0 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs, ← body.eventStep_forget, Fintype.sum_option, Fintype.sum_unique]
  by_cases hc : s () = .calm ∨ q = .vacuum
  · have hn : next true true (s ()) q = q := by
      rcases hc with h | h
      · by_cases hq : q = .vacuum <;> simp [next, h, hq]
      · simp [next, h]
    simp [Medium.eventStep, body, plant, hc, hn, funext_iff]
    simp only [eq_comm]
  · simp [Medium.eventStep, body, plant, hc, funext_iff]
    simp only [eq_comm]

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
  | zero => simp [Medium.lawCut, early, body, plant, world, GraceRecovery.world,
      GraceRecovery.worldWith, funext_iff]
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
        norm_num [early, ghost, moves, GraceRecovery.moves, RenewableLearning.moves, policy_table, kernel_table, sense, next]

private theorem move_five (x : Unit → Cell) (w v : Env) :
    Medium.cutMove world (body.lawCut world (fun _ => world.sense) 5) x w v =
      (1 : ℝ) / 2 * moves w (fun _ => .search) v +
      (1 : ℝ) / 2 * moves w (fun _ => .aReady) v := by
  change (∑ g : Unit → Cell, (∑ u : Env,
    body.lawCut world (fun _ => world.sense) 5 g u) * moves w g v) = _
  have hi (q : Cell) (w : Env) := cut_early 5 (by omega) q w
  simp only [sum_configs, sum_env, hi]
  norm_num [early]

private theorem bad_six :
    body.lawCut world (fun _ => world.sense) 6 (fun _ => .search) .missB = 1 / 4 := by
  change (∑ x : Unit → Cell, ∑ w : Env,
    body.lawCut world (fun _ => world.sense) 5 x w *
      Medium.cutMove world (body.lawCut world (fun _ => world.sense) 5) x w .missB *
        body.kernel 1 (sense w) x (fun _ => .search)) = _
  have hi (q : Cell) (w : Env) := cut_early 5 (by omega) q w
  simp only [sum_configs, sum_env, hi, move_five]
  norm_num [early, kernel_table, sense, next, moves, GraceRecovery.moves,
    RenewableLearning.moves, policy_table]

private theorem no_dead_six (v : Env) :
    body.lawCut world (fun _ => world.sense) 6 (fun _ => .vacuum) v = 0 := by
  change (∑ x : Unit → Cell, ∑ w : Env,
    body.lawCut world (fun _ => world.sense) 5 x w *
      Medium.cutMove world (body.lawCut world (fun _ => world.sense) 5) x w v *
        body.kernel 1 (sense w) x (fun _ => .vacuum)) = _
  have hi (q : Cell) (w : Env) := cut_early 5 (by omega) q w
  simp only [sum_configs, sum_env, hi]
  norm_num [early, kernel_table, sense, next]

private theorem miss_to_probe_six (x : Unit → Cell) :
    Medium.cutMove world (body.lawCut world (fun _ => world.sense) 6) x .missB .probeB = 1 := by
  change (∑ g : Unit → Cell, (∑ u : Env,
    body.lawCut world (fun _ => world.sense) 6 g u) * moves .missB g .probeB) = _
  have hs := body.lawCut_sum world (fun _ => world.sense) 6
  simp only [sum_configs] at hs ⊢
  norm_num [moves, GraceRecovery.moves, RenewableLearning.moves, no_dead_six] at hs ⊢
  exact hs

private noncomputable def dead (n : ℕ) : ℝ :=
  ∑ w : Env, body.lawCut world (fun _ => world.sense) n (fun _ => .vacuum) w

private theorem step_lower (n : ℕ) (x y : S) :
    body.lawCut world (fun _ => world.sense) n x.1 x.2 *
      Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x.1 x.2 y.2 *
        body.kernel 1 (sense x.2) x.1 y.1 ≤
    body.lawCut world (fun _ => world.sense) (n + 1) y.1 y.2 := by
  change _ ≤ ∑ z : Unit → Cell, ∑ w : Env,
    body.lawCut world (fun _ => world.sense) n z w *
      Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) z w y.2 *
        body.kernel 1 (sense w) z y.1
  apply le_trans _ (Finset.single_le_sum (f := fun z : Unit → Cell =>
    ∑ w : Env, body.lawCut world (fun _ => world.sense) n z w *
      Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) z w y.2 *
        body.kernel 1 (sense w) z y.1) ?_ (Finset.mem_univ x.1))
  · apply Finset.single_le_sum (f := fun w : Env =>
      body.lawCut world (fun _ => world.sense) n x.1 w *
        Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x.1 w y.2 *
          body.kernel 1 (sense w) x.1 y.1) _ (Finset.mem_univ x.2)
    intro w _
    exact mul_nonneg (mul_nonneg
      (body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n _ _)
      (body.eventWorldStep_nonneg world ec true _ n (x.1,w) _))
      (body.kernel_nonneg (body.eventClocked_implies_clocked ec) _ _ _)
  · intro z _
    apply Finset.sum_nonneg
    intro w _
    exact mul_nonneg (mul_nonneg
      (body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n _ _)
      (body.eventWorldStep_nonneg world ec true _ n (z,w) _))
      (body.kernel_nonneg (body.eventClocked_implies_clocked ec) _ _ _)

/-- A quarter of mass has physically lost its body by tick seven. -/
theorem dead_seed : (1 : ℝ) / 4 ≤ dead 7 := by
  have h := step_lower 6 ((fun _ => .search), .missB) ((fun _ => .vacuum), .probeB)
  rw [bad_six, miss_to_probe_six, kernel_table] at h
  norm_num [sense, next] at h
  exact h.trans (Finset.single_le_sum (fun w _ =>
    body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ 7 _ w)
      (Finset.mem_univ Env.probeB))

private theorem empty_kernel (s : Unit → Signal) :
    body.kernel 1 s (fun _ => .vacuum) (fun _ => .vacuum) = 1 := by
  rw [← body.eventStep_forget, Fintype.sum_option, Fintype.sum_unique]
  simp [Medium.eventStep, body, plant, next]

private theorem dead_monotone (n : ℕ) : dead n ≤ dead (n + 1) := by
  let L := body.lawCut world (fun _ => world.sense) n
  calc
    _ = ∑ w : Env, ∑ v : Env, L (fun _ => .vacuum) w *
        Medium.cutMove world L (fun _ => .vacuum) w v := by
      apply Finset.sum_congr rfl
      intro w _
      symm
      rw [← Finset.mul_sum, Medium.cutMove_sum world (body.lawCut_sum world _ n), mul_one]
    _ = ∑ v : Env, ∑ w : Env, L (fun _ => .vacuum) w *
        Medium.cutMove world L (fun _ => .vacuum) w v := Finset.sum_comm
    _ ≤ _ := by
      unfold dead
      simp only [Medium.lawCut]
      apply Finset.sum_le_sum
      intro v _
      have he : (∑ w : Env, L (fun _ => .vacuum) w *
          Medium.cutMove world L (fun _ => .vacuum) w v) =
        ∑ w : Env, L (fun _ => .vacuum) w * Medium.cutMove world L (fun _ => .vacuum) w v *
          body.kernel 1 (sense w) (fun _ => .vacuum) (fun _ => .vacuum) := by
        simp only [empty_kernel, mul_one]
      rw [he]
      apply Finset.single_le_sum (f := fun x : Unit → Cell =>
        ∑ w : Env, L x w * Medium.cutMove world L x w v *
          body.kernel 1 (sense w) x (fun _ => .vacuum)) _
            (Finset.mem_univ (fun _ : Unit => Cell.vacuum))
      intro x _
      apply Finset.sum_nonneg
      intro w _
      exact mul_nonneg (mul_nonneg
        (body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n _ _)
        (body.eventWorldStep_nonneg world ec true _ n (x,w) _))
        (body.kernel_nonneg (body.eventClocked_implies_clocked ec) _ _ _)

private theorem cut_mass_bound (n : ℕ) :
    Exchange.survival body world ec material true (fun _ => world.sense) n ≤ 1 - dead n := by
  classical
  have point (x : Unit → Cell) (w : Env) :
      Exchange.alive body world ec material true (fun _ => world.sense) n (x,w) ≤
        if x () = .vacuum then 0 else 1 := by
    by_cases hx : x () = .vacuum
    · have he : x = fun _ => .vacuum := by funext u; cases u; exact hx
      rw [he]
      simp [Exchange.alive, Viability.alive, body, plant, Medium.total, cost, buffer, credit]
    · simp only [hx, ↓reduceIte]
      exact (Viability.alive_bounds _ _ _).2
  calc
    _ ≤ ∑ x : Unit → Cell, ∑ w : Env, (if x () = .vacuum then 0 else 1) *
        body.lawCut world (fun _ => world.sense) n x w := by
      apply Finset.sum_le_sum
      intro x _
      apply Finset.sum_le_sum
      intro w _
      exact mul_le_mul_of_nonneg_right (point x w)
        (body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n x w)
    _ = _ := by
      have hs := body.lawCut_sum world (fun _ => world.sense) n
      unfold dead
      simp only [sum_configs] at hs ⊢
      simp only [↓reduceIte, zero_mul, one_mul, Finset.sum_const_zero] at hs ⊢
      norm_num at hs ⊢
      linarith

theorem cut_survival (n : ℕ) (hn : 7 ≤ n) :
    Exchange.survival body world ec material true (fun _ => world.sense) n ≤ 3 / 4 := by
  have hd : (1 : ℝ) / 4 ≤ dead n := by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hn
    induction k with
    | zero => simpa using dead_seed
    | succ k ih => exact (ih (by omega)).trans (dead_monotone (7 + k))
  linarith [cut_mass_bound n]

theorem maintenance : Exchange.Persists body world ec material 0 ∧
    Exchange.ToleratesSensoryInterruption body world ec material 0 1 ∧
    Exchange.SelfMaintains body world ec material := by
  refine ⟨⟨by norm_num, ?_⟩, GraceRecovery.one_tick_tolerance, 7, (1 : ℝ) / 4,
    by norm_num, ?_⟩
  · intro n; rw [GraceRecovery.natural_survival]; norm_num
  · intro n hn
    rw [GraceRecovery.natural_survival]
    linarith [cut_survival n hn]

end CGB.GraceRecoveryClosure

