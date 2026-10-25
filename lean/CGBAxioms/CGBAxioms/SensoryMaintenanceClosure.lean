import CGBAxioms.SensoryMaintenanceRecovery

/-! The original marginal-ghost cut includes the new physical reserve. -/
open Finset

namespace CGB.SensoryMaintenanceClosure
open RenewableLearning (Cell Signal Env sense)
open SensoryMaintenance

private theorem sum_old (f : (Unit → Cell) → ℝ) : ∑ x, f x = ∑ q : Cell, f (fun _ => q) := by
  exact Fintype.sum_equiv (Equiv.funUnique Unit Cell) _ _ (by intro x; congr 1)

private theorem sum_new (f : (Unit → State) → ℝ) :
    ∑ x, f x = ∑ q : Cell, ∑ r : Fin 3, f (fun _ => (q,r)) := by
  rw [Fintype.sum_equiv (Equiv.funUnique Unit State) _ (fun q => f (fun _ => q))
    (by intro x; congr 1)]
  exact Fintype.sum_prod_type _

private theorem awake_projection (a : Signal) (x : State) :
    (next (some a) x).1 = RenewableLearning.next true true a x.1 := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases a <;> cases q <;> decide

private theorem old_kernel (s : Unit → Signal) (q r : Cell) :
    GraceRecovery.body.kernel 1 s (fun _ => q) (fun _ => r) =
      if r = RenewableLearning.next true true (s ()) q then 1 else 0 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs, ← GraceRecovery.body.eventStep_forget, Fintype.sum_option, Fintype.sum_unique]
  by_cases hc : s () = .calm ∨ q = .vacuum
  · have hn : RenewableLearning.next true true (s ()) q = q := by
      rcases hc with h | h
      · by_cases hq : q = .vacuum <;> simp [RenewableLearning.next, h, hq]
      · simp [RenewableLearning.next, h]
    simp [Medium.eventStep, GraceRecovery.body, RenewableLearning.plant, hc, hn, funext_iff]
    simp only [eq_comm]
  · simp [Medium.eventStep, GraceRecovery.body, RenewableLearning.plant, hc, funext_iff]
    simp only [eq_comm]

private theorem kernel_marginal (w : Env) (x : State) (q : Cell) :
    (∑ r : Fin 3, body.kernel 1 (world.sense w) (fun _ => x) (fun _ => (q,r))) =
      GraceRecovery.body.kernel 1 (sense w) (fun _ => x.1) (fun _ => q) := by
  simp_rw [kernel_table]
  rw [old_kernel, ← awake_projection (sense w ()) x]
  generalize next (some (sense w ())) x = y
  rcases y with ⟨p,s⟩
  by_cases h : q = p
  · subst p; simp [Prod.mk.injEq]
  · simp [Prod.mk.injEq, h]

private theorem cut_move (L : (Unit → State) → Env → ℝ)
    (l : (Unit → Cell) → Env → ℝ)
    (h : ∀ q w, (∑ r : Fin 3, L (fun _ => (q,r)) w) = l (fun _ => q) w)
    (x : Unit → State) (w v : Env) :
    Medium.cutMove world L x w v =
      Medium.cutMove GraceRecovery.world l (fun u => (x u).1) w v := by
  unfold Medium.cutMove
  rw [sum_new, sum_old]
  apply Finset.sum_congr rfl
  intro q _
  change (∑ r : Fin 3, (∑ u : Env, L (fun _ => (q,r)) u) *
    GraceRecovery.moves w (fun _ => q) v) = _
  rw [← Finset.sum_mul, Finset.sum_comm]
  simp_rw [h]

/-- Recompute the actual cut with the augmented body's own marginal; summing
its physical reserve gives exactly the old cut's memory/world law. -/
theorem cut_marginal (n : ℕ) (q : Cell) (w : Env) :
    (∑ r : Fin 3, body.lawCut world (fun _ => world.sense) n (fun _ => (q,r)) w) =
      GraceRecovery.body.lawCut GraceRecovery.world GraceRecovery.natural n (fun _ => q) w := by
  induction n generalizing q w with
  | zero =>
    cases q <;> cases w <;>
      norm_num [Medium.lawCut, body, funext_iff, GraceRecovery.body, RenewableLearning.plant]
  | succ n ih =>
    have hm (x : Unit → State) (v u : Env) := cut_move
      (body.lawCut world (fun _ => world.sense) n)
      (GraceRecovery.body.lawCut GraceRecovery.world GraceRecovery.natural n) ih x v u
    simp only [Medium.lawCut]
    rw [Finset.sum_comm]
    calc
      _ = ∑ x : Unit → State, ∑ v : Env,
          body.lawCut world (fun _ => world.sense) n x v *
          Medium.cutMove world (body.lawCut world (fun _ => world.sense) n) x v w *
          (∑ r : Fin 3, body.kernel 1 (world.sense v) x (fun _ => (q,r))) := by
        apply Finset.sum_congr rfl
        intro x _
        rw [Finset.sum_comm]
        simp only [← Finset.mul_sum]
      _ = _ := by
        simp_rw [hm]
        rw [sum_new, sum_old]
        apply Finset.sum_congr rfl
        intro p _
        simp_rw [kernel_marginal]
        rw [Finset.sum_comm]
        apply Finset.sum_congr rfl
        intro v _
        rw [← Finset.sum_mul, ← Finset.sum_mul, ih]

private theorem vacuum_target (w : Env) (x : State) :
    (next (some (sense w ())) x).1 = .vacuum →
      (next (some (sense w ())) x).2.val = 0 := by
  rcases x with ⟨q,r⟩
  fin_cases r <;> cases w <;> cases q <;> decide

private theorem vacuum_reserve_zero (n : ℕ) (r : Fin 3) (hr : r ≠ 0) (w : Env) :
    body.lawCut world (fun _ => world.sense) n (fun _ => (.vacuum,r)) w = 0 := by
  cases n with
  | zero => simp [Medium.lawCut, body, funext_iff]
  | succ n =>
    simp only [Medium.lawCut]
    apply Finset.sum_eq_zero; intro x _
    apply Finset.sum_eq_zero; intro v _
    have he : x = fun _ => x () := by funext u; cases u; rfl
    rw [he, kernel_table]
    have hn : (.vacuum,r) ≠ next (world.sense v ()) (x ()) := by
      intro hh
      have hq := congrArg Prod.fst hh.symm
      have hz := vacuum_target v (x ()) hq
      rw [← hh] at hz
      exact hr (Fin.ext hz)
    simp [hn]

noncomputable def dead (n : ℕ) : ℝ :=
  ∑ w : Env, body.lawCut world (fun _ => world.sense) n (fun _ => empty) w

theorem empty_mass (n : ℕ) : dead n =
    ∑ w : Env, GraceRecovery.body.lawCut GraceRecovery.world GraceRecovery.natural n
      (fun _ => .vacuum) w := by
  unfold dead
  apply Finset.sum_congr rfl; intro w _
  rw [← cut_marginal n .vacuum w]
  symm
  apply Finset.sum_eq_single (0 : Fin 3)
  · intro r _ hr; exact vacuum_reserve_zero n r hr w
  · simp

theorem cut_survival (n : ℕ) (hn : 7 ≤ n) :
    Exchange.survival body world ec material true (fun _ => world.sense) n ≤ 3 / 4 := by
  have hd : (1 : ℝ) / 4 ≤ dead n := by
    -- Use the old absorbing-body lower bound, rather than a changed counterfactual.
    obtain ⟨k,rfl⟩ := Nat.exists_eq_add_of_le hn
    clear hn
    have hseed := GraceRecoveryClosure.dead_seed
    rw [empty_mass]
    change (1 : ℝ) / 4 ≤ ∑ w : Env,
      GraceRecovery.body.lawCut GraceRecovery.world GraceRecovery.natural (7 + k)
        (fun _ => .vacuum) w
    -- The legacy public cut bound is complemented below by monotonicity.
    induction k with
    | zero => exact hseed
    | succ k ih =>
      exact ih.trans (by
        have h := GraceRecoveryClosure.dead_seed
        clear h hseed
        -- This is a generic absorbing-content expectation inequality.
        have em (s : Unit → Signal) :
            GraceRecovery.body.kernel 1 s (fun _ => .vacuum) (fun _ => .vacuum) = 1 := by
          rw [old_kernel]; simp [RenewableLearning.next]
        let L := GraceRecovery.body.lawCut GraceRecovery.world GraceRecovery.natural (7 + k)
        calc
          _ = ∑ v : Env, ∑ w : Env, L (fun _ => .vacuum) w *
              Medium.cutMove GraceRecovery.world L (fun _ => .vacuum) w v := by
            rw [Finset.sum_comm]
            apply Finset.sum_congr rfl; intro w _
            rw [← Finset.mul_sum, Medium.cutMove_sum GraceRecovery.world
              (GraceRecovery.body.lawCut_sum GraceRecovery.world GraceRecovery.natural (7 + k)), mul_one]
          _ ≤ _ := by
            apply Finset.sum_le_sum; intro v _
            change _ ≤ ∑ x : Unit → Cell, ∑ w : Env, L x w *
              Medium.cutMove GraceRecovery.world L x w v *
                GraceRecovery.body.kernel 1 (sense w) x (fun _ => .vacuum)
            have he : (∑ w : Env, L (fun _ => .vacuum) w *
                Medium.cutMove GraceRecovery.world L (fun _ => .vacuum) w v) =
                ∑ w : Env, L (fun _ => .vacuum) w *
                Medium.cutMove GraceRecovery.world L (fun _ => .vacuum) w v *
                  GraceRecovery.body.kernel 1 (sense w) (fun _ => .vacuum) (fun _ => .vacuum) := by
              simp [em]
            rw [he]
            apply Finset.single_le_sum (f := fun x : Unit → Cell => ∑ w : Env,
              L x w * Medium.cutMove GraceRecovery.world L x w v *
                GraceRecovery.body.kernel 1 (sense w) x (fun _ => .vacuum)) _
                  (Finset.mem_univ (fun _ : Unit => Cell.vacuum))
            intro x _; apply Finset.sum_nonneg; intro w _
            exact mul_nonneg (mul_nonneg
              (GraceRecovery.body.lawCut_nonneg_of_clocked GraceRecovery.world
                (GraceRecovery.body.eventClocked_implies_clocked GraceRecovery.ec) _ _ _ _)
              (GraceRecovery.body.eventWorldStep_nonneg GraceRecovery.world GraceRecovery.ec
                true GraceRecovery.natural (7+k) (x,w) v))
              (GraceRecovery.body.kernel_nonneg
                (GraceRecovery.body.eventClocked_implies_clocked GraceRecovery.ec) _ _ _))
  have point (x : Unit → State) (w : Env) :
      Exchange.alive body world ec material true (fun _ => world.sense) n (x,w) ≤
        if x = (fun _ => empty) then 0 else 1 := by
    split_ifs with hx
    · rw [hx]
      simp [Exchange.alive, Viability.alive, Medium.total, body, SensoryMaintenance.cost,
        empty, RenewableLearning.cost, RenewableLearning.buffer, RenewableLearning.credit]
    · exact (Viability.alive_bounds _ _ _).2
  have hs := body.lawCut_sum world (fun _ => world.sense) n
  have bound : Exchange.survival body world ec material true (fun _ => world.sense) n ≤
      ∑ x : Unit → State, ∑ w : Env, (if x = (fun _ => empty) then 0 else 1) *
        body.lawCut world (fun _ => world.sense) n x w := by
    apply Finset.sum_le_sum; intro x _
    apply Finset.sum_le_sum; intro w _
    exact mul_le_mul_of_nonneg_right (point x w)
      (body.lawCut_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n x w)
  have eq : (∑ x : Unit → State, ∑ w : Env, (if x = (fun _ => empty) then 0 else 1) *
      body.lawCut world (fun _ => world.sense) n x w) = 1 - dead n := by
    unfold dead
    have pointid (x : Unit → State) (w : Env) :
        (if x = (fun _ => empty) then 0 else 1) *
          body.lawCut world (fun _ => world.sense) n x w =
        body.lawCut world (fun _ => world.sense) n x w -
          (if x = (fun _ => empty) then 1 else 0) *
            body.lawCut world (fun _ => world.sense) n x w := by split_ifs <;> ring
    simp_rw [pointid, Finset.sum_sub_distrib]
    rw [hs]
    simp [ite_mul, Finset.sum_ite_irrel]
  rw [eq] at bound
  linarith

end CGB.SensoryMaintenanceClosure
