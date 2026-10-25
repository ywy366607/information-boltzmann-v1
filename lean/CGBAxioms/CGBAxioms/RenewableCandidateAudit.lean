import CGBAxioms.RenewableLearningCarrier

/-!
# A concrete obstruction to the missing sensory-interruption obligation

Keep the original body, world, clock and interruption protocol. A supported
path reaches conflict, loses its reopening opportunity during interruption,
and then fails the probe. Its mass is at least one half. This diagnoses this
specific witness; it is not an impossibility theorem for living systems.
-/

open Finset

namespace CGB.RenewableCandidateAudit
open RenewableLearning

noncomputable abbrev body := plant true true
abbrev clocked := (body.eventClocked_implies_clocked (he true true))
noncomputable def schedule (T : ℕ) := body.interrupt world 4 T

private def path : ℕ → S
  | 0 => ((fun _ => .search), .probeA)
  | 1 => ((fun _ => .search), .answerA)
  | 2 => ((fun _ => .aSpent), .feedA)
  | 3 => ((fun _ => .aReady), .feedB)
  | 4 => ((fun _ => .aReady), .missB)
  | 5 => ((fun _ => .aReady), .probeB)
  | _ => ((fun _ => .aReady), .trap)

private theorem kernel_at_next (s : Unit → Signal) (q : Cell) :
    body.kernel 1 s (fun _ => q) (fun _ => next true true (s ()) q) = 1 := by
  have hs : s = fun _ => s () := by funext u; cases u; rfl
  rw [hs, ← body.eventStep_forget, Fintype.sum_option, Fintype.sum_unique]
  generalize s () = a
  cases a <;> cases q <;> norm_num [Medium.eventStep, body, plant, next, funext_iff]

private theorem policy_values : policy .search = 2 ∧ policy .aSpent = 0 ∧
    policy .aReady = 0 := by decide

private theorem edge (T : ℕ) (hT : 0 < T) (i : Fin 6) :
    moves (path i.val).2 (path i.val).1 (path (i.val + 1)).2 *
      body.kernel 1 (schedule T i.val (path i.val).2) (path i.val).1
        (path (i.val + 1)).1 = if i.val = 2 then 1 / 2 else 1 := by
  fin_cases i <;>
    simp [path, schedule, Medium.interrupt, world, sense, moves,
      policy_values.1, policy_values.2.1, policy_values.2.2]
  all_goals try omega
  all_goals try split_ifs
  all_goals exact kernel_at_next _ _

private theorem step_lower (sched : ℕ → Env → Unit → Signal) (n : ℕ) (x y : S) :
    body.law world sched n x.1 x.2 *
      (moves x.2 x.1 y.2 * body.kernel 1 (sched n x.2) x.1 y.1) ≤
        body.law world sched (n + 1) y.1 y.2 := by
  change _ ≤ ∑ z : Unit → Cell, ∑ w : Env,
    body.law world sched n z w * moves w z y.2 * body.kernel 1 (sched n w) z y.1
  calc
    _ = body.law world sched n x.1 x.2 * moves x.2 x.1 y.2 *
        body.kernel 1 (sched n x.2) x.1 y.1 := by ring
    _ ≤ ∑ w : Env, body.law world sched n x.1 w * moves w x.1 y.2 *
        body.kernel 1 (sched n w) x.1 y.1 := by
      apply Finset.single_le_sum (f := fun w : Env =>
        body.law world sched n x.1 w * moves w x.1 y.2 *
          body.kernel 1 (sched n w) x.1 y.1) _ (Finset.mem_univ x.2)
      intro w _
      exact mul_nonneg (mul_nonneg (body.law_nonneg_of_clocked world clocked sched n x.1 w)
        (world.moves_nonneg _ _ _)) (body.kernel_nonneg clocked _ _ _)
    _ ≤ _ := by
      apply Finset.single_le_sum (f := fun z : Unit → Cell =>
        ∑ w : Env, body.law world sched n z w * moves w z y.2 *
          body.kernel 1 (sched n w) z y.1) _ (Finset.mem_univ x.1)
      intro z _
      apply Finset.sum_nonneg
      intro w _
      exact mul_nonneg (mul_nonneg (body.law_nonneg_of_clocked world clocked sched n z w)
        (world.moves_nonneg _ _ _)) (body.kernel_nonneg clocked _ _ _)

private theorem path_lower (sched : ℕ → Env → Unit → Signal) (n : ℕ) :
    (∏ t ∈ range n, moves (path t).2 (path t).1 (path (t + 1)).2 *
      body.kernel 1 (sched t (path t).2) (path t).1 (path (t + 1)).1) ≤
        body.law world sched n (path n).1 (path n).2 := by
  induction n with
  | zero => simp [Medium.law, path, body, plant, world, funext_iff]
  | succ n ih =>
    rw [Finset.prod_range_succ]
    exact (mul_le_mul_of_nonneg_right ih
      (mul_nonneg (world.moves_nonneg _ _ _) (body.kernel_nonneg clocked _ _ _))).trans
        (step_lower sched n (path n) (path (n + 1)))

theorem interrupted_trap_mass (T : ℕ) (hT : 0 < T) :
    1 / 2 ≤ body.law world (schedule T) 6 (fun _ => .aReady) .trap := by
  have h := path_lower (schedule T) 6
  have he : (∏ t ∈ range 6, moves (path t).2 (path t).1 (path (t + 1)).2 *
      body.kernel 1 (schedule T t (path t).2) (path t).1 (path (t + 1)).1) =
      ∏ t ∈ range 6, (if t = 2 then (1 : ℝ) / 2 else 1) := by
    apply Finset.prod_congr rfl
    intro t ht
    exact edge T hT ⟨t, Finset.mem_range.mp ht⟩
  rw [he] at h
  norm_num [Finset.prod_range_succ, path] at h ⊢
  exact h

private theorem trap_recovery (T t k : ℕ) (x : Unit → Cell) :
    Exchange.recovery body world (he true true) (ledger true true) false
      (schedule T) t k (x, .trap) = 0 := by
  apply RecoveryCriteria.firstHit_zero_of_closed _ _ (fun z => z.2 = .trap)
  · intro n z e z' hg hp
    change 0 < moves z.2 z.1 z'.2 *
      body.eventStep 1 (schedule T n z.2) z.1 e z'.1 at hp
    rw [hg] at hp
    by_contra hn
    simp [moves, hn] at hp
  · intro n z e z' hg _
    simp only [Exchange.intake]
    cases e with
    | none => rfl
    | some u =>
      cases u
      change decide (0 < incoming true (schedule T n z.2 ()) (z.1 ())) = false
      rw [hg]
      have hz : incoming true (schedule T n .trap ()) (z.1 ()) = 0 := by
        simp only [schedule, Medium.interrupt]
        split_ifs <;> simp [world, body, plant, sense, incoming]
      simp [hz]
  · rfl

theorem interruption_failure (T : ℕ) (hT : 0 < T) :
    Exchange.survival body world (he true true) (ledger true true) false
      (schedule T) 6 ≤ 1 / 2 := by
  classical
  let bad : S := ((fun _ => .aReady), .trap)
  let L : S → ℝ := fun z => body.law world (schedule T) 6 z.1 z.2
  let a : S → ℝ := Exchange.alive body world (he true true) (ledger true true)
    false (schedule T) 6
  have hz : a bad = 0 := by
    change Viability.alive _ _ (Exchange.recovery body world (he true true)
      (ledger true true) false (schedule T) 6 world.horizon ((fun _ => .aReady), .trap)) = 0
    rw [trap_recovery]
    norm_num [Viability.alive, world]
  have mass : (1 : ℝ) / 2 ≤ L bad := interrupted_trap_mass T hT
  have total : (∑ z : S, L z) = 1 := by
    simpa only [Fintype.sum_prod_type] using body.law_sum world (schedule T) 6
  have delete : (∑ z : S, if z = bad then 0 else L z) = 1 - L bad := by
    have hh : (∑ z : S, ((if z = bad then 0 else L z) + (if z = bad then L z else 0))) =
        ∑ z : S, L z := by
      apply Finset.sum_congr rfl
      intro z _
      split_ifs <;> simp
    simp only [Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte,
      total] at hh
    linarith
  have bound : (∑ z : S, a z * L z) ≤ ∑ z : S, if z = bad then 0 else L z := by
    apply Finset.sum_le_sum
    intro z _
    by_cases he : z = bad
    · subst z; simp [hz]
    · rw [if_neg he]
      exact mul_le_of_le_one_left (body.law_nonneg_of_clocked world clocked _ _ _ _)
        (Viability.alive_bounds _ _ _).2
  change (∑ x, ∑ w, _ * _) ≤ _
  change (∑ x, ∑ w, a (x, w) * L (x, w)) ≤ _
  simpa only [Fintype.sum_prod_type] using
    (show (∑ z : S, a z * L z) ≤ 1 / 2 from by linarith)

theorem no_positive_interruption_tolerance (T : ℕ) :
    ¬ Exchange.ToleratesSensoryInterruption body world (he true true) (ledger true true) 0 T := by
  rintro ⟨hT, hp⟩
  have h := interruption_failure T hT
  have hh := hp 4 6
  change 1 - (0 : ℝ) ≤ Exchange.survival body world (he true true) (ledger true true)
    false (schedule T) 6 at hh
  linarith

end CGB.RenewableCandidateAudit
