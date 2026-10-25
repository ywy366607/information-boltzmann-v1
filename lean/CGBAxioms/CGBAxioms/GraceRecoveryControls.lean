import CGBAxioms.GraceRecovery
import CGBAxioms.RenewableCandidateAudit

/-!
# Separate controls for diagnostic grace and recovery lookahead

Changing the lookahead alone does not repair the old absorbing failure. With
grace present, two ticks still miss a genuine intake delayed by one idle tick.
The original body and interruption protocol are used on both sides.
-/

open Finset

namespace CGB.GraceRecoveryControls
open RenewableLearning

noncomputable abbrev body := GraceRecovery.body
noncomputable abbrev material := ledger true true
noncomputable abbrev offWorld : World Unit Cell Signal := { RenewableLearning.world with horizon := 3 }
abbrev offClock : body.EventClocked offWorld.clock := he true true
noncomputable def offProtocol := body.interrupt offWorld 4 1

private theorem point_loss (μ a : S → ℝ) (bad : S)
    (hn : ∀ z, 0 ≤ μ z) (hs : (∑ z, μ z) = 1) (hb : ∀ z, a z ≤ 1) (hz : a bad = 0) :
    (∑ z, a z * μ z) ≤ 1 - μ bad := by
  classical
  have delete : (∑ z : S, if z = bad then 0 else μ z) = 1 - μ bad := by
    have h : (∑ z : S, ((if z = bad then 0 else μ z) + (if z = bad then μ z else 0))) =
        ∑ z : S, μ z := by
      apply Finset.sum_congr rfl
      intro z _
      split_ifs <;> simp
    simp only [Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, hs] at h
    linarith
  calc
    _ ≤ ∑ z : S, if z = bad then 0 else μ z := by
      apply Finset.sum_le_sum
      intro z _
      by_cases he : z = bad
      · subst z; simp [hz]
      · rw [if_neg he]
        exact mul_le_of_le_one_left (hn z) (hb z)
    _ = _ := delete

private theorem off_trap_recovery (t k : ℕ) (x : Unit → Cell) :
    Exchange.recovery body offWorld offClock material false offProtocol t k (x, .trap) = 0 := by
  apply RecoveryCriteria.firstHit_zero_of_closed _ _ (fun z => z.2 = .trap)
  · intro n z e z' hg hp
    change 0 < RenewableLearning.moves z.2 z.1 z'.2 *
      body.eventStep 1 (offProtocol n z.2) z.1 e z'.1 at hp
    rw [hg] at hp
    by_contra hn
    simp [RenewableLearning.moves, hn] at hp
  · intro n z e z' hg _
    cases e with
    | none => rfl
    | some u =>
      cases u
      change decide (0 < incoming true (offProtocol n z.2 ()) (z.1 ())) = false
      rw [hg]
      have hz : incoming true (offProtocol n .trap ()) (z.1 ()) = 0 := by
        simp only [offProtocol, Medium.interrupt]
        split_ifs <;> simp [offWorld, body, plant, sense, incoming]
      simp [hz]
  · rfl

theorem horizon_change_alone_fails :
    Exchange.survival body offWorld offClock material false offProtocol 6 ≤ 1 / 2 := by
  classical
  let bad : S := ((fun _ => .aReady), .trap)
  let μ : S → ℝ := fun z => body.law offWorld offProtocol 6 z.1 z.2
  let a := Exchange.alive body offWorld offClock material false offProtocol 6
  have mass : (1 : ℝ) / 2 ≤ μ bad :=
    RenewableCandidateAudit.interrupted_trap_mass 1 (by norm_num)
  have hs : (∑ z : S, μ z) = 1 := by
    simpa only [Fintype.sum_prod_type] using body.law_sum offWorld offProtocol 6
  have hz : a bad = 0 := by
    change Viability.alive _ _ (Exchange.recovery body offWorld offClock material false
      offProtocol 6 offWorld.horizon ((fun _ => .aReady), .trap)) = 0
    rw [off_trap_recovery]
    norm_num [Viability.alive, offWorld]
  have h := point_loss μ a bad
    (fun z => body.law_nonneg_of_clocked offWorld (body.eventClocked_implies_clocked offClock)
      offProtocol 6 z.1 z.2) hs (fun _ => (Viability.alive_bounds _ _ _).2) hz
  change (∑ x, ∑ w, a (x,w) * μ (x,w)) ≤ _
  simpa only [Fintype.sum_prod_type] using
    (show (∑ z : S, a z * μ z) ≤ 1 / 2 from by linarith)

theorem two_tick_lookahead_insufficient :
    Exchange.recovery body GraceRecovery.world GraceRecovery.ec material false
      (GraceRecovery.interrupted 5) 4 2 ((fun _ => .aReady), .missB) = 0 := by
  let K := body.jointEventKernel GraceRecovery.world GraceRecovery.ec false (GraceRecovery.interrupted 5)
  let mark := Exchange.intake body GraceRecovery.world material (GraceRecovery.interrupted 5)
  have hm5 (z z' : S) (e : Option Unit) : mark 5 z e z' = false := by
    cases e <;> simp [mark, Exchange.intake, material, ledger, GraceRecovery.interrupted,
      Medium.interrupt, body, plant, incoming]
  have hz5 (z : S) : EventChannels.Kernel.firstHit K mark 5 1 z = 0 := by
    simp [EventChannels.Kernel.firstHit, hm5]
  have hm4 (e : Option Unit) (z' : S) : mark 4 ((fun _ => .aReady), .missB) e z' = false := by
    cases e <;> simp [mark, Exchange.intake, material, ledger, GraceRecovery.interrupted,
      Medium.interrupt, GraceRecovery.world, GraceRecovery.worldWith, sense, incoming]
  change (∑ e, ∑ z' : S, (K 4).step ((fun _ => .aReady), .missB) e z' *
    (if mark 4 ((fun _ => .aReady), .missB) e z' then 1
    else EventChannels.Kernel.firstHit K mark 5 1 z')) = 0
  simp only [hm4, hz5, Bool.false_eq_true, ↓reduceIte, mul_zero, Finset.sum_const_zero]

end CGB.GraceRecoveryControls
