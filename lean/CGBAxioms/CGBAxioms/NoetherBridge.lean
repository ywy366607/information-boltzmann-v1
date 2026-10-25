import CGBAxioms.Medium

/-!
# Noether correspondence for the actual medium event language

A closed-event restriction is an analysis object, not a change to the running
medium. P2a is equivalent to conserved content, two zero moment drifts, and an
observable multiplication symmetry of this restricted generator. Open events
are retained separately in the original jump kernel and supply boundary flux.
-/

open Finset

namespace CGB.JumpProcess

variable {S : Type*} [Fintype S]

/-- Multiplication by Q commutes with the observable generator. This is an
operator identity, with no assertion that multiplication is a stochastic map. -/
def ObservableSymmetry (P : JumpProcess S) (Q : S → ℝ) : Prop :=
  ∀ f : S → ℝ, ∀ x,
    P.gen (fun y => Q y * f y) x = Q x * P.gen f x

theorem observableSymmetry_iff_conserved (P : JumpProcess S) (Q : S → ℝ) :
    P.ObservableSymmetry Q ↔ P.Conserved Q := by
  constructor
  · intro h
    have hQ : ∀ x, P.gen Q x = 0 := by
      intro x
      simpa [gen] using h (fun _ => 1) x
    apply (noether P Q).mp
    refine ⟨hQ, ?_⟩
    intro x
    simpa [pow_two, hQ x] using h Q x
  · intro h f x
    simp only [gen, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro y _
    rcases (P.rate_nonneg x y).lt_or_eq with hp | hz
    · rw [h x y hp]
      ring
    · rw [← hz]
      ring

end CGB.JumpProcess

namespace CGB.Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- Sum event contributions before aggregating transitions with the same output. -/
theorem jump_gen_events [Fintype A] [Fintype (U → Λ)] [DecidableEq (U → Λ)]
    (s : U → I) (f : (U → Λ) → ℝ) (x : U → Λ) :
    (M.jump s).gen f x = ∑ a, M.rate a s x * (f (M.act a s x) - f x) := by
  simp only [JumpProcess.gen, jump, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a _
  by_cases he : M.act a s x = x
  · simp [he]
  · rw [Finset.sum_eq_single (M.act a s x)]
    · simp [Ne.symm he]
    · intro y _ hy
      simp [Ne.symm hy]
    · simp

/-- Retain precisely the existing closed events. Nothing is added to their rates
or effects, and no boundary event is claimed to conserve body content. -/
noncomputable def closedPart [DecidableEq U] : Medium U Λ A I :=
  { M with
    rate := fun a s x => if M.Closed a then M.rate a s x else 0
    rate_nonneg := fun a s x => by
      split_ifs
      · exact M.rate_nonneg a s x
      · exact le_rfl }

noncomputable def closedJump [DecidableEq U] [Fintype A] [DecidableEq (U → Λ)]
    (s : U → I) : JumpProcess (U → Λ) := M.closedPart.jump s

theorem closedJump_conserved_iff [DecidableEq U] [Fintype A]
    [DecidableEq (U → Λ)] (s : U → I) (Q : (U → Λ) → ℝ) :
    (M.closedJump s).Conserved Q ↔
      ∀ a x, M.Closed a → 0 < M.rate a s x → Q (M.act a s x) = Q x := by
  constructor
  · intro h a x ha hp
    by_cases he : M.act a s x = x
    · rw [he]
    have hl : M.rate a s x ≤ (M.closedJump s).rate x (M.act a s x) := by
      have hb := Finset.single_le_sum
        (s := (univ : Finset A))
        (f := fun b => if M.closedPart.act b s x = M.act a s x ∧ x ≠ M.act a s x
          then M.closedPart.rate b s x else 0)
        (fun b _ => by split_ifs; exacts [M.closedPart.rate_nonneg b s x, le_rfl])
        (Finset.mem_univ a)
      simpa [closedJump, jump, closedPart, ha, Ne.symm he] using hb
    exact (h x (M.act a s x) (lt_of_lt_of_le hp hl)).symm
  · intro h x y hp
    by_contra hQ
    have hz : (M.closedJump s).rate x y = 0 := by
      apply Finset.sum_eq_zero
      intro a _
      change (if M.act a s x = y ∧ x ≠ y then
        (if M.Closed a then M.rate a s x else 0) else 0) = 0
      split_ifs with he ha
      · by_cases hr : 0 < M.rate a s x
        · have hc := h a x ha hr
          rw [he.1] at hc
          exact False.elim (hQ hc.symm)
        · exact le_antisymm (le_of_not_gt hr) (M.rate_nonneg a s x)
      all_goals rfl
    rw [hz] at hp
    exact lt_irrefl _ hp

variable [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]

/-- The selected physical observable is total content, cast to the real numbers. -/
def contentObservable (x : U → Λ) : ℝ := M.total x

omit [Fintype Λ] in
theorem p2a_iff_closed_conserved :
    M.P2a ↔ ∀ s, (M.closedJump s).Conserved M.contentObservable := by
  constructor
  · intro h s
    apply (M.closedJump_conserved_iff s _).mpr
    intro a x ha hp
    dsimp only [contentObservable]
    exact_mod_cast h a s x ha hp
  · intro h a s x ha hp
    have he := (M.closedJump_conserved_iff s _).mp (h s) a x ha hp
    dsimp only [contentObservable] at he
    exact_mod_cast he

theorem p2a_iff_closed_moments :
    M.P2a ↔ ∀ s,
      (∀ x, (M.closedJump s).gen M.contentObservable x = 0) ∧
      (∀ x, (M.closedJump s).gen (fun y => M.contentObservable y ^ 2) x = 0) := by
  rw [M.p2a_iff_closed_conserved]
  exact forall_congr' fun s => (JumpProcess.noether (M.closedJump s) _).symm

/-- Every normalized initial distribution has zero first and second moment
DRIFT. Constancy along a time evolution follows when that evolution solves the
forward equation; this statement does not introduce a continuous semigroup. -/
theorem p2a_iff_closed_expected_drifts :
    M.P2a ↔ ∀ s,
      (∀ p : (U → Λ) → ℝ, (∀ x, 0 ≤ p x) → ∑ x, p x = 1 →
        ∑ x, p x * (M.closedJump s).gen M.contentObservable x = 0) ∧
      (∀ p : (U → Λ) → ℝ, (∀ x, 0 ≤ p x) → ∑ x, p x = 1 →
        ∑ x, p x * (M.closedJump s).gen (fun y => M.contentObservable y ^ 2) x = 0) := by
  rw [M.p2a_iff_closed_moments]
  apply forall_congr'
  intro s
  exact and_congr (JumpProcess.expect_const_iff _ _).symm
    (JumpProcess.expect_const_iff _ _).symm

theorem p2a_iff_closed_observable_symmetry :
    M.P2a ↔ ∀ s, (M.closedJump s).ObservableSymmetry M.contentObservable := by
  rw [M.p2a_iff_closed_conserved]
  exact forall_congr' fun s =>
    (JumpProcess.observableSymmetry_iff_conserved (M.closedJump s) _).symm

/-- In the original OPEN generator, all drift of any function of total content
comes from boundary-touching events. This is an event balance, not a heat or
negentropy identification. -/
theorem content_drift_boundary (h : M.P2a) (s : U → I) (g : ℕ → ℝ) (x : U → Λ) :
    (M.jump s).gen (fun y => g (M.total y)) x =
      ∑ a, if M.Closed a then 0 else
        M.rate a s x * (g (M.total (M.act a s x)) - g (M.total x)) := by
  rw [M.jump_gen_events]
  apply Finset.sum_congr rfl
  intro a _
  by_cases ha : M.Closed a
  · simp only [ha, ite_true]
    by_cases hp : 0 < M.rate a s x
    · rw [h a s x ha hp, sub_self, mul_zero]
    · have hz : M.rate a s x = 0 :=
        le_antisymm (le_of_not_gt hp) (M.rate_nonneg a s x)
      rw [hz, zero_mul]
  · simp only [ha, ite_false]

end CGB.Medium
