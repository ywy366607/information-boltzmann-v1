import CGBAxioms.Inference

/-!
# Foundation audit: premises, specialization, and clocks

These results audit the existing predicates; they add no physical axiom and do not
change the organism dynamics or the life target. Independence is relative to the
class of admissible models, not a statement that this is a globally minimal theory.
-/

open Finset

namespace CGB.Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- P1a itself supplies local-state finiteness, before a finite enumeration is chosen. -/
theorem finite_local_states_of_p1a (h : M.P1a) : Finite Λ := by
  obtain ⟨K, hK⟩ := h.2.1
  have hfin := h.1 K
  rw [show {ℓ : Λ | M.content ℓ ≤ K} = Set.univ from
    Set.eq_univ_of_forall hK] at hfin
  exact Set.finite_univ_iff.1 hfin

/-- On an already finite local state type, P1a's first two conjuncts are automatic.
The infinite tape/bag independence witnesses belong to the more general Medium layer. -/
theorem p1a_iff_unique_vacuum [Finite Λ] :
    M.P1a ↔ ∃! ℓ, M.content ℓ = 0 := by
  classical
  let _ : Fintype Λ := Fintype.ofFinite Λ
  constructor
  · exact fun h => h.2.2
  · intro hv
    exact ⟨fun _ => Set.toFinite _,
      ⟨∑ ℓ, M.content ℓ, fun ℓ => Finset.single_le_sum (fun _ _ => Nat.zero_le _)
        (Finset.mem_univ ℓ)⟩, hv⟩

variable [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]

/-- A globally bounded clock follows if the stimulus alphabet is also finite.
Finiteness of the body alone does not provide this missing uniformity premise. -/
theorem exists_clock_of_finite_stimuli [Finite I] : ∃ R, M.Clocked R := by
  classical
  let _ : Fintype I := Fintype.ofFinite I
  let q := fun (s : U → I) (x : U → Λ) => ∑ y, (M.jump s).rate x y
  have hq : ∀ s x, 0 ≤ q s x := fun s x =>
    Finset.sum_nonneg fun y _ => (M.jump s).rate_nonneg x y
  let total := ∑ s, ∑ x, q s x
  have ht : 0 ≤ total := Finset.sum_nonneg fun s _ =>
    Finset.sum_nonneg fun x _ => hq s x
  refine ⟨total + 1, by dsimp [total] at *; linarith, ?_⟩
  intro s x
  have hlocal : q s x ≤ ∑ z, q s z :=
    Finset.single_le_sum (fun z _ => hq s z) (Finset.mem_univ x)
  have hglobal : (∑ z, q s z) ≤ total :=
    Finset.single_le_sum (fun s' _ => Finset.sum_nonneg fun z _ => hq s' z)
      (Finset.mem_univ s)
  change q s x ≤ total + 1
  linarith

variable (Wd : World U Λ I)

/-- The current recovery menu does not consult actual world transitions. This audits
the existing definition rather than calling its value a reachable-world probability. -/
theorem recover_ignores_world_moves
    (moves' : Wd.W → (U → Λ) → Wd.W → ℝ)
    (hnonneg : ∀ w x w', 0 ≤ moves' w x w')
    (hsum : ∀ w x, ∑ w', moves' w x w' = 1) (k : ℕ) (x : U → Λ) :
    M.recover { Wd with moves := moves', moves_nonneg := hnonneg, moves_sum := hsum } k x
      = M.recover Wd k x := by
  induction k generalizing x with
  | zero => rfl
  | succ k ih => simp only [recover, ih]

/-- Recovery is a genuine probability only with the clock-admission hypothesis.
Its favourable-stimulus supremum is not an actual-world reachability probability. -/
theorem recover_bounds (hc : M.Clocked Wd.clock) (k : ℕ) (x : U → Λ) :
    0 ≤ M.recover Wd k x ∧ M.recover Wd k x ≤ 1 := by
  classical
  induction k generalizing x with
  | zero => simp [recover]
  | succ k ih =>
    let f : Wd.W → ℝ := fun w => if Wd.sense w = M.calm then 0 else
      ∑ y, M.kernel Wd.clock (Wd.sense w) x y *
        (if M.total x < M.total y then 1 else M.recover Wd k y)
    have hf : ∀ w, 0 ≤ f w ∧ f w ≤ 1 := by
      intro w
      dsimp [f]
      split_ifs
      · norm_num
      · have hv : ∀ y, 0 ≤ (if M.total x < M.total y then (1 : ℝ)
            else M.recover Wd k y) ∧
            (if M.total x < M.total y then (1 : ℝ) else M.recover Wd k y) ≤ 1 := by
          intro y
          split_ifs
          · norm_num
          · exact ih y
        constructor
        · exact Finset.sum_nonneg fun y _ =>
            mul_nonneg (M.kernel_nonneg hc _ _ _) (hv y).1
        · calc
            _ ≤ ∑ y, M.kernel Wd.clock (Wd.sense w) x y :=
              Finset.sum_le_sum fun y _ =>
                mul_le_of_le_one_right (M.kernel_nonneg hc _ _ _) (hv y).2
            _ = 1 := M.kernel_sum _ _ _
    have hb : BddAbove (Set.range f) := ⟨1, by rintro _ ⟨w, rfl⟩; exact (hf w).2⟩
    let _ : Nonempty Wd.W := ⟨Wd.start⟩
    change 0 ≤ ⨆ w, f w ∧ (⨆ w, f w) ≤ 1
    exact ⟨(hf Wd.start).1.trans (le_ciSup hb Wd.start), ciSup_le fun w => (hf w).2⟩

/-- Normal and loop-cut laws must be nonnegative as well as normalized. -/
theorem law_nonneg_of_clocked (hc : M.Clocked Wd.clock)
    (sched : ℕ → Wd.W → U → I) (n : ℕ) (x : U → Λ) (w : Wd.W) :
    0 ≤ M.law Wd sched n x w := by
  induction n generalizing x w with
  | zero => simp only [law]; split_ifs <;> norm_num
  | succ n ih =>
    exact Finset.sum_nonneg fun y _ => Finset.sum_nonneg fun v _ =>
      mul_nonneg (mul_nonneg (ih y v) (Wd.moves_nonneg v y w))
        (M.kernel_nonneg hc _ _ _)

theorem lawCut_nonneg_of_clocked (hc : M.Clocked Wd.clock)
    (sched : ℕ → Wd.W → U → I) (n : ℕ) (x : U → Λ) (w : Wd.W) :
    0 ≤ M.lawCut Wd sched n x w := by
  induction n generalizing x w with
  | zero => simp only [lawCut]; split_ifs <;> norm_num
  | succ n ih =>
    have hcut : ∀ y v, 0 ≤ cutMove Wd (M.lawCut Wd sched n) y v w := by
      intro y v
      exact Finset.sum_nonneg fun z _ => mul_nonneg
        (Finset.sum_nonneg fun v' _ => ih z v') (Wd.moves_nonneg v z w)
    exact Finset.sum_nonneg fun y _ => Finset.sum_nonneg fun v _ =>
      mul_nonneg (mul_nonneg (ih y v) (hcut y v)) (M.kernel_nonneg hc _ _ _)

theorem aliveProb_bounds (hc : M.Clocked Wd.clock)
    (sched : ℕ → Wd.W → U → I) (n : ℕ) :
    0 ≤ M.aliveProb Wd sched n ∧ M.aliveProb Wd sched n ≤ 1 := by
  classical
  have hi : ∀ x, 0 ≤ M.aliveInd Wd x ∧ M.aliveInd Wd x ≤ 1 := by
    intro x
    simp only [aliveInd]
    split_ifs <;> norm_num
  constructor
  · exact Finset.sum_nonneg fun x _ => Finset.sum_nonneg fun w _ =>
      mul_nonneg (hi x).1 (M.law_nonneg_of_clocked Wd hc sched n x w)
  · calc
      _ ≤ ∑ x, ∑ w, M.law Wd sched n x w :=
        Finset.sum_le_sum fun x _ => Finset.sum_le_sum fun w _ =>
          mul_le_of_le_one_left (M.law_nonneg_of_clocked Wd hc sched n x w) (hi x).2
      _ = 1 := M.law_sum Wd sched n

theorem aliveProb_zero (sched : ℕ → Wd.W → U → I) :
    M.aliveProb Wd sched 0 = M.aliveInd Wd M.init := by
  classical
  simp [aliveProb, law, mul_ite, ite_and]

/-- Probability/horizon parameter bounds follow from a persistence witness and clock
admission; they need not be introduced as additional physical axioms. -/
theorem persists_parameter_bounds (hc : M.Clocked Wd.clock) {δ : ℝ} {T : ℕ}
    (hp : M.Persists Wd δ T) :
    0 ≤ δ ∧ 0 < Wd.horizon ∧ Wd.recovery ≤ 1 := by
  have hn := hp.2.2.1 0
  have hu := (M.aliveProb_bounds Wd hc (fun _ => Wd.sense) 0).2
  have hd : 0 ≤ δ := by linarith
  rw [M.aliveProb_zero Wd] at hn
  have hpos : 0 < M.aliveInd Wd M.init := by linarith [hp.1]
  have hr : Wd.recovery ≤ M.recover Wd Wd.horizon M.init := by
    classical
    unfold aliveInd at hpos
    split_ifs at hpos with h
    · exact h.2
    · norm_num at hpos
  refine ⟨hd, ?_, hr.trans (M.recover_bounds Wd hc _ _).2⟩
  by_contra h
  have hz : Wd.horizon = 0 := Nat.eq_zero_of_not_pos h
  rw [hz] at hr
  change Wd.recovery ≤ 0 at hr
  linarith [Wd.recovery_pos]

end CGB.Medium

namespace CGB.FoundationCounterexamples

/-- A single finite site with rates set by an unbounded natural-valued stimulus.
It satisfies all six existing clauses; no seventh clause has been added. -/
def unboundedStimulus : Medium (Fin 1) Bool Unit ℕ where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content := Bool.toNat
  parts _ := univ
  act _ _ x := fun u => !x u
  rate _ s _ := s 0
  rate_nonneg _ _ _ := Nat.cast_nonneg _
  init _ := false
  calm _ := 0

theorem unboundedStimulus_axioms : unboundedStimulus.Axioms := by
  refine ⟨Models.bool_p1a rfl, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨fun u v => ⟨Equiv.swap u v, fun _ _ => Iff.rfl,
        Equiv.swap_apply_left u v⟩, fun _ _ _ _ _ => trivial, ?_, ?_⟩
    · intro a s x u hu
      exact False.elim (hu (Finset.mem_univ u))
    · intro a s s' x y h
      have hs : s = s' := funext fun u => (h u (Finset.mem_univ u)).2
      have hx : x = y := funext fun u => (h u (Finset.mem_univ u)).1
      subst hs hx
      exact ⟨rfl, fun _ _ => rfl⟩
  · intro a s x hc _
    have : (0 : Fin 1) ∉ unboundedStimulus.parts a :=
      fun hp => Finset.disjoint_left.1 hc hp (Finset.mem_univ 0)
    exact False.elim (this (Finset.mem_univ 0))
  · refine ⟨fun _ => 1, fun _ => one_pos, ?_⟩
    intro x y
    simp [Medium.jump, unboundedStimulus]
  · intro σ _
    refine ⟨Equiv.refl Unit, fun a s x => ⟨?_, ?_⟩⟩
    · have h0 : σ.symm 0 = 0 := Subsingleton.elim _ _
      simp [unboundedStimulus, h0]
    · rfl
  · refine ⟨?_, fun _ _ => rfl⟩
    intro a s s' x hs
    have h0 := hs 0 (Finset.mem_univ 0)
    exact ⟨by simp [unboundedStimulus, h0], rfl⟩

/-- Six clauses plus finite body/state/event types do not imply a global Clocked bound. -/
theorem unboundedStimulus_no_clock (R : ℝ) : ¬ unboundedStimulus.Clocked R := by
  intro h
  obtain ⟨n, hn⟩ := exists_nat_gt R
  let s : Fin 1 → ℕ := fun _ => n
  let x : Fin 1 → Bool := fun _ => false
  let y : Fin 1 → Bool := fun _ => true
  have hne : x ≠ y := by
    intro he
    have hz := congrFun he 0
    change false = true at hz
    cases hz
  have hj : (unboundedStimulus.jump s).rate x y = (n : ℝ) := by
    simp [Medium.jump, unboundedStimulus, s, x, y, hne]
  have hsingle : (n : ℝ) ≤ ∑ z, (unboundedStimulus.jump s).rate x z := by
    rw [← hj]
    exact Finset.single_le_sum
      (fun z _ => (unboundedStimulus.jump s).rate_nonneg x z) (Finset.mem_univ y)
  exact (not_lt_of_ge (hsingle.trans (h.2 s x))) hn

end CGB.FoundationCounterexamples

namespace CGB.TheoryAudit

/-- Weakening admissibility cannot destroy an already constructed existence witness.
Consequently a bad model is not a necessity proof for an existential life conjecture. -/
theorem existence_survives_weakening {S : Type*} (strong weak target : S → Prop)
    (hweak : ∀ x, strong x → weak x) (hex : ∃ x, strong x ∧ target x) :
    ∃ x, weak x ∧ target x := by
  obtain ⟨x, hx, ht⟩ := hex
  exact ⟨x, hweak x hx, ht⟩

/-- A counterexample does refute a universal guarantee from the remaining premises. -/
theorem counterexample_blocks_guarantee {S : Type*} (remaining target : S → Prop)
    (hcounter : ∃ x, remaining x ∧ ¬ target x) :
    ¬ ∀ x, remaining x → target x := by
  intro hall
  obtain ⟨x, hx, hnot⟩ := hcounter
  exact hnot (hall x hx)

end CGB.TheoryAudit
