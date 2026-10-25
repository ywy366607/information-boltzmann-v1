import CGBAxioms.WorldRecovery

/-!
# Finite probabilistic interaction coalgebras

This is a semantic layer, not a physical axiom. A machine unfolds its current
state into an output/next-state distribution for each admitted external input.
The registered world tick is one transition. Histories are observer data, not
additional uncharged memory inside the organism. All claims quantify over every
finite history/continuation; no infinite-path measure is postulated here.
-/

open Finset

namespace CGB.Interaction

structure Machine (S I O : Type*) [Fintype S] [Fintype O] where
  step : I → S → O → S → ℝ
  nonneg : ∀ i s o s', 0 ≤ step i s o s'
  sum_one : ∀ i s, ∑ o, ∑ s', step i s o s' = 1

variable {S T I O : Type*} [Fintype S] [Fintype T] [Fintype O]
  (P : Machine S I O) (Q : Machine T I O)

/-- The input is selected causally from the common observed output history.
This is an external test context, not a planner installed in the body. -/
abbrev Policy (I O : Type*) := List O → I

namespace Machine

/-- A positive-mass observation uniquely fixes the normalized next-state law.
This conditions a given joint kernel; it does not identify its physical law. -/
theorem conditioned_next (i : I) (s : S) (o : O)
    (positive : 0 < ∑ t, P.step i s o t) :
    let mass := ∑ t, P.step i s o t
    let post := fun t => P.step i s o t / mass
    (∀ t, 0 ≤ post t) ∧ (∑ t, post t) = 1 ∧
      ∀ candidate : S → ℝ, (∀ t, mass * candidate t = P.step i s o t) → candidate = post := by
  dsimp
  refine ⟨fun t => div_nonneg (P.nonneg i s o t) (le_of_lt positive), ?_, ?_⟩
  · simp only [div_eq_mul_inv, ← Finset.sum_mul]
    exact div_self (ne_of_gt positive)
  · intro candidate law
    funext t
    exact (eq_div_iff (ne_of_gt positive)).mpr (by simpa [mul_comm] using law t)

noncomputable def trace (policy : Policy I O) : List O → List O → S → ℝ
  | _, [] => fun _ => 1
  | history, o :: future => fun s => ∑ s', P.step (policy history) s o s' *
      trace policy (history ++ [o]) future s'

end Machine

theorem trace_nonneg (policy : Policy I O) (history future : List O) (s : S) :
    0 ≤ P.trace policy history future s := by
  induction future generalizing history s with
  | nil => norm_num [Machine.trace]
  | cons o future ih =>
    exact Finset.sum_nonneg fun s' _ => mul_nonneg (P.nonneg _ _ _ _) (ih _ _)

/-- Cylinder consistency: summing all one-symbol extensions recovers the prefix
probability. Together with trace [] = 1 and nonnegativity this is a consistent
family of finite observation laws, not just a collection of named numbers. -/
theorem trace_prefix_consistent (policy : Policy I O) (history future : List O) (s : S) :
    (∑ o, P.trace policy history (future ++ [o]) s) = P.trace policy history future s := by
  induction future generalizing history s with
  | nil => simpa [Machine.trace] using P.sum_one (policy history) s
  | cons o future ih =>
    simp only [List.cons_append, Machine.trace]
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, ih]

/-- Equality of every finite output cylinder under every causal input policy.
This defines observable stream behaviour without assuming a final coalgebra or
an already-constructed measure on infinite paths. It is not literal state equality. -/
def SameBehavior (s : S) (t : T) : Prop :=
  ∀ policy history future, P.trace policy history future s = Q.trace policy history future t

/-- A common-output coupling of the two one-step laws. Its support must remain
inside R; separate matching of output and state marginals would be too weak. -/
structure Coupling (R : S → T → Prop) (s : S) (t : T) where
  mass : I → O → S → T → ℝ
  nonneg : ∀ i o s' t', 0 ≤ mass i o s' t'
  left : ∀ i o s', ∑ t', mass i o s' t' = P.step i s o s'
  right : ∀ i o t', ∑ s', mass i o s' t' = Q.step i t o t'
  supported : ∀ i o s' t', 0 < mass i o s' t' → R s' t'

def Bisimulation (R : S → T → Prop) : Prop :=
  ∀ s t, R s t → Nonempty (Coupling P Q R s t)

def Bisimilar (s : S) (t : T) : Prop :=
  ∃ R, Bisimulation P Q R ∧ R s t

variable {P Q}

/-- One-step transfer of any observable continuation value. Nonnegativity is
used to eliminate zero-mass pairs, not to assume the observable is positive. -/
theorem Coupling.expectation_at_output {R : S → T → Prop} {s : S} {t : T}
    (c : Coupling P Q R s t) (i : I) (o : O) (f : S → ℝ) (g : T → ℝ)
    (hfg : ∀ s' t', R s' t' → f s' = g t') :
    (∑ s', P.step i s o s' * f s') = ∑ t', Q.step i t o t' * g t' := by
  calc
    _ = ∑ s', ∑ t', c.mass i o s' t' * f s' := by
      simp only [← Finset.sum_mul, c.left]
    _ = ∑ s', ∑ t', c.mass i o s' t' * g t' := by
      apply Finset.sum_congr rfl
      intro s' _
      apply Finset.sum_congr rfl
      intro t' _
      rcases (c.nonneg i o s' t').lt_or_eq with hp | hp
      · rw [hfg s' t' (c.supported i o s' t' hp)]
      · simp [hp.symm]
    _ = _ := by rw [Finset.sum_comm]; simp only [← Finset.sum_mul, c.right]

theorem Bisimulation.trace_eq {R : S → T → Prop} (hR : Bisimulation P Q R)
    (policy : Policy I O) (history future : List O) {s : S} {t : T} (hst : R s t) :
    P.trace policy history future s = Q.trace policy history future t := by
  induction future generalizing history s t with
  | nil => rfl
  | cons o future ih =>
    obtain ⟨c⟩ := hR s t hst
    exact c.expectation_at_output (policy history) o _ _ (fun s' t' hr => ih _ hr)

/-- Coinductive soundness for observed infinite behaviour, represented by ALL
finite cylinders. A finite sample of cylinders does not establish this premise. -/
theorem Bisimulation.sameBehavior {R : S → T → Prop} (hR : Bisimulation P Q R)
    {s : S} {t : T} (hst : R s t) : SameBehavior P Q s t :=
  fun policy history future => Bisimulation.trace_eq hR policy history future hst

theorem coinduction {R : S → T → Prop} (hR : Bisimulation P Q R)
    {s : S} {t : T} (hst : R s t) : Bisimilar P Q s t := ⟨R, hR, hst⟩

theorem bisimilar_is_bisimulation : Bisimulation P Q (Bisimilar P Q) := by
  rintro s t ⟨R, hR, hst⟩
  obtain ⟨c⟩ := hR s t hst
  exact ⟨{ c with supported := fun i o s' t' hp => ⟨R, hR, c.supported i o s' t' hp⟩ }⟩

theorem bisimilar_sameBehavior {s : S} {t : T} (h : Bisimilar P Q s t) :
    SameBehavior P Q s t := Bisimulation.sameBehavior bisimilar_is_bisimulation h

theorem distinguishing_trace {R : S → T → Prop} (hR : Bisimulation P Q R)
    (policy : Policy I O) (history future : List O) {s : S} {t : T}
    (hne : P.trace policy history future s ≠ Q.trace policy history future t) : ¬ R s t :=
  fun hst => hne (Bisimulation.trace_eq hR policy history future hst)

/-- An exact state representation/quotient must push forward the JOINT output and
next-state distribution. Hidden state may be merged only when this equation holds
for EVERY source state and admitted input. f need not be injective or surjective. -/
def Refines (P : Machine S I O) (Q : Machine T I O) [DecidableEq T] (f : S → T) : Prop :=
  ∀ i s o t', (∑ s', if f s' = t' then P.step i s o s' else 0) = Q.step i (f s) o t'

theorem Refines.bisimulation [DecidableEq T] {f : S → T} (h : Refines P Q f) :
    Bisimulation P Q (fun s t => f s = t) := by
  intro s t hst
  subst t
  refine ⟨{
    mass := fun i o s' t' => if f s' = t' then P.step i s o s' else 0
    nonneg := by intros; split <;> simp_all [P.nonneg]
    left := by intros; simp
    right := by intros; exact h _ _ _ _
    supported := ?_ }⟩
  intro i o s' t' hp
  split_ifs at hp with he
  · exact he
  · norm_num at hp

theorem Refines.sameBehavior [DecidableEq T] {f : S → T} (h : Refines P Q f) (s : S) :
    SameBehavior P Q s (f s) := Bisimulation.sameBehavior (Refines.bisimulation h) rfl

theorem Refines.refl [DecidableEq S] : Refines P P id := by
  simp [Refines]

/-- Certificates compose, so a chain of representation changes does not need
to recheck arbitrary long sample traces after each implementation pass. -/
theorem Refines.comp {V : Type*} [Fintype V] [DecidableEq T] [DecidableEq V]
    {R : Machine V I O} {f : S → T} {g : T → V}
    (hPQ : Refines P Q f) (hQR : Refines Q R g) : Refines P R (g ∘ f) := by
  intro i s o v
  obtain ⟨c⟩ := Refines.bisimulation hPQ s (f s) rfl
  have he := c.expectation_at_output i o
    (fun s' => if g (f s') = v then 1 else 0)
    (fun t' => if g t' = v then 1 else 0)
    (by intro s' t' hh; rw [hh])
  simp only [mul_ite, mul_one, mul_zero] at he
  exact he.trans (hQR i (f s) o v)

/-- Step-indexed behavioural equality alone does not fix physical time units.
A physical implementation certificate must ALSO identify the two world clocks.
The clock is supplied by the existing worlds, not selected by the interpreter. -/
structure TimedRefinement (P : Machine S I O) (Q : Machine T I O) [DecidableEq T]
    (f : S → T) (sourceClock targetClock : ℝ) : Prop where
  source_positive : 0 < sourceClock
  same_clock : sourceClock = targetClock
  representation : Refines P Q f

theorem TimedRefinement.elapsed_eq [DecidableEq T] {f : S → T} {a b : ℝ}
    (h : TimedRefinement P Q f a b) (n : ℕ) : (n : ℝ) / a = (n : ℝ) / b := by
  rw [h.same_clock]

variable (P)

namespace Machine

/-- First occurrence of an observed event under the same causal input context. -/
noncomputable def firstHit (policy : Policy I O) (mark : O → Bool) : List O → ℕ → S → ℝ
  | _, 0 => fun _ => 0
  | history, k + 1 => fun s => ∑ o, ∑ s', P.step (policy history) s o s' *
      (if mark o then 1 else firstHit policy mark (history ++ [o]) k s')

end Machine

variable {P}

theorem Bisimulation.firstHit_eq {R : S → T → Prop} (hR : Bisimulation P Q R)
    (policy : Policy I O) (mark : O → Bool) (history : List O) (k : ℕ)
    {s : S} {t : T} (hst : R s t) :
    P.firstHit policy mark history k s = Q.firstHit policy mark history k t := by
  induction k generalizing history s t with
  | zero => rfl
  | succ k ih =>
    obtain ⟨c⟩ := hR s t hst
    apply Finset.sum_congr rfl
    intro o _
    apply c.expectation_at_output (policy history) o
    intro s' t' hr
    split_ifs
    · rfl
    · exact ih _ hr

theorem Refines.firstHit_eq [DecidableEq T] {f : S → T} (h : Refines P Q f)
    (policy : Policy I O) (mark : O → Bool) (history : List O) (k : ℕ) (s : S) :
    P.firstHit policy mark history k s = Q.firstHit policy mark history k (f s) :=
  Bisimulation.firstHit_eq (Refines.bisimulation h) policy mark history k rfl

/-- To preserve the existing nonempty-body + recovery-threshold test, content
must ALSO be represented faithfully. Matching emitted words alone is insufficient. -/
theorem Refines.recovery_test_iff [DecidableEq T] {f : S → T} (h : Refines P Q f)
    (contentS : S → ℕ) (contentT : T → ℕ) (hcontent : ∀ s, contentS s = contentT (f s))
    (policy : Policy I O) (mark : O → Bool) (history : List O) (k : ℕ) (p : ℝ) (s : S) :
    (0 < contentS s ∧ p ≤ P.firstHit policy mark history k s) ↔
      (0 < contentT (f s) ∧ p ≤ Q.firstHit policy mark history k (f s)) := by
  rw [hcontent, Refines.firstHit_eq h]

variable (P)

theorem firstHit_bounds (policy : Policy I O) (mark : O → Bool) (history : List O)
    (k : ℕ) (s : S) :
    0 ≤ P.firstHit policy mark history k s ∧ P.firstHit policy mark history k s ≤ 1 := by
  induction k generalizing history s with
  | zero => simp [Machine.firstHit]
  | succ k ih =>
    constructor
    · exact Finset.sum_nonneg fun o _ => Finset.sum_nonneg fun s' _ =>
        mul_nonneg (P.nonneg _ _ _ _) (by split_ifs <;> simp_all [(ih (history ++ [o]) s').1])
    · calc
        _ ≤ ∑ o, ∑ s', P.step (policy history) s o s' * 1 :=
          Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun s' _ =>
            mul_le_mul_of_nonneg_left (by split_ifs <;> simp_all [(ih (history ++ [o]) s').2])
              (P.nonneg _ _ _ _)
        _ = 1 := by simp [P.sum_one]

end CGB.Interaction
