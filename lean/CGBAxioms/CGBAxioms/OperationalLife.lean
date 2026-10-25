import CGBAxioms.RuleEncoding
import CGBAxioms.WorldRecovery

/-!
# Versioned operational target: finite descriptions and actual-world intake

The old LifeConjecture remains a historical statement, not a proved equivalent.
This target uses actual current joint states and the actual future protocol on
each branch, including the loop-cut future. Targets are desired properties, not
assumptions added to P1--P3. Material intake is not yet usable-energy intake.
-/

open Finset

/- Shared viability calculus. Observer-specific recovery and world laws stay
at the call site; these abbreviations add neither dynamics nor admission rules. -/
namespace CGB.Viability

open Classical in
@[simp] noncomputable abbrev alive (positive : Prop) (threshold prospect : ℝ) : ℝ :=
  if positive ∧ threshold ≤ prospect then 1 else 0

theorem alive_bounds (positive : Prop) (threshold prospect : ℝ) :
    0 ≤ alive positive threshold prospect ∧ alive positive threshold prospect ≤ 1 := by
  classical
  unfold alive
  split_ifs <;> norm_num

@[simp] noncomputable abbrev survival {X W : Type*} [Fintype X] [Fintype W]
    (score law : X → W → ℝ) : ℝ := ∑ x, ∑ w, score x w * law x w

theorem survival_bounds {X W : Type*} [Fintype X] [Fintype W]
    (score law : X → W → ℝ)
    (ha : ∀ x w, 0 ≤ score x w ∧ score x w ≤ 1)
    (hl : ∀ x w, 0 ≤ law x w) (hs : (∑ x, ∑ w, law x w) = 1) :
    0 ≤ survival score law ∧ survival score law ≤ 1 := by
  constructor
  · exact Finset.sum_nonneg fun x _ => Finset.sum_nonneg fun w _ =>
      mul_nonneg (ha x w).1 (hl x w)
  · calc
      _ ≤ ∑ x, ∑ w, law x w := Finset.sum_le_sum fun x _ =>
        Finset.sum_le_sum fun w _ => mul_le_of_le_one_left (hl x w) (ha x w).2
      _ = 1 := hs

abbrev Persists (natural : ℕ → ℝ) (δ : ℝ) : Prop :=
  δ < 1 ∧ ∀ n, 1 - δ ≤ natural n

abbrev ToleratesSensoryInterruption (interrupted : ℕ → ℕ → ℝ) (δ : ℝ) (T : ℕ) : Prop :=
  0 < T ∧ ∀ n₀ n, 1 - δ ≤ interrupted n₀ n

abbrev SelfMaintains (natural cut : ℕ → ℝ) : Prop :=
  ∃ n₀ ε, 0 < ε ∧ ∀ n, n₀ ≤ n → cut n + ε ≤ natural n

end CGB.Viability

namespace CGB.Operational

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
  (M : Medium U Λ A I) (Wd : World U Λ I) (hc : M.Clocked Wd.clock)

/-- Success prospect from the actual state and clock time, under either the
closed protocol or the registered marginal-ghost cut. No rescue-world oracle. -/
noncomputable def recovery (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ)
    (z : (U → Λ) × Wd.W) : ℝ :=
  FirstHit.during (if cut then M.scheduledCutKernel Wd hc sched else
    M.scheduledKernel Wd hc sched) (M.materialIntake Wd) t k z

theorem recovery_bounds (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ)
    (z : (U → Λ) × Wd.W) :
    0 ≤ recovery M Wd hc cut sched t k z ∧ recovery M Wd hc cut sched t k z ≤ 1 :=
  FirstHit.during_bounds _ _ _ _ _

/-- For the natural uncut world, the general protocol reduces to actualRecover. -/
theorem natural_recovery (t k : ℕ) (x : U → Λ) (w : Wd.W) :
    recovery M Wd hc false (fun _ => Wd.sense) t k (x, w) =
      M.actualRecover Wd hc k x w := by
  have he : M.scheduledKernel Wd hc (fun _ => Wd.sense) = fun _ => M.actualKernel Wd hc := rfl
  simp only [recovery, Bool.false_eq_true, ↓reduceIte, he, Medium.actualRecover]
  exact FirstHit.during_constant _ _ _ _ _

noncomputable def alive (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ)
    (z : (U → Λ) × Wd.W) : ℝ :=
  Viability.alive (0 < M.total z.1) Wd.recovery (recovery M Wd hc cut sched t Wd.horizon z)

noncomputable def survival (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ) : ℝ :=
  Viability.survival (fun x w => alive M Wd hc cut sched n (x, w))
    (fun x w => if cut then M.lawCut Wd sched n x w else M.law Wd sched n x w)

theorem survival_bounds (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ) :
    0 ≤ survival M Wd hc cut sched n ∧ survival M Wd hc cut sched n ≤ 1 := by
  apply Viability.survival_bounds
  · intro x w; exact Viability.alive_bounds _ _ _
  · intro x w
    cases cut
    · exact M.law_nonneg_of_clocked Wd hc sched n x w
    · exact M.lawCut_nonneg_of_clocked Wd hc sched n x w
  · cases cut <;> simp [M.law_sum, M.lawCut_sum]

/-- Desired reliability in a specified actual protocol, not a new physical law. -/
def Persists (δ : ℝ) : Prop :=
  Viability.Persists (survival M Wd hc false (fun _ => Wd.sense)) δ

/-- This protocol cuts sensing; resource interventions use their own world laws. -/
def ToleratesSensoryInterruption (δ : ℝ) (T : ℕ) : Prop :=
  Viability.ToleratesSensoryInterruption
    (fun n₀ n => survival M Wd hc false (M.interrupt Wd n₀ T) n) δ T

def SelfMaintains : Prop :=
  Viability.SelfMaintains (survival M Wd hc false (fun _ => Wd.sense))
    (survival M Wd hc true (fun _ => Wd.sense))

/-- Parameter bounds follow from an actual persistence witness, not extra axioms. -/
theorem persistence_parameter_bounds {δ : ℝ} (h : Persists M Wd hc δ) :
    0 ≤ δ ∧ 0 < Wd.horizon ∧ Wd.recovery ≤ 1 := by
  have hl := h.2 0
  have hd : 0 ≤ δ := by
    linarith [(survival_bounds M Wd hc false (fun _ => Wd.sense) 0).2]
  have hz : survival M Wd hc false (fun _ => Wd.sense) 0 =
      alive M Wd hc false (fun _ => Wd.sense) 0 (M.init, Wd.start) := by
    classical
    simp [survival, Medium.law, mul_ite, ite_and]
  rw [hz] at hl
  have ha : 0 < alive M Wd hc false (fun _ => Wd.sense) 0 (M.init, Wd.start) := by linarith [h.1]
  have hr : Wd.recovery ≤ recovery M Wd hc false (fun _ => Wd.sense) 0 Wd.horizon
      (M.init, Wd.start) := by
    classical
    unfold alive Viability.alive at ha
    split_ifs at ha with hh
    · exact hh.2
    · norm_num at ha
  refine ⟨hd, ?_, hr.trans (recovery_bounds M Wd hc false _ _ _ _).2⟩
  by_contra hh
  have hzero : Wd.horizon = 0 := Nat.eq_zero_of_not_pos hh
  rw [hzero] at hr
  change Wd.recovery ≤ 0 at hr
  linarith [Wd.recovery_pos]

/-- The representation-aware target. Every property remains a proof obligation.
It deliberately does not claim the full life definition: resource starvation,
active recovery, non-frozen useful memory and autonomous inference remain open. -/
structure EncodedCandidate (b : Description.Book)
    (Wd : World b.Site b.State b.Stim) (graph : Description.WiringReference b.sites)
    (num den : ℕ) (δ : ℝ) (κ T : ℕ) : Prop where
  axioms : b.medium.Axioms
  clocked : b.medium.Clocked Wd.clock
  budget : Description.GenomeBudget b graph num den
  body : b.medium.ReadsBoundary Wd
  rich : κ * Fintype.card (b.Site → b.State) ≤ Fintype.card Wd.W
  persists : Persists b.medium Wd clocked δ
  tolerates : ToleratesSensoryInterruption b.medium Wd clocked δ T
  selfMaintains : SelfMaintains b.medium Wd clocked

end CGB.Operational
