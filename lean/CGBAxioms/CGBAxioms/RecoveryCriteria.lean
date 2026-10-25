import CGBAxioms.EnergyBudget

/-!
# Local sufficient criteria for actual resource recovery

Regions and ranks are proof objects, not new organism state, policy inputs or
life axioms. Every unmarked supported transition must make strict progress
toward actual registered intake; closure keeps that progress physically usable.
-/

open Finset

namespace CGB.RecoveryCriteria

variable {S E : Type*} [Fintype S] [Fintype E]

structure Ranked (K : ℕ → EventChannels.Kernel S E)
    (mark : ℕ → S → E → S → Bool) (horizon : ℕ) where
  region : S → Prop
  rank : S → ℕ
  bounded : ∀ z, region z → rank z < horizon
  closed : ∀ t z e z', region z → 0 < (K t).step z e z' → region z'
  progresses : ∀ t z e z', region z → 0 < (K t).step z e z' →
    mark t z e z' = false → rank z' < rank z

/-- A local rank certificate proves recovery in the actual future kernel,
without assuming a favourable stimulus sequence or the desired recovery score. -/
theorem ranked_firstHit {K : ℕ → EventChannels.Kernel S E}
    {mark : ℕ → S → E → S → Bool} {horizon : ℕ} (C : Ranked K mark horizon)
    (t k : ℕ) (z : S) (hg : C.region z) (hk : C.rank z < k) :
    EventChannels.Kernel.firstHit K mark t k z = 1 := by
  induction k generalizing t z with
  | zero => omega
  | succ k ih =>
    change (∑ e, ∑ z', (K t).step z e z' *
      (if mark t z e z' then 1 else EventChannels.Kernel.firstHit K mark (t + 1) k z')) = 1
    calc
      _ = ∑ e, ∑ z', (K t).step z e z' := by
        apply Finset.sum_congr rfl
        intro e _
        apply Finset.sum_congr rfl
        intro z' _
        rcases ((K t).nonneg z e z').lt_or_eq with hp | hz
        · cases hm : mark t z e z' with
          | true => simp [hm]
          | false =>
            have hd := C.progresses t z e z' hg hp hm
            rw [show EventChannels.Kernel.firstHit K mark (t + 1) k z' = 1 from
              ih (t + 1) z' (C.closed t z e z' hg hp) (by omega)]
            simp
        · simp [← hz]
      _ = 1 := (K t).sum_one z

theorem firstHit_zero_of_closed (K : ℕ → EventChannels.Kernel S E)
    (mark : ℕ → S → E → S → Bool) (region : S → Prop)
    (closed : ∀ t z e z', region z → 0 < (K t).step z e z' → region z')
    (unmarked : ∀ t z e z', region z → 0 < (K t).step z e z' → mark t z e z' = false)
    (t k : ℕ) (z : S) (hg : region z) :
    EventChannels.Kernel.firstHit K mark t k z = 0 := by
  induction k generalizing t z with
  | zero => rfl
  | succ k ih =>
    change (∑ e, ∑ z', (K t).step z e z' *
      (if mark t z e z' then 1 else EventChannels.Kernel.firstHit K mark (t + 1) k z')) = 0
    apply Finset.sum_eq_zero
    intro e _
    apply Finset.sum_eq_zero
    intro z' _
    rcases ((K t).nonneg z e z').lt_or_eq with hp | hz
    · rw [unmarked t z e z' hg hp, ih (t + 1) z' (closed t z e z' hg hp)]
      simp
    · simp [← hz]

private theorem support_closed [DecidableEq S] (K : ℕ → EventChannels.Kernel S E)
    (initial : S) (region : S → Prop) (hi : region initial)
    (hc : ∀ t z e z', region z → 0 < (K t).step z e z' → region z')
    (n : ℕ) (z : S) (hz : ¬region z) :
    EventChannels.Kernel.stateLaw K (fun z => if z = initial then 1 else 0) n z = 0 := by
  classical
  induction n generalizing z with
  | zero =>
    have hn : z ≠ initial := by intro h; subst z; exact hz hi
    simp [EventChannels.Kernel.stateLaw, hn]
  | succ n ih =>
    change (∑ y, EventChannels.Kernel.stateLaw K (fun z => if z = initial then 1 else 0) n y *
      (∑ e, (K n).step y e z)) = 0
    apply Finset.sum_eq_zero
    intro y _
    by_cases hy : region y
    · have he (e : E) : (K n).step y e z = 0 := by
        by_cases hp : 0 < (K n).step y e z
        · exact False.elim (hz (hc n y e z hy hp))
        · exact le_antisymm (le_of_not_gt hp) ((K n).nonneg y e z)
      simp [he]
    · rw [ih y hy, zero_mul]

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
  [Fintype A] (M : Medium U Λ A I) (Wd : World U Λ I) (he : M.EventClocked Wd.clock)
  (L : M.MaterialLedger)

/-- Reusable sufficient criterion for the original Exchange score. Physical
admission, actual marked resource channels and the original world clock stay
at the call site. It proves a target; it does not define life by the certificate. -/
theorem survival_of_ranked (ht : Wd.recovery ≤ 1)
    (C : Ranked (M.jointEventKernel Wd he false (fun _ => Wd.sense))
      (Exchange.intake M Wd L (fun _ => Wd.sense)) Wd.horizon)
    (hi : C.region (M.init, Wd.start))
    (positive : ∀ z, C.region z → 0 < M.total z.1) (n : ℕ) :
    Exchange.survival M Wd he L false (fun _ => Wd.sense) n = 1 := by
  classical
  let K := M.jointEventKernel Wd he false (fun _ => Wd.sense)
  have law_eq (z : (U → Λ) × Wd.W) :
      EventChannels.Kernel.stateLaw K
          (fun z => if z = (M.init, Wd.start) then (1 : ℝ) else 0) n z =
        M.law Wd (fun _ => Wd.sense) n z.1 z.2 := by
    have hb : (fun z : (U → Λ) × Wd.W => if z = (M.init, Wd.start) then (1 : ℝ) else 0) =
        (fun z => if z.1 = M.init ∧ z.2 = Wd.start then 1 else 0) := by
      funext z
      simp [Prod.ext_iff]
    rw [hb]
    exact M.jointEventKernel_stateLaw Wd he false (fun _ => Wd.sense) n z
  have point (x : U → Λ) (w : Wd.W) :
      Exchange.alive M Wd he L false (fun _ => Wd.sense) n (x, w) *
        M.law Wd (fun _ => Wd.sense) n x w = M.law Wd (fun _ => Wd.sense) n x w := by
    by_cases hg : C.region (x, w)
    · have hr := ranked_firstHit C n Wd.horizon (x, w) hg (C.bounded (x, w) hg)
      have ha : Exchange.alive M Wd he L false (fun _ => Wd.sense) n (x, w) = 1 := by
        unfold Exchange.alive Viability.alive
        apply ite_eq_left
        refine ⟨positive (x, w) hg, ?_⟩
        change Wd.recovery ≤ EventChannels.Kernel.firstHit K
          (Exchange.intake M Wd L (fun _ => Wd.sense)) n Wd.horizon (x, w)
        rw [hr]
        exact ht
      rw [ha, one_mul]
    · have hz := support_closed K (M.init, Wd.start) C.region hi C.closed n (x, w) hg
      rw [law_eq] at hz
      rw [hz, mul_zero]
  change (∑ x, ∑ w, Exchange.alive M Wd he L false (fun _ => Wd.sense) n (x, w) *
    M.law Wd (fun _ => Wd.sense) n x w) = 1
  simp only [point]
  exact M.law_sum Wd (fun _ => Wd.sense) n

end CGB.RecoveryCriteria
