import CGBAxioms.RecoveryCriteria

/-!
# Time-dependent certificates for a registered recovery protocol

Regions and ranks may refer to external protocol time. They are analysis
objects, not organism state or advance knowledge supplied to its policy. The
actual dynamics, marked intake and world horizon remain at the call site.
-/

open Finset

namespace CGB.ScheduledRecovery

variable {S E : Type*} [Fintype S] [Fintype E]

structure Ranked (K : ℕ → EventChannels.Kernel S E)
    (mark : ℕ → S → E → S → Bool) (horizon : ℕ) where
  region : ℕ → S → Prop
  rank : ℕ → S → ℕ
  bounded : ∀ t z, region t z → rank t z < horizon
  closed : ∀ t z e z', region t z → 0 < (K t).step z e z' → region (t + 1) z'
  progresses : ∀ t z e z', region t z → 0 < (K t).step z e z' →
    mark t z e z' = false → rank (t + 1) z' < rank t z

/-- A local rank certificate proves recovery in the actual future kernel,
without assuming a favourable stimulus sequence or the desired recovery score. -/
theorem ranked_firstHit {K : ℕ → EventChannels.Kernel S E}
    {mark : ℕ → S → E → S → Bool} {horizon : ℕ} (C : Ranked K mark horizon)
    (t k : ℕ) (z : S) (hg : C.region t z) (hk : C.rank t z < k) :
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

private theorem support_closed [DecidableEq S] (K : ℕ → EventChannels.Kernel S E)
    (initial : S) (region : ℕ → S → Prop) (hi : region 0 initial)
    (hc : ∀ t z e z', region t z → 0 < (K t).step z e z' → region (t + 1) z')
    (n : ℕ) (z : S) (hz : ¬region n z) :
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
    by_cases hy : region n y
    · have he (e : E) : (K n).step y e z = 0 := by
        by_cases hp : 0 < (K n).step y e z
        · exact False.elim (hz (hc n y e z hy hp))
        · exact le_antisymm (le_of_not_gt hp) ((K n).nonneg y e z)
      simp [he]
    · rw [ih y hy, zero_mul]

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
  [Fintype A] (M : Medium U Λ A I) (Wd : World U Λ I) (he : M.EventClocked Wd.clock)
  (L : M.MaterialLedger) (sched : ℕ → Wd.W → U → I)

/-- Reusable sufficient criterion for the original Exchange score. Physical
admission, actual marked resource channels and the original world clock stay
at the call site. It proves a target; it does not define life by the certificate. -/
theorem survival_of_ranked (ht : Wd.recovery ≤ 1)
    (C : Ranked (M.jointEventKernel Wd he false sched)
      (Exchange.intake M Wd L sched) Wd.horizon)
    (hi : C.region 0 (M.init, Wd.start))
    (positive : ∀ t z, C.region t z → 0 < M.total z.1) (n : ℕ) :
    Exchange.survival M Wd he L false sched n = 1 := by
  classical
  let K := M.jointEventKernel Wd he false sched
  have law_eq (z : (U → Λ) × Wd.W) :
      EventChannels.Kernel.stateLaw K
          (fun z => if z = (M.init, Wd.start) then (1 : ℝ) else 0) n z =
        M.law Wd sched n z.1 z.2 := by
    have hb : (fun z : (U → Λ) × Wd.W => if z = (M.init, Wd.start) then (1 : ℝ) else 0) =
        (fun z => if z.1 = M.init ∧ z.2 = Wd.start then 1 else 0) := by
      funext z
      simp [Prod.ext_iff]
    rw [hb]
    exact M.jointEventKernel_stateLaw Wd he false sched n z
  have point (x : U → Λ) (w : Wd.W) :
      Exchange.alive M Wd he L false sched n (x, w) *
        M.law Wd sched n x w = M.law Wd sched n x w := by
    by_cases hg : C.region n (x, w)
    · have hr := ranked_firstHit C n Wd.horizon (x, w) hg (C.bounded n (x, w) hg)
      have ha : Exchange.alive M Wd he L false sched n (x, w) = 1 := by
        unfold Exchange.alive Viability.alive
        apply ite_eq_left
        refine ⟨positive n (x, w) hg, ?_⟩
        change Wd.recovery ≤ EventChannels.Kernel.firstHit K
          (Exchange.intake M Wd L sched) n Wd.horizon (x, w)
        rw [hr]
        exact ht
      rw [ha, one_mul]
    · have hz := support_closed K (M.init, Wd.start) C.region hi C.closed n (x, w) hg
      rw [law_eq] at hz
      rw [hz, mul_zero]
  change (∑ x, ∑ w, Exchange.alive M Wd he L false sched n (x, w) *
    M.law Wd sched n x w) = 1
  simp only [point]
  exact M.law_sum Wd sched n

end CGB.ScheduledRecovery
