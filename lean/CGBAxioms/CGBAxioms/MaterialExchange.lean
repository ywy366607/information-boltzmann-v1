import CGBAxioms.EventChannels

/-!
# Versioned gross material exchange observer

The old net-growth observer remains intact. A ledger is registered channel data
with balance/boundary certificates; those certificates do not identify chemical
energy or prove that a physical implementation actually delivers the labelled
matter. Finite ExchangeBook encodings charge the additional channel data.
-/

open Finset

namespace CGB.Medium

variable {U Λ A I : Type*} [Fintype U] (M : Medium U Λ A I)

structure MaterialLedger where
  incoming : A → (U → I) → (U → Λ) → ℕ
  outgoing : A → (U → I) → (U → Λ) → ℕ
  balance : ∀ a s x, 0 < M.rate a s x →
    M.total (M.act a s x) + outgoing a s x = M.total x + incoming a s x
  closed_zero : ∀ a s x, M.Closed a → 0 < M.rate a s x →
    incoming a s x = 0 ∧ outgoing a s x = 0

theorem ledger_p2a (L : M.MaterialLedger) : M.P2a := by
  intro a s x ha hp
  have hb := L.balance a s x hp
  obtain ⟨hi, ho⟩ := L.closed_zero a s x ha hp
  simpa [hi, ho] using hb

/-- A positive net increase necessarily includes incoming material. The reverse
need not hold because incoming and outgoing can be equal and positive. -/
theorem net_increase_implies_incoming (L : M.MaterialLedger)
    (a : A) (s : U → I) (x : U → Λ) (hp : 0 < M.rate a s x)
    (hi : M.total x < M.total (M.act a s x)) : 0 < L.incoming a s x := by
  have hb := L.balance a s x hp
  omega

theorem equal_content_iff_balanced_exchange (L : M.MaterialLedger)
    (a : A) (s : U → I) (x : U → Λ) (hp : 0 < M.rate a s x) :
    M.total (M.act a s x) = M.total x ↔ L.incoming a s x = L.outgoing a s x := by
  have hb := L.balance a s x hp
  omega

end CGB.Medium

namespace CGB.Exchange

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ]
  [Fintype A] (M : Medium U Λ A I) (Wd : World U Λ I) (he : M.EventClocked Wd.clock)
  (L : M.MaterialLedger)

/-- Actual registered channel intake, including state-preserving replacement.
Idle slots carry no material; protocol time is read only by the observer. -/
def intake (sched : ℕ → Wd.W → U → I) (t : ℕ) (z : (U → Λ) × Wd.W) :
    Option A → ((U → Λ) × Wd.W) → Bool
  | none, _ => false
  | some a, _ => decide (0 < L.incoming a (sched t z.2) z.1)

noncomputable def recovery (cut : Bool) (sched : ℕ → Wd.W → U → I)
    (t k : ℕ) (z : (U → Λ) × Wd.W) : ℝ :=
  EventChannels.Kernel.firstHit (M.jointEventKernel Wd he cut sched)
    (intake M Wd L sched) t k z

theorem recovery_bounds (cut : Bool) (sched : ℕ → Wd.W → U → I)
    (t k : ℕ) (z : (U → Λ) × Wd.W) :
    0 ≤ recovery M Wd he L cut sched t k z ∧ recovery M Wd he L cut sched t k z ≤ 1 :=
  EventChannels.Kernel.firstHit_bounds _ _ _ _ _

theorem recovery_zero_without_incoming
    (hz : ∀ a s x, L.incoming a s x = 0)
    (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ)
    (z : (U → Λ) × Wd.W) : recovery M Wd he L cut sched t k z = 0 := by
  have hm : intake M Wd L sched = fun _ _ _ _ => false := by
    funext t z e z'
    cases e <;> simp [intake, hz]
  unfold recovery
  rw [hm]
  exact EventChannels.Kernel.firstHit_unmarked _ _ _ _

noncomputable def alive (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ)
    (z : (U → Λ) × Wd.W) : ℝ :=
  Viability.alive (0 < M.total z.1) Wd.recovery (recovery M Wd he L cut sched t Wd.horizon z)

noncomputable def survival (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ) : ℝ :=
  Viability.survival (fun x w => alive M Wd he L cut sched n (x, w))
    (fun x w => if cut then M.lawCut Wd sched n x w else M.law Wd sched n x w)

theorem survival_bounds (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ) :
    0 ≤ survival M Wd he L cut sched n ∧ survival M Wd he L cut sched n ≤ 1 := by
  have hc := M.eventClocked_implies_clocked he
  apply Viability.survival_bounds
  · intro x w; exact Viability.alive_bounds _ _ _
  · intro x w
    cases cut
    · exact M.law_nonneg_of_clocked Wd hc sched n x w
    · exact M.lawCut_nonneg_of_clocked Wd hc sched n x w
  · cases cut <;> simp [M.law_sum, M.lawCut_sum]

def Persists (δ : ℝ) : Prop :=
  Viability.Persists (survival M Wd he L false (fun _ => Wd.sense)) δ

def ToleratesSensoryInterruption (δ : ℝ) (T : ℕ) : Prop :=
  Viability.ToleratesSensoryInterruption
    (fun n₀ n => survival M Wd he L false (M.interrupt Wd n₀ T) n) δ T

def SelfMaintains : Prop :=
  Viability.SelfMaintains (survival M Wd he L false (fun _ => Wd.sense))
    (survival M Wd he L true (fun _ => Wd.sense))

theorem not_persists_without_incoming (hz : ∀ a s x, L.incoming a s x = 0) (δ : ℝ) :
    ¬ Persists M Wd he L δ := by
  rintro ⟨hδ, hp⟩
  have hn : ¬ Wd.recovery ≤ 0 := not_le_of_gt Wd.recovery_pos
  have ha : ∀ cut sched t z, alive M Wd he L cut sched t z = 0 := by
    intros
    classical
    simp [alive, recovery_zero_without_incoming M Wd he L hz, hn]
  have hzero : survival M Wd he L false (fun _ => Wd.sense) 0 = 0 := by
    simp [survival, ha]
  have h := hp 0
  rw [hzero] at h
  linarith

end CGB.Exchange

namespace CGB.Description

/-- Complete additional event-count data, encoded WITH the original book.
No uncharged arbitrary ledger is carried beside a short base description. -/
structure ExchangeBook where
  base : Book
  transfers : List (ℕ × ℕ)
  deriving Encodable, DecidableEq

namespace ExchangeBook

def transfer (b : ExchangeBook) (a : b.base.Action) : ℕ × ℕ :=
  b.transfers[a.val]?.getD (0, 0)

/-- Rates/guards remain in the base book. Transfer counts are per occurrence of
that channel; disabled channels cannot create an observed transfer. -/
structure Accountable (b : ExchangeBook) : Prop where
  balance : ∀ a s x, 0 < b.base.medium.rate a s x →
    b.base.medium.total (b.base.medium.act a s x) + (b.transfer a).2 =
      b.base.medium.total x + (b.transfer a).1
  closed_zero : ∀ a s x, b.base.medium.Closed a → 0 < b.base.medium.rate a s x →
    (b.transfer a).1 = 0 ∧ (b.transfer a).2 = 0

noncomputable def ledger (b : ExchangeBook) (h : b.Accountable) :
    b.base.medium.MaterialLedger where
  incoming a _ _ := (b.transfer a).1
  outgoing a _ _ := (b.transfer a).2
  balance := h.balance
  closed_zero := h.closed_zero

def GenomeBudget (b : ExchangeBook) (graph : WiringReference b.base.sites)
    (num den : ℕ) : Prop :=
  0 < graph.payload.length ∧ 0 < den ∧ den * bits b ≤ num * graph.payload.length

theorem one_percent_budget (b : ExchangeBook) (graph : WiringReference b.base.sites) :
    b.GenomeBudget graph 1 100 ↔
      0 < b.base.sites * b.base.sites ∧ 100 * bits b ≤ b.base.sites * b.base.sites := by
  simp [GenomeBudget, WiringReference.payload_length]

end ExchangeBook
end CGB.Description

namespace CGB.Exchange

/-- Versioned MATERIAL-throughput subtarget. Resource realization, usable
free-energy flow and cognitive function remain separate obligations. -/
structure EncodedCandidate (b : Description.ExchangeBook) (h : b.Accountable)
    (Wd : World b.base.Site b.base.State b.base.Stim)
    (graph : Description.WiringReference b.base.sites) (num den : ℕ)
    (δ : ℝ) (κ T : ℕ) : Prop where
  axioms : b.base.medium.Axioms
  clocked : b.base.medium.EventClocked Wd.clock
  budget : b.GenomeBudget graph num den
  body : b.base.medium.ReadsBoundary Wd
  rich : κ * Fintype.card (b.base.Site → b.base.State) ≤ Fintype.card Wd.W
  persists : Persists b.base.medium Wd clocked (b.ledger h) δ
  tolerates : ToleratesSensoryInterruption b.base.medium Wd clocked (b.ledger h) δ T
  selfMaintains : SelfMaintains b.base.medium Wd clocked (b.ledger h)

end CGB.Exchange
