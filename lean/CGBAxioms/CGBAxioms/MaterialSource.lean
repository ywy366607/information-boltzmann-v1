import CGBAxioms.MaterialExchange
import CGBAxioms.InteractionBridge

/-!
# Source-resolved material transport

Fixed token identities locate material in the body, accessible source, return
sink or exterior. Counts are derived from transport, not supplied as a second
free ledger. Recycling and exterior supply are explicit; neither material
conservation nor a finite source is by itself an energy or mortality theorem.
-/

open Finset

namespace CGB.MaterialSource

inductive Place where
  | body | source | sink | outside
  deriving DecidableEq

variable {C : Type*} [Fintype C]

def count (owner : C → Place) (p : Place) : ℕ :=
  ∑ c, if owner c = p then 1 else 0

def moved (before after : C → Place) (p q : Place) : ℕ :=
  ∑ c, if before c = p ∧ after c = q then 1 else 0

/-- Declared interface: uptake, release, exterior supply/export, and recycling.
The choice of accessible source/return sink is part of the carrier model. -/
def Allowed (before after : C → Place) : Prop := ∀ c,
  before c = after c ∨
  (before c = .source ∧ after c = .body) ∨
  (before c = .body ∧ after c = .sink) ∨
  (before c = .outside ∧ after c = .source) ∨
  (before c = .source ∧ after c = .outside) ∨
  (before c = .sink ∧ after c = .source) ∨
  (before c = .sink ∧ after c = .outside)

/-- One transport law yields both body balance and the accessible-source budget.
Positive gross turnover at constant body count consumes source stock unless
there is explicit supply or recycling. No irreversibility is postulated. -/
theorem balance (before after : C → Place) (h : Allowed before after) :
    count after .body + moved before after .body .sink =
      count before .body + moved before after .source .body ∧
    count after .source + moved before after .source .body + moved before after .source .outside =
      count before .source + moved before after .outside .source +
        moved before after .sink .source ∧
    count after .body + count after .source + count after .sink +
        moved before after .source .outside + moved before after .sink .outside =
      count before .body + count before .source + count before .sink +
        moved before after .outside .source := by
  refine ⟨?_, ?_, ?_⟩
  all_goals simp only [count, moved, ← Finset.sum_add_distrib]
  all_goals
    apply Finset.sum_congr rfl
    intro c _
    have hc := h c
    cases hb : before c <;> cases ha : after c <;> simp_all

theorem moved_le_stock (before after : C → Place) (p q : Place) :
    moved before after p q ≤ count before p := by
  apply Finset.sum_le_sum
  intro c _
  by_cases hb : before c = p <;> by_cases ha : after c = q <;> simp [hb, ha]

/-- An empty accessible source cannot feed the body in this tick. Supply first
arrives in the source; there is no unregistered exterior-to-body shortcut. -/
theorem no_intake_from_empty_source (before after : C → Place)
    (h : count before .source = 0) : moved before after .source .body = 0 := by
  have hh := moved_le_stock before after .source .body
  omega

/-- With identities fixed, an unchanged ownership state cannot be labelled as
positive gross transport. A coarse constant-content state may still refine it. -/
theorem no_transport_without_relocation (owner : C → Place) :
    moved owner owner .source .body = 0 ∧ moved owner owner .body .sink = 0 := by
  constructor <;> apply Finset.sum_eq_zero
  all_goals
    intro c _
    cases owner c <;> simp

/-- Exact finite-prefix stock law, applicable to each supported sample path.
It includes both replenishment and recycling instead of asserting that matter
must be consumed. All sums are in the existing external tick count. -/
theorem source_budget (path : ℕ → C → Place)
    (h : ∀ t, Allowed (path t) (path (t + 1))) (n : ℕ) :
    count (path n) .source +
      ∑ t ∈ Finset.range n, (moved (path t) (path (t + 1)) .source .body +
        moved (path t) (path (t + 1)) .source .outside) =
    count (path 0) .source +
      ∑ t ∈ Finset.range n, (moved (path t) (path (t + 1)) .outside .source +
        moved (path t) (path (t + 1)) .sink .source) := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hb := (balance (path n) (path (n + 1)) (h n)).2.1
    simp only [Finset.sum_range_succ]
    omega

/-- A finite unreplenished, unrecycled source bounds cumulative intake;
this is a material-throughput conclusion, not a death/free-energy theorem. -/
theorem intake_bound_without_refill (path : ℕ → C → Place)
    (h : ∀ t, Allowed (path t) (path (t + 1)))
    (hz : ∀ t, moved (path t) (path (t + 1)) .outside .source +
      moved (path t) (path (t + 1)) .sink .source = 0) (n : ℕ) :
    (∑ t ∈ Finset.range n, moved (path t) (path (t + 1)) .source .body) ≤
      count (path 0) .source := by
  have hb := source_budget path h n
  simp only [hz, Finset.sum_const_zero, add_zero] at hb
  have hs : (∑ t ∈ Finset.range n, moved (path t) (path (t + 1)) .source .body) ≤
      ∑ t ∈ Finset.range n, (moved (path t) (path (t + 1)) .source .body +
        moved (path t) (path (t + 1)) .source .outside) :=
    Finset.sum_le_sum fun _ _ => Nat.le_add_right _ _
  omega

/-- Use the existing interaction/refinement calculus, not another lifetime or
probability framework. One input is the same external world tick/protocol. -/
noncomputable def machine {S E : Type*} [Fintype S] [Fintype E]
    (K : ℕ → EventChannels.Kernel S E) : Interaction.Machine S ℕ E where
  step t := (K t).step
  nonneg t := (K t).nonneg
  sum_one t := (K t).sum_one

/-- A carrier interpretation certificate, not an additional general axiom.
The timed refinement must be supplied: an arbitrary coarse kernel need not have
resources to realize its declared uptake. The microscopic data cannot alter the
coarse event/state law or add organism memory through that law. -/
structure Realization {R S E : Type*} [Fintype R] [Fintype S] [DecidableEq S] [Fintype E]
    (P : Interaction.Machine R ℕ E) (Q : Interaction.Machine S ℕ E)
    (owner : R → C → Place) (project : R → S) (content : S → ℕ)
    (incoming outgoing : ℕ → S → E → S → ℕ) (clock : ℝ) : Prop where
  process : Interaction.TimedRefinement P Q project clock clock
  body_count : ∀ r, count (owner r) .body = content (project r)
  transport : ∀ t r e r', 0 < P.step t r e r' →
    Allowed (owner r) (owner r') ∧
    moved (owner r) (owner r') .source .body = incoming t (project r) e (project r') ∧
    moved (owner r) (owner r') .body .sink = outgoing t (project r) e (project r')

end CGB.MaterialSource
