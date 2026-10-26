import CGBAxioms.Nesting

/-!
Exploratory consequences of nesting, not a new infinite Medium instance.
No production module or general axiom is added. The resource/delay contract
below is explicit; it does not follow from nesting or from P1b alone.
-/
open Finset

namespace CGB.InfiniteNestingReview

-- A finite spatial carrier permits infinite names, but a decreasing family
-- eventually stops exposing new subsets. The stabilization INDEX is unbounded.
theorem finite_regions_stabilize {U : Type*} [DecidableEq U]
    (O : ℕ → Finset U) (hO : Antitone O) :
    ∃ N, ∀ n, N ≤ n → O n = O N := by
  classical
  let P : ℕ → Prop := fun k => ∃ n, (O n).card = k
  have hp : ∃ k, P k := ⟨(O 0).card, 0, rfl⟩
  obtain ⟨N, hN⟩ := Nat.find_spec hp
  refine ⟨N, fun n hn => Finset.eq_of_subset_of_card_le (hO hn) ?_⟩
  rw [hN]
  exact Nat.find_min' hp ⟨n, rfl⟩

theorem strict_region_depth {U : Type*} [DecidableEq U]
    (O : ℕ → Finset U) (n : ℕ)
    (hO : ∀ k < n, O (k + 1) ⊂ O k) : n + (O n).card ≤ (O 0).card := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hi := ih (fun k hk => hO k (by omega))
    have hd := Finset.card_lt_card (hO n (by omega))
    omega

-- Every layer carrying an independently selectable bit needs REAL state
-- distinctions. Infinite products of bits cannot fit in a finite body.
theorem independent_bits_bound {S : Type*} [Fintype S] {n : ℕ}
    (encode : (Fin n → Bool) → S) (hi : Function.Injective encode) :
    2 ^ n ≤ Fintype.card S := by
  simpa using Fintype.card_le_of_injective encode hi

theorem no_unbounded_independent_bits {S : Type*} [Fintype S] :
    ¬ (∀ n, ∃ encode : (Fin n → Bool) → S, Function.Injective encode) := by
  intro h
  obtain ⟨encode, hi⟩ := h (Fintype.card S)
  exact (Nat.lt_two_pow_self (n := Fintype.card S)).not_ge (independent_bits_bound encode hi)

-- One lemma serves either disjoint layer costs or sequential crossing delays.
-- It does NOT count the same matter once per enclosing boundary.
theorem depth_budget_bound (cost : ℕ → ℝ) (ε B : ℝ) (n : ℕ)
    (hlower : ∀ k < n, ε ≤ cost k) (hbudget : ∑ k ∈ range n, cost k ≤ B) :
    (n : ℝ) * ε ≤ B := by
  calc
    (n : ℝ) * ε = ∑ _k ∈ range n, ε := by simp
    _ ≤ ∑ k ∈ range n, cost k := Finset.sum_le_sum fun k hk => hlower k (mem_range.mp hk)
    _ ≤ B := hbudget

theorem no_uniform_cost_infinite_depth (cost : ℕ → ℝ) {ε : ℝ} (he : 0 < ε)
    (hlower : ∀ k, ε ≤ cost k) : ¬ ∃ B, ∀ n, ∑ k ∈ range n, cost k ≤ B := by
  rintro ⟨B, hb⟩
  obtain ⟨n, hn⟩ := exists_nat_gt (B / ε)
  have hlo : B < (n : ℝ) * ε := (div_lt_iff₀ he).mp hn
  have hhi := depth_budget_bound cost ε B n (fun k _ => hlower k) (hb n)
  linarith

-- Infinite positive costs may fit a finite prefix budget if there is no
-- positive lower bound. This alone constructs no infinite physical dynamics.
theorem geometric_prefix (n : ℕ) :
    (∑ k ∈ range n, ((1 : ℝ) / 2) ^ (k + 1)) = 1 - ((1 : ℝ) / 2) ^ n := by
  induction n with
  | zero => simp
  | succ n ih => rw [sum_range_succ, ih, pow_succ]; ring

theorem geometric_costs_fit :
    (∀ k : ℕ, 0 < ((1 : ℝ) / 2) ^ (k + 1)) ∧
    (∀ n : ℕ, (∑ k ∈ range n, ((1 : ℝ) / 2) ^ (k + 1)) < 1) := by
  constructor
  · intro k; positivity
  · intro n
    rw [geometric_prefix]
    have : 0 < ((1 : ℝ) / 2) ^ n := by positivity
    linarith

-- For the SAME fine state, repeated freezing is intersection, not a fresh
-- bath, coarse-graining operation, or source of irreversibility.
theorem freeze_twice {U Λ : Type*} (K : CGB.JumpProcess (U → Λ)) (O P : Set U)
    (x y : U → Λ) :
    ((K.freezeOutside O).freezeOutside P).rate x y =
      (K.freezeOutside (O ∩ P)).rate x y := by
  classical
  have he : (∀ u, u ∉ O ∩ P → x u = y u) ↔
      (∀ u, u ∉ O → x u = y u) ∧ (∀ u, u ∉ P → x u = y u) := by
    constructor
    · intro h
      exact ⟨fun u hu => h u (fun hp => hu hp.1),
        fun u hu => h u (fun hp => hu hp.2)⟩
    · rintro ⟨ho, hp⟩ u hu
      by_cases h : u ∈ O
      · exact hp u (fun h' => hu ⟨h, h'⟩)
      · exact ho u h
  simp only [CGB.JumpProcess.freezeOutside, he]
  split_ifs <;> simp_all

theorem infinite_freeze_balance {U Λ : Type*} (K : CGB.JumpProcess (U → Λ))
    (O : ℕ → Set U) {π : (U → Λ) → ℝ}
    (hdb : ∀ x y, π x * K.rate x y = π y * K.rate y x) (x y : U → Λ) :
    π x * (K.freezeOutside (⋂ n, O n)).rate x y =
      π y * (K.freezeOutside (⋂ n, O n)).rate y x :=
  K.freezeOutside_balance (⋂ n, O n) hdb x y

-- A reversible microscopic generator can lose exact Markov closure when the
-- observer merges states. This is NOT an experimental capability benchmark.
theorem reversible_projection_not_closed :
    ∃ (rate : Fin 3 → Fin 3 → ℝ) (view : Fin 3 → Bool),
      (∀ i j, 0 ≤ rate i j) ∧ (∀ i j, rate i j = rate j i) ∧
      ¬ ∃ coarse : Bool → Bool → ℝ, ∀ i b,
        (∑ j : Fin 3, if view j = b then rate i j else 0) = coarse (view i) b := by
  let rate : Fin 3 → Fin 3 → ℝ := fun i j =>
    if (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0) ∨
       (i = 1 ∧ j = 2) ∨ (i = 2 ∧ j = 1) then 1 else 0
  let view : Fin 3 → Bool := fun i => decide (i = 2)
  refine ⟨rate, view, ?_, ?_, ?_⟩
  · intro i j; dsimp [rate]; split_ifs <;> norm_num
  · intro i j; fin_cases i <;> fin_cases j <;> norm_num [rate]
  · rintro ⟨coarse, hc⟩
    have h0 := hc 0 true
    have h1 := hc 1 true
    norm_num [view, rate, Fin.sum_univ_succ] at h0 h1
    linarith

#print axioms finite_regions_stabilize
#print axioms strict_region_depth
#print axioms independent_bits_bound
#print axioms no_unbounded_independent_bits
#print axioms depth_budget_bound
#print axioms no_uniform_cost_infinite_depth
#print axioms geometric_prefix
#print axioms geometric_costs_fit
#print axioms freeze_twice
#print axioms infinite_freeze_balance
#print axioms reversible_projection_not_closed

end CGB.InfiniteNestingReview
