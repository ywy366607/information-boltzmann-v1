import Mathlib.Data.Nat.Log
import Mathlib.Order.MinMax
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# What division alone gives a tree (language organism, 2026-11-03)

A unit divides into two. This file reads the combinatorics of that one fact, and of `G8b`
(`three_part_balance_impossible`): one balanced site serves at most two independent parts.

* `leaves_eq_splits_succ`: every division adds exactly one leaf, whatever the shape.
* `leaves_le_two_pow_depth`, `clog_leaves_le_depth`: reading `k` parts through two-part units needs
  depth at least `⌈log₂ k⌉`.
* `exists_tree_of_clog_depth`: that depth is reached.
* `sharedPrefix_ultrametric`: units named by their division path form nested groups. If `x` shares
  `m` divisions with `y` and `y` shares `m` with `z`, then `x` shares `m` with `z`.

This is combinatorics of lineage. It says nothing about geometry, says nothing about where a
division happens, and the bridge from `Medium.jump` to a sum over parts is not formalised, so `G8b`
motivates the two-part node here and does not prove it for any body.
-/

namespace CGB.DivisionTree

/-- A division history: a unit, or a unit that divided into two. -/
inductive Lineage
  | leaf
  | split (l r : Lineage)

namespace Lineage

def leaves : Lineage → ℕ
  | leaf => 1
  | split l r => leaves l + leaves r

def splits : Lineage → ℕ
  | leaf => 0
  | split l r => splits l + splits r + 1

def depth : Lineage → ℕ
  | leaf => 0
  | split l r => max (depth l) (depth r) + 1

theorem leaves_eq_splits_succ (t : Lineage) : leaves t = splits t + 1 := by
  induction t with
  | leaf => rfl
  | split l r hl hr => simp only [leaves, splits, hl, hr]; ring

theorem leaves_pos (t : Lineage) : 0 < leaves t := by
  rw [leaves_eq_splits_succ]; omega

theorem leaves_le_two_pow_depth (t : Lineage) : leaves t ≤ 2 ^ depth t := by
  induction t with
  | leaf => simp [leaves, depth]
  | split l r hl hr =>
    have h1 : 2 ^ depth l ≤ 2 ^ max (depth l) (depth r) :=
      Nat.pow_le_pow_right (by norm_num) (le_max_left _ _)
    have h2 : 2 ^ depth r ≤ 2 ^ max (depth l) (depth r) :=
      Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)
    simp only [leaves, depth, pow_succ]
    omega

/-- Reading `k` parts through two-part units takes at least `⌈log₂ k⌉` levels. -/
theorem clog_leaves_le_depth (t : Lineage) : Nat.clog 2 (leaves t) ≤ depth t :=
  Nat.clog_le_of_le_pow (leaves_le_two_pow_depth t)

/-- The depth bound is reached for every number of parts. -/
theorem exists_tree_of_clog_depth (k : ℕ) (hk : 0 < k) :
    ∃ t : Lineage, leaves t = k ∧ depth t = Nat.clog 2 k := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    by_cases h1 : k = 1
    · exact ⟨leaf, by simp [leaves, h1], by simp [depth, h1]⟩
    · have hk2 : 2 ≤ k := by omega
      obtain ⟨l, hl, hdl⟩ := ih (k / 2) (by omega) (by omega)
      obtain ⟨r, hr, hdr⟩ := ih ((k + 1) / 2) (by omega) (by omega)
      refine ⟨split l r, by simp only [leaves, hl, hr]; omega, ?_⟩
      have hmono : Nat.clog 2 (k / 2) ≤ Nat.clog 2 ((k + 1) / 2) :=
        Nat.clog_mono_right 2 (by omega)
      have hclog : Nat.clog 2 k = Nat.clog 2 ((k + 1) / 2) + 1 := by
        rw [Nat.clog_of_two_le (by norm_num) hk2, show k + 2 - 1 = k + 1 by omega]
      simp only [depth, hdl, hdr, max_eq_right hmono, hclog]

end Lineage

/-- Number of divisions two units share, reading each unit's path of left and right from the root. -/
def sharedPrefix : List Bool → List Bool → ℕ
  | a :: as, b :: bs => if a = b then sharedPrefix as bs + 1 else 0
  | _, _ => 0

/-- **Nested groups**: sharing `m` divisions is transitive, so each level of the tree partitions
the units. No group straddles another. -/
theorem sharedPrefix_ultrametric (x y z : List Bool) :
    min (sharedPrefix x y) (sharedPrefix y z) ≤ sharedPrefix x z := by
  induction x generalizing y z with
  | nil => simp [sharedPrefix]
  | cons a as ih =>
    cases y with
    | nil => simp [sharedPrefix]
    | cons b bs =>
      cases z with
      | nil => simp [sharedPrefix]
      | cons c cs =>
        have h := ih bs cs
        by_cases hab : a = b
        · subst hab
          by_cases hbc : a = c
          · subst hbc
            simp [sharedPrefix]
            omega
          · simp [sharedPrefix, hbc]
        · simp [sharedPrefix, hab]

theorem sharedPrefix_comm (x y : List Bool) : sharedPrefix x y = sharedPrefix y x := by
  induction x generalizing y with
  | nil => cases y <;> simp [sharedPrefix]
  | cons a as ih =>
    cases y with
    | nil => simp [sharedPrefix]
    | cons b bs =>
      by_cases hab : a = b
      · simp [sharedPrefix, hab, ih bs]
      · simp [sharedPrefix, hab, Ne.symm hab]

end CGB.DivisionTree
