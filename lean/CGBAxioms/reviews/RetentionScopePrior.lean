import Mathlib.Tactic

/-! Actual binding: relation_retention_use.UseSnapshot stores own moments
and a distinct, identity-filtered parent snapshot. effective_rows first reads
positive-count parent rows and then overwrites with positive-count own rows.
advance updates only own moments once on the actual arrived event. A parent
read never increments the own observation count or changes the parent.
-/
namespace CGB.RetentionScopePrior

structure Moment where
  first : ℝ
  second : ℝ
  count : ℕ

def read (own parent : Moment) : Moment :=
  if 0 < own.count then own else parent

def attach {Id : Type} [DecidableEq Id] (allowed : Finset Id)
    (parent : Id → Option Moment) (i : Id) : Option Moment :=
  if i ∈ allowed then parent i else none

theorem cold_reads_parent (own parent : Moment) (h : own.count = 0) :
    read own parent = parent := by simp [read, h]

theorem observed_reads_local (own parent : Moment) (h : 0 < own.count) :
    read own parent = own := by simp [read, h]

theorem identity_filter {Id : Type} [DecidableEq Id] (allowed : Finset Id)
    (parent : Id → Option Moment) (i : Id) (h : i ∉ allowed) :
    attach allowed parent i = none := by simp [attach, h]

theorem matching_identity_read {Id : Type} [DecidableEq Id] (allowed : Finset Id)
    (parent : Id → Option Moment) (i : Id) (h : i ∈ allowed) :
    attach allowed parent i = parent i := by simp [attach, h]

def arrive (own : Moment) (value b1 b2 : ℝ) : Moment :=
  ⟨b1*own.first+(1-b1)*value,
   b2*own.second+(1-b2)*value^2, own.count+1⟩

theorem exactly_one_local_observation (own : Moment) (value b1 b2 : ℝ) :
    (arrive own value b1 b2).count = own.count+1 := rfl

theorem first_arrival_is_local (own parent : Moment) (value b1 b2 : ℝ) :
    read (arrive own value b1 b2) parent = arrive own value b1 b2 := by
  apply observed_reads_local
  simp [arrive]

theorem prior_does_not_enter_update (own p1 p2 : Moment) (value b1 b2 : ℝ) :
    read (arrive own value b1 b2) p1 = read (arrive own value b1 b2) p2 := by
  rw [first_arrival_is_local, first_arrival_is_local]

end CGB.RetentionScopePrior
