import Mathlib.Data.Rat.Defs
import Mathlib.Data.Fintype.Sum
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic

/-!
Measurement reference for `relation_variables`: relation id+generation,
unknown id and observed version metadata. The graph is an incidence view of
that table; it is not an asserted simplicial model of the complete body.
No theta/phi phase is manufactured from a version counter.
-/
namespace CGB.NativeMeasurement

inductive Port where
  | condition | action | mother
  deriving Inhabited, DecidableEq, Repr

structure Dependency where
  relation : Nat
  generation : Nat
  unknown : String
  port : Port
  seenVersion : Nat
  deriving Inhabited, DecidableEq, Repr

inductive Vertex where
  | relation (identity generation : Nat)
  | unknown (identity : String)
  deriving DecidableEq, Repr

def source (e : Dependency) : Vertex := .relation e.relation e.generation
def target (e : Dependency) : Vertex := .unknown e.unknown

theorem source_ne_target (e : Dependency) : source e ≠ target e := by
  simp [source, target]

/-- Version advances do not, by themselves, add or remove a dependency. -/
theorem seenVersion_does_not_change_edge (e : Dependency) (version : Nat) :
    (source { e with seenVersion := version }, target { e with seenVersion := version }) =
      (source e, target e) := rfl

/-- Scope identity includes generation: recycling a relation does change it. -/
theorem relation_scope_injective (i j g h : Nat) :
    Vertex.relation i g = Vertex.relation j h ↔ i = j ∧ g = h := by
  simp

/-- Dense indexes are bookkeeping only; the exporter retains the identities. -/
structure Graph (r u e : Nat) where
  relationEnd : Fin e → Fin r
  unknownEnd : Fin e → Fin u

variable {r u e : Nat}
abbrev Nodes (r u : Nat) := Sum (Fin r) (Fin u)

def divergence (g : Graph r u e) (flow : Fin e → ℚ) (v : Nodes r u) : ℚ :=
  match v with
  | .inl i => -∑ k, if g.relationEnd k = i then flow k else 0
  | .inr i => ∑ k, if g.unknownEnd k = i then flow k else 0

def gradient (g : Graph r u e) (x : Nodes r u → ℚ) (k : Fin e) : ℚ :=
  x (.inr (g.unknownEnd k)) - x (.inl (g.relationEnd k))

def dirac (g : Graph r u e) (x : (Nodes r u → ℚ) × (Fin e → ℚ)) :=
  (divergence g x.2, gradient g x.1)

def nodeLaplacian (g : Graph r u e) (x : Nodes r u → ℚ) :=
  divergence g (gradient g x)

def edgeLaplacian (g : Graph r u e) (x : Fin e → ℚ) :=
  gradient g (divergence g x)

/-- Box 3 on the actual graph abstraction: the square has no cross blocks. -/
theorem dirac_squared (g : Graph r u e)
    (x : (Nodes r u → ℚ) × (Fin e → ℚ)) :
    dirac g (dirac g x) = (nodeLaplacian g x.1, edgeLaplacian g x.2) := rfl

theorem constant_node_signal_has_zero_gradient (g : Graph r u e) (c : ℚ) :
    gradient g (fun _ => c) = 0 := by
  funext k
  simp [gradient]

/-- Summed signed divergence is zero on the exported real finite graph. -/
theorem total_divergence_zero (g : Graph r u e) (flow : Fin e → ℚ) :
    ∑ v : Nodes r u, divergence g flow v = 0 := by
  classical
  have left : (∑ i : Fin r, ∑ k : Fin e,
      if g.relationEnd k = i then flow k else 0) = ∑ k, flow k := by
    rw [Finset.sum_comm]
    simp
  have right : (∑ i : Fin u, ∑ k : Fin e,
      if g.unknownEnd k = i then flow k else 0) = ∑ k, flow k := by
    rw [Finset.sum_comm]
    simp
  simp only [Nodes, Fintype.sum_sum_type, divergence, Finset.sum_neg_distrib]
  rw [left, right]
  simp

/-- The exported oriented incidence really is adjoint to the gradient. -/
theorem divergence_gradient_adjoint (g : Graph r u e)
    (x : Nodes r u → ℚ) (flow : Fin e → ℚ) :
    (∑ v : Nodes r u, x v * divergence g flow v) =
      ∑ k : Fin e, gradient g x k * flow k := by
  classical
  have left : (∑ i : Fin r, x (.inl i) *
      (∑ k : Fin e, if g.relationEnd k = i then flow k else 0)) =
      ∑ k : Fin e, x (.inl (g.relationEnd k)) * flow k := by
    simp_rw [Finset.mul_sum, mul_ite, mul_zero]
    rw [Finset.sum_comm]
    simp
  have right : (∑ i : Fin u, x (.inr i) *
      (∑ k : Fin e, if g.unknownEnd k = i then flow k else 0)) =
      ∑ k : Fin e, x (.inr (g.unknownEnd k)) * flow k := by
    simp_rw [Finset.mul_sum, mul_ite, mul_zero]
    rw [Finset.sum_comm]
    simp
  simp only [Nodes, Fintype.sum_sum_type, divergence, mul_neg,
    Finset.sum_neg_distrib]
  rw [left, right]
  simp only [gradient, sub_mul, Finset.sum_sub_distrib]
  ring

theorem node_laplacian_energy (g : Graph r u e) (x : Nodes r u → ℚ) :
    (∑ v, x v * nodeLaplacian g x v) =
      ∑ k, (gradient g x k) ^ 2 := by
  change (∑ v, x v * divergence g (gradient g x) v) = _
  rw [divergence_gradient_adjoint]
  simp [pow_two]

/-- If edge values are manufactured as node differences, a graph harmonic
measurement of them is identically zero, regardless of body activity.
This rules out that particular encoder for measuring independent circulation.
-/
theorem gradient_orthogonal_to_divergence_free (g : Graph r u e)
    (x : Nodes r u → ℚ) (flow : Fin e → ℚ) (h : divergence g flow = 0) :
    (∑ k, gradient g x k * flow k) = 0 := by
  rw [← divergence_gradient_adjoint, h]
  simp

/-- A divergence-free edge signal is graph harmonic, with no fabricated faces. -/
theorem zero_divergence_implies_graph_harmonic (g : Graph r u e)
    (flow : Fin e → ℚ) (h : divergence g flow = 0) :
    edgeLaplacian g flow = 0 := by
  unfold edgeLaplacian
  rw [h]
  exact constant_node_signal_has_zero_gradient g 0

/-- Pure information test: a recorder cannot recover a quantity it aliases.
This is the obligation for a proposed phase/edge-state decoder from body logs. -/
theorem aliased_quantity_has_no_decoder {State Obs Quantity : Type*}
    (record : State → Obs) (quantity : State → Quantity)
    (a b : State) (same : record a = record b) (different : quantity a ≠ quantity b) :
    ¬ ∃ decode : Obs → Quantity, ∀ s, decode (record s) = quantity s := by
  rintro ⟨decode, correct⟩
  apply different
  rw [← correct a, ← correct b, same]

end CGB.NativeMeasurement
