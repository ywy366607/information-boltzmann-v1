import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Complex.Circle

/-! Measurement formulas for the actual impedance-normalized S/P rotation,
integer pair-graph incidence, and the certified relation edge-response channel.
No extra oscillator, face, prediction branch or runtime update is introduced. -/
namespace CGB.NativePaperPhaseDirac

open scoped BigOperators

/-- One coordinate of the actual normalized free rotation in quiet():
a=sqrt(Z)S, b=P/sqrt(Z), c=cos(omega dt), d=sin(omega dt). -/
theorem normalized_rotation (a b c d : ℝ) :
    (((c*a+d*b : ℝ) : ℂ) + Complex.I*((c*b-d*a : ℝ) : ℂ)) =
      ((c : ℂ)-Complex.I*(d : ℂ))*((a : ℂ)+Complex.I*(b : ℂ)) := by
  apply Complex.ext <;> simp <;> ring

variable {F : Type*} [Fintype F]

def overlap (left right : F → ℂ) : ℂ := ∑ k, left k * star (right k)

/-- Native free rotations act on the overlap through their relative phase. -/
theorem overlap_rotation (left right : F → ℂ) (a b : ℂ) :
    overlap (fun k => a*left k) (fun k => b*right k) =
      a*star b*overlap left right := by
  simp only [overlap, star_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- Circle-valued inputs make the integer-incidence paper readout
independent of the choice of a real phase lift. -/
theorem native_phase_lift (phase : ℝ) (turns : ℤ) :
    Circle.exp (phase + turns*(2*Real.pi)) = Circle.exp phase := by
  rw [Circle.exp_add, Circle.exp_int_mul_two_pi, mul_one]

variable {V E : Type*} [Fintype V] [Fintype E]

theorem circle_integer_sum (indices : Finset E) (weight : E → ℤ) (phase : E → ℝ) :
    Circle.exp (∑ e ∈ indices, (weight e : ℝ)*phase e) =
      ∏ e ∈ indices, Circle.exp (phase e) ^ weight e := by
  classical
  induction indices using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
      simp only [Finset.sum_insert ha, Finset.prod_insert ha, Circle.exp_add,
        Circle.exp_intCast_mul, ih]

theorem integer_boundary_lift (weight : E → ℤ) (phase : E → ℝ) (turns : E → ℤ) :
    Circle.exp (∑ e, (weight e : ℝ)*(phase e + turns e*(2*Real.pi))) =
      Circle.exp (∑ e, (weight e : ℝ)*phase e) := by
  rw [circle_integer_sum, circle_integer_sum]
  apply Finset.prod_congr rfl
  intro e _
  rw [native_phase_lift]

def multiply (boundary : V → E → ℝ) (signal : E → ℝ) (v : V) : ℝ :=
  ∑ e, boundary v e * signal e

def adjoint (boundary : V → E → ℝ) (signal : V → ℝ) (e : E) : ℝ :=
  ∑ v, boundary v e * signal v

theorem native_adjoint_identity (boundary : V → E → ℝ)
    (edge : E → ℝ) (node : V → ℝ) :
    (∑ v, multiply boundary edge v * node v) =
      ∑ e, edge e * adjoint boundary node e := by
  unfold multiply adjoint
  simp only [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro e _
  apply Finset.sum_congr rfl
  intro v _
  ring

variable {T : Type*} [Fintype T]

/-- The known grade-1 contribution determines two adjacent D readouts.
Other native grade signals are not assumed to vanish in the body. -/
theorem edge_dirac_energy (down : V → E → ℝ) (up : T → E → ℝ)
    (edge : E → ℝ) :
    (∑ v, multiply down edge v ^ 2) + (∑ t, multiply up edge t ^ 2) =
      ∑ e, edge e * (adjoint down (multiply down edge) e +
        adjoint up (multiply up edge) e) := by
  simp_rw [pow_two]
  rw [native_adjoint_identity, native_adjoint_identity]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro e _
  ring

-- The actual eight-head potential pairs. This carrier is distinct from
-- the relation/shared-unknown complex and does not replace it.
def actualPairs : Array (Nat × Nat) := #[(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
  (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,4),(2,5),(2,6),(2,7),
  (3,4),(3,5),(3,6),(3,7),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]

def pairBoundary (v e : Nat) : Int :=
  (if v == actualPairs[e]!.2 then 1 else 0) -
  (if v == actualPairs[e]!.1 then 1 else 0)

example : actualPairs.size = 28 := by native_decide
example : actualPairs.toList.Nodup := by native_decide
example : actualPairs.all (fun e => e.1 < e.2 && e.2 < 8) = true := by native_decide

def actualTriangles : List (Nat × Nat × Nat) :=
  (List.range 8).flatMap (fun i => (List.range 8).flatMap (fun j =>
    (List.range 8).filterMap (fun k => if i < j ∧ j < k then some (i,j,k) else none)))

def edgeIndex (i j : Nat) : Nat :=
  (actualPairs.toList.findIdx? (fun e => e == (i,j))).getD 28

def actualCycleBoundary (c : Nat × Nat × Nat) (v : Nat) : Int :=
  pairBoundary v (edgeIndex c.1 c.2.1) + pairBoundary v (edgeIndex c.2.1 c.2.2) -
    pairBoundary v (edgeIndex c.1 c.2.2)

example : actualTriangles.length = 56 := by native_decide
example : actualTriangles.all (fun c =>
    (List.range 8).all (fun v => actualCycleBoundary c v == 0)) = true := by native_decide

end CGB.NativePaperPhaseDirac
