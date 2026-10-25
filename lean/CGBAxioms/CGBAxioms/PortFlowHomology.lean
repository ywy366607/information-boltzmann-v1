import CGBAxioms.AlgebraicHomology
import CGBAxioms.PortSeparation

/-!
# Finite flow chain for the separated port witness

Each current state/input pair is a directed transition edge. Its integer
boundary is target minus source; multiplying by the registered outgoing flow
keeps the same zero augmented boundary. This is a finite chain instance, not a
claim that biological topology has already been reconstructed.
-/

namespace CGB.PortFlowHomology
open PortSeparation

abbrev Edge := State × Input

def source (e : Edge) : State := e.1

def target (e : Edge) : State := next e.2 e.1

def incidence (e : Edge) (s : State) : Int :=
  (if s = target e then 1 else 0) - (if s = source e then 1 else 0)

def weightedIncidence (e : Edge) (s : State) : Int :=
  (outgoing e.2 e.1 : Int) * incidence e s

def augmentation (s : State) (_ : Unit) : Int := 1

theorem incidence_augmented_zero :
    AlgebraicHomology.boundaryLaw incidence augmentation := by
  intro e u
  classical
  have ht : (∑ s : State, if s = target e then (1 : Int) else 0) = 1 := by
    simp
  have hs : (∑ s : State, if s = source e then (1 : Int) else 0) = 1 := by
    simp
  simp only [AlgebraicHomology.boundaryLaw, AlgebraicHomology.matrixCompose,
    incidence, augmentation, mul_one, Finset.sum_sub_distrib]
  rw [ht, hs]
  norm_num

theorem weighted_incidence_augmented_zero :
    AlgebraicHomology.boundaryLaw weightedIncidence augmentation := by
  intro e u
  classical
  have hz : (∑ s : State, incidence e s) = 0 := by
    have ht : (∑ s : State, if s = target e then (1 : Int) else 0) = 1 := by
      simp
    have hs : (∑ s : State, if s = source e then (1 : Int) else 0) = 1 := by
      simp
    simp only [incidence, Finset.sum_sub_distrib]
    rw [ht, hs]
    norm_num
  simp only [AlgebraicHomology.boundaryLaw, AlgebraicHomology.matrixCompose,
    weightedIncidence, augmentation, mul_one]
  rw [← Finset.mul_sum, hz, mul_zero]

theorem weighted_flow_boundary_zero (e : Edge) :
    AlgebraicHomology.matrixAction augmentation
        (AlgebraicHomology.matrixAction weightedIncidence (fun e' =>
          if e' = e then 1 else 0)) () = 0 := by
  exact AlgebraicHomology.boundary_boundary_zero weightedIncidence augmentation
    weighted_incidence_augmented_zero (fun e' => if e' = e then 1 else 0) ()

end CGB.PortFlowHomology
