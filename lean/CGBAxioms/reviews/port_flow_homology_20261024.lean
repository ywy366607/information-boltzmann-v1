import CGBAxioms.PortFlowHomology

open CGB.AlgebraicHomology CGB.PortFlowHomology

namespace PortFlowHomologyReview20261024

example : CGB.AlgebraicHomology.boundaryLaw incidence augmentation :=
  incidence_augmented_zero

example : CGB.AlgebraicHomology.boundaryLaw weightedIncidence augmentation :=
  weighted_incidence_augmented_zero

example (e : Edge) :
    CGB.AlgebraicHomology.matrixAction augmentation
        (CGB.AlgebraicHomology.matrixAction weightedIncidence (fun e' =>
          if e' = e then 1 else 0)) () = 0 :=
  weighted_flow_boundary_zero e

end PortFlowHomologyReview20261024
