import CGBAxioms.GenerationContactPlasticity

/-!
# Dynamic boundary teaching has a necessary cross term

`GenerationContactPlasticity` proves the current spatial pullback while its
returned teaching coefficient is stopped.  That is the correct local object.
If the boundary decoder which supplies that coefficient also changes, the
complete derivative of the coupled score has one additional product-rule term.

This is an obstruction theorem, not a proposed new state, gate, or optimizer:
the extra term can be omitted only when the teaching signal is locally
stationary or the current contact response is zero.  Any O(1) replacement must
therefore obtain this information from an existing H/C/S carrier or explicitly
establish a separation of time scales; it cannot declare the frozen local VJP
to be a total future derivative.
-/

namespace CGB.Generation.DynamicTeaching

/-- A boundary teaching coefficient times a contact response.  When both vary
with the same parameter, its derivative contains both the local spatial term
and the teaching-motion term. -/
theorem teaching_response_deriv (teaching response : ℝ → ℝ)
    (x teaching' response' : ℝ)
    (hteaching : HasDerivAt teaching teaching' x)
    (hresponse : HasDerivAt response response' x) :
    HasDerivAt (fun θ => teaching θ * response θ)
      (teaching' * response x + teaching x * response') x := by
  exact hteaching.mul hresponse

/-- The stopped-teaching spatial term is complete exactly when its omitted
teaching-motion product vanishes.  This equality is the admission condition
for reusing a frozen-coefficient local derivative as a coupled derivative. -/
theorem stopped_teaching_complete_iff (teachingAt responseAt teaching' response' : ℝ) :
    teaching' * responseAt + teachingAt * response' = teachingAt * response' ↔
      teaching' * responseAt = 0 := by
  constructor <;> intro h <;> linarith

end CGB.Generation.DynamicTeaching
