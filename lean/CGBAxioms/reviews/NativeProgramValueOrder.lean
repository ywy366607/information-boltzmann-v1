import Mathlib.Tactic

/-! Mapping to generative_relations.execute_programs and causal_use_value.utility.
vm/vd/vc are actual continuation returns on mediated/direct/coordinated inputs.
u is the unchanged calibrated finite-U term, including its factor of ten.
This decomposition separates continuation compression from immediate utility.
-/
namespace CGB.NativeProgramValueOrder
noncomputable section

structure Scores where
  u : ℝ
  vm : ℝ
  vd : ℝ
  vc : ℝ
  w : ℝ
  gamma : ℝ

def mixed (s : Scores) := s.u + s.gamma * (s.w*s.vm+(1-s.w)*s.vd)
def coordinated (s : Scores) := s.u + s.gamma*s.vc
def shift (s : Scores) := s.gamma*(s.w*s.vm+(1-s.w)*s.vd-s.vc)

theorem value_shift (s : Scores) : mixed s - coordinated s = shift s := by
  simp only [mixed, coordinated, shift]; ring

theorem ranking_shift (a b : Scores) :
    mixed a - mixed b = coordinated a - coordinated b + shift a - shift b := by
  simp only [mixed, coordinated, shift]; ring

theorem constant_readout (s : Scores) (hm : s.vm = s.vc) (hd : s.vd = s.vc) :
    mixed s = coordinated s := by
  simp only [mixed, coordinated, hm, hd]; ring

/-- The scalar affine specialization of the actual continuation map. -/
theorem affine_mix (w x y slope bias : ℝ) :
    slope*(w*x+(1-w)*y)+bias =
      w*(slope*x+bias)+(1-w)*(slope*y+bias) := by ring

theorem program_swap (w vm vd : ℝ) :
    w*vm+(1-w)*vd = (1-w)*vd+(1-(1-w))*vm := by ring
end
end CGB.NativeProgramValueOrder
