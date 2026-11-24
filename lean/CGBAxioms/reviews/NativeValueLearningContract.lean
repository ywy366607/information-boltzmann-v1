import Mathlib.Tactic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add

/-! Actual C2 gain coordinate (index 3), in units 10 * finite-U nats.
`storedBase + contextual` maps to v4q._training_base + CausalWood._residual.
`currentBase + plain` maps to navigation_internal_rollout.continuation.
`inlet` is contextual - plain for the same stored source/action/option.
The repair uses the SAME current base and plain head in gain loss, forecast
receipt and continuation. Other three channels retain their original graph.
-/
namespace CGB.NativeValueLearningContract
noncomputable section

structure Sample where
  target : ℝ
  storedBase : ℝ
  currentBase : ℝ
  plain : ℝ
  inlet : ℝ

def trainOld (s : Sample) := s.storedBase + s.plain + s.inlet
def readRaw (s : Sample) := s.currentBase + s.plain
def gap (s : Sample) := s.currentBase - s.storedBase - s.inlet
def trainShared (s : Sample) := s.currentBase + s.plain

theorem actual_contract_gap (s : Sample) :
    readRaw s - trainOld s = gap s := by
  simp only [readRaw, trainOld, gap]; ring

theorem perfect_old_fit_leaves_read_error (s : Sample)
    (fit : trainOld s = s.target) : readRaw s - s.target = gap s := by
  rw [← fit]; exact actual_contract_gap s

theorem shared_fit_delivers (s : Sample)
    (fit : trainShared s = s.target) : readRaw s = s.target := fit

/-- Half squared-error derivative in the head output coordinate. A zero
legacy gradient can coexist with a nonzero gradient of the actual read. -/
theorem derivative_gap (s : Sample) :
    (readRaw s - s.target) - (trainOld s - s.target) = gap s := by
  simp only [readRaw, trainOld, gap]; ring

/-- When a new baseline b replaces an old baseline a, replay of absolute
targets must rebase its residual by a-b to preserve the delivered value. -/
theorem residual_rebase (a b r : ℝ) : b + (r + a - b) = a + r := by ring

/-- With the shared contract, fitting remains live even when confidence is
zero; the confidence gates use, not the raw training residual. -/
theorem raw_loss_derivative (b r y : ℝ) :
    HasDerivAt (fun t : ℝ => (b+t-y)^2/2) (b+r-y) r := by
  have h : HasDerivAt (fun t : ℝ => b+t-y) 1 r := by
    simpa only [id_eq] using ((hasDerivAt_id r).const_add b).sub_const y
  convert (h.pow 2).div_const 2 using 1; ring

/-- Conditional descent for a real last-row gradient step: k is the squared
feature norm (or its positive preconditioned counterpart), eta the step size.
This states the precise sufficient condition, rather than assuming every
AdamW step contracts. -/
theorem shared_output_step_contracts (e eta k : ℝ)
    (h : 0 < eta*k) (h2 : eta*k < 2) (he : e ≠ 0) :
    ((1-eta*k)*e)^2 < e^2 := by
  have ee : 0 < e^2 := sq_pos_of_ne_zero he
  have hk : 0 < (eta*k)*(2-eta*k) := mul_pos h (sub_pos.mpr h2)
  have hp := mul_pos hk ee
  nlinarith

end
end CGB.NativeValueLearningContract
