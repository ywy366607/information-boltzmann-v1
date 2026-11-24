import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Tactic

/-! The existing causal_tag_compression_20261108 factor/ownership and silent
quotient laws apply to the lifted (M,T,E,H) state. This file adds the dated
source recurrence and a minimum dimension certificate for joint delayed
targets. Body ports: autonomous_cycle.poll inserts selector eligibility;
take_boundary transfers its owner to the real action; prev.actor captures
the third factor at the following real update. ExternalTask supplies clocks.
-/
namespace CGB.NativeDelayedMemoryClock

/-- Common transport preserves source ages by carrying both moments. -/
theorem dated_source_step (a e₁ e₂ t₁ t₂ dt : ℝ) :
    a * (t₁*e₁+t₂*e₂) + dt*(a*(e₁+e₂)) =
      (t₁+dt)*(a*e₁)+(t₂+dt)*(a*e₂) := by ring

/-- Every future common wait can be answered from these two moments. -/
theorem future_wait_read (e₁ e₂ t₁ t₂ wait m : ℝ) :
    m * ((t₁*e₁+t₂*e₂)+wait*(e₁+e₂)) =
      m*((t₁+wait)*e₁+(t₂+wait)*e₂) := by ring

/-- Distinct ages make endpoint eligibility AND its time moment determine
the two separate source coordinates. This supplies a minimality witness. -/
theorem distinct_ages_determine_sources (a b c d t u : ℝ) (htu : t ≠ u)
    (he : a+b=c+d) (ht : t*a+u*b=t*c+u*d) : a=c ∧ b=d := by
  have hd : b-d = -(a-c) := by linarith
  have hz : (t-u)*(a-c)=0 := calc
    (t-u)*(a-c) = t*(a-c) + u*(-(a-c)) := by ring
    _ = t*(a-c) + u*(b-d) := by rw [hd]
    _ = 0 := by linarith only [ht]
  have hac : a=c := sub_eq_zero.mp ((mul_eq_zero.mp hz).resolve_left (sub_ne_zero.mpr htu))
  constructor
  · exact hac
  · linarith

/-- A finite delayed input/output matrix factored through k memory
coordinates has rank at most k. The lifted joint response uses the SAME
factorization including time and credit rows, rather than mass alone. -/
theorem delayed_target_dimension_lower_bound {o k s : Type*}
    [Fintype o] [Fintype k] [Fintype s] [DecidableEq k]
    (read : Matrix o k ℝ) (write : Matrix k s ℝ) :
    (read * write).rank ≤ Fintype.card k := by
  exact le_trans (Matrix.rank_mul_le_left read write) (Matrix.rank_le_card_width read)

/-- Pending history remains an independent term until its own target reads
it: elapsed time transports it even in stages with no fresh insertion. -/
theorem pending_age_transport (a e h dt : ℝ) :
    a*h+dt*(a*e) = a*(h+dt*e) := by ring

/-- Dating an accumulated score by its last insertion drops the older
source times the time between insertions, independent of the future wait. -/
theorem last_stamp_defect (first last t₁ t₂ read : ℝ) :
    read * (t₁*first+t₂*last) - read * (t₂*(first+last)) =
      read * ((t₁-t₂)*first) := by ring

end CGB.NativeDelayedMemoryClock
