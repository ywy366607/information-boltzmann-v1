import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Exp

/-! Actual binding: relation_retention_use.py. h is the existing geometric
leverage, b is tanh of bias-corrected signed finite-use EMA, e arrives after
the old support. The finite state is scope + revision + <=8 identity moments.
Both raw and encoded U select with old moments, then update retained moments.
The real commit CAS is separate from this pure transition.
-/
namespace CGB.NativeRelationRetentionUse
noncomputable section

def priority (h b : ℝ) := h * Real.exp b

theorem neutral (h : ℝ) : priority h 0 = h := by simp [priority]

theorem bounded (h b : ℝ) (hh : 0 ≤ h) (hb : -1 ≤ b ∧ b ≤ 1) :
    h * Real.exp (-1) ≤ priority h b ∧ priority h b ≤ h * Real.exp 1 := by
  exact ⟨mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hb.1) hh,
         mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hb.2) hh⟩

theorem protects (h b : ℝ) (hh : 0 ≤ h) (hb : 0 ≤ b) : h ≤ priority h b := by
  have e : 1 ≤ Real.exp b := by simpa using Real.exp_le_exp.mpr hb
  dsimp [priority]; nlinarith

theorem releases (h b : ℝ) (hh : 0 ≤ h) (hb : b ≤ 0) : priority h b ≤ h := by
  have e : Real.exp b ≤ 1 := by simpa using Real.exp_le_exp.mpr hb
  dsimp [priority]; nlinarith

/-- A legal priority conflict, instantiated numerically on the body as well. -/
theorem ranking_can_change :
    (1 : ℝ) < 2 ∧ priority 2 (-1) < priority 1 1 := by
  have h : 1 < Real.exp (1:ℝ) := by simpa using Real.exp_lt_exp.mpr (by norm_num : (0:ℝ)<1)
  have h2 := Real.add_one_lt_exp (by norm_num : (1:ℝ) ≠ 0)
  rw [priority, priority, Real.exp_neg]
  constructor
  · norm_num
  · apply (div_lt_iff₀ (Real.exp_pos (1:ℝ))).mpr
    nlinarith

def step {K E A : Type*} (choose : K → A) (update : K → E → K)
    (k : K) (e : E) := (choose k, update k e)

theorem no_current_label_leak {K E A : Type*} (choose : K → A)
    (update : K → E → K) (k : K) (e f : E) :
    (step choose update k e).1 = (step choose update k f).1 := rfl

theorem next_reads_updated {K E A : Type*} (choose : K → A)
    (update : K → E → K) (k : K) (e f : E) :
    (step choose update (step choose update k e).2 f).1 = choose (update k e) := rfl

/-- Production binding: revise_raw selects local support, organize compresses
the shared union, then the returned RelationState exposes successor moments.
Both selectors may depend on this event's GEOMETRY; only the use snapshot is
lagged. organize(selection_state=state) supplies that same old snapshot. -/
def compoundU {K E L A : Type*} (selectLocal : K → E → L)
    (share : K → E → L → A) (update : K → E → K) (k : K) (e : E) :=
  ((selectLocal k e, share k e (selectLocal k e)), update k e)

theorem organization_reads_old_use {K E L A : Type*} (selectLocal : K → E → L)
    (share : K → E → L → A) (update : K → E → K) (k : K) (e : E) :
    (compoundU selectLocal share update k e).1.2 = share k e (selectLocal k e) := rfl

theorem current_selection_independent_of_successor_moments {K E L A : Type*}
    (selectLocal : K → E → L) (share : K → E → L → A)
    (update₁ update₂ : K → E → K) (k : K) (e : E) :
    (compoundU selectLocal share update₁ k e).1 =
      (compoundU selectLocal share update₂ k e).1 := rfl

theorem compound_next_reads_updated {K E L A : Type*} (selectLocal : K → E → L)
    (share : K → E → L → A) (update : K → E → K) (k : K) (e f : E) :
    (compoundU selectLocal share update (compoundU selectLocal share update k e).2 f).1.2 =
      share (update k e) f (selectLocal (update k e) f) := rfl

def commit {K : Type*} (sameRevision : Bool) (old new : K) :=
  if sameRevision then new else old

theorem stale_unpaid {K : Type*} (old new : K) : commit false old new = old := rfl

theorem encode_transition {S K E : Type*} (encode : S → K)
    (raw : S → E → S) (latent : K → E → K)
    (bound : ∀ s e, encode (raw s e) = latent (encode s) e)
    (s : S) (es : List E) :
    encode (es.foldl raw s) = es.foldl latent (encode s) := by
  induction es generalizing s with
  | nil => rfl
  | cons e es ih => simp only [List.foldl_cons, ih, bound]

/-- audit_native_relation_use_admission.py, complete 2481, unknown:1726:
episode:2482 vs episode:2369. CPU/GPU binding separately validates the actual
priority computation; these concrete rational inequalities bind its readout. -/
theorem actual_body_release_conflict :
    (16155952215194702 : ℚ)/10^17 < 3426612615585327/10^16 ∧
    (1599959284067154 : ℚ)/10^16 < 16155952215194702/10^17 := by norm_num

theorem identity_reindex {I J : Type*} (h b : I → ℝ) (p : J → I) (j : J) :
    priority (h (p j)) (b (p j)) = (fun i => priority (h i) (b i)) (p j) := rfl

end
end CGB.NativeRelationRetentionUse
