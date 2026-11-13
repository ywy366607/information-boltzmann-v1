import CGBAxioms.GenerationLinearMemory
import CGBAxioms.GenerationRoutingCredit
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Asymptotics.Lemmas
import Mathlib.Analysis.Calculus.FDeriv.Mul

/-!
# Installed GDN operator directions

Real-arithmetic operator refinement, separately kernel checked. Functions of
epsilon are coordinate curves of the ACTUAL current input, weights and cache;
the premises are their coordinate derivatives, not a future loss derivative.
All incoming directions are allowed to vary together. No optimizer, CUDA
operator, floating-point error, stochastic token intervention or empirical
benefit is certified by this file. q/k normalization differs between the
installed short-inference and chunk implementations; both are explicit below.
-/

namespace CGB.GDNOperatorTags

open Finset

section FiniteRead

variable {I : Type*} [Fintype I]

/-- Same bilinear local law for each projection row, depthwise convolution
window and query read. Frozen weights mean dw=0, not dx=0. -/
theorem bilinear_read_deriv (w x : I → ℝ → ℝ) (dw dx : I → ℝ) (θ : ℝ)
    (hw : ∀ i, HasDerivAt (w i) (dw i) θ)
    (hx : ∀ i, HasDerivAt (x i) (dx i) θ) :
    HasDerivAt (fun t => ∑ i, w i t * x i t)
      (∑ i, (dw i * x i θ + w i θ * dx i)) θ := by
  simpa using HasDerivAt.fun_sum (u := univ) (fun i _ => (hw i).mul (hx i))

/-- Squared norm source; the factor two is essential for both normalizers. -/
theorem square_sum_deriv (x : I → ℝ → ℝ) (dx : I → ℝ) (θ : ℝ)
    (hx : ∀ i, HasDerivAt (x i) (dx i) θ) :
    HasDerivAt (fun t => ∑ i, (x i t) ^ 2)
      (2 * ∑ i, x i θ * dx i) θ := by
  have h := bilinear_read_deriv x x dx dx θ hx hx
  convert h using 1
  · funext t
    simp [pow_two]
  · rw [mul_sum]
    apply sum_congr rfl
    intro i _
    ring

end FiniteRead

noncomputable def silu (x : ℝ) : ℝ := x * Real.sigmoid x

/-- SiLU used by the three causal convolutions, MLP and output gate. -/
theorem silu_deriv (x : ℝ → ℝ) (dx θ : ℝ) (hx : HasDerivAt x dx θ) :
    HasDerivAt (fun t => silu (x t))
      (dx * (Real.sigmoid (x θ) +
        x θ * Real.sigmoid (x θ) * (1 - Real.sigmoid (x θ)))) θ := by
  have h := hx.mul ((Real.hasDerivAt_sigmoid (x θ)).comp θ hx)
  convert h using 1
  · rfl
  · dsimp
    ring

noncomputable def softplus (x : ℝ) : ℝ := Real.log (1 + Real.exp x)

/-- Smooth branch only. Torch softplus uses a linear branch above its
threshold; its boundary and finite-precision behavior are not erased here. -/
theorem softplus_deriv (x : ℝ → ℝ) (dx θ : ℝ) (hx : HasDerivAt x dx θ) :
    HasDerivAt (fun t => softplus (x t))
      (Real.exp (x θ) * dx / (1 + Real.exp (x θ))) θ := by
  exact (hx.exp.const_add 1).log (by positivity)

/-- sigmoid write gate: raw projection directions still have a local path. -/
theorem beta_deriv (b : ℝ → ℝ) (db θ : ℝ) (hb : HasDerivAt b db θ) :
    HasDerivAt (fun t => Real.sigmoid (b t))
      (Real.sigmoid (b θ) * (1 - Real.sigmoid (b θ)) * db) θ :=
  (Real.hasDerivAt_sigmoid (b θ)).comp θ hb

/-- log retention in the installed layer: -exp(A_log)*softplus(a+dt_bias).
The caller supplies a+dt_bias as z, including both parameter/input sources. -/
theorem log_retention_deriv (a z : ℝ → ℝ) (da dz θ : ℝ)
    (ha : HasDerivAt a da θ) (hz : HasDerivAt z dz θ) :
    HasDerivAt (fun t => -(Real.exp (a t) * softplus (z t)))
      (-(Real.exp (a θ) * da * softplus (z θ) +
        Real.exp (a θ) * (Real.exp (z θ) * dz / (1 + Real.exp (z θ))))) θ := by
  exact (ha.exp.mul (softplus_deriv z dz θ hz)).neg

/-- Delta receives alpha=exp(g), not g itself. -/
theorem alpha_deriv (g : ℝ → ℝ) (dg θ : ℝ) (hg : HasDerivAt g dg θ) :
    HasDerivAt (fun t => Real.exp (g t)) (Real.exp (g θ) * dg) θ := hg.exp

section Norm

/-- Chunk q/k norm (d=1), RMS norm (d=feature count); n is the squared
sum. This factor is sqrt(n/d+eps), NOT sqrt(n)+eps. -/
theorem smooth_normalizer_deriv (n : ℝ → ℝ) (dn d eps θ : ℝ)
    (hn : HasDerivAt n dn θ) (hpositive : 0 < n θ / d + eps) :
    HasDerivAt (fun t => Real.sqrt (n t / d + eps))
      ((dn / d) / (2 * Real.sqrt (n θ / d + eps))) θ := by
  exact ((hn.div_const d).add_const eps).sqrt hpositive.ne'

/-- Fused recurrent short-input q/k norm uses sqrt(n)+eps. Nonzero norm
is an explicit premise for this scalar sqrt proof. -/
theorem recurrent_normalizer_deriv (n : ℝ → ℝ) (dn eps θ : ℝ)
    (hn : HasDerivAt n dn θ) (hpositive : 0 < n θ) :
    HasDerivAt (fun t => Real.sqrt (n t) + eps)
      (dn / (2 * Real.sqrt (n θ))) θ := by
  exact (hn.sqrt hpositive.ne').add_const eps

/-- One normalized coordinate, including the norm's response to all other
coordinates. Suppressing dr is not an exact normalization direction. -/
theorem normalized_coordinate_deriv (x r : ℝ → ℝ) (dx dr θ : ℝ)
    (hx : HasDerivAt x dx θ) (hr : HasDerivAt r dr θ) (hne : r θ ≠ 0) :
    HasDerivAt (fun t => x t / r t)
      ((dx * r θ - x θ * dr) / (r θ) ^ 2) θ := hx.div hr hne

/-- RMS affine weight is trainable in the algebra; a frozen block still
passes the incoming direction. -/
theorem weighted_norm_deriv (w x r : ℝ → ℝ) (dw dx dr θ : ℝ)
    (hw : HasDerivAt w dw θ) (hx : HasDerivAt x dx θ)
    (hr : HasDerivAt r dr θ) (hne : r θ ≠ 0) :
    HasDerivAt (fun t => w t * (x t / r t))
      (dw * (x θ / r θ) + w θ *
        ((dx * r θ - x θ * dr) / (r θ) ^ 2)) θ := by
  exact hw.mul (normalized_coordinate_deriv x r dx dr θ hx hr hne)

end Norm

section Conv

/-- Last four raw projections after ingesting the current token, in the
oldest-to-newest order used by streaming_short_convolution. -/
def shift4 (cache : Fin 4 → ℝ) (x : ℝ) (i : Fin 4) : ℝ :=
  if h : i.val < 3 then cache ⟨i.val + 1, by omega⟩ else x

/-- The convolution CACHE direction must shift as well as its output.
Keeping S tags but dropping these buffers omits a real temporal path. -/
theorem shift4_deriv (cache : Fin 4 → ℝ → ℝ) (x : ℝ → ℝ)
    (dcache : Fin 4 → ℝ) (dx θ : ℝ)
    (hc : ∀ i, HasDerivAt (cache i) (dcache i) θ)
    (hx : HasDerivAt x dx θ) (i : Fin 4) :
    HasDerivAt (fun t => shift4 (fun j => cache j t) (x t) i)
      (shift4 dcache dx i) θ := by
  unfold shift4
  split
  · exact hc _
  · exact hx

/-- Four-tap convolution plus SiLU, with weight and incoming-cache sources
on the same graph. The current projection enters the last tap. -/
theorem causal_conv4_deriv (w cache : Fin 4 → ℝ → ℝ) (x : ℝ → ℝ)
    (dw dcache : Fin 4 → ℝ) (dx θ : ℝ)
    (hw : ∀ i, HasDerivAt (w i) (dw i) θ)
    (hc : ∀ i, HasDerivAt (cache i) (dcache i) θ)
    (hx : HasDerivAt x dx θ) :
    HasDerivAt
      (fun t => silu (∑ i, w i t * shift4 (fun j => cache j t) (x t) i))
      ((∑ i, (dw i * shift4 (fun j => cache j θ) (x θ) i +
        w i θ * shift4 dcache dx i)) *
        (Real.sigmoid (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i) +
          (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i) *
          Real.sigmoid (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i) *
          (1 - Real.sigmoid (∑ i, w i θ * shift4 (fun j => cache j θ) (x θ) i)))) θ := by
  exact silu_deriv _ _ θ (bilinear_read_deriv w
    (fun i t => shift4 (fun j => cache j t) (x t) i) dw
    (shift4 dcache dx) θ hw (fun i => shift4_deriv cache x dcache dx θ hc hx i))

end Conv

section Delta

variable {I J : Type*} [Fintype I]

/-- Explicit decay-before-erase step, S indexed by key and value. -/
def deltaNext (S : I → J → ℝ) (k : I → ℝ) (v : J → ℝ) (α β : ℝ)
    (i : I) (j : J) : ℝ :=
  α * S i j + β * k i * (v j - ∑ a, k a * (α * S a j))

/-- Joint source and transport: alpha,beta,key,value AND old S all vary.
No assumption that the old effect stays in its original layer is introduced. -/
theorem delta_next_deriv (S : I → J → ℝ → ℝ) (k : I → ℝ → ℝ)
    (v : J → ℝ → ℝ) (α β : ℝ → ℝ) (dS : I → J → ℝ)
    (dk : I → ℝ) (dv : J → ℝ) (dα dβ θ : ℝ)
    (hS : ∀ i j, HasDerivAt (S i j) (dS i j) θ)
    (hk : ∀ i, HasDerivAt (k i) (dk i) θ)
    (hv : ∀ j, HasDerivAt (v j) (dv j) θ)
    (hα : HasDerivAt α dα θ) (hβ : HasDerivAt β dβ θ) (i : I) (j : J) :
    HasDerivAt
      (fun t => deltaNext (fun a b => S a b t) (fun a => k a t)
        (fun b => v b t) (α t) (β t) i j)
      (dα * S i j θ + α θ * dS i j +
        (dβ * k i θ + β θ * dk i) *
          (v j θ - ∑ a, k a θ * (α θ * S a j θ)) +
        β θ * k i θ * (dv j - ∑ a, (dk a * (α θ * S a j θ) +
          k a θ * (dα * S a j θ + α θ * dS a j)))) θ := by
  have hp := bilinear_read_deriv k (fun a t => α t * S a j t) dk
    (fun a => dα * S a j θ + α θ * dS a j) θ hk (fun a => hα.mul (hS a j))
  have h := (hα.mul (hS i j)).add ((hβ.mul (hk i)).mul ((hv j).sub hp))
  convert h using 1
  · rfl
  · dsimp
    ring

/-- Actual query read has BOTH query credit and transported S credit; scale
is the fixed head-dimension factor. Frozen query projections do not kill dq
if their inputs depend on the old participation. -/
theorem query_read_deriv (q : I → ℝ → ℝ) (S : I → J → ℝ → ℝ)
    (dq : I → ℝ) (dS : I → J → ℝ) (scale θ : ℝ)
    (hq : ∀ i, HasDerivAt (q i) (dq i) θ)
    (hS : ∀ i j, HasDerivAt (S i j) (dS i j) θ) (j : J) :
    HasDerivAt (fun t => scale * ∑ i, q i t * S i j t)
      (scale * ∑ i, (dq i * S i j θ + q i θ * dS i j)) θ := by
  exact (bilinear_read_deriv q (fun i => S i j) dq
    (fun i => dS i j) θ hq (fun i => hS i j)).const_mul scale

end Delta

/-- SwiGLU intermediate coordinate AND the swish output norm gate share the
same product rule. u's source survives even if the gate weights are frozen. -/
theorem swish_product_deriv (g u : ℝ → ℝ) (dg du θ : ℝ)
    (hg : HasDerivAt g dg θ) (hu : HasDerivAt u du θ) :
    HasDerivAt (fun t => silu (g t) * u t)
      (dg * (Real.sigmoid (g θ) +
        g θ * Real.sigmoid (g θ) * (1 - Real.sigmoid (g θ))) * u θ +
        silu (g θ) * du) θ := (silu_deriv g dg θ hg).mul hu

/-- Each residual adds the incoming path instead of replacing it with an
isolated local loss. Apply twice per GDN block. -/
theorem residual_deriv (x f : ℝ → ℝ) (dx df θ : ℝ)
    (hx : HasDerivAt x dx θ) (hf : HasDerivAt f df θ) :
    HasDerivAt (fun t => x t + f t) (dx + df) θ := hx.add hf

section LayerComposition

variable {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]

/-- The next frozen block is a map on activation AND its own recurrent cache.
Its weights may be fixed while its state/activation directions pass through.
This is spatial composition, not a second historical BPTT window. -/
theorem frozen_successor_carries_direction (input : ℝ → E) (e : E) (θ : ℝ)
    (lower : E → F) (upper : F → G)
    (dlower : E →L[ℝ] F) (dupper : F →L[ℝ] G)
    (hi : HasDerivAt input e θ)
    (hl : HasFDerivAt lower dlower (input θ))
    (hu : HasFDerivAt upper dupper (lower (input θ))) :
    HasDerivAt (fun t => upper (lower (input t))) (dupper (dlower e)) θ := by
  exact hu.comp_hasDerivAt θ (hl.comp_hasDerivAt θ hi)

end LayerComposition

section Stack

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Each spatial stage acts on the joint package of residual activation and
the layer caches; untouched caches pass through. This definition does not
claim that an existing CUDA/hook implementation already equals that package. -/
noncomputable def spatial (stage : ℕ → ℝ × E → E) (input : ℝ → E)
    (ε : ℝ) : ℕ → E
  | 0 => input ε
  | n + 1 => stage n (ε, spatial stage input ε n)

noncomputable def spatialTag (D : ℕ → (ℝ × E →L[ℝ] E)) (initial : E) : ℕ → E
  | 0 => initial
  | n + 1 => D n (1, spatialTag D initial n)

/-- Assemble any finite number of differentiable stages, hence also 24,
from LOCAL stage derivatives. It does not assume the whole-block/whole-stack
derivative as a premise. Stage instantiation must cover every cache field. -/
theorem spatial_stages_deriv (stage : ℕ → ℝ × E → E) (input : ℝ → E)
    (D : ℕ → (ℝ × E →L[ℝ] E)) (initial : E)
    (hi : HasDerivAt input initial 0)
    (hs : ∀ n, HasFDerivAt (stage n) (D n) (0, spatial stage input 0 n))
    (layers : ℕ) :
    HasDerivAt (fun ε => spatial stage input ε layers)
      (spatialTag D initial layers) 0 := by
  induction layers with
  | zero => exact hi
  | succ n ih =>
    simpa only [spatial, spatialTag, Function.comp_def, id] using
      (hs n).comp_hasDerivAt 0 ((hasDerivAt_id (0 : ℝ)).prodMk ih)

/-- A frozen spatial stage contributes no new parameter source, but passes
the old participation via its actual state/input differential. -/
theorem frozen_stage_transport (D : ℝ × E →L[ℝ] E) (e : E)
    (frozen : D (1, 0) = 0) : D (1, e) = D (0, e) := by
  have hsplit : ((1 : ℝ), e) = (1, (0 : E)) + (0, e) := by simp
  rw [hsplit, map_add, frozen, zero_add]

end Stack

section Boundary

variable {V : Type*} [Fintype V] [Nonempty V]

/-- The original language boundary supplies a vector innovation. Contracting
it with the old participation's logit direction does not give hidden layers
private vocabulary heads or substitute a scalar reward for language error. -/
theorem boundary_nll_deriv (z : V → ℝ → ℝ) (dz : V → ℝ) (θ : ℝ)
    (target : V) (hz : ∀ i, HasDerivAt (z i) (dz i) θ) :
    HasDerivAt (fun t => Real.log (∑ i, Real.exp (z i t)) - z target t)
      ((∑ i, CGB.Medium.softmax (fun i => z i θ) i * dz i) - dz target) θ := by
  have hp : 0 < ∑ i, Real.exp (z i θ) :=
    sum_pos (fun _ _ => Real.exp_pos _) univ_nonempty
  have hs := HasDerivAt.fun_sum (u := univ) (fun i _ => (hz i).exp)
  have h := (hs.log hp.ne').sub (hz target)
  have heq : (∑ i, CGB.Medium.softmax (fun i => z i θ) i * dz i) =
      (∑ i, Real.exp (z i θ) * dz i) / (∑ i, Real.exp (z i θ)) := by
    simp only [CGB.Medium.softmax]
    simp_rw [div_mul_eq_mul_div]
    rw [sum_div]
  rw [heq]
  exact h

/-- The declared real-valued boundary loss is the original negative log
softmax, not a new auxiliary target for the interior layers. -/
theorem boundary_nll_eq_original (z : V → ℝ) (target : V) :
    Real.log (∑ i, Real.exp (z i)) - z target =
      -Real.log (CGB.Medium.softmax z target) := by
  have hp : 0 < ∑ i, Real.exp (z i) :=
    sum_pos (fun _ _ => Real.exp_pos _) univ_nonempty
  rw [CGB.Medium.softmax, Real.log_div (Real.exp_pos _).ne' hp.ne', Real.log_exp]
  ring

end Boundary

section BranchAdmission

/-- Installed Torch softplus default: beta=1, threshold=20. This piecewise
real reference is not identified with the smooth formula at the threshold. -/
noncomputable def torchSoftplus (x : ℝ) : ℝ := if 20 < x then x else softplus x

theorem torch_softplus_below (x : ℝ) (h : x < 20) :
    HasDerivAt torchSoftplus (Real.exp x / (1 + Real.exp x)) x := by
  have hd := softplus_deriv (fun t : ℝ => t) 1 x (hasDerivAt_id x)
  simp only [mul_one] at hd
  apply hd.congr_of_eventuallyEq
  filter_upwards [gt_mem_nhds h] with y hy
  simp [torchSoftplus, not_lt.mpr hy.le]

theorem torch_softplus_above (x : ℝ) (h : 20 < x) :
    HasDerivAt torchSoftplus 1 x := by
  apply (hasDerivAt_id x).congr_of_eventuallyEq
  filter_upwards [lt_mem_nhds h] with y hy
  simp [torchSoftplus, hy]

/-- Numerical error must fit inside the actual branch margin before the
same differential is used. Equality/ties have no smooth admission here. -/
theorem scalar_branch_preserved (x approximate threshold error : ℝ)
    (he : |x - approximate| ≤ error) (hm : error < |x - threshold|) :
    (threshold < approximate ↔ threshold < x) := by
  rcases lt_trichotomy x threshold with h | h | h
  · rw [abs_of_neg (sub_neg.mpr h)] at hm
    have hl := (abs_le.mp he).1
    constructor <;> intro ha <;> linarith
  · subst threshold
    simp only [sub_self, abs_zero] at hm
    exact False.elim ((lt_of_le_of_lt (abs_nonneg _) (he.trans_lt hm)).false)
  · rw [abs_of_pos (sub_pos.mpr h)] at hm
    have hu := (abs_le.mp he).2
    constructor <;> intro ha <;> linarith

/-- Shared scalar coordinate row for the actual normalized derivative.
r is sqrt(sum+eps), sqrt(sum)+eps, or the active recall clamp denominator;
the caller must use the corresponding branch's dr. -/
theorem normalized_coordinate_gain (x dx r dr : ℝ) (hr : 0 < r) :
    |(dx * r - x * dr) / r ^ 2| ≤ |dx| / r + |x| * |dr| / r ^ 2 := by
  rw [abs_div, abs_of_nonneg (sq_nonneg r)]
  have ht : |dx * r - x * dr| ≤ |dx| * r + |x| * |dr| := by
    simpa [abs_mul, abs_of_pos hr] using abs_sub (dx * r) (x * dr)
  calc
    _ ≤ (|dx| * r + |x| * |dr|) / r ^ 2 :=
      div_le_div_of_nonneg_right ht (sq_nonneg r)
    _ = _ := by field_simp

/-- Recall F.normalize has max(norm,eps), not either native normalizer.
Strictly below eps its denominator is constant, including at a zero vector. -/
theorem clamped_coordinate_below (x n : ℝ → ℝ) (dx eps θ : ℝ)
    (hx : HasDerivAt x dx θ) (hn : ContinuousAt n θ) (h : n θ < eps) :
    HasDerivAt (fun t => x t / max (n t) eps) (dx / eps) θ := by
  apply (hx.div_const eps).congr_of_eventuallyEq
  filter_upwards [hn.eventually (gt_mem_nhds h)] with t ht
  rw [max_eq_right ht.le]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

noncomputable def recurrentNormalize (eps : ℝ) (x : E) : E := (‖x‖ + eps)⁻¹ • x

/-- The recurrent VECTOR map is differentiable at zero, although norm itself
is not. This quadratic remainder also records its large 1/eps^2 sensitivity. -/
theorem recurrent_zero_remainder (eps : ℝ) (heps : 0 < eps) (x : E) :
    ‖recurrentNormalize eps x - eps⁻¹ • x‖ ≤ (eps ^ 2)⁻¹ * ‖x‖ ^ 2 := by
  have hr : 0 < ‖x‖ + eps := add_pos_of_nonneg_of_pos (norm_nonneg _) heps
  have hid : (‖x‖ + eps)⁻¹ - eps⁻¹ = -‖x‖ / (eps * (‖x‖ + eps)) := by
    field_simp; ring
  rw [recurrentNormalize, ← sub_smul, norm_smul, hid, Real.norm_eq_abs,
    abs_div, abs_neg, abs_of_nonneg (norm_nonneg x), abs_of_pos (mul_pos heps hr)]
  have hh : eps * eps ≤ eps * (‖x‖ + eps) := by nlinarith [norm_nonneg x]
  have hd := div_le_div_of_nonneg_left (norm_nonneg x) (mul_pos heps heps) hh
  have hh' := mul_le_mul_of_nonneg_right hd (norm_nonneg x)
  convert hh' using 1; field_simp

theorem recurrent_derivative_at_zero (eps : ℝ) (heps : 0 < eps) :
    HasFDerivAt (recurrentNormalize (E := E) eps)
      (eps⁻¹ • ContinuousLinearMap.id ℝ E) 0 := by
  have hb : (fun x : E => recurrentNormalize eps x - eps⁻¹ • x) =O[nhds 0]
      (fun x => ‖x‖ ^ 2) := Asymptotics.IsBigO.of_bound ((eps ^ 2)⁻¹) (by
    filter_upwards [] with x
    simpa only [Real.norm_eq_abs, abs_pow, abs_of_nonneg (norm_nonneg x)] using
      recurrent_zero_remainder eps heps x)
  have hl := hb.trans_isLittleO (Asymptotics.isLittleO_norm_pow_id (E' := E) one_lt_two)
  apply HasFDerivAt.of_isLittleO
  simpa [recurrentNormalize] using hl

/-- Reciprocal error becomes unbounded near zero. A denominator floor is an
explicit numerical obligation, not something fixed by U/V compression. -/
theorem reciprocal_error (r approximate floor : ℝ) (hf : 0 < floor)
    (hr : floor ≤ r) (ha : floor ≤ approximate) :
    |r⁻¹ - approximate⁻¹| ≤ |r - approximate| / floor ^ 2 := by
  have rp : 0 < r := hf.trans_le hr
  have ap : 0 < approximate := hf.trans_le ha
  have hid : r⁻¹ - approximate⁻¹ = -(r - approximate) / (r * approximate) := by
    field_simp; ring
  rw [hid, abs_div, abs_neg, abs_of_pos (mul_pos rp ap)]
  exact div_le_div_of_nonneg_left (abs_nonneg _) (sq_pos_of_pos hf)
    (by nlinarith [mul_le_mul hr ha hf.le rp.le])

end BranchAdmission

end CGB.GDNOperatorTags
