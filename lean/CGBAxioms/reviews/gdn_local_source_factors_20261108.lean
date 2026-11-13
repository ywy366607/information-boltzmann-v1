import CGBAxioms.GenerationRoutingCredit
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Local source factors of the installed decay-before-delta GDN

One reached event, fixed incoming activation/cache, real arithmetic. These
identities specialize the ownership-preserving compression proposal: matrix
rank, rank across parameter sources, and rank accumulated over time differ.
No new runtime, historical archive, optimizer step or CUDA equality is assumed.
The actual joint-step derivative must supply the common downstream linear map,
including changed convolution caches and record fields wherever applicable.
-/

namespace CGB.GDNLocalSourceFactors

open Finset

section Native

variable {I J : Type*} [Fintype I]

def dot (k x : I → ℝ) : ℝ := ∑ i, k i * x i

/-- Pull a scalar out of the actual key-column contraction. -/
theorem column_scale (k x : I → ℝ) (c : ℝ) :
    (∑ a, k a * (c * x a)) = c * dot k x := by
  rw [dot, mul_sum]
  apply sum_congr rfl
  intro a _
  ring

def native (S : I → J → ℝ) (k : I → ℝ) (v : J → ℝ)
    (α β : ℝ) (i : I) (j : J) : ℝ :=
  α * S i j + β * k i * (v j - ∑ a, k a * (α * S a j))

def residual (S : I → J → ℝ) (k : I → ℝ) (v : J → ℝ)
    (α : ℝ) (j : J) : ℝ := v j - ∑ a, k a * (α * S a j)

def retentionSource (S : I → J → ℝ) (k : I → ℝ) (β : ℝ)
    (i : I) (j : J) : ℝ := S i j - β * k i * dot k (fun a => S a j)

def writeSource (S : I → J → ℝ) (k : I → ℝ) (v : J → ℝ)
    (α : ℝ) (i : I) (j : J) : ℝ := k i * residual S k v α j

def keySource (S : I → J → ℝ) (k dk : I → ℝ) (v : J → ℝ)
    (α β : ℝ) (i : I) (j : J) : ℝ :=
  β * dk i * residual S k v α j -
    β * k i * dot dk (fun a => α * S a j)

/-- Varying retention alone changes every preserved column through the same
matrix-valued source. This is exact for finite scalar edits with fixed k. -/
theorem retention_edit_exact (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α β d : ℝ) (i : I) (j : J) :
    native S k v (α + d) β i j - native S k v α β i j =
      d * retentionSource S k β i j := by
  simp only [native, retentionSource, column_scale]
  ring

/-- The write-gate source is an outer product k times the current innovation;
all weights of its scalar projection row share this source, not its owners. -/
theorem write_edit_exact (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α β d : ℝ) (i : I) (j : J) :
    native S k v α (β + d) i j - native S k v α β i j =
      d * writeSource S k v α i j := by
  unfold native writeSource residual
  ring

/-- A value direction gives one outer product. Its parameter source rank and
the rank of a sum over different events are separate quantities. -/
theorem value_edit_exact (S : I → J → ℝ) (k : I → ℝ)
    (v dv : J → ℝ) (α β : ℝ) (i : I) (j : J) :
    native S k (fun j => v j + dv j) α β i j - native S k v α β i j =
      β * k i * dv j := by
  unfold native
  ring

/-- A key intervention has two first-order outer products AND a quadratic
remainder. The first-order tag must not be advertised as a finite-edit law. -/
theorem key_edit_with_remainder (S : I → J → ℝ) (k dk : I → ℝ)
    (v : J → ℝ) (α β t : ℝ) (i : I) (j : J) :
    native S (fun a => k a + t * dk a) v α β i j =
      native S k v α β i j + t * keySource S k dk v α β i j -
        t ^ 2 * β * dk i * dot dk (fun a => α * S a j) := by
  simp only [native, keySource, residual, dot, add_mul, sum_add_distrib,
    mul_assoc, ← mul_sum]
  ring

/-- The native derivative already proved in gdn_operator_tags is here split
into transported old S and four actual local sources, without dropping key,
retention or old-state terms. Incoming activation effects enter dk,dv,da,db. -/
theorem native_direction_source_split (S dS : I → J → ℝ)
    (k dk : I → ℝ) (v dv : J → ℝ) (α β da db : ℝ) (i : I) (j : J) :
    da * S i j + α * dS i j +
        (db * k i + β * dk i) * residual S k v α j +
        β * k i * (dv j - ∑ a, (dk a * (α * S a j) +
          k a * (da * S a j + α * dS a j))) =
      α * retentionSource dS k β i j +
        da * retentionSource S k β i j + db * writeSource S k v α i j +
        keySource S k dk v α β i j + β * k i * dv j := by
  simp only [retentionSource, writeSource, keySource, mul_add, sum_add_distrib,
    column_scale]
  have hdk : dot dk (fun a => α * S a j) = α * dot dk (fun a => S a j) :=
    column_scale dk (fun a => S a j) α
  rw [hdk]
  ring

/-- Query is absent from the S update; this does not erase its activation,
raw-convolution-cache, downstream-layer or stored-record causal paths. -/
theorem query_read_source (S : I → J → ℝ) (q dq : I → ℝ) (j : J) :
    (∑ i, (q i + dq i) * S i j) - (∑ i, q i * S i j) =
      ∑ i, dq i * S i j := by
  simp [add_mul, sum_add_distrib]

/-- Retention can preserve all directions: away from the exact erase
singularity, x -> x-beta*k*(k dot x) is injective. Thus one gate-source
column in effect space does NOT imply a rank-one native S matrix. -/
theorem erase_injective (k : I → ℝ) (β : ℝ)
    (h : 1 - β * dot k k ≠ 0) :
    Function.Injective (fun x : I → ℝ => fun i => x i - β * k i * dot k x) := by
  intro x y heq
  have hd := congrArg (dot k) heq
  have hx : dot k (fun i => x i - β * k i * dot k x) =
      (1 - β * dot k k) * dot k x := by
    have hs : (∑ i, k i * (β * k i * dot k x)) = β * dot k k * dot k x := by
      change (∑ i, k i * (β * k i * dot k x)) = β * (∑ i, k i * k i) * dot k x
      simp only [sum_mul, mul_sum]
      apply sum_congr rfl
      intro i _
      ring
    calc
      _ = dot k x - (∑ i, k i * (β * k i * dot k x)) := by
        simp only [dot, mul_sub, sum_sub_distrib]
      _ = dot k x - β * dot k k * dot k x := by rw [hs]
      _ = _ := by ring
  have hy : dot k (fun i => y i - β * k i * dot k y) =
      (1 - β * dot k k) * dot k y := by
    have hs : (∑ i, k i * (β * k i * dot k y)) = β * dot k k * dot k y := by
      change (∑ i, k i * (β * k i * dot k y)) = β * (∑ i, k i * k i) * dot k y
      simp only [sum_mul, mul_sum]
      apply sum_congr rfl
      intro i _
      ring
    calc
      _ = dot k y - (∑ i, k i * (β * k i * dot k y)) := by
        simp only [dot, mul_sub, sum_sub_distrib]
      _ = dot k y - β * dot k k * dot k y := by rw [hs]
      _ = _ := by ring
  rw [hx, hy] at hd
  have hd' : dot k x = dot k y := mul_left_cancel₀ h hd
  funext i
  have hi := congrFun heq i
  change x i - β * k i * dot k x = y i - β * k i * dot k y at hi
  rw [hd'] at hi
  linarith

/-- The same injectivity holds columnwise for the whole native matrix. No
rank-one compression of the retention source follows from scalar gating. -/
theorem retention_matrix_injective (k : I → ℝ) (β : ℝ)
    (h : 1 - β * dot k k ≠ 0) :
    Function.Injective (fun S : I → J → ℝ => retentionSource S k β) := by
  intro S T heq
  funext i j
  have hc : (fun a => S a j - β * k a * dot k (fun b => S b j)) =
      (fun a => T a j - β * k a * dot k (fun b => T b j)) := by
    funext a
    exact congrFun (congrFun heq a) j
  exact congrFun (erase_injective k β h hc) i

end Native

section GateChain

noncomputable def alphaOf (A z : ℝ) : ℝ :=
  Real.exp (-(Real.exp A * Real.log (1 + Real.exp z)))

noncomputable def rawSlope (A z : ℝ) : ℝ :=
  -(alphaOf A z * Real.exp A * (Real.exp z / (1 + Real.exp z)))

noncomputable def logSlope (A z : ℝ) : ℝ :=
  alphaOf A z * (-(Real.exp A * Real.log (1 + Real.exp z)))

/-- The installed smooth softplus branch: a_proj and dt_bias share rawSlope;
A_log has logSlope. The thresholded torch branch requires its own admission. -/
theorem alpha_gate_deriv (A z : ℝ → ℝ) (dA dz θ : ℝ)
    (hA : HasDerivAt A dA θ) (hz : HasDerivAt z dz θ) :
    HasDerivAt (fun t => alphaOf (A t) (z t))
      (rawSlope (A θ) (z θ) * dz + logSlope (A θ) (z θ) * dA) θ := by
  have hs := (hz.exp.const_add 1).log (by positivity : 1 + Real.exp (z θ) ≠ 0)
  have hh := (hA.exp.mul hs).neg.exp
  convert hh using 1
  · rfl
  · dsimp [rawSlope, logSlope, alphaOf]
    ring

variable {I J : Type*} [Fintype I]

/-- For fixed reached S/key/value, the native update is affine in retention.
This connects the actual nonlinear gate derivative to its shared S source. -/
theorem native_affine_retention (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α β : ℝ) (i : I) (j : J) :
    native S k v α β i j =
      α * retentionSource S k β i j + β * k i * v j := by
  simp only [native, retentionSource, column_scale]
  ring

/-- Original raw retention gate parameters generate one exact current native
source per head. Old-state/input directions are additional transported terms. -/
theorem native_retention_gate_deriv (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (β : ℝ) (A z : ℝ → ℝ) (dA dz θ : ℝ)
    (hA : HasDerivAt A dA θ) (hz : HasDerivAt z dz θ) (i : I) (j : J) :
    HasDerivAt (fun t => native S k v (alphaOf (A t) (z t)) β i j)
      ((rawSlope (A θ) (z θ) * dz + logSlope (A θ) (z θ) * dA) *
        retentionSource S k β i j) θ := by
  simp_rw [native_affine_retention]
  exact ((alpha_gate_deriv A z dA dz θ hA hz).mul_const _).add_const _

/-- One scalar write gate has the ordinary sigmoid slope, shared by all
coordinates in its projection row at the current incoming feature. -/
theorem write_gate_deriv (b : ℝ → ℝ) (db θ : ℝ)
    (hb : HasDerivAt b db θ) :
    HasDerivAt (fun t => Real.sigmoid (b t))
      (Real.sigmoid (b θ) * (1 - Real.sigmoid (b θ)) * db) θ :=
  (Real.hasDerivAt_sigmoid (b θ)).comp θ hb

/-- Write projection weights share the innovation outer product after the
sigmoid slope. The current input and old S are held only for this source leg. -/
theorem native_write_gate_deriv (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (α : ℝ) (b : ℝ → ℝ) (db θ : ℝ)
    (hb : HasDerivAt b db θ) (i : I) (j : J) :
    HasDerivAt (fun t => native S k v α (Real.sigmoid (b t)) i j)
      ((Real.sigmoid (b θ) * (1 - Real.sigmoid (b θ)) * db) *
        writeSource S k v α i j) θ := by
  have h := (hasDerivAt_const θ (α * S i j)).add
    ((write_gate_deriv b db θ hb).mul_const (writeSource S k v α i j))
  convert h using 1
  · funext t
    dsimp
    unfold native writeSource residual
    ring
  · simp

variable {P : Type*} [Fintype P]

/-- Dense row ownership is finite: each incoming feature coordinate supplies
its own coefficient even though the downstream effect direction is shared. -/
theorem projection_row_direction (w dw x : P → ℝ) :
    HasDerivAt (fun t => ∑ p, (w p + t * dw p) * x p)
      (∑ p, dw p * x p) 0 := by
  convert HasDerivAt.fun_sum (u := univ) (fun p _ =>
    ((hasDerivAt_const (0 : ℝ) (w p)).add
      ((hasDerivAt_id (0 : ℝ)).mul_const (dw p))).mul_const (x p)) using 1 <;> simp

/-- Combine the actual a_proj row, dt_bias and A_log interventions. This
proves their local parameter source from operators instead of assuming it. -/
theorem retention_projection_family_deriv (S : I → J → ℝ) (k : I → ℝ)
    (v : J → ℝ) (β : ℝ) (w dw x : P → ℝ) (A bias dlog dbias : ℝ)
    (i : I) (j : J) :
    HasDerivAt (fun t => native S k v
      (alphaOf (A + t * dlog) ((∑ p, (w p + t * dw p) * x p) + bias + t * dbias)) β i j)
      ((rawSlope A ((∑ p, w p * x p) + bias) * ((∑ p, dw p * x p) + dbias) +
        logSlope A ((∑ p, w p * x p) + bias) * dlog) * retentionSource S k β i j) 0 := by
  have hA := (hasDerivAt_const (0 : ℝ) A).add
    ((hasDerivAt_id (0 : ℝ)).mul_const dlog)
  have hz := ((projection_row_direction w dw x).add_const bias).add
    ((hasDerivAt_id (0 : ℝ)).mul_const dbias)
  convert native_retention_gate_deriv S k v β _ _ _ _ 0 hA hz i j using 1 <;> simp

/-- Raw retention-projection weights, dt_bias and A_log all use ONE native
retention matrix source. Their distinct coefficients are not summed away. -/
theorem retention_owner_contraction (x dw : P → ℝ) (A z dbias dlog : ℝ) :
    rawSlope A z * ((∑ p, dw p * x p) + dbias) + logSlope A z * dlog =
      (∑ p, dw p * (rawSlope A z * x p)) +
        dbias * rawSlope A z + dlog * logSlope A z := by
  rw [mul_add, mul_sum]
  have hs : (∑ p, rawSlope A z * (dw p * x p)) =
      ∑ p, dw p * (rawSlope A z * x p) := by
    apply sum_congr rfl
    intro p _
    ring
  rw [hs]
  ring

end GateChain

section JointEffect

variable {E F : Type*} [AddCommGroup E] [Module ℝ E]
    [AddCommGroup F] [Module ℝ F]
variable {P O : Type*} [Fintype P] [Fintype O] [DecidableEq O]

def coordinate (o : O) (a : ℝ) : O → ℝ := fun j => if j = o then a else 0

omit [Fintype P] [Fintype O] in
/-- All weights in ONE raw projection row share its propagated coordinate
source. L must include the raw conv cache as well as current S and output. -/
theorem projection_row_shared_source (L : (O → ℝ) →ₗ[ℝ] E)
    (o : O) (x : P → ℝ) (p : P) :
    L (coordinate o (x p)) = x p • L (coordinate o 1) := by
  have h : coordinate o (x p) = x p • coordinate o 1 := by
    funext j
    by_cases hj : j = o <;> simp [coordinate, hj]
  rw [h, map_smul]

/-- A dense projected direction is represented by finitely many raw-output
sources; this is a one-event factorization, not a bound over a lifetime. -/
theorem projection_all_rows_factor (L : (O → ℝ) →ₗ[ℝ] E)
    (dw : O → P → ℝ) (x : P → ℝ) :
    L (fun o => ∑ p, dw o p * x p) =
      ∑ o, (∑ p, dw o p * x p) • L (coordinate o 1) := by
  have h : (fun o => ∑ p, dw o p * x p) =
      ∑ o, (∑ p, dw o p * x p) • coordinate o 1 := by
    funext j
    simp [coordinate, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  rw [h, map_sum]
  simp only [map_smul]

omit [Fintype P] in
/-- Shared ownership survives frozen successor layers and joint record
operations whenever they supply the SAME admitted linearized map. -/
theorem shared_source_survives_transport (A : E →ₗ[ℝ] F)
    (U : E) (owner : P → ℝ) (p : P) :
    A (owner p • U) = owner p • A U := map_smul A _ _

omit [Fintype P] in
/-- The delayed third factor is contracted with the shared effect once;
ownership distributes the resulting scalar to the original parameters. -/
theorem shared_source_delayed_capture (m : E →ₗ[ℝ] ℝ)
    (U : E) (owner : P → ℝ) (p : P) :
    m (owner p • U) = owner p * m U := by
  rw [map_smul, smul_eq_mul]

/-- A raw projection may change persistent cache even when its immediate
output differential vanishes. Observing only current S/output is insufficient
whenever a future read can distinguish that cache direction. -/
theorem visible_zero_does_not_erase_cache (visible : E →ₗ[ℝ] F)
    (future : E →ₗ[ℝ] ℝ) (e : E) (_hv : visible e = 0)
    (hf : future e ≠ 0) : e ≠ 0 := by
  intro he
  exact hf (by rw [he, map_zero])

end JointEffect

end CGB.GDNLocalSourceFactors
