import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Logic.Function.Iterate
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination

/-!
# Holding versus correcting, and the latch transition (generation chain, 2026-11-04)

The user separated two abilities of a memory:
* **holding**: when nobody rewrites it, the content stays;
* **attractor correction**: when a small disturbance changes it, the dynamics pull it back.
and asked whether tuning decay against feedback is a phase transition: mistuned, the system
falls to one side; tuned, a richer picture appears.

Runs that motivated this: `results/cgb_mode_task_20261103/RESULT.md` (E2: the trained
latch holds a bit for 32k events when the text written into its direction is at most
0.1-0.3 of its real strength, and loses it within a few hundred events at full strength).

Part 1, any scalar map, perturbed orbit `x (t+1) = f (x t) + ε t`:
* `hold_accumulates`, `hold_error_unbounded`: the pure hold `f = id` keeps every content and
  every disturbance: errors add up without bound. (`linear_hold_keeps_error` in
  `GenerationLinearMemory` is the linear-map version of the same fact.)
* `contraction_noise_bound`: a map contracting toward `r` with factor `L < 1` keeps the
  error below `L^n |x₀ - r| + e / (1 - L)` forever under disturbances `|ε t| ≤ e`.
* `contraction_merges`: the same contraction maps any two contents together at rate `L^n`.
  Correction and holding a continuum exclude each other: what is corrected is only *which*
  attractor, not where inside its basin.

Part 2, the feedback renewal of `feedback_delta` read along one key (value 0):
`fb a c g r = a r + c · softsign (g r)`, retention `a ∈ [0,1)`, feedback `c, g > 0`.
Only the ratio `κ = c g / (1 - a)` decides the regime.
* `fb_sub_fixed` (the one identity everything follows from): with `r* = c/(1-a) - 1/g`,
  for `x ≥ 0`, `fb x - r* = (x - r*) · (a + (1-a)/(1+g x))`.
* `subcritical_forgets` (κ ≤ 1): every content shrinks, by at least the factor `a + c g`
  per event, whose distance from 1 is `(1-a)(1-κ)`.
* `supercritical_fixed`, `supercritical_origin_repels`, `fb_odd` (κ > 1): two attractors
  `± r*`, and small content grows away from 0 toward them.
* `latch_converges`, `latch_corrects` (κ > 1): on `[ρ, ∞)` with `0 < ρ ≤ r*` the map
  contracts toward `r*` with factor `L ρ = a + (1-a)/(1+g ρ) < 1`; disturbances up to
  `(1 - L ρ) B` keep the state within `B ≤ r* - ρ` of `r*` forever. Larger disturbances can
  carry it out of the basin: this is the E2 threshold.
* `latch_tolerance`, `tolerance_falls_with_decay` (κ > 1): at the best basin margin
  `ρ = (√κ - 1)/g` the disturbance `latch_corrects` tolerates per event is
  `(√c - √((1-a)/g))²`. Stronger decay (smaller `a`) lowers it: decay does not protect a
  latch, it weakens it. What protects it is a smaller write into its direction.
* `rates_vanish_at_transition`: the forgetting rate below and the correction rate at `r*`
  above are `(1-a)(1-κ)` and `(1-a)(1-1/κ)`: both go to zero at κ = 1 (critical slowing).

What this does and does not say. The "rich picture" at the transition is slow time: near
κ = 1 neither forgetting nor correcting is fast, so content lingers and responds strongly to
input. It is not holding: at κ = 1 content still shrinks (`subcritical_forgets` includes it).
Holding without correction is the marginal point of the linear map (`a = 1`, nothing
written), which is the self-state S; correction requires κ > 1 and discreteness.
-/

namespace CGB.Generation.HoldCorrect

open Finset

/-! ## Part 1: holding and correcting for a general scalar map -/

/-- Perturbed orbit `x 0 = x₀`, `x (t+1) = f (x t) + ε t`. -/
def orbit (f : ℝ → ℝ) (ε : ℕ → ℝ) (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | t + 1 => f (orbit f ε x₀ t) + ε t

/-- **Holding keeps every disturbance**: under the pure hold the state is the start plus the
sum of all disturbances. -/
theorem hold_accumulates (ε : ℕ → ℝ) (x₀ : ℝ) (n : ℕ) :
    orbit id ε x₀ n = x₀ + ∑ t ∈ range n, ε t := by
  induction n with
  | zero => simp [orbit]
  | succ n ih => rw [orbit, id, ih, sum_range_succ]; ring

/-- **A held content drifts without bound** under any constant disturbance `e > 0`: for
every bound `B` some time exceeds it. -/
theorem hold_error_unbounded {e : ℝ} (he : 0 < e) (x₀ B : ℝ) :
    ∃ n : ℕ, B < |orbit id (fun _ => e) x₀ n - x₀| := by
  obtain ⟨n, hn⟩ := exists_nat_gt (B / e)
  refine ⟨n, ?_⟩
  have hsum : orbit id (fun _ => e) x₀ n - x₀ = n * e := by
    rw [hold_accumulates]; simp
  rw [hsum, abs_of_nonneg (by positivity)]
  rwa [div_lt_iff₀ he] at hn

/-- **Contraction corrects**: if `f` pulls every state toward `r` by the factor `L < 1`,
disturbances bounded by `e` leave an error of at most `L^n |x₀ - r| + e / (1 - L)` at every
time, a bound that does not grow with `n`. -/
theorem contraction_noise_bound (f : ℝ → ℝ) {r L e : ℝ} (hL0 : 0 ≤ L) (hL1 : L < 1)
    (hf : ∀ x, |f x - r| ≤ L * |x - r|) (ε : ℕ → ℝ) (he : ∀ t, |ε t| ≤ e) (x₀ : ℝ)
    (n : ℕ) : |orbit f ε x₀ n - r| ≤ L ^ n * |x₀ - r| + e / (1 - L) := by
  have hpos : 0 < 1 - L := by linarith
  have he0 : 0 ≤ e := (abs_nonneg _).trans (he 0)
  induction n with
  | zero =>
    simp only [orbit, pow_zero, one_mul]
    linarith [div_nonneg he0 hpos.le]
  | succ n ih =>
    have step : |orbit f ε x₀ (n + 1) - r| ≤ L * |orbit f ε x₀ n - r| + e := by
      simp only [orbit]
      calc |f (orbit f ε x₀ n) + ε n - r| = |(f (orbit f ε x₀ n) - r) + ε n| := by ring_nf
        _ ≤ |f (orbit f ε x₀ n) - r| + |ε n| := abs_add_le _ _
        _ ≤ L * |orbit f ε x₀ n - r| + e := add_le_add (hf _) (he n)
    calc |orbit f ε x₀ (n + 1) - r| ≤ L * |orbit f ε x₀ n - r| + e := step
      _ ≤ L * (L ^ n * |x₀ - r| + e / (1 - L)) + e := by gcongr
      _ = L ^ (n + 1) * |x₀ - r| + e / (1 - L) := by field_simp; ring

/-- **Correction merges contents**: a map contracting all pairs by `L` brings any two
contents within `L^n` of their initial gap. -/
theorem contraction_merges (f : ℝ → ℝ) {L : ℝ} (hL0 : 0 ≤ L)
    (hf : ∀ x y, |f x - f y| ≤ L * |x - y|) (x y : ℝ) (n : ℕ) :
    |f^[n] x - f^[n] y| ≤ L ^ n * |x - y| := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', pow_succ]
    calc |f (f^[n] x) - f (f^[n] y)| ≤ L * |f^[n] x - f^[n] y| := hf _ _
      _ ≤ L * (L ^ n * |x - y|) := by gcongr
      _ = L ^ n * L * |x - y| := by ring

/-! ## Part 2: the feedback renewal along one key -/

/-- One `feedback_delta` renewal of the read along a key that is read at every event, with
zero value: `a r + c · softsign (g r)`; `c = β η` and `a = α (1 - β)` in the layer. -/
noncomputable def fb (a c g r : ℝ) : ℝ := a * r + c * (g * r / (1 + |g * r|))

/-- The nonzero fixed point when `κ = c g / (1 - a) > 1`. -/
noncomputable def rstar (a c g : ℝ) : ℝ := c / (1 - a) - 1 / g

/-- Correction factor toward `r*` at a state `x ≥ 0`. -/
noncomputable def factor (a g x : ℝ) : ℝ := a + (1 - a) / (1 + g * x)

/-- Two copies of the latch: the renewal is odd, so `-r*` mirrors `r*`. -/
theorem fb_odd (a c g r : ℝ) : fb a c g (-r) = -fb a c g r := by
  simp only [fb, mul_neg, abs_neg]; ring

theorem fb_nonneg_eq {a c g x : ℝ} (hg : 0 < g) (hx : 0 ≤ x) :
    fb a c g x = a * x + c * (g * x / (1 + g * x)) := by
  rw [fb, abs_of_nonneg (by positivity)]

/-- **The identity**: for `x ≥ 0`, `fb x - r* = (x - r*) · (a + (1-a)/(1+g x))`. -/
theorem fb_sub_fixed {a c g x : ℝ} (ha1 : a < 1) (hg : 0 < g) (hx : 0 ≤ x) :
    fb a c g x - rstar a c g = (x - rstar a c g) * factor a g x := by
  have h1 : (1 - a) ≠ 0 := by linarith
  have h2 : (1 + g * x) ≠ 0 := by positivity
  rw [fb_nonneg_eq hg hx, rstar, factor]
  field_simp
  ring

theorem factor_lt_one {a g x : ℝ} (ha1 : a < 1) (hg : 0 < g) (hx : 0 < x) : factor a g x < 1 := by
  have hgx : 1 < 1 + g * x := by nlinarith
  have : (1 - a) / (1 + g * x) < 1 - a := div_lt_self (by linarith) hgx
  simp only [factor]; linarith

theorem factor_nonneg {a g x : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (hg : 0 < g) (hx : 0 ≤ x) :
    0 ≤ factor a g x := by
  have : 0 ≤ (1 - a) / (1 + g * x) := div_nonneg (by linarith) (by positivity)
  simp only [factor]; linarith

theorem factor_le_of_le {a g ρ x : ℝ} (ha1 : a < 1) (hg : 0 < g) (hρ : 0 ≤ ρ) (hx : ρ ≤ x) :
    factor a g x ≤ factor a g ρ := by
  simp only [factor]
  gcongr

/-- **κ ≤ 1, forgetting**: every positive content shrinks by at least the factor `a + c g`
per event (and stays positive): only 0 is left in the long run. -/
theorem subcritical_forgets {a c g x : ℝ} (ha0 : 0 ≤ a) (hc : 0 < c) (hg : 0 < g)
    (hsub : c * g ≤ 1 - a) (hx : 0 < x) :
    0 < fb a c g x ∧ fb a c g x ≤ (a + c * g) * x ∧ fb a c g x < x := by
  rw [fb_nonneg_eq hg hx.le]
  have hgx : 0 < g * x := mul_pos hg hx
  have hfrac : g * x / (1 + g * x) < g * x := div_lt_self hgx (by linarith)
  have hfrac0 : 0 < g * x / (1 + g * x) := by positivity
  refine ⟨by positivity, ?_, ?_⟩
  · nlinarith
  · nlinarith

/-- **κ > 1, the latch exists**: `r* > 0` and it is fixed. -/
theorem supercritical_fixed {a c g : ℝ} (ha1 : a < 1) (hg : 0 < g) (hsup : 1 - a < c * g) :
    0 < rstar a c g ∧ fb a c g (rstar a c g) = rstar a c g := by
  have hpos : 0 < rstar a c g := by
    rw [rstar, sub_pos, div_lt_div_iff₀ hg (by linarith)]; linarith
  refine ⟨hpos, ?_⟩
  have := fb_sub_fixed (c := c) ha1 hg hpos.le
  rw [sub_self, zero_mul] at this
  linarith

/-- **κ > 1, the origin repels**: small positive content grows toward `r*`. -/
theorem supercritical_origin_repels {a c g x : ℝ} (ha1 : a < 1) (hg : 0 < g)
    (hx : 0 < x) (hxr : x < rstar a c g) : x < fb a c g x := by
  have hid := fb_sub_fixed (c := c) ha1 hg hx.le
  have hf := factor_lt_one ha1 hg hx
  nlinarith

/-- **κ > 1, convergence inside the basin**: from any `x₀ ≥ ρ`, `0 < ρ ≤ r*`, the
undisturbed latch stays above `min x₀ r* ≥ ρ` and approaches `r*` at rate `factor ρ < 1`. -/
theorem latch_converges {a c g ρ x₀ : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (hg : 0 < g)
    (hρ : 0 < ρ) (hρr : ρ ≤ rstar a c g) (hx : ρ ≤ x₀) (n : ℕ) :
    ρ ≤ (fb a c g)^[n] x₀ ∧
      |(fb a c g)^[n] x₀ - rstar a c g| ≤ factor a g ρ ^ n * |x₀ - rstar a c g| := by
  induction n with
  | zero => simp [hx]
  | succ n ih =>
    obtain ⟨hlow, hbound⟩ := ih
    set x := (fb a c g)^[n] x₀
    set m := factor a g x
    have hx0 : 0 ≤ x := hρ.le.trans hlow
    have hm0 : 0 ≤ m := factor_nonneg ha0 ha1 hg hx0
    have hm1 : m < 1 := factor_lt_one ha1 hg (hρ.trans_le hlow)
    have hmρ : m ≤ factor a g ρ := factor_le_of_le ha1 hg hρ.le hlow
    have hid := fb_sub_fixed (c := c) ha1 hg hx0
    rw [Function.iterate_succ_apply']
    refine ⟨?_, ?_⟩
    · -- fb x = (1 - m) r* + m x, a convex combination of two numbers ≥ ρ
      have : fb a c g x = (1 - m) * rstar a c g + m * x := by linarith
      rw [this]; nlinarith
    · rw [show fb a c g x - rstar a c g = (x - rstar a c g) * m from hid, abs_mul,
        abs_of_nonneg hm0, pow_succ]
      calc |x - rstar a c g| * m ≤ |x - rstar a c g| * factor a g ρ :=
            mul_le_mul_of_nonneg_left hmρ (abs_nonneg _)
        _ ≤ factor a g ρ ^ n * |x₀ - rstar a c g| * factor a g ρ :=
            mul_le_mul_of_nonneg_right hbound (factor_nonneg ha0 ha1 hg hρ.le)
        _ = factor a g ρ ^ n * factor a g ρ * |x₀ - rstar a c g| := by ring

/-- **κ > 1, attractor correction under disturbance**: if the state starts within `B` of
`r*`, the basin margin covers it (`B ≤ r* - ρ`, `ρ > 0`), and every disturbance is at most
`(1 - factor ρ) · B`, the state stays within `B` of `r*` forever. Disturbances larger than
this can carry it out of the basin. -/
theorem latch_corrects {a c g ρ B e : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (hg : 0 < g)
    (hρ : 0 < ρ) (hB : B ≤ rstar a c g - ρ) (he : e ≤ (1 - factor a g ρ) * B)
    (ε : ℕ → ℝ) (hε : ∀ t, |ε t| ≤ e) (x₀ : ℝ) (h₀ : |x₀ - rstar a c g| ≤ B) (n : ℕ) :
    |orbit (fb a c g) ε x₀ n - rstar a c g| ≤ B := by
  induction n with
  | zero => simpa [orbit] using h₀
  | succ n ih =>
    set x := orbit (fb a c g) ε x₀ n
    have hlow : ρ ≤ x := by
      have := (abs_le.mp ih).1; linarith
    have hx0 : 0 ≤ x := hρ.le.trans hlow
    have hm0 := factor_nonneg ha0 ha1 hg hx0
    have hmρ : factor a g x ≤ factor a g ρ := factor_le_of_le ha1 hg hρ.le hlow
    have hid := fb_sub_fixed (c := c) ha1 hg hx0
    have hcontract : |fb a c g x - rstar a c g| ≤ factor a g ρ * B := by
      rw [hid, abs_mul, abs_of_nonneg hm0, mul_comm]
      exact mul_le_mul hmρ ih (abs_nonneg _) (factor_nonneg ha0 ha1 hg hρ.le)
    simp only [orbit]
    calc |fb a c g x + ε n - rstar a c g| = |(fb a c g x - rstar a c g) + ε n| := by ring_nf
      _ ≤ |fb a c g x - rstar a c g| + |ε n| := abs_add_le _ _
      _ ≤ factor a g ρ * B + e := add_le_add hcontract (hε n)
      _ ≤ B := by linarith

/-- **The latch's noise tolerance** (κ > 1). Write `u = √((1-a)/g)` and `v = √c` (given as
positive numbers with these squares, so no square root is needed): `√κ = v/u > 1`. With the
margin `ρ = (v/u - 1)/g` the condition of `latch_corrects` holds (`0 < ρ ≤ r*`) and the
tolerated disturbance per event, `(1 - factor ρ) · (r* - ρ)`, equals `(v - u)²`. -/
theorem latch_tolerance {a c g u v : ℝ} (ha1 : a < 1) (hg : 0 < g) (hu : 0 < u) (hv : 0 < v)
    (huu : u ^ 2 = (1 - a) / g) (hvv : v ^ 2 = c) (huv : u < v) :
    let ρ := (v / u - 1) / g
    0 < ρ ∧ ρ ≤ rstar a c g ∧ (1 - factor a g ρ) * (rstar a c g - ρ) = (v - u) ^ 2 := by
  intro ρ
  have hs : 1 - a = u ^ 2 * g := by rw [huu]; field_simp
  have hu0 : u ≠ 0 := hu.ne'
  have hg0 : g ≠ 0 := hg.ne'
  have hs0 : (1 - a) ≠ 0 := by rw [hs]; positivity
  have hw1 : 1 < v / u := by rw [lt_div_iff₀ hu]; linarith
  have hstar : rstar a c g - ρ = (v / u) * (v / u - 1) / g := by
    simp only [rstar, ρ]; rw [← hvv, hs]; field_simp; ring
  refine ⟨div_pos (by linarith) hg, ?_, ?_⟩
  · have : 0 ≤ (v / u) * (v / u - 1) / g := div_nonneg (by nlinarith) hg.le
    linarith
  · rw [hstar]
    simp only [factor, ρ]
    rw [hs]
    have h1 : 1 + g * ((v / u - 1) / g) = v / u := by field_simp; ring
    rw [h1]
    field_simp
    linear_combination v * hs

/-- **Decay weakens the latch**: the tolerance `(v - u)²` with `u = √((1-a)/g)` falls as
decay grows: a larger `u` (stronger decay, smaller retention `a`) with the latch still present
(`u ≤ v`) tolerates no more disturbance. -/
theorem tolerance_falls_with_decay {u₁ u₂ v : ℝ} (h : u₁ ≤ u₂) (hlatch : u₂ ≤ v) :
    (v - u₂) ^ 2 ≤ (v - u₁) ^ 2 :=
  pow_le_pow_left₀ (by linarith) (by linarith) 2

/-- **Critical slowing at κ = 1**: the forgetting rate below the transition and the
correction rate at `r*` above it are `(1-a)(1-κ)` and `(1-a)(1-1/κ)` with
`κ = c g / (1 - a)`; both vanish as κ → 1. -/
theorem rates_vanish_at_transition {a c g : ℝ} (ha1 : a < 1) (hg : 0 < g) (hc : 0 < c) :
    1 - (a + c * g) = (1 - a) * (1 - c * g / (1 - a)) ∧
      1 - factor a g (rstar a c g) = (1 - a) * (1 - 1 / (c * g / (1 - a))) := by
  have h1 : (1 - a) ≠ 0 := by linarith
  have hcg : c * g ≠ 0 := by positivity
  constructor
  · field_simp
    ring
  · simp only [factor, rstar]
    have : 1 + g * (c / (1 - a) - 1 / g) = c * g / (1 - a) := by field_simp; ring
    rw [this]
    field_simp
    ring

/-- Relaxation time near the transition: the inverse of the rate `(1-a)|1-κ|` (exact below
the transition by `rates_vanish_at_transition`; above it the rate is `(1-a)(1-1/κ)`, the same
to first order in `κ - 1`). -/
noncomputable def relaxTime (a κ : ℝ) : ℝ := 1 / ((1 - a) * |1 - κ|)

/-- **Scale-free timescales near criticality**: a head's relaxation time exceeds `T` exactly
when its `κ` lies within `1 / ((1-a) T)` of 1. The band shrinks as `1/T`, so a population whose
`κ` is spread over a neighbourhood of 1 contains every timescale, the share above `T` falling
as `1/T`: one mechanism family, many timescales, without assigning them. -/
theorem long_timescale_band {a κ T : ℝ} (ha : a < 1) (hT : 0 < T) (hκ : κ ≠ 1) :
    T < relaxTime a κ ↔ |κ - 1| < 1 / ((1 - a) * T) := by
  have hs : 0 < 1 - a := by linarith
  have hd : 0 < |1 - κ| := abs_pos.mpr (sub_ne_zero.mpr (Ne.symm hκ))
  rw [relaxTime, abs_sub_comm κ 1, lt_div_iff₀ (mul_pos hs hd), lt_div_iff₀ (mul_pos hs hT)]
  constructor <;> intro h <;> nlinarith [mul_pos hs hd, mul_pos hs hT]

end CGB.Generation.HoldCorrect
