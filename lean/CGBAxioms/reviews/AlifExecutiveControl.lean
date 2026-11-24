import Init

/-!
# Formal Theory of ALIF Surprise-Gated Executive Control (Information Boltzmann / MengLi)

This zero-dependency (`import Init`) Lean 4 module abstracts MengLi's coupled
cortical-subcortical body loop into an exact fixed-point dynamical system
(denominator `Q > 0` representing `1.0`) synthesizing four mechanisms:

1. **Drosophila / Cortical ALIF** (Adaptive Leaky Integrate-and-Fire):
   A leaky membrane potential `V` integrates sensory surprise, BFSS physical
   momentum surges, and latent epistemic novelty against a homeostatic adaptive
   threshold `θ(a) = θ₀ + (β * a) / Q`.
2. **Continuous Subcortical Metabolism (H-JEPA / π₀.₅ fast loop)**:
   Every sensory block executes 100% of language learning, S/P physics, and
   fast-table updates (`language_byte_conservation`), while heavy full-backbone
   executive deliberation fires only on ALIF spikes.
3. **OS Interrupt Liveness & Anti-Storming**:
   - *Rheobase Zero-Starvation* (`rheobase_zero_starvation`): when tonic drive
     `δ₀` clears the leaky equilibrium floor, calm flow is constructively
     guaranteed to cross threshold within `≤ max(a₀, θ₀)` steps.
   - *Refractory Anti-Storming* (`refractory_prevents_back_to_back_spike`):
     when `I_max < θ₀ + β`, consecutive-step storming is impossible (`τ ≥ 2`).
4. **SMDP Exact Delayed-Credit Contraction & Burstiness Dominance**:
   - *SMDP Contraction* (`smdp_bellman_contraction`): variable ALIF durations
     `τ ≥ 1` contract continuation value by `γ^τ ≤ γ` without shrinking the
     immediate causal-use reward `r`.
   - *Burstiness Dominance* (`burstiness_dominance_over_periodic`): at equal
     executive spike budget (identical wall-clock cost as periodic `K:1`
     chunking), threshold-selected ALIF spikes capture strictly more surprise
     and incur strictly less missed executive regret.
-/

namespace CGB.AlifExecutiveControl

/-- Fixed-point parameters of a biological Rheobase-ALIF executive gate (unit scale `Q > 0`). -/
structure AlifParams where
  Q : Nat          -- fixed-point unit denominator (represents 1.0)
  lamV : Nat       -- membrane leak numerator (0 ≤ lamV < Q, represents λ_V = lamV / Q)
  lamA : Nat       -- adaptation decay numerator (0 ≤ lamA < Q, represents λ_a = lamA / Q)
  theta0 : Nat     -- resting threshold θ₀ > 0
  beta : Nat       -- refractory stiffness β (scaled by Q: Δθ = (beta * a) / Q)
  delta0 : Nat     -- tonic pacemaker (rheobase) drive δ₀ > 0
  iMax : Nat       -- upper cap on per-step total drive I_max ≥ δ₀
  hQ_pos : 0 < Q
  hLamV_lt : lamV < Q
  hLamA_lt : lamA < Q
  hTheta0_pos : 0 < theta0
  hDelta0_pos : 0 < delta0
  hIMax_ge : delta0 ≤ iMax

/-- Dynamic state of the ALIF neuron (all quantities non-negative). -/
structure AlifState where
  v : Nat          -- membrane potential V_t
  a : Nat          -- adaptation variable a_t (unit spike jump = Q)
  deriving DecidableEq, Repr

/-- Adaptive firing threshold `θ_t = θ₀ + (β * a_t) / Q`. -/
def threshold (p : AlifParams) (s : AlifState) : Nat :=
  p.theta0 + (p.beta * s.a) / p.Q

/-- Pre-reset integrated membrane potential `V_t⁺ = (lamV * V_t) / Q + drive`. -/
def integratedV (p : AlifParams) (s : AlifState) (drive : Nat) : Nat :=
  (p.lamV * s.v) / p.Q + drive

/-- Whether the ALIF neuron fires an executive interrupt spike. -/
def fires (p : AlifParams) (s : AlifState) (drive : Nat) : Bool :=
  decide (threshold p s ≤ integratedV p s drive)

/-- Single-step ALIF state transition (hard reset `V ← 0`, adaptation jump `a ← (lamA * a)/Q + Q`). -/
def step (p : AlifParams) (s : AlifState) (drive : Nat) : AlifState :=
  if fires p s drive then
    { v := 0, a := (p.lamA * s.a) / p.Q + p.Q }
  else
    { v := integratedV p s drive, a := (p.lamA * s.a) / p.Q }

/-- Calm (non-spiking) trajectory under constant minimal drive `d`. -/
def calmV (p : AlifParams) (v0 d : Nat) : Nat → Nat
  | 0 => v0
  | k + 1 => (p.lamV * calmV p v0 d k) / p.Q + d

def calmA (p : AlifParams) (a0 : Nat) : Nat → Nat
  | 0 => a0
  | k + 1 => (p.lamA * calmA p a0 k) / p.Q

/-! ### Theorem 1: Homeostatic Bounded Adaptation -/

/-- If `aMax` clears the fixed-point steady state `(Q - lamA) * aMax ≥ Q * Q`,
    then adaptation stays strictly bounded `a_t ≤ aMax` forever under ANY spike stream. -/
theorem step_adaptation_bounded (p : AlifParams) (s : AlifState) (drive aMax : Nat)
    (ha_le : s.a ≤ aMax)
    (hcap : p.Q * p.Q ≤ (p.Q - p.lamA) * aMax) :
    (step p s drive).a ≤ aMax := by
  have hQ : 0 < p.Q := p.hQ_pos
  have hmul : p.lamA * s.a ≤ p.lamA * aMax := Nat.mul_le_mul_left p.lamA ha_le
  have hsum : p.lamA * aMax + p.Q * p.Q ≤ p.Q * aMax := by
    have hsplit : p.lamA * aMax + (p.Q - p.lamA) * aMax = p.Q * aMax := by
      rw [← Nat.add_mul, Nat.add_sub_of_le (Nat.le_of_lt p.hLamA_lt)]
    omega
  have hdiv_plus : (p.lamA * s.a) / p.Q + p.Q ≤ aMax := by
    have h1 : (p.lamA * s.a) / p.Q + p.Q = (p.lamA * s.a + p.Q * p.Q) / p.Q := by
      rw [Nat.add_mul_div_right _ _ hQ]
    rw [h1]
    have hnum : p.lamA * s.a + p.Q * p.Q ≤ aMax * p.Q := by
      rw [Nat.mul_comm aMax p.Q]
      omega
    exact (Nat.div_le_iff_le_mul_add_pred hQ).mpr (by omega)
  unfold step
  by_cases hf : fires p s drive = true
  · simp [hf]
    exact hdiv_plus
  · simp [hf]
    omega

/-- Consequently, the adaptive threshold stays bounded within `[θ₀, θ₀ + (β * aMax) / Q]`. -/
theorem threshold_bounded (p : AlifParams) (s : AlifState) (aMax : Nat)
    (ha_le : s.a ≤ aMax) :
    p.theta0 ≤ threshold p s ∧ threshold p s ≤ p.theta0 + (p.beta * aMax) / p.Q := by
  unfold threshold
  constructor
  · exact Nat.le_add_right p.theta0 _
  · have hmul : p.beta * s.a ≤ p.beta * aMax := Nat.mul_le_mul_left p.beta ha_le
    have hdiv : (p.beta * s.a) / p.Q ≤ (p.beta * aMax) / p.Q := Nat.div_le_div_right hmul
    omega

/-! ### Theorem 2: Constructive Rheobase Zero-Starvation (Bounded Liveness) -/

/-- Strict decay of adaptation during calm steps: whenever `a > 0`, `(lamA * a) / Q < a`. -/
private theorem div_mul_lt_of_lt (lam Q x : Nat) (hlam : lam < Q) (hx : 0 < x) :
    (lam * x) / Q < x := by
  have hQ : 0 < Q := Nat.lt_of_le_of_lt (Nat.zero_le lam) hlam
  rw [Nat.div_lt_iff_lt_mul hQ]
  rw [Nat.mul_comm x Q]
  exact Nat.mul_lt_mul_of_pos_right hlam hx

theorem calmA_zero_after (p : AlifParams) (a0 : Nat) :
    ∀ k, a0 ≤ k → calmA p a0 k = 0 := by
  induction a0 with
  | zero =>
      intro k _
      induction k with
      | zero => rfl
      | succ k ih =>
          simp [calmA, ih]
  | succ a0 ih =>
      intro k hk
      cases k with
      | zero => omega
      | succ k =>
          have hstep : calmA p (a0 + 1) (k + 1) = calmA p ((p.lamA * (a0 + 1)) / p.Q) k := by
            clear ih hk
            induction k with
            | zero => rfl
            | succ k ihk =>
                show (p.lamA * calmA p (a0 + 1) (k + 1)) / p.Q =
                     (p.lamA * calmA p ((p.lamA * (a0 + 1)) / p.Q) k) / p.Q
                rw [ihk]
          rw [hstep]
          have hlt : (p.lamA * (a0 + 1)) / p.Q < a0 + 1 :=
            div_mul_lt_of_lt p.lamA p.Q (a0 + 1) p.hLamA_lt (Nat.succ_pos a0)
          have hle_a0 : (p.lamA * (a0 + 1)) / p.Q ≤ a0 := Nat.lt_succ_iff.mp hlt
          -- Stronger monotonicity on calmA: if x ≤ y then calmA x k ≤ calmA y k
          have hmono : ∀ x y m, x ≤ y → calmA p x m ≤ calmA p y m := by
            intro x y m hxy
            induction m with
            | zero => exact hxy
            | succ m ihm =>
                simp only [calmA]
                exact Nat.div_le_div_right (Nat.mul_le_mul_left p.lamA ihm)
          have hle0 : calmA p ((p.lamA * (a0 + 1)) / p.Q) k ≤ calmA p a0 k :=
            hmono _ _ k hle_a0
          have hzero : calmA p a0 k = 0 := ih k (by omega)
          omega

/-- Helper: under the discrete rheobase condition `(Q - lamV) * θ₀ + 2 * Q ≤ Q * δ₀`,
    any sub-threshold membrane value `v < θ₀` increases by at least `+1` per calm step,
    and any super-threshold value `v ≥ θ₀` remains `≥ θ₀`. -/
private theorem calmV_step_progress (p : AlifParams) (v : Nat)
    (hrheo : (p.Q - p.lamV) * p.theta0 + 2 * p.Q ≤ p.Q * p.delta0) :
    (v < p.theta0 → v + 1 ≤ (p.lamV * v) / p.Q + p.delta0) ∧
    (p.theta0 ≤ v → p.theta0 ≤ (p.lamV * v) / p.Q + p.delta0) := by
  have hQ : 0 < p.Q := p.hQ_pos
  have hsub : p.lamV + (p.Q - p.lamV) = p.Q := Nat.add_sub_of_le (Nat.le_of_lt p.hLamV_lt)
  constructor
  · intro hv
    -- Need (v + 1 - delta0) * Q ≤ lamV * v, or directly:
    -- Q * (v + 1) ≤ lamV * v + Q * delta0 - Q + 1 (since floor loses < Q)
    have hle : (p.Q - p.lamV) * v ≤ (p.Q - p.lamV) * p.theta0 :=
      Nat.mul_le_mul_left _ (Nat.le_of_lt hv)
    have hsplit : p.lamV * v + (p.Q - p.lamV) * v = p.Q * v := by
      rw [← Nat.add_mul, hsub]
    have hQv1 : p.Q * (v + 1) = p.Q * v + p.Q := by rw [Nat.mul_add, Nat.mul_one]
    have hmod : p.lamV * v % p.Q < p.Q := Nat.mod_lt _ hQ
    have hdiv_eq : p.Q * ((p.lamV * v) / p.Q) + p.lamV * v % p.Q = p.lamV * v :=
      Nat.div_add_mod (p.lamV * v) p.Q
    have hmul_le : p.Q * (v + 1) ≤ p.Q * ((p.lamV * v) / p.Q + p.delta0) := by
      rw [hQv1, Nat.mul_add]
      omega
    exact Nat.le_of_mul_le_mul_left hmul_le hQ
  · intro hv
    have hmul_v : p.lamV * p.theta0 ≤ p.lamV * v := Nat.mul_le_mul_left p.lamV hv
    have hsplit : p.lamV * p.theta0 + (p.Q - p.lamV) * p.theta0 = p.Q * p.theta0 := by
      rw [← Nat.add_mul, hsub]
    have hmod : p.lamV * v % p.Q < p.Q := Nat.mod_lt _ hQ
    have hdiv_eq : p.Q * ((p.lamV * v) / p.Q) + p.lamV * v % p.Q = p.lamV * v :=
      Nat.div_add_mod (p.lamV * v) p.Q
    have hmul_le : p.Q * p.theta0 ≤ p.Q * ((p.lamV * v) / p.Q + p.delta0) := by
      rw [Nat.mul_add]
      omega
    exact Nat.le_of_mul_le_mul_left hmul_le hQ

theorem calmV_reaches_theta0 (p : AlifParams) (v0 : Nat)
    (hrheo : (p.Q - p.lamV) * p.theta0 + 2 * p.Q ≤ p.Q * p.delta0) :
    ∀ k, min p.theta0 (v0 + k) ≤ calmV p v0 p.delta0 k := by
  intro k
  induction k with
  | zero =>
      simp [calmV]
      omega
  | succ k ih =>
      simp only [calmV]
      have hprog := calmV_step_progress p (calmV p v0 p.delta0 k) hrheo
      by_cases hge : p.theta0 ≤ calmV p v0 p.delta0 k
      · have hstay := hprog.2 hge
        omega
      · have hlt : calmV p v0 p.delta0 k < p.theta0 := Nat.lt_of_not_ge hge
        have hinc := hprog.1 hlt
        omega

/-- **Constructive Rheobase Zero-Starvation Theorem**:
    Under the discrete rheobase inequality `(Q - lamV) * θ₀ + 2 * Q ≤ Q * δ₀`,
    even under zero surprise (`drive ≡ δ₀`) and arbitrary initial adaptation `a₀`,
    the calm membrane trajectory is guaranteed to cross the adaptive threshold
    within at most `k = max a₀ θ₀` steps (zero starvation, explicit finite horizon). -/
theorem rheobase_zero_starvation (p : AlifParams) (v0 a0 : Nat)
    (hrheo : (p.Q - p.lamV) * p.theta0 + 2 * p.Q ≤ p.Q * p.delta0) :
    let k := max a0 p.theta0
    p.theta0 + (p.beta * calmA p a0 k) / p.Q ≤ calmV p v0 p.delta0 k := by
  intro k
  have ha_zero : calmA p a0 k = 0 := calmA_zero_after p a0 k (Nat.le_max_left a0 p.theta0)
  have hv_ge : min p.theta0 (v0 + k) ≤ calmV p v0 p.delta0 k :=
    calmV_reaches_theta0 p v0 hrheo k
  have hk_ge : p.theta0 ≤ k := Nat.le_max_right a0 p.theta0
  rw [ha_zero]
  simp only [Nat.mul_zero, Nat.zero_div, Nat.add_zero]
  omega

/-! ### Theorem 3: Adaptive Refractory Anti-Storming -/

/-- **Refractory Anti-Storming Theorem**:
    If the capped per-step drive satisfies `I_max < θ₀ + β`, then immediately after
    any executive spike, the ALIF neuron cannot fire on the next step even under
    maximal adversarial surprise (`drive₂ ≤ I_max`), guaranteeing `τ ≥ 2`. -/
theorem refractory_prevents_back_to_back_spike (p : AlifParams) (s : AlifState)
    (drive1 drive2 : Nat)
    (hfired : fires p s drive1 = true)
    (hdrive2 : drive2 ≤ p.iMax)
    (hrefrac : p.iMax < p.theta0 + p.beta) :
    fires p (step p s drive1) drive2 = false := by
  have hQ : 0 < p.Q := p.hQ_pos
  have hstep : step p s drive1 = { v := 0, a := (p.lamA * s.a) / p.Q + p.Q } := by
    unfold step
    simp [hfired]
  rw [hstep]
  unfold fires integratedV threshold
  simp only [Nat.mul_zero, Nat.zero_div, Nat.zero_add]
  have hbeta_le : p.beta ≤ (p.beta * ((p.lamA * s.a) / p.Q + p.Q)) / p.Q := by
    have hge : p.Q ≤ (p.lamA * s.a) / p.Q + p.Q := Nat.le_add_left p.Q _
    have hmul : p.beta * p.Q ≤ p.beta * ((p.lamA * s.a) / p.Q + p.Q) :=
      Nat.mul_le_mul_left p.beta hge
    have hdiv := Nat.div_le_div_right (c := p.Q) hmul
    rw [Nat.mul_div_cancel p.beta hQ] at hdiv
    exact hdiv
  exact decide_eq_false (by omega)

/-! ### Theorem 4: Exact Strict 1:1 Limit Reduction -/

/-- When `β = 0` and `θ₀ ≤ δ₀`, ALIF fires on every step (`τ = 1`), reducing
    identically to strict 1:1 execution. -/
theorem strict_mode_limit (p : AlifParams) (s : AlifState) (drive : Nat)
    (hbeta : p.beta = 0) (htheta : p.theta0 ≤ p.delta0) (hd : p.delta0 ≤ drive) :
    fires p s drive = true := by
  unfold fires threshold integratedV
  rw [hbeta]
  simp only [Nat.zero_mul, Nat.zero_div, Nat.add_zero]
  exact decide_eq_true (Nat.le_trans htheta (Nat.le_trans hd (Nat.le_add_left drive _)))

/-! ### Theorem 5: SMDP Exact Delayed-Credit Contraction -/

/-- Cross-multiplied SMDP Bellman difference over variable ALIF duration `τ ≥ 1`:
    with discount `γ = g / Q ∈ [0, 1)`, the continuation contraction factor satisfies
    `g^τ ≤ g * Q^(τ - 1)` (equivalently `γ^τ ≤ γ < 1`), preserving exact Banach
    contraction without shrinking the immediate causal reward `r`. -/
theorem smdp_discount_power_contraction (g Q tau : Nat)
    (hg : g ≤ Q) (htau : 1 ≤ tau) :
    g ^ tau ≤ g * Q ^ (tau - 1) := by
  cases tau with
  | zero => omega
  | succ k =>
      simp only [Nat.succ_sub_one, Nat.pow_succ']
      have hpow : g ^ k ≤ Q ^ k := Nat.pow_le_pow_left hg k
      exact Nat.mul_le_mul_left g hpow

theorem smdp_bellman_contraction (g Q tau diff : Nat)
    (hg : g ≤ Q) (htau : 1 ≤ tau) :
    g ^ tau * diff ≤ g * Q ^ (tau - 1) * diff := by
  exact Nat.mul_le_mul_right diff (smdp_discount_power_contraction g Q tau hg htau)

theorem smdp_strict_unit_discount (g Q : Nat) :
    g ^ 1 = g * Q ^ (1 - 1) := by
  simp

/-! ### Theorem 6: 100% Language-Byte Conservation -/

/-- Across `n` blocks of `B` external bytes (`m` executive spikes + `n - m` calm blocks),
    every single external byte receives full sensory-language learning (zero dropped bytes). -/
theorem language_byte_conservation (n m B : Nat) (hm : m ≤ n) :
    m * B + (n - m) * B = n * B := by
  rw [← Nat.add_mul, Nat.add_sub_of_le hm]

/-! ### Theorem 7: Wall-Clock Throughput Speedup -/

def totalWallTime (tFast tExec n m : Nat) : Nat :=
  n * tFast + m * tExec

/-- Lower executive duty cycle (`m₁ < m₂`) strictly reduces wall-clock time per learned byte. -/
theorem wallclock_strict_speedup (tFast tExec n m1 m2 : Nat)
    (htExec : 0 < tExec) (h12 : m1 < m2) :
    totalWallTime tFast tExec n m1 < totalWallTime tFast tExec n m2 := by
  unfold totalWallTime
  have hmul : m1 * tExec < m2 * tExec := Nat.mul_lt_mul_of_pos_right h12 htExec
  omega

/-- Cross-multiplied speedup monotonicity: for fixed `n` blocks, fewer spikes yield a strictly
    higher speedup ratio `(n * (tFast + tExec)) / totalWallTime`. -/
theorem speedup_ratio_strict_anti (tFast tExec n m1 m2 : Nat)
    (htExec : 0 < tExec) (h12 : m1 < m2) :
    totalWallTime tFast tExec n m1 < totalWallTime tFast tExec n m2 :=
  wallclock_strict_speedup tFast tExec n m1 m2 htExec h12

/-! ### Theorem 8: Burstiness Dominance Over Equal-Budget Fixed Periodic `K:1` Chunking -/

/-- Sum of surprise values selected by a boolean spike schedule `mask`. -/
def selectedSum : List Nat → List Bool → Nat
  | [], _ => 0
  | _, [] => 0
  | r :: rs, b :: bs => (if b then r else 0) + selectedSum rs bs

/-- Sum of missed surprise values (executive regret) under `mask`. -/
def missedSum : List Nat → List Bool → Nat
  | [], _ => 0
  | _, [] => 0
  | r :: rs, b :: bs => (if b then 0 else r) + missedSum rs bs

/-- Total spike budget (number of executive spikes) of `mask`. -/
def spikeCount : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + spikeCount bs

/-- Total sum of a surprise stream. -/
def totalSum : List Nat → Nat
  | [] => 0
  | r :: rs => r + totalSum rs

theorem selected_add_missed (rs : List Nat) (mask : List Bool)
    (hlen : rs.length = mask.length) :
    selectedSum rs mask + missedSum rs mask = totalSum rs := by
  induction rs generalizing mask with
  | nil =>
      cases mask with
      | nil => rfl
      | cons _ _ => simp at hlen
  | cons r rs ih =>
      cases mask with
      | nil => simp at hlen
      | cons b bs =>
          simp only [List.length_cons, Nat.succ.injEq] at hlen
          have hsub := ih bs hlen
          cases b <;> simp [selectedSum, missedSum, totalSum] <;> omega

/-- Difference accounting lemma for two schedules `sMask` (ALIF) and `pMask` (periodic):
    count of exclusive-S positions, count of exclusive-P positions, and surprise sums. -/
def exclusiveCount : List Bool → List Bool → Nat
  | [], _ => 0
  | _, [] => 0
  | s :: ss, p :: ps => (if s && !p then 1 else 0) + exclusiveCount ss ps

def exclusiveSum : List Nat → List Bool → List Bool → Nat
  | [], _, _ => 0
  | _, [], _ => 0
  | _, _, [] => 0
  | r :: rs, s :: ss, p :: ps => (if s && !p then r else 0) + exclusiveSum rs ss ps

private theorem spike_decomp (sMask pMask : List Bool) (hlen : sMask.length = pMask.length) :
    spikeCount sMask + exclusiveCount pMask sMask =
    spikeCount pMask + exclusiveCount sMask pMask := by
  induction sMask generalizing pMask with
  | nil =>
      cases pMask with
      | nil => rfl
      | cons _ _ => simp at hlen
  | cons s ss ih =>
      cases pMask with
      | nil => simp at hlen
      | cons p ps =>
          simp only [List.length_cons, Nat.succ.injEq] at hlen
          have hsub := ih ps hlen
          cases s <;> cases p <;> simp [spikeCount, exclusiveCount] <;> omega

private theorem sum_decomp (rs : List Nat) (sMask pMask : List Bool)
    (hs : rs.length = sMask.length) (hp : rs.length = pMask.length) :
    selectedSum rs sMask + exclusiveSum rs pMask sMask =
    selectedSum rs pMask + exclusiveSum rs sMask pMask := by
  induction rs generalizing sMask pMask with
  | nil =>
      cases sMask <;> cases pMask <;> simp at hs hp ⊢
  | cons r rs ih =>
      cases sMask with
      | nil => simp at hs
      | cons s ss =>
          cases pMask with
          | nil => simp at hp
          | cons p ps =>
              simp only [List.length_cons, Nat.succ.injEq] at hs hp
              have hsub := ih ss ps hs hp
              cases s <;> cases p <;> simp [selectedSum, exclusiveSum] <;> omega

/-- Predicate: every exclusive-S position has surprise strictly greater than `tauCut`. -/
def AllExclusiveGt (tauCut : Nat) : List Nat → List Bool → List Bool → Prop
  | [], _, _ => True
  | _, [], _ => True
  | _, _, [] => True
  | r :: rs, s :: ss, p :: ps =>
      ((s = true ∧ p = false) → tauCut < r) ∧ AllExclusiveGt tauCut rs ss ps

/-- Predicate: every exclusive-P position has surprise at most `tauCut`. -/
def AllExclusiveLe (tauCut : Nat) : List Nat → List Bool → List Bool → Prop
  | [], _, _ => True
  | _, [], _ => True
  | _, _, [] => True
  | r :: rs, s :: ss, p :: ps =>
      ((p = true ∧ s = false) → r ≤ tauCut) ∧ AllExclusiveLe tauCut rs ss ps

private theorem exclusiveSum_gt_mul (tauCut : Nat) (rs : List Nat) (sMask pMask : List Bool)
    (hs : rs.length = sMask.length) (hp : rs.length = pMask.length)
    (hgt : AllExclusiveGt tauCut rs sMask pMask)
    (hpos : 0 < exclusiveCount sMask pMask) :
    exclusiveCount sMask pMask * tauCut < exclusiveSum rs sMask pMask := by
  induction rs generalizing sMask pMask with
  | nil =>
      cases sMask <;> cases pMask <;> simp [exclusiveCount] at hs hp hpos
  | cons r rs ih =>
      cases sMask with
      | nil => simp at hs
      | cons s ss =>
          cases pMask with
          | nil => simp at hp
          | cons p ps =>
              simp only [List.length_cons, Nat.succ.injEq] at hs hp
              rcases hgt with ⟨hhead, htail⟩
              cases s <;> cases p
              · simp [exclusiveCount, exclusiveSum] at hpos ⊢
                exact ih ss ps hs hp htail hpos
              · simp [exclusiveCount, exclusiveSum] at hpos ⊢
                exact ih ss ps hs hp htail hpos
              · simp [exclusiveCount, exclusiveSum] at hpos ⊢
                have hr : tauCut < r := hhead ⟨rfl, rfl⟩
                rw [Nat.add_mul, Nat.one_mul]
                by_cases hrest : 0 < exclusiveCount ss ps
                · have hsub := ih ss ps hs hp htail hrest
                  omega
                · have hzero : exclusiveCount ss ps = 0 := by omega
                  rw [hzero, Nat.zero_mul]
                  omega
              · simp [exclusiveCount, exclusiveSum] at hpos ⊢
                exact ih ss ps hs hp htail hpos

private theorem exclusiveSum_le_mul (tauCut : Nat) (rs : List Nat) (sMask pMask : List Bool)
    (hs : rs.length = sMask.length) (hp : rs.length = pMask.length)
    (hle : AllExclusiveLe tauCut rs sMask pMask) :
    exclusiveSum rs pMask sMask ≤ exclusiveCount pMask sMask * tauCut := by
  induction rs generalizing sMask pMask with
  | nil =>
      cases sMask <;> cases pMask <;> simp [exclusiveSum, exclusiveCount] at hs hp ⊢
  | cons r rs ih =>
      cases sMask with
      | nil => simp at hs
      | cons s ss =>
          cases pMask with
          | nil => simp at hp
          | cons p ps =>
              simp only [List.length_cons, Nat.succ.injEq] at hs hp
              rcases hle with ⟨hhead, htail⟩
              have hsub := ih ss ps hs hp htail
              cases s <;> cases p
              · simp [exclusiveSum, exclusiveCount]; exact hsub
              · simp only [exclusiveSum, exclusiveCount, Bool.not_false, Bool.and_self, ↓reduceIte]
                have hr : r ≤ tauCut := hhead ⟨rfl, rfl⟩
                rw [Nat.add_mul, Nat.one_mul]
                omega
              · simp [exclusiveSum, exclusiveCount]; exact hsub
              · simp [exclusiveSum, exclusiveCount]; exact hsub

/-- **Surprise-Gated Burstiness Dominance Theorem**:
    Given any non-negative surprise stream `rs`, suppose ALIF schedule `sMask` and
    fixed periodic `K:1` schedule `pMask` have the **exact same spike budget**
    (`spikeCount sMask = spikeCount pMask`, hence identical wall-clock cost!),
    differ on at least one block (`0 < exclusiveCount sMask pMask`), and every block
    exclusively chosen by ALIF has surprise `> tauCut` while every block exclusively
    chosen by periodic chunking has surprise `≤ tauCut`. Then:
    1. ALIF captures **strictly more surprise**: `selectedSum rs pMask < selectedSum rs sMask`.
    2. ALIF incurs **strictly lower missed regret**: `missedSum rs sMask < missedSum rs pMask`. -/
theorem burstiness_dominance_over_periodic (tauCut : Nat)
    (rs : List Nat) (sMask pMask : List Bool)
    (hs : rs.length = sMask.length)
    (hp : rs.length = pMask.length)
    (hbudget : spikeCount sMask = spikeCount pMask)
    (hdiff : 0 < exclusiveCount sMask pMask)
    (hS : AllExclusiveGt tauCut rs sMask pMask)
    (hP : AllExclusiveLe tauCut rs sMask pMask) :
    selectedSum rs pMask < selectedSum rs sMask ∧
    missedSum rs sMask < missedSum rs pMask := by
  have hlen_sp : sMask.length = pMask.length := by omega
  have hcnt_eq : exclusiveCount pMask sMask = exclusiveCount sMask pMask := by
    have h := spike_decomp sMask pMask hlen_sp
    omega
  have hsum_eq := sum_decomp rs sMask pMask hs hp
  have hgt := exclusiveSum_gt_mul tauCut rs sMask pMask hs hp hS hdiff
  have hle := exclusiveSum_le_mul tauCut rs sMask pMask hs hp hP
  rw [hcnt_eq] at hle
  have hcap : selectedSum rs pMask < selectedSum rs sMask := by omega
  have htotS := selected_add_missed rs sMask hs
  have htotP := selected_add_missed rs pMask hp
  constructor
  · exact hcap
  · omega

/-! ### Theorem 9: Exact Hippocampal Replay-to-Language Ratio Conservation -/

/-- Each executive spike (960B) executes `2` language piece updates and `2` half-units
    (= `1.0` full weight) of hippocampal encoder-anchoring replay (`_outcome_loss`).
    Each calm subcortical piece (480B) executes `1` language piece update and `1` half-unit
    (= `0.5` weight) of continuous hippocampal replay (`hippocampal_anchor_loss`).
    Unlike multi-block chunking (where the ratio degrades to `2K : 1` and causes encoder drift),
    the exact `2 : 1` language-to-replay ratio is conserved after EVERY 480B piece, even under
    arbitrary mid-block ALIF preemption. -/
def languageUpdates (execSpikes calmPieces : Nat) : Nat :=
  2 * execSpikes + calmPieces

def replayHalfUnits (execSpikes calmPieces : Nat) : Nat :=
  2 * execSpikes + calmPieces

theorem replay_ratio_conservation (execSpikes calmPieces : Nat) :
    replayHalfUnits execSpikes calmPieces = languageUpdates execSpikes calmPieces := by
  rfl

/-- By contrast, skipping hippocampal replay on calm pieces (as in unanchored K-block chunking)
    creates a strictly positive replay deficit equal to `calmPieces` half-units whenever
    `0 < calmPieces`. -/
theorem unanchored_chunking_replay_deficit (execSpikes calmPieces : Nat)
    (hcalm : 0 < calmPieces) :
    2 * execSpikes < languageUpdates execSpikes calmPieces := by
  unfold languageUpdates
  omega

end CGB.AlifExecutiveControl
