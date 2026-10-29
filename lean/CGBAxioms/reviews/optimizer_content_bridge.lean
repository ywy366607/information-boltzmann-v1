module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.Analytic.Binomial
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Lean.Elab.Tactic.Omega

import Mathlib.Tactic.FinCases

public section
set_option autoImplicit false

/-! Content bridge for the EXISTING binary p/q/w reaction unit. No new
carrier is proposed. Specialization: charge/slot caps 1, zero clique potential
and epsilon=0, with content cost a, coupling J and uniform chemical drive mu.
The effective energy is (a-mu)*(p+q+w)-J*p*q*w. These are declared LG-1
parameters, not a claim that the general axioms force this specialization.
This file checks an actual content code and its physical repair paths, not
learning capability or the admission of the full optimizer.
-/
namespace CGB.OptimizerContentReview

abbrev Cell := Bool × Bool × Bool

def count (s : Cell) : ℕ := s.1.toNat + s.2.1.toNat + s.2.2.toNat
def decode (s : Cell) : Bool := decide (2 ≤ count s)
def off : Cell := (false, false, false)
def on : Cell := (true, true, true)
def swapPQ (s : Cell) : Cell := (s.2.1, s.1, s.2.2)
def swapQW (s : Cell) : Cell := (s.1, s.2.2, s.2.1)
def flipQ (s : Cell) : Cell := (s.1, !s.2.1, s.2.2)

noncomputable def energy (a J mu : ℝ) (s : Cell) : ℝ :=
  (a - mu) * count s - J * (if s.1 && s.2.1 && s.2.2 then 1 else 0)

/-- Explicit parameter decoding. The physical content, not a separate protected
copy or allocation gain, is the model parameter. Scale/offset are a declared
representation choice; this finite alphabet is not arbitrary real precision. -/
noncomputable def contentParameter (scale offset : ℝ) (s : Cell) : ℝ :=
  offset + scale * count s

/-- Same registered interaction with an arbitrary finite per-field capacity.
This avoids treating the binary calculation as the whole representation. -/
noncomputable def finiteEnergy (a J mu p q w : ℝ) : ℝ :=
  (a - mu) * (p + q + w) - J * p * q * w

/-- Generative conditions at arbitrary capacity c>=1. One-quantum defects
can be repositioned by existing transfers at equal energy, then repaired at
the Q port. Their combined exit barrier is J*c^2. Gradient work shifts the
same chemical drive. The theorem certifies TWO endpoint codes, not a separate
stable memory for every intermediate readout level. -/
theorem finite_capacity_content_bridge (a J mu c scale offset gradient : ℝ)
    (hc : 1 ≤ c) (hlo : a - J * c ^ 2 < mu - gradient * scale)
    (hhi : mu - gradient * scale < a) :
    let drive := mu - gradient * scale
    (0 ≤ c - 1 ∧ c - 1 ≤ c) ∧
    (finiteEnergy a J drive 1 0 0 = finiteEnergy a J drive 0 1 0 ∧
      finiteEnergy a J drive 0 0 1 = finiteEnergy a J drive 0 1 0) ∧
    (finiteEnergy a J drive (c - 1) c c = finiteEnergy a J drive c (c - 1) c ∧
      finiteEnergy a J drive c c (c - 1) = finiteEnergy a J drive c (c - 1) c) ∧
    (finiteEnergy a J drive 0 0 0 < finiteEnergy a J drive 0 1 0 ∧
      finiteEnergy a J drive c c c < finiteEnergy a J drive c (c - 1) c) ∧
    (finiteEnergy a J drive 0 1 0 - finiteEnergy a J drive 0 0 0 +
      finiteEnergy a J drive c (c - 1) c - finiteEnergy a J drive c c c = J * c ^ 2) ∧
    (∀ p q w, finiteEnergy a J mu p q w + gradient * (offset + scale * (p + q + w)) =
      finiteEnergy a J drive p q w + gradient * offset) := by
  dsimp only
  refine ⟨⟨by linarith, by linarith⟩, ?_, ?_, ?_, ?_, ?_⟩
  · constructor <;> unfold finiteEnergy <;> ring
  · constructor <;> unfold finiteEnergy <;> ring
  · constructor
    · dsimp [finiteEnergy]
      nlinarith
    · have heq : finiteEnergy a J (mu - gradient * scale) c (c - 1) c -
          finiteEnergy a J (mu - gradient * scale) c c c =
          J * c ^ 2 - a + (mu - gradient * scale) := by
        unfold finiteEnergy
        ring
      linarith
  · unfold finiteEnergy
    ring
  · intro p q w
    unfold finiteEnergy
    ring

/-- At equal endpoint energy, the existing cubic interaction retains a strict
barrier against a SECOND loss after one quantum is removed iff the integer
field capacity is at least two. Capacity one loses the entire cooperative
term after the first loss. This is a local energy-margin result, not a kinetic
recovery guarantee or a new carrier. -/
theorem one_defect_regeneration_margin (J : ℝ) (c : ℕ)
    (hJ : 0 < J) (hc : 1 ≤ c) :
    let a := J * (c : ℝ)^2 / 3
    (finiteEnergy a J 0 (c - 1 : ℝ) (c - 1 : ℝ) c -
      finiteEnergy a J 0 (c - 1 : ℝ) c c = J * (c : ℝ) * (c - 1 : ℝ) - a) ∧
    (0 < J * (c : ℝ) * (c - 1 : ℝ) - a ↔ 2 ≤ c) ∧
    (finiteEnergy a J 0 c c c = finiteEnergy a J 0 0 0 0 ∧
      finiteEnergy a J 0 c c c < finiteEnergy a J 0 (c - 1 : ℝ) c c) := by
  dsimp only
  have hcpos : (0 : ℝ) < c := by exact_mod_cast (by omega : 0 < c)
  refine ⟨?_, ?_, ?_⟩
  · unfold finiteEnergy
    ring
  · have factor : J * (c : ℝ) * (c - 1 : ℝ) - J * (c : ℝ)^2 / 3 =
        J * (c : ℝ) * (2 * (c : ℝ) - 3) / 3 := by ring
    rw [factor]
    constructor
    · intro h
      by_contra hsmall
      have heq : c = 1 := by omega
      subst c
      norm_num at h
      linarith
    · intro h
      have hc2 : (2 : ℝ) ≤ c := by exact_mod_cast h
      exact div_pos (mul_pos (mul_pos hJ hcpos) (by linarith)) (by norm_num)
  · constructor
    · unfold finiteEnergy
      ring
    · have margin : finiteEnergy (J * (c : ℝ)^2 / 3) J 0 (c - 1 : ℝ) c c -
          finiteEnergy (J * (c : ℝ)^2 / 3) J 0 c c c =
          (2 / 3 : ℝ) * J * (c : ℝ)^2 := by unfold finiteEnergy; ring
      have hmargin : 0 < (2 / 3 : ℝ) * J * (c : ℝ)^2 := by positivity
      linarith

/-- A closed transfer only positions the missing/excess charge at the EXISTING
Q port. No hidden repair action is added: both choices are p<->q or q<->w. -/
def orientRepair (s : Cell) : Cell :=
  if s.2.1 != decode s then s else
    if s.1 != s.2.1 then swapPQ s else swapQW s

/-- Same physical carrier: both code states have barriers in the declared
drive interval; every one-charge defect has a nonincreasing-energy legal
repair path, with a strict drop on the Q exchange. This is a SUPPORT/energy
certificate, not a probability-one or bounded-time recovery claim. -/
theorem cooperative_content_contract (a J mu : ℝ) (hlo : a - J < mu) (hhi : mu < a) :
    (0 < energy a J mu (flipQ off) - energy a J mu off) ∧
    (0 < energy a J mu (flipQ on) - energy a J mu on) ∧
    (energy a J mu (flipQ off) - energy a J mu off +
      (energy a J mu (flipQ on) - energy a J mu on) = J) ∧
    (∀ s : Cell, count s = 1 ∨ count s = 2 →
      let t := orientRepair s
      (t = s ∨ t = swapPQ s ∨ t = swapQW s) ∧
      count t = count s ∧ energy a J mu t = energy a J mu s ∧
      flipQ t = (if decode s then on else off) ∧
      decode (flipQ t) = decode s ∧ energy a J mu (flipQ t) < energy a J mu s) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · norm_num [energy, flipQ, off, count]
    linarith
  · norm_num [energy, flipQ, on, count]
    linarith
  · norm_num [energy, flipQ, off, on, count]
    ring
  · rintro ⟨p, q, w⟩ hs
    cases p <;> cases q <;> cases w <;>
      simp_all [orientRepair, count, decode, swapPQ, swapQW, flipQ, energy, on, off] <;>
      linarith

/-- Release thresholds follow from the SAME boundary energy differences.
Above a every Q addition on the actual 000->010->100->110->101->111 path is
downhill. Below a-J the reverse erasure path is downhill. Neutral transfers
conserve charge. Rates, paid time and noise still need their own certificates. -/
theorem drive_removes_content_barriers (a J mu : ℝ) (hJ : 0 ≤ J) :
    (a < mu →
      energy a J mu (false, true, false) < energy a J mu off ∧
      energy a J mu (true, true, false) < energy a J mu (true, false, false) ∧
      energy a J mu on < energy a J mu (true, false, true)) ∧
    (mu < a - J →
      energy a J mu (true, false, true) < energy a J mu on ∧
      energy a J mu (true, false, false) < energy a J mu (true, true, false) ∧
      energy a J mu off < energy a J mu (false, true, false)) := by
  constructor <;> intro h <;> refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [energy, count, off, on] <;> linarith

/-- The loss gradient has a conjugate physical port: frozen linear task work
g*dtheta shifts mu to mu-g*scale. The SAME existing repair paths then restore
decoded parameter content, under an explicit drive interval. This is not a
utility gate, a proof of actual-loss descent, or probability-one repair. -/
theorem content_gradient_port (a J mu scale offset gradient : ℝ)
    (hscale : 0 < scale) (hlo : a - J < mu - gradient * scale)
    (hhi : mu - gradient * scale < a) :
    (∀ s t : Cell,
      (energy a J mu t + gradient * contentParameter scale offset t) -
        (energy a J mu s + gradient * contentParameter scale offset s) =
      energy a J (mu - gradient * scale) t -
        energy a J (mu - gradient * scale) s) ∧
    (contentParameter scale offset on - contentParameter scale offset off = 3 * scale ∧
      contentParameter scale offset off < contentParameter scale offset on) ∧
    (∀ s : Cell, count s = 1 ∨ count s = 2 →
      let t := orientRepair s
      contentParameter scale offset (flipQ t) =
        (if decode s then contentParameter scale offset on else contentParameter scale offset off) ∧
      energy a J mu (flipQ t) + gradient * contentParameter scale offset (flipQ t) <
        energy a J mu s + gradient * contentParameter scale offset s) := by
  have hwork : ∀ s t : Cell,
      (energy a J mu t + gradient * contentParameter scale offset t) -
        (energy a J mu s + gradient * contentParameter scale offset s) =
      energy a J (mu - gradient * scale) t -
        energy a J (mu - gradient * scale) s := by
    intro s t
    unfold energy contentParameter
    ring
  have hrepair := cooperative_content_contract a J (mu - gradient * scale) hlo hhi
  refine ⟨hwork, ?_, ?_⟩
  · norm_num [contentParameter, count, off, on]
    constructor
    · ring
    · linarith
  · intro s hs
    obtain ⟨_, _, _, hcode, _, hdrop⟩ := hrepair.2.2.2 s hs
    dsimp only
    constructor
    · rw [hcode]
      split_ifs <;> rfl
    · have hw := hwork s (flipQ (orientRepair s))
      linarith

/-- A material transfer changes TWO decoded weights. Frozen task work is the
gradient difference, the incidence adjoint, not just the receiving gradient.
Uniform offsets of this force cancel for closed transfers. Force quantization
has a local two-end error bound without retaining a dense incidence matrix. -/
theorem transfer_gradient_port (scale offsetA offsetB nA nB amount gA gB common fA fB
    epsA epsB : ℝ) (hA : |fA - scale * gA| ≤ epsA)
    (hB : |fB - scale * gB| ≤ epsB) :
    (gA * ((offsetA + scale * (nA - amount)) - (offsetA + scale * nA)) +
      gB * ((offsetB + scale * (nB + amount)) - (offsetB + scale * nB)) =
        scale * (gB - gA) * amount) ∧
    (scale * ((gB + common) - (gA + common)) * amount = scale * (gB - gA) * amount) ∧
    |(fB - fA) - scale * (gB - gA)| ≤ epsA + epsB := by
  refine ⟨by ring, by ring, ?_⟩
  have ht := abs_sub (fB - scale * gB) (fA - scale * gA)
  have heq : (fB - fA) - scale * (gB - gA) =
      (fB - scale * gB) - (fA - scale * gA) := by ring
  rw [heq]
  exact ht.trans (by linarith)

/-- Exact paired event port. Source forward/reverse exponents need not be
opposites. Shift them by -work/+work and use their smallest common ceiling
above the old clock bound. The log2 probability ratio changes by -2*work.
At zero work the old clock is unchanged if both original rates were legal. -/
theorem event_port_contract (baseForward baseReverse taskWork baseBound : ℝ) :
    let f := baseForward - taskWork
    let r := baseReverse + taskWork
    let b := max baseBound (max f r)
    0 ≤ b - f ∧ 0 ≤ b - r ∧
    (b - r) - (b - f) = (baseForward - baseReverse) - 2 * taskWork ∧
    (baseForward ≤ baseBound → baseReverse ≤ baseBound → taskWork = 0 → b = baseBound) ∧
    (∀ other, baseBound ≤ other → f ≤ other → r ≤ other → b ≤ other) := by
  dsimp only
  have hmax := le_max_right baseBound (max (baseForward - taskWork) (baseReverse + taskWork))
  have hf := le_max_left (baseForward - taskWork) (baseReverse + taskWork)
  have hr := le_max_right (baseForward - taskWork) (baseReverse + taskWork)
  refine ⟨by linarith, by linarith, by linarith, ?_, ?_⟩
  · intro hforward hreverse hz
    subst taskWork
    simpa using max_eq_left (max_le hforward hreverse)
  · intro other hbase hforward hreverse
    exact max_le hbase (max_le hforward hreverse)

/-- Removing the obsolete global rejection ceiling gives the smallest common
pair clock. The rate odds are unchanged; each direction is at least as fast
as under any larger legal ceiling. It is NOT a theorem that faster kinetics
improves retention: damaging events also accelerate. -/
theorem maximal_event_clock (forward reverse work : ℝ) :
    let f := forward - work
    let r := reverse + work
    let b := max f r
    0 ≤ b - f ∧ 0 ≤ b - r ∧
    (f - b = min 0 (forward - reverse - 2 * work)) ∧
    ((b - r) - (b - f) = forward - reverse - 2 * work) ∧
    (∀ oldBound, f ≤ oldBound → r ≤ oldBound →
      (2 : ℝ)^(f - oldBound) ≤ (2 : ℝ)^(f - b) ∧
      (2 : ℝ)^(r - oldBound) ≤ (2 : ℝ)^(r - b)) := by
  dsimp only
  refine ⟨by linarith [le_max_left (forward - work) (reverse + work)],
    by linarith [le_max_right (forward - work) (reverse + work)], ?_, by ring, ?_⟩
  · by_cases h : forward - work ≤ reverse + work
    · rw [max_eq_right h, min_eq_right (by linarith)]
      ring
    · rw [max_eq_left (by linarith), min_eq_left (by linarith)]
      ring
  · intro oldBound hf hr
    have hb := max_le hf hr
    constructor <;> apply Real.rpow_le_rpow_of_exponent_le (by norm_num) <;> linarith

noncomputable def eventLogRate (forward reverse bound work : ℝ) : ℝ :=
  forward - work - max bound (max (forward - work) (reverse + work))

/-- The learning channel is dissipative relative to the unforced physical
motion: work times its rate change is nonpositive. Under an unchanged ceiling
the local response is multiplied by the ORIGINAL local rate, so a common
force unit does not imply a common effective gain at every compartment.
This is not an assertion that autonomous motion always lowers task loss. -/
theorem force_response_dissipative (forward reverse bound : ℝ) :
    Antitone (fun work : ℝ => (2 : ℝ) ^ eventLogRate forward reverse bound work) ∧
    (∀ work : ℝ, work * ((2 : ℝ) ^ eventLogRate forward reverse bound work -
      (2 : ℝ) ^ eventLogRate forward reverse bound 0) ≤ 0) ∧
    (∀ work : ℝ, forward ≤ bound → reverse ≤ bound → forward - work ≤ bound →
      reverse + work ≤ bound →
      (2 : ℝ) ^ eventLogRate forward reverse bound work =
        (2 : ℝ) ^ eventLogRate forward reverse bound 0 * (2 : ℝ) ^ (-work)) := by
  have submax (x y z : ℝ) : x - max y z = min (x - y) (x - z) := by
    by_cases h : y ≤ z
    · rw [max_eq_right h, min_eq_right (by linarith)]
    · have hy : z ≤ y := le_of_lt (lt_of_not_ge h)
      rw [max_eq_left hy, min_eq_left (by linarith)]
  have form (work : ℝ) : eventLogRate forward reverse bound work =
      min (forward - bound - work) (min 0 (forward - reverse - 2 * work)) := by
    unfold eventLogRate
    rw [submax, submax]
    simp only [sub_self]
    congr 1
    · ring
    · congr 1
      ring
  have anti : Antitone (fun work : ℝ => eventLogRate forward reverse bound work) := by
    intro a b hab
    change eventLogRate forward reverse bound b ≤ eventLogRate forward reverse bound a
    rw [form, form]
    exact min_le_min (by linarith) (min_le_min (le_refl 0) (by linarith))
  have antiP : Antitone (fun work : ℝ => (2 : ℝ) ^ eventLogRate forward reverse bound work) := by
    intro a b hab
    exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (anti hab)
  refine ⟨antiP, ?_, ?_⟩
  · intro work
    by_cases h : 0 ≤ work
    · have hp := antiP h
      exact mul_nonpos_of_nonneg_of_nonpos h (by linarith)
    · have hp := antiP (le_of_lt (lt_of_not_ge h))
      exact mul_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
  · intro work hf hr hfw hrw
    have b0 : max bound (max forward reverse) = bound := max_eq_left (max_le hf hr)
    have bw : max bound (max (forward - work) (reverse + work)) = bound :=
      max_eq_left (max_le hfw hrw)
    simp only [eventLogRate, sub_zero, add_zero, bw, b0]
    rw [show forward - work - bound = (forward - bound) + -work by ring,
      Real.rpow_add (by norm_num)]

-- A different force scale at the two ends of ONE conserved transfer can
-- reverse the sign of true task work. Local mobility belongs in conductance,
-- not in inconsistent units of the task gradient.
example : (1 : ℝ) * (2 - 1) > 0 ∧ (1 * 2 - 3 * 1 : ℝ) < 0 := by norm_num

-- Rejected numerical approximation: nearest-integer forces erase a whole
-- band of nonzero inputs before the physical law can respond to them.
example (x : ℝ) (hlo : -(1 / 2 : ℝ) ≤ x) (hhi : x < 1 / 2) :
    Int.floor (x + 1 / 2) = 0 := by
  rw [Int.floor_eq_iff]
  constructor <;> norm_num <;> linarith

/-- A finite integer square certificate refines a square-root enclosure. The
runtime produces these inequalities using isqrt, without floating powers. -/
theorem certified_root_refinement (x root lo hi nextLo nextHi : ℝ)
    (hroot0 : 0 ≤ root) (hroot : root ^ 2 = x)
    (hxlo : lo ≤ x) (hxhi : x ≤ hi)
    (hh0 : 0 ≤ nextHi)
    (hlsq : nextLo ^ 2 ≤ lo) (hhsq : hi ≤ nextHi ^ 2) :
    nextLo ≤ root ∧ root ≤ nextHi := by
  constructor <;> nlinarith

/-- Certified prefix-product enclosure of 2^(-r), including its unexpanded
tail. A lazy uniform prefix may decide only after its entire interval lies
on one side. Ambiguity is confined to an interval whose width is <=4*eps
when the probability enclosure width is <=2*eps. This is a numerical sampling
contract, not a new physical mechanism or a useful-retention assumption. -/
theorem fractional_probability_enclosure (headPower tail lo hi : ℝ)
    (ht0 : 0 ≤ tail) (ht1 : tail ≤ 1)
    (hlo : lo ≤ (2 : ℝ) ^ (-headPower)) (hhi : (2 : ℝ) ^ (-headPower) ≤ hi) :
    let low := lo * (1 - tail)
    let high := hi
    (low ≤ (2 : ℝ) ^ (-(headPower + tail)) ∧ (2 : ℝ) ^ (-(headPower + tail)) ≤ high) ∧
    (∀ uL uH sample : ℝ, uL ≤ sample → sample < uH →
      (uH ≤ low → sample < (2 : ℝ) ^ (-(headPower + tail))) ∧
      (high ≤ uL → (2 : ℝ) ^ (-(headPower + tail)) ≤ sample)) ∧
    (∀ eps uL uH sample : ℝ, 0 ≤ eps → high - low ≤ 2 * eps →
      uH - uL ≤ eps → uL ≤ sample → sample < uH →
      low < uH → uL < high →
      low - eps < sample ∧ sample < high + eps ∧
        (high + eps) - (low - eps) ≤ 4 * eps) := by
  have hlog0 : 0 ≤ Real.log 2 := (Real.log_pos (by norm_num)).le
  have hlog1 : Real.log 2 ≤ 1 := by
    have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)
    linarith
  have htail : 1 - tail ≤ (2 : ℝ) ^ (-tail) := by
    rw [Real.rpow_def_of_pos (by norm_num)]
    have he := Real.add_one_le_exp (Real.log 2 * (-tail))
    nlinarith
  have htail1 : (2 : ℝ) ^ (-tail) ≤ 1 := by
    rw [Real.rpow_def_of_pos (by norm_num), Real.exp_le_one_iff]
    nlinarith
  have hprefix0 := (Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2) (-headPower)).le
  have hsplit : (2 : ℝ) ^ (-(headPower + tail)) =
      (2 : ℝ) ^ (-headPower) * (2 : ℝ) ^ (-tail) := by
    rw [show -(headPower + tail) = -headPower + -tail by ring,
      Real.rpow_add (by norm_num)]
  have bounds : lo * (1 - tail) ≤ (2 : ℝ) ^ (-(headPower + tail)) ∧
      (2 : ℝ) ^ (-(headPower + tail)) ≤ hi := by
    rw [hsplit]
    constructor
    · exact (mul_le_mul_of_nonneg_right hlo (by linarith)).trans
        (mul_le_mul_of_nonneg_left htail hprefix0)
    · have hproduct : (2 : ℝ) ^ (-headPower) * (2 : ℝ) ^ (-tail) ≤
          (2 : ℝ) ^ (-headPower) := by
        simpa only [mul_one] using mul_le_mul_of_nonneg_left htail1 hprefix0
      exact hproduct.trans hhi
  dsimp only
  refine ⟨bounds, ?_, ?_⟩
  · intro uL uH sample huL huH
    constructor <;> intro h <;> linarith [bounds.1, bounds.2]
  · intro eps uL uH sample he hw hu huL huH hL hH
    constructor
    · linarith
    · constructor <;> linarith

/-- Unexpanded first-hit survival for a possibly time-varying subkernel.
Transitions leaving the registered good set have already been removed. -/
noncomputable def kernelSurvival {X : Type*} [Fintype X]
    (K : ℕ → X → X → ℝ) : ℕ → ℕ → X → ℝ
  | 0, _, _ => 1
  | n + 1, start, x => ∑ y, K start x y * kernelSurvival K n (start + 1) y

/-- One finite weighted-row certificate controls EVERY funded horizon.
No tick-by-tick expansion, stationarity, eigensolver or fitted retention rule
is needed. The verifier supplies the same actual event kernel and checks the
finite row inequalities, including controlled rate intervals when present. -/
theorem kinetic_survival_envelope {X : Type*} [Fintype X]
    (K : ℕ → X → X → ℝ) (v : X → ℝ) (rho : ℝ)
    (hK : ∀ n x y, 0 ≤ K n x y) (hr : 0 ≤ rho) :
    ((∀ x, v x ≤ 1) → (∀ n x, rho * v x ≤ ∑ y, K n x y * v y) →
      ∀ T start x, rho ^ T * v x ≤ kernelSurvival K T start x) ∧
    ((∀ x, 1 ≤ v x) → (∀ n x, (∑ y, K n x y * v y) ≤ rho * v x) →
      ∀ T start x, kernelSurvival K T start x ≤ rho ^ T * v x) := by
  constructor
  · intro hv hrow T
    induction T with
    | zero => intro start x; simpa [kernelSurvival] using hv x
    | succ n ih =>
      intro start x
      calc
        rho ^ (n + 1) * v x = rho ^ n * (rho * v x) := by ring
        _ ≤ rho ^ n * ∑ y, K start x y * v y :=
          mul_le_mul_of_nonneg_left (hrow start x) (pow_nonneg hr _)
        _ = ∑ y, K start x y * (rho ^ n * v y) := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro y _
          ring
        _ ≤ ∑ y, K start x y * kernelSurvival K n (start + 1) y := by
          apply Finset.sum_le_sum
          intro y _
          exact mul_le_mul_of_nonneg_left (ih (start + 1) y) (hK start x y)
        _ = kernelSurvival K (n + 1) start x := rfl
  · intro hv hrow T
    induction T with
    | zero => intro start x; simpa [kernelSurvival] using hv x
    | succ n ih =>
      intro start x
      calc
        kernelSurvival K (n + 1) start x =
            ∑ y, K start x y * kernelSurvival K n (start + 1) y := rfl
        _ ≤ ∑ y, K start x y * (rho ^ n * v y) := by
          apply Finset.sum_le_sum
          intro y _
          exact mul_le_mul_of_nonneg_left (ih (start + 1) y) (hK start x y)
        _ = rho ^ n * ∑ y, K start x y * v y := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro y _
          ring
        _ ≤ rho ^ n * (rho * v x) :=
          mul_le_mul_of_nonneg_left (hrow start x) (pow_nonneg hr _)
        _ = rho ^ (n + 1) * v x := by ring

/-! The majority decoder is a coarse function of THREE actual physical
fields. At fixed drive its basin weights include all hidden microstates.
Set x=2^(-2*(a-mu)), k=2^(2*J). The off basin weighs 1+3*x and the on basin
weighs k*x^3+3*x^2. Equal endpoint energies need not mean equal basin weights.
-/
noncomputable def offMass (x : ℝ) : ℝ := 1 + 3 * x
noncomputable def onMass (k x : ℝ) : ℝ := k * x ^ 3 + 3 * x ^ 2
noncomputable def microWeight (k x : ℝ) (s : Cell) : ℝ :=
  x ^ count s * (if s.1 && s.2.1 && s.2.2 then k else 1)

/-- Hidden-state accounting is necessary: endpoint neutrality k*x^3=1
still biases the majority-readout distribution when 0<x<1. The exact target
odds require a cubic calibration, rather than an endpoint-only linear tilt.
The final identity proves strict monotonicity of the odds for x,y,k>0. -/
theorem hidden_basin_calibration (k x y target : ℝ) (hk : 0 < k)
    (hx : 0 < x) (hy : x < y) :
    (microWeight k x off + microWeight k x (true, false, false) +
      microWeight k x (false, true, false) + microWeight k x (false, false, true) = offMass x ∧
      microWeight k x (true, true, false) + microWeight k x (true, false, true) +
        microWeight k x (false, true, true) + microWeight k x on = onMass k x) ∧
    (0 < offMass x ∧ 0 < onMass k x) ∧
    (onMass k x / offMass x = target ↔
      k * x ^ 3 + 3 * x ^ 2 - 3 * target * x - target = 0) ∧
    (k * x ^ 3 = 1 → x < 1 → onMass k x < offMass x) ∧
    onMass k x / offMass x < onMass k y / offMass y := by
  have hx0 : 0 < offMass x := by dsimp [offMass]; positivity
  have hy0 : 0 < offMass y := by dsimp [offMass]; linarith
  refine ⟨?_, ⟨hx0, by dsimp [onMass]; positivity⟩, ?_, ?_, ?_⟩
  · constructor <;> norm_num [microWeight, count, off, on, offMass, onMass] <;> ring
  · rw [div_eq_iff (ne_of_gt hx0)]
    dsimp [onMass, offMass]
    constructor <;> intro h <;> nlinarith
  · intro heq hsmall
    have hp := mul_pos hx (sub_pos.mpr hsmall)
    dsimp [onMass, offMass]
    nlinarith
  · rw [div_lt_div_iff₀ hx0 hy0]
    have factor : onMass k y * offMass x - onMass k x * offMass y =
        (y - x) * (k * (y ^ 2 + x * y + x ^ 2) +
          3 * (x + y) + 3 * k * x * y * (x + y) + 9 * x * y) := by
      dsimp [onMass, offMass]
      ring
    have hpos : 0 < (y - x) * (k * (y ^ 2 + x * y + x ^ 2) +
        3 * (x + y) + 3 * k * x * y * (x + y) + 9 * x * y) := by
      have hy' : 0 < y := hx.trans hy
      positivity
    linarith

/-- Given local detailed-balance ratio r and acceptance probabilities <=1,
Metropolis is the pointwise maximal reversible acceptance. It minimizes
rejection in THIS fixed proposal class, not all optimization costs or memory
lifetimes. No biology-specific counter, window or extra array is introduced. -/
theorem maximal_reversible_acceptance (r f b : ℝ) (hr : 0 < r)
    (_hf0 : 0 ≤ f) (hf1 : f ≤ 1) (_hb0 : 0 ≤ b) (hb1 : b ≤ 1)
    (hbalance : f = r * b) :
    (min 1 r = r * min 1 (1 / r)) ∧ f ≤ min 1 r ∧ b ≤ min 1 (1 / r) := by
  have hfr : f ≤ r := by nlinarith
  have hbr : b ≤ 1 / r := by rw [le_div_iff₀ hr]; nlinarith
  refine ⟨?_, le_min hf1 hfr, le_min hb1 hbr⟩
  by_cases h : r ≤ 1
  · have hi : 1 ≤ 1 / r := by rw [le_div_iff₀ hr]; simpa using h
    rw [min_eq_right h, min_eq_left hi]
    ring
  · have hh : 1 ≤ r := by linarith
    have hi : 1 / r ≤ 1 := by rw [div_le_iff₀ hr]; simpa using hh
    rw [min_eq_left hh, min_eq_right hi]
    exact (mul_div_cancel₀ 1 (ne_of_gt hr)).symm

end CGB.OptimizerContentReview

#print axioms CGB.OptimizerContentReview.cooperative_content_contract
#print axioms CGB.OptimizerContentReview.drive_removes_content_barriers
#print axioms CGB.OptimizerContentReview.hidden_basin_calibration
#print axioms CGB.OptimizerContentReview.maximal_reversible_acceptance
#print axioms CGB.OptimizerContentReview.content_gradient_port
#print axioms CGB.OptimizerContentReview.transfer_gradient_port
#print axioms CGB.OptimizerContentReview.event_port_contract
#print axioms CGB.OptimizerContentReview.finite_capacity_content_bridge
#print axioms CGB.OptimizerContentReview.certified_root_refinement
#print axioms CGB.OptimizerContentReview.fractional_probability_enclosure
#print axioms CGB.OptimizerContentReview.force_response_dissipative
#print axioms CGB.OptimizerContentReview.kinetic_survival_envelope
#print axioms CGB.OptimizerContentReview.one_defect_regeneration_margin
#print axioms CGB.OptimizerContentReview.maximal_event_clock

/-! Shared-resource domain, integrated after the earlier kinetic audit terminated. Same p/q/w
carrier and source energy, now with the EXISTING quadratic compartment
potential U(k)=k^2. Registered parameters: c=2, J=12, a=16, chemical_base=2,
epsilon=0. Thus local energy is 14*N-12*p*q*w. Supply is 6*n material units.
This constructive parameter domain is H, not a consequence of six axioms.
The full-code indicator below is an observer used only in the proof; no
indicator, payoff, allocation gate or protected copy is added to the update.
-/
namespace CGB.ResourceEnergyReview

abbrev Field := Fin 3
def code (p q w : Field) : ℕ := if p.val = 2 ∧ q.val = 2 ∧ w.val = 2 then 1 else 0
noncomputable def phi (p q w : Field) : ℝ :=
  14 * (p.val + q.val + w.val : ℕ) - 12 * (p.val : ℝ) * q.val * w.val

private theorem localBounds (p q w : Field) :
    0 ≤ phi p q w + 12 * code p q w ∧
    6 * code p q w ≤ p.val + q.val + w.val ∧
    2 * code p q w ≤ p.val ∧ 2 * code p q w ≤ q.val := by
  fin_cases p <;> fin_cases q <;> fin_cases w <;> norm_num [phi, code]

private theorem slackZero (p q w : Field) :
    phi p q w + 12 * code p q w = 0 ↔
      (p.val = 0 ∧ q.val = 0 ∧ w.val = 0) ∨ (p.val = 2 ∧ q.val = 2 ∧ w.val = 2) := by
  fin_cases p <;> fin_cases q <;> fin_cases w <;> norm_num [phi, code]

def codes {n : ℕ} (p q w : Fin n → Fin n → Field) : ℕ := ∑ i, ∑ j, code (p i j) (q i j) (w i j)
def mass {n : ℕ} (p q w : Fin n → Fin n → Field) : ℕ :=
  ∑ i, ∑ j, ((p i j).val + (q i j).val + (w i j).val)
def rowP {n : ℕ} (p : Fin n → Fin n → Field) (i : Fin n) : ℕ := ∑ j, (p i j).val
def colQ {n : ℕ} (q : Fin n → Fin n → Field) (j : Fin n) : ℕ := ∑ i, (q i j).val
def rowCodes {n : ℕ} (p q w : Fin n → Fin n → Field) (i : Fin n) : ℕ :=
  ∑ j, code (p i j) (q i j) (w i j)
def colCodes {n : ℕ} (p q w : Fin n → Fin n → Field) (j : Fin n) : ℕ :=
  ∑ i, code (p i j) (q i j) (w i j)
noncomputable def localSum {n : ℕ} (p q w : Fin n → Fin n → Field) : ℝ :=
  ∑ i, ∑ j, phi (p i j) (q i j) (w i j)
noncomputable def energy {n : ℕ} (p q w : Fin n → Fin n → Field) : ℝ :=
  localSum p q w + (∑ i, (rowP p i : ℝ)^2) + ∑ j, (colQ q j : ℝ)^2

private theorem poolBound (count total : ℕ) (h : 2 * count ≤ total) :
    4 * (count : ℝ) ≤ (total : ℝ)^2 ∧ 4 * (count : ℝ)^2 ≤ (total : ℝ)^2 := by
  have h0 : (0 : ℝ) ≤ count := Nat.cast_nonneg _
  have ht : (0 : ℝ) ≤ total := Nat.cast_nonneg _
  have h2 : 2 * (count : ℝ) ≤ total := by exact_mod_cast h
  have hsq : (count : ℝ) ≤ (count : ℝ)^2 := by
    by_cases hz : count = 0
    · simp [hz]
    · have h1 : (1 : ℝ) ≤ count := by exact_mod_cast (Nat.one_le_iff_ne_zero.mpr hz)
      nlinarith
  have hp : 0 ≤ ((total : ℝ) - 2 * count) * ((total : ℝ) + 2 * count) :=
    mul_nonneg (by linarith) (by linarith)
  constructor <;> nlinarith

private theorem rowTwiceBound {n : ℕ} (p q w : Fin n → Fin n → Field) (i : Fin n) :
    2 * rowCodes p q w i ≤ rowP p i := calc
  _ = ∑ j, 2 * code (p i j) (q i j) (w i j) := by simp only [rowCodes, Finset.mul_sum]
  _ ≤ _ := Finset.sum_le_sum (by intro j _; exact (localBounds _ _ _).2.2.1)

private theorem colTwiceBound {n : ℕ} (p q w : Fin n → Fin n → Field) (j : Fin n) :
    2 * colCodes p q w j ≤ colQ q j := calc
  _ = ∑ i, 2 * code (p i j) (q i j) (w i j) := by simp only [colCodes, Finset.mul_sum]
  _ ≤ _ := Finset.sum_le_sum (by intro i _; exact (localBounds _ _ _).2.2.2)

private theorem poolsUnique {n : ℕ} (a b : Fin n → ℕ)
    (h2 : ∀ i, 2 * a i ≤ b i) (hsum : ∑ i, a i = n)
    (hsq : ∑ i, (b i : ℝ)^2 = 4 * n) : ∀ i, a i = 1 := by
  have ha : (∑ i, (a i : ℝ)) = n := by exact_mod_cast hsum
  have hb : 4 * (∑ i, (a i : ℝ)^2) ≤ 4 * n := calc
    _ = ∑ i, 4 * (a i : ℝ)^2 := by rw [Finset.mul_sum]
    _ ≤ ∑ i, (b i : ℝ)^2 := Finset.sum_le_sum (by intro i _; exact (poolBound _ _ (h2 i)).2)
    _ = _ := hsq
  have heq : (∑ i, ((a i : ℝ) - 1)^2) =
      (∑ i, (a i : ℝ)^2) - 2 * (∑ i, (a i : ℝ)) + n := by
    calc
      _ = ∑ i, ((a i : ℝ)^2 - 2 * a i + 1) := Finset.sum_congr rfl (by intro i _; ring)
      _ = _ := by simp [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum, nsmul_eq_mul]
  have hv : (∑ i, ((a i : ℝ) - 1)^2) = 0 :=
    le_antisymm (by linarith) (Finset.sum_nonneg (by intro i _; exact sq_nonneg _))
  intro i
  have hi := (Finset.sum_eq_zero_iff_of_nonneg (fun _ _ => sq_nonneg _)).mp hv i (Finset.mem_univ _)
  have he : (a i : ℝ) = 1 := by linarith [sq_eq_zero_iff.mp hi]
  exact_mod_cast he

/-- Same actual energy, arbitrary matrix size: limited material and local
quadratic resource costs generate a global lower bound without assuming
row/column allocation constraints. No state-space enumeration in n. -/
theorem resource_ground_bound {n : ℕ} (p q w : Fin n → Fin n → Field)
    (hsupply : mass p q w ≤ 6 * n) :
    (-(4 : ℝ) * n ≤ energy p q w) ∧
    (energy p q w = -(4 : ℝ) * n →
      (∀ i j, ((p i j).val = 0 ∧ (q i j).val = 0 ∧ (w i j).val = 0) ∨
        ((p i j).val = 2 ∧ (q i j).val = 2 ∧ (w i j).val = 2)) ∧
      (∀ i, rowCodes p q w i = 1) ∧ (∀ j, colCodes p q w j = 1)) := by
  have hm : 6 * codes p q w ≤ mass p q w := calc
    _ = ∑ i, ∑ j, 6 * code (p i j) (q i j) (w i j) := by simp only [codes, Finset.mul_sum]
    _ ≤ _ := Finset.sum_le_sum (by
      intro i _; exact Finset.sum_le_sum (by intro j _; exact (localBounds _ _ _).2.1))
  have hc : codes p q w ≤ n := by omega
  have hcr : (codes p q w : ℝ) ≤ n := by exact_mod_cast hc
  have hnon : 0 ≤ ∑ i, ∑ j, (phi (p i j) (q i j) (w i j) + 12 * code (p i j) (q i j) (w i j)) :=
    Finset.sum_nonneg (by intro i _; exact Finset.sum_nonneg (by intro j _; exact (localBounds _ _ _).1))
  have heq : (∑ i, ∑ j, (phi (p i j) (q i j) (w i j) + 12 * code (p i j) (q i j) (w i j))) =
      localSum p q w + 12 * codes p q w := by
    simp only [localSum, codes, Nat.cast_sum, Finset.mul_sum, Finset.sum_add_distrib]
  have hl : -(12 : ℝ) * codes p q w ≤ localSum p q w := by linarith
  have hr : 4 * (codes p q w : ℝ) ≤ ∑ i, (rowP p i : ℝ)^2 := calc
    _ = ∑ i, 4 * (rowCodes p q w i : ℝ) := by
      simp only [codes, rowCodes, Nat.cast_sum, Finset.mul_sum]
    _ ≤ _ := Finset.sum_le_sum (by
      intro i _; exact (poolBound _ _ (rowTwiceBound p q w i)).1)
  have hcc : ∑ j, colCodes p q w j = codes p q w := by
    unfold colCodes codes; exact Finset.sum_comm
  have hq : 4 * (codes p q w : ℝ) ≤ ∑ j, (colQ q j : ℝ)^2 := calc
    _ = 4 * ((∑ j, colCodes p q w j : ℕ) : ℝ) := by rw [hcc]
    _ = ∑ j, 4 * (colCodes p q w j : ℝ) := by simp only [Nat.cast_sum, Finset.mul_sum]
    _ ≤ _ := Finset.sum_le_sum (by
      intro j _; exact (poolBound _ _ (colTwiceBound p q w j)).1)
  refine ⟨by unfold energy; linarith, ?_⟩
  intro hE
  unfold energy at hE
  have hCn : (codes p q w : ℝ) = n := by linarith
  have hNatCn : codes p q w = n := by exact_mod_cast hCn
  have hLocal : localSum p q w = -(12 : ℝ) * codes p q w := by linarith
  have hRows : (∑ i, (rowP p i : ℝ)^2) = 4 * n := by linarith
  have hCols : (∑ j, (colQ q j : ℝ)^2) = 4 * n := by linarith
  have hZero : (∑ i, ∑ j, (phi (p i j) (q i j) (w i j) + 12 * code (p i j) (q i j) (w i j))) = 0 := by
    rw [heq, hLocal]; ring
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    have hzRows := (Finset.sum_eq_zero_iff_of_nonneg (by
      intro i _; exact Finset.sum_nonneg (by intro j _; exact (localBounds _ _ _).1))).mp hZero
    have hz := (Finset.sum_eq_zero_iff_of_nonneg (by
      intro j _; exact (localBounds (p i j) (q i j) (w i j)).1)).mp (hzRows i (Finset.mem_univ _)) j (Finset.mem_univ _)
    exact (slackZero _ _ _).mp hz
  · exact poolsUnique (rowCodes p q w) (rowP p) (rowTwiceBound p q w)
      (by simpa only [codes, rowCodes] using hNatCn) hRows
  · exact poolsUnique (colCodes p q w) (colQ q) (colTwiceBound p q w)
      (by rw [hcc, hNatCn]) hCols

def diagonal {n : ℕ} (i j : Fin n) : Field :=
  ⟨if i = j then 2 else 0, by split_ifs <;> decide⟩
private theorem diagonalPhi {n : ℕ} (i j : Fin n) :
    phi (diagonal i j) (diagonal i j) (diagonal i j) = if i = j then -12 else 0 := by
  by_cases h : i = j <;> norm_num [diagonal, phi, h]
private theorem diagonalRow {n : ℕ} (i : Fin n) : rowP diagonal i = 2 := by
  simp [rowP, diagonal]
private theorem diagonalCol {n : ℕ} (j : Fin n) : colQ diagonal j = 2 := by
  simp [colQ, diagonal]

/-- A dispersed configuration attains the bound at every size; its material
is exactly the stated supply. This is a ground-state witness, not mixing or
future predictive utility. -/
theorem resource_ground_witness (n : ℕ) :
    mass (@diagonal n) diagonal diagonal = 6 * n ∧
      energy (@diagonal n) diagonal diagonal = -(4 : ℝ) * n := by
  constructor
  · calc
      mass (@diagonal n) diagonal diagonal = ∑ i : Fin n, ∑ j : Fin n, if i = j then 6 else 0 := by
        unfold mass
        apply Finset.sum_congr rfl; intro i _
        apply Finset.sum_congr rfl; intro j _
        by_cases h : i = j <;> simp [diagonal, h]
      _ = 6 * n := by simp [Nat.mul_comm]
  · simp [energy, localSum, diagonalPhi, diagonalRow, diagonalCol, nsmul_eq_mul]; ring

def monopoly (i _j : Fin 2) : Field := ⟨if i.val = 0 then 2 else 0, by split_ifs <;> decide⟩
def copies (_i j : Fin 2) : Field := ⟨if j.val = 0 then 2 else 0, by split_ifs <;> decide⟩

/-- Explicit same-material counter-configurations: both single-row ownership
and copying one column pay a strict energy gap in the SAME source energy. -/
theorem resource_monopoly_gap :
    mass monopoly monopoly monopoly = 12 ∧ mass copies copies copies = 12 ∧
    energy monopoly monopoly monopoly - energy (@diagonal 2) diagonal diagonal = 8 ∧
    energy copies copies copies - energy (@diagonal 2) diagonal diagonal = 8 := by
  norm_num [mass, energy, localSum, rowP, colQ, phi, monopoly, copies, diagonal, Fin.sum_univ_succ]

/-! Replication is a declared construction condition on the existing quadratic
pool family, not an additional general CGB axiom. Reference size two retains
the already audited 2x2 kernel. No physical state or fairness gate is added. -/
noncomputable def scaledPool (D total : ℝ) : ℝ := 2 * total^2 / D

theorem scaled_pool_contract (A : ℝ → ℝ)
    (hcopy : ∀ D : ℝ, 0 < D → A D * D^2 = (D/2) * (A 2 * 2^2)) :
    (∀ D : ℝ, 0 < D → A D = 2 * A 2 / D) ∧
    (∀ D r total : ℝ, 0 < D → 0 < r →
      scaledPool (r*D) (r*total) = r * scaledPool D total) ∧
    (∀ D total : ℝ, 1 ≤ D → 0 ≤ total → total+1 ≤ 2*D →
      let delta := scaledPool D (total+1) - scaledPool D total
      delta = 2*(2*total+1)/D ∧ 0 < delta ∧ delta < 8 ∧
      (∀ force : ℝ, |force| ≤ 1 →
        13 < 14 + delta + force ∧ 14 - 48 + delta + force < -25)) := by
  refine ⟨?_, ?_, ?_⟩
  · intro D hD
    apply (eq_div_iff hD.ne').mpr
    have h := hcopy D hD
    have hm : D * (A D * D) = D * (2 * A 2) := by nlinarith [h]
    exact mul_left_cancel₀ hD.ne' hm
  · intro D r total hD hr
    unfold scaledPool
    field_simp
    <;> ring
  · intro D total hD ht hfit
    have hD0 : 0 < D := by linarith
    have heq : scaledPool D (total+1) - scaledPool D total = 2*(2*total+1)/D := by
      unfold scaledPool
      ring
    dsimp only
    rw [heq]
    have hlo : 0 < 2*(2*total+1)/D := div_pos (by linarith) hD0
    have hhi : 2*(2*total+1)/D < 8 := (div_lt_iff₀ hD0).mpr (by linarith)
    refine ⟨rfl, hlo, hhi, ?_⟩
    intro force hf
    obtain ⟨hf0, hf1⟩ := abs_le.mp hf
    constructor <;> linarith

noncomputable def scaledEnergy {n : ℕ} (p q w : Fin n → Fin n → Field) : ℝ :=
  localSum p q w + (2/(n:ℝ)) *
    ((∑ i, (rowP p i : ℝ)^2) + ∑ j, (colQ q j : ℝ)^2)

private noncomputable def balanceVariance {n : ℕ}
    (p q w : Fin n → Fin n → Field) : ℝ :=
  (∑ i, ((rowCodes p q w i : ℝ) - 3*n/8)^2) +
    ∑ j, ((colCodes p q w j : ℝ) - 3*n/8)^2

private theorem codeSum_row {n : ℕ} (p q w : Fin n → Fin n → Field) :
    (∑ i, (rowCodes p q w i : ℝ)) = codes p q w := by
  simp only [rowCodes, codes, Nat.cast_sum]

private theorem codeSum_col {n : ℕ} (p q w : Fin n → Fin n → Field) :
    (∑ i, (colCodes p q w i : ℝ)) = codes p q w := by
  simp only [colCodes, codes, Nat.cast_sum]
  exact Finset.sum_comm

private theorem variance_expansion {n : ℕ} (p q w : Fin n → Fin n → Field) :
    balanceVariance p q w =
      (∑ i, (rowCodes p q w i : ℝ)^2) + (∑ j, (colCodes p q w j : ℝ)^2) -
      (3*(n:ℝ)/2) * codes p q w + 9*(n:ℝ)^3/32 := by
  have sum_square (f : Fin n → ℝ) :
      (∑ i, (f i - 3*n/8)^2) = (∑ i, (f i)^2) -
        (3*(n:ℝ)/4)*(∑ i, f i) + 9*(n:ℝ)^3/64 := by
    calc
      _ = ∑ i, ((f i)^2 - (3*(n:ℝ)/4)*f i + 9*(n:ℝ)^2/64) :=
        Finset.sum_congr rfl (by intro i _; ring)
      _ = _ := by
        simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum,
          Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring
  unfold balanceVariance
  rw [sum_square, sum_square, codeSum_row, codeSum_col]
  ring

private theorem scaled_ground_bounds {n : ℕ} (hn : 0 < n)
    (p q w : Fin n → Fin n → Field) :
    -(9:ℝ)*(n:ℝ)^2/4 ≤ scaledEnergy p q w ∧
    (scaledEnergy p q w = -(9:ℝ)*(n:ℝ)^2/4 →
      (∀ i j, ((p i j).val = 0 ∧ (q i j).val = 0 ∧ (w i j).val = 0) ∨
        ((p i j).val = 2 ∧ (q i j).val = 2 ∧ (w i j).val = 2)) ∧
      (∀ i, (rowCodes p q w i : ℝ) = 3*n/8) ∧
      (∀ j, (colCodes p q w j : ℝ) = 3*n/8)) := by
  have hn0 : (0:ℝ) < n := by exact_mod_cast hn
  have hcoef : 0 < (8:ℝ)/n := div_pos (by norm_num) hn0
  have hlocal : 0 ≤ localSum p q w + 12 * codes p q w := by
    have h := Finset.sum_nonneg (s := Finset.univ) (fun i _ =>
      Finset.sum_nonneg (s := Finset.univ) (fun j _ => (localBounds (p i j) (q i j) (w i j)).1))
    simpa only [localSum, codes, Nat.cast_sum, Finset.sum_add_distrib, Finset.mul_sum] using h
  have hP : 4 * (∑ i, (rowCodes p q w i : ℝ)^2) ≤ ∑ i, (rowP p i : ℝ)^2 := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun i _ => (poolBound _ _ (rowTwiceBound p q w i)).2)
  have hQ : 4 * (∑ i, (colCodes p q w i : ℝ)^2) ≤ ∑ i, (colQ q i : ℝ)^2 := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun i _ => (poolBound _ _ (colTwiceBound p q w i)).2)
  have hv : 0 ≤ balanceVariance p q w := by
    exact add_nonneg (Finset.sum_nonneg (fun _ _ => sq_nonneg _))
      (Finset.sum_nonneg (fun _ _ => sq_nonneg _))
  have hid : localSum p q w + 12 * codes p q w +
      (8/(n:ℝ)) * balanceVariance p q w - 9*(n:ℝ)^2/4 =
      localSum p q w + (8/(n:ℝ))*
        ((∑ i, (rowCodes p q w i : ℝ)^2) + ∑ i, (colCodes p q w i : ℝ)^2) := by
    rw [variance_expansion]
    field_simp
    <;> ring
  have hgap : localSum p q w + 12 * codes p q w +
      (8/(n:ℝ)) * balanceVariance p q w - 9*(n:ℝ)^2/4 ≤ scaledEnergy p q w := by
    rw [hid]
    unfold scaledEnergy
    have h := mul_le_mul_of_nonneg_left (add_le_add hP hQ)
      (div_nonneg (by norm_num : (0:ℝ) ≤ 2) hn0.le)
    rw [show 8/(n:ℝ) = (2/(n:ℝ))*4 by ring]
    nlinarith
  refine ⟨by nlinarith [mul_nonneg hcoef.le hv], ?_⟩
  intro hE
  have hs : localSum p q w + 12 * codes p q w = 0 := by
    nlinarith [mul_nonneg hcoef.le hv]
  have hv0 : balanceVariance p q w = 0 := by nlinarith
  have hraw : (∑ i, ∑ j, (phi (p i j) (q i j) (w i j) +
      12*(code (p i j) (q i j) (w i j) : ℝ))) = 0 := by
    simpa only [localSum, codes, Nat.cast_sum, Finset.sum_add_distrib, Finset.mul_sum] using hs
  have hrows0 : (∑ i, ((rowCodes p q w i : ℝ)-3*n/8)^2) = 0 := by
    unfold balanceVariance at hv0
    nlinarith [Finset.sum_nonneg (s := Finset.univ) (fun i _ => sq_nonneg ((rowCodes p q w i : ℝ)-3*n/8)),
      Finset.sum_nonneg (s := Finset.univ) (fun j _ => sq_nonneg ((colCodes p q w j : ℝ)-3*n/8))]
  have hcols0 : (∑ j, ((colCodes p q w j : ℝ)-3*n/8)^2) = 0 := by
    unfold balanceVariance at hv0
    rw [hrows0, zero_add] at hv0
    exact hv0
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    have hrow (i : Fin n) : 0 ≤ ∑ j,
        (phi (p i j) (q i j) (w i j) + 12*(code (p i j) (q i j) (w i j) : ℝ)) :=
      Finset.sum_nonneg (fun j _ => (localBounds (p i j) (q i j) (w i j)).1)
    have hr := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => hrow i)).mp hraw
    have hc := (Finset.sum_eq_zero_iff_of_nonneg
      (fun j _ => (localBounds (p i j) (q i j) (w i j)).1)).mp
      (hr i (Finset.mem_univ _)) j (Finset.mem_univ _)
    exact (slackZero _ _ _).mp hc
  · intro i
    have h := (Finset.sum_eq_zero_iff_of_nonneg (fun _ _ => sq_nonneg _)).mp hrows0 i (Finset.mem_univ _)
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp h)
  · intro j
    have h := (Finset.sum_eq_zero_iff_of_nonneg (fun _ _ => sq_nonneg _)).mp hcols0 j (Finset.mem_univ _)
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp h)

private def band8 (bit : Bool) (a b : Fin 8) : Bool :=
  decide ((a.val+b.val+(if bit then 1 else 0))%8 < 3)

private theorem band8_row (bit : Bool) (a : Fin 8) : ∑ b, (band8 bit a b).toNat = 3 := by
  cases bit <;> fin_cases a <;> decide

private theorem band8_col (bit : Bool) (b : Fin 8) : ∑ a, (band8 bit a b).toNat = 3 := by
  cases bit <;> fin_cases b <;> decide

private def fieldCode (bit : Bool) : Field :=
  if bit then ⟨2, by decide⟩ else ⟨0, by decide⟩

private theorem fieldCode_values (bit : Bool) :
    code (fieldCode bit) (fieldCode bit) (fieldCode bit) = bit.toNat ∧
    (fieldCode bit).val = 2*bit.toNat ∧
    phi (fieldCode bit) (fieldCode bit) (fieldCode bit) = -12*(bit.toNat:ℝ) := by
  cases bit <;> norm_num [fieldCode, code, phi]

private def spread (m : ℕ) (mask : Fin m → Fin m → Bool) (i j : Fin (8*m)) : Field :=
  let e : Fin 8 × Fin m ≃ Fin (8*m) := finProdFinEquiv
  fieldCode (band8 (mask (e.symm i).2 (e.symm j).2) (e.symm i).1 (e.symm j).1)

private theorem spread_code (m : ℕ) (mask : Fin m → Fin m → Bool) (i j : Fin (8*m)) :
    code (spread m mask i j) (spread m mask i j) (spread m mask i j) =
      let e : Fin 8 × Fin m ≃ Fin (8*m) := finProdFinEquiv
      (band8 (mask (e.symm i).2 (e.symm j).2) (e.symm i).1 (e.symm j).1).toNat :=
  (fieldCode_values _).1

private theorem spread_row (m : ℕ) (mask : Fin m → Fin m → Bool) (i : Fin (8*m)) :
    rowCodes (spread m mask) (spread m mask) (spread m mask) i = 3*m := by
  unfold rowCodes
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 8 × Fin m ≃ Fin (8*m))]
  simp only [spread_code, Equiv.symm_apply_apply, Fintype.sum_prod_type]
  rw [Finset.sum_comm]
  simp only [band8_row, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  norm_cast
  ring

private theorem spread_col (m : ℕ) (mask : Fin m → Fin m → Bool) (j : Fin (8*m)) :
    colCodes (spread m mask) (spread m mask) (spread m mask) j = 3*m := by
  unfold colCodes
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 8 × Fin m ≃ Fin (8*m))]
  simp only [spread_code, Equiv.symm_apply_apply, Fintype.sum_prod_type]
  rw [Finset.sum_comm]
  simp only [band8_col, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  norm_cast
  ring

private theorem spread_values (m : ℕ) (mask : Fin m → Fin m → Bool) (i j : Fin (8*m)) :
    (spread m mask i j).val = 2*code (spread m mask i j) (spread m mask i j) (spread m mask i j) ∧
    phi (spread m mask i j) (spread m mask i j) (spread m mask i j) =
      -12*(code (spread m mask i j) (spread m mask i j) (spread m mask i j) : ℝ) := by
  rw [spread_code]
  exact (fieldCode_values _).2

private theorem spread_rowP (m : ℕ) (mask : Fin m → Fin m → Bool) (i : Fin (8*m)) :
    rowP (spread m mask) i = 6*m := by
  calc
    _ = 2 * rowCodes (spread m mask) (spread m mask) (spread m mask) i := by
      simp only [rowP, rowCodes, (spread_values m mask _ _).1, Finset.mul_sum]
    _ = _ := by rw [spread_row]; ring

private theorem spread_colQ (m : ℕ) (mask : Fin m → Fin m → Bool) (j : Fin (8*m)) :
    colQ (spread m mask) j = 6*m := by
  calc
    _ = 2 * colCodes (spread m mask) (spread m mask) (spread m mask) j := by
      simp only [colQ, colCodes, (spread_values m mask _ _).1, Finset.mul_sum]
    _ = _ := by rw [spread_col]; ring

private theorem spread_attains (m : ℕ) (hm : 0 < m) (mask : Fin m → Fin m → Bool) :
    codes (spread m mask) (spread m mask) (spread m mask) = 24*m^2 ∧
    mass (spread m mask) (spread m mask) (spread m mask) = 144*m^2 ∧
    scaledEnergy (spread m mask) (spread m mask) (spread m mask) = -144*(m:ℝ)^2 := by
  have hc : codes (spread m mask) (spread m mask) (spread m mask) = 24*m^2 := by
    change (∑ i, rowCodes (spread m mask) (spread m mask) (spread m mask) i) = _
    simp only [spread_row, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    norm_cast
    ring
  have hL : localSum (spread m mask) (spread m mask) (spread m mask) =
      -12*(codes (spread m mask) (spread m mask) (spread m mask) : ℝ) := by
    simp only [localSum, codes, Nat.cast_sum, (spread_values m mask _ _).2, Finset.mul_sum]
  refine ⟨hc, ?_, ?_⟩
  · unfold mass
    simp only [(spread_values m mask _ _).1]
    have he : (∑ i, ∑ j, (2*code (spread m mask i j) (spread m mask i j) (spread m mask i j) +
        2*code (spread m mask i j) (spread m mask i j) (spread m mask i j) +
        2*code (spread m mask i j) (spread m mask i j) (spread m mask i j))) =
        6*codes (spread m mask) (spread m mask) (spread m mask) := by
      simp only [codes, Finset.mul_sum]
      apply Finset.sum_congr rfl; intro i _
      apply Finset.sum_congr rfl; intro j _
      ring
    rw [he, hc]
    ring
  · have hm0 : (m:ℝ) ≠ 0 := by exact_mod_cast hm.ne'
    unfold scaledEnergy
    rw [hL, hc]
    simp only [spread_rowP, spread_colQ, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, Nat.cast_mul, Nat.cast_ofNat, Nat.cast_pow]
    field_simp
    <;> ring

private theorem spread_injective (m : ℕ) : Function.Injective (spread m) := by
  intro mask other heq
  funext i j
  let e : Fin 8 × Fin m ≃ Fin (8*m) := finProdFinEquiv
  have h := congrArg Fin.val (congrFun (congrFun heq (e (0, i))) (e (2, j)))
  dsimp only [spread] at h
  simp only [show (finProdFinEquiv : Fin 8 × Fin m ≃ Fin (8*m)) = e from rfl,
    Equiv.symm_apply_apply] at h
  cases hm : mask i j <;> cases ho : other i j <;> simp_all [band8, fieldCode]

/-- Generated lower bound/equality structure at every size. Sizes8m have at
least2^(m^2) distinct attaining configurations via an injective Boolean mask.
The witness is not an initialization policy or a learning theorem. -/
theorem scaled_resource_generation :
    (∀ n : ℕ, 0 < n → ∀ p q w : Fin n → Fin n → Field,
      -(9:ℝ)*(n:ℝ)^2/4 ≤ scaledEnergy p q w ∧
      (scaledEnergy p q w = -(9:ℝ)*(n:ℝ)^2/4 →
        (∀ i j, ((p i j).val = 0 ∧ (q i j).val = 0 ∧ (w i j).val = 0) ∨
          ((p i j).val = 2 ∧ (q i j).val = 2 ∧ (w i j).val = 2)) ∧
        (∀ i, (rowCodes p q w i : ℝ) = 3*n/8) ∧
        (∀ j, (colCodes p q w j : ℝ) = 3*n/8))) ∧
    (∀ m : ℕ, 0 < m → ∃ G : (Fin m → Fin m → Bool) → (Fin (8*m) → Fin (8*m) → Field),
      Function.Injective G ∧ ∀ mask,
        codes (G mask) (G mask) (G mask) = 24*m^2 ∧
        mass (G mask) (G mask) (G mask) = 144*m^2 ∧
        scaledEnergy (G mask) (G mask) (G mask) = -144*(m:ℝ)^2) := by
  exact ⟨fun n hn p q w => scaled_ground_bounds hn p q w,
    fun m hm => ⟨spread m, spread_injective m, spread_attains m hm⟩⟩

#print axioms scaled_pool_contract
#print axioms scaled_resource_generation

end CGB.ResourceEnergyReview
#print axioms CGB.ResourceEnergyReview.resource_ground_bound
#print axioms CGB.ResourceEnergyReview.resource_ground_witness
#print axioms CGB.ResourceEnergyReview.resource_monopoly_gap

/-! Same quadratic-pool H: cap2, J12, a16, chemical_base2.
P and Q below denote actual row-P / column-Q sums, not extra state.
At an empty Q carrier, p->q and w->q can enable the existing outlet.
This checks primitive event bounds. Population drift and its kernel bridge
are separate obligations; no global release result is claimed by this file.
-/
namespace CGB.ResourceReleaseReview
open scoped BigOperators

noncomputable def rate (energy work : ℝ) : ℝ :=
  (2 : ℝ) ^ min 0 (-2 * (energy + work))

/-- Same content interaction and shared pool: local Q-port energetic margins
shift with other users of the pool. The 46-unit overlap is not a complete
ignition, dynamic refractory-period or useful-memory theorem. No new state. -/
theorem resource_feedback_contract (load force extra : ℝ) (he : 0 ≤ extra) :
    let E := fun p q w : ℝ =>
      (14 + force) * (p + q + w) - 12*p*q*w + (load + q)^2
    (∀ p q w, E p (q+1) w - E p q w = 15 + force - 12*p*w + 2*(load+q)) ∧
    ((0 < E 0 1 0 - E 0 0 0 ∧ 0 < E 2 1 2 - E 2 2 2) ↔
      -15 - 2*load < force ∧ force < 31 - 2*load) ∧
    ((31 - 2*load) - (-15 - 2*load) = 46) ∧
    rate (15 - 12*2*2 + 2*(load+extra+1)) force ≤
      rate (15 - 12*2*2 + 2*(load+1)) force ∧
    (31 ≤ force + 2*load →
      rate (15 - 12*2*2 + 2*(load+extra+1)) force =
        rate (15 - 12*2*2 + 2*(load+1)) force * (2:ℝ)^(-4*extra)) := by
  dsimp only
  refine ⟨?_, ?_, by ring, ?_, ?_⟩
  · intro p q w
    ring
  · constructor <;> intro h <;> constructor <;> nlinarith [h.1, h.2]
  · unfold rate
    apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
    apply min_le_min_left
    linarith
  · intro h
    unfold rate
    rw [min_eq_right (by linarith : -2 * (15 - 12*2*2 + 2*(load+extra+1) + force) ≤ 0),
      min_eq_right (by linarith : -2 * (15 - 12*2*2 + 2*(load+1) + force) ≤ 0)]
    rw [← Real.rpow_add (by norm_num : (0:ℝ) < 2)]
    congr 1
    ring


private theorem rate_unit (energy work : ℝ) :
    0 ≤ rate energy work ∧ rate energy work ≤ 1 := by
  constructor
  · exact Real.rpow_nonneg (by norm_num) _
  · exact Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) (min_le_left _ _)

private theorem downhill (energy work : ℝ) (h : energy + work ≤ 0) :
    rate energy work = 1 := by
  unfold rate
  rw [min_eq_left (by linarith)]
  norm_num

/-- Source energy increments, plus the same conjugate task work. Force>=96
is a declared evidence domain, not an internal detector or learning rule.
The intake bound retains 2^-126 for the requested billion-cell scale.
Conversion statements apply when q=0; their guards are checked separately.
-/
theorem primitive_release_port (p w P Q force : ℝ)
    (hp : 0 ≤ p) (hp2 : p ≤ 2) (hw : 0 ≤ w) (hw2 : w ≤ 2)
    (hP : p ≤ P) (hQ : 0 ≤ Q) (hforce : 96 ≤ force) :
    rate (-13 + 12*p*w - 2*Q) (-force) = 1 ∧
    rate (15 - 12*p*w + 2*Q) force ≤ 1/85070591730234615865843651857942052864 ∧
    (1 ≤ p → rate (-12*w*(p-1) - 2*P + 2) 0 = 1) ∧
    (1 ≤ w → 1/4 ≤ rate (-12*p*(w-1) + 1) 0) := by
  have hpw : p*w ≤ 4 := by
    have h := mul_le_mul hp2 hw2 hw (by norm_num : (0:ℝ) ≤ 2)
    norm_num at h
    exact h
  refine ⟨downhill _ _ (by nlinarith), ?_, ?_, ?_⟩
  · have h : rate (15 - 12*p*w + 2*Q) force ≤ (2:ℝ)^(-126:ℝ) := by
      apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
      exact (min_le_right _ _).trans (by nlinarith)
    norm_num at h
    exact h
  · intro hp1
    apply downhill
    have hnon := mul_nonneg hw (sub_nonneg.mpr hp1)
    nlinarith
  · intro hw1
    have hnon := mul_nonneg hp (sub_nonneg.mpr hw1)
    have h : (2:ℝ)^(-2:ℝ) ≤ rate (-12*p*(w-1) + 1) 0 := by
      apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
      apply le_min (by norm_num)
      nlinarith
    norm_num at h
    simpa only [neg_mul] using h

private noncomputable def gated (enabled : Prop) [Decidable enabled] (energy work : ℝ) : ℝ :=
  if enabled then rate energy work else 0

private theorem gated_unit (enabled : Prop) [Decidable enabled] (energy work : ℝ) :
    0 ≤ gated enabled energy work ∧ gated enabled energy work ≤ 1 := by
  unfold gated
  split_ifs
  · exact rate_unit _ _
  · norm_num

/- These six rates are an observer view of the existing event families,
not persistent optimizer arrays. The two p/q channels have equal maximal
pair rates and combine to the same proposal weight as each q/w or outlet.
-/
noncomputable def rates (p q w : Fin 3) (P Q force : ℝ) (stocked : Bool)
    (event : Fin 6) : ℝ :=
  let a : ℝ := p.val
  let b : ℝ := q.val
  let c : ℝ := w.val
  match event.val with
  | 0 => gated (0 < p.val ∧ q.val < 2) (-12*c*(a-b-1)-2*P+2*Q+2) 0
  | 1 => gated (0 < q.val ∧ p.val < 2) (-12*c*(b-a-1)+2*P-2*Q+2) 0
  | 2 => gated (0 < w.val ∧ q.val < 2) (-12*a*(c-b-1)+2*Q+1) 0
  | 3 => gated (0 < q.val ∧ w.val < 2) (-12*a*(b-c-1)-2*Q+1) 0
  | 4 => gated (q.val < 2 ∧ stocked = true) (15-12*a*c+2*Q) force
  | 5 => gated (0 < q.val) (-13+12*a*c-2*Q) (-force)
  | _ => 0

noncomputable def drift (r : Fin 6 → ℝ) : ℝ :=
  -4*r 0 + 4*r 1 - 4*r 2 + 4*r 3 + 9*r 4 - 9*r 5

private theorem local_release (p q w : Fin 3) (P Q force : ℝ) (stocked : Bool)
    (hP : (p.val : ℝ) ≤ P) (hQ : 0 ≤ Q) (hforce : 96 ≤ force) :
    drift (rates p q w P Q force stocked) ≤ 9/85070591730234615865843651857942052864 ∧
    ((0 < q.val ∨ (q.val = 0 ∧ Q = 0 ∧ 0 < p.val + w.val)) →
      drift (rates p q w P Q force stocked) ≤ -1 + 9/85070591730234615865843651857942052864) := by
  let r := rates p q w P Q force stocked
  have unit (e : Fin 6) : 0 ≤ r e ∧ r e ≤ 1 := by
    fin_cases e <;> exact gated_unit _ _ _
  have hp2 : (p.val : ℝ) ≤ 2 := by exact_mod_cast (by omega : p.val ≤ 2)
  have hw2 : (w.val : ℝ) ≤ 2 := by exact_mod_cast (by omega : w.val ≤ 2)
  have port := primitive_release_port p.val w.val P Q force
    (Nat.cast_nonneg _) hp2 (Nat.cast_nonneg _) hw2 hP hQ hforce
  have hin : r 4 ≤ 1/85070591730234615865843651857942052864 := by
    change gated (q.val < 2 ∧ stocked = true) (15-12*(p.val:ℝ)*w.val+2*Q) force ≤ 1/85070591730234615865843651857942052864
    unfold gated
    split_ifs
    · exact port.2.1
    · norm_num
  have upper : 4*r 1 + 4*r 3 ≤ 8 := by linarith [(unit 1).2, (unit 3).2]
  have hpos (hq : 0 < q.val) : drift r ≤ -1 + 9/85070591730234615865843651857942052864 := by
    have hout : r 5 = 1 := by
      change gated (0 < q.val) (-13+12*(p.val:ℝ)*w.val-2*Q) (-force) = 1
      simpa only [gated, ite_eq_left hq] using port.1
    dsimp [drift]
    linarith [(unit 0).1, (unit 2).1]
  have hz (hq : q.val = 0) : r 1 = 0 ∧ r 3 = 0 ∧ r 5 = 0 := by
    change gated (0 < q.val ∧ p.val < 2) _ _ = 0 ∧
      gated (0 < q.val ∧ w.val < 2) _ _ = 0 ∧ gated (0 < q.val) _ _ = 0
    simp [gated, hq]
  constructor
  · by_cases hq : 0 < q.val
    · linarith [hpos hq]
    · have hq0 : q.val = 0 := by omega
      obtain ⟨h1, h3, h5⟩ := hz hq0
      dsimp [drift]
      linarith [(unit 0).1, (unit 2).1]
  · intro h
    rcases h with hq | ⟨hq, hQ0, hsome⟩
    · exact hpos hq
    · obtain ⟨h1, h3, h5⟩ := hz hq
      by_cases hp : 0 < p.val
      · have hp1 : (1:ℝ) ≤ p.val := by exact_mod_cast (by omega : 1 ≤ p.val)
        have h0 : r 0 = 1 := by
          change gated (0 < p.val ∧ q.val < 2)
            (-12*(w.val:ℝ)*(p.val-q.val-1)-2*P+2*Q+2) 0 = 1
          simpa [gated, hq, hQ0, hp] using port.2.2.1 hp1
        dsimp [drift]
        linarith [(unit 2).1]
      · have hw : 0 < w.val := by omega
        have hw1 : (1:ℝ) ≤ w.val := by exact_mod_cast (by omega : 1 ≤ w.val)
        have h2 : 1/4 ≤ r 2 := by
          change 1/4 ≤ gated (0 < w.val ∧ q.val < 2)
            (-12*(p.val:ℝ)*(w.val-q.val-1)+2*Q+1) 0
          simpa [gated, hq, hQ0, hw] using port.2.2.2 hw1
        dsimp [drift]
        linarith [(unit 0).1]

noncomputable def pool {n : ℕ} (x : Fin n → Fin 3) (clique : Fin n → ℕ)
    (i : Fin n) : ℝ := ∑ j, if clique j = clique i then (x j).val else 0

/-- All-size nonempty-body drift from the actual quadratic-pool local rate
formula. V=13(Ptotal+Wtotal)+9Qtotal is a proof observer only. Its coefficients
balance the outlet and enabling margins; they do not modify the update.
Multiply the displayed sum by the common proposal probability 1/(2*F*n).
Cross-cell p and q transfers preserve V, regardless of their acceptance.
Runtime refinement and conversion to first-hit time are separate obligations.
-/
theorem resource_release_drift {n : ℕ} (p q w : Fin n → Fin 3)
    (row col : Fin n → ℕ) (force : Fin n → ℝ) (stocked : Bool)
    (hforce : ∀ i, 96 ≤ force i)
    (hnonempty : ∃ i, 0 < (p i).val + (q i).val + (w i).val) :
    (∑ i, drift (rates (p i) (q i) (w i) (pool p row i) (pool q col i) (force i) stocked))
      ≤ -1 + n * (9/85070591730234615865843651857942052864) := by
  have hP (i : Fin n) : ((p i).val : ℝ) ≤ pool p row i := by
    have h := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) =>
      show (0:ℝ) ≤ (if row j = row i then ((p j).val:ℝ) else 0) by split_ifs <;> positivity)
      (Finset.mem_univ i)
    simpa [pool] using h
  have hQ (i : Fin n) : 0 ≤ pool q col i := by
    exact Finset.sum_nonneg (by intro j _; split_ifs <;> positivity)
  have localBound (i : Fin n) := local_release (p i) (q i) (w i)
    (pool p row i) (pool q col i) (force i) stocked (hP i) (hQ i) (hforce i)
  have witness : ∃ i, 0 < (q i).val ∨
      ((q i).val = 0 ∧ pool q col i = 0 ∧ 0 < (p i).val + (w i).val) := by
    by_cases hsome : ∃ i, 0 < (q i).val
    · obtain ⟨i, hi⟩ := hsome; exact ⟨i, Or.inl hi⟩
    · have hz : ∀ i, (q i).val = 0 := by intro i; by_contra h; exact hsome ⟨i, by omega⟩
      obtain ⟨i, hi⟩ := hnonempty
      refine ⟨i, Or.inr ⟨hz i, ?_, by simpa [hz i] using hi⟩⟩
      simp [pool, hz]
  obtain ⟨chosen, hc⟩ := witness
  calc
    _ ≤ ∑ i, (9/85070591730234615865843651857942052864 - if i = chosen then (1:ℝ) else 0) := by
      apply Finset.sum_le_sum
      intro i _
      by_cases he : i = chosen
      · subst i
        simp only [ite_true]
        linarith [(localBound chosen).2 hc]
      · simpa [he] using (localBound i).1
    _ = -1 + n * (9/85070591730234615865843651857942052864) := by simp [Finset.sum_sub_distrib]; ring

noncomputable def releaseValue (p q w : Fin 4 → Fin 3) : ℝ :=
  ∑ i, (13 * (((p i).val : ℝ) + (w i).val) + 9 * (q i).val)

private theorem releaseValue_bounds (p q w : Fin 4 → Fin 3)
    (hmass : (∑ i, ((p i).val + (q i).val + (w i).val)) ≤ 12)
    (hne : ∃ i, 0 < (p i).val + (q i).val + (w i).val) :
    9 ≤ releaseValue p q w ∧ releaseValue p q w ≤ 156 := by
  have hm : (∑ i, (((p i).val : ℝ) + (q i).val + (w i).val)) ≤ 12 := by
    exact_mod_cast hmass
  have hlo (i : Fin 4) : 9 * (((p i).val : ℝ) + (q i).val + (w i).val) ≤
      13 * (((p i).val : ℝ) + (w i).val) + 9 * (q i).val := by
    nlinarith [Nat.cast_nonneg (α := ℝ) (p i).val, Nat.cast_nonneg (α := ℝ) (w i).val]
  have hhi : releaseValue p q w ≤ 13 * (∑ i, (((p i).val : ℝ) + (q i).val + (w i).val)) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (by intro i _; have h := Nat.cast_nonneg (α := ℝ) (q i).val; linarith)
  constructor
  · obtain ⟨i, hi⟩ := hne
    have hi1 : (1:ℝ) ≤ ((p i).val : ℝ) + (q i).val + (w i).val := by
      exact_mod_cast (by omega : 1 ≤ (p i).val + (q i).val + (w i).val)
    have hterm := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) =>
      show (0:ℝ) ≤ 13 * (((p j).val : ℝ) + (w j).val) + 9 * (q j).val by positivity)
      (Finset.mem_univ i)
    dsimp [releaseValue]
    linarith [hlo i]
  · linarith

/- Formal view of the SAME four p/q/w cells, finite stock and native moves.
All extra functions below are proof/reference code, not optimizer state. -/
abbrev NativeCell := Fin 3 × Fin 3 × Fin 3
abbrev NativeBody := Fin 4 → NativeCell

def nativeCellMass (c : NativeCell) : ℕ := c.1.val + c.2.1.val + c.2.2.val
noncomputable def nativeCellValue (c : NativeCell) : ℝ :=
  13 * ((c.1.val : ℝ) + c.2.2.val) + 9 * c.2.1.val
def nativeMass (b : NativeBody) : ℕ := ∑ i, nativeCellMass (b i)
noncomputable def nativeValue (b : NativeBody) : ℝ := ∑ i, nativeCellValue (b i)

private def incField (x : Fin 3) : Fin 3 := ⟨(x.val + 1) % 3, Nat.mod_lt _ (by decide)⟩
private def decField (x : Fin 3) : Fin 3 := ⟨(x.val + 2) % 3, Nat.mod_lt _ (by decide)⟩
private def localChange (c : NativeCell) (m : Fin 6) : NativeCell :=
  match m.val with
  | 0 => (decField c.1, incField c.2.1, c.2.2)
  | 1 => (incField c.1, decField c.2.1, c.2.2)
  | 2 => (c.1, incField c.2.1, decField c.2.2)
  | 3 => (c.1, decField c.2.1, incField c.2.2)
  | 4 => (c.1, incField c.2.1, c.2.2)
  | _ => (c.1, decField c.2.1, c.2.2)

private def localEnabled (c : NativeCell) (m : Fin 6) : Prop :=
  match m.val with
  | 0 => 0 < c.1.val ∧ c.2.1.val < 2
  | 1 => 0 < c.2.1.val ∧ c.1.val < 2
  | 2 => 0 < c.2.2.val ∧ c.2.1.val < 2
  | 3 => 0 < c.2.1.val ∧ c.2.2.val < 2
  | 4 => c.2.1.val < 2
  | _ => 0 < c.2.1.val
private instance (c : NativeCell) (m : Fin 6) : Decidable (localEnabled c m) := by
  unfold localEnabled; split <;> infer_instance

private def localMassDelta (m : Fin 6) : ℤ := if m.val < 4 then 0 else if m.val = 4 then 1 else -1
private def localValueDelta (m : Fin 6) : ℝ :=
  match m.val with
  | 0 => -4
  | 1 => 4
  | 2 => -4
  | 3 => 4
  | 4 => 9
  | _ => -9

set_option maxHeartbeats 4000000 in
private theorem local_change_effect (c : NativeCell) (m : Fin 6) (h : localEnabled c m) :
    ((nativeCellMass (localChange c m) : ℤ) - nativeCellMass c = localMassDelta m) ∧
    (nativeCellValue (localChange c m) - nativeCellValue c = localValueDelta m) := by
  rcases c with ⟨p, q, w⟩
  fin_cases p <;> fin_cases q <;> fin_cases w <;> fin_cases m <;>
    norm_num [localEnabled, localChange, incField, decField, nativeCellMass,
      nativeCellValue, localMassDelta, localValueDelta] at *

private theorem sum_update_nat (b : NativeBody) (i : Fin 4) (c : NativeCell) :
    nativeMass (Function.update b i c) + nativeCellMass (b i) = nativeMass b + nativeCellMass c := by
  have hf : (fun j => nativeCellMass (Function.update b i c j)) =
      Function.update (fun j => nativeCellMass (b j)) i (nativeCellMass c) := by
    funext j; by_cases h : j = i <;> simp [h]
  unfold nativeMass
  rw [hf, Finset.sum_update_of_mem (Finset.mem_univ i)]
  have h := Finset.sum_erase_add Finset.univ (fun j => nativeCellMass (b j)) (Finset.mem_univ i)
  simpa [Finset.sdiff_singleton_eq_erase, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
    congrArg (fun n => n + nativeCellMass c) h

private theorem sum_update_value (b : NativeBody) (i : Fin 4) (c : NativeCell) :
    nativeValue (Function.update b i c) - nativeValue b = nativeCellValue c - nativeCellValue (b i) := by
  have hf : (fun j => nativeCellValue (Function.update b i c j)) =
      Function.update (fun j => nativeCellValue (b j)) i (nativeCellValue c) := by
    funext j; by_cases h : j = i <;> simp [h]
  unfold nativeValue
  rw [hf, Finset.sum_update_of_mem (Finset.mem_univ i)]
  have h := Finset.sum_erase_add Finset.univ (fun j => nativeCellValue (b j)) (Finset.mem_univ i)
  simp only [Finset.sdiff_singleton_eq_erase] at *
  linarith

private def selectedField (c : NativeCell) (isQ : Bool) : Fin 3 := if isQ then c.2.1 else c.1
private def shiftCell (c : NativeCell) (isQ up : Bool) : NativeCell :=
  if isQ then (c.1, (if up then incField c.2.1 else decField c.2.1), c.2.2)
  else ((if up then incField c.1 else decField c.1), c.2.1, c.2.2)
private def fieldWeight (isQ : Bool) : ℝ := if isQ then 9 else 13

private theorem shift_up_effect (c : NativeCell) (isQ : Bool)
    (h : (selectedField c isQ).val < 2) :
    nativeCellMass (shiftCell c isQ true) = nativeCellMass c + 1 ∧
    nativeCellValue (shiftCell c isQ true) - nativeCellValue c = fieldWeight isQ := by
  rcases c with ⟨p, q, w⟩
  cases isQ
  · fin_cases p <;> norm_num [selectedField, shiftCell, incField, nativeCellMass,
      nativeCellValue, fieldWeight] at * <;> constructor <;> ring
  · fin_cases q <;> norm_num [selectedField, shiftCell, incField, nativeCellMass,
      nativeCellValue, fieldWeight] at * <;> ring

private theorem shift_down_effect (c : NativeCell) (isQ : Bool)
    (h : 0 < (selectedField c isQ).val) :
    nativeCellMass (shiftCell c isQ false) + 1 = nativeCellMass c ∧
    nativeCellValue (shiftCell c isQ false) - nativeCellValue c = -fieldWeight isQ := by
  rcases c with ⟨p, q, w⟩
  cases isQ
  · fin_cases p <;> norm_num [selectedField, shiftCell, decField, nativeCellMass,
      nativeCellValue, fieldWeight] at * <;> constructor <;> ring
  · fin_cases q <;> norm_num [selectedField, shiftCell, decField, nativeCellMass,
      nativeCellValue, fieldWeight] at * <;> ring

private def partner (isQ : Bool) (i : Fin 4) : Fin 4 :=
  if isQ then (match i.val with | 0 => 2 | 1 => 3 | 2 => 0 | _ => 1)
  else (match i.val with | 0 => 1 | 1 => 0 | 2 => 3 | _ => 2)
private theorem partner_ne (isQ : Bool) (i : Fin 4) : partner isQ i ≠ i := by
  cases isQ <;> fin_cases i <;> decide

private def transferChange (b : NativeBody) (i : Fin 4) (isQ : Bool) : NativeBody :=
  Function.update (Function.update b i (shiftCell (b i) isQ false))
    (partner isQ i) (shiftCell (b (partner isQ i)) isQ true)

private theorem transfer_change_effect (b : NativeBody) (i : Fin 4) (isQ : Bool)
    (hd : 0 < (selectedField (b i) isQ).val)
    (hu : (selectedField (b (partner isQ i)) isQ).val < 2) :
    nativeMass (transferChange b i isQ) = nativeMass b ∧
    nativeValue (transferChange b i isQ) = nativeValue b := by
  have hne := partner_ne isQ i
  have hd' := shift_down_effect (b i) isQ hd
  have hu' := shift_up_effect (b (partner isQ i)) isQ hu
  let middle := Function.update b i (shiftCell (b i) isQ false)
  have hj : middle (partner isQ i) = b (partner isQ i) := by simp [middle, hne]
  have hm1 := sum_update_nat b i (shiftCell (b i) isQ false)
  have hm2 := sum_update_nat middle (partner isQ i) (shiftCell (b (partner isQ i)) isQ true)
  have hv1 := sum_update_value b i (shiftCell (b i) isQ false)
  have hv2 := sum_update_value middle (partner isQ i) (shiftCell (b (partner isQ i)) isQ true)
  rw [hj] at hm2 hv2
  constructor
  · change nativeMass (Function.update middle _ _) = _
    change nativeMass middle + _ = _ at hm1
    omega
  · change nativeValue (Function.update middle _ _) = _
    change nativeValue middle - _ = _ at hv1
    linarith

abbrev NativeEvent := Fin 4 × Fin 10
private def localMode (c : Fin 10) : Option (Fin 6) :=
  match c.val with
  | 0 => some 0 | 1 => some 0 | 2 => some 1 | 3 => some 1
  | 4 => some 3 | 5 => some 2 | 6 => none | 7 => none
  | 8 => some 4 | _ => some 5
private def choiceValueDelta (c : Fin 10) : ℝ := (localMode c).elim 0 localValueDelta

def nativeCandidate (b : NativeBody) (e : NativeEvent) : NativeBody :=
  match localMode e.2 with
  | some m => Function.update b e.1 (localChange (b e.1) m)
  | none => transferChange b e.1 (decide (e.2.val = 7))
def nativeEnabled (b : NativeBody) (e : NativeEvent) : Prop :=
  match localMode e.2 with
  | some m => localEnabled (b e.1) m ∧ (m.val = 4 → nativeMass b < 12)
  | none =>
    0 < (selectedField (b e.1) (decide (e.2.val = 7))).val ∧
      (selectedField (b (partner (decide (e.2.val = 7)) e.1)) (decide (e.2.val = 7))).val < 2
private instance (b : NativeBody) (e : NativeEvent) : Decidable (nativeEnabled b e) := by
  unfold nativeEnabled; split <;> infer_instance

private theorem native_candidate_effect (b : NativeBody) (e : NativeEvent)
    (hmass : nativeMass b ≤ 12) (h : nativeEnabled b e) :
    nativeMass (nativeCandidate b e) ≤ 12 ∧
      nativeValue (nativeCandidate b e) - nativeValue b = choiceValueDelta e.2 := by
  rcases e with ⟨i, c⟩
  cases hmode : localMode c with
  | some m =>
    unfold nativeEnabled at h
    rw [hmode] at h
    change localEnabled (b i) m ∧ (m.val = 4 → nativeMass b < 12) at h
    obtain ⟨hl, hs⟩ := h
    have hd := local_change_effect (b i) m hl
    have hm := sum_update_nat b i (localChange (b i) m)
    have hmZ : (nativeMass (Function.update b i (localChange (b i) m)) : ℤ) + nativeCellMass (b i) =
        (nativeMass b : ℤ) + nativeCellMass (localChange (b i) m) := by exact_mod_cast hm
    constructor
    · simp only [nativeCandidate, hmode]
      fin_cases m <;> norm_num [localMassDelta] at * <;> omega
    · simp only [nativeCandidate, hmode, choiceValueDelta, Option.elim]
      exact (sum_update_value b i (localChange (b i) m)).trans hd.2
  | none =>
    unfold nativeEnabled at h
    rw [hmode] at h
    change 0 < (selectedField (b i) (decide (c.val = 7))).val ∧
      (selectedField (b (partner (decide (c.val = 7)) i)) (decide (c.val = 7))).val < 2 at h
    have hd := transfer_change_effect b i (decide (c.val = 7)) h.1 h.2
    constructor
    · simpa only [nativeCandidate, hmode, hd.1] using hmass
    · simp only [nativeCandidate, hmode, choiceValueDelta, Option.elim, hd.2, sub_self]

abbrev NativeState := {b : NativeBody // 0 < nativeMass b ∧ nativeMass b ≤ 12}
noncomputable instance : Fintype NativeState := Fintype.ofFinite _

def nativePack (b : NativeBody) : Option NativeState :=
  if h : 0 < nativeMass b ∧ nativeMass b ≤ 12 then some ⟨b, h⟩ else none

private theorem native_value_zero (b : NativeBody) (h : nativeMass b = 0) : nativeValue b = 0 := by
  apply Finset.sum_eq_zero
  intro i _
  have hi := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) => Nat.zero_le (nativeCellMass (b j)))
    (Finset.mem_univ i)
  change nativeCellMass (b i) ≤ nativeMass b at hi
  rw [h] at hi
  rcases hcell : b i with ⟨p, q, w⟩
  have hm : p.val + q.val + w.val = 0 := by simpa [hcell, nativeCellMass] using hi
  have hp : p.val = 0 := by omega
  have hq : q.val = 0 := by omega
  have hw : w.val = 0 := by omega
  simp [nativeCellValue, hp, hq, hw]

private theorem native_pack_value (b : NativeBody) (hcap : nativeMass b ≤ 12) :
    (nativePack b).elim 0 (fun x => nativeValue x.val) = nativeValue b := by
  unfold nativePack
  split_ifs with h
  · rfl
  · have hm : nativeMass b = 0 := by omega
    simp [native_value_zero b hm]

private def nativeRow (i : Fin 4) : ℕ := i.val / 2
private def nativeCol (i : Fin 4) : ℕ := i.val % 2
noncomputable def nativeEnergy (b : NativeBody) : ℝ :=
  (∑ i, (14 * (nativeCellMass (b i) : ℝ) - 12 * (b i).1.val * (b i).2.1.val * (b i).2.2.val)) +
    (pool (fun i => (b i).1) nativeRow 0)^2 + (pool (fun i => (b i).1) nativeRow 2)^2 +
    (pool (fun i => (b i).2.1) nativeCol 0)^2 + (pool (fun i => (b i).2.1) nativeCol 1)^2
private noncomputable def nativeWork (f : Fin 4 → ℝ) (b c : NativeBody) : ℝ :=
  ∑ i, (f i * ((nativeCellMass (c i) : ℝ) - nativeCellMass (b i)))

noncomputable def nativeRate (f : Fin 4 → ℝ) (b : NativeBody) (e : NativeEvent) : ℝ :=
  match localMode e.2 with
  | some m => rates (b e.1).1 (b e.1).2.1 (b e.1).2.2
      (pool (fun i => (b i).1) nativeRow e.1) (pool (fun i => (b i).2.1) nativeCol e.1)
      (f e.1) (decide (nativeMass b < 12)) m
  | none => gated (nativeEnabled b e) (nativeEnergy (nativeCandidate b e) - nativeEnergy b)
      (nativeWork f b (nativeCandidate b e))

private theorem local_rate_unit (p q w : Fin 3) (P Q f : ℝ) (stocked : Bool) (m : Fin 6) :
    0 ≤ rates p q w P Q f stocked m ∧ rates p q w P Q f stocked m ≤ 1 := by
  fin_cases m <;> exact gated_unit _ _ _
private theorem local_rate_disabled (p q w : Fin 3) (P Q f : ℝ) (stocked : Bool) (m : Fin 6)
    (h : ¬(localEnabled (p, q, w) m ∧ (m.val = 4 → stocked = true))) :
    rates p q w P Q f stocked m = 0 := by
  fin_cases m <;> simp_all [rates, localEnabled, gated]

private theorem native_rate_unit (f : Fin 4 → ℝ) (b : NativeBody) (e : NativeEvent) :
    0 ≤ nativeRate f b e ∧ nativeRate f b e ≤ 1 := by
  unfold nativeRate
  split
  · exact local_rate_unit _ _ _ _ _ _ _ _
  · exact gated_unit _ _ _
private theorem native_rate_disabled (f : Fin 4 → ℝ) (b : NativeBody) (e : NativeEvent)
    (h : ¬nativeEnabled b e) : nativeRate f b e = 0 := by
  cases hm : localMode e.2 with
  | some m =>
    rw [nativeRate, hm]
    apply local_rate_disabled
    simpa only [nativeEnabled, hm, decide_eq_true_eq] using h
  | none =>
    rw [nativeRate, hm]
    exact ite_eq_right h

private noncomputable def nativeProposal (e : NativeEvent) : ℝ :=
  if e.2.val < 4 then 1/80 else if e.2.val < 6 then 1/40 else if e.2.val < 8 then 1/20 else 1/40

private noncomputable def jumpKernel {X E : Type*} [Fintype E] [DecidableEq X]
    (p : E → ℝ) (a : X → E → ℝ) (d : X → E → Option X) (x y : X) : ℝ :=
  (if x = y then 1 - ∑ e, p e * a x e else 0) +
    ∑ e, if d x e = some y then p e * a x e else 0

private theorem jumpKernel_contract {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    (p : E → ℝ) (a : X → E → ℝ) (d : X → E → Option X)
    (hp : ∀ e, 0 ≤ p e) (hsum : ∑ e, p e = 1)
    (ha : ∀ x e, 0 ≤ a x e ∧ a x e ≤ 1) :
    (∀ x y, 0 ≤ jumpKernel p a d x y) ∧
    (∀ v x, (∑ y, jumpKernel p a d x y * v y) =
      v x + ∑ e, p e * a x e * ((d x e).elim 0 v - v x)) := by
  constructor
  · intro x y
    have hm : ∑ e, p e * a x e ≤ 1 := calc
      _ ≤ ∑ e, p e * 1 := Finset.sum_le_sum (by
        intro e _; exact mul_le_mul_of_nonneg_left (ha x e).2 (hp e))
      _ = 1 := by simpa using hsum
    unfold jumpKernel
    apply add_nonneg
    · split_ifs
      · exact sub_nonneg.mpr hm
      · exact le_refl 0
    · apply Finset.sum_nonneg
      intro e _; split_ifs
      · exact mul_nonneg (hp e) (ha x e).1
      · exact le_refl 0
  · intro v x
    unfold jumpKernel
    simp_rw [add_mul, Finset.sum_add_distrib]
    have hd : (∑ y, (if x = y then 1 - ∑ e, p e * a x e else 0) * v y) =
        (1 - ∑ e, p e * a x e) * v x := by simp
    rw [hd]
    have hj : (∑ y, (∑ e, if d x e = some y then p e * a x e else 0) * v y) =
        ∑ e, p e * a x e * (d x e).elim 0 v := by
      simp_rw [Finset.sum_mul]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro e _
      cases d x e with
      | none => simp
      | some z => simp
    rw [hj]
    simp_rw [mul_sub, Finset.sum_sub_distrib]
    rw [← Finset.sum_mul]
    ring

noncomputable def nativeKernel (f : ℕ → NativeState → Fin 4 → ℝ)
    (t : ℕ) (x y : NativeState) : ℝ := by
  classical
  exact jumpKernel nativeProposal (fun s e => nativeRate (f t s) s.val e)
    (fun s e => nativePack (nativeCandidate s.val e)) x y

private theorem native_event_change (f : Fin 4 → ℝ) (s : NativeState) (e : NativeEvent) :
    nativeRate f s.val e *
        ((nativePack (nativeCandidate s.val e)).elim 0 (fun x => nativeValue x.val) - nativeValue s.val) =
      nativeRate f s.val e * choiceValueDelta e.2 := by
  by_cases h : nativeEnabled s.val e
  · have hc := native_candidate_effect s.val e s.property.2 h
    rw [native_pack_value _ hc.1, hc.2]
  · rw [native_rate_disabled f s.val e h]
    simp

private theorem native_rate_collapse (f : Fin 4 → ℝ) (b : NativeBody) :
    (∑ e, nativeProposal e * nativeRate f b e * choiceValueDelta e.2) =
      (1/40:ℝ) * ∑ i, drift (rates (b i).1 (b i).2.1 (b i).2.2
        (pool (fun j => (b j).1) nativeRow i) (pool (fun j => (b j).2.1) nativeCol i)
        (f i) (decide (nativeMass b < 12))) := by
  rw [Fintype.sum_prod_type, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  norm_num [nativeProposal, nativeRate, choiceValueDelta, localMode, localValueDelta,
    drift, Fin.sum_univ_succ]
  ring

private theorem native_kernel_contract (f : ℕ → NativeState → Fin 4 → ℝ) :
    (∀ t x y, 0 ≤ nativeKernel f t x y) ∧
    (∀ t x, (∑ y, nativeKernel f t x y) ≤ 1) ∧
    (∀ t x, (∑ y, nativeKernel f t x y * nativeValue y.val) = nativeValue x.val +
      (1/40:ℝ) * ∑ i, drift (rates (x.val i).1 (x.val i).2.1 (x.val i).2.2
        (pool (fun j => (x.val j).1) nativeRow i) (pool (fun j => (x.val j).2.1) nativeCol i)
        (f t x i) (decide (nativeMass x.val < 12)))) := by
  classical
  have hp (e : NativeEvent) : 0 ≤ nativeProposal e := by
    unfold nativeProposal; split_ifs <;> norm_num
  have hs : ∑ e, nativeProposal e = 1 := by
    norm_num [Fintype.sum_prod_type, nativeProposal, Fin.sum_univ_succ]
  have hc (t : ℕ) := jumpKernel_contract nativeProposal
    (fun s e => nativeRate (f t s) s.val e) (fun s e => nativePack (nativeCandidate s.val e))
    hp hs (fun s e => native_rate_unit (f t s) s.val e)
  refine ⟨?_, ?_, ?_⟩
  · intro t; exact (hc t).1
  · intro t x
    change (∑ y, jumpKernel _ _ _ x y) ≤ 1
    have h := (hc t).2 (fun _ => (1:ℝ)) x
    simp only [mul_one] at h
    rw [h]
    have hsum : (∑ e, nativeProposal e * nativeRate (f t x) x.val e *
        ((nativePack (nativeCandidate x.val e)).elim 0 (fun _ => (1:ℝ)) - 1)) ≤ 0 := by
      apply Finset.sum_nonpos
      intro e _
      apply mul_nonpos_of_nonneg_of_nonpos
      · exact mul_nonneg (hp e) (native_rate_unit (f t x) x.val e).1
      · cases nativePack (nativeCandidate x.val e) <;> norm_num
    linarith
  · intro t x
    change (∑ y, jumpKernel _ _ _ x y * nativeValue y.val) = _
    rw [(hc t).2 (fun y => nativeValue y.val) x]
    have he : (∑ e, nativeProposal e * nativeRate (f t x) x.val e *
        ((nativePack (nativeCandidate x.val e)).elim 0 (fun y => nativeValue y.val) - nativeValue x.val)) =
        ∑ e, nativeProposal e * nativeRate (f t x) x.val e * choiceValueDelta e.2 := by
      apply Finset.sum_congr rfl
      intro e _
      have h := congrArg (fun z => nativeProposal e * z) (native_event_change (f t x) x e)
      simpa only [mul_assoc] using h
    rw [he, native_rate_collapse]

private def decOrSame (x : Fin 3) : Fin 3 := if 0 < x.val then decField x else x
private theorem decOrSame_le (x : Fin 3) : (decOrSame x).val ≤ x.val := by
  fin_cases x <;> norm_num [decOrSame, decField]
private def damageCell (c : NativeCell) (field : Fin 3) : NativeCell :=
  match field.val with
  | 0 => (decOrSame c.1, c.2.1, c.2.2)
  | 1 => (c.1, decOrSame c.2.1, c.2.2)
  | _ => (c.1, c.2.1, decOrSame c.2.2)
private theorem damage_cell_effect (c : NativeCell) (field : Fin 3) :
    nativeCellMass (damageCell c field) ≤ nativeCellMass c ∧
    nativeCellValue (damageCell c field) ≤ nativeCellValue c := by
  have hp := decOrSame_le c.1
  have hq := decOrSame_le c.2.1
  have hw := decOrSame_le c.2.2
  have hpR : ((decOrSame c.1).val : ℝ) ≤ c.1.val := by exact_mod_cast hp
  have hqR : ((decOrSame c.2.1).val : ℝ) ≤ c.2.1.val := by exact_mod_cast hq
  have hwR : ((decOrSame c.2.2).val : ℝ) ≤ c.2.2.val := by exact_mod_cast hw
  fin_cases field <;> norm_num [damageCell, nativeCellMass, nativeCellValue] <;>
    omega

def nativeDamageCandidate (b : NativeBody) (e : Fin 4 × Fin 3) : NativeBody :=
  Function.update b e.1 (damageCell (b e.1) e.2)
private theorem native_damage_effect (b : NativeBody) (e : Fin 4 × Fin 3) :
    nativeMass (nativeDamageCandidate b e) ≤ nativeMass b ∧
    nativeValue (nativeDamageCandidate b e) ≤ nativeValue b := by
  have hd := damage_cell_effect (b e.1) e.2
  have hm := sum_update_nat b e.1 (damageCell (b e.1) e.2)
  have hv := sum_update_value b e.1 (damageCell (b e.1) e.2)
  constructor
  · change nativeMass (Function.update _ _ _) ≤ _; omega
  · change nativeValue (Function.update _ _ _) ≤ _; linarith

noncomputable def nativeDamageKernel (x y : NativeState) : ℝ := by
  classical
  exact jumpKernel (fun _ : Fin 4 × Fin 3 => (1/12:ℝ)) (fun _ _ => (1/1024:ℝ))
    (fun s e => nativePack (nativeDamageCandidate s.val e)) x y

private theorem native_damage_contract :
    (∀ x y, 0 ≤ nativeDamageKernel x y) ∧
    (∀ x, (∑ y, nativeDamageKernel x y * (nativeValue y.val / 9)) ≤ nativeValue x.val / 9) := by
  classical
  have hc := jumpKernel_contract (fun _ : Fin 4 × Fin 3 => (1/12:ℝ)) (fun _ _ => (1/1024:ℝ))
    (fun s : NativeState => fun e => nativePack (nativeDamageCandidate s.val e))
    (by intro _; norm_num)
    (by norm_num [Fintype.sum_prod_type, Fin.sum_univ_succ]) (by intro _ _; norm_num)
  refine ⟨hc.1, ?_⟩
  intro x
  change (∑ y, jumpKernel _ _ _ x y * (nativeValue y.val / 9)) ≤ _
  rw [hc.2 (fun y => nativeValue y.val / 9) x]
  have hsum : (∑ e : Fin 4 × Fin 3, (1/12:ℝ) * (1/1024) *
      ((nativePack (nativeDamageCandidate x.val e)).elim 0 (fun y => nativeValue y.val / 9) - nativeValue x.val / 9)) ≤ 0 := by
    apply Finset.sum_nonpos
    intro e _
    have he := native_damage_effect x.val e
    have hp := native_pack_value (nativeDamageCandidate x.val e) (he.1.trans x.property.2)
    have hdiv : (nativePack (nativeDamageCandidate x.val e)).elim 0 (fun y => nativeValue y.val / 9) =
        (nativePack (nativeDamageCandidate x.val e)).elim 0 (fun y => nativeValue y.val) / 9 := by
      cases nativePack (nativeDamageCandidate x.val e) <;> simp
    rw [hdiv, hp]
    nlinarith [he.2]
  linarith

private theorem native_nonempty (x : NativeState) :
    ∃ i, 0 < (x.val i).1.val + (x.val i).2.1.val + (x.val i).2.2.val := by
  by_contra hn
  have hz : ∀ i, nativeCellMass (x.val i) = 0 := by
    intro i; by_contra h; exact hn ⟨i, by simpa only [nativeCellMass] using Nat.pos_of_ne_zero h⟩
  have hm : nativeMass x.val = 0 := by simp [nativeMass, hz]
  have hp := x.property.1
  omega

private theorem native_value_bounds (x : NativeState) : 9 ≤ nativeValue x.val ∧ nativeValue x.val ≤ 156 := by
  have h := releaseValue_bounds (fun i => (x.val i).1) (fun i => (x.val i).2.1) (fun i => (x.val i).2.2)
    (by simpa only [nativeMass, nativeCellMass] using x.property.2) (native_nonempty x)
  simpa only [nativeValue, nativeCellValue, releaseValue] using h

private theorem native_weighted_release (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) (t : ℕ) (x : NativeState) :
    (∑ y, nativeKernel f t x y * (nativeValue y.val / 9)) ≤
      (6399/6400:ℝ) * (nativeValue x.val / 9) := by
  have hd := resource_release_drift (fun i => (x.val i).1) (fun i => (x.val i).2.1)
    (fun i => (x.val i).2.2) nativeRow nativeCol (f t x) (decide (nativeMass x.val < 12))
    (hf t x) (native_nonempty x)
  have hv := (native_value_bounds x).2
  have hdiv : (∑ y, nativeKernel f t x y * (nativeValue y.val / 9)) =
      (∑ y, nativeKernel f t x y * nativeValue y.val) / 9 := by
    simp only [div_eq_mul_inv, Finset.sum_mul, mul_assoc]
  rw [hdiv, (native_kernel_contract f).2.2 t x]
  nlinarith

noncomputable def nativeOpenKernel (f : ℕ → NativeState → Fin 4 → ℝ)
    (t : ℕ) (x y : NativeState) : ℝ := ∑ z, nativeDamageKernel x z * nativeKernel f t z y

/-- Same external wear as holding: wear then funded physical opportunity,
first empty hit observed at either substep. Only genuine paid opportunities
are counted; the kernel includes all native proposals and every legal body.
No row-bookkeeping, desired memory or desired release property is assumed. -/
theorem resource_release_budget (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) :
    ∀ T start x, CGB.OptimizerContentReview.kernelSurvival (nativeOpenKernel f) T start x ≤
      (6399/6400:ℝ)^T * (nativeValue x.val / 9) := by
  have hd := native_damage_contract
  have hp := native_kernel_contract f
  apply (CGB.OptimizerContentReview.kinetic_survival_envelope (nativeOpenKernel f)
    (fun x => nativeValue x.val / 9) (6399/6400) ?_ (by norm_num)).2
  · intro x
    apply (le_div_iff₀ (by norm_num : (0:ℝ) < 9)).2
    simpa using (native_value_bounds x).1
  · intro t x
    have heq : (∑ y, nativeOpenKernel f t x y * (nativeValue y.val / 9)) =
        ∑ z, nativeDamageKernel x z * (∑ y, nativeKernel f t z y * (nativeValue y.val / 9)) := by
      simp only [nativeOpenKernel, Finset.sum_mul, Finset.mul_sum, mul_assoc]
      rw [Finset.sum_comm]
    rw [heq]
    calc
      _ ≤ ∑ z, nativeDamageKernel x z * ((6399/6400:ℝ) * (nativeValue z.val / 9)) :=
        Finset.sum_le_sum (by intro z _; exact mul_le_mul_of_nonneg_left (native_weighted_release f hf t z) (hd.1 x z))
      _ = (6399/6400:ℝ) * ∑ z, nativeDamageKernel x z * (nativeValue z.val / 9) := by
        rw [Finset.mul_sum]; apply Finset.sum_congr rfl; intro z _; ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (hd.2 x) (by norm_num)
  · intro t x y
    exact Finset.sum_nonneg (by intro z _; exact mul_nonneg (hd.1 x z) (hp.1 t z y))


end CGB.ResourceReleaseReview
#print axioms CGB.ResourceReleaseReview.primitive_release_port
#print axioms CGB.ResourceReleaseReview.resource_release_drift
#print axioms CGB.ResourceReleaseReview.resource_release_budget

#print axioms CGB.ResourceReleaseReview.resource_feedback_contract

/-! Exact rational mixture for the SAME exponential event probability.
No new learning law. Independent fair-bit primitives remain an explicit
execution assumption; finite-prefix semantics and expected work are proved. -/
open scoped BigOperators
namespace CGB.RationalSamplerReview

noncomputable def coefficient (x : ℝ) (n : ℕ) : ℝ :=
  (-1:ℝ)^n * Ring.choose x (n+1)

private theorem coefficient_step (x : ℝ) (n : ℕ) :
    coefficient x (n+1) = coefficient x n * ((n+1:ℝ)-x) / (n+2) := by
  have h := Ring.choose_smul_choose x (n := n+2) (k := n+1) (by omega)
  have hden : (n+2:ℝ) ≠ 0 := by positivity
  have hc : Ring.choose x (n+2) = Ring.choose x (n+1) * (x-(n+1)) / (n+2) := by
    apply (eq_div_iff hden).mpr
    rw [show n+2-(n+1)=1 by omega, Ring.choose_one_right] at h
    simp only [Nat.choose_succ_self_right, nsmul_eq_mul,
      Nat.cast_add, Nat.cast_one] at h
    nlinarith [h]
  unfold coefficient
  rw [pow_succ, hc]
  ring

private theorem coefficient_bounds (x : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (n : ℕ) :
    0 ≤ coefficient x n ∧ coefficient x n ≤ 1 := by
  induction n with
  | zero => simpa [coefficient] using And.intro hx hx1
  | succ n ih =>
    rw [coefficient_step]
    have hratio0 : 0 ≤ ((n+1:ℝ)-x)/(n+2) := div_nonneg (by nlinarith) (by positivity)
    have hratio1 : ((n+1:ℝ)-x)/(n+2) ≤ 1 := by
      apply (div_le_one (by positivity : (0:ℝ) < n+2)).mpr
      linarith
    constructor
    · simpa [mul_div_assoc] using mul_nonneg ih.1 hratio0
    · have h := mul_le_mul_of_nonneg_left hratio1 ih.1
      simpa [mul_div_assoc] using h.trans (by simpa using ih.2)

private theorem series_identity (x : ℝ) :
    HasSum (fun n : ℕ => coefficient x n * (1/2:ℝ)^(n+1)) (1-(2:ℝ)^(-x)) := by
  have hb := (Real.one_add_rpow_hasFPowerSeriesOnBall_zero (a := x)).hasSum
    (y := (-1/2:ℝ)) (by norm_num [Metric.mem_eball, edist_dist, dist_eq_norm, Real.norm_eq_abs])
  have hb' : HasSum (fun n : ℕ => Ring.choose x n * (-1/2:ℝ)^n) ((1/2:ℝ)^x) := by
    convert hb using 1
    · funext n
      simp [binomialSeries, smul_eq_mul]
      rw [div_pow]
      ring
    · norm_num
  have ht := (hasSum_nat_add_iff' 1).mpr hb'
  simp only [Finset.sum_range_one, Ring.choose_zero_right, pow_zero, mul_one] at ht
  have hn := ht.neg
  convert hn using 1
  · funext n
    unfold coefficient
    rw [show (-1/2:ℝ) = (-1)*(1/2) by norm_num, mul_pow, pow_succ]
    ring
  · rw [show (1/2:ℝ) = (2:ℝ)⁻¹ by norm_num, Real.inv_rpow (by norm_num),
      ← Real.rpow_neg (by norm_num)]
    ring

def coefficientQ (x : ℚ) : ℕ → ℚ
  | 0 => x
  | n+1 => coefficientQ x n * ((n+1:ℚ)-x)/(n+2)

private theorem coefficientQ_refinement (x : ℚ) (n : ℕ) :
    (coefficientQ x n : ℝ) = coefficient x n := by
  induction n with
  | zero => simp [coefficientQ, coefficient]
  | succ n ih => simpa [coefficientQ, coefficient_step, ih]

noncomputable def coinNext (p : ℝ) (bit : Bool) : ℝ :=
  if bit then max (2*p-1) 0 else min (2*p) 1

noncomputable def coinAccepted : ℕ → ℝ → ℝ
  | 0, p => if 1 ≤ p then 1 else 0
  | n+1, p => if p ≤ 0 then 0 else if 1 ≤ p then 1 else
      (coinAccepted n (coinNext p false) + coinAccepted n (coinNext p true))/2

noncomputable def coinPending : ℕ → ℝ → ℝ
  | 0, p => if 0 < p ∧ p < 1 then 1 else 0
  | n+1, p => if p ≤ 0 then 0 else if 1 ≤ p then 0 else
      (coinPending n (coinNext p false) + coinPending n (coinNext p true))/2

def coinNextQ (p : ℚ) (bit : Bool) : ℚ :=
  if bit then max (2*p-1) 0 else min (2*p) 1

def coinAcceptedQ : ℕ → ℚ → ℚ
  | 0, p => if 1 ≤ p then 1 else 0
  | n+1, p => if p ≤ 0 then 0 else if 1 ≤ p then 1 else
      (coinAcceptedQ n (coinNextQ p false) + coinAcceptedQ n (coinNextQ p true))/2

private theorem coinNextQ_refinement (p : ℚ) (bit : Bool) :
    (coinNextQ p bit : ℝ) = coinNext (p:ℝ) bit := by
  cases bit <;> simp [coinNextQ, coinNext]

private theorem coinAcceptedQ_refinement (n : ℕ) (p : ℚ) :
    (coinAcceptedQ n p : ℝ) = coinAccepted n (p:ℝ) := by
  induction n generalizing p with
  | zero =>
    by_cases h : 1 ≤ p
    · have hr : (1:ℝ) ≤ p := by exact_mod_cast h
      simp [coinAcceptedQ, coinAccepted, h, hr]
    · have hr : ¬ (1:ℝ) ≤ p := by exact_mod_cast h
      simp [coinAcceptedQ, coinAccepted, h, hr]
  | succ n ih =>
    by_cases h0 : p ≤ 0
    · have h0r : (p:ℝ) ≤ 0 := by exact_mod_cast h0
      simp [coinAcceptedQ, coinAccepted, h0, h0r]
    · have h0r : ¬ (p:ℝ) ≤ 0 := by exact_mod_cast h0
      by_cases h1 : 1 ≤ p
      · have h1r : (1:ℝ) ≤ p := by exact_mod_cast h1
        simp [coinAcceptedQ, coinAccepted, h0, h0r, h1, h1r]
      · have h1r : ¬ (1:ℝ) ≤ p := by exact_mod_cast h1
        simp [coinAcceptedQ, coinAccepted, h0, h0r, h1, h1r, ih, coinNextQ_refinement]

private theorem coin_endpoints (n : ℕ) :
    coinAccepted n 0 = 0 ∧ coinAccepted n 1 = 1 ∧
      coinPending n 0 = 0 ∧ coinPending n 1 = 0 := by
  cases n <;> norm_num [coinAccepted, coinPending]

private theorem coin_next_bounds (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    (0 ≤ coinNext p false ∧ coinNext p false ≤ 1) ∧
    (0 ≤ coinNext p true ∧ coinNext p true ≤ 1) ∧
    coinNext p false + coinNext p true = 2*p := by
  by_cases h : 2*p ≤ 1
  · simp only [coinNext, Bool.false_eq_true, ite_false, ite_true,
      min_eq_left h, max_eq_right (by linarith : 2*p-1 ≤ 0)]
    constructor
    · constructor <;> linarith
    · exact ⟨⟨by norm_num, by norm_num⟩, by ring⟩
  · simp only [coinNext, Bool.false_eq_true, ite_false, ite_true,
      min_eq_right (by linarith : 1 ≤ 2*p), max_eq_left (by linarith : 0 ≤ 2*p-1)]
    constructor
    · norm_num
    · exact ⟨⟨by linarith, by linarith⟩, by ring⟩

private theorem coin_prefix (n : ℕ) (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ coinAccepted n p ∧ coinAccepted n p ≤ p ∧
    p ≤ coinAccepted n p + coinPending n p ∧
    0 ≤ coinPending n p ∧ coinPending n p ≤ (1/2:ℝ)^n := by
  induction n generalizing p with
  | zero =>
    by_cases h1 : 1 ≤ p
    · have he : p = 1 := by linarith
      subst p
      norm_num [coinAccepted, coinPending]
    · by_cases h0 : p ≤ 0
      · have he : p = 0 := by linarith
        subst p
        norm_num [coinAccepted, coinPending]
      · simp only [coinAccepted, coinPending, ite_eq_right h1,
          ite_eq_left (show 0 < p ∧ p < 1 by constructor <;> linarith)]
        norm_num
        exact ⟨hp, hp1⟩
  | succ n ih =>
    by_cases h0 : p ≤ 0
    · have he : p = 0 := by linarith
      subst p
      norm_num [coinAccepted, coinPending]
    · by_cases h1 : 1 ≤ p
      · have he : p = 1 := by linarith
        subst p
        norm_num [coinAccepted, coinPending]
      · obtain ⟨hfalse, htrue, hmean⟩ := coin_next_bounds p hp hp1
        have left := ih (coinNext p false) hfalse.1 hfalse.2
        have right := ih (coinNext p true) htrue.1 htrue.2
        simp only [coinAccepted, coinPending, ite_eq_right h0, ite_eq_right h1]
        refine ⟨by linarith [left.1, right.1], by linarith [left.2.1, right.2.1],
          by linarith [left.2.2.1, right.2.2.1], by linarith [left.2.2.2.1, right.2.2.2.1], ?_⟩
        rw [pow_succ]
        by_cases h : 2*p ≤ 1
        · have hz : coinNext p true = 0 := by
            simp [coinNext, max_eq_right (by linarith : 2*p-1 ≤ 0)]
          rw [hz, (coin_endpoints n).2.2.1]
          linarith [left.2.2.2.2]
        · have hz : coinNext p false = 1 := by
            simp [coinNext, min_eq_right (by linarith : 1 ≤ 2*p)]
          rw [hz, (coin_endpoints n).2.2.2]
          linarith [right.2.2.2.2]

theorem rational_mixture_contract (x : ℚ) (hx : 0 ≤ x) (hx1 : x ≤ 1) :
    (∀ n, 0 ≤ coefficientQ x n ∧ coefficientQ x n ≤ 1) ∧
    HasSum (fun n : ℕ => (1/2:ℝ)^(n+1) * (coefficientQ x n : ℝ))
      (1-(2:ℝ)^(-(x:ℝ))) ∧
    HasSum (fun n : ℕ => (1/2:ℝ)^(n+1) * (1-(coefficientQ x n : ℝ)))
      ((2:ℝ)^(-(x:ℝ))) ∧
    HasSum (fun n : ℕ => (1/2:ℝ)^(n+1)) 1 ∧
    HasSum (fun n : ℕ => (n:ℝ) * (1/2:ℝ)^(n+1)) 1 ∧
    HasSum (fun n : ℕ => (n+1:ℝ) * (1/2:ℝ)^(n+1)) 2 := by
  have hg : HasSum (fun n : ℕ => (1/2:ℝ)^(n+1)) 1 := by
    convert hasSum_geometric_two.div_const 2 using 1
    · funext n
      rw [pow_succ]
      ring
    · norm_num
  have hr : HasSum (fun n : ℕ => (1/2:ℝ)^(n+1) * (coefficientQ x n : ℝ))
      (1-(2:ℝ)^(-(x:ℝ))) := by
    simpa only [coefficientQ_refinement, mul_comm] using series_identity (x:ℝ)
  have hn : HasSum (fun n : ℕ => (n:ℝ) * (1/2:ℝ)^(n+1)) 1 := by
    have h := (hasSum_coe_mul_geometric_of_norm_lt_one
      (r := (1/2:ℝ)) (by norm_num)).div_const 2
    convert h using 1
    · funext n
      rw [pow_succ]
      ring
    · norm_num
  refine ⟨?_, hr, ?_, hg, hn, ?_⟩
  · intro n
    have h := coefficient_bounds (x:ℝ) (by exact_mod_cast hx) (by exact_mod_cast hx1) n
    rw [← coefficientQ_refinement] at h
    exact_mod_cast h
  · convert hg.sub hr using 1 <;> ring
  · convert hn.add hg using 1 <;> ring

theorem rational_binary_coin_contract (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    (∀ q : ℚ, ∀ n, (coinAcceptedQ n q : ℝ) = coinAccepted n (q:ℝ)) ∧
    (∀ n, 0 ≤ coinAccepted n p ∧ coinAccepted n p ≤ p ∧
      p ≤ coinAccepted n p + coinPending n p ∧
      0 ≤ coinPending n p ∧ coinPending n p ≤ (1/2:ℝ)^n) ∧
    Filter.Tendsto (fun n => coinAccepted n p) Filter.atTop (nhds p) ∧
    Filter.Tendsto (fun n => coinPending n p) Filter.atTop (nhds 0) ∧
    (∀ n, (∑ k ∈ Finset.range n, coinPending k p) ≤ 2) := by
  have h := fun n => coin_prefix n p hp hp1
  have hz := tendsto_pow_atTop_nhds_zero_of_lt_one
    (by norm_num : (0:ℝ) ≤ 1/2) (by norm_num : (1/2:ℝ) < 1)
  refine ⟨fun q n => coinAcceptedQ_refinement n q, h, ?_, ?_, ?_⟩
  · have hl : Filter.Tendsto (fun n => p-(1/2:ℝ)^n) Filter.atTop (nhds p) := by
      simpa using tendsto_const_nhds.sub hz
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le hl tendsto_const_nhds
      (fun n => by linarith [(h n).2.2.1, (h n).2.2.2.2]) (fun n => (h n).2.1)
  · exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hz
      (fun n => (h n).2.2.2.1) (fun n => (h n).2.2.2.2)
  · intro n
    have hg : (∑ k ∈ Finset.range n, (1/2:ℝ)^k) = 2*(1-(1/2:ℝ)^n) := by
      induction n with
      | zero => norm_num
      | succ n ih => rw [Finset.sum_range_succ, ih, pow_succ]; ring
    calc
      _ ≤ ∑ k ∈ Finset.range n, (1/2:ℝ)^k :=
        Finset.sum_le_sum (fun k _ => (h k).2.2.2.2)
      _ = 2*(1-(1/2:ℝ)^n) := hg
      _ ≤ 2 := by nlinarith [pow_nonneg (by norm_num : (0:ℝ) ≤ 1/2) n]

#print axioms rational_mixture_contract
#print axioms rational_binary_coin_contract
end CGB.RationalSamplerReview

/-! Conditional readout-fiber kinetics, with the EXISTING state and energy.
This is a declared change of clock/kernel, not an equality of microscopic
trajectories. The uniform proposal covers two adjacent readout fibers only.
No loss oracle, retention score, protected copy or new physical field appears.
-/
namespace CGB.FiberOptimizerReview
open scoped BigOperators

noncomputable def blockFloor {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) : ℝ :=
  (B.image E).min' (hB.image E)

private theorem blockFloor_le {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) (x : S) (hx : x ∈ B) :
    blockFloor B hB E ≤ E x := by
  unfold blockFloor
  exact Finset.min'_le _ _ (Finset.mem_image.mpr ⟨x, hx, rfl⟩)

private theorem blockFloor_attained {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) :
    ∃ x ∈ B, E x = blockFloor B hB E := by
  exact Finset.mem_image.mp (Finset.min'_mem _ _)

noncomputable def blockRate {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) (y : S) : ℝ :=
  if y ∈ B then (2:ℝ)^(-2*(E y - blockFloor B hB E)) / B.card else 0

noncomputable def blockKernel {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) (x y : S) : ℝ :=
  if x ∈ B then
    (if x = y then 1 - ∑ z, blockRate B hB E z else 0) + blockRate B hB E y
  else if x = y then 1 else 0

private theorem block_rate_bounds {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) (y : S) :
    0 ≤ blockRate B hB E y ∧ blockRate B hB E y ≤ 1/(B.card:ℝ) := by
  have hm : (0:ℝ) < B.card := by exact_mod_cast hB.card_pos
  unfold blockRate
  by_cases hy : y ∈ B
  · rw [ite_eq_left hy]
    constructor
    · positivity
    · apply div_le_div_of_nonneg_right _ hm.le
      exact Real.rpow_le_one_of_one_le_of_nonpos (by norm_num)
        (by linarith [blockFloor_le B hB E y hy])
  · simp [hy, le_of_lt (one_div_pos.mpr hm)]

private theorem block_activity {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) :
    1/(B.card:ℝ) ≤ ∑ z, blockRate B hB E z ∧
      (∑ z, blockRate B hB E z) ≤ 1 := by
  have hm : (0:ℝ) < B.card := by exact_mod_cast hB.card_pos
  obtain ⟨x, hx, hmin⟩ := blockFloor_attained B hB E
  constructor
  · have hs := Finset.single_le_sum (fun z (_ : z ∈ Finset.univ) =>
      (block_rate_bounds B hB E z).1) (Finset.mem_univ x)
    simpa [blockRate, hx, hmin] using hs
  · calc
      _ = ∑ z ∈ B, blockRate B hB E z :=
        (Finset.sum_subset (Finset.subset_univ B)
          (by intro z _ hz; simp [blockRate, hz])).symm
      _ ≤ ∑ _z ∈ B, 1/(B.card:ℝ) :=
        Finset.sum_le_sum (fun z _ => (block_rate_bounds B hB E z).2)
      _ = 1 := by simp [nsmul_eq_mul, hm.ne']

/-- Exact conditional heatbath with a lazy factor >=1/|B|. The floor used by
the implementation is the maximal legal common envelope for uniform block
proposals. Equilibrium weights, not a target-retention premise, determine it.
The identity concerns the new kernel; old time certificates do not transfer. -/
theorem readout_block_kernel_contract {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) :
    (∀ x y, 0 ≤ blockKernel B hB E x y) ∧
    (∀ x, (∑ y, blockKernel B hB E x y) = 1) ∧
    (∀ x y, (2:ℝ)^(-2*E x) * blockKernel B hB E x y =
      (2:ℝ)^(-2*E y) * blockKernel B hB E y x) ∧
    (1/(B.card:ℝ) ≤ ∑ z, blockRate B hB E z ∧
      (∑ z, blockRate B hB E z) ≤ 1) ∧
    (∀ x ∈ B, ∀ y,
      let a := ∑ z, blockRate B hB E z
      blockKernel B hB E x y = a * (blockRate B hB E y / a) +
        (1-a)*(if x = y then 1 else 0)) ∧
    (∀ other : ℝ, (∀ z ∈ B, other ≤ E z) → ∀ y ∈ B,
      (2:ℝ)^(-2*(E y-other))/(B.card:ℝ) ≤ blockRate B hB E y) := by
  have activity := block_activity B hB E
  have hm : (0:ℝ) < B.card := by exact_mod_cast hB.card_pos
  have hpos : 0 < ∑ z, blockRate B hB E z :=
    lt_of_lt_of_le (one_div_pos.mpr hm) activity.1
  refine ⟨?_, ?_, ?_, activity, ?_, ?_⟩
  · intro x y
    unfold blockKernel
    split_ifs
    · exact add_nonneg (by linarith [activity.2]) (block_rate_bounds B hB E y).1
    · exact add_nonneg (by norm_num) (block_rate_bounds B hB E y).1
    · norm_num
    · norm_num
  · intro x
    by_cases hx : x ∈ B
    · simp [blockKernel, hx, Finset.sum_add_distrib]
    · simp [blockKernel, hx]
  · intro x y
    by_cases hxy : x = y
    · subst y; rfl
    · by_cases hx : x ∈ B <;> by_cases hy : y ∈ B
      · simp only [blockKernel, hx, hy, hxy, Ne.symm hxy, ite_true, ite_false, zero_add,
          blockRate]
        rw [← mul_div_assoc, ← mul_div_assoc,
          ← Real.rpow_add (by norm_num : (0:ℝ) < 2),
          ← Real.rpow_add (by norm_num : (0:ℝ) < 2)]
        congr 2
        ring
      · simp [blockKernel, blockRate, hx, hy, hxy, Ne.symm hxy]
      · simp [blockKernel, blockRate, hx, hy, hxy, Ne.symm hxy]
      · simp [blockKernel, blockRate, hx, hy, hxy, Ne.symm hxy]
  · intro x hx y
    dsimp only
    unfold blockKernel
    rw [ite_eq_left hx, mul_div_cancel₀ _ hpos.ne']
    by_cases hxy : x = y <;> simp [hxy] <;> ring
  · intro other hl y hy
    obtain ⟨z, hz, hmin⟩ := blockFloor_attained B hB E
    have hf : other ≤ blockFloor B hB E := by simpa [hmin] using hl z hz
    unfold blockRate
    rw [ite_eq_left hy]
    apply div_le_div_of_nonneg_right _ hm.le
    exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)

abbrev Cell := CGB.ResourceReleaseReview.NativeCell
def cellMass (s : Cell) : ℕ := s.1.val + s.2.1.val + s.2.2.val
def readoutBlock (k : Fin 6) : Finset Cell :=
  Finset.univ.filter (fun s => cellMass s = k.val ∨ cellMass s = k.val+1)

private theorem readout_card_bounds (k : Fin 6) :
    4 ≤ (readoutBlock k).card ∧ (readoutBlock k).card ≤ 13 := by
  fin_cases k <;> decide

noncomputable def rawEnergy (a b R C p q w : ℝ) : ℝ :=
  14*(p+q+w) - 12*p*q*w + 2*(R+p)^2/a + 2*(C+q)^2/b

noncomputable def cellEnergy (a b R C : ℝ) (s : Cell) : ℝ :=
  rawEnergy a b R C s.1.val s.2.1.val s.2.2.val

private theorem coord_bounds (p : Fin 3) : 0 ≤ (p.val:ℝ) ∧ (p.val:ℝ) ≤ 2 := by
  constructor
  · exact Nat.cast_nonneg _
  · exact_mod_cast (by omega : p.val ≤ 2)

private theorem product_bound (p q : Fin 3) : (p.val:ℝ)*q.val ≤ 4 := by
  have h := mul_le_mul (coord_bounds p).2 (coord_bounds q).2
    (coord_bounds q).1 (by norm_num : (0:ℝ) ≤ 2)
  norm_num at h
  exact h

private theorem pool_decrement (D R p : ℝ) (hD : 0 < D) (hR : 0 ≤ R) (hp : 1 ≤ p) :
    2*(R+p-1)^2/D - 2*(R+p)^2/D ≤ 0 := by
  have he : 2*(R+p-1)^2/D - 2*(R+p)^2/D = (-4*(R+p)+2)/D := by ring
  rw [he]
  exact div_nonpos_of_nonpos_of_nonneg (by linarith) hD.le

private theorem pool_increment (D R p : ℝ) (hD : 1 ≤ D) (hR : 0 ≤ R)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hcap : R+2 ≤ 2*D) :
    2*(R+p+1)^2/D - 2*(R+p)^2/D < 8 := by
  have he : 2*(R+p+1)^2/D - 2*(R+p)^2/D = (4*(R+p)+2)/D := by ring
  rw [he]
  apply (div_lt_iff₀ (by linarith : 0 < D)).mpr
  linarith

private def removeOne (s : Cell) : Cell :=
  if 0 < s.2.2.val then (s.1, s.2.1, ⟨s.2.2.val-1, by omega⟩)
  else if 0 < s.2.1.val then (s.1, ⟨s.2.1.val-1, by omega⟩, s.2.2)
  else (⟨s.1.val-1, by omega⟩, s.2.1, s.2.2)

private def addOne (s : Cell) : Cell :=
  if hw : s.2.2.val < 2 then (s.1, s.2.1, ⟨s.2.2.val+1, by omega⟩)
  else if hq : s.2.1.val < 2 then (s.1, ⟨s.2.1.val+1, by omega⟩, s.2.2)
  else (⟨min 2 (s.1.val+1), by omega⟩, s.2.1, s.2.2)

private theorem remove_mass (s : Cell) (hs : 0 < cellMass s) :
    cellMass (removeOne s)+1 = cellMass s := by
  unfold removeOne
  split_ifs <;> simp only [cellMass, Fin.val_mk] at * <;> omega

private theorem add_mass (s : Cell) (hs : cellMass s < 6) :
    cellMass (addOne s) = cellMass s+1 := by
  unfold addOne
  split_ifs <;> simp only [cellMass, Fin.val_mk] at * <;> omega

private theorem remove_energy (a b R C : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (s : Cell) (hs : 0 < cellMass s) :
    cellEnergy a b R C (removeOne s) - cellEnergy a b R C s ≤ 34 := by
  unfold removeOne
  split_ifs with hw hq
  · simp only [cellEnergy, Fin.val_mk, Nat.cast_sub (by omega : 1 ≤ s.2.2.val), Nat.cast_one]
    have he : rawEnergy a b R C s.1.val s.2.1.val ((s.2.2.val:ℝ)-1) -
        rawEnergy a b R C s.1.val s.2.1.val s.2.2.val =
        -14+12*(s.1.val:ℝ)*s.2.1.val := by unfold rawEnergy; ring
    rw [he]
    nlinarith [product_bound s.1 s.2.1]
  · simp only [cellEnergy, Fin.val_mk, Nat.cast_sub (by omega : 1 ≤ s.2.1.val), Nat.cast_one]
    have he : rawEnergy a b R C s.1.val ((s.2.1.val:ℝ)-1) s.2.2.val -
        rawEnergy a b R C s.1.val s.2.1.val s.2.2.val =
        -14+12*(s.1.val:ℝ)*s.2.2.val + (2*(C+s.2.1.val-1)^2/b - 2*(C+s.2.1.val)^2/b) := by
      unfold rawEnergy; ring
    rw [he]
    have hd := pool_decrement b C s.2.1.val hb hC (by exact_mod_cast hq)
    nlinarith [product_bound s.1 s.2.2]
  · have hp : 0 < s.1.val := by unfold cellMass at hs; omega
    simp only [cellEnergy, Fin.val_mk, Nat.cast_sub (by omega : 1 ≤ s.1.val), Nat.cast_one]
    have he : rawEnergy a b R C ((s.1.val:ℝ)-1) s.2.1.val s.2.2.val -
        rawEnergy a b R C s.1.val s.2.1.val s.2.2.val =
        -14+12*(s.2.1.val:ℝ)*s.2.2.val + (2*(R+s.1.val-1)^2/a - 2*(R+s.1.val)^2/a) := by
      unfold rawEnergy; ring
    rw [he]
    have hd := pool_decrement a R s.1.val ha hR (by exact_mod_cast hp)
    nlinarith [product_bound s.2.1 s.2.2]

private theorem add_energy (a b R C : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (s : Cell) (hs : cellMass s < 6) :
    cellEnergy a b R C (addOne s) - cellEnergy a b R C s ≤ 22 := by
  unfold addOne
  split_ifs with hw hq
  · simp only [cellEnergy, Fin.val_mk, Nat.cast_add, Nat.cast_one]
    have he : rawEnergy a b R C s.1.val s.2.1.val ((s.2.2.val:ℝ)+1) -
        rawEnergy a b R C s.1.val s.2.1.val s.2.2.val =
        14-12*(s.1.val:ℝ)*s.2.1.val := by unfold rawEnergy; ring
    rw [he]
    nlinarith [(coord_bounds s.1).1, (coord_bounds s.2.1).1]
  · simp only [cellEnergy, Fin.val_mk, Nat.cast_add, Nat.cast_one]
    have he : rawEnergy a b R C s.1.val ((s.2.1.val:ℝ)+1) s.2.2.val -
        rawEnergy a b R C s.1.val s.2.1.val s.2.2.val =
        14-12*(s.1.val:ℝ)*s.2.2.val + (2*(C+s.2.1.val+1)^2/b - 2*(C+s.2.1.val)^2/b) := by
      unfold rawEnergy; ring
    rw [he]
    have hi := pool_increment b C s.2.1.val hb hC (coord_bounds s.2.1).1 (by exact_mod_cast (by omega : s.2.1.val ≤ 1)) hCq
    nlinarith [(coord_bounds s.1).1, (coord_bounds s.2.2).1]
  · have hp : s.1.val < 2 := by unfold cellMass at hs; omega
    simp only [cellEnergy, Fin.val_mk, min_eq_right (by omega : s.1.val+1 ≤ 2), Nat.cast_add, Nat.cast_one]
    have he : rawEnergy a b R C ((s.1.val:ℝ)+1) s.2.1.val s.2.2.val -
        rawEnergy a b R C s.1.val s.2.1.val s.2.2.val =
        14-12*(s.2.1.val:ℝ)*s.2.2.val + (2*(R+s.1.val+1)^2/a - 2*(R+s.1.val)^2/a) := by
      unfold rawEnergy; ring
    rw [he]
    have hi := pool_increment a R s.1.val ha hR (coord_bounds s.1).1 (by exact_mod_cast (by omega : s.1.val ≤ 1)) hRp
    nlinarith [(coord_bounds s.2.1).1, (coord_bounds s.2.2).1]

private theorem block_nonempty (k : Fin 6) : (readoutBlock k).Nonempty := by
  apply Finset.card_pos.mp
  have := (readout_card_bounds k).1
  omega

def activeBlocks (budget : Fin 7) (x : Cell) : Finset (Fin 6) :=
  Finset.univ.filter (fun k => k.val+1 ≤ budget.val ∧ x ∈ readoutBlock k)

private theorem active_card (budget : Fin 7) (x : Cell) :
    (activeBlocks budget x).card ≤ 2 := by
  have hsub : activeBlocks budget x ⊆
      Finset.univ.filter (fun k : Fin 6 => x ∈ readoutBlock k) := by
    intro k hk
    simp only [activeBlocks, Finset.mem_filter, Finset.mem_univ, true_and] at hk ⊢
    exact hk.2
  apply le_trans (Finset.card_le_card hsub)
  rcases x with ⟨p,q,w⟩
  fin_cases p <;> fin_cases q <;> fin_cases w <;> decide

noncomputable def cellKernel (budget : Fin 7) (E : Cell → ℝ) (x y : Cell) : ℝ :=
  (if x = y then 1 - ((activeBlocks budget x).card:ℝ) / 2 else 0) +
    ∑ k ∈ activeBlocks budget x, blockKernel (readoutBlock k) (block_nonempty k) E x y / 2

private theorem off_block_zero (B : Finset Cell) (hB : B.Nonempty) (E : Cell → ℝ)
    (x y : Cell) (hy : y ∉ B) (hxy : x ≠ y) : blockKernel B hB E x y = 0 := by
  simp [blockKernel, blockRate, hy, hxy]

/-- Uniform cell/direction selection gives this kernel. Each adjacent readout
block is drawn with probability 1/2 when legal, otherwise the attempt stays.
The physical cell has 27 states but at most13 energies are examined per attempt.
Finite stock is the invariant budget N+stock, not a direction-dependent filter
inside a block. All old holding/release time certificates must be rederived. -/
theorem readout_cell_kernel_contract (budget : Fin 7) (E : Cell → ℝ) :
    (∀ k : Fin 6, 4 ≤ (readoutBlock k).card ∧ (readoutBlock k).card ≤ 13) ∧
    (∀ x y, 0 ≤ cellKernel budget E x y) ∧
    (∀ x, (∑ y, cellKernel budget E x y) = 1) ∧
    (∀ x y, (2:ℝ)^(-2*E x)*cellKernel budget E x y =
      (2:ℝ)^(-2*E y)*cellKernel budget E y x) ∧
    (∀ x y, cellMass x ≤ budget.val → budget.val < cellMass y →
      cellKernel budget E x y = 0) ∧
    (∀ x y, cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x →
      cellKernel budget E x y = 0) := by
  refine ⟨readout_card_bounds, ?_, ?_, ?_, ?_, ?_⟩
  · intro x y
    unfold cellKernel
    apply add_nonneg
    · split_ifs
      · have h : (activeBlocks budget x).card ≤ 2 := active_card budget x
        have hr : ((activeBlocks budget x).card:ℝ) ≤ 2 := by exact_mod_cast h
        linarith
      · norm_num
    · apply Finset.sum_nonneg
      intro k hk
      exact div_nonneg ((readout_block_kernel_contract _ (block_nonempty k) E).1 x y) (by norm_num)
  · intro x
    simp only [cellKernel, Finset.sum_add_distrib]
    rw [Finset.sum_comm]
    have hs : (∑ k ∈ activeBlocks budget x,
        ∑ y, blockKernel (readoutBlock k) (block_nonempty k) E x y / 2) =
        ((activeBlocks budget x).card:ℝ)/2 := by
      calc
        _ = ∑ _k ∈ activeBlocks budget x, (1:ℝ)/2 := by
          apply Finset.sum_congr rfl
          intro k _
          simp only [div_eq_mul_inv, ← Finset.sum_mul]
          rw [(readout_block_kernel_contract _ (block_nonempty k) E).2.1 x]
        _ = _ := by simp [nsmul_eq_mul, div_eq_mul_inv]
    rw [hs]
    simp
  · intro x y
    by_cases hxy : x = y
    · subst y; rfl
    · have hterm : ∀ k : Fin 6,
        (2:ℝ)^(-2*E x)*(if k ∈ activeBlocks budget x then
          blockKernel (readoutBlock k) (block_nonempty k) E x y / 2 else 0) =
        (2:ℝ)^(-2*E y)*(if k ∈ activeBlocks budget y then
          blockKernel (readoutBlock k) (block_nonempty k) E y x / 2 else 0) := by
        intro k
        have hdb := (readout_block_kernel_contract _ (block_nonempty k) E).2.2.1 x y
        by_cases hv : k.val+1 ≤ budget.val <;>
          by_cases hx : x ∈ readoutBlock k <;> by_cases hy : y ∈ readoutBlock k
        all_goals simp only [activeBlocks, Finset.mem_filter, Finset.mem_univ, true_and,
          hv, hx, hy, and_false, and_true, false_and, true_and, ite_false, ite_true, mul_zero]
        · simpa only [← mul_div_assoc] using congrArg (fun z : ℝ => z/2) hdb
        · rw [off_block_zero _ _ E x y hy hxy]; simp
        · rw [off_block_zero _ _ E y x hx (Ne.symm hxy)]; simp
      simp only [cellKernel, hxy, Ne.symm hxy, ite_false, zero_add]
      simp only [activeBlocks, Finset.sum_filter]
      rw [Finset.mul_sum, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro k _
      simpa only [activeBlocks, Finset.mem_filter, Finset.mem_univ, true_and] using hterm k
  · intro x y hx hy
    have hxy : x ≠ y := by intro h; subst y; omega
    simp only [cellKernel, hxy, ite_false, zero_add]
    apply Finset.sum_eq_zero
    intro k hk
    have ha : k.val+1 ≤ budget.val := (Finset.mem_filter.mp hk).2.1
    have hnot : y ∉ readoutBlock k := by
      simp only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and]
      omega
    rw [off_block_zero _ _ E x y hnot hxy, zero_div]
  · intro x y hgap
    have hxy : x ≠ y := by intro h; subst y; omega
    simp only [cellKernel, hxy, ite_false, zero_add]
    apply Finset.sum_eq_zero
    intro k hk
    have hx : x ∈ readoutBlock k := (Finset.mem_filter.mp hk).2.2
    have hnot : y ∉ readoutBlock k := by
      simp only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
      omega
    rw [off_block_zero _ _ E x y hnot hxy, zero_div]


noncomputable def tiltedEnergy (a b R C f : ℝ) (s : Cell) : ℝ :=
  cellEnergy a b R C s + f*cellMass s

private theorem remove_tilt (a b R C f : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (s : Cell) (hs : 0 < cellMass s) :
    tiltedEnergy a b R C f (removeOne s) - tiltedEnergy a b R C f s ≤ 34-f := by
  have he := remove_energy a b R C ha hb hR hC s hs
  have hm : (cellMass (removeOne s):ℝ)+1 = cellMass s := by exact_mod_cast remove_mass s hs
  have hmf := congrArg (fun z : ℝ => f*z) hm
  unfold tiltedEnergy
  nlinarith [hmf]

private theorem add_tilt (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (s : Cell) (hs : cellMass s < 6) :
    tiltedEnergy a b R C f (addOne s) - tiltedEnergy a b R C f s ≤ 22+f := by
  have he := add_energy a b R C ha hb hR hC hRp hCq s hs
  have hm : (cellMass (addOne s):ℝ) = cellMass s+1 := by exact_mod_cast add_mass s hs
  have hmf := congrArg (fun z : ℝ => f*z) hm
  unfold tiltedEnergy
  nlinarith [hmf]

private theorem floor_direction (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (k : Fin 6) :
    (34 < f → ∃ y ∈ readoutBlock k, cellMass y = k.val ∧
      blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y =
        1/((readoutBlock k).card:ℝ)) ∧
    (f < -22 → ∃ y ∈ readoutBlock k, cellMass y = k.val+1 ∧
      blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y =
        1/((readoutBlock k).card:ℝ)) := by
  let E := tiltedEnergy a b R C f
  obtain ⟨y, hy, hm⟩ := blockFloor_attained (readoutBlock k) (block_nonempty k) E
  have hmass : cellMass y = k.val ∨ cellMass y = k.val+1 := by
    simpa only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and] using hy
  have hrate : blockRate (readoutBlock k) (block_nonempty k) E y =
      1/((readoutBlock k).card:ℝ) := by simp [blockRate, hy, hm]
  constructor
  · intro hf
    refine ⟨y, hy, ?_, hrate⟩
    rcases hmass with h | h
    · exact h
    · have hs : 0 < cellMass y := by omega
      have ht : removeOne y ∈ readoutBlock k := by
        simp only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and]
        left
        have := remove_mass y hs
        omega
      have hh := blockFloor_le (readoutBlock k) (block_nonempty k) E (removeOne y) ht
      rw [← hm] at hh
      have he := remove_tilt a b R C f (by linarith) (by linarith) hR hC y hs
      change E (removeOne y)-E y ≤ 34-f at he
      linarith
  · intro hf
    refine ⟨y, hy, ?_, hrate⟩
    rcases hmass with h | h
    · have hs : cellMass y < 6 := by omega
      have ht : addOne y ∈ readoutBlock k := by
        simp only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and]
        right
        rw [add_mass y hs, h]
      have hh := blockFloor_le (readoutBlock k) (block_nonempty k) E (addOne y) ht
      rw [← hm] at hh
      have he := add_tilt a b R C f ha hb hR hC hRp hCq y hs
      change E (addOne y)-E y ≤ 22+f at he
      linarith
    · exact h

private theorem cell_block_lower (budget : Fin 7) (E : Cell → ℝ)
    (x y : Cell) (k : Fin 6) (hk : k ∈ activeBlocks budget x) :
    blockRate (readoutBlock k) (block_nonempty k) E y/2 ≤ cellKernel budget E x y := by
  have hx := (Finset.mem_filter.mp hk).2.2
  have hbase : (0:ℝ) ≤ if x = y then 1-((activeBlocks budget x).card:ℝ)/2 else 0 := by
    split_ifs
    · have hc : ((activeBlocks budget x).card:ℝ) ≤ 2 := by exact_mod_cast active_card budget x
      linarith
    · norm_num
  have hs := Finset.single_le_sum (fun j (_ : j ∈ activeBlocks budget x) =>
    div_nonneg ((readout_block_kernel_contract _ (block_nonempty j) E).1 x y)
      (by norm_num : (0:ℝ) ≤ 2)) hk
  have hr : blockRate (readoutBlock k) (block_nonempty k) E y ≤
      blockKernel (readoutBlock k) (block_nonempty k) E x y := by
    unfold blockKernel
    rw [ite_eq_left hx]
    have ha := (block_activity (readoutBlock k) (block_nonempty k) E).2
    split_ifs <;> linarith
  unfold cellKernel
  linarith

/-- In the SAME scaled-pool energy, a sufficiently strong signed force moves
an adjacent-block minimizer in its descent direction. There is a concrete
successor of probability >=1/26 per selected cell, independent of body size.
This is a transition/service bound, not a guarantee that an estimated gradient
is useful, nor a transfer of the old microscopic holding/release time bound. -/
theorem readout_force_response_contract (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (budget : Fin 7) (k : Fin 6) (hk : k.val+1 ≤ budget.val) (x : Cell) :
    (34 < f → cellMass x = k.val+1 → ∃ y, cellMass y+1 = cellMass x ∧
      1/26 ≤ cellKernel budget (tiltedEnergy a b R C f) x y) ∧
    (f < -22 → cellMass x = k.val → ∃ y, cellMass y = cellMass x+1 ∧
      1/26 ≤ cellKernel budget (tiltedEnergy a b R C f) x y) := by
  have hact : cellMass x = k.val ∨ cellMass x = k.val+1 → k ∈ activeBlocks budget x := by
    intro hx
    simp only [activeBlocks, readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨hk, hx⟩
  have hc0 : (0:ℝ) < (readoutBlock k).card := by exact_mod_cast (block_nonempty k).card_pos
  have hc13 : ((readoutBlock k).card:ℝ) ≤ 13 := by exact_mod_cast (readout_card_bounds k).2
  have hc : (1:ℝ)/26 ≤ (1/((readoutBlock k).card:ℝ))/2 := by
    have hd := one_div_le_one_div_of_le hc0 hc13
    norm_num at hd ⊢
    linarith
  constructor
  · intro hf hx
    obtain ⟨y, hy, hm, hr⟩ := (floor_direction a b R C f ha hb hR hC hRp hCq k).1 hf
    refine ⟨y, by omega, ?_⟩
    have hl := cell_block_lower budget (tiltedEnergy a b R C f) x y k (hact (Or.inr hx))
    rw [hr] at hl
    exact le_trans hc hl
  · intro hf hx
    obtain ⟨y, hy, hm, hr⟩ := (floor_direction a b R C f ha hb hR hC hRp hCq k).2 hf
    refine ⟨y, by omega, ?_⟩
    have hl := cell_block_lower budget (tiltedEnergy a b R C f) x y k (hact (Or.inl hx))
    rw [hr] at hl
    exact le_trans hc hl

#print axioms CGB.FiberOptimizerReview.readout_force_response_contract


def fullCell : Cell := (2,2,2)

private theorem full_of_mass (s : Cell) (hs : cellMass s = 6) : s = fullCell := by
  rcases s with ⟨p,q,w⟩
  have hp := p.isLt
  have hq := q.isLt
  have hw := w.isLt
  simp only [cellMass] at hs
  have hp2 : p = (2:Fin 3) := Fin.ext (by simp; omega)
  have hq2 : q = (2:Fin 3) := Fin.ext (by simp; omega)
  have hw2 : w = (2:Fin 3) := Fin.ext (by simp; omega)
  simp [hp2, hq2, hw2, fullCell]

private theorem full_repair_gap (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (s : Cell) (hs : cellMass s = 5) :
    tiltedEnergy a b R C f fullCell - tiltedEnergy a b R C f s ≤ -26+f := by
  have hp : 2*(R+2)^2/a - 2*(R+1)^2/a < 8 := by
    convert pool_increment a R 1 ha hR (by norm_num) (by norm_num) hRp using 1 <;> ring
  have hq : 2*(C+2)^2/b - 2*(C+1)^2/b < 8 := by
    convert pool_increment b C 1 hb hC (by norm_num) (by norm_num) hCq using 1 <;> ring
  rcases s with ⟨p,q,w⟩
  fin_cases p <;> fin_cases q <;> fin_cases w <;> norm_num [cellMass] at hs
  all_goals
    dsimp [tiltedEnergy, cellEnergy, rawEnergy, cellMass, fullCell]
    norm_num at hp hq ⊢
    nlinarith

/-- Existing content restores a missing physical quantum directly: no stored
reference value or loss test selects this successor. With stock available,
all three one-quantum defects have repair probability >=1/8 per selected cell
under the SAME block clock and scaled-pool H, uniformly in body size. Other
transitions and subsequent wear still run; this is not yet a lifetime bound. -/
theorem readout_regeneration_contract (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : f ≤ 1) (x : Cell) (hx : cellMass x = 5) :
    1/8 ≤ cellKernel (6:Fin 7) (tiltedEnergy a b R C f) x fullCell := by
  let E := tiltedEnergy a b R C f
  have hm : ∀ y ∈ readoutBlock (5:Fin 6), E fullCell ≤ E y := by
    intro y hy
    have hyN : cellMass y = 5 ∨ cellMass y = 6 := by
      simpa [readoutBlock] using hy
    rcases hyN with hyN | hyN
    · have hg := full_repair_gap a b R C f ha hb hR hC hRp hCq y hyN
      change E fullCell-E y ≤ -26+f at hg
      linarith
    · rw [full_of_mass y hyN]
  have hin : fullCell ∈ readoutBlock (5:Fin 6) := by decide
  have he : blockFloor (readoutBlock (5:Fin 6)) (block_nonempty _) E = E fullCell := by
    apply le_antisymm (blockFloor_le _ _ E fullCell hin)
    obtain ⟨y,hy,hyE⟩ := blockFloor_attained (readoutBlock (5:Fin 6)) (block_nonempty _) E
    rw [← hyE]
    exact hm y hy
  have hr : blockRate (readoutBlock (5:Fin 6)) (block_nonempty _) E fullCell = 1/4 := by
    rw [blockRate, ite_eq_left hin, he]
    have hc : (readoutBlock (5:Fin 6)).card = 4 := by decide
    norm_num [hc]
  have hk : (5:Fin 6) ∈ activeBlocks (6:Fin 7) x := by
    simp [activeBlocks, readoutBlock, hx]
  have hl := cell_block_lower (6:Fin 7) E x fullCell (5:Fin 6) hk
  rw [hr] at hl
  norm_num at hl
  exact hl

#print axioms CGB.FiberOptimizerReview.readout_regeneration_contract

#print axioms CGB.FiberOptimizerReview.readout_block_kernel_contract
#print axioms CGB.FiberOptimizerReview.readout_cell_kernel_contract
#print axioms remove_energy
#print axioms add_energy

/-! Same readout kernel, quantitative strong-force drift. The observation
functions below do not enter the optimizer state or modify its transitions. -/
private theorem mass_le_six (s : Cell) : cellMass s ≤ 6 := by
  have hp := s.1.isLt
  have hq := s.2.1.isLt
  have hw := s.2.2.isLt
  unfold cellMass
  omega

private theorem upper_block_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hf : 96 ≤ f) (k : Fin 6)
    (y : Cell) (hy : cellMass y = k.val+1) :
    blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y ≤
      (1:ℝ)/2^124 := by
  let E := tiltedEnergy a b R C f
  have hpos : 0 < cellMass y := by omega
  have hz : removeOne y ∈ readoutBlock k := by
    simp only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and]
    left
    have := remove_mass y hpos
    omega
  have hfloor := blockFloor_le (readoutBlock k) (block_nonempty k) E (removeOne y) hz
  have he := remove_tilt a b R C f (by linarith) (by linarith) hR hC y hpos
  have hgap : 62 ≤ E y-blockFloor (readoutBlock k) (block_nonempty k) E := by
    change E (removeOne y)-E y ≤ 34-f at he
    linarith
  have hin : y ∈ readoutBlock k := by simp [readoutBlock, hy]
  have hr : (2:ℝ)^(-2*(E y-blockFloor (readoutBlock k) (block_nonempty k) E)) ≤
      (1:ℝ)/2^124 := by
    calc
      _ ≤ (2:ℝ)^(-(124:ℝ)) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
      _ = _ := by norm_num [Real.rpow_neg]
  have hc : (1:ℝ) ≤ (readoutBlock k).card := by
    exact_mod_cast (by have := (readout_card_bounds k).1; omega : 1 ≤ (readoutBlock k).card)
  change blockRate _ _ E y ≤ _
  rw [blockRate, ite_eq_left hin]
  apply le_trans (div_le_self (by positivity) hc) hr

private theorem upward_cell_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hf : 96 ≤ f) (budget : Fin 7)
    (x y : Cell) (hy : cellMass x < cellMass y) :
    cellKernel budget (tiltedEnergy a b R C f) x y ≤ (1:ℝ)/2^124 := by
  let E := tiltedEnergy a b R C f
  have hxy : x ≠ y := by intro h; subst y; omega
  by_cases hnext : cellMass y = cellMass x+1
  · have hterm : ∀ k ∈ activeBlocks budget x,
        blockKernel (readoutBlock k) (block_nonempty k) E x y / 2 ≤ (1:ℝ)/2^124/2 := by
      intro k hk
      have hx := (Finset.mem_filter.mp hk).2.2
      have hxmass : cellMass x = k.val ∨ cellMass x = k.val+1 := by
        simpa only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and] using hx
      by_cases hin : y ∈ readoutBlock k
      · have hymass : cellMass y = k.val ∨ cellMass y = k.val+1 := by
          simpa only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and] using hin
        have hyup : cellMass y = k.val+1 := by omega
        have hr := upper_block_rate a b R C f ha hb hR hC hf k y hyup
        simp only [blockKernel, hx, hxy, ite_true, ite_false, zero_add]
        exact div_le_div_of_nonneg_right hr (by norm_num)
      · rw [off_block_zero _ _ E x y hin hxy]
        norm_num
    change cellKernel budget E x y ≤ _
    simp only [cellKernel, hxy, ite_false, zero_add]
    calc
      _ ≤ ∑ _k ∈ activeBlocks budget x, (1:ℝ)/2^124/2 := Finset.sum_le_sum hterm
      _ = ((activeBlocks budget x).card:ℝ)*((1:ℝ)/2^124/2) := by simp [nsmul_eq_mul]
      _ ≤ _ := by
        have hc : ((activeBlocks budget x).card:ℝ) ≤ 2 := by exact_mod_cast active_card budget x
        nlinarith
  · have hfar : cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x := by omega
    rw [(readout_cell_kernel_contract budget E).2.2.2.2.2 x y hfar]
    positivity

private theorem drift_up_bound (K : Cell → Cell → ℝ)
    (hn : ∀ x y, 0 ≤ K x y)
    (hfar : ∀ x y, cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x → K x y = 0)
    (x y : Cell) :
    K x y*((cellMass y:ℝ)-cellMass x) ≤ if cellMass x < cellMass y then K x y else 0 := by
  by_cases hu : cellMass x < cellMass y
  · rw [ite_eq_left hu]
    by_cases hnext : cellMass y = cellMass x+1
    · have he : (cellMass y:ℝ) = cellMass x+1 := by exact_mod_cast hnext
      rw [he]; ring_nf; exact le_refl _
    · rw [hfar x y (Or.inl (by omega))]; simp
  · rw [ite_eq_right hu]
    apply mul_nonpos_of_nonneg_of_nonpos (hn x y)
    have hm : (cellMass y:ℝ) ≤ cellMass x := by exact_mod_cast (by omega : cellMass y ≤ cellMass x)
    linarith

/-- Actual adjacent-readout drift under a sustained strong positive force.
Increasing readouts are exponentially rare; a nonempty cell has an explicit
lower-level successor. The state-dependent force may vary between funded
opportunities. This bound is size-independent per selected physical cell. -/
theorem readout_release_drift (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : 96 ≤ f) (budget : Fin 7) (x : Cell) (hx : cellMass x ≤ budget.val) :
    (∑ y, cellKernel budget (tiltedEnergy a b R C f) x y *
      ((cellMass y:ℝ)-cellMass x)) ≤
      (1:ℝ)/2^118 - (if 0 < cellMass x then 1/26 else 0) := by
  let K := cellKernel budget (tiltedEnergy a b R C f)
  have hn := (readout_cell_kernel_contract budget (tiltedEnergy a b R C f)).2.1
  have hfar := (readout_cell_kernel_contract budget (tiltedEnergy a b R C f)).2.2.2.2.2
  have hup : (∑ y : Cell, if cellMass x < cellMass y then K x y else 0) ≤ (1:ℝ)/2^118 := by
    calc
      _ ≤ ∑ _y : Cell, (1:ℝ)/2^124 := Finset.sum_le_sum (by
        intro y _
        split_ifs with hy
        · exact upward_cell_rate a b R C f ha hb hR hC hf budget x y hy
        · positivity)
      _ ≤ _ := by norm_num [Fintype.card_prod, Fintype.card_fin]
  by_cases hpos : 0 < cellMass x
  · let k : Fin 6 := ⟨cellMass x-1, by have := mass_le_six x; omega⟩
    have hk : k.val+1 ≤ budget.val := by dsimp [k]; omega
    have hN : cellMass x = k.val+1 := by dsimp [k]; omega
    obtain ⟨z, hz, hp⟩ := (readout_force_response_contract a b R C f ha hb hR hC hRp hCq
      budget k hk x).1 (by linarith) hN
    have hzup : ¬ cellMass x < cellMass z := by omega
    have hzm : (cellMass z:ℝ)+1 = cellMass x := by exact_mod_cast hz
    have hs : (∑ y, K x y*((cellMass y:ℝ)-cellMass x)) ≤
        (∑ y, if cellMass x < cellMass y then K x y else 0)-K x z := by
      calc
        _ ≤ ∑ y, ((if cellMass x < cellMass y then K x y else 0) -
            (if y = z then K x z else 0)) := Finset.sum_le_sum (by
          intro y _
          by_cases hyz : y = z
          · subst y; simp only [hzup, ite_false, ite_true, zero_sub]
            nlinarith
          · simp only [hyz, ite_false, sub_zero]
            exact drift_up_bound K hn hfar x y)
        _ = _ := by simp [Finset.sum_sub_distrib]
    change (∑ y, K x y*((cellMass y:ℝ)-cellMass x)) ≤ _
    rw [ite_eq_left hpos]
    linarith
  · change (∑ y, K x y*((cellMass y:ℝ)-cellMass x)) ≤ _
    rw [ite_eq_right hpos, sub_zero]
    exact le_trans (Finset.sum_le_sum (fun y _ => drift_up_bound K hn hfar x y)) hup

#print axioms CGB.FiberOptimizerReview.readout_release_drift

open CGB.ResourceReleaseReview

private noncomputable def pushCell (r : Cell → ℝ) (d : Cell → Option NativeState)
    (y : NativeState) : ℝ := ∑ c, if d c = some y then r c else 0

private theorem push_cell_value (r : Cell → ℝ) (d : Cell → Option NativeState)
    (v : NativeState → ℝ) :
    (∑ y, pushCell r d y*v y) = ∑ c, r c*(d c).elim 0 v := by
  simp only [pushCell, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro c _
  cases hd : d c with
  | none => simp
  | some z => simp

private def bodyBudget (x : NativeState) (i : Fin 4) : Fin 7 :=
  ⟨min 6 (cellMass (x.val i)+(12-nativeMass x.val)), by omega⟩

private theorem budget_holds (x : NativeState) (i : Fin 4) :
    cellMass (x.val i) ≤ (bodyBudget x i).val := by
  have := mass_le_six (x.val i)
  simp only [bodyBudget, Fin.val_mk]
  omega

private noncomputable def bodyCellKernel (f : Fin 4 → ℝ) (x : NativeState) (i : Fin 4) : Cell → Cell → ℝ :=
  cellKernel (bodyBudget x i) (tiltedEnergy 2 2
    (x.val (partner false i)).1.val (x.val (partner true i)).2.1.val (f i))

noncomputable def readoutBodyKernel (f : ℕ → NativeState → Fin 4 → ℝ)
    (t : ℕ) (x y : NativeState) : ℝ :=
  (1/4:ℝ)*∑ i, pushCell (bodyCellKernel (f t x) x i (x.val i))
    (fun c => nativePack (Function.update x.val i c)) y

private theorem pack_mass (b : NativeBody) (hcap : nativeMass b ≤ 12) :
    (nativePack b).elim 0 (fun x => (nativeMass x.val:ℝ)) = nativeMass b := by
  unfold nativePack
  split_ifs with h
  · rfl
  · have hz : nativeMass b = 0 := by omega
    simp [hz]

private theorem weighted_update_mass (f : Fin 4 → ℝ) (x : NativeState) (i : Fin 4) (c : Cell) :
    bodyCellKernel f x i (x.val i) c *
      (nativePack (Function.update x.val i c)).elim 0 (fun y => (nativeMass y.val:ℝ)) =
    bodyCellKernel f x i (x.val i) c *
      ((nativeMass x.val:ℝ)+(cellMass c:ℝ)-cellMass (x.val i)) := by
  have hu := sum_update_nat x.val i c
  change nativeMass (Function.update x.val i c)+cellMass (x.val i) =
    nativeMass x.val+cellMass c at hu
  by_cases hc : cellMass c ≤ (bodyBudget x i).val
  · have hb : nativeMass (Function.update x.val i c) ≤ 12 := by
      have hx := x.property.2
      have hb : (bodyBudget x i).val ≤ cellMass (x.val i)+(12-nativeMass x.val) := by
        exact Nat.min_le_right _ _
      omega
    rw [pack_mass _ hb]
    have he : (nativeMass (Function.update x.val i c):ℝ)+(cellMass (x.val i):ℝ) =
        nativeMass x.val+(cellMass c:ℝ) := by exact_mod_cast hu
    congr 1
    linarith
  · have hz := (readout_cell_kernel_contract (bodyBudget x i) (tiltedEnergy 2 2
        (x.val (partner false i)).1.val (x.val (partner true i)).2.1.val (f i))).2.2.2.2.1
        (x.val i) c (budget_holds x i) (by omega)
    change bodyCellKernel f x i (x.val i) c = 0 at hz
    rw [hz]; simp

private theorem body_kernel_contract (f : ℕ → NativeState → Fin 4 → ℝ) :
    (∀ t x y, 0 ≤ readoutBodyKernel f t x y) ∧
    (∀ t x, (∑ y, readoutBodyKernel f t x y) ≤ 1) ∧
    (∀ t x, (∑ y, readoutBodyKernel f t x y*(nativeMass y.val:ℝ)) =
      (nativeMass x.val:ℝ)+(1/4:ℝ)*∑ i, ∑ c,
        bodyCellKernel (f t x) x i (x.val i) c*((cellMass c:ℝ)-cellMass (x.val i))) := by
  have hn (t : ℕ) (x : NativeState) (i : Fin 4) :=
    (readout_cell_kernel_contract (bodyBudget x i) (tiltedEnergy 2 2
      (x.val (partner false i)).1.val (x.val (partner true i)).2.1.val (f t x i))).2.1
  have hr (t : ℕ) (x : NativeState) (i : Fin 4) :
      (∑ c, bodyCellKernel (f t x) x i (x.val i) c) = 1 := by
    unfold bodyCellKernel
    exact (readout_cell_kernel_contract (bodyBudget x i) (tiltedEnergy 2 2
      (x.val (partner false i)).1.val (x.val (partner true i)).2.1.val (f t x i))).2.2.1 (x.val i)
  have heq (t : ℕ) (x : NativeState) (v : NativeState → ℝ) :
      (∑ y, readoutBodyKernel f t x y*v y) =
      (1/4:ℝ)*∑ i, ∑ c, bodyCellKernel (f t x) x i (x.val i) c*
        (nativePack (Function.update x.val i c)).elim 0 v := by
    simp only [readoutBodyKernel, mul_assoc, Finset.mul_sum, Finset.sum_mul]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    rw [← Finset.mul_sum, push_cell_value, Finset.mul_sum]
  refine ⟨?_, ?_, ?_⟩
  · intro t x y
    apply mul_nonneg (by norm_num)
    apply Finset.sum_nonneg
    intro i _
    apply Finset.sum_nonneg
    intro c _
    split_ifs
    · exact hn t x i (x.val i) c
    · exact le_refl 0
  · intro t x
    have h := heq t x (fun _ => 1)
    simp only [mul_one] at h
    rw [h]
    calc
      _ ≤ (1/4:ℝ)*∑ _i : Fin 4, 1 := mul_le_mul_of_nonneg_left (Finset.sum_le_sum (by
        intro i _
        calc
          _ ≤ ∑ c, bodyCellKernel (f t x) x i (x.val i) c*1 := Finset.sum_le_sum (by
            intro c _
            apply mul_le_mul_of_nonneg_left _ (hn t x i (x.val i) c)
            cases nativePack (Function.update x.val i c) <;> norm_num)
          _ = 1 := by simpa only [mul_one] using hr t x i)) (by norm_num)
      _ = 1 := by norm_num
  · intro t x
    rw [heq]
    simp_rw [weighted_update_mass]
    have hi (i : Fin 4) :
        (∑ c, bodyCellKernel (f t x) x i (x.val i) c*
          ((nativeMass x.val:ℝ)+(cellMass c:ℝ)-cellMass (x.val i))) =
        (nativeMass x.val:ℝ)+∑ c, bodyCellKernel (f t x) x i (x.val i) c*
          ((cellMass c:ℝ)-cellMass (x.val i)) := by
      simp_rw [show ∀ c : Cell, (nativeMass x.val:ℝ)+(cellMass c:ℝ)-cellMass (x.val i) =
          (nativeMass x.val:ℝ)+((cellMass c:ℝ)-cellMass (x.val i)) from fun c => by ring,
        mul_add, Finset.sum_add_distrib]
      rw [← Finset.sum_mul, hr t x i, one_mul]
    simp_rw [hi]
    simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring

private theorem body_release_row (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) (t : ℕ) (x : NativeState) :
    (∑ y, readoutBodyKernel f t x y*(nativeMass y.val:ℝ)) ≤
      (1279/1280:ℝ)*(nativeMass x.val:ℝ) := by
  have hd (i : Fin 4) := readout_release_drift 2 2
    (x.val (partner false i)).1.val (x.val (partner true i)).2.1.val (f t x i)
    (by norm_num) (by norm_num) (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    (by have := (coord_bounds (x.val (partner false i)).1).2; linarith)
    (by have := (coord_bounds (x.val (partner true i)).2.1).2; linarith)
    (hf t x i) (bodyBudget x i) (x.val i) (budget_holds x i)
  obtain ⟨j,hj⟩ := native_nonempty x
  have hjN : 0 < cellMass (x.val j) := hj
  have hshare : (1/26:ℝ) ≤ ∑ i : Fin 4, if 0 < cellMass (x.val i) then (1/26:ℝ) else 0 := by
    have hs := Finset.single_le_sum (fun i (_ : i ∈ Finset.univ) =>
      show (0:ℝ) ≤ if 0 < cellMass (x.val i) then (1/26:ℝ) else 0 by split_ifs <;> norm_num)
      (Finset.mem_univ j)
    simpa only [ite_eq_left hjN] using hs
  have hs := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hd i)
  change (∑ i, ∑ c, bodyCellKernel (f t x) x i (x.val i) c*
    ((cellMass c:ℝ)-cellMass (x.val i))) ≤ _ at hs
  simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul] at hs
  rw [(body_kernel_contract f).2.2 t x]
  have hm : (nativeMass x.val:ℝ) ≤ 12 := by exact_mod_cast x.property.2
  norm_num at hs ⊢
  nlinarith

private theorem damage_mass_contract :
    (∀ x y, 0 ≤ nativeDamageKernel x y) ∧
    (∀ x, (∑ y, nativeDamageKernel x y) ≤ 1) ∧
    (∀ x, (∑ y, nativeDamageKernel x y*(nativeMass y.val:ℝ)) ≤ nativeMass x.val) := by
  have hc := jumpKernel_contract (fun _ : Fin 4 × Fin 3 => (1/12:ℝ))
    (fun _ _ => (1/1024:ℝ))
    (fun x : NativeState => fun e => nativePack (nativeDamageCandidate x.val e))
    (by intro _; norm_num) (by norm_num [Fintype.sum_prod_type, Fin.sum_univ_succ])
    (by intro _ _; norm_num)
  refine ⟨hc.1, ?_, ?_⟩
  · intro x
    change (∑ y, jumpKernel _ _ _ x y) ≤ 1
    have he := hc.2 (fun _ => (1:ℝ)) x
    simp only [mul_one] at he
    rw [he]
    have hs : (∑ e : Fin 4 × Fin 3, (1/12:ℝ)*(1/1024)*
        ((nativePack (nativeDamageCandidate x.val e)).elim 0 (fun _ => (1:ℝ))-1)) ≤ 0 := by
      apply Finset.sum_nonpos
      intro e _
      cases nativePack (nativeDamageCandidate x.val e) <;> norm_num
    linarith
  intro x
  change (∑ y, jumpKernel _ _ _ x y*(nativeMass y.val:ℝ)) ≤ _
  rw [hc.2]
  have hs : (∑ e : Fin 4 × Fin 3, (1/12:ℝ)*(1/1024)*
      ((nativePack (nativeDamageCandidate x.val e)).elim 0 (fun y => (nativeMass y.val:ℝ)) -
        (nativeMass x.val:ℝ))) ≤ 0 := by
    apply Finset.sum_nonpos
    intro e _
    have hd := (native_damage_effect x.val e).1
    rw [pack_mass _ (hd.trans x.property.2)]
    have hm : (nativeMass (nativeDamageCandidate x.val e):ℝ) ≤ nativeMass x.val := by exact_mod_cast hd
    nlinarith
  linarith

noncomputable def readoutOpenKernel (f : ℕ → NativeState → Fin 4 → ℝ)
    (t : ℕ) (x y : NativeState) : ℝ := ∑ z, nativeDamageKernel x z*readoutBodyKernel f t z y

/-- First empty hit for the actual four-cell readout clock, including ALL
legal nonempty bodies, finite stock and the SAME continuing external wear.
Forces may vary by time/state/cell but remain>=96. No assumed row certificate,
retention property or target-directed repair selector is used. Fee is13 base
units per counted funded opportunity, so old microscopic budgets do not transfer. -/
theorem readout_release_budget (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) :
    (∀ t x y, 0 ≤ readoutOpenKernel f t x y) ∧
    (∀ t x, (∑ y, readoutOpenKernel f t x y) ≤ 1) ∧
    (∀ T start x, CGB.OptimizerContentReview.kernelSurvival (readoutOpenKernel f) T start x ≤
      (1279/1280:ℝ)^T*(nativeMass x.val:ℝ)) := by
  have hn := (body_kernel_contract f).1
  have hd := damage_mass_contract
  have hkn : ∀ t x y, 0 ≤ readoutOpenKernel f t x y := by
    intro t x y
    apply Finset.sum_nonneg
    intro z _
    exact mul_nonneg (hd.1 x z) (hn t z y)
  refine ⟨hkn, ?_, ?_⟩
  · intro t x
    have heq : (∑ y, readoutOpenKernel f t x y) =
        ∑ z, nativeDamageKernel x z*(∑ y, readoutBodyKernel f t z y) := by
      simp only [readoutOpenKernel, Finset.mul_sum]
      rw [Finset.sum_comm]
    rw [heq]
    calc
      _ ≤ ∑ z, nativeDamageKernel x z*1 := Finset.sum_le_sum (by
        intro z _
        exact mul_le_mul_of_nonneg_left ((body_kernel_contract f).2.1 t z) (hd.1 x z))
      _ ≤ 1 := by simpa only [mul_one] using hd.2.1 x
  apply (CGB.OptimizerContentReview.kinetic_survival_envelope (readoutOpenKernel f)
    (fun x => (nativeMass x.val:ℝ)) (1279/1280) hkn (by norm_num)).2
  · intro x
    exact_mod_cast x.property.1
  · intro t x
    have heq : (∑ y, readoutOpenKernel f t x y*(nativeMass y.val:ℝ)) =
        ∑ z, nativeDamageKernel x z*(∑ y, readoutBodyKernel f t z y*(nativeMass y.val:ℝ)) := by
      simp only [readoutOpenKernel, Finset.sum_mul, Finset.mul_sum, mul_assoc]
      rw [Finset.sum_comm]
    rw [heq]
    calc
      _ ≤ ∑ z, nativeDamageKernel x z*((1279/1280:ℝ)*(nativeMass z.val:ℝ)) :=
        Finset.sum_le_sum (by intro z _; exact mul_le_mul_of_nonneg_left (body_release_row f hf t z) (hd.1 x z))
      _ = (1279/1280:ℝ)*∑ z, nativeDamageKernel x z*(nativeMass z.val:ℝ) := by
        rw [Finset.mul_sum]; apply Finset.sum_congr rfl; intro z _; ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (hd.2.2 x) (by norm_num)

#print axioms CGB.FiberOptimizerReview.readout_release_budget


/-! Actual task risk: reuse the current readout kernel and finite-gradient
error model. These proof observers never select or veto a runtime proposal. -/
private theorem lower_block_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : f ≤ -96) (k : Fin 6) (y : Cell) (hy : cellMass y = k.val) :
    blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y ≤
      (1:ℝ)/2^124 := by
  let E := tiltedEnergy a b R C f
  have hpos : cellMass y < 6 := by omega
  have hz : addOne y ∈ readoutBlock k := by
    simp only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and]
    right
    rw [add_mass y hpos, hy]
  have hfloor := blockFloor_le (readoutBlock k) (block_nonempty k) E (addOne y) hz
  have he := add_tilt a b R C f ha hb hR hC hRp hCq y hpos
  have hgap : 62 ≤ E y-blockFloor (readoutBlock k) (block_nonempty k) E := by
    change E (addOne y)-E y ≤ 22+f at he
    linarith
  have hin : y ∈ readoutBlock k := by simp [readoutBlock, hy]
  have hr : (2:ℝ)^(-2*(E y-blockFloor (readoutBlock k) (block_nonempty k) E)) ≤
      (1:ℝ)/2^124 := by
    calc
      _ ≤ (2:ℝ)^(-(124:ℝ)) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
      _ = _ := by norm_num [Real.rpow_neg]
  have hc : (1:ℝ) ≤ (readoutBlock k).card := by
    exact_mod_cast (by have := (readout_card_bounds k).1; omega : 1 ≤ (readoutBlock k).card)
  change blockRate _ _ E y ≤ _
  rw [blockRate, ite_eq_left hin]
  exact le_trans (div_le_self (by positivity) hc) hr

private theorem downward_cell_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : f ≤ -96) (budget : Fin 7) (x y : Cell) (hy : cellMass y < cellMass x) :
    cellKernel budget (tiltedEnergy a b R C f) x y ≤ (1:ℝ)/2^124 := by
  let E := tiltedEnergy a b R C f
  have hxy : x ≠ y := by intro h; subst y; omega
  by_cases hnext : cellMass y+1 = cellMass x
  · have hterm : ∀ k ∈ activeBlocks budget x,
        blockKernel (readoutBlock k) (block_nonempty k) E x y / 2 ≤ (1:ℝ)/2^124/2 := by
      intro k hk
      have hx := (Finset.mem_filter.mp hk).2.2
      have hxmass : cellMass x = k.val ∨ cellMass x = k.val+1 := by
        simpa only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and] using hx
      by_cases hin : y ∈ readoutBlock k
      · have hymass : cellMass y = k.val ∨ cellMass y = k.val+1 := by
          simpa only [readoutBlock, Finset.mem_filter, Finset.mem_univ, true_and] using hin
        have hylow : cellMass y = k.val := by omega
        have hr := lower_block_rate a b R C f ha hb hR hC hRp hCq hf k y hylow
        simp only [blockKernel, hx, hxy, ite_true, ite_false, zero_add]
        exact div_le_div_of_nonneg_right hr (by norm_num)
      · rw [off_block_zero _ _ E x y hin hxy]
        norm_num
    change cellKernel budget E x y ≤ _
    simp only [cellKernel, hxy, ite_false, zero_add]
    calc
      _ ≤ ∑ _k ∈ activeBlocks budget x, (1:ℝ)/2^124/2 := Finset.sum_le_sum hterm
      _ = ((activeBlocks budget x).card:ℝ)*((1:ℝ)/2^124/2) := by simp [nsmul_eq_mul]
      _ ≤ _ := by
        have hc : ((activeBlocks budget x).card:ℝ) ≤ 2 := by exact_mod_cast active_card budget x
        nlinarith
  · have hfar : cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x := by omega
    rw [(readout_cell_kernel_contract budget E).2.2.2.2.2 x y hfar]
    positivity

noncomputable def moveProbability (K : Cell → Cell → ℝ) (x : Cell) (up : Bool) : ℝ :=
  ∑ y, if (if up then cellMass y = cellMass x+1 else cellMass y+1 = cellMass x) then K x y else 0

private theorem move_lower (K : Cell → Cell → ℝ) (hn : ∀ x y, 0 ≤ K x y)
    (x z : Cell) (up : Bool)
    (hz : if up then cellMass z = cellMass x+1 else cellMass z+1 = cellMass x) :
    K x z ≤ moveProbability K x up := by
  have hs := Finset.single_le_sum (fun y (_ : y ∈ Finset.univ) =>
    show (0:ℝ) ≤ if (if up then cellMass y = cellMass x+1 else cellMass y+1 = cellMass x) then K x y else 0
    by
      by_cases hy : (if up then cellMass y = cellMass x+1 else cellMass y+1 = cellMass x)
      · rw [ite_eq_left hy]; exact hn x y
      · rw [ite_eq_right hy]) (Finset.mem_univ z)
  simpa only [moveProbability, ite_eq_left hz] using hs

private theorem move_small (K : Cell → Cell → ℝ) (x : Cell) (up : Bool)
    (hb : ∀ y, (if up then cellMass y = cellMass x+1 else cellMass y+1 = cellMass x) →
      K x y ≤ (1:ℝ)/2^124) : moveProbability K x up ≤ (1:ℝ)/2^118 := by
  unfold moveProbability
  calc
    _ ≤ ∑ _y : Cell, (1:ℝ)/2^124 := Finset.sum_le_sum (by
      intro y _
      by_cases hy : (if up then cellMass y = cellMass x+1 else cellMass y+1 = cellMass x)
      · rw [ite_eq_left hy]; exact hb y hy
      · rw [ite_eq_right hy]; positivity)
    _ ≤ _ := by norm_num [Fintype.card_prod, Fintype.card_fin]

private theorem finite_risk_moves (K : Cell → Cell → ℝ) (x : Cell) (risk : Cell → ℝ)
    (q estimate error L actual : ℝ) (positive : Bool)
    (hq : 0 ≤ q) (herror : |actual-(if positive then estimate else -estimate)| ≤ error)
    (hn : ∀ x y, 0 ≤ K x y)
    (hfar : ∀ x y, cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x → K x y = 0)
    (hmodel : ∀ y, risk y-risk x ≤ actual*q*((cellMass y:ℝ)-cellMass x) +
      L*q^2/2*((cellMass y:ℝ)-cellMass x)^2) :
    let D := q*estimate
    let c := q*error+L*q^2/2
    (∑ y, K x y*(risk y-risk x)) ≤
      -(D-c)*moveProbability K x (!positive)+(D+c)*moveProbability K x positive := by
  dsimp only
  have he := abs_le.mp herror
  have hterm (y : Cell) : K x y*(risk y-risk x) ≤
      -(q*estimate-(q*error+L*q^2/2))*(if (if !positive then cellMass y = cellMass x+1
          else cellMass y+1 = cellMass x) then K x y else 0) +
      (q*estimate+(q*error+L*q^2/2))*(if (if positive then cellMass y = cellMass x+1
          else cellMass y+1 = cellMass x) then K x y else 0) := by
    have hm := hmodel y
    by_cases hd : cellMass y+1 = cellMass x
    · have heq : (cellMass y:ℝ)-cellMass x = -1 := by
        have : (cellMass y:ℝ)+1 = cellMass x := by exact_mod_cast hd
        linarith
      have hnu : ¬cellMass y = cellMass x+1 := by omega
      rw [heq] at hm
      norm_num at hm
      have hg := mul_nonneg hq (sub_nonneg.mpr he.1)
      have hb : risk y-risk x ≤ -q*(if positive then estimate else -estimate)+q*error+L*q^2/2 := by
        nlinarith
      have hh := mul_le_mul_of_nonneg_left hb (hn x y)
      cases positive <;> simp [hd, hnu] at hh ⊢ <;> nlinarith
    · by_cases hu : cellMass y = cellMass x+1
      · have heq : (cellMass y:ℝ)-cellMass x = 1 := by
          have : (cellMass y:ℝ) = cellMass x+1 := by exact_mod_cast hu
          linarith
        rw [heq] at hm
        norm_num at hm
        have hg := mul_nonneg hq (sub_nonneg.mpr he.2)
        have hb : risk y-risk x ≤ q*(if positive then estimate else -estimate)+q*error+L*q^2/2 := by
          nlinarith
        have hh := mul_le_mul_of_nonneg_left hb (hn x y)
        cases positive <;>
          simp only [Bool.not_false, Bool.not_true, Bool.false_eq_true, Bool.true_eq, ite_true, ite_false] at hh ⊢ <;>
          rw [ite_eq_left hu, ite_eq_right hd] <;> nlinarith
      · by_cases hz : cellMass y = cellMass x
        · have hnu : ¬cellMass y = cellMass x+1 := hu
          have heq : (cellMass y:ℝ)-cellMass x = 0 := by rw [hz]; ring
          rw [heq] at hm
          norm_num at hm
          have hh := mul_le_mul_of_nonneg_left hm (hn x y)
          cases positive <;> simp [hd, hnu] <;> nlinarith
        · rw [hfar x y (by omega)]
          simp
  have hs := Finset.sum_le_sum (fun y (_ : y ∈ Finset.univ) => hterm y)
  simpa only [moveProbability, Finset.sum_add_distrib, Finset.mul_sum] using hs

#print axioms finite_risk_moves

/-- The actual task-risk bridge for this SAME stochastic kernel. A local
Taylor upper model and a gradient-error envelope are explicit properties of
the objective/observation, not an assumed descent or a runtime acceptance
criterion. No loss oracle, curvature array, reference weight or gate is added.
At a feasible strong signed signal the source probabilities close the gain
inequality; the final clause allows50% gradient error with L*q<=estimate/2.
The force is in the existing port's energy units, so a caller must also verify
its positive unit conversion and stale/noisy observation envelope. -/
theorem readout_task_risk_contract (a b R C f q estimate error L actual : ℝ)
    (ha : 1 ≤ a) (hb : 1 ≤ b) (hR : 0 ≤ R) (hC : 0 ≤ C)
    (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b) (hf : 96 ≤ f)
    (hq : 0 ≤ q) (he : 0 ≤ estimate) (herr : 0 ≤ error) (hL : 0 ≤ L)
    (hbudget : q*error+L*q^2/2 ≤ q*estimate)
    (budget : Fin 7) (x : Cell) (hx : cellMass x ≤ budget.val) (risk : Cell → ℝ)
    (hmodel : ∀ y, risk y-risk x ≤ actual*q*((cellMass y:ℝ)-cellMass x) +
      L*q^2/2*((cellMass y:ℝ)-cellMass x)^2) :
    let D := q*estimate
    let c := q*error+L*q^2/2
    let B := -(D-c)/26+(D+c)/2^118
    (0 < cellMass x → |actual-estimate| ≤ error →
      (∑ y, cellKernel budget (tiltedEnergy a b R C f) x y*(risk y-risk x)) ≤ B) ∧
    (cellMass x < budget.val → |actual+estimate| ≤ error →
      (∑ y, cellKernel budget (tiltedEnergy a b R C (-f)) x y*(risk y-risk x)) ≤ B) ∧
    (error ≤ estimate/2 → L*q ≤ estimate/2 → B ≤ -(q*estimate)/208) := by
  dsimp only
  have hc : 0 ≤ q*error+L*q^2/2 := by positivity
  have hD : 0 ≤ q*estimate := mul_nonneg hq he
  have close (K : Cell → Cell → ℝ) (positive : Bool)
      (hg : 1/26 ≤ moveProbability K x (!positive))
      (hb : moveProbability K x positive ≤ (1:ℝ)/2^118)
      (hterm : (∑ y, K x y*(risk y-risk x)) ≤
        -(q*estimate-(q*error+L*q^2/2))*moveProbability K x (!positive)+
          (q*estimate+(q*error+L*q^2/2))*moveProbability K x positive) :
      (∑ y, K x y*(risk y-risk x)) ≤
        -(q*estimate-(q*error+L*q^2/2))/26+(q*estimate+(q*error+L*q^2/2))/2^118 := by
    have hgood := mul_le_mul_of_nonpos_left hg (by linarith : -(q*estimate-(q*error+L*q^2/2)) ≤ 0)
    have hbad := mul_le_mul_of_nonneg_left hb (by linarith : 0 ≤ q*estimate+(q*error+L*q^2/2))
    nlinarith
  refine ⟨?_, ?_, ?_⟩
  · intro hpos herror
    let E := tiltedEnergy a b R C f
    let K := cellKernel budget E
    have hn := (readout_cell_kernel_contract budget E).2.1
    have hfar := (readout_cell_kernel_contract budget E).2.2.2.2.2
    let k : Fin 6 := ⟨cellMass x-1, by have := mass_le_six x; omega⟩
    have hk : k.val+1 ≤ budget.val := by dsimp [k]; omega
    have hN : cellMass x = k.val+1 := by dsimp [k]; omega
    obtain ⟨z,hz,hp⟩ := (readout_force_response_contract a b R C f ha hb hR hC hRp hCq
      budget k hk x).1 (by linarith) hN
    have hg : (1/26:ℝ) ≤ moveProbability K x false := le_trans hp (move_lower K hn x z false hz)
    have hb : moveProbability K x true ≤ (1:ℝ)/2^118 := move_small K x true (by
      intro y hy
      exact upward_cell_rate a b R C f ha hb hR hC hf budget x y (by
        change cellMass y = cellMass x+1 at hy
        omega))
    apply close K true hg hb
    exact finite_risk_moves K x risk q estimate error L actual true hq herror hn hfar hmodel
  · intro hroom herror
    let E := tiltedEnergy a b R C (-f)
    let K := cellKernel budget E
    have hn := (readout_cell_kernel_contract budget E).2.1
    have hfar := (readout_cell_kernel_contract budget E).2.2.2.2.2
    let k : Fin 6 := ⟨cellMass x, by have := budget.isLt; omega⟩
    have hk : k.val+1 ≤ budget.val := by dsimp [k]; omega
    have hN : cellMass x = k.val := rfl
    obtain ⟨z,hz,hp⟩ := (readout_force_response_contract a b R C (-f) ha hb hR hC hRp hCq
      budget k hk x).2 (by linarith) hN
    have hg : (1/26:ℝ) ≤ moveProbability K x true := le_trans hp (move_lower K hn x z true hz)
    have hb : moveProbability K x false ≤ (1:ℝ)/2^118 := move_small K x false (by
      intro y hy
      exact downward_cell_rate a b R C (-f) ha hb hR hC hRp hCq (by linarith) budget x y (by
        change cellMass y+1 = cellMass x at hy
        omega))
    apply close K false hg hb
    apply finite_risk_moves K x risk q estimate error L actual false hq _ hn hfar hmodel
    simpa using herror
  · intro he2 hL2
    have heq := mul_le_mul_of_nonneg_left he2 hq
    have hLq := mul_le_mul_of_nonneg_left hL2 hq
    have hcD : q*error+L*q^2/2 ≤ 3*(q*estimate)/4 := by nlinarith
    have heps : (1:ℝ)/2^118 ≤ 1/100000 := by norm_num
    have htail := mul_le_mul_of_nonneg_left heps (by linarith : 0 ≤ q*estimate+(q*error+L*q^2/2))
    nlinarith

/-- A quantified noise floor is unavoidable for this autonomous finite-rate
kernel: at the minimum of the scalar loss theta^2 and supplied gradient0,
there remains a strictly positive expected loss increment. This is an exact
analytic counterexample, not a capability task or a new optimizer mechanism. -/
theorem autonomous_task_noise_floor :
    let E := tiltedEnergy 1 1 0 0 0
    let x : Cell := (0,0,0)
    (1:ℝ)/2^31 ≤ ∑ y, cellKernel (1:Fin 7) E x y*(cellMass y:ℝ)^2 := by
  dsimp only
  let E := tiltedEnergy 1 1 0 0 0
  let x : Cell := (0,0,0)
  let z : Cell := (0,0,1)
  have hx : x ∈ readoutBlock (0:Fin 6) := by decide
  have hm : ∀ y ∈ readoutBlock (0:Fin 6), 0 ≤ E y := by
    intro y hy
    have hN : cellMass y = 0 ∨ cellMass y = 1 := by simpa [readoutBlock] using hy
    have hz : y.1.val = 0 ∨ y.2.1.val = 0 ∨ y.2.2.val = 0 := by
      unfold cellMass at hN
      omega
    have hp : (y.1.val:ℝ)*y.2.1.val*y.2.2.val = 0 := by
      rcases hz with h | h | h <;> simp [h]
    have h1 := Nat.cast_nonneg (α := ℝ) y.1.val
    have h2 := Nat.cast_nonneg (α := ℝ) y.2.1.val
    have h3 := Nat.cast_nonneg (α := ℝ) y.2.2.val
    dsimp [E, tiltedEnergy, cellEnergy, rawEnergy]
    norm_num
    nlinarith [sq_nonneg (y.1.val:ℝ), sq_nonneg (y.2.1.val:ℝ)]
  have hfloor : blockFloor (readoutBlock (0:Fin 6)) (block_nonempty _) E = 0 := by
    apply le_antisymm
    · have h := blockFloor_le (readoutBlock (0:Fin 6)) (block_nonempty _) E x hx
      simpa [E,x,tiltedEnergy,cellEnergy,rawEnergy,cellMass] using h
    · obtain ⟨y,hy,he⟩ := blockFloor_attained (readoutBlock (0:Fin 6)) (block_nonempty _) E
      rw [← he]; exact hm y hy
  have hz : z ∈ readoutBlock (0:Fin 6) := by decide
  have hc : (readoutBlock (0:Fin 6)).card = 4 := by decide
  have hrate : blockRate (readoutBlock (0:Fin 6)) (block_nonempty _) E z = (1:ℝ)/2^30 := by
    rw [blockRate, ite_eq_left hz, hfloor]
    norm_num [E,z,tiltedEnergy,cellEnergy,rawEnergy,cellMass,hc,Real.rpow_neg]
  have hact : (0:Fin 6) ∈ activeBlocks (1:Fin 7) x := by decide
  have hprob := cell_block_lower (1:Fin 7) E x z (0:Fin 6) hact
  rw [hrate] at hprob
  have hs := Finset.single_le_sum (fun y (_ : y ∈ Finset.univ) =>
    mul_nonneg ((readout_cell_kernel_contract (1:Fin 7) E).2.1 x y) (sq_nonneg (cellMass y:ℝ)))
    (Finset.mem_univ z)
  have hNz : (cellMass z:ℝ)^2 = 1 := by norm_num [z,cellMass]
  rw [hNz, mul_one] at hs
  change (1:ℝ)/2^31 ≤ ∑ y, cellKernel (1:Fin 7) E x y*(cellMass y:ℝ)^2
  calc
    _ = ((1:ℝ)/2^30)/2 := by norm_num
    _ ≤ cellKernel (1:Fin 7) E x z := hprob
    _ ≤ _ := hs

#print axioms CGB.FiberOptimizerReview.readout_task_risk_contract
#print axioms CGB.FiberOptimizerReview.autonomous_task_noise_floor


/-! Uncut rectangular population. These are mathematical views of the SAME
packed fields and conserved stock, not a new runtime carrier or state array.
Zero-mass bodies are ordinary states and may grow again when stock permits. -/
abbrev Site (rows cols : ℕ) := Fin rows × Fin cols
abbrev Population (rows cols : ℕ) := Site rows cols → Cell

def populationMass {r c : ℕ} (b : Population r c) : ℕ := ∑ i, cellMass (b i)

noncomputable def populationDistance {r c : ℕ} (b y : Population r c) : ℝ :=
  ∑ i, |(cellMass (b i):ℝ)-cellMass (y i)|

def populationBudget {r c : ℕ} (supply : ℕ) (b : Population r c) (i : Site r c) : Fin 7 :=
  ⟨min 6 (cellMass (b i)+(supply-populationMass b)), by omega⟩

def outsideRow {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :=
  ∑ j ∈ Finset.univ.erase i.2, (b (i.1,j)).1.val

def outsideCol {r c : ℕ} (b : Population r c) (i : Site r c) : ℕ :=
  ∑ j ∈ Finset.univ.erase i.1, (b (j,i.2)).2.1.val

private theorem outside_bounds {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (b : Population r c) (i : Site r c) :
    outsideRow b i+2 ≤ 2*c ∧ outsideCol b i+2 ≤ 2*r := by
  have hR : outsideRow b i ≤ ∑ _j ∈ Finset.univ.erase i.2, (2:ℕ) := by
    apply Finset.sum_le_sum
    intro j _
    have := (b (i.1,j)).1.isLt
    omega
  have hC : outsideCol b i ≤ ∑ _j ∈ Finset.univ.erase i.1, (2:ℕ) := by
    apply Finset.sum_le_sum
    intro j _
    have := (b (j,i.2)).2.1.isLt
    omega
  simp only [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ _),
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hR hC
  change outsideRow b i ≤ (c-1)*2 at hR
  change outsideCol b i ≤ (r-1)*2 at hC
  constructor <;> omega

private theorem population_budget_holds {r c : ℕ} (supply : ℕ) (b : Population r c) (i : Site r c) :
    cellMass (b i) ≤ (populationBudget supply b i).val := by
  have := mass_le_six (b i)
  simp only [populationBudget, Fin.val_mk]
  omega

private theorem population_mass_update {r c : ℕ} (b : Population r c) (i : Site r c) (z : Cell) :
    populationMass (Function.update b i z)+cellMass (b i) = populationMass b+cellMass z := by
  have he : (fun j => cellMass (Function.update b i z j)) =
      Function.update (fun j => cellMass (b j)) i (cellMass z) := by
    funext j; by_cases h : j=i <;> simp [h]
  unfold populationMass
  rw [he, Finset.sum_update_of_mem (Finset.mem_univ i)]
  have hs := Finset.sum_erase_add Finset.univ (fun j => cellMass (b j)) (Finset.mem_univ i)
  simpa [Finset.sdiff_singleton_eq_erase, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
    congrArg (fun n => n+cellMass z) hs

noncomputable def populationCellKernel {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b : Population r c) (i : Site r c) : Cell → Cell → ℝ :=
  cellKernel (populationBudget supply b i) (tiltedEnergy c r (outsideRow b i) (outsideCol b i) (f i))

private noncomputable def pushPopulation {r c : ℕ} {E : Type*} [Fintype E] (p : E → ℝ)
    (d : E → Population r c) (y : Population r c) : ℝ := by
  classical
  exact ∑ z, if d z=y then p z else 0

noncomputable def populationKernel {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b y : Population r c) : ℝ :=
  (1:ℝ)/(r*c:ℕ)*∑ i, pushPopulation (populationCellKernel supply f b i (b i))
    (fun z => Function.update b i z) y

private theorem push_population_value {r c : ℕ} {E : Type*} [Fintype E] (p : E → ℝ)
    (d : E → Population r c) (v : Population r c → ℝ) :
    (∑ y, pushPopulation p d y*v y) = ∑ z, p z*v (d z) := by
  classical
  simp only [pushPopulation, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro z _
  simp

private theorem population_value {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b : Population r c) (v : Population r c → ℝ) :
    (∑ y, populationKernel supply f b y*v y) =
      (1:ℝ)/(r*c:ℕ)*∑ i, ∑ z, populationCellKernel supply f b i (b i) z*
        v (Function.update b i z) := by
  classical
  simp only [populationKernel, mul_assoc, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  rw [← Finset.mul_sum, push_population_value, Finset.mul_sum]

private theorem population_cell_contract {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b : Population r c) (i : Site r c) :
    (∀ x z, 0 ≤ populationCellKernel supply f b i x z) ∧
    (∑ z, populationCellKernel supply f b i (b i) z) = 1 := by
  have h := readout_cell_kernel_contract (populationBudget supply b i)
    (tiltedEnergy c r (outsideRow b i) (outsideCol b i) (f i))
  exact ⟨h.2.1, h.2.2.1 (b i)⟩

private theorem population_positive_edit {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b y : Population r c) (h : 0 < populationKernel supply f b y) :
    ∃ i z, y=Function.update b i z ∧ cellMass z ≤ (populationBudget supply b i).val ∧
      |(cellMass z:ℝ)-cellMass (b i)| ≤ 1 := by
  classical
  have hn (i : Site r c) (z : Cell) :
      (0:ℝ) ≤ if Function.update b i z=y then populationCellKernel supply f b i (b i) z else 0 := by
    split_ifs
    · exact (population_cell_contract supply f b i).1 _ _
    · exact le_refl 0
  have hs : 0 < ∑ i, pushPopulation (populationCellKernel supply f b i (b i))
      (fun z => Function.update b i z) y := by
    change 0 < (1:ℝ)/(r*c:ℕ)*_ at h
    by_contra hh
    have hsum : (∑ i, pushPopulation (populationCellKernel supply f b i (b i))
      (fun z => Function.update b i z) y) ≤ 0 := le_of_not_gt hh
    have hz := mul_nonpos_of_nonneg_of_nonpos (by positivity : (0:ℝ) ≤ (1:ℝ)/(r*c:ℕ)) hsum
    linarith
  obtain ⟨i,_,hi⟩ := (Finset.sum_pos_iff_of_nonneg (fun i _ => Finset.sum_nonneg (fun z _ => hn i z))).mp hs
  obtain ⟨z,_,hz⟩ := (Finset.sum_pos_iff_of_nonneg (fun z _ => hn i z)).mp hi
  have heq : Function.update b i z=y := by
    by_contra hh
    simp only [ite_eq_right hh] at hz
    exact (lt_irrefl 0) hz
  rw [ite_eq_left heq] at hz
  have hcap : cellMass z ≤ (populationBudget supply b i).val := by
    by_contra hh
    have hzero := (readout_cell_kernel_contract (populationBudget supply b i)
      (tiltedEnergy c r (outsideRow b i) (outsideCol b i) (f i))).2.2.2.2.1
      (b i) z (population_budget_holds supply b i) (by omega)
    change populationCellKernel supply f b i (b i) z=0 at hzero
    rw [hzero] at hz
    exact (lt_irrefl 0) hz
  have hnear : ¬(cellMass (b i)+1 < cellMass z ∨ cellMass z+1 < cellMass (b i)) := by
    intro hh
    have hzero := (readout_cell_kernel_contract (populationBudget supply b i)
      (tiltedEnergy c r (outsideRow b i) (outsideCol b i) (f i))).2.2.2.2.2 (b i) z hh
    change populationCellKernel supply f b i (b i) z=0 at hzero
    rw [hzero] at hz
    exact (lt_irrefl 0) hz
  refine ⟨i,z,heq.symm,hcap,?_⟩
  apply abs_le.mpr
  have hlo : (cellMass (b i):ℝ) ≤ cellMass z+1 := by exact_mod_cast (by omega : cellMass (b i) ≤ cellMass z+1)
  have hhi : (cellMass z:ℝ) ≤ cellMass (b i)+1 := by exact_mod_cast (by omega : cellMass z ≤ cellMass (b i)+1)
  constructor <;> linarith

/-- Full uncut population at EVERY positive rectangular size. Empty bodies
are retained. The same conditional cells and one conserved supply generate
a Markov kernel, preserve the legal supply bound and have an exact task-value
pushforward. No finite population is enumerated and no row identity is assumed. -/
theorem population_kernel_contract {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) :
    (∀ b y, 0 ≤ populationKernel supply f b y) ∧
    (∀ b, (∑ y, populationKernel supply f b y)=1) ∧
    (∀ b y, populationMass b ≤ supply → 0 < populationKernel supply f b y → populationMass y ≤ supply) ∧
    (∀ b (F : Population r c → ℝ),
      (∑ y, populationKernel supply f b y*(F y-F b)) =
      (1:ℝ)/(r*c:ℕ)*∑ i, ∑ z, populationCellKernel supply f b i (b i) z*
        (F (Function.update b i z)-F b)) := by
  classical
  have hP : (0:ℝ) < (r*c:ℕ) := by exact_mod_cast Nat.mul_pos hr hc
  have hn (b y : Population r c) : 0 ≤ populationKernel supply f b y := by
    apply mul_nonneg (by positivity)
    apply Finset.sum_nonneg
    intro i _
    apply Finset.sum_nonneg
    intro z _
    split_ifs
    · exact (population_cell_contract supply f b i).1 _ _
    · exact le_refl 0
  have hrow (b : Population r c) : (∑ y, populationKernel supply f b y)=1 := by
    have he := population_value supply f b (fun _ => 1)
    simp only [mul_one] at he
    rw [he]
    simp only [show ∀ i, (∑ z, populationCellKernel supply f b i (b i) z)=1 from
      fun i => (population_cell_contract supply f b i).2, Finset.sum_const,
      Finset.card_univ, Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul, mul_one]
    field_simp
  refine ⟨hn,hrow,?_,?_⟩
  · intro b y hb hp
    obtain ⟨i,z,he,hcap,_⟩ := population_positive_edit supply f b y hp
    have hm := population_mass_update b i z
    have hg : (populationBudget supply b i).val ≤ cellMass (b i)+(supply-populationMass b) :=
      Nat.min_le_right _ _
    rw [he]
    omega
  · intro b F
    exact population_value supply f b (fun y => F y-F b)

#print axioms CGB.FiberOptimizerReview.population_kernel_contract

noncomputable def populationAllowance {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (q : ℝ) (observed error L : Site r c → ℝ) (b : Population r c) : ℝ :=
  (1:ℝ)/(r*c:ℕ)*∑ i,
    let K := populationCellKernel supply f b i
    let up := moveProbability K (b i) true
    let down := moveProbability K (b i) false
    q*observed i*(up-down)+(q*error i+L i*q^2/2)*(up+down)

private theorem population_risk_row {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (q : ℝ) (hq : 0 ≤ q)
    (actual observed error L : Site r c → ℝ) (b : Population r c) (F : Population r c → ℝ)
    (he : ∀ i, |actual i-observed i| ≤ error i)
    (hm : ∀ i z, F (Function.update b i z)-F b ≤
      actual i*q*((cellMass z:ℝ)-cellMass (b i))+
        L i*q^2/2*((cellMass z:ℝ)-cellMass (b i))^2) :
    (∑ y, populationKernel supply f b y*(F y-F b)) ≤
      populationAllowance supply f q observed error L b := by
  classical
  rw [(population_kernel_contract hr hc supply f).2.2.2 b F]
  unfold populationAllowance
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Finset.sum_le_sum
  intro i _
  let K := populationCellKernel supply f b i
  have hn := (population_cell_contract supply f b i).1
  have hfar : ∀ x z, cellMass x+1 < cellMass z ∨ cellMass z+1 < cellMass x → K x z=0 := by
    unfold K populationCellKernel
    exact (readout_cell_kernel_contract (populationBudget supply b i)
      (tiltedEnergy c r (outsideRow b i) (outsideCol b i) (f i))).2.2.2.2.2
  have hmodel : ∀ z, F (Function.update b i z)-F (Function.update b i (b i)) ≤
      actual i*q*((cellMass z:ℝ)-cellMass (b i))+
        L i*q^2/2*((cellMass z:ℝ)-cellMass (b i))^2 := by
    intro z
    simpa only [Function.update_eq_self] using hm i z
  have h := finite_risk_moves K (b i) (fun z => F (Function.update b i z))
    q (observed i) (error i) (L i) (actual i) true hq (by simpa using he i) hn hfar hmodel
  dsimp only at h ⊢
  simp only [Bool.not_true, Bool.false_eq_true, ite_true, ite_false, Function.update_eq_self] at h
  nlinarith

def populationDamage {r c : ℕ} (b : Population r c)
    (e : Site r c × Fin 3) : Population r c :=
  Function.update b e.1 (damageCell (b e.1) e.2)

noncomputable def populationDamageKernel {r c : ℕ} (eta : ℝ) (b y : Population r c) : ℝ := by
  classical
  exact (if b=y then 1-eta else 0) + eta*pushPopulation
    (fun _ : Site r c × Fin 3 => (1:ℝ)/(3*(r*c):ℕ)) (populationDamage b) y

private theorem population_damage_value {r c : ℕ} (eta : ℝ) (b : Population r c)
    (F : Population r c → ℝ) :
    (∑ y, populationDamageKernel eta b y*F y) =
      (1-eta)*F b+eta*((1:ℝ)/(3*(r*c):ℕ)*∑ e : Site r c × Fin 3, F (populationDamage b e)) := by
  classical
  simp only [populationDamageKernel, add_mul, Finset.sum_add_distrib]
  have he : (∑ y, (if b=y then 1-eta else 0)*F y)=(1-eta)*F b := by simp
  rw [he]
  have hh : (∑ y, eta*pushPopulation (fun _ : Site r c × Fin 3 => (1:ℝ)/(3*(r*c):ℕ))
      (populationDamage b) y*F y) = eta*∑ y, pushPopulation
        (fun _ : Site r c × Fin 3 => (1:ℝ)/(3*(r*c):ℕ)) (populationDamage b) y*F y := by
    simp only [Finset.mul_sum, mul_assoc]
  rw [hh, push_population_value, ← Finset.mul_sum]

private theorem population_damage_markov {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) :
    (∀ b y : Population r c, 0 ≤ populationDamageKernel eta b y) ∧
    (∀ b : Population r c, (∑ y, populationDamageKernel eta b y)=1) := by
  classical
  have hP : (0:ℝ) < (r*c:ℕ) := by exact_mod_cast Nat.mul_pos hr hc
  constructor
  · intro b y
    unfold populationDamageKernel
    apply add_nonneg
    · split_ifs <;> linarith
    · apply mul_nonneg he0
      apply Finset.sum_nonneg
      intro e _
      split_ifs <;> positivity
  · intro b
    have h := population_damage_value eta b (fun _ => 1)
    simp only [mul_one] at h
    rw [h]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod,
      Fintype.card_fin, nsmul_eq_mul, mul_one]
    push_cast
    field_simp
    <;> ring

noncomputable def populationWorldKernel {r c : ℕ} (supply : ℕ) (eta : ℝ)
    (f : Site r c → ℝ) (b y : Population r c) : ℝ :=
  ∑ u, populationDamageKernel eta b u*populationKernel supply f u y

private theorem population_world_value {r c : ℕ} (supply : ℕ) (eta : ℝ)
    (f : Site r c → ℝ) (b : Population r c) (F : Population r c → ℝ) :
    (∑ y, populationWorldKernel supply eta f b y*F y) =
      ∑ u, populationDamageKernel eta b u*(∑ y, populationKernel supply f u y*F y) := by
  classical
  simp only [populationWorldKernel, Finset.sum_mul, Finset.mul_sum, mul_assoc]
  rw [Finset.sum_comm]

private theorem population_world_markov {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) (f : Site r c → ℝ) :
    (∀ b y, 0 ≤ populationWorldKernel supply eta f b y) ∧
    (∀ b, (∑ y, populationWorldKernel supply eta f b y)=1) := by
  have hd := population_damage_markov hr hc eta he0 he1
  have hk := population_kernel_contract hr hc supply f
  constructor
  · intro b y
    apply Finset.sum_nonneg
    intro u _
    exact mul_nonneg (hd.1 b u) (hk.1 u y)
  · intro b
    have h := population_world_value supply eta f b (fun _ => 1)
    simp only [mul_one] at h
    rw [h]
    simp only [hk.2.1, mul_one]
    exact hd.2 b

noncomputable def populationStepAllowance {r c : ℕ} (supply : ℕ) (eta : ℝ) (f : Site r c → ℝ)
    (q : ℝ) (observed : Site r c → ℝ) (error : Population r c → Site r c → ℝ)
    (L : Site r c → ℝ) (F : Population r c → ℝ) (b : Population r c) : ℝ :=
  (∑ u, populationDamageKernel eta b u*(F u-F b))+
    ∑ u, populationDamageKernel eta b u*populationAllowance supply f q observed (error u) L u

private theorem population_world_risk {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) (f : Site r c → ℝ)
    (q : ℝ) (hq : 0 ≤ q) (observed : Site r c → ℝ)
    (actual error : Population r c → Site r c → ℝ) (L : Site r c → ℝ) (F : Population r c → ℝ)
    (he : ∀ b i, |actual b i-observed i| ≤ error b i)
    (hm : ∀ b i z, F (Function.update b i z)-F b ≤
      actual b i*q*((cellMass z:ℝ)-cellMass (b i))+
        L i*q^2/2*((cellMass z:ℝ)-cellMass (b i))^2) (b : Population r c) :
    (∑ y, populationWorldKernel supply eta f b y*(F y-F b)) ≤
      populationStepAllowance supply eta f q observed error L F b := by
  classical
  have hd := population_damage_markov hr hc eta he0 he1
  have hk := population_kernel_contract hr hc supply f
  rw [population_world_value]
  have hs (u : Population r c) :
      (∑ y, populationKernel supply f u y*(F y-F b)) =
      (∑ y, populationKernel supply f u y*(F y-F u))+(F u-F b) := by
    calc
      _ = ∑ y, populationKernel supply f u y*((F y-F u)+(F u-F b)) := by
        apply Finset.sum_congr rfl
        intro y _
        congr 1
        ring
      _ = _ := by
        simp_rw [mul_add,Finset.sum_add_distrib]
        rw [← Finset.sum_mul,hk.2.1 u,one_mul]
  simp_rw [hs]
  unfold populationStepAllowance
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro u _
  have hrisk := population_risk_row hr hc supply f q hq (actual u) observed (error u) L u F (he u) (hm u)
  nlinarith [mul_le_mul_of_nonneg_left hrisk (hd.1 b u)]

noncomputable def populationWindowValue {S : Type*} [Fintype S]
    (K : ℕ → S → S → ℝ) (F : S → ℝ) : ℕ → ℕ → S → ℝ
  | 0,_,b => F b
  | T+1,start,b => ∑ u, K start b u*populationWindowValue K F T (start+1) u

noncomputable def populationWindowAllowance {S : Type*} [Fintype S]
    (K : ℕ → S → S → ℝ) (A : ℕ → S → ℝ) : ℕ → ℕ → S → ℝ
  | 0,_,_ => 0
  | T+1,start,b => A start b+∑ u, K start b u*populationWindowAllowance K A T (start+1) u

/-- Exact whole-window task-risk envelope of the SAME uncut population,
including all weak/blocked coordinates and external wear. Frozen-observation
error at later states is DERIVED from the initial error and objective-gradient
Lipschitz bound. No negative task increment, task oracle, shadow weights or
fitted admission score enters the algorithm. The signed envelope can expose
harmful windows; it is a theorem observer, not an acceptance mechanism. -/
theorem population_task_window_contract {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1)
    (f : ℕ → Site r c → ℝ) (q : ℝ) (hq : 0 ≤ q)
    (observed initialError sensitivity : Site r c → ℝ)
    (actual : Population r c → Site r c → ℝ) (L : Site r c → ℝ)
    (F : Population r c → ℝ) (origin : Population r c)
    (hInitial : ∀ i, |actual origin i-observed i| ≤ initialError i)
    (hLip : ∀ b y i, |actual b i-actual y i| ≤ sensitivity i*q*populationDistance b y)
    (hm : ∀ b i z, F (Function.update b i z)-F b ≤
      actual b i*q*((cellMass z:ℝ)-cellMass (b i))+
        L i*q^2/2*((cellMass z:ℝ)-cellMass (b i))^2) :
    let error := fun b i => initialError i+sensitivity i*q*populationDistance origin b
    let K := fun t => populationWorldKernel supply eta (f t)
    let A := fun t => populationStepAllowance supply eta (f t) q observed error L F
    (∀ t b y, 0 ≤ K t b y) ∧ (∀ t b, (∑ y, K t b y)=1) ∧
    (∀ T start b, populationWindowValue K F T start b-F b ≤
      populationWindowAllowance K A T start b) := by
  dsimp only
  let error := fun b i => initialError i+sensitivity i*q*populationDistance origin b
  let K := fun t => populationWorldKernel supply eta (f t)
  let A := fun t => populationStepAllowance supply eta (f t) q observed error L F
  have he (b : Population r c) (i : Site r c) : |actual b i-observed i| ≤ error b i := by
    have ht := abs_sub_le (actual b i) (actual origin i) (observed i)
    have hl := hLip b origin i
    have hs : populationDistance b origin=populationDistance origin b := by
      unfold populationDistance
      apply Finset.sum_congr rfl
      intro j _
      exact abs_sub_comm _ _
    rw [hs] at hl
    dsimp [error]
    linarith [hInitial i]
  have hk (t : ℕ) := population_world_markov hr hc supply eta he0 he1 (f t)
  refine ⟨fun t => (hk t).1, fun t => (hk t).2, ?_⟩
  intro T
  induction T with
  | zero => intro start b; simp [populationWindowValue,populationWindowAllowance]
  | succ T ih =>
    intro start b
    have hs := population_world_risk hr hc supply eta he0 he1 (f start) q hq
      observed actual error L F he hm b
    have hi : (∑ u, K start b u*populationWindowValue K F T (start+1) u) ≤
        (∑ u, K start b u*(F u+populationWindowAllowance K A T (start+1) u)) :=
      Finset.sum_le_sum (by intro u _; exact mul_le_mul_of_nonneg_left (by linarith [ih (start+1) u]) ((hk start).1 b u))
    have heq : (∑ u, K start b u*(F u+populationWindowAllowance K A T (start+1) u)) =
        (∑ u, K start b u*(F u-F b))+F b+
          (∑ u, K start b u*populationWindowAllowance K A T (start+1) u) := by
      simp_rw [mul_add,Finset.sum_add_distrib,mul_sub,Finset.sum_sub_distrib]
      rw [← Finset.sum_mul,(hk start).2 b]
      ring
    rw [heq] at hi
    change populationWindowValue K F (T+1) start b-F b ≤ populationWindowAllowance K A (T+1) start b
    rw [populationWindowValue,populationWindowAllowance]
    change (∑ u, K start b u*(F u-F b)) ≤ A start b at hs
    linarith

#print axioms CGB.FiberOptimizerReview.population_task_window_contract

private theorem population_distance_triangle {r c : ℕ} (b u y : Population r c) :
    populationDistance b y ≤ populationDistance b u+populationDistance u y := by
  unfold populationDistance
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum (fun i _ => abs_sub_le (cellMass (b i):ℝ) (cellMass (u i):ℝ) (cellMass (y i):ℝ))

private theorem population_distance_update {r c : ℕ} (b : Population r c) (i : Site r c) (z : Cell) :
    populationDistance b (Function.update b i z)=|(cellMass (b i):ℝ)-cellMass z| := by
  classical
  unfold populationDistance
  rw [Finset.sum_eq_single i]
  · simp
  · intro j _ hj
    simp [Function.update,hj,Ne.symm hj]
  · simp

private theorem population_physical_distance {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b y : Population r c) (h : 0 < populationKernel supply f b y) : populationDistance b y ≤ 1 := by
  obtain ⟨i,z,hy,_,hd⟩ := population_positive_edit supply f b y h
  rw [hy,population_distance_update,abs_sub_comm]
  exact hd

private theorem damage_cell_one (s : Cell) (field : Fin 3) :
    cellMass (damageCell s field) ≤ cellMass s ∧
      |(cellMass (damageCell s field):ℝ)-cellMass s| ≤ 1 := by
  rcases s with ⟨p,q,w⟩
  fin_cases p <;> fin_cases q <;> fin_cases w <;> fin_cases field <;>
    norm_num [cellMass,damageCell,decOrSame,decField]

private theorem population_damage_event {r c : ℕ} (b : Population r c) (e : Site r c × Fin 3) :
    populationMass (populationDamage b e) ≤ populationMass b ∧
      populationDistance b (populationDamage b e) ≤ 1 := by
  have hc := damage_cell_one (b e.1) e.2
  have hm := population_mass_update b e.1 (damageCell (b e.1) e.2)
  constructor
  · change populationMass (Function.update b e.1 (damageCell (b e.1) e.2)) ≤ _
    omega
  · rw [populationDamage,population_distance_update,abs_sub_comm]
    exact hc.2

private theorem population_damage_support {r c : ℕ} (eta : ℝ) (he0 : 0 ≤ eta)
    (b y : Population r c) (h : 0 < populationDamageKernel eta b y) :
    populationMass y ≤ populationMass b ∧ populationDistance b y ≤ 1 := by
  classical
  by_cases hby : b=y
  · subst y
    simp [populationDistance]
  · have hp : 0 < pushPopulation (fun _ : Site r c × Fin 3 => (1:ℝ)/(3*(r*c):ℕ))
        (populationDamage b) y := by
      rw [populationDamageKernel,ite_eq_right hby,zero_add] at h
      by_contra hh
      have hz := mul_nonpos_of_nonneg_of_nonpos he0 (le_of_not_gt hh)
      linarith
    have hn (e : Site r c × Fin 3) :
        (0:ℝ) ≤ if populationDamage b e=y then (1:ℝ)/(3*(r*c):ℕ) else 0 := by
      split_ifs <;> positivity
    obtain ⟨e,_,he⟩ := (Finset.sum_pos_iff_of_nonneg (fun e _ => hn e)).mp hp
    have hdest : populationDamage b e=y := by
      by_contra hh
      rw [ite_eq_right hh] at he
      exact (lt_irrefl 0) he
    rw [← hdest]
    exact population_damage_event b e

private theorem population_world_support {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) (f : Site r c → ℝ)
    (b y : Population r c) (h : 0 < populationWorldKernel supply eta f b y) :
    (populationMass b ≤ supply → populationMass y ≤ supply) ∧ populationDistance b y ≤ 2 := by
  have hd := population_damage_markov hr hc eta he0 he1
  have hk := population_kernel_contract hr hc supply f
  have hn (u : Population r c) : 0 ≤ populationDamageKernel eta b u*populationKernel supply f u y :=
    mul_nonneg (hd.1 b u) (hk.1 u y)
  obtain ⟨u,_,hu⟩ := (Finset.sum_pos_iff_of_nonneg (fun u _ => hn u)).mp h
  have hp : 0 < populationDamageKernel eta b u ∧ 0 < populationKernel supply f u y := by
    rcases mul_pos_iff.mp hu with hp | hn
    · exact hp
    · exact False.elim ((not_lt_of_ge (hd.1 b u)) hn.1)
  have hD := population_damage_support eta he0 b u hp.1
  have hK := population_physical_distance supply f u y hp.2
  refine ⟨?_,?_⟩
  · intro hb
    exact hk.2.2.1 u y (hD.1.trans hb) hp.2
  · linarith [population_distance_triangle b u y]

private theorem population_distance_capacity {r c : ℕ} (supply : ℕ) (b y : Population r c)
    (hb : populationMass b ≤ supply) (hy : populationMass y ≤ supply) :
    populationDistance b y ≤ 2*(supply:ℝ) := by
  have hsum : populationDistance b y ≤ ∑ i, ((cellMass (b i):ℝ)+cellMass (y i)) := by
    apply Finset.sum_le_sum
    intro i _
    apply abs_le.mpr
    constructor <;> nlinarith [Nat.cast_nonneg (α := ℝ) (cellMass (b i)),Nat.cast_nonneg (α := ℝ) (cellMass (y i))]
  have heq : (∑ i, ((cellMass (b i):ℝ)+cellMass (y i))) =
      (populationMass b:ℝ)+populationMass y := by simp [populationMass,Finset.sum_add_distrib]
  rw [heq] at hsum
  have hbR : (populationMass b:ℝ) ≤ supply := by exact_mod_cast hb
  have hyR : (populationMass y:ℝ) ≤ supply := by exact_mod_cast hy
  linarith

/-- Frozen observations need no shadow weights or per-parameter age. Source
support bounds each actual funded opportunity (wear then physical event) by
two readout quanta; supply also bounds the total reachable displacement. A
true objective-gradient Lipschitz bound and the INITIAL observation error
therefore derive the later error envelope, rather than assuming it each tick.
The trace is in support of the ACTUAL uncut kernel, including no-change steps. -/
theorem population_frozen_gradient_budget {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1)
    (f : ℕ → Site r c → ℝ) (trace : ℕ → Population r c)
    (hs : populationMass (trace 0) ≤ supply)
    (ht : ∀ t, 0 < populationWorldKernel supply eta (f t) (trace t) (trace (t+1)))
    (q : ℝ) (hq : 0 ≤ q) (sensitivity initialError observed : Site r c → ℝ)
    (hL : ∀ i, 0 ≤ sensitivity i) (actual : Population r c → Site r c → ℝ)
    (hLip : ∀ b y i, |actual b i-actual y i| ≤ sensitivity i*q*populationDistance b y)
    (hInitial : ∀ i, |actual (trace 0) i-observed i| ≤ initialError i) :
    (∀ T, populationMass (trace T) ≤ supply) ∧
    (∀ T, populationDistance (trace 0) (trace T) ≤ min (2*(T:ℝ)) (2*(supply:ℝ))) ∧
    (∀ T i, |actual (trace T) i-observed i| ≤ initialError i+
      sensitivity i*q*min (2*(T:ℝ)) (2*(supply:ℝ))) := by
  have hstep (t : ℕ) := population_world_support hr hc supply eta he0 he1 (f t) (trace t) (trace (t+1)) (ht t)
  have hmass : ∀ T, populationMass (trace T) ≤ supply := by
    intro T
    induction T with
    | zero => exact hs
    | succ T ih => exact (hstep T).1 ih
  have hdist : ∀ T, populationDistance (trace 0) (trace T) ≤ 2*(T:ℝ) := by
    intro T
    induction T with
    | zero => simp [populationDistance]
    | succ T ih =>
      have htri := population_distance_triangle (trace 0) (trace T) (trace (T+1))
      have hh := (hstep T).2
      push_cast
      linarith
  have hmin (T : ℕ) : populationDistance (trace 0) (trace T) ≤ min (2*(T:ℝ)) (2*(supply:ℝ)) :=
    le_min (hdist T) (population_distance_capacity supply (trace 0) (trace T) hs (hmass T))
  refine ⟨hmass,hmin,?_⟩
  intro T i
  have htri := abs_sub_le (actual (trace T) i) (actual (trace 0) i) (observed i)
  have hLipT := hLip (trace T) (trace 0) i
  have hsym : populationDistance (trace T) (trace 0)=populationDistance (trace 0) (trace T) := by
    unfold populationDistance
    apply Finset.sum_congr rfl
    intro j _
    exact abs_sub_comm _ _
  rw [hsym] at hLipT
  have hm := mul_le_mul_of_nonneg_left (hmin T) (mul_nonneg (hL i) hq)
  linarith [hInitial i]

#print axioms CGB.FiberOptimizerReview.population_frozen_gradient_budget


private def levelRepresentative (s : Cell) : Cell :=
  match cellMass s with
  | 0 => (0,0,0)
  | 1 => (0,0,1)
  | 2 => (0,0,2)
  | 3 => (1,1,1)
  | 4 => (1,1,2)
  | 5 => if s.1.val = 2 then (2,1,2) else (1,2,2)
  | _ => (2,2,2)

private def reducedReadoutBlock (k : Fin 6) : Finset Cell :=
  match k.val with
  | 0 => {(0,0,0),(0,0,1)}
  | 1 => {(0,0,1),(0,0,2)}
  | 2 => {(0,0,2),(1,1,1)}
  | 3 => {(1,1,1),(1,1,2)}
  | 4 => {(1,1,2),(1,2,2),(2,1,2)}
  | _ => {(1,2,2),(2,1,2),(2,2,2)}

private theorem reduced_geometry (k : Fin 6) :
    (reducedReadoutBlock k).Nonempty ∧
    reducedReadoutBlock k ⊆ readoutBlock k ∧
    (reducedReadoutBlock k).card ≤ 3 ∧
    (∀ s ∈ readoutBlock k, levelRepresentative s ∈ reducedReadoutBlock k ∧
      cellMass (levelRepresentative s) = cellMass s) := by
  fin_cases k <;> decide

private noncomputable def poolForm (P Q : Fin 3 → ℝ) (f : ℝ) (s : Cell) : ℝ :=
  14*(cellMass s:ℝ)-12*(s.1.val:ℝ)*s.2.1.val*s.2.2.val+P s.1+Q s.2.1+f*cellMass s

private theorem representative_min (P Q : Fin 3 → ℝ) (f : ℝ)
    (hp0 : 0 ≤ P 1-P 0) (hp1 : 0 ≤ P 2-P 1)
    (hq0 : 0 ≤ Q 1-Q 0) (hq1 : 0 ≤ Q 2-Q 1)
    (hp8 : P 1-P 0 ≤ 8) (hq8 : Q 1-Q 0 ≤ 8) (s : Cell) :
    poolForm P Q f (levelRepresentative s) ≤ poolForm P Q f s := by
  rcases s with ⟨p,q,w⟩
  fin_cases p <;> fin_cases q <;> fin_cases w <;>
    norm_num [levelRepresentative,cellMass,poolForm] <;> linarith

private theorem positive_pool_increment (D R p : ℝ) (hD : 0 < D) (hR : 0 ≤ R) (hp : 0 ≤ p) :
    0 < 2*(R+p+1)^2/D-2*(R+p)^2/D := by
  have he : 2*(R+p+1)^2/D-2*(R+p)^2/D=(4*(R+p)+2)/D := by ring
  rw [he]
  exact div_pos (by linarith) hD

/-- For the registered H the exact block floor uses at most three energies.
All original microstates remain in the uniform proposal; only evaluation work
is reduced. Arbitrary signed force is constant within each readout level. -/
theorem readout_floor_reduction (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (k : Fin 6) (hfull : (readoutBlock k).Nonempty) :
    ∃ (B : Finset Cell) (hB : B.Nonempty), B ⊆ readoutBlock k ∧ B.card ≤ 3 ∧
      blockFloor B hB (tiltedEnergy a b R C f) =
        blockFloor (readoutBlock k) hfull (tiltedEnergy a b R C f) := by
  let P : Fin 3 → ℝ := fun p => 2*(R+p.val)^2/a
  let Q : Fin 3 → ℝ := fun q => 2*(C+q.val)^2/b
  have hp0 : 0 ≤ P 1-P 0 := by simpa [P] using (positive_pool_increment a R 0 (by linarith) hR (by norm_num)).le
  have hp1 : 0 ≤ P 2-P 1 := by convert (positive_pool_increment a R 1 (by linarith) hR (by norm_num)).le using 1 <;> norm_num [P] <;> ring
  have hq0 : 0 ≤ Q 1-Q 0 := by simpa [Q] using (positive_pool_increment b C 0 (by linarith) hC (by norm_num)).le
  have hq1 : 0 ≤ Q 2-Q 1 := by convert (positive_pool_increment b C 1 (by linarith) hC (by norm_num)).le using 1 <;> norm_num [Q] <;> ring
  have hp8 : P 1-P 0 ≤ 8 := by simpa [P] using (pool_increment a R 0 ha hR (by norm_num) (by norm_num) hRp).le
  have hq8 : Q 1-Q 0 ≤ 8 := by simpa [Q] using (pool_increment b C 0 hb hC (by norm_num) (by norm_num) hCq).le
  have hform (s : Cell) : poolForm P Q f s = tiltedEnergy a b R C f s := by
    simp only [poolForm,P,Q,tiltedEnergy,cellEnergy,rawEnergy,cellMass,Nat.cast_add] <;> ring
  have hmin (s : Cell) : tiltedEnergy a b R C f (levelRepresentative s) ≤ tiltedEnergy a b R C f s := by
    simpa only [hform] using representative_min P Q f hp0 hp1 hq0 hq1 hp8 hq8 s
  refine ⟨reducedReadoutBlock k,(reduced_geometry k).1,(reduced_geometry k).2.1,(reduced_geometry k).2.2.1,?_⟩
  apply le_antisymm
  · obtain ⟨s,hs,he⟩ := blockFloor_attained (readoutBlock k) hfull (tiltedEnergy a b R C f)
    have hr := (reduced_geometry k).2.2.2 s hs
    exact le_trans (blockFloor_le _ _ _ _ hr.1) (le_trans (hmin s) (le_of_eq he))
  · obtain ⟨s,hs,he⟩ := blockFloor_attained (reducedReadoutBlock k) (reduced_geometry k).1 (tiltedEnergy a b R C f)
    exact le_trans (blockFloor_le _ _ _ _ ((reduced_geometry k).2.1 hs)) (le_of_eq he)

#print axioms readout_floor_reduction

private theorem population_up_choice {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (b : Population r c) (i : Site r c)
    (hf : f i < -22) (ht : cellMass (b i) < 6) (hs : populationMass b < supply) :
    ∃ z, cellMass z=cellMass (b i)+1 ∧
      (1:ℝ)/26 ≤ populationCellKernel supply f b i (b i) z := by
  have hbounds := outside_bounds hr hc b i
  let k : Fin 6 := ⟨cellMass (b i),ht⟩
  have hbudget : k.val+1 ≤ (populationBudget supply b i).val := by
    simp only [k,populationBudget,Fin.val_mk]
    omega
  exact (readout_force_response_contract c r (outsideRow b i) (outsideCol b i) (f i)
    (by exact_mod_cast (show 1 ≤ c by omega)) (by exact_mod_cast (show 1 ≤ r by omega))
    (by positivity) (by positivity) (by exact_mod_cast hbounds.1) (by exact_mod_cast hbounds.2)
    (populationBudget supply b i) k hbudget (b i)).2 hf rfl

private theorem population_down_choice {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (b : Population r c) (i : Site r c)
    (hf : 34 < f i) (ht : 0 < cellMass (b i)) :
    ∃ z, cellMass z+1=cellMass (b i) ∧
      (1:ℝ)/26 ≤ populationCellKernel supply f b i (b i) z := by
  have hbounds := outside_bounds hr hc b i
  have h6 := mass_le_six (b i)
  let k : Fin 6 := ⟨cellMass (b i)-1,by omega⟩
  have hk : cellMass (b i)=k.val+1 := by dsimp [k]; omega
  have hbudget : k.val+1 ≤ (populationBudget supply b i).val := by
    rw [← hk]
    exact population_budget_holds supply b i
  exact (readout_force_response_contract c r (outsideRow b i) (outsideCol b i) (f i)
    (by exact_mod_cast (show 1 ≤ c by omega)) (by exact_mod_cast (show 1 ≤ r by omega))
    (by positivity) (by positivity) (by exact_mod_cast hbounds.1) (by exact_mod_cast hbounds.2)
    (populationBudget supply b i) k hbudget (b i)).1 hf hk

private theorem population_world_expect_lower {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he : 0 ≤ eta) (f : Site r c → ℝ)
    (b : Population r c) (F : Population r c → ℝ) (hF : ∀ y, 0 ≤ F y) :
    (1-eta)*(∑ y, populationKernel supply f b y*F y) ≤
      ∑ y, populationWorldKernel supply eta f b y*F y := by
  classical
  rw [population_world_value,population_damage_value]
  have hn : 0 ≤ eta*((1:ℝ)/(3*(r*c):ℕ)*∑ e : Site r c × Fin 3,
      ∑ y, populationKernel supply f (populationDamage b e) y*F y) := by
    apply mul_nonneg he
    apply mul_nonneg (by positivity)
    exact Finset.sum_nonneg (fun e _ => Finset.sum_nonneg (fun y _ =>
      mul_nonneg ((population_kernel_contract hr hc supply f).1 _ _) (hF y)))
  linarith

private theorem population_up_expect {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (b : Population r c) (i : Site r c)
    (hf : f i < -22) (ht : cellMass (b i) < 6) (hs : populationMass b < supply) :
    (1:ℝ)/(r*c:ℕ)/26 ≤ ∑ y, populationKernel supply f b y*
      (if cellMass (y i)=cellMass (b i)+1 then 1 else 0) := by
  classical
  let V : Population r c → ℝ := fun y => if cellMass (y i)=cellMass (b i)+1 then 1 else 0
  have hv (y) : 0 ≤ V y := by dsimp [V]; split_ifs <;> norm_num
  have hn (j : Site r c) (z : Cell) :
      0 ≤ populationCellKernel supply f b j (b j) z*V (Function.update b j z) :=
    mul_nonneg ((population_cell_contract supply f b j).1 _ _) (hv _)
  obtain ⟨z,hz,hp⟩ := population_up_choice hr hc supply f b i hf ht hs
  have hi : (1:ℝ)/26 ≤ ∑ z, populationCellKernel supply f b i (b i) z*V (Function.update b i z) := by
    have h := Finset.single_le_sum (fun z _ => hn i z) (Finset.mem_univ z)
    have hvz : V (Function.update b i z)=1 := by simp [V,hz]
    rw [hvz,mul_one] at h
    exact hp.trans h
  have hall : (∑ z, populationCellKernel supply f b i (b i) z*V (Function.update b i z)) ≤
      ∑ j, ∑ z, populationCellKernel supply f b j (b j) z*V (Function.update b j z) :=
    Finset.single_le_sum (fun j _ => Finset.sum_nonneg (fun z _ => hn j z)) (Finset.mem_univ i)
  have hs := mul_le_mul_of_nonneg_left (hi.trans hall) (by positivity : (0:ℝ) ≤ (1:ℝ)/(r*c:ℕ))
  change _ ≤ ∑ y, populationKernel supply f b y*V y
  rw [population_value]
  convert hs using 1 <;> ring

/-- The same conservative population can serve a new growth request even
with empty stock: actual donor release, then actual target uptake. Donor set
is a proof observer of signed evidence, never a runtime queue or score. The
bound retains all other choices and external damage. It is a finite-window
service probability, not an assumed task benefit or all-request fairness. -/
theorem population_resource_service {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1)
    (f0 f1 : Site r c → ℝ) (b : Population r c) (target : Site r c)
    (hf : f1 target < -22) (ht : cellMass (b target) < 6) :
    (populationMass b < supply → (1-eta)/(r*c:ℕ)/26 ≤
      ∑ y, populationWorldKernel supply eta f1 b y*
        (if cellMass (y target)=cellMass (b target)+1 then 1 else 0)) ∧
    (populationMass b=supply → ∀ D : Finset (Site r c),
      (∀ i ∈ D, i≠target ∧ 0 < cellMass (b i) ∧ 34 < f0 i) →
      (D.card:ℝ)*((1-eta)/(r*c:ℕ)/26)^2 ≤
        ∑ u, populationWorldKernel supply eta f0 b u*
          (∑ y, populationWorldKernel supply eta f1 u y*
            (if cellMass (y target)=cellMass (b target)+1 then 1 else 0))) := by
  classical
  let V : Population r c → ℝ := fun y => if cellMass (y target)=cellMass (b target)+1 then 1 else 0
  have hv (y) : 0 ≤ V y := by dsimp [V]; split_ifs <;> norm_num
  have hw (u) : 0 ≤ ∑ y, populationWorldKernel supply eta f1 u y*V y :=
    Finset.sum_nonneg (fun y _ => mul_nonneg ((population_world_markov hr hc supply eta he0 he1 f1).1 _ _) (hv y))
  have hp0 : 0 ≤ (1-eta)/(r*c:ℕ)/26 := by positivity
  have up_bound (u : Population r c) (hu : u target=b target) (hs : populationMass u < supply) :
      (1-eta)/(r*c:ℕ)/26 ≤ ∑ y, populationWorldKernel supply eta f1 u y*V y := by
    have h : (1:ℝ)/(r*c:ℕ)/26 ≤ ∑ y, populationKernel supply f1 u y*V y := by
      simpa only [V,hu] using population_up_expect hr hc supply f1 u target hf (by simpa [hu] using ht) hs
    have hl := mul_le_mul_of_nonneg_left h (sub_nonneg.mpr he1)
    have hd := population_world_expect_lower hr hc supply eta he0 f1 u V hv
    have heq : (1-eta)*((1:ℝ)/(r*c:ℕ)/26)=(1-eta)/(r*c:ℕ)/26 := by ring
    rw [heq] at hl
    exact hl.trans hd
  constructor
  · intro hs
    exact up_bound b rfl hs
  · intro hb D hD
    let A : Population r c → ℝ := fun u => ∑ y, populationWorldKernel supply eta f1 u y*V y
    have local0 (i : Site r c) (z : Cell) :
        0 ≤ populationCellKernel supply f0 b i (b i) z*A (Function.update b i z) :=
      mul_nonneg ((population_cell_contract supply f0 b i).1 _ _) (hw _)
    have each (i : Site r c) (hi : i ∈ D) :
        ((1-eta)/(r*c:ℕ)/26)/26 ≤ ∑ z,
          populationCellKernel supply f0 b i (b i) z*A (Function.update b i z) := by
      obtain ⟨hne,hm,hforce⟩ := hD i hi
      obtain ⟨z,hz,hp⟩ := population_down_choice hr hc supply f0 b i hforce hm
      have hu : Function.update b i z target=b target := by simp [Ne.symm hne]
      have hm' := population_mass_update b i z
      have hstock : populationMass (Function.update b i z) < supply := by omega
      have hs := up_bound (Function.update b i z) hu hstock
      have hprod := mul_le_mul hp hs hp0 ((population_cell_contract supply f0 b i).1 _ _)
      have hsum := Finset.single_le_sum (fun z _ => local0 i z) (Finset.mem_univ z)
      change _ ≤ ∑ z, populationCellKernel supply f0 b i (b i) z*A (Function.update b i z)
      convert hprod.trans hsum using 1 <;> ring
    have hcount : (D.card:ℝ)*(((1-eta)/(r*c:ℕ)/26)/26) ≤
        ∑ i, ∑ z, populationCellKernel supply f0 b i (b i) z*A (Function.update b i z) := by
      calc
        _ = ∑ _i ∈ D, (((1-eta)/(r*c:ℕ)/26)/26) := by simp [nsmul_eq_mul]
        _ ≤ ∑ i ∈ D, ∑ z, populationCellKernel supply f0 b i (b i) z*A (Function.update b i z) :=
          Finset.sum_le_sum (fun i hi => each i hi)
        _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ D)
          (fun i _ _ => Finset.sum_nonneg (fun z _ => local0 i z))
    have hphys := mul_le_mul_of_nonneg_left hcount (by positivity : (0:ℝ) ≤ (1:ℝ)/(r*c:ℕ))
    rw [← population_value] at hphys
    have hworld := population_world_expect_lower hr hc supply eta he0 f0 b A hw
    have hfinal := (mul_le_mul_of_nonneg_left hphys (sub_nonneg.mpr he1)).trans hworld
    change _ ≤ ∑ u, populationWorldKernel supply eta f0 b u*A u
    convert hfinal using 1 <;> ring

#print axioms population_resource_service


/-- Two different rows AND columns have no shared pool dependence. This is
exactly the matching restriction used by parallel preparation, not an assumed
independence of arbitrary material sites. -/
private theorem matching_outside {r c : ℕ} (b : Population r c) (i j : Site r c)
    (hr : i.1 ≠ j.1) (hc : i.2 ≠ j.2) (z : Cell) :
    outsideRow (Function.update b j z) i=outsideRow b i ∧
    outsideCol (Function.update b j z) i=outsideCol b i ∧
    Function.update b j z i=b i := by
  classical
  have hij : i≠j := by intro h; exact hr (congrArg Prod.fst h)
  refine ⟨?_,?_,by simp [hij]⟩
  · unfold outsideRow
    apply Finset.sum_congr rfl
    intro k hk
    have h : (i.1,k)≠j := by intro he; have hh := Prod.mk.inj he; exact hr hh.1
    simp [h]
  · unfold outsideCol
    apply Finset.sum_congr rfl
    intro k hk
    have h : (k,i.2)≠j := by intro he; have hh := Prod.mk.inj he; exact hc hh.2
    simp [h]

/-- Exact reflected-cumulative stock accounting. Prefix is cumulative PROPOSED
material change, peak starts at initial stock. At a zero-stock upward step,
reflection cancels the unfunded intake; every other step is conserved. -/
theorem matching_parallel_preparation {r c : ℕ} (b : Population r c) (i j : Site r c)
    (hr : i.1≠j.1) (hc : i.2≠j.2) (z x : Cell) (f : ℝ) :
    tiltedEnergy c r (outsideRow (Function.update b j z) i)
      (outsideCol (Function.update b j z) i) f x =
      tiltedEnergy c r (outsideRow b i) (outsideCol b i) f x ∧
    Function.update (Function.update b i x) j z = Function.update (Function.update b j z) i x ∧
    (∀ cumulative peak delta : ℤ, cumulative ≤ peak → -1 ≤ delta → delta ≤ 1 →
      let stock := peak-cumulative
      let nextPrefix := cumulative+delta
      let nextPeak := max peak nextPrefix
      nextPeak-nextPrefix = (if stock=0 ∧ delta=1 then stock else stock-delta) ∧
      0 ≤ nextPeak-nextPrefix ∧
      nextPeak = peak+max 0 (delta-stock)) := by
  have hout := matching_outside b i j hr hc z
  refine ⟨by rw [hout.1,hout.2.1],?_,?_⟩
  · have hij : i≠j := by intro h; exact hr (congrArg Prod.fst h)
    exact Function.update_comm hij x z b
  · intro cumulative peak delta hstock hlo hhi
    dsimp only
    split_ifs <;> omega

#print axioms matching_parallel_preparation


/-- Finite GPU coin contributes at most one grid unit in addition to the
explicit error of its supplied probability. This does not assume a GPU exp2
implementation is exact; its error remains a separate numerical obligation. -/
private theorem matching_coin_error (p q : ℝ) (levels : ℕ) (hN : 0 < levels)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    let k := Int.floor (q*levels)
    0 ≤ (k:ℝ)/levels ∧ (k:ℝ)/levels ≤ 1 ∧
    |(k:ℝ)/levels-p| ≤ |q-p|+1/(levels:ℝ) := by
  dsimp only
  have hn : (0:ℝ) < levels := by exact_mod_cast hN
  have hlo := Int.floor_le (q*(levels:ℝ))
  have hhi := Int.lt_floor_add_one (q*(levels:ℝ))
  have hk0 : (0:ℝ) ≤ (Int.floor (q*(levels:ℝ)):ℝ) := by
    exact_mod_cast (Int.floor_nonneg.mpr (mul_nonneg hq0 hn.le))
  have hround : |(Int.floor (q*(levels:ℝ)):ℝ)/(levels:ℝ)-q| ≤ 1/(levels:ℝ) := by
    rw [show (Int.floor (q*(levels:ℝ)):ℝ)/(levels:ℝ)-q =
      ((Int.floor (q*(levels:ℝ)):ℝ)-q*levels)/(levels:ℝ) by field_simp <;> ring,
      abs_div,abs_of_pos hn]
    apply (div_le_div_iff_of_pos_right hn).mpr
    rw [abs_le]
    constructor <;> linarith
  refine ⟨div_nonneg hk0 hn.le,?_,?_⟩
  · apply (div_le_one hn).mpr
    nlinarith
  · obtain ⟨hl,hh⟩ := abs_le.mp hround
    have hp := le_abs_self (q-p)
    have hneg := neg_abs_le (q-p)
    apply abs_le.mpr
    constructor <;> linarith

#print axioms matching_coin_error

/-- Propagate the actual finite coin error into task increments. This is an
error budget, not an assumption that the ideal or implemented step descends.
The physical stock guard is shared when comparing the two increments. -/
private theorem matching_coin_task_error {ι : Type*} [Fintype ι]
    (p q d : ι → ℝ) (levels : ℕ) (hN : 0 < levels) (epsilon : ℝ)
    (hq0 : ∀ i, 0 ≤ q i) (hq1 : ∀ i, q i ≤ 1)
    (herr : ∀ i, |q i-p i| ≤ epsilon) :
    (∑ i, (Int.floor (q i*levels):ℝ)/levels*d i) ≤
      (∑ i, p i*d i)+(epsilon+1/(levels:ℝ))*(∑ i, |d i|) := by
  have each (i : ι) :
      ((Int.floor (q i*levels):ℝ)/levels-p i)*d i ≤
        (epsilon+1/(levels:ℝ))*|d i| := by
    have hcoin := (matching_coin_error (p i) (q i) levels hN (hq0 i) (hq1 i)).2.2
    have hbound : |(Int.floor (q i*levels):ℝ)/levels-p i| ≤ epsilon+1/(levels:ℝ) := by
      linarith [herr i]
    calc
      _ ≤ |((Int.floor (q i*levels):ℝ)/levels-p i)*d i| := le_abs_self _
      _ = |(Int.floor (q i*levels):ℝ)/levels-p i| * |d i| := abs_mul _ _
      _ ≤ _ := mul_le_mul_of_nonneg_right hbound (abs_nonneg _)
  have hs := Finset.sum_le_sum (s := Finset.univ) (fun i _ => each i)
  simp only [sub_mul,Finset.sum_sub_distrib,← Finset.mul_sum] at hs
  linarith

#print axioms matching_coin_task_error

end CGB.FiberOptimizerReview
