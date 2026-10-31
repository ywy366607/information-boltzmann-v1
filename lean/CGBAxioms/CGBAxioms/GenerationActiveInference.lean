import CGBAxioms.GenerationInformationEnergy
import CGBAxioms.LanguageLifeKernel

/-!
# Active inference from finite supply (minimal information contract)

This module does not add a physical entropy variable or a new organism state.
It states the smallest world-dependent bridge needed for active behaviour:
the information/maintenance signal available to an eventually passive policy
is bounded, while a viable policy requires an unbounded cumulative signal.
The conclusion is only that viability rules out eventual passivity.  Which
action changes the future is a separate causal contract, covered by the
intervention results in `LanguageLifeKernel`.
-/

namespace CGB.ActiveInference

open Finset

variable {A : Type*}

/-- Cumulative usable signal received by a policy up to a finite prefix. -/
def cumulative (signal : ℕ → ℕ) (T : ℕ) : ℕ :=
  ∑ t ∈ range T, signal t

/-- The policy keeps receiving unboundedly much usable signal. -/
def unbounded (signal : ℕ → ℕ) : Prop :=
  ∀ B : ℕ, ∃ T : ℕ, B < cumulative signal T

/-- A policy is eventually passive with respect to an action `a₀`. -/
def eventuallyPassive (policy : ℕ → A) (a₀ : A) : Prop :=
  ∃ t₀ : ℕ, ∀ t, t₀ ≤ t → policy t = a₀

/-- The signal available to an eventually passive policy is bounded.  This is
the only world-specific premise: it can be witnessed by depletion of a source,
loss of predictive distinctions, or any other registered finite opportunity. -/
def passiveBounded (policy : ℕ → A) (signal : ℕ → ℕ) (a₀ : A) : Prop :=
  eventuallyPassive policy a₀ → ∃ C : ℕ, ∀ T, cumulative signal T ≤ C

/-- A bounded cumulative signal cannot satisfy the unbounded viability contract. -/
theorem not_unbounded_of_bounded {signal : ℕ → ℕ}
    (hbound : ∃ C : ℕ, ∀ T, cumulative signal T ≤ C) :
    ¬ unbounded signal := by
  rintro hunbounded
  obtain ⟨C, hC⟩ := hbound
  obtain ⟨T, hT⟩ := hunbounded C
  exact (Nat.not_lt_of_ge (hC T)) hT

/-- **Active necessity.**  If viability requires an unbounded usable signal,
and every eventually passive policy has a bounded signal in the declared
world, then a viable policy cannot become eventually passive. -/
theorem viable_not_eventually_passive
    (policy : ℕ → A) (signal : ℕ → ℕ) (a₀ : A)
    (hviable : unbounded signal)
    (hpassive : passiveBounded policy signal a₀) :
    ¬ eventuallyPassive policy a₀ := by
  intro hpassive_policy
  exact not_unbounded_of_bounded (hpassive hpassive_policy) hviable

/-- A world response is action-sensitive when some future input has a different
probability under two actions. -/
def actionSensitive {A I : Type*} (response : A → I → ℝ) : Prop :=
  ∃ a a' i, response a i ≠ response a' i

/-- If the world is action-sensitive, an action-blind predictor cannot match
every action-conditioned response.  The predictor need not be a new channel:
the action is simply an argument of the existing causal state/readout. -/
theorem action_condition_required {A I : Type*}
    (response : A → I → ℝ) (predict : I → ℝ)
    (hsensitive : actionSensitive response)
    (hblind : ∀ a i, predict i = response a i) : False := by
  obtain ⟨a, a', i, hdiff⟩ := hsensitive
  apply hdiff
  exact (hblind a i).symm.trans (hblind a' i)

/-- Silence is an ordinary byte outcome.  If the probability of byte `0`
depends on the action, an action-blind predictor still cannot represent it. -/
theorem silence_condition_required {A : Type*}
    (response : A → Fin 256 → ℝ) (predict : Fin 256 → ℝ)
    (hsilent : ∃ a a', response a 0 ≠ response a' 0)
    (hblind : ∀ a i, predict i = response a i) : False := by
  apply action_condition_required response predict
  · exact ⟨hsilent.choose, hsilent.choose_spec.choose, 0,
      hsilent.choose_spec.choose_spec⟩
  · exact hblind

/-- **Active closure.** Under one declared world contract, viability rules out
eventual passivity and action-sensitive future inputs rule out an
action-blind predictor.  This is the minimal logical closure of survival
pressure, intervention and self-conditioned prediction; it does not assume
that every tick produces a nonzero response. -/
theorem active_closure {A I : Type*}
    (policy : ℕ → A) (signal : ℕ → ℕ) (a₀ : A)
    (response : A → I → ℝ) (predict : I → ℝ)
    (hviable : unbounded signal)
    (hpassive : passiveBounded policy signal a₀)
    (hsensitive : actionSensitive response) :
    (¬ eventuallyPassive policy a₀) ∧
      ¬ (∀ a i, predict i = response a i) := by
  constructor
  · exact viable_not_eventually_passive policy signal a₀ hviable hpassive
  · intro hblind
    exact action_condition_required response predict hsensitive hblind

end CGB.ActiveInference
