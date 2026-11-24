import Mathlib.Tactic

namespace CGB.NativePreviewRestoreContract

/- Actual model → Lean reproduction/repair → model (user-required order).
   causal_use_value.configure: schema=1 reentry attaches engine-owned evidence
   then returns, omitting identity_preview_revision=1; first C2 install sets it.
   prepare_continuation loads engine fields before configure (line 768).
   Proposed model repair: rehydrate both runtime fields on every C2 configure;
   also restore an explicit config identity_preview_revision for non-C2 branches.
   This file changes no Python and certifies no learning/utility improvement.

   Payload is opaque retained model/Adam/labels/learning state, NOT a digest.
   Existing one-time C2 calibration migration is outside this runtime projection;
   on schema=1 reentry the entire saved learning payload must remain unchanged.
   Evidence denotes the engine-owned object identity, not just equal tensors. -/
structure State (Payload Evidence : Type) where
  payload : Payload
  evidence : Evidence
  initialized : Bool
  revision : Nat
  attached : Option Evidence

variable {Payload Evidence : Type}

/-- Runtime projection of the actual old C2 branches, including the early return. -/
def oldConfigure (s : State Payload Evidence) : State Payload Evidence :=
  if s.initialized then { s with attached := some s.evidence }
  else { s with initialized := true, revision := 1, attached := some s.evidence }

/-- Minimal C2 repair: no replacement evidence, statistics, labels or optimizer. -/
def configure (s : State Payload Evidence) : State Payload Evidence :=
  { s with initialized := true, revision := 1, attached := some s.evidence }

def PersistedEqual (s t : State Payload Evidence) : Prop :=
  s.payload = t.payload ∧ s.evidence = t.evidence ∧ s.initialized = t.initialized

def RuntimeReady (s : State Payload Evidence) : Prop :=
  s.revision = 1 ∧ s.attached = some s.evidence

/-- The initialized branch reattaches evidence but leaves a lost mode at zero. -/
theorem old_reentry_omits_mode (s : State Payload Evidence)
    (h : s.initialized = true) : (oldConfigure s).revision = s.revision := by
  simp [oldConfigure, h]

/-- Full saved payload equality does not imply the runtime needed by preview. -/
theorem same_payload_different_preview (p : Payload) (e : Evidence) :
    let restored : State Payload Evidence := ⟨p, e, true, 0, none⟩
    PersistedEqual (oldConfigure restored) (configure restored) ∧
      (oldConfigure restored).revision ≠ (configure restored).revision ∧
      ¬ RuntimeReady (oldConfigure restored) := by
  simp [PersistedEqual, RuntimeReady, oldConfigure, configure]

/-- Both first installation and restored/repeated entry reconstruct C2 mode. -/
theorem configure_ready (s : State Payload Evidence) :
    RuntimeReady (configure s) ∧ (configure s).initialized = true := by
  simp [RuntimeReady, configure]

/-- Rehydration preserves opaque learning state and the engine's evidence object. -/
theorem configure_preserves_payload (s : State Payload Evidence) :
    (configure s).payload = s.payload ∧ (configure s).evidence = s.evidence := by
  exact ⟨rfl, rfl⟩

/-- No learned state or runtime field drifts on repeated configure calls. -/
theorem configure_idempotent (s : State Payload Evidence) :
    configure (configure s) = configure s := by rfl

/-- Explicit non-C2 identity-preview config must also restore its runtime flag. -/
def restoreExplicit (savedRevision : Nat) (s : State Payload Evidence) :=
  { s with revision := savedRevision }

theorem explicit_config_restored (r : Nat) (s : State Payload Evidence) :
    (restoreExplicit r s).revision = r ∧ (restoreExplicit r s).payload = s.payload := by
  exact ⟨rfl, rfl⟩

/-- C2 takes precedence after explicit config restoration, without changing payload. -/
theorem c2_after_explicit_restore (r : Nat) (s : State Payload Evidence) :
    configure (restoreExplicit r s) = configure s := by rfl

/- Inherit the exact-real matrix obligations already proved in
   reviews/NativeLargestGroupFactOrder.lean: native_mean_permutation,
   native_draw_permutation, native_conditional_operator_permutation.
   That standalone review has no importable olean here; do not copy its proofs.
   The lookup below is the observation_noise fact-ID binding obligation. It does
   not prove canonical sorting or floating-point solve accuracy. Those remain
   Python numerical checks with tolerances; no post-repair numbers are asserted. -/
theorem identity_keyed_lookup_permutation {Index Fact Noise : Type}
    (key : Index → Fact) (noise : Fact → Noise) (π : Equiv.Perm Index) :
    (fun i => noise (key (π i))) = (fun i => noise (key i)) ∘ π := by rfl

/- Real pre-repair 2437 measurements, candidates 4 and 5, rounded to 8 decimals
   as exact rationals, preserving both strict orders. Source:
   results/published/cgb_use_context_2437_20261118.json
   Same eight facts, reversed presentation only; source_unchanged=true,
   optimizer_updates=0, world_calls=0. These are observations, not repaired output. -/
theorem measured_expected_rank_reversal :
    (0.04107481 : ℚ) > 0.03855537 ∧ (0.03745985 : ℚ) < 0.04349266 := by
  norm_num

theorem measured_total_utility_rank_reversal :
    (0.44021362 : ℚ) > 0.39142635 ∧ (0.40406397 : ℚ) < 0.44079912 := by
  norm_num

end CGB.NativePreviewRestoreContract
