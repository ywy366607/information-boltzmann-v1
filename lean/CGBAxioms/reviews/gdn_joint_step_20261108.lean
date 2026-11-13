import CGBAxioms.GenerationRoutingCredit
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# Joint 24-layer cache plumbing and existing recall ports

This is a typed conditional assembly, not equality to a CUDA implementation.
The explicit cache shape matches the installed 24 x 4 x 256 x 256 GDN and
three width-4 convolution caches per layer. R holds CONTINUOUS record fields;
occupancy, slot identities, versions, schedules and branch choices are fixed
within the declared smooth event. Their discrete updates need separate laws.

The native stage still needs its local operator differential instantiated.
Theorems prove cache handoff, spatial assembly and the recall source terms;
they do not assume an already computed whole-history gradient.
-/

namespace CGB.GDNJointStep

/-- One token or a bounded chunk of residual activations. -/
abbrev Activation (T : ℕ) := Fin T → Fin 1024 → ℝ

abbrev NativeS := Fin 4 → Fin 256 → Fin 256 → ℝ
abbrev ConvCache := Fin 1024 → Fin 4 → ℝ
abbrev NativeCache := NativeS × (ConvCache × (ConvCache × ConvCache))
abbrev InstalledCaches := Fin 24 → NativeCache

/-- Only the original human-numbered layers 2 and 13 have recall inlets. -/
def isPort (i : Fin 24) : Prop := i.val = 1 ∨ i.val = 12

instance (i : Fin 24) : Decidable (isPort i) := inferInstanceAs
  (Decidable (i.val = 1 ∨ i.val = 12))

section Plumbing

variable {A C R : Type*}
    [NormedAddCommGroup A] [NormedSpace ℝ A]
    [NormedAddCommGroup C] [NormedSpace ℝ C]
    [NormedAddCommGroup R] [NormedSpace ℝ R]

/-- Activation, all native caches, continuous reservoir fields. -/
abbrev Frame (A C R : Type*) := A × ((Fin 24 → C) × R)

/-- A native block consumes only its own cache and the actual prepared
activation. Its output cache replaces only this layer's slot. The recalled
increment is computed from PRE-block activation in both placement modes. -/
def layerStep (i : Fin 24) (outputPlacement : Bool)
    (native : ℝ × (A × C) → A × C) (ε : ℝ)
    (frame : Frame A C R) (increment : A) : Frame A C R :=
  let prepared := if outputPlacement then frame.1 else frame.1 + increment
  let result := native (ε, (prepared, frame.2.1 i))
  (if outputPlacement then result.1 + increment else result.1,
    (Function.update frame.2.1 i result.2, frame.2.2))

/-- Same packaging on a directional side-state. No identity of frozen
parameters with zero incoming activation/cache directions is introduced. -/
def layerDirection (i : Fin 24) (outputPlacement : Bool)
    (D : ℝ × (A × C) →L[ℝ] A × C)
    (e : Frame A C R) (dIncrement : A) : Frame A C R :=
  let prepared := if outputPlacement then e.1 else e.1 + dIncrement
  let result := D (1, (prepared, e.2.1 i))
  (if outputPlacement then result.1 + dIncrement else result.1,
    (Function.update e.2.1 i result.2, e.2.2))

omit [NormedSpace ℝ A] [NormedAddCommGroup C] [NormedSpace ℝ C]
  [NormedAddCommGroup R] [NormedSpace ℝ R] in
/-- Structural native-cache locality: an inlet is not permission to overwrite
the entire body's recurrent state with an old snapshot. -/
theorem untouched_cache (i j : Fin 24) (hne : j ≠ i) (placement : Bool)
    (native : ℝ × (A × C) → A × C) (ε : ℝ)
    (z : Frame A C R) (increment : A) :
    (layerStep i placement native ε z increment).2.1 j = z.2.1 j := by
  simp [layerStep, Function.update_of_ne hne]

omit [NormedSpace ℝ A] [NormedAddCommGroup C] [NormedSpace ℝ C]
  [NormedAddCommGroup R] [NormedSpace ℝ R] in
/-- Reading does not itself overwrite stored continuous records. Recording
and consolidation must be represented as later explicit transitions. -/
theorem read_stage_preserves_reservoir (i : Fin 24) (placement : Bool)
    (native : ℝ × (A × C) → A × C) (ε : ℝ)
    (z : Frame A C R) (increment : A) :
    (layerStep i placement native ε z increment).2.2 = z.2.2 := rfl

/-- Exact cache replacement derivative, including all untouched layers. -/
theorem updated_caches_deriv (cache : ℝ → Fin 24 → C) (e : Fin 24 → C)
    (i : Fin 24) (fresh : ℝ → C) (de : C)
    (hc : HasDerivAt cache e 0) (hf : HasDerivAt fresh de 0) :
    HasDerivAt (fun t => Function.update (cache t) i (fresh t))
      (Function.update e i de) 0 := by
  apply hasDerivAt_pi.mpr
  intro j
  by_cases h : j = i
  · subst j
    simpa using hf
  · simpa [Function.update_of_ne h] using (hasDerivAt_pi.mp hc j)

/-- A complete native block plus its actual recall increment, as a joint
activation/cache/reservoir stage. Native derivatives are LOCAL to this block,
and the increment derivative is separately supplied by the recall contract. -/
theorem layer_step_deriv (i : Fin 24) (placement : Bool)
    (native : ℝ × (A × C) → A × C)
    (D : ℝ × (A × C) →L[ℝ] A × C)
    (z : ℝ → Frame A C R) (e : Frame A C R)
    (increment : ℝ → A) (di : A)
    (hz : HasDerivAt z e 0) (hi : HasDerivAt increment di 0)
    (hn : HasFDerivAt native D
      (0, ((if placement then (z 0).1 else (z 0).1 + increment 0), (z 0).2.1 i))) :
    HasDerivAt (fun t => layerStep i placement native t (z t) (increment t))
      (layerDirection i placement D e di) 0 := by
  have hx : HasDerivAt (fun t => (z t).1) e.1 0 :=
    (ContinuousLinearMap.fst ℝ A ((Fin 24 → C) × R)).hasFDerivAt.comp_hasDerivAt 0 hz
  have ht : HasDerivAt (fun t => (z t).2) e.2 0 :=
    (ContinuousLinearMap.snd ℝ A ((Fin 24 → C) × R)).hasFDerivAt.comp_hasDerivAt 0 hz
  have hcs : HasDerivAt (fun t => (z t).2.1) e.2.1 0 :=
    (ContinuousLinearMap.fst ℝ (Fin 24 → C) R).hasFDerivAt.comp_hasDerivAt 0 ht
  have hr : HasDerivAt (fun t => (z t).2.2) e.2.2 0 :=
    (ContinuousLinearMap.snd ℝ (Fin 24 → C) R).hasFDerivAt.comp_hasDerivAt 0 ht
  have hp : HasDerivAt
      (fun t => if placement then (z t).1 else (z t).1 + increment t)
      (if placement then e.1 else e.1 + di) 0 := by
    cases placement
    · exact hx.add hi
    · exact hx
  have hc := hasDerivAt_pi.mp hcs i
  have hy := hn.comp_hasDerivAt 0 ((hasDerivAt_id (0 : ℝ)).prodMk (hp.prodMk hc))
  have hyo := (ContinuousLinearMap.fst ℝ A C).hasFDerivAt.comp_hasDerivAt 0 hy
  have hyc := (ContinuousLinearMap.snd ℝ A C).hasFDerivAt.comp_hasDerivAt 0 hy
  have ho : HasDerivAt
      (fun t => if placement then
        (native (t, ((if placement then (z t).1 else (z t).1 + increment t),
          (z t).2.1 i))).1 + increment t
        else (native (t, ((if placement then (z t).1 else (z t).1 + increment t),
          (z t).2.1 i))).1)
      (if placement then (D (1, ((if placement then e.1 else e.1 + di), e.2.1 i))).1 + di
        else (D (1, ((if placement then e.1 else e.1 + di), e.2.1 i))).1) 0 := by
    cases placement
    · exact hyo
    · exact hyo.add hi
  have hcache := updated_caches_deriv (fun t => (z t).2.1) e.2.1 i
    (fun t => (native (t, ((if placement then (z t).1 else (z t).1 + increment t),
      (z t).2.1 i))).2)
    (D (1, ((if placement then e.1 else e.1 + di), e.2.1 i))).2 hcs hyc
  exact ho.prodMk (hcache.prodMk hr)

/-- The same hook computes an increment from current pre-block activation
and reservoir fields; the parameter source also includes current encoder,
replayed native-block and inlet-projection parameters. -/
theorem hook_increment_deriv (hook : ℝ × (A × R) → A)
    (D : ℝ × (A × R) →L[ℝ] A)
    (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (hh : HasFDerivAt hook D (0, ((z 0).1, (z 0).2.2))) :
    HasDerivAt (fun t => hook (t, ((z t).1, (z t).2.2)))
      (D (1, (e.1, e.2.2))) 0 := by
  exact hh.comp_hasDerivAt 0 ((hasDerivAt_id (0 : ℝ)).prodMk
    (hz.fst.prodMk hz.snd.snd))

/-- All other native layers receive the ordinary zero increment, while
continuing to transport their native-state/input directions. -/
def portIncrement (i : Fin 24) (hook : ℝ × (A × R) → A)
    (ε : ℝ) (z : Frame A C R) : A :=
  if isPort i then hook (ε, (z.1, z.2.2)) else 0

omit [NormedSpace ℝ A] [NormedAddCommGroup C] [NormedSpace ℝ C]
  [NormedAddCommGroup R] [NormedSpace ℝ R] in
theorem outside_ports_no_increment (i : Fin 24) (h : ¬ isPort i)
    (hook : ℝ × (A × R) → A) (ε : ℝ) (z : Frame A C R) :
    portIncrement i hook ε z = 0 := by simp [portIncrement, h]

/-- Derivative of the original two-port schedule; no extra inlets are hidden
inside the coupled directional state. -/
theorem port_increment_deriv (i : Fin 24) (hook : ℝ × (A × R) → A)
    (D : ℝ × (A × R) →L[ℝ] A)
    (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (hh : HasFDerivAt hook D (0, ((z 0).1, (z 0).2.2))) :
    HasDerivAt (fun t => portIncrement i hook t (z t))
      (if isPort i then D (1, (e.1, e.2.2)) else 0) 0 := by
  by_cases h : isPort i
  · simpa [portIncrement, h] using hook_increment_deriv hook D z e hz hh
  · simpa [portIncrement, h] using hasDerivAt_const (0 : ℝ) (0 : A)

end Plumbing

section Assembly

variable {A C R : Type*}
    [NormedAddCommGroup A] [NormedSpace ℝ A]
    [NormedAddCommGroup C] [NormedSpace ℝ C]
    [NormedAddCommGroup R] [NormedSpace ℝ R]

variable (native : Fin 24 → ℝ × (A × C) → A × C)
    (hook : Fin 24 → ℝ × (A × R) → A) (placement : Fin 24 → Bool)

/-- The two actual recall locations are part of the forward program. This
does not add a read port at every layer merely because its cache has a tag. -/
def hookedStep (i : Fin 24) (ε : ℝ) (z : Frame A C R) : Frame A C R :=
  layerStep i (placement i) (native i) ε z (portIncrement i (hook i) ε z)

variable (nativeD : Fin 24 → Frame A C R → (ℝ × (A × C) →L[ℝ] A × C))
    (hookD : Fin 24 → Frame A C R → (ℝ × (A × R) →L[ℝ] A))

def hookedDirection (i : Fin 24) (z e : Frame A C R) : Frame A C R :=
  layerDirection i (placement i) (nativeD i z) e
    (if isPort i then hookD i z (1, (e.1, e.2.2)) else 0)

/-- Only local, reachable-point operator derivative premises. Inactive ports
need no hook derivative; the native block is evaluated at the ACTUAL prepared
input. This makes failed upstream assumptions visible rather than frozen. -/
def localAdmission (i : Fin 24) (z : Frame A C R) : Prop :=
  (isPort i → HasFDerivAt (hook i) (hookD i z) (0, (z.1, z.2.2))) ∧
  HasFDerivAt (native i) (nativeD i z)
    (0, ((if placement i then z.1 else z.1 + portIncrement i (hook i) 0 z), z.2.1 i))

theorem hooked_step_deriv (i : Fin 24) (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (hadmit : localAdmission native hook placement nativeD hookD i (z 0)) :
    HasDerivAt (fun t => hookedStep native hook placement i t (z t))
      (hookedDirection placement nativeD hookD i (z 0) e) 0 := by
  have hincrement : HasDerivAt (fun t => portIncrement i (hook i) t (z t))
      (if isPort i then hookD i (z 0) (1, (e.1, e.2.2)) else 0) 0 := by
    by_cases h : isPort i
    · exact port_increment_deriv i (hook i) (hookD i (z 0)) z e hz (hadmit.1 h)
    · simpa [portIncrement, h] using hasDerivAt_const (0 : ℝ) (0 : A)
  exact layer_step_deriv i (placement i) (native i) (nativeD i (z 0)) z e
    (fun t => portIncrement i (hook i) t (z t)) _ hz hincrement hadmit.2

/-- Process the real layer order; caches live in one frame throughout. -/
def runLayers : List (Fin 24) → ℝ → Frame A C R → Frame A C R
  | [], _, z => z
  | i :: rest, ε, z => runLayers rest ε (hookedStep native hook placement i ε z)

/-- One current side-state, transported through each layer's actual prepared
input, cache and reservoir. It is not 24 copies of historical executions. -/
def runDirections : List (Fin 24) → Frame A C R → Frame A C R → Frame A C R
  | [], _, e => e
  | i :: rest, z, e => runDirections rest
      (hookedStep native hook placement i 0 z)
      (hookedDirection placement nativeD hookD i z e)

/-- A finite chain of local obligations at the states actually reached in
this layer pass. No all-states differentiability or total-stack derivative is
hidden as a premise. The caller must discharge each native/hook obligation. -/
def runAdmission : List (Fin 24) → Frame A C R → Prop
  | [], _ => True
  | i :: rest, z => localAdmission native hook placement nativeD hookD i z ∧
      runAdmission rest (hookedStep native hook placement i 0 z)

/-- Native, hook and joint-cache derivatives assemble along the SAME forward
program, including input- and output-placement variants of the old inlet. -/
theorem run_layers_deriv (layers : List (Fin 24)) (z : ℝ → Frame A C R)
    (e : Frame A C R) (hz : HasDerivAt z e 0)
    (ha : runAdmission native hook placement nativeD hookD layers (z 0)) :
    HasDerivAt (fun t => runLayers native hook placement layers t (z t))
      (runDirections native hook placement nativeD hookD layers (z 0) e) 0 := by
  induction layers generalizing z e with
  | nil => exact hz
  | cons i rest ih =>
    have hlocal := hooked_step_deriv native hook placement nativeD hookD i z e hz ha.1
    exact ih (fun t => hookedStep native hook placement i t (z t))
      (hookedDirection placement nativeD hookD i (z 0) e) hlocal ha.2

/-- Specialization to all 24 original layers in order, without introducing
new inlets or changing the original LM head. -/
theorem all_24_layers_deriv (z : ℝ → Frame A C R) (e : Frame A C R)
    (hz : HasDerivAt z e 0)
    (ha : runAdmission native hook placement nativeD hookD (List.finRange 24) (z 0)) :
    HasDerivAt (fun t => runLayers native hook placement (List.finRange 24) t (z t))
      (runDirections native hook placement nativeD hookD (List.finRange 24) (z 0) e) 0 :=
  run_layers_deriv native hook placement nativeD hookD _ z e hz ha

end Assembly

section Recording

variable {Slot R : Type*} [Fintype Slot] [DecidableEq Slot]
    [NormedAddCommGroup R] [NormedSpace ℝ R]

omit [Fintype Slot] in
/-- A record capture has its own causal source. Detaching it in runtime
autograd does not prove that a historical intervention could not change the
captured contents. Overwrite also replaces the previous record's direction. -/
theorem record_write_deriv (bank : ℝ → Slot → R) (e : Slot → R) (slot : Slot)
    (captured : ℝ → R) (dc : R) (hb : HasDerivAt bank e 0)
    (hc : HasDerivAt captured dc 0) :
    HasDerivAt (fun t => Function.update (bank t) slot (captured t))
      (Function.update e slot dc) 0 := by
  apply hasDerivAt_pi.mpr
  intro j
  by_cases h : j = slot
  · subst j
    simpa using hc
  · simpa [Function.update_of_ne h] using (hasDerivAt_pi.mp hb j)

/-- The source/action feature tables also merge observed vectors by their
fixed current count. That discrete count is held for this one event, while
the old and arriving continuous fields both retain their directions. -/
theorem record_merge_deriv (old arriving : ℝ → R) (de da : R) (n : ℝ)
    (ho : HasDerivAt old de 0) (ha : HasDerivAt arriving da 0) :
    HasDerivAt (fun t => (1 / (n + 1)) • (n • old t + arriving t))
      ((1 / (n + 1)) • (n • de + da)) 0 := by
  exact ((ho.const_smul n).add ha).const_smul (1 / (n + 1))

/-- Finite continuous record fields also cost K times their coordinate count
when K exact coupled tags are retained, independently of life duration. -/
theorem complete_tag_coordinate_count (K native records : ℕ) :
    K * (native + records) = K * native + K * records := Nat.mul_add _ _ _

end Recording

section Recall

variable {I V : Type*} [Fintype I]

/-- All current selected logits, including a null candidate, can vary.
This is the ordinary finite softmax for a fixed occupied support. Invalid
slots and the clamp branch must be excluded/handled before using it. -/
theorem softmax_weight_deriv [Nonempty I] (score : I → ℝ → ℝ) (ds : I → ℝ)
    (hs : ∀ i, HasDerivAt (score i) (ds i) 0) (i : I) :
    HasDerivAt (fun t => Real.exp (score i t) / ∑ j, Real.exp (score j t))
      ((Real.exp (score i 0) * ds i * (∑ j, Real.exp (score j 0)) -
        Real.exp (score i 0) * (∑ j, Real.exp (score j 0) * ds j)) /
        (∑ j, Real.exp (score j 0)) ^ 2) 0 := by
  have hden := HasDerivAt.fun_sum (u := Finset.univ)
    (fun j _ => (hs j).exp)
  exact (hs i).exp.div hden
    (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'

/-- Fixed selected identities include the zero-valued null candidate, if
present. Raw normalization, score, encoder and replay effects enter dweight
or dvalue. A support-changing top-k is not assumed differentiable here. -/
theorem selected_read_deriv (weight : I → ℝ → ℝ) (value : I → V → ℝ → ℝ)
    (dw : I → ℝ) (dv : I → V → ℝ)
    (hw : ∀ i, HasDerivAt (weight i) (dw i) 0)
    (hv : ∀ i j, HasDerivAt (value i j) (dv i j) 0) (j : V) :
    HasDerivAt (fun t => ∑ i, weight i t * value i j t)
      (∑ i, (dw i * value i j 0 + weight i 0 * dv i j)) 0 := by
  simpa using HasDerivAt.fun_sum (u := Finset.univ)
    (fun i _ => (hw i).mul (hv i j))

/-- Holding old native replay derivatives does not erase current query,
weight or encoder derivatives. Their common terms cancel in this difference;
the removed term is the selected OLD-native value leg only. -/
theorem held_replay_leg_difference (w dw value current old : I → ℝ) :
    (∑ i, (dw i * value i + w i * (current i + old i))) -
      (∑ i, (dw i * value i + w i * current i)) = ∑ i, w i * old i := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- A strict finite top-k separation persists in a neighborhood of the
current point for continuous scores. Tied cutoffs do not satisfy this premise;
slot ordering or discrete credit there needs its own convention. -/
theorem strict_support_locally_stable (score : I → ℝ → ℝ) (selected : I → Bool)
    (hcont : ∀ i, ContinuousAt (score i) 0)
    (hgap : ∀ i j, selected i = true → selected j = false → score j 0 < score i 0) :
    ∀ᶠ t in nhds (0 : ℝ), ∀ i j, selected i = true → selected j = false →
      score j t < score i t := by
  apply Filter.eventually_all.mpr
  intro i
  apply Filter.eventually_all.mpr
  intro j
  by_cases hi : selected i = true
  · by_cases hj : selected j = false
    · exact ((hcont j).eventually_lt (hcont i) (hgap i j hi hj)).mono
        (fun t ht _ _ => ht)
    · exact Filter.Eventually.of_forall (fun t _ h => (hj h).elim)
  · exact Filter.Eventually.of_forall (fun t h _ => (hi h).elim)

end Recall

section RecordContract

variable {Slot Q X B : Type*} [Fintype Slot] [DecidableEq Slot]
    [NormedAddCommGroup Q] [NormedSpace ℝ Q]
    [NormedAddCommGroup X] [NormedSpace ℝ X]
    [NormedAddCommGroup B] [NormedSpace ℝ B]

/-- Only fields read by the current local replay: address feature, incoming
65-position activation, and LOCAL boundary captured when the window opened. -/
abbrev CapsuleFields (Q X B : Type*) := Q × (X × B)

noncomputable def sourceMean (n : ℕ) (old fresh : Q) : Q :=
  (1 / ((n : ℝ) + 1)) • ((n : ℝ) • old + fresh)

/-- The real commit rule merges the address source but OVERWRITES capsule
inputs/boundary. The open boundary is an explicit input, not the live end state. -/
noncomputable def commitFields (bank : Slot → CapsuleFields Q X B) (slot : Slot) (n : ℕ)
    (freshKey : Q) (capturedInput : X) (boundaryAtOpen : B) : Slot → CapsuleFields Q X B :=
  Function.update bank slot (sourceMean n (bank slot).1 freshKey, (capturedInput, boundaryAtOpen))

omit [Fintype Slot] in
/-- The same count/slot and the captured OPEN-boundary derivative are used
on the side-state. Runtime detach is not a reason to erase these directions. -/
theorem commit_fields_deriv (bank : ℝ → Slot → CapsuleFields Q X B)
    (dbank : Slot → CapsuleFields Q X B) (slot : Slot) (n : ℕ)
    (freshKey : ℝ → Q) (input : ℝ → X) (openBoundary : ℝ → B)
    (dkey : Q) (dinput : X) (dopen : B)
    (hb : HasDerivAt bank dbank 0) (hk : HasDerivAt freshKey dkey 0)
    (hi : HasDerivAt input dinput 0) (ho : HasDerivAt openBoundary dopen 0) :
    HasDerivAt (fun t => commitFields (bank t) slot n (freshKey t) (input t) (openBoundary t))
      (commitFields dbank slot n dkey dinput dopen) 0 := by
  have hs := (hasDerivAt_pi.mp hb) slot
  have hkey := (ContinuousLinearMap.fst ℝ Q (X × B)).hasFDerivAt.comp_hasDerivAt 0 hs
  have hm := record_merge_deriv (fun t => (bank t slot).1) freshKey
    (dbank slot).1 dkey (n : ℝ) hkey hk
  exact record_write_deriv bank dbank slot _ _ hb (hm.prodMk (hi.prodMk ho))

omit [Fintype Slot] [NormedAddCommGroup X] [NormedSpace ℝ X]
  [NormedAddCommGroup B] [NormedSpace ℝ B] in
/-- Replacement n=0 releases the old address AND old capsule direction. -/
theorem replacement_zero_count (bank : Slot → CapsuleFields Q X B) (slot : Slot)
    (freshKey : Q) (input : X) (openBoundary : B) :
    commitFields bank slot 0 freshKey input openBoundary slot = (freshKey, (input, openBoundary)) := by
  simp [commitFields, sourceMean]

omit [Fintype Slot] [NormedAddCommGroup X] [NormedSpace ℝ X]
  [NormedAddCommGroup B] [NormedSpace ℝ B] in
/-- All untouched records retain their values/directions with no global copy. -/
theorem commit_untouched_slot (bank : Slot → CapsuleFields Q X B) (slot other : Slot)
    (h : other ≠ slot) (n : ℕ) (freshKey : Q) (input : X) (openBoundary : B) :
    commitFields bank slot n freshKey input openBoundary other = bank other := by
  simp [commitFields, Function.update_of_ne h]

omit [Fintype Slot] [NormedAddCommGroup X] [NormedSpace ℝ X]
  [NormedAddCommGroup B] [NormedSpace ℝ B] in
/-- Taking the live end-boundary direction instead of the open snapshot is
observably a different record whenever those two directions differ. -/
theorem open_boundary_cannot_be_replaced_by_end (bank : Slot → CapsuleFields Q X B)
    (slot : Slot) (n : ℕ) (freshKey : Q) (input : X) (atOpen atEnd : B)
    (h : atOpen ≠ atEnd) :
    commitFields bank slot n freshKey input atOpen ≠ commitFields bank slot n freshKey input atEnd := by
  intro heq
  have he := congrArg (fun f => (f slot).2.2) heq
  exact h (by simpa [commitFields] using he)

def nextVersions (versions : Slot → ℕ) (slot : Slot) : Slot → ℕ :=
  Function.update versions slot (versions slot + 1)

omit [Fintype Slot] in
/-- Both merge and replacement invalidate receipts/tags bound to the old
slot version. Reusing the same slot index does not preserve event identity. -/
theorem committed_version_rejects_old (versions : Slot → ℕ) (slot : Slot) :
    nextVersions versions slot slot ≠ versions slot := by
  simp [nextVersions]

/-- The source mean has a concrete gain: old n/(n+1), arriving 1/(n+1).
No all-history or whole-body operator norm is a premise here. -/
theorem source_mean_error_bound (n : ℕ) (old oldApprox fresh freshApprox : Q)
    (oldError freshError : ℝ) (ho : ‖old - oldApprox‖ ≤ oldError)
    (hf : ‖fresh - freshApprox‖ ≤ freshError) :
    ‖sourceMean n old fresh - sourceMean n oldApprox freshApprox‖ ≤
      (1 / ((n : ℝ) + 1)) * ((n : ℝ) * oldError + freshError) := by
  have heq : sourceMean n old fresh - sourceMean n oldApprox freshApprox =
      (1 / ((n : ℝ) + 1)) • ((n : ℝ) • (old - oldApprox) + (fresh - freshApprox)) := by
    simp only [sourceMean, smul_sub]
    module
  rw [heq, norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply (norm_add_le _ _).trans
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg n)]
  exact add_le_add (mul_le_mul_of_nonneg_left ho (Nat.cast_nonneg n)) hf

/-- Averaging does not amplify a common fieldwise error budget. A generic
gain >1 per copied record would throw away this important structure. -/
theorem source_mean_preserves_budget (n : ℕ) (old oldApprox fresh freshApprox : Q)
    (budget : ℝ) (ho : ‖old - oldApprox‖ ≤ budget) (hf : ‖fresh - freshApprox‖ ≤ budget) :
    ‖sourceMean n old fresh - sourceMean n oldApprox freshApprox‖ ≤ budget := by
  have hh := source_mean_error_bound n old oldApprox fresh freshApprox budget budget ho hf
  have heq : (1 / ((n : ℝ) + 1)) * ((n : ℝ) * budget + budget) = budget := by
    field_simp
  simpa only [heq] using hh

structure RecordBudget (Slot : Type*) where
  key : Slot → ℝ
  input : Slot → ℝ
  boundary : Slot → ℝ

noncomputable def commitBudget (old : RecordBudget Slot) (slot : Slot) (n : ℕ)
    (keyError inputError openError : ℝ) : RecordBudget Slot :=
  ⟨Function.update old.key slot ((1 / ((n : ℝ) + 1)) * ((n : ℝ) * old.key slot + keyError)),
   Function.update old.input slot inputError, Function.update old.boundary slot openError⟩

def RecordsWithin (full approximate : Slot → CapsuleFields Q X B) (r : RecordBudget Slot) : Prop :=
  ∀ s, ‖(full s).1 - (approximate s).1‖ ≤ r.key s ∧
    ‖(full s).2.1 - (approximate s).2.1‖ ≤ r.input s ∧
    ‖(full s).2.2 - (approximate s).2.2‖ ≤ r.boundary s

omit [Fintype Slot] [NormedSpace ℝ X] [NormedSpace ℝ B] in
/-- Sparse concrete error handoff: key budgets merge, input/open-boundary
budgets overwrite, and all other slots retain their own budgets. -/
theorem commit_budget_certificate (full approximate : Slot → CapsuleFields Q X B)
    (r : RecordBudget Slot) (hr : RecordsWithin full approximate r) (slot : Slot) (n : ℕ)
    (key keyApprox : Q) (input inputApprox : X) (openBoundary openApprox : B)
    (keyError inputError openError : ℝ)
    (hk : ‖key - keyApprox‖ ≤ keyError) (hi : ‖input - inputApprox‖ ≤ inputError)
    (ho : ‖openBoundary - openApprox‖ ≤ openError) :
    RecordsWithin (commitFields full slot n key input openBoundary)
      (commitFields approximate slot n keyApprox inputApprox openApprox)
      (commitBudget r slot n keyError inputError openError) := by
  intro s
  by_cases hs : s = slot
  · subst s
    simp only [commitFields, commitBudget, Function.update_self]
    exact ⟨source_mean_error_bound n _ _ _ _ _ _ (hr slot).1 hk, hi, ho⟩
  · simpa [commitFields, commitBudget, Function.update_of_ne hs] using hr s

end RecordContract

section ReadGain

variable {I E : Type*} [Fintype I] [NormedAddCommGroup E] [NormedSpace ℝ E]

def weightedRead (w : I → ℝ) (value : I → E) : E := ∑ i, w i • value i

def scoreMean (w ds : I → ℝ) : ℝ := ∑ i, w i * ds i

def readDirection (w ds : I → ℝ) (value dvalue : I → E) : E :=
  ∑ i, ((w i * (ds i - scoreMean w ds)) • value i + w i • dvalue i)

/-- Actual softmax score credit is centered on the read value; common
value shifts cancel. The null candidate must remain in this same sum. -/
theorem read_direction_centered (w ds : I → ℝ) (value dvalue : I → E) :
    readDirection w ds value dvalue = (∑ i, w i • dvalue i) +
      ∑ i, (w i * ds i) • (value i - weightedRead w value) := by
  have hleft : (∑ i, (w i * scoreMean w ds) • value i) =
      scoreMean w ds • weightedRead w value := by
    simp only [weightedRead, Finset.smul_sum, smul_smul]
    apply Finset.sum_congr rfl
    intro i _
    rw [mul_comm]
  have hright : (∑ i, (w i * ds i) • weightedRead w value) =
      scoreMean w ds • weightedRead w value := by
    rw [← Finset.sum_smul]
    rfl
  simp only [readDirection, mul_sub, sub_smul, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, smul_sub, hleft, hright]
  abel

/-- Concrete value/score gain for fixed selected identities. Weights are
the real forward weights, NOT a renormalized history-only distribution. -/
theorem read_direction_gain (w ds : I → ℝ) (value dvalue : I → E)
    (hw : ∀ i, 0 ≤ w i) :
    ‖readDirection w ds value dvalue‖ ≤
      (∑ i, w i * ‖dvalue i‖) +
        ∑ i, w i * |ds i| * ‖value i - weightedRead w value‖ := by
  rw [read_direction_centered]
  apply (norm_add_le _ _).trans
  apply add_le_add
  · have h := norm_sum_le Finset.univ (fun i => w i • dvalue i)
    simpa only [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw _)] using h
  · have h := norm_sum_le Finset.univ
      (fun i => (w i * ds i) • (value i - weightedRead w value))
    simpa only [norm_smul, Real.norm_eq_abs, abs_mul, abs_of_nonneg (hw _)] using h

/-- A checkable gain row from local value and score bounds. No factor equal
to the whole reservoir capacity is introduced by summing a selected read. -/
theorem read_gain_from_local_bounds (w ds valueGain scoreGain : I → ℝ)
    (value dvalue : I → E) (hw : ∀ i, 0 ≤ w i)
    (hv : ∀ i, ‖dvalue i‖ ≤ valueGain i) (hs : ∀ i, |ds i| ≤ scoreGain i) :
    ‖readDirection w ds value dvalue‖ ≤
      (∑ i, w i * valueGain i) +
        ∑ i, w i * scoreGain i * ‖value i - weightedRead w value‖ := by
  apply (read_direction_gain w ds value dvalue hw).trans
  apply add_le_add <;> apply Finset.sum_le_sum
  · intro i _
    exact mul_le_mul_of_nonneg_left (hv i) (hw i)
  · intro i _
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left (hs i) (hw i)) (norm_nonneg _)

/-- Actual record-budget-to-read row: source keys determine weights, local
input/open-boundary errors determine replay values, and query error is shared.
The replay/encoder coefficients are local operator obligations, not a global
whole-history gain assumed as one unexplained number. -/
theorem read_record_budget_row {Slot : Type*} (slots : I → Slot) (r : RecordBudget Slot)
    (w ds : I → ℝ) (value dvalue : I → E) (hw : ∀ i, 0 ≤ w i)
    (inputGain boundaryGain queryGain keyGain : I → ℝ) (queryError : ℝ)
    (hv : ∀ i, ‖dvalue i‖ ≤ inputGain i * r.input (slots i) + boundaryGain i * r.boundary (slots i))
    (hs : ∀ i, |ds i| ≤ queryGain i * queryError + keyGain i * r.key (slots i)) :
    ‖readDirection w ds value dvalue‖ ≤
      (∑ i, w i * ‖value i - weightedRead w value‖ * queryGain i) * queryError +
        ∑ i, (w i * (inputGain i * r.input (slots i) + boundaryGain i * r.boundary (slots i)) +
          w i * ‖value i - weightedRead w value‖ * keyGain i * r.key (slots i)) := by
  have h := read_gain_from_local_bounds w ds _ _ value dvalue hw hv hs
  convert h using 1
  rw [Finset.sum_add_distrib, Finset.sum_mul]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

end ReadGain

section AddressGain

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Inner-product address score differential, including both query and key
paths and an optional additive preference direction, at positive temperature. -/
theorem address_score_gain (q k dq dk : E) (temperature dbias : ℝ)
    (ht : 0 < temperature) :
    |(inner ℝ dq k + inner ℝ q dk) / temperature + dbias| ≤
      (‖dq‖ * ‖k‖ + ‖q‖ * ‖dk‖) / temperature + |dbias| := by
  have habs (a b : ℝ) : |a + b| ≤ |a| + |b| := by
    simpa only [Real.norm_eq_abs] using norm_add_le a b
  calc
    _ ≤ |(inner ℝ dq k + inner ℝ q dk) / temperature| + |dbias| := habs _ _
    _ = |inner ℝ dq k + inner ℝ q dk| / temperature + |dbias| := by
      rw [abs_div, abs_of_pos ht]
    _ ≤ (|inner ℝ dq k| + |inner ℝ q dk|) / temperature + |dbias| :=
      add_le_add (div_le_div_of_nonneg_right (habs _ _) ht.le) le_rfl
    _ ≤ _ := add_le_add (div_le_div_of_nonneg_right
      (add_le_add (abs_real_inner_le_norm dq k) (abs_real_inner_le_norm q dk)) ht.le) le_rfl

end AddressGain

end CGB.GDNJointStep
