import CGBAxioms.GenerationAssociationCapacity

/-!
# Nonlinearity is necessary for language (generation chain, 2026-11-02)

Language is full of interactions: what follows `th` is not what `t` predicts
plus what `h` predicts.  This file makes that precise.

The measure is the **interaction contrast** of a target `f` on two sources:
`f a b + f a' b' - f a b' - f a' b`.  Every map that treats its two sources
separately has contrast zero, so a target with nonzero contrast is out of reach.

* `additive_no_interaction`, `interaction_needs_nonadditive`: the core fact.
* `linear_read_of_superposed_state`, `linear_stack_no_interaction`: width and
  depth do not help; a stack of linear layers still has contrast zero.
* `linear_recurrence_no_interaction`: a linear recurrent memory with a linear
  read is additive in two past arrivals, whatever their embeddings are.
* `additive_logit_cannot_separate`: a final softmax does not help, because the
  log-odds between two outcomes are still additive.
* `additive_error_floor`: the loss is a floor set by the contrast alone,
  independent of the size of the body.
* `association_read_additive_in_contents`: a GDN-style read at a fixed query is
  additive in what is stored.  Retrieval cannot combine what it retrieves.

What is forced is a nonlinearity somewhere on the path between the two arrivals
and the output.  It need not be an MLP:

* `key_query_read_interacts`: a key from one arrival read with a query from the
  other interacts, so a learned address is already a nonlinearity.
* `one_rectifier_reads_interaction`: a single rectifier is enough for XOR.

The premise is a world condition.  On OpenWebText the log-odds contrast of the
next byte being `e` over `th/ta/sh/sa` is about 2.4 nats.  Holding and correcting,
the other reason for nonlinearity, is `GenerationPersistence` and
`linear_hold_keeps_error`.
-/

namespace CGB.Generation.Nonlinearity

variable {A B : Type*}

/-- The interaction contrast of a target on two values of each of two sources. -/
def contrast (f : A → B → ℝ) (a a' : A) (b b' : B) : ℝ :=
  f a b + f a' b' - f a b' - f a' b

/-- The two sources contribute separately. -/
def Additive (f : A → B → ℝ) : Prop :=
  ∃ (c : ℝ) (u : A → ℝ) (v : B → ℝ), ∀ a b, f a b = c + u a + v b

/-- A map that treats its sources separately has no interaction. -/
theorem additive_no_interaction {f : A → B → ℝ} (h : Additive f) (a a' : A) (b b' : B) :
    contrast f a a' b b' = 0 := by
  obtain ⟨c, u, v, hf⟩ := h
  simp only [contrast, hf]
  ring

/-- A target with an interaction is not additive. -/
theorem interaction_needs_nonadditive {f : A → B → ℝ} {a a' : A} {b b' : B}
    (h : contrast f a a' b b' ≠ 0) : ¬ Additive f :=
  fun hf => h (additive_no_interaction hf a a' b b')

/-- The difference of two additive maps is additive, so log-odds of additive logits are. -/
theorem Additive.sub {f g : A → B → ℝ} (hf : Additive f) (hg : Additive g) :
    Additive (fun a b => f a b - g a b) := by
  obtain ⟨c, u, v, hf⟩ := hf
  obtain ⟨c', u', v', hg⟩ := hg
  refine ⟨c - c', fun a => u a - u' a, fun b => v b - v' b, fun a b => ?_⟩
  simp only [hf, hg]
  ring

section Linear

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- A linear read of a superposed state is additive at any width. -/
theorem linear_read_of_superposed_state (p : A → E) (r : B → E) (read : E →ₗ[ℝ] ℝ) (c : ℝ) :
    Additive (fun a b => c + read (p a + r b)) :=
  ⟨c, fun a => read (p a), fun b => read (r b), fun a b => by simp only [map_add]; ring⟩

/-- Any stack of linear layers with biases keeps the two sources superposed. -/
theorem linear_stack_superposes (layers : List ((E →ₗ[ℝ] E) × E)) (p : A → E) (r : B → E) :
    ∃ (p' : A → E) (r' : B → E), ∀ a b,
      layers.foldr (fun layer x => layer.1 x + layer.2) (p a + r b) = p' a + r' b := by
  induction layers with
  | nil => exact ⟨p, r, fun _ _ => rfl⟩
  | cons layer rest ih =>
    obtain ⟨p', r', h⟩ := ih
    refine ⟨fun a => layer.1 (p' a) + layer.2, fun b => layer.1 (r' b), fun a b => ?_⟩
    simp only [List.foldr_cons, h, map_add]
    abel

/-- Depth does not help: a linear read of a linear stack has no interaction. -/
theorem linear_stack_no_interaction (layers : List ((E →ₗ[ℝ] E) × E)) (p : A → E) (r : B → E)
    (read : E →ₗ[ℝ] ℝ) (c : ℝ) :
    Additive (fun a b =>
      c + read (layers.foldr (fun layer x => layer.1 x + layer.2) (p a + r b))) := by
  obtain ⟨p', r', h⟩ := linear_stack_superposes layers p r
  simp only [h]
  exact linear_read_of_superposed_state p' r' read c

/-- A linear recurrent memory with a linear read is additive in two past
arrivals.  The embeddings `W₁, W₂` are arbitrary tables, so a learned embedding
does not help. -/
theorem linear_recurrence_no_interaction (L : E →ₗ[ℝ] E) (W₁ : A → E) (W₂ : B → E) (x₀ : E)
    (read : E →ₗ[ℝ] ℝ) (c : ℝ) :
    Additive (fun a b => c + read (L (L x₀ + W₁ a) + W₂ b)) := by
  refine ⟨c + read (L (L x₀)), fun a => read (L (W₁ a)), fun b => read (W₂ b), fun a b => ?_⟩
  simp only [map_add]
  ring

end Linear

/-- A final squashing does not help: additive log-odds cannot take the XOR sign pattern. -/
theorem additive_logit_cannot_separate {f : A → B → ℝ} (hf : Additive f) {a a' : A} {b b' : B}
    (h₁ : 0 < f a b) (h₂ : 0 < f a' b') (h₃ : f a b' < 0) (h₄ : f a' b < 0) : False := by
  have h := additive_no_interaction hf a a' b b'
  simp only [contrast] at h
  linarith

/-- The squared error of any additive predictor over the four cases is at least
a quarter of the squared contrast.  The bound does not mention the predictor's size. -/
theorem additive_error_floor (f : A → B → ℝ) {g : A → B → ℝ} (hg : Additive g)
    (a a' : A) (b b' : B) :
    contrast f a a' b b' ^ 2 / 4 ≤
      (f a b - g a b) ^ 2 + (f a' b' - g a' b') ^ 2 +
        (f a b' - g a b') ^ 2 + (f a' b - g a' b) ^ 2 := by
  have h := additive_no_interaction hg a a' b b'
  simp only [contrast] at h ⊢
  have hc : f a b + f a' b' - f a b' - f a' b =
      (f a b - g a b) + (f a' b' - g a' b') - (f a b' - g a b') - (f a' b - g a' b) := by
    linarith
  rw [hc]
  generalize f a b - g a b = e₁
  generalize f a' b' - g a' b' = e₂
  generalize f a b' - g a b' = e₃
  generalize f a' b - g a' b = e₄
  nlinarith [sq_nonneg (e₁ - e₂), sq_nonneg (e₁ + e₃), sq_nonneg (e₁ + e₄),
    sq_nonneg (e₂ + e₃), sq_nonneg (e₂ + e₄), sq_nonneg (e₃ - e₄)]

/-- At a fixed query, a GDN-style read is additive in what is stored: retrieval
weighs the contents and cannot combine them. -/
theorem association_read_additive_in_contents {dv dk : ℕ} (k : Fin 2 → Fin dk → ℝ)
    (q : Fin dk → ℝ) (o : Fin dv) (first : A → Fin dv → ℝ) (second : B → Fin dv → ℝ) :
    Additive (fun a b =>
      associationRead (associationState k ![first a, second b]) q o) := by
  refine ⟨0, fun a => first a o * ∑ j, k 0 j * q j, fun b => second b o * ∑ j, k 1 j * q j,
    fun a b => ?_⟩
  change associationRead (associationState k ![first a, second b]) q o = _
  rw [association_state_readout]
  simp [Fin.sum_univ_two]

/-- A key from one arrival read with a query from the other interacts, so a
learned address already supplies the nonlinearity. -/
theorem key_query_read_interacts :
    contrast (fun a b : ℝ =>
      associationRead
        (associationState (fun (_ : Fin 1) (_ : Fin 1) => a)
          (fun (_ : Fin 1) (_ : Fin 1) => (1 : ℝ)))
        (fun _ => b) (0 : Fin 1)) 1 (-1) 1 (-1) = 4 := by
  simp [contrast, associationRead, associationState]
  norm_num

/-- The real value of a bit. -/
def bit (x : Bool) : ℝ := if x then 1 else 0

/-- XOR has an interaction, and one rectifier is enough to read it. -/
theorem one_rectifier_reads_interaction :
    contrast (fun a b : Bool => bit (xor a b)) false true false true ≠ 0 ∧
      ∀ a b : Bool, bit a + bit b - 2 * max (bit a + bit b - 1) 0 = bit (xor a b) := by
  refine ⟨by norm_num [contrast, bit], fun a b => ?_⟩
  cases a <;> cases b <;> norm_num [bit]

end CGB.Generation.Nonlinearity
