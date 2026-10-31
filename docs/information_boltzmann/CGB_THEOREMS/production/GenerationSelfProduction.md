# GenerationSelfProduction

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean) · [返回导航](../README.md)

模块源码说明：Conditional existence of an open self-producing niche This module is deliberately world-independent.

## CGB.SelfProduction.two_phase_finite_score_step_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L33)

源码说明：On a finite alphabet with at least two symbols, a positive finite two-phase score step strictly improves the probability of the observed symbol on the same frozen feature. It does not assert future improvement.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [inst_1 : DecidableEq B] (z : B → ℝ) (target : B) (η : ℝ),
  0 < η →
    (∃ b, b ≠ target) →
      CGB.Medium.softmax z target <
        CGB.Medium.softmax (fun b => z b + η * ((if b = target then 1 else 0) - CGB.Medium.softmax z b)) target
```

## CGB.SelfProduction.softmax_nll_logit_error_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L80)

源码说明：If every byte logit changes by at most δ, the observed-byte negative log likelihood changes by at most 2δ. This is a per-readout bound, not a claim that the approximate state follows the reference on later ticks.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (z z' : B → ℝ) (target : B) (δ : ℝ),
  (∀ (b : B), |z' b - z b| ≤ δ) →
    |-Real.log (CGB.Medium.softmax z' target) - -Real.log (CGB.Medium.softmax z target)| ≤ 2 * δ
```

## CGB.SelfProduction.association_affine_language_nll_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L122)

源码说明：The associative interference certificate propagates through an affine language head into a per-byte NLL bound. The reference logits use the value associated with the same probe. No future distribution or training result is assumed. An old-state residual, if present, must be included separately.

形式陈述：

```lean
∀ {dv dk n nb : ℕ} (k q : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ) (head : Fin nb → Fin dv → ℝ) (bias : Fin nb → ℝ),
  (∀ (i : Fin n) (o : Fin dv), |v i o| ≤ 1) →
    ∀ (i : Fin n) (target : Fin nb) (c : ℝ),
      (∀ (b : Fin nb), ∑ o, |head b o| ≤ c) →
        have err := ∑ j, |∑ t, k j t * q i t - if j = i then 1 else 0|;
        have actual := fun b =>
          bias b + ∑ o, head b o * CGB.Generation.associationRead (CGB.Generation.associationState k v) (q i) o;
        have ideal := fun b => bias b + ∑ o, head b o * v i o;
        |-Real.log (CGB.Medium.softmax actual target) - -Real.log (CGB.Medium.softmax ideal target)| ≤ 2 * (err * c)
```

## CGB.SelfProduction.delayed_readout_capacity_with_retrieval

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L171)

源码说明：For a fixed future query, externally retrieved views share the burden of distinguishing histories with the finite body. If every listed history needs a different answer, the number of histories cannot exceed the product of body states and possible returned views. No assumption stores the external world inside the body; retrieval itself must actually supply the view.

形式陈述：

```lean
∀ {History : Type u_1} {State : Type u_2} {View : Type u_3} {Answer : Type u_4} [inst : Fintype History]
  [inst_1 : Fintype State] [inst_2 : Fintype View] (internal : History → State) (returned : History → View)
  (required : History → Answer) (read : State → View → Answer),
  Function.Injective required →
    (∀ (h : History), read (internal h) (returned h) = required h) →
      Fintype.card History ≤ Fintype.card State * Fintype.card View
```

## CGB.SelfProduction.association_delta_read_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L190)

源码说明：Assemble the existing row delta and associative readout contracts on the same value-by-key matrix. Unit-key write error contracts by `1 - β`; any query changes by its overlap with the written key times the innovation. This is the mean-state reference update, not the Bayesian assimilation path.

形式陈述：

```lean
∀ {dv dk : ℕ} (state : Fin dv → Fin dk → ℝ) (key query : Fin dk → ℝ) (value : Fin dv → ℝ) (β : ℝ),
  CGB.Generation.dot key key = 1 →
    (∀ (o : Fin dv),
        CGB.Generation.associationRead (fun i => CGB.Generation.tiedWrite key β (value i) (state i)) key o - value o =
          (1 - β) * (CGB.Generation.associationRead state key o - value o)) ∧
      ∀ (o : Fin dv),
        CGB.Generation.associationRead (fun i => CGB.Generation.tiedWrite key β (value i) (state i)) query o -
            CGB.Generation.associationRead state query o =
          β * (value o - CGB.Generation.associationRead state key o) * CGB.Generation.dot query key
```

## CGB.SelfProduction.one_hot_delta_is_precision_assimilation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L213)

源码说明：For an exactly one-hot key, no global decay, and precision-derived gain, the delta write agrees with slotwise precision assimilation: only the keyed slot changes. This does not assert equality for mixed keys or nonzero decay.

形式陈述：

```lean
∀ {n : ℕ} (slot : Fin n) (row : Fin n → ℝ) (p w value : ℝ),
  0 < p →
    0 ≤ w →
      CGB.Generation.tiedWrite (fun j => if j = slot then 1 else 0) (w / (p + w)) value row = fun j =>
        if j = slot then (p * row j + w * value) / (p + w) else row j
```

## CGB.SelfProduction.precision_distinguishes_equal_mean_histories

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L229)

源码说明：Two slots with the same current mean but different positive precisions produce different next means on the same nonmatching observation. Hence a deterministic transition reading only that mean cannot reproduce both.

形式陈述：

```lean
∀ (p₁ p₂ w mean observed : ℝ),
  0 < p₁ →
    0 < p₂ →
      0 < w →
        p₁ ≠ p₂ →
          observed ≠ mean →
            have next₁ := (p₁ * mean + w * observed) / (p₁ + w);
            have next₂ := (p₂ * mean + w * observed) / (p₂ + w);
            next₁ ≠ next₂ ∧ ¬∃ transition, transition mean observed = next₁ ∧ transition mean observed = next₂
```

## CGB.SelfProduction.mixed_key_delta_differs_from_slot_assimilation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L253)

源码说明：A normalized mixed key gives a concrete obstruction to general equality: the delta write couples the first slot to the second, while independent precision assimilation with zero target leaves its zero mean unchanged.

形式陈述：

```lean
have key := ![3 / 5, 4 / 5];
have row := ![0, 1];
CGB.Generation.dot key key = 1 ∧
  CGB.Generation.tiedWrite key (1 / 2) 0 row 0 = -(6 / 25) ∧ (1 * row 0 + 1 / 2 * 0) / (1 + 1 / 2) = 0
```

## CGB.SelfProduction.positive_soft_query_changes_under_delta

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L265)

源码说明：Strictly positive soft addresses necessarily overlap. Consequently a nonzero delta write changes every strictly positive raw query in any output coordinate with nonzero innovation. This is exact-real arithmetic; floating point softmax can underflow and dual or signed queries are not covered.

形式陈述：

```lean
∀ {dk : ℕ} [NeZero dk] (key query row : Fin dk → ℝ) (β value : ℝ),
  (∀ (i : Fin dk), 0 < key i) →
    (∀ (i : Fin dk), 0 < query i) →
      β ≠ 0 →
        value ≠ CGB.Generation.dot key row →
          0 < CGB.Generation.dot query key ∧
            CGB.Generation.dot query (CGB.Generation.tiedWrite key β value row) ≠ CGB.Generation.dot query row
```

## CGB.SelfProduction.association_assimilation_query_change

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L288)

源码说明：Query change under the existing natural-parameter assimilation: `weight` is the per-slot sum of observation weights and `weighted` the corresponding value-weighted sum. The identity includes zero observation weight and allows slot contributions to cancel; no separate resource state is introduced.

形式陈述：

```lean
∀ {dv dk : ℕ} (state weighted : Fin dv → Fin dk → ℝ) (precision weight query : Fin dk → ℝ),
  (∀ (j : Fin dk), 0 < precision j) →
    (∀ (j : Fin dk), 0 ≤ weight j) →
      ∀ (o : Fin dv),
        CGB.Generation.associationRead (fun i j => (precision j * state i j + weighted i j) / (precision j + weight j))
              query o -
            CGB.Generation.associationRead state query o =
          ∑ j, query j * ((weighted o j - weight j * state o j) / (precision j + weight j))
```

## CGB.SelfProduction.association_assimilation_improves_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L308)

源码说明：On a fixed query and target, the existing assimilation improves squared prediction error exactly when its query change points sufficiently towards the target without overshooting. The target is an externally specified future value, not assumed to be known during the update.

形式陈述：

```lean
∀ {dv dk : ℕ} (state weighted : Fin dv → Fin dk → ℝ) (precision weight query : Fin dk → ℝ),
  (∀ (j : Fin dk), 0 < precision j) →
    (∀ (j : Fin dk), 0 ≤ weight j) →
      ∀ (o : Fin dv) (target : ℝ),
        have old := CGB.Generation.associationRead state query o;
        have delta := ∑ j, query j * ((weighted o j - weight j * state o j) / (precision j + weight j));
        have updated :=
          CGB.Generation.associationRead
            (fun i j => (precision j * state i j + weighted i j) / (precision j + weight j)) query o;
        (updated - target) ^ 2 < (old - target) ^ 2 ↔ delta * (2 * (old - target) + delta) < 0
```

## CGB.SelfProduction.association_assimilation_softmax_improves_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L336)

源码说明：Under slotwise assimilation and an arbitrary affine-head update, a target's softmax probability rises exactly when its multiplicative change exceeds the old distribution's average. The change includes head, query and state changes; no learned head or update rule is assumed.

形式陈述：

```lean
∀ {dv dk nb : ℕ} (state weighted : Fin dv → Fin dk → ℝ) (precision weight query query' : Fin dk → ℝ)
  (head head' : Fin nb → Fin dv → ℝ) (bias bias' : Fin nb → ℝ),
  (∀ (j : Fin dk), 0 < precision j) →
    (∀ (j : Fin dk), 0 ≤ weight j) →
      ∀ (target : Fin nb),
        have old := fun b => bias b + ∑ i, head b i * CGB.Generation.associationRead state query i;
        have updated := fun b =>
          bias' b +
            ∑ i,
              head' b i *
                CGB.Generation.associationRead
                  (fun i j => (precision j * state i j + weighted i j) / (precision j + weight j)) query' i;
        have delta := fun b =>
          bias' b - bias b +
            ∑ i,
              ((head' b i - head b i) * CGB.Generation.associationRead state query i +
                head' b i *
                  (∑ j, state i j * (query' j - query j) +
                    ∑ j, query' j * ((weighted i j - weight j * state i j) / (precision j + weight j))));
        CGB.Medium.softmax old target < CGB.Medium.softmax updated target ↔
          (∑ i, Real.exp (old i) * Real.exp (delta i)) / ∑ i, Real.exp (old i) < Real.exp (delta target)
```

## CGB.SelfProduction.repeated_assimilation_retention

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L399)

源码说明：With no intervening prior step, repeated identical slot evidence is accumulated by the existing natural/precision update. Its resulting mean retains exactly `p₀ / (p₀ + n*w)` of the initial error. This describes one slot in exact arithmetic, not a learned language-quality guarantee.

形式陈述：

```lean
∀ (precision natural : ℕ → ℝ) (p₀ m₀ w target : ℝ),
  0 < p₀ →
    0 ≤ w →
      precision 0 = p₀ →
        natural 0 = p₀ * m₀ →
          (∀ (n : ℕ), precision (n + 1) = precision n + w) →
            (∀ (n : ℕ), natural (n + 1) = natural n + w * target) →
              ∀ (n : ℕ),
                precision n = p₀ + ↑n * w ∧
                  natural n / precision n - target = p₀ / (p₀ + ↑n * w) * (m₀ - target) ∧
                    (↑n * w ≤ p₀ → 1 / 2 ≤ p₀ / (p₀ + ↑n * w))
```

## CGB.SelfProduction.repeated_assimilation_association_readout

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L434)

源码说明：The one-slot retention law composes into the actual value-by-key associative readout. Each slot may have its own precision and evidence weight. The resulting terms can cancel, so the equality does not claim that an individual association survives merely because a query error is small.

形式陈述：

```lean
∀ {dv dk : ℕ} (precision : ℕ → Fin dk → ℝ) (natural : ℕ → Fin dv → Fin dk → ℝ) (p₀ w query : Fin dk → ℝ)
  (m₀ target : Fin dv → Fin dk → ℝ),
  (∀ (j : Fin dk), 0 < p₀ j) →
    (∀ (j : Fin dk), 0 ≤ w j) →
      (∀ (j : Fin dk), precision 0 j = p₀ j) →
        (∀ (i : Fin dv) (j : Fin dk), natural 0 i j = p₀ j * m₀ i j) →
          (∀ (n : ℕ) (j : Fin dk), precision (n + 1) j = precision n j + w j) →
            (∀ (n : ℕ) (i : Fin dv) (j : Fin dk), natural (n + 1) i j = natural n i j + w j * target i j) →
              ∀ (n : ℕ) (i : Fin dv),
                CGB.Generation.associationRead (fun o j => natural n o j / precision n j) query i -
                    CGB.Generation.associationRead target query i =
                  ∑ j, query j * (p₀ j / (p₀ j + ↑n * w j)) * (m₀ i j - target i j)
```

## CGB.SelfProduction.repeated_assimilation_probe_family

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L460)

源码说明：A family of future queries remains accurate when its target readout is accurate and the summed retained slot deviations fit the remaining tolerance. This gives an explicit no-cancellation sufficient certificate for mixed slots.

形式陈述：

```lean
∀ {dv dk m : ℕ} (precision : ℕ → Fin dk → ℝ) (natural : ℕ → Fin dv → Fin dk → ℝ) (p₀ w : Fin dk → ℝ)
  (m₀ target : Fin dv → Fin dk → ℝ) (keys : Fin m → Fin dk → ℝ) (answers : Fin m → Fin dv → ℝ) (ε₀ ε : ℝ) (n : ℕ),
  (∀ (j : Fin dk), 0 < p₀ j) →
    (∀ (j : Fin dk), 0 ≤ w j) →
      (∀ (j : Fin dk), precision 0 j = p₀ j) →
        (∀ (i : Fin dv) (j : Fin dk), natural 0 i j = p₀ j * m₀ i j) →
          (∀ (t : ℕ) (j : Fin dk), precision (t + 1) j = precision t j + w j) →
            (∀ (t : ℕ) (i : Fin dv) (j : Fin dk), natural (t + 1) i j = natural t i j + w j * target i j) →
              0 ≤ ε →
                CGB.Generation.probeFamilyAccurate target keys answers ε₀ →
                  (∀ (r : Fin m) (i : Fin dv),
                      ∑ j, |keys r j * (p₀ j / (p₀ j + ↑n * w j)) * (m₀ i j - target i j)| + ε₀ ≤ ε) →
                    CGB.Generation.probeFamilyAccurate (fun i j => natural n i j / precision n j) keys answers ε
```

## CGB.SelfProduction.state_conditioned_logloss_le_fixed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L503)

源码说明：In a declared joint distribution, an oracle using the actual conditional byte law at each body state has no higher expected log loss than any fixed strictly positive normalized predictor. The weights are state probabilities; no additional resource state is introduced. This does not establish that the body can learn the conditional law, or that prediction improves viability.

形式陈述：

```lean
∀ {State : Type u_1} {Byte : Type u_2} [inst : Fintype State] [inst_1 : Fintype Byte] (weight : State → ℝ)
  (law : State → Byte → ℝ) (fixed : Byte → ℝ),
  (∀ (s : State), 0 ≤ weight s) →
    (∀ (s : State) (b : Byte), 0 ≤ law s b) →
      (∀ (s : State), ∑ b, law s b = 1) →
        (∀ (b : Byte), 0 < fixed b) →
          ∑ b, fixed b = 1 →
            ∑ s, weight s * CGB.PredictiveHarvest.entropy (law s) ≤
              ∑ s, weight s * CGB.PredictiveHarvest.crossEntropy (law s) fixed
```

## CGB.SelfProduction.merged_histories_binary_logloss_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L522)

源码说明：If two equally likely histories share the same accessible state but require opposite next-byte predictions, any positive binary predictor pays at least `log 2` nats of expected loss. A predictor with the distinguishing information can assign probability one to the correct byte and has zero loss. The equal probabilities are a world condition, not a universal drift law.

形式陈述：

```lean
∀ {History : Type u_1} {State : Type u_2} (encode : History → State) (predict : State → ℝ) (h₀ h₁ : History),
  encode h₀ = encode h₁ →
    0 < predict (encode h₀) →
      predict (encode h₀) < 1 → -(Real.log (1 - predict (encode h₀)) + Real.log (predict (encode h₁))) / 2 ≥ Real.log 2
```

## CGB.SelfProduction.state_conditioned_logloss_decomposition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L547)

源码说明：For a state-dependent predictor, expected log loss is the conditional entropy plus the state-weighted KL mismatch. This uses a specified joint law, not a new resource account, and does not assert the predictor is learned.

形式陈述：

```lean
∀ {State : Type u_1} {Byte : Type u_2} [inst : Fintype State] [inst_1 : Fintype Byte] (weight : State → ℝ)
  (law predictor : State → Byte → ℝ),
  (∀ (s : State), 0 ≤ weight s) →
    (∀ (s : State) (b : Byte), 0 ≤ law s b) →
      (∀ (s : State), ∑ b, law s b = 1) →
        (∀ (s : State) (b : Byte), 0 < predictor s b) →
          (∀ (s : State), ∑ b, predictor s b = 1) →
            ∑ s, weight s * CGB.PredictiveHarvest.crossEntropy (law s) (predictor s) =
                ∑ s, weight s * CGB.PredictiveHarvest.entropy (law s) +
                  ∑ s, weight s * CGB.InformationEnergy.kl (law s) (predictor s) ∧
              0 ≤ ∑ s, weight s * CGB.InformationEnergy.kl (law s) (predictor s)
```

## CGB.SelfProduction.renewable_reopening_not_single_old_advance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L575)

源码说明：The existing renewable witness actually reopens an excluded candidate; no single old `advance` step from the same memory can implement that change.

形式陈述：

```lean
∀ (a : CGB.RenewableLearning.Action) (o : CGB.LearningMaintenance.Outcome),
  CGB.RenewableLearning.memory
      (CGB.RenewableLearning.next true true CGB.RenewableLearning.Signal.miss CGB.RenewableLearning.Cell.aReady) ≠
    CGB.CausalLearning.advance CGB.RenewableLearning.possible
      (CGB.RenewableLearning.memory CGB.RenewableLearning.Cell.aReady) () a o
```

## CGB.SelfProduction.renewable_witness_unbounded_supported_writes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L586)

源码说明：The same existing renewable process has arbitrarily long supported prefixes with more than any requested number of paid writes. This is a finite-prefix support statement, not an almost-sure infinite-path claim.

形式陈述：

```lean
∀ (B : ℕ),
  ∃ T,
    0 <
        CGB.RenewableLearningControls.weight (CGB.RenewableLearning.process true true false)
          CGB.RenewableLearningControls.path T ∧
      B <
        ∑ t ∈ Finset.range T,
          CGB.RenewableLearning.writeCost true (CGB.RenewableLearning.sense (CGB.RenewableLearningControls.path t).2 ())
            ((CGB.RenewableLearningControls.path t).1 ())
```

## CGB.SelfProduction.renewable_maintenance_and_supported_relearning

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L607)

源码说明：One stochastic body-world process has operational maintenance and arbitrarily many paid writes on supported finite prefixes; blocking reopening in the same world and initial conditions lowers the registered survival score by at least one half at tick six. The intervention also changes the declared outgoing accounting, and the prefix claim is not almost-sure infinite writing.

形式陈述：

```lean
CGB.Exchange.Persists CGB.RenewableLearningClosure.body CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
    CGB.RenewableLearningClosure.material 0 ∧
  CGB.Exchange.SelfMaintains CGB.RenewableLearningClosure.body CGB.RenewableLearning.world
      CGB.RenewableLearningClosure.ec CGB.RenewableLearningClosure.material ∧
    (∀ (B : ℕ),
        ∃ T,
          0 <
              CGB.RenewableLearningControls.weight (CGB.RenewableLearning.process true true false)
                CGB.RenewableLearningControls.path T ∧
            B <
              ∑ t ∈ Finset.range T,
                CGB.RenewableLearning.writeCost true
                  (CGB.RenewableLearning.sense (CGB.RenewableLearningControls.path t).2 ())
                  ((CGB.RenewableLearningControls.path t).1 ())) ∧
      CGB.Exchange.survival (CGB.RenewableLearning.plant true true) CGB.RenewableLearning.world ⋯
            (CGB.RenewableLearning.ledger true true) false (fun x => CGB.RenewableLearning.world.sense) 6 =
          1 ∧
        CGB.Exchange.survival (CGB.RenewableLearning.plant false true) CGB.RenewableLearning.world ⋯
            (CGB.RenewableLearning.ledger false true) false (fun x => CGB.RenewableLearning.world.sense) 6 ≤
          1 / 2
```

## CGB.SelfProduction.renewable_maintenance_world_action_contrast

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L641)

源码说明：The already-certified stochastic process also has a concrete world-side action contrast. On live bodies its world law depends on the body only through the selected action. Two body states select different actions and give different next-world probabilities. This is not a fixed-body action intervention or a proof about a language world.

形式陈述：

```lean
CGB.Exchange.SelfMaintains CGB.RenewableLearningClosure.body CGB.RenewableLearning.world CGB.RenewableLearningClosure.ec
    CGB.RenewableLearningClosure.material ∧
  (∀ (w v : CGB.RenewableLearning.Env) (q q' : CGB.RenewableLearning.Cell),
      q ≠ CGB.RenewableLearning.Cell.vacuum →
        q' ≠ CGB.RenewableLearning.Cell.vacuum →
          CGB.RenewableLearning.policy q = CGB.RenewableLearning.policy q' →
            CGB.RenewableLearning.moves w (fun x => q) v = CGB.RenewableLearning.moves w (fun x => q') v) ∧
    CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.aReady = 0 ∧
      CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.bReady = 1 ∧
        CGB.RenewableLearning.moves CGB.RenewableLearning.Env.feedA (fun x => CGB.RenewableLearning.Cell.aReady)
              CGB.RenewableLearning.Env.missA =
            0 ∧
          CGB.RenewableLearning.moves CGB.RenewableLearning.Env.feedA (fun x => CGB.RenewableLearning.Cell.bReady)
              CGB.RenewableLearning.Env.missA =
            1
```

## CGB.SelfProduction.renewable_action_world_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L680)

源码说明：This action kernel is normalized and agrees with the existing world for every live body. Explicit actions induce different next-world laws at a fixed world state. A do-action interpretation assumes this world mechanism remains invariant under intervention; multi-step survival is a separate obligation.

形式陈述：

```lean
(∀ (a : CGB.RenewableLearning.Action),
    CGB.SelfProduction.renewableActionRepresentative a ≠ CGB.RenewableLearning.Cell.vacuum ∧
      CGB.RenewableLearning.policy (CGB.SelfProduction.renewableActionRepresentative a) = a) ∧
  (∀ (w : CGB.RenewableLearning.Env) (a : CGB.RenewableLearning.Action),
      (∀ (v : CGB.RenewableLearning.Env), 0 ≤ CGB.SelfProduction.renewableActionWorld w a v) ∧
        ∑ v, CGB.SelfProduction.renewableActionWorld w a v = 1) ∧
    (∀ (w v : CGB.RenewableLearning.Env) (q : CGB.RenewableLearning.Cell),
        q ≠ CGB.RenewableLearning.Cell.vacuum →
          CGB.RenewableLearning.moves w (fun x => q) v =
            CGB.SelfProduction.renewableActionWorld w (CGB.RenewableLearning.policy q) v) ∧
      CGB.SelfProduction.renewableActionWorld CGB.RenewableLearning.Env.feedA 0 CGB.RenewableLearning.Env.missA = 0 ∧
        CGB.SelfProduction.renewableActionWorld CGB.RenewableLearning.Env.feedA 1 CGB.RenewableLearning.Env.missA = 1
```

## CGB.SelfProduction.renewable_action_machine_reproduces

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L739)

源码说明：Supplying the old policy reproduces every joint event and successor-state probability of the original uncut process, including the vacuum case.

形式陈述：

```lean
∀ (t : ℕ) (z z' : CGB.RenewableLearning.S) (e : Option Unit),
  CGB.SelfProduction.renewableActionMachine.step (CGB.RenewableLearning.policy (z.1 ())) z e z' =
    (CGB.RenewableLearning.process true true false).step t z e z'
```

## CGB.SelfProduction.renewable_action_autonomous_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L767)

源码说明：Closing the explicit action port with the existing policy recovers the whole original machine, not just a one-step marginal or a sample trace.

形式陈述：

```lean
CGB.SelfProduction.renewableActionAutonomous = CGB.RenewableLearning.process true true false
```

## CGB.SelfProduction.renewable_action_zero_failure_steps

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L776)

源码说明：Under the declared action-port intervention, holding action 0 sends the existing initial state to the trap and then to vacuum in two certain steps.

形式陈述：

```lean
CGB.SelfProduction.renewableActionMachine.step 0 CGB.RenewableLearning.initial (some ())
      (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.trap) =
    1 ∧
  CGB.SelfProduction.renewableActionMachine.step 0
      (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.trap) (some ())
      (fun x => CGB.RenewableLearning.Cell.vacuum, CGB.RenewableLearning.Env.trap) =
    1
```

## CGB.SelfProduction.renewable_action_zero_eventual_vacuum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L800)

源码说明：In the specified intervention model, constant action zero reaches vacuum with probability one after two ticks and remains there. This is a distinct intervention from the registered marginal-ghost cut, not a general survival or language-world theorem.

形式陈述：

```lean
∀ (n : ℕ),
  2 ≤ n →
    CGB.Energetics.law (CGB.SelfProduction.renewableActionAlways 0) CGB.RenewableLearning.initial n
        (fun x => CGB.RenewableLearning.Cell.vacuum, CGB.RenewableLearning.Env.trap) =
      1
```

## CGB.SelfProduction.renewable_world_hidden_branch_and_delayed_consequence

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L862)

源码说明：In the same certified world, aligned ready bodies have two equally likely next-world states while their next body state is the same. The current sensory labels of those world states coincide, yet a subsequent action 0 distinguishes them by the world's next `missB` probability. This is a conditional one-step world-law statement, not a claim of infinite entropy or learned prediction.

形式陈述：

```lean
(∀ (q : CGB.RenewableLearning.Cell),
    q = CGB.RenewableLearning.Cell.aReady ∨ q = CGB.RenewableLearning.Cell.bReady →
      have w :=
        if q = CGB.RenewableLearning.Cell.aReady then CGB.RenewableLearning.Env.feedA
        else CGB.RenewableLearning.Env.feedB;
      CGB.SelfProduction.renewableActionMachine.step (CGB.RenewableLearning.policy q) (fun x => q, w) (some ())
            (fun x => q, CGB.RenewableLearning.Env.feedA) =
          1 / 2 ∧
        CGB.SelfProduction.renewableActionMachine.step (CGB.RenewableLearning.policy q) (fun x => q, w) (some ())
            (fun x => q, CGB.RenewableLearning.Env.feedB) =
          1 / 2) ∧
  CGB.RenewableLearning.sense CGB.RenewableLearning.Env.feedA () =
      CGB.RenewableLearning.sense CGB.RenewableLearning.Env.feedB () ∧
    CGB.SelfProduction.renewableActionWorld CGB.RenewableLearning.Env.feedA 0 CGB.RenewableLearning.Env.missB = 0 ∧
      CGB.SelfProduction.renewableActionWorld CGB.RenewableLearning.Env.feedB 0 CGB.RenewableLearning.Env.missB = 1
```

## CGB.SelfProduction.renewable_world_aligned_state_supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L886)

源码说明：The aligned `aReady/feedA` state used above has a positive-probability three-step prefix from the actual initial state in the original process.

形式陈述：

```lean
∃ path,
  path 0 = CGB.RenewableLearning.initial ∧
    path 3 = (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.feedA) ∧
      0 < CGB.RenewableLearningControls.weight (CGB.RenewableLearning.process true true false) path 3
```

## CGB.SelfProduction.renewable_hidden_branch_reopens_and_rewrites

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L912)

源码说明：A positive-probability hidden world change in the certified process can flow through mismatch, reopening, probing and a new write on the same body. At the resulting state the learned action and an action override induce distinct next-world outcomes. This is a supported branch, not a language capability.

形式陈述：

```lean
∀ (t : ℕ),
  have P := CGB.RenewableLearning.process true true false;
  have s0 := (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.feedA);
  have s1 := (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.feedB);
  have s2 := (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.missB);
  have s3 := (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.probeB);
  have s4 := (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.answerB);
  have s5 := (fun x => CGB.RenewableLearning.Cell.bSpent, CGB.RenewableLearning.Env.feedB);
  have s6 := (fun x => CGB.RenewableLearning.Cell.bReady, CGB.RenewableLearning.Env.feedB);
  P.step t s0 (some ()) s1 = 1 / 2 ∧
    P.step (t + 1) s1 (some ()) s2 = 1 ∧
      P.step (t + 2) s2 (some ()) s3 = 1 ∧
        P.step (t + 3) s3 (some ()) s4 = 1 ∧
          P.step (t + 4) s4 (some ()) s5 = 1 ∧
            P.step (t + 5) s5 (some ()) s6 = 1 / 2 ∧
              CGB.RenewableLearning.writeCost true CGB.RenewableLearning.Signal.miss CGB.RenewableLearning.Cell.aReady =
                  1 ∧
                CGB.RenewableLearning.writeCost true CGB.RenewableLearning.Signal.answerB
                      CGB.RenewableLearning.Cell.search =
                    1 ∧
                  CGB.RenewableLearning.topup true CGB.RenewableLearning.Signal.feed CGB.RenewableLearning.Cell.bSpent =
                      2 ∧
                    CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.bReady = 1 ∧
                      CGB.SelfProduction.renewableActionMachine.step 1 s6 (some ())
                            (fun x => CGB.RenewableLearning.Cell.bReady, CGB.RenewableLearning.Env.feedA) =
                          1 / 2 ∧
                        CGB.SelfProduction.renewableActionMachine.step 1 s6 (some ())
                              (fun x => CGB.RenewableLearning.Cell.bReady, CGB.RenewableLearning.Env.feedB) =
                            1 / 2 ∧
                          CGB.SelfProduction.renewableActionMachine.step 0 s6 (some ())
                              (fun x => CGB.RenewableLearning.Cell.bReady, CGB.RenewableLearning.Env.missB) =
                            1
```

## CGB.SelfProduction.renewable_reopen_control_branch

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L951)

源码说明：The recorded normal and reopen-disabled paths agree through tick four. On the same mismatch, the normal body reopens while the disabled body holds the old candidate; the following world move distinguishes the resulting actions. Both paths and the paired survival bound are existing certificates.

形式陈述：

```lean
(∀ n ≤ 4, CGB.RenewableLearningControls.path n = CGB.RenewableLearningControls.failingPath true n) ∧
  CGB.RenewableLearning.next true true CGB.RenewableLearning.Signal.miss CGB.RenewableLearning.Cell.aReady =
      CGB.RenewableLearning.Cell.search ∧
    CGB.RenewableLearning.next false true CGB.RenewableLearning.Signal.miss CGB.RenewableLearning.Cell.aReady =
        CGB.RenewableLearning.Cell.aHeld ∧
      CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.search = 2 ∧
        CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.aHeld = 0 ∧
          CGB.RenewableLearning.moves CGB.RenewableLearning.Env.probeB (fun x => CGB.RenewableLearning.Cell.search)
                CGB.RenewableLearning.Env.answerB =
              1 ∧
            CGB.RenewableLearning.moves CGB.RenewableLearning.Env.probeB (fun x => CGB.RenewableLearning.Cell.aHeld)
                CGB.RenewableLearning.Env.trap =
              1
```

## CGB.SelfProduction.renewable_observable_block_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L996)

源码说明：Every step of either block, including the transition to either next block, has positive probability in the unchanged autonomous process. The world sensor at offset two distinguishes the two blocks. These are finite-path support facts, not an almost-sure statement or a language-performance experiment.

形式陈述：

```lean
(∀ (bit next : Bool) (i : Fin 11) (t : ℕ),
    0 <
      (CGB.RenewableLearning.process true true false).step t (CGB.SelfProduction.renewableObservableBlock bit i)
        (some ())
        (if h : ↑i + 1 < 11 then CGB.SelfProduction.renewableObservableBlock bit ⟨↑i + 1, h⟩
        else CGB.SelfProduction.renewableObservableBlock next ⟨0, ⋯⟩)) ∧
  ∀ (bit : Bool),
    CGB.RenewableLearning.sense (CGB.SelfProduction.renewableObservableBlock bit ⟨2, ⋯⟩).2 () =
      if bit = true then CGB.RenewableLearning.Signal.miss else CGB.RenewableLearning.Signal.feed
```

## CGB.SelfProduction.renewable_observable_path_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1031)

源码说明：Every finite prefix of every selected branch sequence has positive probability from the aligned state, and the existing sensor at offset two of each block recovers its selected bit. This supplies arbitrarily many distinct observable finite prefixes without assuming an infinite-path measure.

形式陈述：

```lean
∀ (bits : ℕ → Bool),
  CGB.SelfProduction.renewableObservablePath bits 0 =
      (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.feedA) ∧
    (∀ (n : ℕ),
        0 <
          (CGB.RenewableLearning.process true true false).step n (CGB.SelfProduction.renewableObservablePath bits n)
            (some ()) (CGB.SelfProduction.renewableObservablePath bits (n + 1))) ∧
      ∀ (k : ℕ),
        CGB.RenewableLearning.sense (CGB.SelfProduction.renewableObservablePath bits (11 * k + 2)).2 () =
          if bits k = true then CGB.RenewableLearning.Signal.miss else CGB.RenewableLearning.Signal.feed
```

## CGB.SelfProduction.renewable_observable_from_birth

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1075)

源码说明：From the same original initial state, every finite prefix of every chosen bit sequence has positive support; sensory symbols distinguish the blocks, and each true block includes two actual paid writes. Individual infinite paths may have probability zero; no entropy-rate or full life claim is made.

形式陈述：

```lean
∀ (bits : ℕ → Bool),
  CGB.SelfProduction.renewableObservableFromBirth bits 0 = CGB.RenewableLearning.initial ∧
    (∀ (N : ℕ),
        0 <
          CGB.RenewableLearningControls.weight (CGB.RenewableLearning.process true true false)
            (CGB.SelfProduction.renewableObservableFromBirth bits) N) ∧
      (∀ (k : ℕ),
          CGB.RenewableLearning.sense (CGB.SelfProduction.renewableObservableFromBirth bits (3 + 11 * k + 2)).2 () =
            if bits k = true then CGB.RenewableLearning.Signal.miss else CGB.RenewableLearning.Signal.feed) ∧
        ∀ (k : ℕ),
          bits k = true →
            CGB.RenewableLearning.writeCost true
                  (CGB.RenewableLearning.sense (CGB.SelfProduction.renewableObservableFromBirth bits (3 + 11 * k + 2)).2
                    ())
                  ((CGB.SelfProduction.renewableObservableFromBirth bits (3 + 11 * k + 2)).1 ()) =
                1 ∧
              CGB.RenewableLearning.writeCost true
                  (CGB.RenewableLearning.sense (CGB.SelfProduction.renewableObservableFromBirth bits (3 + 11 * k + 4)).2
                    ())
                  ((CGB.SelfProduction.renewableObservableFromBirth bits (3 + 11 * k + 4)).1 ()) =
                1
```

## CGB.SelfProduction.trajectory_viable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1169)

源码说明：The same rule generates a valid state at every finite time.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I) (n : ℕ),
  c.viable (CGB.SelfProduction.trajectory c n)
```

## CGB.SelfProduction.actual_actions_reproduce_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1185)

源码说明：Replaying the actual actions in the same world reproduces the generated trajectory.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I) (n : ℕ),
  CGB.SelfProduction.controlledTrajectory c (fun t => c.action (CGB.SelfProduction.trajectory c t)) n =
    CGB.SelfProduction.trajectory c n
```

## CGB.SelfProduction.passive_opportunity_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1196)

源码说明：If each passive use spends from a finite opportunity potential, the total passive intake from that point cannot exceed the starting potential. The potential is a function of the existing state, not an extra state.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I) (actions : ℕ → A) (passive : A)
  (opportunity : S → ℕ) (intake : S → I → ℕ),
  (∀ (s : S), opportunity (c.step s (c.response s passive)) + intake s (c.response s passive) ≤ opportunity s) →
    ∀ (start : ℕ),
      (∀ (n : ℕ), start ≤ n → actions n = passive) →
        ∀ (T : ℕ),
          (opportunity (CGB.SelfProduction.controlledTrajectory c actions (start + T)) +
              ∑ n ∈ Finset.range T,
                have s := CGB.SelfProduction.controlledTrajectory c actions (start + n);
                intake s (c.response s (actions (start + n)))) ≤
            opportunity (CGB.SelfProduction.controlledTrajectory c actions start)
```

## CGB.SelfProduction.passive_opportunity_bounds_eventual_interventions

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1223)

源码说明：A finite passive opportunity potential supplies the universal counterfactual bound required by `counterfactual_passive_bound_forces_activity`.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I) (passive : A)
  (opportunity : S → ℕ) (intake : S → I → ℕ),
  (∀ (s : S), opportunity (c.step s (c.response s passive)) + intake s (c.response s passive) ≤ opportunity s) →
    ∀ (actions : ℕ → A),
      CGB.ActiveInference.eventuallyPassive actions passive →
        ∃ B,
          ∀ (T : ℕ),
            CGB.ActiveInference.cumulative
                (fun n =>
                  have s := CGB.SelfProduction.controlledTrajectory c actions n;
                  intake s (c.response s (actions n)))
                T ≤
              B
```

## CGB.SelfProduction.counterfactual_passive_bound_forces_activity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1259)

源码说明：If every eventually passive intervention in the same world has bounded intake, an actual trajectory with unbounded intake cannot become permanently passive. The universal premise also covers the always-passive intervention.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I) (passive : A)
  (intake : S → I → ℕ),
  (∀ (actions : ℕ → A),
      CGB.ActiveInference.eventuallyPassive actions passive →
        ∃ B,
          ∀ (T : ℕ),
            CGB.ActiveInference.cumulative
                (fun n =>
                  have s := CGB.SelfProduction.controlledTrajectory c actions n;
                  intake s (c.response s (actions n)))
                T ≤
              B) →
    (CGB.ActiveInference.unbounded fun n =>
        have s := CGB.SelfProduction.trajectory c n;
        intake s (c.response s (c.action s))) →
      ¬CGB.ActiveInference.eventuallyPassive (fun n => c.action (CGB.SelfProduction.trajectory c n)) passive
```

## CGB.SelfProduction.passive_opportunity_forces_activity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1283)

源码说明：Finite passive opportunity and unbounded actual intake force recurrent non-passive action in the same deterministic world.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I) (passive : A)
  (opportunity : S → ℕ) (intake : S → I → ℕ),
  (∀ (s : S), opportunity (c.step s (c.response s passive)) + intake s (c.response s passive) ≤ opportunity s) →
    (CGB.ActiveInference.unbounded fun n =>
        have s := CGB.SelfProduction.trajectory c n;
        intake s (c.response s (c.action s))) →
      ¬CGB.ActiveInference.eventuallyPassive (fun n => c.action (CGB.SelfProduction.trajectory c n)) passive
```

## CGB.SelfProduction.exists_infinite_self_producing_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1298)

源码说明：**Unbounded self-production.** The contract generates one trajectory indexed by all natural causal ticks, and every prefix remains in its viable niche. This is the deterministic infinite-duration statement.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I),
  ∃ x,
    x 0 = c.initial ∧
      (∀ (n : ℕ), x (n + 1) = c.step (x n) (c.response (x n) (c.action (x n)))) ∧ ∀ (n : ℕ), c.viable (x n)
```

## CGB.SelfProduction.action_consequence_model_exists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1311)

源码说明：The action-conditioned boundary rule itself is an exact self-consequence model when that rule is available to the organism's persistent state.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I),
  ∃ predict, ∀ (s : S) (a : A), predict s a = c.step s (c.response s a)
```

## CGB.SelfProduction.action_distinction_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1318)

源码说明：If two actions have different boundary consequences at a viable state, the exact self-consequence model preserves that distinction.

形式陈述：

```lean
∀ {S : Type u_1} {A : Type u_2} {I : Type u_3} (c : CGB.SelfProduction.Contract S A I) {s : S},
  c.viable s →
    ∀ {a a' : A},
      c.step s (c.response s a) ≠ c.step s (c.response s a') →
        ∃ predict, predict s a ≠ predict s a' ∧ ∀ (t : S) (b : A), predict t b = c.step t (c.response t b)
```

## CGB.SelfProduction.language_viable_next

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1479)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  {s : CGB.CausalLearning.Memory Model L.base.reserve},
  CGB.SelfProduction.language_viable L s → ∀ (b : Byte), CGB.SelfProduction.language_viable L (L.base.next s b)
```

## CGB.SelfProduction.candidate_erase_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1507)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O),
  (CGB.CausalLearning.advance possible m c a o).candidates.card +
      CGB.SelfProduction.candidate_erase_count possible m c a o =
    m.candidates.card
```

## CGB.SelfProduction.candidate_erase_ge_fee

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1516)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O),
  CGB.CausalLearning.fee possible m c a o ≤ CGB.SelfProduction.candidate_erase_count possible m c a o
```

## CGB.SelfProduction.effective_write_erases_distinction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1524)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {H : Type u_1} {C : Type u_2} {A : Type u_3} {O : Type u_4} [inst : Fintype H] [inst_1 : DecidableEq H]
  (possible : CGB.CausalLearning.Support H C A O) {B : ℕ} (m : CGB.CausalLearning.Memory H B) (c : C) (a : A) (o : O),
  CGB.CausalLearning.fee possible m c a o = 1 → 0 < CGB.SelfProduction.candidate_erase_count possible m c a o
```

## CGB.SelfProduction.language_state_capacity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1533)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  Fintype.card (CGB.CausalLearning.Memory Model L.base.reserve) = 2 ^ Fintype.card Model * (L.base.reserve + 1)
```

## CGB.SelfProduction.generated_history_encodes_self_production

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1574)

源码说明：The generated history and the existing state recursion agree without a separately postulated trajectory-alignment premise.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (n : ℕ),
  L.base.encode (CGB.SelfProduction.language_generated_history L n) =
    CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n
```

## CGB.SelfProduction.generated_boundary_signal_forces_active

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1593)

源码说明：On the generated trajectory, a boundary-derived nonnegative integer signal supports the existing active conclusion when the explicit unbounded intake and passive-depletion conditions hold. No extra stored state is used.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (intake : CGB.CausalLearning.Memory Model L.base.reserve → Byte → ℕ),
  (CGB.ActiveInference.unbounded fun n =>
      have s := CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n;
      intake s (L.world_response s (L.base.policy s))) →
    CGB.ActiveInference.passiveBounded
        (fun n => L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n))
        (fun n =>
          have s := CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n;
          intake s (L.world_response s (L.base.policy s)))
        L.base.passive →
      (∀ (n : ℕ),
          CGB.SelfProduction.language_viable L
            (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)) ∧
        ¬CGB.ActiveInference.eventuallyPassive
            (fun n => L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n))
            L.base.passive
```

## CGB.SelfProduction.language_admission_trajectory_aligned

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1614)

源码说明：The admission history follows the self-production trajectory when its encoded initial state and every action-conditioned transition agree. Only encoded state equality is required; histories themselves need not coincide.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  L.base.encode (L.base.trajectory 0) = L.base.encode L.seed →
    (∀ (n : ℕ),
        L.base.encode (L.base.trajectory (n + 1)) =
          L.base.next (L.base.encode (L.base.trajectory n))
            (L.world_response (L.base.encode (L.base.trajectory n))
              (L.base.policy (L.base.encode (L.base.trajectory n))))) →
      ∀ (n : ℕ),
        L.base.encode (L.base.trajectory n) = CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n
```

## CGB.SelfProduction.language_self_production_active_if_aligned

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1634)

源码说明：With the alignment conditions, the existing active-inference conclusion applies to the same viable state trajectory produced by the language contract.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  L.base.encode (L.base.trajectory 0) = L.base.encode L.seed →
    (∀ (n : ℕ),
        L.base.encode (L.base.trajectory (n + 1)) =
          L.base.next (L.base.encode (L.base.trajectory n))
            (L.world_response (L.base.encode (L.base.trajectory n))
              (L.base.policy (L.base.encode (L.base.trajectory n))))) →
      (∀ (n : ℕ),
          CGB.SelfProduction.language_viable L
            (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)) ∧
        ¬CGB.ActiveInference.eventuallyPassive
            (fun n => L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n))
            L.base.passive
```

## CGB.SelfProduction.language_joint_admission_self_production_active_if_aligned

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1667)

源码说明：The active and self-production halves use one trajectory once the two explicit encoded alignment conditions hold. This is the compact bridge for the joint language-life admission: the base's admission contract, the self-producing viable trajectory, and non-eventual passivity are all stated on the same encoded state sequence.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  L.base.encode (L.base.trajectory 0) = L.base.encode L.seed →
    (∀ (n : ℕ),
        L.base.encode (L.base.trajectory (n + 1)) =
          L.base.next (L.base.encode (L.base.trajectory n))
            (L.world_response (L.base.encode (L.base.trajectory n))
              (L.base.policy (L.base.encode (L.base.trajectory n))))) →
      CGB.LanguageLifeAdmission.joint_admission_spec L.base ∧
        (∀ (n : ℕ),
            CGB.SelfProduction.language_viable L
              (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)) ∧
          ¬CGB.ActiveInference.eventuallyPassive
                (fun n => L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n))
                L.base.passive ∧
            ∀ (n : ℕ),
              L.base.encode (L.base.trajectory n) =
                CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n
```

## CGB.SelfProduction.probeLoss_eq_candidate_erase

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1697)

源码说明：Under explicit injective-probe and recoverability conditions, probe loss is exactly the candidate distinction erased by the update.

形式陈述：

```lean
∀ {H : Type u_1} {P : Type u_2} {S : Type u_3} [Fintype H] [inst : DecidableEq H] [inst_1 : DecidableEq P]
  (probe : H → P) (recoverable : S → P → Bool) (before after : S) (old new : Finset H),
  Function.Injective probe →
    (∀ h ∈ old, recoverable before (probe h) = true) →
      (∀ h ∈ old, h ∉ new → recoverable after (probe h) = false) →
        CGB.SelfProduction.probeLoss probe recoverable before after old new = old.card - (old ∩ new).card
```

## CGB.SelfProduction.language_controlled_trajectory_viable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1731)

源码说明：Every externally controlled trajectory remains in the existing language viability region, since the language contract admits every boundary byte.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (actions : ℕ → Act) (n : ℕ),
  CGB.SelfProduction.language_viable L
    (CGB.SelfProduction.controlledTrajectory (CGB.SelfProduction.language_contract L) actions n)
```

## CGB.SelfProduction.language_controlled_memory_eventually_stops

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1749)

源码说明：The candidate-based language contract eventually stops changing its entire memory, for every action sequence. This is a limitation of this particular one-way, finite-write contract, not of finite-state systems.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (actions : ℕ → Act),
  ∃ t,
    ∀ (n : ℕ),
      t ≤ n →
        CGB.SelfProduction.controlledTrajectory (CGB.SelfProduction.language_contract L) actions (n + 1) =
          CGB.SelfProduction.controlledTrajectory (CGB.SelfProduction.language_contract L) actions n
```

## CGB.SelfProduction.language_self_production_memory_eventually_stops

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1813)

源码说明：The autonomous self-production trajectory inherits the same eventual memory freeze; infinite viability does not by itself imply lifelong learning.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  ∃ t,
    ∀ (n : ℕ),
      t ≤ n →
        CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) (n + 1) =
          CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n
```

## CGB.SelfProduction.language_candidate_eventually_constant_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1835)

源码说明：In the original candidate contract, eventual memory freeze also freezes the policy action and body-determined boundary byte. Hence that particular autonomous witness cannot represent an indefinitely changing boundary stream. This does not apply to the separate open-world contract with external state.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  ∃ t,
    ∀ (n : ℕ),
      t ≤ n →
        CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n =
            CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) t ∧
          L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n) =
              L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) t) ∧
            L.world_response (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)
                (L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)) =
              L.world_response (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) t)
                (L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) t))
```

## CGB.SelfProduction.language_controlled_erasure_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1861)

源码说明：Candidate elimination telescopes along every intervention, not merely along the organism's chosen policy.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (actions : ℕ → Act) (T : ℕ),
  ((CGB.SelfProduction.controlledTrajectory (CGB.SelfProduction.language_contract L) actions T).candidates.card +
      ∑ n ∈ Finset.range T,
        have s := CGB.SelfProduction.controlledTrajectory (CGB.SelfProduction.language_contract L) actions n;
        CGB.SelfProduction.language_dissipation L s (L.world_response s (actions n))) =
    (L.base.encode L.seed).candidates.card
```

## CGB.SelfProduction.language_controlled_erasure_not_unbounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1898)

源码说明：Consequently, permanently eliminating a finite candidate family cannot produce the unbounded signal for any policy, even an active one.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (actions : ℕ → Act),
  ¬CGB.ActiveInference.unbounded fun n =>
      have s := CGB.SelfProduction.controlledTrajectory (CGB.SelfProduction.language_contract L) actions n;
      CGB.SelfProduction.language_dissipation L s (L.world_response s (actions n))
```

## CGB.SelfProduction.language_probe_loss_eq_dissipation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1932)

源码说明：Once the bridge's three recoverability conditions hold, its measured probe loss equals the language candidate dissipation for that byte.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} {P : Type u_8} {S : Type u_9} [inst : Fintype Model] [inst_1 : DecidableEq Model]
  [inst_2 : Fintype Byte] [inst_3 : Nonempty Byte] [inst_4 : DecidableEq P]
  (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (B : CGB.SelfProduction.LanguageProbeBridge L P S) (h : Hist) (b : Byte),
  (∀ x ∈ (L.base.encode h).candidates, B.recoverable (B.state_of (L.base.encode h)) (B.probe x) = true) →
    (∀ x ∈ (L.base.encode h).candidates,
        x ∉ (L.base.next (L.base.encode h) b).candidates →
          B.recoverable (B.state_of (L.base.next (L.base.encode h) b)) (B.probe x) = false) →
      CGB.SelfProduction.probeLoss B.probe B.recoverable (B.state_of (L.base.encode h))
          (B.state_of (L.base.next (L.base.encode h) b)) (L.base.encode h).candidates
          (L.base.next (L.base.encode h) b).candidates =
        CGB.SelfProduction.language_dissipation L (L.base.encode h) b
```

## CGB.SelfProduction.candidate_reference_probe_loss_eq_dissipation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1980)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (h : Hist) (b : Byte),
  CGB.SelfProduction.probeLoss (CGB.SelfProduction.candidateReferenceProbeBridge L).probe
      (CGB.SelfProduction.candidateReferenceProbeBridge L).recoverable (L.base.encode h)
      (L.base.next (L.base.encode h) b) (L.base.encode h).candidates (L.base.next (L.base.encode h) b).candidates =
    CGB.SelfProduction.language_dissipation L (L.base.encode h) b
```

## CGB.SelfProduction.language_dissipation_encoded_eq_candidate_erase

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1998)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (h : Hist) (b : Byte),
  CGB.SelfProduction.language_dissipation L (L.base.encode h) b =
    CGB.SelfProduction.candidate_erase_count L.base.possible (L.base.encode h) (L.base.context h b) (L.base.action h b)
      (L.base.outcome h b)
```

## CGB.SelfProduction.language_cumulative_erasure_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2013)

源码说明：Along an encoded history, cumulative candidate erasure is exactly the initial candidate count minus the current count. This exposes the finite elimination budget already implicit in the admission contract.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (h : ℕ → Hist) (b : ℕ → Byte),
  (∀ (n : ℕ), h (n + 1) = L.base.advanceHistory (h n) (b n)) →
    ∀ (T : ℕ),
      (L.base.encode (h T)).candidates.card +
          ∑ n ∈ Finset.range T, CGB.SelfProduction.language_dissipation L (L.base.encode (h n)) (b n) =
        (L.base.encode (h 0)).candidates.card
```

## CGB.SelfProduction.language_cumulative_erasure_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2041)

源码说明：The total number of eliminated distinctions cannot exceed the finite initial candidate set, regardless of the length of the stream.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (h : ℕ → Hist) (b : ℕ → Byte),
  (∀ (n : ℕ), h (n + 1) = L.base.advanceHistory (h n) (b n)) →
    ∀ (T : ℕ),
      ∑ n ∈ Finset.range T, CGB.SelfProduction.language_dissipation L (L.base.encode (h n)) (b n) ≤
        (L.base.encode (h 0)).candidates.card
```

## CGB.SelfProduction.candidate_erasure_not_unbounded_signal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2056)

源码说明：Finite one-way candidate elimination cannot itself supply the unbounded signal assumed by the active-inference contract.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (h : ℕ → Hist) (b : ℕ → Byte),
  (∀ (n : ℕ), h (n + 1) = L.base.advanceHistory (h n) (b n)) →
    ¬CGB.ActiveInference.unbounded fun n => CGB.SelfProduction.language_dissipation L (L.base.encode (h n)) (b n)
```

## CGB.SelfProduction.language_self_production_bridge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2071)

源码说明：The existing four-way language admission and the open self-production contract can be satisfied by the same persistent memory state.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  CGB.LanguageLifeAdmission.joint_admission_spec L.base →
    ∃ x,
      x 0 = L.base.encode L.seed ∧
        (∀ (n : ℕ), x (n + 1) = L.base.next (x n) (L.world_response (x n) (L.base.policy (x n)))) ∧
          ∀ (n : ℕ), CGB.SelfProduction.language_viable L (x n)
```

## CGB.SelfProduction.language_joint_admission_and_self_production

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2086)

源码说明：The named joint-admission proposition and the infinite self-producing trajectory now form one explicit language-life existence statement.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  CGB.LanguageLifeAdmission.joint_admission_spec L.base ∧
    ∃ x,
      x 0 = L.base.encode L.seed ∧
        (∀ (n : ℕ), x (n + 1) = L.base.next (x n) (L.world_response (x n) (L.base.policy (x n)))) ∧
          ∀ (n : ℕ), CGB.SelfProduction.language_viable L (x n)
```

## CGB.SelfProduction.language_self_consequence_model_exists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2103)

源码说明：In the joined language contract, the self-consequence predictor is a single function of the persistent memory, action, and boundary response.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  ∃ predict,
    ∀ (s : CGB.CausalLearning.Memory Model L.base.reserve) (a : Act), predict s a = L.base.next s (L.world_response s a)
```

## CGB.SelfProduction.language_infinite_stream_life_complete

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2145)

源码说明：Joint candidate-contract theorem. The finite state is the compressed carrier; the two fibre conditions make it a causal sufficient statistic; the next conjuncts expose capacity, dissipation, infinite viability, and a mathematical action-consequence function using the supplied world response. It does not establish a learned self-model.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  CGB.SelfProduction.language_infinite_stream_life_spec L
```

## CGB.SelfProduction.language_candidate_predictive_growth_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2179)

源码说明：If the same language candidate's allocation beats the declared reinvestment cost on every encoded history, its predictive stream contract has positive expected net log growth on every such history. This is the smallest formal bridge from language prediction to the declared information economy; it does not identify that growth with physical energy or trajectory level survival.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer)
  (m : ℝ),
  (∀ (h : Hist),
      CGB.PredictiveHarvest.crossEntropy (L.base.stream h) (L.base.allocation (L.base.encode h)) <
        Real.log ↑(Fintype.card Byte) - m) →
    ∀ (h : Hist), 0 < CGB.PredictiveHarvest.growth (L.base.stream h) (L.base.allocation (L.base.encode h)) - m
```

## CGB.SelfProduction.language_conditional_joint_existence

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2202)

源码说明：Conditional joint existence theorem for the current language candidate. Under the two encoded trajectory-alignment conditions, a cross-entropy threshold, and a block-level learning minorization recurrence, one and the same candidate has joint admission, an infinite viable self-production trajectory, non-eventual passivity, positive expected predictive growth, and the stated finite-block learning-failure bound. Every condition is explicit; none is silently promoted to an axiom.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  L.base.encode (L.base.trajectory 0) = L.base.encode L.seed →
    (∀ (n : ℕ),
        L.base.encode (L.base.trajectory (n + 1)) =
          L.base.next (L.base.encode (L.base.trajectory n))
            (L.world_response (L.base.encode (L.base.trajectory n))
              (L.base.policy (L.base.encode (L.base.trajectory n))))) →
      ∀ (m : ℝ),
        (∀ (h : Hist),
            CGB.PredictiveHarvest.crossEntropy (L.base.stream h) (L.base.allocation (L.base.encode h)) <
              Real.log ↑(Fintype.card Byte) - m) →
          ∀ p ≤ 1,
            ∀ (q : ℕ → ℝ),
              q 0 ≤ 1 →
                (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) →
                  CGB.LanguageLifeAdmission.joint_admission_spec L.base ∧
                    (∀ (n : ℕ),
                        CGB.SelfProduction.language_viable L
                          (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)) ∧
                      ¬CGB.ActiveInference.eventuallyPassive
                            (fun n =>
                              L.base.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n))
                            L.base.passive ∧
                        (∀ (h : Hist),
                            0 <
                              CGB.PredictiveHarvest.growth (L.base.stream h) (L.base.allocation (L.base.encode h)) -
                                m) ∧
                          ∀ (n : ℕ), q n ≤ (1 - p) ^ n
```

## CGB.SelfProduction.minimal_generator_conditional_joint_existence

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2242)

源码说明：Minimal-generator refinement of the conditional joint existence theorem. The generator supplies the finite causal carrier; the only remaining conditions are the same two encoded alignment equations, predictive payment threshold and learning minorization.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte]
  (G : CGB.SelfProduction.MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer),
  (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
        ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory 0) =
      (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
        (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).seed →
    (∀ (n : ℕ),
        (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory (n + 1)) =
          (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.next
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).world_response
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.policy
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))))) →
      ∀ (m : ℝ),
        (∀ (h : Hist),
            CGB.PredictiveHarvest.crossEntropy
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.stream h)
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.allocation
                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode h)) <
              Real.log ↑(Fintype.card Byte) - m) →
          ∀ p ≤ 1,
            ∀ (q : ℕ → ℝ),
              q 0 ≤ 1 →
                (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) →
                  CGB.LanguageLifeAdmission.joint_admission_spec
                      (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base ∧
                    (∀ (n : ℕ),
                        CGB.SelfProduction.language_viable
                          (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G)
                          (CGB.SelfProduction.trajectory
                            (CGB.SelfProduction.language_contract
                              (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G))
                            n)) ∧
                      ¬CGB.ActiveInference.eventuallyPassive
                            (fun n =>
                              (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.policy
                                (CGB.SelfProduction.trajectory
                                  (CGB.SelfProduction.language_contract
                                    (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G))
                                  n))
                            (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.passive ∧
                        (∀ (h : Hist),
                            0 <
                              CGB.PredictiveHarvest.growth
                                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.stream h)
                                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.allocation
                                    ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode h)) -
                                m) ∧
                          ∀ (n : ℕ), q n ≤ (1 - p) ^ n
```

## CGB.SelfProduction.minimal_generator_infinite_stream_life_complete

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2288)

源码说明：The minimal generator is sufficient: `read` and `next` are materialized by history factorization, after which the complete joint theorem applies unchanged.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte]
  (G : CGB.SelfProduction.MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer),
  CGB.SelfProduction.language_infinite_stream_life_spec
    (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G)
```

## CGB.SelfProduction.exists_language_infinite_stream_life_of_minimal_generator

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2301)

源码说明：Given an admissible minimal generator, there is a candidate satisfying the candidate specification. This assumes the generator and does not establish existence of an organism satisfying all original life requirements.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte]
  (G : CGB.SelfProduction.MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer),
  ∃ L, CGB.SelfProduction.language_infinite_stream_life_spec L
```

## CGB.SelfProduction.renewable_language_carrier_correct

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2345)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer),
  (∀ (h : Hist), CGB.SelfProduction.renewableLanguageRead G (G.encode h) = G.required h) ∧
    ∀ (h : Hist) (b : Byte), CGB.SelfProduction.renewableLanguageNext G (G.encode h) b = G.encode (G.advanceHistory h b)
```

## CGB.SelfProduction.renewable_language_carrier_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2375)

源码说明：The same state carrier has its required readout, byte transition, and an unbounded viable trajectory. Continued adaptation, useful prediction and active action remain separate world/transition conditions.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer),
  (∀ (h : Hist), CGB.SelfProduction.renewableLanguageRead G (G.encode h) = G.required h) ∧
    (∀ (h : Hist) (b : Byte),
        CGB.SelfProduction.renewableLanguageNext G (G.encode h) b = G.encode (G.advanceHistory h b)) ∧
      ∀ (n : ℕ),
        (CGB.SelfProduction.renewableLanguageContract G).viable
          (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G) n)
```

## CGB.SelfProduction.precision_binary_update_is_predictive_improvement

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2409)

源码说明：The precision-derived gain used by the active GDN path is a valid binary predictive update whenever incoming precision is positive. This is the smallest analytic bridge from natural-parameter assimilation to the renewable life gate: no fixed learning-rate parameter is introduced.

形式陈述：

```lean
∀ (r : ℝ) (y : Bool) (p w : ℝ),
  0 < r →
    r < 1 →
      0 < p →
        0 < w →
          have β := w / (p + w);
          have r' := r + β * ((if y = true then 1 else 0) - r);
          (0 ≤ r' ∧ r' ≤ 1 ∧ r' ≠ r) ∧
            CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y) (CGB.Generation.binaryAllocation r') <
              CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y) (CGB.Generation.binaryAllocation r)
```

## CGB.SelfProduction.precision_binary_sample_gain_not_future_guarantee

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2429)

源码说明：Fitting an arrived bit does not by itself improve the underlying stream: if the old binary predictor already equals the stream law, its expected log loss cannot improve after the precision-derived update. The same update still improves loss on the arrived bit. This compares both predictors on one law.

形式陈述：

```lean
∀ (r : ℝ) (y : Bool) (p w : ℝ),
  0 < r →
    r < 1 →
      0 < p →
        0 < w →
          have β := w / (p + w);
          have r' := r + β * ((if y = true then 1 else 0) - r);
          CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y) (CGB.Generation.binaryAllocation r') <
              CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryPointLaw y) (CGB.Generation.binaryAllocation r) ∧
            CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryAllocation r) (CGB.Generation.binaryAllocation r) ≤
              CGB.PredictiveHarvest.crossEntropy (CGB.Generation.binaryAllocation r)
                (CGB.Generation.binaryAllocation r')
```

## CGB.SelfProduction.categorical_precision_next_loss_direction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2464)

源码说明：Categorical precision-weighted assimilation improves the next symbol's log loss exactly when that symbol repeats the observed symbol; a different symbol becomes strictly less likely. This is a one-step realised outcome, not a claim about expected loss for a general language distribution.

形式陈述：

```lean
∀ {Byte : Type u_1} [inst : DecidableEq Byte] (c : Byte → ℝ) (observed next : Byte) (p w : ℝ),
  0 < p →
    0 < w →
      0 < c next →
        c next < 1 →
          have q := c next + w / (p + w) * ((if next = observed then 1 else 0) - c next);
          0 < q ∧
            (-Real.log q < -Real.log (c next) ↔ next = observed) ∧ (-Real.log (c next) < -Real.log q ↔ next ≠ observed)
```

## CGB.SelfProduction.renewable_language_life_gate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2521)

源码说明：The renewable carrier, plus explicit predictive payment and recurrent learning conditions, yields one general-state language life gate. Recurrent learning is stated as a change of the same state under the same world response, so the result rules out eventual freezing without adding a clock or a second reserve.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} [inst : Fintype Byte]
  [inst_1 : Nonempty Byte] (G : CGB.SelfProduction.RenewableLanguageLifeGenerator Hist State Byte Act Answer) (m : ℝ),
  (∀ (s : State),
      CGB.PredictiveHarvest.crossEntropy (G.stream s) (G.allocation s) < Real.log ↑(Fintype.card Byte) - m) →
    (∀ (h : Hist), CGB.SelfProduction.renewableLanguageRead G.carrier (G.carrier.encode h) = G.carrier.required h) ∧
      (∀ (h : Hist) (b : Byte),
          CGB.SelfProduction.renewableLanguageNext G.carrier (G.carrier.encode h) b =
            G.carrier.encode (G.carrier.advanceHistory h b)) ∧
        (∀ (n : ℕ),
            (CGB.SelfProduction.renewableLanguageContract G.carrier).viable
              (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G.carrier) n)) ∧
          ¬CGB.ActiveInference.eventuallyPassive
                (fun n =>
                  G.carrier.policy
                    (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G.carrier) n))
                G.passive ∧
            (¬∀ (a : Act) (b : Byte), G.predict b = G.response a b) ∧
              (∀ (s : State), 0 < CGB.PredictiveHarvest.growth (G.stream s) (G.allocation s) - m) ∧
                (∀ (N : ℕ),
                    ∃ n,
                      N ≤ n ∧
                        CGB.SelfProduction.renewablePredictiveUpdateEvent G.carrier G.stream G.allocation
                          (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G.carrier) n)
                          (G.carrier.world_response
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G.carrier) n)
                            (G.carrier.policy
                              (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G.carrier)
                                n)))) ∧
                  ¬∃ N,
                      ∀ (n : ℕ),
                        N ≤ n →
                          CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G.carrier)
                              (n + 1) =
                            CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableLanguageContract G.carrier) n
```

## CGB.SelfProduction.exists_renewable_language_life_of_generator

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2592)

源码说明：Existential form of the renewable gate. The generated trajectory is the same one used by the carrier, active signal, and recurrent-learning clauses; it is not a separately chosen witness.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} [inst : Fintype Byte]
  [inst_1 : Nonempty Byte] (G : CGB.SelfProduction.RenewableLanguageLifeGenerator Hist State Byte Act Answer) (m : ℝ),
  (∀ (s : State),
      CGB.PredictiveHarvest.crossEntropy (G.stream s) (G.allocation s) < Real.log ↑(Fintype.card Byte) - m) →
    ∃ x,
      x 0 = G.carrier.encode G.carrier.seed ∧
        (∀ (n : ℕ),
            x (n + 1) =
              CGB.SelfProduction.renewableLanguageNext G.carrier (x n)
                (G.carrier.world_response (x n) (G.carrier.policy (x n)))) ∧
          (∀ (n : ℕ), (CGB.SelfProduction.renewableLanguageContract G.carrier).viable (x n)) ∧
            ¬CGB.ActiveInference.eventuallyPassive (fun n => G.carrier.policy (x n)) G.passive ∧
              (¬∀ (a : Act) (b : Byte), G.predict b = G.response a b) ∧
                (∀ (s : State), 0 < CGB.PredictiveHarvest.growth (G.stream s) (G.allocation s) - m) ∧
                  (∀ (N : ℕ),
                      ∃ n,
                        N ≤ n ∧
                          CGB.SelfProduction.renewablePredictiveUpdateEvent G.carrier G.stream G.allocation (x n)
                            (G.carrier.world_response (x n) (G.carrier.policy (x n)))) ∧
                    ¬∃ N, ∀ (n : ℕ), N ≤ n → x (n + 1) = x n
```

## CGB.SelfProduction.renewable_open_world_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2675)

源码说明：One open-world trajectory remains viable and realizes the readout for the history of its own delivered inputs. The external world and proof-side history need not be finite or stored inside the organism.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World),
  ∃ x,
    x 0 = (G.encode G.seed, W.initial) ∧
      (∀ (n : ℕ),
          x (n + 1) =
            (CGB.SelfProduction.renewableOpenContract G W).step (x n)
              ((CGB.SelfProduction.renewableOpenContract G W).response (x n)
                ((CGB.SelfProduction.renewableOpenContract G W).action (x n)))) ∧
        (∀ (n : ℕ), (CGB.SelfProduction.renewableOpenContract G W).viable (x n)) ∧
          ∃ h,
            h 0 = G.seed ∧
              (∀ (n : ℕ), h (n + 1) = G.advanceHistory (h n) (W.boundary (x n).2 (x n).1 (G.policy (x n).1))) ∧
                ∀ (n : ℕ),
                  G.encode (h n) = (x n).1 ∧ CGB.SelfProduction.renewableLanguageRead G (x n).1 = G.required (h n)
```

## CGB.SelfProduction.renewable_open_language_joint_gate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2715)

源码说明：Conditional joint gate on the *open* body-world trajectory. The actual boundary selects each learning event; the world may have its own unbounded state. Signal, passive limitation, predictive payment and recurrent useful updates are assumptions on this trajectory, not consequences of a local score step. The world-closure condition is the existing strong `W.closed`.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  [inst : Fintype Byte] [inst_1 : Nonempty Byte]
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World) (stream allocation : State → Byte → ℝ),
  (∀ (s : State) (b : Byte), 0 ≤ stream s b) →
    (∀ (s : State), ∑ b, stream s b = 1) →
      (∀ (s : State) (b : Byte), 0 < allocation s b) →
        (∀ (s : State), ∑ b, allocation s b = 1) →
          ∀ (m : ℝ),
            (∀ (n : ℕ),
                CGB.PredictiveHarvest.crossEntropy
                    (stream (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1)
                    (allocation (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1) <
                  Real.log ↑(Fintype.card Byte) - m) →
              ∀ (signal : ℕ → ℕ) (passive : Act),
                CGB.ActiveInference.unbounded signal →
                  CGB.ActiveInference.passiveBounded
                      (fun n =>
                        G.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1)
                      signal passive →
                    (∃ n a a',
                        W.boundary (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).2
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 a ≠
                          W.boundary (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).2
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 a') →
                      (∀ (N : ℕ),
                          ∃ n,
                            N ≤ n ∧
                              CGB.SelfProduction.renewablePredictiveUpdateEvent G stream allocation
                                (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1
                                (W.boundary
                                  (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).2
                                  (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1
                                  (G.policy
                                    (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W)
                                        n).1))) →
                        ∃ x,
                          x 0 = (G.encode G.seed, W.initial) ∧
                            (∀ (n : ℕ),
                                x (n + 1) =
                                  (CGB.SelfProduction.renewableOpenContract G W).step (x n)
                                    ((CGB.SelfProduction.renewableOpenContract G W).response (x n)
                                      ((CGB.SelfProduction.renewableOpenContract G W).action (x n)))) ∧
                              (∀ (n : ℕ), (CGB.SelfProduction.renewableOpenContract G W).viable (x n)) ∧
                                (∃ h,
                                    h 0 = G.seed ∧
                                      (∀ (n : ℕ),
                                          h (n + 1) =
                                            G.advanceHistory (h n) (W.boundary (x n).2 (x n).1 (G.policy (x n).1))) ∧
                                        ∀ (n : ℕ),
                                          G.encode (h n) = (x n).1 ∧
                                            CGB.SelfProduction.renewableLanguageRead G (x n).1 = G.required (h n)) ∧
                                  ¬CGB.ActiveInference.eventuallyPassive (fun n => G.policy (x n).1) passive ∧
                                    (¬∃ predict,
                                          ∀ (n : ℕ) (a : Act), predict (x n).2 (x n).1 = W.boundary (x n).2 (x n).1 a) ∧
                                      (∀ (n : ℕ),
                                          0 < CGB.PredictiveHarvest.growth (stream (x n).1) (allocation (x n).1) - m) ∧
                                        (∀ (N : ℕ),
                                            ∃ n,
                                              N ≤ n ∧
                                                CGB.SelfProduction.renewablePredictiveUpdateEvent G stream allocation
                                                  (x n).1 (W.boundary (x n).2 (x n).1 (G.policy (x n).1))) ∧
                                          ¬∃ N, ∀ (n : ℕ), N ≤ n → (x (n + 1)).1 = (x n).1
```

## CGB.SelfProduction.open_world_categorical_next_loss_direction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2794)

源码说明：The categorical innovation condition applied to the actual boundary and the next actual boundary on one viable open-world trajectory. The next symbol is used only to assess the result; it is not an input to the earlier update.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  [inst : DecidableEq Byte] (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World) (allocation : State → Byte → ℝ) (n : ℕ) (p w : ℝ),
  0 < p →
    0 < w →
      (∀ (b : Byte),
          0 < allocation (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 b ∧
            allocation (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 b < 1) →
        (∀ (b : Byte),
            allocation (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1 b =
              allocation (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 b +
                w / (p + w) *
                  ((if
                        b =
                          W.boundary (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).2
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1
                            (G.policy
                              (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1) then
                      1
                    else 0) -
                    allocation (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 b)) →
          have C := CGB.SelfProduction.renewableOpenContract G W;
          have s := (CGB.SelfProduction.trajectory C n).1;
          have s' := (CGB.SelfProduction.trajectory C (n + 1)).1;
          have observed := W.boundary (CGB.SelfProduction.trajectory C n).2 s (G.policy s);
          have next := W.boundary (CGB.SelfProduction.trajectory C (n + 1)).2 s' (G.policy s');
          C.viable (CGB.SelfProduction.trajectory C (n + 1)) ∧
            (-Real.log (allocation s' next) < -Real.log (allocation s next) ↔ next = observed) ∧
              (-Real.log (allocation s next) < -Real.log (allocation s' next) ↔ next ≠ observed)
```

## CGB.SelfProduction.open_world_precision_update_improves_next_prediction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2832)

源码说明：An actual open-world update improves the next prediction when the next delivered bit repeats the current one and the same body's probability obeys the existing precision-derived innovation update. Both scores target the actual next bit; the condition is not a claim that arbitrary streams repeat.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Act : Type u_3} {Answer : Type u_4} {World : Type u_5}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Bool Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World) (prob : State → ℝ) (n : ℕ) (p w : ℝ),
  0 < p →
    0 < w →
      0 < prob (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 →
        prob (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 < 1 →
          prob (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1 =
              prob (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 +
                w / (p + w) *
                  ((if
                        W.boundary (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).2
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1
                            (G.policy
                              (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1) =
                          true then
                      1
                    else 0) -
                    prob (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1) →
            W.boundary (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).2
                  (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1
                  (G.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1) =
                W.boundary (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).2
                  (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1
                  (G.policy (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1) →
              (CGB.SelfProduction.renewableOpenContract G W).viable
                  (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)) ∧
                (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1 ≠
                    (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1 ∧
                  CGB.PredictiveHarvest.crossEntropy
                      (CGB.Generation.binaryPointLaw
                        (W.boundary
                          (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).2
                          (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1
                          (G.policy
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1)))
                      (CGB.Generation.binaryAllocation
                        (prob
                          (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1)) <
                    CGB.PredictiveHarvest.crossEntropy
                      (CGB.Generation.binaryPointLaw
                        (W.boundary
                          (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).2
                          (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1
                          (G.policy
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) (n + 1)).1)))
                      (CGB.Generation.binaryAllocation
                        (prob (CGB.SelfProduction.trajectory (CGB.SelfProduction.renewableOpenContract G W) n).1))
```

## CGB.SelfProduction.masked_open_world_information_limit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2902)

源码说明：Under the same fixed delivered symbol, identical starting bodies have identical body states and actions for any two evolving external worlds. If those worlds instead require opposite equally likely next-bit predictions, the common body-only prediction has at least one bit of average log loss. The separation of future targets is an explicit world condition.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World) (quiet : Byte) (s : State) (w₀ w₁ : World)
  (target : State × World → Bool) (predict : State → ℝ) (n : ℕ),
  target (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)) = false →
    target (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)) = true →
      0 < predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1 →
        predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1 < 1 →
          (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1 =
              (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)).1 ∧
            G.policy (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1 =
                G.policy (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)).1 ∧
              target (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)) = false ∧
                target (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)) = true ∧
                  -(Real.log (1 - predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1) +
                          Real.log (predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)).1)) /
                      2 ≥
                    Real.log 2
```

## CGB.SelfProduction.open_world_closure_excludes_conflicting_safe_actions

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2934)

源码说明：The present `RenewableOpenWorld.closed` quantifies over every hidden world state at a viable encoded body. It therefore excludes a pair of worlds that require two different actions for even one-step body viability. This pinpoints a limitation of that contract, not an impossibility of life in such worlds.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World) (h : Hist) (w₀ w₁ : World) (a₀ a₁ : Act),
  G.viable (G.encode h) →
    a₀ ≠ a₁ →
      (∀ (a : Act), G.viable (G.encode (G.advanceHistory h (W.boundary w₀ (G.encode h) a))) → a = a₀) →
        (∀ (a : Act), G.viable (G.encode (G.advanceHistory h (W.boundary w₁ (G.encode h) a))) → a = a₁) → False
```

## CGB.SelfProduction.indistinguishable_joint_worlds_cannot_both_stay_viable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2951)

源码说明：For a joint body-world viability predicate, two worlds hidden behind the same body can require conflicting safe actions. In that case the same body-only policy cannot keep both successor pairs viable. The predicate is an assessment of the existing joint state, not an additional body account.

形式陈述：

```lean
∀ {State : Type u_1} {World : Type u_2} {Act : Type u_3} {Byte : Type u_4} (boundary : World → State → Act → Byte)
  (bodyStep : State → Byte → State) (worldStep : World → State → Act → Byte → World) (viable : State × World → Prop)
  (policy : State → Act) (s : State) (w₀ w₁ : World) (a₀ a₁ : Act),
  a₀ ≠ a₁ →
    (∀ (a : Act), viable (bodyStep s (boundary w₀ s a), worldStep w₀ s a (boundary w₀ s a)) → a = a₀) →
      (∀ (a : Act), viable (bodyStep s (boundary w₁ s a), worldStep w₁ s a (boundary w₁ s a)) → a = a₁) →
        ¬(viable (bodyStep s (boundary w₀ s (policy s)), worldStep w₀ s (policy s) (boundary w₀ s (policy s))) ∧
            viable (bodyStep s (boundary w₁ s (policy s)), worldStep w₁ s (policy s) (boundary w₁ s (policy s))))
```

## CGB.SelfProduction.joint_viability_policy_generation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2976)

源码说明：A body-only policy preserving a joint viable region exists exactly when, for each body state, there is one action safe for every compatible world in that region. Given a viable initial pair, the selected policy generates one joint trajectory that stays viable at every finite tick. The predicate and one-step safe actions are conditions to verify, not a proof that a language world or a complete living organism supplies them. Classical choice here is an existence proof, not a claimed executable or efficient policy.

形式陈述：

```lean
∀ {State : Type u_1} {World : Type u_2} {Act : Type u_3} (step : State → World → Act → State × World)
  (viable : State × World → Prop) (seed : State × World),
  viable seed →
    ((∀ (s : State), ∃ a, ∀ (w : World), viable (s, w) → viable (step s w a)) ↔
      ∃ policy,
        (∀ (s : State) (w : World), viable (s, w) → viable (step s w (policy s))) ∧
          ∃ run,
            run 0 = seed ∧
              (∀ (n : ℕ), run (n + 1) = step (run n).1 (run n).2 (policy (run n).1)) ∧ ∀ (n : ℕ), viable (run n))
```

## _private.CGBAxioms.GenerationSelfProduction.0.CGB.SelfProduction.closed_region_law_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3007)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype E]
  (Q : CGB.Interaction.Machine S ℕ E) (region : S → Prop) (seed : S),
  region seed →
    (∀ (t : ℕ) (x : S) (e : E) (y : S), region x → 0 < Q.step t x e y → region y) →
      ∀ (n : ℕ) (z : S), ¬region z → CGB.Energetics.law Q seed n z = 0
```

## CGB.SelfProduction.stochastic_joint_viability_policy_generation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3039)

源码说明：Stochastic version: a body-only policy keeps every positive-probability successor in the joint viable region iff each body has an action safe for all compatible viable worlds and all supported events. The generated closed-loop law has zero mass outside the region at every finite tick. This is not an infinite-path measure or a proof that the safe-action premise is populated by a real language world.

形式陈述：

```lean
∀ {State : Type u_1} {World : Type u_2} {Act : Type u_3} {Event : Type u_4} [inst : Fintype State]
  [inst_1 : DecidableEq State] [inst_2 : Fintype World] [inst_3 : DecidableEq World] [inst_4 : Fintype Event]
  (P : CGB.Interaction.Machine (State × World) Act Event) (viable : State × World → Prop) (seed : State × World),
  viable seed →
    ((∀ (s : State),
        ∃ a, ∀ (w : World) (e : Event) (z' : State × World), viable (s, w) → 0 < P.step a (s, w) e z' → viable z') ↔
      ∃ policy,
        (∀ (s : State) (w : World) (e : Event) (z' : State × World),
            viable (s, w) → 0 < P.step (policy s) (s, w) e z' → viable z') ∧
          ∃ process,
            (∀ (t : ℕ) (z : State × World) (e : Event) (z' : State × World),
                process.step t z e z' = P.step (policy z.1) z e z') ∧
              ∀ (n : ℕ) (z : State × World), ¬viable z → CGB.Energetics.law process seed n z = 0)
```

## CGB.SelfProduction.stochastic_joint_recovery_policy_generation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3072)

源码说明：Generate one stochastic policy whose same closed loop both stays within the joint region and reaches the declared event within the finite horizon. The decreasing rank is a proof certificate, not stored body state. The mark must be connected separately to an actual useful event in the world.

形式陈述：

```lean
∀ {State : Type u_1} {World : Type u_2} {Act : Type u_3} {Event : Type u_4} [inst : Fintype State]
  [inst_1 : DecidableEq State] [inst_2 : Fintype World] [inst_3 : DecidableEq World] [inst_4 : Fintype Event]
  (P : CGB.Interaction.Machine (State × World) Act Event) (viable : State × World → Prop) (seed : State × World),
  viable seed →
    ∀ (rank : State × World → ℕ) (horizon : ℕ) (mark : ℕ → State × World → Event → State × World → Bool),
      (∀ (z : State × World), viable z → rank z < horizon) →
        (∀ (s : State),
            ∃ a,
              ∀ (t : ℕ) (w : World) (e : Event) (z' : State × World),
                viable (s, w) →
                  0 < P.step a (s, w) e z' → viable z' ∧ (mark t (s, w) e z' = false → rank z' < rank (s, w))) →
          ∃ policy process,
            (∀ (t : ℕ) (z : State × World) (e : Event) (z' : State × World),
                process.step t z e z' = P.step (policy z.1) z e z') ∧
              (∀ (n : ℕ) (z : State × World), ¬viable z → CGB.Energetics.law process seed n z = 0) ∧
                ∀ (t : ℕ) (z : State × World),
                  viable z → CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events process) mark t horizon z = 1
```

## CGB.SelfProduction.ranked_supported_path_recurrent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3115)

源码说明：The existing ranked recovery certificate forces a marked event in every `horizon`-tick window along every infinite sequence of supported transitions. The sequence is a mathematical path, not an asserted infinite-path measure; the mark can denote any actually declared event.

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E)
  (mark : ℕ → S → E → S → Bool) (horizon : ℕ) (C : CGB.RecoveryCriteria.Ranked K mark horizon) (path : ℕ → S)
  (event : ℕ → E),
  C.region (path 0) →
    (∀ (t : ℕ), 0 < (K t).step (path t) (event t) (path (t + 1))) →
      ∀ (start : ℕ), ∃ t, start ≤ t ∧ t < start + horizon ∧ mark t (path t) (event t) (path (t + 1)) = true
```

## CGB.SelfProduction.ranked_exists_recurrently_marked_path

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3162)

源码说明：A normalized finite event kernel has an infinite supported path from any initial state; a ranked certificate makes its marked events recurrent. This constructs a path, not a probability measure assigning it positive mass.

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : Fintype E] (K : ℕ → CGB.EventChannels.Kernel S E)
  (mark : ℕ → S → E → S → Bool) (horizon : ℕ) (C : CGB.RecoveryCriteria.Ranked K mark horizon) (seed : S),
  C.region seed →
    ∃ path event,
      path 0 = seed ∧
        (∀ (t : ℕ), 0 < (K t).step (path t) (event t) (path (t + 1))) ∧
          ∀ (start : ℕ), ∃ t, start ≤ t ∧ t < start + horizon ∧ mark t (path t) (event t) (path (t + 1)) = true
```

## CGB.SelfProduction.masked_world_prediction_and_viability_limit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3201)

源码说明：On the same masked open-world runs, combine the prediction loss with a world-dependent viability requirement. This does not assert that real language worlds meet the two opposite-target or safe-action conditions.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World) (quiet : Byte) (s : State) (w₀ w₁ : World)
  (target : State × World → Bool) (predict : State → ℝ) (n : ℕ) (viable : State × World → Prop) (a₀ a₁ : Act),
  a₀ ≠ a₁ →
    target (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)) = false →
      target (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)) = true →
        0 < predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1 →
          predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1 < 1 →
            viable (CGB.SelfProduction.openActionStep G W (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)) a₀) →
              viable (CGB.SelfProduction.openActionStep G W (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)) a₁) →
                (∀ (a : Act),
                    viable
                        (CGB.SelfProduction.openActionStep G W (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀))
                          a) →
                      a = a₀) →
                  (∀ (a : Act),
                      viable
                          (CGB.SelfProduction.openActionStep G W (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁))
                            a) →
                        a = a₁) →
                    viable
                        (CGB.SelfProduction.openActionStep G W (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀))
                          a₀) ∧
                      viable
                          (CGB.SelfProduction.openActionStep G W (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁))
                            a₁) ∧
                        ¬(viable
                                (CGB.SelfProduction.openActionStep G W
                                  (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀))
                                  (G.policy (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1)) ∧
                              viable
                                (CGB.SelfProduction.openActionStep G W
                                  (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁))
                                  (G.policy (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)).1))) ∧
                          -(Real.log (1 - predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₀)).1) +
                                  Real.log (predict (CGB.SelfProduction.maskedOpenRun G W quiet n (s, w₁)).1)) /
                              2 ≥
                            Real.log 2
```

## CGB.SelfProduction.open_world_input_not_body_determined

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3234)

源码说明：If hidden world states can give different boundary inputs at the same organism state and action, no state-only predictor is correct for both.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World) (s : State) (a : Act) (w₁ w₂ : World),
  W.boundary w₁ s a ≠ W.boundary w₂ s a →
    ∀ (predict : State → Act → Byte), ¬(predict s a = W.boundary w₁ s a ∧ predict s a = W.boundary w₂ s a)
```

## CGB.SelfProduction.open_world_body_prediction_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3247)

源码说明：Exact body-only prediction of the next body state for every world state is possible precisely when hidden world variation has no effect on that next state. This is an expressibility result, not a learned self-model.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World),
  (∃ predict,
      ∀ (s : State) (a : Act) (w : World),
        predict s a = CGB.SelfProduction.renewableLanguageNext G s (W.boundary w s a)) ↔
    ∀ (s : State) (a : Act) (w₁ w₂ : World),
      CGB.SelfProduction.renewableLanguageNext G s (W.boundary w₁ s a) =
        CGB.SelfProduction.renewableLanguageNext G s (W.boundary w₂ s a)
```

## CGB.SelfProduction.reachable_consequence_readout_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3268)

源码说明：Restrict exact consequence prediction to joint states reachable from the initial condition under action interventions. A body-state readout exists iff histories merged into the same body state have identical next-body consequences for every proposed action. This is representability, not identification from observations or proof that the predictor is learned.

形式陈述：

```lean
∀ {Hist : Type u_1} {State : Type u_2} {Byte : Type u_3} {Act : Type u_4} {Answer : Type u_5} {World : Type u_6}
  (G : CGB.SelfProduction.RenewableLanguageGenerator Hist State Byte Act Answer)
  (W : CGB.SelfProduction.RenewableOpenWorld G World),
  have c := CGB.SelfProduction.renewableOpenContract G W;
  have joint := fun p => CGB.SelfProduction.controlledTrajectory c p.1 p.2;
  have body := fun p => (joint p).1;
  have consequence := fun p a =>
    CGB.SelfProduction.renewableLanguageNext G (body p) (W.boundary (joint p).2 (body p) a);
  (∃ predict, ∀ (p : (ℕ → Act) × ℕ) (a : Act), predict (body p) a = consequence p a) ↔
    ∀ (p q : (ℕ → Act) × ℕ), body p = body q → ∀ (a : Act), consequence p a = consequence q a
```

## CGB.SelfProduction.renewable_open_world_is_exogenous

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3364)

源码说明：In the existing open-stream witness, every action intervention leaves the external cursor on exactly the same trajectory. Actions affect the boundary response but not the evolution of this world's own state.

形式陈述：

```lean
∀ (bits : ℕ → Bool) (actions : ℕ → CGB.RenewableLearning.Action) (n : ℕ),
  (CGB.SelfProduction.controlledTrajectory
        (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
          (CGB.SelfProduction.existingRenewableOpenWorld bits))
        actions n).2 =
    n
```

## CGB.SelfProduction.existing_renewable_open_stream_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3380)

源码说明：The external cursor advances each tick while the same finite organism stays in its stated non-vacuum region, for any external bit sequence.

形式陈述：

```lean
∀ (bits : ℕ → Bool),
  ∃ x,
    x 0 = (CGB.RenewableLearning.Cell.search, 0) ∧
      (∀ (n : ℕ),
          x (n + 1) =
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
                  (CGB.SelfProduction.existingRenewableOpenWorld bits)).step
              (x n)
              ((CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
                    (CGB.SelfProduction.existingRenewableOpenWorld bits)).response
                (x n)
                ((CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
                      (CGB.SelfProduction.existingRenewableOpenWorld bits)).action
                  (x n)))) ∧
        (∀ (n : ℕ), (x n).1 ≠ CGB.RenewableLearning.Cell.vacuum) ∧ ∀ (n : ℕ), (x n).2 = n
```

## CGB.SelfProduction.renewable_open_first_input_recorded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3405)

源码说明：On the initial probe, the existing finite carrier records which external answer arrived. The value is supplied by the world, not the body state.

形式陈述：

```lean
∀ (bits : ℕ → Bool),
  (CGB.SelfProduction.trajectory
        (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
          (CGB.SelfProduction.existingRenewableOpenWorld bits))
        1).1 =
    if bits 0 = true then CGB.RenewableLearning.Cell.aSpent else CGB.RenewableLearning.Cell.bSpent
```

## CGB.SelfProduction.renewable_open_distinct_inputs_reach_distinct_states

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3422)

源码说明：Two worlds differing at the first observed bit start with the same body but lead to distinguishable body states after one boundary input.

形式陈述：

```lean
∀ (bits₁ bits₂ : ℕ → Bool),
  bits₁ 0 ≠ bits₂ 0 →
    (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits₁))
            0).1 =
        (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits₂))
            0).1 ∧
      (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits₁))
            1).1 ≠
        (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits₂))
            1).1
```

## CGB.SelfProduction.renewable_open_first_action_affects_body

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3436)

源码说明：In this witness world, replacing the initial probe by the passive action changes the next body state: the passive step retains `search`, while the actual probe records the external answer.

形式陈述：

```lean
∀ (bits : ℕ → Bool),
  (CGB.SelfProduction.controlledTrajectory
          (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
            (CGB.SelfProduction.existingRenewableOpenWorld bits))
          (fun x => 0) 1).1 =
      CGB.RenewableLearning.Cell.search ∧
    (CGB.SelfProduction.trajectory
          (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
            (CGB.SelfProduction.existingRenewableOpenWorld bits))
          1).1 ≠
      CGB.RenewableLearning.Cell.search
```

## CGB.SelfProduction.renewable_open_always_passive_stays_search

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3454)

源码说明：In the open-stream witness an always-passive intervention stays in the non-vacuum `search` state forever. The witness therefore does not establish that activity is necessary for its non-vacuum viability predicate.

形式陈述：

```lean
∀ (bits : ℕ → Bool) (n : ℕ),
  CGB.SelfProduction.controlledTrajectory
      (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
        (CGB.SelfProduction.existingRenewableOpenWorld bits))
      (fun x => 0) n =
    (CGB.RenewableLearning.Cell.search, n)
```

## CGB.SelfProduction.renewable_open_stream_recurrently_writes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3474)

源码说明：The open witness returns to the probing state every three ticks for any external bit stream, and each probe changes the finite body state.

形式陈述：

```lean
∀ (bits : ℕ → Bool) (k : ℕ),
  (CGB.SelfProduction.trajectory
          (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
            (CGB.SelfProduction.existingRenewableOpenWorld bits))
          (3 * k)).1 =
      CGB.RenewableLearning.Cell.search ∧
    (CGB.SelfProduction.trajectory
          (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
            (CGB.SelfProduction.existingRenewableOpenWorld bits))
          (3 * k + 1)).1 ≠
      CGB.RenewableLearning.Cell.search
```

## CGB.SelfProduction.renewable_open_probe_records_then_reopens

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3537)

源码说明：Each probe stores the current external bit for two successive body states, then reopens the body on the third tick. This is bounded retention, not a claim of lasting or useful language memory.

形式陈述：

```lean
∀ (bits : ℕ → Bool) (k : ℕ),
  ((CGB.SelfProduction.trajectory
          (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
            (CGB.SelfProduction.existingRenewableOpenWorld bits))
          (3 * k + 1)).1 =
      if bits (3 * k) = true then CGB.RenewableLearning.Cell.aSpent else CGB.RenewableLearning.Cell.bSpent) ∧
    ((CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits))
            (3 * k + 2)).1 =
        if bits (3 * k) = true then CGB.RenewableLearning.Cell.aReady else CGB.RenewableLearning.Cell.bReady) ∧
      (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits))
            (3 * k + 3)).1 =
        CGB.RenewableLearning.Cell.search
```

## CGB.SelfProduction.renewable_open_probe_enables_conditional_action

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3585)

源码说明：In the witness world the existing certified policy reads the recorded distinction for two ticks; the always-passive intervention remains at the undifferentiated fallback action. This is a task-specific functional contrast.

形式陈述：

```lean
∀ (bits : ℕ → Bool) (k : ℕ),
  (CGB.RenewableLearning.policy
        (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits))
            (3 * k + 1)).1 =
      if bits (3 * k) = true then 0 else 1) ∧
    (CGB.RenewableLearning.policy
          (CGB.SelfProduction.trajectory
              (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
                (CGB.SelfProduction.existingRenewableOpenWorld bits))
              (3 * k + 2)).1 =
        if bits (3 * k) = true then 0 else 1) ∧
      CGB.RenewableLearning.policy
          (CGB.SelfProduction.controlledTrajectory
              (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
                (CGB.SelfProduction.existingRenewableOpenWorld bits))
              (fun x => 0) (3 * k + 2)).1 =
        2
```

## CGB.SelfProduction.renewable_open_rewriting_loses_old_distinction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3614)

源码说明：Recurrent writing is not enough for delayed credit: if two streams differ at one probe but agree at the next, the body first distinguishes them and then merges them. No body-only readout can recover the earlier bit after that merge. A world observation or accessible external record can change the information available to a later rule and is not ruled out here.

形式陈述：

```lean
∀ (bits₁ bits₂ : ℕ → Bool) (k : ℕ),
  bits₁ (3 * k) ≠ bits₂ (3 * k) →
    bits₁ (3 * (k + 1)) = bits₂ (3 * (k + 1)) →
      have body₁ := fun n =>
        (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits₁))
            n).1;
      have body₂ := fun n =>
        (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableOpenContract CGB.SelfProduction.existingRenewableCarrier
              (CGB.SelfProduction.existingRenewableOpenWorld bits₂))
            n).1;
      body₁ (3 * k + 1) ≠ body₂ (3 * k + 1) ∧
        body₁ (3 * (k + 1) + 1) = body₂ (3 * (k + 1) + 1) ∧
          ¬∃ read, read (body₁ (3 * (k + 1) + 1)) = bits₁ (3 * k) ∧ read (body₂ (3 * (k + 1) + 1)) = bits₂ (3 * k)
```

## CGB.SelfProduction.existing_renewable_carrier_reuses_old_rule

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3646)

源码说明：The derived transition admits the old witness's writing, renewal and reopening steps on one finite carrier.

形式陈述：

```lean
CGB.SelfProduction.renewableLanguageNext CGB.SelfProduction.existingRenewableCarrier CGB.RenewableLearning.Cell.search
      CGB.RenewableLearning.Signal.answerA =
    CGB.RenewableLearning.Cell.aSpent ∧
  CGB.SelfProduction.renewableLanguageNext CGB.SelfProduction.existingRenewableCarrier CGB.RenewableLearning.Cell.aSpent
        CGB.RenewableLearning.Signal.feed =
      CGB.RenewableLearning.Cell.aReady ∧
    CGB.SelfProduction.renewableLanguageNext CGB.SelfProduction.existingRenewableCarrier
        CGB.RenewableLearning.Cell.aReady CGB.RenewableLearning.Signal.miss =
      CGB.RenewableLearning.Cell.search
```

## CGB.SelfProduction.existing_renewable_autonomous_cycle

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3659)

源码说明：In the specified deterministic witness world the same finite carrier repeats the writing, refill and reopening cycle at every natural time scale. This is a mathematical interface witness, not a language capability claim.

形式陈述：

```lean
∀ (k : ℕ),
  CGB.SelfProduction.trajectory
        (CGB.SelfProduction.renewableLanguageContract CGB.SelfProduction.existingRenewableCarrier) (3 * k) =
      CGB.RenewableLearning.Cell.search ∧
    CGB.SelfProduction.trajectory
          (CGB.SelfProduction.renewableLanguageContract CGB.SelfProduction.existingRenewableCarrier) (3 * k + 1) =
        CGB.RenewableLearning.Cell.aSpent ∧
      CGB.SelfProduction.trajectory
          (CGB.SelfProduction.renewableLanguageContract CGB.SelfProduction.existingRenewableCarrier) (3 * k + 2) =
        CGB.RenewableLearning.Cell.aReady
```

## CGB.SelfProduction.stationary_nonincreasing_loss_flat_on_existing_cycle

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3701)

源码说明：On the existing closed three-state witness, a fixed state-dependent loss that never increases cannot strictly improve anywhere in the cycle. In particular, recurrent strict improvement against one stationary law requires allowing loss increases, changing the state/world, or changing the comparison. This does not constrain a nonperiodic open-world trajectory.

形式陈述：

```lean
∀ (loss : CGB.RenewableLearning.Cell → ℝ),
  (∀ (n : ℕ),
      loss
          (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableLanguageContract CGB.SelfProduction.existingRenewableCarrier) (n + 1)) ≤
        loss
          (CGB.SelfProduction.trajectory
            (CGB.SelfProduction.renewableLanguageContract CGB.SelfProduction.existingRenewableCarrier) n)) →
    loss CGB.RenewableLearning.Cell.aSpent = loss CGB.RenewableLearning.Cell.search ∧
      loss CGB.RenewableLearning.Cell.aReady = loss CGB.RenewableLearning.Cell.aSpent ∧
        loss CGB.RenewableLearning.Cell.search = loss CGB.RenewableLearning.Cell.aReady
```

## CGB.SelfProduction.existing_renewable_cycle_active_and_viable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3722)

源码说明：The finite witness remains outside vacuum, does not become permanently passive, and has an action-sensitive boundary in the declared witness world.

形式陈述：

```lean
(∀ (n : ℕ),
    CGB.SelfProduction.trajectory
        (CGB.SelfProduction.renewableLanguageContract CGB.SelfProduction.existingRenewableCarrier) n ≠
      CGB.RenewableLearning.Cell.vacuum) ∧
  ¬CGB.ActiveInference.eventuallyPassive
        (fun n =>
          CGB.RenewableLearning.policy
            (CGB.SelfProduction.trajectory
              (CGB.SelfProduction.renewableLanguageContract CGB.SelfProduction.existingRenewableCarrier) n))
        0 ∧
    CGB.SelfProduction.existingRenewableResponse CGB.RenewableLearning.Cell.search 2 ≠
      CGB.SelfProduction.existingRenewableResponse CGB.RenewableLearning.Cell.search 0
```

## CGB.SelfProduction.quiet_identity_memory_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3745)

源码说明：When quiet freezes the body, the historical tagged memory law has the same body marginal after the first observation at every later time. The world may still evolve; this uses only normalization of its transition kernel.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  (∀ (x y : U → Λ), M.kernel W.clock W.quiet x y = if x = y then 1 else 0) →
    ∀ (n₀ : ℕ) (x x₀ : U → Λ) (w₀ : W.W) (m : ℕ), ∑ w, M.memLaw W n₀ (m + 1) x w x₀ w₀ = ∑ w, M.memLaw W n₀ 1 x w x₀ w₀
```

## CGB.SelfProduction.quiet_identity_old_memory_iff_one_write

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3776)

源码说明：For a body frozen under quiet, the old indefinitely retained conditional information requirement reduces exactly to a single positive-information write. This is a property of that historical criterion, not a claim that freezing is sufficient for a useful or adaptive memory.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  (∀ (x y : U → Λ), M.kernel W.clock W.quiet x y = if x = y then 1 else 0) →
    (M.Remembers W ↔ ∃ n₀, 0 < CGB.Medium.condMutualInfo fun x₀ w₀ x => ∑ w, M.memLaw W n₀ 1 x w x₀ w₀)
```

## CGB.SelfProduction.historical_persistence_requires_calm_change

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3797)

源码说明：If calm freezes every body state, the historical `Persists` contract is impossible: under permanent calm its body-only alive score stays at the initial value, which normal persistence requires to be above threshold.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (W : CGB.World U Λ I),
  (∀ (x y : U → Λ), M.kernel W.clock M.calm x y = if x = y then 1 else 0) → ∀ (δ : ℝ) (T : ℕ), ¬M.Persists W δ T
```

## CGB.SelfProduction.grace_not_historical_persists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3834)

源码说明：The grace witness freezes under calm, so it cannot satisfy the old long-disconnection requirement, regardless of its other operational proofs.

形式陈述：

```lean
∀ (δ : ℝ) (T : ℕ), ¬CGB.GraceRecovery.body.Persists CGB.GraceRecovery.world δ T
```

## CGB.SelfProduction.sensory_loss_and_historical_interrupt_differ

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3846)

源码说明：The existing sensory-maintenance witness tolerates one missing-sense tick and loses its body after three, but the historical interrupt selects its different frozen equilibrium stimulus. Thus those results cannot be silently substituted into the old `Persists` contract. The separate material reserve is a property of this witness, not a requirement on information life.

形式陈述：

```lean
∀ (δ : ℝ) (T : ℕ),
  (∀ (n : ℕ),
      CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
          CGB.SensoryMaintenance.material false (fun x => CGB.SensoryMaintenance.world.sense) n =
        1) ∧
    (∀ (n : ℕ),
        7 ≤ n →
          CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
              CGB.SensoryMaintenance.material true (fun x => CGB.SensoryMaintenance.world.sense) n ≤
            3 / 4) ∧
      CGB.SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1 ∧
        (∀ (n₀ duration n : ℕ),
            3 ≤ duration →
              n₀ + 3 ≤ n →
                CGB.Exchange.survival CGB.SensoryMaintenance.body CGB.SensoryMaintenance.world CGB.SensoryMaintenance.ec
                    CGB.SensoryMaintenance.material false (CGB.SensoryMaintenance.sensoryProtocol n₀ duration) n =
                  0) ∧
          (∀ (n₀ : ℕ) (w : CGB.SensoryMaintenance.world.W),
              CGB.SensoryMaintenance.body.interrupt CGB.SensoryMaintenance.world n₀ 1 n₀ w () ≠
                CGB.SensoryMaintenance.sensoryProtocol n₀ 1 n₀ w ()) ∧
            ¬CGB.SensoryMaintenance.body.Persists CGB.SensoryMaintenance.world δ T
```

## CGB.SelfProduction.grace_historical_memory_iff_one_write

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3872)

源码说明：In the same grace witness, the old memory criterion only tests whether one observation can add conditional information before quiet freezing.

形式陈述：

```lean
CGB.GraceRecovery.body.Remembers CGB.GraceRecovery.world ↔
  ∃ n₀,
    0 <
      CGB.Medium.condMutualInfo fun x₀ w₀ x => ∑ w, CGB.GraceRecovery.body.memLaw CGB.GraceRecovery.world n₀ 1 x w x₀ w₀
```

## CGB.SelfProduction.grace_reachable_memory_changes_action

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3885)

源码说明：Two already-certified finite paths also occur in the grace world. At tick five they have the same body but different sensed world states; at tick six the body and its policy differ. Positive support is not a numerical mutual information calculation or a claim of language performance.

形式陈述：

```lean
have p := CGB.SelfProduction.renewableObservableFromBirth fun x => false;
have q := CGB.SelfProduction.renewableObservableFromBirth fun x => true;
0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) p 6 ∧
  0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) q 6 ∧
    (p 5).1 = (q 5).1 ∧
      CGB.RenewableLearning.sense (p 5).2 () ≠ CGB.RenewableLearning.sense (q 5).2 () ∧
        (p 6).1 ≠ (q 6).1 ∧ CGB.RenewableLearning.policy ((p 6).1 ()) ≠ CGB.RenewableLearning.policy ((q 6).1 ())
```

## CGB.SelfProduction.grace_reachable_equal_budget_memory_contrast

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3923)

源码说明：Two histories from the same initial process reach the same world state and bodies with equal total content, buffer and credit, but opposite stored candidates and actions. In that world the existing transition sends one body to `missA` with probability zero and the other with probability one. These are supported histories, not an intervention isolating every physical degree of freedom or a long-term survival comparison.

形式陈述：

```lean
have p := CGB.SelfProduction.renewableObservableFromBirth fun x => false;
have q := CGB.SelfProduction.renewableObservableFromBirth fun x => true;
0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) p 9 ∧
  0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) q 9 ∧
    p 9 = (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.feedA) ∧
      q 9 = (fun x => CGB.RenewableLearning.Cell.bReady, CGB.RenewableLearning.Env.feedA) ∧
        CGB.RenewableLearning.cost CGB.RenewableLearning.Cell.aReady =
            CGB.RenewableLearning.cost CGB.RenewableLearning.Cell.bReady ∧
          CGB.RenewableLearning.buffer CGB.RenewableLearning.Cell.aReady =
              CGB.RenewableLearning.buffer CGB.RenewableLearning.Cell.bReady ∧
            CGB.RenewableLearning.credit CGB.RenewableLearning.Cell.aReady =
                CGB.RenewableLearning.credit CGB.RenewableLearning.Cell.bReady ∧
              CGB.RenewableLearning.incoming true CGB.RenewableLearning.Signal.feed CGB.RenewableLearning.Cell.aReady =
                  CGB.RenewableLearning.incoming true CGB.RenewableLearning.Signal.feed
                    CGB.RenewableLearning.Cell.bReady ∧
                CGB.RenewableLearning.outgoing true true CGB.RenewableLearning.Signal.feed
                      CGB.RenewableLearning.Cell.aReady =
                    CGB.RenewableLearning.outgoing true true CGB.RenewableLearning.Signal.feed
                      CGB.RenewableLearning.Cell.bReady ∧
                  CGB.RenewableLearning.memory CGB.RenewableLearning.Cell.aReady ≠
                      CGB.RenewableLearning.memory CGB.RenewableLearning.Cell.bReady ∧
                    CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.aReady = 0 ∧
                      CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.bReady = 1 ∧
                        CGB.GraceRecovery.moves CGB.RenewableLearning.Env.feedA
                              (fun x => CGB.RenewableLearning.Cell.aReady) CGB.RenewableLearning.Env.missA =
                            0 ∧
                          CGB.GraceRecovery.moves CGB.RenewableLearning.Env.feedA
                              (fun x => CGB.RenewableLearning.Cell.bReady) CGB.RenewableLearning.Env.missA =
                            1
```

## CGB.SelfProduction.grace_action_world_matches

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3971)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (w v : CGB.RenewableLearning.Env) (q : CGB.RenewableLearning.Cell),
  q ≠ CGB.RenewableLearning.Cell.vacuum →
    CGB.GraceRecovery.moves w (fun x => q) v = CGB.SelfProduction.graceActionWorld w (CGB.RenewableLearning.policy q) v
```

## CGB.SelfProduction.grace_action_intervention_contrast

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4015)

源码说明：Reconnecting the body's policy reproduces every transition of the registered grace process; overriding action zero from birth reaches vacuum in two certain steps. Normal operational survival stays one at every tick.

形式陈述：

```lean
(∀ (t : ℕ) (z : CGB.RenewableLearning.S) (e : Option Unit) (z' : CGB.RenewableLearning.S),
    CGB.SelfProduction.graceActionMachine.step (CGB.RenewableLearning.policy (z.1 ())) z e z' =
      (CGB.GraceRecovery.process false CGB.GraceRecovery.natural).step t z e z') ∧
  CGB.SelfProduction.graceActionMachine.step 0 CGB.RenewableLearning.initial (some ())
        (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.missA) =
      1 ∧
    CGB.SelfProduction.graceActionMachine.step 0
          (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.missA) (some ())
          (fun x => CGB.RenewableLearning.Cell.vacuum, CGB.RenewableLearning.Env.probeA) =
        1 ∧
      ∀ (n : ℕ),
        CGB.Exchange.survival CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
            CGB.GraceRecovery.material false CGB.GraceRecovery.natural n =
          1
```

## CGB.SelfProduction.grace_stochastic_joint_generator_exists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4056)

源码说明：The existing grace witness supplies the local conditions of the stochastic joint generator: its already-proved good region is invariant under the original body policy. The selected policy is not asserted to equal that original policy; equality of the original policy's full law is the separate `grace_action_intervention_contrast` result.

形式陈述：

```lean
∃ policy,
  (∀ (s : Unit → CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env) (e : Option Unit)
      (z' : CGB.RenewableLearning.S),
      CGB.GraceRecovery.Good (s ()) w →
        0 < CGB.SelfProduction.graceActionMachine.step (policy s) (s, w) e z' → CGB.GraceRecovery.Good (z'.1 ()) z'.2) ∧
    ∃ process,
      (∀ (t : ℕ) (z : CGB.RenewableLearning.S) (e : Option Unit) (z' : CGB.RenewableLearning.S),
          process.step t z e z' = CGB.SelfProduction.graceActionMachine.step (policy z.1) z e z') ∧
        ∀ (n : ℕ) (z : CGB.RenewableLearning.S),
          ¬CGB.GraceRecovery.Good (z.1 ()) z.2 → CGB.Energetics.law process CGB.RenewableLearning.initial n z = 0
```

## CGB.SelfProduction.grace_joint_recovery_generator_exists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4078)

源码说明：The same finite grace world supplies the strengthened generator's local conditions, using its existing actual-intake mark and recovery rank. The chosen policy is safe and has first hit one; it is not asserted to equal the old policy or to retain its separate loop-cut and learning certificates.

形式陈述：

```lean
have mark :=
  CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material
    CGB.GraceRecovery.natural;
∃ policy process,
  (∀ (t : ℕ) (z : CGB.RenewableLearning.S) (e : Option Unit) (z' : CGB.RenewableLearning.S),
      process.step t z e z' = CGB.SelfProduction.graceActionMachine.step (policy z.1) z e z') ∧
    (∀ (n : ℕ) (z : CGB.RenewableLearning.S),
        ¬CGB.GraceRecovery.Good (z.1 ()) z.2 → CGB.Energetics.law process CGB.RenewableLearning.initial n z = 0) ∧
      ∀ (t : ℕ) (z : CGB.RenewableLearning.S),
        CGB.GraceRecovery.Good (z.1 ()) z.2 →
          CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events process) mark t CGB.GraceRecovery.world.horizon z = 1
```

## CGB.SelfProduction.grace_reachable_one_action_consequence

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4108)

源码说明：On an already-supported prefix, one action override at the same joint state changes the next observation and the following body: the normal probe gets an answer and writes the B candidate; action zero yields conflict and vacuum. Later actions are restored to the body's policy. This is a local causal contrast under the action-kernel invariance assumption, not an attribution of all survival benefit to one prior memory write.

形式陈述：

```lean
have path := CGB.SelfProduction.renewableObservableFromBirth fun x => true;
0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) path 6 ∧
  path 6 = (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.probeB) ∧
    CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.search = 2 ∧
      CGB.SelfProduction.graceActionMachine.step 2 (path 6) (some ())
            (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.answerB) =
          1 ∧
        CGB.SelfProduction.graceActionMachine.step 0 (path 6) (some ())
              (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.missB) =
            1 ∧
          CGB.SelfProduction.graceActionMachine.step 2
                (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.answerB) (some ())
                (fun x => CGB.RenewableLearning.Cell.bSpent, CGB.RenewableLearning.Env.feedB) =
              1 ∧
            CGB.SelfProduction.graceActionMachine.step 2
                  (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.missB) (some ())
                  (fun x => CGB.RenewableLearning.Cell.vacuum, CGB.RenewableLearning.Env.probeB) =
                1 ∧
              CGB.RenewableLearning.writeCost true CGB.RenewableLearning.Signal.answerB
                  CGB.RenewableLearning.Cell.search =
                1
```

## CGB.SelfProduction.grace_one_reopening_intervention_consequence

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4139)

源码说明：On a supported mismatch, override reopening for one body event only, then restore the original body and world rules. The normal event reopens to `search`; the override holds `aHeld`. Their own policies then lead to answer-and-write versus conflict-and-vacuum. The override also changes the event's outgoing-material accounting, so this does not isolate information from all physical effects.

形式陈述：

```lean
have path := CGB.SelfProduction.renewableObservableFromBirth fun x => true;
have off :=
  CGB.MaterialSource.machine
    ((CGB.RenewableLearning.plant false true).jointEventKernel CGB.GraceRecovery.world ⋯ false
      CGB.GraceRecovery.natural);
0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) path 5 ∧
  path 5 = (fun x => CGB.RenewableLearning.Cell.aReady, CGB.RenewableLearning.Env.missB) ∧
    (CGB.GraceRecovery.process false CGB.GraceRecovery.natural).step 5 (path 5) (some ())
          (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.probeB) =
        1 ∧
      off.step 5 (path 5) (some ()) (fun x => CGB.RenewableLearning.Cell.aHeld, CGB.RenewableLearning.Env.probeB) = 1 ∧
        CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.search = 2 ∧
          CGB.RenewableLearning.policy CGB.RenewableLearning.Cell.aHeld = 0 ∧
            CGB.SelfProduction.graceActionMachine.step 2
                  (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.probeB) (some ())
                  (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.answerB) =
                1 ∧
              CGB.SelfProduction.graceActionMachine.step 0
                    (fun x => CGB.RenewableLearning.Cell.aHeld, CGB.RenewableLearning.Env.probeB) (some ())
                    (fun x => CGB.RenewableLearning.Cell.aHeld, CGB.RenewableLearning.Env.missB) =
                  1 ∧
                CGB.SelfProduction.graceActionMachine.step 2
                      (fun x => CGB.RenewableLearning.Cell.search, CGB.RenewableLearning.Env.answerB) (some ())
                      (fun x => CGB.RenewableLearning.Cell.bSpent, CGB.RenewableLearning.Env.feedB) =
                    1 ∧
                  CGB.SelfProduction.graceActionMachine.step 0
                        (fun x => CGB.RenewableLearning.Cell.aHeld, CGB.RenewableLearning.Env.missB) (some ())
                        (fun x => CGB.RenewableLearning.Cell.vacuum, CGB.RenewableLearning.Env.probeB) =
                      1 ∧
                    CGB.RenewableLearning.outgoing true true CGB.RenewableLearning.Signal.miss
                          CGB.RenewableLearning.Cell.aReady =
                        2 ∧
                      CGB.RenewableLearning.outgoing false true CGB.RenewableLearning.Signal.miss
                          CGB.RenewableLearning.Cell.aReady =
                        1
```

## CGB.SelfProduction.grace_action_zero_vacuum_probability

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4191)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Energetics.law (CGB.SelfProduction.graceActionAlways 0) CGB.RenewableLearning.initial 2
    (fun x => CGB.RenewableLearning.Cell.vacuum, CGB.RenewableLearning.Env.probeA) =
  1
```

## CGB.SelfProduction.grace_recovery_learning_joint

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4233)

源码说明：The same grace-world process has operational persistence, one-tick interruption tolerance and a loop-cut maintenance advantage, while supporting arbitrarily many paid rewrites on positive-probability finite prefixes. The finite-prefix statement is not an almost-sure infinite learning result; this material witness is not an instance of the full `Medium.Lives` contract.

形式陈述：

```lean
CGB.Exchange.Persists CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec CGB.GraceRecovery.material 0 ∧
  CGB.Exchange.ToleratesSensoryInterruption CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
      CGB.GraceRecovery.material 0 1 ∧
    CGB.Exchange.SelfMaintains CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
        CGB.GraceRecovery.material ∧
      ∀ (k : ℕ),
        0 <
            CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural)
              CGB.RenewableLearningControls.path (3 + 10 * k) ∧
          ∑ t ∈ Finset.range (3 + 10 * k),
                CGB.RenewableLearning.writeCost true
                  (CGB.RenewableLearning.sense (CGB.RenewableLearningControls.path t).2 ())
                  ((CGB.RenewableLearningControls.path t).1 ()) =
              1 + 4 * k ∧
            ∑ t ∈ Finset.range (3 + 10 * k),
                CGB.RenewableLearning.topup true
                  (CGB.RenewableLearning.sense (CGB.RenewableLearningControls.path t).2 ())
                  ((CGB.RenewableLearningControls.path t).1 ()) =
              2 + 4 * k
```

## CGB.SelfProduction.grace_original_policy_joint_certificate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4255)

源码说明：A concrete policy and its actual process simultaneously satisfy the stochastic generator's one-step closure and ranked recovery conditions, the original loop-cut maintenance and one-tick interruption contracts, and the existing arbitrarily long positive-probability paid-learning prefixes. No property is transferred to an unrelated policy selected by choice.

形式陈述：

```lean
have policy := fun s => CGB.RenewableLearning.policy (s ());
have process := CGB.GraceRecovery.process false CGB.GraceRecovery.natural;
have mark :=
  CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material
    CGB.GraceRecovery.natural;
(∀ (t : ℕ) (z : (Unit → CGB.RenewableLearning.Cell) × CGB.GraceRecovery.world.W) (e : Option Unit)
    (z' : (Unit → CGB.RenewableLearning.Cell) × CGB.GraceRecovery.world.W),
    process.step t z e z' = CGB.SelfProduction.graceActionMachine.step (policy z.1) z e z') ∧
  (∀ (t : ℕ) (z : (Unit → CGB.RenewableLearning.Cell) × CGB.GraceRecovery.world.W) (e : Option Unit)
      (z' : (Unit → CGB.RenewableLearning.Cell) × CGB.GraceRecovery.world.W),
      CGB.GraceRecovery.Good (z.1 ()) z.2 → 0 < process.step t z e z' → CGB.GraceRecovery.Good (z'.1 ()) z'.2) ∧
    (∀ (n : ℕ) (z : (Unit → CGB.RenewableLearning.Cell) × CGB.GraceRecovery.world.W),
        ¬CGB.GraceRecovery.Good (z.1 ()) z.2 → CGB.Energetics.law process CGB.RenewableLearning.initial n z = 0) ∧
      (∀ (t : ℕ) (z : (Unit → CGB.RenewableLearning.Cell) × CGB.GraceRecovery.world.W),
          CGB.GraceRecovery.Good (z.1 ()) z.2 →
            CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events process) mark t CGB.GraceRecovery.world.horizon z =
              1) ∧
        CGB.Exchange.Persists CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
            CGB.GraceRecovery.material 0 ∧
          CGB.Exchange.ToleratesSensoryInterruption CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
              CGB.GraceRecovery.material 0 1 ∧
            CGB.Exchange.SelfMaintains CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.ec
                CGB.GraceRecovery.material ∧
              ∀ (k : ℕ),
                0 < CGB.RenewableLearningControls.weight process CGB.RenewableLearningControls.path (3 + 10 * k) ∧
                  ∑ t ∈ Finset.range (3 + 10 * k),
                      CGB.RenewableLearning.writeCost true
                        (CGB.RenewableLearning.sense (CGB.RenewableLearningControls.path t).2 ())
                        ((CGB.RenewableLearningControls.path t).1 ()) =
                    1 + 4 * k
```

## CGB.SelfProduction.grace_supported_paths_recurrently_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4299)

源码说明：On every infinite supported path of the original grace process, actual intake occurs in every three-tick window. This strengthens finite-prefix support for intake only, not for paid learning writes.

形式陈述：

```lean
∀ (path : ℕ → CGB.RenewableLearning.S) (event : ℕ → Option Unit),
  path 0 = CGB.RenewableLearning.initial →
    (∀ (t : ℕ),
        0 < (CGB.GraceRecovery.process false CGB.GraceRecovery.natural).step t (path t) (event t) (path (t + 1))) →
      ∀ (start : ℕ),
        ∃ t,
          start ≤ t ∧
            t < start + 3 ∧
              CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material
                  CGB.GraceRecovery.natural t (path t) (event t) (path (t + 1)) =
                true
```

## CGB.SelfProduction.grace_exists_recurrently_intaking_path

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4316)

源码说明：The original process has at least one infinite supported path, and each three-tick window on that path contains actual intake.

形式陈述：

```lean
∃ path event,
  path 0 = CGB.RenewableLearning.initial ∧
    (∀ (t : ℕ),
        0 < (CGB.GraceRecovery.process false CGB.GraceRecovery.natural).step t (path t) (event t) (path (t + 1))) ∧
      ∀ (start : ℕ),
        ∃ t,
          start ≤ t ∧
            t < start + 3 ∧
              CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material
                  CGB.GraceRecovery.natural t (path t) (event t) (path (t + 1)) =
                true
```

## CGB.SelfProduction.grace_infinite_supported_learning_and_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4339)

源码说明：The already registered paid-relearning path is itself an infinite supported path of the same grace process. It has unbounded cumulative paid writes and recurrent actual intake. This is one supported path, not a claim that any particular infinite path has positive probability or that learning continues almost surely.

形式陈述：

```lean
have path := CGB.RenewableLearningControls.path;
have event := fun t => CGB.RenewableLearningControls.tag (path t);
path 0 = CGB.RenewableLearning.initial ∧
  (∀ (t : ℕ),
      0 < (CGB.GraceRecovery.process false CGB.GraceRecovery.natural).step t (path t) (event t) (path (t + 1))) ∧
    (∀ (k : ℕ),
        ∑ t ∈ Finset.range (3 + 10 * k),
            CGB.RenewableLearning.writeCost true (CGB.RenewableLearning.sense (path t).2 ()) ((path t).1 ()) =
          1 + 4 * k) ∧
      (∀ (B : ℕ),
          ∃ T,
            B <
              ∑ t ∈ Finset.range T,
                CGB.RenewableLearning.writeCost true (CGB.RenewableLearning.sense (path t).2 ()) ((path t).1 ())) ∧
        ∀ (start : ℕ),
          ∃ t,
            start ≤ t ∧
              t < start + 3 ∧
                CGB.Exchange.intake CGB.GraceRecovery.body CGB.GraceRecovery.world CGB.GraceRecovery.material
                    CGB.GraceRecovery.natural t (path t) (event t) (path (t + 1)) =
                  true
```

## CGB.SelfProduction.grace_supported_path_eventually_no_writes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4380)

源码说明：The same grace process also admits an infinite supported path that stops paid writing after its initial phase. Thus bounded-gap ranked recurrence of paid writes cannot hold on every supported path from the seed, even though a different supported path has unbounded writing. This does not decide the probability of eventual stopping under an infinite-path measure.

形式陈述：

```lean
have path := CGB.SelfProduction.renewableObservableFromBirth fun x => false;
have event := fun t => CGB.RenewableLearningControls.tag (path t);
path 0 = CGB.RenewableLearning.initial ∧
  (∀ (t : ℕ),
      0 < (CGB.GraceRecovery.process false CGB.GraceRecovery.natural).step t (path t) (event t) (path (t + 1))) ∧
    (∀ (t : ℕ),
        3 ≤ t → CGB.RenewableLearning.writeCost true (CGB.RenewableLearning.sense (path t).2 ()) ((path t).1 ()) = 0) ∧
      ¬∃ horizon C, C.region (path 0)
```

## CGB.SelfProduction.uniform_block_learning_failure_decay

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4436)

源码说明：Minimal probabilistic upgrade for persistent learning: if the probability of no useful learning event after `n` completed blocks is `q n`, and every block has conditional success at least `p`, then failure is bounded by `(1-p)^n`. This is an abstract finite-prefix inequality; a real infinite process must separately prove the conditional minorization premise.

形式陈述：

```lean
∀ p ≤ 1, ∀ (q : ℕ → ℝ), q 0 ≤ 1 → (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) → ∀ (n : ℕ), q n ≤ (1 - p) ^ n
```

## CGB.SelfProduction.language_conditional_joint_specification

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4465)

源码说明：The current candidate specification and the conditional infinite-stream guarantees hold for one and the same encoded trajectory. The first conjunct is the complete specification currently defined in this file; the remaining conjuncts add the explicit alignment, predictive-payment, and block-learning conditions needed for active continual operation.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte] (L : CGB.SelfProduction.LanguageSelfProductionCandidate Hist Model Ctx Act Obs Byte Answer),
  L.base.encode (L.base.trajectory 0) = L.base.encode L.seed →
    (∀ (n : ℕ),
        L.base.encode (L.base.trajectory (n + 1)) =
          L.base.next (L.base.encode (L.base.trajectory n))
            (L.world_response (L.base.encode (L.base.trajectory n))
              (L.base.policy (L.base.encode (L.base.trajectory n))))) →
      ∀ (m : ℝ),
        (∀ (h : Hist),
            CGB.PredictiveHarvest.crossEntropy (L.base.stream h) (L.base.allocation (L.base.encode h)) <
              Real.log ↑(Fintype.card Byte) - m) →
          ∀ p ≤ 1,
            ∀ (q : ℕ → ℝ),
              q 0 ≤ 1 →
                (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) →
                  CGB.SelfProduction.language_infinite_stream_life_spec L ∧
                    CGB.LanguageLifeAdmission.joint_admission_spec L.base ∧
                      (∀ (n : ℕ),
                          CGB.SelfProduction.language_viable L
                            (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)) ∧
                        ¬CGB.ActiveInference.eventuallyPassive
                              (fun n =>
                                L.base.policy
                                  (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n))
                              L.base.passive ∧
                          (∀ (h : Hist),
                              0 <
                                CGB.PredictiveHarvest.growth (L.base.stream h) (L.base.allocation (L.base.encode h)) -
                                  m) ∧
                            ∀ (n : ℕ), q n ≤ (1 - p) ^ n
```

## CGB.SelfProduction.minimal_generator_conditional_joint_specification

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4496)

源码说明：Minimal-generator form of the total gate. History factorization supplies the finite causal carrier; the displayed hypotheses are exactly the remaining world alignment, predictive payment, and block-learning obligations.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte]
  (G : CGB.SelfProduction.MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer),
  (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
        ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory 0) =
      (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
        (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).seed →
    (∀ (n : ℕ),
        (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory (n + 1)) =
          (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.next
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).world_response
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.policy
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))))) →
      ∀ (m : ℝ),
        (∀ (h : Hist),
            CGB.PredictiveHarvest.crossEntropy
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.stream h)
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.allocation
                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode h)) <
              Real.log ↑(Fintype.card Byte) - m) →
          ∀ p ≤ 1,
            ∀ (q : ℕ → ℝ),
              q 0 ≤ 1 →
                (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) →
                  CGB.SelfProduction.language_infinite_stream_life_spec
                      (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G) ∧
                    CGB.LanguageLifeAdmission.joint_admission_spec
                        (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base ∧
                      (∀ (n : ℕ),
                          CGB.SelfProduction.language_viable
                            (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G)
                            (CGB.SelfProduction.trajectory
                              (CGB.SelfProduction.language_contract
                                (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G))
                              n)) ∧
                        ¬CGB.ActiveInference.eventuallyPassive
                              (fun n =>
                                (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.policy
                                  (CGB.SelfProduction.trajectory
                                    (CGB.SelfProduction.language_contract
                                      (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G))
                                    n))
                              (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.passive ∧
                          (∀ (h : Hist),
                              0 <
                                CGB.PredictiveHarvest.growth
                                    ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.stream h)
                                    ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.allocation
                                      ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
                                        h)) -
                                  m) ∧
                            ∀ (n : ℕ), q n ≤ (1 - p) ^ n
```

## CGB.SelfProduction.exists_conditionally_operating_language_life_of_minimal_generator

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4548)

源码说明：Existential form of the total gate. Once a minimal generator and its world conditions are supplied, materialization produces one concrete language candidate satisfying the current life specification, active self-production, predictive payment, and block-learning bound together.

形式陈述：

```lean
∀ {Hist : Type u_1} {Model : Type u_2} {Ctx : Type u_3} {Act : Type u_4} {Obs : Type u_5} {Byte : Type u_6}
  {Answer : Type u_7} [inst : Fintype Model] [inst_1 : DecidableEq Model] [inst_2 : Fintype Byte]
  [inst_3 : Nonempty Byte]
  (G : CGB.SelfProduction.MinimalLanguageSelfProductionGenerator Hist Model Ctx Act Obs Byte Answer),
  (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
        ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory 0) =
      (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
        (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).seed →
    (∀ (n : ℕ),
        (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory (n + 1)) =
          (CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.next
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))
            ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).world_response
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))
              ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.policy
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode
                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.trajectory n))))) →
      ∀ (m : ℝ),
        (∀ (h : Hist),
            CGB.PredictiveHarvest.crossEntropy
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.stream h)
                ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.allocation
                  ((CGB.SelfProduction.materializeLanguageSelfProductionGenerator G).base.encode h)) <
              Real.log ↑(Fintype.card Byte) - m) →
          ∀ p ≤ 1,
            ∀ (q : ℕ → ℝ),
              q 0 ≤ 1 →
                (∀ (n : ℕ), q (n + 1) ≤ (1 - p) * q n) →
                  ∃ L,
                    CGB.SelfProduction.language_infinite_stream_life_spec L ∧
                      CGB.LanguageLifeAdmission.joint_admission_spec L.base ∧
                        (∀ (n : ℕ),
                            CGB.SelfProduction.language_viable L
                              (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n)) ∧
                          ¬CGB.ActiveInference.eventuallyPassive
                                (fun n =>
                                  L.base.policy
                                    (CGB.SelfProduction.trajectory (CGB.SelfProduction.language_contract L) n))
                                L.base.passive ∧
                            (∀ (h : Hist),
                                0 <
                                  CGB.PredictiveHarvest.growth (L.base.stream h) (L.base.allocation (L.base.encode h)) -
                                    m) ∧
                              ∀ (n : ℕ), q n ≤ (1 - p) ^ n
```
