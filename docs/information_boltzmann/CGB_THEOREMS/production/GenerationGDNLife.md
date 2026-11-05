# GenerationGDNLife

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean) · [返回导航](../README.md)

模块源码说明：Real-valued GDN carrier on one open trajectory The body is a fixed-dimensional real state, not a finite enumeration.

## CGB.GDNLife.address_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L73)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型参数与字节输入，经指数函数严格正性传递，输出寻址权重在每个槽位上均严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (b : CGB.GDNLife.Byte) (j : Fin k), 0 < CGB.GDNLife.address M b j
```

## CGB.GDNLife.action_head_score_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L107)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型与状态，经G14对数Softmax法则展开，输出动作头对数得分对输出对数几率的导数精确等价于Softmax指示差分。

源码说明：The existing G14 log-softmax law instantiated at the GDN language head. Thus the first two factors of delayed action credit are fixed by the pre-action body and emitted byte; only their persistence until an external outcome is a separate causal-carrier obligation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (emitted coordinate : CGB.GDNLife.Byte) (i : Fin d),
  HasDerivAt
    (fun θ =>
      Real.log
        (CGB.Medium.softmax
          (fun b =>
            CGB.GDNLife.logits s (CGB.GDNLife.feature M s) b +
              if b = coordinate then θ * CGB.GDNLife.feature M s i else 0)
          emitted))
    (CGB.GDNLife.actionHeadScore M s emitted coordinate i) 0
```

## CGB.GDNLife.addressLogitPerturb_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L187)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型与基底扰动，经Softmax求导法则，输出地址对数几率基底坐标扰动的精确Softmax导数。

源码说明：Exact softmax derivative for an address-logit basis coordinate. This is the G14 finite-alphabet score identity applied to the GDN address, rather than an assumed attention-gradient rule.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (b : CGB.GDNLife.Byte) (j₀ j : Fin k),
  HasDerivAt (fun θ => CGB.GDNLife.address (CGB.GDNLife.addressLogitPerturb M b j₀ θ) b j)
    (CGB.GDNLife.address M b j * ((if j = j₀ then 1 else 0) - CGB.GDNLife.address M b j₀)) 0
```

## CGB.GDNLife.assimilate_addressLogitPerturb_natural_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L211)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型、状态与地址扰动，经自然参数积分展开，输出地址侧对GDN自然参数变动的精确导数。

源码说明：The exact address-side contribution to a GDN natural-parameter write. The address and value paths are separate summands of an encoder's total derivative; neither may be silently discarded.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (j₀ j : Fin k)
  (i : Fin d),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.addressLogitPerturb M a.byte j₀ θ) s a).natural j i)
    (CGB.GDNLife.addressEligibility M a j₀ j i) 0
```

## CGB.GDNLife.assimilate_addressLogitPerturb_precision_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L235)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入状态与地址扰动，经精度累加分析，输出地址扰动导致的精度变动严格受地址敏感度调控。

源码说明：The same address perturbation changes precision by the address-side incoming derivative. It is part of the future state sensitivity and cannot be represented by the value-only result.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (j₀ j : Fin k),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.addressLogitPerturb M a.byte j₀ θ) s a).precision j)
    (a.observationPrecision * M.likelihoodPrecision a.byte * CGB.GDNLife.address M a.byte j *
      ((if j = j₀ then 1 else 0) - CGB.GDNLife.address M a.byte j₀))
    0
```

## CGB.GDNLife.likelihoodSoftplus_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L262)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入Softplus置信度参数，经微积分导数计算，输出Softplus置信度参数的精确一阶导数等于标准Logistic Sigmoid函数。

源码说明：Exact derivative of the actual softplus confidence parametrization.

形式陈述：

```lean
∀ (r : ℝ), HasDerivAt (fun θ => CGB.GDNLife.likelihoodSoftplus (r + θ)) (CGB.GDNLife.likelihoodSlope r) 0
```

## CGB.GDNLife.likelihood_posterior_mean_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L301)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在分母非零条件下，GDN 似然 softplus 门控后验均值关于局部未归一化 logits 偏移的一阶导数精确等于解析后验斜率。

源码说明：Exact local posterior sensitivity for the learned likelihood logit. A confidence update changes both C's natural numerator and its precision, so the usable direction is their quotient-rule combination, rather than the natural write alone. This is the runtime refinement of the preceding two arrival sensitivity identities; it adds no trace and no reverse-time graph.

形式陈述：

```lean
∀ (natural precision arrivalWeight value raw : ℝ),
  precision + arrivalWeight * CGB.GDNLife.likelihoodSoftplus raw ≠ 0 →
    HasDerivAt (fun θ => CGB.GDNLife.likelihoodPosteriorMean natural precision arrivalWeight value (raw + θ))
      (CGB.GDNLife.likelihoodPosteriorSlope natural precision arrivalWeight value raw) 0
```

## CGB.GDNLife.assimilate_likelihoodLogitPerturb_natural_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L353)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型与似然对数几率扰动，经自然参数增量求导，输出似然对数几率对自然参数更新的精确导数。

源码说明：The likelihood side of the current GDN write: a learned confidence logit changes the natural parameter by its softplus slope times the existing address, observation precision and value.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (raw : ℝ) (j : Fin k)
  (i : Fin d),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.likelihoodLogitPerturb M a.byte raw θ) s a).natural j i)
    (a.observationPrecision * CGB.GDNLife.address M a.byte j * M.value a.byte i * CGB.GDNLife.likelihoodSlope raw) 0
```

## CGB.GDNLife.assimilate_likelihoodLogitPerturb_precision_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L370)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入学习置信度参数与扰动，经精度更新方程求导，输出置信度对存储自然精度的精确导数。

源码说明：The same learned confidence changes the stored precision. Thus it cannot be treated as a fixed write scale while claiming the encoder is trainable.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (raw : ℝ) (j : Fin k),
  HasDerivAt (fun θ => (CGB.GDNLife.assimilate (CGB.GDNLife.likelihoodLogitPerturb M a.byte raw θ) s a).precision j)
    (a.observationPrecision * CGB.GDNLife.address M a.byte j * CGB.GDNLife.likelihoodSlope raw) 0
```

## CGB.GDNLife.assimilate_likelihoodLogitPerturb_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L389)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入表示当前置信度的原始对数几率，经零扰动求值，输出零摄动下自然参数与精度导数恒为零。

源码说明：When the raw logit represents the current confidence, the perturbation passes through the actual arrival transition at zero exactly. This establishes that the two derivative theorems are tangents of the existing model, rather than of an auxiliary learning rule.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (raw : ℝ),
  M.likelihoodPrecision a.byte = CGB.GDNLife.likelihoodSoftplus raw →
    CGB.GDNLife.assimilate (CGB.GDNLife.likelihoodLogitPerturb M a.byte raw 0) s a = CGB.GDNLife.assimilate M s a
```

## CGB.GDNLife.prepare_valuePerturb

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L397)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入未到达字节编码扰动，经准备算子求解，输出改变未来字节编码无法影响在当前流逝时间下预先准备好的预测状态。

源码说明：Altering a byte value cannot alter the prediction prepared before that byte arrives. This is the causal boundary for its later credit.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ elapsed : ℝ)
  (s : CGB.GDNLife.State k d),
  CGB.GDNLife.prepare (CGB.GDNLife.valuePerturb M direction θ) s elapsed = CGB.GDNLife.prepare M s elapsed
```

## CGB.GDNLife.incoming_valuePerturb

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L404)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入字节编码器扰动，经流入强度核算，输出修改字节值编码严格保持到达包的流入权重不变。

源码说明：Changing the byte encoder leaves the arrival weight unchanged: it does not silently alter the address or likelihood path.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (a : CGB.GDNLife.Arrival)
  (j : Fin k), CGB.GDNLife.incoming (CGB.GDNLife.valuePerturb M direction θ) a j = CGB.GDNLife.incoming M a j
```

## CGB.GDNLife.assimilate_valuePerturb_natural

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L413)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入字节编码器单点扰动，经自然参数累加展开，输出编码坐标扰动对存储自然参数的精确一阶变动。

源码说明：Exact one-arrival finite difference for a byte encoder coordinate. The coefficient is an arrival-local eligibility seed. It is a theorem about the actual GDN assimilation arithmetic, not a claim that delayed future credit can already be recovered without propagating this sensitivity.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a).natural j i =
    (CGB.GDNLife.assimilate M s a).natural j i + θ * CGB.GDNLife.valueEligibility M direction a j i
```

## CGB.GDNLife.assimilate_valuePerturb_precision

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L429)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入字节编码器扰动，经精度更新分析，输出该扰动使存储精度严格保持不变。

源码说明：The same perturbation leaves precision unchanged. A future learner can therefore not attribute an encoder-value effect to a fabricated precision change in this fixed-address GDN subcase.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival) (j : Fin k),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a).precision j =
    (CGB.GDNLife.assimilate M s a).precision j
```

## CGB.GDNLife.assimilate_valuePerturb_difference

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L438)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入两组编码器坐标摄动，经有限差分比较，输出输入编码器不同坐标引起的精确有限差分。

源码说明：Comparing two encoder coordinates gives the exact finite difference, without invoking reverse-time differentiation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ₁ θ₂ : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ₁) s a).natural j i -
      (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ₂) s a).natural j i =
    (θ₁ - θ₂) * CGB.GDNLife.valueEligibility M direction a j i
```

## CGB.GDNLife.valueEligibility_zero_of_no_incoming

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L448)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入无流入到达包（流入强度为零），经资格迹传递链展开，输出该槽位处编码器值的后向资格迹恒等于零。

源码说明：No arriving flow means no encoder-value eligibility at that address.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (a : CGB.GDNLife.Arrival) (j : Fin k)
  (i : Fin d), CGB.GDNLife.incoming M a j = 0 → CGB.GDNLife.valueEligibility M direction a j i = 0
```

## CGB.GDNLife.value_sensitivity_two_arrivals

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L456)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入连续两次到达事件，经自然参数吸收链推演，输出编码器值敏感度由同一自然状态载体连续运载与复合传递。

源码说明：The value-side sensitivity is carried by the same natural state across a second arrival. The second local term is added; no history buffer or reverse computation is introduced.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ)
  (a₀ a₁ : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ)
          (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a₀) a₁).natural
      j i =
    (CGB.GDNLife.assimilate M (CGB.GDNLife.assimilate M s a₀) a₁).natural j i +
      θ * (CGB.GDNLife.valueEligibility M direction a₀ j i + CGB.GDNLife.valueEligibility M direction a₁ j i)
```

## CGB.GDNLife.initial_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L485)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入正的初始精度参数，经初始状态构造，输出初始GDN物理状态满足正精度公理。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (p : ℝ),
  0 < p →
    ∀ (head : CGB.GDNLife.Byte → Fin d → ℝ) (bias : CGB.GDNLife.Byte → ℝ),
      CGB.GDNLife.Positive (CGB.GDNLife.initial p head bias)
```

## CGB.GDNLife.prior_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L489)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入GDN状态与零流逝时间，经先验算子求值，输出零时间流逝下的先验状态严格等于当前状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d), CGB.GDNLife.prior M s 0 = s
```

## CGB.GDNLife.prior_natural_difference

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L502)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入保持槽位精度不变的扰动，经先验时间衰减传递，输出先验演化精确保留自然参数在各槽位上的差分。

源码说明：If a perturbation leaves a slot's precision unchanged, the prior transports its natural-state difference by `priorNaturalScale`; no past input is reread.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s s' : CGB.GDNLife.State k d) (elapsed : ℝ) (j : Fin k) (i : Fin d),
  s'.precision j = s.precision j →
    (CGB.GDNLife.prior M s' elapsed).natural j i - (CGB.GDNLife.prior M s elapsed).natural j i =
      CGB.GDNLife.priorNaturalScale M s elapsed j * (s'.natural j i - s.natural j i)
```

## CGB.GDNLife.value_sensitivity_through_prior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L523)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入编码器写入敏感度，经物理先验时间演化，输出敏感度在经历时间先验后依然保存在相同的自然状态流形中。

源码说明：A value-write sensitivity remains in the same state after the time prior. The result is exact for the present GDN path: the prior rescales the existing trace instead of allocating a separate history carrier.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ)
  (θ elapsed : ℝ) (a : CGB.GDNLife.Arrival) (j : Fin k) (i : Fin d),
  (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed).natural
        j i -
      (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed).natural j i =
    CGB.GDNLife.priorNaturalScale M (CGB.GDNLife.assimilate M s a) elapsed j *
      (θ * CGB.GDNLife.valueEligibility M direction a j i)
```

## CGB.GDNLife.feature_difference_of_shared_query

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L543)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入共享查询与相同精度下的两组自然状态，经特征解码映射，输出两状态特征读出之差精确正比于自然参数之差。

源码说明：With unchanged precision and causal query, an existing natural-state difference is read out linearly by the same GDN feature map. This is the readout side of forward sensitivity, not an auxiliary eligibility array.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s s' : CGB.GDNLife.State k d),
  (∀ (j : Fin k), s'.precision j = s.precision j) →
    s'.lastByte = s.lastByte →
      ∀ (i : Fin d),
        CGB.GDNLife.feature M s' i - CGB.GDNLife.feature M s i =
          ∑ j, (s'.natural j i - s.natural j i) / max (s.precision j) M.precisionFloor * CGB.GDNLife.query M s j
```

## CGB.GDNLife.feature_valuePerturb

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L560)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入字节值编码器摄动，经预测读出分析，输出未到达字节的值编码扰动对当前已准备的特征读出无直接影响。

源码说明：A perturbation of the byte-value encoder has no direct readout path. Its only possible effect is the natural-state flow written at arrival.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ) s = CGB.GDNLife.feature M s
```

## CGB.GDNLife.assimilate_lastByte_of_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L567)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入正置信度到达包，经因果缓存写入，输出该到达字节精确成为下一次预测的因果查询。

源码说明：A positive-confidence arrival makes that byte the next causal query. This is an existing `lastByte` state field, rather than a new trace.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision → (CGB.GDNLife.assimilate M s a).lastByte = some a.byte
```

## CGB.GDNLife.value_read_eligibility_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L585)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入单步字节编码器方向，经因果读出展开，输出该方向对下一因果读出的资格迹精确等于槽位权重归一化因子。

源码说明：One byte-value encoder direction reaches the next causal readout exactly through `valueReadEligibility`. No reverse-time graph, copied history, or separate eligibility state is used: the coefficient is reconstructed from the same material state that already stores the write.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ : ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (i : Fin d),
      CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ)
          (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) i =
        CGB.GDNLife.feature M (CGB.GDNLife.assimilate M s a) i + θ * CGB.GDNLife.valueReadEligibility M direction s a i
```

## CGB.GDNLife.value_future_read_eligibility_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L638)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入后继准备好的读出状态，经前向敏感度递推，输出后继特征读出的前向资格迹精确满足一步前向闭式递推。

源码说明：The exact encoder-value eligibility at the next prepared readout. A write is transported by the GDN prior and read by the same causal query; no backward graph or unbounded trace is introduced.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (i : Fin d),
      CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ)
          (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed)
          i =
        CGB.GDNLife.feature M (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed) i +
          θ * CGB.GDNLife.valueFutureReadEligibility M direction s a elapsed i
```

## CGB.GDNLife.assimilate_head

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L709)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型、状态与到达包，经吸收步进计算，输出吸收操作仅写入物理内存场与因果缓存，输出头权重矩阵保持不变。

源码说明：Assimilation writes only the material memory fields and causal observation cache; it does not create a second copy of the language readout parameters.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  (CGB.GDNLife.assimilate M s a).head = s.head
```

## CGB.GDNLife.assimilate_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L712)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型、状态与到达包，经吸收更新算子，输出吸收变换保持读出头的偏置参数严格不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  (CGB.GDNLife.assimilate M s a).bias = s.bias
```

## CGB.GDNLife.prior_head

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L716)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入物理时间先验转移算子，经参数传递性验证，输出时间先验仅迁移记忆均值与精度，严格保留相同的输出头矩阵与偏置。

源码说明：The time prior transports memory but retains the same head and bias.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (elapsed : ℝ),
  (CGB.GDNLife.prior M s elapsed).head = s.head
```

## CGB.GDNLife.prior_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L720)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入GDN状态与经过时间，经先验衰减演化，输出时间先验演化严格保持输出头偏置不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (elapsed : ℝ),
  (CGB.GDNLife.prior M s elapsed).bias = s.bias
```

## CGB.GDNLife.value_future_logit_eligibility_exact

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L735)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入后继字节对数几率位移，经链式法则推导，输出后继对数几率位移精确等于输出头权重与编码器前向资格迹的内积。

源码说明：The next byte-logit displacement is exactly the forward eligibility read through the unchanged affine head. This is still an identity, before an external outcome is used as a third factor.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (θ elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (b : CGB.GDNLife.Byte),
      CGB.GDNLife.logits
          (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed)
          (CGB.GDNLife.feature (CGB.GDNLife.valuePerturb M direction θ)
            (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
              (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed))
          b =
        CGB.GDNLife.logits (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed)
            (CGB.GDNLife.feature M (CGB.GDNLife.prior M (CGB.GDNLife.assimilate M s a) elapsed)) b +
          θ * CGB.GDNLife.valueFutureLogitEligibility M direction s a elapsed b
```

## CGB.GDNLife.byte_log_softmax_direction_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L772)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入对数几率方向向量，经有限维Softmax求导展开，输出字节对数Softmax沿整个对数几率方向的一阶导数。

源码说明：G14 for a whole finite logit direction. Coordinate G14 is the special case in which `direction` has one nonzero component.

形式陈述：

```lean
∀ (z direction : CGB.GDNLife.Byte → ℝ) (target : CGB.GDNLife.Byte),
  HasDerivAt (fun θ => Real.log (CGB.Medium.softmax (fun b => z b + θ * direction b) target))
    (direction target - ∑ b, CGB.Medium.softmax z b * direction b) 0
```

## CGB.GDNLife.value_future_log_likelihood_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L828)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入未来后继字节到达与GDN单步位移，经G14导数法则，输出未来对数似然对编码器值的全导数精确由前向资格迹给出。

源码说明：G14 applied to the exact next-byte GDN displacement: the derivative of next-byte log likelihood with respect to an encoder-value direction is the forward eligibility weighted by the observed-minus-predicted language error. This derives the local two factors; a delayed external outcome may only enter later as the already-required third factor.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (target : CGB.GDNLife.Byte),
      HasDerivAt
        (fun θ =>
          Real.log
            (CGB.GDNLife.allocation (CGB.GDNLife.valuePerturb M direction θ)
              (CGB.GDNLife.prior (CGB.GDNLife.valuePerturb M direction θ)
                (CGB.GDNLife.assimilate (CGB.GDNLife.valuePerturb M direction θ) s a) elapsed)
              target))
        (CGB.GDNLife.valueFutureScore M direction s a elapsed target) 0
```

## CGB.GDNLife.value_future_teaching_direction_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L865)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：内生教学幅度无需是前向通路的转置；局部方向经教学标量缩放后精确导出对应的未来方向导数，这确立了即时语言路径的导数恒等式，而非假设电路已预先学习了该幅度。

源码说明：An endogenous teaching amplitude need not be a transpose of the forward path. The only admission condition for this immediate language path is its measurable alignment with the already-derived future score: scaling a local direction by `teaching` gives the corresponding directional derivative. This is deliberately an identity, not a claim that a particular circuit has learnt the amplitude or that delayed action credit is solved.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (teaching elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (target : CGB.GDNLife.Byte),
      HasDerivAt
        (fun η =>
          have Mη := CGB.GDNLife.valuePerturb M direction (η * teaching);
          Real.log (CGB.GDNLife.allocation Mη (CGB.GDNLife.prior Mη (CGB.GDNLife.assimilate Mη s a) elapsed) target))
        (teaching * CGB.GDNLife.valueFutureScore M direction s a elapsed target) 0
```

## CGB.GDNLife.positive_teaching_alignment_is_local_ascent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L883)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在 GDN 动力学中，学得教学信号与未来精确分值方向的正对齐是局部梯度上升的充要准入条件；该条件仅要求可测量的对齐，不强制要求预设对称转置权重矩阵。

源码说明：Positive alignment with the exact future score is the minimal local ascent admission for a learned teaching signal. It does not mandate how the signal is represented; a concrete shared-body rule must establish the premise without preset transposed weights.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (teaching elapsed : ℝ)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision →
    ∀ (target : CGB.GDNLife.Byte),
      0 < teaching * CGB.GDNLife.valueFutureScore M direction s a elapsed target →
        ∃ deriv,
          0 < deriv ∧
            HasDerivAt
              (fun η =>
                have Mη := CGB.GDNLife.valuePerturb M direction (η * teaching);
                Real.log
                  (CGB.GDNLife.allocation Mη (CGB.GDNLife.prior Mη (CGB.GDNLife.assimilate Mη s a) elapsed) target))
              deriv 0
```

## CGB.GDNLife.value_future_score_factor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L906)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入编码器值得分函数，经梯度因式分解，输出未来得分精确分解为前向资格迹因子与即时创新误差的乘积。

源码说明：Exact factorization of the encoder-value score. The forward material eligibility and the observed-minus-predicted language error are sufficient statistics for this one-step value path; their product is not a hand-added surrogate for BPTT.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (direction : CGB.GDNLife.Byte → Fin d → ℝ) (s : CGB.GDNLife.State k d)
  (a : CGB.GDNLife.Arrival) (elapsed : ℝ) (target : CGB.GDNLife.Byte),
  CGB.GDNLife.valueFutureScore M direction s a elapsed target =
    ∑ i,
      CGB.GDNLife.valueFutureReadEligibility M direction s a elapsed i *
        CGB.GDNLife.nextFeatureLanguageError M s a elapsed target i
```

## CGB.GDNLife.address_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L956)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型参数与输入字节，经软寻址Softmax计算，输出地址权重向量各槽位分量恒非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (b : CGB.GDNLife.Byte) (j : Fin k), 0 ≤ CGB.GDNLife.address M b j
```

## CGB.GDNLife.incoming_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L960)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入模型参数与到达包，经流入权重映射，输出到达各槽位的流入强度非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (a : CGB.GDNLife.Arrival) (j : Fin k), 0 ≤ CGB.GDNLife.incoming M a j
```

## CGB.GDNLife.prior_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L965)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入满足正精度的GDN状态与非负流逝时间，经时间先验指数衰减，输出衰减后状态的精度依然严格保持正性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s → ∀ (elapsed : ℝ), 0 ≤ elapsed → CGB.GDNLife.Positive (CGB.GDNLife.prior M s elapsed)
```

## CGB.GDNLife.assimilate_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L977)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入正精度初始状态与到达包，经吸收更新，输出更新后状态的所有槽位精度继续保持严格正性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s → ∀ (a : CGB.GDNLife.Arrival), CGB.GDNLife.Positive (CGB.GDNLife.assimilate M s a)
```

## CGB.GDNLife.step_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L982)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入正精度的GDN状态与到达包，经单步复合推进更新，输出完整步进后的后继状态依然满足正精度条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s → ∀ (a : CGB.GDNLife.Arrival), CGB.GDNLife.Positive (CGB.GDNLife.step M s a)
```

## CGB.GDNLife.zero_precision_keeps_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L988)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入零精度输入包，经状态更新判定，输出零精度输入保持既有最后观测缓存不变，避免虚假覆写。

源码说明：Zero-precision input retains an existing last observation, matching `assimilate`'s live-observation mask rather than silently changing query.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival) (b : CGB.GDNLife.Byte),
  s.lastByte = some b → a.observationPrecision = 0 → (CGB.GDNLife.assimilate M s a).lastByte = some b
```

## CGB.GDNLife.first_observation_sets_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L994)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入零初始精度状态与首个输入字节，经缓存写入规则，输出即使在零精度下首个观测字节仍精确写入因果查询缓存。

源码说明：The first input still installs the cache, even at zero precision.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  s.lastByte = none → (CGB.GDNLife.assimilate M s a).lastByte = some a.byte
```

## CGB.GDNLife.positive_observation_sets_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1000)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入任意正置信度到达观测，经因果读出更新，输出该字节直接写入下一次因果查询。

源码说明：Any live arrival writes its byte into the existing causal query field. This is a state transition, not a separate eligibility or action-log array.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision → (CGB.GDNLife.assimilate M s a).lastByte = some a.byte
```

## CGB.GDNLife.step_positive_observation_sets_last_byte

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1009)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入完整GDN单步推进算子与正置信度输入，经端到端追踪，输出完整步进精确保持因果查询缓存写入。

源码说明：The complete GDN step preserves that live causal write through its local head update. A self-produced language byte can therefore use the same body transition as an external byte; source provenance remains a boundary condition for later outcome modulation.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  0 < a.observationPrecision → (CGB.GDNLife.step M s a).lastByte = some a.byte
```

## CGB.GDNLife.distinct_live_bytes_separate_steps

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1019)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入两个不同的有效输入字节，经状态写入推进，输出更新后的后继状态必然可分离且两两不等。

源码说明：Distinct live language emissions produce distinct post-step bodies through an existing causal field. This establishes the positive premise needed by `delayed_action_credit_requires_state_separation` for a byte-level speech act; it does not claim that the distinction survives arbitrary later interference.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a₀ a₁ : CGB.GDNLife.Arrival),
  0 < a₀.observationPrecision →
    0 < a₁.observationPrecision → a₀.byte ≠ a₁.byte → CGB.GDNLife.step M s a₀ ≠ CGB.GDNLife.step M s a₁
```

## CGB.GDNLife.step_source_irrelevant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1032)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入外部源标签，经状态学习转移比对，输出输入源标签不改变语言费用或GDN学习规则本身。

源码说明：Source labels do not invent a speech fee or a different learning rule. The world must still distinguish external provenance from self-produced text when deciding whether an event is new evidence.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival)
  (src : CGB.GDNLife.Source),
  CGB.GDNLife.step M s
      { elapsed := a.elapsed, elapsed_nonneg := ⋯, byte := a.byte, observationPrecision := a.observationPrecision,
        observation_nonneg := ⋯, source := src } =
    CGB.GDNLife.step M s a
```

## CGB.GDNLife.preparation_uses_only_elapsed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1036)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入当前已到达历史与未来未到达字节，经因果隔离分析，输出预测准备过程仅依赖流逝时间与历史，无法窥探未来字节。

源码说明：Prediction cannot inspect the not-yet-arrived byte or its source.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (a a' : CGB.GDNLife.Arrival),
  a.elapsed = a'.elapsed → CGB.GDNLife.prepare M s a.elapsed = CGB.GDNLife.prepare M s a'.elapsed
```

## CGB.GDNLife.head_update_preserves_memory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1041)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入局部头参数更新算子，经状态分量独立性分析，输出局部读出头参数学习绝不改变内部记忆状态或其不确定度（精度）。

源码说明：Local head learning does not modify the memory or its uncertainty.

形式陈述：

```lean
∀ {k d : ℕ} (η : ℝ) (s : CGB.GDNLife.State k d) (f : Fin d → ℝ) (b : CGB.GDNLife.Byte),
  (CGB.GDNLife.headUpdate η s f b).natural = s.natural ∧ (CGB.GDNLife.headUpdate η s f b).precision = s.precision
```

## CGB.GDNLife.head_update_logits

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1047)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入参数更新步长与特征向量，经仿射变换展开，输出参数步进在输出层诱导出大小精确为学习率乘以特征模方的一阶对数几率增量。

源码说明：The actual affine parameter step induces a score step of size `eta * (1 + squared feature norm)`, including the learned bias.

形式陈述：

```lean
∀ {k d : ℕ} (η : ℝ) (s : CGB.GDNLife.State k d) (f : Fin d → ℝ) (target b : CGB.GDNLife.Byte),
  CGB.GDNLife.logits (CGB.GDNLife.headUpdate η s f target) f b =
    CGB.GDNLife.logits s f b + η * (1 + ∑ i, f i * f i) * CGB.GDNLife.innovation s f target b
```

## CGB.GDNLife.head_update_other_feature

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1066)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在 G14 先验更新下，对后续特征的读出变化保持一阶秩一修正形式；该单步更新对后继特征的完整效应可由边界回执与后继特征的单一内积直接恢复，无需提前存储全局历史雅可比矩阵。

源码说明：A G14 head update remains a rank-one change when read on a later feature. The complete effect of this *one known update* is recoverable from its boundary receipt (`innovation`, saved feature) and the later feature through one inner product. This does not supply the derivative of the later feature with respect to an earlier contact parameter; that separate causal factor is the open long-credit obligation.

形式陈述：

```lean
∀ {k d : ℕ} (η : ℝ) (s : CGB.GDNLife.State k d) (written read : Fin d → ℝ) (target b : CGB.GDNLife.Byte),
  CGB.GDNLife.logits (CGB.GDNLife.headUpdate η s written target) read b =
    CGB.GDNLife.logits s read b + η * CGB.GDNLife.innovation s written target b * (1 + ∑ i, written i * read i)
```

## CGB.GDNLife.head_update_improves_saved_feature

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1082)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入保存特征向量，经局部仿射头参数单步更新，输出保存特征上的分类负对数似然损失严格单调下降。

源码说明：Strict improvement on the saved feature, not a future-language claim.

形式陈述：

```lean
∀ {k d : ℕ} (η : ℝ),
  0 < η →
    ∀ (s : CGB.GDNLife.State k d) (f : Fin d → ℝ) (target : CGB.GDNLife.Byte),
      CGB.Medium.softmax (CGB.GDNLife.logits s f) target <
        CGB.Medium.softmax (CGB.GDNLife.logits (CGB.GDNLife.headUpdate η s f target) f) target
```

## CGB.GDNLife.carrier_next

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1121)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入GDN生命载体模型、初始正状态与策略响应映射，经单步推进算子，输出可再生语言后继步进精确等同于标准GDN单步更新算子。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} (M : CGB.GDNLife.Model k d) (s₀ : CGB.GDNLife.State k d) (h₀ : CGB.GDNLife.Positive s₀)
  (policy : CGB.GDNLife.State k d → Act) (response : CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival)
  (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival),
  CGB.SelfProduction.renewableLanguageNext (CGB.GDNLife.carrier M s₀ h₀ policy response) s a = CGB.GDNLife.step M s a
```

## CGB.GDNLife.carrier_fold

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1129)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入实际输入历史序列，经实状态载体连续折叠，输出历史序列折叠轨迹不依赖离散状态替换，直接在连续GDN状态上执行。

源码说明：Actual input histories fold through this same real body, with no reset.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} (M : CGB.GDNLife.Model k d) (s₀ : CGB.GDNLife.State k d) (h₀ : CGB.GDNLife.Positive s₀)
  (policy : CGB.GDNLife.State k d → Act) (response : CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival)
  (inputs : List CGB.GDNLife.Arrival),
  List.foldl (CGB.SelfProduction.renewableLanguageNext (CGB.GDNLife.carrier M s₀ h₀ policy response)) s₀ inputs =
    List.foldl (CGB.GDNLife.step M) s₀ inputs
```

## CGB.GDNLife.exists_open_real_gdn_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1154)

分类：witness；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDNLife 显式开放真实轨迹见证。

解释：输入连续GDN动力学模型，经开流轨迹存在性定理，输出存在一条连续可存活且产生正信息的真实无限流轨迹。

源码说明：An actual continuous GDN, not a finite surrogate, has one infinite open numerical trajectory. Future benefit, active maintenance and the full life definition are not conclusions of positive precision.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (M : CGB.GDNLife.Model k d) (s₀ : CGB.GDNLife.State k d),
  CGB.GDNLife.Positive s₀ →
    ∀ (policy : CGB.GDNLife.State k d → Act) (response : CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival) (w₀ : World)
      (boundary : World → CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival)
      (evolve : World → CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival → World),
      ∃ x,
        x 0 = (s₀, w₀) ∧
          (∀ (n : ℕ),
              x (n + 1) =
                (CGB.GDNLife.step M (x n).1 (boundary (x n).2 (x n).1 (policy (x n).1)),
                  evolve (x n).2 (x n).1 (policy (x n).1) (boundary (x n).2 (x n).1 (policy (x n).1)))) ∧
            ∀ (n : ℕ), CGB.GDNLife.Positive (x n).1
```

## CGB.GDNLife.finish_saved_logits

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1181)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入保存的特征向量与局部头更新，经仿射映射分解，输出吸收后的局部更新不改变基于保存特征的中间对数几率。

源码说明：Assimilation does not change the head. Thus its local update is exactly on the prediction-time feature, despite the changed memory underneath it.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (p : CGB.GDNLife.Prepared k d) (a : CGB.GDNLife.Arrival),
  CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature =
    CGB.GDNLife.logits (CGB.GDNLife.headUpdate M.headStep p.state p.feature a.byte) p.feature
```

## CGB.GDNLife.byte_softmax_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1191)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入任意实对数几率向量与字节目标，经Softmax指数归一化，输出归一化字节预测概率严格大于零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (z : CGB.GDNLife.Byte → ℝ) (b : CGB.GDNLife.Byte), 0 < CGB.Medium.softmax z b
```

## CGB.GDNLife.finish_saved_nll_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1198)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入保存特征与到达字节目标，经梯度下降步进分析，输出基于保存特征的当前字节负对数似然损失在正学习率下严格单调下降。

源码说明：The saved-feature current-byte loss strictly decreases at any positive head step. No assumption about the next byte is hidden in this result.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d),
  0 < M.headStep →
    ∀ (p : CGB.GDNLife.Prepared k d) (a : CGB.GDNLife.Arrival),
      CGB.GDNLife.nll (CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature) a.byte <
        CGB.GDNLife.nll (CGB.GDNLife.logits p.state p.feature) a.byte
```

## CGB.GDNLife.future_ce_improves_of_target_mass

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1211)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入目标字符分布具有正概率质量的条件，经期望梯度分解，输出分布级条件能够保证未来交叉熵严格改进。

源码说明：A distributional, not in-sample, sufficient condition. A known target loss gain beats all possible other-byte loss increases when the *same* future law assigns enough mass to the target.

形式陈述：

```lean
∀ (law old new : CGB.GDNLife.Byte → ℝ) (target : CGB.GDNLife.Byte) (gain bound : ℝ),
  (∀ (b : CGB.GDNLife.Byte), 0 ≤ law b) →
    ∑ b, law b = 1 →
      CGB.GDNLife.nll new target - CGB.GDNLife.nll old target ≤ -gain →
        (∀ (b : CGB.GDNLife.Byte), b ≠ target → CGB.GDNLife.nll new b - CGB.GDNLife.nll old b ≤ bound) →
          bound < law target * (bound + gain) →
            CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax new) <
              CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax old)
```

## CGB.GDNLife.finish_future_ce_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1240)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入局部头更新与未来转移律，经交叉熵求导分析，输出具体头更新能够严格降低系统在未来语言律下的预测交叉熵。

源码说明：A concrete head update can improve a declared future law even after the memory/query changed: the saved-feature gain must exceed the readout drift, and the law must be sufficiently persistent. These are numeric/world conditions, not an assumption that future CE already improved.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (p : CGB.GDNLife.Prepared k d) (a : CGB.GDNLife.Arrival)
  (law : CGB.GDNLife.Byte → ℝ) (δ κ : ℝ),
  (∀ (b : CGB.GDNLife.Byte), 0 ≤ law b) →
    ∑ b, law b = 1 →
      (∀ (b : CGB.GDNLife.Byte),
          |CGB.GDNLife.logits (CGB.GDNLife.finish M p a) (CGB.GDNLife.feature M (CGB.GDNLife.finish M p a)) b -
                CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature b| ≤
            δ) →
        (∀ (b : CGB.GDNLife.Byte),
            |CGB.GDNLife.logits (CGB.GDNLife.finish M p a) (CGB.GDNLife.feature M (CGB.GDNLife.finish M p a)) b -
                  CGB.GDNLife.logits p.state p.feature b| ≤
              κ) →
          2 * κ <
              law a.byte *
                (2 * κ +
                  (CGB.GDNLife.nll (CGB.GDNLife.logits p.state p.feature) a.byte -
                      CGB.GDNLife.nll (CGB.GDNLife.logits (CGB.GDNLife.finish M p a) p.feature) a.byte -
                    2 * δ)) →
            CGB.PredictiveHarvest.crossEntropy law (CGB.GDNLife.allocation M (CGB.GDNLife.finish M p a)) <
              CGB.PredictiveHarvest.crossEntropy law (CGB.Medium.softmax (CGB.GDNLife.logits p.state p.feature))
```

## CGB.GDNLife.real_bias_injective

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1271)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入实数头偏置族，经实坐标嵌入检验，输出实偏置参数族在输出层具备单射可辨识性。

源码说明：The real family already embeds a real coordinate in an existing head bias. No finite body-state enumeration is being smuggled into this carrier.

形式陈述：

```lean
∀ {k d : ℕ}, Function.Injective fun r => CGB.GDNLife.initial 1 (fun x x_1 => 0) fun x => r
```

## CGB.GDNLife.prior_logits

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1276)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入GDN状态、经过时间与特征向量，经仿射头计算，输出时间先验演化不改变由当前特征生成的输出对数几率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) (s : CGB.GDNLife.State k d) (elapsed : ℝ) (f : Fin d → ℝ),
  CGB.GDNLife.logits (CGB.GDNLife.prior M s elapsed) f = CGB.GDNLife.logits s f
```

## CGB.GDNLife.step_ne_of_positive_head_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1285)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入正的真实读出头学习步长，经状态差异比较，输出正头学习步使本体状态在每个到达输入下均发生真实物理改变。

源码说明：A positive real head step changes the body on every arriving byte. This alone is not evidence of useful learning; unpredictable input also moves parameters. Finite-precision stalling is a separate numerical issue.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d),
  0 < M.headStep → ∀ (s : CGB.GDNLife.State k d) (a : CGB.GDNLife.Arrival), CGB.GDNLife.step M s a ≠ s
```

## CGB.GDNLife.Process.run_initial

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1338)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入离散运行过程，经初值提取，输出第0步联合状态精确等于本体与环境的初始状态元组。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World),
  P.run 0 = (P.initial, P.worldInitial)
```

## CGB.GDNLife.Process.run_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1340)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入GDN过程与当前时间步，经联合转移核步进，输出第n+1步联合状态精确等于本体更新与环境演化的直和推进。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  P.run (n + 1) =
    (CGB.GDNLife.step P.model (P.run n).1 (P.arrival n),
      P.evolve (P.run n).2 (P.run n).1 (P.policy (P.run n).1) (P.arrival n))
```

## CGB.GDNLife.Process.run_positive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1348)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入正定初值的GDN离散过程与任意时间步，经数学归纳法推演，输出全轨迹上各时刻本体状态的精度恒严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  CGB.GDNLife.Positive (P.run n).1
```

## CGB.GDNLife.Process.run_body_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1352)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入离散物理过程模型与时间步，经多步迭代推进，输出第n+1步的本体状态精确由前一步快照与到达包经finish算子生成。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  (P.run (n + 1)).1 = CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n)
```

## CGB.GDNLife.Process.self_speech_action_contrast_writes_body

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1361)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入发出不同自生语言字节的两个候选动作，经因果回馈写入，输出不同自生言语动作在本体中写入可区分的物理内部状态。

源码说明：Two candidate language actions that emit different live self-produced bytes write distinct next bodies through the ordinary GDN boundary transition. The byte is the action's existing bodily consequence, not a separately stored action record.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ) (a₀ a₁ : Act),
  (P.boundary (P.run n).2 (P.run n).1 a₀).source = CGB.GDNLife.Source.self →
    (P.boundary (P.run n).2 (P.run n).1 a₁).source = CGB.GDNLife.Source.self →
      0 < (P.boundary (P.run n).2 (P.run n).1 a₀).observationPrecision →
        0 < (P.boundary (P.run n).2 (P.run n).1 a₁).observationPrecision →
          (P.boundary (P.run n).2 (P.run n).1 a₀).byte ≠ (P.boundary (P.run n).2 (P.run n).1 a₁).byte →
            CGB.GDNLife.step P.model (P.run n).1 (P.boundary (P.run n).2 (P.run n).1 a₀) ≠
              CGB.GDNLife.step P.model (P.run n).1 (P.boundary (P.run n).2 (P.run n).1 a₁)
```

## CGB.GDNLife.Process.next_byte_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1377)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入外部世界真实生成的下一字节，经自回归吸收步进，输出更新对世界真实下一字节的预测对数似然严格单调提升。

源码说明：Repetition refers to the world's actual next byte, not a byte chosen for rescoring. The readout-drift budget includes the next physical-time prior.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ) (δ : ℝ),
  (P.arrival (n + 1)).byte = (P.arrival n).byte →
    (∀ (b : CGB.GDNLife.Byte),
        |P.scores (n + 1) b -
              CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature b| ≤
          δ) →
      2 * δ <
          CGB.GDNLife.nll (P.scores n) (P.arrival n).byte -
            CGB.GDNLife.nll
              (CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n)) (P.snapshot n).feature)
              (P.arrival n).byte →
        CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
          CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte
```

## CGB.GDNLife.Process.externallyUseful_improves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1411)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入有用外部事件凭据，经因果响应分析，输出有用事件的发生能严格改善系统的外周适应度与预测增益。

源码说明：The useful-event certificate invokes the earlier causal result; it does not inspect the next byte while preparing the current prediction.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  P.externallyUseful n →
    CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte < CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte
```

## CGB.GDNLife.Process.self_not_externallyUseful

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1420)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入自身复制发出的语言字节，经信息论有用性判据，输出复制自身的自生字节绝不可能凭空转化为有价值的外部独立环境证据。

源码说明：A copied self byte never becomes external evidence just by passing through the same learning rule. It can still affect the body through `step`.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (n : ℕ),
  (P.arrival n).source = CGB.GDNLife.Source.self → ¬P.externallyUseful n
```

## CGB.GDNLife.Process.externalInformation_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1455)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入外部语言转移律，经互信息非负性检验，输出系统外部语言条件互信息恒非负。

源码说明：A valid external language law has nonnegative information opportunity. This is finite conditional mutual information, rather than an arbitrary natural-number intake signal.

形式陈述：

```lean
∀ (joint : CGB.GDNLife.Byte → CGB.GDNLife.Byte → ℝ),
  (∀ (now later : CGB.GDNLife.Byte), 0 ≤ joint now later) → 0 ≤ CGB.GDNLife.Process.externalInformation joint
```

## CGB.GDNLife.Process.ExternalLanguageLaw.information_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1464)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入状态与动作索引的机会测度，经条件互信息非负性继承，输出各状态动作条件下的信息增益测度恒非负。

源码说明：The state/action-indexed opportunity inherits CMI nonnegativity from the registered external language law.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} {P : CGB.GDNLife.Process k d Act World} (L : P.ExternalLanguageLaw)
  (s : CGB.GDNLife.State k d) (a : Act), 0 ≤ L.information s a
```

## CGB.GDNLife.Process.future_reserve_preserved_keeps_signature_distinction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1503)

分类：design；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：针对 GDNLife 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：输入保留的未来语言储备，经特征签名映射，输出保持的未来储备保证不同语言签名在演化中不可默默合并。

源码说明：A preserved future-language reserve cannot silently merge two histories that a registered external continuation could previously distinguish. This is the long-delay retention condition for any causal difference already carried by the body; it adds neither an action log nor a separate eligibility state.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m n : ℕ} (Q : CGB.GDNLife.Process.ExternalFutureProbes m)
  (before after : Fin n → CGB.GDNLife.State k d) (i j : Fin n),
  CGB.GDNLife.Process.futureReservePreserved M Q before after →
    CGB.GDNLife.Process.futureLanguageSignature M Q (before i) ≠
        CGB.GDNLife.Process.futureLanguageSignature M Q (before j) →
      CGB.GDNLife.Process.futureLanguageSignature M Q (after i) ≠
        CGB.GDNLife.Process.futureLanguageSignature M Q (after j)
```

## CGB.GDNLife.Process.future_reserve_recoverable_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1546)

分类：necessity；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDN 动力学中未来储备保持与循环外部证据迫使主动性的充要约束。

解释：输入未来状态区分，经耗散测度分析，输出未来区分度可无损恢复当且仅当该过程的所有离散签名均未发生耗散。

源码说明：A future distinction is not dissipated exactly when all of the prior future language signature remains readable from the post-update signature. This reuses the existing causal-carrier theorem instead of introducing a separate retention mechanism.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m n : ℕ} (Q : CGB.GDNLife.Process.ExternalFutureProbes m)
  (before after : Fin n → CGB.GDNLife.State k d),
  (∃ restore,
      ∀ (i : Fin n),
        restore (CGB.GDNLife.Process.futureLanguageSignature M Q (after i)) =
          CGB.GDNLife.Process.futureLanguageSignature M Q (before i)) ↔
    CGB.GDNLife.Process.futureReservePreserved M Q before after
```

## CGB.GDNLife.Process.loses_future_language_distinction_iff_not_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1561)

分类：necessity；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDN 动力学中未来储备保持与循环外部证据迫使主动性的充要约束。

解释：输入两先前可区分的未来语言签名，经等价性证明，输出二者合并丧失区分度当且仅当未来储备未被有效保留。

源码说明：Merging a prior distinct future-language signature is precisely the failure of that causal retention contract.

形式陈述：

```lean
∀ {k d : ℕ} (M : CGB.GDNLife.Model k d) {m n : ℕ} (Q : CGB.GDNLife.Process.ExternalFutureProbes m)
  (before after : Fin n → CGB.GDNLife.State k d),
  CGB.GDNLife.Process.losesFutureLanguageDistinction M Q before after ↔
    ¬CGB.GDNLife.Process.futureReservePreserved M Q before after
```

## CGB.GDNLife.Process.recurrent_external_information_forces_activity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1584)

分类：necessity；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDN 动力学中未来储备保持与循环外部证据迫使主动性的充要约束。

解释：输入循环外部信息流与GDN物理模型，经信息论主动必要性定理，输出维持循环正信息摄入严格迫使系统必须维持主动干预策略。

源码说明：Information-theoretic active necessity for the concrete GDN port. If settling into `passive` eliminates the conditional information available from the external language boundary, while positive external information recurs on the realised trajectory, the policy cannot settle into `passive`. The premise is deliberately a boundary statement, not a synthetic hunger counter. To turn available information into a learning claim, it must also meet the causal `externallyUseful` scoring certificate above.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (L : P.ExternalLanguageLaw)
  (passive : Act),
  (CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive →
      ∃ N, ∀ (n : ℕ), N ≤ n → L.information (P.run n).1 (P.policy (P.run n).1) = 0) →
    (∀ (N : ℕ),
        ∃ n,
          N ≤ n ∧
            (P.arrival n).source = CGB.GDNLife.Source.external ∧ 0 < L.information (P.run n).1 (P.policy (P.run n).1)) →
      ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive
```

## CGB.GDNLife.Process.recurrent_external_evidence_forces_activity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1603)

分类：necessity；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDN 动力学中未来储备保持与循环外部证据迫使主动性的充要约束。

解释：输入循环出现的外部证据序列与生存约束，经反证法推导，输出若策略最终退化为被动停滞，则无法循环摄取生存所需外部证据。

源码说明：If settling into a passive policy rules out recurrent externally useful language, but such events actually recur on this same trajectory, the policy cannot settle. This is the language form of active necessity: no arbitrary `intake`, opportunity counter, food pool, or maintenance currency is added.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World) (passive : Act),
  (CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive →
      ∃ N, ∀ (n : ℕ), N ≤ n → ¬P.externallyUseful n) →
    (∀ (N : ℕ), ∃ n, N ≤ n ∧ P.externallyUseful n) →
      ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive
```

## CGB.GDNLife.Process.scored_external_language_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1621)

分类：witness；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDNLife 显式开放真实轨迹见证。

解释：输入实数GDN模型与外部语言流，经打分联合轨迹判定，输出存在满足条件语言级打分联合轨迹，且每步维持有效正预测增益。

源码说明：Conditional language-level joint result for the real GDN. Every condition is about the actual language trajectory: source provenance, recurrent post-update improvement, the causal effect of action on the next arrival, and a non-uniform prequential margin. The result intentionally says nothing about physical food, material ingestion, or an auxiliary reserve state. It is still not a full `Medium.Lives` construction.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World),
  0 < P.model.headStep →
    ∀ (passive : Act),
      (CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive →
          ∃ N, ∀ (n : ℕ), N ≤ n → ¬P.externallyUseful n) →
        (∀ (N : ℕ), ∃ n, N ≤ n ∧ P.externallyUseful n) →
          (∃ n a a', (P.boundary (P.run n).2 (P.run n).1 a).byte ≠ (P.boundary (P.run n).2 (P.run n).1 a').byte) →
            (∀ (n : ℕ), CGB.GDNLife.nll (P.scores n) (P.arrival n).byte < Real.log 256) →
              (∀ (n : ℕ), CGB.GDNLife.Positive (P.run n).1) ∧
                ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive ∧
                  (¬∃ predict,
                        ∀ (n : ℕ) (a : Act),
                          predict (P.run n).2 (P.run n).1 = (P.boundary (P.run n).2 (P.run n).1 a).byte) ∧
                    (∀ (n : ℕ), 0 < Real.log 256 - CGB.GDNLife.nll (P.scores n) (P.arrival n).byte) ∧
                      (∀ (N : ℕ),
                          ∃ n,
                            N ≤ n ∧
                              CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
                                CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte) ∧
                        ¬∃ N, ∀ (n : ℕ), N ≤ n → (P.run (n + 1)).1 = (P.run n).1
```

## CGB.GDNLife.Process.scored_joint_trajectory

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1663)

分类：witness；适用范围：candidate；审核：reviewed；候选：GDNLife

分类说明：GDNLife 显式开放真实轨迹见证。

解释：输入连续实数GDN生命过程，经开流联合状态演化，输出全时域联合轨迹在无截断下严格满足条件打分与存活标准。

源码说明：A conditional scored joint trajectory for this real GDN. Conditions are on the actual world: registered intake, a finite passive opportunity budget under the same intervenable transition, an action contrast, a nonnegative predictive margin, and recurrent next-byte persistence with bounded readout drift. Recurrent future score gain is *derived*, not supplied as `hlearn`. Numeric positivity is still not material/biological viability. This deterministic-world statement scores actual future bytes, not a claim about the conditional law of an unspecified stochastic language world.

形式陈述：

```lean
∀ {k d : ℕ} {Act : Type u_1} {World : Type u_2} (P : CGB.GDNLife.Process k d Act World),
  0 < P.model.headStep →
    ∀ (intake : World → CGB.GDNLife.State k d → Act → CGB.GDNLife.Arrival → ℕ) (passive : Act),
      CGB.ActiveInference.unbounded (P.signal intake) →
        ∀ (opportunity : CGB.GDNLife.State k d × World → ℕ),
          (∀ (z : CGB.GDNLife.State k d × World),
              opportunity (P.contract.step z (P.contract.response z passive)) +
                  intake z.2 z.1 passive (P.boundary z.2 z.1 passive) ≤
                opportunity z) →
            (∃ n a a', (P.boundary (P.run n).2 (P.run n).1 a).byte ≠ (P.boundary (P.run n).2 (P.run n).1 a').byte) →
              ∀ (m : ℝ),
                0 ≤ m →
                  (∀ (n : ℕ), CGB.GDNLife.nll (P.scores n) (P.arrival n).byte < Real.log 256 - m) →
                    (∀ (N : ℕ),
                        ∃ n,
                          N ≤ n ∧
                            ∃ δ,
                              (P.arrival (n + 1)).byte = (P.arrival n).byte ∧
                                (∀ (b : CGB.GDNLife.Byte),
                                    |P.scores (n + 1) b -
                                          CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n))
                                            (P.snapshot n).feature b| ≤
                                      δ) ∧
                                  2 * δ <
                                    CGB.GDNLife.nll (P.scores n) (P.arrival n).byte -
                                      CGB.GDNLife.nll
                                        (CGB.GDNLife.logits (CGB.GDNLife.finish P.model (P.snapshot n) (P.arrival n))
                                          (P.snapshot n).feature)
                                        (P.arrival n).byte) →
                      (∀ (n : ℕ), CGB.GDNLife.Positive (P.run n).1) ∧
                        ¬CGB.ActiveInference.eventuallyPassive (fun n => P.policy (P.run n).1) passive ∧
                          (¬∃ predict,
                                ∀ (n : ℕ) (a : Act),
                                  predict (P.run n).2 (P.run n).1 = (P.boundary (P.run n).2 (P.run n).1 a).byte) ∧
                            (∀ (n : ℕ),
                                0 < Real.log 256 - CGB.GDNLife.nll (P.scores n) (P.arrival n).byte ∧
                                  0 < Real.log 256 - CGB.GDNLife.nll (P.scores n) (P.arrival n).byte - m) ∧
                              (∀ (N : ℕ),
                                  ∃ n,
                                    N ≤ n ∧
                                      CGB.GDNLife.nll (P.scores (n + 1)) (P.arrival (n + 1)).byte <
                                        CGB.GDNLife.nll (P.scores n) (P.arrival (n + 1)).byte) ∧
                                ¬∃ N, ∀ (n : ℕ), N ≤ n → (P.run (n + 1)).1 = (P.run n).1
```
