# GenerationNonlinearity

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean) · [返回导航](../README.md)

模块源码说明：Nonlinearity is necessary for language (generation chain, 2026-11-02) Language is full of interactions: what follows `th` is not what `t` predicts plus what `h` predicts.

## CGB.Generation.Nonlinearity.additive_no_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L51)

分类：necessity；适用范围：universal；审核：draft

分类说明：两个来源分开作用的映射，交互对比恒为零。

解释：A map that treats its sources separately has no interaction.

源码说明：A map that treats its sources separately has no interaction.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {f : A → B → ℝ},
  CGB.Generation.Nonlinearity.Additive f → ∀ (a a' : A) (b b' : B), CGB.Generation.Nonlinearity.contrast f a a' b b' = 0
```

## CGB.Generation.Nonlinearity.interaction_needs_nonadditive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L58)

分类：necessity；适用范围：universal；审核：draft

分类说明：目标有非零交互对比时，任何可加映射都不能精确表示它。

解释：A target with an interaction is not additive.

源码说明：A target with an interaction is not additive.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {f : A → B → ℝ} {a a' : A} {b b' : B},
  CGB.Generation.Nonlinearity.contrast f a a' b b' ≠ 0 → ¬CGB.Generation.Nonlinearity.Additive f
```

## CGB.Generation.Nonlinearity.Additive.sub

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L63)

分类：semantics；适用范围：universal；审核：draft

分类说明：可加映射之差可加，所以可加 logit 的对数几率仍可加。

解释：The difference of two additive maps is additive, so log-odds of additive logits are.

源码说明：The difference of two additive maps is additive, so log-odds of additive logits are.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {f g : A → B → ℝ},
  CGB.Generation.Nonlinearity.Additive f →
    CGB.Generation.Nonlinearity.Additive g → CGB.Generation.Nonlinearity.Additive fun a b => f a b - g a b
```

## CGB.Generation.Nonlinearity.linear_read_of_superposed_state

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L76)

分类：necessity；适用范围：universal；审核：draft

分类说明：任意宽度的线性读出作用在叠加状态上仍可加。

解释：A linear read of a superposed state is additive at any width.

源码说明：A linear read of a superposed state is additive at any width.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {E : Type u_3} [inst : AddCommGroup E] [inst_1 : _root_.Module ℝ E] (p : A → E)
  (r : B → E) (read : E →ₗ[ℝ] ℝ) (c : ℝ), CGB.Generation.Nonlinearity.Additive fun a b => c + read (p a + r b)
```

## CGB.Generation.Nonlinearity.linear_stack_superposes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L81)

分类：necessity；适用范围：universal；审核：draft

分类说明：任意深度的线性层加偏置保持两个来源叠加。

解释：Any stack of linear layers with biases keeps the two sources superposed.

源码说明：Any stack of linear layers with biases keeps the two sources superposed.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {E : Type u_3} [inst : AddCommGroup E] [inst_1 : _root_.Module ℝ E]
  (layers : List ((E →ₗ[ℝ] E) × E)) (p : A → E) (r : B → E),
  ∃ p' r', ∀ (a : A) (b : B), List.foldr (fun layer x => layer.1 x + layer.2) (p a + r b) layers = p' a + r' b
```

## CGB.Generation.Nonlinearity.linear_stack_no_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L93)

分类：necessity；适用范围：universal；审核：draft

分类说明：深度不解决问题：线性栈的线性读出没有交互。

解释：Depth does not help: a linear read of a linear stack has no interaction.

源码说明：Depth does not help: a linear read of a linear stack has no interaction.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {E : Type u_3} [inst : AddCommGroup E] [inst_1 : _root_.Module ℝ E]
  (layers : List ((E →ₗ[ℝ] E) × E)) (p : A → E) (r : B → E) (read : E →ₗ[ℝ] ℝ) (c : ℝ),
  CGB.Generation.Nonlinearity.Additive fun a b =>
    c + read (List.foldr (fun layer x => layer.1 x + layer.2) (p a + r b) layers)
```

## CGB.Generation.Nonlinearity.linear_recurrence_no_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L104)

分类：necessity；适用范围：universal；审核：draft

分类说明：线性递推记忆加线性读出，对两个过去到达可加；嵌入是任意表也一样。

解释：A linear recurrent memory with a linear read is additive in two past arrivals. The embeddings `W₁, W₂` are arbitrary tables, so a learned embedding does not help.

源码说明：A linear recurrent memory with a linear read is additive in two past arrivals. The embeddings `W₁, W₂` are arbitrary tables, so a learned embedding does not help.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {E : Type u_3} [inst : AddCommGroup E] [inst_1 : _root_.Module ℝ E] (L : E →ₗ[ℝ] E)
  (W₁ : A → E) (W₂ : B → E) (x₀ : E) (read : E →ₗ[ℝ] ℝ) (c : ℝ),
  CGB.Generation.Nonlinearity.Additive fun a b => c + read (L (L x₀ + W₁ a) + W₂ b)
```

## CGB.Generation.Nonlinearity.additive_logit_cannot_separate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L114)

分类：necessity；适用范围：universal；审核：draft

分类说明：末端 softmax 不解决问题：可加对数几率取不到 XOR 的符号模式。

解释：A final squashing does not help: additive log-odds cannot take the XOR sign pattern.

源码说明：A final squashing does not help: additive log-odds cannot take the XOR sign pattern.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {f : A → B → ℝ},
  CGB.Generation.Nonlinearity.Additive f →
    ∀ {a a' : A} {b b' : B}, 0 < f a b → 0 < f a' b' → f a b' < 0 → f a' b < 0 → False
```

## CGB.Generation.Nonlinearity.additive_error_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L122)

分类：necessity；适用范围：universal；审核：draft

分类说明：可加预测器在四种情形上的平方误差和不低于对比平方的四分之一，与规模无关。

解释：The squared error of any additive predictor over the four cases is at least a quarter of the squared contrast. The bound does not mention the predictor's size.

源码说明：The squared error of any additive predictor over the four cases is at least a quarter of the squared contrast. The bound does not mention the predictor's size.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} (f : A → B → ℝ) {g : A → B → ℝ},
  CGB.Generation.Nonlinearity.Additive g →
    ∀ (a a' : A) (b b' : B),
      CGB.Generation.Nonlinearity.contrast f a a' b b' ^ 2 / 4 ≤
        (f a b - g a b) ^ 2 + (f a' b' - g a' b') ^ 2 + (f a b' - g a b') ^ 2 + (f a' b - g a' b) ^ 2
```

## CGB.Generation.Nonlinearity.association_read_additive_in_contents

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L142)

分类：necessity；适用范围：candidate；审核：draft；候选：Nonlinearity

分类说明：固定查询时 GDN 式读出对存储内容可加：检索只加权，不组合。

解释：At a fixed query, a GDN-style read is additive in what is stored: retrieval weighs the contents and cannot combine them.

源码说明：At a fixed query, a GDN-style read is additive in what is stored: retrieval weighs the contents and cannot combine them.

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {dv dk : ℕ} (k : Fin 2 → Fin dk → ℝ) (q : Fin dk → ℝ) (o : Fin dv)
  (first : A → Fin dv → ℝ) (second : B → Fin dv → ℝ),
  CGB.Generation.Nonlinearity.Additive fun a b =>
    CGB.Generation.associationRead (CGB.Generation.associationState k ![first a, second b]) q o
```

## CGB.Generation.Nonlinearity.key_query_read_interacts

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L154)

分类：witness；适用范围：candidate；审核：draft；候选：Nonlinearity

分类说明：一个到达给键、另一个给查询时读出有交互，所以学到的地址本身就是非线性。

解释：A key from one arrival read with a query from the other interacts, so a learned address already supplies the nonlinearity.

源码说明：A key from one arrival read with a query from the other interacts, so a learned address already supplies the nonlinearity.

形式陈述：

```lean
CGB.Generation.Nonlinearity.contrast
    (fun a b =>
      CGB.Generation.associationRead (CGB.Generation.associationState (fun x x_1 => a) fun x x_1 => 1) (fun x => b) 0)
    1 (-1) 1 (-1) =
  4
```

## CGB.Generation.Nonlinearity.one_rectifier_reads_interaction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L167)

分类：witness；适用范围：universal；审核：draft

分类说明：XOR 有交互，一个整流单元就够读出，说明必要性是紧的。

解释：XOR has an interaction, and one rectifier is enough to read it.

源码说明：XOR has an interaction, and one rectifier is enough to read it.

形式陈述：

```lean
CGB.Generation.Nonlinearity.contrast (fun a b => CGB.Generation.Nonlinearity.bit (a ^^ b)) false true false true ≠ 0 ∧
  ∀ (a b : Bool),
    CGB.Generation.Nonlinearity.bit a + CGB.Generation.Nonlinearity.bit b -
        2 * max (CGB.Generation.Nonlinearity.bit a + CGB.Generation.Nonlinearity.bit b - 1) 0 =
      CGB.Generation.Nonlinearity.bit (a ^^ b)
```
