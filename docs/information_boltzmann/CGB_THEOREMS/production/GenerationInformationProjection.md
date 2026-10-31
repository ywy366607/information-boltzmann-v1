# GenerationInformationProjection

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean) · [返回导航](../README.md)

模块源码说明：Information under finite deterministic observation Data processing for `Medium.condMutualInfo`, including sparse laws and arbitrary nonnegative total mass.

## CGB.InformationProjection.gibbs_term_supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L17)

源码说明：Gibbs' pointwise bound on compatible, possibly zero, supports.

形式陈述：

```lean
∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → (0 < x → 0 < y) → x * (Real.log y - Real.log x) ≤ y - x
```

## CGB.InformationProjection.log_sum_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L25)

源码说明：Finite log-sum, with zero entries allowed and no normalization premise.

形式陈述：

```lean
∀ {ι : Type u_1} [inst : Fintype ι] (x y : ι → ℝ),
  (∀ (i : ι), 0 ≤ x i) →
    (∀ (i : ι), 0 ≤ y i) →
      (∀ (i : ι), 0 < x i → 0 < y i) →
        (∑ i, x i) * (Real.log (∑ i, x i) - Real.log (∑ i, y i)) ≤ ∑ i, x i * (Real.log (x i) - Real.log (y i))
```

## CGB.InformationProjection.kl_nonneg_supported

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L71)

源码说明：The existing relative entropy is nonnegative for normalized compatible supports. The strictly positive case reuses `InformationEnergy.kl_nonneg` unchanged.

形式陈述：

```lean
∀ {ι : Type u_1} [inst : Fintype ι] (x y : ι → ℝ),
  (∀ (i : ι), 0 ≤ x i) →
    (∀ (i : ι), 0 ≤ y i) →
      (∀ (i : ι), 0 < x i → 0 < y i) → ∑ i, x i = 1 → ∑ i, y i = 1 → 0 ≤ CGB.InformationEnergy.kl x y
```

## CGB.InformationProjection.kl_pos_of_support_gap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L82)

源码说明：A support gap makes relative entropy strictly positive for compatible nonnegative weights of equal mass. No normalization is needed.

形式陈述：

```lean
∀ {ι : Type u_1} [inst : Fintype ι] (x y : ι → ℝ),
  (∀ (i : ι), 0 ≤ x i) →
    (∀ (i : ι), 0 ≤ y i) →
      (∀ (i : ι), 0 < x i → 0 < y i) → ∑ i, x i = ∑ i, y i → (∃ i, x i = 0 ∧ 0 < y i) → 0 < CGB.InformationEnergy.kl x y
```

## CGB.InformationProjection.project_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L108)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {γ : Type u_1} {δ : Type u_2} [inst : Fintype γ] (f : γ → δ) (p : γ → ℝ),
  (∀ (c : γ), 0 ≤ p c) → ∀ (d : δ), 0 ≤ CGB.InformationProjection.project f p d
```

## CGB.InformationProjection.sum_project

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L114)

源码说明：Observation preserves total mass, including zero total mass.

形式陈述：

```lean
∀ {γ : Type u_1} {δ : Type u_2} [inst : Fintype γ] [inst_1 : Fintype δ] (f : γ → δ) (p : γ → ℝ),
  ∑ d, CGB.InformationProjection.project f p d = ∑ c, p c
```

## CGB.InformationProjection.kl_project_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L123)

源码说明：Relative entropy contracts under finite deterministic observation, without normalization, provided the reference is positive wherever the source is positive.

形式陈述：

```lean
∀ {γ : Type u_1} {δ : Type u_2} [inst : Fintype γ] [inst_1 : Fintype δ] (f : γ → δ) (x y : γ → ℝ),
  (∀ (c : γ), 0 ≤ x c) →
    (∀ (c : γ), 0 ≤ y c) →
      (∀ (c : γ), 0 < x c → 0 < y c) →
        CGB.InformationEnergy.kl (CGB.InformationProjection.project f x) (CGB.InformationProjection.project f y) ≤
          CGB.InformationEnergy.kl x y
```

## _private.CGBAxioms.GenerationInformationProjection.0.CGB.InformationProjection.marginals_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L151)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {β : Type u_1} {γ : Type u_2} [inst : Fintype β] [inst_1 : Fintype γ] (p : β → γ → ℝ),
  (∀ (b : β) (c : γ), 0 ≤ p b c) →
    ∀ (b : β) (c : γ), 0 < p b c → 0 < ∑ c', p b c' ∧ 0 < ∑ b', p b' c ∧ 0 < ∑ b', ∑ c', p b' c'
```

## CGB.InformationProjection.reference_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L163)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {β : Type u_1} {γ : Type u_2} [inst : Fintype β] [inst_1 : Fintype γ] (p : β → γ → ℝ),
  (∀ (b : β) (c : γ), 0 ≤ p b c) → ∀ (b : β) (c : γ), 0 ≤ CGB.InformationProjection.reference p b c
```

## CGB.InformationProjection.reference_pos_of_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L171)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {β : Type u_1} {γ : Type u_2} [inst : Fintype β] [inst_1 : Fintype γ] (p : β → γ → ℝ),
  (∀ (b : β) (c : γ), 0 ≤ p b c) → ∀ (b : β) (c : γ), 0 < p b c → 0 < CGB.InformationProjection.reference p b c
```

## CGB.InformationProjection.reference_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L179)

源码说明：The conditional-product reference has exactly the source's total mass, including when that mass is zero.

形式陈述：

```lean
∀ {β : Type u_1} {γ : Type u_2} [inst : Fintype β] [inst_1 : Fintype γ] (p : β → γ → ℝ),
  ∑ b, ∑ c, CGB.InformationProjection.reference p b c = ∑ b, ∑ c, p b c
```

## CGB.InformationProjection.condMutualInfo_eq_kl

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L192)

源码说明：CMI is a sum of relative entropies against the conditional-product reference. This identity handles empty slices and zero entries directly.

形式陈述：

```lean
∀ {α : Type u_1} {β : Type u_2} {γ : Type u_3} [inst : Fintype α] [inst_1 : Fintype β] [inst_2 : Fintype γ]
  (p : α → β → γ → ℝ),
  (∀ (a : α) (b : β) (c : γ), 0 ≤ p a b c) →
    CGB.Medium.condMutualInfo p =
      ∑ a, ∑ b, CGB.InformationEnergy.kl (p a b) (CGB.InformationProjection.reference (p a) b)
```

## CGB.InformationProjection.condMutualInfo_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L212)

源码说明：Nonnegative, unnormalized finite conditional mutual information.

形式陈述：

```lean
∀ {α : Type u_1} {β : Type u_2} {γ : Type u_3} [inst : Fintype α] [inst_1 : Fintype β] [inst_2 : Fintype γ]
  (p : α → β → γ → ℝ), (∀ (a : α) (b : β) (c : γ), 0 ≤ p a b c) → 0 ≤ CGB.Medium.condMutualInfo p
```

## CGB.InformationProjection.condMutualInfo_pos_of_missing_rectangle

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L229)

源码说明：A missing entry inside a positive marginal rectangle forces positive CMI. This applies to sparse, unnormalized physical laws without expanding their logs.

形式陈述：

```lean
∀ {α : Type u_1} {β : Type u_2} {γ : Type u_3} [inst : Fintype α] [inst_1 : Fintype β] [inst_2 : Fintype γ]
  (p : α → β → γ → ℝ),
  (∀ (a : α) (b : β) (c : γ), 0 ≤ p a b c) →
    (∃ a b c, p a b c = 0 ∧ 0 < ∑ c', p a b c' ∧ 0 < ∑ b', p a b' c) → 0 < CGB.Medium.condMutualInfo p
```

## _private.CGBAxioms.GenerationInformationProjection.0.CGB.InformationProjection.project_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L254)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {β : Type u_1} {γ : Type u_2} {δ : Type u_3} [inst : Fintype β] [inst_1 : Fintype γ] (f : γ → δ) (p : β → γ → ℝ)
  (d : δ),
  ∑ b, CGB.InformationProjection.project f (p b) d = CGB.InformationProjection.project f (fun c => ∑ b, p b c) d
```

## CGB.InformationProjection.reference_project

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L263)

源码说明：The conditional-product reference commutes with observing the third variable.

形式陈述：

```lean
∀ {β : Type u_1} {γ : Type u_2} {δ : Type u_3} [inst : Fintype β] [inst_1 : Fintype γ] [inst_2 : Fintype δ] (f : γ → δ)
  (p : β → γ → ℝ) (b : β) (d : δ),
  CGB.InformationProjection.reference (fun b => CGB.InformationProjection.project f (p b)) b d =
    CGB.InformationProjection.project f (CGB.InformationProjection.reference p b) d
```

## CGB.InformationProjection.condMutualInfo_project_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L282)

源码说明：**Finite deterministic CMI data processing**, including arbitrary sparse supports and unnormalized nonnegative weights. No strict-positivity or stationarity premise.

形式陈述：

```lean
∀ {α : Type u_1} {β : Type u_2} {γ : Type u_3} {δ : Type u_4} [inst : Fintype α] [inst_1 : Fintype β]
  [inst_2 : Fintype γ] [inst_3 : Fintype δ] (p : α → β → γ → ℝ) (f : γ → δ),
  (∀ (a : α) (b : β) (c : γ), 0 ≤ p a b c) →
    (CGB.Medium.condMutualInfo fun a b => CGB.InformationProjection.project f (p a b)) ≤ CGB.Medium.condMutualInfo p
```
