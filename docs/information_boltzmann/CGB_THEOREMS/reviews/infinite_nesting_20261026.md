# infinite_nesting_20261026

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean) · [返回导航](../README.md)

模块源码说明：Exploratory consequences of nesting, not a new infinite Medium instance.

## CGB.InfiniteNestingReview.finite_regions_stabilize

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L14)

解释：有限集合组成的递减嵌套序列最终稳定，不可能永远严格变小。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem finite_regions_stabilize {U : Type*} [DecidableEq U]
    (O : ℕ → Finset U) (hO : Antitone O) :
    ∃ N, ∀ n, N ≤ n → O n = O N
```

## CGB.InfiniteNestingReview.strict_region_depth

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L25)

解释：若前 n 层有限区域逐层严格缩小，则初始区域大小至少为 n 加第 n 层大小。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem strict_region_depth {U : Type*} [DecidableEq U]
    (O : ℕ → Finset U) (n : ℕ)
    (hO : ∀ k < n, O (k + 1) ⊂ O k) : n + (O n).card ≤ (O 0).card
```

## CGB.InfiniteNestingReview.independent_bits_bound

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L37)

解释：若 n 个独立布尔位可以无冲突编码到有限状态集，则状态数至少为 2^n。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem independent_bits_bound {S : Type*} [Fintype S] {n : ℕ}
    (encode : (Fin n → Bool) → S) (hi : Function.Injective encode) :
    2 ^ n ≤ Fintype.card S
```

## CGB.InfiniteNestingReview.no_unbounded_independent_bits

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L42)

解释：固定有限状态集不可能对任意大的独立布尔位数都提供无冲突编码。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem no_unbounded_independent_bits {S : Type*} [Fintype S] :
    ¬ (∀ n, ∃ encode : (Fin n → Bool) → S, Function.Injective encode)
```

## CGB.InfiniteNestingReview.depth_budget_bound

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L50)

解释：若前 n 层每层成本至少为 ε 且累计成本不超过 B，则 nε 不超过 B。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem depth_budget_bound (cost : ℕ → ℝ) (ε B : ℝ) (n : ℕ)
    (hlower : ∀ k < n, ε ≤ cost k) (hbudget : ∑ k ∈ range n, cost k ≤ B) :
    (n : ℝ) * ε ≤ B
```

## CGB.InfiniteNestingReview.no_uniform_cost_infinite_depth

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L58)

解释：若无限多层每层成本均不小于同一正数，则不存在共同有限上界容纳所有前缀成本。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem no_uniform_cost_infinite_depth (cost : ℕ → ℝ) {ε : ℝ} (he : 0 < ε)
    (hlower : ∀ k, ε ≤ cost k) : ¬ ∃ B, ∀ n, ∑ k ∈ range n, cost k ≤ B
```

## CGB.InfiniteNestingReview.geometric_prefix

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L68)

解释：前 n 项几何成本 Σ2^(−k−1) 恰等于 1−2^(−n)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem geometric_prefix (n : ℕ) :
    (∑ k ∈ range n, ((1 : ℝ) / 2) ^ (k + 1)) = 1 - ((1 : ℝ) / 2) ^ n
```

## CGB.InfiniteNestingReview.geometric_costs_fit

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L74)

解释：每层取正的几何递减成本 2^(−k−1) 时，任意有限前缀总成本都小于一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem geometric_costs_fit :
    (∀ k : ℕ, 0 < ((1 : ℝ) / 2) ^ (k + 1)) ∧
    (∀ n : ℕ, (∑ k ∈ range n, ((1 : ℝ) / 2) ^ (k + 1)) < 1)
```

## CGB.InfiniteNestingReview.freeze_twice

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L86)

解释：先冻结区域 O 外再冻结区域 P 外，等价于一次冻结交集 O∩P 外的转移。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem freeze_twice {U Λ : Type*} (K : CGB.JumpProcess (U → Λ)) (O P : Set U)
    (x y : U → Λ) :
    ((K.freezeOutside O).freezeOutside P).rate x y =
      (K.freezeOutside (O ∩ P)).rate x y
```

## CGB.InfiniteNestingReview.infinite_freeze_balance

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L104)

解释：原跳跃核满足指定细致平衡时，按无限层区域共同冻结得到的转移仍满足该平衡关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem infinite_freeze_balance {U Λ : Type*} (K : CGB.JumpProcess (U → Λ))
    (O : ℕ → Set U) {π : (U → Λ) → ℝ}
    (hdb : ∀ x y, π x * K.rate x y = π y * K.rate y x) (x y : U → Λ) :
    π x * (K.freezeOutside (⋂ n, O n)).rate x y =
      π y * (K.freezeOutside (⋂ n, O n)).rate y x
```

## CGB.InfiniteNestingReview.reversible_projection_not_closed

[源码位置](../../../../lean/CGBAxioms/reviews/infinite_nesting_20261026.lean#L113)

解释：存在对称可逆的三态速率过程，其布尔粗粒化不能由所要求的单一粗粒转移率闭合。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem reversible_projection_not_closed :
    ∃ (rate : Fin 3 → Fin 3 → ℝ) (view : Fin 3 → Bool),
      (∀ i j, 0 ≤ rate i j) ∧ (∀ i j, rate i j = rate j i) ∧
      ¬ ∃ coarse : Bool → Bool → ℝ, ∀ i b,
        (∑ j : Fin 3, if view j = b then rate i j else 0) = coarse (view i) b
```
