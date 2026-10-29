# GenerationLeaveRule

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean) · [返回导航](../README.md)

模块源码说明：When must a forager compare now with the past? (generation chain, C-1) A forager at a patch with local gain rate `q` either stays (earning `q`) or leaves for the rest of the world (worth `c`, the environment's average ra

## CGB.Generation.leaveValue_le

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L26)

解释：对任意离场决策阈值 τ，受约束觅食过程的期望收益均不超过即时收益与机会成本的上界 max(q, c)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (τ c q : ℝ), CGB.Generation.leaveValue τ c q ≤ max q c
```

## CGB.Generation.fixed_threshold_optimal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L30)

解释：当离开收益阈值恰好取当前收益常数 c 时，折现收益函数取得其理论极大值 max(q, c)。

源码说明：In a stationary world the fixed threshold `τ = c` is optimal everywhere.

形式陈述：

```lean
∀ (c q : ℝ), CGB.Generation.leaveValue c c q = max q c
```

## CGB.Generation.no_fixed_leave_threshold

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L38)

解释：在两个成本不同的环境中，不存在一个固定的离开阈值能够在两个环境中都实现最优折现收益，环境切换必然要求动态改变离场决策阈值。

源码说明：**C-1 (a changing world needs comparison)**: with two different averages `c₁ < c₂`, every fixed threshold `τ` loses strictly at some patch rate `q` between them, in one of the two worlds.

形式陈述：

```lean
∀ {c₁ c₂ : ℝ},
  c₁ < c₂ →
    ∀ (τ : ℝ),
      ∃ q, c₁ < q ∧ q < c₂ ∧ (CGB.Generation.leaveValue τ c₁ q < max q c₁ ∨ CGB.Generation.leaveValue τ c₂ q < max q c₂)
```
