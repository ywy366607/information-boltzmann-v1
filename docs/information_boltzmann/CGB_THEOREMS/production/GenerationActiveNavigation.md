# GenerationActiveNavigation

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean) · [返回导航](../README.md)

模块源码说明：Active navigation: sphere slot, branch residuals, sign credit, reactivation (2026-11-04) Design: `docs/information_boltzmann/CGB_ACTIVE_LATCH_NAVIGATION_20261104.md`.

## CGB.Generation.ActiveNavigation.fb_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L53)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：正内容经 fb 后仍为正。

解释：Formal contract for fb_pos; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {a c g x : ℝ}, 0 ≤ a → 0 < c → 0 < g → 0 < x → 0 < CGB.Generation.HoldCorrect.fb a c g x
```

## CGB.Generation.ActiveNavigation.radial_renewal_norm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L61)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：球面槽：读出长度服从标量锁存映射。

解释：The length of the renewed read is the scalar latch map of the old length.

源码说明：The length of the renewed read is the scalar latch map of the old length.

形式陈述：

```lean
∀ {V : Type u_1} [inst : NormedAddCommGroup V] [inst_1 : NormedSpace ℝ V] {a c g : ℝ},
  0 ≤ a →
    0 < c →
      0 < g →
        ∀ {r : V}, r ≠ 0 → ‖CGB.Generation.ActiveNavigation.radial a c g r‖ = CGB.Generation.HoldCorrect.fb a c g ‖r‖
```

## CGB.Generation.ActiveNavigation.radial_renewal_direction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L68)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：球面槽：读出方向不变（连续吸引子的纯保持方向）。

解释：The direction is unchanged: the renewed read is a positive multiple of the old one.

源码说明：The direction is unchanged: the renewed read is a positive multiple of the old one.

形式陈述：

```lean
∀ {V : Type u_1} [inst : NormedAddCommGroup V] [inst_1 : NormedSpace ℝ V] {a c g : ℝ},
  0 ≤ a → 0 < c → 0 < g → ∀ {r : V}, r ≠ 0 → ∃ s, 0 < s ∧ CGB.Generation.ActiveNavigation.radial a c g r = s • r
```

## CGB.Generation.ActiveNavigation.branch_residual_common_invariant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L85)

分类：semantics；适用范围：universal；审核：draft

分类说明：跨分支残差对所有分支共有的成分不变，内部时钟骗不过跨分支防塌缩。

解释：**A common component cancels in every residual** (for at least one branch).

源码说明：**A common component cancels in every residual** (for at least one branch).

形式陈述：

```lean
∀ {K d : ℕ},
  0 < K →
    ∀ (z : Fin K → Fin d → ℝ) (c : Fin d → ℝ) (k : Fin K),
      CGB.Generation.ActiveNavigation.residual (fun j i => z j i + c i) k = CGB.Generation.ActiveNavigation.residual z k
```

## CGB.Generation.ActiveNavigation.latch_keeps_sign_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L98)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：锁存保持正号。

解释：**Positive content stays positive** under the renewal, at every step.

源码说明：**Positive content stays positive** under the renewal, at every step.

形式陈述：

```lean
∀ {a c g x₀ : ℝ}, 0 ≤ a → 0 < c → 0 < g → 0 < x₀ → ∀ (n : ℕ), 0 < (CGB.Generation.HoldCorrect.fb a c g)^[n] x₀
```

## CGB.Generation.ActiveNavigation.latch_keeps_sign_neg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L107)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：锁存保持负号（奇性）。

解释：Negative content stays negative (oddness of the renewal).

源码说明：Negative content stays negative (oddness of the renewal).

形式陈述：

```lean
∀ {a c g x₀ : ℝ}, 0 ≤ a → 0 < c → 0 < g → x₀ < 0 → ∀ (n : ℕ), (CGB.Generation.HoldCorrect.fb a c g)^[n] x₀ < 0
```

## CGB.Generation.ActiveNavigation.latch_reads_forget_magnitude

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L121)

分类：design；适用范围：candidate；审核：draft；候选：LatchSlot

分类说明：同号写入的读出差以 L^n 收缩：读出只携带写入符号。

解释：**The read forgets the magnitude of the write**: two writes of the same sign inside the basin (`ρ ≤ x₀, x₀'`) are read within `L ^ n (|x₀ - r*| + |x₀' - r*|)` of each other, with `L = factor a g ρ < 1` (`latch_converges`).

源码说明：**The read forgets the magnitude of the write**: two writes of the same sign inside the basin (`ρ ≤ x₀, x₀'`) are read within `L ^ n (|x₀ - r*| + |x₀' - r*|)` of each other, with `L = factor a g ρ < 1` (`latch_converges`).

形式陈述：

```lean
∀ {a c g ρ x₀ x₀' : ℝ},
  0 ≤ a →
    a < 1 →
      0 < g →
        0 < ρ →
          ρ ≤ CGB.Generation.HoldCorrect.rstar a c g →
            ρ ≤ x₀ →
              ρ ≤ x₀' →
                ∀ (n : ℕ),
                  |(CGB.Generation.HoldCorrect.fb a c g)^[n] x₀ - (CGB.Generation.HoldCorrect.fb a c g)^[n] x₀'| ≤
                    CGB.Generation.HoldCorrect.factor a g ρ ^ n *
                      (|x₀ - CGB.Generation.HoldCorrect.rstar a c g| + |x₀' - CGB.Generation.HoldCorrect.rstar a c g|)
```

## CGB.Generation.ActiveNavigation.reactivation_exact_held

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L137)

分类：semantics；适用范围：universal；审核：draft

分类说明：被线性保持的写入，读到时重新编码给出精确的跨时间导数。

解释：**Reactivation is exact for a held write**: re-encoding the stored input at read time gives the derivative of the later loss through the hold `a ^ n`.

源码说明：**Reactivation is exact for a held write**: re-encoding the stored input at read time gives the derivative of the later loss through the hold `a ^ n`.

形式陈述：

```lean
∀ {E L : ℝ → ℝ} {θ e l a : ℝ} (n : ℕ),
  HasDerivAt E e θ → HasDerivAt L l (a ^ n * E θ) → HasDerivAt (fun t => L (a ^ n * E t)) (l * (a ^ n * e)) θ
```

## CGB.Generation.ActiveNavigation.reactivation_superposition

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L144)

分类：semantics；适用范围：universal；审核：draft

分类说明：线性记忆读出的导数是逐写入重新编码项之和。

解释：**Superposition**: a linear read `∑ j, k j * E j θ` of several writes has the sum of the per-write re-encoded terms as its derivative.

源码说明：**Superposition**: a linear read `∑ j, k j * E j θ` of several writes has the sum of the per-write re-encoded terms as its derivative.

形式陈述：

```lean
∀ {ι : Type u_1} (s : Finset ι) {E : ι → ℝ → ℝ} {e k : ι → ℝ} {θ : ℝ},
  (∀ j ∈ s, HasDerivAt (E j) (e j) θ) → HasDerivAt (fun t => ∑ j ∈ s, k j * E j t) (∑ j ∈ s, k j * e j) θ
```

## CGB.Generation.ActiveNavigation.reactivation_partial_bias

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L154)

分类：semantics；适用范围：universal；审核：draft

分类说明：只重新编码保留的写入时，偏差恰为未保留写入的项之和。

解释：**The only bias is the writes not retained**: the true derivative minus the retained part is the sum over the dropped writes.

源码说明：**The only bias is the writes not retained**: the true derivative minus the retained part is the sum over the dropped writes.

形式陈述：

```lean
∀ {ι : Type u_1} [inst : DecidableEq ι] (s r : Finset ι),
  r ⊆ s → ∀ (e k : ι → ℝ), ∑ j ∈ s, k j * e j - ∑ j ∈ r, k j * e j = ∑ j ∈ s \ r, k j * e j
```

## CGB.Generation.ActiveNavigation.progress_steps_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L165)

分类：semantics；适用范围：universal；审核：draft

分类说明：超额损失累计有界时，进展≥ε 的步数不超过 B/ε：被动停在一个来源上迟早无可学。

解释：**Progress from one source is used up.** If the excess losses `e t ≥ 0` of a learner on a source have bounded running sums (what `mixture_regret_le` gives a Bayes mixture on a fixed source: at most `log (1 / w θ₀)`), then at most `B / ε` steps can carry progress `≥ ε`, however long the learner stays. Staying passive on one source leaves eventually nothing to learn.

源码说明：**Progress from one source is used up.** If the excess losses `e t ≥ 0` of a learner on a source have bounded running sums (what `mixture_regret_le` gives a Bayes mixture on a fixed source: at most `log (1 / w θ₀)`), then at most `B / ε` steps can carry progress `≥ ε`, however long the learner stays. Staying passive on one source leaves eventually nothing to learn.

形式陈述：

```lean
∀ (e : ℕ → ℝ),
  (∀ (t : ℕ), 0 ≤ e t) →
    ∀ {B ε : ℝ}, (∀ (n : ℕ), ∑ t ∈ Finset.range n, e t ≤ B) → ∀ (n : ℕ), ↑{t ∈ Finset.range n | ε ≤ e t}.card * ε ≤ B
```

## CGB.Generation.ActiveNavigation.mixture_without_entry

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L177)

分类：semantics；适用范围：universal；审核：draft

分类说明：线性混合去掉一条的闭式，反事实无需第二次前向。

解释：**Removing one entry from a linear mixture** `p = w q + (1 - w) p₋`: the mixture without it is `(p - w q) / (1 - w)`, so the counterfactual of one entry needs no second forward pass.

源码说明：**Removing one entry from a linear mixture** `p = w q + (1 - w) p₋`: the mixture without it is `(p - w q) / (1 - w)`, so the counterfactual of one entry needs no second forward pass.

形式陈述：

```lean
∀ {p q w pm : ℝ}, w ≠ 1 → p = w * q + (1 - w) * pm → pm = (p - w * q) / (1 - w)
```
