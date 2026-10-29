# GenerationExport

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean) · [返回导航](../README.md)

模块源码说明：Export and selective export (generation chain, G16-G17) P1a makes the body finite; P2a gives the content ledger `content (t+1) + out t = content t + in t` (`MaterialLedger.balance`).

## CGB.Generation.export_forced

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L25)

解释：在每拍内容守恒且容量有界时，累计摄入不超过累计排出加容量。

源码说明：**G16a (export is forced)**: with the content ledger and a capacity bound, cumulative export is at least cumulative intake minus the capacity.

形式陈述：

```lean
∀ (c inn out : ℕ → ℕ) (cap : ℕ),
  (∀ (t : ℕ), c (t + 1) + out t = c t + inn t) →
    (∀ (t : ℕ), c t ≤ cap) → ∀ (T : ℕ), ∑ t ∈ Finset.range T, inn t ≤ ∑ t ∈ Finset.range T, out t + cap
```

## CGB.Generation.write_only_intake_bounded

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L42)

解释：当只有摄入而无排出且内容不超过容量时，任意有限时段的累计摄入不超过该容量。

源码说明：**G16b (write-only bodies starve)**: with no export, total lifetime intake is at most the capacity.

形式陈述：

```lean
∀ (c inn : ℕ → ℕ) (cap : ℕ),
  (∀ (t : ℕ), c (t + 1) = c t + inn t) → (∀ (t : ℕ), c t ≤ cap) → ∀ (T : ℕ), ∑ t ∈ Finset.range T, inn t ≤ cap
```

## CGB.Generation.endless_intake_endless_export

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L50)

解释：在内容守恒与容量有界的条件下，如果累计摄入可以超过任意界，则累计排出也可以超过任意界。

源码说明：**G16c**: if intake does not stop (its total exceeds every bound), export does not stop either.

形式陈述：

```lean
∀ (c inn out : ℕ → ℕ) (cap : ℕ),
  (∀ (t : ℕ), c (t + 1) + out t = c t + inn t) →
    (∀ (t : ℕ), c t ≤ cap) →
      (∀ (B : ℕ), ∃ T, B < ∑ t ∈ Finset.range T, inn t) → ∀ (B : ℕ), ∃ T, B < ∑ t ∈ Finset.range T, out t
```

## CGB.Generation.survive_le_pow

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L67)

解释：若每次排出删除某内容的概率至少为 q₀ 且不超过一，其经过 n 次排出的存活量不超过 (1−q₀)^n。

源码说明：**G17a (unselective export forgets)**: if every export removes the item with probability at least `q₀ > 0` (for example `1/C` when the exported unit is drawn without regard to what it holds), survival is at most `(1 - q₀)^n`.

形式陈述：

```lean
∀ (q : ℕ → ℝ) (q₀ : ℝ),
  (∀ (k : ℕ), q₀ ≤ q k) → (∀ (k : ℕ), q k ≤ 1) → ∀ (n : ℕ), CGB.Generation.survive q n ≤ (1 - q₀) ^ n
```

## CGB.Generation.unselective_export_forgets

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L79)

解释：若每次删除概率有正下界且不超过一，则经过足够多次排出后，该内容的存活量会低于任何正阈值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : ℕ → ℝ) (q₀ : ℝ),
  0 < q₀ → (∀ (k : ℕ), q₀ ≤ q k) → (∀ (k : ℕ), q k ≤ 1) → ∀ (ε : ℝ), 0 < ε → ∃ n, CGB.Generation.survive q n < ε
```

## CGB.Generation.retention_needs_selectivity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L90)

解释：若内容在任意次数排出后始终有正的存活下界，则其各次删除概率不可能始终高于同一个正数。

源码说明：**G17b (retention needs selectivity)**: if an item must survive every number of exports with probability at least `p > 0`, its per-export removal probability has no positive lower bound: export must eventually spare it, so it cannot be drawn without regard to state (whose removal probability is a fixed `1/C`).

形式陈述：

```lean
∀ (q : ℕ → ℝ),
  (∀ (k : ℕ), q k ≤ 1) → ∀ (p : ℝ), 0 < p → (∀ (n : ℕ), p ≤ CGB.Generation.survive q n) → ∀ q₀ > 0, ∃ k, q k < q₀
```
