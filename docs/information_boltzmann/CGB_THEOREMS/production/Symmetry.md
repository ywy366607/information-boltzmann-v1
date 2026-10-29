# Symmetry

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Symmetry.lean) · [返回导航](../README.md)

模块源码说明：P2: symmetry clauses and their first theorems * P2a (conservation) in pathwise form, and **T-N**: the Baez–Fong Noether correspondence (J.

## CGB.JumpProcess.gen_sq_sub

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L31)

解释：马尔可夫跳跃过程生成元在平方观测量上的作用减去两倍交叉项，等于各跳跃方向平方差的非负加权和（卡拉索导数恒等式）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (P : CGB.JumpProcess S) (Q : S → ℝ) (x : S),
  P.gen (fun s => Q s ^ 2) x - 2 * Q x * P.gen Q x = ∑ y, P.rate x y * (Q y - Q x) ^ 2
```

## CGB.JumpProcess.noether

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L40)

解释：诺特定理马尔可夫形式：观测量及其平方在跳跃过程生成元下均为零，当且仅当该观测量是严格守恒荷。

源码说明：**T-N** (Noether for Markov processes, pointwise form): `Q` and `Q²` are both stationary observables iff every event preserves `Q`. With `expect_const_iff` this is "⟨Q⟩ and ⟨Q²⟩ are constant for every initial distribution iff `Q` is conserved".

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (P : CGB.JumpProcess S) (Q : S → ℝ),
  ((∀ (x : S), P.gen Q x = 0) ∧ ∀ (x : S), P.gen (fun s => Q s ^ 2) x = 0) ↔ P.Conserved Q
```

## CGB.JumpProcess.energy_of_reversible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L64)

解释：在可逆马尔可夫跳跃过程中，转移速率比恒等于微观状态能量差的指数形式，存在自洽的有效能量函数。

源码说明：**T-E** (energy from time-reversal symmetry): a reversible process has an energy `E = -log π` that fixes the ratio of every event to its reverse.

形式陈述：

```lean
∀ {S : Type u_1} (P : CGB.JumpProcess S),
  P.Reversible → ∃ E, ∀ (x y : S), 0 < P.rate y x → P.rate x y / P.rate y x = Real.exp (-(E y - E x))
```

## CGB.JumpProcess.ring3_not_reversible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L83)

解释：三态有向环状跳跃过程因存在非零环流而破坏细致平衡，属于严格不可逆非平衡稳态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.JumpProcess.ring3.Reversible
```
