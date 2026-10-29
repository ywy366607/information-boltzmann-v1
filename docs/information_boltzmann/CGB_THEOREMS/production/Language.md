# Language

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Language.lean) · [返回导航](../README.md)

模块源码说明：CGB language: finite-state Markov jump processes The state–event language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1.

## CGB.JumpProcess.expect_const_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Language.lean#L32)

解释：在有限状态马尔可夫跳跃过程中，测试函数对所有概率测度的无穷小生成元期望为零，当且仅当该生成元在所有状态上处处为零。

源码说明：`⟨f⟩` is constant for every initial distribution iff `L f = 0` pointwise.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] [DecidableEq S] (P : CGB.JumpProcess S) (f : S → ℝ),
  (∀ (p : S → ℝ), (∀ (x : S), 0 ≤ p x) → ∑ x, p x = 1 → ∑ x, p x * P.gen f x = 0) ↔ ∀ (x : S), P.gen f x = 0
```
