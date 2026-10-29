# AlgebraicCore

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean) · [返回导航](../README.md)

模块源码说明：Algebraic finite-state core Finite-state transport and exact budget identities can be expressed over a commutative semiring.

## CGB.AlgebraicCore.pushForward_total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L23)

解释：保质量马尔可夫核前推测度的全空间总测度与原测度总测度严格相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} {R : Type u_3} [inst : Fintype S] [inst_1 : Fintype E] [inst_2 : CommSemiring R]
  (K : S → E → S → R) (ρ : S → R),
  (∀ (x : S), CGB.AlgebraicCore.rowMass K x = 1) → ∑ y, CGB.AlgebraicCore.pushForward K ρ y = ∑ x, ρ x
```

## CGB.AlgebraicCore.pathWeight_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L37)

解释：长度为零的马尔可夫路径在任意核作用下的路径转移权重恒等于幺元一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} {R : Type u_3} [Fintype S] [Fintype E] [inst : CommSemiring R] (K : S → E → S → R)
  (p : ℕ → S) (e : ℕ → E), CGB.AlgebraicCore.pathWeight K p e 0 = 1
```

## CGB.AlgebraicCore.jetMul_product_rule

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L47)

解释：一阶射丛乘法算子严格满足微积分导数乘积法则。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {R : Type u_4} [inst : Mul R] [inst_1 : Add R] (u v : R × R), (CGB.AlgebraicCore.jetMul u v).2 = u.1 * v.2 + u.2 * v.1
```

## CGB.AlgebraicCore.jetHorner_value

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L59)

解释：一阶射丛上的霍纳多项式求值在第一分量上与经典标量霍纳多项式完全一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {R : Type u_4} [inst : AddMonoid R] [inst_1 : Mul R] (coeff : List R) (x dx : R),
  (CGB.AlgebraicCore.jetHorner coeff (x, dx)).1 = CGB.AlgebraicCore.scalarHorner coeff x
```

## CGB.AlgebraicCore.cast_local_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L67)

解释：感官维护局部状态转移成本加上流出量，在任意半环同态下严格等于初始成本加流入量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {R : Type u_4} [inst : Semiring R] (s : CGB.SensoryMaintenance.Stim) (x : CGB.SensoryMaintenance.State),
  ↑(CGB.SensoryMaintenance.cost (CGB.SensoryMaintenance.next s x)) + ↑(CGB.SensoryMaintenance.outgoing s x) =
    ↑(CGB.SensoryMaintenance.cost x) + ↑(CGB.SensoryMaintenance.incoming s x)
```

## CGB.AlgebraicCore.cast_path_budget

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L75)

解释：感官维护全路径累计预算在任意半环上守恒，终端成本与累计流出之和等于初态成本加累计流入。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {R : Type u_4} [inst : Semiring R] (stim : ℕ → CGB.SensoryMaintenance.Stim) (p : ℕ → CGB.SensoryMaintenance.State),
  (∀ (t : ℕ), p (t + 1) = CGB.SensoryMaintenance.next (stim t) (p t)) →
    ∀ (n : ℕ),
      ↑(CGB.SensoryMaintenance.cost (p n)) + ∑ t ∈ Finset.range n, ↑(CGB.SensoryMaintenance.outgoing (stim t) (p t)) =
        ↑(CGB.SensoryMaintenance.cost (p 0)) + ∑ t ∈ Finset.range n, ↑(CGB.SensoryMaintenance.incoming (stim t) (p t))
```
