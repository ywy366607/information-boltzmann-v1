# AlgebraicHomology

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicHomology.lean) · [返回导航](../README.md)

模块源码说明：Minimal algebraic homology core Finite boundary operators are represented by matrices over a commutative semiring.

## CGB.AlgebraicHomology.matrixAction_compose

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicHomology.lean#L27)

解释：向量经由两个关联矩阵相继作用的结果，恒等于经由两矩阵点积复合所得复合矩阵直接作用的结果。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {C : Type u_3} {R : Type u_4} [inst : Fintype A] [inst_1 : Fintype B] [Fintype C]
  [inst_3 : CommSemiring R] (d₁ : A → B → R) (d₂ : B → C → R) (v : A → R) (c : C),
  CGB.AlgebraicHomology.matrixAction d₂ (CGB.AlgebraicHomology.matrixAction d₁ v) c =
    ∑ a, v a * CGB.AlgebraicHomology.matrixCompose d₁ d₂ a c
```

## CGB.AlgebraicHomology.boundary_boundary_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/AlgebraicHomology.lean#L35)

解释：若两个代数关联矩阵 d1 和 d2 满足边界律，则二者的复合作用矩阵满足 d2 ∘ d1 = 0，即复合边界算子的二次作用恒为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {A : Type u_1} {B : Type u_2} {C : Type u_3} {R : Type u_4} [inst : Fintype A] [inst_1 : Fintype B] [Fintype C]
  [inst_3 : CommSemiring R] (d₁ : A → B → R) (d₂ : B → C → R),
  CGB.AlgebraicHomology.boundaryLaw d₁ d₂ →
    ∀ (v : A → R) (c : C), CGB.AlgebraicHomology.matrixAction d₂ (CGB.AlgebraicHomology.matrixAction d₁ v) c = 0
```
