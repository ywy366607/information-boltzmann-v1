# Medium

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Medium.lean) · [返回导航](../README.md)

模块源码说明：The CGB medium and the six axiom clauses The full language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1–2.

## CGB.Medium.total_of_reachable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Medium.lean#L161)

解释：有限步可达状态集合上的总概率和在介质动力学下守恒为一。

源码说明：P2a confines the closed dynamics to a content shell.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  M.P2a → ∀ {s : U → I} {x y : U → Λ}, Relation.ReflTransGen (M.Step s) x y → M.total y = M.total x
```

## CGB.Medium.finite_reachable

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Medium.lean#L172)

解释：在局部有限跳跃介质中，从任意给定初态出发的有限步可达微观状态集合是有限集。

源码说明：**T0a** conservation implies finiteness: under P2a and the first part of P1a, the closed dynamics reaches only finitely many states. The capacity bound is not needed here; it is needed once content can enter through the boundary.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  (∀ (n : ℕ), {ℓ | M.content ℓ ≤ n}.Finite) →
    M.P2a → ∀ (s : U → I) (x : U → Λ), {y | Relation.ReflTransGen (M.Step s) x y}.Finite
```

## CGB.Medium.finite_states

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Medium.lean#L183)

解释：局部介质在指定配置下的总允许物理状态空间是有限类型。

源码说明：With the capacity bound the whole state space is finite: the premise "finite state space" of the life definition.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [Finite U],
  M.P1a → Finite (U → Λ)
```

## CGB.Medium.energy_of_p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Medium.lean#L192)

解释：满足 P2b 细致平衡的介质存在一个与微观状态对应的实值能量函数。

源码说明：**T-E** for media: P2b gives an energy function of the equilibrium dynamics.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P2b →
    ∃ E,
      ∀ (x y : U → Λ),
        0 < (M.jump M.calm).rate y x → (M.jump M.calm).rate x y / (M.jump M.calm).rate y x = Real.exp (-(E y - E x))
```

## CGB.Medium.p2b_of_micro

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Medium.lean#L199)

解释：微观可逆性与对称速率必然导出介质在静默环境下的 P2b 平衡条件。

源码说明：Microscopic reversibility: if in the equilibrium environment every effective event is undone by its reverse event at the same rate, P2b holds (with uniform weights).

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)] (rev : A → A),
  Function.Involutive rev →
    (∀ (a : A) (x : U → Λ),
        M.act a M.calm x ≠ x →
          M.act (rev a) M.calm (M.act a M.calm x) = x ∧ M.rate (rev a) M.calm (M.act a M.calm x) = M.rate a M.calm x) →
      M.P2b
```
