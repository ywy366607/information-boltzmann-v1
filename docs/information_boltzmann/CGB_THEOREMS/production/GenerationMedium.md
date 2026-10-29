# GenerationMedium

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean) · [返回导航](../README.md)

模块源码说明：A generator of admitted media (generation chain, M1) Instead of proving the six clauses case by case for each organism, prove once that a medium built from a local energy satisfies them, so an organism is admitted by giv

## CGB.Medium.p2b_of_energy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L24)

解释：若局部介质的转移对合可逆、基础速率对称，且静默态动作速率由能量差指数形式给出，则该介质满足 P2b 细致平衡条件。

源码说明：**M1a**: symmetric (Metropolis-type) rates of a reversible event set under an energy give P2b.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [Fintype U] [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)] (M : CGB.Medium U Λ A I) (E : (U → Λ) → ℝ) (κ : A → ℝ) (rev : A → A),
  Function.Involutive rev →
    (∀ (a : A), κ (rev a) = κ a) →
      (∀ (a : A) (x : U → Λ), M.act a M.calm x ≠ x → M.act (rev a) M.calm (M.act a M.calm x) = x) →
        (∀ (a : A) (x : U → Λ),
            M.act a M.calm x ≠ x → M.rate a M.calm x = κ a * Real.exp (-(E (M.act a M.calm x) - E x) / 2)) →
          M.P2b
```

## CGB.Medium.rate_local_of_energy

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L45)

解释：若能量差仅取决于参与动作的作用局部区域，则该动作在不同全局背景下的速率仅依赖于局部状态，保持严格空间局域性。

源码说明：**M1b**: a rate that is a function of the participants' states through the event's energy difference reads only the participants, when that difference does.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
  (M : CGB.Medium U Λ A I) (E : (U → Λ) → ℝ) (κ : A → ℝ) (a : A) (s s' : U → I) (x y : U → Λ),
  (∀ (x y : U → Λ), (∀ u ∈ M.parts a, x u = y u) → E (M.act a s x) - E x = E (M.act a s' y) - E y) →
    M.rate a s x = κ a * Real.exp (-(E (M.act a s x) - E x) / 2) →
      M.rate a s' y = κ a * Real.exp (-(E (M.act a s' y) - E y) / 2) →
        (∀ u ∈ M.parts a, x u = y u) → M.rate a s x = M.rate a s' y
```
