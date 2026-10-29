# GenerationAxiomAudit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean) · [返回导航](../README.md)

模块源码说明：Which axiom parts work (generation chain, audit) Usage in the generation chain (G1–G22, M1–M2): P1b's locality parts (participants in contact, writes and reads only participants), P2a, P2b and P3's boundary sensing enter

## CGB.Medium.homogeneous_regular

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L26)

解释：在接触齐次的局部介质中，任意两个空间单元具有完全相等的接触度数（正则图性质）。

源码说明：**Homogeneity forces a regular contact graph**: every unit has the same number of contacts. So with homogeneity in P1b (before 2026-10-26) a body whose contact has units of different degree (a tree with leaves, a hub, a finite grid with a surface, a line graph of compartments of different sizes) was not a model.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  M.HomogeneousContact → ∀ (u v : U), M.degree u = M.degree v
```

## CGB.Medium.not_homogeneous_of_degrees

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L45)

解释：若介质中存在两个度数不同的空间单元，则该系统必然不是接触齐次的。

源码说明：A unit with a different number of contacts than another rules out homogeneity.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U] {u v : U},
  M.degree u ≠ M.degree v → ¬M.HomogeneousContact
```
