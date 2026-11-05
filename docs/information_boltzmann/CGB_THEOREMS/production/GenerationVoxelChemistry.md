# GenerationVoxelChemistry

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean) · [返回导航](../README.md)

模块源码说明：Voxel chemistry: every local reaction table is admitted (generation chain, M5) The law layer of a developing body, proved once.

## CGB.VoxelRxn.swap_swap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L46)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for swap_swap; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} (c : CGB.VoxelRxn S), c.swap.swap = c
```

## CGB.VoxelState.fire_of_fires

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L65)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for fire_of_fires; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {cap : ℕ} {c : CGB.VoxelRxn S} {l : CGB.VoxelState S cap},
  CGB.VoxelState.Fires c l → ∀ (s : S), ↑(CGB.VoxelState.fire c l) s = ↑l s - c.before s + c.after s
```

## CGB.VoxelState.fire_of_not

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L69)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for fire_of_not; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {cap : ℕ} {c : CGB.VoxelRxn S} {l : CGB.VoxelState S cap},
  ¬CGB.VoxelState.Fires c l → CGB.VoxelState.fire c l = l
```

## CGB.VoxelState.fires_swap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L73)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：A reaction that ran can run backwards.

源码说明：A reaction that ran can run backwards.

形式陈述：

```lean
∀ {S : Type u_1} {cap : ℕ} {c : CGB.VoxelRxn S} {l : CGB.VoxelState S cap},
  CGB.VoxelState.Fires c l → CGB.VoxelState.Fires c.swap (CGB.VoxelState.fire c l)
```

## CGB.VoxelState.fire_swap_fire

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L86)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Running backwards restores the voxel.

源码说明：Running backwards restores the voxel.

形式陈述：

```lean
∀ {S : Type u_1} {cap : ℕ} {c : CGB.VoxelRxn S} {l : CGB.VoxelState S cap},
  CGB.VoxelState.Fires c l → CGB.VoxelState.fire c.swap (CGB.VoxelState.fire c l) = l
```

## CGB.VoxelState.weight_fire

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L112)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for weight_fire; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {cap : ℕ} [inst : Fintype S] (w : S → ℕ) {c : CGB.VoxelRxn S} {l : CGB.VoxelState S cap},
  CGB.VoxelState.Fires c l →
    CGB.VoxelState.weight w (CGB.VoxelState.fire c l) + CGB.VoxelState.rxnWeight w c.before =
      CGB.VoxelState.weight w l + CGB.VoxelState.rxnWeight w c.after
```

## CGB.VoxelRxn₂.fire_of_fires

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L144)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for fire_of_fires; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {cap : ℕ} {c : CGB.VoxelRxn₂ S} {p : CGB.VoxelState S cap × CGB.VoxelState S cap},
  c.Fires p → c.fire p = (CGB.VoxelState.fire c.left p.1, CGB.VoxelState.fire c.right p.2)
```

## CGB.VoxelRxn₂.fire_swap_fire

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L148)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for fire_swap_fire; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {cap : ℕ} {c : CGB.VoxelRxn₂ S} {p : CGB.VoxelState S cap × CGB.VoxelState S cap},
  c.Fires p → c.swap.fire (c.fire p) = p
```

## CGB.VoxelChem.weight_eq_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L224)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for weight_eq_zero; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {S : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} {cap : ℕ} [inst : Fintype S]
  (C : CGB.VoxelChem U S I R₁ R₂ cap),
  (∀ (s : S), 0 < C.w s) → ∀ {l : CGB.VoxelState S cap}, CGB.VoxelState.weight C.w l = 0 → l = CGB.VoxelChem.vacuum
```

## CGB.VoxelChem.p1a_data

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L231)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for p1a_data; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {S : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} {cap : ℕ} [inst : Fintype S]
  (C : CGB.VoxelChem U S I R₁ R₂ cap),
  (∀ (s : S), 0 < C.w s) →
    (∀ (n : ℕ), {l | CGB.VoxelState.weight C.w l ≤ n}.Finite) ∧
      (∃ K, ∀ (l : CGB.VoxelState S cap), CGB.VoxelState.weight C.w l ≤ K) ∧ ∃! l, CGB.VoxelState.weight C.w l = 0
```

## CGB.VoxelChem.fires_of_fire_ne₁

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L240)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for fires_of_fire_ne₁; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_2} {cap : ℕ} [Fintype S] {c : CGB.VoxelRxn S} {l : CGB.VoxelState S cap},
  CGB.VoxelState.fire c l ≠ l → CGB.VoxelState.Fires c l
```

## CGB.VoxelChem.fires_of_fire_ne₂

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L245)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for fires_of_fire_ne₂; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_2} {cap : ℕ} [Fintype S] {c : CGB.VoxelRxn₂ S} {p : CGB.VoxelState S cap × CGB.VoxelState S cap},
  c.fire p ≠ p → c.Fires p
```

## CGB.VoxelChem.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L252)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：**M5**: every voxel chemistry that conserves weight in the interior and has reverses is admitted.

源码说明：**M5**: every voxel chemistry that conserves weight in the interior and has reverses is admitted.

形式陈述：

```lean
∀ {U : Type u_1} {S : Type u_2} {I : Type u_3} {R₁ : Type u_4} {R₂ : Type u_5} {cap : ℕ} [inst : Fintype S]
  (C : CGB.VoxelChem U S I R₁ R₂ cap) [inst_1 : Fintype U] [inst_2 : DecidableEq U] [inst_3 : Fintype R₁]
  [DecidableEq R₁] [inst_5 : Fintype R₂] [DecidableEq R₂] [inst_7 : DecidableEq (U → CGB.VoxelState S cap)],
  (∀ (s : S), 0 < C.w s) →
    (∀ (r : R₁),
        ∀ i ∉ C.bid,
          CGB.VoxelState.rxnWeight C.w (C.rxn₁ r i).before = CGB.VoxelState.rxnWeight C.w (C.rxn₁ r i).after) →
      (∀ (r : R₂) (i j : ℕ),
          i ∉ C.bid →
            j ∉ C.bid →
              CGB.VoxelState.rxnWeight C.w (C.rxn₂ r i j).left.before +
                  CGB.VoxelState.rxnWeight C.w (C.rxn₂ r i j).right.before =
                CGB.VoxelState.rxnWeight C.w (C.rxn₂ r i j).left.after +
                  CGB.VoxelState.rxnWeight C.w (C.rxn₂ r i j).right.after) →
        C.μ C.calm = 0 →
          Function.Involutive C.rev₁ →
            Function.Involutive C.rev₂ →
              (∀ (r : R₁) (i : ℕ), C.rxn₁ (C.rev₁ r) i = (C.rxn₁ r i).swap) →
                (∀ (r : R₂) (i j : ℕ), C.rxn₂ (C.rev₂ r) i j = (C.rxn₂ r i j).swap) →
                  (∀ (r : R₁) (i : ℕ), C.κ₁ (C.rev₁ r) i = C.κ₁ r i) →
                    (∀ (r : R₂) (i j : ℕ), C.κ₂ (C.rev₂ r) i j = C.κ₂ r i j) → C.lattice.medium.Axioms
```

## CGB.VoxelRxn.swap_pair

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L330)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for swap_pair; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} (a c b d : S), (CGB.VoxelRxn.pair a c b d).swap = CGB.VoxelRxn.pair b d a c
```

## CGB.VoxelState.rxnWeight_pair

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L338)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：Formal contract for rxnWeight_pair; see its Lean signature for premises and conclusion.

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (w : S → ℕ) (a c : S),
  CGB.VoxelState.rxnWeight w (Pi.single a 1 + Pi.single c 1) = w a + w c
```

## CGB.VoxelState.row_conserves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L344)

分类：design；适用范围：candidate；审核：draft；候选：VoxelChemistry

分类说明：体素化学表的候选介质实例引理（11-02 写入，补分类）。

解释：A row conserves weight exactly when its two sides weigh the same.

源码说明：A row conserves weight exactly when its two sides weigh the same.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (w : S → ℕ) {a c b d : S},
  w a + w c = w b + w d →
    CGB.VoxelState.rxnWeight w (CGB.VoxelRxn.pair a c b d).before =
      CGB.VoxelState.rxnWeight w (CGB.VoxelRxn.pair a c b d).after
```
