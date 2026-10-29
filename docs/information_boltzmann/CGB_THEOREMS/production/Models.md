# Models

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Models.lean) · [返回导航](../README.md)

模块源码说明：Consistency models, independence witnesses and the bottleneck theorem **Consistency.** * `twinFlip`: closed.

## CGB.Models.xhopAct_of_ok

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L54)

解释：当起点占据、终点空且两者不同时，跳跃操作等于交换这两个位置的状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {u v : U} {x : U → Bool},
  u ≠ v ∧ x u = true ∧ x v = false → CGB.Models.xhopAct u v x = x ∘ ⇑(Equiv.swap u v)
```

## CGB.Models.xhopAct_of_not

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L57)

解释：当指定起终点不满足有效跳跃条件时，跳跃操作保持状态不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {u v : U} {x : U → Bool},
  ¬(u ≠ v ∧ x u = true ∧ x v = false) → CGB.Models.xhopAct u v x = x
```

## CGB.Models.xhopAct_rev_ok

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L60)

解释：有效跳跃后原终点被占据、原起点为空，因此交换端点的逆跳跃条件成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {u v : U} {x : U → Bool},
  u ≠ v ∧ x u = true ∧ x v = false → v ≠ u ∧ CGB.Models.xhopAct u v x v = true ∧ CGB.Models.xhopAct u v x u = false
```

## CGB.Models.xhopAct_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L65)

解释：当起点占据、终点空且两者不同时，先正向跳再反向跳恢复原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {u v : U} {x : U → Bool},
  u ≠ v ∧ x u = true ∧ x v = false → CGB.Models.xhopAct v u (CGB.Models.xhopAct u v x) = x
```

## CGB.Models.xhopAct_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L71)

解释：有限位置上，跳跃前后对任意布尔状态函数求和的结果相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] [inst_1 : Fintype U] (f : Bool → ℕ) (u v : U) (x : U → Bool),
  ∑ w, f (CGB.Models.xhopAct u v x w) = ∑ w, f (x w)
```

## CGB.Models.xhopAct_outside

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L78)

解释：跳跃操作不改变起点和终点之外的位置。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {u v w : U} (x : U → Bool), w ≠ u → w ≠ v → CGB.Models.xhopAct u v x w = x w
```

## CGB.Models.xhopAct_congr

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L83)

解释：两个状态在跳跃端点一致时，跳跃后在任一端点的值也一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {u v w : U} {x y : U → Bool},
  x u = y u → x v = y v → w = u ∨ w = v → CGB.Models.xhopAct u v x w = CGB.Models.xhopAct u v y w
```

## CGB.Models.symm_swap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L88)

解释：将交换操作按任意置换共轭后，得到原端点交换与逆置换的对应关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] (σ : Equiv.Perm U) (u v w : U),
  (Equiv.symm σ) ((Equiv.swap (σ u) (σ v)) w) = (Equiv.swap u v) ((Equiv.symm σ) w)
```

## CGB.Models.xhopAct_perm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L97)

解释：同步置换跳跃端点和状态后，跳跃结果与原结果同步置换一致。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] (σ : Equiv.Perm U) (u v : U) (x : U → Bool),
  CGB.Models.xhopAct (σ u) (σ v) (x ∘ ⇑(Equiv.symm σ)) = CGB.Models.xhopAct u v x ∘ ⇑(Equiv.symm σ)
```

## CGB.Models.xhopAct_ind

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L108)

解释：对不同单占据位置，若一次指定跳跃将 a 的单粒子状态变为 b 的单粒子状态，则跳跃起终点分别为 a、b。

源码说明：The content of `a` alone, moved by a hop, can only have gone along `a → b`.

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {a b p q : U},
  a ≠ b → ((CGB.Models.xhopAct p q fun w => decide (w = a)) = fun w => decide (w = b)) → p = a ∧ q = b
```

## CGB.Models.ident_swap

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L121)

解释：两个位置身份相同时，交换这两个位置保持所有位置的身份标签。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {ident : U → ℕ} {a b : U},
  ident a = ident b → ∀ (w : U), ident ((Equiv.swap a b) w) = ident w
```

## CGB.Models.eq_homogeneous

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L130)

解释：对只以相等性定义关系的可判定类型，存在将任意位置映到另一位置且保持相等关系的置换。

源码说明：Homogeneous contact by equality: any swap is a symmetry.

形式陈述：

```lean
∀ {U : Type u_1} [DecidableEq U] (u v : U), ∃ σ, (∀ (a b : U), σ a = σ b ↔ a = b) ∧ σ u = v
```

## CGB.Models.bool_p1a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L135)

解释：若布尔局部状态的内容量等于其自然数值，则介质满足 P1a 有限内容条件。

源码说明：Occupancy content: a site is empty (the vacuum) or holds one unit.

形式陈述：

```lean
∀ {U : Type u_1} {A : Type u_2} {I : Type u_3} {M : CGB.Medium U Bool A I}, M.content = Bool.toNat → M.P1a
```

## CGB.Models.xhop_isAut

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L165)

解释：在指定 xhop 构造中，任何保持位置身份标签的置换都是自同构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {c : U → U → ℝ} {hc : ∀ (u v : U), 0 ≤ c u v} {ident : U → ℕ} {init : U → Bool}
  {σ : Equiv.Perm U}, (∀ (u : U), ident (σ u) = ident u) → (CGB.Models.xhop U c hc ident init).IsAut σ
```

## CGB.Models.xhop_p1b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L170)

解释：指定 xhop 跳跃介质满足 P1b 局部性条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {c : U → U → ℝ} {hc : ∀ (u v : U), 0 ≤ c u v} {ident : U → ℕ} {init : U → Bool},
  (CGB.Models.xhop U c hc ident init).P1b
```

## CGB.Models.xhop_p2a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L180)

解释：有限位置的指定 xhop 跳跃介质满足 P2a 内容守恒。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : Fintype U] [inst_1 : DecidableEq U] {c : U → U → ℝ} {hc : ∀ (u v : U), 0 ≤ c u v}
  {ident : U → ℕ} {init : U → Bool}, (CGB.Models.xhop U c hc ident init).P2a
```

## CGB.Models.xhop_p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L183)

解释：有限位置的 xhop 介质若双向电导相等，则满足 P2b 平静细致平衡。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : Fintype U] [inst_1 : DecidableEq U] {c : U → U → ℝ} {hc : ∀ (u v : U), 0 ≤ c u v}
  {ident : U → ℕ} {init : U → Bool}, (∀ (u v : U), c u v = c v u) → (CGB.Models.xhop U c hc ident init).P2b
```

## CGB.Models.xhop_p2c

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L193)

解释：若所有自同构保持 xhop 电导，则该介质满足 P2c 等变性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {c : U → U → ℝ} {hc : ∀ (u v : U), 0 ≤ c u v} {ident : U → ℕ} {init : U → Bool},
  (∀ (σ : Equiv.Perm U), (CGB.Models.xhop U c hc ident init).IsAut σ → ∀ (u v : U), c (σ u) (σ v) = c u v) →
    (CGB.Models.xhop U c hc ident init).P2c
```

## CGB.Models.xhop_senses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L200)

解释：指定 xhop 介质满足只有边界感知外部刺激的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {c : U → U → ℝ} {hc : ∀ (u v : U), 0 ≤ c u v} {ident : U → ℕ} {init : U → Bool},
  (CGB.Models.xhop U c hc ident init).OnlyBoundarySenses
```

## CGB.Models.xhop_pages

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L206)

解释：指定 xhop 介质满足 P2c 时，相同端点身份页的不同位置对具有相同电导。

源码说明：**Bottleneck theorem.** With all units in contact, P2c forces each conductance to depend only on the identity pages of its two ends.

形式陈述：

```lean
∀ {U : Type u_1} [inst : DecidableEq U] {c : U → U → ℝ} {hc : ∀ (u v : U), 0 ≤ c u v} {ident : U → ℕ} {init : U → Bool},
  (CGB.Models.xhop U c hc ident init).P2c →
    ∀ {u v u' v' : U}, u ≠ v → u' ≠ v' → ident u = ident u' → ident v = ident v' → c u v = c u' v'
```

## CGB.Models.solo_isAut

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L271)

解释：对指定均一单位置事件构造，任意位置置换都是自同构。

源码说明：Every permutation of units is a symmetry of `solo`.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {E : Type u_3} [inst : DecidableEq U] {content : Λ → ℕ} {step : E → Λ → Λ}
  {r : E → Λ → ℝ} {hr : ∀ (e : E) (ℓ : Λ), 0 ≤ r e ℓ} {ℓ₀ : Λ} (σ : Equiv.Perm U),
  (CGB.Models.solo U content step r hr ℓ₀).IsAut σ
```

## CGB.Models.solo_p1b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L274)

解释：指定单位置事件介质满足 P1b 局部性条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {E : Type u_3} [inst : DecidableEq U] {content : Λ → ℕ} {step : E → Λ → Λ}
  {r : E → Λ → ℝ} {hr : ∀ (e : E) (ℓ : Λ), 0 ≤ r e ℓ} {ℓ₀ : Λ}, (CGB.Models.solo U content step r hr ℓ₀).P1b
```

## CGB.Models.solo_p2a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L287)

解释：若单位置事件逐个保持内容，指定 solo 介质满足 P2a。

源码说明：Local content conservation of each event gives P2a.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {E : Type u_3} [inst : DecidableEq U] {content : Λ → ℕ} {step : E → Λ → Λ}
  {r : E → Λ → ℝ} {hr : ∀ (e : E) (ℓ : Λ), 0 ≤ r e ℓ} {ℓ₀ : Λ} [inst_1 : Fintype U],
  (∀ (e : E) (ℓ : Λ), content (step e ℓ) = content ℓ) → (CGB.Models.solo U content step r hr ℓ₀).P2a
```

## CGB.Models.solo_p2c

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L295)

解释：指定均一单位置事件介质满足 P2c 重标记等变条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {E : Type u_3} [inst : DecidableEq U] {content : Λ → ℕ} {step : E → Λ → Λ}
  {r : E → Λ → ℝ} {hr : ∀ (e : E) (ℓ : Λ), 0 ≤ r e ℓ} {ℓ₀ : Λ}, (CGB.Models.solo U content step r hr ℓ₀).P2c
```

## CGB.Models.solo_p3

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L304)

解释：指定 solo 介质满足 P3 边界与初始状态条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {E : Type u_3} [inst : DecidableEq U] {content : Λ → ℕ} {step : E → Λ → Λ}
  {r : E → Λ → ℝ} {hr : ∀ (e : E) (ℓ : Λ), 0 ≤ r e ℓ} {ℓ₀ : Λ}, (CGB.Models.solo U content step r hr ℓ₀).P3
```

## CGB.Models.solo_p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L308)

解释：若 solo 构造的事件具有定理要求的配对逆操作和速率条件，则该介质满足 P2b。

源码说明：Local microscopic reversibility of each unit gives P2b.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {E : Type u_3} [inst : DecidableEq U] {content : Λ → ℕ} {step : E → Λ → Λ}
  {r : E → Λ → ℝ} {hr : ∀ (e : E) (ℓ : Λ), 0 ≤ r e ℓ} {ℓ₀ : Λ} [inst_1 : Fintype U] [inst_2 : Fintype E]
  [inst_3 : DecidableEq (U → Λ)] (rev : E → E),
  Function.Involutive rev →
    (∀ (e : E) (ℓ : Λ), step e ℓ ≠ ℓ → step (rev e) (step e ℓ) = ℓ ∧ r (rev e) (step e ℓ) = r e ℓ) →
      (CGB.Models.solo U content step r hr ℓ₀).P2b
```

## CGB.Models.flip12_flip12

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L328)

解释：三状态翻转操作连续执行两次后返回原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (ℓ : Fin 3), CGB.Models.flip12 (CGB.Models.flip12 ℓ) = ℓ
```

## CGB.Models.flipCost_flip12

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L330)

解释：三状态翻转操作保持所定义的内容量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (ℓ : Fin 3), CGB.Models.flipCost (CGB.Models.flip12 ℓ) = CGB.Models.flipCost ℓ
```

## CGB.Models.flipCost_p1a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L332)

解释：三状态介质若采用指定 flipCost 内容函数，则满足 P1a。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {A : Type u_2} {I : Type u_3} {M : CGB.Medium U (Fin 3) A I}, M.content = CGB.Models.flipCost → M.P1a
```

## CGB.Models.twinFlip_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L343)

解释：所构造的 twinFlip 介质满足当前 Axioms 结构中列出的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.twinFlip.Axioms
```

## CGB.Models.twinFlip_swap_isAut

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L347)

解释：交换 twinFlip 的零号和一号位置是其自同构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.twinFlip.IsAut (Equiv.swap 0 1)
```

## CGB.Models.rock_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L355)

解释：所构造的 rock 介质满足当前 Axioms 结构中列出的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.rock.Axioms
```

## CGB.Models.rock_frozen

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L360)

解释：rock 介质在任意刺激及状态对之间的跳跃速率为零。

源码说明：The rock is frozen: no transition ever has positive rate, in any environment.

形式陈述：

```lean
∀ (s : Fin 2 → Unit) (x y : Fin 2 → Fin 3), (CGB.Models.rock.jump s).rate x y = 0
```

## CGB.Models.ringAdj_refl

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L371)

解释：四位置环形接触关系对每个位置自反。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (u : Fin 4), CGB.Models.ringAdj u u
```

## CGB.Models.ringAdj_symm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L373)

解释：四位置环形接触关系是对称的。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {u v : Fin 4}, CGB.Models.ringAdj u v → CGB.Models.ringAdj v u
```

## CGB.Models.ringAdj_shift

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L375)

解释：四位置环形接触关系在两个端点同时平移后保持不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (k a b : Fin 4), CGB.Models.ringAdj (a + k) (b + k) ↔ CGB.Models.ringAdj a b
```

## CGB.Models.seed_p1b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L409)

解释：seed 介质满足 P1b 局部接触条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.P1b
```

## CGB.Models.seed_p2a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L452)

解释：seed 介质满足 P2a 闭事件内容守恒条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.P2a
```

## CGB.Models.seed_p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L461)

解释：seed 介质满足 P2b 平静环境细致平衡条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.P2b
```

## CGB.Models.seed_aut_fix

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L480)

解释：seed 的任一自同构及其逆都固定被指定的零号和二号位置。

源码说明：Every symmetry of the seed fixes the source and the sink.

形式陈述：

```lean
∀ {σ : Equiv.Perm (Fin 4)}, CGB.Models.seed.IsAut σ → (Equiv.symm σ) 0 = 0 ∧ (Equiv.symm σ) 2 = 2 ∧ σ 0 = 0 ∧ σ 2 = 2
```

## CGB.Models.seed_p2c

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L490)

解释：seed 介质满足 P2c 一致重标记等变条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.P2c
```

## CGB.Models.seed_p3

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L517)

解释：seed 介质满足 P3 边界与初始状态条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.P3
```

## CGB.Models.seed_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L529)

解释：所构造的 seed 介质满足当前 Axioms 结构中列出的条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.Models.seed.Axioms
```

## CGB.Models.seed_swap_isAut

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L533)

解释：交换 seed 的一号与三号位置是其自同构。

源码说明：The reflection swapping the interior sites is a genuine symmetry.

形式陈述：

```lean
CGB.Models.seed.IsAut (Equiv.swap 1 3)
```

## CGB.Models.seed_driven

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L541)

解释：seed 在指定 food 刺激下的跳跃过程不满足可逆性。

源码说明：The seed is driven: under food its open dynamics has no detailed balance. Excreted content never comes back through the sink. In the calm environment it is reversible.

形式陈述：

```lean
¬(CGB.Models.seed.jump CGB.Models.food).Reversible
```

## CGB.Models.seed_grows

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L567)

解释：seed 在 food 下存在从总内容一到总内容三的有限开放事件路径。

源码说明：**Growth.** From one unit at the source, under food, the seed reaches a state with three units: intake and spreading create units at empty sites.

形式陈述：

```lean
(Relation.ReflTransGen (CGB.Models.OpenStep CGB.Models.seed CGB.Models.food) CGB.Models.seed.init fun w =>
    decide (w ≠ 2)) ∧
  CGB.Models.seed.total CGB.Models.seed.init = 1 ∧ (CGB.Models.seed.total fun w => decide (w ≠ 2)) = 3
```

## CGB.Models.twinTape_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L593)

解释：twinTape 见证违反 P1a 的有限子水平集部分，同时满足定理列出的其他公理与内容有界、唯一零态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.twinTape.P1a ∧
  CGB.Models.twinTape.P1b ∧
    CGB.Models.twinTape.P2a ∧
      CGB.Models.twinTape.P2b ∧
        CGB.Models.twinTape.P2c ∧
          CGB.Models.twinTape.P3 ∧
            CGB.Models.twinTape.IsAut (Equiv.swap 0 1) ∧
              (∃ K, ∀ (ℓ : ℕ), CGB.Models.twinTape.content ℓ ≤ K) ∧ ∃! ℓ, CGB.Models.twinTape.content ℓ = 0
```

## CGB.Models.twinBag_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L629)

解释：twinBag 见证违反 P1a 的统一有界部分，同时满足定理列出的其他公理与有限子水平集、唯一零态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.twinBag.P1a ∧
  CGB.Models.twinBag.P1b ∧
    CGB.Models.twinBag.P2a ∧
      CGB.Models.twinBag.P2b ∧
        CGB.Models.twinBag.P2c ∧
          CGB.Models.twinBag.P3 ∧
            CGB.Models.twinBag.IsAut (Equiv.swap 0 1) ∧
              (∀ (n : ℕ), {ℓ | CGB.Models.twinBag.content ℓ ≤ n}.Finite) ∧ ∃! ℓ, CGB.Models.twinBag.content ℓ = 0
```

## CGB.Models.twinBlank_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L657)

解释：twinBlank 见证违反 P1a 的唯一零态部分，同时满足定理列出的其他公理及另外两个内容条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.twinBlank.P1a ∧
  CGB.Models.twinBlank.P1b ∧
    CGB.Models.twinBlank.P2a ∧
      CGB.Models.twinBlank.P2b ∧
        CGB.Models.twinBlank.P2c ∧
          CGB.Models.twinBlank.P3 ∧
            CGB.Models.twinBlank.IsAut (Equiv.swap 0 1) ∧
              (∀ (n : ℕ), {ℓ | CGB.Models.twinBlank.content ℓ ≤ n}.Finite) ∧
                ∃ K, ∀ (ℓ : Fin 3), CGB.Models.twinBlank.content ℓ ≤ K
```

## CGB.Models.twinToggle_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L675)

解释：twinToggle 见证违反 P2a，同时满足列出的 P1a、P1b、P2b、P2c、P3 和指定交换自同构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.twinToggle.P2a ∧
  CGB.Models.twinToggle.P1a ∧
    CGB.Models.twinToggle.P1b ∧
      CGB.Models.twinToggle.P2b ∧
        CGB.Models.twinToggle.P2c ∧ CGB.Models.twinToggle.P3 ∧ CGB.Models.twinToggle.IsAut (Equiv.swap 0 1)
```

## CGB.Models.twinCycle_not_p2b

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L699)

解释：所构造的 twinCycle 介质不满足 P2b 平静细致平衡条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.twinCycle.P2b
```

## CGB.Models.twinCycle_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L721)

解释：twinCycle 见证违反 P2b，同时满足列出的 P1a、P1b、P2a、P2c、P3 和指定交换自同构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.twinCycle.P2b ∧
  CGB.Models.twinCycle.P1a ∧
    CGB.Models.twinCycle.P1b ∧
      CGB.Models.twinCycle.P2a ∧
        CGB.Models.twinCycle.P2c ∧ CGB.Models.twinCycle.P3 ∧ CGB.Models.twinCycle.IsAut (Equiv.swap 0 1)
```

## CGB.Models.perm_fin2_succ

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L746)

解释：任意两个位置的置换与该二元素空间上的加一操作交换。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (σ : Equiv.Perm (Fin 2)) (u : Fin 2), σ (u + 1) = σ u + 1
```

## CGB.Models.farFlip_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L754)

解释：farFlip 见证违反 P1b，同时满足列出的 P1a、P2、P3 条件并允许指定交换自同构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.farFlip.P1b ∧
  CGB.Models.farFlip.P1a ∧
    CGB.Models.farFlip.P2a ∧
      CGB.Models.farFlip.P2b ∧
        CGB.Models.farFlip.P2c ∧ CGB.Models.farFlip.P3 ∧ CGB.Models.farFlip.IsAut (Equiv.swap 0 1)
```

## CGB.Models.triLookup_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L783)

解释：triLookup 见证违反 P2c，同时满足列出的 P1a、P1b、P2a、P2b 和 P3。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.triLookup.P2c ∧
  CGB.Models.triLookup.P1a ∧
    CGB.Models.triLookup.P1b ∧ CGB.Models.triLookup.P2a ∧ CGB.Models.triLookup.P2b ∧ CGB.Models.triLookup.P3
```

## CGB.Models.pairPreset_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L798)

解释：pairPreset 见证违反 P3，同时满足列出的 P1a、P1b、P2 条件和指定交换自同构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.pairPreset.P3 ∧
  CGB.Models.pairPreset.P1a ∧
    CGB.Models.pairPreset.P1b ∧
      CGB.Models.pairPreset.P2a ∧
        CGB.Models.pairPreset.P2b ∧ CGB.Models.pairPreset.P2c ∧ CGB.Models.pairPreset.IsAut (Equiv.swap 0 1)
```

## CGB.Models.innerClock_witness

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Models.lean#L821)

解释：innerClock 见证违反 P3，同时满足列出的 P1a、P1b、P2 条件和指定交换自同构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Models.innerClock.P3 ∧
  CGB.Models.innerClock.P1a ∧
    CGB.Models.innerClock.P1b ∧
      CGB.Models.innerClock.P2a ∧
        CGB.Models.innerClock.P2b ∧ CGB.Models.innerClock.P2c ∧ CGB.Models.innerClock.IsAut (Equiv.swap 0 1)
```
