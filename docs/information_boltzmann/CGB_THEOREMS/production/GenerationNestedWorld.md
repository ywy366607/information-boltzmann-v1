# GenerationNestedWorld

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean) · [返回导航](../README.md)

模块源码说明：An organism nested in its world (T-E2) The joint medium of a bacterium-like organism and its world, as a lattice medium (M4).

## CGB.poolShift_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L48)

解释：有界内容池按整数 d 成功移动后，再按 −d 移动可回到原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {V : ℕ} {k k' : Fin (V + 1)} {d : ℤ}, CGB.poolShift k d = some k' → CGB.poolShift k' (-d) = some k
```

## CGB.poolShift_val

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L61)

解释：有界内容池按整数 d 成功移动后，新池值等于旧池值加 d。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {V : ℕ} {k k' : Fin (V + 1)} {d : ℤ}, CGB.poolShift k d = some k' → ↑↑k' = ↑↑k + d
```

## CGB.NestedWorld.sdir_not

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L191)

解释：切换规则方向后，指定的两个内容变化量分别变为原值的负数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (k : Fin nR) (dir : Bool),
  (W.sdir k !dir) = (-(W.sdir k dir).1, -(W.sdir k dir).2)
```

## CGB.NestedWorld.drain_content

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L279)

解释：若集合 S 内的初始值均为 a，将其更新为 b 后，总内容与原总内容满足加上 S 大小乘对应内容的等式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] {n nR nX nB : ℕ} (W : CGB.NestedWorld Λi Hd n nR nX nB)
  (S : Finset (Fin n)) (a b : Λi) (y : Fin n → Λi),
  (∀ v ∈ S, y v = a) →
    ∑ v, W.contentIn (CGB.NestedWorld.drain S b y v) + S.card * W.contentIn a =
      ∑ v, W.contentIn (y v) + S.card * W.contentIn b
```

## CGB.NestedWorld.drain_drain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L294)

解释：若集合内所有位置初始值相同，先写入另一值再写回原值可恢复整个内部状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {n : ℕ} (S : Finset (Fin n)) (a b : Λi) (y : Fin n → Λi),
  (∀ v ∈ S, y v = a) → CGB.NestedWorld.drain S a (CGB.NestedWorld.drain S b y) = y
```

## CGB.NestedWorld.canDrain_drain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L302)

解释：若给定集合和内部状态满足 canDrain，则将该集合替换为任意值后满足对应的新 canDrain 条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) {rb : Fin nB} {S : Finset (Fin n)} {a : Λi} {y : Fin n → Λi},
  W.canDrain rb S a y → ∀ (b : Λi), W.canDrain rb S b (CGB.NestedWorld.drain S b y)
```

## CGB.NestedWorld.gateCount_drain

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L305)

解释：若分裂页不同于门页且满足 canDrain，指定排空操作保持内部门计数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) {rb : Fin nB},
  W.divPage rb ≠ W.gatePage →
    ∀ {S : Finset (Fin n)} {a : Λi} {y : Fin n → Λi},
      W.canDrain rb S a y → ∀ (b : Λi), W.gateCount (CGB.NestedWorld.drain S b y) = W.gateCount y
```

## CGB.NestedWorld.move_exchanges

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L374)

解释：当前后位置及燃料、运动状态满足定理条件时，指定双体移动事件按所列公式交换两位置的主体字段并更新燃料状态。

源码说明：**A move is an exchange**: whatever occupies the cell ahead (vacuum or a body), a running mover whose fuel rule fires takes its place and the occupant takes the mover's, unchanged; pools stay.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (e d : Fin n) (i j : ℕ) (f w f' w' : Fin (W.V + 1))
  (o : Option ((Fin n → Λi) × Hd)) (y : Fin n → Λi) (h : Hd),
  W.ahead i j h = true ∧ W.fuelAct e d true y ≠ y ∧ y W.motor = W.run ∧ W.fuelAct e d true y W.motor = W.run →
    W.eff2 (Sum.inr (Sum.inl (e, d, true))) i j ((f, w, o), f', w', some (y, h)) =
      ((f, w, some (W.fuelAct e d true y, h)), f', w', o)
```

## CGB.NestedWorld.poolCond_symm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L402)

解释：指定池导率对两个池值的次序对称。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (V a b : ℕ), CGB.NestedWorld.poolCond V a b = CGB.NestedWorld.poolCond V b a
```

## CGB.NestedWorld.poolCond_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L405)

解释：指定有界池导率对任意自然数参数非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (V a b : ℕ), 0 ≤ CGB.NestedWorld.poolCond V a b
```

## CGB.NestedWorld.poolFactor_symm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L411)

解释：指定嵌套世界的池因子交换两个单元状态后保持不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (l l' : W.Cell), W.poolFactor l l' = W.poolFactor l' l
```

## CGB.NestedWorld.poolFactor_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L414)

解释：指定嵌套世界任意两个单元状态之间的池因子非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (l l' : W.Cell), 0 ≤ W.poolFactor l l'
```

## CGB.NestedWorld.poolCond_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L417)

解释：两个池值均不超过容量时，指定池导率严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (V a b : ℕ), a ≤ V → b ≤ V → 0 < CGB.NestedWorld.poolCond V a b
```

## CGB.NestedWorld.poolFactor_pos

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L428)

解释：指定嵌套世界任意两个单元状态之间的池因子严格为正。

源码说明：Pools are bounded by `V`, so every pool change has positive multiplicity.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (l l' : W.Cell), 0 < W.poolFactor l l'
```

## CGB.NestedWorld.poolCond_self

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L432)

解释：同一池值之间的指定池导率等于一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (V a : ℕ), CGB.NestedWorld.poolCond V a a = 1
```

## CGB.NestedWorld.poolCond_step

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L435)

解释：指定池导率在 m 到 m+1 之间等于 √((m+1)(V−m))。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (V m : ℕ), CGB.NestedWorld.poolCond V m (m + 1) = √((↑m + 1) * (↑V - ↑m))
```

## CGB.NestedWorld.mass_action_up

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L442)

解释：池未满时，指定向上池导率与自由能指数因子的乘积化简为 (V−m)exp(−D/2)。

源码说明：**Mass action** from 7.1: with the entropic pool energy `cellF`, the lumped conductance makes a one-quantum gain of a pool with `m` quanta fire at a rate `∝ V - m` (its free places) ...

形式陈述：

```lean
∀ (V m : ℕ),
  m < V →
    ∀ (D : ℝ),
      CGB.NestedWorld.poolCond V m (m + 1) *
          Real.exp (-(CGB.Generation.cellF V (m + 1) - CGB.Generation.cellF V m + D) / 2) =
        (↑V - ↑m) * Real.exp (-D / 2)
```

## CGB.NestedWorld.mass_action_down

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L458)

解释：池未满时，指定向下池导率与自由能指数因子的乘积化简为 (m+1)exp(−D/2)。

源码说明：... and a one-quantum loss of a pool with `m + 1` quanta at a rate `∝ m + 1` (its quanta).

形式陈述：

```lean
∀ (V m : ℕ),
  m < V →
    ∀ (D : ℝ),
      CGB.NestedWorld.poolCond V (m + 1) m *
          Real.exp (-(CGB.Generation.cellF V m - CGB.Generation.cellF V (m + 1) + D) / 2) =
        (↑m + 1) * Real.exp (-D / 2)
```

## CGB.NestedWorld.uniport_selectivity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L479)

解释：在两池均未满且反应系数为正时，指定单池转运的正反速率比等于 (m+1)/(V−q)·exp(−D)。

源码说明：**Uniport selectivity is fixed by free energy**: a reversible exchange that moves one food quantum into the body (inner energy change `D`, food energy included) fills at food `m + 1` and drains back at food `q`; the ratio of the two mass-action rates is `(m + 1) / (V - q) · e^{-D}`, whatever its conductance `κ`. Selection can make the door slow or fast, never selective.

形式陈述：

```lean
∀ (V m q : ℕ),
  m < V →
    q < V →
      ∀ {κ : ℝ},
        0 < κ →
          ∀ (D : ℝ),
            κ * CGB.NestedWorld.poolCond V (m + 1) m *
                  Real.exp (-(CGB.Generation.cellF V m - CGB.Generation.cellF V (m + 1) + D) / 2) /
                (κ * CGB.NestedWorld.poolCond V q (q + 1) *
                  Real.exp (-(CGB.Generation.cellF V (q + 1) - CGB.Generation.cellF V q + -D) / 2)) =
              (↑m + 1) / (↑V - ↑q) * Real.exp (-D)
```

## CGB.NestedWorld.antiport_selectivity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L497)

解释：在池值均未满且反应系数为正时，指定双池反向转运的正反速率比等于所列占据因子乘 exp(−D)。

源码说明：**Antiport selectivity**: the same exchange coupled to the waste pool (one food quantum in, one waste quantum out, inner energy change `D`) fills at food `m + 1`, waste `w` and drains at food `q`, waste `z + 1`; the ratio is `(m + 1)(V - w) / ((V - q)(z + 1)) · e^{-D}`. Where the world keeps waste low (`z` small), the door drains `~ V / (z + 1)` times less than a uniport with the same `D`: dissipating the food-to-waste step at the door makes it one-way.

形式陈述：

```lean
∀ (V m q w z : ℕ),
  m < V →
    q < V →
      w < V →
        z < V →
          ∀ {κ : ℝ},
            0 < κ →
              ∀ (D : ℝ),
                κ * (CGB.NestedWorld.poolCond V (m + 1) m * CGB.NestedWorld.poolCond V w (w + 1)) *
                      Real.exp
                        (-(CGB.Generation.cellF V m - CGB.Generation.cellF V (m + 1) +
                              (CGB.Generation.cellF V (w + 1) - CGB.Generation.cellF V w + D)) /
                          2) /
                    (κ * (CGB.NestedWorld.poolCond V q (q + 1) * CGB.NestedWorld.poolCond V (z + 1) z) *
                      Real.exp
                        (-(CGB.Generation.cellF V (q + 1) - CGB.Generation.cellF V q +
                              (CGB.Generation.cellF V z - CGB.Generation.cellF V (z + 1) + -D)) /
                          2)) =
                  (↑m + 1) * (↑V - ↑w) / ((↑V - ↑q) * (↑z + 1)) * Real.exp (-D)
```

## CGB.NestedWorld.apply_content

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L556)

解释：若二体规则输入输出内容和相等，则任意方向和身份条件下应用该规则保持两端内容总和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} [inst : DecidableEq Λi] {cIn : Λi → ℕ} (s : CGB.RuleSpec Λi),
  cIn s.a.1 + cIn s.a.2 = cIn s.b.1 + cIn s.b.2 →
    ∀ (dir : Bool) (i j : ℕ) (p : Λi × Λi), cIn (s.apply dir i j p).1 + cIn (s.apply dir i j p).2 = cIn p.1 + cIn p.2
```

## CGB.NestedWorld.sum_pair_update

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L569)

解释：若两个不同位置的替换前后 g 值之和相等，则更新两个位置保持整个有限内部状态的 g 值总和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} [DecidableEq Λi] {n : ℕ} [inst : Fintype (Fin n)] (g : Λi → ℕ) {u v : Fin n},
  u ≠ v →
    ∀ (a b : Λi) (y : Fin n → Λi),
      g a + g b = g (y u) + g (y v) → ∑ w, g (Function.update (Function.update y u a) v b w) = ∑ w, g (y w)
```

## CGB.NestedWorld.pair_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L579)

解释：两个不同内部位置先被替换，再用原来的对应值替换，可以恢复原内部状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} [DecidableEq Λi] {n : ℕ} {u v : Fin n},
  u ≠ v →
    ∀ (y : Fin n → Λi) (p q : Λi × Λi),
      q = (y u, y v) →
        Function.update (Function.update (Function.update (Function.update y u p.1) v p.2) u q.1) v q.2 = y
```

## CGB.NestedWorld.fuelAct_content

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L590)

解释：若燃料规则前后的内部内容之和相等，则指定燃料操作保持整个内部状态的内容总和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 = W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2 →
    ∀ (e d : Fin n) (dir : Bool) (y : Fin n → Λi), ∑ w, W.contentIn (W.fuelAct e d dir y w) = ∑ w, W.contentIn (y w)
```

## CGB.NestedWorld.innerAct_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L599)

解释：指定内部二体操作确实改变状态时，切换方向再次操作可恢复原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi),
  W.innerAct k u v dir y ≠ y → W.innerAct k u v (!dir) (W.innerAct k u v dir y) = y
```

## CGB.NestedWorld.fuelAct_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L618)

解释：指定燃料操作确实改变内部状态时，切换方向再次操作可恢复原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (e d : Fin n) (dir : Bool) (y : Fin n → Λi),
  W.fuelAct e d dir y ≠ y → W.fuelAct e d (!dir) (W.fuelAct e d dir y) = y
```

## CGB.NestedWorld.gateCount_update

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L637)

解释：更新一个不属于门页面的位置不会改变门计数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) {u : Fin n},
  W.pageIn u ≠ W.gatePage → ∀ (b : Λi) (y : Fin n → Λi), W.gateCount (Function.update y u b) = W.gateCount y
```

## CGB.NestedWorld.innerAct_delta

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L652)

解释：在规则表平衡且内部操作有效改变状态时，内部内容总和的变化与两个池的方向变化满足所列守恒等式。

源码说明：A pair rule that fired moved the content the pools gave up into the body (or back).

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.SpecBalanced →
    ∀ (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi),
      W.innerAct k u v dir y ≠ y →
        ↑(∑ w, W.contentIn (W.innerAct k u v dir y w)) + (W.sdir k dir).1 + (W.sdir k dir).2 = ↑(∑ w, W.contentIn (y w))
```

## CGB.NestedWorld.gateCount_innerAct

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L690)

解释：若内部规则的两个身份页均不同于门页，则该规则的内部操作保持门计数。

源码说明：A pair rule away from the membrane page leaves the membrane as it is.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi),
  (W.spec k).i ≠ W.gatePage → (W.spec k).j ≠ W.gatePage → W.gateCount (W.innerAct k u v dir y) = W.gateCount y
```

## CGB.NestedWorld.c1_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L715)

解释：指定嵌套世界的单体事件电导非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (r : CGB.NestedWorld.NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell), 0 ≤ W.c1 r i l
```

## CGB.NestedWorld.c2_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L727)

解释：指定嵌套世界的双体事件电导非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (r : CGB.NestedWorld.NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell),
  0 ≤ W.c2 r i j p
```

## CGB.NestedWorld.content_p1a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L761)

解释：内部状态与头部类型有限时，嵌套单元内容有有限子水平集、统一上界和唯一零态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) [Fintype Λi] [Fintype Hd],
  (∀ (m : ℕ), {l | W.content l ≤ m}.Finite) ∧ (∃ K, ∀ (l : W.Cell), W.content l ≤ K) ∧ ∃! l, W.content l = 0
```

## CGB.NestedWorld.not_bid

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L791)

解释：位置标识不属于指定边界页集合时，其按九分组的商既不是一也不是二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {i : ℕ}, i ∉ CGB.NestedWorld.bidW → i / 9 ≠ 1 ∧ i / 9 ≠ 2
```

## CGB.NestedWorld.sum_update_nat

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L795)

解释：替换单个内部位置后，内容总和等于原总和减旧位置内容加新位置内容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} [DecidableEq Λi] {n : ℕ} (g : Λi → ℕ) (u : Fin n) (b : Λi) (y : Fin n → Λi),
  ↑(∑ w, g (Function.update y u b w)) = ∑ w, ↑(g (y w)) - ↑(g (y u)) + ↑(g b)
```

## CGB.NestedWorld.cons1

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L805)

解释：若规则表满足 SpecBalanced，且单体事件所在身份页不属于世界边界页，则该事件保持单元总内容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.SpecBalanced →
    ∀ (r : CGB.NestedWorld.NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell),
      i ∉ CGB.NestedWorld.bidW → W.content (W.eff1 r i l) = W.content l
```

## CGB.NestedWorld.cons2

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L878)

解释：若燃料规则保持内部内容和且分裂规则满足给定内容关系，则任意指定双体事件保持两单元总内容之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 = W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2 →
    (∀ (rb : Fin nB), W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1) →
      ∀ (r : CGB.NestedWorld.NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell),
        W.content (W.eff2 r i j p).1 + W.content (W.eff2 r i j p).2 = W.content p.1 + W.content p.2
```

## CGB.NestedWorld.eff1_xch

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L962)

解释：指定单体交换事件的作用等于按方向选择输入、输出状态后的 xchEff 操作。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (j : Fin nX) (u : Fin n) (dir : Bool) (i : ℕ) (l : W.Cell),
  W.eff1 (Sum.inr (Sum.inl (j, u, dir))) i l =
    W.xchEff (W.xpage j) (W.xfood j) (if dir = true then W.xb j else W.xa j) (if dir = true then W.xa j else W.xb j) u l
```

## CGB.NestedWorld.xchEff_gate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L969)

解释：若交换页不同于门页，则指定交换操作保持所在世界单元的门计数。

源码说明：An exchange through a page other than the membrane leaves the membrane as it is.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) {pg : ℕ},
  pg ≠ W.gatePage →
    ∀ (food : Bool) (a b : Λi) (u : Fin n) (l : W.Cell), W.cellGate (W.xchEff pg food a b u l) = W.cellGate l
```

## CGB.NestedWorld.xchEff_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L982)

解释：指定交换操作确实改变单元状态时，交换输入输出类型再次操作可恢复原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (pg : ℕ) (food : Bool) (a b : Λi) (u : Fin n) (l : W.Cell),
  W.xchEff pg food a b u l ≠ l → W.xchEff pg food b a u (W.xchEff pg food a b u l) = l
```

## CGB.NestedWorld.back1

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1003)

解释：在指定单体事件有效改变状态时，配对反向事件恢复原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (r : CGB.NestedWorld.NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell),
  W.eff1 r i l ≠ l → W.eff1 (CGB.NestedWorld.rev1 r) i (W.eff1 r i l) = l
```

## CGB.NestedWorld.eff2_diff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1086)

解释：指定双体扩散事件的作用等于按方向选择正一或负一位移量的 diffEff 操作。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (food dir : Bool) (i j : ℕ) (p : W.Cell × W.Cell),
  W.eff2 (Sum.inl (food, dir)) i j p = W.diffEff food (if dir = true then 1 else -1) p
```

## CGB.NestedWorld.diffEff_back

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1090)

解释：指定池扩散操作确实改变状态时，反向移动相同量可恢复原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (food : Bool) (δ : ℤ) (p : W.Cell × W.Cell),
  W.diffEff food δ p ≠ p → W.diffEff food (-δ) (W.diffEff food δ p) = p
```

## CGB.NestedWorld.back2

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1111)

解释：在指定双体事件有效改变状态时，配对反向事件恢复原状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (r : CGB.NestedWorld.NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell),
  W.eff2 r i j p ≠ p → W.eff2 (CGB.NestedWorld.rev2 r) i j (W.eff2 r i j p) = p
```

## CGB.NestedWorld.div_gate

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1201)

解释：若指定分裂页不同于门页，则该分裂事件保持第一世界单元的门计数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (rb : Fin nB),
  W.divPage rb ≠ W.gatePage →
    ∀ (S : Finset (Fin n)) (h2 : Hd) (dir : Bool) (i j : ℕ) (p : W.Cell × W.Cell),
      W.cellGate (W.eff2 (Sum.inr (Sum.inr (S, rb, h2, dir))) i j p).1 = W.cellGate p.1
```

## CGB.NestedWorld.cellGate_inner

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1214)

解释：若内部规则的两个身份页均不同于门页，则该单体内部事件不改变所在世界单元的门计数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (k : Fin nR) (u v : Fin n) (dir : Bool) (i : ℕ) (l : W.Cell),
  (W.spec k).i ≠ W.gatePage →
    (W.spec k).j ≠ W.gatePage → W.cellGate (W.eff1 (Sum.inl (k, u, v, dir)) i l) = W.cellGate l
```

## CGB.NestedWorld.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1235)

解释：在规则平衡、门页分离、燃料与分裂内容关系等定理所列前提下，指定嵌套晶格介质满足 Axioms。

源码说明：**T-E2 (an organism nested in its world is admitted)**: the joint medium of organisms and their world satisfies the six axiom clauses as soon as the organism's inner pair rules and its fuel rule conserve content and no exchange goes through the membrane page itself. Sensing, eating, excreting (exchange with the cell's pools), tumbling, diffusion, driven movement and the world's own sources and sinks need no further obligation.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) [inst_2 : Fintype Λi] [inst_3 : Fintype Hd] {U : Type u_3} [inst_4 : Fintype U]
  [inst_5 : DecidableEq U] (adj : U → U → Prop) (hsymm : ∀ (u v : U), adj u v → adj v u) (hirr : ∀ (u : U), ¬adj u u)
  (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd),
  W.SpecBalanced →
    W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 = W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2 →
      (∀ (j : Fin nX), W.xgated j = true → W.xpage j ≠ W.gatePage) →
        (∀ (k : Fin nR), W.sgated k = true → (W.spec k).i ≠ W.gatePage ∧ (W.spec k).j ≠ W.gatePage) →
          (∀ (rb : Fin nB), W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1) →
            (∀ (rb : Fin nB), W.divPage rb ≠ W.gatePage) → (W.lattice adj hsymm hirr page y0 h0).medium.Axioms
```

## CGB.NestedWorld.orgIn_congr

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1324)

解释：两个世界单元包含相同个体字段时，它们的个体内部内容相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) {l l' : W.Cell}, l'.2.2 = l.2.2 → W.orgIn l' = W.orgIn l
```

## CGB.NestedWorld.orgIn_inner

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1332)

解释：在规则表平衡且指定内部规则对两个外部池的净变化之和为零时，该单体事件保持主体内部内容。

源码说明：A pair rule that does not couple to the pools (`sdf + sdw = 0`) keeps the organism's content.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.SpecBalanced →
    ∀ (k : Fin nR) (u v : Fin n) (dir : Bool) (i : ℕ) (l : W.Cell),
      W.sdf k + W.sdw k = 0 → W.orgIn (W.eff1 (Sum.inl (k, u, v, dir)) i l) = W.orgIn l
```

## CGB.NestedWorld.eff1_supply_bug

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1352)

解释：指定单体供给事件不改变世界单元中可选主体字段。

源码说明：The world's supply rules never touch the organism.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (food dir : Bool) (i : ℕ) (l : W.Cell),
  (W.eff1 (Sum.inr (Sum.inr (Sum.inr (Sum.inl (food, dir))))) i l).2.2 = l.2.2
```

## CGB.NestedWorld.intake_is_exchange

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1362)

解释：平衡规则下，单体事件使已有主体的内容增加时，该事件是指定边界交换或带池变化的内部规则。

源码说明：**Only events across the boundary feed the organism**: every single-cell event that raises the content held by the organism is a pool exchange or a pair rule coupled to the pools; a gated one has conductance proportional to the intact membrane.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.SpecBalanced →
    ∀ (r : CGB.NestedWorld.NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell),
      l.2.2 ≠ none →
        W.orgIn l < W.orgIn (W.eff1 r i l) →
          (∃ j u dir,
              r = Sum.inr (Sum.inl (j, u, dir)) ∧
                W.c1 r i l =
                  (W.xκ j * if W.xgated j = true then ↑(W.cellGate l) else 1) * W.poolFactor l (W.eff1 r i l)) ∨
            ∃ k u v dir,
              r = Sum.inl (k, u, v, dir) ∧
                W.sdf k + W.sdw k ≠ 0 ∧
                  W.c1 r i l =
                    ((W.spec k).κ * if W.sgated k = true then ↑(W.cellGate l) else 1) * W.poolFactor l (W.eff1 r i l)
```

## CGB.NestedWorld.dead_intake_is_physics

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1401)

解释：在规则平衡、单元含主体及门计数为零时，若单体事件使主体内部内容增加，则其电导为零或事件属于所列非门控交换或非门控内部规则分支。

源码说明：**Death** (`Part.CanTakeIn` fails through the organism's own doing): with no intact membrane unit, every event that would raise the organism's content has rate zero, except an ungated exchange: physics the organism does not control (the reverse of a leak, membrane re-forming from the cell's pools). Every rate of the medium is a positive multiple of `exp`, so `Part.Recoverable` alone never fails; without the membrane factor, intake would not depend on the body at all, structure would have no use and selection could drop its upkeep.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.SpecBalanced →
    ∀ (r : CGB.NestedWorld.NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell),
      l.2.2 ≠ none →
        W.cellGate l = 0 →
          W.orgIn l < W.orgIn (W.eff1 r i l) →
            W.c1 r i l = 0 ∨
              (∃ j u dir, r = Sum.inr (Sum.inl (j, u, dir)) ∧ W.xgated j = false) ∨
                ∃ k u v dir, r = Sum.inl (k, u, v, dir) ∧ W.sgated k = false
```

## CGB.NestedWorld.dead_output_is_physics

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1420)

解释：当门计数为零且单体事件使主体内部内容减少时，其电导为零或事件属于定理列出的非门控交换、非门控内部规则或溶解分支。

源码说明：**A dead body does not give itself away**: every single-cell event that lowers the content held by the organism is a pool exchange; with no intact membrane the gated ones have rate zero, so what leaves a dead body leaves only by an ungated exchange (physics, a leak). With G26c (`corpses_must_go`), a world of endless mortal lives needs such leaks, or organisms that take dead bodies apart (decomposers, events this table does not yet have).

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB),
  W.SpecBalanced →
    ∀ (r : CGB.NestedWorld.NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell),
      W.cellGate l = 0 →
        W.orgIn (W.eff1 r i l) < W.orgIn l →
          W.c1 r i l = 0 ∨
            (∃ j u dir, r = Sum.inr (Sum.inl (j, u, dir)) ∧ W.xgated j = false) ∨
              (∃ k u v dir, r = Sum.inl (k, u, v, dir) ∧ W.sdf k + W.sdw k ≠ 0 ∧ W.sgated k = false) ∨
                ∃ hd yd, r = Sum.inr (Sum.inr (Sum.inr (Sum.inr (hd, yd, false))))
```

## CGB.NestedWorld.frees_only_by_dissolving

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1461)

解释：在该规则表中，若单体事件将非空主体字段变为空，则该事件属于指定正向溶解分支。

源码说明：**Only dissolution empties a place** (with G26c `corpses_must_go`, why the table needs it): every single-cell event that turns a cell holding a body into a cell without one is the dissolution of a body. (A move only carries the body to the next cell.) Without this event no place ever loses its body, and a world of endless mortal lives fills with corpses. Only a body without intact membrane dissolves (its conductance is zero otherwise), with everything it holds; by P2b the reverse exists too: a dead body forming from waste, suppressed by the entropy of gathering `B` and more quanta.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (r : CGB.NestedWorld.NR1 Λi Hd n nR nX) (i : ℕ) (l : W.Cell),
  l.2.2 ≠ none → (W.eff1 r i l).2.2 = none → ∃ hd yd, r = Sum.inr (Sum.inr (Sum.inr (Sum.inr (hd, yd, false))))
```

## CGB.NestedWorld.births_only_by_division

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1488)

解释：在该嵌套规则表中，若一次双体事件使两个位置的主体数增加，则该事件属于正向分裂分支。

源码说明：**Births come only from division** (the dual of `frees_only_by_dissolving`): a neighbour event after which the pair holds more bodies is a division. Diffusion keeps the bodies where they are, a move carries one body, a fusion removes one. With division, the endless stream of mortal lives of G26c can be fed from inside the world.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (r : CGB.NestedWorld.NR2 Hd n nB) (i j : ℕ) (p : W.Cell × W.Cell),
  W.bodies p.1 + W.bodies p.2 < W.bodies (W.eff2 r i j p).1 + W.bodies (W.eff2 r i j p).2 →
    ∃ S rb h2, r = Sum.inr (Sum.inr (S, rb, h2, false))
```

## CGB.NestedWorld.partner_exit_blocked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1522)

解释：若内部所有对应身份页的位置都不具备所需伙伴状态，则指定内部二体操作保持原内部状态。

源码说明：A pair rule whose partner state is absent leaves the organism as it is.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (k : Fin nR) (u v : Fin n) (dir : Bool) (y : Fin n → Λi),
  (∀ (w : Fin n), W.pageIn w = (W.spec k).j → y w ≠ (if dir = true then (W.spec k).b else (W.spec k).a).2) →
    W.innerAct k u v dir y = y
```

## CGB.NestedWorld.gated_exit_blocked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1537)

解释：若交换规则被标记为门控且世界单元门计数为零，则对应单体事件的电导为零。

源码说明：A gated exchange has no conductance without an intact membrane.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (j : Fin nX),
  W.xgated j = true →
    ∀ (u : Fin n) (dir : Bool) (i : ℕ) (l : W.Cell), W.cellGate l = 0 → W.c1 (Sum.inr (Sum.inl (j, u, dir))) i l = 0
```

## CGB.NestedWorld.poolShift_one

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1541)

解释：容量未满的有界池可以成功增加一单位，且新值为原值加一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {V : ℕ} (w : Fin (V + 1)), ↑w < V → ∃ w', CGB.poolShift w 1 = some w' ∧ ↑w' = ↑w + 1
```

## CGB.NestedWorld.leak_bound

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1553)

解释：在内部势能有界、指定交换非门控及其余定理条件下，该单体事件的电导与指数因子乘积满足给定下界。

源码说明：**Leak bound**: an ungated exchange that moves one quantum from a unit into the waste pool fires at rate at least `κ * exp (-Φ)` whenever the unit holds that quantum and the pool is not full, whatever the rest of the organism and the reservoir potential `μ` (the exchange conserves the cell's content, so `μ` drops out). The bound depends only on physics; under mass action the rate is `κ (V - w) e^{-ΔE_in/2}`, at least one free place.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (Φ : ℝ),
  (∀ (p : ℕ) (s : Λi), |W.φIn p s| ≤ Φ) →
    ∀ (j : Fin nX),
      W.xgated j = false →
        W.xfood j = false →
          W.contentIn (W.xa j) = W.contentIn (W.xb j) + 1 →
            ∀ (u : Fin n) (i : ℕ) (f w : Fin (W.V + 1)) (y : Fin n → Λi) (h : Hd),
              W.pageIn u = W.xpage j ∧ y u = W.xa j →
                ↑w < W.V →
                  ∀ (μ : ℝ),
                    W.xκ j * Real.exp (-Φ) ≤
                      W.c1 (Sum.inr (Sum.inl (j, u, false))) i (f, w, some (y, h)) *
                        Real.exp
                          (-(W.φ i (W.eff1 (Sum.inr (Sum.inl (j, u, false))) i (f, w, some (y, h))) -
                                  W.φ i (f, w, some (y, h)) -
                                μ *
                                  (↑(W.content (W.eff1 (Sum.inr (Sum.inl (j, u, false))) i (f, w, some (y, h)))) -
                                    ↑(W.content (f, w, some (y, h))))) /
                            2)
```

## CGB.NestedWorld.intake_eff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1621)

解释：若指定食品交换规则将内部内容加一、内部页和初态匹配且外部食物池至少有一单位，则该事件使外部池减一并按规则更新内部位置。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (j : Fin nX),
  W.xfood j = true →
    W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1 →
      ∀ (u : Fin n) (i : ℕ) (y : Fin n → Λi) (h : Hd),
        W.pageIn u = W.xpage j ∧ y u = W.xa j →
          ∀ (f w : Fin (W.V + 1)) (hf1 : 1 ≤ ↑f),
            W.eff1 (Sum.inr (Sum.inl (j, u, false))) i (f, w, some (y, h)) =
              (⟨↑f - 1, ⋯⟩, w, some (Function.update y u (W.xb j), h))
```

## CGB.NestedWorld.intake_content

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1636)

解释：若交换目标状态的内容比来源状态多一，且选定位置处于来源状态，则替换该位置使内部内容总和增加一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [DecidableEq Λi] [DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) (j : Fin nX),
  W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1 →
    ∀ (u : Fin n) (y : Fin n → Λi),
      y u = W.xa j → ∑ v, W.contentIn (Function.update y u (W.xb j) v) = ∑ v, W.contentIn (y v) + 1
```

## CGB.NestedWorld.metabolizes_of_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1654)

解释：若指定门控食物交换满足正系数、内容增一、页匹配、门计数和容量等条件，则所构造主体部分满足 Metabolizes。

源码说明：**Nested metabolism** (`Part.Metabolizes`) for every organism whose table has a gated food intake of one quantum, in a state with an intact membrane and a unit ready to take in.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) [Fintype Λi] [Fintype Hd] {U : Type u_3} [inst_4 : Fintype U]
  [inst_5 : DecidableEq U] (adj : U → U → Prop) (hsymm : ∀ (u v : U), adj u v → adj v u) (hirr : ∀ (u : U), ¬adj u u)
  (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd) (u0 : U) (j : Fin nX),
  W.xgated j = true →
    W.xfood j = true →
      0 < W.xκ j →
        W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1 →
          ∀ (u : Fin n) (y : Fin n → Λi) (h : Hd),
            W.pageIn u = W.xpage j ∧ y u = W.xa j →
              0 < W.gateCount y →
                1 ≤ W.V → CGB.Part.Metabolizes (W.lattice adj hsymm hirr page y0 h0).medium (W.orgPart u0)
```

## CGB.NestedWorld.senses_of_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1681)

解释：若指定门控食物交换满足正系数、内容增一、页匹配、正门计数及容量至少二等条件，则所构造主体部分满足 Senses。

源码说明：**Nested sensing** (`Part.Senses`): the same intake event, from the same organism, fires at different rates when only the cell's food differs (one quantum or two).

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) [Fintype Λi] [Fintype Hd] {U : Type u_3} [inst_4 : Fintype U]
  [inst_5 : DecidableEq U] (adj : U → U → Prop) (hsymm : ∀ (u v : U), adj u v → adj v u) (hirr : ∀ (u : U), ¬adj u u)
  (page : U → ℕ) (y0 : Fin n → Λi) (h0 : Hd) (u0 : U) (j : Fin nX),
  W.xgated j = true →
    W.xfood j = true →
      0 < W.xκ j →
        W.contentIn (W.xb j) = W.contentIn (W.xa j) + 1 →
          ∀ (u : Fin n) (y : Fin n → Λi) (h : Hd),
            W.pageIn u = W.xpage j ∧ y u = W.xa j →
              0 < W.gateCount y → 2 ≤ W.V → CGB.Part.Senses (W.lattice adj hsymm hirr page y0 h0).medium (W.orgPart u0)
```

## CGB.NestedWorld.ringAdj_symm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1753)

解释：指定环形世界的相邻关系对称。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (L : ℕ) (u v : Fin L), CGB.NestedWorld.ringAdj L u v → CGB.NestedWorld.ringAdj L v u
```

## CGB.NestedWorld.torusAdj_symm

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1761)

解释：指定二维环面世界的相邻关系对称。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (G : ℕ) (u v : Fin G × Fin G), CGB.NestedWorld.torusAdj G u v → CGB.NestedWorld.torusAdj G v u
```

## CGB.NestedWorld.axioms_torus

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1787)

解释：在嵌套规则平衡和门页、分裂等前提下，以指定二维环面邻接构造的介质满足 Axioms。

源码说明：**The two-dimensional world is a model of the axioms**: organisms on a torus, food pellets from the world's boundary, diffusion, waste sinks, driven movement in four headings.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) [inst_2 : Fintype Λi] [inst_3 : Fintype Hd] (G : ℕ) (kind : Fin G × Fin G → ℕ)
  (y0 : Fin n → Λi) (h0 : Hd),
  W.SpecBalanced →
    W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 = W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2 →
      (∀ (j : Fin nX), W.xgated j = true → W.xpage j ≠ W.gatePage) →
        (∀ (k : Fin nR), W.sgated k = true → (W.spec k).i ≠ W.gatePage ∧ (W.spec k).j ≠ W.gatePage) →
          (∀ (rb : Fin nB), W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1) →
            (∀ (rb : Fin nB), W.divPage rb ≠ W.gatePage) →
              (W.lattice (CGB.NestedWorld.torusAdj G) ⋯ ⋯ (CGB.NestedWorld.torusPage G kind) y0 h0).medium.Axioms
```

## CGB.NestedWorld.axioms_ring

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1799)

解释：在嵌套规则平衡和门页、分裂等前提下，以指定环形邻接构造的介质满足 Axioms。

源码说明：The one-dimensional ring is a model too.

形式陈述：

```lean
∀ {Λi : Type u_1} {Hd : Type u_2} [inst : DecidableEq Λi] [inst_1 : DecidableEq Hd] {n nR nX nB : ℕ}
  (W : CGB.NestedWorld Λi Hd n nR nX nB) [inst_2 : Fintype Λi] [inst_3 : Fintype Hd] (L : ℕ) (page : Fin L → ℕ)
  (y0 : Fin n → Λi) (h0 : Hd),
  W.SpecBalanced →
    W.contentIn W.fuel.a.1 + W.contentIn W.fuel.a.2 = W.contentIn W.fuel.b.1 + W.contentIn W.fuel.b.2 →
      (∀ (j : Fin nX), W.xgated j = true → W.xpage j ≠ W.gatePage) →
        (∀ (k : Fin nR), W.sgated k = true → (W.spec k).i ≠ W.gatePage ∧ (W.spec k).j ≠ W.gatePage) →
          (∀ (rb : Fin nB), W.contentIn (W.divFrom rb) = W.contentIn (W.divTo rb) + 1) →
            (∀ (rb : Fin nB), W.divPage rb ≠ W.gatePage) →
              (W.lattice (CGB.NestedWorld.ringAdj L) ⋯ ⋯ page y0 h0).medium.Axioms
```
