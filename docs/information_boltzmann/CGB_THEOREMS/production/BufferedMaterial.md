# BufferedMaterial

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean) · [返回导航](../README.md)

模块源码说明：Finite identity-preserving buffered material transport Twelve fixed carriers suffice for a body containing at most four units, a four-unit accessible source and at least four returned carriers.

## _private.CGBAxioms.BufferedMaterial.0.CGB.BufferedMaterial.canonical_stock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L30)

解释：对任意不超过容量4的缓冲存量，规范所有权构型均严格实现该存量分配。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ b ≤ 4, CGB.BufferedMaterial.Stock b (CGB.BufferedMaterial.canonical✝ b)
```

## _private.CGBAxioms.BufferedMaterial.0.CGB.BufferedMaterial.total

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L34)

解释：缓冲物质系统中所有空间位置（机体、源、汇与外部）的微观载体总数严格守恒为12。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (o : CGB.BufferedMaterial.Owners),
  CGB.MaterialSource.count o CGB.MaterialSource.Place.body +
          CGB.MaterialSource.count o CGB.MaterialSource.Place.source +
        CGB.MaterialSource.count o CGB.MaterialSource.Place.sink +
      CGB.MaterialSource.count o CGB.MaterialSource.Place.outside =
    12
```

## _private.CGBAxioms.BufferedMaterial.0.CGB.BufferedMaterial.select

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L44)

解释：在任意空间位置存在包含指定数量载体的合法子集以供物质传输调度。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (o : CGB.BufferedMaterial.Owners) (p : CGB.MaterialSource.Place),
  ∀ n ≤ CGB.MaterialSource.count o p, ∃ a, (∀ c ∈ a, o c = p) ∧ a.card = n
```

## CGB.BufferedMaterial.transport_exists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L54)

解释：在给定的摄入、排出与源头补充守恒约束下，必定存在保守恒的局部物质传输转移。

源码说明：Local existence with every intake/release/recharge coming from old stock. The source is replenished by exactly the amount it actually delivers.

形式陈述：

```lean
∀ (o : CGB.BufferedMaterial.Owners) (b b' i d : ℕ),
  CGB.BufferedMaterial.Stock b o →
    b ≤ 4 →
      i ≤ 4 →
        d ≤ b →
          b' + d = b + i →
            ∃ o',
              CGB.BufferedMaterial.Stock b' o' ∧
                CGB.MaterialSource.Allowed o o' ∧
                  CGB.MaterialSource.moved o o' CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body = i ∧
                    CGB.MaterialSource.moved o o' CGB.MaterialSource.Place.body CGB.MaterialSource.Place.sink = d ∧
                      CGB.MaterialSource.moved o o' CGB.MaterialSource.Place.sink CGB.MaterialSource.Place.source = i
```

## _private.CGBAxioms.BufferedMaterial.0.CGB.BufferedMaterial.available

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L145)

解释：只要离散机体执行非零跃迁，就存在满足进出通量计量的微观物质所有权转移。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [DecidableEq S] [inst_2 : Fintype E]
  {P : CGB.Interaction.Machine S ℕ E} (B : CGB.BufferedMaterial.Rules P) (t : ℕ) (r : CGB.BufferedMaterial.State B)
  (e : E) (s : S),
  0 < P.step t (CGB.BufferedMaterial.project B r) e s →
    ∃ o',
      CGB.BufferedMaterial.Stock (B.content s) o' ∧
        CGB.MaterialSource.Allowed (CGB.BufferedMaterial.owner B r) o' ∧
          CGB.MaterialSource.moved (CGB.BufferedMaterial.owner B r) o' CGB.MaterialSource.Place.source
                CGB.MaterialSource.Place.body =
              B.incoming t (CGB.BufferedMaterial.project B r) e s ∧
            CGB.MaterialSource.moved (CGB.BufferedMaterial.owner B r) o' CGB.MaterialSource.Place.body
                  CGB.MaterialSource.Place.sink =
                B.outgoing t (CGB.BufferedMaterial.project B r) e s ∧
              CGB.MaterialSource.moved (CGB.BufferedMaterial.owner B r) o' CGB.MaterialSource.Place.sink
                  CGB.MaterialSource.Place.source =
                B.incoming t (CGB.BufferedMaterial.project B r) e s
```

## CGB.BufferedMaterial.advance_project

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L160)

解释：微观物质缓冲机体的转移步在投影后严格保真原离散机体的状态步。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype E]
  {P : CGB.Interaction.Machine S ℕ E} (B : CGB.BufferedMaterial.Rules P) (t : ℕ) (r : CGB.BufferedMaterial.State B)
  (e : E) (s : S), CGB.BufferedMaterial.project B (CGB.BufferedMaterial.advance B t r e s) = s
```

## _private.CGBAxioms.BufferedMaterial.0.CGB.BufferedMaterial.lift_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L166)

解释：微观物质缓冲机体的总跃迁概率等于原系统相应跃迁核的归一化求和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] [DecidableEq S] {R : Type u_3} [inst_2 : Fintype R] [inst_3 : DecidableEq R]
  (f : R → S) (g : S → R),
  (∀ (s : S), f (g s) = s) → ∀ (v : S → ℝ), (∑ r, if r = g (f r) then v (f r) else 0) = ∑ s, v s
```

## CGB.BufferedMaterial.refines

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L196)

解释：微观带有物质存量跟踪的扩展机体严格弱模拟并细化原抽象机体。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype E]
  {P : CGB.Interaction.Machine S ℕ E} (B : CGB.BufferedMaterial.Rules P),
  CGB.Interaction.Refines (CGB.BufferedMaterial.physical B) P (CGB.BufferedMaterial.project B)
```

## CGB.BufferedMaterial.delivery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L210)

解释：在有效微观转移下，源头、机体与汇之间的物质移动量与宏观物质账本进出通量精确匹配。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype E]
  {P : CGB.Interaction.Machine S ℕ E} (B : CGB.BufferedMaterial.Rules P) (t : ℕ) (r r' : CGB.BufferedMaterial.State B)
  (e : E),
  0 < (CGB.BufferedMaterial.physical B).step t r e r' →
    CGB.MaterialSource.Allowed (CGB.BufferedMaterial.owner B r) (CGB.BufferedMaterial.owner B r') ∧
      CGB.MaterialSource.moved (CGB.BufferedMaterial.owner B r) (CGB.BufferedMaterial.owner B r')
            CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body =
          B.incoming t (CGB.BufferedMaterial.project B r) e (CGB.BufferedMaterial.project B r') ∧
        CGB.MaterialSource.moved (CGB.BufferedMaterial.owner B r) (CGB.BufferedMaterial.owner B r')
              CGB.MaterialSource.Place.body CGB.MaterialSource.Place.sink =
            B.outgoing t (CGB.BufferedMaterial.project B r) e (CGB.BufferedMaterial.project B r') ∧
          CGB.MaterialSource.moved (CGB.BufferedMaterial.owner B r) (CGB.BufferedMaterial.owner B r')
              CGB.MaterialSource.Place.sink CGB.MaterialSource.Place.source =
            B.incoming t (CGB.BufferedMaterial.project B r) e (CGB.BufferedMaterial.project B r')
```

## CGB.BufferedMaterial.realized

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L237)

解释：在离散机体上建立的显式物质传输模型完全实现了抽象缓冲物质规则公理体系。

源码说明：All original states, all supported events, and the same supplied clock.

形式陈述：

```lean
∀ {S : Type u_1} {E : Type u_2} [inst : Fintype S] [inst_1 : DecidableEq S] [inst_2 : Fintype E]
  {P : CGB.Interaction.Machine S ℕ E} (B : CGB.BufferedMaterial.Rules P) (clock : ℝ),
  0 < clock →
    CGB.MaterialSource.Realization (CGB.BufferedMaterial.physical B) P (CGB.BufferedMaterial.owner B)
      (CGB.BufferedMaterial.project B) B.content B.incoming B.outgoing clock
```
