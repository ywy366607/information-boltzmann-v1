# MaterialSourceControls

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean) · [返回导航](../README.md)

模块源码说明：Finite supply / exchange / exhaustion control A mathematical carrier-interpretation witness, not a bacterial/sponge model or a capability experiment.

## CGB.MaterialSourceControls.realized

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L62)

解释：离散物料源控制系统构成了物理介质与粗粒化时钟之间的合法实现关系，满足状态投影与时钟对应。

源码说明：Explicit source realization: no changed body dynamics, no extra physical ticks, and no nonempty-body false positive standing in for actual replacement.

形式陈述：

```lean
∀ (cut : Bool),
  CGB.MaterialSource.Realization CGB.MaterialSourceControls.physical (CGB.MaterialSourceControls.coarse cut)
    CGB.MaterialSourceControls.owner CGB.MaterialSourceControls.project (fun z => CGB.TurnoverControls.body.total z.1)
    CGB.MaterialSourceControls.declared CGB.MaterialSourceControls.declared CGB.MaterialSourceControls.world.clock
```

## CGB.MaterialSourceControls.supply_exchange_exhaustion

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L100)

解释：描述两步物料补给交换与耗竭动力学：物料从外部移入源、源转移至机体、机体排至汇后，源处于耗竭态而机体守恒维持单位占用。

源码说明：One arrival, then one equal input/output replacement, then no accessible source. Body material count stays one throughout. This is not a full life witness.

形式陈述：

```lean
CGB.MaterialSource.moved (CGB.MaterialSourceControls.owner 0) (CGB.MaterialSourceControls.owner 1)
      CGB.MaterialSource.Place.outside CGB.MaterialSource.Place.source =
    1 ∧
  CGB.MaterialSource.moved (CGB.MaterialSourceControls.owner 0) (CGB.MaterialSourceControls.owner 1)
        CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body =
      0 ∧
    CGB.MaterialSource.moved (CGB.MaterialSourceControls.owner 1) (CGB.MaterialSourceControls.owner 2)
          CGB.MaterialSource.Place.source CGB.MaterialSource.Place.body =
        1 ∧
      CGB.MaterialSource.moved (CGB.MaterialSourceControls.owner 1) (CGB.MaterialSourceControls.owner 2)
            CGB.MaterialSource.Place.body CGB.MaterialSource.Place.sink =
          1 ∧
        CGB.MaterialSource.count (CGB.MaterialSourceControls.owner 2) CGB.MaterialSource.Place.source = 0 ∧
          ∀ (p : CGB.MaterialSourceControls.Phase),
            CGB.MaterialSource.count (CGB.MaterialSourceControls.owner p) CGB.MaterialSource.Place.body = 1
```

## CGB.MaterialSourceControls.no_future_intake_after_exhaustion

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L118)

解释：外部物料源一旦耗竭，在后续任意时间步都不会再产生新的摄入事件，粗粒化系统的首次命中概率恒为零。

源码说明：Reuse the existing refinement/first-hit theorem for the WHOLE future. No additional scoring implementation is introduced for the physical carrier.

形式陈述：

```lean
∀ (cut : Bool) (history : List (Option Unit)) (k : ℕ),
  (CGB.MaterialSourceControls.coarse cut).firstHit List.length Option.isSome history k
      (CGB.MaterialSourceControls.project 2) =
    0
```
