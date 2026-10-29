# PortContracts

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/PortContracts.lean) · [返回导航](../README.md)

模块源码说明：Coarse contracts for separated sensory and resource ports Complex bodies need not expose every carrier identity.

## CGB.PortContracts.protocol_sensor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L25)

解释：复合交互协议在感官投影下精确提取出纯环境信号交互轨迹。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {T : Type u_1} {Sensor : Type u_2} {Resource : Type u_3} {I : Type u_4}
  (P : CGB.PortContracts.PortSplit I Sensor Resource) (sensor : T → Sensor) (resource : T → Resource) (t : T),
  P.sensor (CGB.PortContracts.protocol P sensor resource t) = sensor t
```

## CGB.PortContracts.protocol_resource

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L30)

解释：复合交互协议在资源投影下精确还原为单端口纯物质流动协议。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {T : Type u_1} {Sensor : Type u_2} {Resource : Type u_3} {I : Type u_4}
  (P : CGB.PortContracts.PortSplit I Sensor Resource) (sensor : T → Sensor) (resource : T → Resource) (t : T),
  P.resource (CGB.PortContracts.protocol P sensor resource t) = resource t
```

## CGB.PortContracts.admission_port_factorizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L47)

解释：准入端口合约在感官测量与资源补给之间实现完全的笛卡尔因子分解。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {I : Type u_1} {Sensor : Type u_2} {Resource : Type u_3} {S : Type u_4}
  (A : CGB.PortContracts.CoarseAdmission I Sensor Resource S) (s : Sensor) (r : Resource),
  A.ports.sensor (CGB.PortContracts.protocol A.ports (fun x => s) (fun x => r) 0) = s ∧
    A.ports.resource (CGB.PortContracts.protocol A.ports (fun x => s) (fun x => r) 0) = r
```

## CGB.PortContracts.admission_reserve_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L53)

解释：准入储库合约保证资源在有限步骤内被有效利用，且空储库单步不产生虚假利用。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {I : Type u_1} {Sensor : Type u_2} {Resource : Type u_3} {S : Type u_4}
  (A : CGB.PortContracts.CoarseAdmission I Sensor Resource S),
  (∀ (s : S), A.reserve.empty s → A.reserve.rank s = 0) ∧
    (∀ (s : S), A.reserve.empty s → A.reserve.empty (A.reserve.step s)) ∧
      ∀ (s : S), ¬A.reserve.empty s → A.reserve.rank (A.reserve.step s) < A.reserve.rank s
```

## CGB.PortContracts.recovery_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L69)

解释：在满足端口合约的相互作用下，系统的状态可恢复性在端口连接前后保持不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {R : Type u_1} {S : Type u_2} {E : Type u_3} [inst : Fintype R] [inst_1 : DecidableEq R] [inst_2 : Fintype S]
  [inst_3 : DecidableEq S] [inst_4 : Fintype E] {P : CGB.Interaction.Machine R ℕ E} {Q : CGB.Interaction.Machine S ℕ E}
  {project : R → S} {markP : ℕ → R → E → R → Bool} {markQ : ℕ → S → E → S → Bool},
  CGB.PortContracts.RecoveryRefinement P Q project markP markQ →
    ∀ (t k : ℕ) (r : R),
      CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events P) markP t k r =
        CGB.EventChannels.Kernel.firstHit (CGB.Energetics.events Q) markQ t k (project r)
```

## CGB.PortContracts.elapsed_preserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L81)

解释：端口协议作用下，系统所经历的物理流逝时间与各端口局部时钟严格对齐。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {R : Type u_1} {S : Type u_2} {E : Type u_3} [inst : Fintype R] [inst_1 : DecidableEq R] [inst_2 : Fintype S]
  [inst_3 : DecidableEq S] [inst_4 : Fintype E] {P : CGB.Interaction.Machine R ℕ E} {Q : CGB.Interaction.Machine S ℕ E}
  {project : R → S} {markP : ℕ → R → E → R → Bool} {markQ : ℕ → S → E → S → Bool},
  CGB.PortContracts.RecoveryRefinement P Q project markP markQ → ∀ (n : ℕ), ↑n / 1 = ↑n / 1
```
