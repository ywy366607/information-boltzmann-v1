# SpongeContractInstance

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/SpongeContractInstance.lean) · [返回导航](../README.md)

模块源码说明：Coarse admission candidate for a sponge-like continuous body This is a structural interface witness only.

## CGB.SpongeContractInstance.port_factorizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SpongeContractInstance.lean#L54)

解释：海绵合约实例的端口协议在时刻零精确解耦，传感器信号端口与资源端口互不串扰。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.RenewableLearning.Signal) (r : Bool),
  CGB.SpongeContractInstance.coarseAdmission.ports.sensor
        (CGB.PortContracts.protocol CGB.SpongeContractInstance.coarseAdmission.ports (fun x => s) (fun x => r) 0) =
      s ∧
    CGB.SpongeContractInstance.coarseAdmission.ports.resource
        (CGB.PortContracts.protocol CGB.SpongeContractInstance.coarseAdmission.ports (fun x => s) (fun x => r) 0) =
      r
```

## CGB.SpongeContractInstance.reserve_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/SpongeContractInstance.lean#L61)

解释：海绵储库合约满足秩降与空态保持性质：空态秩为零且单步保持为空，非空态单步步进后储量秩严格递减直至排空。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
(∀ (x : CGB.SpongeContractInstance.State),
    CGB.SpongeContractInstance.coarseAdmission.reserve.empty x →
      CGB.SpongeContractInstance.coarseAdmission.reserve.rank x = 0) ∧
  (∀ (x : CGB.SpongeContractInstance.State),
      CGB.SpongeContractInstance.coarseAdmission.reserve.empty x →
        CGB.SpongeContractInstance.coarseAdmission.reserve.empty
          (CGB.SpongeContractInstance.coarseAdmission.reserve.step x)) ∧
    ∀ (x : CGB.SpongeContractInstance.State),
      ¬CGB.SpongeContractInstance.coarseAdmission.reserve.empty x →
        CGB.SpongeContractInstance.coarseAdmission.reserve.rank
            (CGB.SpongeContractInstance.coarseAdmission.reserve.step x) <
          CGB.SpongeContractInstance.coarseAdmission.reserve.rank x
```
