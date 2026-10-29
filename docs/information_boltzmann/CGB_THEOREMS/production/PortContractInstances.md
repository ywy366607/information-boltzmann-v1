# PortContractInstances

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean) · [返回导航](../README.md)

模块源码说明：The current 24-state maintenance witness is admitted through the coarse contracts.

## CGB.PortContractInstances.split_factorizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L41)

解释：复合端口的信号与物质流在劈裂操作下分解为两个相互正交的独立信道。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : CGB.SensoryMaintenance.Stim) (r : Bool),
  CGB.PortContractInstances.sensoryResourceSplit.sensor
        (CGB.PortContracts.protocol CGB.PortContractInstances.sensoryResourceSplit (fun x => s) (fun x => r) 0) =
      s ∧
    CGB.PortContractInstances.sensoryResourceSplit.resource
        (CGB.PortContracts.protocol CGB.PortContractInstances.sensoryResourceSplit (fun x => s) (fun x => r) 0) =
      r
```

## CGB.PortContractInstances.reserve_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L46)

解释：端口合约实例的储量协议在非空态下驱动单步秩递减，满足李雅普诺夫衰减合同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
(∀ (x : CGB.PortContractInstances.State),
    CGB.PortContractInstances.reserveRank.empty x → CGB.PortContractInstances.reserveRank.rank x = 0) ∧
  (∀ (x : CGB.PortContractInstances.State),
      CGB.PortContractInstances.reserveRank.empty x →
        CGB.PortContractInstances.reserveRank.empty (CGB.PortContractInstances.reserveRank.step x)) ∧
    ∀ (x : CGB.PortContractInstances.State),
      ¬CGB.PortContractInstances.reserveRank.empty x →
        CGB.PortContractInstances.reserveRank.rank (CGB.PortContractInstances.reserveRank.step x) <
          CGB.PortContractInstances.reserveRank.rank x
```

## CGB.PortContractInstances.admission_contract

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L52)

解释：端口合约实例的准入协议满足保真度规范与资源有界性要求。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
(∀ (x : CGB.PortContractInstances.State),
    CGB.PortContractInstances.admission.reserve.empty x → CGB.PortContractInstances.admission.reserve.rank x = 0) ∧
  (∀ (x : CGB.PortContractInstances.State),
      CGB.PortContractInstances.admission.reserve.empty x →
        CGB.PortContractInstances.admission.reserve.empty (CGB.PortContractInstances.admission.reserve.step x)) ∧
    ∀ (x : CGB.PortContractInstances.State),
      ¬CGB.PortContractInstances.admission.reserve.empty x →
        CGB.PortContractInstances.admission.reserve.rank (CGB.PortContractInstances.admission.reserve.step x) <
          CGB.PortContractInstances.admission.reserve.rank x
```
