# CGB 定理基座索引（自动生成，勿手改）

本页不是第二份定理清单，而是 Agent 的首轮检索顺序。完整声明和前提仍以 [CGB_PROVEN_INDEX.md](CGB_PROVEN_INDEX.md) 与 Lean 源码为准。

当前审计库存：**1575** 条源码定理，**118** 个生产模块。

## 最小阅读链

```text
一般公理 / 生命定义
        ↓
开放交互 + 因果状态 + 有限容量
        ↓
选择性保留 + 预测收益 + 学习更新
        ↓
主动行动 + 行动条件化预测
        ↓
具体介质、物种实例和运行实现
```

## 四层模块基座

| 层 | 首轮用途 | 定理数 | 模块 |
|---|---|---:|---|
| core | 一般对象、生命目标、开放交互和基本可观测语义。 | 96 | `CausalLearning` (16), `Foundation` (14), `Inference` (6), `Interaction` (19), `InteractionBridge` (6), `InteractionControls` (7), `Language` (1), `Life` (18), `Medium` (5), `Symmetry` (4) |
| causal | 因果压缩、有限容量、学习、收益、主动性和持续状态的生成桥。 | 355 | `Generation` (35), `GenerationActiveInference` (5), `GenerationCascadeMemory` (5), `GenerationColumnMemory` (19), `GenerationExport` (6), `GenerationGradientFilter` (35), `GenerationGrowth` (16), `GenerationInformationEnergy` (7), `GenerationLearning` (7), `GenerationLeaveRule` (3), `GenerationLinearMemory` (7), `GenerationPersistence` (6), `GenerationPredictiveHarvest` (9), `GenerationSelfProduction` (138), `InformationEconomics` (3), `LanguageLifeKernel` (54) |
| medium | 守恒、局部介质、同调、能量和具体生成介质的条件。 | 133 | `AlgebraicCore` (6), `AlgebraicHomology` (2), `EnergyBudget` (5), `EnergyControls` (4), `GenerationCRN` (10), `GenerationLattice` (14), `GenerationMedium` (2), `GenerationNestedWorld` (66), `GenerationNesting` (8), `GenerationTable` (2), `InformationThermodynamics` (3), `NoetherBridge` (8), `PortFlowHomology` (3) |
| instances | 控制、见证、物种实例、恢复协议和历史审计；首轮检索不作为生成骨架。 | 991 | `ActiveLoop` (8), `ActiveMaintenance` (21), `ActiveMaintenanceCarrier` (6), `ActiveMaintenanceEnergy` (4), `AtomicIndependence` (24), `BufferedMaterial` (10), `ConstraintAudit` (15), `ConstraintFreedom` (4), `EventChannels` (15), `GenerationAmplifier` (13), `GenerationAssociationCapacity` (15), `GenerationAxiomAudit` (2), `GenerationClosedLoopCredit` (7), `GenerationColumnRefinement` (5), `GenerationConservedMemory` (24), `GenerationContinualOptimizer` (42), `GenerationGDNLife` (76), `GenerationGraceMemory` (13), `GenerationIgnition` (5), `GenerationInformationProjection` (17), `GenerationLanguageExistence` (18), `GenerationLatchHealth` (3), `GenerationLifeGate` (9), `GenerationOptimizer` (19), `GenerationOrganismEcoliMsS0` (3), `GenerationOrganismEcoliV1` (3), `GenerationOrganismGeneratedS0` (3), `GenerationSeedMemory` (15), `GenerationSpeechTempo` (14), `GenerationSurvivalBounds` (16), `GenerationUnifiedOptimizer` (42), `GraceRecovery` (20), `GraceRecoveryAccounting` (14), `GraceRecoveryClosure` (18), `GraceRecoveryControls` (4), `LanguageLifeAdmission` (1), `LearningMaintenance` (26), `LearningMaintenanceControls` (8), `LearningMaintenanceEnergy` (4), `LifeControls` (25), `MaterialExchange` (8), `MaterialSource` (6), `MaterialSourceControls` (3), `Models` (57), `NestedLife` (2), `Nesting` (8), `OperationalLife` (6), `PortContractInstances` (3), `PortContracts` (6), `PortSeparation` (6), `RecoveryControls` (5), `RecoveryCriteria` (4), `RecoveryTransport` (3), `RenewableCandidateAudit` (9), `RenewableEncoding` (13), `RenewableLearning` (6), `RenewableLearningCarrier` (10), `RenewableLearningClosure` (19), `RenewableLearningControls` (15), `RenewableLearningEnergy` (5), `RenewableLearningLife` (15), `RenewableRecoveryCertificate` (7), `RuleEncoding` (11), `ScheduledRecovery` (3), `SensoryMaintenance` (16), `SensoryMaintenanceBoundaries` (3), `SensoryMaintenanceCarrier` (12), `SensoryMaintenanceClosure` (11), `SensoryMaintenanceLimits` (7), `SensoryMaintenanceProtocolRepairRefutation` (4), `SensoryMaintenanceRecovery` (10), `SensoryMaintenanceRepairAudit` (3), `SensoryMaintenanceRepairCandidates` (25), `SensoryMaintenanceTwoTickGlobal` (7), `SensoryMaintenanceTwoTickObstruction` (3), `SpongeContractInstance` (2), `SubconditionGraph` (37), `TurnoverControls` (15), `WorldRecovery` (15) |

## 三个信息经济学基座

这些是压缩后的共同接口，不是新增机制：

1. `InformationEconomics.causal_carrier_contract_iff`：因果状态必须同时支持未来读出和递归转移。
2. `InformationEconomics.selective_write_contract`：只有排除未来候选的信息才支付写入并占用容量。
3. `InformationEconomics.predictive_growth_contract`：登记的再投资世界中，收益由交叉熵决定并受熵上界约束。

## 使用规则

- 先读 `core`，确定对象和量词；再读 `causal`，寻找可复用桥定理。
- 只有当目标需要具体守恒介质或速率条件时才进入 `medium`。
- `instances` 是证书、反例和控制，不把它们当成新的普遍机制。
- 新定理应说明自己属于哪个基座；无法归类时先检查是否重复或缺少共同桥。
- 本页由 `scripts/build_theorem_basis.py` 从审计库存生成，审计成功后由 `audit.py` 自动刷新。
