# CGB 已证定理索引（自动生成，勿手改）

由 `scripts/build_proven_index.py` 从 Lean 源码生成，并逐条核对当前审计清单。新工作开始前先检索已有机制；点击完整定理名查看源码、前提及结论。
渐进检索入口：[主题导航与逐条完整陈述](CGB_THEOREMS/README.md)；先查主题，再按模块阅读。无源码解释的条目明确标注，避免把机器猜测当作定理含义。

共 **1200** 条已审计源码定理/引理，来自 **104** 个生产模块。编译声明共 **5174** 项，含定义及自动生成项，不作为机制数量。
生成时全部生产源码哈希与通过信任/覆盖检查的库存一致；运行 `--check` 检测过时索引。标准入口 `lean/CGBAxioms/scripts/audit.py` 在构建、审计成功后自动刷新本页；其 `--check-index` 模式只核对索引、不改写索引。

**未入 Lean 的手工推导与数值见证**另列在 `CGB_HAND_DERIVATIONS.md`，检索时两处都要看。
`reviews/` 中的探索证明见渐进检索入口的独立目录，未混入此生产清单。证明通过不替代其局部前提与实现对应检查。

既有中文机制解读见 [Claude 生成链](CGB_GENERATION_LAYER1_20261025.md)；完整接线记录见 [统一参考](CGB_UNIFIED_REFERENCE_20261026.md)。这些文档保留各自证据范围。

## 生成链导航（自动提取源码标注）

以下从全部 `Generation*.lean` 定理注释的开头加粗标注自动提取，不维护手选机制名单。标注供检索，成立范围以链接中的定理签名为准；未带标注的定理仍全部收入后面的完整清单。

| 模块 | 源码标注与位置 |
|---|---|
| Generation | [G3](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L46)；[G13 (prediction is the boundary's Gibbs conditional)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L74)；[G2](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L100)；[G1](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L118)；[G1E](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L171)；[G4](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L197)；[P2b from weighted micro-reversibility](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L224)；[G6](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L255)；[G6DB](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L269)；[G7](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L282)；[G9 (modulation needs the boundary)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L298)；[G10a](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L317)；[G10b (necessity of modulation, minimal form)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L330)；[G10c (heterogeneous compartments need local gains)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L361)；[G11a](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L395)；[G11b (interoceptive memory)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L414)；[G12 (a miss counter is necessary for optimal foraging)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L480)；[G14a (two-phase drift)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L527)；[G14b](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L542)；[G14c (the phases must balance)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L579)；[R2 (leakage loses on every outcome)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L600)；[R1 (conservation makes the readout one-hot)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L608)；[G15a (reward-modulated two-phase drift is the policy gradient)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L642)；[G15b](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L658)；[G15c (outcome-blind modulation learns nothing)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L698)；[G15d (reward phases must balance)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L710)；[G8a](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L751)；[G8b](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L761) |
| GenerationAmplifier | [G25a (H stays off below threshold)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L68)；[G25b (switch width)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L78)；[G25c (equilibrium Hill coefficient is one)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L90)；[G25d (cooperativity from multiple quanta)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L99)；[G25e (gain–range bound)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L140)；[G25f (a fixed threshold fails on a wide world)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L163)；[G25f (a threshold centred on the prediction works everywhere)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L178) |
| GenerationAxiomAudit | [Homogeneity forces a regular contact graph](../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L26) |
| GenerationColumnMemory | [Silence has one rest](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L51)；[Error descent](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L74)；[Decay by use keeps rare topics](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L101)；[No latch below unit gain](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L116)；[A latch above unit gain](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L131) |
| GenerationCRN | [M2](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L263) |
| GenerationExport | [G16a (export is forced)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L25)；[G16b (write-only bodies starve)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L42)；[G16c](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L50)；[G17a (unselective export forgets)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L67)；[G17b (retention needs selectivity)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L90) |
| GenerationGradientFilter | [W2 (drift forbids a vanishing gain)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L108)；[W3 (an occurrence clock)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L163)；[W4 (a relation basis)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L177)；[W5 is implied by W3](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L187)；[W6 (nonconvex data)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L247) |
| GenerationGrowth | [G18a](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L25)；[G18a (uniqueness)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L32)；[G18b (reciprocity)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L41)；[G18c](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L60)；[G19a (one pool grows a rich club)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L77)；[G19b (memoryless ends see only symbol frequencies)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L92)；[G19c (sequence structure needs memory at the ends)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L113)；[G20a (two-state memory is geometric)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L137)；[G20b (symmetric exchange)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L152)；[G21 (no differentiation without fluctuation)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L162)；[G22a (use depletes)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L183)；[G22b (equilibrium inhibition retreats from co-activity)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L189)；[G22c (no rich club needs local potentials)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L196) |
| GenerationIgnition | [G23 (no ignition in equilibrium with additive energies)](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L61)；[G24 (equilibrium memory bound)](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L100) |
| GenerationInformationEnergy | [N1](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L52)；[N1](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L64)；[N2](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L93)；[N3](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L113)；[N4](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L119)；[N5 (learning saturates)](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L143) |
| GenerationLatchHealth | [H1 (lock-in is death)](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L27)；[H2 (latched means blind)](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L41)；[H1, quantitative](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L59) |
| GenerationLattice | [M1a with conductances](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L39)；[M4 (lattice media are admitted)](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L423) |
| GenerationLearning | [L-1](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L27)；[L-1](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L31)；[L-2](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L37)；[L-2](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L53)；[L-3](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L61)；[O-1](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L84)；[O-2](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L92) |
| GenerationLeaveRule | [C-1 (a changing world needs comparison)](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L38) |
| GenerationLifeGate | [No exact latch](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L48)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L78)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L97)；[L-metabolism for rule tables](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L128)；[L-sense for rule tables](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L146) |
| GenerationLinearMemory | [Histories merge](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L75)；[A linear hold cannot clean up](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L102) |
| GenerationMedium | [M1a](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L24)；[M1b](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L45) |
| GenerationNestedWorld | [A move is an exchange](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L374)；[Mass action](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L442)；[Uniport selectivity is fixed by free energy](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L479)；[Antiport selectivity](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L497)；[T-E2 (an organism nested in its world is admitted)](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1235)；[Only events across the boundary feed the organism](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1362)；[Death](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1401)；[A dead body does not give itself away](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1420)；[Only dissolution empties a place](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1461)；[Births come only from division](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1488)；[Leak bound](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1553)；[Nested metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1654)；[Nested sensing](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1681)；[The two-dimensional world is a model of the axioms](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1787) |
| GenerationNesting | [G26a (the sensed potential of a world cell)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L39)；[G26b (staying put starves)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L58)；[T-N7 (the stream never runs dry)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L140)；[G26c (corpses must go: decomposition is necessary)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L161) |
| GenerationOptimizer | [Theorem: CGB Optimizer Strictly Pareto-Dominates the Adam-Muon Hybrid](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L260) |
| GenerationOrganismEcoliMsS0 | [The organism is a model of the CGB axioms](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L52)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L58)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L65) |
| GenerationOrganismEcoliV1 | [The organism is a model of the CGB axioms](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L52)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L58)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L65) |
| GenerationOrganismGeneratedS0 | [The organism is a model of the CGB axioms](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L50)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L56)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L63) |
| GenerationPersistence | [A (Kolmogorov, one direction)](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L46)；[A](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L61)；[B (integral feedback adapts perfectly)](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L73)；[B](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L82)；[C](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L92)；[C](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L102) |
| GenerationPredictiveHarvest | [Gibbs' inequality](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L63)；[Kelly bound](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L75)；[No body persists](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L106) |
| GenerationSpeechTempo | [N7](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L127) |
| GenerationTable | [M3](../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L72) |

## 完整模块目录

- [ActiveLoop（8）](#activeloop8)
- [ActiveMaintenance（21）](#activemaintenance21)
- [ActiveMaintenanceCarrier（6）](#activemaintenancecarrier6)
- [ActiveMaintenanceEnergy（4）](#activemaintenanceenergy4)
- [AlgebraicCore（6）](#algebraiccore6)
- [AlgebraicHomology（2）](#algebraichomology2)
- [AtomicIndependence（24）](#atomicindependence24)
- [BufferedMaterial（10）](#bufferedmaterial10)
- [CausalLearning（16）](#causallearning16)
- [ConstraintAudit（15）](#constraintaudit15)
- [ConstraintFreedom（4）](#constraintfreedom4)
- [EnergyBudget（5）](#energybudget5)
- [EnergyControls（4）](#energycontrols4)
- [EventChannels（15）](#eventchannels15)
- [Foundation（14）](#foundation14)
- [Generation（35）](#generation35)
- [GenerationAmplifier（13）](#generationamplifier13)
- [GenerationAxiomAudit（2）](#generationaxiomaudit2)
- [GenerationCascadeMemory（5）](#generationcascadememory5)
- [GenerationColumnMemory（13）](#generationcolumnmemory13)
- [GenerationContinualOptimizer（42）](#generationcontinualoptimizer42)
- [GenerationCRN（10）](#generationcrn10)
- [GenerationExport（6）](#generationexport6)
- [GenerationGradientFilter（35）](#generationgradientfilter35)
- [GenerationGrowth（16）](#generationgrowth16)
- [GenerationIgnition（5）](#generationignition5)
- [GenerationInformationEnergy（7）](#generationinformationenergy7)
- [GenerationLatchHealth（3）](#generationlatchhealth3)
- [GenerationLattice（14）](#generationlattice14)
- [GenerationLearning（7）](#generationlearning7)
- [GenerationLeaveRule（3）](#generationleaverule3)
- [GenerationLifeGate（9）](#generationlifegate9)
- [GenerationLinearMemory（7）](#generationlinearmemory7)
- [GenerationMedium（2）](#generationmedium2)
- [GenerationNestedWorld（66）](#generationnestedworld66)
- [GenerationNesting（8）](#generationnesting8)
- [GenerationOptimizer（19）](#generationoptimizer19)
- [GenerationOrganismEcoliMsS0（3）](#generationorganismecolimss03)
- [GenerationOrganismEcoliV1（3）](#generationorganismecoliv13)
- [GenerationOrganismGeneratedS0（3）](#generationorganismgenerateds03)
- [GenerationPersistence（6）](#generationpersistence6)
- [GenerationPredictiveHarvest（9）](#generationpredictiveharvest9)
- [GenerationSpeechTempo（14）](#generationspeechtempo14)
- [GenerationTable（2）](#generationtable2)
- [GenerationUnifiedOptimizer（27）](#generationunifiedoptimizer27)
- [GraceRecovery（20）](#gracerecovery20)
- [GraceRecoveryAccounting（14）](#gracerecoveryaccounting14)
- [GraceRecoveryClosure（18）](#gracerecoveryclosure18)
- [GraceRecoveryControls（4）](#gracerecoverycontrols4)
- [Inference（6）](#inference6)
- [InformationThermodynamics（3）](#informationthermodynamics3)
- [Interaction（19）](#interaction19)
- [InteractionBridge（6）](#interactionbridge6)
- [InteractionControls（7）](#interactioncontrols7)
- [Language（1）](#language1)
- [LanguageLifeKernel（53）](#languagelifekernel53)
- [LearningMaintenance（26）](#learningmaintenance26)
- [LearningMaintenanceControls（8）](#learningmaintenancecontrols8)
- [LearningMaintenanceEnergy（4）](#learningmaintenanceenergy4)
- [Life（18）](#life18)
- [LifeControls（25）](#lifecontrols25)
- [MaterialExchange（8）](#materialexchange8)
- [MaterialSource（6）](#materialsource6)
- [MaterialSourceControls（3）](#materialsourcecontrols3)
- [Medium（5）](#medium5)
- [Models（57）](#models57)
- [NestedLife（2）](#nestedlife2)
- [Nesting（8）](#nesting8)
- [NoetherBridge（8）](#noetherbridge8)
- [OperationalLife（6）](#operationallife6)
- [PortContractInstances（3）](#portcontractinstances3)
- [PortContracts（6）](#portcontracts6)
- [PortFlowHomology（3）](#portflowhomology3)
- [PortSeparation（6）](#portseparation6)
- [RecoveryControls（5）](#recoverycontrols5)
- [RecoveryCriteria（4）](#recoverycriteria4)
- [RecoveryTransport（3）](#recoverytransport3)
- [RenewableCandidateAudit（9）](#renewablecandidateaudit9)
- [RenewableEncoding（13）](#renewableencoding13)
- [RenewableLearning（6）](#renewablelearning6)
- [RenewableLearningCarrier（10）](#renewablelearningcarrier10)
- [RenewableLearningClosure（19）](#renewablelearningclosure19)
- [RenewableLearningControls（15）](#renewablelearningcontrols15)
- [RenewableLearningEnergy（5）](#renewablelearningenergy5)
- [RenewableLearningLife（15）](#renewablelearninglife15)
- [RenewableRecoveryCertificate（7）](#renewablerecoverycertificate7)
- [RuleEncoding（11）](#ruleencoding11)
- [ScheduledRecovery（3）](#scheduledrecovery3)
- [SensoryMaintenance（16）](#sensorymaintenance16)
- [SensoryMaintenanceBoundaries（3）](#sensorymaintenanceboundaries3)
- [SensoryMaintenanceCarrier（12）](#sensorymaintenancecarrier12)
- [SensoryMaintenanceClosure（11）](#sensorymaintenanceclosure11)
- [SensoryMaintenanceLimits（7）](#sensorymaintenancelimits7)
- [SensoryMaintenanceProtocolRepairRefutation（4）](#sensorymaintenanceprotocolrepairrefutation4)
- [SensoryMaintenanceRecovery（10）](#sensorymaintenancerecovery10)
- [SensoryMaintenanceRepairAudit（3）](#sensorymaintenancerepairaudit3)
- [SensoryMaintenanceRepairCandidates（25）](#sensorymaintenancerepaircandidates25)
- [SensoryMaintenanceTwoTickGlobal（7）](#sensorymaintenancetwotickglobal7)
- [SensoryMaintenanceTwoTickObstruction（3）](#sensorymaintenancetwotickobstruction3)
- [SpongeContractInstance（2）](#spongecontractinstance2)
- [SubconditionGraph（37）](#subconditiongraph37)
- [Symmetry（4）](#symmetry4)
- [TurnoverControls（15）](#turnovercontrols15)
- [WorldRecovery（15）](#worldrecovery15)

## ActiveLoop（8）
_Proposed responsive loop and algebraic energy account The proved results are medium axioms, clock and boundary admission, world-row normalization, and an expected first-law identity._

- [`CGB.ActiveLoop.unique_vacuum`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L48)
- [`CGB.ActiveLoop.body_clock`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L77)
- [`CGB.ActiveLoop.body_axioms`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L91)
- [`CGB.ActiveLoop.fin6_sum`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L122)
- [`CGB.ActiveLoop.worldMoves_nonneg`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L151)
- [`CGB.ActiveLoop.worldMoves_sum`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L155)
- [`CGB.ActiveLoop.body_readsBoundary`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L180)
- [`CGB.ActiveLoop.active_expected_first_law`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L234)：Average first law holds along the actual active process.

## ActiveMaintenance（21）
_Resource-dependent maintenance witness A separate successor to the refuted ActiveLoop candidate._

- [`CGB.ActiveMaintenance.admission`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L42)
- [`CGB.ActiveMaintenance.reads_boundary`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L107)
- [`CGB.ActiveMaintenance.sum_configs`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L130)
- [`CGB.ActiveMaintenance.kernel_table`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L139)
- [`CGB.ActiveMaintenance.natural_law`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L156)
- [`CGB.ActiveMaintenance.recovery_step`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L179)
- [`CGB.ActiveMaintenance.normal_recovery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L205)
- [`CGB.ActiveMaintenance.natural_alive`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L211)
- [`CGB.ActiveMaintenance.natural_survival`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L227)
- [`CGB.ActiveMaintenance.starve_world_step`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L246)
- [`CGB.ActiveMaintenance.starve_recovery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L258)
- [`CGB.ActiveMaintenance.cut_move_early`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L278)
- [`CGB.ActiveMaintenance.cut_early_law`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L292)
- [`CGB.ActiveMaintenance.starve_mass_three`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L312)
- [`CGB.ActiveMaintenance.starve_mass_monotone`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L322)
- [`CGB.ActiveMaintenance.cut_alive_bound`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L348)
- [`CGB.ActiveMaintenance.cut_survival_mass_bound`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L363)
- [`CGB.ActiveMaintenance.cut_survival`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L390)
- [`CGB.ActiveMaintenance.maintenance_witness`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L403)：The existing gross-exchange targets, for ALL future times.
- [`CGB.ActiveMaintenance.starvation_depletes`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L416)：Scarcity really changes the body reserve, under either process; vacuum then has no action in any stimulus.
- [`CGB.ActiveMaintenance.sustained_exchange`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L432)：Normal maintenance consumes and imports a positive unit EVERY active tick; this demand is a property of this mechanism, not a universal life axiom.

## ActiveMaintenanceCarrier（6）
_Three-token realization of the maintenance process One token is in the body, one in the accessible source, one in the return sink._

- [`CGB.ActiveMaintenanceCarrier.event_delivery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L53)
- [`CGB.ActiveMaintenanceCarrier.physical_delivery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L74)
- [`CGB.ActiveMaintenanceCarrier.token_count`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L91)
- [`CGB.ActiveMaintenanceCarrier.token_transport`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L94)
- [`CGB.ActiveMaintenanceCarrier.realized`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L106)：Full process and clock certificate, for both normal and marginal-ghost cut.
- [`CGB.ActiveMaintenanceCarrier.token_energy`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L121)

## ActiveMaintenanceEnergy（4）
_Reserve, explicit consumption and source-resolved replacement The same registered normal/cut event process pays one reserve unit on a live feeding or scarcity event._

- [`CGB.ActiveMaintenanceEnergy.actual_law`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L28)：The energy law is exactly the state law used in the viability comparison, including the cut's own distribution-dependent future, not a second process.
- [`CGB.ActiveMaintenanceEnergy.body_supported`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L45)
- [`CGB.ActiveMaintenanceEnergy.isothermal`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L86)：Conditional mean second-law admission for the explicitly declared internal entropy and bath units.
- [`CGB.ActiveMaintenanceEnergy.availability_budget`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L108)：Every finite prefix uses the SAME event law and tick.

## AlgebraicCore（6）
_Algebraic finite-state core Finite-state transport and exact budget identities can be expressed over a commutative semiring._

- [`CGB.AlgebraicCore.pushForward_total`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L23)
- [`CGB.AlgebraicCore.pathWeight_zero`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L37)
- [`CGB.AlgebraicCore.jetMul_product_rule`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L47)
- [`CGB.AlgebraicCore.jetHorner_value`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L59)
- [`CGB.AlgebraicCore.cast_local_budget`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L67)
- [`CGB.AlgebraicCore.cast_path_budget`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L75)

## AlgebraicHomology（2）
_Minimal algebraic homology core Finite boundary operators are represented by matrices over a commutative semiring._

- [`CGB.AlgebraicHomology.matrixAction_compose`](../../lean/CGBAxioms/CGBAxioms/AlgebraicHomology.lean#L27)
- [`CGB.AlgebraicHomology.boundary_boundary_zero`](../../lean/CGBAxioms/CGBAxioms/AlgebraicHomology.lean#L35)

## AtomicIndependence（24）
_Full remaining-twelve certificates Since 2026-10-26 contact homogeneity (former A04) is no longer an axiom part; the thirteen remaining atomic conditions keep their ids (A01–A03, A05–A14)._

- [`CGB.Medium.atomized_iff_all_conditions`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L45)
- [`CGB.Medium.a01_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L62)
- [`CGB.Medium.a02_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L71)
- [`CGB.Medium.a03_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L80)
- [`CGB.Medium.a09_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L89)
- [`CGB.Medium.a10_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L98)
- [`CGB.Medium.a11_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L107)
- [`CGB.Medium.a07_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L116)
- [`CGB.Medium.a12_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L125)
- [`CGB.Medium.a14_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L134)
- [`CGB.Medium.globalFlip_other_clauses`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L158)
- [`CGB.Medium.a05_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L172)
- [`CGB.Medium.a06_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L185)
- [`CGB.Medium.farAct_other_clauses`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L203)
- [`CGB.Medium.a08_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L232)
- [`CGB.Medium.innerActClock_other_clauses`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L240)
- [`CGB.Medium.a13_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L276)
- [`CGB.Medium.twinFlip_all_conditions`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L285)：Consistency is witnessed separately from independence.
- [`CGB.Medium.all_atoms_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L305)
- [`CGB.Medium.reducedCore_iff_selected_conditions`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L430)
- [`CGB.Medium.rigidLabel_rigid`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L443)
- [`CGB.Medium.holds_rigidLabel_iff`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L446)
- [`CGB.Medium.isolates_rigidLabel`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L451)
- [`CGB.Medium.every_reduced_condition_independent_in_regime`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L466)：Every selected condition has a finite-local-state countermodel in the SAME rigid-structure regime used by the recovery bridge.

## BufferedMaterial（10）
_Finite identity-preserving buffered material transport Twelve fixed carriers suffice for a body containing at most four units, a four-unit accessible source and at least four returned carriers._

- [`CGB.BufferedMaterial.canonical_stock`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L30)
- [`CGB.BufferedMaterial.total`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L34)
- [`CGB.BufferedMaterial.select`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L44)
- [`CGB.BufferedMaterial.transport_exists`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L54)：Local existence with every intake/release/recharge coming from old stock.
- [`CGB.BufferedMaterial.available`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L145)
- [`CGB.BufferedMaterial.advance_project`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L160)
- [`CGB.BufferedMaterial.lift_sum`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L166)
- [`CGB.BufferedMaterial.refines`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L196)
- [`CGB.BufferedMaterial.delivery`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L210)
- [`CGB.BufferedMaterial.realized`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L237)：All original states, all supported events, and the same supplied clock.

## CausalLearning（16）
_Finite, causal and paid candidate-model learning Models predict possible boundary outcomes of actions._

- [`CGB.CausalLearning.memory_cardinality`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L50)：Candidate masks have |H| bits of state capacity, plus the explicit reserve.
- [`CGB.CausalLearning.write_contract`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L61)：Each effective write strictly removes candidates and spends one declared unit of finite reserve.
- [`CGB.CausalLearning.retain_consistent`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L84)
- [`CGB.CausalLearning.agreement_no_write`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L93)：A predictable observation provides no exclusion evidence.
- [`CGB.CausalLearning.empty_write_reserve`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L101)
- [`CGB.CausalLearning.paid_elimination`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L108)：A funded observation can remove a false candidate using the observation itself, without consulting a true-model identifier.
- [`CGB.CausalLearning.path_budget`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L125)：Analysis may count writes along a path; that counter is not learner state.
- [`CGB.CausalLearning.delivered`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L164)
- [`CGB.CausalLearning.true_model_step`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L182)
- [`CGB.CausalLearning.law_invariant`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L192)
- [`CGB.CausalLearning.true_model_law`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L215)：Positive-probability states retain the faithful candidate at every time; the controller definition itself never receives this candidate's identity.
- [`CGB.CausalLearning.physical_marginal`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L231)：Even with zero write reserve, each physical action/outcome law continues.
- [`CGB.CausalLearning.safe_outcome`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L242)
- [`CGB.CausalLearning.certified_choice_sound`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L268)
- [`CGB.CausalLearning.no_safe_exploration`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L288)
- [`CGB.CausalLearning.paid_mean_budget`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L317)：Exact mean finite-prefix payment on the actual action/outcome process.

## ConstraintAudit（15）
_Joint audit of irredundancy and generative freedom No new physical clause is introduced._

- [`CGB.Medium.axioms_iff_atomized`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L38)
- [`CGB.Medium.p1a_iff_finite_and_unique`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L58)：The first two cost clauses jointly say finite local state space in this language.
- [`CGB.Medium.rigid_of_injective_identity`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L71)
- [`CGB.Medium.p2c_of_rigid`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L77)
- [`CGB.Medium.initial_invariant_of_rigid`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L82)
- [`CGB.Medium.onlyBoundarySenses_of_all_boundary`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L88)
- [`CGB.Medium.p2a_of_every_event_open`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L95)
- [`CGB.Medium.empty_participants_no_effect`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L98)
- [`CGB.Medium.jump_equivariant`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L106)：A genuine joint consequence: event equivariance descends to the aggregate jump kernel, including the single shared event-permutation witness.
- [`CGB.Medium.empty_boundary_every_stimulus_reversible`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L124)：P3 with no boundary removes ALL stimulus dependence.
- [`CGB.Medium.rescale_jump`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L142)
- [`CGB.Medium.rescale_axioms`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L150)
- [`CGB.Medium.rescale_changes_nonzero_jump`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L173)
- [`CGB.JumpProcess.fromConductance_reversible`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L192)
- [`CGB.JumpProcess.reversible_conductance_representation`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L200)

## ConstraintFreedom（4）
_Families admitted by the same clauses These are logical model families, not proposed brain mechanisms or capability experiments._

- [`CGB.ConstraintFreedom.arbitrarily_many_states_at_unit_capacity`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L19)：P1a is qualitative: an arbitrarily large FINITE state alphabet may have capacity 1.
- [`CGB.ConstraintFreedom.arbitrary_driven_tables_admitted`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L49)：Thus the core alone does not determine a driven rule table.
- [`CGB.ConstraintFreedom.openSite_reproduces_process`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L80)：With one event per target state, the exposed-site family reproduces the off-diagonal rates of ANY finite Markov jump process under its driven input.
- [`CGB.ConstraintFreedom.constant_protocol_can_be_driven`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L88)：A constant-in-time stimulus already breaks detailed balance in the existing seed witness.

## EnergyBudget（5）
_Energy balance and conditional mean availability budget Physical energy, work/heat ports, temperature and entropy require a carrier interpretation._

- [`CGB.Energetics.law_probability`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L28)
- [`CGB.Energetics.expected_first_law`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L66)：Average the supported first law along the ORIGINAL process.
- [`CGB.Energetics.availability_balance`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L118)
- [`CGB.Energetics.work_budget`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L133)：One finite-prefix identity gives both the bounded-reserve work budget and the cyclic bound.
- [`CGB.Energetics.no_sustained_use_on_finite_supply`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L169)：A positive mean availability-use requirement (delivered work plus T times production) cannot persist on bounded total supply and reserve.

## EnergyControls（4）
_Energy-interpretation controls on the unchanged material process The same material transport admits different energy calibrations._

- [`CGB.EnergyControls.total_partition`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L24)
- [`CGB.EnergyControls.material_energy_binding`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L43)：On every supported event, the material-energy port is carried by the same exterior token identified in the material-source certificate.
- [`CGB.EnergyControls.isothermal_account`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L54)：Nonnegative mean production for the declared T=1, zero-system-entropy accounting interpretation along the actual process; not a derivation of the microscopic entropy function or of local detailed balance.
- [`CGB.EnergyControls.same_material_different_energy`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L75)：Fixed identities and identical counts do not select chemical energy or work.

## EventChannels（15）
_Event-resolved observations with unchanged body/world marginals State-preserving events can carry gross exchange._

- [`CGB.FirstHit.Kernel.eq_of_step`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L17)
- [`CGB.EventChannels.Kernel.firstHit_bounds`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L50)
- [`CGB.EventChannels.Kernel.state_mark_firstHit`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L79)：A state-only mark is the specialization of the single channel observer that ignores event labels.
- [`CGB.EventChannels.Kernel.firstHit_unmarked`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L92)
- [`CGB.EventChannels.Kernel.labelledLaw_marginal`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L110)
- [`CGB.Medium.exit_rate_events`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L138)
- [`CGB.Medium.eventClocked_implies_clocked`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L153)
- [`CGB.Medium.eventStep_nonneg`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L168)
- [`CGB.Medium.eventStep_sum`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L183)
- [`CGB.Medium.eventStep_forget`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L191)：Forgetting the event labels recovers the ORIGINAL kernel, even when some positive-rate events leave the body state unchanged.
- [`CGB.Medium.eventWorldStep_nonneg`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L229)
- [`CGB.Medium.eventWorldStep_sum`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L239)
- [`CGB.Medium.jointEventKernel_forget`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L257)
- [`CGB.Medium.event_state_mark_matches_operational`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L267)
- [`CGB.Medium.jointEventKernel_stateLaw`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L287)：The entire observed forward process has the original normal/cut state law, not just an equal one-step kernel.

## Foundation（14）
_Foundation audit: premises, specialization, and clocks These results audit the existing predicates; they add no physical axiom and do not change the organism dynamics or the life target._

- [`CGB.Medium.finite_local_states_of_p1a`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L18)：P1a itself supplies local-state finiteness, before a finite enumeration is chosen.
- [`CGB.Medium.p1a_iff_unique_vacuum`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L27)：On an already finite local state type, P1a's first two conjuncts are automatic.
- [`CGB.Medium.exists_clock_of_finite_stimuli`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L42)：A globally bounded clock follows if the stimulus alphabet is also finite.
- [`CGB.Medium.recover_ignores_world_moves`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L65)：The current recovery menu does not consult actual world transitions.
- [`CGB.Medium.recover_bounds`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L77)：Recovery is a genuine probability only with the clock-admission hypothesis.
- [`CGB.Medium.law_nonneg_of_clocked`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L112)：Normal and loop-cut laws must be nonnegative as well as normalized.
- [`CGB.Medium.lawCut_nonneg_of_clocked`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L122)
- [`CGB.Medium.aliveProb_bounds`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L135)
- [`CGB.Medium.aliveProb_zero`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L152)
- [`CGB.Medium.persists_parameter_bounds`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L159)：Probability/horizon parameter bounds follow from a persistence witness and clock admission; they need not be introduced as additional physical axioms.
- [`CGB.FoundationCounterexamples.unboundedStimulus_axioms`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L198)
- [`CGB.FoundationCounterexamples.unboundedStimulus_no_clock`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L226)：Six clauses plus finite body/state/event types do not imply a global Clocked bound.
- [`CGB.TheoryAudit.existence_survives_weakening`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L251)：Weakening admissibility cannot destroy an already constructed existence witness.
- [`CGB.TheoryAudit.counterexample_blocks_guarantee`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L258)：A counterexample does refute a universal guarantee from the remaining premises.

## Generation（35）
_Generation, layer 1: what the axioms force on the equilibrium law The axioms constrain; this file starts deriving._

- [`CGB.JumpProcess.reversible_mul_invariant`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L46)：**G3**: a positive factor that is constant along every move preserves detailed balance.
- [`CGB.JumpProcess.freezeSlow_conditional_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L74)：**G13 (prediction is the boundary's Gibbs conditional)**: if the joint calm dynamics has detailed-balance weight `π`, then with the slow interior `P` frozen, for each fixed `p` the conditional `π (·, p) / Σ π (·, p)` is
- [`CGB.Medium.parts_subset_scope`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L94)：Under P1b an event that changes all of `D` reads only inside `scope D`.
- [`CGB.Medium.content_fixed_by_closed_single_unit`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L100)：**G2**: under P2a a closed event that changes a single unit keeps that unit's content, so content moves inside the body only by events that change at least two units.
- [`CGB.Medium.jump_rate_local`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L118)：**G1**: the total rate of a transition `x → y` with changed set `D` depends only on the new values on `D` and on the old state and stimulus on `D ∪ scope D`.
- [`CGB.Medium.equilibrium_ratio_local`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L171)：**G1E**: across one calm transition with a positive reverse rate, the equilibrium ratio `π y / π x` depends only on the same local data as G1.
- [`CGB.Medium.local_markov_of_single_unit_moves`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L197)：**G4**: if every value of unit `u` can be left by a calm move that changes only `u`, the equilibrium law is locally Markov at `u`: the ratio `π (x with u := l) / π x` depends only on `x` on `{u} ∪ scope {u}`.
- [`CGB.Medium.p2b_of_micro_weighted`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L224)：**P2b from weighted micro-reversibility** (generalizes `p2b_of_micro`, which has uniform weights): if events pair up by an involution `rev` so that every effective calm event is undone by its partner and the pair satisfi
- [`CGB.Medium.rate_interior_stimulus_free`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L255)：**G6**: under P1b and P3 a transition whose changed set and its scope avoid the boundary has the same rate under every stimulus as in the calm environment.
- [`CGB.Medium.interior_detailed_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L269)：**G6DB**: hence, with P2b, interior transitions satisfy detailed balance with the calm weight under every stimulus: all irreversibility enters at the boundary.
- [`CGB.Medium.coupling_vanishes_beyond_scope`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L282)：**G7**: no equilibrium coupling beyond contact.
- [`CGB.Medium.modulation_needs_boundary`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L298)：**G9 (modulation needs the boundary)**: if the rate of a transition (for example a plastic move: its learning rate or, through the forward/backward pair, its learning direction) differs between two stimuli, then some uni
- [`CGB.Medium.fixed_gain_error`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L317)：**G10a**: the fixed-gain error exceeds the optimum `P` by an exact square; the unique optimal fixed gain is the Kalman gain `P / (P + r)`.
- [`CGB.Medium.no_fixed_gain_for_two_volatilities`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L330)：**G10b (necessity of modulation, minimal form)**: in two worlds with different volatility (`P₁ ≠ P₂`), no fixed gain is optimal in both: in at least one world its error is strictly above that world's optimum.
- [`CGB.Medium.no_global_gain_for_heterogeneous_compartments`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L361)：**G10c (heterogeneous compartments need local gains)**: model two compartments by their local volatility parameters `P c`.
- [`CGB.Medium.miss_posterior_bayes`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L385)：The miss posterior is Bayes' rule: with prior `π`, miss probabilities `qr < qp` (`qr = 1 - pr`, `qp = 1 - pp`), `n` misses give rich posterior `1 / (1 + a ρⁿ)` with `a = (1-π)/π` and `ρ = qp/qr`.
- [`CGB.Medium.missPrediction_strictAnti`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L395)：**G11a**: each further miss strictly lowers the best prediction.
- [`CGB.Medium.intake_memory_capacity`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L414)：**G11b (interoceptive memory)**: to predict intake within `ε` after every history of up to `N` consecutive misses, where `2ε` is below every gap between those predictions, an organism's state must take at least `N + 1` d
- [`CGB.Medium.intake_gap_exists`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L431)：The gap condition is satisfiable: strict monotonicity separates any finite family, so some positive precision forces `N + 1` states.
- [`CGB.Medium.exists_leave_threshold`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L452)：The leave threshold exists: whenever going elsewhere is worth more than a known-poor source (`pp < c`), enough misses push the best prediction below `c`.
- [`CGB.Medium.miss_counter_necessary`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L480)：**G12 (a miss counter is necessary for optimal foraging)**: let `nStar` be the first miss count whose best prediction falls below `c` (all earlier counts stay at or above it).
- [`CGB.Medium.merging_across_threshold_loses`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L508)：Merging two histories that lie on opposite sides of the threshold costs expected intake: with positive weights, a single shared action earns strictly less than acting optimally on each.
- [`CGB.Medium.two_phase_drift`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L527)：**G14a (two-phase drift)**: in the drive phase the boundary is clamped to the arrived value `b'` and the site moves by `+λ x` when the boundary shows `b`; in the free phase the boundary samples the readout law and the si
- [`CGB.Medium.softmax_sum`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L535)：The readout law sums to one.
- [`CGB.Medium.log_softmax_deriv`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L542)：**G14b**: the log likelihood of the arrived value `b'` changes, per unit change of that coupling, by `x ([b' = b] - softmax z b)`.
- [`CGB.Medium.phases_must_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L579)：**G14c (the phases must balance)**: if, for every readout state and arrived value, the expected drift vector over boundary values is a positive multiple of the log-likelihood gradient vector, the drive and free phases ca
- [`CGB.Medium.leakage_loses`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L600)：**R2 (leakage loses on every outcome)**: if the free boundary puts mass `ℓ > 0` on configurations that are no symbol, every symbol's predicted probability is `(1 - ℓ) q b` instead of `q b`, and the log likelihood of what
- [`CGB.Medium.one_unit_one_label`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L608)：**R1 (conservation makes the readout one-hot)**: if every label unit holds content `0` or `1` and the readout's total content is `1` (an invariant shell of the closed free dynamics under P2a, `total_of_reachable`), exact
- [`CGB.Medium.sector_law_is_softmax`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L626)：In the one-particle sector a local label energy gives the softmax readout: with weights `exp (z b)` on the configurations `label b on`, the normalized law is `softmax z` (G13 restricted to the sector).
- [`CGB.Medium.reward_two_phase_drift`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L642)：**G15a (reward-modulated two-phase drift is the policy gradient)**: if the motor particle was clamped at the action taken `a ~ softmax z`, the drive phase moved the site by `(R a - bl) x` when label `b` was on, and the f
- [`CGB.Medium.expected_reward_deriv`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L658)：**G15b**: that drift is exactly the derivative of the expected reward `J` with respect to the logit of `b`.
- [`CGB.Medium.outcome_blind_drift_zero`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L698)：**G15c (outcome-blind modulation learns nothing)**: if the modulator's weight `m` does not depend on the action's outcome, the expected two-phase drift is zero for every site.
- [`CGB.Medium.reward_phases_must_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L710)：**G15d (reward phases must balance)**: with drive weight `lam` and free weight `mu`, if the expected drift is a multiple of the expected-reward gradient for every reward table, then `lam = mu`.
- [`CGB.Medium.two_part_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L751)：**G8a** (two parts suffice): with positive part factors `φ₁, φ₂` and constants `a, b ≥ 0`, up rates `a φ₁ + b φ₂` and down rates `b / φ₁ + a / φ₂` (each term reads one part) have ratio exactly `φ₁ φ₂`.
- [`CGB.Medium.three_part_balance_impossible`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L761)：**G8b** (three parts are impossible): if up rates `f₁+f₂+f₃` and down rates `g₁+g₂+g₃` (each term reading one part) satisfy detailed balance with a product factor `φ₁ φ₂ φ₃` that is non-constant on every part, the total

## GenerationAmplifier（13）
_An equilibrium amplifier: ordered filling of a conserved shell (generation chain, G25) Chemotaxis needs an all-or-none response to a tiny difference (a gain of order ten in *E._

- [`CGB.Generation.occupancy_le_exp`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L42)
- [`CGB.Generation.one_sub_occupancy_le_exp`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L49)
- [`CGB.Generation.occupancy_le_one`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L61)
- [`CGB.Generation.high_off_below_threshold`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L68)：**G25a (H stays off below threshold)**: while the low page is at most half full, the high page is occupied at most `e^{-Δ}`.
- [`CGB.Generation.switch_width`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L78)：**G25b (switch width)**: raising the potential from `l1` to `l2` adds at most `NH` times the gain in `H`'s occupancy plus `NL e^{-(l1 - φL)}` content.
- [`CGB.Generation.equilibrium_hill_one`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L90)：**G25c (equilibrium Hill coefficient is one)**: the log-odds of occupation is exactly the potential above the unit's energy, so it moves one-for-one with the sensed potential.
- [`CGB.Generation.quanta_hill`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L99)：**G25d (cooperativity from multiple quanta)**: a state holding `s` units of content, with energy `φs`, has equilibrium weight `exp (s μ - φs)` relative to the empty state, so its log-odds moves `s`-for-one with the poten
- [`CGB.Generation.occupancy_ge_inv_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L106)：Lower edge of the resolvable range: with `N` units, the occupancy is at least one unit's worth (`1/N`) exactly when `-h ≤ log (N - 1)`.
- [`CGB.Generation.occupancy_le_one_sub_inv_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L114)：Upper edge: the occupancy leaves at least one unit empty exactly when `h ≤ log (N - 1)`.
- [`CGB.Generation.resolvable_band`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L140)：**G25e (gain–range bound)**: a switch of gain `s` (a state of `s` quanta, G25d) centred at `θ`, read through `N` units, is resolvable (between one occupied and one empty unit) exactly on `|μ - θ| ≤ log (N - 1) / s`.
- [`CGB.Generation.gain_times_band`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L151)：The product of gain and resolvable width is `2 log (N - 1)`, independent of the gain: an amplified switch covers a proportionally narrower band of the sensed potential.
- [`CGB.Generation.fixed_threshold_fails`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L163)：**G25f (a fixed threshold fails on a wide world)**: if the sensed potential ranges over an interval wider than `2 log (N - 1) / s`, a switch of gain `s` at any fixed `θ` is unresolvable somewhere in it.
- [`CGB.Generation.centred_threshold_resolves`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L178)：**G25f (a threshold centred on the prediction works everywhere)**: if the threshold sits at a stored prediction `μhat` (G13: the slow interior's conditional) and the innovation `μ - μhat` (G14) stays within `δ` with `s δ

## GenerationAxiomAudit（2）
_Which axiom parts work (generation chain, audit) Usage in the generation chain (G1–G22, M1–M2): P1b's locality parts (participants in contact, writes and reads only participants), P2a, P2b and P3's boundary sensing enter_

- [`CGB.Medium.homogeneous_regular`](../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L26)：**Homogeneity forces a regular contact graph**: every unit has the same number of contacts.
- [`CGB.Medium.not_homogeneous_of_degrees`](../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L45)：A unit with a different number of contacts than another rules out homogeneity.

## GenerationCascadeMemory（5）
_A bank of two-state pools keeps a power-law trace (generation chain, cascade memory, 2026-10-27) G20a (`two_state_geometric`) makes the trace of one two-state pool exponential._

- [`CGB.Generation.single_pool_falls_below_power`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L29)：One pool's trace drops below `c / t` once `t > 2 τ / c`.
- [`CGB.Generation.bank_trace_ge_one_term`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L58)：The bank's trace is at least any one of its (nonnegative) terms.
- [`CGB.Generation.pool_in_band_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L63)：A pool with `t ≤ τ ≤ γ t` alone keeps `1 / (e γ t)`.
- [`CGB.Generation.bank_power_law_lower_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L77)
- [`CGB.Generation.geometric_bank_covers`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L84)：Geometric time constants cover the band: every `t ∈ [τ₀, τ₀ γ^m]` has a level `k ≤ m` with `t ≤ τ₀ γ^k ≤ γ t`.

## GenerationColumnMemory（13）
_Column memory for the language organism (generation chain, 2026-10-28) The language organism's synaptic page is a set of columns `S_m` written by the gated delta rule (`GenerationLinearMemory`), with the activity page re_

- [`CGB.Generation.dot_self_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L41)
- [`CGB.Generation.eq_zero_of_dot_self`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L44)
- [`CGB.Generation.silent_rest_unique`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L51)：**Silence has one rest**: a gated delta step without a write (`α < 1`) fixes only zero.
- [`CGB.Generation.tiedWrite_eq_delta_add`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L66)：The tied write is the delta step plus the write of the arrival.
- [`CGB.Generation.tied_write_is_error_descent`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L74)：**Error descent**: the tied write is a gradient step of size `β` on `½ (⟨k, x⟩ - v)²`, whose gradient in `x i` is `(⟨k, x⟩ - v) k i` (predicted minus arrived, times the presynaptic key).
- [`CGB.Generation.tied_write_error_shrinks`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L81)：With a unit key the prediction error shrinks by `1 - β`.
- [`CGB.Generation.exact_prediction_no_write`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L94)：An exact prediction writes nothing.
- [`CGB.Generation.occurrence_decay_retains`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L101)：**Decay by use keeps rare topics**: decaying by `α ^ w t` with `w t ∈ [0, 1]` keeps at least what decaying by `α` every tick keeps.
- [`CGB.Generation.unused_column_kept`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L111)：An unused column keeps everything.
- [`CGB.Generation.no_latch_below_unit_gain`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L116)：**No latch below unit gain**: `x ↦ σ (g x)` with a 1-Lipschitz `σ`, `σ 0 = 0` and `0 ≤ g < 1` rests only at zero.
- [`CGB.Generation.hardSat_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L125)
- [`CGB.Generation.hardSat_lipschitz`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L127)
- [`CGB.Generation.latch_above_unit_gain`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L131)：**A latch above unit gain**: with saturation and `g ≥ 1`, `± 1` rest besides `0`.

## GenerationContinualOptimizer（42）
_A continual optimizer: competition, no winner-take-all, a healthy latch (side branch, 2026-10-27) The optimizer of a stream learner must (i) make units compete so that they differentiate, (ii) keep one unit from taking e_

- [`CGB.Continual.polar_conserves_direction_mass`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L54)：Orthonormal columns carry unit mass along every input direction.
- [`CGB.Continual.payoff_le_two_of_col`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L71)
- [`CGB.Continual.payoff_le_two_of_row`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L80)
- [`CGB.Continual.column_budget_allows_wta`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L89)
- [`CGB.Continual.row_budget_allows_copies`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L97)
- [`CGB.Continual.doubly_budgeted_form`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L105)
- [`CGB.Continual.doubly_budgeted_max_iff_permutation`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L110)
- [`CGB.Continual.doubly_budgeted_max_differentiates`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L125)：At a doubly budgeted maximum the two rows own different directions.
- [`CGB.Continual.linear_protect_implies_slow`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L133)
- [`CGB.Continual.deferLatch_holds`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L145)
- [`CGB.Continual.deferLatch_releases`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L148)
- [`CGB.Continual.deferLatch_senses`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L153)：The latch is not blind: the same held state gives different outputs for different evidence runs (H2 is avoided), and every hold ends after `c` contrary steps (H1).
- [`CGB.Continual.held_budget_starves`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L161)：Consolidated entries that keep a share `s` of a conserved budget `B` and are never released leave `B - n s` free after `n` consolidations; it is gone once `n s ≥ B`.
- [`CGB.Continual.secant_own_noise_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L174)
- [`CGB.Continual.secant_same_batch_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L180)
- [`CGB.Continual.fresh_decrease_mixture`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L204)
- [`CGB.Continual.conditional_prediction_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L207)
- [`CGB.Continual.shared_reference_flips`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L211)：Shared reference: row `x` occurs when `x ≥ f · max x y`.
- [`CGB.Continual.expected_gradient_is_mixture`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L241)
- [`CGB.Continual.positive_part_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L246)：Zero signal, calibrated variance: `m` is `0, 0, c, -c` with `c^2 = 2P`, so `E[m^2] = P`.
- [`CGB.Continual.consistent_prediction_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L253)
- [`CGB.Continual.softmax_gradients_sum_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L259)：Two-class softmax: the class-row gradients `(p - y)` and `((1 - p) - (1 - y))` sum to zero.
- [`CGB.Continual.softmax_rows_coupled`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L263)：The output factor of the softmax curvature, `diag p - p pᵀ` for two classes, has the off-diagonal entry `-p (1 - p)`, nonzero for `0 < p < 1`.
- [`CGB.Continual.ema_forgets_rare_spikes`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L290)：The registered window (factor 0.95) with spikes 200 steps apart: the held share 0.95^200 is below the spike's share 1/200 of the long-run power.
- [`CGB.Continual.partition_is_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L292)
- [`CGB.Continual.event_part_is_rate_times_size`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L299)
- [`CGB.Continual.frozen_hold_loses_on_drift`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L327)
- [`CGB.Continual.mixture_update_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L333)
- [`CGB.Continual.mixture_weight_at_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L343)
- [`CGB.Continual.release_threshold_monotone_in_age`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L346)
- [`CGB.Continual.succession_odds_decreasing`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L352)
- [`CGB.Continual.linear_burst_damage_integral`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L374)
- [`CGB.Continual.beta_hazard_decreasing`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L410)
- [`CGB.Continual.beta_hazard_succession`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L417)
- [`CGB.Continual.accepted_run_variance`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L420)
- [`CGB.Continual.posterior_shrink_error`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L440)：Posterior of the gradient as the two points `m ± √P` (mean `m`, variance `P`): the mean squared error of the action `f m` is `P + (1 - f)^2 m^2`.
- [`CGB.Continual.posterior_mean_is_bayes_action`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L445)
- [`CGB.Continual.newton_contracts_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L469)
- [`CGB.Continual.overestimate_keeps_contraction`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L484)
- [`CGB.Continual.saturating_step_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L499)
- [`CGB.Continual.pooled_signal_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L518)
- [`CGB.Continual.next_value_variance`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L523)

## GenerationCRN（10）
_Well-mixed pair reaction media are admitted (generation chain, M2) One compartment (every unit touches every other: a single clique, which by G8 can integrate anything)._

- [`CGB.PairCRN.sum_pairAct`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L96)：A pair event changes a unit-wise sum only at its two units.
- [`CGB.PairCRN.totalE_pairAct`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L117)
- [`CGB.PairCRN.p1a`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L125)
- [`CGB.PairCRN.p1b`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L128)
- [`CGB.PairCRN.p2a`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L152)
- [`CGB.PairCRN.p2b`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L176)
- [`CGB.PairCRN.p2c`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L213)
- [`CGB.PairCRN.p3`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L236)
- [`CGB.PairCRN.homogeneous`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L256)：The single compartment's contact is homogeneous (an optional model property).
- [`CGB.PairCRN.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L263)：**M2**: a well-mixed pair reaction medium is a model of the CGB axioms as soon as its finite rule table passes the local obligations: content levels (P1a), conservation of closed rules (P2a), reversible rules with symmet

## GenerationExport（6）
_Export and selective export (generation chain, G16-G17) P1a makes the body finite; P2a gives the content ledger `content (t+1) + out t = content t + in t` (`MaterialLedger.balance`)._

- [`CGB.Generation.export_forced`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L25)：**G16a (export is forced)**: with the content ledger and a capacity bound, cumulative export is at least cumulative intake minus the capacity.
- [`CGB.Generation.write_only_intake_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L42)：**G16b (write-only bodies starve)**: with no export, total lifetime intake is at most the capacity.
- [`CGB.Generation.endless_intake_endless_export`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L50)：**G16c**: if intake does not stop (its total exceeds every bound), export does not stop either.
- [`CGB.Generation.survive_le_pow`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L67)：**G17a (unselective export forgets)**: if every export removes the item with probability at least `q₀ > 0` (for example `1/C` when the exported unit is drawn without regard to what it holds), survival is at most `(1 - q₀
- [`CGB.Generation.unselective_export_forgets`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L79)
- [`CGB.Generation.retention_needs_selectivity`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L90)：**G17b (retention needs selectivity)**: if an item must survive every number of exports with probability at least `p > 0`, its per-export removal probability has no positive lower bound: export must eventually spare it,

## GenerationGradientFilter（35）
_An optimizer as a learner of its gradient stream (side branch, 2026-10-27) The outer loop's world is the stream of noisy gradients._

- [`CGB.GradientFilter.memoryless_gain_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L34)：A learner without memory (gain 1) exceeds the optimal error `P` by `r^2 / (P + r)`.
- [`CGB.GradientFilter.memoryless_strictly_worse`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L42)
- [`CGB.GradientFilter.shrink_error_square`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L52)
- [`CGB.GradientFilter.shrink_optimum`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L58)
- [`CGB.GradientFilter.unshrunk_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L64)：Taking the noisy estimate at face value costs exactly `N^2 / (S + N)` more.
- [`CGB.GradientFilter.no_shared_shrink_for_two_snr`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L72)：One shared factor is strictly suboptimal in at least one of two directions whose signal fractions differ.
- [`CGB.GradientFilter.vanishing_gain_loses_under_drift`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L108)：**W2 (drift forbids a vanishing gain)**: the tracking error variance obeys `V (t+1) = (1 - K t)^2 V t + q + K t^2 r`.
- [`CGB.GradientFilter.emaVar_fixed`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L153)：It is the fixed point of `v = x^2 v + (1 - x)^2 r`.
- [`CGB.GradientFilter.occurrence_clock_beats_time_clock`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L163)：**W3 (an occurrence clock)**: a direction observed once every `k ≥ 2` steps, averaged with the per-step factor `β` of the frequent directions, effectively uses `β^k` per observation and has a strictly larger stationary v
- [`CGB.GradientFilter.diagonal_cannot_whiten`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L177)：**W4 (a relation basis)**: if the curvature `[[a, b], [b, d]]` has a nonzero off-diagonal term, no diagonal preconditioner `diag(d₁, d₂)` turns it into a nonzero multiple of the identity (one-step whitening); a basis-awa
- [`CGB.GradientFilter.usage_normalized_gain_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L187)：**W5 is implied by W3**: a gain normalized by that direction's accumulated use (`s / sqrt (c + m)`, the AdaGrad / Adam form) strictly falls when the direction's own demand grows: it is locally anti-dominant without a sep
- [`CGB.GradientFilter.step_error_square`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L215)
- [`CGB.GradientFilter.optimal_step_minimizes`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L222)
- [`CGB.GradientFilter.optimal_step_newton_times_signal`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L227)
- [`CGB.GradientFilter.no_fixed_step_two_distances`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L233)：With noise, the optimal step differs at two different distances to the optimum.
- [`CGB.GradientFilter.no_optimal_step_negative_curvature`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L247)：**W6 (nonconvex data)**: with negative curvature the loss along the step is unbounded below, so no optimal quadratic step exists and the step must be bounded.
- [`CGB.GradientFilter.no_fixed_bound_two_radii`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L292)
- [`CGB.GradientFilter.bound_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L306)
- [`CGB.GradientFilter.newton_step_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L324)：The exact curvature step reaches the optimum of a quadratic direction.
- [`CGB.GradientFilter.momentum_after_exact_step`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L330)：After an exact step, momentum `β ≠ 0` (heavy ball: next error `β (e₁ - e₀)` with `e₁ = 0`) moves a direction that started away from the optimum off it again.
- [`CGB.GradientFilter.gain_error_square`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L354)
- [`CGB.GradientFilter.steady_gain_worse_for_fresh`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L362)：The steady gain `P* / (P* + r)` is strictly worse than the evidence-dependent gain for a direction whose prior variance is different (larger: a fresh or rare direction).
- [`CGB.GradientFilter.in_sample_decrease_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L400)：Decrease measured on the batch that made the step minus the decrease on fresh data.
- [`CGB.GradientFilter.in_sample_decrease_bias_mean`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L405)
- [`CGB.GradientFilter.own_step_biases_noise_estimate`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L412)：Lag-one noise estimate averaged over the noise pair `(x, y)`, `(-x, -y)`.
- [`CGB.GradientFilter.residual_removes_own_step`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L416)
- [`CGB.GradientFilter.secant_needs_diagonal`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L420)
- [`CGB.GradientFilter.secant_exact_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L424)
- [`CGB.GradientFilter.fresh_decrease_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L452)：Fresh-batch decrease averaged over the unseen noise pair `±ξ'`.
- [`CGB.GradientFilter.estimated_curvature_residual_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L458)
- [`CGB.GradientFilter.eigenbasis_decouples`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L463)
- [`CGB.GradientFilter.trust_with_deadband_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L471)
- [`CGB.GradientFilter.linGain_closed`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L495)
- [`CGB.GradientFilter.linear_damage_eq_adoption`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L500)
- [`CGB.GradientFilter.no_linear_gain_separates`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L507)

## GenerationGrowth（16）
_Growth of a connection unit (generation chain, G18) A connection unit sits between two compartments and carries its two end activities `a b : Fin 2` and its slot occupancy `w`._

- [`CGB.Generation.binary_pair_decomposition`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L25)：**G18a**: the Möbius decomposition.
- [`CGB.Generation.interaction_unique`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L32)：**G18a (uniqueness)**: if `f` is written with unary parts and a product coefficient `κ`, then `κ` is the interaction coefficient.
- [`CGB.Generation.slot_reciprocity`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L41)：**G18b (reciprocity)**: for an energy `E w x` of a slot `w` and a co-activity `x`, the effect of `x` on the slot's flip equals the effect of the slot on `x`'s flip.
- [`CGB.Generation.occupancy_odds`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L50)：The detailed-balanced build/dismantle ratio fixes the occupancy: `occ / (1 - occ) = exp h`.
- [`CGB.Generation.occupancy_strictMono`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L60)：**G18c**: occupancy is strictly increasing in the field.
- [`CGB.Generation.one_pool_nested`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L77)：**G19a (one pool grows a rich club)**: with one shared pool, the strong set is an upper set in the rates: if `i` is strongly connected to `j` and `k` fires at least as much as `i`, then `k` is too.
- [`CGB.Generation.memoryless_coactivity`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L92)：**G19b (memoryless ends see only symbol frequencies)**: if every end state depends on the current symbol only, the co-activity, hence the grown structure, depends on the stream only through the frequencies of the current
- [`CGB.Generation.memoryless_same_frequencies_same_structure`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L99)
- [`CGB.Generation.memory_separates`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L113)：**G19c (sequence structure needs memory at the ends)**: the two streams have equal frequencies, so memoryless ends grow the same structure (G19b); an end that holds the previous symbol (`i` active on the previous 1, `j`
- [`CGB.Generation.two_state_geometric`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L137)：**G20a (two-state memory is geometric)**: the deviation from the fixed point `p* = kp / (kp + km)` shrinks by the same factor `1 - kp - km` every tick, so the trace of a past input is exponential; detailed balance fixes
- [`CGB.Generation.symmetric_exchange`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L152)：**G20b (symmetric exchange)**: two compartments exchanging content at the same rate `D` both ways (G6DB) keep their total and their difference shrinks by `1 - 2D`.
- [`CGB.Generation.symmetric_stays_symmetric`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L162)：**G21 (no differentiation without fluctuation)**: deterministic dynamics `f` that commute with a relabelling `σ` (P2c) keep a `σ`-symmetric birth state (P3) symmetric forever.
- [`CGB.Generation.fixed_point_is_occupancy`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L175)：The fixed point of G20a is the occupancy of the log odds `log kp - log km`.
- [`CGB.Generation.available_strictAnti`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L183)：**G22a (use depletes)**: the available fraction strictly falls as demand rises.
- [`CGB.Generation.inhibitory_slot_antitone`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L189)：**G22b (equilibrium inhibition retreats from co-activity)**: a slot species whose energy raises co-activity's cost (`κ < 0`) has occupancy strictly falling in co-activity.
- [`CGB.Generation.shared_potential_forces_nesting`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L196)：**G22c (no rich club needs local potentials)**: if all compartments share one potential and the strong set is not nested in rate order, contradiction (G19a).

## GenerationIgnition（5）
_Ignition needs drive or interaction (generation chain, G23) Ignition (global-workspace theory) is all-or-none, self-sustaining activation: the number of active units has two peaks (off and ignited) with a valley between_

- [`CGB.Generation.choose_log_concave`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L19)：Binomial coefficients are log-concave.
- [`CGB.Generation.activeWeight_log_concave`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L47)
- [`CGB.Generation.no_valley`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L61)：**G23 (no ignition in equilibrium with additive energies)**: the distribution of the number of active units has no strict valley, so it cannot have an "off" peak and an "ignited" peak separated by a gap.
- [`CGB.Generation.pulse_rate_le`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L83)
- [`CGB.Generation.equilibrium_memory_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L100)：**G24 (equilibrium memory bound)**: loading at rate `lp ≤ e^{|μ|/2} l0` for time `tp`, then relaxing at rate `l0` for time `T`, keeps at most `e^{|μ|/2 - 1} tp / T` of the deviation, for every choice of `l0`.

## GenerationInformationEnergy（7）
_Information and energy for a finite body in a stream (generation chain, 2026-10-28) The missing edges N1–N5 of `docs/information_boltzmann/CGB_THEOREM_RELATION_MAP_20261028.md`._

- [`CGB.InformationEnergy.freeEnergy_eq_kl`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L52)：**N1**: the free energy in the landscape `-log π` is the relative entropy to `π`.
- [`CGB.InformationEnergy.freeEnergy_at_law`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L58)：The law itself has free energy zero.
- [`CGB.InformationEnergy.kl_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L64)：**N1**: free energy is never below that of the law (Gibbs' inequality).
- [`CGB.InformationEnergy.snapshot_misses_hidden`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L93)：**N2**: a measurement that reads only the visible part carries no information about the hidden part beyond what it carries about the visible one.
- [`CGB.InformationEnergy.finite_memory_bounds_net_work`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L113)：**N3**: work drawn from acquired information `I` at most `kT · I`, erasure of everything beyond the capacity at least `kT` per unit: the net never exceeds `kT · cap`.
- [`CGB.InformationEnergy.passivity_ends`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L119)：**N4**: with drift `q > 0` and no observation (W2's recursion at gain zero), the information of one observation, `½ log (1 + V / r)`, exceeds any cost `c > 0` after finitely many ticks.
- [`CGB.InformationEnergy.mixture_regret_le`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L143)：**N5 (learning saturates)**: the Bayes mixture codes the sequence at most `log (1 / w θ₀)` worse than the source's law `θ₀` (pathwise, for every sequence and length).

## GenerationLatchHealth（3）
_When is a latch healthy? (generation chain, latch health) A latch (ignition, G23/G24) is a basin of states held by positive feedback._

- [`CGB.Medium.trap_irrecoverable`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L27)：**H1 (lock-in is death)**: if the set `B` is closed under the organism's own events and no state in it can take content in, no state in it is recoverable.
- [`CGB.Medium.trap_insensitive`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L41)：**H2 (latched means blind)**: where rates and effects do not depend on the stimulus, the one-tick kernel out of that state is the same under every stimulus.
- [`CGB.PairCRN.pair_rate_ratio`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L59)：**H1, quantitative**: a cue changes the rate of an event by exactly `exp((reservoir s - reservoir s') · dC / 2)`.

## GenerationLattice（14）
_Lattice media: cells with neighbour contact (generation chain, M4; nesting T-E1) The joint medium of an organism nested in its world._

- [`CGB.Medium.p2b_of_conductance`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L39)：**M1a with conductances**: rates `κ a x · exp (-(E y - E x) / 2)` with `κ` equal on an event and its reverse (read at the respective source states) give P2b.
- [`CGB.LatticeCRN.sum_update`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L151)
- [`CGB.LatticeCRN.sum_update2`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L159)
- [`CGB.LatticeCRN.adj_ne`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L167)
- [`CGB.LatticeCRN.p1a`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L171)
- [`CGB.LatticeCRN.p1b`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L174)
- [`CGB.LatticeCRN.p2a`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L219)
- [`CGB.LatticeCRN.act1_ne`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L263)
- [`CGB.LatticeCRN.act2_ne`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L266)
- [`CGB.LatticeCRN.p2b`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L278)
- [`CGB.LatticeCRN.adj_aut`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L353)
- [`CGB.LatticeCRN.p2c`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L362)
- [`CGB.LatticeCRN.p3`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L396)
- [`CGB.LatticeCRN.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L423)：**M4 (lattice media are admitted)**: a lattice reaction medium satisfies the six clauses when its rule tables pass local obligations: content levels (P1a), conservation by rules among interior cells (P2a), a zero calm po

## GenerationLearning（7）
_When does a world reward learning? (generation chain, L-1 to L-3) C-1 (`GenerationLeaveRule`) says when a forager must compare now with the past within one life._

- [`CGB.Generation.innate_optimal_of_constant`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L27)：**L-1**: a constant average across generations makes the innate threshold optimal.
- [`CGB.Generation.learner_loses_of_constant`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L31)：**L-1**: in a constant world a learner that pays `κ > 0` loses to the instinct at every patch.
- [`CGB.Generation.innate_loss_quarter`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L37)：**L-2**: with two different averages every innate threshold loses at least a quarter of their gap at some patch in one of the two worlds.
- [`CGB.Generation.learner_beats_innate`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L53)：**L-2**: a learner (who uses the world's own average after paying `κ`) beats every instinct somewhere when its cost is below a quarter of the gap.
- [`CGB.Generation.history_useless_of_independent`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L61)：**L-3**: if the next world state `c` (law `p`) is independent of the history `e` (law `r`) from which the rule `σ e` is chosen, the expected value is at most that of the best fixed rule.
- [`CGB.Generation.settled_fixed_optimal`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L84)：**O-1**: if the world settles (its average constant from `T` on), the fixed rule `τ = c T` is optimal at every later time and patch: nothing is left to compare or learn.
- [`CGB.Generation.beaten_forever_iff_never_settles`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L92)：**O-2**: every fixed rule is beaten after every time if and only if the world never settles (after every time it still takes two different averages).

## GenerationLeaveRule（3）
_When must a forager compare now with the past? (generation chain, C-1) A forager at a patch with local gain rate `q` either stays (earning `q`) or leaves for the rest of the world (worth `c`, the environment's average ra_

- [`CGB.Generation.leaveValue_le`](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L26)
- [`CGB.Generation.fixed_threshold_optimal`](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L30)：In a stationary world the fixed threshold `τ = c` is optimal everywhere.
- [`CGB.Generation.no_fixed_leave_threshold`](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L38)：**C-1 (a changing world needs comparison)**: with two different averages `c₁ < c₂`, every fixed threshold `τ` loses strictly at some patch rate `q` between them, in one of the two worlds.

## GenerationLifeGate（9）
_Life necessities as admission obligations (generation chain, gate layer L) The axioms P1–P3 say what a medium is._

- [`CGB.Medium.kernel_self_le`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L28)：One state-changing event of rate `r` makes the chain leave `x` within a tick with probability at least `r / R`.
- [`CGB.Medium.hold_vanishes`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L48)：**No exact latch**: if some state-changing event has rate at least `η > 0` under every stimulus, then the probability of sitting in `x` for `n` ticks under any stimulus sequence is at most `(1 - η / R) ^ n`, which tends
- [`CGB.PairCRN.rate_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L69)：Every rule with a positive prefactor fires on every ordered pair of distinct units at a positive rate, under every stimulus.
- [`CGB.PairCRN.exposed_of_witness`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L78)：**Life gate L-sense** (`Lives.exposed`): a pair medium is exposed as soon as one rule with a positive prefactor moves content through a boundary reservoir whose potential differs between two stimuli.
- [`CGB.PairCRN.total_pairAct`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L89)：The content an event takes in is the change of total content.
- [`CGB.PairCRN.metabolizes_of_witness`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L97)：**Life gate L-metabolism** (`Lives.metabolizes`): one content-changing event with a positive prefactor is a witness.
- [`CGB.tableCRN.dC_witness`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L116)
- [`CGB.tableCRN.metabolizes_of_rule`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L128)：**L-metabolism for rule tables**: one rule with a positive prefactor whose two sides carry different content, on two distinct units of its pages.
- [`CGB.tableCRN.exposed_of_rule`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L146)：**L-sense for rule tables**: the same witness on a rule touching a boundary page, and two stimuli at different reservoir potentials.

## GenerationLinearMemory（7）
_Linear recurrent memory cannot latch (generation chain, corollary of G17/G24) Gated DeltaNet (and every gated linear recurrence: Mamba, RWKV, linear attention with decay) updates each row `x` of its state as `x ↦ α (x -_

- [`CGB.Generation.dot_delta`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L34)
- [`CGB.Generation.delta_nonexpansive`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L48)：The delta step never expands (squared norm), for `0 ≤ β ≤ 2`.
- [`CGB.Generation.gated_delta_contracts`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L58)：One gated delta step on the difference of two states (the write cancels): the squared gap shrinks by `α²`.
- [`CGB.Generation.deltaStep_sub`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L67)
- [`CGB.Generation.histories_merge`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L75)：**Histories merge**: if every step shrinks the gap between two states by at most `α t`, the gap after `T` steps is at most `∏ α t` times the initial gap.
- [`CGB.Generation.histories_merge_uniform`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L87)：With a uniform gate bound the gap decays at least geometrically.
- [`CGB.Generation.linear_hold_keeps_error`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L102)：**A linear hold cannot clean up**: if a linear map holds the signal `s` (`L s = s`), a perturbation along `s` is held exactly as well, `L (s + t • s) - s = t • s`.

## GenerationMedium（2）
_A generator of admitted media (generation chain, M1) Instead of proving the six clauses case by case for each organism, prove once that a medium built from a local energy satisfies them, so an organism is admitted by giv_

- [`CGB.Medium.p2b_of_energy`](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L24)：**M1a**: symmetric (Metropolis-type) rates of a reversible event set under an energy give P2b.
- [`CGB.Medium.rate_local_of_energy`](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L45)：**M1b**: a rate that is a function of the participants' states through the event's energy difference reads only the participants, when that difference does.

## GenerationNestedWorld（66）
_An organism nested in its world (T-E2) The joint medium of a bacterium-like organism and its world, as a lattice medium (M4)._

- [`CGB.poolShift_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L48)
- [`CGB.poolShift_val`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L61)
- [`CGB.NestedWorld.sdir_not`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L191)
- [`CGB.NestedWorld.drain_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L279)
- [`CGB.NestedWorld.drain_drain`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L294)
- [`CGB.NestedWorld.canDrain_drain`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L302)
- [`CGB.NestedWorld.gateCount_drain`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L305)
- [`CGB.NestedWorld.move_exchanges`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L374)：**A move is an exchange**: whatever occupies the cell ahead (vacuum or a body), a running mover whose fuel rule fires takes its place and the occupant takes the mover's, unchanged; pools stay.
- [`CGB.NestedWorld.poolCond_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L402)
- [`CGB.NestedWorld.poolCond_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L405)
- [`CGB.NestedWorld.poolFactor_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L411)
- [`CGB.NestedWorld.poolFactor_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L414)
- [`CGB.NestedWorld.poolCond_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L417)
- [`CGB.NestedWorld.poolFactor_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L428)：Pools are bounded by `V`, so every pool change has positive multiplicity.
- [`CGB.NestedWorld.poolCond_self`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L432)
- [`CGB.NestedWorld.poolCond_step`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L435)
- [`CGB.NestedWorld.mass_action_up`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L442)：**Mass action** from 7.1: with the entropic pool energy `cellF`, the lumped conductance makes a one-quantum gain of a pool with `m` quanta fire at a rate `∝ V - m` (its free places) ...
- [`CGB.NestedWorld.mass_action_down`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L458)：...
- [`CGB.NestedWorld.uniport_selectivity`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L479)：**Uniport selectivity is fixed by free energy**: a reversible exchange that moves one food quantum into the body (inner energy change `D`, food energy included) fills at food `m + 1` and drains back at food `q`; the rati
- [`CGB.NestedWorld.antiport_selectivity`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L497)：**Antiport selectivity**: the same exchange coupled to the waste pool (one food quantum in, one waste quantum out, inner energy change `D`) fills at food `m + 1`, waste `w` and drains at food `q`, waste `z + 1`; the rati
- [`CGB.NestedWorld.apply_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L556)
- [`CGB.NestedWorld.sum_pair_update`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L569)
- [`CGB.NestedWorld.pair_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L579)
- [`CGB.NestedWorld.fuelAct_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L590)
- [`CGB.NestedWorld.innerAct_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L599)
- [`CGB.NestedWorld.fuelAct_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L618)
- [`CGB.NestedWorld.gateCount_update`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L637)
- [`CGB.NestedWorld.innerAct_delta`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L652)：A pair rule that fired moved the content the pools gave up into the body (or back).
- [`CGB.NestedWorld.gateCount_innerAct`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L690)：A pair rule away from the membrane page leaves the membrane as it is.
- [`CGB.NestedWorld.c1_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L715)
- [`CGB.NestedWorld.c2_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L727)
- [`CGB.NestedWorld.content_p1a`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L761)
- [`CGB.NestedWorld.not_bid`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L791)
- [`CGB.NestedWorld.sum_update_nat`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L795)
- [`CGB.NestedWorld.cons1`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L805)
- [`CGB.NestedWorld.cons2`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L878)
- [`CGB.NestedWorld.eff1_xch`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L962)
- [`CGB.NestedWorld.xchEff_gate`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L969)：An exchange through a page other than the membrane leaves the membrane as it is.
- [`CGB.NestedWorld.xchEff_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L982)
- [`CGB.NestedWorld.back1`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1003)
- [`CGB.NestedWorld.eff2_diff`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1086)
- [`CGB.NestedWorld.diffEff_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1090)
- [`CGB.NestedWorld.back2`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1111)
- [`CGB.NestedWorld.div_gate`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1201)
- [`CGB.NestedWorld.cellGate_inner`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1214)
- [`CGB.NestedWorld.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1235)：**T-E2 (an organism nested in its world is admitted)**: the joint medium of organisms and their world satisfies the six axiom clauses as soon as the organism's inner pair rules and its fuel rule conserve content and no e
- [`CGB.NestedWorld.orgIn_congr`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1324)
- [`CGB.NestedWorld.orgIn_inner`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1332)：A pair rule that does not couple to the pools (`sdf + sdw = 0`) keeps the organism's content.
- [`CGB.NestedWorld.eff1_supply_bug`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1352)：The world's supply rules never touch the organism.
- [`CGB.NestedWorld.intake_is_exchange`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1362)：**Only events across the boundary feed the organism**: every single-cell event that raises the content held by the organism is a pool exchange or a pair rule coupled to the pools; a gated one has conductance proportional
- [`CGB.NestedWorld.dead_intake_is_physics`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1401)：**Death** (`Part.CanTakeIn` fails through the organism's own doing): with no intact membrane unit, every event that would raise the organism's content has rate zero, except an ungated exchange: physics the organism does
- [`CGB.NestedWorld.dead_output_is_physics`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1420)：**A dead body does not give itself away**: every single-cell event that lowers the content held by the organism is a pool exchange; with no intact membrane the gated ones have rate zero, so what leaves a dead body leaves
- [`CGB.NestedWorld.frees_only_by_dissolving`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1461)：**Only dissolution empties a place** (with G26c `corpses_must_go`, why the table needs it): every single-cell event that turns a cell holding a body into a cell without one is the dissolution of a body.
- [`CGB.NestedWorld.births_only_by_division`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1488)：**Births come only from division** (the dual of `frees_only_by_dissolving`): a neighbour event after which the pair holds more bodies is a division.
- [`CGB.NestedWorld.partner_exit_blocked`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1522)：A pair rule whose partner state is absent leaves the organism as it is.
- [`CGB.NestedWorld.gated_exit_blocked`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1537)：A gated exchange has no conductance without an intact membrane.
- [`CGB.NestedWorld.poolShift_one`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1541)
- [`CGB.NestedWorld.leak_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1553)：**Leak bound**: an ungated exchange that moves one quantum from a unit into the waste pool fires at rate at least `κ * exp (-Φ)` whenever the unit holds that quantum and the pool is not full, whatever the rest of the org
- [`CGB.NestedWorld.intake_eff`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1621)
- [`CGB.NestedWorld.intake_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1636)
- [`CGB.NestedWorld.metabolizes_of_intake`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1654)：**Nested metabolism** (`Part.Metabolizes`) for every organism whose table has a gated food intake of one quantum, in a state with an intact membrane and a unit ready to take in.
- [`CGB.NestedWorld.senses_of_intake`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1681)：**Nested sensing** (`Part.Senses`): the same intake event, from the same organism, fires at different rates when only the cell's food differs (one quantum or two).
- [`CGB.NestedWorld.ringAdj_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1753)
- [`CGB.NestedWorld.torusAdj_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1761)
- [`CGB.NestedWorld.axioms_torus`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1787)：**The two-dimensional world is a model of the axioms**: organisms on a torus, food pellets from the world's boundary, diffusion, waste sinks, driven movement in four headings.
- [`CGB.NestedWorld.axioms_ring`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1799)：The one-dimensional ring is a model too.

## GenerationNesting（8）
_Nesting: the world is a medium too (generation chain, G26) An organism is a medium; the world it lives in obeys the same axioms, and so does a body that hosts organisms (mitochondria, gut bacteria) or a collective made o_

- [`CGB.Generation.cell_potential`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L39)：**G26a (the sensed potential of a world cell)**.
- [`CGB.Generation.site_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L45)：Conservation at a site: food after `T` ticks plus everything eaten is at most the initial food plus the regeneration.
- [`CGB.Generation.stay_starves`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L58)：**G26b (staying put starves)**: with food never negative, regeneration at most `r`, and a reserve that gains at most what is eaten and pays `c` per tick, the reserve after `T` ticks is at most `R₀ + f₀ + (r - c) T`.
- [`CGB.Generation.stay_dies`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L74)：The reserve is exhausted in finite time when living costs more than the site regenerates.
- [`CGB.Generation.cell_step`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L91)：Adding one quantum to a cell holding `n` of `V` costs exactly `log ((n + 1) / (V - n))`, so to a unit exchanging quanta with it the cell is a reservoir of potential `≈ log (n / V) = ln c`, and one exchange moves that pot
- [`CGB.Generation.missProb_ge`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L118)
- [`CGB.Generation.stream_never_dry`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L140)：**T-N7 (the stream never runs dry)**: under a stationary law `π` of the world–organism process, if some state `x` (weight `π x`) has two possible successors with different boundary readings, then every predictor of the n
- [`CGB.Generation.corpses_must_go`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L161)：**G26c (corpses must go: decomposition is necessary)**: in a world with finitely many places, if new deaths never stop (an endless stream of mortal lives: at every time some place that holds no corpse yet will later hold

## GenerationOptimizer（19）
_Optimizer interfaces forced by the generation chain This file does not classify Adam or Muon._

- [`CGB.Generation.coordinate_separable_has_no_pairwise_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L42)：Coordinate-separable state cannot carry a nonzero pairwise interaction.
- [`CGB.Generation.pairwise_relation_requires_cross_term`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L55)：Equivalently, a nonzero relation forces a cross-term in every unary decomposition.
- [`CGB.Generation.constant_gain_not_local_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L69)：A demand-independent gain cannot satisfy local anti-dominance.
- [`CGB.Generation.reservoir_availability_is_local_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L78)
- [`CGB.Generation.reservoir_gain_is_local_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L87)
- [`CGB.Generation.local_anti_dominance_requires_demand_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L95)：Any anti-dominant gain must read the compartment's local demand: it cannot be constant in that second argument.
- [`CGB.Generation.anti_dominance_favors_underdemanded`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L108)：Under local anti-dominance, an under-demanded compartment receives strictly higher marginal availability than a dominant compartment with greater past demand.
- [`CGB.Generation.shared_global_gain_fails_anti_dominance`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L119)
- [`CGB.Generation.separable_and_global_fails_cgb`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L131)：A coordinate-separable state paired with a shared global gain fails both CGB optimizer requirements.
- [`CGB.Generation.cgb_relation_carrier_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L147)：The CGB carrier has exact interaction `κ`.
- [`CGB.Generation.cgb_carrier_has_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L153)：When coupling `κ ≠ 0`, the CGB carrier is a genuine pairwise relation carrier.
- [`CGB.Generation.cgb_optimizer_admitted`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L165)：The CGB optimizer satisfies both CGB admission requirements.
- [`CGB.Generation.adam_class_has_positive_relational_error`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L178)：Any coordinate-separable (Adam-class) optimizer incurs strictly positive relational error whenever the target coupling is nonzero.
- [`CGB.Generation.cgb_optimizer_zero_relational_error`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L190)：The CGB optimizer achieves zero relational error on the target coupling.
- [`CGB.Generation.cgb_strictly_beats_adam_on_relations`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L197)：Formal Superiority over Adam: CGB optimizer strictly dominates any coordinate-separable optimizer on relational error.
- [`CGB.Generation.muon_class_zero_tail_protection`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L217)：Any shared global/spectral (Muon-class) optimizer provides exactly zero protective margin between compartments: the dominant and tail compartments compete under identical un-depleted gain.
- [`CGB.Generation.cgb_optimizer_positive_tail_protection`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L225)：The CGB reservoir optimizer provides a strictly positive protective margin to the under-demanded compartment whenever the dominant compartment has greater demand.
- [`CGB.Generation.cgb_strictly_beats_muon_on_tail_protection`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L236)：Formal Superiority over Muon: CGB optimizer strictly dominates any shared global gain rule on the tail protection margin, actively suppressing rich-club runaway.
- [`CGB.Generation.cgb_optimizer_pareto_superiority`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L260)：**Theorem: CGB Optimizer Strictly Pareto-Dominates the Adam-Muon Hybrid**.

## GenerationOrganismEcoliMsS0（3）
_Organism `ecoli_ms_s0`: generated from its rule table Source: `information_boltzmann/organisms/ecoli_ms_s0.json`, sha256 38ff4daa364a8d382398d91208cc7908b61c82d1a1324e64111291ef4801fa1b._

- [`CGB.Organisms.EcoliMsS0.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L52)：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).
- [`CGB.Organisms.EcoliMsS0.metabolizes`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L58)：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 1024 and 1735 changes content.
- [`CGB.Organisms.EcoliMsS0.exposed`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L65)：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

## GenerationOrganismEcoliV1（3）
_Organism `ecoli_v1`: generated from its rule table Source: `information_boltzmann/organisms/ecoli_v1.json`, sha256 4ab0545552e85d06779b9d1da4a1c49b88bd976203fae895833f9a148ccb6a65._

- [`CGB.Organisms.EcoliV1.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L52)：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).
- [`CGB.Organisms.EcoliV1.metabolizes`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L58)：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 17 and 1 changes content.
- [`CGB.Organisms.EcoliV1.exposed`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L65)：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

## GenerationOrganismGeneratedS0（3）
_Organism `generated_s0`: generated from its rule table Source: `information_boltzmann/organisms/generated_s0.json`, sha256 d57065ecec7500d38a2b40e1ef94f9e8412f45e47f7a7df63d20421010ccd6b0._

- [`CGB.Organisms.GeneratedS0.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L50)：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).
- [`CGB.Organisms.GeneratedS0.metabolizes`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L56)：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 14 and 0 changes content.
- [`CGB.Organisms.GeneratedS0.exposed`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L63)：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

## GenerationPersistence（6）
_Activity without input: driven circulation, adaptation, latches (generation chain, 2026-10-27) Three different things keep a body active when the input says nothing, and they must not be confused (eco-sphere audit: 154 o_

- [`CGB.Generation.cycle_rates_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L46)：**A (Kolmogorov, one direction)**: detailed balance with a positive weight balances the rate products around every cycle `c 0 → c 1 → … → c (n-1) → c 0`.
- [`CGB.Generation.driven_cycle_not_balanced`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L61)：**A**: a cycle with unequal forward and reverse rate products (net circulation) admits no positive detailed-balance weight: self-repeating activity needs a drive.
- [`CGB.Generation.integral_feedback_adapts`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L73)：**B (integral feedback adapts perfectly)**: if the reference update `m ↦ m + γ (f (s - m) - a₀)` with `γ ≠ 0` has `m` as a fixed point, the activity there is `a₀`, whatever the input `s`.
- [`CGB.Generation.adapted_gap_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L82)：**B**: an act driven only by the gap `g (a - a₀)` with `g 0 = 0` stops at every adapted fixed point of a constant input.
- [`CGB.Generation.contraction_unique_rest`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L92)：**C**: a loop map with gain `L < 1` has at most one rest state: tonic activity without enough loop gain is monostable.
- [`CGB.Generation.latch_needs_unit_gain`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L102)：**C**: two distinct rest states of a differentiable loop map force loop gain (slope) exactly one somewhere between them: a latch needs unit gain.

## GenerationPredictiveHarvest（9）
_Prediction is metabolism under reinvestment (language-stream life, T1) World contract (declared, a niche like the E._

- [`CGB.PredictiveHarvest.growth_eq_log_card_sub_crossEntropy`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L41)
- [`CGB.PredictiveHarvest.gibbs_term`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L53)：One term of Gibbs' inequality: `x (log y - log x) ≤ y - x` (from `log t ≤ t - 1`).
- [`CGB.PredictiveHarvest.gibbs_inequality`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L63)：**Gibbs' inequality**: `Σ p log c ≤ Σ p log p` for a positive allocation summing to one.
- [`CGB.PredictiveHarvest.growth_le`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L75)：**Kelly bound**: no allocation grows faster than `log |B| - H(p)`.
- [`CGB.PredictiveHarvest.growth_at_law`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L83)：Placing capacity as the stream's law attains the bound.
- [`CGB.PredictiveHarvest.uniform_growth_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L96)：A body that knows nothing about the stream (uniform allocation) does not grow.
- [`CGB.PredictiveHarvest.no_allocation_persists`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L106)：**No body persists** on a stream whose entropy exceeds `log |B| - m`: every allocation's net growth (growth minus a maintenance cost `m` per arrival) is negative.
- [`CGB.PredictiveHarvest.persists_of_crossEntropy`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L113)：A body whose cross-entropy is below `log |B| - m` has positive net growth.
- [`CGB.PredictiveHarvest.additive_payoff_mode`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L122)：Without reinvestment (payoff additive in the capacity placed on the arrived byte) the best allocation puts everything on a most likely byte: probability matching is a consequence of the multiplicative contract, not of pr

## GenerationSpeechTempo（14）
_Speech tempo between two finite bodies (generation chain, 2026-10-28, edge N7) A speaker emits units (bytes, syllables) at rate `r` per tick to a listener._

- [`CGB.SpeechTempo.unsettled_eq_rpow`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L59)：The settling factor is the unified reference's geometric bound: `(1 - α) ^ (k / r)` with `L = -log (1 - α)`.
- [`CGB.SpeechTempo.retention_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L65)
- [`CGB.SpeechTempo.retention_le_one`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L67)
- [`CGB.SpeechTempo.settled_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L73)
- [`CGB.SpeechTempo.settled_le_one`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L80)
- [`CGB.SpeechTempo.settled_le`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L84)
- [`CGB.SpeechTempo.useful_le_rate`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L93)
- [`CGB.SpeechTempo.useful_le_capacity`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L105)
- [`CGB.SpeechTempo.useful_le_inv_rate`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L115)
- [`CGB.SpeechTempo.tempo_band`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L127)：**N7**: the good tempo is a bounded band.
- [`CGB.SpeechTempo.slow_retention_vanishes`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L139)：Too slow: per unit, the earlier part is forgotten, whatever the listener's capacity.
- [`CGB.SpeechTempo.fast_settling_vanishes`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L155)：Too fast: per unit, the speaker is unsettled, whatever the listener's capacity.
- [`CGB.SpeechTempo.peaks_wasted`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L166)：Uniform information density, upper bound: with a per-slot capacity `C`, the slots absorb at most the total sent and at most `n C`; anything above capacity in a slot is lost.
- [`CGB.SpeechTempo.uniform_attains`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L173)：Uniform information density, attainment: the even spread reaches the bound.

## GenerationTable（2）
_Rule-table media (generation chain, M3) An organism is a finite table of reversible two-unit transitions: "a pair of units with identity pages `(i, j)` in states `a` may go to states `b`", each with a prefactor `κ`._

- [`CGB.RuleSpec.apply_back`](../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L34)
- [`CGB.tableCRN.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L72)：**M3**: a rule-table medium is a model of the CGB axioms as soon as its content has a unique free state over a finite local state type, the calm reservoir potential is zero, and every rule between two interior pages cons

## GenerationUnifiedOptimizer（27）
_Unified CGB optimizer contract This module is a composition layer, not a new axiom and not a claim about a trained language model._

- [`CGB.Unified.build_is_cgb_candidate`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L58)：The composed relation and gain satisfy the existing CGB admission contract.
- [`CGB.Unified.balanced_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L70)：A bounded exchange fraction gives a nonnegative allocation.
- [`CGB.Unified.balanced_col_budget`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L76)：A balanced allocation preserves every column budget.
- [`CGB.Unified.balanced_row_budget`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L81)：A balanced allocation preserves every row budget.
- [`CGB.Unified.balanced_allocation_contract`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L86)：The double-budget contract bounds the positive-feedback payoff.
- [`CGB.Unified.balanced_max_differentiates`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L97)：At the doubly-budgeted maximum, the two rows own different directions.
- [`CGB.Unified.uniform_allocation_is_budgeted`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L106)
- [`CGB.Unified.balanced_allocation_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L119)
- [`CGB.Unified.balanced_allocation_has_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L125)
- [`CGB.Unified.phase_relation_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L139)
- [`CGB.Unified.phase_relation_has_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L144)
- [`CGB.Unified.phase_history_is_generated`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L150)
- [`CGB.Unified.phase_step_stays_in_unit_interval`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L156)
- [`CGB.Unified.phase_history_stays_in_unit_interval`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L167)
- [`CGB.Unified.real_transfer_total`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L189)
- [`CGB.Unified.real_transfer_target_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L193)
- [`CGB.Unified.phase_exchange_transfer_is_conserved`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L207)
- [`CGB.Unified.phase_exchange_transfer_reads_phase`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L213)
- [`CGB.Unified.fresh_outcome_measurement_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L228)
- [`CGB.Unified.fresh_outcome_trust_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L234)
- [`CGB.Unified.ema_history_collision`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L254)
- [`CGB.Unified.build_has_competition_and_differentiation`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L261)：The built architecture exposes both row and column competition budgets.
- [`CGB.Unified.build_history_is_geometric`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L273)：The retained history is the geometric trace generated by two-state exchange.
- [`CGB.Unified.build_history_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L281)：The supplied typed carrier exposes its telescoping conserved history.
- [`CGB.Unified.build_modulation_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L293)：The outcome gate leaves a dead band and stays inside the trust interval.
- [`CGB.Unified.build_latch_holds_and_releases`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L302)：The latch both holds under short contrary evidence and releases at its threshold.
- [`CGB.Unified.build_mechanical_admission`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L313)：Combined mechanical admission: relation, competition, history and modulation.

## GraceRecovery（20）
_A recoverable diagnostic opportunity after missed sensing Keep the original body, learning payments, food amounts and clock._

- [`CGB.GraceRecovery.policy_table`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L50)
- [`CGB.GraceRecovery.moves_unchanged_on_good`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L68)
- [`CGB.GraceRecovery.good_safe`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L73)
- [`CGB.GraceRecovery.safe_nonvacuum`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L76)
- [`CGB.GraceRecovery.normal_good`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L79)
- [`CGB.GraceRecovery.normal_safe`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L86)
- [`CGB.GraceRecovery.idle_safe`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L92)
- [`CGB.GraceRecovery.normal_zero`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L99)
- [`CGB.GraceRecovery.idle_progress`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L107)
- [`CGB.GraceRecovery.supported`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L117)：Supported body events execute exactly the old local update; an idle slot has zero incoming material.
- [`CGB.GraceRecovery.probe_potential`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L157)
- [`CGB.GraceRecovery.stimulus_before`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L162)
- [`CGB.GraceRecovery.stimulus_idle`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L166)
- [`CGB.GraceRecovery.stimulus_after`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L169)
- [`CGB.GraceRecovery.scheduled_closed`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L173)
- [`CGB.GraceRecovery.scheduled_progress`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L200)
- [`CGB.GraceRecovery.positive`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L240)
- [`CGB.GraceRecovery.one_tick_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L251)
- [`CGB.GraceRecovery.one_tick_tolerance`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L258)
- [`CGB.GraceRecovery.natural_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L291)

## GraceRecoveryAccounting（14）
_Unchanged paid learning and actual source accounting through the grace world All ledgers use the actual registered protocol, including the interrupted tick._

- [`CGB.GraceRecoveryAccounting.body_supported`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L26)
- [`CGB.GraceRecoveryAccounting.credit_step`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L66)
- [`CGB.GraceRecoveryAccounting.path_budgets`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L84)：Exact budgets on EVERY supported finite path, including interrupted and cut paths.
- [`CGB.GraceRecoveryAccounting.realized`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L117)
- [`CGB.GraceRecoveryAccounting.initial_energy`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L155)
- [`CGB.GraceRecoveryAccounting.source_energy_budget`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L166)
- [`CGB.GraceRecoveryAccounting.old_path_good`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L191)
- [`CGB.GraceRecoveryAccounting.normal_step_unchanged`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L206)
- [`CGB.GraceRecoveryAccounting.arbitrary_relearning`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L217)
- [`CGB.GraceRecoveryAccounting.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L239)
- [`CGB.GraceRecoveryAccounting.actual_law`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L251)
- [`CGB.GraceRecoveryAccounting.score_preserved`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L269)
- [`CGB.GraceRecoveryAccounting.physical_one_tick_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L292)
- [`CGB.GraceRecoveryAccounting.calm_has_no_intake`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L297)

## GraceRecoveryClosure（18）
_Original marginal-ghost cut after diagnostic grace._

- [`CGB.GraceRecoveryClosure.policy_table`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L17)
- [`CGB.GraceRecoveryClosure.sum_cells`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L23)
- [`CGB.GraceRecoveryClosure.sum_configs`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L30)
- [`CGB.GraceRecoveryClosure.sum_env`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L38)
- [`CGB.GraceRecoveryClosure.kernel_table`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L45)
- [`CGB.GraceRecoveryClosure.early_move`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L75)：Until tick four the body marginal is a point mass, so the ORIGINAL cut transition is derived from that marginal, not assigned a convenient law.
- [`CGB.GraceRecoveryClosure.cut_early`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L88)
- [`CGB.GraceRecoveryClosure.move_five`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L106)
- [`CGB.GraceRecoveryClosure.bad_six`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L116)
- [`CGB.GraceRecoveryClosure.no_dead_six`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L127)
- [`CGB.GraceRecoveryClosure.miss_to_probe_six`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L137)
- [`CGB.GraceRecoveryClosure.step_lower`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L149)
- [`CGB.GraceRecoveryClosure.dead_seed`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L180)：A quarter of mass has physically lost its body by tick seven.
- [`CGB.GraceRecoveryClosure.empty_kernel`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L188)
- [`CGB.GraceRecoveryClosure.dead_monotone`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L193)
- [`CGB.GraceRecoveryClosure.cut_mass_bound`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L227)
- [`CGB.GraceRecoveryClosure.cut_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L256)
- [`CGB.GraceRecoveryClosure.maintenance`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L265)

## GraceRecoveryControls（4）
_Separate controls for diagnostic grace and recovery lookahead Changing the lookahead alone does not repair the old absorbing failure._

- [`CGB.GraceRecoveryControls.point_loss`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L23)
- [`CGB.GraceRecoveryControls.off_trap_recovery`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L45)
- [`CGB.GraceRecoveryControls.horizon_change_alone_fails`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L67)
- [`CGB.GraceRecoveryControls.two_tick_lookahead_insufficient`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L89)

## Inference（6）
_Active inference, first rungs: the blanket and foresight The life definition orders its emergent part as a ladder: memory → world model and prediction → choosing actions by what is predicted._

- [`CGB.Medium.jump_blanket`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L50)：Under P3, the full dynamics reads the stimulus only at the boundary.
- [`CGB.Medium.kernel_blanket`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L56)
- [`CGB.Medium.blanket_of_axioms`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L73)：P3 gives the organism's half of the blanket; a world that touches only the body gives the other half.
- [`CGB.Models.rock_cannot_eat`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L119)：The rock satisfies all six axioms but can never take anything in: it was never alive.
- [`CGB.Models.rock_not_persists`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L124)：So the rock persists in no world.
- [`CGB.finite_reweighting_improves_iff`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L134)：Any positive finite reweighting improves an observed event's normalized probability exactly when its conditional mean multiplier exceeds the global mean.

## InformationThermodynamics（3）
_Information acquisition as a conditional thermodynamic contract The general medium axioms do not assign physical energy to a recorded symbol._

- [`CGB.InformationThermodynamics.erased_information_floor`](../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L38)
- [`CGB.InformationThermodynamics.positive_erasure_requires_positive_dissipation`](../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L42)
- [`CGB.InformationThermodynamics.efficient_recorder_is_not_free`](../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L51)

## Interaction（19）
_Finite probabilistic interaction coalgebras This is a semantic layer, not a physical axiom._

- [`CGB.Interaction.Machine.conditioned_next`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L33)：A positive-mass observation uniquely fixes the normalized next-state law.
- [`CGB.Interaction.trace_nonneg`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L54)
- [`CGB.Interaction.trace_prefix_consistent`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L64)：Cylinder consistency: summing all one-symbol extensions recovers the prefix probability.
- [`CGB.Interaction.Coupling.expectation_at_output`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L98)：One-step transfer of any observable continuation value.
- [`CGB.Interaction.Bisimulation.trace_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L115)
- [`CGB.Interaction.Bisimulation.sameBehavior`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L126)：Coinductive soundness for observed infinite behaviour, represented by ALL finite cylinders.
- [`CGB.Interaction.coinduction`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L130)
- [`CGB.Interaction.bisimilar_is_bisimulation`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L133)
- [`CGB.Interaction.bisimilar_sameBehavior`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L138)
- [`CGB.Interaction.distinguishing_trace`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L141)
- [`CGB.Interaction.Refines.bisimulation`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L152)
- [`CGB.Interaction.Refines.sameBehavior`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L167)
- [`CGB.Interaction.Refines.refl`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L170)
- [`CGB.Interaction.Refines.comp`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L175)：Certificates compose, so a chain of representation changes does not need to recheck arbitrary long sample traces after each implementation pass.
- [`CGB.Interaction.TimedRefinement.elapsed_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L196)
- [`CGB.Interaction.Bisimulation.firstHit_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L214)
- [`CGB.Interaction.Refines.firstHit_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L230)
- [`CGB.Interaction.Refines.recovery_test_iff`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L237)：To preserve the existing nonempty-body + recovery-threshold test, content must ALSO be represented faithfully.
- [`CGB.Interaction.firstHit_bounds`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L246)

## InteractionBridge（6）
_The existing medium is a probabilistic interaction coalgebra The observation wrapper preserves the full joint transition and its world tick._

- [`CGB.Interaction.Machine.ofKernels_step`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L27)
- [`CGB.Interaction.Machine.ofKernels_marginal`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L33)：Forgetting the observer's label recovers EXACTLY the original transition.
- [`CGB.Interaction.Machine.ofKernels_firstHit`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L40)：Observe a time protocol in the existing external world clock.
- [`CGB.Medium.intakeCoalgebra_firstHit`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L91)
- [`CGB.Medium.intakeCoalgebra_actualRecover`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L101)：This normal-world observation yields the existing actualRecover, not a different coalgebraic notion of success or a favourable-stimulus menu.
- [`CGB.Medium.refinement_preserves_recovery_and_time`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L112)：Actual recovery AND elapsed time transfer along a certified implementation.

## InteractionControls（7）
_Semantic certificates and obstructions These mathematical controls exercise exact representation, not brain capacity._

- [`CGB.InteractionControls.independent_bit_refines`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L29)：Removing a bit is justified HERE by a universal transition equation.
- [`CGB.InteractionControls.independent_bit_sameBehavior`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L35)
- [`CGB.InteractionControls.same_first_observation`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L46)
- [`CGB.InteractionControls.different_future`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L51)
- [`CGB.InteractionControls.no_bisimulation_from_first_observation`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L57)
- [`CGB.InteractionControls.physical_certificate_rejects_time_rescaling`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L63)
- [`CGB.InteractionControls.actual_recovery_bridge`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L71)：The tested representation is the identity on the actual continuous state; the bridge must work for BOTH normal and cut futures with the same world clock.

## Language（1）
_CGB language: finite-state Markov jump processes The state–event language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1._

- [`CGB.JumpProcess.expect_const_iff`](../../lean/CGBAxioms/CGBAxioms/Language.lean#L32)：`⟨f⟩` is constant for every initial distribution iff `L f = 0` pointwise.

## LanguageLifeKernel（53）
_Compact algebraic reference kernel for the language-life model This is a finite exact reference object, not a neural-network capability claim._

- [`CGB.LanguageLifeKernel.incidence_ternary`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L22)
- [`CGB.LanguageLifeKernel.incidence_sum_zero`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L28)
- [`CGB.LanguageLifeKernel.charge_total_preserved`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L38)
- [`CGB.LanguageLifeKernel.reserve_balance`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L50)
- [`CGB.LanguageLifeKernel.step_clock_advances`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L66)
- [`CGB.LanguageLifeKernel.step_charge_conserved`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L71)
- [`CGB.LanguageLifeKernel.step_reserve_paid`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L77)
- [`CGB.LanguageLifeKernel.no_visible_only_delayed_credit_rule`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L99)
- [`CGB.LanguageLifeKernel.delayed_credit_requires_history_carrier`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L107)
- [`CGB.LanguageLifeKernel.conserved_transfer_target_trace`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L135)
- [`CGB.LanguageLifeKernel.conserved_transfer_source_trace`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L140)
- [`CGB.LanguageLifeKernel.conserved_transfer_total_preserved`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L145)
- [`CGB.LanguageLifeKernel.observable_transfer_recovers_amount`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L165)
- [`CGB.LanguageLifeKernel.superposed_transfer_readout_is_net`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L169)
- [`CGB.LanguageLifeKernel.conservation_does_not_identify_components`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L173)
- [`CGB.LanguageLifeKernel.leaky_transfer_readout_is_net`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L185)
- [`CGB.LanguageLifeKernel.conserved_carrier_path_balance`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L204)
- [`CGB.LanguageLifeKernel.conserved_carrier_path_no_output`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L213)
- [`CGB.LanguageLifeKernel.history_readout_exists_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L230)
- [`CGB.LanguageLifeKernel.history_transition_exists_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L238)
- [`CGB.LanguageLifeKernel.external_causal_readout_exists_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L260)
- [`CGB.LanguageLifeKernel.external_trace_requires_retrieved_distinction`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L284)
- [`CGB.LanguageLifeKernel.observational_record_does_not_fix_intervention`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L297)：Two world equations agree on every recorded A=U trajectory but disagree under the intervention U=false, A=true.
- [`CGB.LanguageLifeKernel.SensoryRecord.exact_record_contract`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L346)
- [`CGB.LanguageLifeKernel.SensoryRecord.action_content_distinguishes`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L360)：All sixteen records already have distinct action/content observations.
- [`CGB.LanguageLifeKernel.SensoryRecord.record_state_lower_bound`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L366)
- [`CGB.LanguageLifeKernel.SensoryRecord.fold_refines`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L380)：The finite record supports arbitrary-length replay by a forward fold, without consulting the world regime or keeping an autograd history.
- [`CGB.LanguageLifeKernel.SensoryRecord.joint_successor_reconstruct`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L392)：Every positive-probability successor in the original joint body/world process is exactly representable, for either registered cut and any schedule.
- [`CGB.LanguageLifeKernel.CausalRecord.policy_cases`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L431)
- [`CGB.LanguageLifeKernel.CausalRecord.weights_exact`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L438)
- [`CGB.LanguageLifeKernel.CausalRecord.sensor_alias_witness`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L447)：Same present sensor, different consequences of the same actuator setting.
- [`CGB.LanguageLifeKernel.CausalRecord.update_sound`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L458)
- [`CGB.LanguageLifeKernel.CausalRecord.update_least`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L473)：Any candidate set retaining every compatible successor contains this one.
- [`CGB.LanguageLifeKernel.CausalRecord.action_feedback_witness`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L484)：An identical later sensor has different interpretations when the earlier action differs.
- [`CGB.LanguageLifeKernel.PaidRecord.settlement_contract`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L519)
- [`CGB.LanguageLifeKernel.PaidRecord.world_feedback_possible`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L538)
- [`CGB.LanguageLifeKernel.PaidRecord.combined_write_budget`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L554)：Add the recorder's separately endowed reserve to the ORIGINAL body's credit ledger.
- [`CGB.LanguageLifeKernel.PaidRecord.historical_frame_witness`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L567)：Holding a source hypothesis is sound about its origin, not about a moving world.
- [`CGB.LanguageLifeKernel.PaidRecord.joint_source_possible`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L576)：The ORIGINAL normal joint kernel, with any registered sensory schedule, licenses a receipt only for its pre-update body action and its successor's actual sensor.
- [`CGB.LanguageLifeKernel.PaidRecord.attach_refines`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L602)：Projection recovers the whole original kernel, hence its registered finite-future observations by Interaction.Refines.sameBehavior.
- [`CGB.LanguageLifeKernel.LocalEdit.binary_factor_balance`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L615)：A binary factor edit depends on joint contrast, not just either marginal.
- [`CGB.LanguageLifeKernel.LocalEdit.accepts_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L628)：With positive total masses, the executable comparison exactly detects improvement of the same event; inference and resource contracts are separate.
- [`CGB.LanguageLifeKernel.FirstHit.first_hit_sum`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L660)：Conditional on a fixed lifetime, first-success probabilities telescope.
- [`CGB.LanguageLifeKernel.FirstHit.truncation`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L671)：Stop unfolding after `horizon` trials.
- [`CGB.LanguageLifeKernel.FirstHit.probability_bounds`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L700)：A normalized nonnegative finite lifetime distribution gives a genuine probability, independently of the chosen support representation.
- [`CGB.LanguageLifeKernel.CausalState.realizes`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L748)：Canonical readout and single-input recursion realize the same contract.
- [`CGB.LanguageLifeKernel.CausalState.fold_refines`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L758)：Arbitrarily long forward recursion; not a bounded-memory implementation.
- [`CGB.LanguageLifeKernel.CausalState.minimal`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L767)：Every exact deterministic realization projects onto the canonical states.
- [`CGB.LanguageLifeKernel.CausalState.finite_bound`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L787)：Finiteness follows from a finite exact realization; it is not assumed.
- [`CGB.LanguageLifeKernel.CausalState.finite_alphabets_unbounded`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L797)：Finite input and output alphabets alone do not imply finite exact memory.
- [`CGB.LanguageLifeKernel.CausalState.separated_histories_capacity`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L810)：Uniform precision requires distinct readable codes for separated futures.
- [`CGB.LanguageLifeKernel.Approximation.fold_error`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L845)：Full one-step error certificates compose over any finite continuation.
- [`CGB.LanguageLifeKernel.Approximation.invariant_error`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L864)：A locally closed error budget stays uniform in elapsed sequence length.

## LearningMaintenance（26）
_A finite learning-and-maintenance mathematical witness Safe informative boundary feedback precedes resource-dependent action._

- [`CGB.LearningMaintenance.policy_table`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L39)
- [`CGB.LearningMaintenance.plant_admission`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L66)
- [`CGB.LearningMaintenance.admission`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L103)
- [`CGB.LearningMaintenance.reads_boundary`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L139)
- [`CGB.LearningMaintenance.sum_configs`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L165)
- [`CGB.LearningMaintenance.kernel_table`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L174)
- [`CGB.LearningMaintenance.natural_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L191)
- [`CGB.LearningMaintenance.recovery_step`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L214)
- [`CGB.LearningMaintenance.normal_recovery`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L241)
- [`CGB.LearningMaintenance.natural_alive`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L247)
- [`CGB.LearningMaintenance.natural_survival`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L263)
- [`CGB.LearningMaintenance.starve_world_step`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L282)
- [`CGB.LearningMaintenance.starve_recovery`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L294)
- [`CGB.LearningMaintenance.cut_move_early`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L314)
- [`CGB.LearningMaintenance.cut_early_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L328)
- [`CGB.LearningMaintenance.starve_mass_three`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L348)
- [`CGB.LearningMaintenance.starve_mass_monotone`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L358)
- [`CGB.LearningMaintenance.cut_alive_bound`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L384)
- [`CGB.LearningMaintenance.cut_survival_mass_bound`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L399)
- [`CGB.LearningMaintenance.cut_survival`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L426)
- [`CGB.LearningMaintenance.maintenance_witness`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L439)：The original gross-exchange goals hold during paid identification and all subsequent times.
- [`CGB.LearningMaintenance.starvation_depletes`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L452)：Scarcity really changes the body reserve, under either process; vacuum then has no action in any stimulus.
- [`CGB.LearningMaintenance.sustained_exchange`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L468)：Normal maintenance consumes and imports a positive unit EVERY active tick; this demand is a property of this mechanism, not a universal life axiom.
- [`CGB.LearningMaintenance.paid_learning`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L479)：The finite physical feedback update realizes the previous candidate filter, using the action chosen BEFORE feedback.
- [`CGB.LearningMaintenance.acquired_control`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L488)：Both branches use one common initial controller; actual feedback determines which singleton memory and specialized resource action it acquires.
- [`CGB.LearningMaintenance.normal_resources`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L497)：Physical fuel and registered resource counts along the exact normal law.

## LearningMaintenanceControls（8）
_A paired write-disabled control: same initial fuel, sensing, world and clock._

- [`CGB.LearningMaintenanceControls.sum_configs`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L27)
- [`CGB.LearningMaintenanceControls.kernel_table`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L36)
- [`CGB.LearningMaintenanceControls.policy_value`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L53)
- [`CGB.LearningMaintenanceControls.frozen_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L59)
- [`CGB.LearningMaintenanceControls.starve_recovery`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L85)
- [`CGB.LearningMaintenanceControls.starve_alive`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L98)
- [`CGB.LearningMaintenanceControls.frozen_survival`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L108)：With writing disabled the initial two fuel units cannot preserve access to food.
- [`CGB.LearningMaintenanceControls.frozen_not_persists`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L126)

## LearningMaintenanceEnergy（4）
_Declared fuel, actual consumption and paid learning The same registered normal/cut event process pays maintenance fuel at active steps and an additional write-fuel unit on the effective feedback update._

- [`CGB.LearningMaintenanceEnergy.actual_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L29)：The energy law is exactly the state law used in the viability comparison, including the cut's own distribution-dependent future, not a second process.
- [`CGB.LearningMaintenanceEnergy.body_supported`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L46)
- [`CGB.LearningMaintenanceEnergy.isothermal`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L87)：Conditional mean second-law admission for the explicitly declared internal entropy and bath units.
- [`CGB.LearningMaintenanceEnergy.availability_budget`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L110)：Every finite prefix uses the SAME event law and tick.

## Life（18）
_The infinite-stream life statement **Historical menu semantics (versioned 2026-10-23).** The `recover`, `aliveInd`, `Lives` and `LifeConjecture` below retain the old favourable-stimulus menu for compatibility._

- [`CGB.Medium.kernel_nonneg`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L104)
- [`CGB.Medium.kernel_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L111)：The kernel is stochastic.
- [`CGB.Medium.sum4`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L131)
- [`CGB.Medium.law_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L138)：The law is a probability distribution at every time.
- [`CGB.Medium.irrecoverable_absorbing`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L175)：**Losing recoverability is irreversible**: no event leads from an irrecoverable state back to a recoverable one.
- [`CGB.Medium.kernel_up_zero`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L196)：Without any possible intake, no step raises content.
- [`CGB.Medium.recover_eq_zero`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L211)：Without any possible intake, recovery is impossible.
- [`CGB.Medium.aliveInd_eq_zero`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L229)
- [`CGB.Medium.stepSum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L242)：One joint step, weighted by a test function `a` that the organism kernel preserves.
- [`CGB.Medium.margStep`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L261)：One joint step of the organism's marginal, when its kernel ignores the world.
- [`CGB.Medium.cutMove_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L276)
- [`CGB.Medium.lawCut_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L291)
- [`CGB.Medium.never_alive_without_intake`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L311)：An organism that can never take anything in is never alive, whatever it senses.
- [`CGB.Medium.not_persists_of_no_intake`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L332)：Without any possible intake there is no persistence.
- [`CGB.Medium.not_selfMaintains_of_unexposed`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L367)：An organism that ignores its stimuli cannot self-maintain: its own law does not depend on the world, so cutting the loop changes nothing.
- [`CGB.Medium.not_selfMaintains_of_closedMatter`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L397)：An organism that exchanges no content with the world cannot self-maintain: whether it holds content never changes, in the loop or out of it.
- [`CGB.Medium.Lives.exposed`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L431)：A living organism necessarily senses its world.
- [`CGB.Medium.Lives.metabolizes`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L435)：A living organism necessarily exchanges content with its world (metabolism).

## LifeControls（25）
_Definition controls prompted by passive absorption and frozen-copy objections The old reflex/self-model advantages remain withdrawn._

- [`CGB.FirstHit.during_zero_on_closed`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L20)：Protocol-dependent extension of zero_on_closed, also valid for the future marginal-ghost cut.
- [`CGB.Operational.recovery_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L44)：No possible positive-rate intake excludes recovery on BOTH registered future protocols, for every current joint state and every horizon.
- [`CGB.Operational.alive_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L62)
- [`CGB.Operational.survival_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L69)
- [`CGB.Operational.not_persists_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L74)
- [`CGB.Operational.not_selfMaintains_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L81)
- [`CGB.Operational.not_persists_of_frozen`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L92)：This migrates the fully-frozen-copy exclusion to the actual-world target.
- [`CGB.Operational.not_persists_of_closed_moments_and_empty_boundary`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L104)：Direct downstream use of the new Noether bridge: if closed moment drifts vanish and there are no boundary participants, actual intake is impossible.
- [`CGB.Medium.kernel_from_frozen_state`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L126)：The full initial state need only be absorbing for all possible stimuli.
- [`CGB.Operational.recovery_zero_from_frozen_state`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L149)
- [`CGB.Operational.alive_zero_from_frozen_state`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L172)
- [`CGB.Operational.not_persists_of_frozen_initial`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L181)：Even holding material at birth cannot rescue an absorbing initial body.
- [`CGB.RecoveryControls.passive_body_axioms`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L199)：The existing passive-absorption control also satisfies all six physical clauses; its failure of the life target is not a violation of those clauses.
- [`CGB.RecoveryControls.passive_body_can_take_in`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L232)：Passive absorption really occurs, so this is stronger than a frozen body with no intake transition anywhere.
- [`CGB.RecoveryControls.passive_full_frozen`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L238)：Once full, the passive body's absorption stops in every environment.
- [`CGB.RecoveryControls.passive_full_not_alive`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L245)：The current actual-world alive test excludes the full passive state for both normal and cut futures, at every horizon and in every world.
- [`CGB.RecoveryControls.filledPassive_axioms`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L256)
- [`CGB.RecoveryControls.filledPassive_clock`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L260)
- [`CGB.RecoveryControls.filledPassive_not_persists`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L262)
- [`CGB.RecoveryControls.passive_intake_not_persistence`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L269)：Explicit definition control: physical admissibility, initial material, and an intake event somewhere do not imply actual persistence in this world.
- [`CGB.NoetherControls.seed_closed_symmetry`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L284)：Existing admissible seed: its closed-event content symmetry is certified.
- [`CGB.NoetherControls.toggle_fails_closed_moments`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L289)：Existing nonconserving closed toggle is rejected by the new moment bridge.
- [`CGB.NoetherControls.meanOnly_first_drift`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L303)
- [`CGB.NoetherControls.meanOnly_second_drift`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L307)
- [`CGB.NoetherControls.meanOnly_not_conserved`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L310)

## MaterialExchange（8）
_Versioned gross material exchange observer The old net-growth observer remains intact._

- [`CGB.Medium.ledger_p2a`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L26)
- [`CGB.Medium.net_increase_implies_incoming`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L34)：A positive net increase necessarily includes incoming material.
- [`CGB.Medium.equal_content_iff_balanced_exchange`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L40)
- [`CGB.Exchange.recovery_bounds`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L66)
- [`CGB.Exchange.recovery_zero_without_incoming`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L71)
- [`CGB.Exchange.survival_bounds`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L90)
- [`CGB.Exchange.not_persists_without_incoming`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L112)
- [`CGB.Description.ExchangeBook.one_percent_budget`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L162)

## MaterialSource（6）
_Source-resolved material transport Fixed token identities locate material in the body, accessible source, return sink or exterior._

- [`CGB.MaterialSource.balance`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L43)：One transport law yields both body balance and the accessible-source budget.
- [`CGB.MaterialSource.moved_le_stock`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L61)
- [`CGB.MaterialSource.no_intake_from_empty_source`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L69)：An empty accessible source cannot feed the body in this tick.
- [`CGB.MaterialSource.no_transport_without_relocation`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L76)：With identities fixed, an unchanged ownership state cannot be labelled as positive gross transport.
- [`CGB.MaterialSource.source_budget`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L86)：Exact finite-prefix stock law, applicable to each supported sample path.
- [`CGB.MaterialSource.intake_bound_without_refill`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L103)：A finite unreplenished, unrecycled source bounds cumulative intake; this is a material-throughput conclusion, not a death/free-energy theorem.

## MaterialSourceControls（3）
_Finite supply / exchange / exhaustion control A mathematical carrier-interpretation witness, not a bacterial/sponge model or a capability experiment._

- [`CGB.MaterialSourceControls.realized`](../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L62)：Explicit source realization: no changed body dynamics, no extra physical ticks, and no nonempty-body false positive standing in for actual replacement.
- [`CGB.MaterialSourceControls.supply_exchange_exhaustion`](../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L100)：One arrival, then one equal input/output replacement, then no accessible source.
- [`CGB.MaterialSourceControls.no_future_intake_after_exhaustion`](../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L118)：Reuse the existing refinement/first-hit theorem for the WHOLE future.

## Medium（5）
_The CGB medium and the six axiom clauses The full language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1–2._

- [`CGB.Medium.total_of_reachable`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L161)：P2a confines the closed dynamics to a content shell.
- [`CGB.Medium.finite_reachable`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L172)：**T0a** conservation implies finiteness: under P2a and the first part of P1a, the closed dynamics reaches only finitely many states.
- [`CGB.Medium.finite_states`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L183)：With the capacity bound the whole state space is finite: the premise "finite state space" of the life definition.
- [`CGB.Medium.energy_of_p2b`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L192)：**T-E** for media: P2b gives an energy function of the equilibrium dynamics.
- [`CGB.Medium.p2b_of_micro`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L199)：Microscopic reversibility: if in the equilibrium environment every effective event is undone by its reverse event at the same rate, P2b holds (with uniform weights).

## Models（57）
_Consistency models, independence witnesses and the bottleneck theorem **Consistency.** * `twinFlip`: closed._

- [`CGB.Models.xhopAct_of_ok`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L54)
- [`CGB.Models.xhopAct_of_not`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L57)
- [`CGB.Models.xhopAct_rev_ok`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L60)
- [`CGB.Models.xhopAct_back`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L65)
- [`CGB.Models.xhopAct_sum`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L71)
- [`CGB.Models.xhopAct_outside`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L78)
- [`CGB.Models.xhopAct_congr`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L83)
- [`CGB.Models.symm_swap`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L88)
- [`CGB.Models.xhopAct_perm`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L97)
- [`CGB.Models.xhopAct_ind`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L108)：The content of `a` alone, moved by a hop, can only have gone along `a → b`.
- [`CGB.Models.ident_swap`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L121)
- [`CGB.Models.eq_homogeneous`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L130)：Homogeneous contact by equality: any swap is a symmetry.
- [`CGB.Models.bool_p1a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L135)：Occupancy content: a site is empty (the vacuum) or holds one unit.
- [`CGB.Models.xhop_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L165)
- [`CGB.Models.xhop_p1b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L170)
- [`CGB.Models.xhop_p2a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L180)
- [`CGB.Models.xhop_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L183)
- [`CGB.Models.xhop_p2c`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L193)
- [`CGB.Models.xhop_senses`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L200)
- [`CGB.Models.xhop_pages`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L206)：**Bottleneck theorem.** With all units in contact, P2c forces each conductance to depend only on the identity pages of its two ends.
- [`CGB.Models.solo_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L271)：Every permutation of units is a symmetry of `solo`.
- [`CGB.Models.solo_p1b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L274)
- [`CGB.Models.solo_p2a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L287)：Local content conservation of each event gives P2a.
- [`CGB.Models.solo_p2c`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L295)
- [`CGB.Models.solo_p3`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L304)
- [`CGB.Models.solo_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L308)：Local microscopic reversibility of each unit gives P2b.
- [`CGB.Models.flip12_flip12`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L328)
- [`CGB.Models.flipCost_flip12`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L330)
- [`CGB.Models.flipCost_p1a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L332)
- [`CGB.Models.twinFlip_axioms`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L343)
- [`CGB.Models.twinFlip_swap_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L347)
- [`CGB.Models.rock_axioms`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L355)
- [`CGB.Models.rock_frozen`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L360)：The rock is frozen: no transition ever has positive rate, in any environment.
- [`CGB.Models.ringAdj_refl`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L371)
- [`CGB.Models.ringAdj_symm`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L373)
- [`CGB.Models.ringAdj_shift`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L375)
- [`CGB.Models.seed_p1b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L409)
- [`CGB.Models.seed_p2a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L452)
- [`CGB.Models.seed_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L461)
- [`CGB.Models.seed_aut_fix`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L480)：Every symmetry of the seed fixes the source and the sink.
- [`CGB.Models.seed_p2c`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L490)
- [`CGB.Models.seed_p3`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L517)
- [`CGB.Models.seed_axioms`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L529)
- [`CGB.Models.seed_swap_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L533)：The reflection swapping the interior sites is a genuine symmetry.
- [`CGB.Models.seed_driven`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L541)：The seed is driven: under food its open dynamics has no detailed balance.
- [`CGB.Models.seed_grows`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L567)：**Growth.** From one unit at the source, under food, the seed reaches a state with three units: intake and spreading create units at empty sites.
- [`CGB.Models.twinTape_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L593)
- [`CGB.Models.twinBag_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L629)
- [`CGB.Models.twinBlank_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L657)
- [`CGB.Models.twinToggle_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L675)
- [`CGB.Models.twinCycle_not_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L699)
- [`CGB.Models.twinCycle_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L721)
- [`CGB.Models.perm_fin2_succ`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L746)
- [`CGB.Models.farFlip_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L754)
- [`CGB.Models.triLookup_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L783)
- [`CGB.Models.pairPreset_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L798)
- [`CGB.Models.innerClock_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L821)

## NestedLife（2）
_Life of a nested organism Once the organism is part of one medium with its world (`Nesting`), it has no outside stimulus: what it senses is the state of the world next to it, and what it takes in comes out of the world._

- [`CGB.Part.irrecoverable_absorbing`](../../lean/CGBAxioms/CGBAxioms/NestedLife.lean#L52)
- [`CGB.Part.no_intake_of_closed`](../../lean/CGBAxioms/CGBAxioms/NestedLife.lean#L58)：An organism that never exchanges content cannot take anything in, so nothing is recoverable for it: metabolism is necessary for nested life.

## Nesting（8）
_Nesting: an organism is a part of one medium, its boundary induced by contact There is one medium._

- [`CGB.Medium.boundary_eq_outer_compl`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L44)：`S`'s own boundary: the units of `S` touching the outside.
- [`CGB.Medium.participant_near`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L49)：A participant of an event that also has a participant in `S` is in `S` or touches it.
- [`CGB.Medium.blanket_of_contact`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L57)：**T-N1 (the blanket from contact)**: two states that agree on `S` and on the units touching it give every event with a participant in `S` the same rate and the same effect on `S`.
- [`CGB.Medium.untouched_unchanged`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L71)：Events with no participant in `S` leave `S` unchanged.
- [`CGB.Medium.organism_blanket`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L76)：The organism's half: what happens to `O` reads only `O` and the units touching it.
- [`CGB.Medium.world_blanket`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L83)：The world's half (formerly the assumption `ReadsBoundary`): what happens to the rest reads only the rest and the organism's own boundary units.
- [`CGB.Medium.exchange_conserves`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L96)：**T-N2 (exchange conserves)**: a closed event with positive rate changes the content of `O` by exactly minus the change of the rest.
- [`CGB.JumpProcess.freezeOutside_balance`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L127)：**T-N3 (the organism's own moves are balanced with the world frozen)**: if the joint calm dynamics has detailed-balance weight `π`, the dynamics with the outside of `O` frozen has the same detailed balance.

## NoetherBridge（8）
_Noether correspondence for the actual medium event language A closed-event restriction is an analysis object, not a change to the running medium._

- [`CGB.JumpProcess.observableSymmetry_iff_conserved`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L24)
- [`CGB.Medium.jump_gen_events`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L52)：Sum event contributions before aggregating transitions with the same output.
- [`CGB.Medium.closedJump_conserved_iff`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L80)
- [`CGB.Medium.p2a_iff_closed_conserved`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L120)
- [`CGB.Medium.p2a_iff_closed_moments`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L133)
- [`CGB.Medium.p2a_iff_closed_expected_drifts`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L143)：Every normalized initial distribution has zero first and second moment DRIFT.
- [`CGB.Medium.p2a_iff_closed_observable_symmetry`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L155)
- [`CGB.Medium.content_drift_boundary`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L164)：In the original OPEN generator, all drift of any function of total content comes from boundary-touching events.

## OperationalLife（6）
_Versioned operational target: finite descriptions and actual-world intake The old LifeConjecture remains a historical statement, not a proved equivalent._

- [`CGB.Viability.alive_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L23)
- [`CGB.Viability.survival_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L32)
- [`CGB.Operational.recovery_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L68)
- [`CGB.Operational.natural_recovery`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L74)：For the natural uncut world, the general protocol reduces to actualRecover.
- [`CGB.Operational.survival_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L89)
- [`CGB.Operational.persistence_parameter_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L113)：Parameter bounds follow from an actual persistence witness, not extra axioms.

## PortContractInstances（3）
_The current 24-state maintenance witness is admitted through the coarse contracts._

- [`CGB.PortContractInstances.split_factorizes`](../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L41)
- [`CGB.PortContractInstances.reserve_contract`](../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L46)
- [`CGB.PortContractInstances.admission_contract`](../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L52)

## PortContracts（6）
_Coarse contracts for separated sensory and resource ports Complex bodies need not expose every carrier identity._

- [`CGB.PortContracts.protocol_sensor`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L25)
- [`CGB.PortContracts.protocol_resource`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L30)
- [`CGB.PortContracts.admission_port_factorizes`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L47)
- [`CGB.PortContracts.admission_reserve_contract`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L53)
- [`CGB.PortContracts.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L69)
- [`CGB.PortContracts.elapsed_preserved`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L81)

## PortFlowHomology（3）
_Finite flow chain for the separated port witness Each current state/input pair is a directed transition edge._

- [`CGB.PortFlowHomology.incidence_augmented_zero`](../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L30)
- [`CGB.PortFlowHomology.weighted_incidence_augmented_zero`](../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L43)
- [`CGB.PortFlowHomology.weighted_flow_boundary_zero`](../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L59)

## PortSeparation（6）
_Independent sensory and resource ports on the same finite body Absent sensing with resources supplied is a noninformative maintenance inlet._

- [`CGB.PortSeparation.local_budgets`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L54)
- [`CGB.PortSeparation.starvation_progress`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L65)
- [`CGB.PortSeparation.next_empty`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L74)
- [`CGB.PortSeparation.admission`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L92)
- [`CGB.PortSeparation.legacy_rules`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L129)
- [`CGB.PortSeparation.actual_transition`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L148)

## RecoveryControls（5）
_Recovery semantic controls (not organism capability experiments) Identical body and stimulus menus; only world accessibility differs._

- [`CGB.RecoveryControls.sum_functions`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L41)
- [`CGB.RecoveryControls.body_kernel`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L50)
- [`CGB.RecoveryControls.body_clock`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L60)
- [`CGB.RecoveryControls.oracle_ignores_access`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L72)
- [`CGB.RecoveryControls.actual_recovery_distinguishes_world_moves`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L78)

## RecoveryCriteria（4）
_Local sufficient criteria for actual resource recovery Regions and ranks are proof objects, not new organism state, policy inputs or life axioms._

- [`CGB.RecoveryCriteria.ranked_firstHit`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L28)：A local rank certificate proves recovery in the actual future kernel, without assuming a favourable stimulus sequence or the desired recovery score.
- [`CGB.RecoveryCriteria.firstHit_zero_of_closed`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L54)
- [`CGB.RecoveryCriteria.support_closed`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L74)
- [`CGB.RecoveryCriteria.survival_of_ranked`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L104)：Reusable sufficient criterion for the original Exchange score.

## RecoveryTransport（3）
_Recovery and state-score transport through existing exact refinements State-dependent intake must match on supported transitions, in addition to the joint event/state law and body content._

- [`CGB.RecoveryTransport.mean_next`](../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L15)
- [`CGB.RecoveryTransport.firstHit_eq`](../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L32)：Supported actual-intake correspondence preserves every recovery horizon, including in a distribution-dependent cut supplied as a separate kernel.
- [`CGB.RecoveryTransport.mean_eq`](../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L61)：Exact state-score expectations at every time, without assuming that the score is a probability or replacing physical state by output histories.

## RenewableCandidateAudit（9）
_A concrete obstruction to the missing sensory-interruption obligation Keep the original body, world, clock and interruption protocol._

- [`CGB.RenewableCandidateAudit.kernel_at_next`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L30)
- [`CGB.RenewableCandidateAudit.policy_values`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L37)
- [`CGB.RenewableCandidateAudit.edge`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L40)
- [`CGB.RenewableCandidateAudit.step_lower`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L51)
- [`CGB.RenewableCandidateAudit.path_lower`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L78)
- [`CGB.RenewableCandidateAudit.interrupted_trap_mass`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L90)
- [`CGB.RenewableCandidateAudit.trap_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L103)
- [`CGB.RenewableCandidateAudit.interruption_failure`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L127)
- [`CGB.RenewableCandidateAudit.no_positive_interruption_tolerance`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L165)

## RenewableEncoding（13）
_Concrete rule-book compilation and honest candidate boundaries The original one-site rules are compiled into the EXISTING tabular language._

- [`CGB.RenewableEncoding.code_roundtrip`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L56)
- [`CGB.RenewableEncoding.book_layout`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L58)
- [`CGB.RenewableEncoding.row_lookup`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L62)
- [`CGB.RenewableEncoding.transfer_lookup`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L65)
- [`CGB.RenewableEncoding.compiled_rows`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L72)
- [`CGB.RenewableEncoding.compiled_rates`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L82)
- [`CGB.RenewableEncoding.data_balance`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L92)
- [`CGB.RenewableEncoding.accountable`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L99)
- [`CGB.RenewableEncoding.description_positive`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L137)：Necessary bit count for EVERY book under the registered convention.
- [`CGB.RenewableEncoding.small_reference_obstruction`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L141)
- [`CGB.RenewableEncoding.no_one_percent`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L147)
- [`CGB.RenewableEncoding.world_richness_limit`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L151)
- [`CGB.RenewableEncoding.no_encoded_candidate`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L160)：A full encoded subtarget cannot be obtained by attaching the already proved persistence/source certificates to this one-site book.

## RenewableLearning（6）
_Renewable paid learning in a continuously changing finite world Resource access can fail after an unannounced regime change._

- [`CGB.RenewableLearning.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L69)
- [`CGB.RenewableLearning.local_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L145)
- [`CGB.RenewableLearning.learning_contract`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L153)：Paid reopening is driven by a contradictory consequence, and the ensuing probe uses exactly the previously verified candidate-screening update.
- [`CGB.RenewableLearning.admission`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L177)
- [`CGB.RenewableLearning.sum_env`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L235)
- [`CGB.RenewableLearning.actual_law`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L280)

## RenewableLearningCarrier（10）
_Source and recovery certificates for the unchanged renewable-learning process Fixed material identities realize every supported normal/cut transition._

- [`CGB.RenewableLearningCarrier.body_supported`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L23)
- [`CGB.RenewableLearningCarrier.realized`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L72)
- [`CGB.RenewableLearningCarrier.intake_matches`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L87)
- [`CGB.RenewableLearningCarrier.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L97)
- [`CGB.RenewableLearningCarrier.recovery_by_certificate`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L104)
- [`CGB.RenewableLearningCarrier.score_preserved`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L119)
- [`CGB.RenewableLearningCarrier.maintenance_preserved`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L144)
- [`CGB.RenewableLearningCarrier.stock_and_recharge`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L186)
- [`CGB.RenewableLearningCarrier.initial_energy`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L197)：The extra source stock is external to the old two-unit body at birth.
- [`CGB.RenewableLearningCarrier.source_energy_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L207)

## RenewableLearningClosure（19）
_Close the ORIGINAL distribution-dependent marginal-ghost comparison._

- [`CGB.RenewableLearningClosure.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L16)
- [`CGB.RenewableLearningClosure.sum_cells`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L22)
- [`CGB.RenewableLearningClosure.sum_configs`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L29)
- [`CGB.RenewableLearningClosure.sum_env`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L37)
- [`CGB.RenewableLearningClosure.kernel_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L46)
- [`CGB.RenewableLearningClosure.early_move`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L70)：Until tick four the body marginal is a point mass, so the ORIGINAL cut transition is derived from that marginal, not assigned a convenient law.
- [`CGB.RenewableLearningClosure.cut_early`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L83)
- [`CGB.RenewableLearningClosure.move_five`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L100)
- [`CGB.RenewableLearningClosure.trapped_next`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L113)
- [`CGB.RenewableLearningClosure.trapped_six`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L124)
- [`CGB.RenewableLearningClosure.trap_moves`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L130)
- [`CGB.RenewableLearningClosure.trap_world_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L134)
- [`CGB.RenewableLearningClosure.trapped_monotone`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L148)
- [`CGB.RenewableLearningClosure.trap_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L163)
- [`CGB.RenewableLearningClosure.cut_alive`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L180)
- [`CGB.RenewableLearningClosure.cut_mass_bound`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L193)
- [`CGB.RenewableLearningClosure.cut_survival`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L216)：The original cut loses at least a quarter of viable mass by tick six; absorption extends the bound to ALL later times, including future cut marginals.
- [`CGB.RenewableLearningClosure.maintenance`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L225)
- [`CGB.RenewableLearningClosure.cut_seed_mass`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L233)：Exact seed mass for the absorbing failure region under the original cut.

## RenewableLearningControls（15）
_Finite-path probabilities are products of the actual registered step law._

- [`CGB.RenewableLearningControls.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L13)
- [`CGB.RenewableLearningControls.cycle_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L35)
- [`CGB.RenewableLearningControls.cycle_payment`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L42)
- [`CGB.RenewableLearningControls.path_after`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L54)
- [`CGB.RenewableLearningControls.path_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L56)
- [`CGB.RenewableLearningControls.path_active`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L80)
- [`CGB.RenewableLearningControls.write_blocks`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L94)
- [`CGB.RenewableLearningControls.arbitrary_relearning`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L126)：Every requested number of paid A-to-B-to-A cycles has a positive-probability finite prefix of this one process, with explicit equal credit replenishment.
- [`CGB.RenewableLearningControls.law_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L135)
- [`CGB.RenewableLearningControls.path_law_bound`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L148)
- [`CGB.RenewableLearningControls.failure_weight`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L171)
- [`CGB.RenewableLearningControls.trap_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L178)
- [`CGB.RenewableLearningControls.trap_alive`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L191)
- [`CGB.RenewableLearningControls.survival_mass_bound`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L199)
- [`CGB.RenewableLearningControls.paired_failure`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L231)：Blocking either paid reopening or storage renewal creates a half-probability loss of viability by tick six, under the same world, initial state and food rule.

## RenewableLearningEnergy（5）
_The total fuel account contains both buffer and write credit._

- [`CGB.RenewableLearningEnergy.supported`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L16)
- [`CGB.RenewableLearningEnergy.allocation`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L81)：Allocating credit uses part of the specified food input.
- [`CGB.RenewableLearningEnergy.telescope`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L86)
- [`CGB.RenewableLearningEnergy.fuel_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L107)
- [`CGB.RenewableLearningEnergy.credit_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L114)

## RenewableLearningLife（15）
_All-time viability in the same changing body/world process._

- [`CGB.RenewableLearningLife.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L19)
- [`CGB.RenewableLearningLife.nonvacuum`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L26)
- [`CGB.RenewableLearningLife.active`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L29)
- [`CGB.RenewableLearningLife.next_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L33)
- [`CGB.RenewableLearningLife.supported_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L39)
- [`CGB.RenewableLearningLife.law_invariant`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L61)
- [`CGB.RenewableLearningLife.reachable_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L82)
- [`CGB.RenewableLearningLife.sum_configs`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L91)
- [`CGB.RenewableLearningLife.sum_env`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L95)
- [`CGB.RenewableLearningLife.recovery_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L106)
- [`CGB.RenewableLearningLife.immediate_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L117)
- [`CGB.RenewableLearningLife.recovery_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L123)
- [`CGB.RenewableLearningLife.alive_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L143)
- [`CGB.RenewableLearningLife.natural_survival`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L162)：Every supported continuation remains viable through missed food, paid reopening, probing, identification and actual resource-funded renewal.
- [`CGB.RenewableLearningLife.persistence`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L189)

## RenewableRecoveryCertificate（7）
_Instantiate the reusable local recovery criterion on the existing model._

- [`CGB.RenewableRecoveryCertificate.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L12)
- [`CGB.RenewableRecoveryCertificate.nonvacuum`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L19)
- [`CGB.RenewableRecoveryCertificate.active`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L22)
- [`CGB.RenewableRecoveryCertificate.next_good`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L26)
- [`CGB.RenewableRecoveryCertificate.supported_good`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L32)
- [`CGB.RenewableRecoveryCertificate.unmarked_progress`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L55)
- [`CGB.RenewableRecoveryCertificate.survival_by_criterion`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L94)：The original all-time score now also follows from the shared local criterion; it is not inserted as a premise of that criterion.

## RuleEncoding（11）
_A fixed finite rule language and an honest description budget This is a representation layer, not a seventh physical axiom._

- [`CGB.Description.code_roundtrip`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L52)
- [`CGB.Description.different_books_different_codes`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L55)
- [`CGB.Description.bounded_books`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L60)：At most 2^B distinct descriptions fit in B bits (an intentionally loose bound).
- [`CGB.Description.one_percent_budget`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L81)
- [`CGB.Description.WiringReference.payload_length`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L94)
- [`CGB.Description.Book.findWithin_complete`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L119)
- [`CGB.Description.Book.updateWithin_complete`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L153)
- [`CGB.Description.Book.weight_nonneg`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L161)
- [`CGB.Description.Book.same_code_same_medium`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L189)：Equality of codes entails equality of the whole interpreted instance.
- [`CGB.Description.Book.unchanged_outside`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L195)：Locality of effects is a theorem of the interpreter, not a user-supplied promise.
- [`CGB.Description.Book.admits_some_clock`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L208)：Finite alphabets and this finite rulebook admit some uniform clock.

## ScheduledRecovery（3）
_Time-dependent certificates for a registered recovery protocol Regions and ranks may refer to external protocol time._

- [`CGB.ScheduledRecovery.ranked_firstHit`](../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L28)：A local rank certificate proves recovery in the actual future kernel, without assuming a favourable stimulus sequence or the desired recovery score.
- [`CGB.ScheduledRecovery.support_closed`](../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L54)
- [`CGB.ScheduledRecovery.survival_of_ranked`](../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L84)：Reusable sufficient criterion for the original Exchange score.

## SensoryMaintenance（16）
_Sensory loss does not suspend physical maintenance `some calm` remains the equilibrium stimulus._

- [`CGB.SensoryMaintenance.local_budgets`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L60)
- [`CGB.SensoryMaintenance.transfer_bounds`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L68)
- [`CGB.SensoryMaintenance.active_loss`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L75)
- [`CGB.SensoryMaintenance.missing_no_intake`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L84)
- [`CGB.SensoryMaintenance.next_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L87)
- [`CGB.SensoryMaintenance.admission`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L102)
- [`CGB.SensoryMaintenance.equilibrium_distinct_from_loss`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L154)
- [`CGB.SensoryMaintenance.actual_transition`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L161)
- [`CGB.SensoryMaintenance.actual_law`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L183)
- [`CGB.SensoryMaintenance.kernel_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L194)
- [`CGB.SensoryMaintenance.burn_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L210)
- [`CGB.SensoryMaintenance.burn_three`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L218)
- [`CGB.SensoryMaintenance.burn_all_later`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L222)
- [`CGB.SensoryMaintenance.funded_two_tick_bridge`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L228)
- [`CGB.SensoryMaintenance.path_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L234)：Exact actual-channel fuel balance for every registered input path.
- [`CGB.SensoryMaintenance.path_credit_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L246)

## SensoryMaintenanceBoundaries（3）
_Exact local blackout boundary and recharge condition The two-loss boundary is a state-level result: a supported nonvacuum cell has zero reserve but remains nonempty after two absent-sensing ticks, and reaches the absorbi_

- [`CGB.SensoryMaintenanceBoundaries.two_tick_functional_boundary`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L16)
- [`CGB.SensoryMaintenanceBoundaries.sensed_recharge`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L29)
- [`CGB.SensoryMaintenanceBoundaries.two_tick_recharge_condition`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L41)

## SensoryMaintenanceCarrier（12）
_Two existing finite material buffers realize structural fuel and the new reserve._

- [`CGB.SensoryMaintenanceCarrier.component_rules`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L28)
- [`CGB.SensoryMaintenanceCarrier.component_sums`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L36)
- [`CGB.SensoryMaintenanceCarrier.coarse_positive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L78)
- [`CGB.SensoryMaintenanceCarrier.count_sum`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L117)
- [`CGB.SensoryMaintenanceCarrier.moved_sum`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L125)
- [`CGB.SensoryMaintenanceCarrier.realized`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L136)
- [`CGB.SensoryMaintenanceCarrier.source_stock`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L167)
- [`CGB.SensoryMaintenanceCarrier.recharge_paid`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L176)
- [`CGB.SensoryMaintenanceCarrier.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L195)
- [`CGB.SensoryMaintenanceCarrier.score_preserved`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L212)
- [`CGB.SensoryMaintenanceCarrier.initial_energy`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L260)
- [`CGB.SensoryMaintenanceCarrier.source_energy_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L269)

## SensoryMaintenanceClosure（11）
_The original marginal-ghost cut includes the new physical reserve._

- [`CGB.SensoryMaintenanceClosure.sum_old`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L10)
- [`CGB.SensoryMaintenanceClosure.sum_new`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L13)
- [`CGB.SensoryMaintenanceClosure.awake_projection`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L19)
- [`CGB.SensoryMaintenanceClosure.old_kernel`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L24)
- [`CGB.SensoryMaintenanceClosure.kernel_marginal`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L39)
- [`CGB.SensoryMaintenanceClosure.cut_move`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L50)
- [`CGB.SensoryMaintenanceClosure.cut_marginal`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L67)：Recompute the actual cut with the augmented body's own marginal; summing its physical reserve gives exactly the old cut's memory/world law.
- [`CGB.SensoryMaintenanceClosure.vacuum_target`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L100)
- [`CGB.SensoryMaintenanceClosure.vacuum_reserve_zero`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L106)
- [`CGB.SensoryMaintenanceClosure.empty_mass`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L127)
- [`CGB.SensoryMaintenanceClosure.cut_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L138)

## SensoryMaintenanceLimits（7）
_Finite blackout loss and actual-event energy accounting._

- [`CGB.SensoryMaintenanceLimits.support_closed`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L10)
- [`CGB.SensoryMaintenanceLimits.loss_closed`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L36)
- [`CGB.SensoryMaintenanceLimits.blackout_body_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L63)：After three consecutive absent-sensing ticks, every supported body is empty.
- [`CGB.SensoryMaintenanceLimits.blackout_survival_zero`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L81)
- [`CGB.SensoryMaintenanceLimits.long_loss_not_tolerated`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L98)
- [`CGB.SensoryMaintenanceLimits.no_input_active_duration`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L108)：A declared positive loss at every active tick yields a finite duration bound from the actual first law.
- [`CGB.SensoryMaintenanceLimits.fuel_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L155)

## SensoryMaintenanceProtocolRepairRefutation（4）

- [`CGB.SensoryMaintenanceProtocolRepairRefutation.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L7)
- [`CGB.SensoryMaintenanceProtocolRepairRefutation.path_step`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L29)
- [`CGB.SensoryMaintenanceProtocolRepairRefutation.protected_protocol_empty_mass_positive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L44)
- [`CGB.SensoryMaintenanceProtocolRepairRefutation.protected_protocol_survival_lt_one`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L63)

## SensoryMaintenanceRecovery（10）

- [`CGB.SensoryMaintenanceRecovery.intake_same`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L14)
- [`CGB.SensoryMaintenanceRecovery.old_event`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L22)
- [`CGB.SensoryMaintenanceRecovery.supported`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L34)
- [`CGB.SensoryMaintenanceRecovery.sensed_not_calm`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L87)
- [`CGB.SensoryMaintenanceRecovery.decode_protocol`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L90)
- [`CGB.SensoryMaintenanceRecovery.positive_region`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L96)
- [`CGB.SensoryMaintenanceRecovery.reserve_sensed`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L104)
- [`CGB.SensoryMaintenanceRecovery.one_tick_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L154)
- [`CGB.SensoryMaintenanceRecovery.natural_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L187)
- [`CGB.SensoryMaintenanceRecovery.one_tick_tolerance`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L205)

## SensoryMaintenanceRepairAudit（3）
_Audit of state-only repairs after the two-tick refutation The existing material ports make the `aSpent`/`miss` failure structurally forced._

- [`CGB.SensoryMaintenanceRepairAudit.cost_zero_iff_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L16)
- [`CGB.SensoryMaintenanceRepairAudit.spent_miss_budget_forces_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L23)
- [`CGB.SensoryMaintenanceRepairAudit.no_state_only_two_tick_repair`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L42)

## SensoryMaintenanceRepairCandidates（25）
_Minimal repair candidates after the two-tick refutation The registered model is unchanged._

- [`CGB.SensoryMaintenanceRepairCandidates.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L16)
- [`CGB.SensoryMaintenanceRepairCandidates.protected_second_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L26)
- [`CGB.SensoryMaintenanceRepairCandidates.protected_first_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L30)
- [`CGB.SensoryMaintenanceRepairCandidates.protocol_repair_preserves_one_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L34)
- [`CGB.SensoryMaintenanceRepairCandidates.probe_recharges_after_one_loss`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L38)
- [`CGB.SensoryMaintenanceRepairCandidates.protected_probe_path_outgoing`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L53)
- [`CGB.SensoryMaintenanceRepairCandidates.protected_probe_path_incoming`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L66)
- [`CGB.SensoryMaintenanceRepairCandidates.protected_branch_excludes_local_blackout`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L78)
- [`CGB.SensoryMaintenanceRepairCandidates.initial_guarded_two_tick_start`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L102)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_third_tick_recharges`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L116)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_repair_preserves_one_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L149)
- [`CGB.SensoryMaintenanceRepairCandidates.firstHit_three_tick_layered`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L153)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_firstHit`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L216)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_alive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L313)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_two_tick_local_tolerance`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L333)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_survival_at`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L344)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_support_zero`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L396)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_support_path_step`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L421)
- [`CGB.SensoryMaintenanceRepairCandidates.not_guarded_support_two`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L436)
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_initial_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L460)
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_excludes_spent`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L466)
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_excludes_verified_dangerous_branch`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L472)
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_two_loss_preserves_cell`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L479)
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_keeps_existing_budget_contract`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L491)
- [`CGB.SensoryMaintenanceRepairCandidates.region_repair_preserves_one_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L498)

## SensoryMaintenanceTwoTickGlobal（7）

- [`CGB.SensoryMaintenanceTwoTickGlobal.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L9)
- [`CGB.SensoryMaintenanceTwoTickGlobal.stateLaw_nonneg`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L14)
- [`CGB.SensoryMaintenanceTwoTickGlobal.stateLaw_pos_of_path`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L28)
- [`CGB.SensoryMaintenanceTwoTickGlobal.path_step`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L79)
- [`CGB.SensoryMaintenanceTwoTickGlobal.empty_mass_positive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L92)
- [`CGB.SensoryMaintenanceTwoTickGlobal.two_tick_survival_lt_one`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L108)
- [`CGB.SensoryMaintenanceTwoTickGlobal.not_two_tick_tolerance`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L169)

## SensoryMaintenanceTwoTickObstruction（3）

- [`CGB.SensoryMaintenanceTwoTickObstruction.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L7)
- [`CGB.SensoryMaintenanceTwoTickObstruction.two_tick_world_branch`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L14)：A concrete world branch that remains physically possible through two lost ticks while the projected cell is `aSpent`.
- [`CGB.SensoryMaintenanceTwoTickObstruction.two_tick_local_failure`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L22)：The same branch drains the two-unit reserve and the next sensed miss sends the cell to vacuum.

## SpongeContractInstance（2）
_Coarse admission candidate for a sponge-like continuous body This is a structural interface witness only._

- [`CGB.SpongeContractInstance.port_factorizes`](../../lean/CGBAxioms/CGBAxioms/SpongeContractInstance.lean#L54)
- [`CGB.SpongeContractInstance.reserve_contract`](../../lean/CGBAxioms/CGBAxioms/SpongeContractInstance.lean#L61)

## SubconditionGraph（37）
_Structural Graph of the 14 Atomic Sub-conditions This file establishes the formal dependency and redundancy relations among the 14 atomic sub-conditions (A01–A14) from the 6 compound CGB axioms._

- [`CGB.Medium.p1b_iff_atoms`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L61)
- [`CGB.Medium.onlyBoundarySenses_iff_atoms`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L72)
- [`CGB.Medium.p3_iff_atoms`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L80)
- [`CGB.Medium.a01_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L88)：In any finite local state space, sublevel sets are automatically finite (A01 is redundant).
- [`CGB.Medium.a02_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L94)：In any finite local state space, content is automatically bounded (A02 is redundant).
- [`CGB.Medium.a01_and_a02_redundant_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L104)：Under `[Finite Λ]`, A01 and A02 hold unconditionally for ANY content function.
- [`CGB.Medium.uniqueVacuum_independent_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L110)：A03 (uniqueVacuum) is strictly independent of finiteness: on `Fin 2`, we can have zero vacuums, multiple vacuums, or a unique vacuum.
- [`CGB.Medium.farFlip_not_rateReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L136)：`farFlip` violates rateReadsParticipants (A07).
- [`CGB.Medium.farFlip_satisfies_effectReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L150)：`farFlip` SATISFIES effectReadsParticipants (A08).
- [`CGB.Medium.farFlip_satisfies_writes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L158)：`farFlip` satisfies writesParticipants (A06).
- [`CGB.Medium.farFlip_satisfies_partsContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L164)：`farFlip` satisfies participantsContact (A05).
- [`CGB.Medium.farFlip_satisfies_homoContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L171)：`farFlip` satisfies homogeneousContact (A04).
- [`CGB.Medium.farFlip_isolates_a07`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L176)：Decoupling summary for `farFlip`: it isolates and violates A07 while satisfying A08, A06, A05, A04.
- [`CGB.Medium.farAct_not_effectReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L199)
- [`CGB.Medium.farAct_satisfies_rateReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L214)
- [`CGB.Medium.farAct_satisfies_writes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L218)
- [`CGB.Medium.farAct_satisfies_partsContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L223)
- [`CGB.Medium.farAct_satisfies_homoContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L229)
- [`CGB.Medium.farAct_isolates_a08`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L234)：Decoupling summary for `farAct`: it isolates and violates A08 while satisfying A07, A06, A05, A04.
- [`CGB.Medium.innerClock_not_boundaryRate`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L247)：`innerClock` violates boundaryRate (A12).
- [`CGB.Medium.innerClock_satisfies_boundaryEffect`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L256)：`innerClock` SATISFIES boundaryEffect (A13).
- [`CGB.Medium.innerClock_satisfies_initialInvariant`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L261)：`innerClock` SATISFIES initialInvariant (A14).
- [`CGB.Medium.innerClock_isolates_a12`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L268)：Decoupling summary for `innerClock`: it isolates and violates A12 while satisfying A13 and A14.
- [`CGB.Medium.innerActClock_not_boundaryEffect`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L289)
- [`CGB.Medium.innerActClock_satisfies_boundaryRate`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L299)
- [`CGB.Medium.innerActClock_satisfies_initialInvariant`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L303)
- [`CGB.Medium.innerActClock_isolates_a13`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L310)：Decoupling summary for `innerActClock`: it isolates and violates A13 while satisfying A12 and A14.
- [`CGB.Medium.pairPreset_isolates_a14`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L319)：`pairPreset` isolates A14: satisfies boundaryRate (A12) and boundaryEffect (A13), but violates A14.
- [`CGB.Medium.complete_graph_satisfies_a04_a05`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L339)：Under a complete graph (`near u v ↔ True`), homogeneousContact (A04) and participantsContact (A05) are vacuously satisfied.
- [`CGB.Medium.universal_parts_satisfies_a06_a07_a08`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L348)：Under universal participants (`∀ a, parts a = univ`), writesParticipants (A06), rateReadsParticipants (A07), and effectReadsParticipants (A08) are vacuously satisfied.
- [`CGB.Medium.all_boundary_satisfies_a12_a13`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L361)：When boundary = univ, boundaryRate (A12) and boundaryEffect (A13) are vacuously satisfied.
- [`CGB.Medium.path3_not_homogeneous`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L381)：This specific three-site contact relation is not vertex-transitive.
- [`CGB.Medium.atomized_implies_reducedCore`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L426)
- [`CGB.Medium.reducedCore_implies_atomized_of_regimes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L433)
- [`CGB.Medium.reducedCore_iff_atomized_of_regimes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L447)：Exact recovery under finite local states and rigid structural data (contact homogeneity is no longer part of the axioms since 2026-10-26).
- [`CGB.Medium.path3Rigid_reducedCore`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L467)
- [`CGB.Medium.path3Rigid_atomized_not_homogeneous`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L485)：Since contact homogeneity left P1b (2026-10-26), the rigid three-unit chain, whose contact is not homogeneous (its ends have one contact, its middle two), is a model; before, it was the countermodel showing that finitene

## Symmetry（4）
_P2: symmetry clauses and their first theorems * P2a (conservation) in pathwise form, and **T-N**: the Baez–Fong Noether correspondence (J._

- [`CGB.JumpProcess.gen_sq_sub`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L31)
- [`CGB.JumpProcess.noether`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L40)：**T-N** (Noether for Markov processes, pointwise form): `Q` and `Q²` are both stationary observables iff every event preserves `Q`.
- [`CGB.JumpProcess.energy_of_reversible`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L64)：**T-E** (energy from time-reversal symmetry): a reversible process has an energy `E = -log π` that fixes the ratio of every event to its reverse.
- [`CGB.JumpProcess.ring3_not_reversible`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L83)

## TurnoverControls（15）
_Constant-content turnover controls These are mathematical observation controls, not bacterial/sponge models or capability experiments._

- [`CGB.TurnoverControls.body_axioms`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L22)
- [`CGB.TurnoverControls.body_unchanged`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L27)
- [`CGB.TurnoverControls.body_eventClock`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L33)
- [`CGB.TurnoverControls.event_at_full`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L52)
- [`CGB.TurnoverControls.old_net_recovery_zero`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L63)
- [`CGB.TurnoverControls.balanced_exchange_recovery_one`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L75)：A one-world-tick exchange occurs with certainty although net content never increases.
- [`CGB.TurnoverControls.zero_ledger_recovery_zero`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L84)
- [`CGB.TurnoverControls.balanced_exchange_alive`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L90)
- [`CGB.TurnoverControls.natural_law`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L109)：The state process stays at the same full-body/world state forever, even though the registered replacement channel fires each tick.
- [`CGB.TurnoverControls.natural_survival_one`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L122)
- [`CGB.TurnoverControls.balanced_exchange_persists`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L130)：A material-throughput subtarget witness, not the complete life or cognitive candidate: the separate SelfMaintains and cognitive obligations are not assumed.
- [`CGB.TurnoverControls.old_net_and_balanced_exchange_disagree`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L138)：Same original state process, two different first-intake observations.
- [`CGB.TurnoverControls.ledger_not_identified_by_state_dynamics`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L148)：Two algebraically accountable ledgers give different gross observations on IDENTICAL body/world dynamics.
- [`CGB.TurnoverControls.fastBody_old_clock`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L160)
- [`CGB.TurnoverControls.fastBody_no_event_clock`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L165)

## WorldRecovery（15）
_First-hit probabilities in the actual body--world chain No best-stimulus oracle is used._

- [`CGB.FirstHit.bounds`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L31)
- [`CGB.FirstHit.monotone_horizon`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L44)
- [`CGB.FirstHit.flagged_true`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L62)
- [`CGB.FirstHit.flagged_false`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L68)
- [`CGB.FirstHit.zero_on_closed`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L82)：A closed region with no marked positive-probability transition has zero success probability at every finite horizon, even if success exists elsewhere.
- [`CGB.FirstHit.impossible_absorbing`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L101)
- [`CGB.FirstHit.zero_of_impossible`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L106)
- [`CGB.FirstHit.during_bounds`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L129)
- [`CGB.FirstHit.during_constant`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L143)
- [`CGB.Medium.jointStep_sum`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L162)
- [`CGB.Medium.law_uses_jointStep`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L168)
- [`CGB.Medium.lawCut_uses_scheduledCutKernel`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L202)
- [`CGB.Medium.actualRecover_bounds`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L216)
- [`CGB.Medium.actualRecover_mono`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L220)
- [`CGB.Medium.actualRecover_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L226)：No possible material intake implies zero actual recovery, without assuming that a favourable stimulus can be summoned from some other world state.

