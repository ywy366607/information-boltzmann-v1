# CGB 已证定理索引（自动生成，勿手改）

由 `scripts/build_proven_index.py` 从 Lean 源码生成，并逐条核对当前审计清单。新工作开始前先检索已有机制；点击完整定理名查看源码、前提及结论。
渐进检索入口：[主题导航与逐条完整陈述](CGB_THEOREMS/README.md)；先查主题，再按模块阅读。无源码解释的条目明确标注，避免把机器猜测当作定理含义。
分类标注来自 `CGB_THEOREM_CLASSES.json`；draft 为未审核种子，不代表定稿。

共 **1807** 条已审计源码定理/引理，来自 **136** 个生产模块。编译声明共 **7347** 项，含定义及自动生成项，不作为机制数量。
生成时全部生产源码哈希与通过信任/覆盖检查的库存一致；运行 `--check` 检测过时索引。标准入口 `lean/CGBAxioms/scripts/audit.py` 在构建、审计成功后自动刷新本页；其 `--check-index` 模式只核对索引、不改写索引。

**未入 Lean 的手工推导与数值见证**另列在 `CGB_HAND_DERIVATIONS.md`，检索时两处都要看。
`reviews/` 中的探索证明见渐进检索入口的独立目录，未混入此生产清单。证明通过不替代其局部前提与实现对应检查。

既有中文机制解读见 [Claude 生成链](CGB_GENERATION_LAYER1_20261025.md)；完整接线记录见 [统一参考](CGB_UNIFIED_REFERENCE_20261026.md)。这些文档保留各自证据范围。

## 生成链导航（自动提取源码标注）

以下从全部 `Generation*.lean` 定理注释的开头加粗标注自动提取，不维护手选机制名单。标注供检索，成立范围以链接中的定理签名为准；未带标注的定理仍全部收入后面的完整清单。

| 模块 | 源码标注与位置 |
|---|---|
| Generation | [G3](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L46)；[G13 (prediction is the boundary's Gibbs conditional)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L74)；[G2](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L100)；[G1](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L118)；[G1E](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L171)；[G4](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L197)；[P2b from weighted micro-reversibility](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L224)；[G6](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L255)；[G6DB](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L269)；[G7](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L282)；[G9 (modulation needs the boundary)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L298)；[G10a](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L317)；[G10b (necessity of modulation, minimal form)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L330)；[G10c (heterogeneous compartments need local gains)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L361)；[G11a](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L395)；[G11b (interoceptive memory)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L414)；[G12 (a miss counter is necessary for optimal foraging)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L480)；[G14a (two-phase drift)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L527)；[G14b](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L542)；[G14c (the phases must balance)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L579)；[R2 (leakage loses on every outcome)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L600)；[R1 (conservation makes the readout one-hot)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L608)；[G15a (reward-modulated two-phase drift is the policy gradient)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L642)；[G15b](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L658)；[G15c (outcome-blind modulation learns nothing)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L698)；[G15d (reward phases must balance)](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L710)；[G8a](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L751)；[G8b](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L761) |
| GenerationActiveInference | [Active necessity.](../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L52)；[Active closure.](../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L92) |
| GenerationActiveNavigation | [A common component cancels in every residual](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L85)；[Positive content stays positive](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L98)；[The read forgets the magnitude of the write](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L121)；[Reactivation is exact for a held write](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L137)；[Superposition](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L144)；[The only bias is the writes not retained](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L154)；[Progress from one source is used up.](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L165)；[Removing one entry from a linear mixture](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L177) |
| GenerationAmplifier | [G25a (H stays off below threshold)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L68)；[G25b (switch width)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L78)；[G25c (equilibrium Hill coefficient is one)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L90)；[G25d (cooperativity from multiple quanta)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L99)；[G25e (gain–range bound)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L140)；[G25f (a fixed threshold fails on a wide world)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L163)；[G25f (a threshold centred on the prediction works everywhere)](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L178) |
| GenerationAxiomAudit | [Homogeneity forces a regular contact graph](../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L26) |
| GenerationColumnMemory | [Silence has one rest](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L53)；[Error descent](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L76)；[Decay by use keeps rare topics](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L263)；[No latch below unit gain](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L278)；[A latch above unit gain](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L293) |
| GenerationContextAssociation | [Order blindness.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L93)；[Necessity.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L127)；[Conditional statistic.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L209)；[No interference.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L239)；[Order separation.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L252)；[Address must refine what the future separates.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L271)；[Fading context.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L281)；[Repeatable address.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L298)；[A held coordinate never fades.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L314)；[Separation of roles.](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L332) |
| GenerationContextBody | [O01.](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L277)；[O22.](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L285)；[O25.](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L314)；[O07.](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L357) |
| GenerationCountedPopulation | [Lock-in](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L118)；[Recovery after any history](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L209)；[Price equation](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L224)；[The same law at the next scale](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L262)；[A compartment keeps its total](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L281)；[A compartment reads only its own bets](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L297)；[Delayed credit without a backward pass](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L379)；[The tag is the exact credit](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L428)；[One step of forgetting](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L479)；[The credit horizon](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L541) |
| GenerationCRN | [M2](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L263) |
| GenerationDivisionTree | [Nested groups](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L94) |
| GenerationExport | [G16a (export is forced)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L25)；[G16b (write-only bodies starve)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L42)；[G16c](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L50)；[G17a (unselective export forgets)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L67)；[G17b (retention needs selectivity)](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L90) |
| GenerationGradientFilter | [W2 (drift forbids a vanishing gain)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L108)；[W3 (an occurrence clock)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L163)；[W4 (a relation basis)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L177)；[W5 is implied by W3](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L187)；[W6 (nonconvex data)](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L247) |
| GenerationGrowth | [G18a](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L25)；[G18a (uniqueness)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L32)；[G18b (reciprocity)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L41)；[G18c](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L60)；[G19a (one pool grows a rich club)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L77)；[G19b (memoryless ends see only symbol frequencies)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L92)；[G19c (sequence structure needs memory at the ends)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L113)；[G20a (two-state memory is geometric)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L137)；[G20b (symmetric exchange)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L152)；[G21 (no differentiation without fluctuation)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L162)；[G22a (use depletes)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L183)；[G22b (equilibrium inhibition retreats from co-activity)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L189)；[G22c (no rich club needs local potentials)](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L196) |
| GenerationHoldCorrect | [Holding keeps every disturbance](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L74)；[A held content drifts without bound](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L82)；[Contraction corrects](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L94)；[Correction merges contents](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L115)；[The identity](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L147)；[κ ≤ 1, forgetting](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L172)；[κ > 1, the latch exists](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L184)；[κ > 1, the origin repels](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L194)；[κ > 1, convergence inside the basin](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L202)；[κ > 1, attractor correction under disturbance](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L234)；[The latch's noise tolerance](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L261)；[Decay weakens the latch](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L287)；[Critical slowing at κ = 1](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L294)；[Scale-free timescales near criticality](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L317) |
| GenerationIgnition | [G23 (no ignition in equilibrium with additive energies)](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L61)；[G24 (equilibrium memory bound)](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L100) |
| GenerationInformationEnergy | [N1](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L53)；[N1](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L65)；[N2](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L94)；[N3](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L114)；[N4](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L120)；[N5 (mixture regret)](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L144) |
| GenerationInformationProjection | [Finite deterministic CMI data processing](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L282) |
| GenerationLatchHealth | [H1 (lock-in is death)](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L27)；[H2 (latched means blind)](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L41)；[H1, quantitative](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L59) |
| GenerationLatchSlot | [A small gate is subcritical](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L54)；[The gate floor of a coupled latch](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L69)；[The disturbance floor](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L77)；[When the slot latches](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L94) |
| GenerationLattice | [M1a with conductances](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L39)；[M4 (lattice media are admitted)](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L423) |
| GenerationLearning | [L-1](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L27)；[L-1](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L31)；[L-2](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L37)；[L-2](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L53)；[L-3](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L61)；[O-1](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L84)；[O-2](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L92) |
| GenerationLeaveRule | [C-1 (a changing world needs comparison)](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L38) |
| GenerationLifeGate | [No exact latch](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L48)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L78)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L97)；[L-metabolism for rule tables](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L128)；[L-sense for rule tables](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L146) |
| GenerationLinearMemory | [Histories merge](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L78)；[A linear hold cannot clean up](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L105) |
| GenerationMedium | [M1a](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L24)；[M1b](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L45) |
| GenerationNestedWorld | [A move is an exchange](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L374)；[Mass action](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L442)；[Uniport selectivity is fixed by free energy](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L479)；[Antiport selectivity](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L497)；[T-E2 (an organism nested in its world is admitted)](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1235)；[Only events across the boundary feed the organism](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1362)；[Death](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1401)；[A dead body does not give itself away](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1420)；[Only dissolution empties a place](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1461)；[Births come only from division](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1488)；[Leak bound](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1553)；[Nested metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1654)；[Nested sensing](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1681)；[The two-dimensional world is a model of the axioms](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1787) |
| GenerationNesting | [G26a (the sensed potential of a world cell)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L39)；[G26b (staying put starves)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L58)；[T-N7 (the stream never runs dry)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L140)；[G26c (corpses must go: decomposition is necessary)](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L161) |
| GenerationOptimizer | [Theorem: CGB Optimizer Strictly Pareto-Dominates the Adam-Muon Hybrid](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L260) |
| GenerationOrganismEcoliMsS0 | [The organism is a model of the CGB axioms](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L52)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L58)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L65) |
| GenerationOrganismEcoliV1 | [The organism is a model of the CGB axioms](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L52)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L58)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L65) |
| GenerationOrganismGeneratedS0 | [The organism is a model of the CGB axioms](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L50)；[Life gate L-metabolism](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L56)；[Life gate L-sense](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L63) |
| GenerationPersistence | [A (Kolmogorov, one direction)](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L46)；[A](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L61)；[B (integral feedback adapts perfectly)](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L73)；[B](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L82)；[C](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L92)；[C](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L102) |
| GenerationPredictiveHarvest | [Gibbs' inequality](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L63)；[Kelly bound](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L75) |
| GenerationReinvestment | [Gross balanced exchange](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L64)；[Exact prediction exchanges nothing](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L77)；[Adaptation of the total](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L93)；[Adaptation of the level](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L100)；[Kelly equals Bayes](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L124)；[Credit from the same event](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L161) |
| GenerationSelfProduction | [Unbounded self-production.](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1298) |
| GenerationSpeechTempo | [N7](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L127) |
| GenerationTable | [M3](../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L72) |
| GenerationVoxelChemistry | [M5](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L252) |

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
- [GenerationActiveInference（5）](#generationactiveinference5)
- [GenerationActiveNavigation（12）](#generationactivenavigation12)
- [GenerationAmplifier（13）](#generationamplifier13)
- [GenerationAssociationCapacity（15）](#generationassociationcapacity15)
- [GenerationAxiomAudit（2）](#generationaxiomaudit2)
- [GenerationCascadeMemory（7）](#generationcascadememory7)
- [GenerationClosedLoopCredit（7）](#generationclosedloopcredit7)
- [GenerationColumnMemory（21）](#generationcolumnmemory21)
- [GenerationColumnRefinement（5）](#generationcolumnrefinement5)
- [GenerationConservedMemory（24）](#generationconservedmemory24)
- [GenerationContactCompetition（18）](#generationcontactcompetition18)
- [GenerationContactEligibility（5）](#generationcontacteligibility5)
- [GenerationContactPlasticity（15）](#generationcontactplasticity15)
- [GenerationContextAssociation（21）](#generationcontextassociation21)
- [GenerationContextBody（24）](#generationcontextbody24)
- [GenerationContinualOptimizer（42）](#generationcontinualoptimizer42)
- [GenerationCountedPopulation（30）](#generationcountedpopulation30)
- [GenerationCRN（10）](#generationcrn10)
- [GenerationDivisionTree（7）](#generationdivisiontree7)
- [GenerationDynamicTeaching（2）](#generationdynamicteaching2)
- [GenerationExport（6）](#generationexport6)
- [GenerationFutureValue（4）](#generationfuturevalue4)
- [GenerationGDNLife（80）](#generationgdnlife80)
- [GenerationGraceMemory（13）](#generationgracememory13)
- [GenerationGradientFilter（35）](#generationgradientfilter35)
- [GenerationGrowth（16）](#generationgrowth16)
- [GenerationHoldCorrect（19）](#generationholdcorrect19)
- [GenerationIgnition（5）](#generationignition5)
- [GenerationInformationEnergy（7）](#generationinformationenergy7)
- [GenerationInformationProjection（17）](#generationinformationprojection17)
- [GenerationLanguageExistence（18）](#generationlanguageexistence18)
- [GenerationLatchHealth（3）](#generationlatchhealth3)
- [GenerationLatchSlot（8）](#generationlatchslot8)
- [GenerationLattice（14）](#generationlattice14)
- [GenerationLearning（7）](#generationlearning7)
- [GenerationLeaveRule（3）](#generationleaverule3)
- [GenerationLifeGate（9）](#generationlifegate9)
- [GenerationLinearMemory（7）](#generationlinearmemory7)
- [GenerationMedium（2）](#generationmedium2)
- [GenerationNestedWorld（66）](#generationnestedworld66)
- [GenerationNesting（8）](#generationnesting8)
- [GenerationNonlinearity（12）](#generationnonlinearity12)
- [GenerationOptimizer（19）](#generationoptimizer19)
- [GenerationOrganismEcoliMsS0（3）](#generationorganismecolimss03)
- [GenerationOrganismEcoliV1（3）](#generationorganismecoliv13)
- [GenerationOrganismGeneratedS0（3）](#generationorganismgenerateds03)
- [GenerationPersistence（6）](#generationpersistence6)
- [GenerationPredictiveHarvest（9）](#generationpredictiveharvest9)
- [GenerationReciprocalCredit（2）](#generationreciprocalcredit2)
- [GenerationReinvestment（14）](#generationreinvestment14)
- [GenerationResponseGatedCarrier（12）](#generationresponsegatedcarrier12)
- [GenerationRoutingCredit（3）](#generationroutingcredit3)
- [GenerationSeedMemory（15）](#generationseedmemory15)
- [GenerationSelfProduction（138）](#generationselfproduction138)
- [GenerationSpeechTempo（14）](#generationspeechtempo14)
- [GenerationSurvivalBounds（16）](#generationsurvivalbounds16)
- [GenerationTable（2）](#generationtable2)
- [GenerationUnifiedOptimizer（42）](#generationunifiedoptimizer42)
- [GenerationVoxelChemistry（16）](#generationvoxelchemistry16)
- [GraceRecovery（20）](#gracerecovery20)
- [GraceRecoveryAccounting（14）](#gracerecoveryaccounting14)
- [GraceRecoveryClosure（18）](#gracerecoveryclosure18)
- [GraceRecoveryControls（4）](#gracerecoverycontrols4)
- [Inference（6）](#inference6)
- [InformationEconomics（3）](#informationeconomics3)
- [InformationThermodynamics（3）](#informationthermodynamics3)
- [Interaction（19）](#interaction19)
- [InteractionBridge（6）](#interactionbridge6)
- [InteractionControls（7）](#interactioncontrols7)
- [Language（1）](#language1)
- [LanguageLifeAdmission（1）](#languagelifeadmission1)
- [LanguageLifeKernel（54）](#languagelifekernel54)
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

- [`CGB.ActiveLoop.unique_vacuum`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L48) [design / candidate / reviewed]
- [`CGB.ActiveLoop.body_clock`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L77) [design / candidate / reviewed]
- [`CGB.ActiveLoop.body_axioms`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L91) [design / candidate / reviewed]
- [`CGB.ActiveLoop.fin6_sum`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L122) [design / candidate / reviewed]
- [`CGB.ActiveLoop.worldMoves_nonneg`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L151) [design / candidate / reviewed]
- [`CGB.ActiveLoop.worldMoves_sum`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L155) [design / candidate / reviewed]
- [`CGB.ActiveLoop.body_readsBoundary`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L180) [design / candidate / reviewed]
- [`CGB.ActiveLoop.active_expected_first_law`](../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L234) [design / candidate / reviewed]：Average first law holds along the actual active process.

## ActiveMaintenance（21）
_Resource-dependent maintenance witness A separate successor to the refuted ActiveLoop candidate._

- [`CGB.ActiveMaintenance.admission`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L42) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.reads_boundary`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L107) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.sum_configs`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L130) [witness / universal / reviewed]
- [`CGB.ActiveMaintenance.kernel_table`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L139) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.natural_law`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L156) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.recovery_step`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L179) [witness / universal / reviewed]
- [`CGB.ActiveMaintenance.normal_recovery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L205) [witness / universal / reviewed]
- [`CGB.ActiveMaintenance.natural_alive`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L211) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.natural_survival`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L227) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.starve_world_step`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L246) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.starve_recovery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L258) [witness / universal / reviewed]
- [`CGB.ActiveMaintenance.cut_move_early`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L278) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.cut_early_law`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L292) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.starve_mass_three`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L312) [witness / universal / reviewed]
- [`CGB.ActiveMaintenance.starve_mass_monotone`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L322) [witness / universal / reviewed]
- [`CGB.ActiveMaintenance.cut_alive_bound`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L348) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.cut_survival_mass_bound`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L363) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.cut_survival`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L390) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenance.maintenance_witness`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L403) [witness / candidate / reviewed]：The existing gross-exchange targets, for ALL future times.
- [`CGB.ActiveMaintenance.starvation_depletes`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L416) [witness / candidate / reviewed]：Scarcity really changes the body reserve, under either process; vacuum then has no action in any stimulus.
- [`CGB.ActiveMaintenance.sustained_exchange`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenance.lean#L432) [witness / candidate / reviewed]：Normal maintenance consumes and imports a positive unit EVERY active tick; this demand is a property of this mechanism, not a universal life axiom.

## ActiveMaintenanceCarrier（6）
_Three-token realization of the maintenance process One token is in the body, one in the accessible source, one in the return sink._

- [`CGB.ActiveMaintenanceCarrier.event_delivery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L53) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenanceCarrier.physical_delivery`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L74) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenanceCarrier.token_count`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L91) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenanceCarrier.token_transport`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L94) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenanceCarrier.realized`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L106) [witness / candidate / reviewed]：Full process and clock certificate, for both normal and marginal-ghost cut.
- [`CGB.ActiveMaintenanceCarrier.token_energy`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceCarrier.lean#L121) [witness / candidate / reviewed]

## ActiveMaintenanceEnergy（4）
_Reserve, explicit consumption and source-resolved replacement The same registered normal/cut event process pays one reserve unit on a live feeding or scarcity event._

- [`CGB.ActiveMaintenanceEnergy.actual_law`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L28) [witness / candidate / reviewed]：The energy law is exactly the state law used in the viability comparison, including the cut's own distribution-dependent future, not a second process.
- [`CGB.ActiveMaintenanceEnergy.body_supported`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L45) [witness / candidate / reviewed]
- [`CGB.ActiveMaintenanceEnergy.isothermal`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L86) [witness / universal / reviewed]：Conditional mean second-law admission for the explicitly declared internal entropy and bath units.
- [`CGB.ActiveMaintenanceEnergy.availability_budget`](../../lean/CGBAxioms/CGBAxioms/ActiveMaintenanceEnergy.lean#L108) [witness / universal / reviewed]：Every finite prefix uses the SAME event law and tick.

## AlgebraicCore（6）
_Algebraic finite-state core Finite-state transport and exact budget identities can be expressed over a commutative semiring._

- [`CGB.AlgebraicCore.pushForward_total`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L23) [semantics / universal / reviewed]
- [`CGB.AlgebraicCore.pathWeight_zero`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L37) [semantics / universal / reviewed]
- [`CGB.AlgebraicCore.jetMul_product_rule`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L47) [semantics / universal / reviewed]
- [`CGB.AlgebraicCore.jetHorner_value`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L59) [semantics / universal / reviewed]
- [`CGB.AlgebraicCore.cast_local_budget`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L67) [semantics / universal / reviewed]
- [`CGB.AlgebraicCore.cast_path_budget`](../../lean/CGBAxioms/CGBAxioms/AlgebraicCore.lean#L75) [semantics / universal / reviewed]

## AlgebraicHomology（2）
_Minimal algebraic homology core Finite boundary operators are represented by matrices over a commutative semiring._

- [`CGB.AlgebraicHomology.matrixAction_compose`](../../lean/CGBAxioms/CGBAxioms/AlgebraicHomology.lean#L27) [semantics / universal / reviewed]
- [`CGB.AlgebraicHomology.boundary_boundary_zero`](../../lean/CGBAxioms/CGBAxioms/AlgebraicHomology.lean#L35) [semantics / universal / reviewed]

## AtomicIndependence（24）
_Full remaining-twelve certificates Since 2026-10-26 contact homogeneity (former A04) is no longer an axiom part; the thirteen remaining atomic conditions keep their ids (A01–A03, A05–A14)._

- [`CGB.Medium.atomized_iff_all_conditions`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L45) [semantics / universal / reviewed]
- [`CGB.Medium.a01_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L62) [witness / universal / reviewed]
- [`CGB.Medium.a02_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L71) [witness / universal / reviewed]
- [`CGB.Medium.a03_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L80) [witness / universal / reviewed]
- [`CGB.Medium.a09_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L89) [witness / universal / reviewed]
- [`CGB.Medium.a10_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L98) [witness / universal / reviewed]
- [`CGB.Medium.a11_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L107) [witness / universal / reviewed]
- [`CGB.Medium.a07_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L116) [witness / universal / reviewed]
- [`CGB.Medium.a12_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L125) [witness / universal / reviewed]
- [`CGB.Medium.a14_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L134) [witness / universal / reviewed]
- [`CGB.Medium.globalFlip_other_clauses`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L158) [witness / universal / reviewed]
- [`CGB.Medium.a05_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L172) [witness / universal / reviewed]
- [`CGB.Medium.a06_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L185) [witness / universal / reviewed]
- [`CGB.Medium.farAct_other_clauses`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L203) [witness / universal / reviewed]
- [`CGB.Medium.a08_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L232) [witness / universal / reviewed]
- [`CGB.Medium.innerActClock_other_clauses`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L240) [witness / universal / reviewed]
- [`CGB.Medium.a13_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L276) [witness / universal / reviewed]
- [`CGB.Medium.twinFlip_all_conditions`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L285) [witness / universal / reviewed]：Consistency is witnessed separately from independence.
- [`CGB.Medium.all_atoms_independent`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L305) [witness / universal / reviewed]
- [`CGB.Medium.reducedCore_iff_selected_conditions`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L430) [semantics / universal / reviewed]
- [`CGB.Medium.rigidLabel_rigid`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L443) [semantics / universal / reviewed]
- [`CGB.Medium.holds_rigidLabel_iff`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L446) [semantics / universal / reviewed]
- [`CGB.Medium.isolates_rigidLabel`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L451) [semantics / universal / reviewed]
- [`CGB.Medium.every_reduced_condition_independent_in_regime`](../../lean/CGBAxioms/CGBAxioms/AtomicIndependence.lean#L466) [witness / universal / reviewed]：Every selected condition has a finite-local-state countermodel in the SAME rigid-structure regime used by the recovery bridge.

## BufferedMaterial（10）
_Finite identity-preserving buffered material transport Twelve fixed carriers suffice for a body containing at most four units, a four-unit accessible source and at least four returned carriers._

- [`CGB.BufferedMaterial.canonical_stock`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L30) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.total`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L34) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.select`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L44) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.transport_exists`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L54) [witness / universal / reviewed]：Local existence with every intake/release/recharge coming from old stock.
- [`CGB.BufferedMaterial.available`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L145) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.advance_project`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L160) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.lift_sum`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L166) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.refines`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L196) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.delivery`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L210) [witness / universal / reviewed]
- [`CGB.BufferedMaterial.realized`](../../lean/CGBAxioms/CGBAxioms/BufferedMaterial.lean#L237) [witness / universal / reviewed]：All original states, all supported events, and the same supplied clock.

## CausalLearning（16）
_Finite, causal and paid candidate-model learning Models predict possible boundary outcomes of actions._

- [`CGB.CausalLearning.memory_cardinality`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L50) [design / candidate / reviewed]：Candidate masks have |H| bits of state capacity, plus the explicit reserve.
- [`CGB.CausalLearning.write_contract`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L61) [design / candidate / reviewed]：Each effective write strictly removes candidates and spends one declared unit of finite reserve.
- [`CGB.CausalLearning.retain_consistent`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L84) [design / candidate / reviewed]
- [`CGB.CausalLearning.agreement_no_write`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L93) [design / candidate / reviewed]：A predictable observation provides no exclusion evidence.
- [`CGB.CausalLearning.empty_write_reserve`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L101) [design / candidate / reviewed]
- [`CGB.CausalLearning.paid_elimination`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L108) [design / candidate / reviewed]：A funded observation can remove a false candidate using the observation itself, without consulting a true-model identifier.
- [`CGB.CausalLearning.path_budget`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L125) [design / candidate / reviewed]：Analysis may count writes along a path; that counter is not learner state.
- [`CGB.CausalLearning.delivered`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L164) [design / candidate / reviewed]
- [`CGB.CausalLearning.true_model_step`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L182) [design / candidate / reviewed]
- [`CGB.CausalLearning.law_invariant`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L192) [design / candidate / reviewed]
- [`CGB.CausalLearning.true_model_law`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L215) [design / candidate / reviewed]：Positive-probability states retain the faithful candidate at every time; the controller definition itself never receives this candidate's identity.
- [`CGB.CausalLearning.physical_marginal`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L231) [design / candidate / reviewed]：Even with zero write reserve, each physical action/outcome law continues.
- [`CGB.CausalLearning.safe_outcome`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L242) [design / candidate / reviewed]
- [`CGB.CausalLearning.certified_choice_sound`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L268) [design / candidate / reviewed]
- [`CGB.CausalLearning.no_safe_exploration`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L288) [necessity / world_contract / reviewed]
- [`CGB.CausalLearning.paid_mean_budget`](../../lean/CGBAxioms/CGBAxioms/CausalLearning.lean#L317) [design / candidate / reviewed]：Exact mean finite-prefix payment on the actual action/outcome process.

## ConstraintAudit（15）
_Joint audit of irredundancy and generative freedom No new physical clause is introduced._

- [`CGB.Medium.axioms_iff_atomized`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L38) [semantics / universal / reviewed]
- [`CGB.Medium.p1a_iff_finite_and_unique`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L58) [semantics / universal / reviewed]：The first two cost clauses jointly say finite local state space in this language.
- [`CGB.Medium.rigid_of_injective_identity`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L71) [semantics / universal / reviewed]
- [`CGB.Medium.p2c_of_rigid`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L77) [semantics / universal / reviewed]
- [`CGB.Medium.initial_invariant_of_rigid`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L82) [semantics / universal / reviewed]
- [`CGB.Medium.onlyBoundarySenses_of_all_boundary`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L88) [semantics / universal / reviewed]
- [`CGB.Medium.p2a_of_every_event_open`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L95) [semantics / universal / reviewed]
- [`CGB.Medium.empty_participants_no_effect`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L98) [semantics / universal / reviewed]
- [`CGB.Medium.jump_equivariant`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L106) [semantics / universal / reviewed]：A genuine joint consequence: event equivariance descends to the aggregate jump kernel, including the single shared event-permutation witness.
- [`CGB.Medium.empty_boundary_every_stimulus_reversible`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L124) [semantics / universal / reviewed]：P3 with no boundary removes ALL stimulus dependence.
- [`CGB.Medium.rescale_jump`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L142) [semantics / universal / reviewed]
- [`CGB.Medium.rescale_axioms`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L150) [semantics / universal / reviewed]
- [`CGB.Medium.rescale_changes_nonzero_jump`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L173) [semantics / universal / reviewed]
- [`CGB.JumpProcess.fromConductance_reversible`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L192) [semantics / universal / reviewed]
- [`CGB.JumpProcess.reversible_conductance_representation`](../../lean/CGBAxioms/CGBAxioms/ConstraintAudit.lean#L200) [semantics / universal / reviewed]

## ConstraintFreedom（4）
_Families admitted by the same clauses These are logical model families, not proposed brain mechanisms or capability experiments._

- [`CGB.ConstraintFreedom.arbitrarily_many_states_at_unit_capacity`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L19) [witness / universal / reviewed]：P1a is qualitative: an arbitrarily large FINITE state alphabet may have capacity 1.
- [`CGB.ConstraintFreedom.arbitrary_driven_tables_admitted`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L49) [semantics / universal / reviewed]：Thus the core alone does not determine a driven rule table.
- [`CGB.ConstraintFreedom.openSite_reproduces_process`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L80) [semantics / universal / reviewed]：With one event per target state, the exposed-site family reproduces the off-diagonal rates of ANY finite Markov jump process under its driven input.
- [`CGB.ConstraintFreedom.constant_protocol_can_be_driven`](../../lean/CGBAxioms/CGBAxioms/ConstraintFreedom.lean#L88) [witness / universal / reviewed]：A constant-in-time stimulus already breaks detailed balance in the existing seed witness.

## EnergyBudget（5）
_Energy balance and conditional mean availability budget Physical energy, work/heat ports, temperature and entropy require a carrier interpretation._

- [`CGB.Energetics.law_probability`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L28) [semantics / universal / reviewed]
- [`CGB.Energetics.expected_first_law`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L66) [semantics / universal / reviewed]：Average the supported first law along the ORIGINAL process.
- [`CGB.Energetics.availability_balance`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L118) [semantics / universal / reviewed]
- [`CGB.Energetics.work_budget`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L133) [semantics / universal / reviewed]：One finite-prefix identity gives both the bounded-reserve work budget and the cyclic bound.
- [`CGB.Energetics.no_sustained_use_on_finite_supply`](../../lean/CGBAxioms/CGBAxioms/EnergyBudget.lean#L169) [necessity / world_contract / reviewed]：A positive mean availability-use requirement (delivered work plus T times production) cannot persist on bounded total supply and reserve.

## EnergyControls（4）
_Energy-interpretation controls on the unchanged material process The same material transport admits different energy calibrations._

- [`CGB.EnergyControls.total_partition`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L24) [witness / universal / reviewed]
- [`CGB.EnergyControls.material_energy_binding`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L43) [witness / universal / reviewed]：On every supported event, the material-energy port is carried by the same exterior token identified in the material-source certificate.
- [`CGB.EnergyControls.isothermal_account`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L54) [witness / universal / reviewed]：Nonnegative mean production for the declared T=1, zero-system-entropy accounting interpretation along the actual process; not a derivation of the microscopic entropy function or of local detailed balance.
- [`CGB.EnergyControls.same_material_different_energy`](../../lean/CGBAxioms/CGBAxioms/EnergyControls.lean#L75) [witness / universal / reviewed]：Fixed identities and identical counts do not select chemical energy or work.

## EventChannels（15）
_Event-resolved observations with unchanged body/world marginals State-preserving events can carry gross exchange._

- [`CGB.FirstHit.Kernel.eq_of_step`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L17) [witness / universal / reviewed]
- [`CGB.EventChannels.Kernel.firstHit_bounds`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L50) [witness / universal / reviewed]
- [`CGB.EventChannels.Kernel.state_mark_firstHit`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L79) [witness / universal / reviewed]：A state-only mark is the specialization of the single channel observer that ignores event labels.
- [`CGB.EventChannels.Kernel.firstHit_unmarked`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L92) [witness / universal / reviewed]
- [`CGB.EventChannels.Kernel.labelledLaw_marginal`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L110) [witness / universal / reviewed]
- [`CGB.Medium.exit_rate_events`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L138) [witness / universal / reviewed]
- [`CGB.Medium.eventClocked_implies_clocked`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L153) [witness / universal / reviewed]
- [`CGB.Medium.eventStep_nonneg`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L168) [witness / universal / reviewed]
- [`CGB.Medium.eventStep_sum`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L183) [witness / universal / reviewed]
- [`CGB.Medium.eventStep_forget`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L191) [witness / universal / reviewed]：Forgetting the event labels recovers the ORIGINAL kernel, even when some positive-rate events leave the body state unchanged.
- [`CGB.Medium.eventWorldStep_nonneg`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L229) [witness / universal / reviewed]
- [`CGB.Medium.eventWorldStep_sum`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L239) [witness / universal / reviewed]
- [`CGB.Medium.jointEventKernel_forget`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L257) [witness / universal / reviewed]
- [`CGB.Medium.event_state_mark_matches_operational`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L267) [witness / universal / reviewed]
- [`CGB.Medium.jointEventKernel_stateLaw`](../../lean/CGBAxioms/CGBAxioms/EventChannels.lean#L287) [witness / universal / reviewed]：The entire observed forward process has the original normal/cut state law, not just an equal one-step kernel.

## Foundation（14）
_Foundation audit: premises, specialization, and clocks These results audit the existing predicates; they add no physical axiom and do not change the organism dynamics or the life target._

- [`CGB.Medium.finite_local_states_of_p1a`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L18) [semantics / universal / reviewed]：P1a itself supplies local-state finiteness, before a finite enumeration is chosen.
- [`CGB.Medium.p1a_iff_unique_vacuum`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L27) [semantics / universal / reviewed]：On an already finite local state type, P1a's first two conjuncts are automatic.
- [`CGB.Medium.exists_clock_of_finite_stimuli`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L42) [semantics / universal / reviewed]：A globally bounded clock follows if the stimulus alphabet is also finite.
- [`CGB.Medium.recover_ignores_world_moves`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L65) [semantics / universal / reviewed]：The current recovery menu does not consult actual world transitions.
- [`CGB.Medium.recover_bounds`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L77) [semantics / universal / reviewed]：Recovery is a genuine probability only with the clock-admission hypothesis.
- [`CGB.Medium.law_nonneg_of_clocked`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L112) [semantics / universal / reviewed]：Normal and loop-cut laws must be nonnegative as well as normalized.
- [`CGB.Medium.lawCut_nonneg_of_clocked`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L122) [semantics / universal / reviewed]
- [`CGB.Medium.aliveProb_bounds`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L135) [semantics / universal / reviewed]
- [`CGB.Medium.aliveProb_zero`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L152) [semantics / universal / reviewed]
- [`CGB.Medium.persists_parameter_bounds`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L159) [semantics / universal / reviewed]：Probability/horizon parameter bounds follow from a persistence witness and clock admission; they need not be introduced as additional physical axioms.
- [`CGB.FoundationCounterexamples.unboundedStimulus_axioms`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L198) [semantics / universal / reviewed]
- [`CGB.FoundationCounterexamples.unboundedStimulus_no_clock`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L226) [semantics / universal / reviewed]：Six clauses plus finite body/state/event types do not imply a global Clocked bound.
- [`CGB.TheoryAudit.existence_survives_weakening`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L251) [semantics / universal / reviewed]：Weakening admissibility cannot destroy an already constructed existence witness.
- [`CGB.TheoryAudit.counterexample_blocks_guarantee`](../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L258) [semantics / universal / reviewed]：A counterexample does refute a universal guarantee from the remaining premises.

## Generation（35）
_Generation, layer 1: what the axioms force on the equilibrium law The axioms constrain; this file starts deriving._

- [`CGB.JumpProcess.reversible_mul_invariant`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L46) [design / universal / reviewed]：**G3**: a positive factor that is constant along every move preserves detailed balance.
- [`CGB.JumpProcess.freezeSlow_conditional_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L74) [design / universal / reviewed]：**G13 (prediction is the boundary's Gibbs conditional)**: if the joint calm dynamics has detailed-balance weight `π`, then with the slow interior `P` frozen, for each fixed `p` the conditional `π (·, p) / Σ π (·, p)` is
- [`CGB.Medium.parts_subset_scope`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L94) [design / universal / reviewed]：Under P1b an event that changes all of `D` reads only inside `scope D`.
- [`CGB.Medium.content_fixed_by_closed_single_unit`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L100) [design / universal / reviewed]：**G2**: under P2a a closed event that changes a single unit keeps that unit's content, so content moves inside the body only by events that change at least two units.
- [`CGB.Medium.jump_rate_local`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L118) [design / universal / reviewed]：**G1**: the total rate of a transition `x → y` with changed set `D` depends only on the new values on `D` and on the old state and stimulus on `D ∪ scope D`.
- [`CGB.Medium.equilibrium_ratio_local`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L171) [design / universal / reviewed]：**G1E**: across one calm transition with a positive reverse rate, the equilibrium ratio `π y / π x` depends only on the same local data as G1.
- [`CGB.Medium.local_markov_of_single_unit_moves`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L197) [design / universal / reviewed]：**G4**: if every value of unit `u` can be left by a calm move that changes only `u`, the equilibrium law is locally Markov at `u`: the ratio `π (x with u := l) / π x` depends only on `x` on `{u} ∪ scope {u}`.
- [`CGB.Medium.p2b_of_micro_weighted`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L224) [design / universal / reviewed]：**P2b from weighted micro-reversibility** (generalizes `p2b_of_micro`, which has uniform weights): if events pair up by an involution `rev` so that every effective calm event is undone by its partner and the pair satisfi
- [`CGB.Medium.rate_interior_stimulus_free`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L255) [design / universal / reviewed]：**G6**: under P1b and P3 a transition whose changed set and its scope avoid the boundary has the same rate under every stimulus as in the calm environment.
- [`CGB.Medium.interior_detailed_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L269) [design / universal / reviewed]：**G6DB**: hence, with P2b, interior transitions satisfy detailed balance with the calm weight under every stimulus: all irreversibility enters at the boundary.
- [`CGB.Medium.coupling_vanishes_beyond_scope`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L282) [design / universal / reviewed]：**G7**: no equilibrium coupling beyond contact.
- [`CGB.Medium.modulation_needs_boundary`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L298) [necessity / universal / reviewed]：**G9 (modulation needs the boundary)**: if the rate of a transition (for example a plastic move: its learning rate or, through the forward/backward pair, its learning direction) differs between two stimuli, then some uni
- [`CGB.Medium.fixed_gain_error`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L317) [design / universal / reviewed]：**G10a**: the fixed-gain error exceeds the optimum `P` by an exact square; the unique optimal fixed gain is the Kalman gain `P / (P + r)`.
- [`CGB.Medium.no_fixed_gain_for_two_volatilities`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L330) [design / universal / reviewed]：**G10b (necessity of modulation, minimal form)**: in two worlds with different volatility (`P₁ ≠ P₂`), no fixed gain is optimal in both: in at least one world its error is strictly above that world's optimum.
- [`CGB.Medium.no_global_gain_for_heterogeneous_compartments`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L361) [necessity / world_contract / reviewed]：**G10c (heterogeneous compartments need local gains)**: model two compartments by their local volatility parameters `P c`.
- [`CGB.Medium.miss_posterior_bayes`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L385) [design / universal / reviewed]：The miss posterior is Bayes' rule: with prior `π`, miss probabilities `qr < qp` (`qr = 1 - pr`, `qp = 1 - pp`), `n` misses give rich posterior `1 / (1 + a ρⁿ)` with `a = (1-π)/π` and `ρ = qp/qr`.
- [`CGB.Medium.missPrediction_strictAnti`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L395) [design / universal / reviewed]：**G11a**: each further miss strictly lowers the best prediction.
- [`CGB.Medium.intake_memory_capacity`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L414) [design / universal / reviewed]：**G11b (interoceptive memory)**: to predict intake within `ε` after every history of up to `N` consecutive misses, where `2ε` is below every gap between those predictions, an organism's state must take at least `N + 1` d
- [`CGB.Medium.intake_gap_exists`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L431) [design / universal / reviewed]：The gap condition is satisfiable: strict monotonicity separates any finite family, so some positive precision forces `N + 1` states.
- [`CGB.Medium.exists_leave_threshold`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L452) [design / universal / reviewed]：The leave threshold exists: whenever going elsewhere is worth more than a known-poor source (`pp < c`), enough misses push the best prediction below `c`.
- [`CGB.Medium.miss_counter_necessary`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L480) [design / universal / reviewed]：**G12 (a miss counter is necessary for optimal foraging)**: let `nStar` be the first miss count whose best prediction falls below `c` (all earlier counts stay at or above it).
- [`CGB.Medium.merging_across_threshold_loses`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L508) [design / universal / reviewed]：Merging two histories that lie on opposite sides of the threshold costs expected intake: with positive weights, a single shared action earns strictly less than acting optimally on each.
- [`CGB.Medium.two_phase_drift`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L527) [design / universal / reviewed]：**G14a (two-phase drift)**: in the drive phase the boundary is clamped to the arrived value `b'` and the site moves by `+λ x` when the boundary shows `b`; in the free phase the boundary samples the readout law and the si
- [`CGB.Medium.softmax_sum`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L535) [design / universal / reviewed]：The readout law sums to one.
- [`CGB.Medium.log_softmax_deriv`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L542) [design / universal / reviewed]：**G14b**: the log likelihood of the arrived value `b'` changes, per unit change of that coupling, by `x ([b' = b] - softmax z b)`.
- [`CGB.Medium.phases_must_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L579) [necessity / universal / reviewed]：**G14c (the phases must balance)**: if, for every readout state and arrived value, the expected drift vector over boundary values is a positive multiple of the log-likelihood gradient vector, the drive and free phases ca
- [`CGB.Medium.leakage_loses`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L600) [design / universal / reviewed]：**R2 (leakage loses on every outcome)**: if the free boundary puts mass `ℓ > 0` on configurations that are no symbol, every symbol's predicted probability is `(1 - ℓ) q b` instead of `q b`, and the log likelihood of what
- [`CGB.Medium.one_unit_one_label`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L608) [design / universal / reviewed]：**R1 (conservation makes the readout one-hot)**: if every label unit holds content `0` or `1` and the readout's total content is `1` (an invariant shell of the closed free dynamics under P2a, `total_of_reachable`), exact
- [`CGB.Medium.sector_law_is_softmax`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L626) [design / universal / reviewed]：In the one-particle sector a local label energy gives the softmax readout: with weights `exp (z b)` on the configurations `label b on`, the normalized law is `softmax z` (G13 restricted to the sector).
- [`CGB.Medium.reward_two_phase_drift`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L642) [design / universal / reviewed]：**G15a (reward-modulated two-phase drift is the policy gradient)**: if the motor particle was clamped at the action taken `a ~ softmax z`, the drive phase moved the site by `(R a - bl) x` when label `b` was on, and the f
- [`CGB.Medium.expected_reward_deriv`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L658) [design / universal / reviewed]：**G15b**: that drift is exactly the derivative of the expected reward `J` with respect to the logit of `b`.
- [`CGB.Medium.outcome_blind_drift_zero`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L698) [design / universal / reviewed]：**G15c (outcome-blind modulation learns nothing)**: if the modulator's weight `m` does not depend on the action's outcome, the expected two-phase drift is zero for every site.
- [`CGB.Medium.reward_phases_must_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L710) [necessity / universal / reviewed]：**G15d (reward phases must balance)**: with drive weight `lam` and free weight `mu`, if the expected drift is a multiple of the expected-reward gradient for every reward table, then `lam = mu`.
- [`CGB.Medium.two_part_balance`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L751) [design / universal / reviewed]：**G8a** (two parts suffice): with positive part factors `φ₁, φ₂` and constants `a, b ≥ 0`, up rates `a φ₁ + b φ₂` and down rates `b / φ₁ + a / φ₂` (each term reads one part) have ratio exactly `φ₁ φ₂`.
- [`CGB.Medium.three_part_balance_impossible`](../../lean/CGBAxioms/CGBAxioms/Generation.lean#L761) [necessity / universal / reviewed]：**G8b** (three parts are impossible): if up rates `f₁+f₂+f₃` and down rates `g₁+g₂+g₃` (each term reading one part) satisfy detailed balance with a product factor `φ₁ φ₂ φ₃` that is non-constant on every part, the total

## GenerationActiveInference（5）
_Active inference from finite supply (minimal information contract) This module does not add a physical entropy variable or a new organism state._

- [`CGB.ActiveInference.not_unbounded_of_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L41) [necessity / world_contract / reviewed]：A bounded cumulative signal cannot satisfy the unbounded viability contract.
- [`CGB.ActiveInference.viable_not_eventually_passive`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L52) [necessity / world_contract / reviewed]：**Active necessity.** If viability requires an unbounded usable signal, and every eventually passive policy has a bounded signal in the declared world, then a viable policy cannot become eventually passive.
- [`CGB.ActiveInference.action_condition_required`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L68) [necessity / world_contract / reviewed]：If the world is action-sensitive, an action-blind predictor cannot match every action-conditioned response.
- [`CGB.ActiveInference.silence_condition_required`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L78) [necessity / world_contract / reviewed]：Silence is an ordinary byte outcome.
- [`CGB.ActiveInference.active_closure`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveInference.lean#L92) [necessity / world_contract / reviewed]：**Active closure.** Under one declared world contract, viability rules out eventual passivity and action-sensitive future inputs rule out an action-blind predictor.

## GenerationActiveNavigation（12）
_Active navigation: sphere slot, branch residuals, sign credit, reactivation (2026-11-04) Design: `docs/information_boltzmann/CGB_ACTIVE_LATCH_NAVIGATION_20261104.md`._

- [`CGB.Generation.ActiveNavigation.fb_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L53) [design / candidate / draft]
- [`CGB.Generation.ActiveNavigation.radial_renewal_norm`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L61) [design / candidate / draft]：The length of the renewed read is the scalar latch map of the old length.
- [`CGB.Generation.ActiveNavigation.radial_renewal_direction`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L68) [design / candidate / draft]：The direction is unchanged: the renewed read is a positive multiple of the old one.
- [`CGB.Generation.ActiveNavigation.branch_residual_common_invariant`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L85) [semantics / universal / draft]：**A common component cancels in every residual** (for at least one branch).
- [`CGB.Generation.ActiveNavigation.latch_keeps_sign_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L98) [design / candidate / draft]：**Positive content stays positive** under the renewal, at every step.
- [`CGB.Generation.ActiveNavigation.latch_keeps_sign_neg`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L107) [design / candidate / draft]：Negative content stays negative (oddness of the renewal).
- [`CGB.Generation.ActiveNavigation.latch_reads_forget_magnitude`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L121) [design / candidate / draft]：**The read forgets the magnitude of the write**: two writes of the same sign inside the basin (`ρ ≤ x₀, x₀'`) are read within `L ^ n (|x₀ - r*| + |x₀' - r*|)` of each other, with `L = factor a g ρ < 1` (`latch_converges`
- [`CGB.Generation.ActiveNavigation.reactivation_exact_held`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L137) [semantics / universal / draft]：**Reactivation is exact for a held write**: re-encoding the stored input at read time gives the derivative of the later loss through the hold `a ^ n`.
- [`CGB.Generation.ActiveNavigation.reactivation_superposition`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L144) [semantics / universal / draft]：**Superposition**: a linear read `∑ j, k j * E j θ` of several writes has the sum of the per-write re-encoded terms as its derivative.
- [`CGB.Generation.ActiveNavigation.reactivation_partial_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L154) [semantics / universal / draft]：**The only bias is the writes not retained**: the true derivative minus the retained part is the sum over the dropped writes.
- [`CGB.Generation.ActiveNavigation.progress_steps_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L165) [semantics / universal / draft]：**Progress from one source is used up.** If the excess losses `e t ≥ 0` of a learner on a source have bounded running sums (what `mixture_regret_le` gives a Bayes mixture on a fixed source: at most `log (1 / w θ₀)`), the
- [`CGB.Generation.ActiveNavigation.mixture_without_entry`](../../lean/CGBAxioms/CGBAxioms/GenerationActiveNavigation.lean#L177) [semantics / universal / draft]：**Removing one entry from a linear mixture** `p = w q + (1 - w) p₋`: the mixture without it is `(p - w q) / (1 - w)`, so the counterfactual of one entry needs no second forward pass.

## GenerationAmplifier（13）
_An equilibrium amplifier: ordered filling of a conserved shell (generation chain, G25) Chemotaxis needs an all-or-none response to a tiny difference (a gain of order ten in *E._

- [`CGB.Generation.occupancy_le_exp`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L42) [design / candidate / reviewed]
- [`CGB.Generation.one_sub_occupancy_le_exp`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L49) [design / candidate / reviewed]
- [`CGB.Generation.occupancy_le_one`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L61) [design / candidate / reviewed]
- [`CGB.Generation.high_off_below_threshold`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L68) [design / candidate / reviewed]：**G25a (H stays off below threshold)**: while the low page is at most half full, the high page is occupied at most `e^{-Δ}`.
- [`CGB.Generation.switch_width`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L78) [design / candidate / reviewed]：**G25b (switch width)**: raising the potential from `l1` to `l2` adds at most `NH` times the gain in `H`'s occupancy plus `NL e^{-(l1 - φL)}` content.
- [`CGB.Generation.equilibrium_hill_one`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L90) [design / candidate / reviewed]：**G25c (equilibrium Hill coefficient is one)**: the log-odds of occupation is exactly the potential above the unit's energy, so it moves one-for-one with the sensed potential.
- [`CGB.Generation.quanta_hill`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L99) [design / candidate / reviewed]：**G25d (cooperativity from multiple quanta)**: a state holding `s` units of content, with energy `φs`, has equilibrium weight `exp (s μ - φs)` relative to the empty state, so its log-odds moves `s`-for-one with the poten
- [`CGB.Generation.occupancy_ge_inv_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L106) [design / candidate / reviewed]：Lower edge of the resolvable range: with `N` units, the occupancy is at least one unit's worth (`1/N`) exactly when `-h ≤ log (N - 1)`.
- [`CGB.Generation.occupancy_le_one_sub_inv_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L114) [design / candidate / reviewed]：Upper edge: the occupancy leaves at least one unit empty exactly when `h ≤ log (N - 1)`.
- [`CGB.Generation.resolvable_band`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L140) [design / candidate / reviewed]：**G25e (gain–range bound)**: a switch of gain `s` (a state of `s` quanta, G25d) centred at `θ`, read through `N` units, is resolvable (between one occupied and one empty unit) exactly on `|μ - θ| ≤ log (N - 1) / s`.
- [`CGB.Generation.gain_times_band`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L151) [design / candidate / reviewed]：The product of gain and resolvable width is `2 log (N - 1)`, independent of the gain: an amplified switch covers a proportionally narrower band of the sensed potential.
- [`CGB.Generation.fixed_threshold_fails`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L163) [design / candidate / reviewed]：**G25f (a fixed threshold fails on a wide world)**: if the sensed potential ranges over an interval wider than `2 log (N - 1) / s`, a switch of gain `s` at any fixed `θ` is unresolvable somewhere in it.
- [`CGB.Generation.centred_threshold_resolves`](../../lean/CGBAxioms/CGBAxioms/GenerationAmplifier.lean#L178) [design / candidate / reviewed]：**G25f (a threshold centred on the prediction works everywhere)**: if the threshold sits at a stored prediction `μhat` (G13: the slow interior's conditional) and the innovation `μ - μhat` (G14) stays within `δ` with `s δ

## GenerationAssociationCapacity（15）
_Exact and mixed capacity of an associative linear state A GDN fast-weight state is a superposition of value-weighted keys._

- [`CGB.Generation.associationState_row`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L29) [design / candidate / reviewed]：Each output row of `V Kᵀ` is one scalar key superposition.
- [`CGB.Generation.association_slot_retention`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L38) [design / candidate / reviewed]：Slotwise natural-state retention transports each stored key by the same slot factors.
- [`CGB.Generation.association_state_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L56) [design / candidate / reviewed]：Reading `V Kᵀ` with a query is the corresponding weighted combination of values; this is the algebraic reference for GDN probe readout.
- [`CGB.Generation.association_readout_interference_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L74) [design / candidate / reviewed]：A single interference matrix bounds the readout error uniformly over all bounded values.
- [`CGB.Generation.probeFamilyAccurate_mono`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L111) [design / candidate / reviewed]：Accuracy is monotone in the declared tolerance.
- [`CGB.Generation.exact_decoder_capacity`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L127) [necessity / universal / reviewed]：Exact decoding itself forces the same key-dimension bound.
- [`CGB.Generation.exact_decoder_association_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L158) [design / candidate / reviewed]
- [`CGB.Generation.bounded_decoder_association_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L167) [design / candidate / reviewed]
- [`CGB.Generation.normalized_assimilation_decoder_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L185) [design / candidate / reviewed]：A weighted natural-parameter write followed by slotwise normalization is an associative superposition with effective keys `wᵢ kᵢⱼ / pⱼ`, plus the normalized old state.
- [`CGB.Generation.exact_decoder_is_bounded_decoder`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L224) [design / candidate / reviewed]
- [`CGB.Generation.precision_assimilation_is_innovation`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L237) [design / candidate / reviewed]：A natural-parameter assimilation is an innovation update, with gain `w / (p + w)` for the individual slot.
- [`CGB.Generation.precision_assimilation_gain_bounds`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L246) [design / candidate / reviewed]：The precision-derived slot gain is bounded when prior precision is positive and incoming precision is nonnegative.
- [`CGB.Generation.process_noise_gain_floor`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L258) [design / candidate / reviewed]：With strictly positive process variance and a lower bound on incoming precision, one predict-then-assimilate step has a positive gain floor.
- [`CGB.Generation.exact_key_capacity`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L271) [necessity / universal / reviewed]
- [`CGB.Generation.collision_of_excess_keys`](../../lean/CGBAxioms/CGBAxioms/GenerationAssociationCapacity.lean#L278) [necessity / universal / reviewed]：Beyond key dimension, linear superposition is not injective: two distinct coefficient assignments produce the same stored state.

## GenerationAxiomAudit（2）
_Which axiom parts work (generation chain, audit) Usage in the generation chain (G1–G22, M1–M2): P1b's locality parts (participants in contact, writes and reads only participants), P2a, P2b and P3's boundary sensing enter_

- [`CGB.Medium.homogeneous_regular`](../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L26) [semantics / universal / reviewed]：**Homogeneity forces a regular contact graph**: every unit has the same number of contacts.
- [`CGB.Medium.not_homogeneous_of_degrees`](../../lean/CGBAxioms/CGBAxioms/GenerationAxiomAudit.lean#L45) [semantics / universal / reviewed]：A unit with a different number of contacts than another rules out homogeneity.

## GenerationCascadeMemory（7）
_A bank of two-state pools keeps a power-law trace (generation chain, cascade memory, 2026-10-27) G20a (`two_state_geometric`) makes the trace of one two-state pool exponential._

- [`CGB.Generation.single_pool_falls_below_power`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L29) [design / candidate / reviewed]：One pool's trace drops below `c / t` once `t > 2 τ / c`.
- [`CGB.Generation.bank_trace_ge_one_term`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L58) [design / candidate / reviewed]：The bank's trace is at least any one of its (nonnegative) terms.
- [`CGB.Generation.pool_in_band_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L63) [design / candidate / reviewed]：A pool with `t ≤ τ ≤ γ t` alone keeps `1 / (e γ t)`.
- [`CGB.Generation.bank_power_law_lower_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L77) [necessity / candidate / reviewed]
- [`CGB.Generation.geometric_bank_covers`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L84) [design / candidate / reviewed]：Geometric time constants cover the band: every `t ∈ [τ₀, τ₀ γ^m]` has a level `k ≤ m` with `t ≤ τ₀ γ^k ≤ γ t`.
- [`CGB.Generation.ranked_release_eq_inv_time`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L123) [design / candidate / reviewed]：A ranked release is exactly the reciprocal of its ranked relaxation time.
- [`CGB.Generation.ranked_time_succ`](../../lean/CGBAxioms/CGBAxioms/GenerationCascadeMemory.lean#L130) [design / candidate / reviewed]：Adjacent ranks have the geometric time-scale ratio used by `geometric_bank_covers`.

## GenerationClosedLoopCredit（7）
_Closed-loop credit with a shared policy and predictor G15 (`Medium.expected_reward_deriv`) differentiates a fixed reward table._

- [`CGB.ClosedLoopCredit.objective_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L43) [design / candidate / reviewed]：Product rule for the shared policy/predictor, including direct score credit.
- [`CGB.ClosedLoopCredit.policy_deriv_sum_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L61) [semantics / universal / reviewed]：Differentiating normalization forces the policy tangents to sum to zero.
- [`CGB.ClosedLoopCredit.scalar_baseline_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L72) [semantics / universal / reviewed]：A scalar baseline cancels against a normalized conditional outcome law.
- [`CGB.ClosedLoopCredit.objective_deriv_advantage`](../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L82) [design / candidate / reviewed]：Three-factor advantage/score credit PLUS the complete direct score derivative.
- [`CGB.ClosedLoopCredit.log_policy_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L106) [design / candidate / reviewed]：For a positive policy, the ratio appearing above really is the log score.
- [`CGB.ClosedLoopCredit.missing_score_derivative_counterexample`](../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L114) [witness / candidate / reviewed]：Singleton action/outcome, π=K=1, Rθ=θ: both laws are fixed and normalized; the policy-only estimator is zero while the actual objective derivative is one.
- [`CGB.ClosedLoopCredit.constant_likelihood_posterior`](../../lean/CGBAxioms/CGBAxioms/GenerationClosedLoopCredit.lean#L135) [semantics / universal / reviewed]：Constant positive likelihood leaves a normalized prior unchanged.

## GenerationColumnMemory（21）
_Column memory for the language organism (generation chain, 2026-10-28) The language organism's synaptic page is a set of columns `S_m` written by the gated delta rule (`GenerationLinearMemory`), with the activity page re_

- [`CGB.Generation.dot_self_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L43) [semantics / universal / reviewed]
- [`CGB.Generation.eq_zero_of_dot_self`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L46) [semantics / universal / reviewed]
- [`CGB.Generation.silent_rest_unique`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L53) [necessity / candidate / reviewed]：**Silence has one rest**: a gated delta step without a write (`α < 1`) fixes only zero.
- [`CGB.Generation.tiedWrite_eq_delta_add`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L68) [design / candidate / reviewed]：The tied write is the delta step plus the write of the arrival.
- [`CGB.Generation.tied_write_is_error_descent`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L76) [design / candidate / reviewed]：**Error descent**: the tied write is a gradient step of size `β` on `½ (⟨k, x⟩ - v)²`, whose gradient in `x i` is `(⟨k, x⟩ - v) k i` (predicted minus arrived, times the presynaptic key).
- [`CGB.Generation.tied_write_error_shrinks`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L83) [design / candidate / reviewed]：With a unit key the prediction error shrinks by `1 - β`.
- [`CGB.Generation.binary_delta_crossEntropy_decreases`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L104) [design / candidate / reviewed]
- [`CGB.Generation.tied_write_cross_read`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L168) [design / candidate / reviewed]：The change in any other readout is exactly the prediction error times the key overlap and write gain.
- [`CGB.Generation.reciprocal_shared_update_mismatch`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L203) [design / candidate / reviewed]：A matched reciprocal update contracts its splice mismatch by its release factor.
- [`CGB.Generation.reciprocal_iter_mismatch`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L219) [design / candidate / reviewed]：The mismatch is geometric even when the common local statistic changes at each event.
- [`CGB.Generation.reused_key_tracks_and_rewrites`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L239) [design / candidate / reviewed]：Every write recovers the latest value.
- [`CGB.Generation.exact_prediction_no_write`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L256) [design / candidate / reviewed]：An exact prediction writes nothing.
- [`CGB.Generation.occurrence_decay_retains`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L263) [design / candidate / reviewed]：**Decay by use keeps rare topics**: decaying by `α ^ w t` with `w t ∈ [0, 1]` keeps at least what decaying by `α` every tick keeps.
- [`CGB.Generation.unused_column_kept`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L273) [side / universal / reviewed]：An unused column keeps everything.
- [`CGB.Generation.no_latch_below_unit_gain`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L278) [necessity / universal / reviewed]：**No latch below unit gain**: `x ↦ σ (g x)` with a 1-Lipschitz `σ`, `σ 0 = 0` and `0 ≤ g < 1` rests only at zero.
- [`CGB.Generation.hardSat_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L287) [semantics / universal / reviewed]
- [`CGB.Generation.hardSat_lipschitz`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L289) [semantics / universal / reviewed]
- [`CGB.Generation.latch_above_unit_gain`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L293) [design / candidate / reviewed]：**A latch above unit gain**: with saturation and `g ≥ 1`, `± 1` rest besides `0`.
- [`CGB.Generation.self_listening_increment`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L311) [design / candidate / reviewed]：Exact two-trajectory increment, including the feedback contribution.
- [`CGB.Generation.self_listening_frozen_state_gap`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L318) [side / universal / reviewed]：Freezing the self-listening input loses the feedback part of state credit.
- [`CGB.Generation.self_listening_frozen_state_counterexample`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnMemory.lean#L324) [witness / candidate / reviewed]：A concrete nonzero feedback path makes the frozen-input credit incorrect.

## GenerationColumnRefinement（5）
_Column refinement: one body, several readable projections This file does not construct a column organism._

- [`CGB.Generation.LocalColumnProjections.activity_unchanged_outside`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L83) [design / candidate / reviewed]：P1b makes a nonparticipant's activity readout unchanged under the same physical event.
- [`CGB.Generation.LocalColumnProjections.carrier_unchanged_outside`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L89) [design / candidate / reviewed]：The same event leaves a nonparticipant's local causal reading unchanged.
- [`CGB.Generation.LocalColumnProjections.contact_unchanged_outside`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L95) [design / candidate / reviewed]：A contact reading changes only when at least one endpoint participates.
- [`CGB.Generation.BodyCausalCarrier.satisfies_causal_carrier_contract`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L111) [design / candidate / reviewed]：The carrier fields meet the existing exact readout-and-successor contract.
- [`CGB.Generation.BodyCausalCarrier.merged_history_preserves_read_and_successor`](../../lean/CGBAxioms/CGBAxioms/GenerationColumnRefinement.lean#L120) [design / candidate / reviewed]：Equivalently, any two histories merged by the carrier agree on every declared readout and on the carrier successor for each arrival.

## GenerationConservedMemory（24）
_Memory carried by a conserved observable, not a frozen body Generalize the existing quiet-identity memory marginal._

- [`CGB.ConservedMemory.observable_test_preserved`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L29) [design / candidate / reviewed]：Any test of a preserved observable has constant conditional expectation under the specified stimulus.
- [`CGB.ConservedMemory.observable_indicator_preserved`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L46) [design / candidate / reviewed]：A stochastic kernel preserving an observable also preserves each of its indicator test functions.
- [`CGB.ConservedMemory.quiet_observable_memory_marginal`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L64) [design / candidate / reviewed]：Quiet can be nontrivial: each future observable marginal equals the one installed by the actual first observation, even as the world evolves.
- [`CGB.ConservedMemory.tagged_memory_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L90) [design / candidate / reviewed]：The tagged law used by the original memory definition is nonnegative under the actual clock admission, including the observation and quiet steps.
- [`CGB.ConservedMemory.one_write_body_marginal`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L106) [design / candidate / reviewed]：The first actual write marginalizes the world's normalized successor law; this is not a hand-assigned probability on a supported path.
- [`CGB.ConservedMemory.registered_event_kernel_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L117) [design / candidate / reviewed]：A positive registered event that changes the body has a positive actual uniformized transition, at the same world clock.
- [`CGB.ConservedMemory.observed_memory_as_project`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L142) [design / candidate / reviewed]：The observable law is exactly the deterministic projection used by finite conditional-information data processing.
- [`CGB.ConservedMemory.observed_memory_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L152) [design / candidate / reviewed]
- [`CGB.ConservedMemory.one_write_observed_row`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L162) [design / candidate / reviewed]：A first-write coarse row is its actual prefix mass times the observed transition probability.
- [`CGB.ConservedMemory.one_write_observed_mass`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L177) [design / candidate / reviewed]：Summing the observed outcomes returns the same actual prefix mass.
- [`CGB.ConservedMemory.observed_write_pos_of_transition`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L189) [design / candidate / reviewed]：A supported actual transition from a positive prefix gives positive observed write mass, with no assigned path weights.
- [`CGB.ConservedMemory.observed_memory_information_le`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L208) [necessity / candidate / reviewed]：Coarse information is a certified lower bound on the original full-body CMI, including zeros in the actual probability support.
- [`CGB.ConservedMemory.remembers_of_conserved_observable`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L227) [design / candidate / reviewed]：A positive actual information write into a conserved finite observable implies the original full-body `Remembers`, without freezing the body or assuming a positive CMI at all later times.
- [`CGB.ConservedMemory.remembers_of_quiet_and_actual_write`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L248) [design / candidate / reviewed]：An actual counterfactual write creates positive information: at the same reachable body, one world input is quiet and another has a positive transition into a different observable class.
- [`CGB.ConservedMemory.event_observable_kernel`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L280) [design / candidate / reviewed]：A local event invariant lifts to the uniformized quiet kernel.
- [`CGB.ConservedMemory.quiet_event_observable_kernel`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L304) [design / candidate / reviewed]：Quiet specialization of the local event invariant.
- [`CGB.ConservedMemory.quiet_total_kernel`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L312) [design / candidate / reviewed]：Conservation is applied only to positive quiet events which really are closed, not silently to arbitrary boundary-driven writes.
- [`CGB.ConservedMemory.quiet_total_memory_marginal`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L321) [design / candidate / reviewed]：P2a protects the total-content memory marginal under positive closed quiet exchange.
- [`CGB.ConservedMemory.seed_calm_event_conserves`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L333) [design / candidate / reviewed]：In the existing four-site physical seed, quiet shuts the two reservoir ports but leaves the hopping dynamics intact.
- [`CGB.ConservedMemory.seed_event_clocked`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L346) [design / candidate / reviewed]：A structural finite clock bound: at most eighteen registered events, each of rate at most one.
- [`CGB.ConservedMemory.seed_calm_has_nontrivial_event`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L358) [witness / universal / reviewed]：The same physical seed really moves at quiet: this is not the old identity-kernel memory shortcut.
- [`CGB.ConservedMemory.seed_quiet_total_memory_marginal`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L370) [design / candidate / reviewed]：Content distinctions written through the original seed's reservoir ports have invariant quiet marginals despite its nontrivial hopping.
- [`CGB.ConservedMemory.historical_persistence_requires_alive_change`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L386) [necessity / candidate / reviewed]：Historical persistence requires calm to change the viability observable, not merely to move some microscopic states.
- [`CGB.ConservedMemory.conserving_calm_needs_organizational_viability`](../../lean/CGBAxioms/CGBAxioms/GenerationConservedMemory.lean#L416) [necessity / candidate / reviewed]：If calm conserves total and viability reads *only* total, the original long-disconnection death requirement is impossible.

## GenerationContactCompetition（18）
_A shared finite contact reservoir `S` is already a bounded, typed local contact state._

- [`CGB.Generation.receipt_rate_in_unit_interval`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L31) [design / candidate / reviewed]：A nonnegative receipt produces a legal exchange rate, even when its raw magnitude is unbounded.
- [`CGB.Generation.shared_contact_pool_total`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L49) [design / candidate / reviewed]：The whole contact bundle follows one two-state exchange.
- [`CGB.Generation.full_shared_pool_blocks_admission`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L77) [design / candidate / reviewed]：Once a local bundle is full, no positive admission can add to any of its contacts.
- [`CGB.Generation.shared_pool_marginal_admission_is_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L89) [design / candidate / reviewed]：For a positive local admission gate, an already more occupied bundle has strictly less remaining marginal capacity.
- [`CGB.Generation.shared_pool_closed_gate_holds`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L100) [design / candidate / reviewed]：When a local evidence gate is closed, the complete bundle holds exactly.
- [`CGB.Generation.shared_contact_pool_vector_constant_release`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L114) [design / candidate / reviewed]：Constant vector release recovers the original interface exactly.
- [`CGB.Generation.shared_contact_pool_vector_total`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L122) [design / candidate / reviewed]
- [`CGB.Generation.shared_contact_pool_vector_conserves_total`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L152) [design / candidate / reviewed]
- [`CGB.Generation.shared_contact_pool_vector_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L161) [design / candidate / reviewed]
- [`CGB.Generation.shared_contact_pool_free_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L179) [design / candidate / reviewed]
- [`CGB.Generation.shared_contact_pool_vector_total_le_one`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L191) [design / candidate / reviewed]
- [`CGB.Generation.full_shared_pool_vector_blocks_admission`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L203) [design / candidate / reviewed]
- [`CGB.Generation.shared_pool_vector_closed_gate_holds`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L212) [design / candidate / reviewed]
- [`CGB.Generation.shared_contact_pool_vector_endpoint_agreement`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L220) [design / candidate / reviewed]：Endpoints applying the same inputs to the same shared state cannot write inconsistent copies of that edge's reservoir.
- [`CGB.Generation.shared_edge_response_gate_symmetric`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L229) [design / candidate / reviewed]
- [`CGB.Generation.shared_edge_response_gate_bounds`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L233) [design / candidate / reviewed]
- [`CGB.Generation.shared_edge_response_gate_closed`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L238) [design / candidate / reviewed]
- [`CGB.Generation.shared_edge_response_gate_one`](../../lean/CGBAxioms/CGBAxioms/GenerationContactCompetition.lean#L241) [design / candidate / reviewed]

## GenerationContactEligibility（5）
_Local contact eligibility is a forward response, not occupancy alone For a column contact `w` with local participation `s`, the next activity is `hardSat (base + w * s)`._

- [`CGB.Generation.hardSat_eq_self`](../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L26) [design / candidate / reviewed]
- [`CGB.Generation.hardSat_eq_one`](../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L31) [design / candidate / reviewed]
- [`CGB.Generation.contact_response_unsaturated_increment`](../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L38) [design / candidate / reviewed]：On one unsaturated response segment, participation is the exact local forward finite-difference factor of a contact-weight perturbation.
- [`CGB.Generation.contact_response_saturated_increment_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L53) [design / candidate / reviewed]：A bounded activity face carries no additional local contact evidence: as long as both responses remain saturated at `+1`, their difference is zero.
- [`CGB.Generation.contact_score_unsaturated_increment`](../../lean/CGBAxioms/CGBAxioms/GenerationContactEligibility.lean#L66) [design / candidate / reviewed]：A score whose local read coefficient is `q` inherits both cases.

## GenerationContactPlasticity（15）
_Compartment tags modulate a current-event spatial pullback The coefficient `c = lambda * eligibility` is fixed during this event's parameter differentiation._

- [`CGB.Generation.ContactPlasticity.linear_contact_return`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L25) [design / candidate / reviewed]：Transposing a finite linear contact read returns exactly the same scalar contraction; the return is not an independently learned feedback matrix.
- [`CGB.Generation.ContactPlasticity.periodic_flux_return`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L39) [design / candidate / reviewed]：Incoming-minus-outgoing periodic flux has the next-minus-local adjoint.
- [`CGB.Generation.ContactPlasticity.two_direction_flux_return`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L53) [design / candidate / reviewed]：Two transport directions add their local adjoints without retaining a transport trajectory or its past pullbacks.
- [`CGB.Generation.ContactPlasticity.tanh_offset_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L66) [design / candidate / reviewed]：The actual bounded transport law has the exact local factor used by the runtime: an infinitesimal change of a pre-flow changes `tanh flow` by `1 - tanh(flow)^2`.
- [`CGB.Generation.ContactPlasticity.allocation_normalization_cancels`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L95) [design / candidate / reviewed]：Positive total drive cancels between admission and normalized target.
- [`CGB.Generation.ContactPlasticity.normalized_actual_pool`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L105) [design / candidate / reviewed]：The cancellation holds inside the actual shared reservoir, with its old occupation and coordinatewise release unchanged.
- [`CGB.Generation.ContactPlasticity.sigmoid_runtime_formula`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L128) [design / candidate / reviewed]：Mathlib's sigmoid is exactly the runtime logistic expression.
- [`CGB.Generation.ContactPlasticity.endpoint_allocation_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L134) [design / candidate / reviewed]：Exact allocation response to one endpoint's local pre-weight perturbation; `pre` is fixed and the sigmoid supplies the current post-response factor.
- [`CGB.Generation.ContactPlasticity.endpoint_retention_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L145) [design / candidate / reviewed]：Exact mean-retention response at one perturbed endpoint.
- [`CGB.Generation.ContactPlasticity.local_potential_zero_eligibility`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L163) [design / candidate / reviewed]：Vanishing participation tags make the entire parameter-dependent proxy identically zero, not merely its value at a chosen parameter.
- [`CGB.Generation.ContactPlasticity.local_potential_zero_return`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L169) [design / candidate / reviewed]：A zero current local return also makes the stopped-coefficient proxy zero.
- [`CGB.Generation.ContactPlasticity.local_potential_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L177) [design / candidate / reviewed]：Exact current-event derivative through the actual inflow and coordinatewise release.
- [`CGB.Generation.ContactPlasticity.allocation_pool_pre_post_coefficient`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L199) [design / candidate / reviewed]：Actual normalized allocation, allowing target and total admission to vary together.
- [`CGB.Generation.ContactPlasticity.retention_pool_pre_post_coefficient`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L233) [design / candidate / reviewed]：The actual release is A*(1-retention), so a retention logit's response has a positive sign and is contracted with old S across all source coordinates.
- [`CGB.Generation.ContactPlasticity.local_direction_self_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L257) [design / candidate / reviewed]：A parameter direction contracted with itself is nonnegative.

## GenerationContextAssociation（21）
_Context association: what a language memory must write, and where `GDNLife.assimilate` writes an arrival at that arrival's own address and adds its value._

- [`CGB.GDNLife.state_ext`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L34) [design / candidate / reviewed]
- [`CGB.GDNLife.foldl_assimilate_natural`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L46) [design / candidate / reviewed]：Natural parameters after a list of arrivals are the initial array plus one summand per arrival.
- [`CGB.GDNLife.foldl_assimilate_precision`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L58) [design / candidate / reviewed]
- [`CGB.GDNLife.foldl_assimilate_velocity`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L70) [design / candidate / reviewed]
- [`CGB.GDNLife.assimilate_last_fields`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L81) [design / candidate / reviewed]：A positive-confidence arrival replaces both last-arrival fields.
- [`CGB.GDNLife.assimilate_order_blind`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L93) [necessity / candidate / reviewed]：**Order blindness.** Any reordering of the arrivals before a common positive-confidence arrival reaches exactly the same complete state.
- [`CGB.GDNLife.order_blind_same_allocation`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L118) [design / candidate / reviewed]：Order-blind states issue the same byte distribution.
- [`CGB.GDNLife.no_read_separates_order`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L127) [necessity / candidate / reviewed]：**Necessity.** No readout of this state can serve a requirement that separates two orderings.
- [`CGB.Generation.ContextAssociation.foldl_write_natural`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L185) [design / candidate / reviewed]：A cell's natural total is the evidence-weighted sum of the values that followed its contexts.
- [`CGB.Generation.ContextAssociation.foldl_write_precision`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L196) [design / candidate / reviewed]
- [`CGB.Generation.ContextAssociation.cell_mean_is_conditional_statistic`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L209) [design / candidate / reviewed]：**Conditional statistic.** A cell's read is the precision-weighted mean of what followed its contexts, combined with its prior.
- [`CGB.Generation.ContextAssociation.hard_cell_innovation`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L217) [design / candidate / reviewed]：A hard-addressed write is the Bayesian innovation step at that cell.
- [`CGB.Generation.ContextAssociation.write_outside_support`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L232) [design / candidate / reviewed]：Writes leave every cell outside the writing context's support unchanged.
- [`CGB.Generation.ContextAssociation.disjoint_context_read_unchanged`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L239) [design / candidate / reviewed]：**No interference.** A context whose support is disjoint from the writer reads exactly what it read before.
- [`CGB.Generation.ContextAssociation.context_write_separates_order`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L252) [design / candidate / reviewed]：**Order separation.** Unlike `assimilate_order_blind`, the context write holds the order of two arrivals whenever the cell of one successor context sees the value of the other arrival.
- [`CGB.Generation.ContextAssociation.same_address_same_read`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L271) [design / candidate / reviewed]：**Address must refine what the future separates.** Two contexts sharing an address share one prediction, and a common prediction pays at least this squared excess across two different required means.
- [`CGB.Generation.ContextAssociation.common_prediction_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L275) [side / universal / reviewed]
- [`CGB.Generation.ContextAssociation.common_suffix_distance`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L281) [design / candidate / reviewed]：**Fading context.** If each arrival contracts distance, a common suffix of arrivals shrinks the difference geometrically, whatever the older history.
- [`CGB.Generation.ContextAssociation.common_suffix_address_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L298) [design / candidate / reviewed]：**Repeatable address.** A Lipschitz address of a fading context returns to within this bound after a common suffix.
- [`CGB.Generation.ContextAssociation.held_coordinate_survives_suffix`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L314) [design / candidate / reviewed]：**A held coordinate never fades.** If arrivals leave a coordinate untouched, two contexts differing there still differ after every common suffix, so an address reading it is not a function of recent language.
- [`CGB.Generation.ContextAssociation.contraction_has_no_two_latches`](../../lean/CGBAxioms/CGBAxioms/GenerationContextAssociation.lean#L332) [design / candidate / reviewed]：**Separation of roles.** A contraction under a held arrival has at most one rest point, so an address-forming state cannot also hold two latched alternatives.

## GenerationContextBody（24）
_The context-cell language body as a Medium `ConstraintFreedom.arbitrary_driven_tables_admitted` shows that one exposed site satisfies every clause with an arbitrary table, so a Medium instance says nothing about a body u_

- [`CGB.Generation.ContextBody.sense_mem`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L142) [design / candidate / draft]
- [`CGB.Generation.ContextBody.context_mem`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L143) [design / candidate / draft]
- [`CGB.Generation.ContextBody.pool_mem`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L144) [design / candidate / draft]
- [`CGB.Generation.ContextBody.cell_mem`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L145) [design / candidate / draft]
- [`CGB.Generation.ContextBody.parts_subset`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L148) [design / candidate / draft]
- [`CGB.Generation.ContextBody.sense_part`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L157) [design / candidate / draft]
- [`CGB.Generation.ContextBody.context_part`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L162) [design / candidate / draft]
- [`CGB.Generation.ContextBody.body_rigid`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L167) [design / candidate / draft]
- [`CGB.Generation.ContextBody.write_outside`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L171) [design / candidate / draft]
- [`CGB.Generation.ContextBody.write_congr`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L190) [design / candidate / draft]
- [`CGB.Generation.ContextBody.body_p1b`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L213) [design / candidate / draft]
- [`CGB.Generation.ContextBody.body_p2a`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L237) [design / candidate / draft]
- [`CGB.Generation.ContextBody.calm_inactive`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L241) [design / candidate / draft]
- [`CGB.Generation.ContextBody.body_p2b`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L246) [design / candidate / draft]
- [`CGB.Generation.ContextBody.body_p3`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L259) [design / candidate / draft]
- [`CGB.Generation.ContextBody.body_p1a`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L270) [design / candidate / draft]
- [`CGB.Generation.ContextBody.body_axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L277) [design / candidate / draft]：**O01.** The body with finite unit states satisfies every clause.
- [`CGB.Generation.ContextBody.events_touch_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L285) [design / candidate / draft]：**O22.** An event touches at most `B + 3` units, whatever the cell count, and with more cells than the address reaches, some unit is left untouched.
- [`CGB.Generation.ContextBody.modulation_at_boundary`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L314) [design / candidate / draft]：**O25.** Plasticity that depends on the stimulus sits at the boundary (G9, `modulation_needs_boundary`).
- [`CGB.Generation.ContextBody.step_active`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L346) [design / candidate / draft]
- [`CGB.Generation.ContextBody.order_separated`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L357) [design / candidate / draft]：**O07.** The actual write holds order.
- [`CGB.Generation.ContextBody.card_site`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L436) [witness / candidate / draft]
- [`CGB.Generation.ContextBody.suffixOne_is_local`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L453) [witness / candidate / draft]：The premises of `events_touch_bounded` are met by a real instance.
- [`CGB.Generation.ContextBody.suffixOne_holds_order`](../../lean/CGBAxioms/CGBAxioms/GenerationContextBody.lean#L459) [witness / candidate / draft]：The premises of `order_separated` are met from birth by a real instance.

## GenerationContinualOptimizer（42）
_A continual optimizer: competition, no winner-take-all, a healthy latch (side branch, 2026-10-27) The optimizer of a stream learner must (i) make units compete so that they differentiate, (ii) keep one unit from taking e_

- [`CGB.Continual.polar_conserves_direction_mass`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L54) [design / candidate / reviewed]：Orthonormal columns carry unit mass along every input direction.
- [`CGB.Continual.payoff_le_two_of_col`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L71) [design / candidate / reviewed]
- [`CGB.Continual.payoff_le_two_of_row`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L80) [design / candidate / reviewed]
- [`CGB.Continual.column_budget_allows_wta`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L89) [witness / candidate / reviewed]
- [`CGB.Continual.row_budget_allows_copies`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L97) [design / candidate / reviewed]
- [`CGB.Continual.doubly_budgeted_form`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L105) [design / candidate / reviewed]
- [`CGB.Continual.doubly_budgeted_max_iff_permutation`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L110) [design / candidate / reviewed]
- [`CGB.Continual.doubly_budgeted_max_differentiates`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L125) [design / candidate / reviewed]：At a doubly budgeted maximum the two rows own different directions.
- [`CGB.Continual.linear_protect_implies_slow`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L133) [design / candidate / reviewed]
- [`CGB.Continual.deferLatch_holds`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L145) [design / candidate / reviewed]
- [`CGB.Continual.deferLatch_releases`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L148) [design / candidate / reviewed]
- [`CGB.Continual.deferLatch_senses`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L153) [design / candidate / reviewed]：The latch is not blind: the same held state gives different outputs for different evidence runs (H2 is avoided), and every hold ends after `c` contrary steps (H1).
- [`CGB.Continual.held_budget_starves`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L161) [side / universal / reviewed]：Consolidated entries that keep a share `s` of a conserved budget `B` and are never released leave `B - n s` free after `n` consolidations; it is gone once `n s ≥ B`.
- [`CGB.Continual.secant_own_noise_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L174) [design / candidate / reviewed]
- [`CGB.Continual.secant_same_batch_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L180) [design / candidate / reviewed]
- [`CGB.Continual.fresh_decrease_mixture`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L204) [side / universal / reviewed]
- [`CGB.Continual.conditional_prediction_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L207) [side / universal / reviewed]
- [`CGB.Continual.shared_reference_flips`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L211) [design / candidate / reviewed]：Shared reference: row `x` occurs when `x ≥ f · max x y`.
- [`CGB.Continual.expected_gradient_is_mixture`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L241) [side / universal / reviewed]
- [`CGB.Continual.positive_part_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L246) [design / candidate / reviewed]：Zero signal, calibrated variance: `m` is `0, 0, c, -c` with `c^2 = 2P`, so `E[m^2] = P`.
- [`CGB.Continual.consistent_prediction_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L253) [side / universal / reviewed]
- [`CGB.Continual.softmax_gradients_sum_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L259) [design / candidate / reviewed]：Two-class softmax: the class-row gradients `(p - y)` and `((1 - p) - (1 - y))` sum to zero.
- [`CGB.Continual.softmax_rows_coupled`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L263) [design / candidate / reviewed]：The output factor of the softmax curvature, `diag p - p pᵀ` for two classes, has the off-diagonal entry `-p (1 - p)`, nonzero for `0 < p < 1`.
- [`CGB.Continual.ema_forgets_rare_spikes`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L290) [side / universal / reviewed]：The registered window (factor 0.95) with spikes 200 steps apart: the held share 0.95^200 is below the spike's share 1/200 of the long-run power.
- [`CGB.Continual.partition_is_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L292) [design / candidate / reviewed]
- [`CGB.Continual.event_part_is_rate_times_size`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L299) [side / universal / reviewed]
- [`CGB.Continual.frozen_hold_loses_on_drift`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L327) [necessity / world_contract / reviewed]
- [`CGB.Continual.mixture_update_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L333) [design / candidate / reviewed]
- [`CGB.Continual.mixture_weight_at_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L343) [design / candidate / reviewed]
- [`CGB.Continual.release_threshold_monotone_in_age`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L346) [design / candidate / reviewed]
- [`CGB.Continual.succession_odds_decreasing`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L352) [design / candidate / reviewed]
- [`CGB.Continual.linear_burst_damage_integral`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L374) [design / candidate / reviewed]
- [`CGB.Continual.beta_hazard_decreasing`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L410) [side / universal / reviewed]
- [`CGB.Continual.beta_hazard_succession`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L417) [side / universal / reviewed]
- [`CGB.Continual.accepted_run_variance`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L420) [side / universal / reviewed]
- [`CGB.Continual.posterior_shrink_error`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L440) [design / candidate / reviewed]：Posterior of the gradient as the two points `m ± √P` (mean `m`, variance `P`): the mean squared error of the action `f m` is `P + (1 - f)^2 m^2`.
- [`CGB.Continual.posterior_mean_is_bayes_action`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L445) [design / candidate / reviewed]
- [`CGB.Continual.newton_contracts_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L469) [design / candidate / reviewed]
- [`CGB.Continual.overestimate_keeps_contraction`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L484) [design / candidate / reviewed]
- [`CGB.Continual.saturating_step_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L499) [design / candidate / reviewed]
- [`CGB.Continual.pooled_signal_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L518) [design / candidate / reviewed]
- [`CGB.Continual.next_value_variance`](../../lean/CGBAxioms/CGBAxioms/GenerationContinualOptimizer.lean#L523) [design / candidate / reviewed]

## GenerationCountedPopulation（30）
_Reinvestment by counted units (language organism, 2026-11-03) `GenerationReinvestment` treats the shares as real numbers._

- [`CGB.CountedPopulation.moran_expected_change`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L51) [design / candidate / draft]：One birth in proportion to reinvestment and one recycling in proportion to share: the mean change of the share of `i` is the reinvestment step.
- [`CGB.CountedPopulation.exchange_moves_a_unit`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L66) [design / candidate / draft]：Counts of a fixed total change only by moving at least one whole unit.
- [`CGB.CountedPopulation.extinct_stays_extinct`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L81) [design / candidate / draft]：An address at zero stays at zero.
- [`CGB.CountedPopulation.concentrated_absorbing`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L85) [design / candidate / draft]：A population on one address never leaves it.
- [`CGB.CountedPopulation.reinvest_ratio`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L102) [design / candidate / draft]：The ratio of two shares moves by the ratio of the bets.
- [`CGB.CountedPopulation.ratio_after_le`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L118) [design / candidate / draft]：**Lock-in**: when each byte pays `i` at most `ρ` times what it pays `j`, the ratio of `i` to `j` falls by at least `ρ` per byte, however good `i` was before.
- [`CGB.CountedPopulation.mix_total`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L143) [design / candidate / draft]
- [`CGB.CountedPopulation.mix_floor`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L153) [design / candidate / draft]：Every address keeps a floor after one transport.
- [`CGB.CountedPopulation.mix_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L159) [design / candidate / draft]
- [`CGB.CountedPopulation.mixWealth_ge_address`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L170) [design / candidate / draft]：Wealth with transport tracks any one address, paying `1 - ε` per byte.
- [`CGB.CountedPopulation.recovers_from_floor`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L209) [design / candidate / draft]：**Recovery after any history**: from any state with the floor, the wealth over a later stretch is at least the floor times what the address earns there, minus the transport toll.
- [`CGB.CountedPopulation.price_equation`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L224) [design / candidate / draft]：**Price equation**: the mean of any carried trait changes by the covariance of the trait with the multiplier minus one.
- [`CGB.CountedPopulation.groupGain_sum`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L248) [design / candidate / draft]
- [`CGB.CountedPopulation.coarse_predicted`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L254) [design / candidate / draft]：Groups predict as addresses do.
- [`CGB.CountedPopulation.coarse_reinvest`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L262) [design / candidate / draft]：**The same law at the next scale**: reinvestment of the group total is reinvestment of the group as one address.
- [`CGB.CountedPopulation.local_keeps_group_total`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L281) [design / candidate / draft]：**A compartment keeps its total**: the sum of a group after a local exchange is its sum before, whatever the bets are.
- [`CGB.CountedPopulation.local_ignores_outside`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L297) [design / candidate / draft]：**A compartment reads only its own bets**: changing bets outside the group changes nothing inside it.
- [`CGB.CountedPopulation.step_smul`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L320) [design / candidate / draft]
- [`CGB.CountedPopulation.lik_smul`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L326) [design / candidate / draft]
- [`CGB.CountedPopulation.fwd_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L346) [design / candidate / draft]
- [`CGB.CountedPopulation.fwd_total`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L353) [design / candidate / draft]
- [`CGB.CountedPopulation.predicted_pos_of_total`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L364) [design / candidate / draft]
- [`CGB.CountedPopulation.tag_fraction_eq_posterior`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L379) [design / candidate / draft]：**Delayed credit without a backward pass**: the tagged fraction of the mass that survives a stream equals the likelihood of that stream through the tag over its likelihood overall, which is the posterior of the tag given
- [`CGB.CountedPopulation.step_add`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L394) [design / candidate / draft]
- [`CGB.CountedPopulation.lik_add`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L400) [design / candidate / draft]
- [`CGB.CountedPopulation.lik_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L411) [design / candidate / draft]
- [`CGB.CountedPopulation.tag_is_exact_credit`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L428) [design / candidate / draft]：**The tag is the exact credit**: the derivative of the log likelihood of every later byte with respect to mass added at an earlier event equals the tag's fraction of the surviving mass.
- [`CGB.CountedPopulation.tagReinvest_total_bracket`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L456) [design / candidate / draft]
- [`CGB.CountedPopulation.bracket_step`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L479) [design / candidate / draft]：**One step of forgetting**: uniform transport pulls the bracket of a tag toward its mean, by `ε / n` of the gap at every byte.
- [`CGB.CountedPopulation.bracket_after`](../../lean/CGBAxioms/CGBAxioms/GenerationCountedPopulation.lean#L541) [design / candidate / draft]：**The credit horizon**: after `L` bytes the tag is the same fraction of the mass at every address, up to `(1 - ε / n) ^ L` of the original gap.

## GenerationCRN（10）
_Well-mixed pair reaction media are admitted (generation chain, M2) One compartment (every unit touches every other: a single clique, which by G8 can integrate anything)._

- [`CGB.PairCRN.sum_pairAct`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L96) [semantics / candidate / reviewed]：A pair event changes a unit-wise sum only at its two units.
- [`CGB.PairCRN.totalE_pairAct`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L117) [semantics / candidate / reviewed]
- [`CGB.PairCRN.p1a`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L125) [semantics / candidate / reviewed]
- [`CGB.PairCRN.p1b`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L128) [semantics / candidate / reviewed]
- [`CGB.PairCRN.p2a`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L152) [semantics / candidate / reviewed]
- [`CGB.PairCRN.p2b`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L176) [semantics / candidate / reviewed]
- [`CGB.PairCRN.p2c`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L213) [semantics / candidate / reviewed]
- [`CGB.PairCRN.p3`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L236) [semantics / candidate / reviewed]
- [`CGB.PairCRN.homogeneous`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L256) [semantics / candidate / reviewed]：The single compartment's contact is homogeneous (an optional model property).
- [`CGB.PairCRN.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationCRN.lean#L263) [semantics / candidate / reviewed]：**M2**: a well-mixed pair reaction medium is a model of the CGB axioms as soon as its finite rule table passes the local obligations: content levels (P1a), conservation of closed rules (P2a), reversible rules with symmet

## GenerationDivisionTree（7）
_What division alone gives a tree (language organism, 2026-11-03) A unit divides into two._

- [`CGB.DivisionTree.Lineage.leaves_eq_splits_succ`](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L45) [design / candidate / draft]
- [`CGB.DivisionTree.Lineage.leaves_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L50) [design / candidate / draft]
- [`CGB.DivisionTree.Lineage.leaves_le_two_pow_depth`](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L53) [design / candidate / draft]
- [`CGB.DivisionTree.Lineage.clog_leaves_le_depth`](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L65) [design / candidate / draft]：Reading `k` parts through two-part units takes at least `⌈log₂ k⌉` levels.
- [`CGB.DivisionTree.Lineage.exists_tree_of_clog_depth`](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L69) [design / candidate / draft]：The depth bound is reached for every number of parts.
- [`CGB.DivisionTree.sharedPrefix_ultrametric`](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L94) [design / candidate / draft]：**Nested groups**: sharing `m` divisions is transitive, so each level of the tree partitions the units.
- [`CGB.DivisionTree.sharedPrefix_comm`](../../lean/CGBAxioms/CGBAxioms/GenerationDivisionTree.lean#L115) [design / candidate / draft]

## GenerationDynamicTeaching（2）
_Dynamic boundary teaching has a necessary cross term `GenerationContactPlasticity` proves the current spatial pullback while its returned teaching coefficient is stopped._

- [`CGB.Generation.DynamicTeaching.teaching_response_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationDynamicTeaching.lean#L24) [design / candidate / reviewed]：A boundary teaching coefficient times a contact response.
- [`CGB.Generation.DynamicTeaching.stopped_teaching_complete_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationDynamicTeaching.lean#L35) [design / candidate / reviewed]：The stopped-teaching spatial term is complete exactly when its omitted teaching-motion product vanishes.

## GenerationExport（6）
_Export and selective export (generation chain, G16-G17) P1a makes the body finite; P2a gives the content ledger `content (t+1) + out t = content t + in t` (`MaterialLedger.balance`)._

- [`CGB.Generation.export_forced`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L25) [design / candidate / reviewed]：**G16a (export is forced)**: with the content ledger and a capacity bound, cumulative export is at least cumulative intake minus the capacity.
- [`CGB.Generation.write_only_intake_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L42) [design / candidate / reviewed]：**G16b (write-only bodies starve)**: with no export, total lifetime intake is at most the capacity.
- [`CGB.Generation.endless_intake_endless_export`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L50) [design / candidate / reviewed]：**G16c**: if intake does not stop (its total exceeds every bound), export does not stop either.
- [`CGB.Generation.survive_le_pow`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L67) [design / candidate / reviewed]：**G17a (unselective export forgets)**: if every export removes the item with probability at least `q₀ > 0` (for example `1/C` when the exported unit is drawn without regard to what it holds), survival is at most `(1 - q₀
- [`CGB.Generation.unselective_export_forgets`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L79) [design / candidate / reviewed]
- [`CGB.Generation.retention_needs_selectivity`](../../lean/CGBAxioms/CGBAxioms/GenerationExport.lean#L90) [design / candidate / reviewed]：**G17b (retention needs selectivity)**: if an item must survive every number of exports with probability at least `p > 0`, its per-export removal probability has no positive lower bound: export must eventually spare it,

## GenerationFutureValue（4）
_Future-paid language distinctions A finite body cannot retain every distinction that *might* be queried._

- [`CGB.GDNLife.Process.future_cross_entropy_congr_on_positive_support`](../../lean/CGBAxioms/CGBAxioms/GenerationFutureValue.lean#L52) [design / candidate / reviewed]：A continuation of zero world-side recurrence has no effect on the future code-length audit.
- [`CGB.GDNLife.Process.positive_continuation_improvement_is_future_paid`](../../lean/CGBAxioms/CGBAxioms/GenerationFutureValue.lean#L71) [design / candidate / reviewed]：If a body change never worsens any declared future continuation and strictly improves one continuation that recurs with positive mass, it strictly improves expected future language code length.
- [`CGB.GDNLife.Process.empirical_future_cross_entropy_congr`](../../lean/CGBAxioms/CGBAxioms/GenerationFutureValue.lean#L109) [design / candidate / reviewed]：If no actual re-entered continuation can tell two body states apart in language code length, the empirical future-value audit is equal.
- [`CGB.GDNLife.Process.observed_continuation_improvement_is_empirically_paid`](../../lean/CGBAxioms/CGBAxioms/GenerationFutureValue.lean#L126) [design / candidate / reviewed]：One strictly better re-entered continuation, with no worse observed continuation, strictly improves the empirical future language code length.

## GenerationGDNLife（80）
_Real-valued GDN carrier on one open trajectory The body is a fixed-dimensional real state, not a finite enumeration._

- [`CGB.GDNLife.address_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L73) [design / candidate / reviewed]
- [`CGB.GDNLife.action_head_score_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L107) [design / candidate / reviewed]：The existing G14 log-softmax law instantiated at the GDN language head.
- [`CGB.GDNLife.addressLogitPerturb_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L187) [design / candidate / reviewed]：Exact softmax derivative for an address-logit basis coordinate.
- [`CGB.GDNLife.assimilate_addressLogitPerturb_natural_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L211) [design / candidate / reviewed]：The exact address-side contribution to a GDN natural-parameter write.
- [`CGB.GDNLife.assimilate_addressLogitPerturb_precision_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L235) [design / candidate / reviewed]：The same address perturbation changes precision by the address-side incoming derivative.
- [`CGB.GDNLife.likelihoodSoftplus_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L262) [design / candidate / reviewed]：Exact derivative of the actual softplus confidence parametrization.
- [`CGB.GDNLife.likelihood_posterior_mean_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L301) [design / candidate / reviewed]：Exact local posterior sensitivity for the learned likelihood logit.
- [`CGB.GDNLife.assimilate_likelihoodLogitPerturb_natural_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L353) [design / candidate / reviewed]：The likelihood side of the current GDN write: a learned confidence logit changes the natural parameter by its softplus slope times the existing address, observation precision and value.
- [`CGB.GDNLife.assimilate_likelihoodLogitPerturb_precision_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L370) [design / candidate / reviewed]：The same learned confidence changes the stored precision.
- [`CGB.GDNLife.assimilate_likelihoodLogitPerturb_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L389) [design / candidate / reviewed]：When the raw logit represents the current confidence, the perturbation passes through the actual arrival transition at zero exactly.
- [`CGB.GDNLife.prepare_valuePerturb`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L397) [design / candidate / reviewed]：Altering a byte value cannot alter the prediction prepared before that byte arrives.
- [`CGB.GDNLife.incoming_valuePerturb`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L404) [design / candidate / reviewed]：Changing the byte encoder leaves the arrival weight unchanged: it does not silently alter the address or likelihood path.
- [`CGB.GDNLife.assimilate_valuePerturb_natural`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L413) [design / candidate / reviewed]：Exact one-arrival finite difference for a byte encoder coordinate.
- [`CGB.GDNLife.assimilate_valuePerturb_precision`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L429) [design / candidate / reviewed]：The same perturbation leaves precision unchanged.
- [`CGB.GDNLife.assimilate_valuePerturb_difference`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L438) [design / candidate / reviewed]：Comparing two encoder coordinates gives the exact finite difference, without invoking reverse-time differentiation.
- [`CGB.GDNLife.valueEligibility_zero_of_no_incoming`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L448) [design / candidate / reviewed]：No arriving flow means no encoder-value eligibility at that address.
- [`CGB.GDNLife.value_sensitivity_two_arrivals`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L456) [design / candidate / reviewed]：The value-side sensitivity is carried by the same natural state across a second arrival.
- [`CGB.GDNLife.initial_positive`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L485) [design / candidate / reviewed]
- [`CGB.GDNLife.prior_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L489) [design / candidate / reviewed]
- [`CGB.GDNLife.prior_natural_difference`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L502) [design / candidate / reviewed]：If a perturbation leaves a slot's precision unchanged, the prior transports its natural-state difference by `priorNaturalScale`; no past input is reread.
- [`CGB.GDNLife.value_sensitivity_through_prior`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L523) [design / candidate / reviewed]：A value-write sensitivity remains in the same state after the time prior.
- [`CGB.GDNLife.feature_difference_of_shared_query`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L543) [design / candidate / reviewed]：With unchanged precision and causal query, an existing natural-state difference is read out linearly by the same GDN feature map.
- [`CGB.GDNLife.feature_valuePerturb`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L560) [design / candidate / reviewed]：A perturbation of the byte-value encoder has no direct readout path.
- [`CGB.GDNLife.assimilate_lastByte_of_positive`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L567) [design / candidate / reviewed]：A positive-confidence arrival makes that byte the next causal query.
- [`CGB.GDNLife.value_read_eligibility_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L585) [design / candidate / reviewed]：One byte-value encoder direction reaches the next causal readout exactly through `valueReadEligibility`.
- [`CGB.GDNLife.value_future_read_eligibility_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L638) [design / candidate / reviewed]：The exact encoder-value eligibility at the next prepared readout.
- [`CGB.GDNLife.assimilate_head`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L709) [design / candidate / reviewed]：Assimilation writes only the material memory fields and causal observation cache; it does not create a second copy of the language readout parameters.
- [`CGB.GDNLife.assimilate_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L712) [design / candidate / reviewed]
- [`CGB.GDNLife.prior_head`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L716) [design / candidate / reviewed]：The time prior transports memory but retains the same head and bias.
- [`CGB.GDNLife.prior_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L720) [design / candidate / reviewed]
- [`CGB.GDNLife.value_future_logit_eligibility_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L735) [design / candidate / reviewed]：The next byte-logit displacement is exactly the forward eligibility read through the unchanged affine head.
- [`CGB.GDNLife.byte_log_softmax_direction_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L772) [design / candidate / reviewed]：G14 for a whole finite logit direction.
- [`CGB.GDNLife.value_future_log_likelihood_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L828) [design / candidate / reviewed]：G14 applied to the exact next-byte GDN displacement: the derivative of next-byte log likelihood with respect to an encoder-value direction is the forward eligibility weighted by the observed-minus-predicted language erro
- [`CGB.GDNLife.value_future_teaching_direction_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L865) [design / candidate / reviewed]：An endogenous teaching amplitude need not be a transpose of the forward path.
- [`CGB.GDNLife.positive_teaching_alignment_is_local_ascent`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L883) [design / candidate / reviewed]：Positive alignment with the exact future score is the minimal local ascent admission for a learned teaching signal.
- [`CGB.GDNLife.value_future_score_factor`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L906) [design / candidate / reviewed]：Exact factorization of the encoder-value score.
- [`CGB.GDNLife.address_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L956) [design / candidate / reviewed]
- [`CGB.GDNLife.incoming_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L960) [design / candidate / reviewed]
- [`CGB.GDNLife.prior_positive`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L965) [design / candidate / reviewed]
- [`CGB.GDNLife.assimilate_positive`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L977) [design / candidate / reviewed]
- [`CGB.GDNLife.step_positive`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L982) [design / candidate / reviewed]
- [`CGB.GDNLife.zero_precision_keeps_last_byte`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L988) [design / candidate / reviewed]：Zero-precision input retains an existing last observation, matching `assimilate`'s live-observation mask rather than silently changing query.
- [`CGB.GDNLife.first_observation_sets_last_byte`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L994) [design / candidate / reviewed]：The first input still installs the cache, even at zero precision.
- [`CGB.GDNLife.positive_observation_sets_last_byte`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1000) [design / candidate / reviewed]：Any live arrival writes its byte into the existing causal query field.
- [`CGB.GDNLife.step_positive_observation_sets_last_byte`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1009) [design / candidate / reviewed]：The complete GDN step preserves that live causal write through its local head update.
- [`CGB.GDNLife.distinct_live_bytes_separate_steps`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1019) [design / candidate / reviewed]：Distinct live language emissions produce distinct post-step bodies through an existing causal field.
- [`CGB.GDNLife.step_source_irrelevant`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1032) [design / candidate / reviewed]：Source labels do not invent a speech fee or a different learning rule.
- [`CGB.GDNLife.preparation_uses_only_elapsed`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1036) [design / candidate / reviewed]：Prediction cannot inspect the not-yet-arrived byte or its source.
- [`CGB.GDNLife.head_update_preserves_memory`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1041) [design / candidate / reviewed]：Local head learning does not modify the memory or its uncertainty.
- [`CGB.GDNLife.head_update_logits`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1047) [design / candidate / reviewed]：The actual affine parameter step induces a score step of size `eta * (1 + squared feature norm)`, including the learned bias.
- [`CGB.GDNLife.head_update_other_feature`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1066) [design / candidate / reviewed]：A G14 head update remains a rank-one change when read on a later feature.
- [`CGB.GDNLife.head_update_improves_saved_feature`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1082) [design / candidate / reviewed]：Strict improvement on the saved feature, not a future-language claim.
- [`CGB.GDNLife.carrier_next`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1121) [design / candidate / reviewed]
- [`CGB.GDNLife.carrier_fold`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1129) [design / candidate / reviewed]：Actual input histories fold through this same real body, with no reset.
- [`CGB.GDNLife.exists_open_real_gdn_trajectory`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1154) [witness / candidate / reviewed]：An actual continuous GDN, not a finite surrogate, has one infinite open numerical trajectory.
- [`CGB.GDNLife.finish_saved_logits`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1181) [design / candidate / reviewed]：Assimilation does not change the head.
- [`CGB.GDNLife.byte_softmax_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1191) [design / candidate / reviewed]
- [`CGB.GDNLife.finish_saved_nll_improves`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1198) [design / candidate / reviewed]：The saved-feature current-byte loss strictly decreases at any positive head step.
- [`CGB.GDNLife.future_ce_improves_of_target_mass`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1211) [design / candidate / reviewed]：A distributional, not in-sample, sufficient condition.
- [`CGB.GDNLife.finish_future_ce_improves`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1240) [design / candidate / reviewed]：A concrete head update can improve a declared future law even after the memory/query changed: the saved-feature gain must exceed the readout drift, and the law must be sufficiently persistent.
- [`CGB.GDNLife.real_bias_injective`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1271) [design / candidate / reviewed]：The real family already embeds a real coordinate in an existing head bias.
- [`CGB.GDNLife.prior_logits`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1276) [design / candidate / reviewed]
- [`CGB.GDNLife.step_ne_of_positive_head_step`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1285) [design / candidate / reviewed]：A positive real head step changes the body on every arriving byte.
- [`CGB.GDNLife.Process.run_initial`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1338) [design / candidate / reviewed]
- [`CGB.GDNLife.Process.run_step`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1340) [design / candidate / reviewed]
- [`CGB.GDNLife.Process.run_positive`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1348) [design / candidate / reviewed]
- [`CGB.GDNLife.Process.run_body_step`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1352) [design / candidate / reviewed]
- [`CGB.GDNLife.Process.self_speech_action_contrast_writes_body`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1361) [design / candidate / reviewed]：Two candidate language actions that emit different live self-produced bytes write distinct next bodies through the ordinary GDN boundary transition.
- [`CGB.GDNLife.Process.next_byte_improves`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1377) [design / candidate / reviewed]：Repetition refers to the world's actual next byte, not a byte chosen for rescoring.
- [`CGB.GDNLife.Process.externallyUseful_improves`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1411) [design / candidate / reviewed]：The useful-event certificate invokes the earlier causal result; it does not inspect the next byte while preparing the current prediction.
- [`CGB.GDNLife.Process.self_not_externallyUseful`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1420) [design / candidate / reviewed]：A copied self byte never becomes external evidence just by passing through the same learning rule.
- [`CGB.GDNLife.Process.externalInformation_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1455) [design / candidate / reviewed]：A valid external language law has nonnegative information opportunity.
- [`CGB.GDNLife.Process.ExternalLanguageLaw.information_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1464) [design / candidate / reviewed]：The state/action-indexed opportunity inherits CMI nonnegativity from the registered external language law.
- [`CGB.GDNLife.Process.future_reserve_preserved_keeps_signature_distinction`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1503) [design / candidate / reviewed]：A preserved future-language reserve cannot silently merge two histories that a registered external continuation could previously distinguish.
- [`CGB.GDNLife.Process.future_reserve_recoverable_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1546) [necessity / candidate / reviewed]：A future distinction is not dissipated exactly when all of the prior future language signature remains readable from the post-update signature.
- [`CGB.GDNLife.Process.loses_future_language_distinction_iff_not_preserved`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1561) [necessity / candidate / reviewed]：Merging a prior distinct future-language signature is precisely the failure of that causal retention contract.
- [`CGB.GDNLife.Process.recurrent_external_information_forces_activity`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1584) [necessity / candidate / reviewed]：Information-theoretic active necessity for the concrete GDN port.
- [`CGB.GDNLife.Process.recurrent_external_evidence_forces_activity`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1603) [necessity / candidate / reviewed]：If settling into a passive policy rules out recurrent externally useful language, but such events actually recur on this same trajectory, the policy cannot settle.
- [`CGB.GDNLife.Process.scored_external_language_trajectory`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1621) [witness / candidate / reviewed]：Conditional language-level joint result for the real GDN.
- [`CGB.GDNLife.Process.scored_joint_trajectory`](../../lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean#L1663) [witness / candidate / reviewed]：A conditional scored joint trajectory for this real GDN.

## GenerationGraceMemory（13）
_Historical memory in the unchanged grace process The natural law at tick four has one body state and three world states._

- [`CGB.GraceMemory.sum_cells`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L17) [design / candidate / reviewed]
- [`CGB.GraceMemory.sum_configs`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L24) [design / candidate / reviewed]
- [`CGB.GraceMemory.sum_env`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L32) [design / candidate / reviewed]
- [`CGB.GraceMemory.kernel_table`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L40) [design / candidate / reviewed]：The original body kernel, obtained by forgetting its actual event label.
- [`CGB.GraceMemory.policy_table`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L55) [design / candidate / reviewed]
- [`CGB.GraceMemory.natural_law_early`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L75) [design / candidate / reviewed]：Each step rewrites the preceding sparse row before any finite summation; there is no expansion of a Cartesian product of histories.
- [`CGB.GraceMemory.actual_event_law_four`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L96) [design / candidate / reviewed]：The same row is the law of the registered event process.
- [`CGB.GraceMemory.one_write_marginal`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L110) [design / candidate / reviewed]：Marginalize the actual world transition, whose row sums to one.
- [`CGB.GraceMemory.one_write_joint`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L122) [design / candidate / reviewed]：The three nonzero tagged outcomes have masses 1/4, 1/4 and 1/2.
- [`CGB.GraceMemory.one_write_information`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L139) [design / candidate / reviewed]：Exact conditional mutual information in nats: one bit, not a support argument.
- [`CGB.GraceMemory.remembers`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L155) [design / candidate / reviewed]：The original `Life.Remembers` predicate holds for the unchanged grace body/world.
- [`CGB.GraceMemory.retained_information`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L162) [design / candidate / reviewed]：Quiet retains exactly the same bit at every later delay.
- [`CGB.GraceMemory.remembers_with_reachable_action_change`](../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L177) [design / candidate / reviewed]：Combine the numerical memory witness with the already-proved supported histories whose stored states select different actions.

## GenerationGradientFilter（35）
_An optimizer as a learner of its gradient stream (side branch, 2026-10-27) The outer loop's world is the stream of noisy gradients._

- [`CGB.GradientFilter.memoryless_gain_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L34) [design / candidate / reviewed]：A learner without memory (gain 1) exceeds the optimal error `P` by `r^2 / (P + r)`.
- [`CGB.GradientFilter.memoryless_strictly_worse`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L42) [design / candidate / reviewed]
- [`CGB.GradientFilter.shrink_error_square`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L52) [design / candidate / reviewed]
- [`CGB.GradientFilter.shrink_optimum`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L58) [design / candidate / reviewed]
- [`CGB.GradientFilter.unshrunk_excess`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L64) [design / candidate / reviewed]：Taking the noisy estimate at face value costs exactly `N^2 / (S + N)` more.
- [`CGB.GradientFilter.no_shared_shrink_for_two_snr`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L72) [design / candidate / reviewed]：One shared factor is strictly suboptimal in at least one of two directions whose signal fractions differ.
- [`CGB.GradientFilter.vanishing_gain_loses_under_drift`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L108) [necessity / world_contract / reviewed]：**W2 (drift forbids a vanishing gain)**: the tracking error variance obeys `V (t+1) = (1 - K t)^2 V t + q + K t^2 r`.
- [`CGB.GradientFilter.emaVar_fixed`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L153) [design / candidate / reviewed]：It is the fixed point of `v = x^2 v + (1 - x)^2 r`.
- [`CGB.GradientFilter.occurrence_clock_beats_time_clock`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L163) [design / candidate / reviewed]：**W3 (an occurrence clock)**: a direction observed once every `k ≥ 2` steps, averaged with the per-step factor `β` of the frequent directions, effectively uses `β^k` per observation and has a strictly larger stationary v
- [`CGB.GradientFilter.diagonal_cannot_whiten`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L177) [design / candidate / reviewed]：**W4 (a relation basis)**: if the curvature `[[a, b], [b, d]]` has a nonzero off-diagonal term, no diagonal preconditioner `diag(d₁, d₂)` turns it into a nonzero multiple of the identity (one-step whitening); a basis-awa
- [`CGB.GradientFilter.usage_normalized_gain_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L187) [design / candidate / reviewed]：**W5 is implied by W3**: a gain normalized by that direction's accumulated use (`s / sqrt (c + m)`, the AdaGrad / Adam form) strictly falls when the direction's own demand grows: it is locally anti-dominant without a sep
- [`CGB.GradientFilter.step_error_square`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L215) [design / candidate / reviewed]
- [`CGB.GradientFilter.optimal_step_minimizes`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L222) [design / candidate / reviewed]
- [`CGB.GradientFilter.optimal_step_newton_times_signal`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L227) [design / candidate / reviewed]
- [`CGB.GradientFilter.no_fixed_step_two_distances`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L233) [design / candidate / reviewed]：With noise, the optimal step differs at two different distances to the optimum.
- [`CGB.GradientFilter.no_optimal_step_negative_curvature`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L247) [design / candidate / reviewed]：**W6 (nonconvex data)**: with negative curvature the loss along the step is unbounded below, so no optimal quadratic step exists and the step must be bounded.
- [`CGB.GradientFilter.no_fixed_bound_two_radii`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L292) [design / candidate / reviewed]
- [`CGB.GradientFilter.bound_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L306) [design / candidate / reviewed]
- [`CGB.GradientFilter.newton_step_exact`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L324) [design / candidate / reviewed]：The exact curvature step reaches the optimum of a quadratic direction.
- [`CGB.GradientFilter.momentum_after_exact_step`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L330) [design / candidate / reviewed]：After an exact step, momentum `β ≠ 0` (heavy ball: next error `β (e₁ - e₀)` with `e₁ = 0`) moves a direction that started away from the optimum off it again.
- [`CGB.GradientFilter.gain_error_square`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L354) [design / candidate / reviewed]
- [`CGB.GradientFilter.steady_gain_worse_for_fresh`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L362) [design / candidate / reviewed]：The steady gain `P* / (P* + r)` is strictly worse than the evidence-dependent gain for a direction whose prior variance is different (larger: a fresh or rare direction).
- [`CGB.GradientFilter.in_sample_decrease_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L400) [design / candidate / reviewed]：Decrease measured on the batch that made the step minus the decrease on fresh data.
- [`CGB.GradientFilter.in_sample_decrease_bias_mean`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L405) [design / candidate / reviewed]
- [`CGB.GradientFilter.own_step_biases_noise_estimate`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L412) [design / candidate / reviewed]：Lag-one noise estimate averaged over the noise pair `(x, y)`, `(-x, -y)`.
- [`CGB.GradientFilter.residual_removes_own_step`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L416) [design / candidate / reviewed]
- [`CGB.GradientFilter.secant_needs_diagonal`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L420) [design / candidate / reviewed]
- [`CGB.GradientFilter.secant_exact_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L424) [design / candidate / reviewed]
- [`CGB.GradientFilter.fresh_decrease_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L452) [design / candidate / reviewed]：Fresh-batch decrease averaged over the unseen noise pair `±ξ'`.
- [`CGB.GradientFilter.estimated_curvature_residual_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L458) [design / candidate / reviewed]
- [`CGB.GradientFilter.eigenbasis_decouples`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L463) [design / candidate / reviewed]
- [`CGB.GradientFilter.trust_with_deadband_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L471) [design / candidate / reviewed]
- [`CGB.GradientFilter.linGain_closed`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L495) [design / candidate / reviewed]
- [`CGB.GradientFilter.linear_damage_eq_adoption`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L500) [design / candidate / reviewed]
- [`CGB.GradientFilter.no_linear_gain_separates`](../../lean/CGBAxioms/CGBAxioms/GenerationGradientFilter.lean#L507) [design / candidate / reviewed]

## GenerationGrowth（16）
_Growth of a connection unit (generation chain, G18) A connection unit sits between two compartments and carries its two end activities `a b : Fin 2` and its slot occupancy `w`._

- [`CGB.Generation.binary_pair_decomposition`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L25) [design / candidate / reviewed]：**G18a**: the Möbius decomposition.
- [`CGB.Generation.interaction_unique`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L32) [design / candidate / reviewed]：**G18a (uniqueness)**: if `f` is written with unary parts and a product coefficient `κ`, then `κ` is the interaction coefficient.
- [`CGB.Generation.slot_reciprocity`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L41) [design / candidate / reviewed]：**G18b (reciprocity)**: for an energy `E w x` of a slot `w` and a co-activity `x`, the effect of `x` on the slot's flip equals the effect of the slot on `x`'s flip.
- [`CGB.Generation.occupancy_odds`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L50) [design / candidate / reviewed]：The detailed-balanced build/dismantle ratio fixes the occupancy: `occ / (1 - occ) = exp h`.
- [`CGB.Generation.occupancy_strictMono`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L60) [design / candidate / reviewed]：**G18c**: occupancy is strictly increasing in the field.
- [`CGB.Generation.one_pool_nested`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L77) [design / candidate / reviewed]：**G19a (one pool grows a rich club)**: with one shared pool, the strong set is an upper set in the rates: if `i` is strongly connected to `j` and `k` fires at least as much as `i`, then `k` is too.
- [`CGB.Generation.memoryless_coactivity`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L92) [design / candidate / reviewed]：**G19b (memoryless ends see only symbol frequencies)**: if every end state depends on the current symbol only, the co-activity, hence the grown structure, depends on the stream only through the frequencies of the current
- [`CGB.Generation.memoryless_same_frequencies_same_structure`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L99) [design / candidate / reviewed]
- [`CGB.Generation.memory_separates`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L113) [design / candidate / reviewed]：**G19c (sequence structure needs memory at the ends)**: the two streams have equal frequencies, so memoryless ends grow the same structure (G19b); an end that holds the previous symbol (`i` active on the previous 1, `j`
- [`CGB.Generation.two_state_geometric`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L137) [design / candidate / reviewed]：**G20a (two-state memory is geometric)**: the deviation from the fixed point `p* = kp / (kp + km)` shrinks by the same factor `1 - kp - km` every tick, so the trace of a past input is exponential; detailed balance fixes
- [`CGB.Generation.symmetric_exchange`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L152) [design / candidate / reviewed]：**G20b (symmetric exchange)**: two compartments exchanging content at the same rate `D` both ways (G6DB) keep their total and their difference shrinks by `1 - 2D`.
- [`CGB.Generation.symmetric_stays_symmetric`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L162) [design / candidate / reviewed]：**G21 (no differentiation without fluctuation)**: deterministic dynamics `f` that commute with a relabelling `σ` (P2c) keep a `σ`-symmetric birth state (P3) symmetric forever.
- [`CGB.Generation.fixed_point_is_occupancy`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L175) [design / candidate / reviewed]：The fixed point of G20a is the occupancy of the log odds `log kp - log km`.
- [`CGB.Generation.available_strictAnti`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L183) [design / candidate / reviewed]：**G22a (use depletes)**: the available fraction strictly falls as demand rises.
- [`CGB.Generation.inhibitory_slot_antitone`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L189) [design / candidate / reviewed]：**G22b (equilibrium inhibition retreats from co-activity)**: a slot species whose energy raises co-activity's cost (`κ < 0`) has occupancy strictly falling in co-activity.
- [`CGB.Generation.shared_potential_forces_nesting`](../../lean/CGBAxioms/CGBAxioms/GenerationGrowth.lean#L196) [design / candidate / reviewed]：**G22c (no rich club needs local potentials)**: if all compartments share one potential and the strong set is not nested in rate order, contradiction (G19a).

## GenerationHoldCorrect（19）
_Holding versus correcting, and the latch transition (generation chain, 2026-11-04) The user separated two abilities of a memory: * **holding**: when nobody rewrites it, the content stays; * **attractor correction**: when_

- [`CGB.Generation.HoldCorrect.hold_accumulates`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L74) [semantics / universal / draft]：**Holding keeps every disturbance**: under the pure hold the state is the start plus the sum of all disturbances.
- [`CGB.Generation.HoldCorrect.hold_error_unbounded`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L82) [semantics / universal / draft]：**A held content drifts without bound** under any constant disturbance `e > 0`: for every bound `B` some time exceeds it.
- [`CGB.Generation.HoldCorrect.contraction_noise_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L94) [semantics / universal / draft]：**Contraction corrects**: if `f` pulls every state toward `r` by the factor `L < 1`, disturbances bounded by `e` leave an error of at most `L^n |x₀ - r| + e / (1 - L)` at every time, a bound that does not grow with `n`.
- [`CGB.Generation.HoldCorrect.contraction_merges`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L115) [semantics / universal / draft]：**Correction merges contents**: a map contracting all pairs by `L` brings any two contents within `L^n` of their initial gap.
- [`CGB.Generation.HoldCorrect.fb_odd`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L139) [design / candidate / draft]：Two copies of the latch: the renewal is odd, so `-r*` mirrors `r*`.
- [`CGB.Generation.HoldCorrect.fb_nonneg_eq`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L142) [design / candidate / draft]
- [`CGB.Generation.HoldCorrect.fb_sub_fixed`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L147) [design / candidate / draft]：**The identity**: for `x ≥ 0`, `fb x - r* = (x - r*) · (a + (1-a)/(1+g x))`.
- [`CGB.Generation.HoldCorrect.factor_lt_one`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L155) [design / candidate / draft]
- [`CGB.Generation.HoldCorrect.factor_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L160) [design / candidate / draft]
- [`CGB.Generation.HoldCorrect.factor_le_of_le`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L165) [design / candidate / draft]
- [`CGB.Generation.HoldCorrect.subcritical_forgets`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L172) [design / candidate / draft]：**κ ≤ 1, forgetting**: every positive content shrinks by at least the factor `a + c g` per event (and stays positive): only 0 is left in the long run.
- [`CGB.Generation.HoldCorrect.supercritical_fixed`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L184) [design / candidate / draft]：**κ > 1, the latch exists**: `r* > 0` and it is fixed.
- [`CGB.Generation.HoldCorrect.supercritical_origin_repels`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L194) [design / candidate / draft]：**κ > 1, the origin repels**: small positive content grows toward `r*`.
- [`CGB.Generation.HoldCorrect.latch_converges`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L202) [design / candidate / draft]：**κ > 1, convergence inside the basin**: from any `x₀ ≥ ρ`, `0 < ρ ≤ r*`, the undisturbed latch stays above `min x₀ r* ≥ ρ` and approaches `r*` at rate `factor ρ < 1`.
- [`CGB.Generation.HoldCorrect.latch_corrects`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L234) [design / candidate / draft]：**κ > 1, attractor correction under disturbance**: if the state starts within `B` of `r*`, the basin margin covers it (`B ≤ r* - ρ`, `ρ > 0`), and every disturbance is at most `(1 - factor ρ) · B`, the state stays within
- [`CGB.Generation.HoldCorrect.latch_tolerance`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L261) [design / candidate / draft]：**The latch's noise tolerance** (κ > 1).
- [`CGB.Generation.HoldCorrect.tolerance_falls_with_decay`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L287) [design / candidate / draft]：**Decay weakens the latch**: the tolerance `(v - u)²` with `u = √((1-a)/g)` falls as decay grows: a larger `u` (stronger decay, smaller retention `a`) with the latch still present (`u ≤ v`) tolerates no more disturbance.
- [`CGB.Generation.HoldCorrect.rates_vanish_at_transition`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L294) [design / candidate / draft]：**Critical slowing at κ = 1**: the forgetting rate below the transition and the correction rate at `r*` above it are `(1-a)(1-κ)` and `(1-a)(1-1/κ)` with `κ = c g / (1 - a)`; both vanish as κ → 1.
- [`CGB.Generation.HoldCorrect.long_timescale_band`](../../lean/CGBAxioms/CGBAxioms/GenerationHoldCorrect.lean#L317) [design / candidate / draft]：**Scale-free timescales near criticality**: a head's relaxation time exceeds `T` exactly when its `κ` lies within `1 / ((1-a) T)` of 1.

## GenerationIgnition（5）
_Ignition needs drive or interaction (generation chain, G23) Ignition (global-workspace theory) is all-or-none, self-sustaining activation: the number of active units has two peaks (off and ignited) with a valley between_

- [`CGB.Generation.choose_log_concave`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L19) [design / candidate / reviewed]：Binomial coefficients are log-concave.
- [`CGB.Generation.activeWeight_log_concave`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L47) [design / candidate / reviewed]
- [`CGB.Generation.no_valley`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L61) [design / candidate / reviewed]：**G23 (no ignition in equilibrium with additive energies)**: the distribution of the number of active units has no strict valley, so it cannot have an "off" peak and an "ignited" peak separated by a gap.
- [`CGB.Generation.pulse_rate_le`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L83) [design / candidate / reviewed]
- [`CGB.Generation.equilibrium_memory_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationIgnition.lean#L100) [design / candidate / reviewed]：**G24 (equilibrium memory bound)**: loading at rate `lp ≤ e^{|μ|/2} l0` for time `tp`, then relaxing at rate `l0` for time `T`, keeps at most `e^{|μ|/2 - 1} tp / T` of the deviation, for every choice of `l0`.

## GenerationInformationEnergy（7）
_Information and energy for a finite body in a stream (generation chain, 2026-10-28) Selected mathematical identities and conditional bounds for the information economics map._

- [`CGB.InformationEnergy.freeEnergy_eq_kl`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L53) [design / candidate / reviewed]：**N1**: the free energy in the landscape `-log π` is the relative entropy to `π`.
- [`CGB.InformationEnergy.freeEnergy_at_law`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L59) [design / candidate / reviewed]：The law itself has free energy zero.
- [`CGB.InformationEnergy.kl_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L65) [design / candidate / reviewed]：**N1**: free energy is never below that of the law (Gibbs' inequality).
- [`CGB.InformationEnergy.snapshot_misses_hidden`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L94) [design / candidate / reviewed]：**N2**: a measurement that reads only the visible part carries no information about the hidden part beyond what it carries about the visible one.
- [`CGB.InformationEnergy.finite_memory_bounds_net_work`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L114) [design / candidate / reviewed]：**N3**: work drawn from acquired information `I` at most `kT · I`, erasure of everything beyond the capacity at least `kT` per unit: the net never exceeds `kT · cap`.
- [`CGB.InformationEnergy.passivity_ends`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L120) [design / candidate / reviewed]：**N4**: with drift `q > 0` and no observation (W2's recursion at gain zero), the information of one observation, `½ log (1 + V / r)`, exceeds any cost `c > 0` after finitely many ticks.
- [`CGB.InformationEnergy.mixture_regret_le`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationEnergy.lean#L144) [design / candidate / reviewed]：**N5 (mixture regret)**: the Bayes mixture codes the sequence at most `log (1 / w θ₀)` worse than the source's law `θ₀` (pathwise, for every sequence and length).

## GenerationInformationProjection（17）
_Information under finite deterministic observation Data processing for `Medium.condMutualInfo`, including sparse laws and arbitrary nonnegative total mass._

- [`CGB.InformationProjection.gibbs_term_supported`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L17) [design / candidate / reviewed]：Gibbs' pointwise bound on compatible, possibly zero, supports.
- [`CGB.InformationProjection.log_sum_le`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L25) [design / candidate / reviewed]：Finite log-sum, with zero entries allowed and no normalization premise.
- [`CGB.InformationProjection.kl_nonneg_supported`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L71) [design / candidate / reviewed]：The existing relative entropy is nonnegative for normalized compatible supports.
- [`CGB.InformationProjection.kl_pos_of_support_gap`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L82) [design / candidate / reviewed]：A support gap makes relative entropy strictly positive for compatible nonnegative weights of equal mass.
- [`CGB.InformationProjection.project_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L108) [design / candidate / reviewed]
- [`CGB.InformationProjection.sum_project`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L114) [design / candidate / reviewed]：Observation preserves total mass, including zero total mass.
- [`CGB.InformationProjection.kl_project_le`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L123) [design / candidate / reviewed]：Relative entropy contracts under finite deterministic observation, without normalization, provided the reference is positive wherever the source is positive.
- [`CGB.InformationProjection.marginals_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L151) [design / candidate / reviewed]
- [`CGB.InformationProjection.reference_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L163) [design / candidate / reviewed]
- [`CGB.InformationProjection.reference_pos_of_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L171) [design / candidate / reviewed]
- [`CGB.InformationProjection.reference_sum`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L179) [design / candidate / reviewed]：The conditional-product reference has exactly the source's total mass, including when that mass is zero.
- [`CGB.InformationProjection.condMutualInfo_eq_kl`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L192) [design / candidate / reviewed]：CMI is a sum of relative entropies against the conditional-product reference.
- [`CGB.InformationProjection.condMutualInfo_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L212) [design / candidate / reviewed]：Nonnegative, unnormalized finite conditional mutual information.
- [`CGB.InformationProjection.condMutualInfo_pos_of_missing_rectangle`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L229) [design / candidate / reviewed]：A missing entry inside a positive marginal rectangle forces positive CMI.
- [`CGB.InformationProjection.project_sum`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L254) [design / candidate / reviewed]
- [`CGB.InformationProjection.reference_project`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L263) [design / candidate / reviewed]：The conditional-product reference commutes with observing the third variable.
- [`CGB.InformationProjection.condMutualInfo_project_le`](../../lean/CGBAxioms/CGBAxioms/GenerationInformationProjection.lean#L282) [design / candidate / reviewed]：**Finite deterministic CMI data processing**, including arbitrary sparse supports and unnormalized nonnegative weights.

## GenerationLanguageExistence（18）
_Necessary bridges for a language-life existence theorem Keep continuous GDN mathematics separate from its physical realization._

- [`CGB.LanguageExistence.no_global_finite_gdn_codec`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L26) [design / candidate / reviewed]：A finite physical code cannot losslessly represent every real GDN state.
- [`CGB.LanguageExistence.p1a_no_global_real_codec`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L39) [design / candidate / reviewed]：P1a plus a finite number of physical sites rules out a global exact real-state codec; it does not rule out a continuous macro model or finite precision matrix arithmetic.
- [`CGB.LanguageExistence.finite_range_nonincreasing_eventually_constant`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L48) [design / candidate / reviewed]：A nonincreasing real sequence with finite range eventually attains a fixed value.
- [`CGB.LanguageExistence.finite_body_stationary_loss_eventually_constant`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L72) [design / candidate / reviewed]：A fixed loss on a finite body cannot keep strictly improving while remaining nonincreasing at every step.
- [`CGB.LanguageExistence.recurrent_gain_excludes_eventual_stationary_descent`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L83) [design / candidate / reviewed]：Even *eventually* nonincreasing stationary loss is incompatible with arbitrarily late strict improvements on the same finite body.
- [`CGB.LanguageExistence.strict_descent_has_no_finite_exact_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L107) [design / candidate / reviewed]：The obstruction also applies on one actual trajectory: a finite body cannot reproduce an endlessly strictly descending fixed real readout exactly.
- [`CGB.LanguageExistence.restricted_step_refines`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L132) [design / candidate / reviewed]
- [`CGB.LanguageExistence.restricted_fold_refines`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L137) [design / candidate / reviewed]
- [`CGB.LanguageExistence.gdn_prior_preserves_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L147) [design / candidate / reviewed]：The physical-time prior never rewrites learned head bias.
- [`CGB.LanguageExistence.gdn_silent_continuation_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L154) [design / candidate / reviewed]：With no new arrived byte, any sequence of physical-time priors preserves the head bias.
- [`CGB.LanguageExistence.gdn_alternative_arrival_bias`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L165) [design / candidate / reviewed]：Alternative bytes arriving at the same prepared body install a head-bias difference determined by the actual local update.
- [`CGB.LanguageExistence.gdn_written_distinction_survives_silence`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L183) [design / candidate / reviewed]：A distinction written by different bytes survives arbitrarily many silent physical-time steps in this real GDN.
- [`CGB.LanguageExistence.precision_assimilation_is_two_state_step`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L204) [design / candidate / reviewed]：The existing two-state population law gives exactly the actual GDN precision-weighted mean update, with up/down probabilities determined by the same local gain.
- [`CGB.LanguageExistence.precision_two_state_rates_valid`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L215) [design / candidate / reviewed]：For a bounded occupancy target the two rates are valid probabilities; calm `v=1/2` balances them.
- [`CGB.LanguageExistence.bounded_mean`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L228) [design / candidate / reviewed]
- [`CGB.LanguageExistence.prior_unit_memory`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L238) [design / candidate / reviewed]
- [`CGB.LanguageExistence.assimilate_unit_memory`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L252) [design / candidate / reviewed]
- [`CGB.LanguageExistence.step_unit_memory`](../../lean/CGBAxioms/CGBAxioms/GenerationLanguageExistence.lean#L267) [design / candidate / reviewed]：Bounded byte embeddings imply bounded memory means through the *full* time-prior/assimilation/head step.

## GenerationLatchHealth（3）
_When is a latch healthy? (generation chain, latch health) A latch (ignition, G23/G24) is a basin of states held by positive feedback._

- [`CGB.Medium.trap_irrecoverable`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L27) [necessity / candidate / reviewed]：**H1 (lock-in is death)**: if the set `B` is closed under the organism's own events and no state in it can take content in, no state in it is recoverable.
- [`CGB.Medium.trap_insensitive`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L41) [design / candidate / reviewed]：**H2 (latched means blind)**: where rates and effects do not depend on the stimulus, the one-tick kernel out of that state is the same under every stimulus.
- [`CGB.PairCRN.pair_rate_ratio`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchHealth.lean#L59) [design / candidate / reviewed]：**H1, quantitative**: a cue changes the rate of an event by exactly `exp((reservoir s - reservoir s') · dC / 2)`.

## GenerationLatchSlot（8）
_Latch slot: renewal that does not ride on the input gate (generation chain, 2026-11-04) Motivation (`results/cgb_rich_world_20261104/RESULT.md`, `results/cgb_mode_task_20261103/`): the trained latch heads are erased by o_

- [`CGB.Generation.LatchSlot.coupled_closed_gate_decays`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L48) [design / candidate / draft]：Closed gate in the coupled rule: no renewal is left, only the decay `α`.
- [`CGB.Generation.LatchSlot.coupled_small_gate_subcritical`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L54) [design / candidate / draft]：**A small gate is subcritical** in the coupled rule: with `0 ≤ α`, `0 ≤ β` and `β η g ≤ 1 - α`, the renewal satisfies `c g ≤ 1 - a` (κ ≤ 1).
- [`CGB.Generation.LatchSlot.coupled_small_gate_forgets`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L61) [necessity / candidate / draft]：With a small gate the content is forgotten every event (`subcritical_forgets`).
- [`CGB.Generation.LatchSlot.coupled_latch_gate_floor`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L69) [design / candidate / draft]：**The gate floor of a coupled latch**: while it latches (κ > 1), `1 - α < β (η g - α)`.
- [`CGB.Generation.LatchSlot.coupled_disturbance_floor`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L77) [necessity / candidate / draft]：**The disturbance floor**: a latching coupled head admits from a value of size `V` at least `(1-α)(1-η)V/(η g - α)` per event (`D = β (1 - η) V`).
- [`CGB.Generation.LatchSlot.slot_composite`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L87) [design / candidate / draft]：Decay by `α` followed by the slot renewal is the same family with rescaled parameters.
- [`CGB.Generation.LatchSlot.slot_supercritical_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L94) [design / candidate / draft]：**When the slot latches**: with `a = 1 - ρ`, `c = ρ η` after decay `α`, the composite map is supercritical (`1 - a α < c g α`) iff `α (1 + ρ (η g - 1)) > 1`.
- [`CGB.Generation.LatchSlot.slot_supercritical_no_decay`](../../lean/CGBAxioms/CGBAxioms/GenerationLatchSlot.lean#L100) [design / candidate / draft]：Without decay the slot latches iff `η g > 1`, for any renewal rate `ρ > 0`.

## GenerationLattice（14）
_Lattice media: cells with neighbour contact (generation chain, M4; nesting T-E1) The joint medium of an organism nested in its world._

- [`CGB.Medium.p2b_of_conductance`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L39) [semantics / candidate / reviewed]：**M1a with conductances**: rates `κ a x · exp (-(E y - E x) / 2)` with `κ` equal on an event and its reverse (read at the respective source states) give P2b.
- [`CGB.LatticeCRN.sum_update`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L151) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.sum_update2`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L159) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.adj_ne`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L167) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.p1a`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L171) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.p1b`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L174) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.p2a`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L219) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.act1_ne`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L263) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.act2_ne`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L266) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.p2b`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L278) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.adj_aut`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L353) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.p2c`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L362) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.p3`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L396) [semantics / candidate / reviewed]
- [`CGB.LatticeCRN.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationLattice.lean#L423) [semantics / candidate / reviewed]：**M4 (lattice media are admitted)**: a lattice reaction medium satisfies the six clauses when its rule tables pass local obligations: content levels (P1a), conservation by rules among interior cells (P2a), a zero calm po

## GenerationLearning（7）
_When does a world reward learning? (generation chain, L-1 to L-3) C-1 (`GenerationLeaveRule`) says when a forager must compare now with the past within one life._

- [`CGB.Generation.innate_optimal_of_constant`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L27) [witness / world_contract / reviewed]：**L-1**: a constant average across generations makes the innate threshold optimal.
- [`CGB.Generation.learner_loses_of_constant`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L31) [witness / world_contract / reviewed]：**L-1**: in a constant world a learner that pays `κ > 0` loses to the instinct at every patch.
- [`CGB.Generation.innate_loss_quarter`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L37) [necessity / world_contract / reviewed]：**L-2**: with two different averages every innate threshold loses at least a quarter of their gap at some patch in one of the two worlds.
- [`CGB.Generation.learner_beats_innate`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L53) [witness / world_contract / reviewed]：**L-2**: a learner (who uses the world's own average after paying `κ`) beats every instinct somewhere when its cost is below a quarter of the gap.
- [`CGB.Generation.history_useless_of_independent`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L61) [necessity / world_contract / reviewed]：**L-3**: if the next world state `c` (law `p`) is independent of the history `e` (law `r`) from which the rule `σ e` is chosen, the expected value is at most that of the best fixed rule.
- [`CGB.Generation.settled_fixed_optimal`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L84) [witness / world_contract / reviewed]：**O-1**: if the world settles (its average constant from `T` on), the fixed rule `τ = c T` is optimal at every later time and patch: nothing is left to compare or learn.
- [`CGB.Generation.beaten_forever_iff_never_settles`](../../lean/CGBAxioms/CGBAxioms/GenerationLearning.lean#L92) [necessity / world_contract / reviewed]：**O-2**: every fixed rule is beaten after every time if and only if the world never settles (after every time it still takes two different averages).

## GenerationLeaveRule（3）
_When must a forager compare now with the past? (generation chain, C-1) A forager at a patch with local gain rate `q` either stays (earning `q`) or leaves for the rest of the world (worth `c`, the environment's average ra_

- [`CGB.Generation.leaveValue_le`](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L26) [design / candidate / reviewed]
- [`CGB.Generation.fixed_threshold_optimal`](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L30) [design / candidate / reviewed]：In a stationary world the fixed threshold `τ = c` is optimal everywhere.
- [`CGB.Generation.no_fixed_leave_threshold`](../../lean/CGBAxioms/CGBAxioms/GenerationLeaveRule.lean#L38) [design / candidate / reviewed]：**C-1 (a changing world needs comparison)**: with two different averages `c₁ < c₂`, every fixed threshold `τ` loses strictly at some patch rate `q` between them, in one of the two worlds.

## GenerationLifeGate（9）
_Life necessities as admission obligations (generation chain, gate layer L) The axioms P1–P3 say what a medium is._

- [`CGB.Medium.kernel_self_le`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L28) [design / candidate / reviewed]：One state-changing event of rate `r` makes the chain leave `x` within a tick with probability at least `r / R`.
- [`CGB.Medium.hold_vanishes`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L48) [design / candidate / reviewed]：**No exact latch**: if some state-changing event has rate at least `η > 0` under every stimulus, then the probability of sitting in `x` for `n` ticks under any stimulus sequence is at most `(1 - η / R) ^ n`, which tends
- [`CGB.PairCRN.rate_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L69) [design / candidate / reviewed]：Every rule with a positive prefactor fires on every ordered pair of distinct units at a positive rate, under every stimulus.
- [`CGB.PairCRN.exposed_of_witness`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L78) [design / candidate / reviewed]：**Life gate L-sense** (`Lives.exposed`): a pair medium is exposed as soon as one rule with a positive prefactor moves content through a boundary reservoir whose potential differs between two stimuli.
- [`CGB.PairCRN.total_pairAct`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L89) [design / candidate / reviewed]：The content an event takes in is the change of total content.
- [`CGB.PairCRN.metabolizes_of_witness`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L97) [design / candidate / reviewed]：**Life gate L-metabolism** (`Lives.metabolizes`): one content-changing event with a positive prefactor is a witness.
- [`CGB.tableCRN.dC_witness`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L116) [design / candidate / reviewed]
- [`CGB.tableCRN.metabolizes_of_rule`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L128) [design / candidate / reviewed]：**L-metabolism for rule tables**: one rule with a positive prefactor whose two sides carry different content, on two distinct units of its pages.
- [`CGB.tableCRN.exposed_of_rule`](../../lean/CGBAxioms/CGBAxioms/GenerationLifeGate.lean#L146) [design / candidate / reviewed]：**L-sense for rule tables**: the same witness on a rule touching a boundary page, and two stimuli at different reservoir potentials.

## GenerationLinearMemory（7）
_Linear recurrent memory cannot latch (generation chain, corollary of G17/G24) The results below concern the specific recurrence `x ↦ α (x - β ⟪k, x⟫_ℝ k) + (write)`, with `‖k‖ = 1`, `β ∈ [0, 2]`, `α < 1`._

- [`CGB.Generation.dot_delta`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L37) [design / candidate / reviewed]
- [`CGB.Generation.delta_nonexpansive`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L51) [design / candidate / reviewed]：The delta step never expands (squared norm), for `0 ≤ β ≤ 2`.
- [`CGB.Generation.gated_delta_contracts`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L61) [design / candidate / reviewed]：One gated delta step on the difference of two states (the write cancels): the squared gap shrinks by `α²`.
- [`CGB.Generation.deltaStep_sub`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L70) [design / candidate / reviewed]
- [`CGB.Generation.histories_merge`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L78) [design / candidate / reviewed]：**Histories merge**: if every step shrinks the gap between two states by at most `α t`, the gap after `T` steps is at most `∏ α t` times the initial gap.
- [`CGB.Generation.histories_merge_uniform`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L90) [design / candidate / reviewed]：With a uniform gate bound the gap decays at least geometrically.
- [`CGB.Generation.linear_hold_keeps_error`](../../lean/CGBAxioms/CGBAxioms/GenerationLinearMemory.lean#L105) [design / candidate / reviewed]：**A linear hold cannot clean up**: if a linear map holds the signal `s` (`L s = s`), a perturbation along `s` is held exactly as well, `L (s + t • s) - s = t • s`.

## GenerationMedium（2）
_A generator of admitted media (generation chain, M1) Instead of proving the six clauses case by case for each organism, prove once that a medium built from a local energy satisfies them, so an organism is admitted by giv_

- [`CGB.Medium.p2b_of_energy`](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L24) [semantics / universal / reviewed]：**M1a**: symmetric (Metropolis-type) rates of a reversible event set under an energy give P2b.
- [`CGB.Medium.rate_local_of_energy`](../../lean/CGBAxioms/CGBAxioms/GenerationMedium.lean#L45) [semantics / universal / reviewed]：**M1b**: a rate that is a function of the participants' states through the event's energy difference reads only the participants, when that difference does.

## GenerationNestedWorld（66）
_An organism nested in its world (T-E2) The joint medium of a bacterium-like organism and its world, as a lattice medium (M4)._

- [`CGB.poolShift_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L48) [design / candidate / reviewed]
- [`CGB.poolShift_val`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L61) [design / candidate / reviewed]
- [`CGB.NestedWorld.sdir_not`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L191) [design / candidate / reviewed]
- [`CGB.NestedWorld.drain_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L279) [design / candidate / reviewed]
- [`CGB.NestedWorld.drain_drain`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L294) [design / candidate / reviewed]
- [`CGB.NestedWorld.canDrain_drain`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L302) [design / candidate / reviewed]
- [`CGB.NestedWorld.gateCount_drain`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L305) [design / candidate / reviewed]
- [`CGB.NestedWorld.move_exchanges`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L374) [design / candidate / reviewed]：**A move is an exchange**: whatever occupies the cell ahead (vacuum or a body), a running mover whose fuel rule fires takes its place and the occupant takes the mover's, unchanged; pools stay.
- [`CGB.NestedWorld.poolCond_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L402) [design / candidate / reviewed]
- [`CGB.NestedWorld.poolCond_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L405) [design / candidate / reviewed]
- [`CGB.NestedWorld.poolFactor_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L411) [design / candidate / reviewed]
- [`CGB.NestedWorld.poolFactor_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L414) [design / candidate / reviewed]
- [`CGB.NestedWorld.poolCond_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L417) [design / candidate / reviewed]
- [`CGB.NestedWorld.poolFactor_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L428) [design / candidate / reviewed]：Pools are bounded by `V`, so every pool change has positive multiplicity.
- [`CGB.NestedWorld.poolCond_self`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L432) [design / candidate / reviewed]
- [`CGB.NestedWorld.poolCond_step`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L435) [design / candidate / reviewed]
- [`CGB.NestedWorld.mass_action_up`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L442) [design / candidate / reviewed]：**Mass action** from 7.1: with the entropic pool energy `cellF`, the lumped conductance makes a one-quantum gain of a pool with `m` quanta fire at a rate `∝ V - m` (its free places) ...
- [`CGB.NestedWorld.mass_action_down`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L458) [design / candidate / reviewed]：...
- [`CGB.NestedWorld.uniport_selectivity`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L479) [design / candidate / reviewed]：**Uniport selectivity is fixed by free energy**: a reversible exchange that moves one food quantum into the body (inner energy change `D`, food energy included) fills at food `m + 1` and drains back at food `q`; the rati
- [`CGB.NestedWorld.antiport_selectivity`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L497) [design / candidate / reviewed]：**Antiport selectivity**: the same exchange coupled to the waste pool (one food quantum in, one waste quantum out, inner energy change `D`) fills at food `m + 1`, waste `w` and drains at food `q`, waste `z + 1`; the rati
- [`CGB.NestedWorld.apply_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L556) [design / candidate / reviewed]
- [`CGB.NestedWorld.sum_pair_update`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L569) [design / candidate / reviewed]
- [`CGB.NestedWorld.pair_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L579) [design / candidate / reviewed]
- [`CGB.NestedWorld.fuelAct_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L590) [design / candidate / reviewed]
- [`CGB.NestedWorld.innerAct_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L599) [design / candidate / reviewed]
- [`CGB.NestedWorld.fuelAct_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L618) [design / candidate / reviewed]
- [`CGB.NestedWorld.gateCount_update`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L637) [design / candidate / reviewed]
- [`CGB.NestedWorld.innerAct_delta`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L652) [design / candidate / reviewed]：A pair rule that fired moved the content the pools gave up into the body (or back).
- [`CGB.NestedWorld.gateCount_innerAct`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L690) [design / candidate / reviewed]：A pair rule away from the membrane page leaves the membrane as it is.
- [`CGB.NestedWorld.c1_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L715) [design / candidate / reviewed]
- [`CGB.NestedWorld.c2_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L727) [design / candidate / reviewed]
- [`CGB.NestedWorld.content_p1a`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L761) [design / candidate / reviewed]
- [`CGB.NestedWorld.not_bid`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L791) [design / candidate / reviewed]
- [`CGB.NestedWorld.sum_update_nat`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L795) [design / candidate / reviewed]
- [`CGB.NestedWorld.cons1`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L805) [design / candidate / reviewed]
- [`CGB.NestedWorld.cons2`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L878) [design / candidate / reviewed]
- [`CGB.NestedWorld.eff1_xch`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L962) [design / candidate / reviewed]
- [`CGB.NestedWorld.xchEff_gate`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L969) [design / candidate / reviewed]：An exchange through a page other than the membrane leaves the membrane as it is.
- [`CGB.NestedWorld.xchEff_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L982) [design / candidate / reviewed]
- [`CGB.NestedWorld.back1`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1003) [design / candidate / reviewed]
- [`CGB.NestedWorld.eff2_diff`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1086) [design / candidate / reviewed]
- [`CGB.NestedWorld.diffEff_back`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1090) [design / candidate / reviewed]
- [`CGB.NestedWorld.back2`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1111) [design / candidate / reviewed]
- [`CGB.NestedWorld.div_gate`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1201) [design / candidate / reviewed]
- [`CGB.NestedWorld.cellGate_inner`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1214) [design / candidate / reviewed]
- [`CGB.NestedWorld.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1235) [design / candidate / reviewed]：**T-E2 (an organism nested in its world is admitted)**: the joint medium of organisms and their world satisfies the six axiom clauses as soon as the organism's inner pair rules and its fuel rule conserve content and no e
- [`CGB.NestedWorld.orgIn_congr`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1324) [design / candidate / reviewed]
- [`CGB.NestedWorld.orgIn_inner`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1332) [design / candidate / reviewed]：A pair rule that does not couple to the pools (`sdf + sdw = 0`) keeps the organism's content.
- [`CGB.NestedWorld.eff1_supply_bug`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1352) [design / candidate / reviewed]：The world's supply rules never touch the organism.
- [`CGB.NestedWorld.intake_is_exchange`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1362) [design / candidate / reviewed]：**Only events across the boundary feed the organism**: every single-cell event that raises the content held by the organism is a pool exchange or a pair rule coupled to the pools; a gated one has conductance proportional
- [`CGB.NestedWorld.dead_intake_is_physics`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1401) [design / candidate / reviewed]：**Death** (`Part.CanTakeIn` fails through the organism's own doing): with no intact membrane unit, every event that would raise the organism's content has rate zero, except an ungated exchange: physics the organism does
- [`CGB.NestedWorld.dead_output_is_physics`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1420) [design / candidate / reviewed]：**A dead body does not give itself away**: every single-cell event that lowers the content held by the organism is a pool exchange; with no intact membrane the gated ones have rate zero, so what leaves a dead body leaves
- [`CGB.NestedWorld.frees_only_by_dissolving`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1461) [design / candidate / reviewed]：**Only dissolution empties a place** (with G26c `corpses_must_go`, why the table needs it): every single-cell event that turns a cell holding a body into a cell without one is the dissolution of a body.
- [`CGB.NestedWorld.births_only_by_division`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1488) [design / candidate / reviewed]：**Births come only from division** (the dual of `frees_only_by_dissolving`): a neighbour event after which the pair holds more bodies is a division.
- [`CGB.NestedWorld.partner_exit_blocked`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1522) [design / candidate / reviewed]：A pair rule whose partner state is absent leaves the organism as it is.
- [`CGB.NestedWorld.gated_exit_blocked`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1537) [design / candidate / reviewed]：A gated exchange has no conductance without an intact membrane.
- [`CGB.NestedWorld.poolShift_one`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1541) [design / candidate / reviewed]
- [`CGB.NestedWorld.leak_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1553) [design / candidate / reviewed]：**Leak bound**: an ungated exchange that moves one quantum from a unit into the waste pool fires at rate at least `κ * exp (-Φ)` whenever the unit holds that quantum and the pool is not full, whatever the rest of the org
- [`CGB.NestedWorld.intake_eff`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1621) [design / candidate / reviewed]
- [`CGB.NestedWorld.intake_content`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1636) [design / candidate / reviewed]
- [`CGB.NestedWorld.metabolizes_of_intake`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1654) [design / candidate / reviewed]：**Nested metabolism** (`Part.Metabolizes`) for every organism whose table has a gated food intake of one quantum, in a state with an intact membrane and a unit ready to take in.
- [`CGB.NestedWorld.senses_of_intake`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1681) [design / candidate / reviewed]：**Nested sensing** (`Part.Senses`): the same intake event, from the same organism, fires at different rates when only the cell's food differs (one quantum or two).
- [`CGB.NestedWorld.ringAdj_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1753) [design / candidate / reviewed]
- [`CGB.NestedWorld.torusAdj_symm`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1761) [design / candidate / reviewed]
- [`CGB.NestedWorld.axioms_torus`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1787) [design / candidate / reviewed]：**The two-dimensional world is a model of the axioms**: organisms on a torus, food pellets from the world's boundary, diffusion, waste sinks, driven movement in four headings.
- [`CGB.NestedWorld.axioms_ring`](../../lean/CGBAxioms/CGBAxioms/GenerationNestedWorld.lean#L1799) [design / candidate / reviewed]：The one-dimensional ring is a model too.

## GenerationNesting（8）
_Nesting: the world is a medium too (generation chain, G26) An organism is a medium; the world it lives in obeys the same axioms, and so does a body that hosts organisms (mitochondria, gut bacteria) or a collective made o_

- [`CGB.Generation.cell_potential`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L39) [semantics / candidate / reviewed]：**G26a (the sensed potential of a world cell)**.
- [`CGB.Generation.site_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L45) [semantics / candidate / reviewed]：Conservation at a site: food after `T` ticks plus everything eaten is at most the initial food plus the regeneration.
- [`CGB.Generation.stay_starves`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L58) [semantics / candidate / reviewed]：**G26b (staying put starves)**: with food never negative, regeneration at most `r`, and a reserve that gains at most what is eaten and pays `c` per tick, the reserve after `T` ticks is at most `R₀ + f₀ + (r - c) T`.
- [`CGB.Generation.stay_dies`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L74) [semantics / candidate / reviewed]：The reserve is exhausted in finite time when living costs more than the site regenerates.
- [`CGB.Generation.cell_step`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L91) [semantics / candidate / reviewed]：Adding one quantum to a cell holding `n` of `V` costs exactly `log ((n + 1) / (V - n))`, so to a unit exchanging quanta with it the cell is a reservoir of potential `≈ log (n / V) = ln c`, and one exchange moves that pot
- [`CGB.Generation.missProb_ge`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L118) [semantics / candidate / reviewed]
- [`CGB.Generation.stream_never_dry`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L140) [semantics / candidate / reviewed]：**T-N7 (the stream never runs dry)**: under a stationary law `π` of the world–organism process, if some state `x` (weight `π x`) has two possible successors with different boundary readings, then every predictor of the n
- [`CGB.Generation.corpses_must_go`](../../lean/CGBAxioms/CGBAxioms/GenerationNesting.lean#L161) [semantics / candidate / reviewed]：**G26c (corpses must go: decomposition is necessary)**: in a world with finitely many places, if new deaths never stop (an endless stream of mortal lives: at every time some place that holds no corpse yet will later hold

## GenerationNonlinearity（12）
_Nonlinearity is necessary for language (generation chain, 2026-11-02) Language is full of interactions: what follows `th` is not what `t` predicts plus what `h` predicts._

- [`CGB.Generation.Nonlinearity.additive_no_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L51) [necessity / universal / draft]：A map that treats its sources separately has no interaction.
- [`CGB.Generation.Nonlinearity.interaction_needs_nonadditive`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L58) [necessity / universal / draft]：A target with an interaction is not additive.
- [`CGB.Generation.Nonlinearity.Additive.sub`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L63) [semantics / universal / draft]：The difference of two additive maps is additive, so log-odds of additive logits are.
- [`CGB.Generation.Nonlinearity.linear_read_of_superposed_state`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L76) [necessity / universal / draft]：A linear read of a superposed state is additive at any width.
- [`CGB.Generation.Nonlinearity.linear_stack_superposes`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L81) [necessity / universal / draft]：Any stack of linear layers with biases keeps the two sources superposed.
- [`CGB.Generation.Nonlinearity.linear_stack_no_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L93) [necessity / universal / draft]：Depth does not help: a linear read of a linear stack has no interaction.
- [`CGB.Generation.Nonlinearity.linear_recurrence_no_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L104) [necessity / universal / draft]：A linear recurrent memory with a linear read is additive in two past arrivals.
- [`CGB.Generation.Nonlinearity.additive_logit_cannot_separate`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L114) [necessity / universal / draft]：A final squashing does not help: additive log-odds cannot take the XOR sign pattern.
- [`CGB.Generation.Nonlinearity.additive_error_floor`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L122) [necessity / universal / draft]：The squared error of any additive predictor over the four cases is at least a quarter of the squared contrast.
- [`CGB.Generation.Nonlinearity.association_read_additive_in_contents`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L142) [necessity / candidate / draft]：At a fixed query, a GDN-style read is additive in what is stored: retrieval weighs the contents and cannot combine them.
- [`CGB.Generation.Nonlinearity.key_query_read_interacts`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L154) [witness / candidate / draft]：A key from one arrival read with a query from the other interacts, so a learned address already supplies the nonlinearity.
- [`CGB.Generation.Nonlinearity.one_rectifier_reads_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationNonlinearity.lean#L167) [witness / universal / draft]：XOR has an interaction, and one rectifier is enough to read it.

## GenerationOptimizer（19）
_Optimizer interfaces forced by the generation chain This file does not classify Adam or Muon._

- [`CGB.Generation.coordinate_separable_has_no_pairwise_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L42) [design / candidate / reviewed]：Coordinate-separable state cannot carry a nonzero pairwise interaction.
- [`CGB.Generation.pairwise_relation_requires_cross_term`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L55) [design / candidate / reviewed]：Equivalently, a nonzero relation forces a cross-term in every unary decomposition.
- [`CGB.Generation.constant_gain_not_local_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L69) [design / candidate / reviewed]：A demand-independent gain cannot satisfy local anti-dominance.
- [`CGB.Generation.reservoir_availability_is_local_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L78) [design / candidate / reviewed]
- [`CGB.Generation.reservoir_gain_is_local_anti_dominant`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L87) [design / candidate / reviewed]
- [`CGB.Generation.local_anti_dominance_requires_demand_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L95) [design / candidate / reviewed]：Any anti-dominant gain must read the compartment's local demand: it cannot be constant in that second argument.
- [`CGB.Generation.anti_dominance_favors_underdemanded`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L108) [design / candidate / reviewed]：Under local anti-dominance, an under-demanded compartment receives strictly higher marginal availability than a dominant compartment with greater past demand.
- [`CGB.Generation.shared_global_gain_fails_anti_dominance`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L119) [design / candidate / reviewed]
- [`CGB.Generation.separable_and_global_fails_cgb`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L131) [design / candidate / reviewed]：A coordinate-separable state paired with a shared global gain fails both CGB optimizer requirements.
- [`CGB.Generation.cgb_relation_carrier_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L147) [design / candidate / reviewed]：The CGB carrier has exact interaction `κ`.
- [`CGB.Generation.cgb_carrier_has_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L153) [design / candidate / reviewed]：When coupling `κ ≠ 0`, the CGB carrier is a genuine pairwise relation carrier.
- [`CGB.Generation.cgb_optimizer_admitted`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L165) [design / candidate / reviewed]：The CGB optimizer satisfies both CGB admission requirements.
- [`CGB.Generation.adam_class_has_positive_relational_error`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L178) [design / candidate / reviewed]：Any coordinate-separable (Adam-class) optimizer incurs strictly positive relational error whenever the target coupling is nonzero.
- [`CGB.Generation.cgb_optimizer_zero_relational_error`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L190) [design / candidate / reviewed]：The CGB optimizer achieves zero relational error on the target coupling.
- [`CGB.Generation.cgb_strictly_beats_adam_on_relations`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L197) [design / candidate / reviewed]：Formal Superiority over Adam: CGB optimizer strictly dominates any coordinate-separable optimizer on relational error.
- [`CGB.Generation.muon_class_zero_tail_protection`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L217) [design / candidate / reviewed]：Any shared global/spectral (Muon-class) optimizer provides exactly zero protective margin between compartments: the dominant and tail compartments compete under identical un-depleted gain.
- [`CGB.Generation.cgb_optimizer_positive_tail_protection`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L225) [design / candidate / reviewed]：The CGB reservoir optimizer provides a strictly positive protective margin to the under-demanded compartment whenever the dominant compartment has greater demand.
- [`CGB.Generation.cgb_strictly_beats_muon_on_tail_protection`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L236) [design / candidate / reviewed]：Formal Superiority over Muon: CGB optimizer strictly dominates any shared global gain rule on the tail protection margin, actively suppressing rich-club runaway.
- [`CGB.Generation.cgb_optimizer_pareto_superiority`](../../lean/CGBAxioms/CGBAxioms/GenerationOptimizer.lean#L260) [design / candidate / reviewed]：**Theorem: CGB Optimizer Strictly Pareto-Dominates the Adam-Muon Hybrid**.

## GenerationOrganismEcoliMsS0（3）
_Organism `ecoli_ms_s0`: generated from its rule table Source: `information_boltzmann/organisms/ecoli_ms_s0.json`, sha256 38ff4daa364a8d382398d91208cc7908b61c82d1a1324e64111291ef4801fa1b._

- [`CGB.Organisms.EcoliMsS0.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L52) [witness / candidate / reviewed]：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).
- [`CGB.Organisms.EcoliMsS0.metabolizes`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L58) [witness / candidate / reviewed]：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 1024 and 1735 changes content.
- [`CGB.Organisms.EcoliMsS0.exposed`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L65) [witness / candidate / reviewed]：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

## GenerationOrganismEcoliV1（3）
_Organism `ecoli_v1`: generated from its rule table Source: `information_boltzmann/organisms/ecoli_v1.json`, sha256 4ab0545552e85d06779b9d1da4a1c49b88bd976203fae895833f9a148ccb6a65._

- [`CGB.Organisms.EcoliV1.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L52) [witness / candidate / reviewed]：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).
- [`CGB.Organisms.EcoliV1.metabolizes`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L58) [witness / candidate / reviewed]：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 17 and 1 changes content.
- [`CGB.Organisms.EcoliV1.exposed`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L65) [witness / candidate / reviewed]：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

## GenerationOrganismGeneratedS0（3）
_Organism `generated_s0`: generated from its rule table Source: `information_boltzmann/organisms/generated_s0.json`, sha256 d57065ecec7500d38a2b40e1ef94f9e8412f45e47f7a7df63d20421010ccd6b0._

- [`CGB.Organisms.GeneratedS0.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L50) [witness / candidate / reviewed]：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).
- [`CGB.Organisms.GeneratedS0.metabolizes`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L56) [witness / candidate / reviewed]：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 14 and 0 changes content.
- [`CGB.Organisms.GeneratedS0.exposed`](../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L63) [witness / candidate / reviewed]：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

## GenerationPersistence（6）
_Activity without input: driven circulation, adaptation, latches (generation chain, 2026-10-27) Three different things keep a body active when the input says nothing, and they must not be confused (eco-sphere audit: 154 o_

- [`CGB.Generation.cycle_rates_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L46) [design / candidate / reviewed]：**A (Kolmogorov, one direction)**: detailed balance with a positive weight balances the rate products around every cycle `c 0 → c 1 → … → c (n-1) → c 0`.
- [`CGB.Generation.driven_cycle_not_balanced`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L61) [design / candidate / reviewed]：**A**: a cycle with unequal forward and reverse rate products (net circulation) admits no positive detailed-balance weight: self-repeating activity needs a drive.
- [`CGB.Generation.integral_feedback_adapts`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L73) [design / candidate / reviewed]：**B (integral feedback adapts perfectly)**: if the reference update `m ↦ m + γ (f (s - m) - a₀)` with `γ ≠ 0` has `m` as a fixed point, the activity there is `a₀`, whatever the input `s`.
- [`CGB.Generation.adapted_gap_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L82) [design / candidate / reviewed]：**B**: an act driven only by the gap `g (a - a₀)` with `g 0 = 0` stops at every adapted fixed point of a constant input.
- [`CGB.Generation.contraction_unique_rest`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L92) [design / candidate / reviewed]：**C**: a loop map with gain `L < 1` has at most one rest state: tonic activity without enough loop gain is monostable.
- [`CGB.Generation.latch_needs_unit_gain`](../../lean/CGBAxioms/CGBAxioms/GenerationPersistence.lean#L102) [design / candidate / reviewed]：**C**: two distinct rest states of a differentiable loop map force loop gain (slope) exactly one somewhere between them: a latch needs unit gain.

## GenerationPredictiveHarvest（9）
_Prediction is metabolism under reinvestment (language-stream life, T1) World contract (declared, a niche like the E._

- [`CGB.PredictiveHarvest.growth_eq_log_card_sub_crossEntropy`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L41) [design / candidate / reviewed]
- [`CGB.PredictiveHarvest.gibbs_term`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L53) [design / candidate / reviewed]：One term of Gibbs' inequality: `x (log y - log x) ≤ y - x` (from `log t ≤ t - 1`).
- [`CGB.PredictiveHarvest.gibbs_inequality`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L63) [design / candidate / reviewed]：**Gibbs' inequality**: `Σ p log c ≤ Σ p log p` for a positive allocation summing to one.
- [`CGB.PredictiveHarvest.growth_le`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L75) [design / candidate / reviewed]：**Kelly bound**: no allocation grows faster than `log |B| - H(p)`.
- [`CGB.PredictiveHarvest.growth_at_law`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L83) [design / candidate / reviewed]：Placing capacity as the stream's law attains the bound.
- [`CGB.PredictiveHarvest.uniform_growth_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L96) [design / candidate / reviewed]：A body that knows nothing about the stream (uniform allocation) does not grow.
- [`CGB.PredictiveHarvest.no_allocation_persists`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L106) [necessity / world_contract / reviewed]：When entropy exceeds `log |B| - m`, every admissible allocation has negative expected net log growth under this niche contract.
- [`CGB.PredictiveHarvest.persists_of_crossEntropy`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L113) [design / candidate / reviewed]：Below the cross-entropy threshold, expected net log growth is positive.
- [`CGB.PredictiveHarvest.additive_payoff_mode`](../../lean/CGBAxioms/CGBAxioms/GenerationPredictiveHarvest.lean#L122) [design / candidate / reviewed]：Without reinvestment (payoff additive in the capacity placed on the arrived byte) the best allocation puts everything on a most likely byte: probability matching is a consequence of the multiplicative contract, not of pr

## GenerationReciprocalCredit（2）
_Reciprocal local credit This is the common algebraic core for every slow map in a generated local medium._

- [`CGB.Generation.reciprocal_contact_response_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationReciprocalCredit.lean#L41) [design / candidate / reviewed]：The exact local forward response of one reciprocal contact.
- [`CGB.Generation.reciprocal_contact_teaching_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationReciprocalCredit.lean#L98) [design / candidate / reviewed]：Scaling a local response by a fresh boundary teaching signal preserves its identity as the exact directional derivative.

## GenerationReinvestment（14）
_Reinvestment as one balanced exchange (language organism, 2026-11-03) User decisions of 2026-11-03: occupancy is reinvested multiplicatively, and metabolism is gross balanced exchange._

- [`CGB.Reinvestment.reinvest_total`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L45) [design / candidate / draft]
- [`CGB.Reinvestment.gains_eq_losses`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L51) [design / candidate / draft]：Two distributions of the same total: the summed gains equal the summed losses.
- [`CGB.Reinvestment.gross_balanced`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L64) [design / candidate / draft]：**Gross balanced exchange**: reinvestment imports into the winners exactly what it exports from the losers.
- [`CGB.Reinvestment.grows_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L69) [design / candidate / draft]：An address with a positive share grows exactly when its bet beats the prediction.
- [`CGB.Reinvestment.no_exchange_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L77) [design / candidate / draft]：**Exact prediction exchanges nothing**, and only then: the shares stay put exactly when every live address bet the prediction.
- [`CGB.Reinvestment.mean_multiplier_one`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L93) [design / candidate / draft]：**Adaptation of the total**: the share-weighted multiplier is one for every byte.
- [`CGB.Reinvestment.reinvest_scale`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L100) [design / candidate / draft]：**Adaptation of the level**: multiplying every bet by the same positive number changes nothing.
- [`CGB.Reinvestment.predicted_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L107) [design / candidate / draft]
- [`CGB.Reinvestment.reinvest_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L111) [design / candidate / draft]
- [`CGB.Reinvestment.wealth_eq_mixture`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L124) [design / candidate / draft]：**Kelly equals Bayes**: the product of the predictions equals the mixture of the addresses' own products of bets.
- [`CGB.Reinvestment.wealth_ge_address`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L138) [design / candidate / draft]：The wealth is at least what one address would have made from its first share.
- [`CGB.Reinvestment.predicted_update`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L148) [design / candidate / draft]：Changing one bet changes the prediction by the share times the change.
- [`CGB.Reinvestment.log_predicted_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L161) [design / candidate / draft]：**Credit from the same event**: the derivative of the arrived byte's log prediction with respect to one bet is the share over the prediction.
- [`CGB.Reinvestment.credit_eq_new_share`](../../lean/CGBAxioms/CGBAxioms/GenerationReinvestment.lean#L173) [design / candidate / draft]：The same credit written with the new share: the update needs only the share after the arrival and the bet itself.

## GenerationResponseGatedCarrier（12）
_Response-gated conserved contact carrier `GenerationContactEligibility` establishes that a saturated local H contact has zero forward response._

- [`CGB.Generation.response_gated_contact_closed_holds`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L25) [design / candidate / reviewed]：A response gate closed at the source event leaves every coordinate of the existing shared contact reservoir unchanged.
- [`CGB.Generation.response_gated_contact_write_distinguishes`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L34) [design / candidate / reviewed]：If opening the existing shared exchange produces a nonzero local transfer, the S coordinate distinguishes it from a saturated (closed-gate) event.
- [`CGB.Generation.shared_contact_pool_vector_affine`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L58) [design / candidate / reviewed]：Exact connection of the retention/injection decomposition to the executable coordinatewise-release shared reservoir.
- [`CGB.Generation.contact_retention_bounds`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L66) [design / candidate / reviewed]
- [`CGB.Generation.affine_forward_history`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L78) [design / candidate / reviewed]：Algebraic forward expansion only.
- [`CGB.Generation.response_gated_vector_history_expansion`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L102) [design / candidate / reviewed]：Every actual coordinate is its retained initial value plus its finitely many injections, each weighted by subsequent retention.
- [`CGB.Generation.response_gated_vector_no_injection_retention`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L118) [design / candidate / reviewed]：With no new injection into this type during the interval, only the product of its actual coordinatewise retention acts on its initial content.
- [`CGB.Generation.response_gated_vector_zero_without_injection`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L135) [design / candidate / reviewed]：A type absent at birth cannot appear without an injection of that type.
- [`CGB.Generation.response_gated_vector_history_difference`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L148) [design / candidate / reviewed]：Equal external inputs alone do not guarantee equal injections: the free pool depends on each trajectory.
- [`CGB.Generation.response_gated_vector_nonzero_retention_distinguishes`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L173) [design / candidate / reviewed]：Nonzero retained product preserves an initial scalar distinction, not a claim that the complete old pre×gate vector can be decoded.
- [`CGB.Generation.response_gated_vector_source_difference`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L193) [design / candidate / reviewed]：Equal rank-release rates give a direct source-contrast recurrence inside one and the same reservoir; the common free pool is retained in the formula.
- [`CGB.Generation.positive_demand_preserves_contact_distinction`](../../lean/CGBAxioms/CGBAxioms/GenerationResponseGatedCarrier.lean#L204) [design / candidate / reviewed]

## GenerationRoutingCredit（3）
_Finite sparse-route credit A fixed sparse support is a finite boundary alphabet._

- [`CGB.Generation.log_softmax_direction_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationRoutingCredit.lean#L28) [design / candidate / reviewed]：The log-softmax derivative for an arbitrary finite direction.
- [`CGB.Generation.softmax_direction_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationRoutingCredit.lean#L72) [design / candidate / reviewed]：Each selected route probability has the expected softmax response.
- [`CGB.Generation.softmax_read_direction_deriv`](../../lean/CGBAxioms/CGBAxioms/GenerationRoutingCredit.lean#L98) [design / candidate / reviewed]：A route read changes by the sum of its local, centered softmax responses.

## GenerationSeedMemory（15）
_An actual memory world for the unchanged physical seed The first quiet tick moves the source particle and independently selects a quiet or food world._

- [`CGB.SeedMemory.clocked`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L53) [design / candidate / reviewed]：The existing structural event bound admits this world's exact clock.
- [`CGB.SeedMemory.law_one`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L57) [design / candidate / reviewed]：Collapse only the initial Dirac law, without enumerating body histories.
- [`CGB.SeedMemory.first_hop_act`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L76) [design / candidate / reviewed]
- [`CGB.SeedMemory.first_hop_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L81) [design / candidate / reviewed]：A registered positive-rate hop yields a positive actual first transition.
- [`CGB.SeedMemory.prefix_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L90) [design / candidate / reviewed]：Both input tags have positive probability at the very same reachable body.
- [`CGB.SeedMemory.write_act`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L97) [design / candidate / reviewed]
- [`CGB.SeedMemory.write_totals`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L102) [design / candidate / reviewed]：Food really increases total from one to two through the existing source.
- [`CGB.SeedMemory.write_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L106) [design / candidate / reviewed]
- [`CGB.SeedMemory.quiet_preserves`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L116) [design / candidate / reviewed]：Positive quiet transitions preserve the readout even though hops continue.
- [`CGB.SeedMemory.remembers`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L125) [design / candidate / reviewed]：The original full-body CMI memory clause, derived from actual positive prefixes and source intake, with no assumed positive-information premise.
- [`CGB.SeedMemory.quiet_has_nontrivial_event`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L135) [design / candidate / reviewed]：Quiet is not an identity dynamics: a positive-rate event changes the body.
- [`CGB.SeedMemory.partial_has_hop`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L142) [design / candidate / reviewed]：Finite Boolean graph fact only: every partially occupied ring has an occupied-to-vacant contact.
- [`CGB.SeedMemory.partial_rate_one`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L153) [design / candidate / reviewed]：Every partial microstate has a state-changing rate-one hop under every stimulus, including quiet.
- [`CGB.SeedMemory.partial_hold_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L169) [design / candidate / reviewed]：The probability of never leaving a partial microstate has a geometric bound, while the coarse memory remains.
- [`CGB.SeedMemory.physical_memory_component`](../../lean/CGBAxioms/CGBAxioms/GenerationSeedMemory.lean#L181) [design / candidate / reviewed]：One unchanged admitted physical medium jointly has actual memory and nontrivial quiet events in this world.

## GenerationSelfProduction（138）
_Conditional existence of an open self-producing niche This module is deliberately world-independent._

- [`CGB.SelfProduction.two_phase_finite_score_step_improves`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L33) [design / universal / reviewed]：On a finite alphabet with at least two symbols, a positive finite two-phase score step strictly improves the probability of the observed symbol on the same frozen feature.
- [`CGB.SelfProduction.softmax_nll_logit_error_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L80) [design / universal / reviewed]：If every byte logit changes by at most δ, the observed-byte negative log likelihood changes by at most 2δ.
- [`CGB.SelfProduction.association_affine_language_nll_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L122) [design / universal / reviewed]：The associative interference certificate propagates through an affine language head into a per-byte NLL bound.
- [`CGB.SelfProduction.delayed_readout_capacity_with_retrieval`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L171) [design / universal / reviewed]：For a fixed future query, externally retrieved views share the burden of distinguishing histories with the finite body.
- [`CGB.SelfProduction.association_delta_read_contract`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L190) [design / universal / reviewed]：Assemble the existing row delta and associative readout contracts on the same value-by-key matrix.
- [`CGB.SelfProduction.one_hot_delta_is_precision_assimilation`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L213) [design / universal / reviewed]：For an exactly one-hot key, no global decay, and precision-derived gain, the delta write agrees with slotwise precision assimilation: only the keyed slot changes.
- [`CGB.SelfProduction.precision_distinguishes_equal_mean_histories`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L229) [necessity / universal / reviewed]：Two slots with the same current mean but different positive precisions produce different next means on the same nonmatching observation.
- [`CGB.SelfProduction.mixed_key_delta_differs_from_slot_assimilation`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L253) [design / universal / reviewed]：A normalized mixed key gives a concrete obstruction to general equality: the delta write couples the first slot to the second, while independent precision assimilation with zero target leaves its zero mean unchanged.
- [`CGB.SelfProduction.positive_soft_query_changes_under_delta`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L265) [design / universal / reviewed]：Strictly positive soft addresses necessarily overlap.
- [`CGB.SelfProduction.association_assimilation_query_change`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L288) [design / universal / reviewed]：Query change under the existing natural-parameter assimilation: `weight` is the per-slot sum of observation weights and `weighted` the corresponding value-weighted sum.
- [`CGB.SelfProduction.association_assimilation_improves_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L308) [design / universal / reviewed]：On a fixed query and target, the existing assimilation improves squared prediction error exactly when its query change points sufficiently towards the target without overshooting.
- [`CGB.SelfProduction.association_assimilation_softmax_improves_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L336) [design / universal / reviewed]：Under slotwise assimilation and an arbitrary affine-head update, a target's softmax probability rises exactly when its multiplicative change exceeds the old distribution's average.
- [`CGB.SelfProduction.repeated_assimilation_retention`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L399) [design / universal / reviewed]：With no intervening prior step, repeated identical slot evidence is accumulated by the existing natural/precision update.
- [`CGB.SelfProduction.repeated_assimilation_association_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L434) [design / universal / reviewed]：The one-slot retention law composes into the actual value-by-key associative readout.
- [`CGB.SelfProduction.repeated_assimilation_probe_family`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L460) [design / universal / reviewed]：A family of future queries remains accurate when its target readout is accurate and the summed retained slot deviations fit the remaining tolerance.
- [`CGB.SelfProduction.state_conditioned_logloss_le_fixed`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L503) [design / universal / reviewed]：In a declared joint distribution, an oracle using the actual conditional byte law at each body state has no higher expected log loss than any fixed strictly positive normalized predictor.
- [`CGB.SelfProduction.merged_histories_binary_logloss_floor`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L522) [necessity / universal / reviewed]：If two equally likely histories share the same accessible state but require opposite next-byte predictions, any positive binary predictor pays at least `log 2` nats of expected loss.
- [`CGB.SelfProduction.state_conditioned_logloss_decomposition`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L547) [design / universal / reviewed]：For a state-dependent predictor, expected log loss is the conditional entropy plus the state-weighted KL mismatch.
- [`CGB.SelfProduction.renewable_reopening_not_single_old_advance`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L575) [design / universal / reviewed]：The existing renewable witness actually reopens an excluded candidate; no single old `advance` step from the same memory can implement that change.
- [`CGB.SelfProduction.renewable_witness_unbounded_supported_writes`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L586) [design / universal / reviewed]：The same existing renewable process has arbitrarily long supported prefixes with more than any requested number of paid writes.
- [`CGB.SelfProduction.renewable_maintenance_and_supported_relearning`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L607) [design / world_contract / reviewed]：One stochastic body-world process has operational maintenance and arbitrarily many paid writes on supported finite prefixes; blocking reopening in the same world and initial conditions lowers the registered survival scor
- [`CGB.SelfProduction.renewable_maintenance_world_action_contrast`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L641) [design / world_contract / reviewed]：The already-certified stochastic process also has a concrete world-side action contrast.
- [`CGB.SelfProduction.renewable_action_world_contract`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L680) [design / world_contract / reviewed]：This action kernel is normalized and agrees with the existing world for every live body.
- [`CGB.SelfProduction.renewable_action_machine_reproduces`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L739) [design / universal / reviewed]：Supplying the old policy reproduces every joint event and successor-state probability of the original uncut process, including the vacuum case.
- [`CGB.SelfProduction.renewable_action_autonomous_eq`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L767) [design / universal / reviewed]：Closing the explicit action port with the existing policy recovers the whole original machine, not just a one-step marginal or a sample trace.
- [`CGB.SelfProduction.renewable_action_zero_failure_steps`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L776) [design / universal / reviewed]：Under the declared action-port intervention, holding action 0 sends the existing initial state to the trap and then to vacuum in two certain steps.
- [`CGB.SelfProduction.renewable_action_zero_eventual_vacuum`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L800) [design / universal / reviewed]：In the specified intervention model, constant action zero reaches vacuum with probability one after two ticks and remains there.
- [`CGB.SelfProduction.renewable_world_hidden_branch_and_delayed_consequence`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L862) [design / world_contract / reviewed]：In the same certified world, aligned ready bodies have two equally likely next-world states while their next body state is the same.
- [`CGB.SelfProduction.renewable_world_aligned_state_supported`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L886) [design / universal / reviewed]：The aligned `aReady/feedA` state used above has a positive-probability three-step prefix from the actual initial state in the original process.
- [`CGB.SelfProduction.renewable_hidden_branch_reopens_and_rewrites`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L912) [design / universal / reviewed]：A positive-probability hidden world change in the certified process can flow through mismatch, reopening, probing and a new write on the same body.
- [`CGB.SelfProduction.renewable_reopen_control_branch`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L951) [design / universal / reviewed]：The recorded normal and reopen-disabled paths agree through tick four.
- [`CGB.SelfProduction.renewable_observable_block_contract`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L996) [design / universal / reviewed]：Every step of either block, including the transition to either next block, has positive probability in the unchanged autonomous process.
- [`CGB.SelfProduction.renewable_observable_path_contract`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1031) [design / universal / reviewed]：Every finite prefix of every selected branch sequence has positive probability from the aligned state, and the existing sensor at offset two of each block recovers its selected bit.
- [`CGB.SelfProduction.renewable_observable_from_birth`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1075) [design / universal / reviewed]：From the same original initial state, every finite prefix of every chosen bit sequence has positive support; sensory symbols distinguish the blocks, and each true block includes two actual paid writes.
- [`CGB.SelfProduction.trajectory_viable`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1169) [design / world_contract / reviewed]：The same rule generates a valid state at every finite time.
- [`CGB.SelfProduction.actual_actions_reproduce_trajectory`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1185) [design / world_contract / reviewed]：Replaying the actual actions in the same world reproduces the generated trajectory.
- [`CGB.SelfProduction.passive_opportunity_budget`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1196) [design / world_contract / reviewed]：If each passive use spends from a finite opportunity potential, the total passive intake from that point cannot exceed the starting potential.
- [`CGB.SelfProduction.passive_opportunity_bounds_eventual_interventions`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1223) [design / world_contract / reviewed]：A finite passive opportunity potential supplies the universal counterfactual bound required by `counterfactual_passive_bound_forces_activity`.
- [`CGB.SelfProduction.counterfactual_passive_bound_forces_activity`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1259) [necessity / world_contract / reviewed]：If every eventually passive intervention in the same world has bounded intake, an actual trajectory with unbounded intake cannot become permanently passive.
- [`CGB.SelfProduction.passive_opportunity_forces_activity`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1283) [design / world_contract / reviewed]：Finite passive opportunity and unbounded actual intake force recurrent non-passive action in the same deterministic world.
- [`CGB.SelfProduction.exists_infinite_self_producing_trajectory`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1298) [design / world_contract / reviewed]：**Unbounded self-production.** The contract generates one trajectory indexed by all natural causal ticks, and every prefix remains in its viable niche.
- [`CGB.SelfProduction.action_consequence_model_exists`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1311) [design / world_contract / reviewed]：The action-conditioned boundary rule itself is an exact self-consequence model when that rule is available to the organism's persistent state.
- [`CGB.SelfProduction.action_distinction_preserved`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1318) [necessity / world_contract / reviewed]：If two actions have different boundary consequences at a viable state, the exact self-consequence model preserves that distinction.
- [`CGB.SelfProduction.language_viable_next`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1479) [design / universal / reviewed]
- [`CGB.SelfProduction.candidate_erase_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1507) [design / universal / reviewed]
- [`CGB.SelfProduction.candidate_erase_ge_fee`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1516) [design / universal / reviewed]
- [`CGB.SelfProduction.effective_write_erases_distinction`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1524) [design / universal / reviewed]
- [`CGB.SelfProduction.language_state_capacity`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1533) [design / universal / reviewed]
- [`CGB.SelfProduction.generated_history_encodes_self_production`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1574) [design / world_contract / reviewed]：The generated history and the existing state recursion agree without a separately postulated trajectory-alignment premise.
- [`CGB.SelfProduction.generated_boundary_signal_forces_active`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1593) [design / world_contract / reviewed]：On the generated trajectory, a boundary-derived nonnegative integer signal supports the existing active conclusion when the explicit unbounded intake and passive-depletion conditions hold.
- [`CGB.SelfProduction.language_admission_trajectory_aligned`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1614) [design / world_contract / reviewed]：The admission history follows the self-production trajectory when its encoded initial state and every action-conditioned transition agree.
- [`CGB.SelfProduction.language_self_production_active_if_aligned`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1634) [design / world_contract / reviewed]：With the alignment conditions, the existing active-inference conclusion applies to the same viable state trajectory produced by the language contract.
- [`CGB.SelfProduction.language_joint_admission_self_production_active_if_aligned`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1667) [design / world_contract / reviewed]：The active and self-production halves use one trajectory once the two explicit encoded alignment conditions hold.
- [`CGB.SelfProduction.probeLoss_eq_candidate_erase`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1697) [design / universal / reviewed]：Under explicit injective-probe and recoverability conditions, probe loss is exactly the candidate distinction erased by the update.
- [`CGB.SelfProduction.language_controlled_trajectory_viable`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1731) [design / world_contract / reviewed]：Every externally controlled trajectory remains in the existing language viability region, since the language contract admits every boundary byte.
- [`CGB.SelfProduction.language_controlled_memory_eventually_stops`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1749) [design / world_contract / reviewed]：The candidate-based language contract eventually stops changing its entire memory, for every action sequence.
- [`CGB.SelfProduction.language_self_production_memory_eventually_stops`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1813) [design / world_contract / reviewed]：The autonomous self-production trajectory inherits the same eventual memory freeze; infinite viability does not by itself imply lifelong learning.
- [`CGB.SelfProduction.language_candidate_eventually_constant_boundary`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1835) [design / world_contract / reviewed]：In the original candidate contract, eventual memory freeze also freezes the policy action and body-determined boundary byte.
- [`CGB.SelfProduction.language_controlled_erasure_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1861) [design / world_contract / reviewed]：Candidate elimination telescopes along every intervention, not merely along the organism's chosen policy.
- [`CGB.SelfProduction.language_controlled_erasure_not_unbounded`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1898) [design / world_contract / reviewed]：Consequently, permanently eliminating a finite candidate family cannot produce the unbounded signal for any policy, even an active one.
- [`CGB.SelfProduction.language_probe_loss_eq_dissipation`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1932) [design / universal / reviewed]：Once the bridge's three recoverability conditions hold, its measured probe loss equals the language candidate dissipation for that byte.
- [`CGB.SelfProduction.candidate_reference_probe_loss_eq_dissipation`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1980) [design / universal / reviewed]
- [`CGB.SelfProduction.language_dissipation_encoded_eq_candidate_erase`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L1998) [design / universal / reviewed]
- [`CGB.SelfProduction.language_cumulative_erasure_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2013) [design / universal / reviewed]：Along an encoded history, cumulative candidate erasure is exactly the initial candidate count minus the current count.
- [`CGB.SelfProduction.language_cumulative_erasure_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2041) [design / universal / reviewed]：The total number of eliminated distinctions cannot exceed the finite initial candidate set, regardless of the length of the stream.
- [`CGB.SelfProduction.candidate_erasure_not_unbounded_signal`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2056) [necessity / world_contract / reviewed]：Finite one-way candidate elimination cannot itself supply the unbounded signal assumed by the active-inference contract.
- [`CGB.SelfProduction.language_self_production_bridge`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2071) [design / world_contract / reviewed]：The existing four-way language admission and the open self-production contract can be satisfied by the same persistent memory state.
- [`CGB.SelfProduction.language_joint_admission_and_self_production`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2086) [design / world_contract / reviewed]：The named joint-admission proposition and the infinite self-producing trajectory now form one explicit language-life existence statement.
- [`CGB.SelfProduction.language_self_consequence_model_exists`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2103) [design / world_contract / reviewed]：In the joined language contract, the self-consequence predictor is a single function of the persistent memory, action, and boundary response.
- [`CGB.SelfProduction.language_infinite_stream_life_complete`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2145) [design / universal / reviewed]：Joint candidate-contract theorem.
- [`CGB.SelfProduction.language_candidate_predictive_growth_positive`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2179) [design / universal / reviewed]：If the same language candidate's allocation beats the declared reinvestment cost on every encoded history, its predictive stream contract has positive expected net log growth on every such history.
- [`CGB.SelfProduction.language_conditional_joint_existence`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2202) [design / world_contract / reviewed]：Conditional joint existence theorem for the current language candidate.
- [`CGB.SelfProduction.minimal_generator_conditional_joint_existence`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2242) [design / world_contract / reviewed]：Minimal-generator refinement of the conditional joint existence theorem.
- [`CGB.SelfProduction.minimal_generator_infinite_stream_life_complete`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2288) [design / universal / reviewed]：The minimal generator is sufficient: `read` and `next` are materialized by history factorization, after which the complete joint theorem applies unchanged.
- [`CGB.SelfProduction.exists_language_infinite_stream_life_of_minimal_generator`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2301) [design / universal / reviewed]：Given an admissible minimal generator, there is a candidate satisfying the candidate specification.
- [`CGB.SelfProduction.renewable_language_carrier_correct`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2345) [design / universal / reviewed]
- [`CGB.SelfProduction.renewable_language_carrier_trajectory`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2375) [design / world_contract / reviewed]：The same state carrier has its required readout, byte transition, and an unbounded viable trajectory.
- [`CGB.SelfProduction.precision_binary_update_is_predictive_improvement`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2409) [design / universal / reviewed]：The precision-derived gain used by the active GDN path is a valid binary predictive update whenever incoming precision is positive.
- [`CGB.SelfProduction.precision_binary_sample_gain_not_future_guarantee`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2429) [design / universal / reviewed]：Fitting an arrived bit does not by itself improve the underlying stream: if the old binary predictor already equals the stream law, its expected log loss cannot improve after the precision-derived update.
- [`CGB.SelfProduction.categorical_precision_next_loss_direction`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2464) [design / universal / reviewed]：Categorical precision-weighted assimilation improves the next symbol's log loss exactly when that symbol repeats the observed symbol; a different symbol becomes strictly less likely.
- [`CGB.SelfProduction.renewable_language_life_gate`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2521) [design / world_contract / reviewed]：The renewable carrier, plus explicit predictive payment and recurrent learning conditions, yields one general-state language life gate.
- [`CGB.SelfProduction.exists_renewable_language_life_of_generator`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2592) [design / world_contract / reviewed]：Existential form of the renewable gate.
- [`CGB.SelfProduction.renewable_open_world_trajectory`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2675) [design / world_contract / reviewed]：One open-world trajectory remains viable and realizes the readout for the history of its own delivered inputs.
- [`CGB.SelfProduction.renewable_open_language_joint_gate`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2715) [design / world_contract / reviewed]：Conditional joint gate on the *open* body-world trajectory.
- [`CGB.SelfProduction.open_world_categorical_next_loss_direction`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2794) [design / world_contract / reviewed]：The categorical innovation condition applied to the actual boundary and the next actual boundary on one viable open-world trajectory.
- [`CGB.SelfProduction.open_world_precision_update_improves_next_prediction`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2832) [design / world_contract / reviewed]：An actual open-world update improves the next prediction when the next delivered bit repeats the current one and the same body's probability obeys the existing precision-derived innovation update.
- [`CGB.SelfProduction.masked_open_world_information_limit`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2902) [design / world_contract / reviewed]：Under the same fixed delivered symbol, identical starting bodies have identical body states and actions for any two evolving external worlds.
- [`CGB.SelfProduction.open_world_closure_excludes_conflicting_safe_actions`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2934) [design / world_contract / reviewed]：The present `RenewableOpenWorld.closed` quantifies over every hidden world state at a viable encoded body.
- [`CGB.SelfProduction.indistinguishable_joint_worlds_cannot_both_stay_viable`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2951) [necessity / world_contract / reviewed]：For a joint body-world viability predicate, two worlds hidden behind the same body can require conflicting safe actions.
- [`CGB.SelfProduction.joint_viability_policy_generation`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L2976) [design / world_contract / reviewed]：A body-only policy preserving a joint viable region exists exactly when, for each body state, there is one action safe for every compatible world in that region.
- [`CGB.SelfProduction.closed_region_law_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3007) [design / universal / reviewed]
- [`CGB.SelfProduction.stochastic_joint_viability_policy_generation`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3039) [design / world_contract / reviewed]：Stochastic version: a body-only policy keeps every positive-probability successor in the joint viable region iff each body has an action safe for all compatible viable worlds and all supported events.
- [`CGB.SelfProduction.stochastic_joint_recovery_policy_generation`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3072) [design / world_contract / reviewed]：Generate one stochastic policy whose same closed loop both stays within the joint region and reaches the declared event within the finite horizon.
- [`CGB.SelfProduction.ranked_supported_path_recurrent`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3115) [design / universal / reviewed]：The existing ranked recovery certificate forces a marked event in every `horizon`-tick window along every infinite sequence of supported transitions.
- [`CGB.SelfProduction.ranked_exists_recurrently_marked_path`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3162) [design / universal / reviewed]：A normalized finite event kernel has an infinite supported path from any initial state; a ranked certificate makes its marked events recurrent.
- [`CGB.SelfProduction.masked_world_prediction_and_viability_limit`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3201) [design / world_contract / reviewed]：On the same masked open-world runs, combine the prediction loss with a world-dependent viability requirement.
- [`CGB.SelfProduction.open_world_input_not_body_determined`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3234) [design / world_contract / reviewed]：If hidden world states can give different boundary inputs at the same organism state and action, no state-only predictor is correct for both.
- [`CGB.SelfProduction.open_world_body_prediction_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3247) [necessity / world_contract / reviewed]：Exact body-only prediction of the next body state for every world state is possible precisely when hidden world variation has no effect on that next state.
- [`CGB.SelfProduction.reachable_consequence_readout_iff`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3268) [design / world_contract / reviewed]：Restrict exact consequence prediction to joint states reachable from the initial condition under action interventions.
- [`CGB.SelfProduction.renewable_open_world_is_exogenous`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3364) [design / world_contract / reviewed]：In the existing open-stream witness, every action intervention leaves the external cursor on exactly the same trajectory.
- [`CGB.SelfProduction.existing_renewable_open_stream_witness`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3380) [design / world_contract / reviewed]：The external cursor advances each tick while the same finite organism stays in its stated non-vacuum region, for any external bit sequence.
- [`CGB.SelfProduction.renewable_open_first_input_recorded`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3405) [design / world_contract / reviewed]：On the initial probe, the existing finite carrier records which external answer arrived.
- [`CGB.SelfProduction.renewable_open_distinct_inputs_reach_distinct_states`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3422) [design / world_contract / reviewed]：Two worlds differing at the first observed bit start with the same body but lead to distinguishable body states after one boundary input.
- [`CGB.SelfProduction.renewable_open_first_action_affects_body`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3436) [design / world_contract / reviewed]：In this witness world, replacing the initial probe by the passive action changes the next body state: the passive step retains `search`, while the actual probe records the external answer.
- [`CGB.SelfProduction.renewable_open_always_passive_stays_search`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3454) [design / world_contract / reviewed]：In the open-stream witness an always-passive intervention stays in the non-vacuum `search` state forever.
- [`CGB.SelfProduction.renewable_open_stream_recurrently_writes`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3474) [design / world_contract / reviewed]：The open witness returns to the probing state every three ticks for any external bit stream, and each probe changes the finite body state.
- [`CGB.SelfProduction.renewable_open_probe_records_then_reopens`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3537) [design / world_contract / reviewed]：Each probe stores the current external bit for two successive body states, then reopens the body on the third tick.
- [`CGB.SelfProduction.renewable_open_probe_enables_conditional_action`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3585) [design / world_contract / reviewed]：In the witness world the existing certified policy reads the recorded distinction for two ticks; the always-passive intervention remains at the undifferentiated fallback action.
- [`CGB.SelfProduction.renewable_open_rewriting_loses_old_distinction`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3614) [design / world_contract / reviewed]：Recurrent writing is not enough for delayed credit: if two streams differ at one probe but agree at the next, the body first distinguishes them and then merges them.
- [`CGB.SelfProduction.existing_renewable_carrier_reuses_old_rule`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3646) [design / universal / reviewed]：The derived transition admits the old witness's writing, renewal and reopening steps on one finite carrier.
- [`CGB.SelfProduction.existing_renewable_autonomous_cycle`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3659) [design / world_contract / reviewed]：In the specified deterministic witness world the same finite carrier repeats the writing, refill and reopening cycle at every natural time scale.
- [`CGB.SelfProduction.stationary_nonincreasing_loss_flat_on_existing_cycle`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3701) [design / world_contract / reviewed]：On the existing closed three-state witness, a fixed state-dependent loss that never increases cannot strictly improve anywhere in the cycle.
- [`CGB.SelfProduction.existing_renewable_cycle_active_and_viable`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3722) [design / world_contract / reviewed]：The finite witness remains outside vacuum, does not become permanently passive, and has an action-sensitive boundary in the declared witness world.
- [`CGB.SelfProduction.quiet_identity_memory_marginal`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3745) [design / world_contract / reviewed]：When quiet freezes the body, the historical tagged memory law has the same body marginal after the first observation at every later time.
- [`CGB.SelfProduction.quiet_identity_old_memory_iff_one_write`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3776) [design / world_contract / reviewed]：For a body frozen under quiet, the old indefinitely retained conditional information requirement reduces exactly to a single positive-information write.
- [`CGB.SelfProduction.historical_persistence_requires_calm_change`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3797) [design / world_contract / reviewed]：If calm freezes every body state, the historical `Persists` contract is impossible: under permanent calm its body-only alive score stays at the initial value, which normal persistence requires to be above threshold.
- [`CGB.SelfProduction.grace_not_historical_persists`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3834) [design / world_contract / reviewed]：The grace witness freezes under calm, so it cannot satisfy the old long-disconnection requirement, regardless of its other operational proofs.
- [`CGB.SelfProduction.sensory_loss_and_historical_interrupt_differ`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3846) [design / world_contract / reviewed]：The existing sensory-maintenance witness tolerates one missing-sense tick and loses its body after three, but the historical interrupt selects its different frozen equilibrium stimulus.
- [`CGB.SelfProduction.grace_historical_memory_iff_one_write`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3872) [design / world_contract / reviewed]：In the same grace witness, the old memory criterion only tests whether one observation can add conditional information before quiet freezing.
- [`CGB.SelfProduction.grace_reachable_memory_changes_action`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3885) [design / universal / reviewed]：Two already-certified finite paths also occur in the grace world.
- [`CGB.SelfProduction.grace_reachable_equal_budget_memory_contrast`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3923) [design / universal / reviewed]：Two histories from the same initial process reach the same world state and bodies with equal total content, buffer and credit, but opposite stored candidates and actions.
- [`CGB.SelfProduction.grace_action_world_matches`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L3971) [design / world_contract / reviewed]
- [`CGB.SelfProduction.grace_action_intervention_contrast`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4015) [design / world_contract / reviewed]：Reconnecting the body's policy reproduces every transition of the registered grace process; overriding action zero from birth reaches vacuum in two certain steps.
- [`CGB.SelfProduction.grace_stochastic_joint_generator_exists`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4056) [design / universal / reviewed]：The existing grace witness supplies the local conditions of the stochastic joint generator: its already-proved good region is invariant under the original body policy.
- [`CGB.SelfProduction.grace_joint_recovery_generator_exists`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4078) [design / world_contract / reviewed]：The same finite grace world supplies the strengthened generator's local conditions, using its existing actual-intake mark and recovery rank.
- [`CGB.SelfProduction.grace_reachable_one_action_consequence`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4108) [design / universal / reviewed]：On an already-supported prefix, one action override at the same joint state changes the next observation and the following body: the normal probe gets an answer and writes the B candidate; action zero yields conflict and
- [`CGB.SelfProduction.grace_one_reopening_intervention_consequence`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4139) [design / world_contract / reviewed]：On a supported mismatch, override reopening for one body event only, then restore the original body and world rules.
- [`CGB.SelfProduction.grace_action_zero_vacuum_probability`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4191) [design / universal / reviewed]
- [`CGB.SelfProduction.grace_recovery_learning_joint`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4233) [design / world_contract / reviewed]：The same grace-world process has operational persistence, one-tick interruption tolerance and a loop-cut maintenance advantage, while supporting arbitrarily many paid rewrites on positive-probability finite prefixes.
- [`CGB.SelfProduction.grace_original_policy_joint_certificate`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4255) [design / world_contract / reviewed]：A concrete policy and its actual process simultaneously satisfy the stochastic generator's one-step closure and ranked recovery conditions, the original loop-cut maintenance and one-tick interruption contracts, and the e
- [`CGB.SelfProduction.grace_supported_paths_recurrently_intake`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4299) [design / world_contract / reviewed]：On every infinite supported path of the original grace process, actual intake occurs in every three-tick window.
- [`CGB.SelfProduction.grace_exists_recurrently_intaking_path`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4316) [design / world_contract / reviewed]：The original process has at least one infinite supported path, and each three-tick window on that path contains actual intake.
- [`CGB.SelfProduction.grace_infinite_supported_learning_and_intake`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4339) [design / world_contract / reviewed]：The already registered paid-relearning path is itself an infinite supported path of the same grace process.
- [`CGB.SelfProduction.grace_supported_path_eventually_no_writes`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4380) [design / universal / reviewed]：The same grace process also admits an infinite supported path that stops paid writing after its initial phase.
- [`CGB.SelfProduction.uniform_block_learning_failure_decay`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4436) [design / universal / reviewed]：Minimal probabilistic upgrade for persistent learning: if the probability of no useful learning event after `n` completed blocks is `q n`, and every block has conditional success at least `p`, then failure is bounded by
- [`CGB.SelfProduction.language_conditional_joint_specification`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4465) [design / world_contract / reviewed]：The current candidate specification and the conditional infinite-stream guarantees hold for one and the same encoded trajectory.
- [`CGB.SelfProduction.minimal_generator_conditional_joint_specification`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4496) [design / world_contract / reviewed]：Minimal-generator form of the total gate.
- [`CGB.SelfProduction.exists_conditionally_operating_language_life_of_minimal_generator`](../../lean/CGBAxioms/CGBAxioms/GenerationSelfProduction.lean#L4548) [design / world_contract / reviewed]：Existential form of the total gate.

## GenerationSpeechTempo（14）
_Speech tempo between two finite bodies (generation chain, 2026-10-28, edge N7) A speaker emits units (bytes, syllables) at rate `r` per tick to a listener._

- [`CGB.SpeechTempo.unsettled_eq_rpow`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L59) [design / candidate / reviewed]：The settling factor is the unified reference's geometric bound: `(1 - α) ^ (k / r)` with `L = -log (1 - α)`.
- [`CGB.SpeechTempo.retention_pos`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L65) [design / candidate / reviewed]
- [`CGB.SpeechTempo.retention_le_one`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L67) [design / candidate / reviewed]
- [`CGB.SpeechTempo.settled_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L73) [design / candidate / reviewed]
- [`CGB.SpeechTempo.settled_le_one`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L80) [design / candidate / reviewed]
- [`CGB.SpeechTempo.settled_le`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L84) [design / candidate / reviewed]
- [`CGB.SpeechTempo.useful_le_rate`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L93) [design / candidate / reviewed]
- [`CGB.SpeechTempo.useful_le_capacity`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L105) [design / candidate / reviewed]
- [`CGB.SpeechTempo.useful_le_inv_rate`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L115) [design / candidate / reviewed]
- [`CGB.SpeechTempo.tempo_band`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L127) [design / candidate / reviewed]：**N7**: the good tempo is a bounded band.
- [`CGB.SpeechTempo.slow_retention_vanishes`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L139) [design / candidate / reviewed]：Too slow: per unit, the earlier part is forgotten, whatever the listener's capacity.
- [`CGB.SpeechTempo.fast_settling_vanishes`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L155) [design / candidate / reviewed]：Too fast: per unit, the speaker is unsettled, whatever the listener's capacity.
- [`CGB.SpeechTempo.peaks_wasted`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L166) [design / candidate / reviewed]：Uniform information density, upper bound: with a per-slot capacity `C`, the slots absorb at most the total sent and at most `n C`; anything above capacity in a slot is lost.
- [`CGB.SpeechTempo.uniform_attains`](../../lean/CGBAxioms/CGBAxioms/GenerationSpeechTempo.lean#L173) [design / candidate / reviewed]：Uniform information density, attainment: the even spread reaches the bound.

## GenerationSurvivalBounds（16）
_Conditional maintenance, not immortality Maintenance means longer expected structural lifetime with drive than without (`CGB_AXIOMS_V0_20261022.md`, section 4)._

- [`CGB.SurvivalBounds.decay_eventually_lt`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L22) [design / universal / reviewed]：Reuse the existing geometric bound; no nonnegativity of `q` is needed for this upper-tail conclusion.
- [`CGB.SurvivalBounds.no_uniform_positive_lower`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L31) [necessity / universal / reviewed]：Uniform positive irreversible block loss rules out a fixed positive floor.
- [`CGB.SurvivalBounds.geometric_lower`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L38) [design / universal / reviewed]：Lower counterpart to the reused geometric upper bound.
- [`CGB.SurvivalBounds.cumulative_risk_lower`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L51) [design / universal / reviewed]：Telescoping lower bound; no independence, absorption, or sign premise is needed for this algebraic statement.
- [`CGB.SurvivalBounds.risk_budget_uniform_lower`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L62) [design / universal / reviewed]：A prefix budget allows strictly positive but summably bounded per-step risk.
- [`CGB.SurvivalBounds.geometric_tail_bound`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L82) [design / universal / reviewed]：Uniform scalar decay bounds every finite cut tail, reusing the existing pointwise geometric bound rather than repeating its power induction.
- [`CGB.SurvivalBounds.prefix_gap_persists`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L103) [design / universal / reviewed]：A prefix advantage exceeding the entire cut-tail allowance persists at all later horizons; normal needs only nonnegative scores, not contraction.
- [`CGB.SurvivalBounds.finite_window_lifetime_transfer`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L117) [design / universal / reviewed]：Finite-window cumulative errors plus an actual cut-tail certificate transfer reference lifetime advantage to every actual horizon beyond the window.
- [`CGB.SurvivalBounds.expected_step`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L137) [design / universal / reviewed]：General expected-step identity extracted from the rearrangement in `Energetics.expected_first_law`.
- [`CGB.SurvivalBounds.mean_initial`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L149) [design / universal / reviewed]
- [`CGB.SurvivalBounds.mean_unit_interval`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L155) [design / universal / reviewed]：A bounded observable gives a bounded score under the actual process law.
- [`CGB.SurvivalBounds.mean_contraction`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L168) [design / universal / reviewed]：A local row contraction transfers to the original ensemble score.
- [`CGB.SurvivalBounds.mean_risk_step`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L181) [design / universal / reviewed]：A local additive risk allowance transfers without independence assumptions.
- [`CGB.SurvivalBounds.mean_retention_step`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L195) [design / universal / reviewed]：Lower multiplicative row bounds transfer along the same actual law.
- [`CGB.SurvivalBounds.expectedLifetime_strict_of_row_gap`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L216) [design / universal / reviewed]：Positive maintenance without immortality: the same initial individual and observable have strictly longer truncated lifetime under the normal process.
- [`CGB.SurvivalBounds.normal_cut_eventual_gap`](../../lean/CGBAxioms/CGBAxioms/GenerationSurvivalBounds.lean#L246) [design / universal / reviewed]：Stronger, separate all-time-floor criterion: SAME initial state and observable, normal prefix risk budget versus uniform cut contraction.

## GenerationTable（2）
_Rule-table media (generation chain, M3) An organism is a finite table of reversible two-unit transitions: "a pair of units with identity pages `(i, j)` in states `a` may go to states `b`", each with a prefactor `κ`._

- [`CGB.RuleSpec.apply_back`](../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L34) [semantics / candidate / reviewed]
- [`CGB.tableCRN.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationTable.lean#L72) [semantics / candidate / reviewed]：**M3**: a rule-table medium is a model of the CGB axioms as soon as its content has a unique free state over a finite local state type, the calm reservoir potential is zero, and every rule between two interior pages cons

## GenerationUnifiedOptimizer（42）
_Unified CGB optimizer contract This module is a composition layer, not a new axiom and not a claim about a trained language model._

- [`CGB.Unified.build_is_cgb_candidate`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L58) [design / candidate / reviewed]：The composed relation and gain satisfy the existing CGB admission contract.
- [`CGB.Unified.balanced_nonneg`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L70) [design / candidate / reviewed]：A bounded exchange fraction gives a nonnegative allocation.
- [`CGB.Unified.balanced_col_budget`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L76) [design / candidate / reviewed]：A balanced allocation preserves every column budget.
- [`CGB.Unified.balanced_row_budget`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L81) [design / candidate / reviewed]：A balanced allocation preserves every row budget.
- [`CGB.Unified.balanced_allocation_contract`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L86) [design / candidate / reviewed]：The double-budget contract bounds the positive-feedback payoff.
- [`CGB.Unified.balanced_max_differentiates`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L97) [design / candidate / reviewed]：At the doubly-budgeted maximum, the two rows own different directions.
- [`CGB.Unified.uniform_allocation_is_budgeted`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L106) [design / candidate / reviewed]
- [`CGB.Unified.balanced_allocation_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L119) [design / candidate / reviewed]
- [`CGB.Unified.balanced_allocation_has_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L125) [design / candidate / reviewed]
- [`CGB.Unified.phase_relation_interaction`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L139) [design / candidate / reviewed]
- [`CGB.Unified.phase_relation_has_relation`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L144) [design / candidate / reviewed]
- [`CGB.Unified.phase_history_is_generated`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L150) [design / candidate / reviewed]
- [`CGB.Unified.phase_step_stays_in_unit_interval`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L156) [design / candidate / reviewed]
- [`CGB.Unified.phase_history_stays_in_unit_interval`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L167) [design / candidate / reviewed]
- [`CGB.Unified.real_transfer_total`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L189) [design / candidate / reviewed]
- [`CGB.Unified.real_transfer_target_readout`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L193) [design / candidate / reviewed]
- [`CGB.Unified.phase_exchange_transfer_is_conserved`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L207) [design / candidate / reviewed]
- [`CGB.Unified.phase_exchange_transfer_reads_phase`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L213) [design / candidate / reviewed]
- [`CGB.Unified.fresh_outcome_measurement_unbiased`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L228) [design / candidate / reviewed]
- [`CGB.Unified.fresh_outcome_trust_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L234) [design / candidate / reviewed]
- [`CGB.Unified.bernoulli_expected_brier_gap`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L254) [design / candidate / reviewed]
- [`CGB.Unified.bernoulli_expected_brier_calibrated`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L260) [design / candidate / reviewed]
- [`CGB.Unified.bernoulli_prediction_error_expectation`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L271) [design / candidate / reviewed]
- [`CGB.Unified.brier_score_is_not_outcome_prediction_error`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L276) [design / candidate / reviewed]
- [`CGB.Unified.centered_bernoulli_evidence_zero_mean`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L280) [design / candidate / reviewed]
- [`CGB.Unified.calibrated_evidence_penalty_is_square`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L286) [design / candidate / reviewed]
- [`CGB.Unified.eligibility_gated_phase_stays_in_unit_interval`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L299) [design / candidate / reviewed]
- [`CGB.Unified.eligibility_gate_zero_holds`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L312) [design / candidate / reviewed]
- [`CGB.Unified.eligibility_gate_one_recovers`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L316) [design / candidate / reviewed]
- [`CGB.Unified.eligibility_gated_signed_phase_stays_bounded`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L323) [design / candidate / reviewed]
- [`CGB.Unified.eligibility_gated_signed_phase_zero_holds`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L336) [design / candidate / reviewed]
- [`CGB.Unified.latch_phase_holds_before_release`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L343) [design / candidate / reviewed]
- [`CGB.Unified.latch_phase_releases_at_threshold`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L349) [design / candidate / reviewed]
- [`CGB.Unified.eligibility_gated_transfer_is_conserved`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L365) [design / candidate / reviewed]
- [`CGB.Unified.eligibility_gated_transfer_reads_phase`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L371) [design / candidate / reviewed]
- [`CGB.Unified.ema_history_collision`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L388) [design / candidate / reviewed]
- [`CGB.Unified.build_has_competition_and_differentiation`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L395) [design / candidate / reviewed]：The built architecture exposes both row and column competition budgets.
- [`CGB.Unified.build_history_is_geometric`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L407) [design / candidate / reviewed]：The retained history is the geometric trace generated by two-state exchange.
- [`CGB.Unified.build_history_balance`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L415) [design / candidate / reviewed]：The supplied typed carrier exposes its telescoping conserved history.
- [`CGB.Unified.build_modulation_stays_in_band`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L427) [design / candidate / reviewed]：The outcome gate leaves a dead band and stays inside the trust interval.
- [`CGB.Unified.build_latch_holds_and_releases`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L436) [design / candidate / reviewed]：The latch both holds under short contrary evidence and releases at its threshold.
- [`CGB.Unified.build_mechanical_admission`](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean#L447) [design / candidate / reviewed]：Combined mechanical admission: relation, competition, history and modulation.

## GenerationVoxelChemistry（16）
_Voxel chemistry: every local reaction table is admitted (generation chain, M5) The law layer of a developing body, proved once._

- [`CGB.VoxelRxn.swap_swap`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L46) [design / candidate / draft]
- [`CGB.VoxelState.fire_of_fires`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L65) [design / candidate / draft]
- [`CGB.VoxelState.fire_of_not`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L69) [design / candidate / draft]
- [`CGB.VoxelState.fires_swap`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L73) [design / candidate / draft]：A reaction that ran can run backwards.
- [`CGB.VoxelState.fire_swap_fire`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L86) [design / candidate / draft]：Running backwards restores the voxel.
- [`CGB.VoxelState.weight_fire`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L112) [design / candidate / draft]
- [`CGB.VoxelRxn₂.fire_of_fires`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L144) [design / candidate / draft]
- [`CGB.VoxelRxn₂.fire_swap_fire`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L148) [design / candidate / draft]
- [`CGB.VoxelChem.weight_eq_zero`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L224) [design / candidate / draft]
- [`CGB.VoxelChem.p1a_data`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L231) [design / candidate / draft]
- [`CGB.VoxelChem.fires_of_fire_ne₁`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L240) [design / candidate / draft]
- [`CGB.VoxelChem.fires_of_fire_ne₂`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L245) [design / candidate / draft]
- [`CGB.VoxelChem.axioms`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L252) [design / candidate / draft]：**M5**: every voxel chemistry that conserves weight in the interior and has reverses is admitted.
- [`CGB.VoxelRxn.swap_pair`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L330) [design / candidate / draft]
- [`CGB.VoxelState.rxnWeight_pair`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L338) [design / candidate / draft]
- [`CGB.VoxelState.row_conserves`](../../lean/CGBAxioms/CGBAxioms/GenerationVoxelChemistry.lean#L344) [design / candidate / draft]：A row conserves weight exactly when its two sides weigh the same.

## GraceRecovery（20）
_A recoverable diagnostic opportunity after missed sensing Keep the original body, learning payments, food amounts and clock._

- [`CGB.GraceRecovery.policy_table`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L50) [witness / universal / reviewed]
- [`CGB.GraceRecovery.moves_unchanged_on_good`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L68) [witness / universal / reviewed]
- [`CGB.GraceRecovery.good_safe`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L73) [witness / universal / reviewed]
- [`CGB.GraceRecovery.safe_nonvacuum`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L76) [witness / universal / reviewed]
- [`CGB.GraceRecovery.normal_good`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L79) [witness / universal / reviewed]
- [`CGB.GraceRecovery.normal_safe`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L86) [witness / universal / reviewed]
- [`CGB.GraceRecovery.idle_safe`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L92) [witness / universal / reviewed]
- [`CGB.GraceRecovery.normal_zero`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L99) [witness / universal / reviewed]
- [`CGB.GraceRecovery.idle_progress`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L107) [witness / universal / reviewed]
- [`CGB.GraceRecovery.supported`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L117) [witness / candidate / reviewed]：Supported body events execute exactly the old local update; an idle slot has zero incoming material.
- [`CGB.GraceRecovery.probe_potential`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L157) [witness / universal / reviewed]
- [`CGB.GraceRecovery.stimulus_before`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L162) [witness / universal / reviewed]
- [`CGB.GraceRecovery.stimulus_idle`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L166) [witness / universal / reviewed]
- [`CGB.GraceRecovery.stimulus_after`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L169) [witness / universal / reviewed]
- [`CGB.GraceRecovery.scheduled_closed`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L173) [witness / universal / reviewed]
- [`CGB.GraceRecovery.scheduled_progress`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L200) [witness / candidate / reviewed]
- [`CGB.GraceRecovery.positive`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L240) [witness / candidate / reviewed]
- [`CGB.GraceRecovery.one_tick_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L251) [witness / candidate / reviewed]
- [`CGB.GraceRecovery.one_tick_tolerance`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L258) [witness / candidate / reviewed]
- [`CGB.GraceRecovery.natural_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecovery.lean#L291) [witness / candidate / reviewed]

## GraceRecoveryAccounting（14）
_Unchanged paid learning and actual source accounting through the grace world All ledgers use the actual registered protocol, including the interrupted tick._

- [`CGB.GraceRecoveryAccounting.body_supported`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L26) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryAccounting.credit_step`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L66) [witness / universal / reviewed]
- [`CGB.GraceRecoveryAccounting.path_budgets`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L84) [witness / universal / reviewed]：Exact budgets on EVERY supported finite path, including interrupted and cut paths.
- [`CGB.GraceRecoveryAccounting.realized`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L117) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryAccounting.initial_energy`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L155) [witness / universal / reviewed]
- [`CGB.GraceRecoveryAccounting.source_energy_budget`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L166) [witness / universal / reviewed]
- [`CGB.GraceRecoveryAccounting.old_path_good`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L191) [witness / universal / reviewed]
- [`CGB.GraceRecoveryAccounting.normal_step_unchanged`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L206) [witness / universal / reviewed]
- [`CGB.GraceRecoveryAccounting.arbitrary_relearning`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L217) [witness / universal / reviewed]
- [`CGB.GraceRecoveryAccounting.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L239) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryAccounting.actual_law`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L251) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryAccounting.score_preserved`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L269) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryAccounting.physical_one_tick_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L292) [witness / universal / reviewed]
- [`CGB.GraceRecoveryAccounting.calm_has_no_intake`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryAccounting.lean#L297) [witness / universal / reviewed]

## GraceRecoveryClosure（18）
_Original marginal-ghost cut after diagnostic grace._

- [`CGB.GraceRecoveryClosure.policy_table`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L17) [witness / universal / reviewed]
- [`CGB.GraceRecoveryClosure.sum_cells`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L23) [witness / universal / reviewed]
- [`CGB.GraceRecoveryClosure.sum_configs`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L30) [witness / universal / reviewed]
- [`CGB.GraceRecoveryClosure.sum_env`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L38) [witness / universal / reviewed]
- [`CGB.GraceRecoveryClosure.kernel_table`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L45) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.early_move`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L75) [witness / candidate / reviewed]：Until tick four the body marginal is a point mass, so the ORIGINAL cut transition is derived from that marginal, not assigned a convenient law.
- [`CGB.GraceRecoveryClosure.cut_early`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L88) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.move_five`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L106) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.bad_six`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L116) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.no_dead_six`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L127) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.miss_to_probe_six`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L137) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.step_lower`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L149) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.dead_seed`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L180) [witness / universal / reviewed]：A quarter of mass has physically lost its body by tick seven.
- [`CGB.GraceRecoveryClosure.empty_kernel`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L188) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.dead_monotone`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L193) [witness / universal / reviewed]
- [`CGB.GraceRecoveryClosure.cut_mass_bound`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L227) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.cut_survival`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L256) [witness / candidate / reviewed]
- [`CGB.GraceRecoveryClosure.maintenance`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryClosure.lean#L265) [witness / candidate / reviewed]

## GraceRecoveryControls（4）
_Separate controls for diagnostic grace and recovery lookahead Changing the lookahead alone does not repair the old absorbing failure._

- [`CGB.GraceRecoveryControls.point_loss`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L23) [witness / universal / reviewed]
- [`CGB.GraceRecoveryControls.off_trap_recovery`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L45) [witness / universal / reviewed]
- [`CGB.GraceRecoveryControls.horizon_change_alone_fails`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L67) [witness / universal / reviewed]
- [`CGB.GraceRecoveryControls.two_tick_lookahead_insufficient`](../../lean/CGBAxioms/CGBAxioms/GraceRecoveryControls.lean#L89) [witness / universal / reviewed]

## Inference（6）
_Active inference, first rungs: the blanket and foresight The life definition orders its emergent part as a ladder: memory → world model and prediction → choosing actions by what is predicted._

- [`CGB.Medium.jump_blanket`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L50) [design / universal / reviewed]：Under P3, the full dynamics reads the stimulus only at the boundary.
- [`CGB.Medium.kernel_blanket`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L56) [design / universal / reviewed]
- [`CGB.Medium.blanket_of_axioms`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L73) [design / universal / reviewed]：P3 gives the organism's half of the blanket; a world that touches only the body gives the other half.
- [`CGB.Models.rock_cannot_eat`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L119) [witness / universal / reviewed]：The rock satisfies all six axioms but can never take anything in: it was never alive.
- [`CGB.Models.rock_not_persists`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L124) [witness / universal / reviewed]：So the rock persists in no world.
- [`CGB.finite_reweighting_improves_iff`](../../lean/CGBAxioms/CGBAxioms/Inference.lean#L134) [semantics / universal / reviewed]：Any positive finite reweighting improves an observed event's normalized probability exactly when its conditional mean multiplier exceeds the global mean.

## InformationEconomics（3）
_Information economics: a small reusable basis The project has many mechanism-level theorems, but their common economic core is small._

- [`CGB.InformationEconomics.causal_carrier_contract_iff`](../../lean/CGBAxioms/CGBAxioms/InformationEconomics.lean#L33) [necessity / universal / reviewed]：**Causal carrier basis.** A finite state is a valid compressed carrier exactly when it supports both the required readout and the encoded successor for every future input.
- [`CGB.InformationEconomics.selective_write_contract`](../../lean/CGBAxioms/CGBAxioms/InformationEconomics.lean#L60) [design / world_contract / reviewed]：**Selective write basis.** Agreement over all live candidates consumes no write capacity; a funded observation that rules out a live candidate spends one unit and removes that candidate.
- [`CGB.InformationEconomics.predictive_growth_contract`](../../lean/CGBAxioms/CGBAxioms/InformationEconomics.lean#L81) [design / world_contract / reviewed]：**Predictive-growth basis.** Under the declared reinvestment contract, growth decomposes into a fixed alphabet term minus cross-entropy and is bounded by the stream entropy.

## InformationThermodynamics（3）
_Information acquisition as a conditional thermodynamic contract The general medium axioms do not assign physical energy to a recorded symbol._

- [`CGB.InformationThermodynamics.erased_information_floor`](../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L38) [design / universal / reviewed]
- [`CGB.InformationThermodynamics.positive_erasure_requires_positive_dissipation`](../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L42) [semantics / universal / reviewed]
- [`CGB.InformationThermodynamics.efficient_recorder_is_not_free`](../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L51) [semantics / universal / reviewed]

## Interaction（19）
_Finite probabilistic interaction coalgebras This is a semantic layer, not a physical axiom._

- [`CGB.Interaction.Machine.conditioned_next`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L33) [semantics / universal / reviewed]：A positive-mass observation uniquely fixes the normalized next-state law.
- [`CGB.Interaction.trace_nonneg`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L54) [semantics / universal / reviewed]
- [`CGB.Interaction.trace_prefix_consistent`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L64) [semantics / universal / reviewed]：Cylinder consistency: summing all one-symbol extensions recovers the prefix probability.
- [`CGB.Interaction.Coupling.expectation_at_output`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L98) [semantics / universal / reviewed]：One-step transfer of any observable continuation value.
- [`CGB.Interaction.Bisimulation.trace_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L115) [semantics / universal / reviewed]
- [`CGB.Interaction.Bisimulation.sameBehavior`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L126) [semantics / universal / reviewed]：Coinductive soundness for observed infinite behaviour, represented by ALL finite cylinders.
- [`CGB.Interaction.coinduction`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L130) [semantics / universal / reviewed]
- [`CGB.Interaction.bisimilar_is_bisimulation`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L133) [semantics / universal / reviewed]
- [`CGB.Interaction.bisimilar_sameBehavior`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L138) [semantics / universal / reviewed]
- [`CGB.Interaction.distinguishing_trace`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L141) [semantics / universal / reviewed]
- [`CGB.Interaction.Refines.bisimulation`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L152) [semantics / universal / reviewed]
- [`CGB.Interaction.Refines.sameBehavior`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L167) [semantics / universal / reviewed]
- [`CGB.Interaction.Refines.refl`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L170) [semantics / universal / reviewed]
- [`CGB.Interaction.Refines.comp`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L175) [semantics / universal / reviewed]：Certificates compose, so a chain of representation changes does not need to recheck arbitrary long sample traces after each implementation pass.
- [`CGB.Interaction.TimedRefinement.elapsed_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L196) [semantics / universal / reviewed]
- [`CGB.Interaction.Bisimulation.firstHit_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L214) [semantics / universal / reviewed]
- [`CGB.Interaction.Refines.firstHit_eq`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L230) [semantics / universal / reviewed]
- [`CGB.Interaction.Refines.recovery_test_iff`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L237) [semantics / universal / reviewed]：To preserve the existing nonempty-body + recovery-threshold test, content must ALSO be represented faithfully.
- [`CGB.Interaction.firstHit_bounds`](../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L246) [semantics / universal / reviewed]

## InteractionBridge（6）
_The existing medium is a probabilistic interaction coalgebra The observation wrapper preserves the full joint transition and its world tick._

- [`CGB.Interaction.Machine.ofKernels_step`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L27) [semantics / universal / reviewed]
- [`CGB.Interaction.Machine.ofKernels_marginal`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L33) [semantics / universal / reviewed]：Forgetting the observer's label recovers EXACTLY the original transition.
- [`CGB.Interaction.Machine.ofKernels_firstHit`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L40) [semantics / universal / reviewed]：Observe a time protocol in the existing external world clock.
- [`CGB.Medium.intakeCoalgebra_firstHit`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L91) [semantics / universal / reviewed]
- [`CGB.Medium.intakeCoalgebra_actualRecover`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L101) [semantics / universal / reviewed]：This normal-world observation yields the existing actualRecover, not a different coalgebraic notion of success or a favourable-stimulus menu.
- [`CGB.Medium.refinement_preserves_recovery_and_time`](../../lean/CGBAxioms/CGBAxioms/InteractionBridge.lean#L112) [semantics / universal / reviewed]：Actual recovery AND elapsed time transfer along a certified implementation.

## InteractionControls（7）
_Semantic certificates and obstructions These mathematical controls exercise exact representation, not brain capacity._

- [`CGB.InteractionControls.independent_bit_refines`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L29) [witness / universal / reviewed]：Removing a bit is justified HERE by a universal transition equation.
- [`CGB.InteractionControls.independent_bit_sameBehavior`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L35) [witness / universal / reviewed]
- [`CGB.InteractionControls.same_first_observation`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L46) [witness / universal / reviewed]
- [`CGB.InteractionControls.different_future`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L51) [witness / universal / reviewed]
- [`CGB.InteractionControls.no_bisimulation_from_first_observation`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L57) [witness / universal / reviewed]
- [`CGB.InteractionControls.physical_certificate_rejects_time_rescaling`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L63) [witness / universal / reviewed]
- [`CGB.InteractionControls.actual_recovery_bridge`](../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L71) [witness / universal / reviewed]：The tested representation is the identity on the actual continuous state; the bridge must work for BOTH normal and cut futures with the same world clock.

## Language（1）
_CGB language: finite-state Markov jump processes The state–event language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1._

- [`CGB.JumpProcess.expect_const_iff`](../../lean/CGBAxioms/CGBAxioms/Language.lean#L32) [semantics / universal / reviewed]：`⟨f⟩` is constant for every initial distribution iff `L f = 0` pointwise.

## LanguageLifeAdmission（1）
_Historical finite-candidate language admission This is a contract, not an existence theorem._

- [`CGB.LanguageLifeAdmission.joint_admission`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeAdmission.lean#L97) [design / universal / reviewed]

## LanguageLifeKernel（54）
_Compact algebraic reference kernel for the language-life model This is a finite exact reference object, not a neural-network capability claim._

- [`CGB.LanguageLifeKernel.incidence_ternary`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L22) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.incidence_sum_zero`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L28) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.charge_total_preserved`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L38) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.reserve_balance`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L50) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.step_clock_advances`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L66) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.step_charge_conserved`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L71) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.step_reserve_paid`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L77) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.no_visible_only_delayed_credit_rule`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L99) [necessity / universal / reviewed]
- [`CGB.LanguageLifeKernel.delayed_credit_requires_history_carrier`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L107) [necessity / universal / reviewed]
- [`CGB.LanguageLifeKernel.delayed_action_credit_requires_state_separation`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L119) [necessity / universal / reviewed]：Delayed action credit needs a state distinction between alternatives whose later consequences require different credit.
- [`CGB.LanguageLifeKernel.conserved_transfer_target_trace`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L155) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.conserved_transfer_source_trace`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L160) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.conserved_transfer_total_preserved`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L165) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.observable_transfer_recovers_amount`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L185) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.superposed_transfer_readout_is_net`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L189) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.conservation_does_not_identify_components`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L193) [witness / universal / reviewed]
- [`CGB.LanguageLifeKernel.leaky_transfer_readout_is_net`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L205) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.conserved_carrier_path_balance`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L224) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.conserved_carrier_path_no_output`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L233) [semantics / universal / reviewed]
- [`CGB.LanguageLifeKernel.history_readout_exists_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L250) [necessity / universal / reviewed]
- [`CGB.LanguageLifeKernel.history_transition_exists_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L258) [necessity / universal / reviewed]
- [`CGB.LanguageLifeKernel.external_causal_readout_exists_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L280) [necessity / universal / reviewed]
- [`CGB.LanguageLifeKernel.external_trace_requires_retrieved_distinction`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L304) [necessity / universal / reviewed]
- [`CGB.LanguageLifeKernel.observational_record_does_not_fix_intervention`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L317) [witness / universal / reviewed]：Two world equations agree on every recorded A=U trajectory but disagree under the intervention U=false, A=true.
- [`CGB.LanguageLifeKernel.SensoryRecord.exact_record_contract`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L366) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.SensoryRecord.action_content_distinguishes`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L380) [design / universal / reviewed]：All sixteen records already have distinct action/content observations.
- [`CGB.LanguageLifeKernel.SensoryRecord.record_state_lower_bound`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L386) [necessity / universal / reviewed]
- [`CGB.LanguageLifeKernel.SensoryRecord.fold_refines`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L400) [design / universal / reviewed]：The finite record supports arbitrary-length replay by a forward fold, without consulting the world regime or keeping an autograd history.
- [`CGB.LanguageLifeKernel.SensoryRecord.joint_successor_reconstruct`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L412) [design / universal / reviewed]：Every positive-probability successor in the original joint body/world process is exactly representable, for either registered cut and any schedule.
- [`CGB.LanguageLifeKernel.CausalRecord.policy_cases`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L451) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.CausalRecord.weights_exact`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L458) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.CausalRecord.sensor_alias_witness`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L467) [design / universal / reviewed]：Same present sensor, different consequences of the same actuator setting.
- [`CGB.LanguageLifeKernel.CausalRecord.update_sound`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L478) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.CausalRecord.update_least`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L493) [design / universal / reviewed]：Any candidate set retaining every compatible successor contains this one.
- [`CGB.LanguageLifeKernel.CausalRecord.action_feedback_witness`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L504) [design / universal / reviewed]：An identical later sensor has different interpretations when the earlier action differs.
- [`CGB.LanguageLifeKernel.PaidRecord.settlement_contract`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L539) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.PaidRecord.world_feedback_possible`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L558) [design / universal / reviewed]
- [`CGB.LanguageLifeKernel.PaidRecord.combined_write_budget`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L574) [design / universal / reviewed]：Add the recorder's separately endowed reserve to the ORIGINAL body's credit ledger.
- [`CGB.LanguageLifeKernel.PaidRecord.historical_frame_witness`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L587) [design / universal / reviewed]：Holding a source hypothesis is sound about its origin, not about a moving world.
- [`CGB.LanguageLifeKernel.PaidRecord.joint_source_possible`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L596) [design / universal / reviewed]：The ORIGINAL normal joint kernel, with any registered sensory schedule, licenses a receipt only for its pre-update body action and its successor's actual sensor.
- [`CGB.LanguageLifeKernel.PaidRecord.attach_refines`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L622) [design / universal / reviewed]：Projection recovers the whole original kernel, hence its registered finite-future observations by Interaction.Refines.sameBehavior.
- [`CGB.LanguageLifeKernel.LocalEdit.binary_factor_balance`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L635) [design / universal / reviewed]：A binary factor edit depends on joint contrast, not just either marginal.
- [`CGB.LanguageLifeKernel.LocalEdit.accepts_iff`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L648) [design / universal / reviewed]：With positive total masses, the executable comparison exactly detects improvement of the same event; inference and resource contracts are separate.
- [`CGB.LanguageLifeKernel.FirstHit.first_hit_sum`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L680) [design / universal / reviewed]：Conditional on a fixed lifetime, first-success probabilities telescope.
- [`CGB.LanguageLifeKernel.FirstHit.truncation`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L691) [design / universal / reviewed]：Stop unfolding after `horizon` trials.
- [`CGB.LanguageLifeKernel.FirstHit.probability_bounds`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L720) [design / universal / reviewed]：A normalized nonnegative finite lifetime distribution gives a genuine probability, independently of the chosen support representation.
- [`CGB.LanguageLifeKernel.CausalState.realizes`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L768) [design / universal / reviewed]：Canonical readout and single-input recursion realize the same contract.
- [`CGB.LanguageLifeKernel.CausalState.fold_refines`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L778) [design / universal / reviewed]：Arbitrarily long forward recursion; not a bounded-memory implementation.
- [`CGB.LanguageLifeKernel.CausalState.minimal`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L787) [design / universal / reviewed]：Every exact deterministic realization projects onto the canonical states.
- [`CGB.LanguageLifeKernel.CausalState.finite_bound`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L807) [design / universal / reviewed]：Finiteness follows from a finite exact realization; it is not assumed.
- [`CGB.LanguageLifeKernel.CausalState.finite_alphabets_unbounded`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L817) [design / universal / reviewed]：Finite input and output alphabets alone do not imply finite exact memory.
- [`CGB.LanguageLifeKernel.CausalState.separated_histories_capacity`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L830) [necessity / universal / reviewed]：Uniform precision requires distinct readable codes for separated futures.
- [`CGB.LanguageLifeKernel.Approximation.fold_error`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L865) [design / universal / reviewed]：Full one-step error certificates compose over any finite continuation.
- [`CGB.LanguageLifeKernel.Approximation.invariant_error`](../../lean/CGBAxioms/CGBAxioms/LanguageLifeKernel.lean#L884) [design / universal / reviewed]：A locally closed error budget stays uniform in elapsed sequence length.

## LearningMaintenance（26）
_A finite learning-and-maintenance mathematical witness Safe informative boundary feedback precedes resource-dependent action._

- [`CGB.LearningMaintenance.policy_table`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L39) [witness / universal / reviewed]
- [`CGB.LearningMaintenance.plant_admission`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L66) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.admission`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L103) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.reads_boundary`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L139) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.sum_configs`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L165) [witness / universal / reviewed]
- [`CGB.LearningMaintenance.kernel_table`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L174) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.natural_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L191) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.recovery_step`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L214) [witness / universal / reviewed]
- [`CGB.LearningMaintenance.normal_recovery`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L241) [witness / universal / reviewed]
- [`CGB.LearningMaintenance.natural_alive`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L247) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.natural_survival`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L263) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.starve_world_step`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L282) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.starve_recovery`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L294) [witness / universal / reviewed]
- [`CGB.LearningMaintenance.cut_move_early`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L314) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.cut_early_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L328) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.starve_mass_three`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L348) [witness / universal / reviewed]
- [`CGB.LearningMaintenance.starve_mass_monotone`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L358) [witness / universal / reviewed]
- [`CGB.LearningMaintenance.cut_alive_bound`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L384) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.cut_survival_mass_bound`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L399) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.cut_survival`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L426) [witness / candidate / reviewed]
- [`CGB.LearningMaintenance.maintenance_witness`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L439) [witness / candidate / reviewed]：The original gross-exchange goals hold during paid identification and all subsequent times.
- [`CGB.LearningMaintenance.starvation_depletes`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L452) [witness / candidate / reviewed]：Scarcity really changes the body reserve, under either process; vacuum then has no action in any stimulus.
- [`CGB.LearningMaintenance.sustained_exchange`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L468) [witness / candidate / reviewed]：Normal maintenance consumes and imports a positive unit EVERY active tick; this demand is a property of this mechanism, not a universal life axiom.
- [`CGB.LearningMaintenance.paid_learning`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L479) [witness / universal / reviewed]：The finite physical feedback update realizes the previous candidate filter, using the action chosen BEFORE feedback.
- [`CGB.LearningMaintenance.acquired_control`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L488) [witness / universal / reviewed]：Both branches use one common initial controller; actual feedback determines which singleton memory and specialized resource action it acquires.
- [`CGB.LearningMaintenance.normal_resources`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenance.lean#L497) [witness / candidate / reviewed]：Physical fuel and registered resource counts along the exact normal law.

## LearningMaintenanceControls（8）
_A paired write-disabled control: same initial fuel, sensing, world and clock._

- [`CGB.LearningMaintenanceControls.sum_configs`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L27) [witness / universal / reviewed]
- [`CGB.LearningMaintenanceControls.kernel_table`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L36) [witness / universal / reviewed]
- [`CGB.LearningMaintenanceControls.policy_value`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L53) [witness / universal / reviewed]
- [`CGB.LearningMaintenanceControls.frozen_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L59) [witness / universal / reviewed]
- [`CGB.LearningMaintenanceControls.starve_recovery`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L85) [witness / universal / reviewed]
- [`CGB.LearningMaintenanceControls.starve_alive`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L98) [witness / universal / reviewed]
- [`CGB.LearningMaintenanceControls.frozen_survival`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L108) [witness / universal / reviewed]：With writing disabled the initial two fuel units cannot preserve access to food.
- [`CGB.LearningMaintenanceControls.frozen_not_persists`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L126) [witness / universal / reviewed]

## LearningMaintenanceEnergy（4）
_Declared fuel, actual consumption and paid learning The same registered normal/cut event process pays maintenance fuel at active steps and an additional write-fuel unit on the effective feedback update._

- [`CGB.LearningMaintenanceEnergy.actual_law`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L29) [witness / candidate / reviewed]：The energy law is exactly the state law used in the viability comparison, including the cut's own distribution-dependent future, not a second process.
- [`CGB.LearningMaintenanceEnergy.body_supported`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L46) [witness / candidate / reviewed]
- [`CGB.LearningMaintenanceEnergy.isothermal`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L87) [witness / universal / reviewed]：Conditional mean second-law admission for the explicitly declared internal entropy and bath units.
- [`CGB.LearningMaintenanceEnergy.availability_budget`](../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceEnergy.lean#L110) [witness / universal / reviewed]：Every finite prefix uses the SAME event law and tick.

## Life（18）
_The infinite-stream life statement **Historical menu semantics (versioned 2026-10-23).** The `recover`, `aliveInd`, `Lives` and `LifeConjecture` below retain the old favourable-stimulus menu for compatibility._

- [`CGB.Medium.kernel_nonneg`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L104) [semantics / universal / reviewed]
- [`CGB.Medium.kernel_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L111) [semantics / universal / reviewed]：The kernel is stochastic.
- [`CGB.Medium.sum4`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L131) [semantics / universal / reviewed]
- [`CGB.Medium.law_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L138) [semantics / universal / reviewed]：The law is a probability distribution at every time.
- [`CGB.Medium.irrecoverable_absorbing`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L175) [semantics / universal / reviewed]：**Losing recoverability is irreversible**: no event leads from an irrecoverable state back to a recoverable one.
- [`CGB.Medium.kernel_up_zero`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L196) [semantics / universal / reviewed]：Without any possible intake, no step raises content.
- [`CGB.Medium.recover_eq_zero`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L211) [semantics / universal / reviewed]：Without any possible intake, recovery is impossible.
- [`CGB.Medium.aliveInd_eq_zero`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L229) [semantics / universal / reviewed]
- [`CGB.Medium.stepSum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L242) [semantics / universal / reviewed]：One joint step, weighted by a test function `a` that the organism kernel preserves.
- [`CGB.Medium.margStep`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L261) [semantics / universal / reviewed]：One joint step of the organism's marginal, when its kernel ignores the world.
- [`CGB.Medium.cutMove_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L276) [semantics / universal / reviewed]
- [`CGB.Medium.lawCut_sum`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L291) [semantics / universal / reviewed]
- [`CGB.Medium.never_alive_without_intake`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L311) [semantics / universal / reviewed]：An organism that can never take anything in is never alive, whatever it senses.
- [`CGB.Medium.not_persists_of_no_intake`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L332) [semantics / universal / reviewed]：Without any possible intake there is no persistence.
- [`CGB.Medium.not_selfMaintains_of_unexposed`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L367) [semantics / universal / reviewed]：An organism that ignores its stimuli cannot self-maintain: its own law does not depend on the world, so cutting the loop changes nothing.
- [`CGB.Medium.not_selfMaintains_of_closedMatter`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L397) [semantics / universal / reviewed]：An organism that exchanges no content with the world cannot self-maintain: whether it holds content never changes, in the loop or out of it.
- [`CGB.Medium.Lives.exposed`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L431) [semantics / universal / reviewed]：A living organism necessarily senses its world.
- [`CGB.Medium.Lives.metabolizes`](../../lean/CGBAxioms/CGBAxioms/Life.lean#L435) [semantics / universal / reviewed]：A living organism necessarily exchanges content with its world (metabolism).

## LifeControls（25）
_Definition controls prompted by passive absorption and frozen-copy objections The old reflex/self-model advantages remain withdrawn._

- [`CGB.FirstHit.during_zero_on_closed`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L20) [witness / universal / reviewed]：Protocol-dependent extension of zero_on_closed, also valid for the future marginal-ghost cut.
- [`CGB.Operational.recovery_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L44) [witness / universal / reviewed]：No possible positive-rate intake excludes recovery on BOTH registered future protocols, for every current joint state and every horizon.
- [`CGB.Operational.alive_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L62) [witness / universal / reviewed]
- [`CGB.Operational.survival_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L69) [witness / universal / reviewed]
- [`CGB.Operational.not_persists_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L74) [witness / universal / reviewed]
- [`CGB.Operational.not_selfMaintains_without_intake`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L81) [witness / universal / reviewed]
- [`CGB.Operational.not_persists_of_frozen`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L92) [witness / universal / reviewed]：This migrates the fully-frozen-copy exclusion to the actual-world target.
- [`CGB.Operational.not_persists_of_closed_moments_and_empty_boundary`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L104) [witness / universal / reviewed]：Direct downstream use of the new Noether bridge: if closed moment drifts vanish and there are no boundary participants, actual intake is impossible.
- [`CGB.Medium.kernel_from_frozen_state`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L126) [witness / universal / reviewed]：The full initial state need only be absorbing for all possible stimuli.
- [`CGB.Operational.recovery_zero_from_frozen_state`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L149) [witness / universal / reviewed]
- [`CGB.Operational.alive_zero_from_frozen_state`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L172) [witness / universal / reviewed]
- [`CGB.Operational.not_persists_of_frozen_initial`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L181) [witness / universal / reviewed]：Even holding material at birth cannot rescue an absorbing initial body.
- [`CGB.RecoveryControls.passive_body_axioms`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L199) [witness / universal / reviewed]：The existing passive-absorption control also satisfies all six physical clauses; its failure of the life target is not a violation of those clauses.
- [`CGB.RecoveryControls.passive_body_can_take_in`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L232) [witness / universal / reviewed]：Passive absorption really occurs, so this is stronger than a frozen body with no intake transition anywhere.
- [`CGB.RecoveryControls.passive_full_frozen`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L238) [witness / universal / reviewed]：Once full, the passive body's absorption stops in every environment.
- [`CGB.RecoveryControls.passive_full_not_alive`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L245) [witness / universal / reviewed]：The current actual-world alive test excludes the full passive state for both normal and cut futures, at every horizon and in every world.
- [`CGB.RecoveryControls.filledPassive_axioms`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L256) [witness / universal / reviewed]
- [`CGB.RecoveryControls.filledPassive_clock`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L260) [witness / universal / reviewed]
- [`CGB.RecoveryControls.filledPassive_not_persists`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L262) [witness / universal / reviewed]
- [`CGB.RecoveryControls.passive_intake_not_persistence`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L269) [witness / universal / reviewed]：Explicit definition control: physical admissibility, initial material, and an intake event somewhere do not imply actual persistence in this world.
- [`CGB.NoetherControls.seed_closed_symmetry`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L284) [witness / universal / reviewed]：Existing admissible seed: its closed-event content symmetry is certified.
- [`CGB.NoetherControls.toggle_fails_closed_moments`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L289) [witness / universal / reviewed]：Existing nonconserving closed toggle is rejected by the new moment bridge.
- [`CGB.NoetherControls.meanOnly_first_drift`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L303) [witness / universal / reviewed]
- [`CGB.NoetherControls.meanOnly_second_drift`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L307) [witness / universal / reviewed]
- [`CGB.NoetherControls.meanOnly_not_conserved`](../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L310) [witness / universal / reviewed]

## MaterialExchange（8）
_Versioned gross material exchange observer The old net-growth observer remains intact._

- [`CGB.Medium.ledger_p2a`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L26) [witness / universal / reviewed]
- [`CGB.Medium.net_increase_implies_incoming`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L34) [witness / universal / reviewed]：A positive net increase necessarily includes incoming material.
- [`CGB.Medium.equal_content_iff_balanced_exchange`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L40) [witness / universal / reviewed]
- [`CGB.Exchange.recovery_bounds`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L66) [witness / universal / reviewed]
- [`CGB.Exchange.recovery_zero_without_incoming`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L71) [witness / universal / reviewed]
- [`CGB.Exchange.survival_bounds`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L90) [witness / universal / reviewed]
- [`CGB.Exchange.not_persists_without_incoming`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L112) [necessity / world_contract / reviewed]
- [`CGB.Description.ExchangeBook.one_percent_budget`](../../lean/CGBAxioms/CGBAxioms/MaterialExchange.lean#L162) [witness / universal / reviewed]

## MaterialSource（6）
_Source-resolved material transport Fixed token identities locate material in the body, accessible source, return sink or exterior._

- [`CGB.MaterialSource.balance`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L43) [witness / universal / reviewed]：One transport law yields both body balance and the accessible-source budget.
- [`CGB.MaterialSource.moved_le_stock`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L61) [witness / universal / reviewed]
- [`CGB.MaterialSource.no_intake_from_empty_source`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L69) [witness / universal / reviewed]：An empty accessible source cannot feed the body in this tick.
- [`CGB.MaterialSource.no_transport_without_relocation`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L76) [witness / universal / reviewed]：With identities fixed, an unchanged ownership state cannot be labelled as positive gross transport.
- [`CGB.MaterialSource.source_budget`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L86) [witness / universal / reviewed]：Exact finite-prefix stock law, applicable to each supported sample path.
- [`CGB.MaterialSource.intake_bound_without_refill`](../../lean/CGBAxioms/CGBAxioms/MaterialSource.lean#L103) [witness / universal / reviewed]：A finite unreplenished, unrecycled source bounds cumulative intake; this is a material-throughput conclusion, not a death/free-energy theorem.

## MaterialSourceControls（3）
_Finite supply / exchange / exhaustion control A mathematical carrier-interpretation witness, not a bacterial/sponge model or a capability experiment._

- [`CGB.MaterialSourceControls.realized`](../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L62) [witness / universal / reviewed]：Explicit source realization: no changed body dynamics, no extra physical ticks, and no nonempty-body false positive standing in for actual replacement.
- [`CGB.MaterialSourceControls.supply_exchange_exhaustion`](../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L100) [witness / universal / reviewed]：One arrival, then one equal input/output replacement, then no accessible source.
- [`CGB.MaterialSourceControls.no_future_intake_after_exhaustion`](../../lean/CGBAxioms/CGBAxioms/MaterialSourceControls.lean#L118) [witness / universal / reviewed]：Reuse the existing refinement/first-hit theorem for the WHOLE future.

## Medium（5）
_The CGB medium and the six axiom clauses The full language of `docs/information_boltzmann/CGB_AXIOMS_V0_20261022.md` §1–2._

- [`CGB.Medium.total_of_reachable`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L161) [semantics / universal / reviewed]：P2a confines the closed dynamics to a content shell.
- [`CGB.Medium.finite_reachable`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L172) [semantics / universal / reviewed]：**T0a** conservation implies finiteness: under P2a and the first part of P1a, the closed dynamics reaches only finitely many states.
- [`CGB.Medium.finite_states`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L183) [semantics / universal / reviewed]：With the capacity bound the whole state space is finite: the premise "finite state space" of the life definition.
- [`CGB.Medium.energy_of_p2b`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L192) [semantics / universal / reviewed]：**T-E** for media: P2b gives an energy function of the equilibrium dynamics.
- [`CGB.Medium.p2b_of_micro`](../../lean/CGBAxioms/CGBAxioms/Medium.lean#L199) [semantics / universal / reviewed]：Microscopic reversibility: if in the equilibrium environment every effective event is undone by its reverse event at the same rate, P2b holds (with uniform weights).

## Models（57）
_Consistency models, independence witnesses and the bottleneck theorem **Consistency.** * `twinFlip`: closed._

- [`CGB.Models.xhopAct_of_ok`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L54) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_of_not`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L57) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_rev_ok`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L60) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_back`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L65) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_sum`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L71) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_outside`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L78) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_congr`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L83) [witness / universal / reviewed]
- [`CGB.Models.symm_swap`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L88) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_perm`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L97) [witness / universal / reviewed]
- [`CGB.Models.xhopAct_ind`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L108) [witness / universal / reviewed]：The content of `a` alone, moved by a hop, can only have gone along `a → b`.
- [`CGB.Models.ident_swap`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L121) [witness / universal / reviewed]
- [`CGB.Models.eq_homogeneous`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L130) [witness / universal / reviewed]：Homogeneous contact by equality: any swap is a symmetry.
- [`CGB.Models.bool_p1a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L135) [witness / universal / reviewed]：Occupancy content: a site is empty (the vacuum) or holds one unit.
- [`CGB.Models.xhop_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L165) [witness / universal / reviewed]
- [`CGB.Models.xhop_p1b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L170) [witness / universal / reviewed]
- [`CGB.Models.xhop_p2a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L180) [witness / universal / reviewed]
- [`CGB.Models.xhop_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L183) [witness / universal / reviewed]
- [`CGB.Models.xhop_p2c`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L193) [witness / universal / reviewed]
- [`CGB.Models.xhop_senses`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L200) [witness / universal / reviewed]
- [`CGB.Models.xhop_pages`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L206) [witness / universal / reviewed]：**Bottleneck theorem.** With all units in contact, P2c forces each conductance to depend only on the identity pages of its two ends.
- [`CGB.Models.solo_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L271) [witness / universal / reviewed]：Every permutation of units is a symmetry of `solo`.
- [`CGB.Models.solo_p1b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L274) [witness / universal / reviewed]
- [`CGB.Models.solo_p2a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L287) [witness / universal / reviewed]：Local content conservation of each event gives P2a.
- [`CGB.Models.solo_p2c`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L295) [witness / universal / reviewed]
- [`CGB.Models.solo_p3`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L304) [witness / universal / reviewed]
- [`CGB.Models.solo_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L308) [witness / universal / reviewed]：Local microscopic reversibility of each unit gives P2b.
- [`CGB.Models.flip12_flip12`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L328) [witness / universal / reviewed]
- [`CGB.Models.flipCost_flip12`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L330) [witness / universal / reviewed]
- [`CGB.Models.flipCost_p1a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L332) [witness / universal / reviewed]
- [`CGB.Models.twinFlip_axioms`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L343) [witness / universal / reviewed]
- [`CGB.Models.twinFlip_swap_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L347) [witness / universal / reviewed]
- [`CGB.Models.rock_axioms`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L355) [witness / universal / reviewed]
- [`CGB.Models.rock_frozen`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L360) [witness / universal / reviewed]：The rock is frozen: no transition ever has positive rate, in any environment.
- [`CGB.Models.ringAdj_refl`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L371) [witness / universal / reviewed]
- [`CGB.Models.ringAdj_symm`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L373) [witness / universal / reviewed]
- [`CGB.Models.ringAdj_shift`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L375) [witness / universal / reviewed]
- [`CGB.Models.seed_p1b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L409) [witness / universal / reviewed]
- [`CGB.Models.seed_p2a`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L452) [witness / universal / reviewed]
- [`CGB.Models.seed_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L461) [witness / universal / reviewed]
- [`CGB.Models.seed_aut_fix`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L480) [witness / universal / reviewed]：Every symmetry of the seed fixes the source and the sink.
- [`CGB.Models.seed_p2c`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L490) [witness / universal / reviewed]
- [`CGB.Models.seed_p3`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L517) [witness / universal / reviewed]
- [`CGB.Models.seed_axioms`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L529) [witness / universal / reviewed]
- [`CGB.Models.seed_swap_isAut`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L533) [witness / universal / reviewed]：The reflection swapping the interior sites is a genuine symmetry.
- [`CGB.Models.seed_driven`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L541) [witness / universal / reviewed]：The seed is driven: under food its open dynamics has no detailed balance.
- [`CGB.Models.seed_grows`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L567) [witness / universal / reviewed]：**Growth.** From one unit at the source, under food, the seed reaches a state with three units: intake and spreading create units at empty sites.
- [`CGB.Models.twinTape_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L593) [witness / universal / reviewed]
- [`CGB.Models.twinBag_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L629) [witness / universal / reviewed]
- [`CGB.Models.twinBlank_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L657) [witness / universal / reviewed]
- [`CGB.Models.twinToggle_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L675) [witness / universal / reviewed]
- [`CGB.Models.twinCycle_not_p2b`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L699) [witness / universal / reviewed]
- [`CGB.Models.twinCycle_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L721) [witness / universal / reviewed]
- [`CGB.Models.perm_fin2_succ`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L746) [witness / universal / reviewed]
- [`CGB.Models.farFlip_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L754) [witness / universal / reviewed]
- [`CGB.Models.triLookup_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L783) [witness / universal / reviewed]
- [`CGB.Models.pairPreset_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L798) [witness / universal / reviewed]
- [`CGB.Models.innerClock_witness`](../../lean/CGBAxioms/CGBAxioms/Models.lean#L821) [witness / universal / reviewed]

## NestedLife（2）
_Life of a nested organism Once the organism is part of one medium with its world (`Nesting`), it has no outside stimulus: what it senses is the state of the world next to it, and what it takes in comes out of the world._

- [`CGB.Part.irrecoverable_absorbing`](../../lean/CGBAxioms/CGBAxioms/NestedLife.lean#L52) [witness / candidate / reviewed]
- [`CGB.Part.no_intake_of_closed`](../../lean/CGBAxioms/CGBAxioms/NestedLife.lean#L58) [witness / candidate / reviewed]：An organism that never exchanges content cannot take anything in, so nothing is recoverable for it: metabolism is necessary for nested life.

## Nesting（8）
_Nesting: an organism is a part of one medium, its boundary induced by contact There is one medium._

- [`CGB.Medium.boundary_eq_outer_compl`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L44) [semantics / candidate / reviewed]：`S`'s own boundary: the units of `S` touching the outside.
- [`CGB.Medium.participant_near`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L49) [semantics / candidate / reviewed]：A participant of an event that also has a participant in `S` is in `S` or touches it.
- [`CGB.Medium.blanket_of_contact`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L57) [semantics / candidate / reviewed]：**T-N1 (the blanket from contact)**: two states that agree on `S` and on the units touching it give every event with a participant in `S` the same rate and the same effect on `S`.
- [`CGB.Medium.untouched_unchanged`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L71) [semantics / candidate / reviewed]：Events with no participant in `S` leave `S` unchanged.
- [`CGB.Medium.organism_blanket`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L76) [semantics / candidate / reviewed]：The organism's half: what happens to `O` reads only `O` and the units touching it.
- [`CGB.Medium.world_blanket`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L83) [semantics / candidate / reviewed]：The world's half (formerly the assumption `ReadsBoundary`): what happens to the rest reads only the rest and the organism's own boundary units.
- [`CGB.Medium.exchange_conserves`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L96) [semantics / candidate / reviewed]：**T-N2 (exchange conserves)**: a closed event with positive rate changes the content of `O` by exactly minus the change of the rest.
- [`CGB.JumpProcess.freezeOutside_balance`](../../lean/CGBAxioms/CGBAxioms/Nesting.lean#L127) [semantics / candidate / reviewed]：**T-N3 (the organism's own moves are balanced with the world frozen)**: if the joint calm dynamics has detailed-balance weight `π`, the dynamics with the outside of `O` frozen has the same detailed balance.

## NoetherBridge（8）
_Noether correspondence for the actual medium event language A closed-event restriction is an analysis object, not a change to the running medium._

- [`CGB.JumpProcess.observableSymmetry_iff_conserved`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L24) [semantics / universal / reviewed]
- [`CGB.Medium.jump_gen_events`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L52) [semantics / universal / reviewed]：Sum event contributions before aggregating transitions with the same output.
- [`CGB.Medium.closedJump_conserved_iff`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L80) [semantics / universal / reviewed]
- [`CGB.Medium.p2a_iff_closed_conserved`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L120) [semantics / universal / reviewed]
- [`CGB.Medium.p2a_iff_closed_moments`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L133) [semantics / universal / reviewed]
- [`CGB.Medium.p2a_iff_closed_expected_drifts`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L143) [semantics / universal / reviewed]：Every normalized initial distribution has zero first and second moment DRIFT.
- [`CGB.Medium.p2a_iff_closed_observable_symmetry`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L155) [semantics / universal / reviewed]
- [`CGB.Medium.content_drift_boundary`](../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L164) [semantics / universal / reviewed]：In the original OPEN generator, all drift of any function of total content comes from boundary-touching events.

## OperationalLife（6）
_Versioned operational target: finite descriptions and actual-world intake The old LifeConjecture remains a historical statement, not a proved equivalent._

- [`CGB.Viability.alive_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L23) [witness / candidate / reviewed]
- [`CGB.Viability.survival_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L32) [witness / candidate / reviewed]
- [`CGB.Operational.recovery_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L68) [witness / candidate / reviewed]
- [`CGB.Operational.natural_recovery`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L74) [witness / candidate / reviewed]：For the natural uncut world, the general protocol reduces to actualRecover.
- [`CGB.Operational.survival_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L89) [witness / candidate / reviewed]
- [`CGB.Operational.persistence_parameter_bounds`](../../lean/CGBAxioms/CGBAxioms/OperationalLife.lean#L113) [witness / candidate / reviewed]：Parameter bounds follow from an actual persistence witness, not extra axioms.

## PortContractInstances（3）
_The current 24-state maintenance witness is admitted through the coarse contracts._

- [`CGB.PortContractInstances.split_factorizes`](../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L41) [witness / candidate / reviewed]
- [`CGB.PortContractInstances.reserve_contract`](../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L46) [witness / candidate / reviewed]
- [`CGB.PortContractInstances.admission_contract`](../../lean/CGBAxioms/CGBAxioms/PortContractInstances.lean#L52) [witness / candidate / reviewed]

## PortContracts（6）
_Coarse contracts for separated sensory and resource ports Complex bodies need not expose every carrier identity._

- [`CGB.PortContracts.protocol_sensor`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L25) [witness / universal / reviewed]
- [`CGB.PortContracts.protocol_resource`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L30) [witness / universal / reviewed]
- [`CGB.PortContracts.admission_port_factorizes`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L47) [witness / universal / reviewed]
- [`CGB.PortContracts.admission_reserve_contract`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L53) [witness / universal / reviewed]
- [`CGB.PortContracts.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L69) [witness / universal / reviewed]
- [`CGB.PortContracts.elapsed_preserved`](../../lean/CGBAxioms/CGBAxioms/PortContracts.lean#L81) [witness / universal / reviewed]

## PortFlowHomology（3）
_Finite flow chain for the separated port witness Each current state/input pair is a directed transition edge._

- [`CGB.PortFlowHomology.incidence_augmented_zero`](../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L30) [semantics / universal / reviewed]
- [`CGB.PortFlowHomology.weighted_incidence_augmented_zero`](../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L43) [semantics / universal / reviewed]
- [`CGB.PortFlowHomology.weighted_flow_boundary_zero`](../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L59) [semantics / universal / reviewed]

## PortSeparation（6）
_Independent sensory and resource ports on the same finite body Absent sensing with resources supplied is a noninformative maintenance inlet._

- [`CGB.PortSeparation.local_budgets`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L54) [witness / universal / reviewed]
- [`CGB.PortSeparation.starvation_progress`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L65) [witness / universal / reviewed]
- [`CGB.PortSeparation.next_empty`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L74) [witness / universal / reviewed]
- [`CGB.PortSeparation.admission`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L92) [witness / universal / reviewed]
- [`CGB.PortSeparation.legacy_rules`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L129) [witness / universal / reviewed]
- [`CGB.PortSeparation.actual_transition`](../../lean/CGBAxioms/CGBAxioms/PortSeparation.lean#L148) [witness / universal / reviewed]

## RecoveryControls（5）
_Recovery semantic controls (not organism capability experiments) Identical body and stimulus menus; only world accessibility differs._

- [`CGB.RecoveryControls.sum_functions`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L41) [witness / universal / reviewed]
- [`CGB.RecoveryControls.body_kernel`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L50) [witness / universal / reviewed]
- [`CGB.RecoveryControls.body_clock`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L60) [witness / universal / reviewed]
- [`CGB.RecoveryControls.oracle_ignores_access`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L72) [witness / universal / reviewed]
- [`CGB.RecoveryControls.actual_recovery_distinguishes_world_moves`](../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L78) [witness / universal / reviewed]

## RecoveryCriteria（4）
_Local sufficient criteria for actual resource recovery Regions and ranks are proof objects, not new organism state, policy inputs or life axioms._

- [`CGB.RecoveryCriteria.ranked_firstHit`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L28) [semantics / universal / reviewed]：A local rank certificate proves recovery in the actual future kernel, without assuming a favourable stimulus sequence or the desired recovery score.
- [`CGB.RecoveryCriteria.firstHit_zero_of_closed`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L54) [semantics / universal / reviewed]
- [`CGB.RecoveryCriteria.support_closed`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L74) [semantics / universal / reviewed]
- [`CGB.RecoveryCriteria.survival_of_ranked`](../../lean/CGBAxioms/CGBAxioms/RecoveryCriteria.lean#L104) [semantics / universal / reviewed]：Reusable sufficient criterion for the original Exchange score.

## RecoveryTransport（3）
_Recovery and state-score transport through existing exact refinements State-dependent intake must match on supported transitions, in addition to the joint event/state law and body content._

- [`CGB.RecoveryTransport.mean_next`](../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L15) [semantics / universal / reviewed]
- [`CGB.RecoveryTransport.firstHit_eq`](../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L32) [semantics / universal / reviewed]：Supported actual-intake correspondence preserves every recovery horizon, including in a distribution-dependent cut supplied as a separate kernel.
- [`CGB.RecoveryTransport.mean_eq`](../../lean/CGBAxioms/CGBAxioms/RecoveryTransport.lean#L61) [semantics / universal / reviewed]：Exact state-score expectations at every time, without assuming that the score is a probability or replacing physical state by output histories.

## RenewableCandidateAudit（9）
_A concrete obstruction to the missing sensory-interruption obligation Keep the original body, world, clock and interruption protocol._

- [`CGB.RenewableCandidateAudit.kernel_at_next`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L30) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.policy_values`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L37) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.edge`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L40) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.step_lower`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L51) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.path_lower`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L78) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.interrupted_trap_mass`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L90) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.trap_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L103) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.interruption_failure`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L127) [witness / universal / reviewed]
- [`CGB.RenewableCandidateAudit.no_positive_interruption_tolerance`](../../lean/CGBAxioms/CGBAxioms/RenewableCandidateAudit.lean#L165) [witness / universal / reviewed]

## RenewableEncoding（13）
_Concrete rule-book compilation and honest candidate boundaries The original one-site rules are compiled into the EXISTING tabular language._

- [`CGB.RenewableEncoding.code_roundtrip`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L56) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.book_layout`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L58) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.row_lookup`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L62) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.transfer_lookup`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L65) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.compiled_rows`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L72) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.compiled_rates`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L82) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.data_balance`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L92) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.accountable`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L99) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.description_positive`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L137) [witness / universal / reviewed]：Necessary bit count for EVERY book under the registered convention.
- [`CGB.RenewableEncoding.small_reference_obstruction`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L141) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.no_one_percent`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L147) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.world_richness_limit`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L151) [witness / universal / reviewed]
- [`CGB.RenewableEncoding.no_encoded_candidate`](../../lean/CGBAxioms/CGBAxioms/RenewableEncoding.lean#L160) [witness / universal / reviewed]：A full encoded subtarget cannot be obtained by attaching the already proved persistence/source certificates to this one-site book.

## RenewableLearning（6）
_Renewable paid learning in a continuously changing finite world Resource access can fail after an unannounced regime change._

- [`CGB.RenewableLearning.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L69) [witness / universal / reviewed]
- [`CGB.RenewableLearning.local_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L145) [witness / universal / reviewed]
- [`CGB.RenewableLearning.learning_contract`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L153) [witness / universal / reviewed]：Paid reopening is driven by a contradictory consequence, and the ensuing probe uses exactly the previously verified candidate-screening update.
- [`CGB.RenewableLearning.admission`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L177) [witness / candidate / reviewed]
- [`CGB.RenewableLearning.sum_env`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L235) [witness / universal / reviewed]
- [`CGB.RenewableLearning.actual_law`](../../lean/CGBAxioms/CGBAxioms/RenewableLearning.lean#L280) [witness / candidate / reviewed]

## RenewableLearningCarrier（10）
_Source and recovery certificates for the unchanged renewable-learning process Fixed material identities realize every supported normal/cut transition._

- [`CGB.RenewableLearningCarrier.body_supported`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L23) [witness / candidate / reviewed]
- [`CGB.RenewableLearningCarrier.realized`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L72) [witness / candidate / reviewed]
- [`CGB.RenewableLearningCarrier.intake_matches`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L87) [witness / candidate / reviewed]
- [`CGB.RenewableLearningCarrier.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L97) [witness / candidate / reviewed]
- [`CGB.RenewableLearningCarrier.recovery_by_certificate`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L104) [witness / candidate / reviewed]
- [`CGB.RenewableLearningCarrier.score_preserved`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L119) [witness / candidate / reviewed]
- [`CGB.RenewableLearningCarrier.maintenance_preserved`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L144) [witness / candidate / reviewed]
- [`CGB.RenewableLearningCarrier.stock_and_recharge`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L186) [witness / universal / reviewed]
- [`CGB.RenewableLearningCarrier.initial_energy`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L197) [witness / universal / reviewed]：The extra source stock is external to the old two-unit body at birth.
- [`CGB.RenewableLearningCarrier.source_energy_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningCarrier.lean#L207) [witness / universal / reviewed]

## RenewableLearningClosure（19）
_Close the ORIGINAL distribution-dependent marginal-ghost comparison._

- [`CGB.RenewableLearningClosure.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L16) [witness / universal / reviewed]
- [`CGB.RenewableLearningClosure.sum_cells`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L22) [witness / universal / reviewed]
- [`CGB.RenewableLearningClosure.sum_configs`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L29) [witness / universal / reviewed]
- [`CGB.RenewableLearningClosure.sum_env`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L37) [witness / universal / reviewed]
- [`CGB.RenewableLearningClosure.kernel_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L46) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.early_move`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L70) [witness / candidate / reviewed]：Until tick four the body marginal is a point mass, so the ORIGINAL cut transition is derived from that marginal, not assigned a convenient law.
- [`CGB.RenewableLearningClosure.cut_early`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L83) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.move_five`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L100) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.trapped_next`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L113) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.trapped_six`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L124) [witness / universal / reviewed]
- [`CGB.RenewableLearningClosure.trap_moves`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L130) [witness / universal / reviewed]
- [`CGB.RenewableLearningClosure.trap_world_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L134) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.trapped_monotone`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L148) [witness / universal / reviewed]
- [`CGB.RenewableLearningClosure.trap_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L163) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.cut_alive`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L180) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.cut_mass_bound`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L193) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.cut_survival`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L216) [witness / candidate / reviewed]：The original cut loses at least a quarter of viable mass by tick six; absorption extends the bound to ALL later times, including future cut marginals.
- [`CGB.RenewableLearningClosure.maintenance`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L225) [witness / candidate / reviewed]
- [`CGB.RenewableLearningClosure.cut_seed_mass`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningClosure.lean#L233) [witness / candidate / reviewed]：Exact seed mass for the absorbing failure region under the original cut.

## RenewableLearningControls（15）
_Finite-path probabilities are products of the actual registered step law._

- [`CGB.RenewableLearningControls.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L13) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.cycle_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L35) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.cycle_payment`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L42) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.path_after`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L54) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.path_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L56) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.path_active`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L80) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.write_blocks`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L94) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.arbitrary_relearning`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L126) [witness / universal / reviewed]：Every requested number of paid A-to-B-to-A cycles has a positive-probability finite prefix of this one process, with explicit equal credit replenishment.
- [`CGB.RenewableLearningControls.law_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L135) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.path_law_bound`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L148) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.failure_weight`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L171) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.trap_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L178) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.trap_alive`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L191) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.survival_mass_bound`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L199) [witness / universal / reviewed]
- [`CGB.RenewableLearningControls.paired_failure`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningControls.lean#L231) [witness / universal / reviewed]：Blocking either paid reopening or storage renewal creates a half-probability loss of viability by tick six, under the same world, initial state and food rule.

## RenewableLearningEnergy（5）
_The total fuel account contains both buffer and write credit._

- [`CGB.RenewableLearningEnergy.supported`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L16) [witness / candidate / reviewed]
- [`CGB.RenewableLearningEnergy.allocation`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L81) [witness / universal / reviewed]：Allocating credit uses part of the specified food input.
- [`CGB.RenewableLearningEnergy.telescope`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L86) [witness / universal / reviewed]
- [`CGB.RenewableLearningEnergy.fuel_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L107) [witness / universal / reviewed]
- [`CGB.RenewableLearningEnergy.credit_budget`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningEnergy.lean#L114) [witness / universal / reviewed]

## RenewableLearningLife（15）
_All-time viability in the same changing body/world process._

- [`CGB.RenewableLearningLife.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L19) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.nonvacuum`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L26) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.active`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L29) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.next_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L33) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.supported_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L39) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.law_invariant`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L61) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.reachable_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L82) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.sum_configs`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L91) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.sum_env`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L95) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.recovery_step`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L106) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.immediate_recovery`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L117) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.recovery_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L123) [witness / universal / reviewed]
- [`CGB.RenewableLearningLife.alive_good`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L143) [witness / candidate / reviewed]
- [`CGB.RenewableLearningLife.natural_survival`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L162) [witness / candidate / reviewed]：Every supported continuation remains viable through missed food, paid reopening, probing, identification and actual resource-funded renewal.
- [`CGB.RenewableLearningLife.persistence`](../../lean/CGBAxioms/CGBAxioms/RenewableLearningLife.lean#L189) [witness / candidate / reviewed]

## RenewableRecoveryCertificate（7）
_Instantiate the reusable local recovery criterion on the existing model._

- [`CGB.RenewableRecoveryCertificate.policy_table`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L12) [witness / universal / reviewed]
- [`CGB.RenewableRecoveryCertificate.nonvacuum`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L19) [witness / universal / reviewed]
- [`CGB.RenewableRecoveryCertificate.active`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L22) [witness / universal / reviewed]
- [`CGB.RenewableRecoveryCertificate.next_good`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L26) [witness / universal / reviewed]
- [`CGB.RenewableRecoveryCertificate.supported_good`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L32) [witness / universal / reviewed]
- [`CGB.RenewableRecoveryCertificate.unmarked_progress`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L55) [witness / universal / reviewed]
- [`CGB.RenewableRecoveryCertificate.survival_by_criterion`](../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L94) [witness / universal / reviewed]：The original all-time score now also follows from the shared local criterion; it is not inserted as a premise of that criterion.

## RuleEncoding（11）
_A fixed finite rule language and an honest description budget This is a representation layer, not a seventh physical axiom._

- [`CGB.Description.code_roundtrip`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L52) [semantics / universal / reviewed]
- [`CGB.Description.different_books_different_codes`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L55) [semantics / universal / reviewed]
- [`CGB.Description.bounded_books`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L60) [semantics / universal / reviewed]：At most 2^B distinct descriptions fit in B bits (an intentionally loose bound).
- [`CGB.Description.one_percent_budget`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L81) [semantics / universal / reviewed]
- [`CGB.Description.WiringReference.payload_length`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L94) [semantics / universal / reviewed]
- [`CGB.Description.Book.findWithin_complete`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L119) [semantics / universal / reviewed]
- [`CGB.Description.Book.updateWithin_complete`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L153) [semantics / universal / reviewed]
- [`CGB.Description.Book.weight_nonneg`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L161) [semantics / universal / reviewed]
- [`CGB.Description.Book.same_code_same_medium`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L189) [semantics / universal / reviewed]：Equality of codes entails equality of the whole interpreted instance.
- [`CGB.Description.Book.unchanged_outside`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L195) [semantics / universal / reviewed]：Locality of effects is a theorem of the interpreter, not a user-supplied promise.
- [`CGB.Description.Book.admits_some_clock`](../../lean/CGBAxioms/CGBAxioms/RuleEncoding.lean#L208) [semantics / universal / reviewed]：Finite alphabets and this finite rulebook admit some uniform clock.

## ScheduledRecovery（3）
_Time-dependent certificates for a registered recovery protocol Regions and ranks may refer to external protocol time._

- [`CGB.ScheduledRecovery.ranked_firstHit`](../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L28) [witness / universal / reviewed]：A local rank certificate proves recovery in the actual future kernel, without assuming a favourable stimulus sequence or the desired recovery score.
- [`CGB.ScheduledRecovery.support_closed`](../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L54) [witness / universal / reviewed]
- [`CGB.ScheduledRecovery.survival_of_ranked`](../../lean/CGBAxioms/CGBAxioms/ScheduledRecovery.lean#L84) [witness / universal / reviewed]：Reusable sufficient criterion for the original Exchange score.

## SensoryMaintenance（16）
_Sensory loss does not suspend physical maintenance `some calm` remains the equilibrium stimulus._

- [`CGB.SensoryMaintenance.local_budgets`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L60) [witness / universal / reviewed]
- [`CGB.SensoryMaintenance.transfer_bounds`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L68) [witness / universal / reviewed]
- [`CGB.SensoryMaintenance.active_loss`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L75) [witness / universal / reviewed]
- [`CGB.SensoryMaintenance.missing_no_intake`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L84) [witness / universal / reviewed]
- [`CGB.SensoryMaintenance.next_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L87) [witness / universal / reviewed]
- [`CGB.SensoryMaintenance.admission`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L102) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.equilibrium_distinct_from_loss`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L154) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.actual_transition`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L161) [witness / universal / reviewed]
- [`CGB.SensoryMaintenance.actual_law`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L183) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.kernel_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L194) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.burn_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L210) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.burn_three`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L218) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.burn_all_later`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L222) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.funded_two_tick_bridge`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L228) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenance.path_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L234) [witness / universal / reviewed]：Exact actual-channel fuel balance for every registered input path.
- [`CGB.SensoryMaintenance.path_credit_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenance.lean#L246) [witness / universal / reviewed]

## SensoryMaintenanceBoundaries（3）
_Exact local blackout boundary and recharge condition The two-loss boundary is a state-level result: a supported nonvacuum cell has zero reserve but remains nonempty after two absent-sensing ticks, and reaches the absorbi_

- [`CGB.SensoryMaintenanceBoundaries.two_tick_functional_boundary`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L16) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceBoundaries.sensed_recharge`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L29) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceBoundaries.two_tick_recharge_condition`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceBoundaries.lean#L41) [witness / candidate / reviewed]

## SensoryMaintenanceCarrier（12）
_Two existing finite material buffers realize structural fuel and the new reserve._

- [`CGB.SensoryMaintenanceCarrier.component_rules`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L28) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceCarrier.component_sums`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L36) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceCarrier.coarse_positive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L78) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceCarrier.count_sum`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L117) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceCarrier.moved_sum`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L125) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceCarrier.realized`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L136) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceCarrier.source_stock`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L167) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceCarrier.recharge_paid`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L176) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceCarrier.recovery_preserved`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L195) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceCarrier.score_preserved`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L212) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceCarrier.initial_energy`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L260) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceCarrier.source_energy_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceCarrier.lean#L269) [witness / universal / reviewed]

## SensoryMaintenanceClosure（11）
_The original marginal-ghost cut includes the new physical reserve._

- [`CGB.SensoryMaintenanceClosure.sum_old`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L10) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceClosure.sum_new`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L13) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceClosure.awake_projection`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L19) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceClosure.old_kernel`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L24) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceClosure.kernel_marginal`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L39) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceClosure.cut_move`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L50) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceClosure.cut_marginal`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L67) [witness / candidate / reviewed]：Recompute the actual cut with the augmented body's own marginal; summing its physical reserve gives exactly the old cut's memory/world law.
- [`CGB.SensoryMaintenanceClosure.vacuum_target`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L100) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceClosure.vacuum_reserve_zero`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L106) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceClosure.empty_mass`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L127) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceClosure.cut_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceClosure.lean#L138) [witness / candidate / reviewed]

## SensoryMaintenanceLimits（7）
_Finite blackout loss and actual-event energy accounting._

- [`CGB.SensoryMaintenanceLimits.support_closed`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L10) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceLimits.loss_closed`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L36) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceLimits.blackout_body_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L63) [witness / candidate / reviewed]：After three consecutive absent-sensing ticks, every supported body is empty.
- [`CGB.SensoryMaintenanceLimits.blackout_survival_zero`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L81) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceLimits.long_loss_not_tolerated`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L98) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceLimits.no_input_active_duration`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L108) [witness / universal / reviewed]：A declared positive loss at every active tick yields a finite duration bound from the actual first law.
- [`CGB.SensoryMaintenanceLimits.fuel_budget`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceLimits.lean#L155) [witness / universal / reviewed]

## SensoryMaintenanceProtocolRepairRefutation（4）

- [`CGB.SensoryMaintenanceProtocolRepairRefutation.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L7) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceProtocolRepairRefutation.path_step`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L29) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceProtocolRepairRefutation.protected_protocol_empty_mass_positive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L44) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceProtocolRepairRefutation.protected_protocol_survival_lt_one`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceProtocolRepairRefutation.lean#L63) [witness / candidate / reviewed]

## SensoryMaintenanceRecovery（10）

- [`CGB.SensoryMaintenanceRecovery.intake_same`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L14) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceRecovery.old_event`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L22) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRecovery.supported`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L34) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRecovery.sensed_not_calm`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L87) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceRecovery.decode_protocol`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L90) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceRecovery.positive_region`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L96) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceRecovery.reserve_sensed`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L104) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceRecovery.one_tick_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L154) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRecovery.natural_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L187) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRecovery.one_tick_tolerance`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRecovery.lean#L205) [witness / universal / reviewed]

## SensoryMaintenanceRepairAudit（3）
_Audit of state-only repairs after the two-tick refutation The existing material ports make the `aSpent`/`miss` failure structurally forced._

- [`CGB.SensoryMaintenanceRepairAudit.cost_zero_iff_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L16) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceRepairAudit.spent_miss_budget_forces_empty`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L23) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceRepairAudit.no_state_only_two_tick_repair`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairAudit.lean#L42) [witness / universal / reviewed]

## SensoryMaintenanceRepairCandidates（25）
_Minimal repair candidates after the two-tick refutation The registered model is unchanged._

- [`CGB.SensoryMaintenanceRepairCandidates.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L16) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.protected_second_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L26) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.protected_first_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L30) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.protocol_repair_preserves_one_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L34) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.probe_recharges_after_one_loss`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L38) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.protected_probe_path_outgoing`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L53) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.protected_probe_path_incoming`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L66) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.protected_branch_excludes_local_blackout`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L78) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.initial_guarded_two_tick_start`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L102) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_third_tick_recharges`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L116) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_repair_preserves_one_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L149) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.firstHit_three_tick_layered`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L153) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_firstHit`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L216) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_alive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L313) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_two_tick_local_tolerance`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L333) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_survival_at`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L344) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_support_zero`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L396) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_support_path_step`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L421) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.not_guarded_support_two`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L436) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.guarded_initial_survival`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L460) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_excludes_spent`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L466) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_excludes_verified_dangerous_branch`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L472) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_two_loss_preserves_cell`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L479) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.eligible_keeps_existing_budget_contract`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L491) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceRepairCandidates.region_repair_preserves_one_tick`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceRepairCandidates.lean#L498) [witness / candidate / reviewed]

## SensoryMaintenanceTwoTickGlobal（7）

- [`CGB.SensoryMaintenanceTwoTickGlobal.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L9) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceTwoTickGlobal.stateLaw_nonneg`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L14) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceTwoTickGlobal.stateLaw_pos_of_path`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L28) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceTwoTickGlobal.path_step`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L79) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceTwoTickGlobal.empty_mass_positive`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L92) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceTwoTickGlobal.two_tick_survival_lt_one`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L108) [witness / candidate / reviewed]
- [`CGB.SensoryMaintenanceTwoTickGlobal.not_two_tick_tolerance`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickGlobal.lean#L169) [witness / universal / reviewed]

## SensoryMaintenanceTwoTickObstruction（3）

- [`CGB.SensoryMaintenanceTwoTickObstruction.policy_table`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L7) [witness / universal / reviewed]
- [`CGB.SensoryMaintenanceTwoTickObstruction.two_tick_world_branch`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L14) [witness / universal / reviewed]：A concrete world branch that remains physically possible through two lost ticks while the projected cell is `aSpent`.
- [`CGB.SensoryMaintenanceTwoTickObstruction.two_tick_local_failure`](../../lean/CGBAxioms/CGBAxioms/SensoryMaintenanceTwoTickObstruction.lean#L22) [witness / candidate / reviewed]：The same branch drains the two-unit reserve and the next sensed miss sends the cell to vacuum.

## SpongeContractInstance（2）
_Coarse admission candidate for a sponge-like continuous body This is a structural interface witness only._

- [`CGB.SpongeContractInstance.port_factorizes`](../../lean/CGBAxioms/CGBAxioms/SpongeContractInstance.lean#L54) [witness / candidate / reviewed]
- [`CGB.SpongeContractInstance.reserve_contract`](../../lean/CGBAxioms/CGBAxioms/SpongeContractInstance.lean#L61) [witness / candidate / reviewed]

## SubconditionGraph（37）
_Structural Graph of the 14 Atomic Sub-conditions This file establishes the formal dependency and redundancy relations among the 14 atomic sub-conditions (A01–A14) from the 6 compound CGB axioms._

- [`CGB.Medium.p1b_iff_atoms`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L61) [semantics / universal / reviewed]
- [`CGB.Medium.onlyBoundarySenses_iff_atoms`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L72) [semantics / universal / reviewed]
- [`CGB.Medium.p3_iff_atoms`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L80) [semantics / universal / reviewed]
- [`CGB.Medium.a01_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L88) [semantics / universal / reviewed]：In any finite local state space, sublevel sets are automatically finite (A01 is redundant).
- [`CGB.Medium.a02_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L94) [semantics / universal / reviewed]：In any finite local state space, content is automatically bounded (A02 is redundant).
- [`CGB.Medium.a01_and_a02_redundant_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L104) [semantics / universal / reviewed]：Under `[Finite Λ]`, A01 and A02 hold unconditionally for ANY content function.
- [`CGB.Medium.uniqueVacuum_independent_of_finite`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L110) [witness / universal / reviewed]：A03 (uniqueVacuum) is strictly independent of finiteness: on `Fin 2`, we can have zero vacuums, multiple vacuums, or a unique vacuum.
- [`CGB.Medium.farFlip_not_rateReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L136) [witness / universal / reviewed]：`farFlip` violates rateReadsParticipants (A07).
- [`CGB.Medium.farFlip_satisfies_effectReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L150) [witness / universal / reviewed]：`farFlip` SATISFIES effectReadsParticipants (A08).
- [`CGB.Medium.farFlip_satisfies_writes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L158) [witness / universal / reviewed]：`farFlip` satisfies writesParticipants (A06).
- [`CGB.Medium.farFlip_satisfies_partsContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L164) [witness / universal / reviewed]：`farFlip` satisfies participantsContact (A05).
- [`CGB.Medium.farFlip_satisfies_homoContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L171) [witness / universal / reviewed]：`farFlip` satisfies homogeneousContact (A04).
- [`CGB.Medium.farFlip_isolates_a07`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L176) [witness / universal / reviewed]：Decoupling summary for `farFlip`: it isolates and violates A07 while satisfying A08, A06, A05, A04.
- [`CGB.Medium.farAct_not_effectReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L199) [witness / universal / reviewed]
- [`CGB.Medium.farAct_satisfies_rateReads`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L214) [witness / universal / reviewed]
- [`CGB.Medium.farAct_satisfies_writes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L218) [witness / universal / reviewed]
- [`CGB.Medium.farAct_satisfies_partsContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L223) [witness / universal / reviewed]
- [`CGB.Medium.farAct_satisfies_homoContact`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L229) [witness / universal / reviewed]
- [`CGB.Medium.farAct_isolates_a08`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L234) [witness / universal / reviewed]：Decoupling summary for `farAct`: it isolates and violates A08 while satisfying A07, A06, A05, A04.
- [`CGB.Medium.innerClock_not_boundaryRate`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L247) [witness / universal / reviewed]：`innerClock` violates boundaryRate (A12).
- [`CGB.Medium.innerClock_satisfies_boundaryEffect`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L256) [witness / universal / reviewed]：`innerClock` SATISFIES boundaryEffect (A13).
- [`CGB.Medium.innerClock_satisfies_initialInvariant`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L261) [witness / universal / reviewed]：`innerClock` SATISFIES initialInvariant (A14).
- [`CGB.Medium.innerClock_isolates_a12`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L268) [witness / universal / reviewed]：Decoupling summary for `innerClock`: it isolates and violates A12 while satisfying A13 and A14.
- [`CGB.Medium.innerActClock_not_boundaryEffect`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L289) [witness / universal / reviewed]
- [`CGB.Medium.innerActClock_satisfies_boundaryRate`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L299) [witness / universal / reviewed]
- [`CGB.Medium.innerActClock_satisfies_initialInvariant`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L303) [witness / universal / reviewed]
- [`CGB.Medium.innerActClock_isolates_a13`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L310) [witness / universal / reviewed]：Decoupling summary for `innerActClock`: it isolates and violates A13 while satisfying A12 and A14.
- [`CGB.Medium.pairPreset_isolates_a14`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L319) [witness / universal / reviewed]：`pairPreset` isolates A14: satisfies boundaryRate (A12) and boundaryEffect (A13), but violates A14.
- [`CGB.Medium.complete_graph_satisfies_a04_a05`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L339) [semantics / universal / reviewed]：Under a complete graph (`near u v ↔ True`), homogeneousContact (A04) and participantsContact (A05) are vacuously satisfied.
- [`CGB.Medium.universal_parts_satisfies_a06_a07_a08`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L348) [semantics / universal / reviewed]：Under universal participants (`∀ a, parts a = univ`), writesParticipants (A06), rateReadsParticipants (A07), and effectReadsParticipants (A08) are vacuously satisfied.
- [`CGB.Medium.all_boundary_satisfies_a12_a13`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L361) [semantics / universal / reviewed]：When boundary = univ, boundaryRate (A12) and boundaryEffect (A13) are vacuously satisfied.
- [`CGB.Medium.path3_not_homogeneous`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L381) [witness / universal / reviewed]：This specific three-site contact relation is not vertex-transitive.
- [`CGB.Medium.atomized_implies_reducedCore`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L426) [semantics / universal / reviewed]
- [`CGB.Medium.reducedCore_implies_atomized_of_regimes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L433) [semantics / universal / reviewed]
- [`CGB.Medium.reducedCore_iff_atomized_of_regimes`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L447) [semantics / universal / reviewed]：Exact recovery under finite local states and rigid structural data (contact homogeneity is no longer part of the axioms since 2026-10-26).
- [`CGB.Medium.path3Rigid_reducedCore`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L467) [witness / universal / reviewed]
- [`CGB.Medium.path3Rigid_atomized_not_homogeneous`](../../lean/CGBAxioms/CGBAxioms/SubconditionGraph.lean#L485) [witness / universal / reviewed]：Since contact homogeneity left P1b (2026-10-26), the rigid three-unit chain, whose contact is not homogeneous (its ends have one contact, its middle two), is a model; before, it was the countermodel showing that finitene

## Symmetry（4）
_P2: symmetry clauses and their first theorems * P2a (conservation) in pathwise form, and **T-N**: the Baez–Fong Noether correspondence (J._

- [`CGB.JumpProcess.gen_sq_sub`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L31) [semantics / universal / reviewed]
- [`CGB.JumpProcess.noether`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L40) [semantics / universal / reviewed]：**T-N** (Noether for Markov processes, pointwise form): `Q` and `Q²` are both stationary observables iff every event preserves `Q`.
- [`CGB.JumpProcess.energy_of_reversible`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L64) [semantics / universal / reviewed]：**T-E** (energy from time-reversal symmetry): a reversible process has an energy `E = -log π` that fixes the ratio of every event to its reverse.
- [`CGB.JumpProcess.ring3_not_reversible`](../../lean/CGBAxioms/CGBAxioms/Symmetry.lean#L83) [semantics / universal / reviewed]

## TurnoverControls（15）
_Constant-content turnover controls These are mathematical observation controls, not bacterial/sponge models or capability experiments._

- [`CGB.TurnoverControls.body_axioms`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L22) [witness / universal / reviewed]
- [`CGB.TurnoverControls.body_unchanged`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L27) [witness / universal / reviewed]
- [`CGB.TurnoverControls.body_eventClock`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L33) [witness / universal / reviewed]
- [`CGB.TurnoverControls.event_at_full`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L52) [witness / universal / reviewed]
- [`CGB.TurnoverControls.old_net_recovery_zero`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L63) [witness / universal / reviewed]
- [`CGB.TurnoverControls.balanced_exchange_recovery_one`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L75) [witness / universal / reviewed]：A one-world-tick exchange occurs with certainty although net content never increases.
- [`CGB.TurnoverControls.zero_ledger_recovery_zero`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L84) [witness / universal / reviewed]
- [`CGB.TurnoverControls.balanced_exchange_alive`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L90) [witness / universal / reviewed]
- [`CGB.TurnoverControls.natural_law`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L109) [witness / universal / reviewed]：The state process stays at the same full-body/world state forever, even though the registered replacement channel fires each tick.
- [`CGB.TurnoverControls.natural_survival_one`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L122) [witness / universal / reviewed]
- [`CGB.TurnoverControls.balanced_exchange_persists`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L130) [witness / universal / reviewed]：A material-throughput subtarget witness, not the complete life or cognitive candidate: the separate SelfMaintains and cognitive obligations are not assumed.
- [`CGB.TurnoverControls.old_net_and_balanced_exchange_disagree`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L138) [witness / universal / reviewed]：Same original state process, two different first-intake observations.
- [`CGB.TurnoverControls.ledger_not_identified_by_state_dynamics`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L148) [witness / universal / reviewed]：Two algebraically accountable ledgers give different gross observations on IDENTICAL body/world dynamics.
- [`CGB.TurnoverControls.fastBody_old_clock`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L160) [witness / universal / reviewed]
- [`CGB.TurnoverControls.fastBody_no_event_clock`](../../lean/CGBAxioms/CGBAxioms/TurnoverControls.lean#L165) [witness / universal / reviewed]

## WorldRecovery（15）
_First-hit probabilities in the actual body--world chain No best-stimulus oracle is used._

- [`CGB.FirstHit.bounds`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L31) [semantics / universal / reviewed]
- [`CGB.FirstHit.monotone_horizon`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L44) [semantics / universal / reviewed]
- [`CGB.FirstHit.flagged_true`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L62) [semantics / universal / reviewed]
- [`CGB.FirstHit.flagged_false`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L68) [semantics / universal / reviewed]
- [`CGB.FirstHit.zero_on_closed`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L82) [semantics / universal / reviewed]：A closed region with no marked positive-probability transition has zero success probability at every finite horizon, even if success exists elsewhere.
- [`CGB.FirstHit.impossible_absorbing`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L101) [semantics / universal / reviewed]
- [`CGB.FirstHit.zero_of_impossible`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L106) [semantics / universal / reviewed]
- [`CGB.FirstHit.during_bounds`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L129) [semantics / universal / reviewed]
- [`CGB.FirstHit.during_constant`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L143) [semantics / universal / reviewed]
- [`CGB.Medium.jointStep_sum`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L162) [semantics / universal / reviewed]
- [`CGB.Medium.law_uses_jointStep`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L168) [semantics / universal / reviewed]
- [`CGB.Medium.lawCut_uses_scheduledCutKernel`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L202) [semantics / universal / reviewed]
- [`CGB.Medium.actualRecover_bounds`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L216) [semantics / universal / reviewed]
- [`CGB.Medium.actualRecover_mono`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L220) [semantics / universal / reviewed]
- [`CGB.Medium.actualRecover_zero_without_intake`](../../lean/CGBAxioms/CGBAxioms/WorldRecovery.lean#L226) [semantics / universal / reviewed]：No possible material intake implies zero actual recovery, without assuming that a favourable stimulus can be summoned from some other world state.

