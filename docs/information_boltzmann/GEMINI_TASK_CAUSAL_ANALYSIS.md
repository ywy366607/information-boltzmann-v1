# 任务委托书：第一代 3D 连续介质检查点深度因果消融与 AG 度量分析

**致 Gemini 合作助手：**

本项目（3D 连续玻尔兹曼介质 PlasticMedium3D）的第一代连续学习训练已经在 Step 2689、86,048 fresh tokens 处安全停止，检查点位于：
`results/medium_d768_streaming_pathway_8x8x8_160k/last.pt`

由于主开发团队正在攻关第三代（Hopf 拓扑重构与无损分束分支），现委托你完成第一代权重的**离线因果归因消融实验**与**全生命周期 AG / 记忆留存指标分析**。该任务为只读评估，无需更新任何模型参数。

---

## 任务一：各物理算子因果承担量消融（Causal Attribution Audit）

### 1. 目标
通过对前向物理演化中各个算子的精确开闭，测定它们对下一个 token 预测负对数似然（NLL）的实际贡献，检验是否存在输运与碰撞的“超加和协同效应（Super-additivity）”，以及材料各向异性分化贡献了多少能力。

### 2. 消融工况（在 4 段独立的 128-token OWT 序列上评估）
1. **Full (全算子基线)**：完整物理演化前向，记录 $L_{\text{full}}$。
2. **No Transport (消融输运)**：在 `advance` 时跳过波动传输步（令 $B = 0$ 或仅保留局部阻尼），记录 $L_{\text{no\_trans}}$，计算 $\Delta \text{NLL}_{\text{trans}} = L_{\text{no\_trans}} - L_{\text{full}}$。
3. **No Collision (消融非线性碰撞)**：跳过多层局部碰撞算子，记录 $L_{\text{no\_coll}}$，计算 $\Delta \text{NLL}_{\text{coll}} = L_{\text{no\_coll}} - L_{\text{full}}$。
4. **No Both (双重消融)**：同时关闭输运与碰撞，记录 $L_{\text{no\_both}}$，计算 $\Delta \text{NLL}_{\text{joint}} = L_{\text{no\_both}} - L_{\text{full}}$。
   * **协同效应度量**：计算协同比率 $S = \frac{\Delta \text{NLL}_{\text{joint}}}{\Delta \text{NLL}_{\text{trans}} + \Delta \text{NLL}_{\text{coll}}}$。若 $S > 1$，证明三维空间路由与局部非线性变换形成正向协同（Synergy）。
5. **Homogeneous Material (消融材料各向异性)**：将介质中学习出的各向异性剪切和电导张量抹平为初始均匀空间场，计算 $\Delta \text{NLL}_{\text{material}}$，衡量自组织河床贡献的净收益。
6. **No Temporal Probes (消融时域探针)**：关闭 16 个探针的复指数滤波器银行，仅用瞬时场读出，计算 $\Delta \text{NLL}_{\text{temporal}}$。

### 3. 输出要求
将测试脚本与结果保存为：
`results/published/medium_causal_attribution_86k_20261009.json`，并包含各个工况的 Mean NLL 与 Delta NLL。

---

## 任务二：生命周期 AG 指标与 Ebbinghaus 记忆留存全谱分析

### 1. 目标
从长期日志中定量提取系统随时间演化的两大核心生命指标：
1. **AG (Adaptation-Generalization) 指标**：面对生疏环境 B 时的自适应能力。
2. **Ebbinghaus 记忆留存率**：在经历 B 冲击后，回到 A 时的留存（Savings）与重学加速。

### 2. 数据来源
日志文件：`results/medium_d768_streaming_pathway_8x8x8_160k/lifelong_evaluation.jsonl`（包含全部 17 次周期性评估快照）。

### 3. 具体计算指标
1. **AG 指标随时间的演化趋势**：
   * 提取每个快照的 `fresh_training_tokens`、`B_nll`（环境 B 预测损失）与 `B_prior_nll`（无信息先验损失）。
   * 计算对数收益 $\text{log\_gain} = B_{\text{prior\_nll}} - B_{\text{nll}}$ 与泛化优势比 $\text{gain\_ratio} = \exp(\text{log\_gain})$。
   * 给出均值、峰值和变化斜率，评估随着时间推移，介质的生疏适应能力是增强、平稳还是退化。
2. **A -> B -> A 记忆留存率（Savings）**：
   * 提取快照中的 `savings` 字段，分析 `initial_nll`（首次学 A 的损失）、`revisit_nll`（二次学 A 的损失）与 `relative_nll_saving`。
   * 分析记忆巩固程度是否随生命周期训练而提升。

### 4. 输出要求
将分析结果和精简 Markdown 报告保存为：
`results/published/medium_lifelong_ag_and_savings_audit_20261009.json`

---

感谢配合！主团队正在专注第三代 Hopf 拓扑重构与二代对照实验，完成分析后请直接输出结构化指标报告。
