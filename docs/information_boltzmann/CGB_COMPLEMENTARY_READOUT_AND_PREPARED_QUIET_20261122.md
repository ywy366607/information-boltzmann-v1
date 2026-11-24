# 互补读出、目标条件数与保留学习资格的内部步进

2026-11-22。接续 `CGB_TARGET_CLOSURE_AND_QUIET_CONTRACTS_20261122.md` 的两项合同。研究顺序为数学推导、既有证明去重、实码绑定、数值核对、等价实现。当前交付包括五条Lean命题、2537存档上的八个条件绑定，以及已接通的内部步进复用。

## 1. 木与土共同提供一组互补坐标

设源状态为 h∈R²⁵⁶，encoder为 E:R²⁵⁶→R¹²⁸。原生条件packet同时包含：

\[
Eh,\qquad R_P(h)=M\,[P_l^Tq_l(h)]_{l=1}^{32}.
\]

第二项来自 `decision_momentum_read` 和 `momentum_to_boundary`。其中q经过各层原norm、q_proj和归一化；P是身体当前实际动量。`native_amortized_decision.recipe` 将该读出经 `lift_boundary` 写入第二个raw角色块，随后由同一encoder进入features。

因此，源状态的局部可辨识性由联合导数决定：

\[
J_{\rm joint}(h)=\begin{bmatrix} E\\D_hR_P(h)\end{bmatrix},\quad
\ker J_{\rm joint}=\ker E\cap\ker D_hR_P.
\]

记N为ker E的正交基，J=D_hR_P，C=Eᵀ(EEᵀ)⁻¹。若JN可逆，取K=N(JN)⁻¹，便得到明确的重构：

\[
\delta h=C(E\delta h)+K[J\delta h-JC(E\delta h)].
\]

这把encoder可见部分与动量补回的部分逐项分开。Lean的 `joint_readout_injective_iff` 和 `reconstruct_from_complementary_readouts` 在实际256/128维度上验证上述条件及结论。

### 原存档数值绑定

从2537原生银行选取前八个不同的真实源条件，固定该存档P、权重和其余packet角色，计算源h的导数。原32层RMSNorm的eps沿float32原值固定，计算转double。

| 对象 | 八个条件的结果 |
|---|---|
| encoder E的秩 | 全部128 |
| J在ker E上的秩 | 全部128 |
| 联合读出的秩 | 全部256 |
| 实际归一化packet导数的秩 | 全部256 |
| 256个坐标同时重构的最大坐标误差 | 8.13×10⁻¹⁴ |
| 动量导数与中央差分相对误差 | 3.40×10⁻¹¹—6.57×10⁻¹¹ |
| packet导数与生产features中央差分相对误差 | 3.15×10⁻¹⁰—1.82×10⁻⁹ |

P=0时，动量读出及其h导数为0，联合秩回到128。于是土的持存动量在这里同时承担状态演化和木的内容寻址坐标两种作用。其贡献落在实际读取路径上。

这也更新了上一轮关于encoder核空间的执行方向：以联合读出的共同核及其数值条件为检查对象。单独E的128维核空间继续作为构造互补方向的工具。

## 2. 信息存在与目标容易学到之间，还有一层几何

实际packet的局部奇异值强弱比为2554.57—16081.01。不同源状态方向在木特征中的变化幅度相差数千至一万余倍。联合秩刻画可分辨的方向数量；这些奇异值刻画各方向的分辨强度。

现用二次核 k(q,r)=(1+q·r)² 的显式特征为 ψ(q)=(1,√2q,vec(qqᵀ))，故：

\[
\|D\psi(q)v\|^2=2(1+\|q\|^2)\|v\|^2+2(q\cdot v)^2.
\]

在当前固定角色有无的切片上，每个有效方向块被归一化，幅度/标量的增广块也被归一化，整个q范数恒定。因此 q·Dq=0，得到：

\[
G_k=2(1+\|q\|^2)J_q^TJ_q.
\]

**二次核在这个切片上等比例放大所有局部方向，完整继承原特征的条件数。** 八个实际条件中q范数为0.915811，qᵀJ_q最大坐标3.13×10⁻¹⁹；按上式计算的核度量等比例残差为0。核条件数仍为2554.57—16081.01。Lean验证了单位球内的方向长度界和垂直切向的比例保持。

这一结果将木的升级问题收紧到有实际用途的方向：对同一完整条件下的两个合法行动，令T_ab=T_a−T_b，g_ab=∇T_ab，下一项计算：

\[
\chi_{ab}^2=g_{ab}^T(J_q^TJ_q)^{-1}g_{ab}.
\]

它等于用特征变化表达该行动收益差所需的最小线性读出系数平方范数。借助已算出的SVD，可以同时列出收益差在各奇异方向上的分量。这样将目标敏感度、表示分辨率及行动排序放在同一对象里：优先强化对真实收益差贡献大、而特征变化弱的方向。原生教师与含学习的G₂分别使用各自完整条件、跨度和目标生成状态；沿已有C(B)工具验收实际选择贡献。

### 零角色块的微分定义

固定缺席角色保持为0，其导数取0。生产features中的逐块sqrt(RMS²)在这些零块上直接做forward AD会出现0/0；本次数值绑定使用分块导数及中央差分。八个条件均通过上述差分核对。原银行当前按无梯度拟合/预测执行；新增微分观察器沿此分块定义实现。

## 3. 势能对称性允许合并计算，同时保持学习历史

原势能的头对项为：

\[
V(S)=\frac{\lambda}{16}\sum_{i,j}w_{ij}(S)
\left(\|S_iS_j^T-S_jS_i^T\|_F^2+
\|S_i^TS_j-S_j^TS_i\|_F^2\right).
\]

内容门控w对i、j对称，交换i、j使交换子变号，平方能量保持；对角项为0。于是：

\[
V(S)=\frac{\lambda}{8}\sum_{i<j}w_{ij}(S)\,e_{ij}(S).
\]

8头的64个有序位置化为28个独立配对。Lean的 `ordered_pair_energy_eq_twice_upper` 验证有限求和恒等式；随机内容门控、矩形状态矩阵及1/2/8头的数值检查同时核对势能、状态力、参数梯度和混合方向导数。

对F=(∂V/∂S)，其方向传播为：

\[
DF[\delta S,\delta\theta]=H_{SS}\delta S+H_{S\theta}\delta\theta.
\]

标量势能的混合偏导对称性允许用同一导数图计算这两部分。保留原四方向分块，处理全部12类逐层theta资格及2类contact资格。沿旧因果资格定理继续计算 E′=AE+B。

### 已接通的提交顺序

1. 木试走内部一步时，用原势能力计算前向S/P，保存当前状态、旋转系数及后继状态。
2. 火选择内部续行后，延迟构建两个kick位置上的独立配对导数图，输运theta/contact旧资格并加入本次参数源项。
3. 提交原S/P前向终点一次，继续原事件钟、关系重读、后到学习及世界交互。
4. 火选择交话时释放提案；该路径保留前向试算成本与约0.13 MiB的峰值额外分配。

前向继续使用原势能求和顺序，保证本次核对的S/P逐项相同。导数使用等价的28对势能。提案核对状态/参数/资格身份、版本、实际参数值和dt，支持检测绕过Tensor._version的参数写入；参数对象替换也会触发重新计算。提案从持存快照派生字段排除，内存中只跨同一次选择与提交。

实现：`scripts/quiet_force_jet.py`。接入 `_sample_internal_choice` 与 `continue_internal`。CUDA默认启用，CPU以原实现为默认并支持检查时显式启用。原多子步路径沿现有积分器执行。

## 4. GTX1650上的32层端口结果

同一2537完整端口，32层/8头/32×32，原S/P、context、频率、阻抗及资格。每种情况两次预热、三次计时，显式CUDA同步。

| 原操作 | 原实现中位耗时 | 新实现中位耗时 | 原峰值分配 | 新峰值分配 |
|---|---:|---:|---:|---:|
| 提案后交话 | 22.87 ms | 19.69 ms | 260.52 MiB | 260.65 MiB |
| 提案后内部续行 | 612.98 ms | 325.48 ms | 973.15 MiB | 515.81 MiB |

内部续行节省287.50 ms、457.33 MiB峰值，约1.88倍端口速度。若两种选择各占一半，端口混合成本为1.84倍速度；自然生活采用实际选择频率加权。

S/P最大坐标差均0。theta资格最大差2.38×10⁻⁷，contact资格最大差7.45×10⁻⁹。原存档的历史theta资格处于更新后状态；定向检查另覆盖非零theta及contact历史资格、阻抗/非阻抗、零/正/微负dt及真实读出后的梯度。

相关五文件42项检查通过；延迟建立导数图后再次运行16项专用检查通过，包括两次内部续行、交话、世界反馈、真实优化更新、恢复后的话语/选择/参数核对。Lean五条命题编译通过。本轮运行使用冻结端口与数值接口，新增自然训练和世界调用均0。

## 5. 在完整五行中的位置与下一执行对象

后续行动通路已完成第一轮绑定，见 `CGB_ACTION_CONTRAST_PORT_TRANSPORT_20261122.md`：源状态与候选角色分别定义可观结构；原生教师四格读取将误差沿局部q、终点、共同状态传播到选择；收益先验与校准门控进入DT-Dprior残差检查。用户确认该路线作为整体升级入口，最终沿实际C(B)验收。

木通过内容和动量两条实际路径读取土；火在试算结果间作出选择；土按相同物理时间推进；金得到输运后的原学习资格；水的持存及待结算身份沿原恢复合同继续。由此形成两项可用结论：

- 表示升级按行动收益差与联合读出几何对接，优先定位值得放大的区别。
- 等价计算按前向与一阶学习资格共同保持实施，已经减少32层内部续行端口的成本。

下一项沿第一条继续：在现有完整边界取原生候选收益差，计算g_ab在联合packet奇异方向上的分量及χ_ab；先对现有教师条件作可重构绑定，再将确定的缺口交给原行动贡献C(B)验收。谱、首达及长期核沿已保持的完整学习状态和物理时间继续接入。

## 6. 复现材料

- Lean：`lean/CGBAxioms/reviews/NativeComplementaryReadoutAndPairs.lean`。
- 联合读出：`scripts/audit_joint_target_coordinates.py`；结果 `results/published/cgb_joint_target_coordinates_20261122.json`。
- 性能：`scripts/benchmark_prepared_quiet.py`；结果 `results/published/cgb_prepared_quiet_20261122.json`。
- 专用检查：`tests/test_quiet_force_jet.py`。
- 整体验收摘要与源码SHA：`results/published/cgb_complementary_quiet_contracts_20261122.json`。
- 输入沿原2537完整归档提取的 `source_ports_2537.pt` 和 `query_ports_2537.pt`，SHA写入对应结果。

命令均在仓库根目录执行：`D:/conda_envs/vox/python.exe -m scripts.audit_joint_target_coordinates`、`D:/conda_envs/vox/python.exe -m scripts.benchmark_prepared_quiet`、`D:/conda_envs/vox/python.exe -m pytest tests/test_quiet_force_jet.py -q`。Lean在 `lean/CGBAxioms` 执行 `E:/lean/elan/bin/lake.exe env lean reviews/NativeComplementaryReadoutAndPairs.lean`。
