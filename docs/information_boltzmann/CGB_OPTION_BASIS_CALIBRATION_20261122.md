# 选项基础校准：从先验几何到实际选择

2026-11-22。承接 `CGB_ACTION_CONTRAST_PORT_TRANSPORT_20261122.md` §5—§6；对应数学总表中的目标闭合、实际读出和历史校准路线。

## 1. 问题与本轮交付

原生2537边界中，一对生成/复用候选具有相同名义向量和相同执行终点，教师q相差约3e-7，学生q却相差约0.511。其中历史选项先验贡献约0.546，条件银行抵消约0.036。当前内容收益门控依据真实校准退回选项均值，外层核同时承担类别均值修正和内容条件拟合。

本轮把选项整体修正作为一个至多三维的自由基底，与原内容核联合求解。完整32层身体读取、原教师用途下的选择、原银行的整批次留出均已计算；独立实验开关已接入完整生活工作器。验收目标依次为：数学合同→实际候选差与选择→后续学习贡献C(B)。

## 2. 数学模型与身体绑定

原预测是 p(x)+k(x,X)Dα，p沿用原守卫门控先验，D沿用native/world来源平衡，K沿用原二次核，λ沿用relation_wood.ridge=0.1。

令C为银行中实际出现的选项one-hot矩阵；每列都有记录，D各对角元严格为正。新预测为：

\[
\widehat q(x)=p(x)+c(x)^T\beta+k(x,X)D\alpha.
\]

其中 G=DKD，H=DC，y=D(t−p)，A=G+λI。解以下两个正规方程：

\[
A\alpha+H\beta=y,\qquad H^T\alpha=0.
\]

利用原拟合的Cholesky分解，先算α₀=A⁻¹y、Z=A⁻¹H，再解至多3×3的Schur系统：

\[
\beta=(H^TZ)^{-1}H^T\alpha_0,
\qquad \alpha=\alpha_0-Z\beta.
\]

它最小化加权残差平方加原核惩罚：

\[
\|y-G\alpha-H\beta\|^2+\lambda\alpha^TG\alpha.
\]

候选相对这个解的增量代价，恰为
\(\|G\delta\alpha+H\delta\beta\|^2+\lambda\delta\alpha^TG\delta\alpha\)。

**得到的关键对称性：** 固定银行标签，若先验在每个选项上整体移动c(x)ᵀb，则β移动−b，α和最终预测保持。这样，类别先验的坐标选择由小基底承接，核容量用于条件结构。银行尚未出现的选项沿用原先验与内容核的预测。

Lean文件 `reviews/NativeOptionBasisCalibration.lean` 已编译6条命题：实际加权one-hot设计的单射性、Schur正规方程、先验移动后的正规方程、查询预测保持、Schur正性及正则目标增量。正岭核结论沿用 `NativeAmortizedDecision.lean` / `NativeJointSuccessor.lean`。

代码绑定 `scripts/option_basis_calibration.py` 的 fit/evaluate/predict_q/calibrated_predictions。数值求解采用float64；身体的post/endpoint、原始novelty、教师、结算与actor沿原路径计算。实验作用域只替换q列，并按原拟合cache身份复用校准，作用域退出恢复函数。

## 3. 完整身体的现场结果

原检查点SHA256：102f52357b46be060a196ce83cf7cb29e2e06f0e6030fec86340032386284336。
完整2537身体及原绑定档案恢复后，沿恢复源、合法沉默和原均值后继读取一次原生教师。两个问题分支各6个候选，采用同一组教师q评价各自实际草稿/actor概率。

| 现场量 | 世界回应问题：原→校准 | 动作问题：原→校准 |
|---|---:|---:|
| 同终点生成减复用q差 | −0.510699→−0.174281 | −0.511285→−0.174690 |
| 候选差MSE | 0.110824→0.028286 | 0.105947→0.028721 |
| 候选差MSE降幅 | 74.48% | 72.89% |
| 教师用途下的概率加权q | 0.388415→0.390853 | 0.389152→0.391404 |
| 概率加权用途增量 | +0.002438 | +0.002252 |
| 相对于仅教师q替换的改善比例 | 25.27% | 23.45% |

校准后的沉默/生成/复用概率为23.37%/23.44%/53.19%及23.27%/23.73%/53.00%；原先复用约84.04%/83.21%。教师q本身的完整替换参照沿用原四格报告。

两支使用相同银行解，β=[+0.271562,+0.068442,−0.300651]。正规方程最大残差4.55e-15，Hᵀα最大残差8.88e-16，Schur条件数165.64。

### 绝对误差与选择误差的分离

对于同一候选集合，eᵢ=q̂ᵢ−qᵢ，精确分解为

\[
\operatorname{MSE}(e)=\bar e^2+\operatorname{mean}(e_i-\bar e)^2.
\]

对n个候选，全部无序候选对的差误差MSE等于2n/(n−1)乘后一个分量。

| 分量 | 世界回应问题：原→校准 | 动作问题：原→校准 |
|---|---:|---:|
| 公共误差平方 | 0.035979→0.093441 | 0.064758→0.131139 |
| 中心化误差 | 0.046177→0.011786 | 0.044144→0.011967 |
| 绝对q MSE | 0.082156→0.105226 | 0.108902→0.143106 |

这定位了一个具体机制：校准更好地区分了当前候选，同时把这一问题下的整组q整体压低。原actor共同模式定理解释了组内概率对公共偏移的保持；跨问题、跨后继状态的价值比较继续使用其实际绝对用途。后续C(B)因此同时承接这两类分量的真实传播。

## 4. 同一银行的整批次留出

固定原64条记录，每次拿出一个完整commit revision，用其余记录重新计算来源权重并拟合。38次留出覆盖全部64行。每个revision内的native/world标签一同留出，避免同批次的另一候选继续充当训练记录。

| 留出记录 | 原MSE | 校准MSE | 降幅 |
|---|---:|---:|---:|
| 全部64行 | 0.152870 | 0.130349 | 14.73% |
| native 32行 | 0.146733 | 0.120448 | 17.91% |
| world 32行 | 0.159006 | 0.140250 | 11.80% |

离线分析复用原Cholesky因子的乘积构造加权系统；其原完整查询重构差2.16e-6。标签与2537端口提取中的银行逐行核对，选项顺序逐项一致。38是留出批次数，统计对象是这64行历史记录。

## 5. 资源和完整生活接入

GTX1650上用捕获的实际64行矩阵计时，100次拟合均值1.69 ms，200次已缓存查询均值0.20 ms。此计时覆盖新增小系统及查询代数；原特征编码和核矩阵继续沿原路径。

只读观察器完整端口的首支诊断47.92 ms、次支9.69 ms，包含额外的生产predict/prior/features重读与诊断。实验作用域按原cache复用拟合，教师与学习更新导致原cache变化时重新求解。

完整生活工作器新增 `--option-basis-calibration`，沿原自然线/两步条件分支执行，并记录 `option_basis_runtime.json` 的拟合/查询次数。两次真实step接口检查覆盖银行结算、原教师标签、cache复用及学习后更新；状态/终点列与novelty逐项一致，退出和异常恢复均已检查。

2026-11-22最终选择接线核对：当前持存节拍路径直接读native_value_bank的标量整次决策用途，逐行动q校准处于另一条件路径。原q开关在持存最终选择端的直接增量为0。新增持存接通版将相同后继上的F_cal−F_original沿原质量传回该标量价值，并在同一真实菜单下取得两套最终概率；其协议、Lean和G2实验单列于 `CGB_OPTION_CALIBRATION_USE_20261122.md`。上一节逐行动教师与留出结果继续对应原来的逐行动目标。

15项相关检查通过，涉及校准正规方程、目标最小化、先验移动保持、未出现选项、完整step学习/结算、cache、异常恢复、原生教师观察及原条件actor。6条Lean命题编译通过，5个修改/新增Python入口语法编译通过。

本次完整身体读取世界调用0、优化更新0；原检查点哈希保持。数值capsule仅包含小矩阵和标签，后续数值分析直接复用。

## 6. 得到的认识与下一步

1. **可观性要落到有效读出。** 源信息的联合满秩、内容门控和残差银行共同决定目标可读程度。本轮从“增加表示维度”推进到“先验与核如何分工”。
2. **选择所用的是收益差。** 这次同一真实菜单上，绝对MSE上升和选择用途增加同时发生，差额已落到公共分量与中心化分量。验收需沿实际概率和后续学习收益继续传播。
3. **历史类别均值具有一种可消除的坐标负担。** 让原核惩罚同时管类别常量和细节，会让弱选项方向额外承担抵消先验的任务。小维自由基底把这个任务解析分离，给出可验证的先验移动保持性。
4. **一次至多3×3的校准已经改变了实际概率。** 当前银行的整批次留出和当前教师点均给出具体改善，适合作为下一项C(B)实验候选。

2026-11-22该完整选择验收已完成：原持存路径C=+.000493555，显式接通版C=+.000309906，ΔC=−.000183649；当前保留原持存路径，详见 `CGB_OPTION_CALIBRATION_USE_20261122.md`。以下保留事前执行顺序与合同：

- 在一个事先固定的新完整决策边界，保留自然形成的问题、草稿、合法候选和完整学习策略，记录原模型与校准模型的实际最终概率。
- 对同一组首步取得共同独立材料上的G₂，使用原两步条件分支工具。已有轨迹在边界、候选、原学习策略和材料逐项相符时复用。
- 计算 C_old=Σ(π_old−π_habit)G₂，C_new=Σ(π_new−π_habit)G₂，及ΔC=Σ(π_new−π_old)G₂。这个配对直接量化“这次校准究竟把概率移向了后续更可学的行动多少”。
- 当校准参与了候选生成或后续学习时，相应轨迹按其真实条件取得G₂；账目显式携带这项条件。原冻结菜单上的概率效应与完整算法改变后的生活收益各用对应的实际轨迹。
- 以完整C结果决定此校准进入持续生长的方式；公共价值偏移沿问题选择和后继价值继续做原四格归因。更广的谱/时间/持存路线继续挂在同一个轨迹核上。

## 7. 复现与文件

- `scripts/option_basis_calibration.py`：小系统及显式实验作用域。
- `scripts/native_action_contrast_geometry.py`：现场校准、原四格、策略与capsule记录。
- `scripts/audit_option_basis_calibration.py`：公共/候选差分解及38批次留出。
- `scripts/run_wood_policy_use_worker.py --option-basis-calibration`：完整生活实验入口。
- `results/published/cgb_option_basis_calibration_20261122.json`：实际身体两支读数。
- `results/published/cgb_option_basis_holdout_20261122.json`：逐折结果。
- `results/published/cgb_option_basis_cost_20261122.json`：新增小系统计时。
- `lean/CGBAxioms/reviews/NativeOptionBasisCalibration.lean`：6条命题。

实际身体复现：

```powershell
& D:/conda_envs/vox/python.exe -m scripts.run_native_action_geometry_readonly --checkpoint scratch/cgb_competing_action_teachers_2521_20261120/competing/body/last.pt.gz --work <新的D盘私有目录> --output <结果JSON> --option-calibration
```

离线留出复现：

```powershell
& D:/conda_envs/vox/python.exe -m scripts.audit_option_basis_calibration --ports scratch/cgb_math_contracts_20261122/source_ports_2537.pt --measurement results/published/cgb_option_basis_calibration_20261122.json --output results/published/cgb_option_basis_holdout_20261122.json
```
