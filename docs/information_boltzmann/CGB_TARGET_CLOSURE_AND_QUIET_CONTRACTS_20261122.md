# 木表示的目标闭合与内部续行合同：第一轮数学结果

2026-11-22。执行入口承接`CGB_MATHEMATICAL_FIVE_PHASE_MAP_20261122.md`。用户批准先做两份合同，再向完整轨迹核、谱、持存时间与动力学零点生长。

本轮交付：8项Lean命题编译通过；真实2537身体两类条件银行128条记录的特征恢复；真实32层/8头内部动力学及切向输运绑定；旧完整轨迹999段内部—交话路径的停时绑定。训练更新0，世界调用0。生产机制沿原实现执行。

## 1. 目标及已有证明的分工

木的读出目标首先固定为同一实际候选的原生下一决策端口。按需要另声明当前actor分布、真实后续学习、旧事实保持目标。合法候选身份、条件、期限、单位和源类型一起进入目标定义。

已有`NativeDelayedActionContrast.exact_credit_iff`和`NativeActorExpectation.actual_centered_logit_difference`已经给出行动对比的共同平移不变性。本轮继承它们，选择有关目标时使用动作差或合法行动概率，绝对价值目标继续按原用途保留。

已有`GenerationColumnRefinement.BodyCausalCarrier.merged_history_preserves_read_and_successor`承接读出与后继一致性；11月8日因果资格压缩承接源基输运、未来信用可见性和误差界。本轮新增证明聚焦下述尚缺的实码性质。

## 2. 结果一：固定锚点使当前特征归一化可逆

两类实际features都具有如下结构：每个编码块保留方向u_i及对数幅度a_i，再将幅度和scalar拼接后归一化。scalar中含有在本身体配置下固定的非零gamma=0.8。

令v=(a,scalar)，归一化后w=v/r，锚点v_j=c≠0，则：

\[
r=\frac{c}{w_j},\qquad v_i=c\frac{w_i}{w_j}.
\]

实际增强向量有非零锚点，其范数处于F.normalize的正常分支。各编码块由幅度恢复范数，再结合已保存的方向恢复；对零块及epsilon分支按原分母规则处理。

Lean：`anchor_recovers_coordinate`、`anchored_view_injective`。它们验证固定锚点切片上的归一化可逆性。编码器以前的压缩与原始事实摘要各自属于前级表示。

### 真实2537银行的恢复

| 原银行 | 记录数 | 角色块×宽度 | 当前特征宽度 | 编码块相对L2误差 | 编码块最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| native_value_bank | 64 | 31×128 | 4074 | 1.13743e-7 | 2.11147e-6 |
| latent_action_bank | 64 | 13×128 | 1729 | 1.04985e-7 | 1.53434e-6 |

使用当前生产features及该存档实际encoder；原生银行包括360个零编码块，沿同一恢复过程核对。两处锚点scalar索引分别37与6。

**设计含义：**当前特征归一化具有明确的信息恢复路径。木的目标闭合检查优先考察证据摘要、encoder映射、条件状态和读出模型。目标所需的信息在前级已经合并时，增加后级拟合宽度需要配合前级表示修订；若前级已足够，则沿原表示处理拟合和校准。

同一实际encoder的矩阵为128×256，数值秩128，核空间维数128；最小/最大非零奇异值0.1026814/1.2799003。后续联合读出已完成：同一packet中的实际动量读出，在八个源条件上补齐这128个局部方向；联合及完整packet导数均为256秩。执行对象更新为联合读出的数值条件及其与行动收益差的对应，见 `CGB_COMPLEMENTARY_READOUT_AND_PREPARED_QUIET_20261122.md`。该页同时给出实际重构、二次核的局部等比例性质及已接通的续行复用。

## 3. 结果二：熟悉度与可预测性具有独立坐标

当前核为k(x,y)=(1+x·y)²，ridge=λ。取同一特征x重复n次，自核κ=k(x,x)>0，实际Cholesky Schur读出的归一化新颖度可化为：

\[
\nu_n=\frac{\lambda}{\lambda+n\kappa}.
\]

它随重复数严格下降。若同一表示对应的两个残差目标为±δ，任意单一预测m均满足：

\[
\frac{(m-\delta)^2+(m+\delta)^2}{2}=m^2+\delta^2\ge\delta^2.
\]

这给出明确机制：条件覆盖逐渐充分，条件内剩余误差仍可由被压缩的区别、真实后果波动或目标随学习变化产生。分别记录覆盖度、同条件残差及目标生成状态，可以区分它们。

Lean：`repeated_feature_schur`、`repeated_novelty_strictly_decreases`、`same_feature_error_floor`、`novelty_below_threshold`。

### 与生产公式的数值绑定

取真实银行的一条特征κ=3.380853503，λ=0.1，使用原_kernel和Cholesky，给出声明的交替±1残差目标，核对上面的代数性质：

| 重复数 | 实际Schur新颖度 | 声明目标的均方残差 |
|---:|---:|---:|
| 2 | 0.0145736330 | 1 |
| 4 | 0.0073403040 | 1 |
| 8 | 0.0036836716 | 1 |
| 16 | 0.0018452344 | 1 |

这一组是原核公式的数值见证。真实身体是否存在相应的目标混叠，交给下节的目标/状态绑定检查。

### 木合同的具体检查对象

1. 当前recipe/packet及features确定q；原生目标确定T。完整条件包括候选身份、问题、关系版本、模型/身体生成状态及目标类型。
2. 同一可微执行分支，检查`Dq(v)=0 ⇒ DT(v)=0`；离散检索与合法性分支使用实际身份及原执行分支。
3. 优先利用原生参照已有prediction/error/priority、bank的kind/revision和原始packet；针对需要的方向补记录或重算当前目标。
4. 对含真实学习的目标，把优化器、待结算资格及实际世界转移加入所需状态。对固定参数的局部原生目标沿其原合同。
5. `native_value_bank.quality`已有原生残差，teacher_checks已有当前误差；novelty使用其几何覆盖含义。新增分配策略须在真实行动贡献工具C(B)上验收。

此合同让升级选择具有具体对象：补条件、修目标时效、处理随机后果或修拟合，各自有对应的数据与数学条件。

## 4. 结果三：前向状态一致与后续学习一致具有不同的保持对象

设内部一步F_θ(x)，x=(S,P)。已有资格E记录过去参数参与对当前状态的影响，则：

\[
x'=F_\theta(x),\qquad
E'=D_xF_\theta(x)E+D_\theta F_\theta(x).
\]

写成A E+B后，直接复用原因果资格输运定理。内部提案的前向终点可以复用；后续学习还要携带相应一阶信息（jet）及旧E。

### 实际32层绑定

从同一2537源提取32层、8头、32×32矩阵的S/P和原动力学参数。沿原自适应积分器：

- capture=False提案与capture=True实际执行，S最大绝对差0，P最大绝对差0。
- 第一拍后保留实际12类逐层theta资格及2类逐层contact资格。
- 以第一拍所得状态和theta资格作为第二拍输入，分别计算整体JVP与A E、B两部分。
- 全部32层、12类theta方向的相对重构误差最大1.15520e-7，绝对误差最大1.19209e-7。
- 完整下一步theta敏感度的联合L2范数29.22774；清空旧E所移除的A E范数14.61400，差值/完整范数约0.5000。

这些导数沿生产实现的已分离context、原固定子步规则和局部theta方向定义。原contact资格也由实际capture推进，合并S/P范数58.40512；该数是接口中的实际资格量。

**意义：**已经得到一个具体的“前向相同、以后学法不同”的绑定。学习所需历史可以由资格承接，并在真实后果到达时兑现。把同一S/P当作学习状态的全部，会把这部分历史合并；完整学习状态包含其资格、优化器和待结算身份。

## 5. 内部提案复用合同

生产入口：`navigation_generative_cycle._sample_internal_choice`试走quiet；`navigation_joint_cycle.continue_internal`提交实际quiet。前者主要服务木/火比较，后者推进土及后到学习资格。

一个合格的两阶段提案包含：

- 前态标识：S/P、theta、context、frequency_scale、impedance、dt及solver细分。
- 前向结果及可用的一阶传播信息，或对应的延迟计算配方。
- theta与contact旧资格的读取身份。
- 提案可能产生的context/cache副作用及其提交归属。
- 选中提交恰好一次；未选中沿原语义保留状态及资格。

复用的等价目标为：前向结果、A E+B、contact输运、原RNG/选择、关系绑定和后到学习读出。已有资格压缩定理负责源方向与未来读出的保持。

本轮数值已经验证前向一致和一阶分解。后续实现已将独立配对势能、共享导数图和延迟资格输运接入原选择/提交端口；32层CUDA内部采纳端口612.98→325.48 ms，峰值973.15→515.81 MiB。完整说明及检查见互补读出与内部步进文档。

## 6. 结果四：主动交话的有限段同时携带概率与信用尾部

令c_n为沿真实内部续行路径在第n个选择处继续的条件概率：

\[
\sigma_0=1,\quad\sigma_{n+1}=\sigma_nc_n,
\qquad h_n=\sigma_n(1-c_n).
\]

有限前缀有：

\[
\sum_{n<N}h_n+\sigma_N=1,
\qquad\sum_{n<N}\dot h_n+\dot\sigma_N=0.
\]

Lean：`prefix_release_and_tail`、`prefix_tangent_and_tail`。前者保留尚在内部活动的路径质量，后者同时保留其学习方向。原`autonomous_cycle.poll`对实际internal或release选择累计log-softmax score，`take_boundary/add_boundary_credit`将这段参与交给真实后到因子。

### 既有完整轨迹绑定

从已保存真实轨迹读取999段从完整起点开始的内部—交话前缀；文件起点已经携带部分历史的1段单独跳过。最长前缀9次选择。

- 概率守恒最大误差2.22045e-16。
- 共同内部相对logit方向的导数守恒最大误差5.55112e-17。
- 从逐次实际概率重构整段已选路径logp，与原eligible_logp最大差2.83049e-7。
- 在各自已观测前缀长度处，反事实继续路径质量的平均值0.335374。这个平均基于实际抽样停止长度；它表示各前缀的数学待续质量，按原抽样规则解释。

谱/首达计算中的有限前缀因此保留σ_N及其导数；实际身体的快照继续携带pending资格。这个绑定将内部节律、首次行动与真实后果学习接到同一条路径上。

## 7. 本轮对升级的决定

1. 当前归一化沿已验证的恢复结构保留，木的下一检查指向前级摘要/编码及目标条件。
2. novelty、已有teacher残差和目标状态分别入账。下一步在真实同条件目标上检查混叠、随机性及学习漂移，依据实际失配决定表示升级。
3. 内部提案复用以“前向＋一阶资格＋提交身份”为合同；把theta/contact及待结算参与一起接通。
4. 首达/轨迹核使用完整学习状态及等待标记，有限前缀保留质量和信用尾部。谱、持存时间和Lee–Yang后续工作沿此状态对象继续。

## 8. 文件、命令与来源

- Lean：`lean/CGBAxioms/reviews/NativeTargetClosureAndStopping.lean`，8项命题，`lake env lean reviews/NativeTargetClosureAndStopping.lean`通过。
- 数值入口：`scripts/audit_native_math_contracts.py`。
- 已提取的35.13 MB私有只读输入：`scratch/cgb_math_contracts_20261122/source_ports_2537.pt`，来自报告中的原完整存档，含encoder、两银行与S/P/资格。
- 结果：`results/published/cgb_native_math_contracts_20261122.json`；源路径、提取输入SHA及原事件SHA在JSON内。
- 运行：`D:/conda_envs/vox/python.exe -m scripts.audit_native_math_contracts --input scratch/cgb_math_contracts_20261122/source_ports_2537.pt --output results/published/cgb_native_math_contracts_20261122.json`。

文献定理沿上轮数学入口引用；本轮新增内容为锚定归一化、实际核的显式覆盖/误差分离见证、有限停时的质量/切向尾部，以及当前32层资格绑定。既有actor共同模式、因果资格输运与完整事件合同直接继承。
