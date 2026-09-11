# 局部可学习信息玻尔兹曼：数学与编码规格 v1

日期：2026-09-11。状态：设计规格，尚非新实现或性能结论。
读者：执行编码 AI、测试 AI、审查者。MUST 为必须；禁止自行补充物理机制。

## 0. 任务与边界

目标是嵌入外部环境、可持续接收输入并学习的持存个体。开放驱动与耗散属于模型主体；临界耗散是用户确定的设计要求。本文给出其测量接口，不声称仅凭守恒/有界性已经证明临界或无限学习。

必须保留两个可学习机制：(A) 外部输入条件化的出生分布与持续驱动；(B) 局部二体碰撞率 Attention。碰撞保持质量、动量、动能；完整系统不要求封闭守恒。不得替换为全局 Slice、BGK、MRT 或固定不可学习碰撞率。

当前 `results/ib_bpe_512_5000` 是独立全局 Slice 对照。不得中断、修改其已加载模型或把其权重称为局部版本。不得并发运行新 GPU 性能实验来争抢资源。新代码放独立模块；不改公共导出。

性能硬门槛：真实 OWT，至少 L=128 BPE targets/optimizer update，预热后完整更新 p95 <=1.5 s；同时给 median、tokens/s、冷启动、验证与监控开销。首选 N=512,d=4,H=128，另报 L=256/512；无法达标必须如实返回 profiling，不擅自降低 N/H/梯度长度。显存分别报 allocated/reserved/全卡专用/共享，未测共享就写 unavailable，不能把 allocator 上限当全卡保证。此前 2 GiB 限制用户随后撤回并同意原配置，不声称它已经满足。

## 1. 状态、因果顺序与生命周期

输入 token 为 y_0,y_1,...；BOS=50256，V=50257，GPT-2 BPE。时钟 s 是动力学时间，k 是 token 事件，u 是 optimizer update，三者不能混用。

状态 Z=(x,v), x,v∈R^{N×d}；经验测度 μ_N=N^{-1}Σ_i δ_(x_i,v_i)。质量归一为1。θ 包含出生/驱动/阻尼，φ 为碰撞率参数，ψ 为读出。出生用仅含已知上下文的 F_θ 推送基分布；以后不在文档边界或窗口边界重置 Z。

事件 k：用 y_{k-1}（k=0 时 BOS）演化 Z_k→Z_{k+1}，读出 p(y_k|past)，评分 y_k，随后允许它成为下一事件输入。禁止用目标 y_k 驱动同一事件。每 L 个事件做一次 AdamW，参数窗口内固定；状态值持续，窗口边界 detach。它是截断 BPTT，不是无限梯度。

出生参数只在首个窗口收到梯度（共享参数可在其他路径继续学习）。不得宣称出生映射一直在线单独受训；若需要跨生命学习，另立实验。加速度 a 是漂移的一部分，非第三个独立持存状态。

## 2. 输入算子与开放演化

冻结一次 token 输入与参数时，连续参考动力学为

    dx_i = v_i ds
    dv_i = [-κ x_i + b_θ(x_i,y_{k-1},s) - γ_θ(x_i)v_i] ds
           + sqrt(2 γ_θ(x_i) T) dW_i + 局部碰撞跳跃.

默认 κ=1,T=0.1,A=1；b=A/sqrt(d) tanh(MLP([x,embedding(y),sin s,cos s]))，所以 ||b||₂<=A。γ=γ_min+(γ_max-γ_min)sigmoid(w·x+bγ)，默认 min=.01,max=2,初值.3。先沿用原公式，不加 controller。

出生是现有 logistic base + affine coupling flow（`density.py`），非标准 DeepONet；驱动也是现有条件 MLP。不得以命名替代实现。若以后改 branch/trunk，须独立版本。

形式上的分布方程为

    ∂_s f + v·∇_x f + ∇_v·[(-κx+b_θ)f]
      = ∇_v·(γ_θ v f) + γ_θ T Δ_v f + Q_φ[f; f_background].

γ仅依赖x，所以速度扩散可写以上形式；若以后依赖v需重新推导。Q的有限N实现和背景冻结见第4节。本文不声称已经证明N→∞闭合、h→0局部极限或传播混沌。

## 3. 二体碰撞、三守恒、空间定义

设 g=v_i-v_j，n∈S^{d-1}，α=g·n：

    v_i' = v_i - αn;  v_j' = v_j + αn.

动量变化为0。平方速度和变化=-2α(v_i-v_j)·n+2α²=0；位置不变。固定粒子数量与等权重使质量不变。法向必须归一。只能对同一对同时写回，不允许平均速度来替代反射。

空间核：K_h(r)=h^{-d} Π_a max(1-|r_a|/h,0)。默认h=1，固定写入config；不得调小h暗中减少候选或改变邻域，Λ也随h变化。这是 L∞ 盒支撑，不是欧氏球。若某坐标距离>=h，直接碰撞率为0；K_max=h^{-d}。有限h时为有限范围相互作用；全局动量守恒不等于每个位置的局部动量源为0。

pair center m=(x_i+x_j)/2。上下文作用域也定义为 K_h(x_l-m)>0，且背景坐标/速度来自当前碰撞子步开始的快照。

版本必须区分：
- `legacy_eps`：现有 scores 加 log(max(K,1e-12))，严格重现旧函数；远处背景不严格隔离。
- `strict_local_v1`：K=0处 score=-inf；正支撑内用真实log K。若上下文为空，用query-only分支raw_score=mean(output(tanh(q)))，再执行相同clamp、sigmoid、B_max，不能直接把raw_score当碰撞率。活动pair正常含端点；空集合仅用于接口健壮性。

strict_local_v1 是一项明确的语义修正；只与自己的串行参考进行精确等价验收。最终主线使用它。禁止把改 mask 后的结果称为与旧 epsilon 版本完全相同。

局部性的范围：token可全局广播，读出可全局池化，共享参数也由全局loss更新。因此本文的严格局部性约束的是固定参数/固定外部输入下的内部碰撞依赖，不是整个训练系统不存在全局通路。传播测量必须冻结参数；在线更新后的变化另报，不能称为纯局部传播。另外，有限邻域不自动给严格有限传播速度：Poisson次数无上界、连续速度无硬上界，均需以概率/响应尺度表述。

pair端点距中心最多h/2（L∞），所以背景对端点的直接条件依赖可达1.5h，尽管直接动量交换范围是h。测量与文档必须同时标注这两个范围。

## 4. 可学习碰撞率与冻结背景过程

对每个活动pair生成8个 query：对 (vi,vj),(vj,vi),(vi',vj'),(vj',vi') 各配 ±n。每行orbit=concat(第一速度,第二速度,带符号法向)∈R^{3d}，orbit整体[8,3d]，q为[8,Hc]；不拼位置或替换成相对速度。r_l=[x_l-m,v_l^background]∈R^{2d}。线性投影含原bias：q=Wq orbit+bq，k=Wk r+bk，a=Wv r+bv。

    attention = softmax(q kᵀ/sqrt(Hc) + log K_h(x_l-m)) a
    B_φ = B_max sigmoid(clamp(mean(output(tanh(attention+q))), -12,12)).

默认 Hc=128,B_max=1。对8个输出标量先平均再sigmoid；不得交换此顺序。在同一冻结背景下，交换pair、n反号、碰撞前后反射均使8元素集合置换，因此B保持不变。这不等于背景随碰撞变化时的完整详细平衡证明。

每个无序pair目标速率 λ_ij(n)=K_h(x_i-x_j)B_φ/N。法向测度采用均匀球面概率测度，总质量1；若使用球面积测度须改B归一。

用 uniformization：总候选率 Λ=(N-1)B_max K_max/2；M~Poisson(Λ Δ)。每候选均匀抽i和j≠i，n为归一高斯，ξ~Uniform[0,1)。接受概率 p=K_h(x_i-x_j)B/(K_max B_max)。因为无序pair抽中概率2/[N(N-1)]，Λ×该概率×p=K_h B/N，归一正确。

子步中位置及背景冻结，pair当下速度持续更新。参数独立Λ，允许仅抽总次数并按顺序执行，不必模拟时间戳。若驱动/背景/参数在子步内变动，该理由不再直接适用。不得cap Poisson次数。候选几何可批量生成，但RNG调用布局变化会改变seed轨迹；精确比较必须注入同一张候选表。

弱形式定义（给定背景g，形式层面）：

    <χ,Q_h[f;g]> = 1/2 ∫ dx dy dv dw dσ(n)
      K_h(x-y) B_φ(m,v,w,n;g) f(x,v)f(y,w)
      [χ(x,v')+χ(y,w')-χ(x,v)-χ(y,w)].

χ=1,v,|v|²给全局三守恒；χ含x依赖时不自动消失。原背景冻结是离散模型约定；到自洽Q[f;f]的近似误差要测，不能称为解析闭合已完成。

## 5. 依赖保持的并行算法

输入候选表固定，先筛 K_h>0；保持原event_id。对每个粒子维护last_level，初始-1：

```python
for e in active_events_in_original_order:
    level[e] = 1 + max(last_level[i[e]], last_level[j[e]])
    last_level[i[e]] = level[e]
    last_level[j[e]] = level[e]
for layer in increasing_levels:
    vi, vj = gather(current_v, pairs[layer])
    p = batched_attention_rate(vi, vj, frozen_context, normals[layer]) * geometry[layer]
    accepted = uniforms[layer] < p
    current_v = functional_scatter_unique_pairs(current_v, reflected_or_original)
```

同层pair不共享粒子；任一粒子的事件相对顺序保持。冻结背景下，两个不相交pair的状态变换相互交换，接受概率也不依赖对方更新。因此上述拓扑顺序保留给定候选表的计算语义（浮点舍入除外）。若背景改为即时状态，即使pair不共享粒子也可能有Attention依赖，证明失效。

冻结指值不随子步内部更新，**不是detach**。背景必须保留到子步入口状态的autograd路径。gather/scatter需函数式或经过验证的custom backward，禁止原地覆盖反向所需张量。

CPU参考允许顺序调度；GPU版本可先用单GPU线程构建小整数DAG、再按level排序压缩。不可预先假定它快。记录候选数、活动数、层数、最大/平均层宽。禁止高冲突时强行并行。

K/V共享：先对[x_l,v_l^background]投影；相对坐标的key平移对同一query所有l相同，在softmax中消去；value需显式减去Wv_x m。query、mask、距离核仍是pair相关。缓存生命周期仅一个冻结子步，不能跨token复用。

动态图与CUDA Graph：先eager正确实现，随后做容量bucket。固定buffer容量只限制一次提交，不限制事件总数；溢出保留全部候选，分块按序执行或fallback并计数。重复播放时必须更新随机数、几何及DAG，不能固定一次抽样。性能报告包含fallback。禁止把计算p时所需的vi/vj提前冻结。

## 6. 外力输运的精确子更新与整体分裂

冻结x、驱动c=-κx+b、γ和时钟，在时长a内：

    v_new = e^(-γa)v + (1-e^(-γa))/γ * c
            + sqrt(T(1-e^(-2γa))) ξ, ξ~N(0,I).

使用expm1计算小量；γ范围严格正，仍应测试小γ极限。参数梯度必须包含噪声幅度对γ的导数。ξ作为外部随机张量。

Transport(τ)：kick(τ/2) → x+=τv → kick(τ/2)，两次kick用各自位置，同一中点时钟。每token interval=1，S=4，Δ=1/S：Transport(Δ/2)→Collision(Δ)→Transport(Δ/2)，重复S次。随机增量独立。这里只规定对称分裂，**不未经证明声称整个随机/冻结背景方法是弱二阶**。

时钟以float64计算sin/cos，再转模型dtype；5000×256事件后float32不足以保留部分子步时间偏移。

## 7. 梯度：连续路径 + 离散接受事件

事件e的A_e∈{0,1}，logπ_e=A_e log p_e+(1-A_e)log(1-p_e)。只对活动候选记录，p=0的几何外事件贡献0。p的上界由sigmoid(clamp12)保证小于1；正确处理padding，避免0×log0。

strict参考/生产的稳定实现：logp=Σ_a log1p(-|r_a|/h)+logsigmoid(clamp(raw,-12,12))，仅对所有因子严格正的活动pair计算。接受判断用log(uniform)<logp（uniform=0对应-inf）。log_reject在logp<-log2时用log1p(-exp(logp))，其余用log(-expm1(logp))；通过有效索引分支避免无效分支产生NaN梯度。以A选择logp或log_reject，不做A乘logp。不得随意给p设最小epsilon改变概率律。legacy精确复现保留旧概率计算；strict串行与批量统一使用log域版本，并测试极小概率、uniform=0与概率近上界。

窗口目标J=E[L^{-1}Σ_k ℓ_k]，包含离散碰撞随机性。条件于历史、基噪声与候选表：

    ∇J = E[ L^{-1}Σ_k (∇path ℓ_k
          + (ℓ_k-b_k) Σ_{e≼k} ∇logπ_e ) ].

直接用autograd实现surrogate：

    mean_k [CE_k + stopgrad(CE_k-b_k) * prefix_logprob_k].

默认b_k=log(V)，常数独立于路径；不得拿当前路径的未来损失做未推导baseline。训练日志报告原CE，不报告surrogate当NLL。prefix只累计导致该预测之前的事件，窗口结束归零。候选数/对/法向的采样分布参数独立，所以无额外score项；核宽h和B_max在首版固定，不可随意学习改变proposal。

局部支撑的硬筛选按分段可微处理，零点与核折点不在gradcheck处；点态autograd通过不等于已经证明跨边界交换微分与期望合法。需要第10节的枚举检查。梯度裁剪改变优化更新，不能作为动力学稳定或无偏梯度证明。

## 8. 能量、耗散与持存：能推导什么

E_N=(2N)^-1 Σ_i (|v_i|²+κ|x_i|²)。连续参考由Ito公式，势阱与输运交叉项抵消，碰撞跳跃ΔE=0：

    dE_N = N^-1 Σ_i [v_i·b_i - γ_i|v_i|² + d γ_i T] ds
          + N^-1 Σ_i sqrt(2γ_iT) v_i·dW_i.

这是开放能量账，不是E恒定。期望稳态时平均输入功+热浴注入=耗散，仅是能量平衡条件，不自动给出临界或有效记忆。

用||b||<=A及γ>=γ_min：v·b<=γ_min|v|²/2+A²/(2γ_min)。故

    d E[E_N]/ds <= -(γ_min/2)E[mean |v|²]
                     + A²/(2γ_min)+d γ_max T.

此式只直接控制速度耗散，不能独自推出相空间全部矩的统一界。若要引入x·v的Lyapunov交叉项，有限h碰撞会改变该项，必须另外界定其生成元贡献。编码AI禁止把这一步省掉写成无限存续定理。

离散账采用逐子操作恒等式：kick分离确定性mean动能变化与实际噪声端点动能变化，drift记录势能差，collision记录动能差；它们之和必须重构E_end-E_start。OU端点噪声能量项不等于连续式dγTΔ，也不能与连续期望混加。记账用detached tensors但不得detach动力学本身。

若需要分列功/耗散，采用冻结kick确定性路径约定：q=c/γ,z=v-q,I1=q a+z(1-exp(-γa))/γ；I2=|q|²a+2q·z(1-exp(-γa))/γ+|z|²(1-exp(-2γa))/(2γ)。驱动功=b·I1，势阱功=(-κx)·I1，确定性阻尼损失=γI2；三者带符号相加等于mean动能差。实际随机端点残差另列，不能称为原始热注入。全部按粒子平均，记录正负号。

## 审核记录

独立审查者 local_kinetic_review 审核了归一、反射、冻结背景DAG、score梯度及能量账，未发现实质推导错误；提出的h默认值、orbit形状、空上下文、有界速率、稳定概率、1.5h依赖范围和功分解已补充。审查者质疑的128/1.5s门槛已与用户最新明确要求核对；256–512/~1s保留为扩展目标。审核是推导/规格审查，不代表生产实现、速度或临界性已通过。

## 9. 临界耗散：操作性规格，不暗造证明

用户约束：系统需通过开放耗散维持可塑且不塌缩的组织。本文不把该要求偷换为γ=0或谐振子的临界阻尼。

线性无驱动变化、常γ下，单坐标漂移矩阵[[0,1],[-κ,-γ]]的根是 (-γ±sqrt(γ²-4κ))/2。γ=2sqrtκ是欠/过阻尼边界；它不是历史响应指数为0，也不是自组织临界的充分条件。

必须同时报告：
1. 能量/幅度轨迹和账目；2. 标准化相空间协方差谱与有效秩；3. 同输入同噪声局部扰动响应；4. 历史状态对未来验证NLL的影响；5. 训练梯度与分布变化后的恢复。不把这些合并成未经标定的一个分数。

响应协议：固定checkpoint参数，从真实保存状态复制两个副本；对局部子集施加ε扰动，以出生或参考期固定尺度标准化x/v（不能每时刻重新归一隐藏爆炸）；ε∈{1e-4,3e-4,1e-3}，同OWT后续、同OU噪声、同候选随机表。分别测H∈{32,128,512,2048}事件。r(H)=||δZ_H||/||δZ_0||，λ_H=log r/H。

随机接受可能因扰动改变，必须报告接受分支分歧率。区分固定离散分支的路径导数与真正共同随机数的有限扰动；后者不自动收敛成Lyapunov导数。多扰动方向/同一checkpoint的方向采样不是多种群训练。有限窗口λ_H不能冒充最大渐近指数或完整在线学习系统的指数。

秩定义：标准化中心化样本矩阵的协方差特征值η_a>=0，p_a=η_a/Ση，r_eff=exp(-Σp logp)。trace近0时单独报告零尺度，禁止把归一化后rank正常称为未塌缩。不能要求16个四维方向正交。

临界判据中的目标响应区间、容许能量/秩下限和调节时间尺度目前 **OPEN**，必须基于明确理论/预注册标定确定。它们不阻塞参考实现与测量接口，但阻塞启用反馈controller和宣布临界达标。编码AI不得自动把临界设为λ=0并优化到它；不得把OU的γ参数和碰撞松弛/碰撞率混为同一量。

## 10. 必须实现的接口与验收

新目录 `scripts/ib_local/`，不改运行中的Slice文件。

```python
CandidateTable(i: int64[M], j: int64[M], normal: float[M,d], uniform: float[M])
FrozenContext(x: float[N,d], v: float[N,d])  # 保留autograd
CollisionResult(v: float[N,d], log_prob: scalar, accepted: bool[M], stats: dict)
sample_candidates(N, d, duration, width, max_rate, generator) -> CandidateTable
collision_serial(x,v,context,table,mode) -> CollisionResult
schedule_dependencies(active_i, active_j) -> levels, permutation, offsets
collision_batched(x,v,context,table,mode) -> CollisionResult
```

forward只返回离散事件log_prob总和；prefix与CE因果连接在trainer中处理。CPU float64参考、CUDA float32生产；法向/随机表/上下文来源均显式。固定候选表构造与路径状态分开，方便复现与梯度检查。

数值测试（非toy能力研究）：
- 反射两次恢复、法向反号等价、pair交换等价；三守恒。
- 活动/非活动盒边界、严格mask外状态扰动不影响该pair速率；保留identity不改变粒子编号。
- 无共享、链状共享、重复pair、多层、M=0、全不活动和overflow；分层调度逐粒子顺序不变。
- 注入同一候选表，serial/batched的accepted完全相同，状态/log_prob/各参数及入口x/v梯度close。预先检查uniform不接近接受阈值；临界阈值例另报浮点分支敏感性。
- FP64参考 atol1e-10 rtol1e-8；CUDA FP32起始 atol2e-5 rtol2e-4。梯度相消处同时报告绝对误差，不擅自放宽容差；多事件累计误差单独曲线。
- 1–2个候选完整枚举所有接受分支：直接求Σ_branch P_branch L_branch并autograd，与Σ_branch stopgrad(P_branch)×surrogate梯度比较；包含后事件p依赖前事件状态、输入位置梯度。有限差分仅作辅助。
- OU输出与梯度、子步能量账、事件因果对齐；出生仅一次；恢复checkpoint后下一窗口状态/参数/RNG/事件计数一致。

## 11. 训练/性能协议

数据复用 `data/ib_owt_gpt2` 的已固定OWT子集/划分/manifest哈希，不称完整OWT。读出沿用H128 tied embedding候选，参数数量/共享/初始密度与梯度长度在config写明。AdamW lr3e-4, weight_decay.01,clip1；不顺手改超参。

性能先验证N512,d4,H128,S4,L128，再L256/512。计时前后同步，至少10个预热真实更新、100个测量更新，排除初次编译但另报；CPU数据准备、RNG、GPU前后向、score项、clip、optimizer、状态传递都计入。单个体不能用batch多个独立个体凑吞吐。不与现有训练并行压测。记录GPU利用率、专用/共享内存、温度、事件层宽、fallback与峰值；不把NLL改善当性能验收。

达到门槛才启动下一版长训练。比较匹配训练targets预算（不是只匹配updates），同时报告墙钟。验证使用独立状态/RNG，同一固定验证流；保存best、last、阶段权重及输入cursor、动力学时钟、优化器、RNG、proposal算法版本、mask模式、训练历史和源文件哈希。未完成状态/参数续训一致性测试，不允许长跑。

## 12. 决策表与开放问题

- 原过程同义加速达标：主线继续，不引入LBM替换。
- DAG长期近串行：提交瓶颈证据；局部分轮近似作为新版本提案，不自动实施。
- loss下降但rank/历史响应失效：检查归因，不能宣布临界。
- 数值正确但未收敛：记录预算受限，不否定架构。
- strict mask改变旧结果：预期的模型修正，不能伪装纯优化。
- OPEN：完整自洽碰撞闭合、有限h空间误差、长期持存充分条件、临界判据阈值。执行者不得在文档中自行补上定理。

参考定位：`collision.py`（原随机模型），`force.py`（OU/输运），`density.py`（出生flow），`streaming.py`（prefix score），`scripts/ib_bpe_window.py`（仅作为批量投影/读出工程参考）。

外部方法边界：XLB BGK源码 https://github.com/Autodesk/XLB/blob/main/xlb/operator/collision/bgk.py；Lettuce源码 https://lettuceboltzmann.readthedocs.io/en/stable/_modules/lettuce/collision.html。仅借鉴实现组织；它们的局部平衡松弛不是本文Attention跳跃核，禁止以引用这些库证明本文模型。
