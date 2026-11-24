# 梦漓：先提高吞吐，再持续训练基础语言能力

用户于2026-11-23将主线切换为训练吞吐与独立材料NLL；随后明确“先继续优化速度吧”。当前只推进速度工作，下一段基础训练待本轮加速验收后接续。

## 已确定的成本与执行对象

### 2026-11-23 突破交付：完整 1:1 加速栈（14.39秒/轮）＋事件驱动脑波相位耦合节律（9.09秒/轮）

完成 `stacked_pyprofile` 全量剖析后，先落地 4 批严格 1:1 优化并完成 2 轮无干扰核对（`batch4_quiet_full`）和 8 轮基准（`strict8_baseline`），随后按用户批准实现**事件驱动＋脑波相位耦合多速率节律**（`--phase-coupled-rhythm`，详见 [CGB_PHASE_COUPLED_RHYTHM_20261123.md](CGB_PHASE_COUPLED_RHYTHM_20261123.md)）并完成同源 2813 的 8 轮头对头对照（32 更新、7680 外界字节、独立 FineWeb 8192 字节验证＋64 旧事实＋存档恢复核对）：

1. **Batch 1—4 严格 1:1 加速栈**（`gdn_decode_step` 预切片＋`_foreach_copy_`、`causal_archive` 向量化脏槽跳过与批量 `warm_relation`、`variable_expression` 合并 readback 与右填充变长策略梯度合批 `--policy-proposal-batch-size 4`、`noise_prior` 作用域缓存、`_retain_constraints` 提核移出循环并 CPU 求解、修复 `_release_if_tight` `gpu_budget_bytes` 键名彻底消除冗余 `gc.collect()`）：
   - 2 轮无干扰基线 `batch4_quiet_full`（[cgb_throughput_batch4_quiet_full_20261123.json](../../results/published/cgb_throughput_batch4_quiet_full_20261123.json)）：纯生活 `34.144 秒`（较参照 `50.624 秒` **1.483×**、较初始 `64.428 秒` **1.887×**），存档 `2.462 秒`（**9.77×**），端到端 `36.607 秒`（**2.040×**）；全部实际端口、9542 银行叶项、S/P、RNG 零差异，模型最大浮点差 `5.96e-8`。
   - 8 轮严格 1:1 基准 `strict8_baseline`：纯生活 **`115.115 秒`**（**`14.389 秒/轮`**、`66.72 B/s`，较初始 `32.214 秒/轮` 提速 **2.24×**），存档 `2.427 秒`，峰值显存 `2.158 GiB`，独立 FineWeb NLL `1.583043 → 1.580249`（`+0.002794`），64 旧事实 `-1.137590 → -1.132076`（`+0.005514`）。
2. **事件驱动＋脑波相位耦合多速率节律**（`phase_coupled_rhythm.py`、Lean 4 [PhaseCoupledEventRhythm.lean](../../lean/CGBAxioms/reviews/PhaseCoupledEventRhythm.lean) 7 条定理通过、单元测试 7/7 通过、显式开关 `--phase-coupled-rhythm`）：
   - 8 轮实机 `phase8_rhythm`（[cgb_phase_coupled_rhythm_8cycles_20261123.json](../../results/published/cgb_phase_coupled_rhythm_8cycles_20261123.json)）：纯生活 **`72.686 秒`**（**`9.086 秒/轮`**，纯 Gamma 轮最低 **`5.939 秒/轮`**，**`105.66 B/s`**，相对 `strict8_baseline` 再提速 **1.584×**、相对初始提速 **3.545×**），存档 `2.547 秒`，峰值显存降至 **`1.983 GiB`**。
   - 独立 FineWeb NLL `1.583043 → 1.582251`（`+0.000792`），64 旧事实 `-1.137590 → -1.133782`（`+0.003808`），独立恢复审计确认振荡器相位、SWR 压力、H-JEPA 活动主题与配置完全一致。原 2813 与主 2401 保持，全部 GPU 任务已结束。

### 2026-11-23 继续交付：经历撤回合批与完整存档压缩

完整2813源、相同冻结源码和实际两轮生活已完成串行/合批比较。每臂2外界轮、8次更新、1920外界字节、128自生字节，两个原生教师预算和全部候选沿用。

| 实际对象 | 串行参照 | 私有合批及复用 | 结果 |
|---|---:|---:|---:|
| 4组真实经历撤回用途计算，32笔身份/版本记录 | 5.887857秒 | 1.757503秒 | 3.350倍 |
| 全部完整前向调用 | 66次 | 34次 | 省32次调用 |
| 生活调度及末尾64旧事实评分，保存检查点前 | 50.624107秒 | 48.940001秒 | 省1.684107秒，3.327% |
| 从生活开始到最终检查点发布 | 74.690015秒 | 75.015124秒 | 两次实际原始计时，含各自存档 |

原来每组撤回先重算一次事实预测，再串行撤回8份经历。现在更新前直接复用实际学习目标的logits，8个独立撤回沿B轴并行通过原32层；每个分支带自己的slot/version、共演化valid掩码、深读屏蔽和问题作用域。原参数、身体起点、字节、快表概率及问题均来自同一实际计算时点。各分支的结果用于原保留统计，原学习图继续完成反传。`reviews/NativeReplayBatch.lean`证明逐层独立复合在各B分支还原串行轨迹，实码绑定由CPU/CUDA张量和完整存档核对完成。

两实际行动、全部话语字节、世界位置/内容、支付、预测读数、优化器5382叶项、快表、全部RNG、S/P、待结算和两个学习银行逐项相同。模型705项中703项精确相同；两项保留用途统计最大差2.765446e-7。32笔用途身份/版本/年龄相同，benefit最大差1.907349e-6（原单位10*nats/byte），归一化后的preference最大差2.357662e-4。64旧事实终点评分精确相同。显存峰值2,131,032,064→2,314,707,968字节，增加约175 MiB。27项分支隔离、问题作用域、梯度、资格、共演化和原生端点检查通过。

真实32层只读面板另有镜像顺序局部计时：9分支（事实加8撤回）宽1/2/4/8中位数1.315133/0.799104/0.536290/0.479212秒；最大logits差9.536743e-6。此项保留事实重算，生产合批另外省去已有事实预测的重复计算。普通续行通过`--native-replay-batch-size 8 --reuse-objective-logits`启用并保存，恢复沿用`config.native_replay_runtime`。

**存档是本次继续解决的大块。** 计时对账发现最终约0.52GB gzip存档另占24—26秒，独立于约49秒生活/评分。`result.runtime_profile`原先在保存后再次snapshot，其seconds混入了存档，而`runtime_profile.json`在保存前落盘；现已分别记录`life_seconds`和`checkpoint_commit_seconds`，结果内的runtime_profile也固定在保存前。

在相同872,629,881个完整检查点字节上，标准gzip level1为24.392700秒，ISA-L level1单工作线程为1.567164秒、4线程为1.703636秒；选择单工作线程。文件大小519,062,128→519,143,383字节，增加81,255字节（约0.0157%）。三者经原标准gzip读取器解压后的SHA256均为`143e8f729da0d9ada696b7cd4536a97ddd5da1801512b46fa6833261f322bd5c`，完整原文件SHA保持，0更新、0世界调用。压缩输出保持单一标准gzip流和原原子提交/完整压缩字节哈希。实现依赖[python-isal官方流式接口](https://python-isal.readthedocs.io/en/stable/index.html)，有界1MiB分块；运行环境已安装isal1.9.1，可选依赖为`pip install -e ".[runtime]"`。

普通续行启用`--checkpoint-compressor isal --checkpoint-compression-workers 1`，存档保存`checkpoint_compressor`并在恢复时核对依赖。17项完整footer哈希、标准旧读取、spool读取、配置继承、原子提交及写盘失败线程退出检查通过。首次`replay_fast_checkpoint`因`run_native_throughput_worker.py`解析器默认`allow_abbrev=True`导致`--checkpoint`与`--checkpoint-compressor`/`--checkpoint-compression-workers`前缀冲突以`exit_code=2`退出（原失败日志保留）；修复为`allow_abbrev=False`后，叠加全部已验收加速项（单字节五核融合、撤回合批B=8、复用目标logits、直接类型问题、后继分解、热表预热、E盘spool加载与ISA-L单线程压缩）完成同源两轮`replay_fast_checkpoint_v2`：

- 生活计算及末尾64旧事实评分：`life_seconds = 39.608171`秒（`runtime_profile.seconds = 39.606924`秒），相对`replay_reference_ready`（50.624107秒）达**1.278倍**，相对初始64.428秒达**1.627倍**；峰值CUDA分配2,314,707,968字节（约2.156 GiB）。
- 完整检查点序列化、ISA-L压缩与原子提交：`checkpoint_commit_seconds = 2.910539`秒（原gzip约24.07—26.08秒，提速**8.3—9.0倍**），归档大小517,766,851字节，标准`gzip.open`一次读回通过。
- 端到端总耗时（生活＋存档）：**42.519秒**，对比原版`replay_reference_ready`的74.690秒节省**32.17秒（1.757倍）**。
- 与`fused_hot_candidate`逐项审计（`results/published/cgb_throughput_replay_fast_checkpoint_20261123.json`）：全部实际端口、优化器5382叶项、表、世界、全部RNG、S/P、待结算813叶项、两个学习银行9542叶项及64旧事实520叶项全部零差；模型705项中703项精确相同，仅`memory.retention_utility_{first,second}`两项差`2.765446e-7`。

报告：`results/published/cgb_replay_batch_20261123.json`、`cgb_throughput_replay_batch_20261123.json`、`cgb_checkpoint_compression_20261123.json`、`cgb_throughput_replay_fast_checkpoint_20261123.json`。完整参照、合批与全叠加速身体分别为`D:/mengli-throughput-20261123/replay_reference_ready/body/last.pt.gz`、`replay_batched_ready/body/last.pt.gz`、`replay_fast_checkpoint_v2/body/last.pt.gz`；共同原2813及主2401保持。执行状态位于`scratch/cgb_throughput_20261123/replay_fast_checkpoint_v2_status.json`。

### 2026-11-23 用户要求改抓大块后的交付

已完成单字节GDN执行器融合与低内存检查点一次落盘读取，完整结果如下。下一步以整圈总时间为分母选择对象。此前“下一块并行撤回”调整为完整前向调用的来源与共用条件核对；现有66次前向中同时含学习、草稿重算、端点和用途测量，按真实调用来源组织批执行与复用。

| 验收对象 | 原版 | 改版 | 实际收益 |
|---|---:|---:|---:|
| 真实2815身体，32层，128个相同真实文本字节，4次中位数 | 0.641959秒 | 0.312723秒 | 执行段2.0528倍 |
| 原2813同源完整两轮，统一冻结源码及热编译缓存 | 50.194393秒 | 48.330678秒 | 省1.863716秒，时间减少3.713%，吞吐1.03856倍 |
| 原2813完整检查点，低内存gzip寻址/一次落盘读取 | 28.745335秒 | 4.345585秒 | 加载6.61484倍，省24.399751秒 |

**为什么整体仍只节省约两秒：** 709次字节步按固定重放实测折算，原神经解码约`709/128*0.641959=3.56`秒，占50.19秒约7.1%；将它减半，整圈预计节省约1.82秒，实测1.86秒。开口前整个任务的19.86秒还包含关系执行、预演、候选用途和教师计算。用这个分解直接安排下一个对象。

融合实现为`gdn_decode_step.py`：归一化与6路投影、原QKV卷积/归一化/衰减与delta递推/容量投影、输出残差、SwiGLU和下投影残差，d256/H8/C4/M768的一层落为5个核心CUDA核。参数始终来自原层；图引用的投影缓冲在每次start重新写入。序列训练和导数继续走原实现。Lean `reviews/NativeDecodeProjection.lean`核对角色投影的打包索引，深度独立复合继承`NativeWordBatch`；真实FP32的归约次序由数值比较验收。

固定重放logits最大差5.722e-6，持存态最大差9.060e-6。完整两轮两实际选择、全部生成字节、导航、快表、世界、所有RNG相同；学习后模型最大差4.292e-6，优化器最大差1.682e-5，64份旧事实评分最大差9.537e-7。数值差同时进入内容哈希和两个学习银行的样本身份，按数值与功能核对记录。完整GPU峰值2,131,032,064字节，约1.985 GiB。

低内存加载路径使用显式`CGB_CHECKPOINT_SPOOL`目录中的有界匿名临时文件，顺向解压后由PyTorch正常寻址读取；文件关闭自动释放。实际原2813的29,622个完整状态叶项全部相同，原存档SHA不变，暂存目录最后为空。命令启动器提供`--checkpoint-spool E:/mengli-runtime-staging-20261123`，普通续行可用同一环境变量。普通生产续行已提供并保存`--fused-decode-step`配置，恢复继承该执行设置。旧身体沿原配置恢复。

计时另补`--reuse-runtime`：逐项验证已有冻结树的SHA后复用同一路径及编译缓存。首个冷融合任务77.03秒保留为冷启动记录；配对结论使用统一热缓存的50.19/48.33秒。完整状态参照位于`E:/mengli-throughput-20261123/fused_hot_reference`，候选位于`D:/mengli-throughput-20261123/fused_hot_candidate`；冷融合记录迁往E盘保留。新增三组各两轮，共6外界轮、24次优化更新、5760外界字节。主2401与原2813保持。

验证：16项单事件/图刷新/梯度与加载接口检查，加13项原GDN融合前向和全导数检查，共29项通过。完整报告为`cgb_decode_fusion_20261123.json`、`cgb_throughput_fused_hot_20261123.json`、`cgb_checkpoint_spool_20261123.json`。当前全部任务已结束。

**接续目标：** 当前热参照66次`predict_observed`共12.98秒，策略梯度共7.03秒；生成核与缓存的后续优先级按上述整圈比例安排。完整前向的计时现附调用位置，下一次采集直接拆出真实学习、草稿、端点、撤回和未来后果评分的各自工作量；执行器按固定参数、字节前缀、身体起点、经历版本和掩码这些实际条件合并，真实更新与世界到达负责切分生命周期。动作、教师、优化更新与输入预算沿当前设置。

共同源为完整原版2813：`E:/mengli-native-supply-completed-20261122/original/last.pt.gz`，SHA256 `3f890fda0a56518a88bc66ea47d0e8a5f991a8b04a464c046b995d8ad819b27e`。32层、30,500,893参数，完整自然文本闭环。每轮960外界字节、4次优化更新，主2401另行保留。

此前2805→2813八轮：整圈dispatch229.011秒，28.626秒/轮、33.535外界字节/秒。新2813同源两轮含运行中末次64旧事实评分的profile为64.428秒。调用包含重叠，以下按父子关系阅读：

| 调用 | 两轮累计秒 |
|---|---:|
| 开口前 `_utter` | 31.667 |
| 两个原生教师预算的展开 `flush_teachers`，位于上行内部 | 7.975 |
| 实际策略梯度 `_policy_grad` | 5.788 |
| 四次实际 `_advance` | 3.736 |
| 强制GC与CUDA缓存清理 | 0.830 |

`world_arrived`阶段从事务开始累计到真实回应抵达，包含问题选择、草稿、预演、行动与导航。该阶段耗时通过调用树进一步拆解。

## 已实施与已核对

1. **不可变关系快照复用。** `prospective_decision.preview_bind`只在一次只读预演内共享同一关系的引用解析与支持筛选；各分支随后单独覆盖其经历。`causal_relation_program.mechanism`直接构造正典路径的空前缀，省去旧热槽路径的空神经运算。
2. **按实际显存余量分配激活。** 运行配置由GPU512/host512 MiB调到GPU1536/host768 MiB，沿用原tiered自动余量限制与原始stride，超额才进入临时磁盘。
3. **省去被当前问题替换的旧式菜单。** 父类先加旧问题激活，子类又减去同一激活并换成实际抽样的typed filler；`capture_actual`同时清空此前教师队列。`typed_replacement=True`保留真实绑定、历史问题贡献、引用观察与实际菜单，跳过旧菜单的完整预演。当前用`--direct-typed-question`显式启用。Lean绑定见`reviews/NativeTypedQuestionRuntime.lean`。
4. **固定数据的编码投影复用。** `encoder_read`绑定实际拥有该encoder的预演；按参数与输入Tensor版本区分。整张经历表、关系地址键与真实事实的编码可在多个查询之间共享。梯度调用执行原图。
5. **完整只读菜单共用缓存。** 实际问题菜单、两个原生教师的展开各拥有一个只读作用域，内部嵌套的功能性分支继承缓存。各假想事实、上下文、资格和teacher target按各自键保留；真实参数/经历写入发生在作用域结束后。
6. **检查点单次解压。** `compressed_checkpoint.checkpoint_stream`在内存充分时顺向解压一次，再让PyTorch读取同一字节流；内存紧张与较大存档使用原gzip流。上限1GiB，实际读入同样受限。原文件格式、哈希、原子提交和恢复内容沿用。

## 同源验收

| 版本 | 两轮profile秒 | 相对初始 |
|---|---:|---:|
| 初始 | 64.428 | 1.000× |
| 关系快照/空前缀/激活放置 | 62.629 | 1.029× |
| 再省去被替换的菜单 | 57.171 | 1.127× |

第二行的完整最终状态与初始逐项相同：705项模型状态、5382项优化器叶项、表、世界、全部RNG、S/P、待结算及两个学习银行均相同。报告：`results/published/cgb_native_throughput_snapshot_20261123.json`。

第三行是代数等价的上下文简化。两实际行动、生成字节、导航、世界、快表和RNG相同；支付最大差2.235e-7，实际actor概率最大差3.845e-6，64旧事实读数最大差2.921e-6。浮点加减消去改变了少量数值，完整模型最大差1.648e-5，内容哈希选存的学习银行随后也出现差别。该行按数值精度与实际功能验收，沿用显式运行开关。报告：`cgb_native_throughput_direct_20261123.json`。

新增固定投影复用的两轮与第三行完整状态逐项相同，报告`cgb_native_throughput_projections_20261123.json`。该次计时同时遇到Claude的GPU试验与CUDA单元检查，计时单列为受竞争影响的原始记录；速度结论使用随后串行计时。

## 当前执行游标

2026-11-23 接管 Claude 后更新：下列 `scoped_serial` / `factorized` 状态属于早先任务；其完成、重试及后续等价核对见后文。当前接续为 `takeover_reference_status.json`：冻结当前源码，从原2813运行两轮，GPU2048/host256 MiB。之后同配置检验 `--internal-step-review`。加载准入显式设2048 MiB；检查点加载器依实际余量选择流式读取，运行文件位于D盘。此调度选项保留为显式候选，比较教师预算、真实更新与后续动作后作决定。

- `scripts/start_native_throughput_profile.py`：冻结仅scripts源码，等待GPU使用者完成和30秒安静窗口，再运行同源完整两轮；启动前也识别尚在CPU恢复的已知GPU试验进程。
- 最新串行候选：`scratch/cgb_throughput_20261123/scoped_serial_status.json`，日志同目录`scoped_serial.log`，输出`D:/mengli-throughput-20261123/scoped_serial`。
- `scoped`是与另一组试验同时恢复后开始的较早候选，保留其事件与完整结果用于数值核对。运行过程依实际status核对。
- CPU数值/梯度/生命周期测试已完成；新loader覆盖缓冲、低内存回退、大小回退、异常关闭四种情况。涉及GPU的数值检查与吞吐计时串行安排。
- 完成串行候选后：对照第三行的同两轮真实事件与完整状态，读出教师展开与整圈所节省的秒数，随后处理剩余最大的实际调用。用户当前要求继续优化速度。

## 2813→2815 串行续验：话语前缀、激活放置和后继拟合

factorized 首次任务因 D 盘反传溢写空间不足退出。冻结版本在 C 盘私有目录从完整2813重跑成功，两轮53.815秒。与保存的direct参照，实际行动、支付、模型/优化器、快表、世界和RNG相同；原生价值银行四个质量统计差≤1.02e-8。旧scoped_serial已由direct完整存档承接。

本次额外改动：一个行动的观测、源、有限证据及模型固定，先生成共同原坐标锚点、编码与右逆，再分别投影每个后继积分点；教师键保留原字节顺序，点及质量一起回CPU。完整候选耗时50.486秒，另以相同显存准入配置、冻结前缀版做单因素对照50.977秒。局部估值44调用5.951→5.062秒，整圈节省0.490秒。与该对照逐项核对，含原生价值银行与行动概率在内，两次实际行动/支付、模型、优化器、S/P、世界、RNG、待信用及64旧事实读数均相同。峰值CUDA分配约1.90 GiB。与历史基线64.428秒比，整套配置观测吞吐1.276倍；仅各两轮单次profile，不外推长期速度。当前最大排他计时是 `body.predict_observed`（66调用、13.25秒）。结果：`results/published/cgb_throughput_factorized_completed_20261123.json`、`cgb_throughput_prepared_20261123.json`、`cgb_throughput_prepared_matched_20261123.json`。基础训练未启动。

## 撤回同一经历时复用固定投影

草稿快表资格包候选两轮57.595秒，热跑60.561秒，已撤回。随后定位到真实文本学习的保留用途测量：每段做一次事实重放，再逐个撤回同批版本经历。多字节只读测量复用既有解码作用域，按该段的当前参数重新投影事实；每次撤回仍重新运行32层并应用各自的有效掩码。单字节不切换到融合解码。实际训练、作用域外读出和支付不改。

首次冻结候选57.637秒。相同冻结目录热缓存，依次运行原测量和优化测量，各从原2813完成2轮、8更新、1920外界字节：整圈profile 50.493→49.935秒；40次`body.predict` 6.750→5.902秒，66次`body.predict_observed` 13.285→11.896秒。实际端口、模型、优化器、快表、世界、随机状态、所查持存及64旧事实评分逐项相同。7项定向测试通过。`_utter`两轮21.143秒（含预演），是下一段主要目标；主机侧嵌套项不可加总。单次热配对不作持续吞吐保证。结果 `results/published/cgb_throughput_withdrawal_matched_20261123.json`，首次对照 `results/published/cgb_throughput_withdrawal_scope_20261123.json`。

## 内部交叉核算：一次确定逐假设评分的旧支持

`flush_teachers`在真实行动前最多选两个假想节点，用同一身体原生的下一步问题、话语、行动和有限U估值产生内部目标。后到世界样本另行结算。即时评分只消费本次释放后的事实集合：共用旧支持的释放决定，保留每个假想回应及其独立后验评分，省去只会影响再下一次释放的统计量；完整`revise`保持原行为。

同源2813两轮：C盘参照/候选的预演核算6.388/4.995秒，局部预期用途2.347/.565秒，整圈49.935/50.385秒。G盘工作目录复测（表与激活仍在C）核算4.436秒、整圈105.022秒，不与C盘做速度归因。两次完整状态与行动审计所查项均零差，36项测试通过。保留局部计算优化，整圈基准未刷新。结果：`cgb_throughput_immediate_support_20261123.json`、`cgb_throughput_immediate_support_repeat_20261123.json`。

## 可选多速率交接：语言更新照常，慢复核随内部行动

`--internal-step-review` 随 `native_amortization.review_cadence=internal_step` 存档。直接释放时依赖现有预测器；已释放行动的回执含内部步时才展开原生参考（仍至多两个）。不消费下一行动正在累积的资格，不调整字节预算，不跨更新缓存预测，不改变经历身份与延迟回报的归属。旧档默认逐次复核。独立等价改动：同次持存方案读取合并价值/先验/新颖度标量的主机传输。`tests/test_persistent_rhythm.py` 七项通过，包括实际语言更新、延迟信用与完整保存恢复接下一行动。全尺寸私有计时在启动前因主机可用2.62 GiB低于原3 GiB门槛停止；本项无新吞吐或NLL结果。

### 接管后的完整验收

原2813、同一D盘布局、GPU2048/host256 MiB、各两轮八次真实更新。参照69.280215秒，候选66.336012秒，节省2.944203秒（4.2497%）；教师展开7.8201→4.5761秒，整个开口前计算28.5327→25.2249秒。第一轮内部步0、教师2→0；第二轮内部步2、教师仍2。实际行动、话语、概率、支付、参数、优化器、S/P、快表、世界、RNG和64份旧事实评分逐项相同。后继规划银行有207个叶项变化，其训练样本随减少的内部参照发生变化；此项继续作为显式调度候选。真实世界标签两轮均结算。报告 `results/published/cgb_throughput_internal_review_20261123.json`。

同D盘参照的完整状态已与保留的C盘 `immediate-support` 终点逐项核对，报告 `cgb_throughput_takeover_reference_20261123.json`；计时使用本轮D盘参照69.280215秒。C盘历史50.385139秒单列保存，避免混入本轮配对。

启动器另修订两项准入：显式指定主机可用量门槛（下限2048 MiB，原默认3072）；已确认的桌面GPU进程可通过PID记录并共存，仍要求至少3072 MiB空闲显存，并继续排队等待训练Python。此次GameViewer PID23136此前被当成训练占用，导致等待；保持该桌面进程运行后完成上述配对。

## 同一固定身体边界共享候选比较与教师展开

接续候选 `--shared-utter-preview`：位置在 `GenerativeCycleMixin._value`。先完成实际内部S/P运动、关系保护和真实问题绑定，再共享当前候选比较与教师计算中的只读投影。问题菜单保留自己的只读作用域。沿用 `NativeWordBatch` 固定前缀/独立端口合同、现有 `preview_runtime` 的输入版本与分支键、梯度绕过及作用域退出；各问题的FiLM仍在原端口单独施加。实际话语交付发生在作用域退出之后。

生产函数检查覆盖已走过一轮实际学习的身体：话语、实际记录、完整模型、快表、决策状态与RNG逐项相同，作用域正常释放。另有实际归档两轮回归：每次真实 `shared_unknown_bindings.select` 都位于缓存作用域之外。两组检查合计11项通过。全尺寸同源两轮见 `scratch/cgb_throughput_20261123/takeover_decision_preview_status.json`；本候选保留原每轮两次教师预算，与多速率调度分别验收。

此前 `takeover_shared_preview` 将作用域放在整个开口过程，跨过真实问题绑定的SQLite写入；第二轮一个教师值从−.196087变为−.198143，进而改变银行的两项质量统计。其模型、实际行动与旧事实评分虽相同，教师目标存在这一具体差异，故该实现已撤回，日志/冻结代码/完整比较保留于 `cgb_throughput_shared_preview_20261123.json`。上述回归直接锁定该真实写入边界。

修正后 `takeover_decision_preview` 完整两轮的实际端口、参数、优化器、S/P、快表、世界、RNG、两个学习银行及64旧事实评分全部零差。计时71.735789秒，继续保留显式实验开关；默认沿用此前的读取范围。报告 `cgb_throughput_decision_preview_20261123.json`。

## 4 GiB显卡的原生激活驻留核对

`takeover_device` 从同原2813完整执行两轮、8更新、1920外界字节，草稿、教师与学习预算沿原设置，激活选择现有 `device` 模式。完整最终状态及实际端口全部逐项相同，峰值CUDA分配2,033,854,976字节（1.894 GiB）。本组30.5M身体、128字节话语上限和480+480字节外界配置可以完成原生显存反传。

计时70.203635秒，对应同盘tiered参照69.280215秒；当前分层放置已经容纳这组工作负载，后续速度预算转向重复的前向与闭环核算。报告 `results/published/cgb_throughput_device_20261123.json`，完整终点 `D:/mengli-throughput-20261123/takeover_device/body/last.pt.gz`。

## 下一块：同一真实语言片段的并行撤回重放

当前两轮仍有66次 `predict_observed`、709次单字节决策步、44次摊销估值，真实优化更新8次。`native_retention_feedback.measure` 对同一语言片段逐个撤回版本经历，每次重跑32层；这些轨迹具有相同输入字节与起点，仅旧经历有效掩码不同。下一步沿已验证的 `NativeWordBatch.original_layers_batch_pointwise` 做私有B轴：为各撤回分支分别携带记忆和共演化工作集掩码，整批执行原GDN，再分别计算同一语言损失。实际身体、学习和结算仍逐事件执行。先在身体抽象中核对批轴可分离前提及原代码映射，再完成逐分支原预测/梯度状态核对和完整闭环验收；目标是减少重复32层调用，而非调整用途目标或预算。

## 空间与恢复材料

接管时核对六份完整比较报告，七份C盘测速模型均已由 `C:/Users/85750/mengli-speed-immediate-support/body/last.pt.gz` 承接其所核对的完整模型、优化器、表、世界、RNG、持存和旧事实读数。保留日志、结果、源码和各份配置；清理重复模型3,633,460,204字节。逐文件SHA、保留终点SHA及报告链见 `scratch/cgb_throughput_20261123/equivalent_checkpoint_reclaim.json`。原2813及主2401继续保留。

D盘早期 `snapshot`、`direct` 两轮测速模型在各自对照交付后退役；源2813、冻结源码、完整比较和事件保留，删除前均核对SHA并记入 `retired_snapshot_checkpoint.json` / `retired_direct_checkpoint.json`。本轮D盘参照由已核对的 `immediate-support` 完整身体承接，记录 `takeover_reference_reclaim.json`。多速率候选完整2815继续保留。

原2813、主2401、日志和冻结源码保留。原有已完成实验的临时快表，经对应完整2612/2613检查点SHA256核对后回收。新初始profile2815由完整状态逐项相同的`snapshot/body/last.pt.gz`承接；`projections`由完整逐项相同的`direct/body/last.pt.gz`承接。对应替换与快表回收记录在`scratch/cgb_throughput_20261123/`。这样每个保留状态都有实际完整恢复材料，训练数据和唯一身体存档保持。
