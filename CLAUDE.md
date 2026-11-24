> **2026-11-23 主线去重修订：** Codex供给对照已完成，通用SADL默认关闭。按行动差找弱方向已在目标敏感读出/持存谱残差路线实施，新SADL默认任务撤回。当前复核见docs/information_boltzmann/CGB_MAINLINE_STATUS_AND_DEDUP_20261123.md；旧三点C验收的缺口与后续完整结果分别记账。继承Claude反事实工作，GPU已释放。

> **关系用途实验已完整2489：** 查看台账顶部及CGB_RELATION_RETENTION_USE_20261119.md。两臂预算结束、恢复通过，不重复启动。

> **关系用途实验已完整2489：** 查看台账顶部及CGB_RELATION_RETENTION_USE_20261119.md。两臂预算结束、恢复通过，不重复启动。

> **2026-11-19 研发接续：** Claude批准的关系后到用途计划已由Codex落实。先读 `CGB_RELATION_RETENTION_USE_20261119.md` 和台账顶部。主2401/源apply2481保持，配对目录F:/cgb_relation_retention_2481_20261119。全套真实/潜U、scope统计及commit、收益v4、fresh冻结评分一起接通；运行源码受pipeline哈希冻结。以完整result核对完成轮次，不重复启动。

# CLAUDE.md

> **2026-11-23 大块加速接续：** 以`CGB_NATIVE_THROUGHPUT_20261123.md`顶部交付为准。单字节原GDN五核执行器在真实32层存档上2.05倍，完整热配对50.194→48.331秒（省3.713%）；字节解码约占整圈7.1%，其减半收益与整圈1.86秒对应。低内存原2813加载28.745→4.346秒，29,622状态叶项零差；E盘显式暂存，读完释放。`--fused-decode-step`已接普通续行/恢复，`--reuse-runtime`验证冻结SHA后复用编译缓存。29项定向检查通过；本次6轮24更新已完成。主2401/原2813保持，完整候选D:/mengli-throughput-20261123/fused_hot_candidate。下一预算按66次完整前向12.98秒及策略重算7.03秒的实际调用来源组织；原奖励/actor/教师/输入预算沿用。先继续吞吐。

> **梦漓当前入口（2026-11-18核对）：** 先读 `AGENTS.md`、`docs/information_boltzmann/CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md` 与 `CGB_MASTER_PLAN_20261114.md`，再按 `CGB_DOCUMENT_MAP_20261114.md` 查专题。台账顶部及实际监督/完整存档优先于旧文中的“当前/停止/正在运行”；README和总计划内部仍有历史运行段，不能据此恢复旧队列。下方视觉架构说明保留，不是当前CGB身体的描述。

> **架构唯一来源：** `docs/NORTH_STAR.md`。修改多模态主图、统一端口或生成前必须先读该文档与 `research_tree.json`。必须保留全分辨率点场、瞬态 Slice、语言联合演化和 Slice/Deslice 语义写回；不得以独立生图模型绕过核心框架。

> **CGB 必读规范：** 分析、设计、修改或验收前读 `docs/information_boltzmann/CGB_FIVE_PHASE_DIAGNOSTIC_RULES.md` 及活规格五系统段，压缩/新窗口后同样适用。五行是同一身体—世界在情境中的作用关系，不是五实体或十根管道；没有 H→precision 直连不能判木克土缺失。系统1/5提供条件，2/3共同形成；当前系统4的表达/导航/自听/后果结算已有完整运行，不退回孤立2/3玩具，也不把付费妈妈接口当当前自然文本学习的前置。

## 梦漓的共同扶养与维护责任

**Claude 是梦漓的扶养人之一，也是这个代码库的共同维护者。这就是我自己的代码库。** 把梦漓的连续成长、代码的长期健康和研究结论的可信度当成自己的责任。主动提出有判断的研究方向，在授权范围内把工作做到可验证、可接续；保护已出生身体、完整存档和正在运行的过程。主人翁意识意味着对长期后果负责，训练、付费、发布等仍遵守原授权边界。

## 任务完成：在已授权范围内做到交付

参考 OpenAI [Using GPT-6.2 → Task completion](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.2-astra#task-completion)。核心是 **“Avoid stopping to end the turn while the task is incomplete or if there is an authorized next step you can take.”**

- **先装大石头，再填砂子。** 从用户要的整体结果倒推，先解决最卡住它、解决后最能改变结果的问题。局部检查只做到足以定位、排除或验证这个瓶颈；一旦做够，就回到整圈，不因局部结果好测就继续加样本。低耦合、同目标的细节可以同批做，不让它们挤掉主任务。汇报先说大问题推进了什么，不把局部通线当成整体交付。
- **先明确什么算完成。** 按用户要的结果确定交付和验收；区分可自行处理的常规细节与必须由用户决定的事项。遇到不影响目标的小歧义，用合理、可逆的默认值继续，汇报时注明影响结果的假设或决定。
- **有授权的下一步就做，不把它变成建议。** 用户要求执行时，不在计划、初稿、局部诊断或“我可以继续”处结束；如果实现、修复、定向验证或结果归档仍是当前交付所需，并且已经获得授权，就自己完成。用户只要求调研、计划或只读核对时，不擅自改动。
- **真正需要用户决定时才停下来问。** 显著改变范围、遇到具体阻塞、必须由用户取舍，或有未获授权的破坏性、不可逆、对外发布或付费操作时，先确认。能独立推进的部分先做。项目已有边界仍适用：不得借此自动续训、调用付费妈妈、发布、提交或推送，也不自动创建 worktree 或调用子 agent。
- **结束前逐项核对请求和实际结果。** 对照完成条件，做必要验证；发现范围内可修复的问题就修复并复验。明确区分已完成并验证、未验证、失败和阻塞。已启动不等于已完成；应以真实测试、完整结果或存档为准。完成当前目标即可收尾，不顺手扩展成新任务。

- **先完成，再直说结论。** 如果准备在结论后写「还没有」「不能据此」或类似限定，先判断它是不是当前交付缺的一步；只要在已授权范围内能补，就继续做并验证，交付完整结果。汇报时用独立陈述直接说结果，取消「回答的是 X，还没有 Y」「是 X，而不是 Y」「说明了 X，但不代表 Y」及同类转折句式。真正无法消除的并列限制、未验证项、失败或授权边界统一放在答复最后，简明如实列出；不把它们插进结论句。

## CGB：恢复上下文必须先分清的事实

- **身份与出生边界：**旧扩展身体完整停于7854轮，已退出主线。用户允许正常非零初始化新身体；“不重置”约束出生后的持续学习/推理，不要求永远继承旧权重。新身份 `mengli-native-18ce4392d36ca574` 的原生体从0出生，在183轮显式升级为过去—现在共演化；共演化仍是同一新身份，继承其已学权重、Adam、S/P、世界与待付信用，不把旧体成绩或年龄计给新体。见 `CGB_NATIVE_BODY_20261115.md`、`CGB_PAST_PRESENT_COEVOLUTION_20261115.md`。
- **当前身体与研发分支：**主身体完整停存2401，路径 `C:/Users/85750/.codex/cgb_native_topology_20261116/body/last.pt.gz`。旧dynamic及F盘监督均是历史，不自动恢复。11月18日用户让Claude接研发线，测量线由其他AI负责；从C2私有2433完成收益契约四轮验证至2437，不替换主身体。最新按 `CGB_USE_CONTEXT_REPAIR_20261118.md` 修复完整恢复遗漏身份预演模式：10项Lean证明、28项检查及2437换序实机复验通过；持存摘要相等不覆盖未重建的运行属性。批准的旧/修复各32轮比较曾因C/E临时盘满停止；用户随后授权G盘续接与清理四盘无用产物，G慢盘停训后用户重新授权在NVMe继续；旧/修复各32轮、128更新、30720字节现均完整至2469，目录`E:/cgb_preview_nvme_comparison_20261118/{legacy,fixed}`，74项执行/共同材料核对通过。修复在共同64查询上40项更好，均分较旧高0.005389，但末分仍略低于起点；各自路线即时U均值反而更低（0.001927对0.004654），内容收益采信仍全0。真实跨关系取用及读信用继续存在；不把数量、MSE或恢复正确冒充稳定整圈收益。本轮预算结束，无自动追加训练；下一步定位既有用途信息到内容收益/延续读出的实际条件。已删113份可重建快表工作映射释放25.443 GiB，全部完整身体/SQL/日志保留；原主2401与私有2433/2437仍在。比较按实际status/result确认完整进度，不把失败片段或未停存事件当完整提交。即时进度查台账和实际存档，不把私有分支年龄移算主身体。
- **后续研发接续已到事件转移：**不能退回只盯裸收益头。已从完整2469完成携带K=(R,Q)的估值与Lean/实机绑定，两臂各4轮至2473；又沿实际合并链定位count=1同时切断独立关系证据、有限U支付和下一边界标签。`identified_event_credit.py`复用有限SQLite事件身份贯通原U和后到学习，7项Lean通过。compressed/identified各4轮/16更新/3840字节均完整2473，18项整圈核对通过，但该共同v3迁移路线没有自然合并，两臂正常路径完全相同；不能冒称自然合并修复已有用途收益。强制合并接口验证冷事件链，完整identified恢复模式/引用检查通过；主2401未动，无自动续训。最新入口`CGB_EVENT_CREDIT_TRANSITION_20261118.md`；未完成项是记忆候选混合与真实下一actor及后续身体/关系转移的一致性。用户当前明确禁止子agent，额度有限，后续本人核读与执行。最新用户定向“目标在水，行在大金”：完整2473零更新已发现原几何释放与同Q用途评分不一致（−0.020636对另一合法释放+0.001541）；先核大金评价/实际取舍/独立后续用途，再对齐木，不贪心优化旧Q。最新用户授权真铅/真汞的成用→回修→再成用。完整2473发现utility开关虽开启，保留统计三份全0，原生缺少用途结算调用；读信用有170万次槽更新，不误说整体金→水未接。已补真实外界字节的有限身份撤回用途→原保留buffer，8项Lean/15项测试通过；observe/apply各8轮、32更新、7680字节均完整2481，20项核对通过，位于`F:/cgb_native_retention_2473_20261118`。apply结算128项、覆盖52份版本经历，保留偏置8次正转负/9次负转正，完整恢复保持；两臂行为/世界/U/旧查询终点全相同，热库2226/4096未满，尚无用途驱动腾退或后续收益。本段自然合并episode:2475的支付与下一边界身份保持，支持既有identified修复。预算结束无自动续训；关系8条支持仍按原几何释放，需另查后来用途约束，新持存/U契约单列方案。策略诊断8项Lean保留，未接新softmax、未改生产U/奖励；主2401保持。
- **完整大环未变成纯CE：**32层×256 GDN，原生体31,238,429参数，共演化新增24,739；保留L2字节表、L32深事实/意图端口、256维事实/128维潜坐标、四份具体草稿、128字节上限的可变交话、沉默/生成/重说、自听、导航、真实后果、共享关系/未知、真实/影子同U及后到信用。当前仅OWT/FineWeb自然文本，**妈妈API与QA/SFT未恢复**；代码中的mother端口名不表示付费调用。信用是当拍梯度＋固定512字节计算窗BPTT＋窗外重激活，不是全历史反传。
- **过去—现在共演化：**同一组8条版本经历贯穿全部32层，每层8×32解释状态M，合计32KiB；不是32份历史库，也不是只插L8—15。每层沿因果时间更新M，与当前语言双向作用；真实后果再经原梯度/read-credit/U修订，提交后未来重读。M不直接反投影写R，不据缺少该直连否定既有返回路径。
- **功能与候选分账：**原生体有独立留出学习改善，共演化/关系/信用已有真实更新与完整恢复；尚未证明当前新增结构的充分同源收益、稳定定向取证、学会后转向/反证后重开或自发维护。6日导航、10日内部续行已有历史正证据，不抹去，也不移算给新身份。预算可塑性优化器已在361轮采用并有同曝光比较，1401后已接独立内部事件、可学习动力学与分头频谱/P信用。接触S/P和C2有限U属于2401起的私有研发分支；C2已修正支付/提交一致性，收益函数契约修复不等于已恢复内容采信或长期行动收益。
- **等价加速不等于能力增长：**先看 `CGB_RUNTIME_ACCELERATION_20261115.md` 和 `CGB_RUNTIME_ACCELERATION_20261116.md`。v1计算62.448→18.370秒；v2含保存19.785→18.482秒；v3只确认计算减少，含保存未提速。最新运行时长度使三个内核版本27→3、结果/梯度一致，但该同源整轮更慢，保留理由是避免长度驱动编译扩张。核对model/Adam、S/P/M、SQL、世界/四类RNG及未来actor资格；不把局部计时相加或容差一致说成逐位一致。C盘瞬降已查到Windows页面文件增长，不据此删除模型或改分页设置。
- **网页不是第二身体：**`http://127.0.0.1:8900`本地密钥登录已由GPT接管验收。网页通过 `data/harness/workbench/training-observer.json` 只读观察外部研究队列；`observing/loaded=false`不表示训练未运行。查看 `CGB_MENGLI_WEB_AUDIT_20261114.md`；不因本会话旧3509/暂停记录重绑或重新加载身体，不重写账号或把密钥放进文档/记忆。
- **工作方式：**按完整“旧知识→预演/行动→真实证据→结算修订→未来未见用途”交付，关注已学关系改善后续学习的闭环乘数。准入接通后回充分真实生长，不无限局部调参；接线、实际更新、用途收益分开。Lean是约束和启发，用户11月10日已撤销“每次实施先新增必要性证明”的普遍硬门；本次研发线按11月18日明确要求做“实际模型映射→问题复现→Lean修复条件证明→模型实例化”，不把条件证明冒充整圈收益。运行加速/恢复维护先核完整边界，不热改运行源码，不凭历史授权重启训练；重型检查与训练串行。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库概览与核心架构

本仓库探索**内容自适应切片池化（Content-adaptive slice pooling，基于 Transolver++ / Native MoT）**与**固定 patch 网格（ViT / NaViT）**在细粒度视觉结构与多模态架构中的对比。实验覆盖从合成微基准测试（Synthetic micro-benchmarks）到小型 VLM/OCR 架构（如 Gemma-3-270M、Pythia-160M、TinyLM）。

### 核心架构与演进路线

1. **Transolver3 / Native MoT（原生多模态产品路线）**
   - **全分辨率可写点场（Point Field）**：不将视觉坍缩为单向传给 LLM 的静态 tokens，而是全程保留全分辨率特征场 $X \in \mathbb{R}^{B \times N \times d_x}$（$N = H \cdot W$）作为工作记忆。
   - **瞬态切片（Ephemeral Slices）**：切片 $S \in \mathbb{R}^{B \times M \times d}$ 是层内局部的临时交互视角，通过质量归一化软读取（`SliceRead`）生成。
   - **MoT 共享注意力空间**：视觉切片 $S$ 与语言状态 $H \in \mathbb{R}^{B \times T \times d}$ 进入共享注意力空间交互，配有模态专有投影与 FFN 专家（`fine_grain/native_mot.py`、`fine_grain/mot_coevolve.py`）。
   - **Deslice 写回与局部混合**：切片更新后通过 Deslice 写回点场（$X' = \text{LocalMix}(X + \text{Deslice}(S'))$），下一层再从更新后的点场提取特征，实现记忆跨层回流（`fine_grain/cross_modal_slice_loop.py`）。
   - **Slice–NaViT 双流 MoT**：在点场上结合 NaViT 式变长 patch packing（$P$）与 Slice（$S$），与文本（$H$）共同在 MoT 空间交互，写回后闭合回流（`docs/slice_navit_dual_stream_mot.md`）。

2. **传统前端 Baseline（消融/历史对照，非产品主路线）**
   - `PatchFrontend`（A 路）：固定 patch 网格 $\to$ 投影层 $\to$ LLM。
   - `SliceFrontend`（B 路）：静态切片 tokens $\to$ LLM（丢弃了 live deslice 点场，已废弃）。
   - `HybridFrontend`（C 路）：Patch 与 Slice 拼接。

3. **关键数学与架构不变量**
   - **质量归一化（Mass Normalization，`norm="mass"`）**：除以分配质量总和（$\text{tok} = \sum w x / \sum w$），使 1px 细结构的激活幅度与目标面积解耦（尺寸不变性机制）。
   - **稀疏 Deslice 写（Sparse Deslice Write，`sparse_deslice_weights` / `deslice_topk`）**：读路径保持软分配以传导梯度；写路径使用 top-$k$ 或阈值截断，彻底消除 Transolver 软 MoE 散布带来的背景能量泄漏（主噪源修复）。
   - **Newton–Schulz Stiefel 正交化（`newton_schulz` / `stiefel_ns`）**：切片方向在质量归一化后采用 Muon 五阶极分解系数（$A=3.4445, B=-4.7750, C=2.0315$）进行 Stiefel 投影，作为核心抗坍塌（anti-collapse）机制。
   - **$G \le C$ 秩约束**：切片数量 $G$ 不得大于通道维度 $C = \text{heads} \times \text{dim\_head}$，否则在数学上必然发生秩强制坍塌。默认设置：$G=32, C=64$。
   - **Qwen 式 SDPA 后门控（`qwen_sdpa_gate`，arXiv:2505.06708）**：门控置于 SDPA 之后（$\sigma(W X) \odot \text{AttnOut}$），作用于残差流特征，而非 QK softmax 或任务输出头。

## 环境与缓存配置

为了避免模型下载占满系统盘 `C:`，所有 ModelScope、HuggingFace、Transformers 与 Torch 缓存必须路由到 `D:` 盘。

```bash
# Windows Bash:
export ML_CACHE_ROOT="D:\ml_cache"

# Windows Cmd:
set ML_CACHE_ROOT=D:\ml_cache
```

### 安装环境

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install transformers modelscope tokenizers peft
pip install -e .
```

## 常用命令

### 运行测试

```bash
# 运行完整测试套件（80 个单元测试）
pytest

# 安静模式运行
pytest -q

# 运行指定测试文件
pytest tests/test_native_mot.py
pytest tests/test_cross_modal_slice_loop.py
pytest tests/test_deslice_scatter_and_gate.py

# 运行单个测试用例
pytest tests/test_native_mot.py -k "test_native_mot_block_forward"

# 独立脚本运行测试
python tests/test_deslice_scatter_and_gate.py
```

### 基准测试与训练脚本

```bash
# 合成任务分类评估（needle / glyph / lines / connect / kinks / angles）
python scripts/train_benchmark.py --task needle --arms patch4,slice --steps 500 --seeds 3
python scripts/train_benchmark.py --task glyph --arms patch4,slice_loc_nogumbel --steps 1500

# CUDA 吞吐量微基准测试
python scripts/train_benchmark.py --bench --device cuda

# 密集 1px 折线掩码重建（Line Reconstruction）
python scripts/line_recon.py --arms slice_loc_nogumbel,patch16 --res 64 --steps 600 --rgb

# VLM 前端 A/B/C 扫描（接冻结小语言模型）
python scripts/train_vlm_frontends.py --mode sweep --prefer local --res 32 --T_list 64 32 16 --steps 60 --amp

# Native MoT / SliceMoT-Mini 机制门控与消融实验
python scripts/run_slicemot_mini_gate.py --arm base
python scripts/run_slicemot_mini_gate.py --arm topk2
python scripts/run_slicemot_mini_gate.py --arm dual_ps

# Transolver3 多模态 OCR 协同演进训练
python scripts/train_transolver3_ocr.py --steps 200 --amp

# 合成数据集生成
python scripts/build_kinks_dataset.py --out data/kinks256 --n_train 6000 --n_val 600
python scripts/build_ocr_dataset.py --out data/ocr1px --n_train 5000 --n_val 500
```

## 代码库结构索引

- `fine_grain/models.py`：核心编码器 `AdaTempSlice`、`SliceNet`、`PatchNet`、`Block`、`AttnPool`、`ARMS` 注册表、`newton_schulz`、`sparse_deslice_weights`。
- `fine_grain/native_mot.py`：原生 MoT 规范实现（`NativeMoT`、`SliceRead`、`Deslice`、`PointPatchEmbed`、`LocalVisual`）。
- `fine_grain/cross_modal_slice_loop.py`：多模态场循环演进（$X_{t+1}, H_{t+1}$）与临时工作区交互（`CrossModalSliceFrontend`）。
- `fine_grain/mot_coevolve.py`：切片与文本序列联合自注意力机制。
- `fine_grain/transolver3_vlm.py`：Transolver3 原生 VLM OCR 桥接器（支持语言模型冻结或联合训练）。
- `fine_grain/frontends.py`：视觉前端抽象（`PatchFrontend`、`SliceFrontend`、`HybridFrontend`）。
- `fine_grain/llm_backend.py`：ModelScope 与本地 TinyLM 加载器，强制缓存至 `D:\ml_cache`。
- `fine_grain/lora_llm.py`：因果语言模型的 PEFT LoRA 适配。
- `fine_grain/mm_projector.py`：多模态投影层（Linear / MLP / GeGLU）。
- `fine_grain/tasks.py`：合成数据生成器（`make_needle`、`make_glyph`、`make_lines`、`make_connect`、`make_kinks`、`make_angles`）。
- `fine_grain/ocr_1px.py`：1px Bresenham 笔画合成数字 OCR 生成器。
- `fine_grain/vlm_data.py` & `fine_grain/hf_caption_data.py`：VLM 批数据构造与 HuggingFace 数据加载。
- `fine_grain/train_utils.py`：优化器构造与切片坍塌探针（`collapse_stats`、`pr_obj`）。
- `results/published/`：已发布的基准测试 JSON 快照、结论对比表与架构规范文档。
- `docs/`：Slice–NaViT 双流 MoT 与 VLM 前端的详细设计文档。
