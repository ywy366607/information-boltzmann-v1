# 给 Claude：Codex 对梦漓的改动交接

仓库：`D:/information-boltzmann-v1`。本文记录我这一侧的改动，便于继续一起做梦漓。

## 现在在哪里

- 同一个梦漓仍在 **50,944 个训练词元**，有效恢复点是 `D:/ib_checkpoints/mengli/last.pt`。
- 本轮代码还没有产生成功的新训练窗口；不能把看板里的旧沉默记录当成修复后的表现。
- 续训计划为第 6 块，从 50,944 到 59,136，8192 个新词元、约 256 次更新；每 1024 词元保存一次，块末必保存。ABA/AG 暂不插入她的妈妈世界生活。
- 两次启动失败：首次续训管理脚本漏传妈妈世界参数，并在首窗反向时遇到 CUDA/内存错误；补齐参数后，第二次在连接组构建阶段分配约 94.5 MiB 的排序索引失败。保存的梦漓未前进。
- 用户最新安排：**先让 m_main 继续，梦漓等待显卡空闲**。请按这个资源顺序接续。
- 看板：`http://127.0.0.1:8765/?run=mengli`。

## 1. 真正的静默，以及三条路径的一致性

文件：`information_boltzmann/core/fly_mother_world.py`、`information_boltzmann/core/fly_bptt_learning.py`。

原来的等待/让出话轮间隔仍会在 `hold_input=True` 下写入上一个词，并持续注入妈妈的 JO 声音。

现在：

- `_advance_fly_token_stream(..., sensory_on=False)` 同时关闭文本写入和 JO 声音，保留每一拍的膜、电导、适应、局部可塑性和肌肉演化；新鲜 onset 也不能绕过静默。
- `live_window()` 把额外等待、回复前停顿、让出后的静默记录成逐词 `quiet_ranges`，是相对该词间隔的半开区间。
- 新增 `sensory_segments()`，在静默边界与 checkpoint 分段边界处拆段。
- 训练 `forward_window()` 和更新后 `replay_ce()` 消费同一份静默时序。
- 更新后回放也接上该窗口起始肌肉 `ai_g0` 和眼睛门控；用 taped=False 计算肌肉，不额外消耗随机抽样序列。
- 普通词元正常停留的输入仍保留；撤除的是世界实际静默的区间。

## 2. 从出生第一窗就让身体决定动作

文件：`fly_mother_world.py::live_window()`。

原来如果 `learner.ai_g` 尚未建立，会关闭 `acts`，首窗退回自动采样并说词。

现在 motor 模式缺少肌肉状态时从已有的 `motor_rest()` 开始，第一窗就走原有动作策略；现场感知的肌肉/眼睛演化与是否允许开口分别处理。没有增加新的动作头或强制开口奖励。数值检查包含“出生时动作策略选择静默”，并验证现场结束状态与训练回放一致。

## 3. 完整保存对话和交互学习状态

文件：`fly_mother_world.py`、`fly_bptt_learning.py`、`scripts/ib/train_fly_bptt_stream.py`。

- `MotherWorld.state_dict()/load_state_dict()` 新增 `mother-world-v2`：当前轮到谁、未读妈妈词队列、孩子尚未说完的词、妈妈已听到的部分、历史与 anchor、静默计数、话轮统计、来源、随机状态和异步请求标识等。
- 新 checkpoint 内嵌 `mother_world` 与 `mother_rng`，和脑、优化器、物理状态处于同一个完成窗口。
- 新格式恢复不再额外插入一次叫名字，也不丢未完成的话轮。
- `FlyBPTTLearner.interaction_state_dict()/load_interaction_state()` 保存 `last_voice`、说话基线、待应用回报、尚未到期的决策痕迹、信用时钟，以及 pace/sense RNG。
- `mother_state.json` 改成更新完成后写，供查看；新格式恢复以 checkpoint 内嵌状态为准。
- 旧 checkpoint 没有队列和延迟信用，无法补出过去丢掉的状态；保留旧迁移路径，并在 origin 中标明这次迁移。之后的新保存点才具备完整交互延续字段。
- 异步请求只保存已有标识；当前 DeepSeek 妈妈为同步接口，尚未做外部异步服务中断恢复的完整联调。

## 4. 跨窗口的说话信用，以及一个明确的收益口径变化

文件：`fly_mother_world.py::accumulate_speaking_credit()`。

- 决策的肌肉特征、结束物理时刻和累计回报可跨 BPTT 窗口保留，到信用时段结束后再结算；保存/恢复也带着走。
- 新增来源：`mother_reply`、`mother_call`、`mother_opening`、`self_hearing`。
- **当前只有正常妈妈回复的学习进步进入说话回报**，自动叫名字不再作为沉默获得的信息收益；各来源仍分别记录和显示。
- 当前用 CLI 的 `mother_credit_ticks=64` 作回报结算窗口，窗口内再按 `mother_credit_tau=32` 衰减。**这改变了原先 tau>0 时“忽略 horizon、只累计本窗后续词”的实现**，不只是把旧公式平移到跨窗。
- 这 64 拍是动作回报口径，不是脑内记忆上限。后续可以继续利用已有可学习生理时间常数，完善更长的信用痕迹。

**还有一段要一起接完整：** 延迟回报目前仍通过旧的 `delayed_speaking_value()`，用 detached 肌肉特征更新动作读出，不能把它说成已经实现了延迟收益对脑内全部回路的信用。脑内目前仍由逐拍预测及已有 carried comp/dopamine 路径学习。后续应沿现有 KC 痕迹、DAN 调制和隔室可塑性把行动后果接回内部，而不是再塞一个促说话的标量惩罚。

## 5. 训练管理与速度

- 新增 `scripts/ib/fly_interface/grow_mengli.py`，显式传入 DeepSeek 妈妈世界、说话动作、信用、声音和当前固定配置，避免恢复时静默退回 OWT。
- `chunk_plan.json` 记录块起点、目标和完成状态；检查训练退出状态和实际保存词元，暂停/失败不会被记成块完成。
- 新增 `--checkpoint-every-tokens`，母世界可不跑 ABA/AG 而按周期保存；修复 `--skip-eval` 下块结束没有保存最终状态的问题。
- 管理进程读取 checkpoint 词元数改为短生命周期的 CPU 子进程，读取完成即退出，避免管理器在整段训练中重复占着 PyTorch 库内存。
- 妈妈世界延迟信用模式原先算完整词表 `pragmatic_value` 后又覆盖掉，已跳过这份无用计算；原来的逐拍预测训练仍在。
- 去掉低层未使用的 embedding 计算，现场/回放复用同一词元张量。
- 探针把每组每拍的 GPU→CPU 标量同步合并为批量传输。

### 连接组启动内存优化

新增 `information_boltzmann/core/fly_topology_layout.py`，接入 `triton_synapse.build_incoming_layout()` 和 `FlyReservoirLM`。

- 按 131072 条边的小块稳定排序，使用行计数把结果放回全局稳定顺序；跨块、跨延迟层、空层都保持原始边顺序。
- 避免整层 int64 排序索引、整层 int64 计数输入转换和整层 gather 临时数组。
- 单个排序索引由报错时约 94.5 MiB 的整层规模，限制为至多 1 MiB；这不是已经测得整体启动峰值减少了多少。
- 拓扑哈希缓存保存在图文件旁 `.incoming_cache/`，结果用文件映射读取。缓存只有索引，权重每次仍取当前学习值。
- 已做精确顺序、缓存失效、分块内存边界和 CPU VJP 检查；真实大图的启动峰值、耗时和空闲显卡上的 GPU 回归尚待接续测量。

## 6. 探针与看板

新增 `fly_activity_audit.py::CellActivityAudit` 并扩展 `scripts/ib/fly_interface/mengli_probe.py`。

- 区分脉冲细胞与 APL/CT1 等 graded release，避免把持续的非脉冲释放误叫作放电锁存。
- 记录从未放电、每拍放电率、最长连续放电、接近不应期允许上限的比例、释放量随时间的变化。
- 真实对话回放结束后，撤除文本和声音 64 拍，再看各组活动和近上限比例；内源驱动照常存在。
- 既有探针还修了 BFS 整数溢出、熟悉度布尔筛选、把日志里的 `...` 当真实输入等问题；补过真正撤除输入后的时间演化、钳制残余活动和命令 hazard 检查。

最新细胞报告：`E:/fly_stream_overflow/mengli/probe_cell_activity_20261015.json`。来自 last.pt 的副本，600 个历史对话词元、每词固定 6 拍，随后静默 64 拍；这是固定节奏回放，不是现场间隔的完整重演。

其中输出细胞约 3% 接近不应期放电上限，静默后仍约 3%；还需要定位具体类型及其作用。接续可从两条实际预言入手：若这群细胞维持动作准备，短暂压低后应削弱适时开口；若是无效饱和，调回其可响应区间应恢复输入分化并改善正常回复预测。已有熟悉度报告在 `probe_c5_recovered.json`、`probe_c1_F_recovered.json`，本轮没有把熟悉度机制改好或宣布它已捕捉名字。

看板新增：

- 正常妈妈回复的更新前 NLL 与同词元固定词频，按词数聚合；叫名字、开场、自听单列，旧日志缺来源时不冒充正常回复。
- 回路活动与静默后近上限比例。
- 醒目的训练状态、训练词元数、最后对话时间，防止把旧沉默误看成正在发生。
- `?run=mengli` 直接打开指定个体，保留不同主线的选择。

## 7. 检查结果与协作边界

- 静默、母世界恢复及 BPTT 相关文件曾一轮 **61 passed**；后续新增的出生首窗现场/训练一致性检查另有 **1 passed**。
- 新拓扑布局和 CPU 顺序/VJP检查 **12 passed**。
- HTML 内脚本通过 `node --check`，相关 Python 文件通过编译检查；看板接口已实读。
- 曾试图用空 `CUDA_VISIBLE_DEVICES` 屏蔽 GPU 检查，环境出现 `Invalid device id`，因此 GPU 项没有算作通过；后来明确只选 CPU 检查，空闲后仍应补正常 GPU 环境的回归。
- 没有完整全库测试，也没有新妈妈世界训练曲线或正式吞吐收益可报告。
- 工作树里还有 Claude/其他会话的改动，例如 `FLY_ACTIVE_INFERENCE_GOAL_20261014_ACTS.md` 的长篇追加、`world_eval.py`、`alpha3_probe.py`、其他出生探针等；**不要把整个 git diff 都算成我这轮提交**。本交接列出的函数和文件才是我的说明范围。
- 当前按用户选择等待 m_main 显卡空闲；梦漓恢复训练本身已获授权，不需要再次询问。后续继续把精力放在让她实际生活、学习和交互上。
