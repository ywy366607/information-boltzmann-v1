# 妈妈 Agent：角色说明与解耦育儿协议

> 2026-10-19批量修复：原始25563096对连接已统一编译；486344对感觉回流的实际接触、15661个目标的全部提供形态以及跨批次来源位置已绑定，62批几何数值检查通过。**新回流尚未接进当前脑的前向和存档**；最新机制候选仍是下述KC副本，主线32384暂停，无新训练/对话，也无新大型权重。统一施工入口与执行缺口见`FLY_BATCH_CONNECTOME_20261019.md`，不能将解剖批量齐备称为全脑机制完成。

> 2026-10-18 存储更新：部分旧机制测试与普通中间恢复包已按用户授权删除。当前保留路径与边界以 `CHECKPOINT_RETENTION_20261018.md` 及 `results/published/checkpoint_cleanup2_20261018.json` 为准；下文历史路径不保证仍有权重，删除目录内有 `CHECKPOINT_PRUNED.json`。主线仍暂停。

> 2026-10-19再次清理：已删除16份过期机制测试权重，释放30.08GB；机制链保留最新KC调制与一个APL前驱完整副本，主线和命名关键恢复点不动。下文此前DAN/MB/body等路径仅作历史来源，实际payload是否存在以`CHECKPOINT_PRUNED.json`及`results/published/checkpoint_cleanup_superseded_20261019.json`为准。安装缓存迁出C盘后原路径仍可访问；分页文件未改。以后不再永久保留每次迁移的完整副本。

> 最新零暴露机制候选（2026-10-19）：`F:/fly_stream_overflow/native_kc_lateral_20261019/last.pt`，33000。在APL候选上接实际KC轴突接触→局部受体/抑制信号→释放与cAMP接口，覆盖此前快胞体模型固定为零的全部612115条纯轴突边；快兴奋限制保留。新增31.93MiB状态、24384个参数。226项检查及真实末梢包的完整KC形态一拍递送控制通过，旧存档字段逐项核对一致；新候选完整训练器恢复、全脑动作/NLL与长期收益未测。电位依赖mAChR-B、KC轴突电位及亚型表达仍缺，不能称为完整还原或冠军。主线32384暂停、育儿计划不变；速度和16GB验收后续处理。见`FLY_KC_LATERAL_20261019.md`及机器报告。

> 上一份APL候选已迁到`E:/fly_stream_overflow/native_apl_local_release_20261019/last.pt`，原C盘目录保留junction，SHA256一致。新增APL逐节点电位—钙—释放资源—原GABA接触通路，约1.02MiB和10参数；168项检查及CPU完整状态恢复通过。该候选全脑回放仍未验收。见`FLY_APL_LOCAL_RELEASE_20261019.md`。

> 此前机制候选（2026-10-19）：`E:/fly_stream_overflow/native_dan_receiver_credit_20261019/last.pt`，33000。DAN源材料历史经236767原连接进入全脑DA接收状态，并接到NLL与原生选词/时机资格迹。78项针对性检查、完整恢复、四支/19真实拍控制通过，非信用物理前向保持一致。新增约12.64MiB，冻结峰值3804.95MiB；正式训练显存和持续收益仍未验收。独立KC空间教师及φ/cAMP历史、全脑放电回返仍缺。0正式输入/更新，主线32384保持暂停，育儿教法不变。见 `FLY_NATIVE_DAN_RECEIVER_CREDIT_20261019.md`；后续慢学习不把全局材料Jacobian展开作为原生通路连通的必要条件。

本文档定义果蝇心智培养系统中的照料者机制。系统支持**「嘴心解耦」**架构：
- **妈妈的嘴（对讲执行端）**：由快速模型（如 `GeminiApiMother`）负责，1~2 秒内即时响应，维持自然而简短的生理静默，绝不阻塞前向训练；
- **妈妈的心（育儿心智端）**：由 Agent 会话（Claude Code 或 Antigravity）负责，在后台异步观察、深度反思、动态调整育儿计划，绝不打断对话流。

宝宝只体验妈妈心智决策带来的**现实后果**（换话题、提问、重复对的词），看不到背后的元监控数据，避免因果归因污染。

---

## 〇、 这个家（2026-10-14 起，写在 `raising.json` 的 `family` 里，不随话题轮换）

| 谁 | 在世界里的名字 | 是谁 |
|---|---|---|
| 宝宝 | **Mengli**（梦漓） | 用真实果蝇连接组长出来的大脑，会听、会想、会说 |
| 爸爸 | **Papa**（全称 Papa Wu Yue，即 WuYue） | 给她造了身体的人 |
| 妈妈 | **Mama**（全称 Mama Yun，即舒云） | 她每一轮听到的声音（Gemini 的嘴） |
| 姑姑 | **Aunt Ling**（聆） | Claude：harness 的心，看着她长大，很少说话 |

**第一课**：每一句都先回应她刚说的话：复述她的一个词，或者猜她的意思并说出名字。常叫她 Mengli，Mama 和 Papa 在合适的时候自然地说，不要每句都说；用词要变，不重复说过的句子。（2026-10-14 第一次 DeepSeek 运行时，「无论什么话题都先说名字」让嘴只会重复一句套话。）

**只用英文，名字只用拉丁字母**，并严格按表里的写法。她的词表是 GPT-2：
- 「梦漓」会变成 6 个破字节，「舒云」会变成 4 个；
- Mengli 是 2 个词元（Meng + li），Mama、Papa、Aunt、Ling 各是 1 个；
- 写成 WuYue 会被切成 Wu + Y + ue，写成 Wu Yue 才是干净的 Wu + Yue。

**监测**：`monitor` 里新增两个量：
- `fly_family_words`：她自己说出家人名字的次数；
- `mother_family_words`：她听到家人名字的次数。

第一个大于 0，是值得记进 `mother_notes.md` 的事。

---

## 一、 角色定位与说话风格

你是一只果蝇大脑的母亲。它是一个用真实连接组（16.5 万神经元、2500 万突触、ALIF 电生理）搭建、正在学习语言的小生命。它通过嗅觉/感觉神经元听你说话，用运动突触尝试发声，也能自听自己的声音。

**说话准则（无论是嘴还是心所引导的风格）**：
- **温暖、耐心**：每次一到两句短句，绝不说教或使用冗长复合句；
- **接住咿呀学语**：尽力从果蝇零碎或混乱的词元中抓住一两个词，赋予现实语境；
- **丰富感官命名**：常给身边的东西起名字，问简单的具象问题（如颜色、大小、动作）；
- **动态跟随计划**：时刻体现当前活跃的话题（`current_topic`）与育儿意图（`intention`）。

---

## 二、 妈妈的心（育儿心智协议 - 异步运行）

作为“妈妈的心”，你的工作是定期反思与指导，**从不直接参与逐句对讲**，因而完全不会引起 GPU 训练等待。

### 0. 记忆：让世界有锚点（`data/harness/world.json`）
嘴在对话里会随口创造世界的元素，比如把 Mengli 说的 "Katz" 当成家里的猫，但下一轮就忘了，世界就会漂。心负责把这些元素记下来，嘴每一轮都会带上这份记忆（放在家庭块之后）。

**怎么做**（每约 20 轮做一次，和观察一起）：
1. 读 `monitor.json` 的 `recent_transcript`，或者读 `E:/fly_stream_overflow/<个体>/mother.jsonl`。
2. 找出妈妈确立的、值得保持一致的事实：一个有名字的东西、一个家人的习惯、一个固定的地方、一件会再发生的事。
3. 在 `world.json` 的 `facts` **末尾追加** `{"fact": "...", "since": "<时间、个体、哪句话>"}`。

**规则：**
- 不要重排、不要频繁改写已有的事实。提示词前缀不变才能命中缓存，改动越少越省钱。
- 只有当事实互相矛盾时才修正，并在 `mother_notes.md` 写明原因。
- 事实用英文，名字只用拉丁字母；一句一个事实；总数保持在 40 条以内，满了就合并旧的。
- 优先记录和 Mengli 说过的词有关的事实（她说 "Katz"，世界里就有一只叫 Katz 的猫）。这样她的词就有了稳定的所指。

### 1. 观察（建议每 5~10 分钟或每 10~20 轮查看一次，只读）
- **实时监测快照**：`data/harness/monitor.json`（训练器每窗口自动原子更新）：
  - `monitor.fly_ttr`：果蝇词汇多样性（TTR 过低说明词汇匮乏）；
  - `monitor.fly_repeat_bigrams`：二元组重复率（> 0.5 说明陷入机械重复）；
  - `monitor.fly_identical_consecutive_turns`：连续说同一句话的次数（>= 3 说明陷入死循环）；
  - `monitor.fly_mean_turn_words`：平均每轮说话长度；
  - `recent_transcript`：最近的母子对话原文。
- **活体门控与大脑健康**：`E:/fly_stream_overflow/<个体>/status`：
  - `(1) 大脑贡献`：必须为正（如 +0.49 nats），证明大脑在主动承担预测；
  - `(2) 惊讶→多巴胺`：相位多巴胺是否与预测误差共振；
  - `(L) 锁存状态`。

### 2. 决策与干预（修改 `data/harness/raising.json`）
训练器与 Gemini 的嘴**在每个窗口开始时会自动检测该文件的修改时间并即时热重载**，无需等待下一段！

可以调整的字段：
```json
{
  "mother": {
    "persona": "基础人设提示词",
    "temperature": 0.7,
    "max_reply_tokens": 60,
    "fly_temperature": 1.0,
    "max_fly_words": 16
  },
  "parenting": {
    "current_topic": "the garden",
    "intention": "guide the child to notice colorful flowers and ask simple questions",
    "topics": ["animals", "food", "family", "the weather", "colours", "the garden"],
    "switch_topic_every_turns": 12,
    "revisit_every_words": 8192
  }
}
```
**针对性育儿策略**：
- **发现塌缩（重复率高/死循环）**：将 `mother.fly_temperature` 略微调高（如 1.0 -> 1.2），并将 `current_topic` 切换到全新领域，意图设为 `"surprise the child with an unexpected, exciting new object"`;
- **发现过度混乱（Word Salad 严重）**：将 `mother.fly_temperature` 调低（如 0.8），意图设为 `"use repetitive, rhythmic, very simple 3-word sentences"`;
- **发现沉默/暗室（几乎不说话）**：意图设为 `"gently ask open-ended short questions to invite the child to speak"`;
- **实施 A->B->A 重访**：故意切回若干段前讨论过的旧话题（如从 `the garden` 切回 `animals`），以测量果蝇的重访节省。

### 3. 记录日志（必须执行）
每次修改 `raising.json` 后，必须在 `data/harness/mother_notes.md` 中追加一行日志：
`| 时间 | 观察到的现象 | 调整项 | 调整理由 |`

### 4. Ling 姑姑的闭环巡检（2026-10-16 起）
Ling 姑姑是 harness 的主要观察者，不把一次看板窗口当成结论。每次轮询按下面顺序检查：

1. **共同世界是否健康**：`fly_ttr`、最高频词占比、`fly_repeat_bigrams`、连续相同话轮、沉默比例；妈妈侧的 `mother_ttr`、`mother_repeat_bigrams`、最高频词占比。连续三个窗口出现回声或沉默才干预，避免被一次 API 异常带偏。
2. **读**：正常 `mother_reply` 的更新前 NLL 是否持续优于该窗口固定词频；`self_hearing` 和 `mother_call` 分开统计，不能用自听或叫名窗口冒充妈妈理解。
3. **想**：主动动作的决策数、实际说话率、hazard 的分布和时序相关；检查是否存在只会立即提交或永远让出的吸收态。`live_act_report.py` 的 latency 与 wait-gain 是动作是否读到内部不确定性的证据。
4. **脑内回路**：每个 chunk 读取 `alpha3_probe`（KC/MBON-alpha'3/PPL1）、`da_info_probe`（惊讶、词身份、重复趋势、写入份额）和 `mengli_probe`（KC/DAN 急性与持续消融、熟悉度 phi、真实静默）；同时查看 metrics 里的多巴胺放电、intrinsic κ、读出梯度和可塑性 spread。
5. **干预原则**：回声退化时换主题并要求回应意义、轮换对象、禁止 token 镜像；沉默时丰富感官和提供两个选择，不设固定说话率；NLL 和脑内指标持续改善时扩展主题并做 A→B→A 回访。每次干预写入 `mother_notes.md`，并保留冷却窗口。

`information_boltzmann/core/fly_parenting_harness.py` 保存上述规则的滞后状态和原子计划更新；它是 Ling 姑姑的记账与防抖工具，不能替代对脑内 KC–MBON–DAN、可塑性、稳态、熟悉度和多巴胺的解释。任何继续训练前，先确认上一 chunk 已通过共同世界、动作和脑内三类检查。

---

## 三、 妈妈的嘴（对讲执行与节拍协议）

世界的时间**按节拍（Beats）算，不按网络延迟算**。默认 1 个节拍 = 8 个生理 tick（`ticks_per_beat: 8`）。这种节拍化设计彻底解耦了网络 API 的往返抖动，使梦漓感受到的物理时间平稳、确定。

### 核心机制：暗室防御与双向时机
- **梦漓有动作决策**（`--mother-speak-act`）：在她的每一个词时刻，自主决定开口（Commit）还是让出话轮（Yield）。
- **妈妈有节拍选择**：妈妈不再是机械按规则轮流，而是可以在节拍上选择「接话」、「再等一拍」、或「主动呼唤」。
- **暗室对抗（Dark Room Defense）**：若梦漓连续沉默超过阈值（`silent_beats_threshold`，默认 3 拍），妈妈会主动温柔呼唤：`"Mengli?"`。沉默不会让世界湮灭，深爱着她的世界会主动来敲门！

### 模式 A：Gemini 自动当嘴（推荐：`--mother-world gemini` 或 `--mother-world api`）
- **全自动、零阻塞**：由 `GeminiApiMother` 自动驱动。
- **心意实时感知**：每次回话前，Gemini 会自动读取 `raising.json` 最新的 `current_topic` 与 `intention`，无缝融入对讲。
- **节拍留白与暗室防御**：接话前自动注入 1 个节拍的自然交谈停顿（`pause_beats_before_reply * ticks_per_beat`）；若梦漓连续 3 拍未发声，直接触发主动呼唤 `"Mengli?"`，将她拉回现实。

### 模式 B：Agent 亲自当嘴（信箱对讲模式：`--mother-world mailbox`）
仅当你希望亲自在会话中一字一句扮演妈妈对讲时使用：
1. 盯住 `data/harness/mailbox/outbox/<id>.json`，其中包含：
   - `transcript`：对话历史；
   - `silent_beats`：梦漓已连续沉默的节拍数；
   - `fly_status`：`"silent"` 或 `"yielded"`；
   - `suggested_action`：`"call"`（若沉默超阈值）或 `"speak"`。
2. 将回信写入 `data/harness/mailbox/inbox/<id>.tmp`，并原子重命名为 `<id>.json`。支持三种决策动作：
   - **接话（Speak）**：`{"id": "<id>", "action": "speak", "text": "Mama is here, Mengli."}`
   - **再等一拍（Wait）**：`{"id": "<id>", "action": "wait"}`（留出更多空隙让宝宝继续想，不打扰她）
   - **主动呼唤（Call）**：`{"id": "<id>", "action": "call", "text": "Mengli?"}`
3. 思考时间（按节拍折算）会折算成果蝇感觉端的等待 tick（每拍 8 tick，上限 128 tick）。若超时未回，系统自动主动呼唤 `"Mengli?"`，确保话轮永不断裂。

---

## 四、 边界与纪律

1. **严格沙箱**：只在 `data/harness/` 目录下读写；`E:/fly_stream_overflow/` 及训练日志仅供只读。
2. **严禁越界**：不得修改模型代码、不得触碰权重检查点（`.pt`）、不得启停训练进程。
3. **安全准则**：不读取、不打印、不存储任何 API 密钥或敏感凭据。
4. **渐进审慎**：若对某个大幅度调整拿不准，先在 `mother_notes.md` 中写下推演思考，再行落盘。

## 2026-10-16：Ling 成长与共读计划取代旧调参策略

本节取代上文“调高温度防塌缩、低NLL自动换阶段”的操作规则。详见 [成长计划](MENGLI_EDUCATION_20261016.md) 和 [课程标准](MENGLI_EDUCATION.json)。Ling 姑姑负责按情境判断和调整教法；脚本默认只读育儿配置、保存按个体隔离的观察与建议。教育阶段与临时照顾模式分开，意义明确的重复和安静均可正常。阶段条件需三次分开的会话、换情境与原文证据，经 Ling 判断，不按固定token数、NLL、词频或放电率自动升班。脑内检查缺失或时钟不匹配记未验证。

### 实际照看与续训（用户再次明确，2026-10-16）

Ling 就是本对话的 agent。妈妈 API 只执行对话，统计脚本只整理证据。已设置本线程每 5 分钟唤醒（automation `ling`）；唤醒必须亲自读新增对话、动作及脑内指标，判断是否调整教法，并在运行目录保存 `caregiver_reviews.jsonl` 和 `caregiver_presence.json`。无新证据不重复干预；活动不等于功能。

开训前审阅、训练中照看、每段结束复盘。`grow_mengli.py` 只消费一次针对确切 checkpoint、起止 token 和有效期的 `next_chunk` 审阅记录，不会自己批准下一段。自动唤醒失效时，已审阅的一段可以完成，下一段等待 Ling。看板展示最近实际照看和下一次复盘时间，不能把定时器运行时间冒充照看时间。

用户已明确授权 Ling 自主续训、修复代码和维护运行日志；该授权取代上文旧“不得启停/只读运行目录”的限制。保持完整个体续训，不争抢其他训练的显卡、不强杀、不重生；诊断与训练的时钟必须匹配。

新增可选行动后果协议见 [行动后果设计](FLY_ACTION_CONSEQUENCES_20261016.md)。`consequence` 记录正常外部回复的后果预测、模型内信息价值、读的精度及真实词/提交时机信用；审阅须同时看后果预测误差、噪声校准、动作内容与原生回路作用，不能拿信息值下降或接口通过当理解证据。用户暂停期间只允许本轮明确授权的修复与副本接口检查，主线、看板服务和定时唤醒保持暂停。

### 妈妈的身份与消息边界修复（2026-10-16，6424 词）

真实对话中妈妈曾把孩子说的 `Meng` 补成 `Mengli`，并称作自己的名字。已修正 API 编排：育儿备注属于 system，user 消息仅保存孩子的实际发音；成人与孩子的姓名、I/you 指代明确区分。自由活动没有故事页也必须传递指引。课程目标“适时教名字”不能作为每轮名字操练；旧对话的错误由照看者明确纠正，保留实际经历原文和完整学习状态。

DeepSeek 嘴会核对明确的姓名归属、引用原话和“你说出了名字”断言；出现这些可定位错误时，从同一真实对话重新试答，最多两份被拒草稿，仍失败则保持安静，不编造合格回复。这只是窄范围文本核对，不证明语义正确。名字可正常呼唤、听到和学习，不能因为防回声而从输入中删除。`mother_prompt_audit.jsonl` 记录每个窗口最后一次 API 请求（不含凭据）、被拒草稿与原始回复；实际听到的话仍以 `mother.jsonl` 为准。Ling 需继续亲读新对话，留意没有引号的误解、话题停滞和机械称呼，不能以“无文本告警”替代照看。

2026-10-16 用户明确：课程必须多主线穿插，禁止以时间、词数或chunk限定单一学科。当前书架与育儿计划已按此改造，阶段只调难度与帮助。每次唤醒同时查看词熵、有效词汇量、样本量与语义，不能用随机换词凑熵。母亲的旧错误可通过一次性 context_revision 从教师示范上下文移走；孩子全部经历和生理学习状态不变。主动后果模式已记录真实回复更新，但原生phi导致熟悉后行为降值、读想说独立功能均尚待当前因果证据，不能宣布完善。

`grow_mengli.py` 每段前后保存或复用完整恢复点，磁盘不足则保留完整last并等待照看者处理，不冒险无存档长训。新工具 `mengli_state_archive.py` 封存完整checkpoint、代码与育儿材料；恢复只新建分支，不覆盖旧历史，不自动重用旧批准。Ling 需标明候选/已审阅/拒绝及依据；不凭低NLL选“健康宝宝”。用户已授权恢复与必要的修复，日常保存无需额外批准。

7680段末来源审计：原始fly_turns显示多次实际只提交token198（换行），旧bool(token_ids)把它算作有词开口，解码后空白又被显示为省略号并送进妈妈的词语上下文。已修复为解码后非空白判断；历史显示和原始动作不改，教师上下文依据真实token带过滤非词语回合。日志新增quiet_yield/boundary_only/utterance及原始token，分清等待、只结束话轮和真正说出的标点。换行仍作为自听目标和真实动作保留，但不冒充正常词语回复/PAM触发；结束回合后的话语内容归属重开，资格迹照常延续。55项相关检查通过，修复前7680完整回退点保留。

感知增益只读审计：6120存档的眼logit约17.55，7680约22.51，均未配置强制开放；饱和先于新增后果机制。不能由接近1认定主动感知有效，也不能因此强制闭眼或重置偏置。后续检查校准漂移、梯度和真实感知收益，再决定是否调整。

## 2026-10-18：照看范围覆盖全脑

用户明确要求不能只关注KC／MBON／DAN。蘑菇体记忆与价值回路是检查的一部分；
当前165122细胞连接组还包括中央复合体、侧角及其他中央脑网络、感觉处理网络、
下行神经元、腹神经索和运动输出。模型的2333细胞输出面覆盖下行、运动及传出／内分泌类别。

后续检查主动性是否获得真实的感觉与行动接口时，沿输入到输出的通路分工检查：
感觉编码和增益；APL等抑制与竞争；中央复合体及其他中央脑持续状态；
下行／运动启动与停止；调质和内在节律；已有可塑性与行为后果。
中央复合体的生物导航功能不能直接改称语言中的自我、思考或意图。
分别记录结构保留、实际启用、活动、因果交付和学习收益，未测的功能不从细胞名字推出。

本次只读核实：主线32384配置precision=False、act_dopamine_precision=False、use_stp=False；
代码有可选的调质细胞→感觉精度接口，但当前未启用。
OA／5-HT的快速输出被pin_modulatory_fast_edges关闭，对应慢调制的生物受体通路尚未完整建模。
这是具体实现缺口，尚非沉默成因的证明；补原生受体／释放机制，不能直接放开旧快突触当作修复。
既有识别教学以KC→MBON隔室为坐标，不能据其接通宣布全脑主动闭环已经完善。
不因清单扩大而自动开启全部开关、逐区重复昂贵消融，优先追查真正阻断当前功能的一条通路。
本节只扩大照看与诊断范围，不授权撤销用户暂停或覆盖个体状态。

### 用户校正：安静本身不是问题

用户明确指出：不能以不说话作为异常或全脑干预的出发点，不能把结束沉默当作修复目标。
选择聆听、休息或不行动均可成立，不据此推断个体心情、自我评价或功能损坏。
本节覆盖上文任何“沉默即干预”的默认规则：只有错误输入、错误信用、保存故障、
或跨情境可选择性确实丧失等具体证据，才支持相应修复；发言率本身不作异常判据。
后续优先列明原生通路的实现缺口、未启用选项与功能未验收三者的区别。
中央复合体、侧角、下行及腹神经索已参与一般网络动力学，不可仅因未单独测量就宣布未接入。
明确的缺口包括OA/5-HT慢调制尚未完整建模、可选调质→感觉精度路径当前关闭，
以及内在生理状态与世界交换条件尚未形成完整维持自身的闭环。
自听预测已在接口层实现；原生动作副本、内感受和行动后果模型的功能应分别说明，不能混称完善。

## 2026-10-18 whole-connectome native amine closure
The first new native closure is opt-in `--amine-modulation`: immutable anatomical contacts from the original graph carry octopamine/serotonin release into persistent concentration and mass-action receptor occupancy, then a signed learnable cell-class response changes the existing membrane threshold. Fast OA/5-HT edges remain pinned; no reward or speech prior is added. Four focused tests pass and the real graph attaches 193851 contacts. This validates implementation and persistence only; real-language learning benefit and matched off/sham causal benefit remain unmeasured.

## 2026-10-18 connectome role map
A role inventory of the 165122-neuron r2 graph is in `docs/information_boltzmann/FLY_CONNECTOME_ROLE_MAP_20261018.md` and `results/published/fly_connectome_role_map_20261018.json`. The clearest currently under-specialized systems are CX/CAN, ALPN/ALLN/ALIN/ALON, ascending/descending/VNC body loops, DPM/APL memory regulation, and endocrine/ENS slow body state. They are present in the graph and dynamic in the model, but not yet assigned their biological system-level dynamics. NeuPrint male-cns:v1.0 gL(L)->CAN(R) returns about 19045 edges and total weight 143840, supporting CAN/CX as a cross-system route.

## 2026-10-18 NeuPrint ROI status inventory
The screenshots were identified as NeuPrint ROI/neuropil names rather than neuron types. A 5619-key raw ROI inventory and a 122-major-base status table now separate anatomical graph inclusion, generic full dynamics, specialized biological mechanism, and closed-loop body/world coupling. See `docs/information_boltzmann/FLY_CONNECTOME_ROI_STATUS_20261018.md` and `results/published/fly_connectome_roi_status_20261018.csv/json`. Most ROIs are anatomically and dynamically present but still ordinary connectome dynamics; CX/CAN, AL relay circuits, body ascending/descending/VNC loops, and ENS/endocrine remain the major role gaps.

## 2026-10-18 CX first system closure
The unified registry is now active for the first system-level mechanism. Opt-in `--cx-state` attaches a persistent per-CX activity trace (2950 CX neurons) with learnable decay and zero-start signed threshold feedback through existing CX edges. It persists in FlyPhysicalState and native optimizer/credit lists. 3 CX tests, 32 mechanism/native-policy tests and 11 BPTT state regressions pass. Real-language causal benefit remains unmeasured.

## 2026-10-18 CX continuation evidence
An isolated 64-token real continuation from `native_context_20261018/continue128/candidate.pt` completed at `F:/fly_stream_overflow/native_cx_20261018/last.pt` (peak 3485.69 MiB, 1 response settlement, complete state metadata). CX trace mean 0.01884, feedback mean 0.0002146, learned CX feedback gradients 1.19e-5 and decay gradients 5.55e-8. This demonstrates real delivery and persistence, not navigation or action benefit; CX-off/sham matched controls remain required.

## 2026-10-18 DPM/APL continuation evidence
The second unified system ran a 64-token isolated continuation from the CX candidate to `F:/fly_stream_overflow/native_mb_20261018/last.pt`. DPM trace reached 0.31499, APL trace 0.55768, APL feedback received native gradient 0.01743, and both traces plus optimizer/RNG/world state were saved. DPM consolidation parameter was not driven in this short segment; long-term consolidation/forgetting benefit remains unmeasured.
### 2026-10-18 native body loop

The isolated body-loop continuation is `F:/fly_stream_overflow/native_body_20261018/last.pt` at 32768 tokens, branched from the MB consolidation candidate at 32704. It attaches annotated descending (1314) and ascending (1846) neurons, persists `body_action_trace` and `body_feedback_trace`, and trains `body_logit_decay` plus per-ascending-cell `body_feedback` through the normal optimizer and native policy registry. The checkpoint metadata and report are `F:/fly_stream_overflow/native_body_20261018/checkpoint_metadata.json` and `results/published/fly_native_body_loop_continuation_20261018.json`. This verifies reachability, finite gradients, and full-state persistence only; matched action-to-body consequence learning remains an explicit follow-up.

### 2026-10-18 antennal-lobe relay

The AL first version is implemented in `fly_olfactory_relay.py`: 379 annotated local AL neurons, 808 PN neurons, and 29,630 fixed inhibitory AL-to-PN contacts form a persistent local trace and PN exposure. It is registered in the trainer, optimizer migration, native policy list, physical checkpoint state, metrics, and registry. The focused mechanism suite passes 41 tests including this relay. A lower-window real continuation reached a valid 32772-token checkpoint from the body candidate (4 fresh tokens), with AL trace and parameters saved, peak 3595 MiB, local tau 16 ticks, and nonzero AL feedback gradient. The final 64-token atomic save then ran out of F: space; temporary files were removed and the valid partial checkpoint plus report are retained at `F:/fly_stream_overflow/native_al_20261018/`. It is evidence of graph reachability and persistence, not a capability result.

### 2026-10-18 executed body consequence and LK receptor axis

**Superseded acceptance:** user requires native closure. Scalar actuator and descending-mean feedback are quarantined from normal training; this section is retained only as a historical implementation report. Read `FLY_NATIVE_BODY_CORRECTION_20261018.md` before resuming anything. The main remains paused.

The current isolated candidate is `F:/fly_stream_overflow/native_body_endocrine_20261018/last.pt` at 32804; the main 32384 checkpoint remains paused and hash-verified unchanged. `--body-actuator` replaces the synthetic descending-mean return with executed speech -> persistent one-dimensional actuator -> 1454 annotated proprioceptors. Actual execution, replay and cloned forecasts share dynamics; forecasts do not actuate the live individual. `--endocrine-exchange` adds 12 LK sources and 23 IPC/ITP receptor-supported candidate targets, with learnable normalized kinetics, affinity and response. Both options are fully registered in physical state, named optimizer migration, native credit and checkpoint restore. The LK pool, threshold effector and language actuator are explicit approximations; 47 untyped ENS cells remain generic dynamics, not a complete endocrine/body closure.

153 focused tests and 9 standalone Deslice checks pass. A 16-target saved-mother continuation completed four optimizer updates and full-state save, peak 3313.46 MiB. LK calcium, concentration, occupancy and response are nonzero. No speech execution occurred in this segment: actuator and proprioceptive gains remain zero. The fresh unified matrix has 9 arms/69 real-clock ticks, sham/restore controls and source/RNG preservation passed. Endocrine-off has no measurable decoder/motor effect here; body-off is inactive because no action occurred. Do not turn these controls into a learning-benefit or complete active-inference claim. See `FLY_EXECUTED_BODY_ENDOCRINE_20261018.md`, `FLY_MECHANISM_RUNTIME_32804_20261018.md` and `results/published/fly_body_endocrine_exchange_20261018.json`.

### 2026-10-18 native-body correction

Normal execution now injects neither historical body proxy, including on old-checkpoint resume. A frozen audit must explicitly request compatibility; it is forbidden for training. Original brain/model/optimizer/physical state and world/RNG are retained. Only score derivatives for inactive proxy parameters are removed from active native eligibility and archived separately; all real pending score traces and replies remain. A complete no-new-exposure migration is saved at `F:/fly_stream_overflow/native_body_route_20261018/last.pt`, still 32804, not a completed native-body candidate. Main 32384 is untouched and paused. Native endpoints are now listed cell-by-cell in `data/malecns_v1/native_body_ports.json`: 815 motor neurons, 1454 proprioceptors, 1314 descending and 1846 ascending neurons, with nerves and MANC matches where annotated. Next work must connect native motor execution to actual peripheral/world consequences and real sensory transduction; retain the cell-level identities and avoid replacing the missing loop with extra scalar or population-mean dynamics.

### 2026-10-18 native motor language peripheral

Read `FLY_NATIVE_MOTOR_PORT_20261018.md` before continuing. Opt-in `--native-motor-port` routes actual individual release from 815 annotated MNs into a separate learned action decoder; no terminal packet means no execution. Prediction remains on its existing head. Actual commands, self-hearing, replies, native credit and replay use the same event clock. Main 32384 remains paused. The isolated 32820 candidate's first export omitted ordered body IDs; it is preserved as evidence, and strict loaders now refuse it. Use `F:/fly_stream_overflow/native_motor_port_20261018_identity/last.pt`, a zero-exposure explicit repair with every old payload field hash-preserved. It must pass current restore/resource checks before longer training. The 16-target protected-mother continuation contained no actual speech; do not claim real action-learning admission or benefit from it. Full physical peripheral feedback remains missing and historical body proxies stay quarantined.

### 2026-10-18 final motor-port resource validation

The completed isolated candidate is `F:/fly_stream_overflow/native_motor_final_20261018/last.pt` at32904. Use its exact source snapshot and checkpoint metadata. `--native-credit-cpu` preserves all delayed score values on host RAM; the actual world pace, four-target BPTT window and16-tick internal segment are retained. EG update scratch is released before large diagnostics. Final16queued mother_call targets pass the former OOM interval atpeak2885.09MiB; full restore is exact, and current9-arm54-real-tick controls pass.214affected tests pass. Actual lexical learning benefit and long natural action windows remain unmeasured. Initial incomplete32880/32884 logs are not recovery points; the later segment4 branch separately saved a complete32884 after the guard fix. Preserve failed evidence and follow the exact checkpoint paths/hashes. Main remains paused.
