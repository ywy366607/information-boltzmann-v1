# 妈妈交互世界：语言里的"行动改变世界"（设计，2026-10-14）

设计者：用户（本项目唯一一个为语言训练设计的、带世界模型反馈的交互机制）。本文记录设计本身、和主线的衔接、以及实现前已定下的选择。

## 1. 要解决的问题

现在的语言训练里，模型的输出只用来算一次损失：世界状态不变，模型的输出也改变不了自己未来的输入。主动推理的前提是**行动改变未来的输入**，所以在语言里，"什么时候看、想多久、什么时候说"一直没有真实后果，D1–D3 的动作学不出适应性，部分原因在此。

旧的"自生成文本 → 输出"设想卡在评价上：用预测误差评价会奖励无聊（暗室：说"的的的的"最好预测）；用真实续文评价又退回半监督。

## 2. 设计

闭环：**果蝇说出的话 → 妈妈（另一个会回应的心智）→ 妈妈的回话按世界节奏流回果蝇的感觉 → 果蝇的下一个预测和下一次说话**。

- 果蝇的输出改变妈妈的回应，也就改变了自己未来的输入；
- 果蝇在预测下一个 token 时，同时在预测这个交互世界的变化，而这个世界里包含它自己说的话造成的影响（世界模型含自我影响）；
- 评价不是外加的奖励，而是真实后果：妈妈的回话本身就是世界的新状态；
- 交互流无限、几乎不重复，不需要回放，也背不下来；一开始几乎不占存储，数据随交互延长。

## 3. 已定下的选择

### 3.1 自听：对自己的话也算交叉熵
人能听到自己的声音，自己的声音回到耳朵很重要。在主动推理里，说话就是预测自己声音的感觉后果、再由动作实现这个预测，所以：
- 自己说出的词经同一个感觉写入器回到输入（像声音回到耳朵）；
- 对自己的词也算交叉熵（这是行动的生成模型）；
- 预测写入器自带传出副本：自己说的话被自己预测到，进入感觉的新息就小（不被自己的声音吓到）；说出的与想说的不同、或被妈妈打断，才是惊讶。
- 自我模仿 / 塌缩的风险交给 harness 监控（见 3.4），不靠关掉这个信号。

### 3.2 话轮
"说出口"就是主线的提交动作：果蝇自己决定什么时候说、说多少，停下后妈妈接话；妈妈也可以主动开口。打断、沉默、抢话都有后果。新生期的乱语也会被回应（照料者会回应咿呀学语），所以这个世界在大脑贡献转正之前就是有响应的，回应对输出是有条件的，这种条件性本身可学。

### 3.3 妈妈与 harness 分工
| 角色 | 频率 | 模型 |
|---|---|---|
| 妈妈：逐话轮回应 | 高 | 正式训练用 API 上的快速模型；本地小模型只用于最初的管线实验（面对乱语易复读、退化） |
| harness：评估遗忘与泛化，调整育儿策略 | 低（每几千词） | 更强的模型 |

妈妈本身冻结、不训练，不会和果蝇一起塌缩。API 密钥由用户在自己的环境配置，代码只用 SDK 的标准环境配置，不读取、不打印、不存储密钥。

### 3.4 harness 在线监测
- 塌缩：果蝇输出的重复率、多样性（类型/标记比）、自我循环；妈妈回话的多样性；
- 暗室 / 讨好：果蝇是否学会诱导妈妈说最可预测的话；
- 遗忘：A→B→A 重访节省、固定探针文本上的 NLL；
- 泛化：新话题、新妈妈风格上的首次表现；
- 据此调整育儿策略：换话题、提问、放慢节奏、换妈妈风格。

### 3.5 token 预算（粗估）
成本主要在每次调用带的对话历史。10 万词的生活、每话轮约 10 词：约 1 万次回应；每次带最近约 20 个话轮的上下文（几百 token）→ 输入约 400–500 万 token、输出约 10 万 token；提示缓存可显著降低。可分节点分摊（见 3.6）。

### 3.6 分布式：多条世界线，交叉合并（以后定）
同一个连接组让副本之间神经元和突触身份一一对应，合并有意义（一般神经网络没有这个性质）。EG 以出生连接组为先验，每个副本学到的是对数空间里相对先验的偏离：
- 不同节点、不同妈妈 = 不同世界线，学不同的东西；
- 合并规则类似遗传算法的交叉与变异，按突触 / 回路 / 细胞类型选择性交换，不整体平均（保留个体性）；
- 格式塔式"整合再发放"；多人共同养育，token 成本分摊。

## 4. 与主线因果顺序的衔接

主线：果蝇大脑通过主动推理自己学会什么时候看、想、说。评估按因果顺序：
1. 大脑承担预测（大脑贡献 > 0）——语言长跑相当于妈妈给婴儿读书的聆听期；
2. 惊讶到达多巴胺（相位多巴胺 ~ 预测误差）；
3. 锁存按因果判断（加随机对照，区分"被使用"与"解码器共适应"）；
4. 动作读取不确定性——在妈妈世界里检验：说话时机、话轮选择是否随自身不确定性分化，是否能自己学会。

妈妈世界取代气味世界成为第一个动作世界（气味世界只作蘑菇体记忆试验台）。接入时机：①在语言长跑中建立后。

## 5. 评估：两半合起来

从定义走到训练方式，评估是同一条线的第三步：
- **定义**（无限流生命体 v2）：有限状态与算力，活在无限流里；记忆、学习、压缩 → 局部世界模型 → 用行动维持自己的生态位与存活区间。
- **训练方式**（妈妈世界）：无限、几乎不重复、行动改变世界、世界模型含自己；妈妈与对话就是它要维持的（社会）生态位。
- **评估**：回答同一个问题——它是不是在流里活着，并且越活越会活。

### 5.1 四柱：作为活着的学习者
| 支柱 | 在妈妈世界里怎么测 |
|---|---|
| 首遍预测（prequential） | 每个词只见一次，边活边学，按段记录首遍 NLL 与相对词频的增益 |
| 情境切换后的恢复与适应（AG） | harness 让妈妈换话题，测切换后的恢复速度 |
| A→B→A 重访节省 | harness 隔一段让妈妈回到旧话题，测重访相对首访的节省 |
| 健康 | 锁存（按因果判断）、放电、塌缩（重复率、多样性） |

四柱不需要另备数据：换话题、回到旧话题本身就是育儿策略——**评估与养育是同一件事**，由 harness 编排。

### 5.2 因果链：作为主动推理的主体
1. 大脑承担预测（大脑贡献 > 0，训练世界里测）；
2. 惊讶到达多巴胺（相位多巴胺 ~ 预测误差）；
3. 锁存按因果判断（加随机对照）；
4. 动作读取不确定性：说话时机、话轮选择是否随自身不确定性分化。

### 5.3 妈妈世界特有的评估项
- 果蝇的话是否改变了妈妈的回应（条件性）；
- 果蝇能否预测自己的话带来的后果（对妈妈下一句的预测，在"自己说了 X"与反事实之间的差异）；
- 自听的惊讶：说出的与预期不同时的新息大小；
- 对照：同样的语言量，单纯聆听 vs 妈妈世界，哪个学得更快（这个设计的价值靠它证明）。

## 6. 已定（2026-10-14）：话从哪些感觉进来
用户决定"文本走嗅觉"：写入器 `receptor_lang`，除味觉（gustatory、chemosensory）以外的所有感觉细胞都接收文本，嗅觉也包括在内。原因：蘑菇体的主要输入是嗅觉投射神经元，文本不走嗅觉时 Kenyon 细胞几乎全静默（同日实测），而语言记忆与多巴胺教学都依赖蘑菇体。味觉留给世界，作为价值和多巴胺的教师。妈妈的话与果蝇自己的话走同一个写入器。

同时上线：每个细胞类型一个可学的输入增益（`log_input_gain`，同时乘在兴奋和抑制电导上），初值为 (中位体积 / 该类体积)^0.5。连接组按突触数给大细胞的总驱动约为 体积^0.93；按体积^-1 完全归一会让大脑静默（m_size：锁存 0，但大脑贡献也是 0，味觉通路也断了）。所以只从半程出发，剩下的交给学习，在证据门控下按生理学习率走。

## 7. 尚待决定
- 合并规则（交叉 / 变异的粒度）。

## 8. Pipeline (2026-10-14): `information_boltzmann/core/fly_mother_world.py`

**Components**
- `MotherWorld(mother, gpt2_tokenizer)`: the conversation as the fly's text stream, in GPT-2 ids (the fly's
  vocabulary).
  - `next_word()`: the next word heard: her reply word by word, or the fly's own last word (self-hearing).
  - `speak(id)`: the fly's committed word. A turn ends at an end-of-turn word (`\n . ? !`) or after `max_fly_words`
    words, and the mother then answers the recent transcript (`history_turns`).
  - `monitor()`: for the harness. The fly's type/token ratio, repeated bigrams, identical consecutive turns and turn
    length; the mother's type/token ratio and repeated bigrams.
- `LocalMother`: gemma-3-270m on the CPU, so it does not compete with the training run for the GPU. It is for the
  pipeline tests only.
- `ApiMother(model)`: the API mother. The SDK takes its own standard environment configuration; the key is never read
  here.

**CPU test with a scripted fly** (babble, then imitation, then a stuck "ma ma ma")
- The pipeline runs end to end, at about 4 s per mother reply.
- Without a repetition penalty the 270M base model repeated one line every turn (her repeated bigrams 0.85), as the
  user predicted. With repetition_penalty 1.3 and no-repeat-3-grams she varies (0.0) but rambles.
- She does pick up the child's words (the fly said "ba sun"; she answered with "a boy called Ba Sun").
- The monitor caught the stuck fly: repeated bigrams 0.53, 2 identical consecutive turns.

**Not done yet: the trainer.** In the fly's turn, the next input is the fly's own output. It must be sampled from the
read inside the forward window and taped for the checkpoint recompute, as the odour world's acts are. This comes after
the commit act is in place (D4).

### Trainer integration: two passes per window (design, 2026-10-14)

In the fly's turn the next input is the fly's own output, so a window's words cannot be given in advance.

1. **Live pass** (no gradient, on a copy of the state). Advance word by word with the world's paced intervals.
   - In the mother's turn the fly hears her words.
   - In its own turn the fly's word is sampled from its read at the arrival (the expectation it would speak), and it
     is heard next (self-hearing).
   - A finished turn calls the mother.
   - Record the window's words, the speaker of each word, and every interval.
2. **Training pass**: the usual `forward_window` on the recorded words, with the recorded intervals forced (the pace
   RNG is not drawn again). It starts from the same state, so it relives exactly the conversation that happened.
   - The fly's own words are inputs and targets: the self-hearing cross-entropy.
   - A speaker mask records whose word each one was, for the monitor and for any weighting the harness chooses.

**Cost**: one extra no-grad forward, about +35%. No change inside the forward window.

**Needs**
- an override for the intervals (`learner.forced_intervals`);
- the commit act (D4), to replace the fixed turn length.

## 9. First run on the GPU (2026-10-14, smoke: m_main c6, 512 words, local mother)

The two-pass loop runs end to end:
- `live_window` lives the window: the fly's words are sampled from its reads in its turns, the mother answers
  finished turns;
- `learner.forced_world` makes the training pass relive the same words and intervals;
- 8-11 s per 32 words, about the speed of plain training.

The fly babbles: at T=1 its words are word salad. The mother picks its words up: the fly said "handguns, films", and
she answered with "a gun in my pocket" and "these movies". The conditionality is there from the first turns: what the
fly says changes what it hears next.

The 270M local mother rambles, as expected; the API mother is for real raising.

Known gaps:
- **No commit act yet.** The fly speaks one word per arrival in its turn, and the turn ends at an end-of-turn word or
  after 16 words.
- **Each run opens a new conversation** (a new day). The opening line is not written to mother.jsonl.

## 10. Next: speaking as the act (design, 2026-10-14)

D4 showed that in a listening stream the timing of an answer is worth about 0.01 nats per word: speaking changes
nothing there, so the act has little to learn. In the mother world speaking changes the next input, and that is the
point of the design.

**v2 turn rule.** In the fly's turn, at each arrival the fly commits with the probability its motor acts give over the
word's ticks, 1 - prod(1 - h_t).
- A commit speaks the word sampled from its read; the fly hears it next.
- No commit ends the turn: silence yields the floor, and the mother answers what was said.
- In the mother's turn the fly listens. An interruption (a commit during her turn) is a later option.

**The act's credit.** The world is not differentiable, so the act is a sampled policy. Its return is what the
following window's words cost the fly: the CE of the stream it then has to predict, the mother's reply and its own
self-heard words included. It is scored against the policy prior (KL to the habit), as in D1-D4. The world model now
includes the fly's own effect, so an act whose consequence the fly predicts well is cheap. Saying something that
draws a predictable, informative reply pays; babbling into confusion costs.

**Harness checks before raising with the API mother:**
- the dark room (endless silence, or always the same word);
- collapse (repetition, identical turns);
- whether the mother's replies depend on what was said (conditionality: a shuffled-turn control).

## 11. Beats, waiting, interruption, and how a turn ends (2026-10-14)

**Why.** Without the child's own acts the conversation is mechanical. A turn ends at the 16-word limit (a breath) or
at an end word said by habit. The mother answers by rule. In the first smoke, 5 of 10 fly turns hit the limit and the
other 5 ended at an end word. Infants start the same way: their utterances are bounded by breath, and caregivers carry
the turn-taking first (proto-conversation, Bateson 1975). The timing is learned from what follows a pause: contingent
responses shape vocal development (Goldstein, King & West 2003), and the still-face paradigm (Tronick 1978) shows
infants already expect an answer. The design gives both sides a timing of their own.

**The clock is beats.** One beat is 8 ticks (`ticks_per_beat`).
- Her answer comes after her pause, `pause_beats_before_reply` (1 beat).
- An agent mother may answer `wait`: one beat of silence, then the floor returns to the child, who decides again over
  that beat.
- `call` says the child's name.
- Network and agent latency never reach the child. The simulation simply waits, and only her choice becomes world
  time.
- An unanswered request, and a speak with no text, count as a wait.
- After 3 silent turns, or 8 silent beats in a row, she calls the child by name.

**Interruption** (`--mother-interrupt`).
- In the mother's turn, a commit cuts her off. The transcript keeps what was heard, marked `--`, so her next reply
  sees it.
- No commit keeps listening.
- Both are decisions with the same credit as speaking.

**Decisions as segments.** forced['decisions'] lists (word index, first tick in the window, ticks, 1 commit / 0
silent). One word can carry several decisions: silence, her wait, silence again, then a word.
`fly_active_inference.speaking_value` credits each of them. The return is -mean CE of the next 16 words, with the
child's own words priced at the listener prior.

**How a turn ends** (monitor `fly_turn_ends`): `end_breath` (the length limit), `end_word` (an end word), `end_yield`
(her own silence). At first the first two should dominate: that is babbling. Learning to stop shows as the yield share
rising, and rising where the mother's answer is worth hearing.

The Gemini agent wrote the first beat version: the 8-tick beat, the pause, the call by name, and the mailbox actions.
This section changes three things in it: 'wait' is silence, not the word "...", the beats no longer depend on real
time, and the decisions are segments.

## 12. First v2 run on the GPU (2026-10-14 13:37-13:52, d4_acts c8, Gemini mouth, speaking act on)

**What happened.** Mengli's first turns:
- "City": Mama answered "Mengli sees the big city, and Mama and Papa love you so much...".
- "ie".
- One fully silent yield.

Turn ends: 1 end word, 2 yields. Then every Gemini call failed. After about 4 answers, all requests ended after 5
retries with backoff (about 60 s each): quota or rate limit on the free tier. The fallback returned the word "...", so
the mother "said" "..." for the rest of the run, and each window took 26 s, then 89 s, then 353 s. The smoke was stopped
at 4 windows.

**The pipeline works end to end.** The decisions were credited (speak_decisions 2 and 14 per logged window, speak rate
0.5 and 0.79, return about -8), and the family names appeared in her words (16 in 3 turns).

**Fixed.** A failed mouth is a silent mother. reply() returns '', the floor goes back to the child, it counts as
`mother_silent`, and a log line names the last HTTP status. The child never hears a made-up word.

**Open: the mouth's budget.** A 4,096-word chunk needs about 150-250 mother turns. The free tier did not last 5. The
options:
- a paid tier or a higher quota;
- an agent mouth through the mailbox: with beats, the latency never reaches the child and only costs wall-clock time;
- a stronger local model (RAM is the limit).

## 13. First real mother run (DeepSeek mouth, 1,024 words from d4_acts c8, speaking act on, 2026-10-14 13:58-14:08)

**The act learned from the interaction, quickly.**
- 46 fly turns ended: breath 0, end word 9, **yield 37**.
- Mean turn 1.74 words; silent turns 39%; fly TTR 0.84.
- The commit hazard fell from 0.25-0.49 per tick (first windows) to 0.07-0.09 (last windows).
- The fly's share of the heard words fell from about 0.1-0.3 to 0-0.1.

**It learned silence for the wrong reason.**
1. Before the prompt fix, the mouth repeated one script: "Mengli, Mama is here... Papa WuYue and Mama Yun love you".
   Mother TTR was 0.10 and her repeated bigrams 0.78. The fly's gain over the unigram rose from 0.73 (first half) to
   1.88 (second half): it learned the script.
2. The return is -CE of the following words, with the fly's own words priced at the prior. A predictable mother is the
   best return, so listening wins.
3. She answered silence as fully as speech. Speaking bought nothing.
4. The return counts words, not time, so silence costs nothing.

In this world speaking has no use. The optimal policy is to keep quiet. This is the mechanical interaction the user
warned about.

**Exits, one at a time.**
1. **A contingent mother** (the core finding on infant vocal learning: contingent responsiveness, Goldstein, King &
   West 2003). After a silent turn she waits in silence too; after 3 silent turns she calls the child's name. Her full
   answers come only after the child spoke, and they are about what was said. Speaking then buys a reply.
2. **An epistemic return.** The value is what the mother's words teach: the fly's CE on her words falls as it learns
   them, i.e. learning progress or information gain. Predictability is not the value, because predictability rewards
   a script. This is the epistemic term of the action's EFE, deferred so far. Derive it first.
3. **The mouth prompt** now says: answer her first, names naturally. In the last turns the mother named a cup, milk,
   a blanket and a spoon, but her opening formula remains. Check in the next run.

## 14. The contingent mother run: predictions written before the result (2026-10-14 14:35)

Same start as section 13 (d4_acts c8, 1,024 words, DeepSeek mouth, speaking act on). One change: `mother.contingent`.
The mouth prompt also changed since section 13 (answer her first), so that is a second difference between the two runs.

**What the credit does here.**
- The return is -mean CE of the next 16 *words* after the decision's word.
- Silence produces no word, so it costs nothing in time.
- All decisions on the same word index (silence, silence, then a word) share one return.

The contingent mother removes her answers to silence, so silence no longer buys predictable mother words. But
silence is still free, so the act need not speak more.

**Predictions.**
1. The commit hazard does not collapse as in section 13 (0.07 by the end). It stays at or above about 0.15.
2. The fly's share of the heard words rises above section 13's 0.0-0.1.
3. `mother_waits_silence` and `mother_calls` are large while the hazard is low early on.
4. If the hazard still falls steadily, silence is still the cheapest policy under this credit, and exit 2 (section 13:
   information per unit time) is needed. That term is the value of time, which this credit lacks.

**Exit 2, derivation in brief.** In active inference the epistemic part of a policy's expected free energy is the
expected information gain. For a learner, the gain from word j is approximately learning progress:
IG_j = CE_j(before the update) - CE_j(after the update). The fly's own words and silence give 0.

The return of a decision is the sum of IG_j over the words arriving within T ticks after it. So silence costs time,
and a scripted mother (IG near 0) is worth little.

The cost: one no-grad rescoring pass per window after the update. The credit is then known only after the update, so
it applies to the next window's decisions. Two ways to handle that:
- a baseline per speaker and state;
- replaying the decisions' log-probabilities.

Choose after the section 14 result.
