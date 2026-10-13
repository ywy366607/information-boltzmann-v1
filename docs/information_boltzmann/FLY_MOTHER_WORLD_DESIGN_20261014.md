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
