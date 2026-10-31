# 实数 GDN：开放轨迹、预测更新与联合条件

## 1. 本轮选择

保留固定维度实数状态，不先量化，不枚举身体状态，不以查表代替矩阵运算。

`GenerationSelfProduction.lean` 的 renewable 确定性载体、开放世界及联合门已去除不参与证明的 `[Fintype State]`。字节字母表仍有限；容量基数、对身体状态求和及有限随机核的局部定理仍保留各自有限性前提。

新增模块：`lean/CGBAxioms/CGBAxioms/GenerationGDNLife.lean`。其 `State k d` 是自然参数、精度、速度、上一观测摘要、仿射 head/bias 的实数结构；`real_bias_injective` 将任意实数注入已有 head bias，明确它不是有限类型的改名。

这里不把无限精度实数当成免费硬件。有限精度、完整快照恢复、表示预算是下一层实现精化，不是泛化连续状态联合门的前置条件。

## 2. 唯一递推与实现对应

分析对象是 `GDNByteReadoutPort` 的**单观测、关闭运动更新、有 bias 的仿射 head**路径。运行时保留历史共享 encoder 的默认行为，并显式区分 value、address、likelihood 三个已有输入项；只有 value 子路径已按 Lean 接入在线更新。

| 组成 | 本模块定义 | 与实际接口的关系 |
|---|---|---|
| 内容与不确定性 | `State.natural/precision` | 分别保存 η、p，不以均值代替整个状态 |
| 编码/地址 | `Model.value/addressLogits/likelihoodPrecision` | 默认可共享同一表示；启用 value 学习时，运行时要求固定且独立的 address/likelihood 视图，避免一次 value 更新暗中改变另两条未准入路径 |
| 因果 query | `query` | 使用上一到达字节的地址；出生用 seed query。是一个明确候选，不是所有 Python callable 的自动精化 |
| 时间 prior | `prior` | p′=1/(1/max(p,ε)+Δt Q)，η′=p′·η/max(p,ε)；零时间严格不更新 |
| 准备预测 | `prepare` | prior 后读取特征，保存 feature；字节和来源不进入准备函数 |
| 同化 | `assimilate` | w=address×observationPrecision×likelihoodPrecision；η+=wv，p+=w |
| 局部 head 学习 | `finish/headUpdate` | 同化后仍使用预测时 feature；只更新 head/bias，不反传历史 |
| 独立世界 | `Process` | 外部世界有自己的状态、初态、边界响应、演化；策略是明确提供的参数，不声称已由 GDN 学出 |

`LastByteGDNReadoutPort` 是这份固定递推的实际入口：出生时用 `seedLogits`；其后用状态所保留的上一到达 byte 的地址查询同一 `natural/precision` 状态。它移除了原 `GDNByteReadoutPort` 中任意 Python query 回调的歧义。Python 现有 affine head 的 G14 更新，并新增仅针对上一 byte 的 value encoder 的一拍前向信用：到达写入时保留一个标量读出资格，下一字节到达后把它乘以准备预测的“观察到的 byte − 预测分布”语言误差，写回该 value 行。address、likelihood、动作和延迟外部结果仍没有在线规则。参数带有 `requires_grad` 也不构成其余生命学习机制，仍须完成外部结果闭环。

### 固定骨架与可学习部分

| 固定于架构/世界合同 | 在同一骨架内学习或演化 |
|---|---|
| 有限 `natural/precision` 槽、因果拍序、加性同化、softmax 归一化、softplus 正性、外部来源与预序评分边界 | value encoder、内容地址及其尺度、likelihood 的 raw logit、head/bias、动态 prior 的可训练系数、行动策略 |
| 这些约束保证状态可解释、写入不倒流、置信度非负；它们不替生命挑选记忆内容 | 参数从语言预测误差和新鲜外部结果中形成表示；状态则在每拍按递推真实改变 |

这对应生物中相对稳定的细胞/时序/化学可行域，与会被经验改写的突触、受体增益、联结和行为策略。这里的“固定”是发展骨架和安全边界，不是手写知识或手调学习方向。
### 编码器信用：已得到的精确部分与尚未越过的边界

`valuePerturb` 只在固定地址、固定似然下沿一个 byte-value 方向作有限差分，并不创建运行时数组、奖励账本或第二条记忆。它给出三条可直接核对的结果：

1. `prepare_valuePerturb`：当前 byte 到达前的预测完全不受这次 value 变化影响，因而没有把未来 byte 偷放进预测；
2. `incoming_valuePerturb`：同一到达的地址权重不因 value 方向改变；
3. `assimilate_valuePerturb_natural` 与 `assimilate_valuePerturb_difference`：到达后每个自然参数的精确有限差分为
   \[
   (\theta_1-\theta_2)\,[\mathrm{incoming}\times\mathrm{direction}],
   \]
   其中方括号内是该到达的局部资格种子；`assimilate_valuePerturb_precision` 还排除了把该效应误归因于精度变化。

这说明“局部痕迹”不是参照生物硬加的机制：在 GDN 的自然参数同化公式里它已经是编码方向造成的精确差分。`valuePerturb` 仍只是**一拍、固定地址的 value 子路径**。本轮补上的 `addressLogitPerturb_deriv` 则对同一写入式给出地址 logit 的精确导数：

\[
  \frac{\partial a_j}{\partial z_{j_0}}
  =a_j\,[\mathbf 1_{j=j_0}-a_{j_0}].
\]

于是 `assimilate_addressLogitPerturb_natural_deriv` 与 `...precision_deriv` 证明：地址侧会同时改变自然参数和精度，系数正是 observation precision × likelihood precision × 上式（自然参数再乘 value）。这不是运行时挑选地址方向；`j₀` 只是在证明中展开现有 softmax 的坐标基，真实 encoder 的普通局部 Jacobian 自行线性组合这些分量。

实际 `LastByteGDNReadoutPort` 的 byte encoder 同时影响 value、地址以及 likelihood。likelihood 这条路径现已按运行时的 `softplus(raw)` 参数化：`likelihoodSoftplus_deriv` 给出 raw logit 的精确斜率，两个 `assimilate_likelihoodLogitPerturb_*_deriv` 定理证明它同时改变自然参数和精度；`assimilate_likelihoodLogitPerturb_zero` 则证明 raw logit 表示当前置信度时，这确实是既有 GDN 同化的切线，而非新造规则。

跨拍部分先得到一个最小而精确的结果：`value_sensitivity_two_arrivals` 表明两次到达后的同一自然状态恰好携带两次局部 value 敏感度之和。`prior_natural_difference` 与 `value_sensitivity_through_prior` 再证明，时间 prior 只是把这一已有差分按其由精度和时间决定的系数输运；它不需要重读历史或另存资格迹。若 precision 与因果 query 未变，`feature_difference_of_shared_query` 证明同一差分会线性进入下一次读出。没有回看第一拍输入，也没有额外历史缓冲区；这是前向因果留痕的实际递推起点。

完整 Jacobian 只保留为小规模、短视界的**精确审计上界**：它列出某个候选留痕不能静默漏掉的 value、地址、likelihood、prior 与读出路径。实际在线个体需要的是由当前有限状态与当前事件递推的、压缩的因果载体；它不被预设为梯度，不保存 BPTT 图，也不要求逐参数复刻 RTRL。`delayed_credit_requires_history_carrier` 仅证明延迟结果需要某个足够统计量，并明确不选择任何特定资格迹形式；`causal_carrier_contract_iff` 给出其合格条件：它必须能读出后续所需区别，并决定每种未来输入下的后继。候选载体是否足以在第三因子到来时归因，应与精确短视界和路径消融对照，再以新鲜外部语言结果验收。当前 Python 已实现其中唯一完整闭合的 value→下一字节评分子路径，并以自动微分有限审计逐分量核对；它没有把局部子定理扩张为 address、likelihood 或外部结果的可训练声明。

### 长程信用的准入：递推因果载体，而非 Jacobian 替身

这里的时间长度不是存一段越来越长的过去，而是同一身体连续经历任意多次事件。`carrier_fold` 已证明任意有限 `List Arrival` 都经由同一个 `step` 递推而不重置；因此，**递推本身**没有 BPTT 的窗口截断。它并没有推出“有限身体能精确区分无限多条独立旧原因”：`collision_of_excess_keys` 与 `superposed_transfer_readout_is_net` 正好给出这一容量边界。

因此候选资格迹必须先满足以下准入合同，而不是先被规定为 \(\partial s/\partial\theta\)：

1. **同体递推**：\(e_{t+1}=T(e_t,s_t,u_t)\)，其中 \(e_t\) 是现有状态的可读部分或其受限投影；运行时不得保存事件历史或反向图。
2. **延迟可区分**：如果两个过去在未来外部结果到达时要求不同信用，则它们的 \(e_t\) 不能已经合并。这是 `delayed_credit_requires_history_carrier` 的直接准入条件。
3. **结果调制**：写入只由新的外部预测误差或行动后果调制；复制自身字节不能伪造第三因子。
4. **容量诚实**：若真实语言续接仍可区分的历史已在 \(e_t\) 中碰撞，拒绝该投影或扩展可路由状态；不能以“守恒”宣称来源身份仍可恢复。

当前 GDN 的分工由此变得明确。`natural/precision` 已是 value 写入差分的载体；head/bias 使用同拍预序误差，根本没有延迟信用问题。真正尚未通过准入的是**动作的延迟世界后果**与 encoder 的多路径可塑性：前者必须让动作在结果到达前留下可读的身体—世界因果状态，后者必须从现有局部写入式中导出一个可递推投影。两者都不能靠附加“每参数 Jacobian”解决。

短视界精确 Jacobian 只做两项审计：列出候选投影漏掉的路径；在小规模数值检查中比较其对干预删除/替换的因果排序。长程有效性由递推合同、容量/碰撞检验和连续真实语言流上的外部结果验证，而不是由把短 Jacobian 拉长来宣称。

### 行动—世界回路的当前准入结论

新增 `delayed_action_credit_requires_state_separation` 将上述必要性接到动作：若同一身体经两个动作后进入相同状态、而两个动作应得到不同的延迟信用，则任何只读该状态的规则必然失败。它不要求动作日志；可读的区分可由行动造成的身体变化、写入同一状态的自发语言，或世界返回并进入身体的后果承担。

形式 GDN 已给出字节级的正面前提：输入精度为正的到达必把该字节写入既有 `lastByte` 因果 query（`step_positive_observation_sets_last_byte`）；不同的有效字节故必进入不同的 post-step 身体（`distinct_live_bytes_separate_steps`）。更具体地，`self_speech_action_contrast_writes_body` 把同一时刻的两个候选动作接到 `Process.boundary`：若它们产生不同且有效的 `Source.self` 字节，普通 `step` 必给出不同的后继身体。`step_source_irrelevant` 同时保证 `Source.self` 不另开一套学习或储存算法：它与外部字节经同一身体转移，来源只在边界判定“是否为新证据”时有意义。

这条身体差异本身只是一拍结论。若注册的外部语言续接确实能把两条自发语言后的身体区分开，并且中间递推满足 `futureReservePreserved`，新定理 `future_reserve_preserved_keeps_signature_distinction` 证明这种可用的未来语言差异不会被静默合并。再由 `future_reserve_recoverable_iff`，结果到来时可从后状态的未来签名恢复该先前区别。它正是长程行动信用所需的条件，既不使用动作日志，也不要求全 Jacobian。

当前 Python 端仍没有把“模型选出的自发字节”送为 `Source.self` 的 `Arrival`：`ActiveInferenceGDN2.query/dynamic_prior` 能接收即时 action，但 `LastByteGDNReadoutPort` 的递推没有这一 action/emission 输入，`advance_time` 也没有把 action 传给 prior。因此运行时尚未具备 source—speech—response 接线，也还没有外部回应到第三因子/策略更新的规则；不能宣称行动学习已经实现。这是一个明确的接口桥，而非再造资格迹或另设动作账本。

现有端口已做一个受控的有限见证：分别写入自发字节 65/66，再写入同一个外部字节 67；两支的 `last_tokens` 已相同，但 `natural` 仍不同，且再回放同一外部字节 68 得到不同的 `external_future_signature`。这验证了 Lean 所说的同体写入在当前 GDN 算术中的一段实例；它不替代 `futureReservePreserved` 的任意延迟合同，也不是语言能力实验。

第三因子接入还须经过一个更严格的门：`objective_deriv_advantage` 表明完整的闭环导数同时含有“外部结果减基线 × 当时动作 score”与直接预测评分导数。`action_head_score_deriv` 已把前一项的局部 score 实例化到 GDN：对一个 head 坐标，它严格等于当前特征 ×（实际发出的字节指示 − 当前预测概率），来自 G14，而不是人为定义的资格迹。当前有限见证只显示动作差异留在身体里，尚未显示身体可在延迟后读出当时的这一个 score 方向；因此不能把外部回应乘任意旧状态就称为正确学习。下一步应把该 score 作为同一因果载体的**所需读出**，用 `causal_carrier_contract_iff` 检验现有 GDN 状态是否足够；若不够，先寻找可由自发语言写入自然状态的最小可读统计量，而不增加独立动作日志或全 Jacobian。

同一状态也不能被神化为“无代价地保存每一个过去原因”。`associationState_row` 说明自然矩阵存的是 value-weighted key 的线性叠加；`collision_of_excess_keys` 则证明独立 keys 超过键维度时，存在不同来源系数写成同一状态。`superposed_transfer_readout_is_net` 给出同一个事实的守恒形式：只读到净流，读不到每条来源流。

这不要求人从外部 future probe 手挑“该学的方向”。probe 只是在留出妈妈语言续接上验收：当前身体是否仍区分了对未来有影响的历史。训练中，有限的 `natural/precision/head` 与可训练 encoder/address 共同构成压缩瓶颈；预序语言损失和来自新鲜外部结果的第三因子才是让它自己形成可复用方向的选择压力，正如普通 LLM 由有限参数和 next-token loss 自行形成表示。`causal_carrier_contract_iff` 给出合格压缩的判据：同一压缩状态必须同时足以读出所需预测、并确定每个后续输入下的压缩转移。它不承诺任何优化过程必然找到最小压缩；这要由真实语言流训练和留出 probe 验收决定。容量定理在这里是拒绝条件：若留出世界仍需要比当前 state 能承载的独立区别更多的维度，才证明必须扩容，而不是事前替生命体造一份资格迹账本。

速度在该路径中保持不变。已有上一观测且输入精度为零时，保留旧的上一观测；没有缓存时首次输入仍建立缓存，与 Python 的 live mask 一致。固定 encoder 下，缓存字节足以重建对应 token 内容；这不允许任意带隐态的 encoder/query 偷藏额外历史。输入的 `elapsed/source/byte` 与预测字母 `Fin 256` 分开，信息基准仍为 log256。

`carrier_next/carrier_fold` 将显式 GDN 算术递推接入已有历史因子化载体，证明任意有限输入续接都使用同一个更新。证明侧历史摘要不作为第二套运行时记忆。

## 3. 已证链条

### A. 数值状态与开放轨迹

`initial_positive → prior_positive → assimilate_positive → step_positive`。

`exists_open_real_gdn_trajectory`、`Process.run_step/run_positive` 给出一条从同一初态出发、按实际边界输入与世界演化推进的无限实数轨迹。这里的 `Positive` **只表示槽精度严格为正**，不等同于膜完整、物质摄入或生物可维持性。

### B. 实际参数更新如何影响评分

`head_update_logits` 给出实际仿射参数步在保存特征上的有效 score 步长：

η_eff = η · (1 + Σᵢ fᵢ²)。

其中 1 来自 bias 更新。复用既有 `two_phase_finite_score_step_improves` 后，`finish_saved_nll_improves` 推出正步长降低刚观察字节在保存特征上的 NLL。它不宣称改变记忆/query 后仍然改善未来。

`step_ne_of_positive_head_step` 还证明实数 head 在每个到达字节上都会变化。**参数不停动不是有益学习**：不可预测噪声也会触发更新；浮点子 ULP 停滞也不在这一实数结论内。

### C. 从局部收益到未来收益的新增条件桥

两条互补结果：

1. `future_ce_improves_of_target_mass/finish_future_ce_improves`：对**同一个指定未来分布**，若目标字节的收益足够大、其分布质量足够高，并覆盖其他字节的最坏损失增加及实际读出漂移，则未来 CE 严格下降。未来分布与真实随机世界之间的对应仍须另证，不能任填一个 law 就称为真实世界。
2. `Process.next_byte_improves`：在明确的确定性开放世界中，若**实际下一字节**重复刚到达字节，且下一拍完整读出（包括下一次时间 prior）相对更新 head 的保存特征 logits 漂移不超过 δ，局部收益大于 2δ，则实际下一字节的 NLL 相对更新前预测更好。

第二条用的是世界实际后继字节，不是挑选一个符号重评分；没有在当前预测中读取未来字节。它是可预测持续窗口的充分条件，不是一般语言必须连续重复字节的必要条件。

## 4. 同轨迹条件联合结果

语言生命主线采用 `Process.scored_external_language_trajectory`。它的所有量由同一个 `Process.run` 生成；不引入摄入函数、机会势、食物池或维护货币。原 `scored_joint_trajectory` 保留为一般开放系统的旧工具，不再作为语言生命准入来源。

`ExternalLanguageLaw` 进一步给出这个“机会”的信息论语言版，而不把它变回任意数值工具。对**同一** GDN 状态 `s` 与行动 `a`，世界注册当前外部字节与一个后续外部字节的有限联合分布 `joint s a`；机会定义为

\[
I(B_{\mathrm{now}};B_{\mathrm{later}}\mid S=s,A=a).
\]

状态和行动在条件中，并不被塞进有限求和或另存成账户。该量非负；它是边界世界的可测性质，既不是奖励，也不是身体的“食物条”。自听可以进入同一个 GDN 更新，但来源标记阻止它仅凭复制自己而冒充外部证据。正信息机会只有在满足 `externallyUseful` 的实际后继评分桥时，才可进一步称为对当前模型有用。

运行时对应的只读分叉接口是 `GDNByteReadoutPort.snapshot/restore/external_future_signature`：它在一个因果边界完整复制现有 GDN 状态；仅在副本上回放带来源的外部 `StreamEvent` 续接；读取续接后的 256 字节 logits 后恢复原个体。局部 head 学习开启时，head/bias 也在快照内；固定的 memory kernel、query 与 encoder 在一次审计中共享且不得改变。它没有把续接、快照或 logits 留在个体状态内，也没有替代真实妈妈世界对“哪些续接合法、何时两份 logits 算同一签名”的声明。

**储备与耗散。** `ExternalFutureProbes` 注册妈妈世界允许的有限组外部语言延续；GDN 当前状态在每个延续后的字节预测组成 `futureLanguageSignature`。因此储备不是权重范数、另设余额或一个“记忆页”，而是同一状态对这些未来语言问题保留了多少可区分的预测签名。`futureReserveClasses` 仅把这一组签名的类别数作为审计量；只有显式声明有限探针与表示精度时，其对数才可解释为 bits。

更新后的签名若仍能读出更新前的每一份签名，则 `futureReservePreserved`。Lean 已证明此条件等价于存在一个恢复读出；若某两个此前可区分的历史在更新后签名相同，则 `losesFutureLanguageDistinction`，并且它等价于上述恢复条件失败。`arrivalLosesFutureLanguageDistinction` 将此判据直接作用于同一外部到达对替代历史的 `step`。这给出遗忘与串扰的语言语义，不增加任何 GDN 状态字段或“耗散账户”。

| 条件/生成元 | 来源 | 可得结果 |
|---|---|---|
| 合法初始精度、非负观测精度/过程方差 | 明确 GDN 算术，已推导闭合 | 任意时刻精度正，数值递推可定义 |
| 正 head 步长 | 当前局部更新参数 | 不永久冻结，但不自动有用 |
| `ExternalLanguageLaw.information s a > 0` | 当前外部字节—后续外部字节的有限条件互信息 | 信息论的语言机会；不添加状态字段、食物币或任意 `Nat` 摄入 |
| `futureLanguageSignature` 与 `futureReservePreserved` | 同一 GDN 状态在已注册外部延续后的预测；更新后能恢复更新前签名 | 状态内的有用语言结构与其保持；不是权重大小或额外储备值 |
| `losesFutureLanguageDistinction` | 两个先前不同的未来语言签名被同一更新合并 | 遗忘/串扰的精确定义；待在实际妈妈世界的合法探针上测量 |
| 实际到达标为 `external`，且该拍更新使实际后继字节评分严格改善 | `Process.externallyUseful`：同一更新、同一下一拍、读出漂移界与收益界 | 这拍是对未来语言有可核验作用的外部证据；自听字节可被整合，但不计为新增外部证据 |
| 若最终采用被动动作，则从某拍起外部条件互信息为零；实际轨迹却任意晚有正外部信息 | `recurrent_external_information_forces_activity` | **推导**实际策略不可能最终被动；行动的必要性来自失去后续语言证据，而非虚构饥饿 |
| 某个实际到达身体—世界状态上，两个动作得到不同字节 | 世界反事实合同 | 无法用忽略行动的函数匹配所有动作后果 |
| 实际预序 NLL < log256 | 同一预测读出 | 相对均匀字节预测有严格信息优势；它不是可自由兑换的资源 |

这是已固定 GDN 递推的条件联合结果，不是对真实语言流已经成立的报告。外部来源的真实性、被动为何失去有益交流，以及上述实际评分条件仍须在同一妈妈语言世界中兑现；储备与耗散须由这一有限状态的可用未来区别定义，而不能添加独立账户。

## 5. 完整生命定义仍需的桥

不能把本轮 `Positive` 或旧联合门的任意 `viable` 谓词当成 `Medium.Lives`。

- 六条物理公理、局部接触/守恒/可逆性到 GDN 宏观递推的精化；
- 原定义要求的边界、实际规则描述预算、世界丰富度；
- 有限状态中可用未来区别的保持、遗忘/串扰损失、有限时恢复与 loop-cut 自维持优势；
- 原定义的记忆条件，以及允许遗忘/再学习时采用哪一版操作性语义；
- 查询/编码、策略和自听—妈妈闭环的在线信用；
- 真实随机语言世界的条件法则、未来收益及持续机会；
- 浮点递推、完整 checkpoint 续接与资源预算。

这些是下一层证明/机制义务；本轮不增加储备账户，不把语言分数兑换成物料，不启动训练，不用离散旧见证代替 GDN。后续执行环境统一用 `vox`。

## 6. 验证记录

- `lake build CGBAxioms.GenerationSelfProduction`：通过，原有限实例兼容。
- `lake build CGBAxioms.GenerationGDNLife`：通过。
- `D:/conda_envs/vox/python.exe lean/CGBAxioms/scripts/audit.py`：最终全量构建、信任/覆盖审计及来源/basis/catalog 刷新均通过；10 项审计守卫通过。
- `D:/conda_envs/vox/python.exe -m pytest tests/test_information_boltzmann.py -q`：42 passed；包含 value→下一字节前向信用与自动微分的精确核对，以及快照恢复该载体。
- 自动索引当前收录 1570 条来源定理；本模块的条目、源码链接和形式陈述由审计生成。这个数字是可检索库存，不是机制数量或能力声明。
- 真实 OWT UTF-8 连续流已做一次预序接线验收：100,000 个训练源事件后不重置地继续 100,000 个独立 validation 源事件；冻结 head 为 8.068 BPB，G14 head 为 4.571 BPB，value→下一字节信用为 4.568 BPB。后者相对 head-only 的 −0.00345 BPB 是正向单种子信号，尚无多种子区间、长期尺度或因果打乱对照，不能写成该学习机制已通过。完整记录在 `results/published/gdn_language_value_credit_owt100k_20261031.json`。
- 未修改优化器。
