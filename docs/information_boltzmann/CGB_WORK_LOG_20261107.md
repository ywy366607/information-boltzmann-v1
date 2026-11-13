# 2026-11-07 CGB 当日工作核对

核对截至22:52，北京时间。此处从当前源码、原始运行JSON、精选报告及NTFS时间戳重建工作，不把“计划”“准入通过”“代码可运行”写成能力成功。文件未提交且工作区有并行代理，不能凭git dirty或文件名归属作者；下文分开列出本聊天已有执行证据的工作和另发现的并行产物。

## 时间依据

- 下表运行起止优先用原始报告的`seconds`与完成文件时间反推；它是分钟级约值，包含该报告计时范围内的加载/检查。
- 没有运行起始字段的，使用目录创建时间、运行产物与源码修改时间给出约略区间，标“约”或“截至”；不把最后修改时刻当作整段工作时长。
- JSON原子替换会改变文件创建时间，不能据其CreationTime断言运行从那一刻才开始。NTFS主机时区已核对为China Standard Time。
- 测试数量是各阶段相关测试通过数，彼此有重复，不相加成独立测试总数。

## 按时序逐项核对

| 时间 | 代码/实际操作 | 实际运行与结论 |
| --- | --- | --- |
| 约00:29–00:34 | `pretrained_gdn_experience.py`的`read_interval/query_positions`、`pretrained_gdn_learning.py.set_read_interval`及`probe_pretrained_gdn_read_cadence.py`：经历读取与128词元训练窗口解耦，旧胶囊每窗只重算一次、每位置独立查询 | 491源零保留更新迁移；两层各128查询，24/27条胶囊重算，磁盘下一步恢复与因果前缀检查通过。得到逐词元读取能力，未证明任务收益。证据`cgb_read_cadence_wiring_20261107.json`。 |
| 约01:03–01:05 | `audit_pretrained_gdn_gradient_balance.py`：分解当前/旧信用、CE/SIGReg、裁剪及实际Adam位移 | 三个真实窗口、两种读取周期，12个丢弃影子步。旧信用确实改变门控/投影和FP16存储，但没有由此测出任务收益。证据`cgb_gradient_balance_20261107.json`。 |
| 约08:29–08:48 | `prepare_pretrained_gdn_credit_stream.py`、`run_pretrained_gdn_retention.py`、`analyze_pretrained_gdn_credit_pair.py`：真实OWT连续X→Y→X对照 | held/full各96更新、12288评分词元；源491到587。流内NLL held2.808548/full2.810348；部分旧域续文保持略好、当前域适应略差，未获得全面收益。证据`credit_continuous_20261107/report.json`。 |
| 约09:08–09:27 | `prepare_pretrained_gdn_ruler.py`、`evaluate_pretrained_gdn_ruler.py`：准备官方RULER子集，做预训练答题/长度准入 | 证实该纯GDN底座确实能答若干单针题；此时评估每题初始化原生S，尚不是持续个体验收。证据`cgb_ruler_admission_20261107.json`、`cgb_ruler_lengths_initial_20261107.json`。 |
| 09:27–10:33 | `train_pretrained_gdn_ruler.py`：预训练held/full答案监督，各64题 | 两组都退化。后来补完整数字匹配后，held三长度均2/4、full均1/4；原官方包含率掩盖了多余数字。此轮还有答案单独分词的边界错误，不能作修正后配对的结果。原始`ruler_20261107/learning_pair/report.json`，3964.32秒。 |
| 截至10:55 | `analyze_ruler_generation.py`、`audit_pretrained_gdn_ruler_generation.py`及作答编码适配 | 查出SentencePiece单独编码答案会多出句首空白；改为联合分词、验证提示边界。补首个完整数字及EOS诊断，保留历史包含率；未将适配修复说成学习成功。 |
| 10:55–11:30 | 修正分词后`train_pretrained_gdn_ruler.py`复测 | `learning_pair_joint` held64题后因用户STOP退出，full0题，报告failed/InterruptedError。不是完成的两臂实验。2088.37秒。 |
| 11:32–11:33 | `small_gdn_ruler.py`及完整恢复准入，切回原1792轮四层语言身体 | 参数/Adam/原生状态等恢复，GDN和MLP梯度通过；选定源与原平台SHA相同。此时原个体没有被失败终态覆盖。 |
| 11:45–11:48 | `train_small_gdn_ruler.py`：四层身体短布局学习 | 1024实际更新，始终未答对完整数字（0/8），OWT退化；没有验证旧原生重算信用的开关。原始`short_task_learning/report.json`，183.72秒。 |
| 约11:50–12:04 | `audit_small_gdn_ruler_routes.py`：交换快表/神经权重、检查首位寻址 | 先前数字替换效应主要来自快表；首位词法地址跨题相同。不能称神经身体本来完全无记忆，也不能把快表效应当成长信用收益。 |
| 约12:14–12:20 | `small_gdn_experience.py`、`train_small_native_credit.py`、`run_causal_water_reply.py`：每层保存前层输入、本层S/卷积边界，重算GDN+MLP | 首次接线12:17因CPU/CUDA张量混用失败；12:18修复重跑。schema1每窗前65位置、4层各16槽，非零旧梯度与恢复检查通过；只是接口准入。 |
| 12:20–12:31 | schema1 held/full各128题学习 | 两臂短/长完整数字均0/8。669.52秒；终态未晋升。 |
| 约12:31–12:49 | `audit_small_native_failure.py`及schema2保存/查询修复 | 查出65位置切片漏掉部分数字参与、浅层查询偏静态；改整128位置及尾段保存、递推上下文查询、按位置取回。当前相关记录的直接梯度接通，但不预定任务成功。 |
| 12:51–13:02；截至13:05 | schema2 `context_pair`及`audit_small_native_content.py` | 两臂各128题，仍0/8。固定当前状态仅替换旧数字内容后，正确值方向仅1/8，说明删记录的影响不等于正确内容使用。634.34秒。 |
| 13:22–13:37；截至13:40 | `predictive_responsibility.py`：schema3把候选各自预测混成概率，用观测后验结算责任 | held/full各128题，短/长仍0/8；后验与导数数值通过，但正确旧内容方向未成立。944.30秒，不晋升。 |
| 约14:50–15:05 | `audit_small_native_information.py`、`audit_small_native_successor.py`：编码内容与同刻/后继关联诊断 | 原有连续表征可以解码数字；浅层后继关联更好、深层同刻更好。研究者指定记录的读出不是身体已学会检索，未据此报能力成功。 |
| 15:46–15:47；截至15:53 | schema4双视图：同刻和已观察后继，各层新增选择先验；`analyze_small_dual_view.py` | 仅8次校准更新；65相关测试通过。旧内容选择性功能门未过，未开长对照。不能据8步认定学不会。53.68秒。 |
| 16:24–17:19 | schema5直接注意力：候选不再各跑全后层，旧响应作为value，一次真实后层前向；`small_gdn_experience.py`、`train_small_native_credit.py` | held/full各1024题；两组完整答案均0/16，多题输出同一数字，OWT约3.08。3299.45秒；这是今天一条明确失败的学习路线。 |
| 约17:24–17:28 | `audit_small_cross_window_credit.py`：只切>128的旧原生导数，并加错配历史 | 固定前向，真实Adam更新确实变化；正确旧教学相对切断/错配的内容收益不稳定。补上原full/held可能包含窗内记录的检查缺口。 |
| 约17:28–18:00 | `audit_small_native_delivery.py`：逐层跟踪内容、读出和入口符号 | 第2层确实读到数字区域，但首位质量小；入口增益学成负值。仅翻符号可翻转值效应，但顺序对比仍4/8，不能永久翻号或称成功。 |
| 18:20–19:16（中断） | 显式可学习输出矩阵`g*r+W*r`，新增65536参数 | held完成1024题，full记录到969题；原始report仍running但进程已结束，没有完整两臂终点。已记录的答题仍失败；不能报告完成或假定中断原因。 |
| 19:18首次启动未进入训练；19:20–19:39正式完成 | `small_gdn_counterfactual.py`、`train_small_counterfactual.py`：近距离等长数字替换配对，不新增对比损失 | 512对/1024更新，真实间隔17–36字节；固定0/32、新鲜0/64。Top1和指定旧首位活动没有形成可靠输出，1158.51秒。之后停止小身体数字课程。 |
| 19:49–19:58 | `audit_pretrained_gdn_memory_location.py`：查预训练是否靠窗口注意力；清S和换数字 | 运行时24层全纯GDN，无标准窗口注意力。12题完整11/12、关经历读取10/12、读题后清全部S为0/12、换隐藏数12/12。552.16秒；清题后S不同于题前reset。 |
| 约20:09–20:14（GPU检查约24.76秒） | `small_gdn_stream_eval.py.CarriedStream`、`audit_small_state_consistency.py`：修正持续评估状态协议 | 两检查点、真实OWT512字节，训练前向/持续推理logits差0，末态/记忆/前字节对齐；22相关测试。尚未把自由生成及后果接成不可逆主评估。 |
| 20:24–20:36 | `native_state_policy.py`、`train_small_layer_reset_pair.py`：四层全携带vs第1/3层每128字节清S，训练评估同规则 | 各4096更新、524288到达字节，A→B→A→C。终验证1.363070/1.366546，交替清S没有优势；两组较起点均退化约.0455。683.93秒。此为历史候选，不再当生命运行规则。 |
| 21:36–21:37 | `evaluate_layer_reset_answers.py`：给上述两终态补连续状态策略下的自由答题 | 现有三长度各8题；两组均0/24。生成按同128边界，16相关检查。92.10秒；答题没有区分两策略。 |
| 21:46–21:51 | 新预训练答题审计加载失败和修复：`pretrained_gdn_stream_load.py` | 直接GPU读取、捕栈重试、CPU后转GPU三次在Safetensors映射存储访问失败，均0题/0训练。逐张量普通二进制读入解决；435参数张量、原FP16主体/F32时间参数保留，只忽略24个确认为零的旧D。 |
| 21:51–22:32 | `audit_pretrained_continued_answers.py`：491共同源及587 held/full终态，原题与真实OWT长干扰 | 三身体原题9/12、8/12、7/12；插16384词元真实文本后全部0/4。48次冻结作答、2429.51秒，没有训练。每题仍恢复源且生成不提交，因此仅外部诊断，不能算不可逆生命史。 |
| 晚间（讨论与只读记录，未取得逐条时间） | 明确不人为reset、会话不可逆，核对官方实现及本地调用；更新执行规范 | 区分chunk/detach/状态清零；取消默认新增预训练交替清S。发现“不清零但每题恢复存档”仍不符合生命主评估。没有因此自动实现新的不可逆运行器。 |
| 截至22:28 | 从现有96步逐层记录统计年龄、质量与命中：`cgb_credit_age_scale_20261107.json` | 所选记录约98%年龄>128，中位约2100词元；远距读取并非稀少。没有有效责任标签，未宣称Lévy或幂律拟合成立。 |
| 22:36–22:38；截至22:40整理 | `credit_scale_diagnostics.py`、`audit_pretrained_credit_direction_scale.py`：参数量、同幅错配、等层RMS放大 | 3真实窗口、15个全部丢弃影子步，125.33秒。原旧信用实际改33万–68万坐标；放大改190万–239万。原/放大分别0/3、1/3续文优于held，均只1/3胜等幅错配；不安装直接放大。教学能量集中，但未证明正确责任。 |

## 单列：发现但不据此认领的并行产物

`scripts/ib_local/audit_pretrained_gdn_stream_pair.py`文件最后修改20:33，原始`stream_pair_20261107.json`明确started_at对应20:42、总235.78秒，约20:46完成。实际只运行`reset_native`和`continuous_native`，不是注释列出的四个臂；分别10/12和9/12。

它在生成旁路之后调用`advance_answer(gold)`，把标准答案而非刚生成的内容带入后续状态，并且起点新建空cache。它有“跨题携带状态”的研究信息，但不满足用户后来明确的“生成自身也留在唯一不可逆历史、从原完整生命态续接”要求。不能把这份代码的存在称作已经完成生命主评估；文件作者/执行者未由本聊天工具记录确认，单列避免混为自己的工作。

## 今天实际交付与尚未交付

实际留下的工程结果：读取周期解耦；答案联合分词修复；整段参与保存和上下文查询；数版可恢复候选；纯GDN记忆位置的干预证据；持续状态前向一致入口；非mmap的GPU权重加载；逐层参数量及错配/尺度诊断。重复测试次数不当作独立证据数量。

**没有交付：**跨窗口原生信用的稳定答题增益；能同时满足不可逆生活、自听、后果和持续学习的主验收；有效归一化/资格迹学习规则；预训练交替清S对照；新增Lean定理覆盖；嵌入CPU或在线Lean执行器。原1792轮已接受平台点没有被这些失败终态覆盖。

**工作失误：**小模型反复在准确率地板上改结构、训练后仍未建立有效取用；多次用接口/数值通过推进下一版而没有任务收益；主生命史要求没有从开始落实到评估；最后仍持续用冻结旁路作为主要报告。AGENTS顶部不断叠加“当前/最新”记录，留下过时指令，也是文档管理问题。本核对不再由“又跑了一项诊断”宣称生命体进展。

git当前HEAD为`fc57109`，时间2026-11-05 17:41+08:00；今天这些工作尚未形成新提交。本次只核对/记录，没有清理、提交或启动训练。

## 核心证据入口

- `results/published/cgb_credit_continuous_20261107.json`
- `results/published/cgb_small_native_credit_20261107.json`
- `results/published/cgb_small_native_context_repair_20261107.json`
- `results/published/cgb_predictive_responsibility_20261107.json`
- `results/published/cgb_small_dual_view_20261107.json`
- `results/published/cgb_small_direct_attention_20261107.json`
- `results/published/cgb_strict_cross_window_credit_20261107.json`
- `results/published/cgb_counterfactual_retrieval_20261107.json`
- `results/published/cgb_pretrained_memory_location_20261107.json`
- `results/published/cgb_state_consistency_20261107.json`
- `results/published/cgb_layer_reset_pair_20261107.json`
- `results/published/cgb_layer_reset_answers_20261107.json`
- `results/published/cgb_pretrained_continued_answers_20261107.json`
- `results/published/cgb_credit_direction_scale_20261107.json`
