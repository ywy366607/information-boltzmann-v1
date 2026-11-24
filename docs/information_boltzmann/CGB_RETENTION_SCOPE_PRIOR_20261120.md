# 父关系经验接入未知项：跨作用域保留先验

## 从完整生活定位到的接点

承接`CGB_ROOT_NATIVE_VALUE_DELIVERY_20261120.md`末节，同源observe/apply2553→2569均完成整圈。实际U已按同一当前坐标分解为新证据加入收益与释放旧支持的影响：apply末4轮均值分别-0.001459705、-0.017173279，合计-0.018632984；释放项占这段净负值92.17%。同样分解的observe末段为+0.013089741、+0.008644355，合计+0.021734096。

现有规则在relation与unknown分别持存first/second/count，已有文档`CGB_RELATION_RETENTION_USE_20261119.md`明确各scope只积累自己的反馈。这次复核新未知项的真实构造：`_open/_seed`复制父关系的真实事实引用，而`_with_constraints/candidates`为未知项读出独立的空用途快照。因此同一事实已经被关系反复评分，新未知项仍可能以零偏置释放它。

旧记录直接重放：observe16轮中12轮没有本地正计数，8轮可读到同一事实的父关系经验，身份绑定先验会改变6次原释放选择；apply13轮没有本地正计数，其中10轮有可取用父经验，会改变4次释放选择。重放核对生产原优先度再计算新优先度，只有release选择的回顾性重放，未生成替代世界后果。apply的4次变化发生2555/2559/2561/2563；末4轮原选择保持，后续完整生活比较检验早期接续对以后路径的作用。原始结果：`results/published/cgb_retention_scope_handoff_20261120.json`；复现入口`scripts/audit_retention_scope_handoff.py`。

## 同一生成规则

对一个真实事实i：未知项有自己的已结算观察时读自己的moments；本地count为零且当前父关系有同一identity的既有moments时，读父关系作为保留先验；其余按原几何中性值。优先度仍为原`h_i exp(tanh(first_hat/(sqrt(second_hat)+eps)))`，betas/eps沿用现有read-credit。

父先验与本地观察分开存储。attach只读身份交集和父scope/revision，既不增加本地count，也不修改父关系。真实后果到达后，原advance仅更新本地first/second/count；第一次本地证据以后，该事实由本地统计接管。两个scope对同一事件的观察不相加。每个快照仍最多原8个本地事实统计，再加最多8个只读父引用统计，额外状态为少量标量。

这是一项明确的模型候选：把既有文档中的严格独立scope统计改成“本地观察优先、同身份父经验作冷启动先验”。它利用过去实际发生的证据，未来真实结算负责验证是否值得使用。

## 全部接点

- 水：SQLite原relation_retention/unknown_retention继续持存，payload增添prior_scope/prior_revision/prior_moments；旧payload默认空先验。
- 木：`shared_unknown_bindings.candidates/_with_constraints`在相同身份上附父先验，原raw/encoded U共同读取；`native_amortized_decision.recipe`以有效保留读数填入原三个统计标量槽位，输入维度保持，银行读出也感知本次保留条件。
- 火：原question/actor读取同一预演；保留原概率策略、两次原生参照、持存读出及内部S/P交接。本轮root_native_correction固定observe，独立测先验接续。
- 土：原自然文本导航提供实际回应，原冻结fresh评分核对到达的新后果。
- 金：原有限U/v5的加入与释放分解、单位和原后果结算保持；原验证核对实际支持及完整用途快照；advance只把这次真实后果记到本地count。

源码：`relation_retention_use.py`、`retention_scope_prior.py`、`shared_unknown_bindings.py`、`native_amortized_decision.py`；恢复入口`run_mother_continuation.py`按config重接。预演和真实提交相同规则，父版本是选择时读到的冻结收据；本地提交继续使用原revision/CAS。

## Lean与身体契约

已检索CGB_PROVEN_INDEX、既有NativeRelationRetentionUse/GenerationExport，复用选择性保持、优先度有界、旧统计选择与后到观察更新合同。`reviews/RetentionScopePrior.lean`7项编译通过：冷启动读父、本地观察优先、身份过滤、同身份读、一次真实到达只加一个本地count、首次本地观察接管、父统计不混入本地更新。

`tests/test_retention_scope_prior.py`检查身份过滤、本地计数、幂等、先验只读、真实未知项生成/提交；`test_persistent_rhythm.py`新增先验版本的完整保存恢复后下一事件/权重/Adam/交织时序核对。复用原relation retention与shared unknown整圈契约。数值契约与生活收益分别报告。

## 完整生活协议与资源

共同起点为上一observe完整2569，比较off与relation两臂，各16轮至2585、64次生产更新、15360自然文本字节。两臂保留完整导航、学习、原有限U、内部活动、持存修订；付费妈妈0、QA/SFT0。验收真实U、加入/释放分量、固定64旧事实、后续收益预测、实际父先验消费、耗时与用途/分钟。

为减少临时副本，每臂采用一个16轮段，结束保存和独立恢复；4个串行子进程各创建一个私有UUID快表工作副本，快表本体嵌入各自完整checkpoint。新副本总量约0.9GiB，统一放C指定目录，训练/恢复不同时持有同一工作副本。D保存两个完整身体、归档和报告。启动要求D至少3GiB、C临时盘6GiB，逐臂检查。删除上一已完成实验的退役审计副本命令被自动审批以blocked by policy拒绝，原件保持；本轮通过减少新副本数量安排运行。

入口`scripts/run_retention_scope_prior_comparison.py`；工作目录`scratch/cgb_retention_scope_prior_2569_20261120`；报告`results/published/cgb_retention_scope_prior_2569_20261120.json`。原2401/2493及原所有研发源保留；启动后运行源码冻结至两臂和恢复报告结束。

## 当前游标

7项Lean已编译；真实身体与恢复回归执行中。通过后按上述完整协议启动后台对照。


已通过7项Lean及25项针对性身体检查（112.32秒）：`pytest tests/test_retention_scope_prior.py tests/test_relation_retention_use.py tests/test_persistent_rhythm.py tests/test_root_native_correction.py tests/test_shared_unknown_bindings.py -q -x -k "not versioned_clock_full_restore_keeps_next_loop"`。本轮覆盖当前持存/交接/新先验的完整磁盘恢复；24项旧schema笛卡尔恢复矩阵未纳入本次针对性结果。后台supervisor27584已启动off/relation各16轮，源码冻结至两臂及恢复报告完成。

具体旧事件例：2555轮unknown:1767本地统计为空，父relation:241:5对episode:1616已有3次真实结算、first=0.01323358。原几何杠杆0.340756使其先于无父统计的episode:2465（0.386464）释放；同身份父先验会改为释放2465。原实际加入收益+0.02491154、释放影响-0.05773658、净U=-0.03282505。替代选择的后果由新完整生活对照检验。

## 恢复审计兼容修复及原进度接续

2026-11-20 13:24，off臂已完整保存2585，16轮/64更新/15360字节，训练636.016秒。随后off_restore在逐字比较归档JSON与加载后的dataclass时失败：25条relation与68条unknown旧payload没有新增prior_scope/prior_revision/prior_moments，加载正确补空值，旧审计仍要求JSON键集完全相同。

审计现只为缺失的三个新字段补各自空默认，再逐项严格比较全部字段。原保存的非空先验、旧计数/动量/版本/单位仍严格比较。`tests/test_retention_restore_audit.py`11项通过（0.23秒），覆盖旧字段兼容、原值改变拒绝及非空先验精确保留。训练源码、原始归档、完成检查点和共同2569源保持；原失败日志保持。

接续入口`scripts/resume_retention_scope_prior_comparison.py`，supervisor19356，13:42启动。复用完成off结果，从独立恢复重试开始，然后启动同源relation16轮、恢复和原汇总。`restore_audit_amendment.json`逐项记录唯一审计修改的原/新SHA256；原hash清单保持，所有训练源码及共同源继续严格核对。恢复重试会新增一份私有快表，原失败副本保持供审查。

## 同源完整结果与逐项接续

两臂均完成2569→2585、16轮/64更新/15360自然文本字节，两次独立完整恢复通过，18项运行核对通过；付费妈妈0，原生参照预算、有限U、内部交接和学习均保持。报告`results/published/cgb_retention_scope_prior_2569_20261120.json`。

|指标|off：本地经验|relation：来源经验迁移|
|---|---:|---:|
|实际U均值|+0.002095961|-0.003536731|
|加入新证据收益均值|+0.014574943|+0.013231484|
|释放旧支持影响均值|-0.012478983|-0.016768215|
|正U轮数|10/16|7/16|
|固定旧事实分数变化|-0.000204877|-0.001540276|
|后续实际收益预测总MSE|0.129778281|0.136916193|
|前后策略价值衔接分量MSE|0.047259898|0.080555402|
|到达后实际收益偏差分量MSE|0.058604032|0.034165606|
|训练秒数|636.016|734.288|

来源经验12轮被读取，按当前同一几何和冻结本地统计撤去prior重放，7次释放身份改变。真正无本地观察、读取来源评价且获得当前真实标签的43对事实中，29对同号、14对异号。例2575：episode:263来源first=-0.00947141、count=1，当前真实MOTHER边际用途+0.04467237；迁移后释放263，原本地规则在同一当前状态会释放2553。原事实身份相同，旧用途及当前用途已经反转。上述单端口边际分数与完整U分别保留；替代释放只重放身份，实际替代后果尚未生成。

收益均值差-0.005632692，其中释放项差-0.004289232，占76.15%。末四轮off U+0.011023447，relation-0.005984627。候选到达后实际收益偏差分量降低41.70%，前后策略价值衔接分量升70.45%，总后续收益MSE升5.50%；这些量分别对应`admitted-actual`、`contextual_future-on_policy`及整体误差，分量平方带交叉项，按原有符号分解解释。

保留off2585作后续续接参照，relation2585保留研究分支，后台已结束，源码冻结解除。下一复用43对真实记录，检查来源评价的适用条件与当前调用/行动/世界后果，以及MOTHER单端口边际用途与整次有限U释放收益的对应；将信用迁移是否适用与木的后续价值排序放在同一事件链核对，再确定接线修改。

逐项核对可复现：`python -m scripts.audit_retention_scope_prior_followthrough --work scratch/cgb_retention_scope_prior_2569_20261120 --output results/published/cgb_retention_scope_prior_followthrough_20261120.json`。逐次重建原优先度和原释放身份核对通过；归档SQLite只读，新增训练/世界调用0。
