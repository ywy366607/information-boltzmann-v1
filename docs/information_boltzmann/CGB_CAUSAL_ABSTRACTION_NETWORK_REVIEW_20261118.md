# 因果抽象网络与水库、木估值的对应

2026-11-18。论文为[arXiv:2509.25236v3](https://arxiv.org/html/2509.25236v3)，*Networks of Causal Abstractions: A Sheaf-theoretic Framework*，D’Acunto、Di Lorenzo、Barbarossa，2026-04-28。完成文献与实码核对，生产机制保持原样。

## 继承与新增

`CGB_CAUSAL_LEARNING_UNIFICATION_20261110.md`已选择因果抽象、合法重叠处的胞腔层一致性，以及绑定—执行—真实证据修订；全局截面ker(delta)=H0及H1含义已经讨论。当前目标仍为木的有效估值带动火选择、金结算、水持存及后续再用。

论文以层/余层组织不同局部混合因果模型的抽象与嵌入，传递概率混合（Remark17）；全局截面表达传递后的相容。§6.5—6.6将截面、连接拉普拉斯核和扩散固定点联系起来，讨论存在与收敛条件。身体接入对象取实际有类型端口。

## 已有实码

|身体对象|实码|核对点|
|---|---|---|
|条件、动作、后果及事实身份|`causal_relation_program.py`、`relation_evidence.py`|各证据参与哪个预测|
|共享未知项及版本|`joint_relation_update.py: relation_variables/register_use`|实际SQL引用确定重叠与再用时间|
|两条后果路径|`generative_relations.py: execute_programs`|回应→运输→下一边界，与直接下一边界|
|同端口协调|`joint_relation_update.py: overlap_consensus`|均值、scale2及程序权重形成精度中心|
|延续价值|`navigation_internal_rollout.py: continuation`|边界预测触发经历寻址、候选加权、收益读出|
|即时有限U净用途|`finite_use_value.py: expected/observed`|同一revise计入新增、替换和提交|
|行动总估值|`causal_use_value.py: utility`|10×校准后即时净用途＋gamma×延续价值|

现有局部对象是有证据的预测程序。进一步实例化论文MCM时，需将结构选择、机制、外生变量与干预语义逐项绑定；首先比较现有真实程序的计算。

## 找到的具体计算差别

`execute_programs`保存`program_predictions=(mediated,direct)`和权重。启用`joint_relation_update_schema==1`后，先执行`overlap_consensus`，再将其mean送给`continuation`。configure的revision1—4都设置schema1；用途实验要求revision≥4，走的是该分支。

逐坐标协调中心：`z_c = sum_i(w_i / scale2_i * mu_i) / sum_i(w_i / scale2_i)`。

原函数另一分支已有`sum_i w_i * continuation(mu_i)`。因此可复用生成器，直接比较“各程序先推演再汇总价值”与“先协调预测再算价值”。差额发生在木的后果表征进入未来经历寻址之前。

自己的推导：即使采用普通均值，非线性V一般也有`V(sum_i w_i mu_i) != sum_i w_i V(mu_i)`。实际V含有限top-k经历选择，跨寻址边界会更换候选。应同时记录程序分歧、召回身份、收益采信系数、价值差额及实际排序，由原身体计算判断影响大小。

这一路直接进入C2的`gamma * path['value']`。即时净用途从`path['mother']`的后验假设进入有限U积分，两项分别记录后合成总排序。不能把延续分支的压缩直接当成全部C2结果的原因。

## 数学实例化

局部适用条件和动作端口决定合法信息传递。学习抽象映射时，用实际行动后预测保持作为约束；跨作用域私有细节仍由局部状态保存。已有5082维Dirac描述依赖复形；论文的连接拉普拉斯应由实际抽象/嵌入块构造，读出相容自由度。既有62维一阶调和空间保留原测量含义。

Lean先审查实际方程。原文Lemma22、Theorem23把lambda取值写为[0,1]；由Eq7代入lambda=1，更新恒等，任意初态都是固定点。由固定点推出相容需要有效交流条件。实例化写入正的实际交换、自保持、连通及映射一致性，再证明身体传播的目标性质。这是明确的代数端点检查。

## 下一执行项

1. 复用C2记录/完整边界，同参数同候选分别读取即时U价值、协调后延续价值、逐程序延续价值；保持事实及样本身份。
2. 将两种实际计算映射至Lean：同端口绑定、权重归一、换序对应、仿射V的混合恒等；非线性/寻址差额由身体实算。
3. 对真实候选比较总排序及匹配后果评分，据实际差额决定整圈学习对照；同时评价实际支持再用与共同旧查询保持。
4. 跨作用域抽象与分布扩散沿共享端口扩展，每条新连接绑定真实证据和再次使用收益。五行检查整条作用链覆盖。

本轮新增训练0、世界调用0。C2完成状态及32轮结果见`CGB_VALUATION_ACTUAL_U_20261118.md`；主身体保持2401。下一项为上述第1项。
