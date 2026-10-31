# 果蝇耦合线收尾（2026-10-19）

用户 10-19 决定关闭这条线。这里只记录最后的结果和尚未回答的问题，方便以后归因。

## 最后一次对照：两个个体都活在妈妈世界里

两个个体都只用局部学习（只算一遍，不重演，没有 BPTT），回路观察器开着。

| 项目 | 补回路前（32384 → 32896，512 词） | 原生（33000，6 词后主机内存溢出） |
|---|---|---|
| 多巴胺随意外度变化（gate 2） | 不成立：PAM ρ −0.095，PPL1 +0.04，MBON 处多巴胺 +0.01 | 词数太少，无法计算；多巴胺从 3.4 单调降到 1.5，与意外度无关 |
| 熟悉度到 MBON | ρ −0.08（p 0.08）；KC 约 0 | 无法计算 |
| 读出在学（重复次数 vs 意外度） | ρ −0.73 | — |
| 开口 | 轮到 12 次，开口 1 次，11 次沉默让出 | 没有轮到它 |
| KC 发放率（每细胞每拍） | 0.024 | 0.0001–0.0014，第一个词起就几乎为零 |
| DPM | 0.31 | 第 3 个词起为 0 |
| PAM / PPL1 | 0.026 / 0.040 | 约 0.11，高且各词几乎相同 |

原生个体的速度约每拍 3 秒（GTX 1650，4GB）。显存峰值 5.8GB，溢出到共享内存。主机提交上限是 23.8GB，其中约 16GB 被系统占用，第 6 个词时在 `diffuse_rows` 里分配失败。

## 尚未回答的问题

1. **原生个体的 KC 为什么几乎沉默。** 有两种解释：
   - 原生机制（嫌疑最大的是 KC 末梢的局部 GABA）是装在按"未接通大脑"学出来的权重上的，两者不匹配；
   - 原生机制本身就过强。

   区分的办法是出生检验：在 `E:/ib_checkpoints/newborn/birth.pt` 上只迁移和蘑菇体有关的几步，听几十个词，看 KC、DPM、DAN 是否活着。这一步没有做。
2. **补回路前个体的 gate 2 和开口为什么失败。** 原因未定位。
3. **从头养一个个体需要快的运行器。** 设计见 `FLY_LEAN_LIFE_20261019.md`，未实现。

## 留下的东西

- 代码（未提交）：
  - `fly_kc_terminals.py`：按树并行求解，fp32；
  - `fly_native_runtime.py`：索引压缩为 int32；
  - `fly_bptt_learning.py`：`--brain-learning local`、`live_commit`、`TICK_OBSERVERS`；
  - `fly_mother_world.py`：实时提交；
  - `fly_circuit_observer.py`、`circuit_report.py`、`profile_native_forward.py`；
  - 相关测试。
- 结果：
  - `F:/fly_stream_overflow/coupling_compare_20261019/`：两边的 circuits、日志和妈妈对话，没有 .pt；
  - `F:/fly_stream_overflow/profile_native_forward_cpu*/`。
- 全脑接触数据：
  - 分桶在 `E:/fly_stream_overflow/whole_brain_contacts_buckets`；
  - 投影没有做完。可以用 `project_whole_brain_contacts.py` 续跑（`--stage post`，然后 `--stage pre`）。
