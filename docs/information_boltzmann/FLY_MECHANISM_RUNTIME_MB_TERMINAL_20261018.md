# 当前候选的统一接线与验收清单：MB末梢接入

历史快照：本表对应32968存档。后续APL局部形态接入见
`FLY_NATIVE_APL_CABLE_20261018.md`，当前验收清单见
`FLY_MECHANISM_RUNTIME_APL_CABLE_20261018.md`；不要把本表旧缺口当成新存档的状态。

候选：`E:\fly_stream_overflow\native_mb_terminal_20261018\last.pt`，32968词、4146次累计更新。主线32384保持暂停。

165122个细胞的注释分组可以重叠，不能把下列计数相加。专用开关、急性作用送达和持续学习收益分别记录。

| 系统 | 注释细胞 | 当前专用开关 | 当前存档急性检查 |
|---|---:|---|---|
| 感官入口 | 15896 | early_visual_graded=开, sensory_precision=关 | visual: 预测 1.009 / 选词 4.331 |
| 嗅觉中继 | 3783 | local_relay=开 | al_relay: 预测 0.9023 / 选词 2.51 |
| 蘑菇体/熟悉度 | 4501 | kc_mbon_memory=开, kc_dan_feedback=开, dpm_apl_axonal_projection=开 | mb_regulators: 预测 0 / 选词 0；mb_terminal: 预测 0.9131 / 选词 2.079；mb_terminal_gaba: 预测 0.522 / 选词 1.548；dpm_5ht: 预测 0.9692 / 选词 3.749 |
| 神经调质 | 389 | dopamine=开, oa_5ht_extension=开 | amines: 预测 1.513 / 选词 3.77 |
| 中央复合体 | 2950 | cx_extension=开 | cx: 预测 0.0003471 / 选词 0 |
| 关联/上下文 | 121550 | 通用图动力学 | 未独立验收 |
| 上行/身体反馈 | 21362 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关 | 未独立验收 |
| 下行/VNC/运动 | 2227 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关, native_motor_language_port=开 | motor_terminal: 预测 0 / 选词 10.87 |
| 视觉/长程投射 | 99154 | early_visual_graded=开 | visual: 预测 1.009 / 选词 4.331 |
| 内分泌/ENS | 141 | lk_receptor_exchange=开 | endocrine: 预测 0 / 选词 0 |

表内数值是配对干预相对完整支的最大log概率差，**不是NLL改善或学习收益**。

当前13支统一检查使用存档接下来的4个真实输入间隔 `[7,6,8,4]`，共25拍。
同状态、实际感觉采样、sham、恢复、来源与RNG控制通过；旧MB全局代理关闭的影响为0。
新增整条末梢、GABA和DPM5-HT干预都影响MBON/DAN及独立选词分布。
这批极小样本的三个急性关断支NLL均更低，须保留这个结果，不能把作用强度当收益。

本轮32个已保存妈妈主动输入、8次更新、271实际拍；更新前NLL **5.6357**，同批固定词频 **7.6067**。
当前妈妈持有话轮，未发生新的自然发声或对话轮次；不能声称互动已改善。

196项相关检查通过，包含CPU/CUDA、重算梯度/状态及恢复；独立Deslice检查通过。
全量入口受到无关导入即训练的基准脚本阻塞；排除这8个入口后仍是之前同样5项缺失诊断脚本的收集错误。
完整训练器的模型、脑状态、优化器、信用、世界队列和RNG恢复逐项一致。

**仍缺**：APL胞内局部非脉冲空间机制、KC各bouton状态、DPM其他肽/受体与靶点、新慢动力学跨窗敏感度；真实外周与完整ENS。
原生熟悉度的选择性降值、自然选词后果的持续学习收益、完整探索与自我仍未验收。

本轮机制与生物依据：`FLY_NATIVE_MB_TERMINAL_20261018.md`；完整缺口表：`FLY_NATIVE_COMPLETENESS_20261018.md`。
本轮报告：`results/published/fly_native_mb_terminal_20261018.json`；机器清单：`results/published/fly_mechanism_runtime_mb_terminal_20261018.json`。
