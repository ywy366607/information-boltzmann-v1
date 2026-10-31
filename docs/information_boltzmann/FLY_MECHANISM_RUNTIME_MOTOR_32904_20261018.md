# 32904候选的统一接线与验证清单

检查点：`F:/fly_stream_overflow/native_motor_final_20261018/last.pt`。主线32384仍暂停。

全图包含165122个神经元；下表的系统注释可以重叠，不能相加。原生连边存在、通用动力学参与、专用规则开启和学习收益分别记录。

| 系统 | 注释匹配细胞 | 实际专用开关 | 本次急性作用 |
|---|---:|---|---|
| 感觉入口 | 15896 | early_visual_graded=开, sensory_precision=关 | visual: 预测Δ+0.0868，动作logp最大Δ3.2830 |
| 嗅觉中继 | 3783 | local_relay=开 | al_relay: 预测Δ+0.0203，动作logp最大Δ3.2390 |
| 蘑菇体记忆 | 4501 | kc_mbon_memory=开, kc_dan_feedback=开, dpm_apl_extension=开 | mb_regulators: 预测Δ+0.0374，动作logp最大Δ5.2194 |
| 神经调质 | 389 | dopamine=开, oa_5ht_extension=关 | 未独立测量 |
| 中央复合体 | 2950 | cx_extension=开 | cx: 预测Δ+0.0167，动作logp最大Δ2.5860 |
| 关联/上下文 | 121550 |  | 未独立测量 |
| 上行与身体反馈 | 21362 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关 | 未独立测量 |
| 下行/VNC/运动末端 | 2227 | historical_body_state_retained=开, historical_proxy_diagnostic_only=关, native_peripheral_exchange=关, native_motor_language_port=开 | motor_terminal: 预测Δ+0.0000，动作logp最大Δ9.7891 |
| 视觉与长程投射 | 99154 | early_visual_graded=开 | visual: 预测Δ+0.0868，动作logp最大Δ3.2830 |
| 内分泌/ENS | 141 | lk_receptor_exchange=开 | endocrine: 预测Δ+0.0000，动作logp最大Δ0.0000 |

当前统一矩阵9支/54个真实时钟tick，sham/恢复与来源/RNG控制通过。运动末端关闭使指令可用拍数54→0，原感知预测不变。这个读出分布的急性结果不等于真实行为学习收益。

独立词语行动外设与原预测头解耦；实际词语、时机与自听/妈妈回复按同一事件时钟回放。接口测试和完整恢复已通过，当前资源短段未自然发声。

仍缺：真实肌肉/受体及本体感觉、完整ENS与外周交换；OA/5-HT扩展当前关闭。原生phi选择性熟悉降值、自然选词带来的后果收益和完整探索待验收。DPM/APL合并分支的急性作用不能单独证明DPM巩固功能。

详情与故障/恢复沿革见`FLY_NATIVE_MOTOR_PORT_20261018.md`；完整数据见`results/published/fly_mechanism_runtime_motor_32904_20261018.json`。
