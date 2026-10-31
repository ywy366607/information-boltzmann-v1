# 当前机制候选：KC位点资源与窗口1

候选 `E:/fly_stream_overflow/native_kc_site_stp_20261018/last.pt`，33000，新增真实输入/优化器更新均为0；后续BPTT窗口1词元。主线32384暂停。

| 系统 | 注释细胞 | 当前专用开关 | 本次实际因果检查 |
|---|---:|---|---|
| sensory_surfaces | 15896 | early_visual_graded=开，sensory_precision=关 | 本次未重测 |
| olfactory_relay | 3783 | local_relay=开 | 本次未重测 |
| mushroom_body_memory | 4501 | kc_mbon_memory=开，kc_dan_feedback=开，dpm_apl_axonal_projection=开，apl_local_cable=开，kc_local_terminals=开，kc_site_resource_dynamics=开 | kc_site_stp: 预测log概率差0.000837326 |
| neuromodulatory_state | 389 | dopamine=开，oa_5ht_extension=开 | 本次未重测 |
| central_complex_state_action | 2950 | cx_extension=开 | 本次未重测 |
| association_and_context | 121550 | 通用图动力学 | 本次未重测 |
| ascending_body_feedback | 21362 | historical_body_state_retained=开，historical_proxy_diagnostic_only=关，native_peripheral_exchange=关 | 本次未重测 |
| descending_and_vnc_motor | 2227 | historical_body_state_retained=开，historical_proxy_diagnostic_only=关，native_peripheral_exchange=关，native_motor_language_port=开 | 本次未重测 |
| visual_and_long_range_projection | 99154 | early_visual_graded=开 | 本次未重测 |
| endocrine_and_ens | 141 | lk_receptor_exchange=开 | 本次未重测 |

这是一套统一入口的针对性4支检查：intact、sham、kc_site_stp关断、restored，
使用当前完整存档、同RNG和一个真实19拍间隔。全部原生系统仍参与前向，
本次只干预新增资源机制；其他系统不能借用旧候选报告冒充本存档的新因果验收。

结果：资源最低0.099811，利用率最高0.656225；关断资源机制的φ最大差0.00033766，
预测log概率最大差0.00083733。此段没有动作提交，动作读出logit差0，未观察到实际选词/时机收益。
sham与恢复控制及来源/RNG保持通过；完整训练器恢复通过。

窗口1的数值更新/重算、完整资源历史保存、守恒/恢复/局部抑制及材料梯度在226项相关检查中通过。
独立Deslice通过。全量入口仍有5项旧诊断文件缺失，未称全套通过。

冻结完整支19拍重放约21.50秒，CUDA张量分配峰值2194.14MiB；4支最大2495.16MiB。
这是前向与配对状态的分配峰值，不是总进程VRAM，也不是实际BPTT学习峰值。
新增资源状态约5.41MiB。完整4.18M节点运输仍有速度成本，统一执行优化后置。

最初18支全矩阵因主存提交余量约600MiB，由Ling停止本只读进程；它未完成，不能称通过。
随后在完整恢复之后释放冻结前向不使用的训练副本，保留实际脑/权重/时钟并完成本4支检查。
原始完整存档中的优化器、信用和世界队列未删除；训练器本身的完整恢复路径未简化。

当前是实际位置上的资源模型，分子priming、钙通道/轴突电位波、每位点φ、局部电耦合、
慢参数完整跨窗信用及持续学习收益仍列缺口。无新训练NLL或母女对话，不改教法、不升级阶段。

详细机制：`FLY_NATIVE_KC_SITE_STP_20261018.md`；完整性：`FLY_NATIVE_COMPLETENESS_20261018.md`。
