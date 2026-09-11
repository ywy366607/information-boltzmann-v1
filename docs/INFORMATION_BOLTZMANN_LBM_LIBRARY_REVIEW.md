# LBM代码库与下一轮优化依据（2026-09-11）

当前耗时不是已证明的架构上限。1024粒子64字节6.720秒是4次参数更新，约1.680秒/update、105ms/byte，仍然很慢，但不是6.7秒/token或/update。跨大模型比较必须记录同设备、tokenizer、batch/sequence和前后向范围；不能以增大update窗口人为压低更新频次当提速。

## 实际剩余成本

最终Slice候选1024粒子：64事件前向输运CPU区间2.766秒，前向碰撞.541秒，反向加更新2.801秒；区间嵌套且CPU提交/GPU等待混合，不是exclusive kernel FLOPs。前向每token仍16次drive、8次transport、4次Cayley，此外能量账目和同步仍未整段融合。继续只替换碰撞不是完整解决方案。

新增同设置关闭子步能量记账，64事件5.3768秒，约1.344秒/update，相比6.720减少20%短窗口耗时；并不表示可永久丢弃耗散测量，也没有消除阶段计时/观测开销。应尝试账目在GPU融合与批量归约，完整审计窗口与低开销常态记录分开，记录采样变化。

## 已查阅的代码库

1. XLB（Autodesk）：https://github.com/Autodesk/XLB 。2D/3D可微LBM，支持JAX、Warp等后端。直接核对 `xlb/operator/stepper/nse_stepper.py`：Warp后端在一个kernel内完成pull streaming、边界、宏观量、平衡态、collision和回写。最有价值的是整个时间步融合与局部固定形状数据流；不能假设所有backend和边界组合都支持我们所需的反向，必须单独验证。源码 https://github.com/Autodesk/XLB/blob/main/xlb/operator/stepper/nse_stepper.py 。
2. Lettuce：https://github.com/lettucecfd/lettuce 。PyTorch LBM，支持自动微分研究和原生CUDA扩展，最贴近现有张量栈。优先核对生成/融合碰撞迁移内核及具体原生路径的梯度支持，不能由PyTorch接口自动推断所有自定义kernel可反向。
3. lbmpy：当前 https://github.com/lssfau/lbmpy ，旧mabau镜像已归档并指向此地址；基于符号表达式生成CPU/GPU LBM代码，值得借鉴公共子表达式消除、矩变换和自动生成代码。不是现成PyTorch训练模块。工作论文题名为 lbmpy: Automatic code generation for efficient parallel lattice Boltzmann methods。
4. FluidX3D：https://github.com/ProjectPhysX/FluidX3D 。OpenCL高性能LBM，可参考内存布局与存储/计算权衡；不视为现成可微训练替换件。仓库标明非商业使用许可范围，若未来直接复用代码用于产品必须检查具体许可；本次只查看未复制代码。

## 下一优化目标

优先借鉴XLB整步融合和Lettuce张量接口，减少输运/驱动/记账及反向的调用数量。进一步排查Cayley solve的host同步与小矩阵求解成本、每事件loss/cross指标回CPU、recompute导致重复forward。固定形状后才评估CUDA Graph；先拆出跨token依赖的数据驱动和仪表，不能假定整流可并行。

时间离散步骤4→2→1属于数值近似候选，需要真实输入同权重的轨迹/能量/响应误差以及之后训练效果比较，不能改steps后只报加速。纯LBM擅长固定格点和有限速度人口，而当前是4D信息位置+4D连续速度与可学习MLP；直接替换会改变状态表示及动力学，不是安装库就获得同一个模型的加速。

本次只阅读库及新增运行对照，未安装新依赖、未采用外部代码、未宣称已经达到百毫秒/update。优先进行整步与反向融合的下一轮实现。
