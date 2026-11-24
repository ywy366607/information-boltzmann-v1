# 梦漓训练终端网页实查（2026-11-14）

## 2026-11-15：扩展身体已接入现有网页

本次修复的是运行绑定：网页观察配置仍选择旧5356轮小身体，但共享生命卡已由32层扩展身体更新，触发了正确的`Life card does not match selected body`拒绝。已原子更新`data/harness/workbench/training-observer.json`，明确选择`results/cgb_grown_body_20261114/learning_v2`、`stage_10m/status.json`及原身份/妈妈分支；保留声明的历史日志段并追加扩展身体日志，目标15773轮。绑定旧件保存于`results/cgb_grown_body_20261114/web_attachment/previous-binding.json`。

不改模型、不加载检查点、不重启训练或妈妈服务。现有热读绑定和SSE自动恢复页面。实际浏览器验收：学习面板5533轮、目标15773、学习曲线；随后生命面板5536轮，分层明细L1—L32全部有效，仍为同一个体`mengli-851f47bb4a72b46c`。32层继续用原有三段汇总曲面及完整逐层读数，不增加32张重叠膜。

`test_mengli_training_observer.mjs`及`native-family-client/tests/life-core.test.mjs`共12项通过，覆盖身份拒绝与恢复、SSE、只读控制、分层汇总。实机来源与浏览器证据在`results/cgb_grown_body_20261114/web_attachment/verification.json`及`learning-live.jpg`。本次只改本地绑定与记录，没有新增前端实现；modern-web-guidance已搜索实时状态/SSE，未采用无关的样式或流式HTML改造。

## 已修复并部署：统一当前研究运行（同日后续）

`TrainingObserver`现经本地显式绑定接入生产FamilyRuntime。开场生命卡、family状态、生命/训练面板、SSE和梦漓页均读取当前研究身体；原家长—妈妈管理对话及其管理轨迹仍保留原始历史，不冒充当前照护事件。绑定文件`data/harness/workbench/training-observer.json`在Git忽略区，记录运行ID、身体身份/分支、当前输出、生命卡、守护状态和明确的连续日志段；不扫描目录发现个体，不加载权重/快表，不执行世界动作。

读取预算：最多8个显式日志段、各自尾部8MiB/128行缓存，合并后最多128点、64轮事件；文件未变复用缓存。每2秒仅对已打开的family SSE订阅者检查，推送只在状态改变时发生。完整JSON行且cycle不超过生命卡已保存边界才可显示；页面分列已完成事件与已保存检查点，两者允许短暂差一轮。回复显示为“妈妈 · 回应原文”，保留真实摄入的妈妈前缀字节数；不把整段回复或背景填充伪装成完整摄入收据。

控制权：观察模式的load/control/step/say/plan均在服务端拒绝（409），网页隐藏旧连接/接管入口，输入与计划编辑关闭。即使绑定损坏或身份不符，也进入观察不可用，不能退回旧3509身体。原工作台的计划只显示为未应用于当前研究的旧管理计划。外部研究队列仍是唯一身体拥有者。

部署按事件边界进行：原队列在4506轮完整停止，检查点SHA256 `4b58289b971cda45028b6317e9a98ce5c60d60acea78acc09f77b8068922d823`与原分支生命卡一致。重启原妈妈/工作台服务后，从4506续214轮至原4720目标；原4506源、实际会话历史和Adam沿用。新守护状态`results/cgb_shared_unknown_20261114/continued_after_web_binding/status.json`，实际身体在其`body/last.pt.gz`；不从旧3509或4208重新运行。

验收：64项整套接口/客户端/原生视图回归通过；新增进度元素及最后客户端检查20项通过（与前者有重叠，不相加作独立项目数）。实际8902三端口投影一致，128曲线点、64轮原话，控制请求409；SSE实收4511→4512连续推进，随后HTTP核对4513、running、worker PID3964。浏览器工具当前被Work设置禁用，未取得修复后浏览器截图，未声称完成新版视觉/移动端验收；这些是实际HTTP、SSE和原生组件树检查。旧截图仅代表修复前。

modern-web-guidance的progress建议用于有名称的原生`<progress>`，保留默认样式及不支持accent-color时的原生回退，不把每个训练事件做成抢焦点播报。侧栏搜索标签和热更新恢复仍是独立的小项，未在这次核心接线中改写。

## 结论与下一项

网页当前是旧3509身体的管理工作台，研究队列正在推进共同未知升级后的身体，两者未接入同一个运行状态源。应先统一运行归属、只读状态和控制权，再优化图表与样式。不能通过点击旧的“连接身体”解决：现有`FamilyRuntime.load`会从自身保存的旧检查点启动BodyWorker，不会附着到研究队列的在跑进程。

## 已核对的证据

- 浏览器实际打开8900，完成开场进入、学习训练、梦漓原话、管理轨迹、家庭照护菜单及Escape关闭。1280×720下文档宽高均与视口一致，无整页横向溢出；页面语言为zh-CN。
- 8902 `/health`为ready。`/api/family`为cycle3509、mode=paused、loaded=false；绑定`results/cgb_mother_dsh_20261113/body_resumed/last.pt.gz`。
- `/api/family/telemetry`仍读取该旧目录相邻metrics，只返回1条3509记录；学习面板显示本轮4次更新、response_nll=1.4693、“仅1条记录，暂无趋势”。这是旧数据源的准确投影，不是当前训练只学了一轮。
- 首次对照时研究队列`results/cgb_shared_unknown_20261114/pipeline_512_service_restored`已有27条learning记录，最新4237；与网页3509不同。后续训练仍在推进，以该队列实际metrics及guard状态为准。
- 梦漓页从旧绑定妈妈历史展示16条消息/8对交流；管理轨迹页展示的是家长—妈妈管理agent的3轮/6步，不是这次身体续训的事件流。
- 可访问性小项：侧栏“搜索会话名称”输入没有关联label/aria-label/aria-labelledby，AX树也没有名称。应补持久标签。图表已有文字数值和原始表格入口，继续保留。

## 回归与运行影响

`node --test --test-concurrency=1 --test-timeout=60000 tests/test_mengli_readmodel.mjs tests/test_mengli_workbench.mjs`：17项通过，0失败。它们验证显式绑定与工具合同，未覆盖网页和外部研究训练进程的共同运行归属。

本次回归中的前端重建触发了共享客户端资源热更新，观察到`ui-conversation: conversation service unavailable`白屏。完整刷新后工作台、历史和面板恢复。这是本次检查触发的热更新恢复问题，不记作首次打开就已失败。没有重启妈妈桥、修改身体生产源码、加载旧身体或改变在跑训练。

检查截图：`results/mengli_web_audit_20261114/workbench.png`。

## 修复顺序

1. 以显式声明的当前研究运行建立只读观察端口，携带运行ID、身体身份、最新完成事件、检查点保存边界与数据时间；不靠扫描目录猜哪个个体最新。
2. 开场、生命体征、学习曲线、孩子—妈妈实际对话统一读取该端口；区分最新完成事件与已保存检查点，不能把两者混成“当前轮数”。
3. 网页现有控制器若不拥有训练进程，连接/照护/接管须明确禁止再启动另一身体；以后需要控制时，接原队列事件边界控制协议。
4. 补侧栏搜索的可访问名称，并隔离测试构建与线上共享客户端热更新。当前未改应用源代码。

使用modern-web-guidance查阅accessibility与responsive-design；本次仅检查，没有据此改写网页代码。手机版与完整屏幕阅读器验收未开展。
