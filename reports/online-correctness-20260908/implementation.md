# 实施与验收记录

对应规格：docs/sanma-online-correctness-spec-20260908.md。
范围：A/B 修复及 C 的独立配置、dry-run；没有启动正式 10k 训练，也没有晋级、提交或推送。

## A：统计与历史复算

- runner 复用共享评分解析，显式匹配 challenger，不再猜测非 baseline 的第一个名字。
- 检查完整终局、名字、分数/增量、文件与牌谱 seed 一致性、三次轮座及重复种子。
- 合并使用原始种子均值；中途面板只观察，固定 2,997 局终点判定。
- 八个历史面板共 7,992 局全部复算，顺位计数分别与原生控制台输出一致。
- phase1.json / phase2.json 保留种子记录和输入文件哈希；manifest.json 保留配置文本、原始结果哈希和计算代码哈希。
- 两轮仍未显著超过冠军。1.81 被标记为项目验收线，不解释成非劣效证明。原始结果及模型未覆写。
- 历史候选在评估当时的模型哈希没有记录，报告明确标为未知，没有拿现在的 deployment 冒充。

## B：动态 Self

- opponents.source = 'current' 显式选择跟随参数的对手；默认 checkpoint，兼容旧固定对手配置。
- client 接收完整参数包后、下一批对局前同步实际 engine 的 Brain/DQN。
- 固定对手不更新；每批记录对手来源、参数身份、训练者版本和 seed 范围。
- smoke 开启 verify_self_sync，直接比较实际 engine 与输入参数哈希；加载失败时不开始新批次。

## 有界运行证据

正式模型未训练，以下输出均位于独立 .cache 子目录：

| 尝试 | 结果 |
|---|---|
| online-self-correctness-smoke-01 | 沙箱拒绝 MIOpen 用户缓存锁；未证明 GPU 训练通过。相关进程已退出。 |
| online-self-correctness-smoke-02 | GPU 更新与恢复启动成功，但先耗尽总计 60 局预算，按限制停止。 |
| online-self-correctness-smoke-03 | 20 个训练计数、19 次优化器更新，18 局；末尾检查遗漏 oracle=False，后经 --verify-only 复核现有产物通过。原始 result.json 保留，verification.json 为复核证据。 |
| online-self-correctness-smoke-04 | 最终有效更新计数验证通过：20 次实际 AdamW 更新、24 局、43.484 秒；真实从 10 恢复到 20，Brain/DQN 严格重载成功，6 份 drain 日志只产生 trainee 样本。 |

smoke 与正式配置的差别：batch=128、num_workers=0、submit_every=2、save_every=5、warmup=10、每批3局、compat workers=4、总20次更新。没有将这些值带入正式配置。每次 smoke 上限为60局、10分钟，无无限重试。

最后一次进程核查未发现本项目 Python 后台进程；没有清理 Hermes 或其他未知进程。

## 附加修正：AMP 与更新预算

smoke03 证明旧 steps 是训练迭代数，AMP 溢出会跳过 optimizer.step。因此新增显式 control.count_optimizer_updates：

- 默认 false，保留旧配置的计数行为。
- 新独立实验为 true，且要求 opt_step_every=1。
- 跳过的更新不增加 steps、不推进 scheduler、不污染已成功更新的统计窗口。
- 恢复时拒绝混用两种计数语义；旧模型可通过 init_from 热启动新状态，不直接接续旧计数状态。
- smoke04 中确有一次 AMP 跳步，但最终 optimizer state 仍达到20，证明预算不是仅按日志步数判断。

## C：配置已准备，尚未启动

配置：Mortal/config/sanma-online-128x8-dynamic-self.toml。
起点为封存128x8冠军，GRP冻结、expert_ratio=0.5、LR=3e-5、batch=1024、网络128x8；预算为10,000次成功优化器更新。

输出隔离于 Mortal/training/online_dynamic_self_20260908/；评估目录和配置使用独立前缀。共享只读索引的69,213条记录与配置 globs 完全相同，无重复。

dry-run 输出：dynamic-self-dry-run.json，含所有路径、输入模型哈希、数据索引哈希、有效配置及预留 seed keys 20260980–20260983。这些数值是随机种子标签，不是日历日期。正式运行前仍会重新检查冲突。

从项目根目录执行只读预检：

    rtk proxy .venv-rocm/Scripts/python.exe scripts/run_online_phase1_overnight.py --config Mortal/config/sanma-online-128x8-dynamic-self.toml --dry-run --seed-base 20260980

只有获得用户单独的正式训练授权后，才可去掉 --dry-run。该入口现在要求显式 --config，并拒绝非空的初始输出；不再通过旧的 --result/--log 参数改变输出。一个运行内部仍按保存状态重启 trainer；本次没有新增跨会话的自动 resume 入口。

在 Codex 沙箱中，ROCm 可能因用户缓存锁写权限而失败；遇到确切权限错误时请求受控提升，不改全局缓存权限，也不用 CPU 冒充 GPU 验证。

## 检查入口

    rtk proxy .venv-rocm/Scripts/python.exe checks/online_panel_stats_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/online_self_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/online_runner_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/optimizer_update_budget_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/online_state_resume_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/scheduler_resume_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/three_way_paired_check.py --self-check

有界 GPU 检查必须指定一个未存在的 .cache 子目录：

    rtk proxy .venv-rocm/Scripts/python.exe checks/online_self_smoke.py --output .cache/online-self-smoke-new-run

不要调用旧 scripts/run_online_smoke.ps1：它仍有全局 Python 进程清理逻辑，本次新入口不使用它。

Rust 编码器及 ABI 未修改，本次没有重新编译 Rust 或重跑 native parity；这不构成新的原生验收声明。
独立代码审查尚未完成：审查子任务因模型使用额度限制退出，没有返回完整审查结论。此验收门保持未通过；上述 smoke 不代表模型棋力已经提升。

## 2026-09-08 交付前复验

- 最新代码重新运行上述7项回归检查，全部退出0；14个相关Python文件编译检查通过，git diff --check通过（仅换行符提醒）。
- 重新执行正式配置dry-run：10,000次成功更新，69,213条索引与globs一致，4个评估seed key检查通过。
- 对比已存dry-run清单，全部输入模型SHA256未变；native pyd哈希仍与spec一致。
- 本轮读取并核对smoke04的passed结果，未重复GPU训练。正式输出目录尚不存在。
- 只读进程查询未发现命令行包含本项目路径的Python进程；没有执行任何进程终止操作。
- 未提交、推送、晋级或启动C阶段正式训练。剩余门槛：完成独立只读审查，以及用户明确授权正式10k实验。

## 2026-09-09 独立审查重试

- 独立只读审查任务已重新派发；在请求进度和中断要求返回已有结果后仍无报告，已关闭子任务，未保留后台等待。
- 本次没有获得可核实的审查覆盖范围或结论，因此不能将独立审查标记为通过。此次无响应的原因未查明，不等同于此前的额度错误。
- 主控只读核对历史manifest的13项哈希（3份复算源码、2份原始结果、8份评估配置），全部一致；HEAD仍为188d8bcf09f822e6b406720d95d43f5730632abc。
- 本轮没有改动训练代码、运行GPU任务或启动C阶段。下一步仍需有效的独立审查结果。

## 2026-09-09 主控替代复核（用户批准的验收例外）

用户明确允许由主控逐项代码复核和回归验证替代本次独立审查。本节取代此前“仍需独立审查”的阻塞说明，但不能称为独立审查通过；不包含正式训练授权。

### 覆盖范围与结论

- 统计：复核online_panel_stats.py、arena_log_results.py及调用方。显式challenger严格匹配；三轮座位/种子身份、完整日志预算、分数与增量、供托、同分规则有检查。种子均值直接合并，SE使用样本标准差除以sqrt(N)，N=1不推断；正式runner不因中途显著而提前结束。旧调用保留默认行为，1.81仍仅为policy threshold。
- Self：复核client.py -> TrainPlayer.sync_opponents -> CompatMjaiEngine.delegate -> _BatchContext推理路径。更新实际Brain/DQN，不是替换无效外层引用；固定对手不更新，动态对手未同步时拒绝开局；同步发生在两批对局之间，会话UUID区分重启版本。
- 更新与恢复：复核train.py和online_training.py。init_from只初始化权重，新编号从0起；成功更新才推进steps/scheduler，恢复拒绝切换编号语义；在线loader限定trainee。AMP跳步仍会消耗输入批次，因此expert_ratio是批次混合比例，不保证成功更新来源精确各半。
- 入口与隔离：复核describe_run/main/run_trainer/spawn/run_panel及独立TOML。预检拒绝既有输出及seed冲突；新旧配置逐字段对比只涉及输出路径、端口、动态Self和成功更新计数开关，训练超参数不变。正式预算、评估预算及不自动晋级边界保留。
- 证据：复核历史重算manifest和smoke04记录；未将既有GPU smoke说成本轮重新训练。原生编码器/ABI未修改。

### 发现并修复

1. 预检错误通过全局log()写入历史online_phase1.log。入口异常改为只输出stderr。用临时目录中的真实入口和缺失配置复现；修复前断言失败，修复后无日志写入。
2. 子进程已启动、owned_processes.jsonl写入失败时，spawn()未返回，进程尚未加入外层清理列表。现在在该OSError分支终止并等待回收已创建的进程。故障注入检查修复前失败、修复后通过；此检查使用进程替身，未模拟真实磁盘写满或执行真实taskkill。

本轮修改仅scripts/run_online_phase1_overnight.py、checks/online_runner_check.py及本报告，没有修改网络、优化器或模型产物。

### 最新验收

- 自动化：7项检查全部退出0，包括新增两个异常路径回归。
- 静态：14个相关Python文件py_compile通过，git diff --check通过（仅LF/CRLF提醒）。
- dry-run：10,000次成功更新；69,213条索引与globs一致；seed keys 20260980–20260983检查通过。
- 保护：全部输入模型哈希及历史manifest的13项哈希保持一致，正式输出目录未创建。
- 未执行：新的GPU smoke、正式训练/对局面板、Rust构建、提交、推送或晋级。

在上述已复核范围内未发现剩余阻塞项。A/B及C准备工作可按用户批准的替代验收方式交付；下一步只待用户明确授权启动正式10k GPU实验及其固定预算评估。
