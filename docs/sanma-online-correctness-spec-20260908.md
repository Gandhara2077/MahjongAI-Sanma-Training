# 三麻在线训练正确性修复与动态 Self 实验 Spec

日期：2026-09-08。状态：待实施，供后续实现者执行。
项目：本仓库（路径无关，均使用仓库相对路径）。
编写时分支：stage/04-training；最近提交：188d8bc。

## 0. 执行入口与授权

先读取本文件，再读取适用的 AGENTS.md 和当前相关代码。用户本轮仅要求生成 spec；收到本文不等于获得修改、训练或发布授权。

- 用户明确要求按 spec 实施后，可执行 A/B 阶段代码修复、回归检查、旧日志重算及本文规定的有界 smoke。
- C 阶段的 10k GPU 训练及正式模拟面板需用户明确批准启动；不得因 A/B 通过自动启动。
- 不授权提交、推送、删除数据、晋级模型或覆盖 baselines/。
- 按 L2 管理实现：根因定位、失败检查、最小修改、验证、独立只读审查。
- 本文规定行为和验收，不要求照搬函数名或引入新框架。复用已有实现，不新建第二套统计系统。

目标：修正在线评估统计，使 self 对手真正随参数版本更新，生成隔离实验配置及可审计证据。不是继续堆训练步数或重写 Rust 编码器。

## 1. 已知现状与证据等级

接手时重新核对可变状态；不要以旧交接覆盖新事实。

### 1.1 Native v4 已上线

- Mortal/config/sanma.toml 使用 observation_encoder = 'native_v4'。
- Mortal/mortal/libriichi.pyd 本次实测 SHA256：6117493924c3a18d4291f9e7423688228ca4cd33f73b78f418eae96fdb86fd3d。
- reports/native-v4-rollout-20260906.md 记录 17,934 samples 的 obs/mask/metadata parity 零差异及 training smoke 通过。本次没有重跑这些测试。
- 报告中的 8.21x 是固定文件数据管线 benchmark，不是整体训练提速。
- 本任务不修改观测 ABI、Rust 计分或生产扩展，不恢复旧 pyd。

### 1.2 历史结果复算

原始证据：online_phase1_result.json、online_phase2_result.json、两份同名 log，以及 online_phase1_logs/、online_phase2_logs/。
牌谱位于 Mortal/training/sanma_1v2_online_p1_*，八个面板目录，每个 999 局。

下表为上一分析轮复用 checks/arena_log_results.py 的分数解析、显式匹配 challenger、按种子三次轮座聚合的只读复算。未写回旧报告，尚未完成全部 native summary 交叉核验。

| 阶段 | 对手两席 | 局数 | 种子数 | 平均顺位 | 95% CI |
|---|---|---:|---:|---:|---|
| Phase 1 / 10k | 128x8 冠军 | 2997 | 999 | 2.004671 | [1.983961, 2.025381] |
| Phase 2 / 20k | 128x8 冠军 | 2997 | 999 | 1.997664 | [1.975046, 2.020283] |
| Phase 1 / 10k | 原始基线 | 999 | 333 | 1.801802 | [1.755877, 1.847727] |
| Phase 2 / 20k | 原始基线 | 999 | 333 | 1.828829 | [1.785281, 1.872377] |

与旧 runner 相比，85 局顺位改变。两轮均未证明超过冠军；不能由不同面板的点估计宣称 Phase 2 相对 Phase 1 显著退步。

### 1.3 已定位偏差

1. scripts/run_online_phase1_overnight.py 的 paired_stats 重写计分，漏处理 reach_accepted 扣款及终局剩余供托。
2. paired_stats_pooled 使用 sqrt(sum(se_stage^2 * n_stage) / N)，不是正确的合并均值标准误；不得继续由已舍入 CI 反推。
3. sanma-online-128x8.toml 的 deployed-self 与 anchor-128x8-champion 指向同一固定冠军。player.py 初始化时加载对手；client.py 后续只更新训练者。
4. 客户端会话选择次数：P1 baseline=175、anchor=287、self=440；P2 baseline=168、anchor=283、self=431。约 81% 会话选固定冠军。选择次数不等于成功入库样本数。
5. mean_rank <= 1.81 是历史项目验收阈值，不是统计非劣效结论。

## 2. 资产保护与非目标

- 不覆盖或删除 baselines/，尤其 sanma_champion_128x8_46500_20260831.pth、sanma_baseline_mortal3p_original.pth、sanma_grp_10k_last_frozen.pth。
- 保留 online_128x8 的 state/deployment/best/snapshots、旧评估配置及原始结果。不要假定 10k 独立模型仍存在，先盘点。
- 不清理未跟踪文件，不杀未知进程；只能清理本次创建且身份已核实的进程树。
- 不同时改 LR、expert_ratio、GRP、网络大小、奖励、探索参数或数据窗口。
- 不新增完整历史联盟、淘汰系统、缓存服务、数据库或新调度平台。
- 不把当前训练称为纯行为克隆：train.py 使用动作 Q 回归，CQL 仅作用于 expert 批次，reward_calculator.py 已接入真实终局排名。新瓶颈判断需要新证据。

## 3. 接手检查

- [ ] 记录 git status、分支、HEAD、相关差异，保留用户修改。
- [ ] 盘点历史面板、summary、模型与当前任务。无法检查进程时报告限制，不声称后台已停止。
- [ ] 阅读 runner、player.py、client.py、online_training.py、train.py 的调用链。
- [ ] 阅读 checks/arena_log_results.py、arena_log_paired.py、three_way_paired_check.py，复用解析和验证。
- [ ] 新输出目录若已存在且非空则拒绝覆盖。

命令通过 rtk；PowerShell 内建命令通过 rtk proxy powershell -NoProfile 调用。Python 优先用项目 .venv-rocm/Scripts/python.exe；训练脚本从 Mortal 工作目录运行，显式指定 MORTAL_CFG。

## 4. A 阶段：评估正确性与历史重算

### A1. 统一评分解析

复用已修正的最终分数解析，不在 runner 新写事件计分循环。扩展共享接口时保持旧调用方兼容，并检查所有调用方。

- challenger 按明确名称唯一匹配，不能猜测第一个非 mortal-baseline 名称；两个冠军席同名合法。
- 验证 start_game、完整 end_game、三席、有效分数及唯一 challenger。
- 正确处理立直棒、局间绝对分数、和牌/流局增量、终局供托和同分座次。
- 缺失、损坏、重复、不完整日志使正式面板无效，不得静默跳过后发布通过结论。
- 如有 native summary，核对 challenger 的三种顺位计数和总局数，不仅核对均值。缺 summary 时记录缺失，不伪造验收。

### A2. 种子级记录与统计

每局保存：phase/panel ID、seed key、seed number、轮座标识、challenger 座位、最终三席分数、challenger 顺位、源日志相对路径。
每个种子恰好 a/b/c 三局，challenger 遍历三席。以完整随机种子身份去重；不同 seed key 下同数字不合并；同一实际种子重复导入报错。

对第 i 个种子的三次顺位：

    x_i = (r_i1 + r_i2 + r_i3) / 3
    mean_rank = sum(x_i) / N
    variance = sum((x_i - mean_rank)^2) / (N - 1)
    standard_error = sqrt(variance / N)
    ci95 = mean_rank ± 1.96 * standard_error

这是种子聚类的正态近似区间，不是精确检验。N < 2 时拒绝推断。跨阶段直接合并 x_i；不得平均区间或从舍入汇总值恢复方差。保存未舍入值，展示时再舍入。

区分两种统计：单候选对两席冠军的 1v2 为轮座聚类评估；两个候选在同一批 seed 上各自对同一对手评估后的差异才是候选间配对比较。

### A3. 正式判据

- 新实验冠军面板固定 2997 局 / 999 seeds；999/1998 中间值仅作观察，不提前晋级。
- 固定终点 ci95 上界 < 2.0 表示该面板支持优于两席冠军。未达标表示证据不足，不是证明两者相等。
- 记录 effect = mean_rank - 2.0，不仅输出布尔值。
- 保留 1.81 历史验收线，明确标记 policy threshold，不为通过而改值。
- 若声称相对冠军无基线回退，需要候选与冠军在共享基线面板上的种子差异和预先指定非劣效界值。界值未经用户确定则只报告差异/区间，标记未判定，不自选。
- 筛选多个 checkpoint 后，用未参与筛选的新面板正式确认。
- 任何统计通过均不自动复制模型到 baselines/。

### A4. 回归检查与交付

新增或复用一个可直接运行的集中检查，覆盖立直影响顺位、终局供托、同分座次、名称匹配、缺日志、重复轮座、跨 seed key、零/一个种子、summary 不一致与 pooled 计算。

具体合并回归输入：两个阶段分别 [1,2,3] 和 [1,2,3]；合并 N=6、mean=2、variance=0.8、standard_error=sqrt(0.8/6)。检查直接合并与统计函数一致。

- [ ] 先证明旧逻辑被检查抓住，再修复并通过。
- [ ] 重算八个面板，共 7992 局；核对第 1.2 节数值。若不同，定位具体日志，不硬编码预期输出。
- [ ] 新建 reports/online-correctness-20260908/，保存 corrected JSON、种子记录、manifest 和报告。
- [ ] manifest 记录面板配置、源日志身份、统计方法、代码版本、可获得的实际输入模型哈希。原模型不可追溯时明确标记，不用当前模型哈希冒充。
- [ ] 原始两份 result JSON 保持不变。

## 5. B 阶段：动态 Self 最小实现

### B1. 更新契约

在既有 opponents 配置增加最小的显式来源标记，区分固定 checkpoint 与当前参数版本 self；旧配置默认固定。不得根据名字含 self 推断行为。

- baseline/anchor 保持启动权重，不随参数更新变化。
- client 收到完整参数包后、下一批对局前，将同一版本 mortal 与 dqn 同步到 self；批次中途不更新。
- 保持 self 与训练者探索配置独立，沿用各自现有推理/探索设置。
- 复用 client 收到的参数，不持续读取可能正在覆写的 deployment 文件。
- 检查 compat wrapper 的模型复制/缓存，证明实际参加对局的 engine 收到权重，而非仅修改外层字段。
- 更新失败或结构/版本不完整时停止该批，不能静默退回固定冠军却标记 self。
- 每批记录实际对手、来源、完整参数版本身份、训练者版本、局数和 seed 范围；重启可能重号，版本需组合会话身份或内容哈希。
- 保持 baseline/anchor/self 权重 0.2/0.3/0.5、weighted 选择方式不变。

### B2. 验证及有界 smoke

最小 fake engine 回归证明：两个不同版本传播到 self 的两个网络；固定对手不变；批次边界同步；旧固定配置仍可用。
至少一次真实引擎检查实际参数指纹/版本；不能只看日志文本，也不能要求不同权重必然产生不同动作。

实施授权下 smoke 的上限：独立输出、最多 50 updates、最多 60 局、最多 10 分钟，任一上限先到即停止。允许仅为覆盖两次更新缩短 smoke 的 submit/test 周期，记录差异；正式配置不得继承此缩短值。

保持 val_loss_every=0，限制 smoke 的 worker/BLAS 数。不得用 CPU 训练替代 GPU smoke 并声称验证 ROCm；GPU 不可用则报告未验证。

验收：至少两个完整参数版本进入真实 self engine；固定锚点未变；在线训练只取 trainee 样本；保存/恢复正常；无本次遗留进程。限额内未覆盖即标记不充分，不无限重跑。

## 6. C 阶段：批准后独立实验

先生成配置和启动说明，不直接启动。配置建议：Mortal/config/sanma-online-128x8-dynamic-self.toml。

- 从封存 128x8 冠军启动，而非 Phase 2 的 20k 状态。
- 网络 128x8；冻结 GRP 为 sanma_grp_10k_last_frozen.pth；expert_ratio=0.5；peak LR=3e-5；warmup=1000；final LR=1e-5；batch=1024。
- 预算为起点之后 10000 个实际 optimizer updates。核验 control/scheduler 编号语义，不能因离线 checkpoint 的 46500 steps 跳过预算。
- 其余数据、增强、CQL、奖励、探索设置不变，只变 self 刷新行为。
- state/deployment/best/snapshot/tensorboard/buffer/drain/train_play/test_play/正式评估全部用独立路径。共享只读索引前验证其文件清单与配置一致。
- runner 不得通过常量仍写旧 online_128x8 路径；接受显式配置并解析实际状态/输出。已有入口可满足则复用。
- dry-run 列出起点与哈希、有效配置、更新预算、种子计划、所有输出路径；路径冲突拒绝执行。
- 冠军面板固定 2997 局，基线检查固定 999 局；预先选择未使用的 seed key 并保存。
- 历史静态两轮仅作背景。若归因于动态 self，另行批准同起点/同预算静态对照，并用共享评估种子比较；不要把历史不同面板点估计当因果证明。
- 达预算即停止，不自动启动下一 phase 或晋级。

## 7. 验证入口与修改范围

以下为现有检查入口，执行前核对当前参数和工作目录；列出不表示已经运行：

    rtk proxy .venv-rocm/Scripts/python.exe checks/online_state_resume_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/scheduler_resume_check.py
    rtk proxy .venv-rocm/Scripts/python.exe checks/three_way_paired_check.py --self-check

新增检查提供同样直接的命令；对实际修改的 Python 文件运行 py_compile。不涉及编码行为则不重编 Rust。

预计修改：scripts/run_online_phase1_overnight.py、checks/arena_log_results.py（仅共享接口确有需要）、既有统计检查、Mortal/mortal/player.py、Mortal/mortal/client.py、新配置和报告。其他文件需说明其调用链必要性，不无关重构。

## 8. 完成标准

- [ ] A：统一评分、种子统计、完整性检查和旧日志校正完成。
- [ ] B：真实 self 更新通过回归及有界 smoke，固定锚点不变。
- [ ] 独立审查覆盖统计公式、旧调用兼容、engine 更新、恢复编号和输出隔离；发现项修复或明确列为阻塞。
- [ ] C：独立配置和 dry-run 就绪，未启动；如获另行授权，单独报告训练与正式评估结果。
- [ ] 交接分别报告静态检查、自动化检查、真实运行验证、未运行项、修改文件及待授权事项。
- [ ] 不以配置存在声称 smoke 成功，不以退出码 0 声称模型提升，不以未显著更差声称非劣效。

## 给执行模型的指令

先只读核对本文与当前仓库。获得实施授权后按 A → B 最小修复并验证，保留旧模型和日志，生成 C 的配置和 dry-run 后停止。无单独训练授权不启动 10k；无发布授权不提交、推送、晋级。证据与本文不一致时定位并记录，不修改数据迎合本文数值。
