# 在线自对弈训练方案（2026-09-06 grilling 共识）

## 已确认的七项决策

1. **奖励信号**：GRP 奖励 + 既有 BC/CQL 损失（Mortal online 方案），不做
   终局奖励 RL（ADR 0001）。
2. **对手池**：锚定池起步——原始基线 20%、128×8 冠军 30%、自己 50%；
   管线验证后滑向滚动联盟（定期把 best 快照入池）。
3. **起点模型**：128×8 46.5k 冠军（现任、推理最快、晋级判定干净）。
4. **数据配比**：expert_ratio = 0.5（专家批 : 在线批 = 1 : 1），按证据调节。
5. **节奏**：peak LR 3e-5，验证 phase 10,000 步；之后按"phase → 面板"
   循环滚动。
6. **晋级门**：阶梯面板 1,000 → 2,000 → 3,000 局（配对 CI 排除 0 即终止），
   晋级条件 = CI 上界 < 0 + 对基线 1,000 局轻量面板无回退。
7. **工程债先行**：修复 online 状态恢复的 step 回归 bug（静默错误，必须
   在第一个 phase 前堵死）。

## 执行顺序

### 第 0 步：修复 online 状态恢复 bug（约半天）
- 定位 `Mortal/mortal/automation_pipeline.py` 中 `effective_start_step` 的
  回归路径（Codex 审查发现：首次选择较早 offline snapshot 时恢复错误、
  可能超额训练）。
- 修复 + 新增定向 check（参照 `checks/scheduler_resume_check.py` 模式）：
  构造"从较早 snapshot 恢复"的场景，断言恢复后 step/调度器/数据游标
  一致。
- 顺带把本次离线训练暴露的稳定性问题固化：val 评估的中途 worker 孵化在
  本机不可靠（两次崩溃），在线管线设计上避免同类操作。

### 第 1 步：在线配置与冒烟（约半天）
- 新建 `Mortal/config/sanma-online-128x8.toml`：起点 = 128×8 冠军、
  expert_ratio 0.5、peak LR 3e-5、max_steps 10,000、对手池
  {baseline 20%, champion 30%, self 50%}、submit/save/snapshot 沿用现值。
- 冒烟：server + client 各起一次，确认对局生成 → drain → 训练 → 参数回
  提交的完整闭环，跑 ~200 步即停，不产出正式模型。

### 第 2 步：Phase 1（10,000 步，约 1-1.5 小时 + 面板）
- 全程监控对局生成吞吐与训练批速率。
- 结束后跑阶梯面板：对 128×8 冠军 1v2（新种子）+ 对基线 1,000 局。
- 产出三种结论之一：显著更强（晋级或续跑确认）、平局（续跑下一个
  phase 或调 expert_ratio/LR）、显著更弱（回滚并诊断）。

### 第 3 步：Phase 循环
- 每个 phase 结束过一次晋级门；连续两个 phase 无进展则停止并复盘
  （候选动作：滑向滚动联盟、调 expert_ratio、GRP 再训练——最后者需
  另行立项）。

## 明确不做

- 不做终局奖励 RL（ADR 0001）。
- 不再加大参数量（ADR 0002）。
- 不动评估/客户端部署路径。
- 阶梯面板未达显著前，任何在线 checkpoint 不得进入 `baselines/`。
