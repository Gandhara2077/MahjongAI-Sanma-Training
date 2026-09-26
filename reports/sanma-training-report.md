# 三麻模型训练报告

生成时间：2026-08-29（Asia/Shanghai）

## 结论

已完成一轮可复现的三麻训练：使用用户提供的三麻 Mortal 权重初始化，在 AMD Radeon RX 9070 XT 的 ROCm PyTorch 环境下完成 1000 steps 主模型训练，并通过三麻规则、数据、ABI、模型加载和 1v2 运行验收。

这是一版可运行的训练候选模型，不是经过大规模牌谱和长周期训练的最终强力模型。当前数据只有 1802 局，主模型只有 1000 steps；后续提升强度的第一优先级是扩大数据窗口并增加训练步数。

## 环境与模型契约

| 项目 | 实际值 |
| --- | --- |
| Python | 3.12（`.venv-rocm`） |
| PyTorch | 2.9.1+rocm7.2.1 |
| GPU | AMD Radeon RX 9070 XT |
| 训练设备 | `cuda:0`（ROCm 使用 PyTorch 的 CUDA 兼容接口） |
| Sanma 原生 ABI | v5，观测 `(780, 34)`，3 玩家 |
| 部署 ABI | v4，观测 `(775, 34)`，动作空间 44 |
| 网络 | 32 channels，2 ResNet blocks |
| 初始化权重 | `.cache/models/mortal3p.pth` |

初始化权重 SHA-256：

`7b77cab4cd9782f48b0a8538b264840e5f5d20f9a8469914cdb52b7d0912f384`

## 数据集

来源为 Tenhou 官方三麻牌谱，日期窗口为 2026-07-01 至 2026-07-07（含首尾）。

| 数据 | 文件数 | 字节数 | SHA-256 |
| --- | ---: | ---: | --- |
| 原始 `.mjlog` | 1802 | 22,360,331 | `b6ae07a3934aaa034b3dd6b45005cd0fc550fdd7c3fa1741cd8b7de4308d6431` |
| 转换 `.mjson` | 1802 | 4,854,818 | `33d373559b8b9a096fe2dc46421a0c9a11ebacc20b0aff3f57e082a872e0ad18` |

GRP 训练使用 7 月 1 日至 6 日作为训练集、7 月 7 日作为验证集：

- 训练牌谱：1565 局
- 验证牌谱：237 局
- GRP 验证样本：2170
- 主模型：使用全部 1802 局，2 epochs，启用数据增强

数据文件本身没有提交到 Git；可复现清单在 [`data/manifests/sanma-20260701-20260707.json`](../data/manifests/sanma-20260701-20260707.json)。

## GRP 辅助网络

配置：500 steps，batch size 128，训练集/验证集按上述日期划分，最佳模型选择指标为 `expected_pt_mse`。

| step | train loss | train accuracy | val NLL | val accuracy | expected_pt_mse |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.78588 | 17.52% | 1.77923 | 22.67% | 5.94438 |
| 200 | 1.77517 | 21.06% | 1.77058 | 24.84% | 5.90026 |
| 300 | 1.76858 | 23.62% | 1.76151 | 25.25% | 5.85451 |
| 400 | 1.76213 | 22.93% | 1.75355 | 27.19% | 5.80919 |
| 500 | 1.74890 | 27.25% | 1.74375 | 28.25% | 5.75970 |

GRP 产物位于 `Mortal/training/`：

- `sanma_grp.pth`
- `sanma_best_grp.pth`
- `sanma_grp_metrics.json`

## 主模型训练

使用配置 [`Mortal/config/sanma-training.toml`](../Mortal/config/sanma-training.toml)：

- 从 `.cache/models/mortal3p.pth` 初始化 Brain/DQN
- batch size 512
- 1000 steps
- 2 epochs
- 启用数据增强
- 学习率从 `1e-6` warm up，最终到 `3e-7`
- 使用已训练的三麻 GRP
- ROCm GPU 自动混合精度

最终产物：

- `Mortal/training/sanma_mortal3p.pth`：部署模型
- `Mortal/training/sanma_train_state.pth`：可恢复训练状态，step 1000
- `Mortal/training/sanma_snapshots/mortal_step500.pth`
- `Mortal/training/sanma_snapshots/mortal_step1000.pth`

## 1v2 评测

配置为 300 局三麻 hanchan：10 个种子，每个种子让训练模型分别坐一次 0/1/2 号位，因此共 30 局。对手为原始三麻基模 `mortal3p.pth` 的两个副本。

评测程序原始输出：

```text
challenger rankings: [14  9  7] (1.7666666666666666, 21.0pt)
```

解释：

- 第 1 名：14 局
- 第 2 名：9 局
- 第 3 名：7 局
- 平均名次：1.7667
- 按当前配置的显示点数计算：21.0pt

完整压缩牌局日志在 `Mortal/training/sanma_1v2/`，共 30 个 `.json.gz` 文件。这个样本量只能作为运行和方向性验收，不能作为统计显著的强度结论。

## 验收证据

以下检查均已实际运行：

- Rust 三麻原生测试：`31 passed; 0 failed; 9 ignored`
- `cargo fmt --check`：通过
- `git diff --check`：通过
- `SANMA_SETUP_OK`
- `NATIVE_SANMA_OK`
- `SANMA_ABI_OK`
- `SANMA_DATA_OK games=1802 raw_bytes=22360331 mjson_bytes=4854818`
- 最终模型 GPU 前向：`SANMA_FINAL_CHECK_OK`
- 最终前向输入 `(1, 775, 34)`，输出 `(1, 44)`，合法动作数 5，所有权重和合法 Q 值有限，非法动作均为负无穷
- 三麻 `1v2` 评测进程正常退出

三麻规则层已覆盖并回归验证拔北相关行为：拔北取消一发、对手拔北可抢和，以及拔北抢和不额外标记为抢杠。

## Git 状态

规则、配置和数据清单阶段提交已保留在 `stage/04-training` 分支：

`81c7bcb train: finalize sanma rules and offline config`

训练产物和原始牌谱按 `.gitignore` 规则不进入版本库；报告和数据 manifest 可以提交。

当前远程为 `https://github.com/JinxianRen/MahjongAI-Sanma-Training.git`。该 Private 仓库已自动创建，`stage/04-training` 已成功推送并设置跟踪 `origin/stage/04-training`。训练产物和原始牌谱仍按 `.gitignore` 规则保留在本地；代码、报告和数据 manifest 已进入远程版本库。

## 复现主训练

从仓库根目录进入 `Mortal`，使用 Python 3.12 ROCm 环境：

```powershell
$env:MORTAL_CFG = 'config/sanma-training.toml'
$env:MORTAL_PYTHON = 'D:\deepcode\objects\MahjongAI-Sanma-Training\.venv-rocm\Scripts\python.exe'
$env:PYO3_PYTHON = $env:MORTAL_PYTHON
$env:MIOPEN_USER_DB_PATH = 'D:\deepcode\objects\MahjongAI-Sanma-Training\.cache\miopen\db'
$env:MIOPEN_CUSTOM_CACHE_DIR = 'D:\deepcode\objects\MahjongAI-Sanma-Training\.cache\miopen\cache'

& $env:MORTAL_PYTHON mortal/train.py
```

原始牌谱和转换文件需要先按 `data/manifests/README.md` 的说明重新下载；模型权重需要放回 `.cache/models/mortal3p.pth`。

## 下一轮建议

1. 将 Tenhou 三麻数据扩展到至少数万局，并保持按完整牌局/日期划分训练集和验证集，避免同一牌局泄漏。
2. 将主训练从 1000 steps 提升到与数据规模匹配的长周期，再用固定种子、多批次 1v2 评测。
3. 只有在三麻长期基线稳定后，才接入 MahjongCopilot；它是部署/前端桥接层，不是训练必需组件。
