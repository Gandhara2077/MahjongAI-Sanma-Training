# 基线与本地模型输入

本目录保存本地训练和评测所需的 sanma 权重。`*.pth` 被 `.gitignore` 忽略，
这些文件不会随 Git 克隆或发布包提供；新克隆需要从有权限的模型存储中恢复，
再按文件名放置。

## 本地文件清单

| 文件 | 用途 |
| --- | --- |
| `sanma_baseline_mortal3p_original.pth` | sanma 默认初始化和参考评测模型；`Mortal/config/sanma-training.toml` 指向它 |
| `sanma_champion_lr2_40k_20260830.pth` | 本地历史评测快照 |
| `sanma_champion_128x8_46500_20260831.pth` | 本地历史评测快照 |
| `sanma_champion_192x12_best_92000.pth` | 本地历史评测快照 |
| `sanma_grp_10k_last_frozen.pth` | 冻结的 GRP 状态；不能当作主模型初始化权重 |

文件名中的历史命名不构成当前强度或晋级结论。若只运行默认 sanma 流程，至少
需要 `sanma_baseline_mortal3p_original.pth`。

yonma 配置另需 `baselines/baseline.pth`；它不属于上述 sanma 清单，也不随仓库
提供。外部 sanma reference 扩展应放在 `.cache/libriichi3p/`，具体路径和校验
方式见 [`docs/USAGE.md`](../docs/USAGE.md) 与 `artifacts/sanma-assets.json`。
