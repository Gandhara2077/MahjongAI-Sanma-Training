# 三麻 Mortal 训练项目规范

## 目标

在雀魂三麻 3E、北拔、无吃规则下，使用现有 MahjongCopilot 三麻模型作为初始化权重，基于公开三麻牌谱完成可复现的离线训练，并保留 1v2 自对弈和模型导出路径。

## 固定兼容边界

- 训练规则引擎：`Mortal/libriichi` 的 `sanma` Cargo feature。
- 部署观测版本：`control.version = 4`。
- 部署观测形状：`(775, 34)`。
- 原生三麻元数据版本：`dataset.native_version = 5`。
- 动作空间：44 个动作，禁止吃；`40` 为拔北，`43` 为 pass。
- 训练初始化：`C:\Users\Administrator\Downloads\MahjongCopilot - 副本\models\mortal.pth` 的内容复制到本地缓存后，通过 `control.init_from` 读取；不会把用户原模型改写成历史训练状态。
- 参考 ABI：固定加载 Python 3.12 的 `libriichi3p`，并在启动时检查版本、动作空间和观测形状。

## 数据方案

数据来源为 Tenhou 公开牌谱下载接口，限定三凤南（`sanma`）半庄。原始 `.mjlog` 和转换后的 `.mjson` 只保存在本地数据目录并被 Git 忽略；仓库只保存下载命令、日期清单、文件数量、SHA-256 清单和转换验证结果。

训练集按完整牌局文件划分，不能把同一牌局拆到训练和验证两侧。默认使用日期和稳定哈希分组，验证集保留独立日期；每次扩充数据时重新生成索引，不覆盖原始日志。

## 阶段验收

1. ABI check：Python 能加载 3p 参考扩展，模型配置和张量形状吻合。
2. Rules check：Rust `sanma` 单元测试通过，能拒绝四人事件和吃事件。
3. Data check：下载文件能转换为完整 gzip JSONL，日志边界和三人字段通过验证。
4. Loader check：至少一个三麻牌局能生成合法动作 mask、775 通道观测和 GRP 样本。
5. Training check：短跑离线训练写出 `mortal3p.pth`，并能再次加载。
6. Evaluation check：1v2 评估生成 JSON/Markdown 结果；只有通过门槛的 checkpoint 才标记为 best。

## 不纳入本项目

MahjongCopilot 的 GUI、自动登录和雀魂桥接不属于训练运行链路；它只提供已验证的三麻推理 ABI 和初始化模型。数据和模型二进制默认不进入 Git，避免泄露大文件和不可复现的本地路径。
