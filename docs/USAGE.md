# 使用说明

本文按新克隆的 Windows 工作树编写。仓库提供 Mortal 的源码、Rust native
源码、配置、检查脚本和数据下载/转换脚本；训练权重、牌谱数据、native 构建物
以及评测对局记录都是本地或外部输入，不会由 Git 克隆提供。

## 1. 前置条件与版本边界

需要：

- 64 位 Windows、PowerShell 5.1 或 PowerShell 7；
- CPython 3.12。脚本会拒绝其他 Python 小版本；
- 支持的 AMD GPU、AMD Windows 驱动/ROCm 运行时，以及与 GPU、Windows、Python
  版本匹配的 AMD PyTorch wheel；
- Rust stable/MSVC 工具链和 Cargo。没有可复用的 native `.pyd` 时，编译还需要
  Cargo 可用的 Visual C++/Windows SDK 构建工具；
- Git，以及用于下载牌谱和安装依赖的网络连接；
- Conda（推荐）或一个能运行 CPython 3.12 的虚拟环境。

本项目不是一个需要 `pip install` 的 Python 包，没有单独的包版本号。要复现一
次运行，应同时记录 Git commit、`config/training-profile.toml`、所用权重和
`artifacts/sanma-assets.json` 中的 SHA-256，以及实际安装的 PyTorch/ROCm 版本。

## 2. 创建环境并安装 AMD PyTorch

从仓库根目录创建环境：

```powershell
conda env create -f Mortal/environment.yml
conda activate mortal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`environment.yml` 和 `requirements.txt` 都只列项目实际直接使用的非 PyTorch
依赖：`numpy`、`tqdm`、`toml`、`tensorboard`。`orjson` 只是可选加速项，代码在
未安装时使用标准库 JSON；`torchvision` 和 `torchaudio` 不被本项目导入。

Windows AMD 安装必须先查阅 AMD 的[官方 PyTorch 安装页](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
和[兼容矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)，
按当前 GPU、Windows 和 Python 版本选择 wheel。页面当前给出的 CPython 3.12
示例可写成下面的 PowerShell 命令；如果官方页面已更新，以页面和矩阵为准：

```powershell
python -m pip install --no-cache-dir `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
```

不要执行裸 `pip install torch`，也不要使用 CUDA 的 PyTorch index；那样不能证明
本项目获得 AMD ROCm 支持。安装后先验证：

```powershell
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

必须看到 Python 3.12、`True` 和 AMD GPU 名称。脚本使用 `cuda:0`，ROCm PyTorch
仍通过 PyTorch 的 `torch.cuda` API 暴露设备；GPU 不可用时一键流程会失败，不会
静默改用 CPU。

若不用 Conda，可使用仓库根目录的 `.venv-rocm`：

```powershell
py -3.12 -m venv .venv-rocm
.\.venv-rocm\Scripts\python.exe -m pip install --upgrade pip
.\.venv-rocm\Scripts\python.exe -m pip install -r requirements.txt
# 再用同一个解释器按上面的 AMD 官方 wheel 命令安装 torch
$torchWheel = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl'
.\.venv-rocm\Scripts\python.exe -m pip install --no-cache-dir $torchWheel
$env:MORTAL_PYTHON = (Resolve-Path .\.venv-rocm\Scripts\python.exe).Path
```

## 3. 外部模型与 sanma reference

权重和历史 `libriichi3p` 扩展可能受外部项目许可约束，仓库不重新分发它们。
默认 sanma 配置要求下列精确布局：

| 作用 | 工作树路径 | 说明 |
| --- | --- | --- |
| sanma 初始化/参考权重 | `baselines/sanma_baseline_mortal3p_original.pth` | 配置中的 `control.init_from`、baseline 和默认 `1v2` champion 都使用它 |
| sanma reference | `.cache/libriichi3p/libriichi3p-3.12-x86_64-pc-windows-msvc.pyd` | 用于兼容回放/评测；不是本仓库编译的 native 变体 |

`artifacts/sanma-assets.json` 是小型 manifest，不是模型或二进制本身；
`checks/abi/sanma_setup_check.py` 会按 manifest 检查文件存在、字节数和 SHA-256。
它还记录 sanma reference ABI：版本 4、观测形状 `775×34`、动作空间 44。

如果有授权的 MahjongCopilot 工作树，可手动复制对应文件（不要把外部路径写
进 Git 配置）：

```powershell
$mc = 'D:\path\to\MahjongCopilot'
New-Item -ItemType Directory -Force -Path .\baselines, .\.cache\libriichi3p | Out-Null
Copy-Item -LiteralPath (Join-Path $mc 'models\mortal.pth') `
  -Destination .\baselines\sanma_baseline_mortal3p_original.pth
Copy-Item -LiteralPath (Join-Path $mc 'libriichi3p\libriichi3p-3.12-x86_64-pc-windows-msvc.pyd') `
  -Destination .\.cache\libriichi3p\libriichi3p-3.12-x86_64-pc-windows-msvc.pyd
```

也可以将 reference 文件或其目录通过 `MORTAL_LIBRIICHI3P` 指定；默认配置已
指向上表的 `.cache` 路径。当前 `tools/prepare_sanma_assets.ps1` 仍是历史辅助
脚本，会把模型写到旧的 `.cache/models/mortal3p.pth`；在它被同步到新配置前，
使用上面的手动布局，或把生成文件复制到默认路径后再运行检查。

`baselines/README.md` 列出了本地可保留的五个 sanma 权重。它们都被忽略且不随
仓库发布；yonma 还需要配置所指的 `baselines/baseline.pth`，请从有权限的外部
存储提供，不能用 sanma 权重替代。

## 4. 选择 sanma 或 yonma

只修改 `config/training-profile.toml` 的 active variant：

```toml
[active]
variant = 'sanma' # 或 'yonma'
```

profile 负责同时选择训练配置、自动化配置、玩家数、数据目录和下载参数：

- `sanma`：3 人，使用 `Mortal/config/sanma-training.toml`、`koromo/mjlog` 和
  `koromo/mjson`，默认模型为 `baselines/sanma_baseline_mortal3p_original.pth`；
- `yonma`：4 人，使用 `Mortal/config/yonma.toml`、`koromo4p/mjlog` 和
  `koromo4p/mjson`，默认模型路径为 `baselines/baseline.pth`。

`Mortal/mortal/libriichi.pyd` 是单一活动 native 槽位。切换 variant 后重新运行
`01_prepare.ps1`，它会按当前 Python ABI 编译/选择 `Mortal/native/<variant>/`
下的构建，并激活对应变体；不能在同一工作树中同时激活两个变体。

## 5. 数据

训练数据不随仓库发布。`02_fetch_data.ps1` 使用 profile 的变体、日期、下载根
目录和 worker 参数，从 Tenhou 原始牌谱索引下载 `.mjlog`，转换为 Mortal 使用的
gzip JSON Lines `.mjson`，并在 `data/manifests/` 生成校验清单。数据源入口是
[Tenhou official logs](https://tenhou.net/sc/raw/)；使用前请遵守其条款和相关
数据许可。

按 profile 下载并转换：

```powershell
.\scripts\02_fetch_data.ps1
```

指定日期或限速时，参数会覆盖 profile 的默认值：

```powershell
$startDate = 'YYYYMMDD'
$endDate = 'YYYYMMDD'
.\scripts\02_fetch_data.ps1 -StartDate $startDate -EndDate $endDate -Delay 0.5
```

修改数据目录或日期后，要同步检查所选 `Mortal/config/*.toml` 的 dataset globs，
并确保训练集和验证集没有按局交叉。不要把下载数据、转换数据或 manifest 之外的
本地数据路径写入版本库。

## 6. 一键阶段 01–07

从仓库根目录执行。首次执行前先完成环境、外部资产和 Rust/Cargo 准备：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

先做不下载、不训练的流程预览。预览只打印将要执行的命令，因此不要求外部权重
或牌谱已经就位：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -DryRun -SkipData
```

实际阶段如下；`04`–`06` 可能长时间运行：

```powershell
.\scripts\01_prepare.ps1       # 编译并激活当前 sanma/yonma native 变体
.\scripts\02_fetch_data.ps1    # 下载、转换并检查牌谱
.\scripts\03_validate.ps1      # ROCm、profile、资产、ABI 和 bounded smoke
.\scripts\04_train_grp.ps1     # 训练 GRP 状态
.\scripts\05_train_offline.ps1 # 训练 Mortal 主模型
.\scripts\06_evaluate.ps1      # sanma: one_vs_two；yonma: one_vs_three
.\scripts\07_export.ps1        # 从完整训练状态导出部署权重
```

`run_all.ps1` 会按同样顺序串联阶段：

```powershell
.\scripts\run_all.ps1
```

已有 native 和数据时，可以从中间阶段恢复：

```powershell
.\scripts\run_all.ps1 -FromStep validate -SkipPrepare -SkipData
```

阶段脚本从当前 profile 和所选 TOML 读取步数、batch、输入和输出路径；不要把
`.pth` 文件名硬编码到 PowerShell 脚本中。`03_validate.ps1` 包含有限的训练/编码
smoke，只用于验证环境，不等同于完整训练。

## 7. 输出、对局记录与重建

训练和评测输出都落在所选 TOML 的相对路径下，默认集中在 `Mortal/training/`，
并被 Git 忽略。sanma 配置中的代表性输出是：

- `control.state_file` / `best_state_file`：可恢复的完整训练状态；
- `control.deployment_file` / `best_deployment_file`：部署格式权重；
- `grp.state_file`：GRP 状态；
- `dataset.file_index`、TensorBoard 目录和 snapshot 目录：可重建索引/日志；
- `[test_play].log_dir`、`[train_play.default].log_dir` 和 `[1v2].log_dir`：测试、
  训练对局或评测 match records。

这些生成物不是发布输入。缺失时按“阶段 02 → 03 → 04 → 05 → 06 → 07”重建；
只重新评测/导出时可从 `evaluate` 或 `export` 开始，但相应 state、deployment、
数据和 reference 必须已存在。`07_export.ps1` 的准确输出目标由当前
`control.deployment_file` 决定，执行后以终端输出的 `EXPORT_OK path=...` 为准。

保留基线时不要用部署模型替代完整训练状态，也不要用 GRP 状态替代主模型；三者
格式不同。更换外部权重后先运行 manifest/ABI 检查，再运行评测。

## 8. 检查命令

下面的命令不会启动完整训练：

```powershell
python checks/abi/training_profile_check.py --generic
python checks/abi/one_click_dry_run_check.py
git diff --check
```

拥有外部资产、PyTorch、native 构建和数据后，再运行需要真实环境的检查：

```powershell
python checks/abi/sanma_setup_check.py
python checks/abi/sanma_abi_check.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\03_validate.ps1
```

只检查 Rust native 源码时，可在 `Mortal/` 目录运行上游提供的 workspace 测试：

```powershell
Push-Location Mortal
cargo test --locked --workspace --no-default-features --features flate2/zlib -- --nocapture
Pop-Location
```

不要把“静态检查通过”写成“GPU 运行通过”；还应分别记录 wheel/ROCm 探测、native
ABI、数据存在性、bounded smoke、完整训练和固定评测的结果。本文不预设任何当前
实验数字或模型强度结论。

## 9. native online 的边界

native online/self-play 路径是实验功能。它的吞吐、续跑、内部 smoke 或自对弈结果
不等于 `06_evaluate.ps1` 使用的兼容 reference evaluator，也不能证明模型强度、
晋级或部署安全。需要比较模型时，应在固定、可复现的兼容评测面板上重新评估，并
保存配置、输入权重、reference ABI 和结果；不要从历史文件名或旧实验报告推导当前
“冠军”结论。

## 10. 回滚与恢复

- 变体切换失败时，运行 `Mortal/tools/use-variant.ps1` 查看活动槽位，再重新执行
  `01_prepare.ps1`；每次只保留一个 `mortal/libriichi.pyd` 活动副本。
- 默认 sanma 输入路径现在是 `baselines/sanma_baseline_mortal3p_original.pth`。
  回滚配置只能指向同时存在且 ABI 匹配的旧权重；不要把旧 `.cache/models` 路径
  写回后却不恢复该文件。
- 外部模型或 reference 丢失时，从有权限的 artifact 存储恢复，再用
  `artifacts/sanma-assets.json` 校验；manifest 不能恢复二进制内容。
- `Mortal/training/` 的 state、deployment 和 match records 缺失时，按上一节的
  阶段顺序重建。未保存的训练状态不能由导出模型逆向恢复。
