# MahjongAI-Sanma-Training

[简体中文](../README.md) · [English](README.en.md) · **日本語**

[Mortal](https://github.com/Equim-chan/Mortal) をベースにした、**三人麻雀（三麻）AI の再現可能な
学習ワークフロー**です。`pip install` で導入できるパッケージではありません。

上流の Mortal は四人麻雀（四麻）向けです。本リポジトリが追加したのは次の点です。

| 機能 | 内容 |
| --- | --- |
| 三麻ルールと符号化 | 抜きドラ・三人局の進行に対応した三人打ち native 拡張、観測 `775×34`・行動空間 44 |
| variant 切替 | `config/training-profile.toml` の `variant` 一つで学習設定・自動化設定・人数・データディレクトリを切替 |
| データパイプライン | 天鳳の生 `.mjlog` をダウンロードし、gzip JSON Lines の `.mjson` へ変換、検証付き manifest を生成 |
| GRP 学習 | 順位点予測器（GRP）を先に学習し、凍結した GRP がオンライン自己対局に稠密な報酬を与える |
| 統計的評価 | 三者対局（`one_vs_two` / `one_vs_three`）、固定予算の段階的パネル、95% 信頼区間と対応検定 |
| フェイルクローズ | ROCm が使えない・資産が無い・ABI が不一致なら停止し、CPU へ黙ってフォールバックしない |

> **範囲に関する注記**：native のオンライン自己対局経路は実験的機能です。そのスループット、
> 再開動作、内部 smoke テスト、自己対局の結果は、互換 reference evaluator による評価と等価では
> なく、棋力・昇格・配備安全性を示すものでもありません。本リポジトリは「最強」「チャンピオン」
> といった主張を行いません。

## 動作要件

| 項目 | 要件 |
| --- | --- |
| OS | 64bit Windows、PowerShell 5.1 または PowerShell 7 |
| Python | **CPython 3.12**（他のマイナーバージョンは拒否されます） |
| GPU | 対応 AMD GPU と ROCm ランタイム。スクリプトは `cuda:0` を使用し、ROCm 版 PyTorch が `torch.cuda` API で公開します |
| PyTorch | GPU・Windows・Python のバージョンに一致する AMD ROCm wheel |
| Rust | stable/MSVC ツールチェーンと Cargo（再利用できる `.pyd` が無い場合は C++ ビルドツールも必要） |

素の `pip install torch` や CUDA 用の PyTorch index は**使用しないでください**。いずれも AMD
ROCm 対応を示すものではありません。GPU が利用できない場合、パイプラインは黙って CPU に切り替
えるのではなくエラー終了します。

## クイックスタート

詳細は [`../USAGE.md`](../USAGE.md) にまとまっています。最短手順：

```powershell
git clone https://github.com/Gandhara2077/MahjongAI-Sanma-Training.git
cd MahjongAI-Sanma-Training

conda env create -f Mortal/environment.yml
conda activate mortal
python -m pip install -r requirements.txt
# GPU に一致する ROCm 版 torch wheel を導入（../USAGE.md の第 2 節）

# ダウンロードも学習も行わず、外部ウェイトも不要でパイプラインを予覧
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -DryRun -SkipData

# その後、外部の三麻用ウェイトと libriichi3p reference を配置（../USAGE.md の第 3 節）
```

## パイプライン

| 段階 | スクリプト | 内容 |
| --- | --- | --- |
| 01 | `01_prepare.ps1` | 現在の variant の native 拡張をビルドして有効化 |
| 02 | `02_fetch_data.ps1` | 天鳳牌譜のダウンロード・変換・検証 |
| 03 | `03_validate.ps1` | ROCm・profile・資産・ABI と範囲限定 smoke の検査 |
| 04 | `04_train_grp.ps1` | GRP（順位点予測器）の状態を学習 |
| 05 | `05_train_offline.ps1` | メインモデルのオフライン学習 |
| 06 | `06_evaluate.ps1` | 評価：三麻は `one_vs_two`、四麻は `one_vs_three` |
| 07 | `07_export.ps1` | 完全な学習状態から配備形式のウェイトを書き出し |

native とデータが既にあれば途中から再開できます。

```powershell
.\scripts\run_all.ps1 -FromStep validate -SkipPrepare -SkipData
```

## variant の切替

`config/training-profile.toml` だけを編集します。

```toml
[active]
variant = 'sanma' # または 'yonma'
```

- `sanma`：3 人打ち。`Mortal/config/sanma-training.toml`、データは `koromo/`、既定モデルは
  `baselines/sanma_baseline_mortal3p_original.pth`。
- `yonma`：4 人打ち。`Mortal/config/yonma.toml`、データは `koromo4p/`、既定モデルは
  `baselines/baseline.pth`。

`Mortal/mortal/libriichi.pyd` は**単一の有効 native スロット**です。variant を切り替えたら
`01_prepare.ps1` を再実行してください。同一ワークツリーで 2 つの variant を同時に有効化できません。

## データと生成物

学習データは本リポジトリに**含まれません**。牌譜は
[Tenhou official logs](https://tenhou.net/sc/raw/) から取得します。利用規約を守り、同一対局を
学習セットと検証セットに分割しないでください。

ウェイト、`Mortal/training/` 配下の生成物（学習状態・配備ウェイト・GRP 状態・インデックス・
TensorBoard ログ・対局記録）、`koromo/{mjlog,mjson}`、reference 拡張はいずれもローカル生成物
であり `.gitignore` で除外されています。失われた場合は `04 → 05 → 06 → 07` の順に再構築します。
`artifacts/sanma-assets.json` は外部資産のサイズ・SHA-256・ABI（バージョン 4、観測 `775×34`、
行動空間 44）のみを記録し、バイナリ自体は含みません。

## 検査と CI

GPU 不要で実行できる検査：

```powershell
python checks/abi/training_profile_check.py --generic
python checks/abi/one_click_dry_run_check.py
git diff --check
```

CI（[`../.github/workflows/ci.yml`](../.github/workflows/ci.yml)）は push と pull request ごとに
上記 2 つの契約検査、プロジェクト自身の Python コードのバイトコンパイル、ruff のエラー級
ルールを実行します。vendored された上流ツリー `Mortal/` はスタイル検査から除外されます。CI は
GPU 学習や棋力を検証**しません**。

## ライセンス

本リポジトリは [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) の改変コピーであり、
**GNU AGPL v3.0** の下で配布されます（[`../LICENSE`](../LICENSE) および
[`../Mortal/LICENSE`](../Mortal/LICENSE)）。帰属と配布の境界は
[`../NOTICE.md`](../NOTICE.md) に記載されています。モデルのウェイト、天鳳の牌譜、
MahjongCopilot の `libriichi3p` reference は外部入力であり、ここでは再配布しません。
