# プログラム移行手順（日本語）

この文書は、現在の SAM3 学習プログラムを別の PC に移行する手順を説明します。
GitHub 上の `book01` リポジトリには、業務コード、設定、manifest、ドキュメント
のみが含まれます。学習データ、画像、重み、過去の run、SAM3 ソースコード、
conda 環境は別途用意する必要があります。

## 1. パスの自動設定

新 PC のディレクトリは旧 PC と同じである必要はありません。`book01`、`sam301`、
および `sam301` conda 環境を準備した後、新しい `book01` で実行します。

```bash
conda run -n sam301 python scripts/migrate_environment.py
```

2 つのフォルダ選択画面で、新しい `book01`、次に `sam301` を選択します。スクリプトは
必要なコード、SAM3 ソース、学習 YAML、BPE、`sam3.pt` を検証し、
`config/local_paths.json` を生成します。また SAM3 の import 元、trainer patch、CUDA
を確認し、editable install または既知の patch が必要な場合は実行前に確認します。
UNKNOWN 状態の trainer は自動変更しません。

本機専用の設定と `config/migration_report.json` は Git の対象外です。既存の設定は
`config/local_paths.<timestamp>.bak.json` にバックアップされます。

注意：新しい場所の `book01` では、移行ウィザードを実行して
`config/local_paths.json` が生成されるまでプログラムは起動を拒否し、旧 PC の
パスを黙って使う代わりに `scripts/migrate_environment.py` の実行を促す
エラーを表示します。

コマンドラインでも指定できます。

```bash
conda run -n sam301 python scripts/migrate_environment.py \
  --book-root "/new/path/book01" \
  --sam301-root "/new/path/sam301"
```

書き込み、再インストール、patch 適用を行わない事前確認：

```bash
conda run -n sam301 python scripts/migrate_environment.py \
  --book-root "/new/path/book01" \
  --sam301-root "/new/path/sam301" \
  --dry-run
```

非対話モードでは 2 つのパスが必須です。修復も許可する場合は
`--repair-install --apply-patch` を明示的に追加します。

本機設定がない場合は、互換性のため次の旧デフォルトが使われます。

```text
/home/book/book01
/home/book/sam301
```

`core/config.py` を手動で編集しないでください。production patch manifest の trainer
パスは SAM301 root からの相対パスになっているため、PC のパス変更では編集不要です。

## 2. GitHub からプロジェクトを取得

現在の汎用単一ターゲット学習ブランチを使います。

```bash
git clone -b codex-stage-e5-general-target-training \
  git@github.com:lin050120/SAM_training.git \
  /home/book/book01
```

すでに clone 済みの場合：

```bash
cd /home/book/book01
git checkout codex-stage-e5-general-target-training
git pull
```

## 3. conda 環境を準備

すべてのコマンドは基本的に次の conda 環境を使います。

```text
sam301
```

旧 PC で環境を書き出します。

```bash
conda env export -n sam301 > sam301_environment.yml
```

新 PC にコピーして作成します。

```bash
conda env create -f sam301_environment.yml
```

確認：

```bash
conda run -n sam301 python --version
```

## 4. SAM3 ソースコードと重みを準備

GitHub の `book01` には SAM3 ソースコードは含まれません。SAM3 は以下に配置します。

```text
/home/book/sam301
```

最低限必要なもの：

```text
/home/book/sam301/sam3/
/home/book/sam301/sam3.pt
/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml
```

旧 PC から `/home/book/sam301` をコピーしてもよいですし、SAM3 を再インストール
してから `sam3.pt` と学習 YAML を配置しても構いません。

## 5. SAM3 パッケージをインストール

`sam301` 環境が `/home/book/sam301` から SAM3 を import できるようにします。
自動移行スクリプトはこの状態を検査し、誤っている場合は確認後に editable install を
修復します。以下は手動対応が必要な場合のコマンドです。

```bash
cd /home/book/sam301
conda run -n sam301 pip install -e ".[train,dev]"
```

import 先を確認します。

```bash
env -u PYTHONPATH conda run -n sam301 python -c \
  "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
```

期待される出力：

```text
/home/book/sam301/sam3/__init__.py
```

`/home/book/sam3` など古いパスを指してはいけません。

## 6. trainer patch を適用・検証

このプロジェクトでは `/home/book/sam301/sam3/train/trainer.py` に
gradient accumulation の loss scaling patch が必要です。学習 preflight、launcher、
学習 subprocess はすべてこの patch を fail-closed で検証します。

自動移行スクリプトは UNPATCHED の場合だけ確認後に適用します。UNKNOWN の場合は
ファイルを上書きせず、手動確認を要求します。

`/home/book/book01` で実行：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/manage_sam301_patch.py status
conda run -n sam301 python scripts/manage_sam301_patch.py apply
conda run -n sam301 python scripts/manage_sam301_patch.py verify
```

`verify` は終了コード 0 で PATCHED を返す必要があります。  
`status` が UNKNOWN の場合は強制上書きせず、SAM3 ファイルの由来を確認してください。

## 7. データ、モデル、過去 run をコピー

`.gitignore` では大きなデータや成果物が除外されています。

```text
data/
runs/
*.pt
*.jpg
*.png
*.npz
```

そのため GitHub から取得しただけでは以下は含まれません。

- 学習データセット
- 元画像
- 推論/学習の過去 run
- checkpoint / inference model
- `.npz` 中間結果

必要に応じて手動でコピーします。

```text
/home/book/book01/data/
/home/book/book01/runs/          # 過去 run が必要な場合
/home/book/sam301/sam3.pt
```

`data_manifests/dataset_identity_registry.json` は GitHub にありますが、そこに
登録されている `annotations_path` の実ファイルも新 PC に存在している必要があります。

## 8. データセット identity 登録

formal 学習または multi-epoch 学習には、データセットが以下として登録されている
必要があります。

```json
"allowed_for_formal_training": true
```

新しいターゲット（例：cable）の場合、UI の `データセット登録` タブで登録できます。
ディレクトリ構造は次の形式にしてください。

```text
data/cable_sam3_dataset/
  train/images/
  train/annotations.json
  val/images/
  val/annotations.json
  test/images/              # 任意
  test/annotations.json     # 任意
```

登録後、`学習プリフライト` タブに戻り、同じ train/val COCO を選び、
`training mode` を `formal` にします。

## 9. GPU / CUDA を確認

新 PC には NVIDIA GPU、ドライバ、PyTorch と互換性のある CUDA が必要です。

```bash
conda run -n sam301 python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`False` の場合、UI は CUDA 推論と実学習を拒否します。

## 10. UI を起動

```bash
cd /home/book/book01
conda run -n sam301 python app.py
```

ブラウザで開きます。

```text
http://127.0.0.1:7860
```

UI は `127.0.0.1` のみで待ち受けます。デフォルトでは外部公開されません。

## 11. 最小確認

移行後、以下を実行することを推奨します。

```bash
cd /home/book/book01
conda run -n sam301 python scripts/manage_sam301_patch.py verify
env -u PYTHONPATH conda run -n sam301 python -c \
  "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
conda run -n sam301 pytest -q tests/test_sam301_patch.py tests/test_ui.py::UiImportsTest
```

完全な回帰テスト：

```bash
conda run -n sam301 pytest -q tests
```

## 12. 移行時の要点

GitHub のコードだけでは不十分です。新 PC には以下が必要です。

- `sam301` conda 環境
- `/home/book/sam301` の SAM3 ソースコード
- `/home/book/sam301/sam3.pt`
- SAM3 の editable install
- trainer patch の検証成功
- 学習/推論データとモデルファイル
- GPU/CUDA が正常
- `scripts/migrate_environment.py` で本機パス設定を生成
