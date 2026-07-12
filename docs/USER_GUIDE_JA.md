# SAM3 学習と推論 使用説明（日本語版）

> 言語: [中文](USER_GUIDE_ZH_CN.md) | **日本語**

対象プロジェクト：`/home/book/book01`

対象 SAM3 ソース：`/home/book/sam301`

対象 Conda 環境：`sam301`

UI 起動コマンド：

```bash
cd /home/book/book01
conda run -n sam301 python /home/book/book01/app.py
```

本ドキュメントは現在のコード実装に基づいて書かれています。過去のドキュメントにある旧 `sam3` 環境、旧 run、旧コマンドをそのまま実行しないでください。

## 1. プロジェクトの用途

本プロジェクトは背表紙インスタンスセグメンテーションの SAM3 ワークフローであり、以下を含みます：

- 背表紙画像の推論；
- raw mask / NMS mask の出力；
- COCO / CVAT エクスポート；
- 学習設定のプリフライト；
- ワンショットの学習起動；
- 学習状態・ログ・summary・checkpoint の閲覧；
- trainer checkpoint の inference checkpoint へのエクスポート；
- inference checkpoint の strict ロードと推論での使用。

現段階の重点は学習と推論のフローが完全に動くことの検証であり、モデル効果の向上を証明することではありません。

明確にしておくべきこと：

- 現在の 184 画像・7185 アノテーションのデータセットは **SAM3 machine pre-annotation smoke dataset** である。
- このデータセットは人手修正済みの ground truth ではない。
- `human_reviewed=false`。
- `allowed_for_formal_training=false`。
- 現在許可されるのは `smoke + max_epochs=1` のフローテストのみ。
- 現在のデータは正式なモデル効果評価に使えない。
- 学習ログの bbox AP は mask 品質とも、ファインチューニング効果の向上とも解釈できない。
- 正式学習の前に人手修正アノテーションデータを作成し、dataset identity を登録し直す必要がある。

## 2. ディレクトリと重要ファイル

主要パス：

| 用途 | パス |
|---|---|
| プロジェクトルート | `/home/book/book01` |
| SAM301 ソース | `/home/book/sam301` |
| Conda 環境 | `sam301` |
| Python | `/home/book/anaconda3/envs/sam301/bin/python` |
| UI エントリ | `/home/book/book01/app.py` |
| 学習ベース YAML | `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml` |
| 学習 wrapper | `/home/book/book01/scripts/launch_sam3_training.py` |
| 学習 preflight CLI | `/home/book/book01/scripts/training_preflight.py` |
| 推論 CLI | `/home/book/book01/scripts/run_unified_inference.py` |
| checkpoint exporter | `/home/book/book01/scripts/export_sam3_inference_checkpoint.py` |
| SAM301 patch manifest | `/home/book/book01/config/sam301_patch_manifest.json` |
| SAM301 patch ファイル | `/home/book/book01/patches/sam301_trainer_grad_accum_loss_scaling.patch` |
| dataset identity registry | `/home/book/book01/data_manifests/dataset_identity_registry.json` |
| 現在の smoke split | `/home/book/book01/data/formal_book_spine_sam3_dataset` |
| 元画像集合 | `/home/book/book01/data/dataset_raw` |
| 学習 run ルート | `/home/book/book01/runs/training` |
| 推論 run ルート | `/home/book/book01/runs/inference` |
| 人手/レビュー出力 | `/home/book/book01/runs/review_artifacts` |
| ドキュメント | `/home/book/book01/docs` |

日常的に作成・変更してよいもの：

- 新しい `runs/training/<run_id>/`；
- 新しい `runs/inference/<run_id>/`；
- `runs/review_artifacts/` 配下の一時的な人手受け入れ画像；
- 新ドキュメント；
- 明確にコミットが必要な manifest やレポート。

上書き・削除してはいけないもの：

- `/home/book/sam301/sam3.pt`；
- 過去の `runs/training/*/checkpoints/checkpoint.pt`；
- 過去の `runs/training/*/checkpoints/inference_model.pt`；
- 元 COCO；
- 元画像；
- 過去の run；
- `data_manifests/formal_dataset_manifest.json`；
- `data_manifests/formal_split_manifest.json`；
- `/home/book/sam301` のソース（個別承認がある場合を除く）。

Git にコミットしてはいけないもの：

- `runs/`；
- `data/`；
- 画像；
- checkpoint；
- 大きなログ；
- `review_artifacts/`。

## 3. 起動前の環境チェック

コマンドは通常のターミナルで実行することを推奨します。GPU が見えない制限付き agent sandbox で学習可用性を判断しないでください。

### 3.1 プロジェクトディレクトリへ移動

```bash
cd /home/book/book01
pwd
```

正常な期待値：

```text
/home/book/book01
```

このパスでない場合、以降の相対パスコマンドが誤ったファイルを読む可能性があります。

### 3.2 Git の確認

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

正常な期待値：

- branch は `e3-formal-training-prep`；
- HEAD はレビュー済みの現在のコミット；
- `git status --short` は既知の未追跡ドキュメントのみ、または空。

異常の意味：

- branch が違う：学習しない。ブランチを切り間違えていないか先に確認；
- 未知のソース変更がある：学習しない。先に diff をレビュー；
- 何を捨てるか完全に確信がない限り、`git reset --hard` や `git clean` で直接掃除しない。

### 3.3 Python、PyTorch、CUDA、SAM3 import の確認

```bash
env -u PYTHONPATH \
conda run -n sam301 python - <<'PY'
from pathlib import Path
import sys
import torch
import sam3

print("python:", sys.executable)
print("python version:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("sam3:", Path(sam3.__file__).resolve())
PY
```

正常な期待値：

```text
python: /home/book/anaconda3/envs/sam301/bin/python
cuda: True
device_count: 1
gpu: NVIDIA GeForce RTX 5090
sam3: /home/book/sam301/sam3/__init__.py
```

異常の意味：

- `sam3` が `/home/book/sam3/...` を指す：新環境のバインドが誤っており、学習不可；
- `cuda: False`：Codex/Claude sandbox 内で出た場合は sandbox が GPU をマップしていないだけの可能性；通常のターミナルで再実行；
- 通常のターミナルでも `cuda: False`：学習しない。NVIDIA driver、PyTorch CUDA build、Conda 環境を確認；
- `python` が `/home/book/anaconda3/envs/sam301/bin/python` でない：新環境を使っていない。

### 3.4 GPU の確認

```bash
nvidia-smi
```

正常な期待値：

- NVIDIA GeForce RTX 5090 が見える；
- driver 正常；
- 未知の大型 compute プロセスがない；
- デスクトップ描画による少量の VRAM 使用は許容。

異常の意味：

- `nvidia-smi` が通信できない：通常のターミナルでこの状態なら学習すべきでない；
- VRAM が未知のタスクに大量占有されている：プロセスを殺さず、まず由来を確認。

### 3.5 SAM301 パッチ状態の確認

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py --json status
conda run -n sam301 python scripts/manage_sam301_patch.py --json verify
```

正常な期待値：

```json
{
  "state": "PATCHED",
  "actual_sha256": "bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2"
}
```

`verify` は `"ok": true` を返すこと。

異常の意味：

- `UNPATCHED`：現在の trainer に loss-scaling パッチが適用されておらず、学習不可；
- `UNKNOWN`：hash 不一致、学習不可；
- `MISSING`：対象ファイルが存在せず、学習不可；
- `/home/book/sam301/sam3/train/trainer.py` を手で編集しないこと。

### 3.6 exporter と UI import の確認

```bash
conda run -n sam301 python - <<'PY'
import app
from core.checkpoint_export import identify_checkpoint
print("app import: OK")
print("exporter import: OK")
print("base:", identify_checkpoint("/home/book/sam301/sam3.pt").type)
print("trainer:", identify_checkpoint("/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt").type)
print("inference:", identify_checkpoint("/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt").type)
PY
```

正常な期待値：

```text
app import: OK
exporter import: OK
base: base
trainer: trainer
inference: inference
```

異常の意味：

- import 失敗：先に traceback を見る。UI や学習を起動しない；
- checkpoint 種別が違う：パスの書き間違いを確認。

## 4. UI の起動

起動コマンド：

```bash
cd /home/book/book01
conda run -n sam301 python /home/book/book01/app.py
```

現在の `app.py` の固定値：

- `server_name="127.0.0.1"`；
- `server_port=7860`；
- `share=False`。

ブラウザでアクセス：

```text
http://127.0.0.1:7860
```

UI 上部タイトル：

```text
SAM3 Fine-tuning ツール（言語切替で中文表示は「SAM3 Fine-tuning 工具」）
```

現在のタブ：

- `推論タスク設定`
- `実行履歴`
- `結果表示`
- `CVAT エクスポート`
- `学習プリフライト`
- `データセット登録`
- `Checkpoint 評価`

UI の停止：

- UI を起動したターミナルで `Ctrl+C`；
- アクティブな学習タスクがある場合、プロセスマネージャは正常終了時に学習プロセスグループの停止を試みる；
- `kill -9`、電源断、カーネルクラッシュではクリーンアップは保証されない。

UI 起動失敗時のチェック：

- カレントディレクトリが `/home/book/book01` か；
- Conda 環境が `sam301` か；
- `app import` が成功するか；
- 7860 が既に使用されていないか；
- Gradio が import できるか。

明確な必要がない限り、ポートや share 設定を勝手に変えないこと。

## 5. データ準備とデータ識別

学習データに必要なこと：

- 画像ディレクトリが読めること；
- COCO `annotations.json` が読めること；
- COCO `images[].file_name` が画像ルートディレクトリで見つかること；
- categories が空でないこと（プリフライトは id 最小の category をターゲットカテゴリとする。val の categories
  はそのカテゴリを含む必要がある。ターゲットは任意の名前でよく、例えば `book_spine` や `cable`）；
- segmentation が現在のフローで解析できること；
- train と val がどちらも空でないこと。

正式（複数 epoch）学習にはさらに、そのデータセットが
`data_manifests/dataset_identity_registry.json` に
`allowed_for_formal_training=true` として登録済みであることが必要。未登録のデータセット
（新ターゲットのデータセットを含む）は smoke モードかつ `max_epochs<=1` しか実行できない。

現在の smoke データパス：

```text
/home/book/book01/data/formal_book_spine_sam3_dataset/train/annotations.json
/home/book/book01/data/formal_book_spine_sam3_dataset/val/annotations.json
/home/book/book01/data/formal_book_spine_sam3_dataset/test/annotations.json
/home/book/book01/data/dataset_raw
```

現在のデータ識別：

```text
annotation_source = sam3_machine_preannotation
human_reviewed = false
independently_corrected_gt = false
intended_use = pipeline_smoke_test
allowed_for_formal_training = false
allowed_for_model_evaluation = false
max_epochs_without_human_review = 1
```

プログラムがデータ識別を知る仕組み：

- `data_manifests/dataset_identity_registry.json` を読む；
- 解決済みの train/val annotation path で registry を照合する；
- registry は manifest SHA256 を記録している；
- 実行時に mask の形を見て自動判定するのではない；
- ファイル名からの推測でもない。

未登録データセットのデフォルト挙動：

- `human_reviewed=false` とみなす；
- `allowed_for_formal_training=false` とみなす；
- smoke one-epoch のみ許可；
- これは fail-closed ポリシーである。

現在の統計：

| 項目 | 数量 |
|---|---:|
| image files | 184 |
| unique images | 136 |
| annotations | 7185 |
| exact duplicate groups | 44 |
| train image files | 160 |
| train unique images | 120 |
| train annotations | 6749 |
| val image files | 12 |
| val unique images | 12 |
| val annotations | 253 |
| test image files | 12 |
| test unique images | 4 |
| test annotations | 183 |

これらの数字は現在の smoke データのみを記述し、将来の正式データを代表しない。

## 6. Smoke と Formal モード

| モード | 用途 | 現在の machine pre-annotation データで許可されるか |
|---|---|---|
| `smoke` | フローテスト、環境テスト、one-epoch 受け入れ | `max_epochs=1` のみ許可 |
| `formal` | 正式学習 | 現在は拒否 |

現在のデータでのルール：

| 設定 | 結果 |
|---|---|
| `training_mode=smoke`, `max_epochs=1` | 許可 |
| `training_mode=smoke`, `max_epochs>1` | 拒否 |
| `training_mode=formal`, 任意の epoch | 拒否 |

拒否が起きる場所：

- runtime YAML の書き込み前；
- launch token の生成前；
- 学習サブプロセスの起動前。

UI で未レビューデータが one-epoch smoke preflight を通過した場合、次のような表示が出るはず：

```text
SMOKE — 数据未经人工审核，非正式训练结果
（データは人手レビューされておらず、正式な学習結果ではない）
```

または中国語/英語混在のデータ識別 warning。

人手レビュー済みの新ターゲットデータセット（例：cable）が既にあるなら、先に UI の
`データセット登録` タブでデータセットルートを選択して登録を書き込み、その後 `学習プリフライト` タブに
戻って `training mode=formal` を選択する。具体的な手順は `docs/DATASET_REGISTRATION_UI_JA.md`。

## 7. Preflight の操作

UI の `学習プリフライト` タブへ。

ステージ A のフィールド：

| UI ラベル | 現在の意味 | よく使う値 |
|---|---|---|
| `authoritative config` | ベース学習 YAML | `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml` |
| `initial checkpoint` | 初期化重み | `/home/book/sam301/sam3.pt` |
| `train images` | train 画像ルート | `/home/book/book01/data/dataset_raw` またはデフォルトの小さい smoke データ |
| `train COCO` | train COCO | `/home/book/book01/data/formal_book_spine_sam3_dataset/train/annotations.json` |
| `val images` | val 画像ルート | `/home/book/book01/data/dataset_raw` またはデフォルトの小さい smoke データ |
| `val COCO` | val COCO | `/home/book/book01/data/formal_book_spine_sam3_dataset/val/annotations.json` |
| `training prompt` | SAM3 テキスト prompt。新ターゲット（例 `cable`）を手動入力可能 | `book spine` または `cable` |
| `output root` | 学習出力ルート | `/home/book/book01/runs/training` |
| `max_epochs` | epoch 数；現在の smoke は 1 | `1` |
| `train batch size` | micro batch size | `1` |
| `gradient accumulation steps` | 勾配累積ステップ数 | `4` |
| `learning rate` | 空欄ならベース YAML の値を継続 | 空欄 |
| `num_workers` | 空欄ならベース YAML の値を継続 | 空欄 |
| `num_gpus` | GPU 数 | `1` |
| `training mode` | `smoke` または `formal` | 現在は `smoke` |

現在の UI に resume フィールドはない。

クリックするボタン：

```text
学習プリフライトを実行（学習は開始しません）
```

preflight がチェックすること：

- ベース YAML の存在；
- train/val の画像と COCO の存在；
- train COCO に category があるか、val COCO が同じターゲット category を含むか；
- missing images；
- training prompt；prompt が COCO category と異なる場合は warning するが直接拒否はしない；
- `max_epochs`、batch size、gradient accumulation；
- effective batch size；
- train 画像数が effective batch より小さくないか；
- train 画像数が effective batch で割り切れない場合は warning；
- dataset identity；
- smoke/formal guard；
- output root の正規化 allowlist；
- SAM301 patch/hash；
- sam3 import path；
- Hydra validate-only；
- per-run distributed port；
- provenance；
- runtime YAML；
- command.txt。

正常に通過した場合：

- `プリフライト状態` に `预检通过，可以启动训练。`（プリフライト合格、学習を開始できます）と表示；
- 未レビューデータには `[SMOKE — 数据未经人工审核，非正式训练结果]` が付く；
- `プリフライト結果` の JSON に以下が見えるはず：
  - `errors: []`
  - `runtime_config_path`
  - `run_dir`
  - `command`
  - `dataset_identity`
  - `training_provenance`
  - `distributed.master_port`

preflight 通過後、新しい run の下に生成される：

```text
config/runtime_config.yaml
dataset_info.json
training_config_summary.json
provenance.json
command.txt
logs/
checkpoints/
```

preflight は学習を開始しない。

## 8. Smoke 学習の開始

現在許可される smoke 学習パラメータ：

```text
max_epochs = 1
train_batch_size = 1
gradient_accumulation_steps = 4
effective_batch_size = 4
num_gpus = 1
resume = 現在の UI では未実装
```

起動手順：

1. 先に preflight を完了し、`errors: []` を確認。
2. チェック：

   ```text
   これが GPU 学習タスクを開始することを確認しました。
   ```

3. クリック：

   ```text
   学習開始
   ```

正式な launcher：

- UI は `ui.training_preflight_page.start_training(...)` を呼ぶ；
- 学習プロセスは `training_process_manager` が管理；
- 実際のコマンドは preflight の `command.txt` に由来；
- wrapper は `/home/book/book01/scripts/launch_sam3_training.py`；
- wrapper は run 内の runtime YAML を公式 `sam3.train.train.main()` に渡す。

これを直接裸で実行しないこと：

```bash
/home/book/sam301/sam3/train/train.py
```

理由：

- launch token をバイパスしてしまう；
- ProcessManager をバイパスしてしまう；
- summary の自動生成をバイパスしてしまう；
- import / patch / provenance / port guard をバイパスしてしまう。

token の仕組み：

- preflight 成功後にワンタイムの launch token が生成される；
- `学習開始` をクリックし全てのサーバー側検証が通ると、token はサブプロセス作成前に消費される；
- 同じ preflight で二度起動することはできない；
- completed、failed、cancelled の後は旧 preflight を再利用できない；
- 起動に失敗した場合も再度 preflight が必要。

学習中に見えるもの：

- `学習タスク状態`；
- `stdout / stderr`；
- `モニタリング情報`；
- 学習終了後の `training_summary.json`。

自動リトライしないこと。失敗後はまず run とログを保全する。

## 9. 学習状態の確認

UI での確認：

- `学習タスク状態`
- `stdout / stderr`
- `モニタリング情報`
- `training_summary.json`

ディスクでの確認：

```bash
RUN=/home/book/book01/runs/training/<run_id>
ls -la "$RUN"
cat "$RUN/command.txt"
cat "$RUN/training_config_summary.json"
cat "$RUN/provenance.json"
cat "$RUN/training_summary.json"
ls -lh "$RUN/checkpoints"
tail -n 100 "$RUN/logs/book_spine/log.txt"
```

背表紙以外のターゲットを学習している場合、ログディレクトリは prompt/category から安全な task slug で生成される。例：

```bash
tail -n 100 "$RUN/logs/cable/log.txt"
```

status の意味：

| status | 意味 |
|---|---|
| `running` | プロセスがまだ実行中 |
| `completed` | exit code 0 |
| `failed` | 非ゼロ exit code |
| `cancelled` | ユーザー停止または shutdown によるキャンセル |

one-epoch 成功の判定基準：

- `training_summary.json.status == "completed"`；
- `exit_code == 0`；
- 1 epoch 完了；
- `checkpoints/checkpoint.pt` が存在しサイズ非ゼロ；
- `discovered_checkpoint_files` が当該 run の checkpoint を列挙している；
- `/home/book/sam301/sam3.pt` に変化がない；
- 学習プロセスの残留がない；
- 残留リッスンポートがない；
- 出力が全て当該 run ディレクトリ内にある。

残留プロセスの確認：

```bash
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml|[t]orchrun"
```

正常な期待値：出力なし。

ベース checkpoint が変更されていないかの確認：

```bash
stat -c 'path=%n size=%s modified=%y permissions=%A' /home/book/sam301/sam3.pt
sha256sum /home/book/sam301/sam3.pt
```

正常な期待値：学習前の記録と一致。

既知の成功 run の例：

```text
/home/book/book01/runs/training/2026-07-03_14-42-27
```

この run：

- `status=completed`；
- `exit_code=0`；
- `max_epochs=1`；
- `train_batch_size=1`；
- `gradient_accumulation_steps=4`；
- `effective_batch_size=4`；
- `checkpoints/checkpoint.pt` あり；
- `checkpoints/inference_model.pt` あり。

## 10. 3 種類の checkpoint の違い

`.pt` 拡張子だけで checkpoint 種別を判断しないこと。

### 10.1 Base checkpoint

パス：

```text
/home/book/sam301/sam3.pt
```

用途：

- モデルの初期化；
- オリジナル SAM3 推論に使用可；
- 上書き不可；
- 学習フローはこれを読むだけで、書いてはならない。

### 10.2 Trainer / resume checkpoint

よくあるパス：

```text
/home/book/book01/runs/training/<run_id>/checkpoints/checkpoint.pt
```

内容：

- `model`
- `optimizer`
- `epoch`
- `scaler`
- `steps`
- その他の resume 状態。

用途：

- 学習時の保存；
- 理論上は resume 用。

現在の制限：

- UI に信頼できる resume 操作は実装されていない；
- trainer checkpoint は通常の推論エントリポイントに直接使えない；
- 推論に直接渡すと `Sam3Adapter` に拒否される。

### 10.3 Inference checkpoint

よくあるパス：

```text
/home/book/book01/runs/training/<run_id>/checkpoints/inference_model.pt
```

内容：

- `format = "sam3_inference"`
- `format_version = 1`
- `model`
- `metadata`

用途：

- 推論に使用；
- exporter が trainer checkpoint から生成；
- ロード時に key、shape、missing/unexpected を strict 検査。

## 11. inference checkpoint のエクスポート

スクリプト：

```text
/home/book/book01/scripts/export_sam3_inference_checkpoint.py
```

引数の確認：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py --help
```

現在の引数：

```text
--input INPUT
--output OUTPUT
--base-checkpoint BASE_CHECKPOINT
--no-base-diff
--overwrite
--dataset-identity-json DATASET_IDENTITY_JSON
--json
```

実際の例：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt \
  --output /home/book/book01/runs/review_artifacts/manual_export/inference_model.pt \
  --json
```

出力の推奨場所：

```text
/home/book/book01/runs/review_artifacts/<your_review_name>/
```

こうすれば Git に入らず、過去の run も上書きしない。

正常な出力に含まれるべきもの：

```text
matched_tensors = 1134
matched_parameters = 841689398
coverage_ratio = 1.0
missing_keys = []
unexpected_keys = []
base_diff.changed_tensors > 0
```

安全規則：

- 出力パス == source trainer checkpoint は不許可；
- `/home/book/sam301/sam3.pt` の上書きは不許可；
- output が既に存在する場合はデフォルト拒否。明示的に `--overwrite` を使った場合のみ；
- coverage が低すぎると失敗；
- shape mismatch で失敗；
- strict load 失敗で失敗；
- エクスポート重みが base と完全同一なら失敗。

注意：

- 現在の CLI は `--dataset-identity-json` を渡した場合のみ dataset identity を inference checkpoint の metadata に書き込む；
- 完全に追跡可能な metadata が必要なら、dataset identity JSON を明示的に渡すか、今後のツール改善を待つこと。

## 12. 推論の操作

現在の実推論エントリポイント：

```text
/home/book/book01/scripts/run_unified_inference.py
```

引数の確認：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/run_unified_inference.py --help
```

主な引数：

| 引数 | 意味 |
|---|---|
| `--input-dir` | 入力画像ディレクトリ |
| `--output-root` | 出力ルート。デフォルト `/home/book/book01/runs` |
| `--sam3-root` | SAM3 ソースルート。デフォルト `/home/book/sam301` |
| `--checkpoint` | base または inference checkpoint |
| `--prompt` | テキスト prompt |
| `--score-threshold` | 最終スコア閾値 |
| `--confidence-threshold` | processor confidence threshold |
| `--dtype-mode` | `bf16`、`fp16`、`none` |
| `--device` | `cuda` または `cpu` |
| `--limit` | 処理画像数の制限 |
| `--nms-iou-thresh` | NMS 閾値 |
| `--nms-metric` | `iou` または `iomin` |
| `--nms-mode` | `suppress` または `merge` |
| `--min-area` | 最小 mask 面積 |
| `--category-name` | COCO クラス名 |

背表紙以外のターゲットも同じエントリポイントを使う。例えば cable の学習や推論では、COCO category は
`cable` を推奨し、推論 prompt も `--prompt "cable"` を渡す；エクスポートする COCO クラス名は
`--category-name cable` が使える。

### 12.1 base checkpoint での推論

```bash
cd /home/book/book01
PYTHONPATH=/home/book/sam301 \
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/sam301/sam3.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 4 \
  --output-root /home/book/book01/runs
```

注意：現在の `data/formal_book_spine_sam3_dataset/{train,val,test}` ディレクトリには
`annotations.json` のみがあり、画像ファイルは split ディレクトリにコピーされていない；COCO の `file_name`
は `data/dataset_raw/.../images` を指している。そのため推論 CLI の `--input-dir` には
実在する画像ディレクトリ、例えば上の `data/dataset_raw/20260622_231208_book_spine/images` を指定すること。

### 12.2 inference checkpoint での推論

```bash
cd /home/book/book01
PYTHONPATH=/home/book/sam301 \
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 4 \
  --output-root /home/book/book01/runs
```

### 12.3 trainer checkpoint を直接推論に使わない

誤った例：

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 1
```

期待される結果：

- プログラムは拒否するはず；
- エラーにこれは trainer checkpoint だと示される；
- エラーに先に `scripts/export_sam3_inference_checkpoint.py` を実行するよう示される。

これは保護が機能している証拠である。

### 12.4 推論出力

推論のたびに新しい run が作られる：

```text
/home/book/book01/runs/inference/<run_id>/
```

よくある出力：

```text
input_images/
raw_npz/
npz_nms/
raw_visualizations/
nms_visualizations/
cvat_export/
manifest.json
run_config.json
validation_report.json
errors.json
```

確認：

```bash
RUN=/home/book/book01/runs/inference/<run_id>
find "$RUN" -maxdepth 2 -type f | sort | head -100
cat "$RUN/run_config.json"
cat "$RUN/manifest.json"
```

UI では：

- `実行履歴` で run 一覧を閲覧；
- `結果表示` で元画像、raw/NMS 可視化、インスタンス表、manifest、errors を閲覧；
- `CVAT エクスポート` で polygon / rle / both を再エクスポート。

silent zero-load でないことの確認：

- trainer checkpoint の直接推論は拒否されるはず；
- inference checkpoint の strict load は成功するはず；
- 少なくとも一部の画像で非空 mask が出力される；
- `run_config.json` の `sam3_checkpoint` が指定した checkpoint であること；
- non-empty mask はチェーンが動くことの証明にすぎず、モデル効果の向上の証明ではない。

## 13. 過去の run と成果物

学習 run の命名：

```text
/home/book/book01/runs/training/YYYY-MM-DD_HH-MM-SS
```

推論 run の命名：

```text
/home/book/book01/runs/inference/<run_id>
```

学習 run のよくあるファイル：

```text
command.txt
config/runtime_config.yaml
dataset_info.json
training_config_summary.json
provenance.json
training_summary.json
logs/<task_slug>/log.txt
checkpoints/checkpoint.pt
checkpoints/inference_model.pt
```

デフォルトの背表紙タスクの `<task_slug>` は `book_spine`；例えば cable タスクは通常
`logs/cable/log.txt`。

成功と失敗の区別：

```bash
cat /home/book/book01/runs/training/<run_id>/training_summary.json
```

見るフィールド：

- `status`
- `exit_code`
- `errors`
- `warnings`
- `discovered_checkpoint_files`

削除してはいけないもの：

- 成功した run；
- 失敗した run；
- `training_summary.json`；
- `command.txt`；
- `runtime_config.yaml`；
- `checkpoint.pt`；
- `inference_model.pt`；
- ログ。

これらは監査と再現のためのファイルであり、Git には入れない。

## 14. Stop / Cancel / Resume

### 14.1 学習の Stop / Cancel

実装済み。

入口：

- UI `学習プリフライト` タブ；
- ボタン：`学習停止`。

挙動：

- `training_process_manager.stop()` を呼ぶ；
- プロセスグループ全体にまず `SIGTERM`；
- タイムアウト後 `SIGKILL`；
- 状態は `cancelled` と記録；
- 可能な限り `training_summary.json` を生成。

制限：

- `kill -9`、電源断、カーネルクラッシュではクリーンアップは保証されない；
- 由来不明の他ユーザープロセスを殺さない；
- 停止後は旧 preflight を再利用できない。

### 14.2 推論の Stop

実装済み。

入口：

- UI `推論タスク設定` タブ；
- ボタン：`停止`。

挙動：

- 現在の推論プロセスを停止；
- 状態は `stopped by user` と表示。

### 14.3 Resume

現在の UI に信頼できる resume 操作は実装されていない。

trainer checkpoint には optimizer、epoch、scaler などの resume 情報が含まれるが、本プロジェクトには検収済みの UI resume フローがなく、RNG の完全復元も宣言されていない。

専用のレビューと受け入れを先に完了しない限り、resume のために runtime YAML を手で変更しないこと。

## 15. よくあるエラーとトラブルシューティング

### 15.1 CUDA unavailable

症状：

```text
torch.cuda.is_available() == False
```

確認：

```bash
nvidia-smi
env -u PYTHONPATH conda run -n sam301 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

安全な対処：

- agent sandbox 内でのみ False なら、通常のターミナルで再試行；
- 通常のターミナルでも False なら、学習しない；
- 個別承認なしに PyTorch/CUDA/driver を再インストールしない。

### 15.2 sam3 import が旧ディレクトリを指す

症状：

```text
sam3: /home/book/sam3/sam3/__init__.py
```

安全な対処：

- 学習しない；
- `sam301` の editable install を確認；
- ターゲットは `/home/book/sam301/sam3/__init__.py` でなければならない。

### 15.3 patch hash mismatch / UNKNOWN / UNPATCHED

確認：

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py --json status
conda run -n sam301 python scripts/manage_sam301_patch.py --json verify
```

安全な対処：

- `PATCHED` 以外では学習しない；
- trainer を手で編集しない；
- UNKNOWN を強制上書きしない。

### 15.4 Hydra config error

症状：

```text
MissingConfigException
hydra config validation failed
```

確認：

```bash
conda run -n sam301 python scripts/launch_sam3_training.py \
  -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1 \
  --validate-only
```

安全な対処：

- `sam3/train/train.py` を直接裸実行しない；
- runtime YAML を `/home/book/sam301` にコピーしない；
- preflight をやり直す。

### 15.5 missing scratch key

症状：

- `scratch.train_batch_size` 欠落；
- `scratch.gradient_accumulation_steps` 欠落；
- 型が正の整数でない。

安全な対処：

- preflight は明確に失敗するはず；
- 空文字列や NaN で回避しない；
- ベース YAML を手で変更しない。

### 15.6 effective batch がデータ量を超える / zero-step guard

症状：

```text
train image count (...) is smaller than the effective batch size (...)
```

原因：

- `train_batch_size x gradient_accumulation_steps x num_gpus` が学習画像数を超えている；
- `drop_last=True` により optimizer step が 0 になる。

安全な対処：

- smoke データには受け入れ済みパラメータ `1 x 4 x 1 = 4` を使う；
- guard を回避するためにコードを勝手に変えない。

### 15.7 port already in use

症状：

```text
TCPStore port ... already in use
```

安全な対処：

- 現在のコードは run ごとに独立した port を割り当てる；
- それでも起きる場合はログを保全；
- 未知のプロセスを殺さない；
- 失敗した run を再利用せず、preflight をやり直して新しい run を作る。

### 15.8 dataset identity 未登録

症状：

```text
no dataset identity record matched...
```

意味：

- プログラムはそのデータが人手 GT だと確認できない；
- デフォルトで未レビューデータとして扱う。

安全な対処：

- one-epoch smoke のみ許可；
- 正式データは新しい registry entry を追加すること；
- 既存の false を直接 true に書き換えない。

### 15.9 formal mode が拒否される

症状：

```text
formal training mode requested, but this dataset is not marked allowed_for_formal_training=true
```

安全な対処：

- 現在のデータでは formal 不可；
- 人手修正 GT を作って登録し直す必要がある。

### 15.10 smoke で epochs が 1 を超える

症状：

```text
max_epochs=20 exceeds the smoke-mode limit (1)
```

安全な対処：

- 現在のデータは `max_epochs=1` のみ許可；
- `smoke` を正式学習として使わない。

### 15.11 trainer checkpoint が inference に使われた

症状：

```text
is a TRAINER checkpoint ... Export it first
```

安全な対処：

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

その後、推論には `inference_model.pt` を使う。

### 15.12 inference checkpoint の coverage 不足 / strict load 失敗

症状：

- `coverage ratio ... below the minimum`
- `shape mismatch`
- `missing`
- `unexpected`
- `strict load ... failed`

安全な対処：

- `strict=False` で回避しない；
- SAM301 コードのバージョンが変わっていないか確認；
- checkpoint が現在のアーキテクチャ由来か確認；
- exporter の完全な出力を保全。

### 15.13 zero predictions

症状：

- 複数の画像で raw/final mask が全て 0。

確認：

- trainer checkpoint を誤用していないか；
- prompt が間違っていないか；
- threshold が高すぎないか；
- input image が正しいか；
- `run_config.json` の checkpoint が期待するパスか。

安全な対処：

- zero predictions を正常通過とみなさない；
- まず base checkpoint とエクスポート済み inference checkpoint で少量の画像を対照実行する。

### 15.14 checkpoint が生成されない

症状：

- `status=completed` だが `checkpoints/` が空。

安全な対処：

- `training_summary.json.warnings` を見る；
- `logs/<task_slug>/log.txt` を見る。例えばデフォルトの背表紙タスクは `logs/book_spine/log.txt`、
  cable タスクは通常 `logs/cable/log.txt`；
- checkpoint を捏造しない；
- その run を受け入れ合格として扱わない。

### 15.15 summary failed

症状：

- `training_summary.json.status == "failed"`；
- `exit_code != 0`。

安全な対処：

- run を保全；
- traceback を保存；
- 自動リトライしない；
- パラメータを変えずに同じ run を直接再実行しない。

### 15.16 残留プロセス

確認：

```bash
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml|[t]orchrun"
```

安全な対処：

- 由来不明のプロセスを殺さない；
- PID/PGID が当該 run のものか確認；
- UI 管理のタスクなら優先的に UI の `学習停止` を使う。

## 16. 人手による機能受け入れ手順

以下のチェックリストはそのまま受け入れ記録にコピーして記入できる。

### A. 環境チェック

操作：

```bash
cd /home/book/book01
git branch --show-current
git status --short
env -u PYTHONPATH conda run -n sam301 python -c "from pathlib import Path; import sys, torch, sam3; print(sys.executable); print(torch.cuda.is_available()); print(Path(sam3.__file__).resolve())"
conda run -n sam301 python scripts/manage_sam301_patch.py --json verify
```

期待値：

- branch は `e3-formal-training-prep`；
- Python は sam301；
- 通常のターミナルで CUDA True；
- sam3 import が `/home/book/sam301/sam3/__init__.py` を指す；
- patch verify ok。

PASS / FAIL：

備考：

### B. UI 起動

操作：

```bash
cd /home/book/book01
conda run -n sam301 python /home/book/book01/app.py
```

開く：

```text
http://127.0.0.1:7860
```

期待値：

- タブが表示される；
- import error がない；
- 推論ページに CUDA 検出情報が表示される。

PASS / FAIL：

備考：

### C. データ識別の表示

操作：

- `学習プリフライト` を開く；
- 現在の smoke train/val COCO を選択；
- `training mode=smoke`；
- `max_epochs=1`；
- `学習プリフライトを実行（学習は開始しません）` をクリック。

期待値：

- preflight 通過；
- 未レビューデータの warning が表示される；
- JSON で `dataset_identity.human_reviewed=false`；
- `allowed_for_formal_training=false`。

PASS / FAIL：

備考：

### D. formal guard

操作：

- `training mode` を `formal` に変更；
- 現在の smoke データのまま；
- preflight をクリック。

期待値：

- preflight 拒否；
- 起動可能な token を生成しない；
- エラーに `allowed_for_formal_training=true` への言及がある。

PASS / FAIL：

備考：

### E. smoke max_epochs guard

操作：

- `training mode=smoke`；
- `max_epochs=2` または `20`；
- preflight をクリック。

期待値：

- preflight 拒否；
- エラーに smoke-mode limit 1 への言及がある。

PASS / FAIL：

備考：

### F. preflight

操作：

- `training mode=smoke`；
- `max_epochs=1`；
- `train_batch_size=1`；
- `gradient_accumulation_steps=4`；
- `num_gpus=1`；
- preflight をクリック。

期待値：

- `errors=[]`；
- 新しい run が生成される；
- runtime YAML、summary config、provenance、command.txt が生成される；
- 学習は開始されない。

PASS / FAIL：

備考：

### G. 学習状態と過去の run

今回学習の再起動が不要なら、既存 run で検証できる：

```text
/home/book/book01/runs/training/2026-07-03_14-42-27
```

操作：

```bash
cat /home/book/book01/runs/training/2026-07-03_14-42-27/training_summary.json
ls -lh /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints
```

期待値：

- `status=completed`；
- `exit_code=0`；
- `checkpoint.pt` がサイズ非ゼロ；
- command が `sam301` を使っている。

PASS / FAIL：

備考：

### H. checkpoint エクスポート

操作：

```bash
mkdir -p /home/book/book01/runs/review_artifacts/manual_acceptance
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt \
  --output /home/book/book01/runs/review_artifacts/manual_acceptance/inference_model.pt \
  --json
```

期待値：

- `matched_tensors=1134`；
- `coverage_ratio=1.0`；
- missing/unexpected が空；
- 出力ファイルが存在する。

PASS / FAIL：

備考：

### I. trainer checkpoint の誤種別拒否

操作：

```bash
conda run -n sam301 python - <<'PY'
from core.sam3_adapter import Sam3Adapter
try:
    Sam3Adapter(
        checkpoint="/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt",
        device="cpu",
    )
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
PY
```

期待値：

- エラーになる；
- `TRAINER checkpoint` への言及がある；
- exporter コマンドへの言及がある。

PASS / FAIL：

備考：

### J. inference checkpoint のロード

操作：

```bash
conda run -n sam301 python - <<'PY'
from core.checkpoint_export import load_inference_checkpoint
model, metadata = load_inference_checkpoint(
    "/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt",
    device="cpu",
)
print("loaded")
print(metadata["key_mapping"]["matched_tensors"])
print(metadata["key_mapping"]["coverage_ratio"])
PY
```

期待値：

- `loaded` が出力される；
- matched tensors が 1134；
- coverage が 1.0。

PASS / FAIL：

備考：

### K. 4 種類の画像での推論

操作：

エクスポート済み inference checkpoint で少量の画像を実行：

```bash
PYTHONPATH=/home/book/sam301 \
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 4 \
  --output-root /home/book/book01/runs
```

期待値：

- 新しい `runs/inference/<run_id>` が生成される；
- raw/NMS 可視化がある；
- NPZ がある；
- 全てが zero predictions ではない。

PASS / FAIL：

備考：

### L. mask / NPZ / 可視化出力

操作：

- UI `結果表示` を開く；
- 先ほどの推論 run を選択；
- 画像を選択；
- `raw` / `nms` を切り替え；
- インスタンス表を確認。

期待値：

- 元画像が表示される；
- raw/NMS 可視化が表示される；
- インスタンス表に annotation id、score、bbox、area が含まれる；
- `errors.json` に致命的エラーがない。

PASS / FAIL：

備考：

### M. Git とファイル完全性チェック

操作：

```bash
git status --short
git ls-files | grep -E '^(runs/|data/|experiments/|test_pic/)|\.(npz|pt|pth|ckpt)$' || true
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml|[t]orchrun" || true
```

期待値：

- run/checkpoint/data が Git に追跡されていない；
- 学習プロセスの残留がない；
- 既知の未追跡ドキュメントのみが表示され、review artifacts は Git に現れない。

PASS / FAIL：

備考：

## 17. 正式学習の前にまだ必要なこと

正式な複数 epoch の前に完了必須：

1. CVAT などのツールで人手修正 GT を作成する；
2. 現在の registry の `false` を直接 `true` に書き換えない；
3. 新データに新しい dataset ID を生成する；
4. dataset manifest を再生成する；
5. split manifest を再生成する；
6. 新しい SHA256 を記録する；
7. `human_reviewed=true` とマークする；
8. `independently_corrected_gt=true` とマークする；
9. `allowed_for_formal_training=true` とマークする；
10. exact duplicate を確認する；
11. train/val/test を固定する；
12. test が学習とチューニングに関与しないことを確認する；
13. オリジナル SAM3 の baseline を実行する；
14. mask IoU、Boundary F1 など mask 品質指標を追加する；
15. best checkpoint 戦略を定義する；
16. 複数 epoch のパラメータを承認する；
17. その後に正式学習を開始する。

現在禁止されていること：

- machine pre-annotation での formal multi-epoch；
- bbox AP を mask 品質とみなすこと；
- one-epoch smoke の結果をモデル効果の向上とみなすこと。

## 18. 安全上の注意事項

禁止：

- `git clean`；
- `git reset --hard`；
- ユーザーファイル未確認での `git stash`；
- 未知の run の削除；
- `/home/book/sam301/sam3.pt` の上書き；
- 過去の `checkpoint.pt` の上書き；
- 過去の `inference_model.pt` の上書き；
- 過去の manifest の変更；
- checkpoint の直接編集；
- 本プロジェクトに属さないプロセスの kill；
- `sudo` による環境の勝手な変更；
- 個別承認なしの `pip install` / `conda install` による環境変更；
- 大型 checkpoint、画像、ログ、`runs/`、`data/` のコミット；
- 正式 launcher をバイパスした `/home/book/sam301/sam3/train/train.py` の裸実行。

安全原則：

- 新しい学習は必ず新しい run；
- 失敗した run は再利用しない；
- 失敗後に自動リトライしない；
- 全ての学習は UI または正式 launcher 経由；
- 全ての推論 checkpoint はまず種別を識別；
- trainer checkpoint は先に export してから推論；
- データが人手レビューされるまでは smoke のみ。
