# SAM3 Fine-tuning ワークフロー（日本語版）

> 言語: [English](README.md) | [中文](README_CN.md) | **日本語**

単一ターゲットの SAM3 ファインチューニングワークフロー（デフォルトは本の背表紙、cable など任意の新ターゲットにも対応）。ローカル Web UI 上部で 中文/日本語 を切り替えられます。

- 簡易使用説明と注意事項：[`docs/QUICK_START_JA.md`](docs/QUICK_START_JA.md)
- 完全な使用説明：[`docs/USER_GUIDE_JA.md`](docs/USER_GUIDE_JA.md)
- プログラム移行ガイド：[`docs/PROGRAM_MIGRATION_JA.md`](docs/PROGRAM_MIGRATION_JA.md)（English: [`docs/PROGRAM_MIGRATION_EN.md`](docs/PROGRAM_MIGRATION_EN.md)，中文: [`docs/PROGRAM_MIGRATION_CN.md`](docs/PROGRAM_MIGRATION_CN.md)）
- 学習の一時停止と再開：[`docs/TRAINING_PAUSE_RESUME_JA.md`](docs/TRAINING_PAUSE_RESUME_JA.md)（English: [`docs/TRAINING_PAUSE_RESUME_EN.md`](docs/TRAINING_PAUSE_RESUME_EN.md)，中文: [`docs/TRAINING_PAUSE_RESUME_CN.md`](docs/TRAINING_PAUSE_RESUME_CN.md)）

以下のコマンドはすべて `book01` プロジェクトルートで実行します。`book01` と SAM301 ソースツリーの本機での場所は、Git 対象外の `config/local_paths.json` で決まります（「別マシンへの移行」参照）。元のマシンでのデフォルトは `/home/book/book01` と `/home/book/sam301` です。以下の `<sam301_root>` は設定された SAM301 ソースディレクトリを指します。

## 別マシンへの移行

`book01` と `sam301` ソースツリーを新しい PC にコピーした後、新しい場所で移行ウィザードを一度実行します：

```bash
conda run -n sam301 python scripts/migrate_environment.py
```

2 つのフォルダ選択画面で新しい `book01` と `sam301` を順に選択します。画面がない環境では `--book-root` / `--sam301-root` を指定します（`--dry-run` で読み取り専用のプレビュー）。ウィザードは両ディレクトリを検証し、`config/local_paths.json` を生成し、SAM3 の editable install・trainer patch・CUDA を確認します。修復はすべて実行前に確認を求め、ハッシュが UNKNOWN の trainer ファイルには決して触れません。最後に `config/migration_report.json` を書き出します。移動後の `book01` は `config/local_paths.json` が生成されるまで起動を拒否し、このウィザードの実行を促します。詳細：[`docs/PROGRAM_MIGRATION_JA.md`](docs/PROGRAM_MIGRATION_JA.md)。

## ローカル Web UI

ローカル Gradio UI は既存の CLI ワークフロー（推論、履歴閲覧、結果表示、CVAT エクスポート、学習プリフライトとオーケストレーション）を可視化するもので、ロジックの再実装は一切行いません。詳細は `docs/stage_d_ui.md` と `docs/stage_e1_training_ui.md` を参照。

```bash
conda run -n sam301 python app.py
```

学習タブは新規学習と永続的な一時停止/再開に対応します：プリフライトは runtime 設定を生成・検証するだけで何も起動しません。開始ボタンはサーバー側でゲートされ、プリフライト合格、checkpoint/データ/runtime YAML が全て存在、他に学習タスクが動いていない、CUDA が利用可能、かつユーザーが明示的に確認した場合にのみ、`scripts/launch_sam3_training.py` 経由で公式 SAM3 トレーナーを起動します（per-run の runtime YAML を `sam3.train.train.main()` に渡す）。プリフライト入力を編集すると保存済みプリフライト結果は即座に無効化されます。少なくとも 1 epoch の完全な checkpoint 生成後は、プロセスを停止して GPU メモリを解放し、ステージ C から最新の完全な `checkpoint.pt` を使って同じ run を再開できます。

通常のターミナルから起動してください（制限された sandbox からではなく）。UI プロセスの CUDA 検出が実際の GPU 可視性を反映するためです。`127.0.0.1:7860` のみで待ち受けます（`share=False`）。UI が学習を黙って開始したり CPU に黙ってフォールバックすることはありません：`device=cuda` を要求して CUDA が利用できない場合、推論の起動を拒否します。

## SAM301 トレーナーパッチガード

SAM301 ソースツリーは git ツリーではありません。`sam3/train/trainer.py` の勾配累積 loss スケーリングパッチは `config/sam301_patch_manifest.json` の完全 SHA256 で固定され、プリフライト・ランチャー・学習サブプロセスの三層で fail-closed に強制されます。正式学習の前（および sam301 再構築後）に実行：

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py verify   # exit 0 (PATCHED) であること
```

`status` / `apply` / `revert` も利用可能。`docs/SAM301_PATCH_MANAGEMENT_JA.md` を参照。

## データ識別（SAM3 事前アノテーション ≠ 人手レビュー済み GT）

現在の `data/formal_book_spine_sam3_dataset` 分割（184 画像 / 7185 アノテーション）は **SAM3 自身の機械事前アノテーション出力**であり、人手修正済み ground truth ではありません。`data_manifests/dataset_identity_registry.json` に `human_reviewed=false`、`allowed_for_formal_training=false` として登録されています。プリフライトは解決済みアノテーションパスで（ファイル名では決して判定せず）registry を照会し、これに対する `--training-mode formal` と `max_epochs>1` の学習を全てブロックします。独立した人手レビュー後にデータセットを正式ステータスへ昇格する方法は `docs/E3_DATASET_IDENTITY_ERRATUM.md` を参照。

このガードは**全ての**データセットに適用されます。新ターゲット（例：cable）も同様です：データセットパスと学習 prompt を与えるだけで smoke 実行（`max_epochs<=1`）は可能ですが、formal または複数 epoch の学習には、人手レビュー後に `allowed_for_formal_training=true` でデータセットを登録することが追加で必要です（UI の「データセット登録」タブが使えます。`docs/DATASET_REGISTRATION_UI_JA.md` 参照）。未登録データセットは未レビューとして扱われます（フェイルセーフのデフォルト）。

## Checkpoint エクスポート（trainer checkpoint → inference checkpoint）

trainer checkpoint（`checkpoints/checkpoint.pt`）は推論エントリポイントに直接渡せません：`sam3/model_builder.py` の loader は trainer checkpoint からゼロ個の重みを静かに読み込んでしまうため、`Sam3Adapter` は実際の構造で checkpoint 種別を識別し、trainer checkpoint をエクスポートのヒント付きで拒否します（`docs/CHECKPOINT_EXPORT_AND_INFERENCE_JA.md` 参照）。先にエクスポート：

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

## Checkpoint 評価（validation で選択 + test は診断のみ）

学習後、まず validation で各ユニーク `checkpoint_N.pt` とオリジナル `sam3.pt` ベースラインを評価し、**validation のみ**から最良 checkpoint を選択し、その後同じ checkpoint 群を test で diagnostic-only の証拠として評価します——test 指標が `best_checkpoint.json` や `inference_best.pt` を変えることはありません。生の予測 mask、マッチ記録、インスタンス毎の指標、GT スナップショット、可視化は `<run_dir>/evaluation/{validation,test}/` に保存されます。UI：「Checkpoint 評価」タブ。ドキュメント：`docs/CHECKPOINT_EVALUATION_JA.md`。

```bash
conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \
  --run-dir runs/training/<run_id> --split all --export-best
```

## 権威 SAM3 学習設定

このワークスペースの権威ファインチューニング設定：

`<sam301_root>/sam3/train/configs/book_spine/book_spine_finetune.yaml`

book_spine という名前ですが、これは**任意の**単一ターゲットカテゴリの固定ベーステンプレートです：プリフライトはこれを決して編集せず、代わりに per-run の `runtime_config.yaml` を生成して prompt、データセットパス、checkpoint、出力/ログディレクトリ（`dumps/<task_slug>`、`logs/<task_slug>`）を上書きします。別ターゲット（例：cable）を学習するには、train/val をそのデータセットに向けて training prompt を設定するだけです——YAML の編集は不要。`--training-prompt` を省略すると、prompt は COCO の最初のカテゴリ名にフォールバックします。

学習を開始せず設定を検査し学習コマンドを生成するプリフライトコマンド：

```bash
conda run -n sam301 python scripts/training_preflight.py
# 任意。COCO カテゴリ名を変更せずに SAM3 学習テキスト prompt を指定：
#   --training-prompt "book spine"
```

プリフライトは `runs/training/<run_id>/config/runtime_config.yaml` に per-run 設定を作り、`dataset_info.json` と `command.txt` を書き、batch size、勾配累積、有効 batch size、checkpoint とデータパス、出力ディレクトリ、解決済み training prompt とその由来を報告します。生成される学習コマンドは以下の通りです：

```bash
conda run -n sam301 python scripts/launch_sam3_training.py \
  -c runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

（train.py 自身の `-c` は `pkg://sam3.train` 内の Hydra 設定名しか受け付けないため、wrapper が run の設定ディレクトリから Hydra を初期化して公式 `sam3.train.train.main()` を呼び出します。）

デフォルトの実行パス：

- checkpoint：`<sam301_root>/sam3.pt`
- train データ：`data/book_spine_sam3_dataset/train`
- val データ：`data/book_spine_sam3_dataset/val`
- 学習出力：`runs/training/<run_id>`

## 統一推論

Legacy NPZ モード：

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --legacy-raw-run data/dataset_raw/20260622_231208_book_spine \
  --limit 2
```

実 SAM3 モード、1 枚：

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir data/book_spine_sam3_dataset/test/images \
  --limit 1 \
  --prompt "book spine" \
  --device cuda
```

実 SAM3 モードはデフォルトで CUDA を要求し、利用できない場合は即座に失敗します。CPU 推論は明示的な `--device cpu` でのみ許可されます。checkpoint のデフォルトは `<sam301_root>/sam3.pt` です（`--checkpoint` で上書き可能）。

CUDA 診断ヘルパー：

```bash
bash scripts/check_cuda_environment.sh
```

## CVAT エクスポート

各推論 run は以下を出力します：

```text
cvat_export/
├── images/
├── annotations/
│   └── instances_default.json
├── polygon/
│   ├── instances_default.json
│   ├── validation_report.json
│   └── polygon_fidelity_report.json
└── rle/
    ├── instances_default.json
    └── validation_report.json
```

SAM3 を再実行せず CVAT パッケージを再エクスポート・検証：

```bash
conda run -n sam301 python scripts/export_cvat_package.py \
  --run-dir runs/inference/<run_id> \
  --segmentation-format both \
  --polygon-fidelity
```

セグメンテーション形式：

- `polygon`：CVAT デフォルト互換パッケージ。最終 NMS mask に対して非可逆。
- `rle`：最終 NMS bool mask からの厳密な mask エクスポート。ローカル検証器がデコード後 mask のピクセル単位一致を確認。
- `both`：`polygon/` と `rle/` の COCO ファイルを別々に書き、形式同士が上書きし合わない。

自動検証されるのは 2 点のみ：pycocotools が segmentation をデコードできること、プロジェクト検証器が通ること。CVAT への実際のインポートと CVAT 再エクスポートの mask 忠実度は手動テストが必要です。1 枚のスモークテストとして、`cvat_export/rle/instances_default.json` と `cvat_export/images/` を一時的な CVAT タスクにインポートし、再エクスポートして `npz_nms/im_000001.npz` と比較してください。
