# SAM3 Fine-tuning ワークフロー（日本語版）

> 言語: [English](README.md) | [中文](README_CN.md) | **日本語**

単一ターゲットの SAM3 ファインチューニングワークフロー（デフォルトは本の背表紙、cable など任意の新ターゲットにも対応）。

- 簡易使用説明と注意事項：[`docs/QUICK_START_JA.md`](docs/QUICK_START_JA.md)
- 完全な使用説明：[`docs/USER_GUIDE_JA.md`](docs/USER_GUIDE_JA.md)
- プログラム移行ガイド：[`docs/PROGRAM_MIGRATION_JA.md`](docs/PROGRAM_MIGRATION_JA.md)（English: [`docs/PROGRAM_MIGRATION_EN.md`](docs/PROGRAM_MIGRATION_EN.md)，中文: [`docs/PROGRAM_MIGRATION_CN.md`](docs/PROGRAM_MIGRATION_CN.md)）

## ローカル Web UI（ステージ D1 / D1.1 / E1）

ローカル Gradio UI は既存の CLI ワークフロー（推論、履歴閲覧、結果表示、CVAT エクスポート、学習プリフライトとオーケストレーション）を可視化するもので、ロジックの再実装は一切行いません。詳細は `docs/stage_d_ui.md`（D1/D1.1）と `docs/stage_e1_training_ui.md`（E1 学習オーケストレーションと監視）を参照。UI 上部で 中文/日本語 を切り替えられます。

学習タブは 2 段階フローです：プリフライトは runtime 設定を生成・検証するだけで何も起動しません。開始ボタン（サーバー側でゲート、単なる無効化ウィジェットではない）は、プリフライト合格、checkpoint/データ/runtime YAML が全て存在、他に学習タスクが動いていない、CUDA が利用可能、かつユーザーが明示的に確認した場合にのみ、`scripts/launch_sam3_training.py` 経由で公式 SAM3 トレーナーを起動します（per-run の runtime YAML を `sam3.train.train.main()` に渡す）。プリフライト入力を編集すると保存済みプリフライト結果は即座に無効化され、古い runtime YAML が学習開始に使われることはありません。

```bash
conda run -n sam301 python /home/book/book01/app.py
```

通常のターミナルから起動してください（制限された coding-agent sandbox からではなく）。UI プロセスの CUDA 検出がターミナルの実際の GPU 可視性を反映するためです。`127.0.0.1:7860` のみで待ち受けます（`share=False`）。`device=cuda` を要求して UI プロセスで CUDA が利用できない場合、暗黙の CPU フォールバックはせず推論の起動を拒否します。

## SAM301 トレーナーパッチガード

`/home/book/sam301` は git ツリーではありません。`sam3/train/trainer.py` の勾配累積 loss スケーリングパッチは `config/sam301_patch_manifest.json` の完全 SHA256 で固定され、プリフライト・ランチャー・学習サブプロセスの三層で fail-closed に強制されます。正式学習の前（および sam301 再構築後）に実行：

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py verify   # exit 0 (PATCHED) であること
```

`status` / `apply` / `revert` も利用可能。`docs/SAM301_PATCH_MANAGEMENT_JA.md` を参照。

## データ識別（SAM3 事前アノテーション ≠ 人手レビュー済み GT）

現在の `data/formal_book_spine_sam3_dataset` 分割（184 画像 / 7185 アノテーション）は **SAM3 自身の機械事前アノテーション出力**であり、人手修正済み ground truth ではありません——各ソース COCO の `info.description` にその旨が明記され、polygon の 100% が頂点数 8 以下で、未編集の機械出力と一致します。`data_manifests/dataset_identity_registry.json` に `human_reviewed=false`、`allowed_for_formal_training=false` として登録されています。プリフライトは解決済みアノテーションパスで（ファイル名では決して判定せず）この registry を照会し、これに対する `--training-mode formal` と `max_epochs>1` の smoke を全てブロックします。完全な経緯と、独立した人手レビュー後にデータセットを正式ステータスへ昇格する方法は `docs/E3_DATASET_IDENTITY_ERRATUM.md` を参照。

このガードは**全ての**データセットに適用されます。新しい非書籍ターゲット（例：cable データセット）も同様です：データセットパスと学習 prompt を与えるだけで smoke 実行（`max_epochs<=1`）は可能ですが、formal または複数 epoch の学習には、人手レビュー後に `data_manifests/dataset_identity_registry.json` へ `allowed_for_formal_training=true` で登録することが追加で必要です（UI の「データセット登録」タブが使えます。`docs/DATASET_REGISTRATION_UI_JA.md` 参照）。未登録データセットは未レビューとして扱われます（フェイルセーフのデフォルト）。

## Checkpoint エクスポート（trainer checkpoint → inference checkpoint）

trainer checkpoint（`checkpoints/checkpoint.pt`）は推論エントリポイントに直接渡せません——`Sam3Adapter` は実際の構造で checkpoint 種別を識別し、trainer checkpoint をエクスポートのヒント付きで即座に拒否します。`sam301/model_builder.py` の loader は trainer checkpoint からゼロ個の重みを静かに読み込んでしまうためです（`docs/CHECKPOINT_EXPORT_AND_INFERENCE_JA.md` 参照）。先にエクスポート：

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

## Checkpoint 評価（validation で選択 + test は診断のみ）

学習後、まず validation で各ユニーク `checkpoint_N.pt` とオリジナル `sam3.pt` ベースラインを評価し、**validation のみ**から最良 checkpoint を選択し、その後同じ checkpoint 群を test で diagnostic-only の証拠として評価します。test 指標が `best_checkpoint.json` や `inference_best.pt` を変えることはありません。評価器は生の予測 mask、マッチ記録、インスタンス毎の指標、GT スナップショット、可視化を `<run_dir>/evaluation/{validation,test}/` に保存します。UI：「Checkpoint 評価」タブ。ドキュメント：`docs/CHECKPOINT_EVALUATION_JA.md`。

```bash
conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \
  --run-dir /home/book/book01/runs/training/<run_id> --split all --export-best
```

## 権威 SAM3 学習設定

このワークスペースの権威ファインチューニング設定：

`/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`

book_spine という名前ですが、これは**任意の**単一ターゲットカテゴリの固定ベーステンプレートです：プリフライトはこれを決して編集せず、代わりに per-run の `runtime_config.yaml` を生成して prompt、データセットパス、checkpoint、出力/ログディレクトリ（`dumps/<task_slug>`、`logs/<task_slug>`）を上書きします。別ターゲット（例：cable）を学習するには、train/val をそのデータセットに向けて training prompt を設定するだけです——YAML の編集は不要。`--training-prompt` を省略すると、prompt は COCO の最初のカテゴリ名にフォールバックします。

学習を開始せず設定を検査し学習コマンドを生成するプリフライトコマンド：

```bash
conda run -n sam301 python scripts/training_preflight.py
```

COCO カテゴリ名を変更せずに SAM3 学習テキスト prompt を手動指定：

```bash
conda run -n sam301 python scripts/training_preflight.py \
  --training-prompt "book spine"
```

生成される学習コマンドは以下の通りです（train.py 自身の `-c` は `pkg://sam3.train` 内の Hydra 設定名であり外部 YAML パスを読み込めないため、wrapper が run の設定ディレクトリから Hydra を初期化して公式 `sam3.train.train.main()` を呼び出します）：

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py \
  -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

プリフライトは権威ベース設定を読み、`/home/book/book01/runs/training/<run_id>/config/runtime_config.yaml` に per-run 設定を作り、`dataset_info.json` と `command.txt` を書き、batch size、GPU 数、勾配累積、有効 batch size、checkpoint、train/val データパス、出力ディレクトリ、COCO カテゴリ、要求された training prompt、解決済み training prompt、prompt の由来を報告します。SAM3 の学習は開始しません。

デフォルトの実行パス：

- checkpoint：`/home/book/sam301/sam3.pt`
- train データ：`/home/book/book01/data/book_spine_sam3_dataset/train`
- val データ：`/home/book/book01/data/book_spine_sam3_dataset/val`
- 学習出力：`/home/book/book01/runs/training/<run_id>`

## 統一推論

Legacy NPZ モード：

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --legacy-raw-run /home/book/book01/data/dataset_raw/20260622_231208_book_spine \
  --limit 2
```

実 SAM3 モード、1 枚：

```bash
PYTHONPATH=/home/book/sam301 conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/book_spine_sam3_dataset/test/images \
  --limit 1 \
  --checkpoint /home/book/sam301/sam3.pt \
  --prompt "book spine" \
  --device cuda
```

実 SAM3 コマンドはデフォルトで CUDA を要求します。CUDA が利用できない場合は即座に失敗し、CPU 推論は明示的な `--device cpu` でのみ許可されます。

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
  --run-dir /home/book/book01/runs/inference/<run_id> \
  --segmentation-format both \
  --polygon-fidelity
```

セグメンテーション形式：

- `polygon`：CVAT デフォルト互換パッケージ。最終 NMS mask に対して非可逆。
- `rle`：最終 NMS bool mask からの厳密な mask エクスポート。ローカル検証器がデコード後 mask のピクセル単位一致を確認。
- `both`：`polygon/` と `rle/` の COCO ファイルを別々に書き、形式同士が上書きし合わない。

自動検証されるのは 2 点のみ：pycocotools が segmentation をデコードできること、プロジェクト検証器が通ること。CVAT への実際のインポートと CVAT 再エクスポートの mask 忠実度は手動テストが必要です。1 枚のスモークテストとして、`cvat_export/rle/instances_default.json` と `cvat_export/images/` を一時的な CVAT タスクにインポートし、再エクスポートして `npz_nms/im_000001.npz` と比較してください。
