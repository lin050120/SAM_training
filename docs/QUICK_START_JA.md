# SAM3 Fine-tuning ツール：簡易使用説明と注意事項

> 言語: [中文](QUICK_START_CN.md) | **日本語**

日常利用のための最短パス説明。詳細は `docs/USER_GUIDE_JA.md` を参照。

## このプログラムは何をするものか

SAM3 を**単一ターゲットカテゴリ**でファインチューニングし（デフォルトは背表紙 `book spine`、cable など任意の新ターゲットにも対応）、
併せて推論、COCO/CVAT エクスポート、学習プリフライト/起動/監視、checkpoint 評価とエクスポートを提供します。
全操作はローカル Web UI で完結し、対応するコマンドラインスクリプトもあります。

## UI の起動

```bash
cd /home/book/book01
conda run -n sam301 python app.py
```

ブラウザで `http://127.0.0.1:7860` を開きます（このマシンからのみアクセス可能）。
タブ：推論タスク設定 / 実行履歴 / 結果表示 / CVAT エクスポート / 学習プリフライト / データセット登録 / Checkpoint 評価。
UI 上部のラジオボタンで 中文/日本語 を切り替えられます。

## 典型フロー：新ターゲットの学習（cable を例に）

1. **データセットの準備**。どちらかを選択：
   - アノテーション済み画像フォルダがある → 「学習プリフライト」ページ上部の自動分割エリア（または
     `scripts/build_training_dataset_split.py`）にアノテーションフォルダ、test フォルダ、
     出力ディレクトリ、`category / training prompt = cable` を入力し、
     `train/val/test` の 3 分割を自動生成（各 split は `images/` + `annotations.json`）；
   - 既製の COCO データセットがある → categories にターゲットカテゴリがあること。推奨：
     `{"id": 1, "name": "cable"}`。
2. **学習プリフライト**。「学習プリフライト」ページで train/val の images ディレクトリと COCO パスを入力し、
   `training prompt` に `cable` を指定（空欄なら COCO の最初のカテゴリ名を自動使用）してプリフライトを実行。
   合格すると `runs/training/<run_id>/config/runtime_config.yaml` に本 run 用の設定が生成されます——
   ベース YAML `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`
   を手で編集する必要はなく、**編集してはいけません**。
3. **学習開始**。同じページで進捗を監視。トレーナーログは `<run_dir>/logs/<task_slug>/`。
4. **Checkpoint 評価**。「Checkpoint 評価」ページで run ディレクトリを指定（または
   `scripts/evaluate_sam3_checkpoints.py --run-dir ... --split all --export-best`）。
   その run に記録された学習 prompt を自動使用し、validation 上で最良 checkpoint を選択します。
5. **推論用重みのエクスポート**。trainer checkpoint は直接推論に使えないため先にエクスポート：
   `scripts/export_sam3_inference_checkpoint.py --input <run_dir>/checkpoints/checkpoint.pt --output <run_dir>/checkpoints/inference_model.pt`
   （評価ページの `--export-best` は `inference_best.pt` を自動エクスポート）。
6. **推論**。「推論タスク設定」ページでモデル重みと入力画像ディレクトリを選択、`prompt` に `cable`、
   `category name` に `cable` を指定。実行後は「結果表示」ページで確認し、
   「CVAT エクスポート」ページで CVAT にインポート可能な COCO パッケージを出力できます。

## 注意事項

- **正式学習の前にデータセット識別の登録が必須**。
  `data_manifests/dataset_identity_registry.json` に `allowed_for_formal_training=true` で
  登録されていないデータセット（新ターゲットのデータセットを含む）は、プリフライトが
  smoke モードかつ `max_epochs<=1` しか許可しません。正式な複数 epoch 学習を行うには、
  人手でデータをレビューした後、UI の「データセット登録」タブ（`docs/DATASET_REGISTRATION_UI_JA.md`）
  または手動編集で登録します。これは機械事前アノテーションによる自己学習を防ぐ安全設計であり、
  回避してはいけません。
- **ベース学習 YAML は常に読み取り専用**。各学習の差分（prompt、データパス、出力ディレクトリ、
  ハイパーパラメータ）は全て per-run の `runtime_config.yaml` に書き込まれます。
  ファイル名の book_spine は歴史的な命名にすぎません。
- **学習前に sam301 パッチを検証**：
  `conda run -n sam301 python scripts/manage_sam301_patch.py verify` が PATCHED を出力すること
  （プリフライトとランチャーも強制チェックし、失敗時は起動を拒否します）。
- **prompt と category の関係**：アノテーションは `category_id` でマッチし、prompt は SAM3 への
  自由テキストであり、両者は同一である必要はありません。prompt 未入力時は COCO の最初の
  カテゴリ名に自動フォールバックします。1 つのデータセットで学習されるのは id 最小のカテゴリのみ
  （本ツールは単一ターゲットのファインチューニング）。
- **評価に使う prompt** はその run に記録された `resolved_training_prompt` を自動使用します。
  通常 `--prompt` を手動で渡す必要はなく、渡した場合はそれが最優先されます。
- **run ディレクトリは再利用不可**：出力ディレクトリが存在して空でない場合プリフライトは拒否します。
  プログラムに新しい `<run_id>` を生成させてください。
- **train セットは有効 batch より小さくできない**（`train_batch_size × grad_accum × num_gpus`）。
  さもないと optimizer step が 1 つも走らず、プリフライトが直接拒否します。
- **環境は固定**：常に `conda run -n sam301 ...` で実行。SAM3 ソースは `/home/book/sam301` にあり、
  本プロジェクトはそれを変更しません（管理されたパッチを除く）。
