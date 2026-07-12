# Checkpoint 評価と最良モデル選択（日本語版）

> 言語: [中文](CHECKPOINT_EVALUATION_CN.md) | **日本語**

生成日時: 2026-07-04（日本語版は中文版と同期して維持）

## 1. checkpoint.pt と checkpoint_N.pt の違い

- `checkpoint_N.pt`（例 `checkpoint_5.pt`）：トレーナーが第 N epoch 終了時に `save_freq` に従って保存する **epoch スナップショット**。model + optimizer + scaler + epoch など完全な再開状態を含む（約 10GB）。
- `checkpoint.pt`：トレーナーの**最新/再開エイリアス**。毎 epoch 最新状態に上書きされる。学習終了時には通常、最後の `checkpoint_N.pt` と**バイト単位で同一**（評価器は SHA256 で判定：同一なら alias として記録し重複評価しない；異なる場合は "latest" 候補として個別に評価し、epoch はファイル内部から読む）。
- どちらも **trainer checkpoint** であり、直接推論には使えない（`docs/CHECKPOINT_EXPORT_AND_INFERENCE_JA.md` 参照）。

## 2. なぜ最後の checkpoint が最良と決めつけられないか

小規模データセットのファインチューニングは過学習しやすい：学習 loss が下がり続けても検証セットの mask 品質が向上し続けるとは限らず、後半の epoch は学習セットのアノテーション瑕疵を記憶して検証セットでの境界が悪化しうる。信頼できる唯一の判定方法は、**固定された・独立した・人手修正済みの検証セット**上で 1 つずつ実測すること。選択ルールの最後のタイブレークも「より早い epoch が勝つ」——根拠なく長く学習したモデルを優遇しない。

## 3. なぜ bbox AP は mask 品質を代表できないか

本プロジェクトの mask は RGB-D 位置合わせ、RANSAC 平面フィッティング、ロボットアームの把持姿勢推定に使われる。bbox AP は外接矩形の重なりしか測らない：mask が系統的に 10 ピクセル外側に膨らむ、隣の背表紙と癒着する、境界がギザギザでも、bbox はほぼ変わらず bbox AP は高いままでありうる。学習ログの `coco_eval_bbox_AP` は学習監視シグナルにのみ使え、選択根拠にはできない。本評価の全指標は **mask レベル**で計算される。

## 4. validation と test の違い

- **validation（検証セット）**：学習中の意思決定用——最良 checkpoint の選択、閾値調整。繰り返し使用可。
- **test（テストセット）**：現在は diagnostic-only の比較用。完全フローは validation で best を確定した後、baseline と全 checkpoint を test でも評価するが、**test の結果が best checkpoint を変えることは決してない**。
- 現在登録済みの `book_spine_human_corrected_v1` は train(44)/val(8)/test(12) を含み、`allowed_for_model_evaluation=false`。本ツールは全 checkpoint の test 指標を表示するため、この test は以後「完全に未見の最終ブラインドテストセット」とは見なせない。人手検証や診断には使えるが、最終的なモデル品質の結論にはできない。

## 5. 検証セットガード（評価が拒否される理由）

評価開始前に、**解決済みパス**（ファイル名では決してない）で `data_manifests/dataset_identity_registry.json` から検証セットを検索：

- 未登録 → `blocked`。先に登録するよう促す；
- 機械事前アノテーションとして登録（`human_reviewed=false`、例 `formal_book_spine_sam3_dataset` の val）→ `blocked`：モデル自身の事前アノテーションで checkpoint を選ぶのは自己循環であり結果に意味がない；
- 検証セットパス == 学習セットパス → `blocked`：学習データでの checkpoint 選択は禁止；
- 合法な検証セットがない場合、best を**決して**捏造しない。`best_checkpoint.json` に `status: blocked` + 理由を書く。

## 6. 評価フローと固定条件

全 checkpoint は**完全に同一**の条件を使う（`evaluation/evaluation_config.yaml` に書き込まれる）：画像と GT、prompt（その run に記録された `resolved_training_prompt` を優先。例 `book spine` や `cable`。run に記録がない場合は `config/checkpoint_evaluation.yaml` の `prompt` にフォールバック。`--prompt` の明示指定が最優先）、score/confidence 閾値、min_area、dtype（bf16）、device、mask 後処理（統一して `core.sam3_adapter.Sam3Adapter.predict` 経由）、SAM3 ソース hash、評価器バージョン。trainer checkpoint は `core.checkpoint_export.load_trainer_checkpoint_model` で **strict=True** により新品モデルへロード（重みの静かな取りこぼしゼロ）。

デフォルトの完全フロー：

1. validation の識別とパスを確認；
2. baseline と `checkpoint_5/10/15/20` などユニーク checkpoint を validation で評価；
3. validation 指標のみで selector を実行し `best_checkpoint.json` を書く；
4. 任意で `checkpoints/inference_best.pt` をエクスポート；
5. test の識別とパスを確認；
6. baseline と全ユニーク checkpoint を test で評価；
7. `test_checkpoint_comparison.json` を書く。フィールド名は `test_highest_metric_checkpoint`、`diagnostic_only=true` を明記。

### インスタンスマッチング

GT × prediction の IoU 行列を構築し、**Hungarian アルゴリズム**（`scipy.optimize.linear_sum_assignment`）で総 IoU 最大の一対一マッチングを行う；IoU=0 のペアは未マッチ扱い。1 つの予測が 2 つの GT に同時マッチすることはない。手法名 `hungarian_max_total_iou` は全結果ファイルに記録される。

### 指標定義

- **mean_iou_all_gt（主指標）**：各 GT インスタンスが IoU を 1 つ提供し、**見逃した GT は 0 と記録**、全 GT インスタンスで平均（インスタンスレベル集計。アノテーションが多い画像ほど多くのインスタンスを提供）。matched-only の平均も出力されるが本質的に過大であり、選択には決して使わない。
- `recall_iou_T` = IoU≥T のマッチ数 / GT 総数；`precision_iou_T` = 同じ分子 / 予測総数（T=0.5/0.75/0.9）。
- `miss_rate_iou_50` = 1 − recall@0.5；`false_positive_count` = 予測総数 − IoU≥0.5 マッチ数。
- **Boundary F1**：**元画像解像度**で計算（予測 mask は adapter が元サイズに復元済み）；境界 = mask とその 1px 収縮の差分；許容誤差はデフォルト **2px**（設定可能、実際の値はレポートに記録）；見逃した GT は 0。
- **面積偏差**：IoU≥0.5 マッチペアの `pred_area/gt_area`。mean/median/p10/p90 と `pred_larger_than_gt_rate` を出力し、SAM3 の既知の系統的外側膨張の監視に使う。

## 7. 最良 checkpoint 選択ルール（selector.py、単体テストあり）

1. 評価失敗/NaN/有効結果なしの checkpoint を除外；baseline は自動選択に決して参加しない；
2. `mean_iou_all_gt` 最大が勝ち；
3. 差 ≤ **0.005** はタイ → `mean_boundary_f1_all_gt` が高い方が勝ち；
4. なおタイ → `miss_rate_iou_50` が低い方が勝ち；
5. なおタイ → `false_positive_per_image` が低い方が勝ち；
6. なおタイ → **より早い epoch** が勝ち。

tie tolerance、ルール全文、段階的な裁定の軌跡は `best_checkpoint.json` の `selection_reason` に書き込まれる。

## 8. オリジナル SAM3 とファインチューニング済みモデルの比較方法

`/home/book/sam301/sam3.pt` は **baseline** として毎回一緒に評価され（`--no-baseline` で無効化可）、ランキング表に独立した行（🏁 マーク）で表示され、ファインチューニング内部の最良選択には**参加しない**。`best_checkpoint.json` の `finetuned_improved_over_baseline` が「ファインチューニングは本当にオリジナルより優れているか」に直接答える（mean_iou_all_gt で比較）。

## 9. 出力ファイルの場所

```
<run_dir>/evaluation/
├── evaluation_config.yaml
├── dataset_split_audit.json
├── dataset_split_audit.csv
├── best_checkpoint.json      # validation のみで選択
├── test_checkpoint_comparison.json
├── evaluation_summary.json
├── evaluation.log
├── validation/
│   ├── checkpoint_metrics.csv
│   ├── checkpoint_metrics.json
│   ├── per_image_metrics.csv
│   ├── per_instance_metrics.csv
│   ├── gt_snapshot.json
│   ├── human_review_index.csv
│   ├── failure_cases.csv
│   ├── raw_predictions/<checkpoint>/*.npz + *.json
│   ├── match_records/<checkpoint>/*.json
│   └── visualizations/<checkpoint>/*.png
├── test/
│   └── validation と同じ構成。ただし test の結果は best selection に関与しない
└── cache/
```

`training_summary.json` には独立した `checkpoint_evaluation` ブロックが追記される（アトミック書き込み、既存フィールドは全て保持）。

キャッシュ key = SHA256(split 名 | checkpoint SHA256 | その split のアノテーション SHA256 | その split の画像 SHA256 集約 | 評価設定 SHA256 | 評価器バージョン | SAM3 ソース hash)。validation/test のキャッシュは物理的に分離され、ファイル名は決してキャッシュの根拠にしない；GT・設定・checkpoint のどれかが変われば再計算される。`--force` で全て強制再計算。

`raw_predictions` の NPZ は mask、score、bbox、instance_id を保存；隣の JSON は画像パス、サイズ、prompt、閾値、推論時間を保存。`match_records` は IoU matrix、Hungarian assignment、accepted matches、unmatched GT、unmatched prediction を保存し、指標の再検算に使える。`human_review_index.csv` はデフォルトで `review_status` と `reviewer_notes` の空カラムを保持し、再評価時にユーザーが記入済みの備考をできる限り保持する。

## 10. 手動での評価起動（コマンドライン）

```bash
conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \
  --run-dir /home/book/book01/runs/training/<run_id> \
  --split all \
  --export-best
```

主なオプション：`--split validation|test|all`、`--test-annotations/--test-images`、`--export-best`（validation で選ばれた best のみエクスポート）、`--force`、`--no-baseline`、`--checkpoints checkpoint_5.pt checkpoint_10.pt`、`--score-threshold/--min-area/--boundary-tolerance/--prompt`（固定条件の上書き——上書き後も全 checkpoint に同一適用）、`--max-images N --smoke`（スモーク；検証セットを切り詰めると自動的に smoke とマークされ、正式ランキングにならない）、`--device cpu`。

## 11. Web 操作

UI を起動（通常のターミナル）：`conda run -n sam301 python /home/book/book01/app.py` → `Checkpoint 評価` タブを開く：

1. ドロップダウンで checkpoint を含む学習 run を選択（自動リスト）；
2. `状態/結果を更新`：評価済みかどうか、validation/test のデータ識別、最良 checkpoint、Mean IoU、baseline より優れているか、validation 表、test 表、`best_checkpoint.json` を表示；
3. `Validation と Test を順に評価`：バックグラウンドのサブプロセスで実行（サーバー側の単一タスクガードにより、重複クリックは BLOCKED）。順序は validation → selector → best export → test diagnostic；
4. `Validation と Test を再評価（キャッシュ無視）`：`--split all --force`；
5. `Validation で全 Checkpoint を評価` / `Test で全 Checkpoint を評価`：対象 split のみ実行；
6. `Validation を再評価` / `Test を再評価`：対象 split のキャッシュのみ無視；
7. `最良の推論モデルをエクスポート`：validation best の trainer checkpoint を `checkpoints/inference_best.pt` にエクスポート（既存の export フローを利用、key カバレッジ/base との差分検証を含む）。

合法な検証セットがない場合、ボタンは静かに成功しない——評価サブプロセスは即座に `blocked` で終了し、ページに理由が表示される。

## 12. 自動評価スイッチ

`config/checkpoint_evaluation.yaml` の `auto_evaluate_after_training: false`（第一段階はデフォルト無効、手動トリガーのみ）。将来有効化すれば、学習成功かつ検証セットガード通過時に自動評価が可能——フックのインターフェースは予約済み（`load_defaults()` がこのスイッチを読む）。現段階では学習フローに未接続。

## 13. 最良モデルのエクスポート

`best_checkpoint.json` に保存されるのは **trainer checkpoint のパス参照**（10GB のファイルはコピーしない）。推論モデルのエクスポートは既存の `scripts/export_sam3_inference_checkpoint.py` を再利用（strict key カバレッジ、base との重み差分確認、アトミック書き込み）し、`<run_dir>/checkpoints/inference_best.pt` を出力。エクスポート失敗は評価結果そのものに影響しない（`export_best.status` が evaluation_summary.json に記録される）。
