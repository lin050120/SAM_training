# SAM モデル選択と Checkpoint エクスポート

> 言語: [中文](SAM_MODEL_SELECTION_AND_EXPORT_CN.md) | **日本語**

## 1. Trainer checkpoint と inference model の違い

`checkpoint_N.pt` と `checkpoint.pt` は学習再開用の trainer checkpoint で、`model`、`optimizer`、epoch、scaler などの学習状態を含みます。通常の推論エントリポイントに直接渡すことはできません。

`inference_*.pt` は exporter が trainer checkpoint から変換した推論モデルで、推論に必要なモデル重みと metadata のみを保存します。初期 mask 生成 / CVAT 事前アノテーションフローで使えるのは、オリジナルの `/home/book/sam301/sam3.pt` またはこの inference model のみです。

`checkpoint_35.pt` を直接推論に選ぶと、システムは拒否し、先に Checkpoint エクスポートページでエクスポートするよう促します。

## 2. Web で任意の checkpoint を選んでエクスポート

Checkpoint 評価ページの「任意の Trainer Checkpoint を手動エクスポート」エリアで：

1. 学習 run ディレクトリを入力または選択。例 `/home/book/book01/runs/training/2026-07-04_14-28-27`。
2. 「Checkpoint 一覧を更新」をクリック。
3. ドロップダウンで `checkpoint_5.pt`、`checkpoint_10.pt`、`checkpoint_20.pt` など任意の trainer checkpoint を選択。
4. 選択中の詳細を確認：epoch、SHA256、alias かどうか、checkpoint 種別、対応する inference model が既にあるか。
5. 出力ディレクトリと出力ファイル名を編集。
6. 「選択した Checkpoint をエクスポート」をクリック。

デフォルト出力ディレクトリはその run の `checkpoints/`。デフォルト出力名の規則：

- `checkpoint_35.pt` -> `inference_checkpoint_35.pt`
- `checkpoint_40.pt` -> `inference_checkpoint_40.pt`
- `checkpoint.pt` -> `inference_checkpoint_latest.pt`

出力ファイル名は手動編集できます。既存ファイルはデフォルトで上書き拒否；「既存出力の上書きを許可」をチェックした場合のみ上書きします。システムは `/home/book/sam301/sam3.pt` を決して上書きしません。

`checkpoint.pt` がいずれかの番号付き checkpoint とバイト単位で同一の場合、alias 情報が表示されます。閲覧・エクスポートは可能ですが、通常は重複エクスポートを推奨しません。

## 3. エクスポート検証と metadata

エクスポートは `core.checkpoint_export.export_inference_checkpoint()` を再利用し、第二の checkpoint パーサーを増やしません。エクスポート時のチェック：

- 入力が本物の trainer checkpoint であること；
- key mapping のカバレッジ、missing keys、unexpected keys、shape mismatch；
- 出力が base `sam3.pt` の単純コピーでないこと；
- 出力ファイルが strict-load できること；
- サイドカー metadata が書き込めること。

各エクスポートモデルの隣に生成されます：

```text
inference_checkpoint_35.pt
inference_checkpoint_35.metadata.json
```

metadata には source trainer checkpoint、source SHA256、epoch、run_id、output SHA256、base model SHA256、missing/unexpected keys、coverage ratio、smoke/load 結果が記録されます。以後、由来をファイル名で推測せず、`.metadata.json` を正とみなしてください。

## 4. 初期 mask 生成時のモデル選択

初期 mask / 統一推論ページの「モデル重みの選択」エリアでモデルソースを選択：

- デフォルトのオリジナル SAM3：`/home/book/sam301/sam3.pt`
- 検出済み inference models：`/home/book/book01/runs/training/*/checkpoints/inference_*.pt` と `inference_best.pt` を自動スキャン
- ディレクトリとファイル名を手動入力：例 ディレクトリ `/home/book/book01/runs/training/2026-07-04_14-28-27/checkpoints`、ファイル名 `inference_checkpoint_35.pt`
- 完全な絶対パスを手動入力

ページには解決後の `resolved model path` が表示されます。「モデルを検査」をクリックするとパス、ファイル種別、SHA256、checkpoint 種別、`Sam3Adapter` ロード互換性が検証されます。

何も変更しない場合は引き続き `/home/book/sam301/sam3.pt` が使われます。最新 checkpoint や最大 epoch が自動選択されることはありません。

## 5. 実際に使われたモデルの確認方法

mask 生成のたびに、run ディレクトリに以下が書き込まれます：

```text
sam_model_provenance.json
run_config.json
run_summary.json
logs/run.log
```

このうち `sam_model_provenance.json` が安定したモデル由来記録で、以下を含みます：

- `sam_model_path`
- `sam_model_filename`
- `sam_model_sha256`
- `sam_model_type`
- `source_trainer_checkpoint`
- `source_epoch`
- `source_run_id`
- prompt、threshold、NMS settings
- generation timestamp

これにより「この初期 mask はどの SAM モデルで生成されたか」に答えられます。

## 6. checkpoint 35 と 40 の比較

まずそれぞれエクスポート：

```text
checkpoint_35.pt -> inference_checkpoint_35.pt
checkpoint_40.pt -> inference_checkpoint_40.pt
```

その後、初期 mask ページで 2 つの inference model をそれぞれ選択し、別々の run に出力します。各 run の `sam_model_provenance.json`、`manifest.json`、`npz_nms/`、`coco/instances_default.json`、可視化ディレクトリを比較してください。

現在の run に `checkpoint_35.pt` や `checkpoint_40.pt` が存在しない場合、Web のリストには表示されません。実在する checkpoint のみ選択できます。

## 7. CLI

統一推論 CLI：

```bash
python scripts/run_unified_inference.py \
  --input-dir /path/to/images \
  --output-root /home/book/book01/runs \
  --model-path /home/book/book01/runs/training/<run_id>/checkpoints/inference_checkpoint_35.pt
```

`--model-path` は `--checkpoint` のエイリアス。未指定時のデフォルトは `/home/book/sam301/sam3.pt`。

## 8. よくあるエラー

- run ディレクトリが存在しない：入力が `/home/book/book01/runs/training/<run_id>` かを確認。
- `checkpoints/` が存在しない：その run には学習 checkpoint がない。
- 出力ファイルが既に存在：ファイル名を変更するか、明示的に上書きを許可。
- `checkpoint_N.pt` を推論に選んだ：先に `inference_checkpoint_N.pt` にエクスポート。
- 手動ディレクトリ・ファイル名の打ち間違い：ページの resolved model path を確認。
- 相対パスを入力した：手動パスは絶対パスであること。
- モデルロード失敗や CUDA OOM：CPU でモデルを確認するか、GPU を解放して再試行。
- metadata 書き込み失敗：エクスポートは失敗し、書きかけの出力を削除して由来不明モデルを残さない。
