# SAM3 学習データの自動分割

> 言語: [中文](DATASET_AUTO_SPLIT_CN.md) | **日本語**

この機能は、アノテーション済み COCO データ 2 組を SAM3 学習ディレクトリに整理します：

- アノテーションデータフォルダ：ランダムに `train` と `val` に分割
- test データフォルダ：丸ごと `test` に入る
- デフォルト `val ratio = 0.10`、つまり検証セットはアノテーションデータの約 1/10

出力構造：

```text
<output_dir>/
  train/images/
  train/annotations.json
  val/images/
  val/annotations.json
  test/images/
  test/annotations.json
  manifest.csv
  dataset_build_summary.json
```

## Web での使用

「学習プリフライト」ページの「データセット自動分割」エリアに入力：

- アノテーションデータフォルダ：train/val 分割用の COCO データディレクトリ
- test データフォルダ：全て test にする COCO データディレクトリ
- 出力データセットディレクトリ：デフォルト `/home/book/book01/data/book_spine_sam3_dataset`
- category / training prompt：デフォルト `book spine`。新ターゲット（例 `cable`）も指定可能
- val ratio：デフォルト `0.10`
- random seed：デフォルト `42`

「train/val/test データセットを自動生成」をクリックすると、生成された：

- `train/images`
- `train/annotations.json`
- `val/images`
- `val/annotations.json`

が下の学習プリフライト入力欄に自動入力されます。

生成された test のパスもページに表示され、後の checkpoint evaluation や人手確認に使えます。

## 入力ディレクトリの形式

各入力ディレクトリはフラットでも、複数バッチのサブディレクトリを含んでもかまいません。プログラムは合法な COCO JSON を再帰的に探索します。

各 COCO JSON の画像は以下の順で探索されます：

1. COCO `file_name` が絶対パスならそのまま使用；
2. `<coco_jsonのあるディレクトリ>/<file_name>`；
3. `<coco_jsonのあるディレクトリ>/<basename>`；
4. `<coco_jsonのあるディレクトリ>/images/<basename>`；
5. `<coco_jsonのあるディレクトリの親>/images/<basename>`。

1 枚でも見つからない場合は全体が失敗し、中途半端なデータセットは生成されません。

## 上書きの挙動

デフォルトでは空でない既存出力ディレクトリを上書きしません。

「出力ディレクトリの上書きを許可」をチェックすると、プログラムはまず一時ディレクトリに構築し、
構築が成功した場合にのみ旧出力ディレクトリを以下にリネームします：

```text
<output_dir>.backup_YYYYMMDD_HHMMSS_microseconds
```

その後、新データセットを `<output_dir>` に移動します。

## CLI

コマンドラインでも生成できます：

```bash
python scripts/build_training_dataset_split.py \
  --annotation-pool-dir /path/to/annotated_pool \
  --test-dir /path/to/test_data \
  --output-dir /home/book/book01/data/book_spine_sam3_dataset \
  --category-name "book spine" \
  --val-ratio 0.10 \
  --seed 42
```

他の単一ターゲットを学習する場合は、出力ディレクトリと category を対応するものに変えるだけです。例：

```bash
python scripts/build_training_dataset_split.py \
  --annotation-pool-dir /path/to/cable_annotated_pool \
  --test-dir /path/to/cable_test_data \
  --output-dir /home/book/book01/data/cable_sam3_dataset \
  --category-name "cable" \
  --val-ratio 0.10 \
  --seed 42
```

上書きが必要な場合：

```bash
python scripts/build_training_dataset_split.py ... --overwrite
```

## 重要な保証

- test データが train/val に混入しない；
- train/val はアノテーションデータフォルダのみに由来する；
- アノテーションデータフォルダ内に内容が重複する画像がある場合、構築を直接拒否する
  （同一画像がランダムに train と val の両方に落ちるのを防ぐ）；
- アノテーションデータフォルダと test データフォルダに同一内容の画像がある場合、構築を直接拒否する
  （test の train/val へのリークを防ぐ）；
- 出力ディレクトリはどの入力ディレクトリの内部にも置けない
  （再ビルド時に前回出力の COCO を再収集してしまうのを防ぐ）；
- 出力画像は全てグローバルに `im_000001.*` 形式へリネームされ、バッチ間の名前衝突を防ぐ；
- COCO の `image_id`、`annotation_id`、`file_name` は再マッピングされる；
- 出力 COCO の category はページまたは CLI で指定した category name に統一される。学習プリフライトは
  その category とユーザーが入力した training prompt で各学習の runtime YAML を生成する；
- `manifest.csv` は各出力画像がどの入力フォルダ・バッチ・元ファイル名に由来するかを記録する。
