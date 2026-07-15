# COCO データ拡張 使用説明

## 1. 使用するタイミング

人手による確認を完了し、元データを train、val、test に分割してから使用します。拡張するのは **train のみ**です。val と test は変更しないでください。

## 2. 入力要件

画像フォルダーと対応する COCO 1.0 JSON を用意します。保存場所は任意です。

```text
train/
  images/
    image_001.jpg
    image_002.png
  annotations.json
```

JSON には空でない `images`、`annotations`、`categories` リストが必要です。`images[].file_name` は、選択した画像フォルダーからの相対パスでなければなりません。画像 ID、アノテーション ID、カテゴリー ID は一意の整数で、JSON の幅と高さは実画像と一致する必要があります。COCO polygon、非圧縮 RLE、圧縮 RLE に対応します。

処理前に全入力を検証します。検証に失敗した場合、最終出力は作成されません。

## 3. GUI の起動

`/home/book/book01` で実行します。

```bash
conda run -n sam301 python "Data_augmentation./data_augmentation_gui.py"
```

Windows では OpenCV、NumPy、pycocotools、tkinter を含む環境を有効化してから、`start_data_augmentation.bat` をダブルクリックします。

## 4. GUI の操作

1. `Input image folder`：train の画像フォルダーを選択します。
2. `COCO annotation JSON`：対応する COCO JSON を選択します。
3. `Output dataset folder`：存在しない、または空の出力先を選択します。
4. `Augmentations per source image`：元画像 1 枚あたりの生成枚数です。既定値は 5 です。
5. `MixUp ratio`：生成画像に占める MixUp の比率です。最初は `0` を推奨します。
6. `Random seed`：同じ入力、設定、seed では同じランダム選択になります。
7. `Segmentation format`：生成アノテーションのセグメンテーション形式です。`Auto (match source)`（既定）は元アノテーションごとの形式を維持します（polygon は polygon、非圧縮 RLE は非圧縮 RLE、圧縮 RLE は圧縮 RLE）。`Polygon (point lists)` はすべて COCO 1.0 の点座標で、`RLE (mask)` はすべて圧縮 RLE で出力します。
8. `Parallel workers`：生成に使うプロセス数です。`0`（既定）は全 CPU コアを使用し、`1` はシングルプロセスです。同じ seed ならプロセス数に関係なく出力は同一です。
9. `Include original images and annotations`：既定で有効です。完全な train データセットを出力します。
10. `Back up and replace...`：空でない出力先を置き換える場合だけ有効にします。旧出力は timestamp 付き backup に移動されます。
11. `Generate dataset` をクリックします。

元画像が `M` 枚、1 枚あたりの生成数が `N`、元画像を含める場合、最終枚数は次の通りです。

```text
M * (N + 1)
```

例：元画像 175 枚、設定 5 の場合、拡張画像 875 枚を含む合計 1050 枚です。

## 5. 拡張方法と mask

各生成画像には 2 または 3 種類の方法がランダムに選ばれます。重みが大きい方法ほど選ばれやすくなります。重みの合計を 1 にする必要はありません。

| 方法 | 処理 | mask の処理 |
| --- | --- | --- |
| Object hue | アノテーション対象領域の色相だけを変更 | 位置は不変 |
| Rotate | 既定で -90～+90 度回転 | 同じ回転行列を使用 |
| Scale | 既定で 0.85～1.15 倍に縮小・拡大 | 同じ拡大縮小行列を使用 |
| Crop/zoom | crop 後に元サイズへ戻して拡大 | 同じ crop と resize |
| Translate | 画像を平行移動 | 同じ移動行列を使用 |
| Flip | 水平または垂直反転 | 同じ反転を適用 |
| Noise | ガウシアンノイズを追加 | 位置は不変 |
| Cutout | mask 外の背景だけを遮蔽 | 位置は不変 |

幾何変換では、画像は線形補間、mask は最近傍補間を使用し、両方に同じ幾何パラメーターを適用します。補間により輪郭に 1 pixel 程度の差が生じる場合がありますが、mask が変換前の位置に残ることはありません。

MixUp は同じ大きさの別画像を優先し、変換後の両画像の mask をすべて保持します。互換サイズの別画像がない場合は、現在の画像を使用します。

## 6. 出力形式

```text
<output-dataset-folder>/
  images/
    image_000001.png
    aug_000001_000176.png
    mixup_000002_000177.png
  annotations.json
  augmentation_stats.json
```

`annotations.json` は完全な COCO ファイルです。元アノテーションの形式は維持され、生成アノテーションの形式は `Segmentation format`（コマンドラインでは `--segmentation-format`）で決まります。既定の `auto` は元アノテーションごとの形式に合わせ、polygon 点座標または圧縮 RLE への強制も可能です。polygon 出力は輪郭を 2 倍精度でトレースするため、細長い物体も変換で崩れません。穴のある mask は polygon 形式では穴が埋まります（COCO polygon は穴を表現できません）。`augmentation_stats.json` には入力パス、seed、生成数、セグメンテーション形式、各方法の使用回数、MixUp 情報が記録されます。

全データを同階層の一時フォルダーに生成し、成功後に最終出力へ切り替えます。空でない出力先は既定で拒否され、置換を有効にした場合は先に backup を保存します。

## 7. SAM3 学習での使用

学習の事前チェック画面で次を設定します。

1. train 画像に `<output>/images` を選択します。
2. train COCO に `<output>/annotations.json` を選択します。
3. val/test は元の未拡張データを使用します。
4. training prompt に実カテゴリー名（`book spine`、`cable` など）を入力します。
5. formal 学習前に、UI のデータセット登録機能で「拡張後 train + 元 val/test」の組み合わせを登録します。train のファイルと JSON が変わるため、旧登録は新データを表しません。
6. smoke を実行してから formal を実行します。

## 8. コマンドライン

```bash
conda run -n sam301 python "Data_augmentation./scripts/random_mixup_aug.py" \
  --input-images /path/to/train/images \
  --input-json /path/to/train/annotations.json \
  --output-dir /path/to/augmented_train \
  --augmentations-per-image 5 \
  --mixup-ratio 0 \
  --seed 42
```

主なオプション：

```text
--segmentation-format auto|polygon|rle   生成アノテーションの形式。既定の auto は元形式に合わせる
--workers 0              並列プロセス数。0 または省略で全 CPU コア、1 でシングルプロセス
--no-include-originals   拡張画像と対応アノテーションだけを出力
--overwrite              既存出力を backup して置換
--limit 2                最初の 2 枚だけで簡易確認
--weight-scale 0.18      各方法の相対的な重み
--min-scale 0.85 --max-scale 1.15
```

互換用の `--num-images N` は生成総数を指定し、1 枚あたりの設定を上書きします。通常は `--augmentations-per-image` を使用し、すべての元画像から同じ枚数を生成してください。

## 9. 枚数の目安

- 実 train 画像が 100 枚以上：最初は 1 枚あたり 3～5 枚。
- 20～100 枚：最初は 5～10 枚。
- 20 枚未満：10～20 枚を試せますが、実画像の追加を優先します。
- 1 枚から 1000 枚を生成して正式学習に使う方法は推奨しません。

最終的な枚数は、元の val/test での指標に基づいて決定します。formal 学習前には、出力画像と mask が対象定義に合っているかを人手でも抽出確認してください。
