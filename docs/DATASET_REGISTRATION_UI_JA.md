# データセット登録 UI 使用説明

> 言語: [中文](DATASET_REGISTRATION_UI_CN.md) | **日本語**

このページは、人手レビュー済みの train/val/test データセットを
`data_manifests/dataset_identity_registry.json` に登録するためのものです。登録後、
学習プリフライトはこのデータが `formal` または複数 epoch 学習を許可されているかを識別できます。

## 入口

UI を起動して開く：

```text
データセット登録
```

## データディレクトリの要件

選択するデータセットルートディレクトリは以下の構造であること：

```text
data/cable_sam3_dataset/
  train/
    images/
    annotations.json
  val/
    images/
    annotations.json
  test/                 # 任意
    images/
    annotations.json
```

`train` と `val` は必須、`test` は任意。COCO の `categories` は空にできず、画像ファイルは
対応する split の `images/` ディレクトリ配下に存在する必要があります。

## フォームの記入方法

- `dataset root`：データセットルート。例 `/home/book/book01/data/cable_sam3_dataset`
- `dataset_id`：一意な名前。例 `cable_human_corrected_v1`
- `annotation source`：アノテーションの由来。SAM3 事前アノテーション後に人手修正した場合は
  `sam3_preannotation_then_human_corrected` を選択
- `annotation source evidence`：必須。**誰が、いつ、何を**レビューしたかを具体的に書く。例
  「2026-07-12 に <氏名> が CVAT で train/val 全 52 枚の全インスタンス境界を 1 枚ずつ確認・修正した」。
  これは後日の監査のための証拠であり、漠然としたテンプレ文を書いてはいけません。
- `human reviewed` / `independently corrected GT` / `allow formal training` /
  `allow final model evaluation`：4 つのチェックはデフォルトで**全て未チェック**。
  登録は承認行為です——自分が責任を持てる項目だけをチェックしてください。
  `allow formal training` は前の 2 つが同時にチェックされていることを要求し、
  そうでなければ書き込みは拒否されます。`allow final model evaluation` は通常未チェックのまま。
- `同じ dataset_id の既存登録と manifest を上書きする`：旧登録を明確に置き換える場合のみチェック

## 推奨操作手順

1. フォームを記入。
2. `登録内容をプレビュー` をクリック（全画像の sha256 を計算するため、大きいデータセットでは時間がかかります）。
3. 出力の split パス、画像数、アノテーション数、category、sha256 を確認。
   `total_unique_image_count` は train/val/test 横断でグローバルに重複排除した数。
   `exact_duplicate_group_count` > 0 は内容が完全に同一の画像が存在することを意味します
   （特に split 間の重複に注意——それはデータリークです）。
4. 問題なければ `登録を書き込む` をクリック。
5. `学習プリフライト` ページに戻り、同じ train/val COCO を選択して `training mode` を
   `formal` にし、プリフライトを実行。

## 登録で生成されるもの

登録は以下を書き込み/更新します：

```text
data_manifests/dataset_identity_registry.json
data_manifests/<dataset_id>_dataset_manifest.json
data_manifests/<dataset_id>_split_manifest.json
```

プリフライトは train/val の `annotations.json` のパスで registry を照合します。パスが
一致しない場合、内容が同じでも未登録データセットとして扱われます。
