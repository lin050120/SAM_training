# COCO Data Augmentation / COCO 数据增强 / COCO データ拡張

[中文](#中文) | [日本語](#日本語) | [English](#english)

## 中文

本工具读取一个图片文件夹及其 COCO 1.0 标注 JSON，为文件夹内的全部图片批量生成增强数据，并输出可直接作为新 train split 使用的图片和 COCO JSON。

```text
输入                         输出
train/images/                augmented_train/images/
train/annotations.json  ->   augmented_train/annotations.json
                              augmented_train/augmentation_stats.json
```

GUI 中的 `Augmentations per source image` 控制每张原图生成多少张增强图，默认是 5。默认同时复制原图和原标注，所以若输入有 `M` 张图片、参数为 `N`，输出共有 `M * (N + 1)` 张图片。

从项目根目录启动：

```bash
conda run -n sam301 python "Data_augmentation./data_augmentation_gui.py"
```

完整步骤、参数和训练接入方式见 [中文使用说明](docs/USAGE_ZH_CN.md)。

## 日本語

このツールは、画像フォルダーと対応する COCO 1.0 アノテーション JSON を読み込み、全画像の拡張データを一括生成します。出力された画像と COCO JSON は、新しい train split としてそのまま使用できます。

```text
入力                         出力
train/images/                augmented_train/images/
train/annotations.json  ->   augmented_train/annotations.json
                              augmented_train/augmentation_stats.json
```

GUI の `Augmentations per source image` で、元画像 1 枚あたりの生成枚数を指定します。既定値は 5 です。元画像を含める設定が既定で有効なため、入力画像が `M` 枚、設定値が `N` の場合、出力は `M * (N + 1)` 枚です。

プロジェクトルートから起動します。

```bash
conda run -n sam301 python "Data_augmentation./data_augmentation_gui.py"
```

詳しい手順、パラメーター、学習への接続方法は [日本語使用説明](docs/USAGE_JA.md) を参照してください。

## English

This tool reads an image folder and its COCO 1.0 annotation JSON, generates augmentations for every source image, and writes images plus a COCO JSON that can be used directly as a new train split.

```text
Input                        Output
train/images/                augmented_train/images/
train/annotations.json  ->   augmented_train/annotations.json
                              augmented_train/augmentation_stats.json
```

`Augmentations per source image` controls how many augmented images are generated from each source image. The default is 5. Original images and annotations are included by default, so `M` input images and a value of `N` produce `M * (N + 1)` output images.

Start the GUI from the project root:

```bash
conda run -n sam301 python "Data_augmentation./data_augmentation_gui.py"
```

See the [English usage guide](docs/USAGE_EN.md) for complete steps, parameters, and training integration.
