# 自动划分 SAM3 训练数据

本功能用于把两组已标注 COCO 数据整理成 SAM3 训练目录：

- 标注数据文件夹：随机划分为 `train` 和 `val`
- test 数据文件夹：整体进入 `test`
- 默认 `val ratio = 0.10`，即验证集约为标注数据的 1/10

输出结构：

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

## 网页使用

打开“训练预检”页面，在“数据集自动划分”区域填写：

- 标注数据文件夹：用于切分 train/val 的 COCO 数据目录
- test 数据文件夹：全部作为 test 的 COCO 数据目录
- 输出数据集目录：默认 `/home/book/book01/data/book_spine_sam3_dataset`
- category / training prompt：默认 `book spine`
- val ratio：默认 `0.10`
- random seed：默认 `42`

点击“自动生成 train/val/test 数据集”后，页面会把生成的：

- `train/images`
- `train/annotations.json`
- `val/images`
- `val/annotations.json`

自动填入下方训练预检输入框。

生成的 test 路径也会显示在页面上，供后续 checkpoint evaluation 或人工检查使用。

## 输入目录格式

每个输入目录可以是扁平目录，也可以包含多个批次子目录。程序会递归查找合法 COCO JSON。

每个 COCO JSON 的图片按以下顺序定位：

1. COCO `file_name` 是绝对路径时直接使用；
2. `<coco_json所在目录>/<file_name>`；
3. `<coco_json所在目录>/<basename>`；
4. `<coco_json所在目录>/images/<basename>`；
5. `<coco_json所在目录的父目录>/images/<basename>`。

找不到任意图片时会整体失败，不会生成半成品数据集。

## 覆盖行为

默认不覆盖已有非空输出目录。

勾选“允许覆盖输出目录”时，程序会先构建临时目录；只有构建成功后，旧输出目录才会被改名为：

```text
<output_dir>.backup_YYYYMMDD_HHMMSS
```

然后新数据集移动到 `<output_dir>`。

## CLI

也可以用命令行生成：

```bash
python scripts/build_training_dataset_split.py \
  --annotation-pool-dir /path/to/annotated_pool \
  --test-dir /path/to/test_data \
  --output-dir /home/book/book01/data/book_spine_sam3_dataset \
  --category-name "book spine" \
  --val-ratio 0.10 \
  --seed 42
```

如需覆盖：

```bash
python scripts/build_training_dataset_split.py ... --overwrite
```

## 关键保证

- test 数据不会混入 train/val；
- train/val 只来自标注数据文件夹；
- 所有输出图片会全局重命名为 `im_000001.*` 形式，避免不同批次重名；
- COCO `image_id`、`annotation_id`、`file_name` 会重新映射；
- 输出 COCO category 统一为页面或 CLI 中填写的 category name；
- `manifest.csv` 记录每张输出图来自哪个输入文件夹、批次和原始文件名。
