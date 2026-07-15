# COCO 数据增强使用说明

## 1. 使用时机

先完成人工审核，并将原始数据划分为 train、val、test。只对 **train** 做增强；val 和 test 保持不变，否则评估结果会失去可比性。

## 2. 输入要求

准备一个图片文件夹和一个对应的 COCO 1.0 JSON。二者可以放在任意位置。

```text
train/
  images/
    image_001.jpg
    image_002.png
  annotations.json
```

JSON 必须包含非空的 `images`、`annotations`、`categories` 列表。每个 `images[].file_name` 必须是相对于所选图片文件夹的路径；图片 ID、标注 ID、类别 ID 必须为唯一整数。JSON 中的宽高必须与实际图片一致。分割标注支持 COCO polygon、未压缩 RLE 和压缩 RLE。

程序启动后会先验证这些内容。验证失败时不会创建最终输出。

## 3. 启动 GUI

在 `/home/book/book01` 下执行：

```bash
conda run -n sam301 python "Data_augmentation./data_augmentation_gui.py"
```

Windows 上先激活包含 OpenCV、NumPy、pycocotools 和 tkinter 的环境，再双击 `start_data_augmentation.bat`。

## 4. GUI 操作

1. `Input image folder`：选择 train 的图片文件夹，例如 `train/images`。
2. `COCO annotation JSON`：选择与这些图片对应的 COCO JSON。
3. `Output dataset folder`：选择一个不存在或为空的输出目录。
4. `Augmentations per source image`：设置每张原图生成多少张增强图，默认 5。
5. `MixUp ratio`：设置增强图中 MixUp 所占比例。`0` 表示不使用，建议先从 0 开始。
6. `Random seed`：相同输入、参数和 seed 会产生相同的随机选择。
7. `Segmentation format`：生成标注的分割格式。`Auto (match source)`（默认）逐条跟随源标注的格式（polygon 保持 polygon，未压缩 RLE 保持未压缩 RLE，压缩 RLE 保持压缩 RLE）；`Polygon (point lists)` 全部输出 COCO 1.0 点坐标；`RLE (mask)` 全部输出压缩 RLE。
8. `Parallel workers`：并行生成的进程数。`0`（默认）自动使用全部 CPU 核心，`1` 为单进程。任何进程数下，相同 seed 的输出完全一致。
9. `Include original images and annotations`：默认勾选，输出将成为完整 train 数据集。
10. `Back up and replace...`：输出目录非空时才勾选。旧目录会被改名为带时间戳的 backup，不会直接删除。
11. 点击 `Generate dataset`。

若原图有 `M` 张，每张生成 `N` 张，且包含原图，则最终图片数为：

```text
M * (N + 1)
```

例如 175 张原图、每张生成 5 张，最终得到 1050 张图片，其中 875 张为增强图。

## 5. 增强方法与 mask

每张增强图随机选择 2 或 3 种方法。权重越大，该方法越容易被选中；权重不要求相加等于 1。

| 方法 | 效果 | mask 处理 |
| --- | --- | --- |
| Object hue | 只改变标注目标区域的色相 | 位置不变 |
| Rotate | 默认在 -90 至 +90 度内旋转 | 使用同一旋转矩阵 |
| Scale | 默认按 0.85 至 1.15 倍缩小或放大 | 使用同一缩放矩阵 |
| Crop/zoom | 裁剪后恢复原尺寸，形成放大效果 | 同步裁剪和缩放 |
| Translate | 平移图片 | 使用同一平移矩阵 |
| Flip | 水平或垂直翻转 | 同步翻转 |
| Noise | 添加高斯噪声 | 位置不变 |
| Cutout | 只遮挡 mask 以外的背景 | 位置不变 |

几何变换中，图片使用线性插值，mask 使用最近邻插值，避免产生无效的半透明类别值。两者共用完全相同的几何参数。边缘可能有 1 像素的插值差异，但 mask 不会停留在变换前的位置。

MixUp 会优先选择尺寸相同的另一张图片，并保留两张图片经过变换后的全部 mask。若没有另一张尺寸兼容图片，则使用当前图片生成 MixUp。

## 6. 输出格式

```text
<output-dataset-folder>/
  images/
    image_000001.png
    aug_000001_000176.png
    mixup_000002_000177.png
  annotations.json
  augmentation_stats.json
```

`annotations.json` 是完整 COCO 文件。原始标注保持原格式，生成标注的格式由 `Segmentation format`（命令行为 `--segmentation-format`）决定：默认 `auto` 逐条跟随源标注格式，也可以强制全部输出 polygon 点坐标或压缩 RLE。选择 polygon 时，mask 会以 2 倍精度描迹轮廓，细长物体也不会因转换而变形；带孔洞的 mask 在 polygon 格式下孔洞会被填充（COCO polygon 无法表示孔洞）。`augmentation_stats.json` 记录输入路径、seed、生成数量、分割格式、方法使用次数和 MixUp 信息。

程序先在同级临时目录生成全部内容，全部成功后才替换为最终输出目录。非空输出默认拒绝覆盖；启用覆盖时会先保留 backup。

## 7. 用于 SAM3 训练

在训练预检页面中：

1. train 图片路径选择 `<output>/images`。
2. train COCO 路径选择 `<output>/annotations.json`。
3. val/test 继续选择原始、未增强的数据。
4. training prompt 填写实际类别，例如 `book spine` 或 `cable`。
5. 正式训练前，通过 UI 的数据集登记功能登记“增强后的 train + 原始 val/test”这一组合。增强会改变 train 文件和 JSON，因此旧登记不能代表新数据。
6. 先运行 smoke，再运行 formal。

## 8. 命令行

```bash
conda run -n sam301 python "Data_augmentation./scripts/random_mixup_aug.py" \
  --input-images /path/to/train/images \
  --input-json /path/to/train/annotations.json \
  --output-dir /path/to/augmented_train \
  --augmentations-per-image 5 \
  --mixup-ratio 0 \
  --seed 42
```

常用可选参数：

```text
--segmentation-format auto|polygon|rle   生成标注的分割格式，默认 auto 跟随源标注
--workers 0              并行进程数；0 或省略使用全部 CPU 核心，1 为单进程
--no-include-originals   只输出增强图和对应标注
--overwrite              备份并替换非空输出目录
--limit 2                只选择前 2 张做快速测试
--weight-scale 0.18      调整某种方法的相对权重
--min-scale 0.85 --max-scale 1.15
```

兼容参数 `--num-images N` 可指定“总共生成 N 张”，并覆盖每张生成数量。常规整套训练数据建议使用 `--augmentations-per-image`，确保每张原图都得到相同数量的增强样本。

## 9. 数量建议

- 100 张以上真实 train 图片：先试每张 3–5 张。
- 20–100 张：先试每张 5–10 张。
- 少于 20 张：可试每张 10–20 张，但应优先增加真实数据。
- 不建议从单张图直接生成 1000 张作为正式训练集。

最终应以原始 val/test 上的指标决定增强数量。正式训练前仍建议人工抽查输出图片及 mask 覆盖是否符合目标定义。
