# 数据集登记 UI 使用说明

这个页面用于把已经人工审核过的 train/val/test 数据集登记到
`data_manifests/dataset_identity_registry.json`。登记后，训练预检才能识别这套
数据是否允许 `formal` 或多 epoch 训练。

## 入口

启动 UI 后打开：

```text
数据集登记
```

## 数据目录要求

选择的数据集根目录需要是这种结构：

```text
data/cable_sam3_dataset/
  train/
    images/
    annotations.json
  val/
    images/
    annotations.json
  test/                 # 可选
    images/
    annotations.json
```

`train` 和 `val` 必须存在；`test` 可选。COCO 的 `categories` 不能为空，图片文件
必须能在对应 split 的 `images/` 目录下找到。

## 表单怎么填

- `dataset root`：数据集根目录，例如 `/home/book/book01/data/cable_sam3_dataset`
- `dataset_id`：唯一名字，例如 `cable_human_corrected_v1`
- `annotation source`：标注来源；如果是 SAM3 预标注后人工修正，选
  `sam3_preannotation_then_human_corrected`
- `annotation source evidence`：一句话说明为什么确认它已人工审核，例如
  “All train/val/test annotations were manually reviewed and corrected before registration.”
- `human reviewed`：确认人工审核过时勾选
- `independently corrected GT`：确认可作为人工修正 GT 时勾选
- `allow formal training`：允许正式训练和多 epoch 时勾选
- `allow final model evaluation`：只有这套数据是最终盲测评估集时才勾选；一般保持不勾选
- `覆盖同 dataset_id 的既有登记和 manifest`：只有你明确要替换旧登记时才勾选

## 推荐操作顺序

1. 填好表单。
2. 点击 `预览登记内容`。
3. 检查输出里的 split 路径、图片数量、标注数量、category、sha256。
4. 确认无误后点击 `写入登记`。
5. 回到 `训练预检` 页面，选择同一套 train/val COCO，`training mode` 设为
   `formal`，再运行预检。

## 登记后会生成什么

登记会写入或更新：

```text
data_manifests/dataset_identity_registry.json
data_manifests/<dataset_id>_dataset_manifest.json
data_manifests/<dataset_id>_split_manifest.json
```

预检会按 train/val 的 `annotations.json` 路径匹配 registry。路径不一致时，即使
内容相同，也会被视为未登记数据集。
