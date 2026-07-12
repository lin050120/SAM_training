# 数据集登记 UI 使用说明

> 语言版本：**中文** | [日本語](DATASET_REGISTRATION_UI_JA.md)

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
- `annotation source evidence`：必填。写清**谁、什么时候、审核了什么**，例如
  「2026-07-12 由 <姓名> 在 CVAT 中逐张检查并修正了 train/val 全部 52 张图的
  所有实例边界」。这是留给日后审计的证据，不要写空泛的模板句。
- `human reviewed` / `independently corrected GT` / `allow formal training` /
  `allow final model evaluation`：四个勾选默认**全部未勾选**。登记即授权——
  只勾选你能亲自负责的项。`allow formal training` 要求前两项同时勾选，
  否则写入会被拒绝。`allow final model evaluation` 一般保持不勾选。
- `覆盖同 dataset_id 的既有登记和 manifest`：只有你明确要替换旧登记时才勾选

## 推荐操作顺序

1. 填好表单。
2. 点击 `预览登记内容`（会对每张图片计算 sha256，大数据集需要等待）。
3. 检查输出里的 split 路径、图片数量、标注数量、category、sha256。
   `total_unique_image_count` 是跨 train/val/test 全局去重的数量；
   `exact_duplicate_group_count` > 0 说明存在内容完全相同的图片
   （尤其注意是否跨 split 重复——那是数据泄漏）。
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
