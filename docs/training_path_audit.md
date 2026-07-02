# Training Path Audit

生成时间: 2026-07-02

本轮没有复制、移动或删除大型数据和 checkpoint，也没有修改 `/home/book/sam3` 或 `/home/book/book`。

## 审计结论

- `/home/book/sam301/sam3.pt` 存在，大小约 3.45GB，已作为默认 initial checkpoint。
- `/home/book/book01/data/book_spine_sam3_dataset` 存在，含 train/val/test 副本，已作为默认训练/验证数据源。
- 训练输出默认根目录已改为 `/home/book/book01/runs/training`。
- 不再默认写入 `/home/book/book/experiments/book_spine_sam3_r1`。
- 权威基础 YAML 保持为模板；每次预检/训练会生成 runtime YAML 到本次 training run 目录。

## 旧路径审计

| 原路径 | 存在 | 类型 | 大小 | 工作副本内 | 只读输入 | 推荐路径 | 是否修改默认 | 依据 |
|---|---:|---|---:|---:|---:|---|---:|---|
| `/home/book/sam3/sam3.pt` | 是 | 文件 | 3450062241 bytes | 否 | 是 | `/home/book/sam301/sam3.pt` | 是 | sam301 中已有同大小 checkpoint，本次开发应优先使用 sam301 |
| `/home/book/book/data/book_spine_sam3_dataset/train/images/` | 是 | 目录 | 32M | 否 | 是 | `/home/book/book01/data/book_spine_sam3_dataset/train/images` | 是 | book01 中已有完整训练数据副本 |
| `/home/book/book/data/book_spine_sam3_dataset/train/annotations.json` | 是 | 文件 | 46639 bytes | 否 | 是 | `/home/book/book01/data/book_spine_sam3_dataset/train/annotations.json` | 是 | book01 中已有对应 COCO |
| `/home/book/book/data/book_spine_sam3_dataset/val/images/` | 是 | 目录 | 7.8M | 否 | 是 | `/home/book/book01/data/book_spine_sam3_dataset/val/images` | 是 | book01 中已有完整验证数据副本 |
| `/home/book/book/data/book_spine_sam3_dataset/val/annotations.json` | 是 | 文件 | 12297 bytes | 否 | 是 | `/home/book/book01/data/book_spine_sam3_dataset/val/annotations.json` | 是 | book01 中已有对应 COCO |
| `/home/book/book/experiments/book_spine_sam3_r1` | 是 | 目录 | 38G | 否 | 否 | `/home/book/book01/runs/training/<run_id>` | 是 | 继续写入旧 experiments 有覆盖历史实验风险 |

## 候选路径检查

| 候选路径 | 状态 |
|---|---|
| `/home/book/sam301/sam3.pt` | 存在，文件，3450062241 bytes，可读，位于 sam301 |
| `/home/book/book01/data/book_spine_sam3_dataset` | 存在，目录，约 61M，可读写，位于 book01 |
| `/home/book/book01/runs/training` | 已由预检创建，训练 run 根目录 |
| `/home/book/book01/experiments` | 存在，约 38G，但不作为新训练默认输出 |

## Runtime YAML 策略

基础 YAML:

`/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`

每次预检/训练创建:

`/home/book/book01/runs/training/<run_id>/config/runtime_config.yaml`

同时写入:

- `/home/book/book01/runs/training/<run_id>/dataset_info.json`
- `/home/book/book01/runs/training/<run_id>/command.txt`
- `/home/book/book01/runs/training/<run_id>/logs/`
- `/home/book/book01/runs/training/<run_id>/checkpoints/`

本轮验证生成的 run:

`/home/book/book01/runs/training/2026-07-02_12-01-48`

## 当前预检结果

- checkpoint: `/home/book/sam301/sam3.pt`
- BPE: `/home/book/sam301/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- train images: `/home/book/book01/data/book_spine_sam3_dataset/train/images`
- train annotations: `/home/book/book01/data/book_spine_sam3_dataset/train/annotations.json`
- val images: `/home/book/book01/data/book_spine_sam3_dataset/val/images`
- val annotations: `/home/book/book01/data/book_spine_sam3_dataset/val/annotations.json`
- output dir: `/home/book/book01/runs/training/2026-07-02_12-01-48`
- train COCO: 8 images, 186 annotations
- val COCO: 2 images, 49 annotations
- missing image files: 0
- effective batch size: `1 × 1 × 4 = 4`

## Warnings

- 当前 train/val COCO 的 category 名称是 `book spine`，不是严格的 `book_spine`。预检将其作为兼容别名通过，并输出 warning。后续 COCO/CVAT 导出阶段应统一默认 category 为 `book_spine`。

## 尚需决定的问题

- 是否要在后续数据规范化阶段把已有训练/验证 COCO 的 category 名从 `book spine` 迁移为 `book_spine`。这会改变标注文件格式，当前未执行。
- 是否要清理或归档 `/home/book/book01/experiments` 下已有 38G 内容。当前未修改。

