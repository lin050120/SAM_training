# SAM301 Changes

## 2026-07-02: Fix book-spine gradient accumulation

- 修改文件: `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`
- 修改函数/配置字段: `scratch.gradient_accumulation_steps`
- 修改内容: `1` -> `4`
- 修改原因: 配置顶部说明第 16 行和行内注释第 220 行均声明该 fine-tune 配置应使用 `gradient_accumulation_steps 1→4`，即 `train_batch_size=1`、`num_gpus=1`、累积 4 步，effective batch size 为 4。实际字段仍为 1，导致训练实际 effective batch size 为 1。
- 兼容性影响: 仍兼容原训练入口和命令。训练优化动态会按文档预期变为等效 batch 4；单次 optimizer step 前会累积 4 个 mini-batch。
- 是否仍兼容原训练命令: 是。
- 如何回滚: 将 `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml` 中 `scratch.gradient_accumulation_steps` 改回 `1`，并同步修正文档注释，避免注释与实际行为不一致。

## 2026-07-02: Add training config preflight guard

- 修改文件: 未修改 `/home/book/sam301` 源文件；新增/修改发生在 `/home/book/book01/core/config.py`, `/home/book/book01/core/training_runner.py`, `/home/book/book01/scripts/training_preflight.py`, `/home/book/book01/README.md` 和 docs。
- 修改函数/配置字段: `DEFAULT_BOOK_SPINE_FINETUNE_CONFIG`, `inspect_training_config()`
- 修改原因: 启动训练前明确显示实际配置路径、batch size、GPU 数、梯度累积、effective batch size、初始 checkpoint、训练/验证数据和输出目录，避免再次使用错误配置路径或错误 batch 设置。
- 兼容性影响: 不改变 SAM3 官方训练入口和配置文件；只增加预检和命令生成能力。用户选择非默认权威配置时会显示明显 warning，但不拒绝运行。
- 是否仍兼容原训练命令: 是。
- 如何回滚: 删除 `/home/book/book01/core/config.py`, `/home/book/book01/core/training_runner.py`, `/home/book/book01/scripts/training_preflight.py` 中对应预检入口，并移除 README/docs 中的预检说明。

## 2026-07-02: Route training runtime paths to current workspace

- 修改文件: 未修改 `/home/book/sam301` 源文件；修改发生在 `/home/book/book01/core/config.py`, `/home/book/book01/core/training_runner.py`, `/home/book/book01/scripts/training_preflight.py`, `/home/book/book01/README.md` 和 docs。
- 修改函数/配置字段: `DEFAULT_SAM3_CHECKPOINT`, `DEFAULT_BOOK_SPINE_DATASET_ROOT`, `DEFAULT_TRAINING_RUN_ROOT`, `write_runtime_yaml()`, `inspect_training_config()`
- 修改原因: 权威基础 YAML 中仍含旧项目路径。预检现在默认使用 `/home/book/sam301/sam3.pt`、`/home/book/book01/data/book_spine_sam3_dataset` 和 `/home/book/book01/runs/training/<run_id>`，避免训练结果写入旧 `/home/book/book/experiments`。
- 兼容性影响: 基础 YAML 不被反复改动；每次训练使用独立 runtime YAML。训练命令的 `-c` 指向 runtime YAML，而不是直接指向基础 YAML。
- 是否仍兼容原训练入口: 是，仍调用 `/home/book/sam301/sam3/train/train.py`。
- 如何回滚: 将训练命令重新指向基础 YAML，并删除 runtime YAML 生成逻辑。但不建议回滚，因为旧输出目录存在误覆盖历史实验风险。

## 2026-07-02: Decouple training prompt from COCO category name

- 修改文件: 未修改 `/home/book/sam301` 源文件；修改发生在 `/home/book/book01/core/training_runner.py`, `/home/book/book01/scripts/training_preflight.py`, `/home/book/book01/README.md` 和 docs。
- 修改函数/配置字段: `inspect_training_config(training_prompt=...)`, `write_runtime_yaml()`, `trainer.data.{train,val}.dataset.coco_json_loader.prompts`
- 修改原因: CVAT/推理 COCO category name、训练 COCO category name 和 SAM3 自然语言 prompt 是不同概念。训练时需要允许手动指定 prompt，例如 `"book spine"`，而不强制改动人工 COCO category name。
- 兼容性影响: 不修改 COCO `category_id`、不修改原始 COCO category name、不影响 annotation 读取。只有在用户传入 `--training-prompt` 时，runtime YAML 通过 SAM3 官方 loader 的 `prompts` 覆盖 query text。
- 是否仍兼容原训练入口: 是，仍调用 `/home/book/sam301/sam3/train/train.py`，并通过 runtime YAML 传入 loader 配置。
- 如何回滚: 移除 `--training-prompt` 参数和 runtime YAML 中 `coco_json_loader.prompts` 写入逻辑；未输入 prompt 时仍会回退为 COCO category name。
