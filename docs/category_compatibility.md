# Category Compatibility

生成时间: 2026-07-02

## 结论

- SAM3 训练数据加载按 `category_id` 分组标注，不按 `"book_spine"` 字符串精确过滤标注。
- 默认情况下 category name 会作为 query text 使用。当前训练数据中的 `"book spine"` 不会导致标注丢失。
- CVAT/推理 COCO category name、训练 COCO category name、SAM3 自然语言 prompt 是三个不同概念，不要求强制相同。
- 当前已确认成功导入 CVAT 的 polygon COCO 使用 category name `"book_spine"`。
- 新推理导出的 COCO/CVAT 包默认使用 `"book_spine"`，以兼容既有 CVAT 导入样本和项目目标名称。
- 当前训练 COCO category name 是 `"book spine"`。
- 默认 SAM3 训练 prompt 可手动指定，例如 `"book spine"`。预检将输出实际 resolved prompt 和来源。
- 不修改人工 COCO。训练预检将 `"book spine"` 作为兼容别名接受，并输出 warning，但不把 category name 和 prompt 强行统一。

## 依据

- `/home/book/sam301/sam3/train/data/coco_json_loaders.py:37` 的 `load_coco_and_group_by_image()` 读取 `categories` 后构建 `cat_id_to_name = {cat["id"]: cat["name"] ...}`。
- `/home/book/sam301/sam3/train/data/coco_json_loaders.py:110` 到 `:147` 的 `COCO_FROM_JSON.__init__()` 接受可选 `prompts`，可用 `{id, name}` 覆盖 query text。
- `/home/book/sam301/sam3/train/data/coco_json_loaders.py:199` 到 `:206` 按 `ann["category_id"]` 将 annotations 分组。
- `/home/book/sam301/sam3/train/data/coco_json_loaders.py:241` 到 `:249` 使用 category name 生成 `query["query_text"]`。
- `/home/book/sam301/sam3/train/data/sam3_image_dataset.py:394` 到 `:421` 将 query text 和 original category id 放入训练 datapoint，不做 `book_spine` 名称过滤。
- `/home/book/book01/data/book_spine_sam3_dataset/train/annotations.json` 的 category 是 `{"id": 1, "name": "book spine"}`。
- `/home/book/book01/data/cvat_import_polygon_20260622_231208_book_spine_iou0.5_book_spine.json` 和 `/home/book/book01/data/dataset_test/lin_0001/instances_default.json` 的 category 是 `{"id": 1, "name": "book_spine"}`。
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_01_pretag.py:28` 的推理 prompt 是 `"book spine"`。

## 兼容策略

- 训练: 保留人工训练 COCO 中的 `"book spine"`，不改原始标注。
- 推理/CVAT 导出: 默认 category name 使用 `"book_spine"`。
- 训练 prompt: `scripts/training_preflight.py --training-prompt "book spine"` 会在 runtime YAML 中写入 SAM3 loader 的 `coco_json_loader.prompts` 覆盖。
- 预检: 接受 `"book spine"` 和 `"book_spine"`，当 category 和 prompt 不同时输出说明但不报错。
- 回退规则: 用户显式输入 training prompt 时使用 `manual_override`；未输入时使用 COCO category name，来源为 `category_name_fallback`。

## 风险

- 如果未来某个新增训练适配层按 category name 精确匹配 `"book_spine"`，当前训练 COCO 的 `"book spine"` 会被过滤。新增适配层必须显式支持兼容别名或 prompt override，不得静默修改人工原始 COCO。
