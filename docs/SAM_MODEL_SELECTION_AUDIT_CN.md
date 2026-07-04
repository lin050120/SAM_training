# SAM 模型选择与 checkpoint 导出审查记录

- Branch: `codex-stage-e4-checkpoint-evaluation`
- 审查前 commit: `5637e07256a8575a6fa63da1f275e0848f937b79`
- Dirty 状态: 有 4 个未跟踪文档文件，未修改已跟踪文件。

## 现有实现

1. checkpoint exporter 的真实入口是 `core/checkpoint_export.py::export_inference_checkpoint()`；CLI 包装为 `scripts/export_sam3_inference_checkpoint.py`。
2. trainer checkpoint 通过真实结构识别：顶层 dict 中存在 `model` dict 且存在 `optimizer`，由 `identify_checkpoint()` 返回 `CHECKPOINT_TYPE_TRAINER`。加载评估时使用 `load_trainer_checkpoint_model()`，对 fresh SAM3 image model 做 `strict=True`。
3. 当前“导出最佳推理模型”在 `ui/checkpoint_evaluation_page.py::export_best_model()`，读取 `evaluation/best_checkpoint.json` 后调用 exporter CLI，固定输出到 `<run>/checkpoints/inference_best.pt` 并带 `--overwrite`。
4. checkpoint evaluation 页面在 `ui/checkpoint_evaluation_page.py`，目前只支持刷新/评价/重评/导出 best。
5. 初始 mask / CVAT 预标注入口为 `ui/inference_page.py`，调用 `scripts/run_unified_inference.py` 和 `core/inference_run.py::run_sam3_image_directory()`。
6. SAM3 model 初始化在 `core/sam3_adapter.py::Sam3Adapter.__init__()`；当前没有全局模型缓存，每次任务构造一个新 adapter/model。
7. `/home/book/sam301/sam3.pt` 作为默认值集中定义在 `core/config.py::DEFAULT_SAM3_CHECKPOINT`，网页和 CLI 默认使用它；没有覆盖该文件的代码路径。
8. CLI 已支持显式传入 checkpoint 路径，参数名为 `--checkpoint`；语义上等同本次要求的 `--model-path`，但尚无别名和模型选择 UI。
9. 当前没有模型 registry、配置文件或最近使用模型记录；inference model 只能靠文件路径手动填写。
10. 示例 run `/home/book/book01/runs/training/2026-07-04_14-28-27` 当前实际包含 `checkpoint_5.pt`、`checkpoint_10.pt`、`checkpoint_15.pt`、`checkpoint_20.pt`、`checkpoint.pt` 和 `inference_best.pt`，未发现 `checkpoint_35.pt` / `checkpoint_40.pt`。

## 主要缺口

- 网页不能选择任意 trainer checkpoint 导出，只能导出 best。
- exporter 不写旁车 `.metadata.json`，metadata 只嵌在 `.pt` 内。
- 导出流程缺少 UI 级 checkpoint 列表、alias 显示、默认输出名和覆盖保护交互。
- 推理页没有模型发现 dropdown、目录+文件名输入、模型验证结果展示。
- 推理输出没有稳定的 SAM 模型 provenance JSON。
- 没有按模型路径/SHA 的缓存键；当前每任务新建模型避免串模型，但无法复用相同模型。
