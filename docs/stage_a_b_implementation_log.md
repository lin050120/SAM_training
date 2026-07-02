# Stage A-B Implementation Log

## Stage A: Current System Analysis

完成内容:

- 确认 `/home/book/book01` 和 `/home/book/sam301` 存在。
- 检查 Git 状态: 两个目录都有空 `.git/` 目录，但都不是有效 Git 仓库，`git status` 失败。
- 确认 Conda 环境 `sam3` 存在。
- 确认 Python `3.12.13`、PyTorch `2.10.0+cu128`。
- 确认当前 `torch.cuda.is_available()` 为 `False`。
- 确认 `/home/book/sam301/sam3/train/train.py` 存在。
- 确认用户示例 YAML 路径不存在，真实路径是 `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`。
- 确认系统默认权威 SAM3 基础训练配置为 `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`。
- 抽样确认旧 NPZ 和 COCO 格式。
- 生成 `/home/book/book01/docs/current_system_analysis.md`。

新增文件:

- `/home/book/book01/docs/current_system_analysis.md`

修改文件:

- 无。

删除文件:

- 无。

执行过的关键命令:

- `find /home/book/book01 -maxdepth 4 ...`
- `find /home/book/sam301 -maxdepth 3 ...`
- `git -C /home/book/book01 status --short --branch`
- `git -C /home/book/sam301 status --short --branch`
- `conda env list`
- `conda run -n sam3 python -c "... import torch; import sam3 ..."`
- `conda run -n sam3 python -c "... np.load(...)"`.
- `conda run -n sam3 python -c "... json.load(...)"`.

测试范围:

- 只读检查。
- 读取 1 个 NPZ 样本。
- 读取 2 个 COCO JSON 样本。

测试结果:

- NPZ 结构确认: `masks/scores/bboxes/instance_ids`。
- COCO polygon 格式确认。
- CUDA 当前不可用。
- `sam3` import 来源当前是 `/home/book/sam3/sam3`，不是 `/home/book/sam301`。

已知问题:

- 当前路径硬编码大量指向 `/home/book/book` 和 `/home/book/sam3`。
- 真实训练/推理无法在当前 CUDA 不可用状态下验证。

尚未验证:

- GPU SAM3 推理。
- 官方训练入口 dry run。

## Stage B: Unified Inference Run Directory

完成内容:

- 新增统一运行目录管理。
- 新增旧 NPZ 兼容读取和保存。
- 新增带删除原因记录的 mask NMS。
- 新增 COCO polygon 导出和一致性校验。
- 新增 raw/NMS 可视化生成。
- 新增 CLI，可从旧 `dataset_raw/<run_id>` 迁移少量样本到新结构。
- 使用旧 run 的前 2 张图片完成离线阶段 B 测试。

新增文件:

- `/home/book/book01/core/__init__.py`
- `/home/book/book01/core/run_manager.py`
- `/home/book/book01/core/npz_io.py`
- `/home/book/book01/core/mask_nms.py`
- `/home/book/book01/core/coco_export.py`
- `/home/book/book01/core/visualization.py`
- `/home/book/book01/core/inference_run.py`
- `/home/book/book01/core/config.py`
- `/home/book/book01/core/training_runner.py`
- `/home/book/book01/scripts/run_unified_inference.py`
- `/home/book/book01/scripts/training_preflight.py`
- `/home/book/book01/.gitignore`
- `/home/book/book01/README.md`
- `/home/book/book01/docs/stage_a_b_implementation_log.md`

修改文件:

- 无旧业务脚本被修改。

删除文件:

- 无。

执行过的关键命令:

```bash
conda run -n sam3 python scripts/run_unified_inference.py \
  --legacy-raw-run /home/book/book01/data/dataset_raw/20260622_231208_book_spine \
  --limit 2
```

```bash
conda run -n sam3 python -m py_compile core/*.py scripts/run_unified_inference.py
```

```bash
conda run -n sam3 python scripts/training_preflight.py
```

测试范围:

- 只处理旧 run 的前 2 张图片。
- 不加载 SAM3 模型。
- 不处理完整数据集。
- 不覆盖旧 run、旧 COCO、旧 checkpoint、人工标注。

测试结果:

- 新运行目录: `/home/book/book01/runs/inference/2026-07-02_11-37-33/`
- 生成 `input_images/`, `npz_raw/`, `npz_nms/`, `visualizations/raw/`, `visualizations/nms/`, `coco/`, `cvat_export/`, `logs/`, `run_config.json`, `manifest.json`, `validation_report.json`, `errors.json`。
- `validation_report.json`: `ok=true`, `images=2`, `annotations=45`, 无一致性错误。
- `py_compile` 通过。
- 训练预检未启动训练，解析结果: `train_batch_size=1`, `num_gpus=1`, `gradient_accumulation_steps=4`, `effective_batch_size=4`。
- 生成训练命令:
  `conda run -n sam3 python /home/book/sam301/sam3/train/train.py -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml --use-cluster 0 --num-gpus 1`

已知问题:

- 阶段 B 已由真实 SAM3 单图推理验收完成。
- 当前 COCO polygon 为有损外轮廓表示。该问题属于阶段 C 导出表示精度，不阻塞阶段 B。
- `run_config.json` 中 `created_at` 当前等于 run_id 字符串，后续应拆成 ISO 时间。
- 权威基础 YAML 仍可作为模板含历史路径；实际训练命令必须使用 runtime YAML，当前预检已覆盖 checkpoint、训练数据、验证数据和输出目录为当前工作副本路径。

尚未验证:

- 重新 NMS CLI。
- 重新 COCO/CVAT 导出 CLI。
- CVAT 实际导入 RLE `cvat_export/rle/instances_default.json`，以及导入后再导出的 mask 是否保持一致。

## Stage B2: Real SAM3 Inference Hook

完成内容:

- 新增 `core/sam3_adapter.py`，接入 `/home/book/sam301` 的 `build_sam3_image_model` 和 `Sam3Processor`。
- `scripts/run_unified_inference.py` 新增 `--input-dir` 真实推理模式，同时保留 `--legacy-raw-run`。
- 真实推理模式与 legacy 模式共用 `run_manager`, `npz_io`, `mask_nms`, `coco_export`, `visualization`, `cvat_export`。

测试结果:

- `PYTHONPATH=/home/book/sam301` 后 `sam3` import 来源正确: `/home/book/sam301/sam3/__init__.py`。
- 普通终端中真实 SAM3 单图推理成功，运行目录 `/home/book/book01/runs/inference/2026-07-02_12-28-40`。
- `im_000001.png`: raw=20, nms=19, COCO annotations=19。
- 模型加载时间: `model_load_seconds=7.995366042014211`。
- 单图推理时间: `inference_seconds=0.5144956979202107`。
- CVAT validation: `ok=true`, `errors=[]`, images=1, annotations=19。
- 阶段 B 状态: `completed`。polygon segmentation 可解码但不能逐像素还原 NMS mask，此问题归入阶段 C 导出表示。
- CUDA 诊断记录在 `/home/book/book01/docs/cuda_environment_diagnosis.md`；当前 PyTorch 是 CUDA 构建，不是 CPU-only 构建。

新增修复:

- `core/run_manager.py` 的 run logger 已设置 `logger.propagate = False`，避免自定义 stream handler 和 root logger 重复输出同一事件。
- `core/inference_run.py` 已增加下一次运行的分阶段计时记录和 CUDA synchronize。

## Training Prompt Decoupling

完成内容:

- `scripts/training_preflight.py` 支持 `--training-prompt "book spine"`。
- `core/training_runner.py` 将手动 prompt 写入 runtime YAML 的 `trainer.data.{train,val}.dataset.coco_json_loader.prompts`。
- 不修改人工 COCO，不修改 `category_id`，不强制统一 CVAT/COCO category name 与 SAM3 prompt。

测试结果:

- `conda run -n sam3 python scripts/training_preflight.py --training-prompt "book spine"` 通过。
- 预检解析 `coco_category_id=1`, `coco_category_name=book spine`, `resolved_training_prompt=book spine`, `prompt_source=manual_override`。

## Stage C: CVAT Export

完成内容:

- 新增 `core/cvat_export.py`。
- 新增 `scripts/export_cvat_package.py`。
- 支持从已有 inference run 重新导出 CVAT 包，不需要重新推理。
- 支持 `--segmentation-format polygon|rle|both`。
- polygon 和 RLE 分别输出到 `cvat_export/polygon/` 与 `cvat_export/rle/`，不会互相覆盖。
- 保留旧兼容路径 `cvat_export/annotations/instances_default.json`。
- 新增 polygon fidelity 报告和 per-instance CSV。
- 新增 NMS pair review 可视化。

测试结果:

- 使用 legacy 2 张图运行目录 `/home/book/book01/runs/inference/2026-07-02_11-37-33` 测试。
- CVAT validation: `ok=true`, images=2, annotations=45, errors=[]。
- 使用真实 run `/home/book/book01/runs/inference/2026-07-02_12-28-40` 测试 polygon/rle/both。
- polygon mode: validator `ok=true`, annotations=19, errors=[]；转换为有损表示，mean IoU=0.9731655038856，median IoU=0.9778061224489796，minimum IoU=0.8740740740740741，exact match=0/19。
- RLE mode: validator `ok=true`, annotations=19, errors=[]；decoded mask exact match=19/19。
- NMS review: `visualizations/nms_review/instance_3_vs_7.png` 和对应 JSON 已生成，semantic correctness 需要人工确认。

下一阶段计划:

- Stage C: 等待用户进行 CVAT RLE 实际导入测试；如 CVAT 不接受 RLE，再保留 polygon 作为兼容模式并在训练/评估内部使用 RLE 或 NPZ mask 保真。
- Stage D 前先补重新 NMS、重新导出 COCO 的 CLI，供 UI 调用。
