# Stage B2 Real SAM3 Inference

生成时间: 2026-07-02

## 当前状态

真实 SAM3 单图推理已在普通终端完成。

真实运行目录:

`/home/book/book01/runs/inference/2026-07-02_12-28-40`

实际日志:

```text
2026-07-02 12:28:40,867 INFO Running SAM3 inference on 1 image(s) from /home/book/book01/data/book_spine_sam3_dataset/test/images
2026-07-02 12:28:49,642 INFO im_000001.png raw=20 nms=19
2026-07-02 12:28:49,727 INFO Completed SAM3 inference run: /home/book/book01/runs/inference/2026-07-02_12-28-40
```

已补充下一次运行可用的精确计时字段。当前真实 run 是补丁前生成，因此只包含旧字段 `model_load_seconds`、每图 `inference_seconds` 和 `nms_npz_visualization_seconds`。

## 已接入的真实推理入口

- `core/sam3_adapter.py`
  - `Sam3Adapter.__init__()`: 使用 `/home/book/sam301` 源码，调用 `sam3.model_builder.build_sam3_image_model`，加载 `/home/book/sam301/sam3.pt`。
  - `Sam3Adapter.predict()`: 对单张图片执行 `Sam3Processor.set_image()` 和 `set_text_prompt()`，输出兼容旧格式的 `InstanceSet`。
- `core/inference_run.py`
  - `run_sam3_image_directory()`: 从图片目录开始执行真实 SAM3 推理，保存 raw NPZ、NMS NPZ、COCO、可视化、manifest、run_config、errors 和 CVAT 包。
  - `migrate_legacy_raw_run()`: 保留旧 NPZ 离线迁移模式。
- `scripts/run_unified_inference.py`
  - `--input-dir`: 真实 SAM3 推理模式。
  - `--legacy-raw-run`: 旧 NPZ 离线模式。

## 计划单图测试配置

测试图片:

`/home/book/book01/data/book_spine_sam3_dataset/test/images/im_000001.png`

checkpoint:

`/home/book/sam301/sam3.pt`

prompt:

`book spine`

推理参数:

- `device`: `cuda`
- `score_threshold`: 0.3
- `processor_confidence_threshold`: 0.05
- `nms_metric`: `iou`
- `nms_iou_thresh`: 0.5
- `nms_mode`: `suppress`
- `min_area`: 200
- `category_name`: `book_spine`

预计显存:

- 当前 Codex 会话中 GPU 不可用，无法测量实际 GPU 显存。
- 未执行 CPU fallback；不记录 CPU 推理估计值作为 GPU 推理时间依据。

计划命令:

```bash
PYTHONPATH=/home/book/sam301 conda run -n sam3 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/book_spine_sam3_dataset/test/images \
  --limit 1 \
  --checkpoint /home/book/sam301/sam3.pt \
  --prompt "book spine" \
  --score-threshold 0.3 \
  --confidence-threshold 0.05 \
  --device cuda \
  --nms-metric iou \
  --nms-iou-thresh 0.5 \
  --nms-mode suppress \
  --min-area 200 \
  --category-name book_spine
```

## 真实单图验收

验收报告:

`/home/book/book01/runs/inference/2026-07-02_12-28-40/acceptance_report.json`

结果:

- 输入图片: `im_000001.png`
- 原图尺寸: 1280x720
- raw NPZ keys: `masks`, `scores`, `bboxes`, `instance_ids`
- raw masks: 20
- raw scores: 20
- raw mask shape: `[20, 720, 1280]`
- raw mask dtype: `bool`
- NMS masks: 19
- NMS threshold: 0.5
- 被删除实例: source instance 7，被 source instance 3 保留项抑制，overlap 0.6423683843038682，reason `iou>=0.5`
- COCO annotations: 19
- bbox: 全部在图片范围内
- area: 19/19 与最终 NMS mask 非零像素数一致
- annotation_id: 唯一
- image_id: 有效且唯一
- manifest 路径: 已全部存在
- errors.json: `[]`
- raw/nms 可视化: 可读取
- CVAT validation: `ok=true`, `errors=[]`, images=1, annotations=19

说明:

- 当前 polygon COCO 可解码但不能逐像素还原原始 NMS mask，这是阶段 C 的导出表示精度问题，不阻塞阶段 B。
- 阶段 B 验收关注真实推理、raw NPZ、NMS NPZ、COCO 数量、bbox、area、ID、manifest 和 errors；这些检查已通过。

## CVAT legacy 验证

已对离线阶段 B 的 2 张 legacy NPZ 结果重新导出并验证 CVAT 包:

运行目录:

`/home/book/book01/runs/inference/2026-07-02_11-37-33`

命令:

```bash
conda run -n sam3 python scripts/export_cvat_package.py \
  --run-dir /home/book/book01/runs/inference/2026-07-02_11-37-33
```

结果:

- `ok`: true
- images: 2
- annotations: 45
- errors: []
- warnings: []
- category: `book_spine`

## 阶段判定

- Stage B: `completed`
- B2 真实推理入口: 已实现。
- B2 真实单图推理测试: 已在普通终端完成。
- raw NPZ 20 个实例，NMS NPZ 19 个实例，COCO 19 个 annotations。
- bbox、area、ID、manifest 和 errors 检查通过。
- Stage C: polygon mode completed with lossy mask conversion; exact RLE mode format/exactness validation completed; CVAT actual RLE import pending.

## Stage C 表示精度

Polygon mode:

- `cvat_export/polygon/instances_default.json`
- `cvat_export/polygon/validation_report.json`
- `cvat_export/polygon_fidelity_report.json`
- `cvat_export/polygon_fidelity_per_instance.csv`
- validator: `ok=true`, annotations=19, errors=[]
- polygon fidelity 是 NMS mask 到 polygon COCO 的转换损失，不是 SAM3 分割精度。
- mean IoU: 0.9731655038856
- median IoU: 0.9778061224489796
- minimum IoU: 0.8740740740740741
- exact match: 0/19

RLE mode:

- `cvat_export/rle/instances_default.json`
- `cvat_export/rle/validation_report.json`
- RLE 直接从最终 NMS bool mask 生成，不从 polygon 反推。
- validator: `ok=true`, annotations=19, errors=[]
- exact match: 19/19 by validator exact mask check.

CVAT 兼容性边界:

- 自动确认: pycocotools 可解码，项目 validator 通过。
- 尚未自动确认: CVAT 实际导入成功，CVAT 导入后再导出的 mask 是否保持一致。

## 后续计时字段

下一次真实推理会写入:

- `run_summary.json`
- `run_config.json.timings`
- `manifest[].timings`

字段包括:

- `environment_setup_seconds`
- `model_load_seconds`
- `image_read_seconds`
- `sam3_inference_seconds`
- `raw_npz_write_seconds`
- `nms_seconds`
- `nms_npz_write_seconds`
- `coco_export_seconds`
- `visualization_seconds`
- `cvat_export_seconds`
- `total_run_seconds`
- 每图 `total_image_seconds`
