# Current System Analysis

生成时间: 2026-07-02

## 1. 当前目录结构

- `/home/book/book01`: 业务副本，包含数据、旧脚本、新增工程化代码。当前有空 `.git/` 目录，但不是有效 Git 仓库，`git status` 报 `fatal: not a git repository`。
- `/home/book/sam301`: SAM3 源码副本，包含 `sam3/`、`sam3.pt`、训练入口和训练配置。同样不是有效 Git 仓库。
- 旧业务脚本主要有两套:
  - `/home/book/book01/sam3_finetune_kit/sam3_finetune_kit`
  - `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit`
- 已有原始预测运行目录: `/home/book/book01/data/dataset_raw/<run_id>/`
- 已有公共 COCO 输出: `/home/book/book01/data/cvat_import_polygon_*.json`
- 已有训练数据: `/home/book/book01/data/book_spine_sam3_dataset/{train,val,test}/annotations.json`

## 2. 主要文件用途

- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_01_pretag.py`: 旧推理入口，批量读取图片、调用 SAM3、保存图片副本、raw mask PNG、NPZ、overlay 和 `sam3_predictions.json`。关键配置在第 24-30 行，主流程在第 65-145 行。
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/sam3_adapter.py`: SAM3 推理适配器，`Sam3Adapter` 在第 48 行定义，`predict()` 在第 112 行定义，`run_sam3()` 懒加载入口在第 154 行。
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_nms.py`: 旧 mask NMS，`masks_nms()` 在第 47 行。
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_02_to_cvat_coco.py`: RLE COCO 导出，`mask_to_rle()` 第 49 行，`load_image_instances()` 第 92 行，`main()` 第 150 行。
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_02_1_to_cvat_polygon.py`: polygon COCO 导出，`mask_to_polygon()` 第 67 行，`main()` 第 152 行。
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_04_eval_paired.py`: 旧 paired evaluation，GT 转 mask 在 `gt_masks_for_image()` 第 37 行，Hungarian matching 在第 95-101 行。
- `/home/book/sam301/sam3/train/train.py`: SAM3 官方训练入口存在。
- `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`: 真实训练 YAML。先前示例中省略 `sam3/train/` 的配置路径不存在，已统一更正为该权威路径。

## 3. 当前完整数据流

旧流程:

1. `ft_01_pretag.py` 读取 `IMAGES_DIR`，第 85-86 行按 `Path.iterdir()` 后 `sorted()` 遍历图片。
2. 第 98 行调用 `run_sam3(img)`。
3. 第 98-103 行按 score、面积过滤，计算 bbox，并按 bbox x 坐标排序。
4. 第 94 行复制图片到 `images/img_XXXX.ext`。
5. 第 106-117 行逐实例保存 mask PNG，并写入 `sam3_predictions.json` 的 `instances`。
6. 第 121-133 行保存每图一个 NPZ。
7. 第 137-138 行保存 raw overlay。
8. `ft_02_to_cvat_coco.py` 或 `ft_02_1_to_cvat_polygon.py` 读取旧 run。
9. COCO 导出脚本第 151-152 行选择 raw run，第 206-207 行把 JSON 写到公共 `OUT_DIR`，导致 COCO 与对应图片/NPZ/overlay 分离。

## 4. 当前程序入口

- 推理入口: `python ft_01_pretag.py`。
- RLE CVAT 导出入口: `python ft_02_to_cvat_coco.py`。
- Polygon CVAT 导出入口: `python ft_02_1_to_cvat_polygon.py`。
- 数据集拆分入口: `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_03_build_dataset.py`。
- 旧评估入口: `python ft_04_eval_paired.py`。
- 官方训练入口: `conda run -n sam3 python /home/book/sam301/sam3/train/train.py -c /home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml --use-cluster 0 --num-gpus 1`。
- 训练预检入口: `conda run -n sam3 python /home/book/book01/scripts/training_preflight.py`。该命令只解析配置、生成 runtime YAML 和训练命令，不启动训练。

## 5. 关键函数和类

- `Sam3Adapter.__init__()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/sam3_adapter.py:48`
- `Sam3Adapter.predict()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/sam3_adapter.py:112`
- `run_sam3()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/sam3_adapter.py:154`
- `mask_to_bbox()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_01_pretag.py:41`
- `make_overlay()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_01_pretag.py:50`
- `masks_nms()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_nms.py:47`
- `mask_to_rle()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_02_to_cvat_coco.py:49`
- `mask_to_polygon()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_02_1_to_cvat_polygon.py:67`
- `gt_masks_for_image()` `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_04_eval_paired.py:37`

## 6. NPZ 的真实格式

抽样文件: `/home/book/book01/data/dataset_raw/20260622_231208_book_spine/sam3_masks_npz/img_0001.npz`

- `masks`: shape `(20, 720, 1280)`, dtype `bool`
- `scores`: shape `(20,)`, dtype `float32`, 样本范围 `0.34179688` 到 `0.8828125`
- `bboxes`: shape `(20, 4)`, dtype `int32`
- `instance_ids`: shape `(20,)`, dtype `int32`, 范围 `1..20`

NPZ 不保存原图尺寸、原文件名、polygon，也不保存 NMS 结果。尺寸和文件对应关系依赖同 run 下的 `sam3_predictions.json`。

## 7. COCO 的真实格式

已确认可用 polygon 样本:

- `/home/book/book01/data/cvat_import_polygon_20260622_231208_book_spine_iou0.5_book_spine.json`
- `/home/book/book01/data/dataset_test/lin_0001/instances_default.json`

结构:

- 顶层 key: `info`, `licenses`, `images`, `annotations`, `categories`
- `categories`: `{"id": 1, "name": "book_spine", "supercategory": ""}`
- `images`: `id`, `file_name`, `width`, `height`
- `annotations`: `id`, `image_id`, `category_id`, `segmentation`, `area`, `bbox`, `iscrowd`
- 当前成功样本的 `segmentation` 是 polygon list，不是 RLE dict。

旧 `ft_02_to_cvat_coco.py` 也支持 RLE，且第 16-19 行注明 CVAT label 需要选 mask。但现有文件名 `cvat_import_polygon_*.json` 和 `dataset_test/lin_0001/instances_default.json` 证明 polygon 格式已经成功使用。

## 8. NMS 的真实实现

旧 NMS 在 `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_nms.py:masks_nms()`。

- 使用 mask overlap，不是 bbox IoU。bbox 只在第 89 行做快速排斥。
- 排序方式: 第 72 行 `np.argsort(scores)[::-1]`，高分优先。
- 默认阈值由调用方设置，`ft_02_to_cvat_coco.py:43` 和 `ft_02_1_to_cvat_polygon.py:40` 默认 `0.5`。
- 支持 `metric="iou"` 和 `metric="iomin"`。
- 支持 `mode="suppress"` 和 `mode="merge"`。
- 不记录被删除实例和原因。
- 对相邻细书脊的风险: `iomin` 会因交集/小面积更激进，旧代码注释第 17-18 行已说明可能误抑制略有重叠的相邻目标。

## 9. 图片和标注的对应规则

- 旧 `ft_01_pretag.py` 第 85-89 行按排序后的图片列表生成 `img_0001`, `img_0002`。
- 第 94 行复制为 `images/img_XXXX.ext`。
- `sam3_predictions.json` 每条记录含 `image_id` 字符串、`file_name`、`orig_path`、`height`、`width`、`npz_path`。
- COCO 导出脚本第 166-168 行用枚举序号作为 COCO `image_id`，并使用 `rec["file_name"]`。
- 潜在风险: 原图文件名不保留在 COCO 中，重新排序或跨 run 混用时只能靠 `sam3_predictions.json` 追溯。

## 10. 当前输出目录问题

- `ft_01_pretag.py:80-83` 把图片、raw mask、NPZ、overlay 写到 `OUT_ROOT/run_id`。
- `ft_02_to_cvat_coco.py:30-32` 和 `ft_02_1_to_cvat_polygon.py:28-30` 把 COCO 写到公共 `/home/book/book/data`。
- 这导致一个 run 的 COCO、图片、NPZ、overlay 不在同一个目录，且脚本里硬编码的是 `/home/book/book/...`，与当前工作副本 `/home/book/book01` 不一致。

## 11. 可以直接复用的代码

- SAM3 推理适配思路: `Sam3Adapter`。
- NPZ 字段格式: `masks/scores/bboxes/instance_ids`。
- mask NMS 核心算法: `ft_nms.py`。
- COCO polygon/RLE 转换思路: `ft_02_to_cvat_coco.py`, `ft_02_1_to_cvat_polygon.py`。
- GT polygon/RLE 转 mask: `ft_04_eval_paired.py:37`。
- Hungarian matching 示例: `ft_04_eval_paired.py:95-101`。

## 12. 重复代码

- `load_image_instances()` 在 RLE 和 polygon 两个导出脚本中重复。
- `resolve_run_dir()` 和 `build_out_path()` 在两个导出脚本中重复。
- `mask_to_bbox_area()` 与 `mask_to_bbox()` 分散在多个脚本。
- overlay 逻辑只在 `ft_01_pretag.py` 和 `sam3_adapter.py` 自检中局部实现。

## 13. 高耦合部分

- 旧脚本通过顶部常量配置路径和参数，缺少 CLI/API 层。
- COCO 导出依赖 `sam3_predictions.json` 的旧 run 结构。
- `sam3_adapter.py` 的 checkpoint 写死为 `/home/book/sam3/sam3.pt`，与当前允许修改的 `/home/book/sam301/sam3.pt` 不一致。
- 模型懒加载用模块级全局 `_adapter`，不利于 UI 停止任务和多 checkpoint 评估。

## 14. 潜在 bug

- `ft_01_pretag.py`、`ft_02*.py`、`ft_04_eval_paired.py` 多处硬编码 `/home/book/book` 或 `/home/book/data`，当前副本路径为 `/home/book/book01`。
- 旧 COCO 导出不记录 NMS 删除原因，后续无法审计被删实例。
- 旧 `ft_04_eval_paired.py` 读取预测时用 raw mask PNG，不使用 NMS 后实例，可能与 CVAT 导出的评估口径不一致。
- Conda 环境中 `import sam3` 指向 `/home/book/sam3/sam3/__init__.py`，不是 `/home/book/sam301`。
- 当前 `torch.cuda.is_available()` 为 `False`，无法完成 GPU 推理/训练验证。

## 15. 路径硬编码问题

- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_01_pretag.py:25-26`
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_02_to_cvat_coco.py:30-32`
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/ft_02_1_to_cvat_polygon.py:28-30`
- `/home/book/book01/sam3_finetune_kit_debug/sam3_finetune_kit/sam3_adapter.py:35`
- `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml:29-32`, `:310`

## 16. SAM3 官方代码和业务代码关系

业务代码通过 `from sam3.model_builder import build_sam3_image_model` 和 `Sam3Processor` 调用 SAM3。训练仍应调用官方入口 `/home/book/sam301/sam3/train/train.py`，不要重写训练器。

注意: 当前 Conda import 的 SAM3 是 `/home/book/sam3`，不是 `/home/book/sam301`。后续运行训练或推理时需要显式处理 `PYTHONPATH=/home/book/sam301` 或确认安装来源。

## 17. 训练配置中的真实数据路径字段

真实 YAML: `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`

- `paths.dataset_root`: 第 30 行，当前 `/home/book/book/data/book_spine_sam3_dataset`
- `paths.experiment_log_dir`: 第 31 行，当前 `/home/book/book/experiments/book_spine_sam3_r1`
- train images: 第 268 行 `${paths.dataset_root}/train/images/`
- train annotations: 第 269 行 `${paths.dataset_root}/train/annotations.json`
- val images: 第 288 行 `${paths.dataset_root}/val/images/`
- val annotations: 第 289 行 `${paths.dataset_root}/val/annotations.json`
- eval GT: 第 327 行 `${paths.dataset_root}/val/annotations.json`

这些路径目前指向 `/home/book/book`，不是 `/home/book/book01`。

## 18. checkpoint 的真实保存规则

- 初始 checkpoint: YAML 第 310 行 `checkpoint_path: /home/book/sam3/sam3.pt`，当前应改为运行时配置指向 `/home/book/sam301/sam3.pt` 或用户选择 checkpoint。
- 保存目录: YAML 第 384-386 行 `trainer.checkpoint.save_dir: ${launcher.experiment_log_dir}/checkpoints`, `save_freq: 5`。
- `trainer.skip_saving_ckpts: false` 在第 234 行。

## 19. 建议的新架构

已开始落地:

- `/home/book/book01/core/config.py`: 权威默认路径，包含默认 SAM301 根目录、训练入口和 book-spine fine-tune 配置路径。
- `/home/book/book01/core/run_manager.py`: 统一 run 目录、日志、runtime 信息。
- `/home/book/book01/core/npz_io.py`: 兼容旧 NPZ 的读取/保存。
- `/home/book/book01/core/mask_nms.py`: 带删除原因的 mask NMS。
- `/home/book/book01/core/coco_export.py`: COCO polygon 导出和一致性检查。
- `/home/book/book01/core/visualization.py`: raw/nms overlay。
- `/home/book/book01/core/inference_run.py`: 旧 raw run 迁移到新 run 结构，后续接 SAM3 推理。
- `/home/book/book01/core/sam3_adapter.py`: 真实 SAM3 图片推理适配器，使用 `/home/book/sam301` 源码和 `/home/book/sam301/sam3.pt` checkpoint。
- `/home/book/book01/core/cvat_export.py`: CVAT COCO 包重新导出和验证。
- `/home/book/book01/core/training_runner.py`: 训练启动前配置解析、路径审计、COCO 摘要、runtime YAML 生成、effective batch size 计算和命令生成，不启动训练。
- `/home/book/book01/scripts/run_unified_inference.py`: 阶段 B CLI。
- `/home/book/book01/scripts/export_cvat_package.py`: 从 inference run 重新导出/验证 CVAT 包。
- `/home/book/book01/scripts/training_preflight.py`: 训练预检 CLI。

后续应继续拆分 `sam3_adapter.py`、真实推理管线、CVAT ZIP、训练 runner 和 checkpoint evaluator。

## 20. 实施阶段

阶段 A 已完成:

- 检查目录和 Git 状态。
- 确认 Conda 环境 `sam3`、Python 3.12.13、PyTorch 2.10.0+cu128。
- 确认 CUDA 当前不可用。
- 确认训练入口存在，用户示例 YAML 路径不存在，真实 YAML 在 `sam3/train/configs/book_spine/`。
- 抽样确认 NPZ 和 COCO 格式。
- 新增训练配置防错预检，默认权威配置为 `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`。

阶段 B: `completed`

- 新增统一 run 目录结构。
- 从旧 raw run 迁移 2 张样本到 `/home/book/book01/runs/inference/2026-07-02_11-37-33/`。
- 新 run 中包含 `input_images/`, `npz_raw/`, `npz_nms/`, `visualizations/raw`, `visualizations/nms`, `coco/instances_default.json`, `cvat_export/annotations/instances_default.json`, `manifest.json`, `run_config.json`, `validation_report.json`, `errors.json`, `logs/run.log`。
- 验证结果: 2 images, 45 annotations, `validation_report.json` 中 `ok: true`。

阶段 B2 真实推理:

- 真实入口: `scripts/run_unified_inference.py --input-dir ...`
- 普通终端中真实单图 SAM3 测试已完成，运行目录 `/home/book/book01/runs/inference/2026-07-02_12-28-40`。
- `im_000001.png` raw=20, nms=19, COCO annotations=19，CVAT validation `ok=true`。
- bbox、area、ID、manifest 和 errors 检查通过。
- polygon segmentation 可解码但不能还原为完全相同的 NMS mask；这是阶段 C 的导出表示问题，不阻塞阶段 B。
- 真实推理入口默认请求 CUDA；如果 CUDA 不可用会立即报错，只有显式 `--device cpu` 才允许 CPU。
- 详情见 `/home/book/book01/docs/stage_b2_real_inference.md`。

阶段 C:

- polygon mode: completed with lossy mask conversion。真实 run validator `ok=true`, annotations=19, errors=[]；polygon fidelity mean IoU=0.9731655038856, median IoU=0.9778061224489796, minimum IoU=0.8740740740740741, exact match=0/19。
- exact RLE mode: implemented and format/exactness validated。RLE 直接从最终 NMS bool mask 生成，validator `ok=true`, exact match=19/19。
- CVAT RLE actual import: pending。当前代码只能自动确认 pycocotools 可解码和项目 validator 通过，不能声称 CVAT 已实际导入成功。

CUDA 诊断:

- 当前 PyTorch 是 CUDA 构建: `torch_version=2.10.0+cu128`, `torch.version.cuda=12.8`, `USE_CUDA=ON`。
- 当前 Codex 会话中 `nvidia-smi` 无法与 driver 通信，只看到空的 `/dev/nvidia-caps` 目录，没有可用 GPU 设备节点，`torch.cuda.is_available()=False`。
- 详见 `/home/book/book01/docs/cuda_environment_diagnosis.md`。

## 21. 兼容性风险

- 当前 polygon COCO 是有损外轮廓表示，不能逐像素还原 NMS mask。已新增 RLE 精确导出模式用于保真。
- 当前阶段 B 用旧 NPZ 离线迁移测试，没有加载 SAM3 模型。真实 SAM3 import 已通过 `PYTHONPATH=/home/book/sam301` 指向 `/home/book/sam301/sam3/__init__.py`。
- 新 NMS NPZ 增加了 `source_instance_ids` 字段，但保留旧核心字段 `masks/scores/bboxes/instance_ids`，读取端保持兼容。
- 训练 COCO category name、CVAT/推理 COCO category name 和 SAM3 prompt 已解耦。训练预检支持 `--training-prompt`，并在 runtime YAML 中通过 SAM3 loader 的 `prompts` 覆盖 query text。

## 22. 尚未确认的信息

- 普通终端中 CUDA 是否可用；当前仅确认 Codex 会话中 CUDA 不可用。
- CVAT 当前生产任务是否接受 RLE COCO，以及导入后再导出的 mask 是否保持一致。
- 官方训练命令在当前 CUDA 不可用状态下不能实际训练验证。
- 训练 YAML 中 `/home/book/book` 路径是历史模板路径；实际训练必须通过 runtime YAML 写入 `/home/book/book01` 和 `/home/book/sam301` 下的解析路径。

## 23a. 阶段 D1: 本地 Web UI

- Git: 本项目从阶段 A 起就没有真正的 Git 仓库（`.git/` 为空目录）。阶段 D1 开始前已经 `git init` 并提交了一次基线快照（`core/`, `scripts/`, `tests/`, `docs/`, `README.md`），后续在 `claude-stage-d` 分支上开发，不包含 `data/`, `runs/`, `experiments/`, `test_pic/` 等数据/输出目录。
- UI 框架: Gradio。检查时 `gradio`/`streamlit`/`fastapi`/`uvicorn` 均未安装，经用户批准安装 `gradio`；因环境里已有的 `huggingface_hub 1.19.0`（`sam3`/`timm` 依赖）与 gradio 4.x 不兼容，二次经用户批准把版本范围从 `gradio>=4.0,<5.0` 调整为 `gradio>=5.0,<6.0`，实际装到 `gradio 5.50.0`，`torch`/`sam3`/`timm` 均验证正常。
- 新增 `app.py` 和 `ui/` 目录，5 个页面（推理任务配置/历史运行记录/结果查看/CVAT 导出/训练预检）全部复用 `core/` 现有模块，没有新写第二套 NMS/COCO/CVAT/训练预检/run manager。
- 有意思的发现: 本轮在 Claude Code 会话内直接检测到真实 GPU（`nvidia-smi` 正常，`torch.cuda.is_available()=True`，`NVIDIA GeForce RTX 5090`），和此前 Codex 会话中 GPU 不可见的情况不同；但本轮仍然没有主动运行真实 SAM3 推理或训练，只做了环境检测。
- 详见 `docs/stage_d_ui.md`。

## 23b. 阶段 E1: 一键训练编排与监控

- 在阶段 D1.1 完成人工浏览器验收后，在新分支 `claude-stage-e1` 上进行（`claude-stage-d` 已提交 baseline+D1/D1.1 后创建此分支）。
- 扩展 `core/training_runner.py::inspect_training_config()`/`write_runtime_yaml()`，新增 `max_epochs`/`train_batch_size`/`gradient_accumulation_steps`/`learning_rate`/`num_workers` 覆盖，全部对照真实 YAML 字段路径确认（不是猜测的），详见 `docs/stage_e1_training_ui.md` 第 2 节。`scratch.lr_transformer` 用到 SAM3 自定义的 `times` OmegaConf resolver，预检没有注册这个 resolver（避免引入沉重的 torch/hydra 依赖），未覆盖时该值展示为 `null` 并附 warning，而不是猜测或手算。
- 新增 `ui/training_process_manager.py`：复用 `ui/process_manager.py::ProcessManager`（同一套进程组管理机制），新增训练专属的状态判定（completed/failed/cancelled）、日志指标 best-effort 解析（未经真实训练日志验证）、`training_summary.json` 生成、`validate_can_start_training()` 纯函数式启动前置条件校验。
- `ui/training_preflight_page.py` 从单阶段预检改为两阶段（预检 + 启动），参数修改会让服务端保存的预检状态立即失效，启动前置条件在服务端强制校验，不只是前端按钮禁用。
- 新增 42 个测试（`tests/test_e1_training.py`），全部使用假 `python3 -c` 命令，未启动 SAM3、未使用 GPU、未产生真实 checkpoint。
- 详见 `docs/stage_e1_training_ui.md`。

## 23. 训练配置预检结果

预检命令:

```bash
conda run -n sam3 python scripts/training_preflight.py
```

当前解析结果:

- 实际配置: `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`
- 配置存在: true
- 是否默认权威配置: true
- `train_batch_size`: 1
- `num_gpus`: 1
- `gradient_accumulation_steps`: 4
- `trainer.gradient_accumulation_steps`: 4
- `effective_batch_size`: 4
- 初始 checkpoint: `/home/book/sam301/sam3.pt`
- 训练数据: `/home/book/book01/data/book_spine_sam3_dataset/train/images`, `/home/book/book01/data/book_spine_sam3_dataset/train/annotations.json`
- 验证数据: `/home/book/book01/data/book_spine_sam3_dataset/val/images`, `/home/book/book01/data/book_spine_sam3_dataset/val/annotations.json`
- 输出目录: `/home/book/book01/runs/training/<run_id>`
- `coco_category_id`: 1
- `coco_category_name`: `book spine`
- 支持手动指定 `--training-prompt "book spine"`，预检输出 `resolved_training_prompt` 和 `prompt_source`。传入该参数时 `prompt_source=manual_override`；未传入时回退为 `category_name_fallback`。

注意: 权威基础 YAML 中仍保留历史路径作为模板内容；实际运行必须使用 runtime YAML。当前预检生成的 runtime YAML 已将 checkpoint、BPE、训练数据、验证数据和输出目录改为本次工作副本路径。
