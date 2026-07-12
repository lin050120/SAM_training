# SAM3 Fine-tuning 工作流（中文版）

> 语言版本：[English](README.md) | **中文** | [日本語](README_JA.md)

单目标 SAM3 微调工作流（默认书脊，也支持 cable 等任意新目标）。

- 简明使用说明与注意事项：[`docs/QUICK_START_CN.md`](docs/QUICK_START_CN.md)
- 中文完整使用说明：[`docs/USER_GUIDE_ZH_CN.md`](docs/USER_GUIDE_ZH_CN.md)

## 本地 Web UI（阶段 D1 / D1.1 / E1）

本地 Gradio UI 把既有 CLI 工作流（推理、历史浏览、结果查看、CVAT 导出、训练预检与编排）可视化，不重新实现任何逻辑。详见 `docs/stage_d_ui.md`（D1/D1.1）和 `docs/stage_e1_training_ui.md`（E1 训练编排与监控）。UI 顶部提供中文/日本語切换。

训练页是两阶段流程：预检只生成并校验 runtime 配置，不启动任何东西；启动按钮（服务端把关，不只是禁用控件）只有在预检通过、checkpoint/数据/runtime YAML 全部存在、当前没有其他训练任务、CUDA 可用、且用户已明确勾选确认后，才通过 `scripts/launch_sam3_training.py` 调起官方 SAM3 训练器（把 per-run runtime YAML 交给 `sam3.train.train.main()`）。编辑任何预检输入都会立即作废已存的预检结果，过期的 runtime YAML 永远不能用于启动训练。

```bash
conda run -n sam301 python /home/book/book01/app.py
```

请在普通终端启动（不要在受限的 coding-agent sandbox 里），这样 UI 进程的 CUDA 检测才反映终端真实的 GPU 可见性。只监听 `127.0.0.1:7860`（`share=False`）。UI 本身绝不静默启动真实 SAM3 训练；请求 `device=cuda` 而 UI 进程中 CUDA 不可用时，会拒绝启动推理而不是静默回退 CPU。

## SAM301 训练器补丁守卫

`/home/book/sam301` 不是 git 仓库；`sam3/train/trainer.py` 上的梯度累积 loss 缩放补丁由 `config/sam301_patch_manifest.json` 中的完整 SHA256 锁定，并由预检、启动器、训练子进程三层 fail-closed 强制校验。正式训练前（以及任何 sam301 重建后）执行：

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py verify   # 必须 exit 0 (PATCHED)
```

`status` / `apply` / `revert` 也可用；见 `docs/SAM301_PATCH_MANAGEMENT.md`。

## 数据身份（SAM3 预标注 ≠ 人工审核 GT）

当前 `data/formal_book_spine_sam3_dataset` 划分（184 图 / 7185 标注）是 **SAM3 自己的机器预标注输出**，不是人工修正的 ground truth——每个源 COCO 的 `info.description` 都写明了这一点，且 100% 的 polygon 顶点数 ≤8，与未编辑的机器导出一致。它在 `data_manifests/dataset_identity_registry.json` 中登记为 `human_reviewed=false`、`allowed_for_formal_training=false`。预检按解析后的标注路径（绝不按文件名）查询该 registry，并拒绝对它运行 `--training-mode formal` 或任何 `max_epochs>1` 的 smoke。完整说明及如何在独立人工审核后把数据集提升为正式状态，见 `docs/E3_DATASET_IDENTITY_ERRATUM.md`。

该守卫适用于**所有**数据集，包括新的非书脊目标（例如 cable 数据集）：只提供数据集路径和训练 prompt 就能跑 smoke（`max_epochs<=1`），但 formal 或多 epoch 训练还需要在人工审核后把数据集登记进 `data_manifests/dataset_identity_registry.json` 且 `allowed_for_formal_training=true`（可用 UI 的「数据集登记」页，见 `docs/DATASET_REGISTRATION_UI_CN.md`）。未登记数据集一律按未审核处理（fail-safe 默认）。

## Checkpoint 导出（trainer checkpoint → inference checkpoint）

trainer checkpoint（`checkpoints/checkpoint.pt`）不能直接传给推理入口——`Sam3Adapter` 按真实结构识别 checkpoint 类型，对 trainer checkpoint 直接拒绝并提示导出命令，因为 `sam301/model_builder.py` 的 loader 会从中静默加载零个权重（见 `docs/CHECKPOINT_EXPORT_AND_INFERENCE_CN.md`）。先导出：

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

## Checkpoint 评估（validation 选优 + test 诊断）

训练后，先在 validation 上评价每个唯一的 `checkpoint_N.pt` 加上原始 `sam3.pt` 基线，**只根据 validation** 选出最佳 checkpoint，再把同一批 checkpoint 放到 test 上作为 diagnostic-only 证据。test 指标绝不改变 `best_checkpoint.json` 或 `inference_best.pt`。评估器把原始预测 mask、匹配记录、逐实例指标、GT 快照和可视化保存在 `<run_dir>/evaluation/{validation,test}/`。UI：「Checkpoint 评估」页。文档：`docs/CHECKPOINT_EVALUATION_CN.md`。

```bash
conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \
  --run-dir /home/book/book01/runs/training/<run_id> --split all --export-best
```

## 权威 SAM3 训练配置

本工作区的权威微调配置是：

`/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`

虽然名字带 book_spine，它是**任何**单目标类别的固定基础模板：预检从不编辑它，而是渲染出 per-run 的 `runtime_config.yaml`，覆盖 prompt、数据路径、checkpoint 和输出/日志目录（`dumps/<task_slug>`、`logs/<task_slug>`）。训练其他目标（如 cable）时，把 train/val 指向对应数据集并填 training prompt 即可——不需要改 YAML。不传 `--training-prompt` 时，prompt 回落为 COCO 第一个 category 名。

预检命令（检查配置并生成训练命令，不启动训练）：

```bash
conda run -n sam301 python scripts/training_preflight.py
```

不改 COCO category 名、手动指定 SAM3 训练文本 prompt：

```bash
conda run -n sam301 python scripts/training_preflight.py \
  --training-prompt "book spine"
```

生成的训练命令如下（train.py 自己的 `-c` 是 `pkg://sam3.train` 内部的 Hydra 配置名，无法加载外部 YAML 路径，所以 wrapper 从 run 的配置目录初始化 Hydra 再调用官方 `sam3.train.train.main()`）：

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py \
  -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

预检读取权威基础配置，在 `/home/book/book01/runs/training/<run_id>/config/runtime_config.yaml` 创建 per-run 运行配置，写入 `dataset_info.json` 和 `command.txt`，并报告 batch size、GPU 数、梯度累积、有效 batch size、checkpoint、train/val 数据路径、输出目录、COCO category、请求的 training prompt、解析后的 training prompt 及 prompt 来源。它不启动 SAM3 训练。

默认运行路径：

- checkpoint：`/home/book/sam301/sam3.pt`
- train 数据：`/home/book/book01/data/book_spine_sam3_dataset/train`
- val 数据：`/home/book/book01/data/book_spine_sam3_dataset/val`
- 训练输出：`/home/book/book01/runs/training/<run_id>`

## 统一推理

Legacy NPZ 模式：

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --legacy-raw-run /home/book/book01/data/dataset_raw/20260622_231208_book_spine \
  --limit 2
```

真实 SAM3 模式，单张图：

```bash
PYTHONPATH=/home/book/sam301 conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/book_spine_sam3_dataset/test/images \
  --limit 1 \
  --checkpoint /home/book/sam301/sam3.pt \
  --prompt "book spine" \
  --device cuda
```

真实 SAM3 命令默认请求 CUDA。CUDA 不可用时快速失败；只有显式 `--device cpu` 才允许 CPU 推理。

CUDA 诊断辅助脚本：

```bash
bash scripts/check_cuda_environment.sh
```

## CVAT 导出

每次推理 run 会写出：

```text
cvat_export/
├── images/
├── annotations/
│   └── instances_default.json
├── polygon/
│   ├── instances_default.json
│   ├── validation_report.json
│   └── polygon_fidelity_report.json
└── rle/
    ├── instances_default.json
    └── validation_report.json
```

不重新运行 SAM3、重新导出并校验 CVAT 包：

```bash
conda run -n sam301 python scripts/export_cvat_package.py \
  --run-dir /home/book/book01/runs/inference/<run_id> \
  --segmentation-format both \
  --polygon-fidelity
```

分割格式：

- `polygon`：CVAT 默认兼容包。相对最终 NMS mask 是有损的。
- `rle`：从最终 NMS bool mask 精确导出。本地校验器检查解码后 mask 的逐像素一致性。
- `both`：分别写 `polygon/` 和 `rle/` COCO 文件，两种格式互不覆盖。

自动验证只有两项：pycocotools 能解码 segmentation、项目校验器通过。CVAT 实际导入和 CVAT 再导出的 mask 保真度必须手动测试。单图冒烟：把 `cvat_export/rle/instances_default.json` 和 `cvat_export/images/` 导入临时 CVAT 任务，再导出回来与 `npz_nms/im_000001.npz` 对比。
