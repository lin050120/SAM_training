# SAM3 书脊训练与推理中文使用说明

> 语言版本：**中文** | [日本語](USER_GUIDE_JA.md)

适用项目：`/home/book/book01`

适用 SAM3 源码：`/home/book/sam301`

适用 Conda 环境：`sam301`

UI 启动命令：

```bash
cd /home/book/book01
conda run -n sam301 python /home/book/book01/app.py
```

本文档按当前代码实现编写。不要把历史文档中的旧 `sam3` 环境、旧 run、旧命令直接拿来执行。

## 1. 项目用途

本项目用于书脊实例分割的 SAM3 工作流，包含：

- 书脊图片推理；
- raw mask / NMS mask 输出；
- COCO / CVAT 导出；
- 训练配置预检；
- 一次性训练启动；
- 训练状态、日志、summary、checkpoint 查看；
- trainer checkpoint 导出为 inference checkpoint；
- inference checkpoint 严格加载并用于推理。

当前阶段的重点是验证训练和推理流程是否完整可用，不是证明模型效果提升。

必须明确：

- 当前 184 张图片、7185 个 annotations 的数据集是 **SAM3 machine pre-annotation smoke dataset**。
- 该数据集不是人工修正的 ground truth。
- `human_reviewed=false`。
- `allowed_for_formal_training=false`。
- 当前只允许 `smoke + max_epochs=1` 的流程测试。
- 当前数据不能用于正式模型效果评价。
- 训练日志里的 bbox AP 不能解释为 mask 质量，也不能解释为微调效果提升。
- 正式训练前必须先制作人工修正标注数据，并重新登记 dataset identity。

## 2. 目录和重要文件

主要路径：

| 用途 | 路径 |
|---|---|
| 项目根目录 | `/home/book/book01` |
| SAM301 源码 | `/home/book/sam301` |
| Conda 环境 | `sam301` |
| Python | `/home/book/anaconda3/envs/sam301/bin/python` |
| UI 入口 | `/home/book/book01/app.py` |
| 训练基础 YAML | `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml` |
| 训练 wrapper | `/home/book/book01/scripts/launch_sam3_training.py` |
| 训练 preflight CLI | `/home/book/book01/scripts/training_preflight.py` |
| 推理 CLI | `/home/book/book01/scripts/run_unified_inference.py` |
| checkpoint exporter | `/home/book/book01/scripts/export_sam3_inference_checkpoint.py` |
| SAM301 patch manifest | `/home/book/book01/config/sam301_patch_manifest.json` |
| SAM301 patch 文件 | `/home/book/book01/patches/sam301_trainer_grad_accum_loss_scaling.patch` |
| dataset identity registry | `/home/book/book01/data_manifests/dataset_identity_registry.json` |
| 当前 smoke split | `/home/book/book01/data/formal_book_spine_sam3_dataset` |
| 原始图片集合 | `/home/book/book01/data/dataset_raw` |
| 训练 run 根目录 | `/home/book/book01/runs/training` |
| 推理 run 根目录 | `/home/book/book01/runs/inference` |
| 人工/审查输出 | `/home/book/book01/runs/review_artifacts` |
| 文档目录 | `/home/book/book01/docs` |

允许日常创建或修改：

- 新的 `runs/training/<run_id>/`；
- 新的 `runs/inference/<run_id>/`；
- `runs/review_artifacts/` 下的临时人工验收图；
- 新文档；
- 明确需要提交的 manifest 或报告。

不要覆盖或删除：

- `/home/book/sam301/sam3.pt`；
- 历史 `runs/training/*/checkpoints/checkpoint.pt`；
- 历史 `runs/training/*/checkpoints/inference_model.pt`；
- 原始 COCO；
- 原始图片；
- 历史 run；
- `data_manifests/formal_dataset_manifest.json`；
- `data_manifests/formal_split_manifest.json`；
- `/home/book/sam301` 源码，除非有单独审批。

不要把以下内容提交进 Git：

- `runs/`；
- `data/`；
- 图片；
- checkpoint；
- 大日志；
- `review_artifacts/`。

## 3. 启动前环境检查

所有命令建议在普通终端运行，不要在 GPU 不可见的受限 agent sandbox 里判断训练可用性。

### 3.1 进入项目目录

```bash
cd /home/book/book01
pwd
```

正常预期：

```text
/home/book/book01
```

如果不是这个路径，后续相对路径命令可能读错文件。

### 3.2 检查 Git

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

正常预期：

- branch 为 `e3-formal-training-prep`；
- HEAD 是当前审查后的提交；
- `git status --short` 只显示你明确知道的未跟踪文档或为空。

异常含义：

- branch 不对：不要训练，先确认是否切错分支；
- 有未知源码改动：不要训练，先审查 diff；
- 不要用 `git reset --hard` 或 `git clean` 直接清理，除非你完全确定要丢弃什么。

### 3.3 检查 Python、PyTorch、CUDA、SAM3 import

```bash
env -u PYTHONPATH \
conda run -n sam301 python - <<'PY'
from pathlib import Path
import sys
import torch
import sam3

print("python:", sys.executable)
print("python version:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("sam3:", Path(sam3.__file__).resolve())
PY
```

正常预期：

```text
python: /home/book/anaconda3/envs/sam301/bin/python
cuda: True
device_count: 1
gpu: NVIDIA GeForce RTX 5090
sam3: /home/book/sam301/sam3/__init__.py
```

异常含义：

- `sam3` 指向 `/home/book/sam3/...`：新环境绑定错误，不能训练；
- `cuda: False`：如果在 Codex/Claude sandbox 里出现，可能只是 sandbox 没映射 GPU；请在普通终端重跑；
- 普通终端仍 `cuda: False`：不要训练，检查 NVIDIA driver、PyTorch CUDA build、Conda 环境；
- `python` 不是 `/home/book/anaconda3/envs/sam301/bin/python`：没有使用新环境。

### 3.4 检查 GPU

```bash
nvidia-smi
```

正常预期：

- 能看到 NVIDIA GeForce RTX 5090；
- driver 正常；
- 当前没有未知的大型 compute 进程；
- 桌面图形占用少量显存可以接受。

异常含义：

- `nvidia-smi` 不能通信：普通终端下不应训练；
- 显存被未知任务大量占用：不要杀进程，先确认进程来源。

### 3.5 检查 SAM301 patch 状态

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py --json status
conda run -n sam301 python scripts/manage_sam301_patch.py --json verify
```

正常预期：

```json
{
  "state": "PATCHED",
  "actual_sha256": "bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2"
}
```

`verify` 应返回 `"ok": true`。

异常含义：

- `UNPATCHED`：当前 trainer 没有应用 loss-scaling patch，不能训练；
- `UNKNOWN`：hash 不匹配，不能训练；
- `MISSING`：目标文件不存在，不能训练；
- 不要手工编辑 `/home/book/sam301/sam3/train/trainer.py`。

### 3.6 检查 exporter 和 UI import

```bash
conda run -n sam301 python - <<'PY'
import app
from core.checkpoint_export import identify_checkpoint
print("app import: OK")
print("exporter import: OK")
print("base:", identify_checkpoint("/home/book/sam301/sam3.pt").type)
print("trainer:", identify_checkpoint("/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt").type)
print("inference:", identify_checkpoint("/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt").type)
PY
```

正常预期：

```text
app import: OK
exporter import: OK
base: base
trainer: trainer
inference: inference
```

异常含义：

- import 失败：先看 traceback，不要启动 UI 或训练；
- checkpoint 类型不对：确认路径是否写错。

## 4. 启动 UI

启动命令：

```bash
cd /home/book/book01
conda run -n sam301 python /home/book/book01/app.py
```

当前 `app.py` 固定：

- `server_name="127.0.0.1"`；
- `server_port=7860`；
- `share=False`。

浏览器访问：

```text
http://127.0.0.1:7860
```

UI 顶部标题：

```text
SAM3 Fine-tuning 工具
```

当前 Tab：

- `推理任务配置`
- `历史运行记录`
- `结果查看`
- `CVAT 导出`
- `训练预检`
- `数据集登记`
- `Checkpoint 评估`

停止 UI：

- 在启动 UI 的终端按 `Ctrl+C`；
- 如果有活动训练任务，进程管理器会尝试在正常退出时停止训练进程组；
- `kill -9`、断电、内核崩溃无法保证清理。

UI 启动失败时检查：

- 当前目录是否 `/home/book/book01`；
- Conda 环境是否 `sam301`；
- `app import` 是否成功；
- 7860 是否已被占用；
- Gradio 是否能 import。

不要随意改端口或 share 设置，除非有明确需要。

## 5. 数据准备和数据身份

训练数据需要：

- 图片目录可读；
- COCO `annotations.json` 可读；
- COCO `images[].file_name` 能在图片根目录下找到；
- categories 非空（预检取 id 最小的 category 作为目标类别；val 的 categories
  必须包含该类别。目标可以是任何名称，例如 `book_spine` 或 `cable`）；
- segmentation 可被当前流程解析；
- train 和 val 均非空。

正式（多 epoch）训练还要求该数据集已在
`data_manifests/dataset_identity_registry.json` 中登记为
`allowed_for_formal_training=true`；未登记的数据集（包括新目标数据集）
只能跑 smoke 模式且 `max_epochs<=1`。

当前 smoke 数据路径：

```text
/home/book/book01/data/formal_book_spine_sam3_dataset/train/annotations.json
/home/book/book01/data/formal_book_spine_sam3_dataset/val/annotations.json
/home/book/book01/data/formal_book_spine_sam3_dataset/test/annotations.json
/home/book/book01/data/dataset_raw
```

当前数据身份：

```text
annotation_source = sam3_machine_preannotation
human_reviewed = false
independently_corrected_gt = false
intended_use = pipeline_smoke_test
allowed_for_formal_training = false
allowed_for_model_evaluation = false
max_epochs_without_human_review = 1
```

程序如何知道数据身份：

- 读取 `data_manifests/dataset_identity_registry.json`；
- 用解析后的 train/val annotation path 匹配 registry；
- registry 记录 manifest SHA256；
- 不是运行时自动看 mask 形状判断；
- 不是靠文件名猜测。

未登记数据集默认行为：

- 视为 `human_reviewed=false`；
- 视为 `allowed_for_formal_training=false`；
- 只允许 smoke one-epoch；
- 这是 fail-closed 策略。

当前统计：

| 项目 | 数量 |
|---|---:|
| image files | 184 |
| unique images | 136 |
| annotations | 7185 |
| exact duplicate groups | 44 |
| train image files | 160 |
| train unique images | 120 |
| train annotations | 6749 |
| val image files | 12 |
| val unique images | 12 |
| val annotations | 253 |
| test image files | 12 |
| test unique images | 4 |
| test annotations | 183 |

这些数字只描述当前 smoke 数据，不代表未来正式数据。

## 6. Smoke 和 Formal 模式

| 模式 | 用途 | 当前 machine pre-annotation 数据是否允许 |
|---|---|---|
| `smoke` | 流程测试、环境测试、one-epoch 验收 | 只允许 `max_epochs=1` |
| `formal` | 正式训练 | 当前拒绝 |

当前数据规则：

| 设置 | 结果 |
|---|---|
| `training_mode=smoke`, `max_epochs=1` | 允许 |
| `training_mode=smoke`, `max_epochs>1` | 拒绝 |
| `training_mode=formal`, 任意 epoch | 拒绝 |

拒绝发生在：

- runtime YAML 写入前；
- launch token 生成前；
- 训练子进程启动前。

UI 中未审核数据通过 one-epoch smoke preflight 时，应看到类似提示：

```text
SMOKE — 数据未经人工审核，非正式训练结果
```

或者中文/英文混合的数据身份 warning。

如果你已经有人工审核过的新目标数据集（例如 cable），先到 UI 的
`数据集登记` Tab 选择数据集根目录并写入登记，再回到 `训练预检` Tab 选择
`training mode=formal`。具体步骤见 `docs/DATASET_REGISTRATION_UI_CN.md`。

## 7. Preflight 操作

进入 UI 的 `训练预检` Tab。

阶段 A 字段：

| UI label | 当前含义 | 常用值 |
|---|---|---|
| `authoritative config` | 基础训练 YAML | `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml` |
| `initial checkpoint` | 初始化权重 | `/home/book/sam301/sam3.pt` |
| `train images` | train 图片根目录 | `/home/book/book01/data/dataset_raw` 或默认小 smoke 数据 |
| `train COCO` | train COCO | `/home/book/book01/data/formal_book_spine_sam3_dataset/train/annotations.json` |
| `val images` | val 图片根目录 | `/home/book/book01/data/dataset_raw` 或默认小 smoke 数据 |
| `val COCO` | val COCO | `/home/book/book01/data/formal_book_spine_sam3_dataset/val/annotations.json` |
| `training prompt` | SAM3 文本 prompt，可手动填写新目标，例如 `cable` | `book spine` 或 `cable` |
| `output root` | 训练输出根目录 | `/home/book/book01/runs/training` |
| `max_epochs` | epoch 数；当前 smoke 填 1 | `1` |
| `train batch size` | micro batch size | `1` |
| `gradient accumulation steps` | 梯度累积步数 | `4` |
| `learning rate` | 留空则沿用基础 YAML | 留空 |
| `num_workers` | 留空则沿用基础 YAML | 留空 |
| `num_gpus` | GPU 数 | `1` |
| `training mode` | `smoke` 或 `formal` | 当前选 `smoke` |

当前 UI 没有 resume 字段。

点击按钮：

```text
运行训练预检 (不会启动训练)
```

preflight 会检查：

- 基础 YAML 是否存在；
- train/val 图片和 COCO 是否存在；
- train COCO 是否有 category，val COCO 是否包含同一个目标 category；
- missing images；
- training prompt；如果 prompt 与 COCO category 不同，会 warning 但不直接拒绝；
- `max_epochs`、batch size、gradient accumulation；
- effective batch size；
- train 图片数量是否小于 effective batch；
- train 图片数量不能整除 effective batch 时 warning；
- dataset identity；
- smoke/formal guard；
- output root canonical allowlist；
- SAM301 patch/hash；
- sam3 import path；
- Hydra validate-only；
- per-run distributed port；
- provenance；
- runtime YAML；
- command.txt。

正常通过时：

- `预检状态` 显示 `预检通过，可以启动训练。`；
- 未审核数据会附带 `[SMOKE — 数据未经人工审核，非正式训练结果]`；
- `预检结果` JSON 中应看到：
  - `errors: []`
  - `runtime_config_path`
  - `run_dir`
  - `command`
  - `dataset_identity`
  - `training_provenance`
  - `distributed.master_port`

preflight 通过后会在新 run 下生成：

```text
config/runtime_config.yaml
dataset_info.json
training_config_summary.json
provenance.json
command.txt
logs/
checkpoints/
```

preflight 不会启动训练。

## 8. 启动 Smoke 训练

当前允许的 smoke 训练参数：

```text
max_epochs = 1
train_batch_size = 1
gradient_accumulation_steps = 4
effective_batch_size = 4
num_gpus = 1
resume = 当前 UI 未实现
```

启动步骤：

1. 先完成 preflight，确认 `errors: []`。
2. 勾选：

   ```text
   我确认这将启动 GPU 训练任务。
   ```

3. 点击：

   ```text
   启动训练
   ```

正式 launcher：

- UI 调用 `ui.training_preflight_page.start_training(...)`；
- 训练进程由 `training_process_manager` 管理；
- 实际命令来自 preflight 的 `command.txt`；
- wrapper 是 `/home/book/book01/scripts/launch_sam3_training.py`；
- wrapper 会把 run 内 runtime YAML 交给官方 `sam3.train.train.main()`。

不要直接裸跑：

```bash
/home/book/sam301/sam3/train/train.py
```

原因：

- 会绕过 launch token；
- 会绕过 ProcessManager；
- 会绕过 summary 自动生成；
- 会绕过 import / patch / provenance / port guard。

token 机制：

- preflight 成功后生成一次性 launch token；
- 点击 `启动训练` 且所有服务端校验通过后，token 在创建子进程前被消费；
- 同一个 preflight 不能重复启动；
- completed、failed、cancelled 后都不能复用旧 preflight；
- 如果启动失败，也需要重新 preflight。

训练中可看到：

- `训练任务状态`；
- `stdout / stderr`；
- `监控信息`；
- 训练结束后的 `training_summary.json`。

不要自动重试。失败后先保留 run 和日志。

## 9. 查看训练状态

UI 中查看：

- `训练任务状态`
- `stdout / stderr`
- `监控信息`
- `training_summary.json`

磁盘中查看：

```bash
RUN=/home/book/book01/runs/training/<run_id>
ls -la "$RUN"
cat "$RUN/command.txt"
cat "$RUN/training_config_summary.json"
cat "$RUN/provenance.json"
cat "$RUN/training_summary.json"
ls -lh "$RUN/checkpoints"
tail -n 100 "$RUN/logs/book_spine/log.txt"
```

如果训练的是非书脊目标，日志目录会按 prompt/category 生成安全 task slug，例如：

```bash
tail -n 100 "$RUN/logs/cable/log.txt"
```

状态含义：

| status | 含义 |
|---|---|
| `running` | 进程仍在运行 |
| `completed` | 退出码 0 |
| `failed` | 非 0 退出码 |
| `cancelled` | 用户停止或 shutdown 取消 |

一次成功 one-epoch 的判断标准：

- `training_summary.json.status == "completed"`；
- `exit_code == 0`；
- 完成 1 epoch；
- `checkpoints/checkpoint.pt` 存在且非 0；
- `discovered_checkpoint_files` 列出当前 run 下 checkpoint；
- `/home/book/sam301/sam3.pt` 没有变化；
- 没有残留训练进程；
- 没有残留监听端口；
- 输出都在当前 run 目录内。

检查残留进程：

```bash
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml|[t]orchrun"
```

正常预期：无输出。

检查基础 checkpoint 是否被改动：

```bash
stat -c 'path=%n size=%s modified=%y permissions=%A' /home/book/sam301/sam3.pt
sha256sum /home/book/sam301/sam3.pt
```

正常预期：与训练前记录一致。

当前已知成功 run 示例：

```text
/home/book/book01/runs/training/2026-07-03_14-42-27
```

该 run：

- `status=completed`；
- `exit_code=0`；
- `max_epochs=1`；
- `train_batch_size=1`；
- `gradient_accumulation_steps=4`；
- `effective_batch_size=4`；
- 有 `checkpoints/checkpoint.pt`；
- 有 `checkpoints/inference_model.pt`。

## 10. 三种 checkpoint 的区别

不要只看 `.pt` 扩展名判断 checkpoint 类型。

### 10.1 Base checkpoint

路径：

```text
/home/book/sam301/sam3.pt
```

用途：

- 初始化模型；
- 可用于原始 SAM3 推理；
- 不能覆盖；
- 训练流程只应该读取它，不应写它。

### 10.2 Trainer / resume checkpoint

常见路径：

```text
/home/book/book01/runs/training/<run_id>/checkpoints/checkpoint.pt
```

内容：

- `model`
- `optimizer`
- `epoch`
- `scaler`
- `steps`
- 其他 resume 状态。

用途：

- 训练保存；
- 理论上用于 resume。

当前限制：

- UI 没有实现可靠 resume 操作；
- trainer checkpoint 不能直接用于正常推理入口；
- 直接传给推理会被 `Sam3Adapter` 拒绝。

### 10.3 Inference checkpoint

常见路径：

```text
/home/book/book01/runs/training/<run_id>/checkpoints/inference_model.pt
```

内容：

- `format = "sam3_inference"`
- `format_version = 1`
- `model`
- `metadata`

用途：

- 用于推理；
- 由 exporter 从 trainer checkpoint 生成；
- 加载时 strict 检查 key、shape、missing/unexpected。

## 11. 导出 inference checkpoint

脚本：

```text
/home/book/book01/scripts/export_sam3_inference_checkpoint.py
```

查看参数：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py --help
```

当前参数：

```text
--input INPUT
--output OUTPUT
--base-checkpoint BASE_CHECKPOINT
--no-base-diff
--overwrite
--dataset-identity-json DATASET_IDENTITY_JSON
--json
```

实际示例：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt \
  --output /home/book/book01/runs/review_artifacts/manual_export/inference_model.pt \
  --json
```

建议输出放到：

```text
/home/book/book01/runs/review_artifacts/<your_review_name>/
```

这样不会进入 Git，也不会覆盖历史 run。

正常输出应包含：

```text
matched_tensors = 1134
matched_parameters = 841689398
coverage_ratio = 1.0
missing_keys = []
unexpected_keys = []
base_diff.changed_tensors > 0
```

安全规则：

- 不允许输出路径等于 source trainer checkpoint；
- 不允许覆盖 `/home/book/sam301/sam3.pt`；
- 已存在 output 时默认拒绝，除非明确使用 `--overwrite`；
- coverage 太低会失败；
- shape mismatch 会失败；
- strict load 失败会失败；
- 导出权重如果和 base 完全相同会失败。

注意：

- 当前 CLI 只有你传 `--dataset-identity-json` 时才会把 dataset identity 写入 inference checkpoint metadata；
- 如果你需要完整可追溯 metadata，请显式传入 dataset identity JSON，或等待后续工具改进。

## 12. 推理操作

当前真实推理入口：

```text
/home/book/book01/scripts/run_unified_inference.py
```

查看参数：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/run_unified_inference.py --help
```

主要参数：

| 参数 | 含义 |
|---|---|
| `--input-dir` | 输入图片目录 |
| `--output-root` | 输出根目录，默认 `/home/book/book01/runs` |
| `--sam3-root` | SAM3 源码根目录，默认 `/home/book/sam301` |
| `--checkpoint` | base 或 inference checkpoint |
| `--prompt` | 文本 prompt |
| `--score-threshold` | 最终分数阈值 |
| `--confidence-threshold` | processor confidence threshold |
| `--dtype-mode` | `bf16`、`fp16`、`none` |
| `--device` | `cuda` 或 `cpu` |
| `--limit` | 限制处理图片数 |
| `--nms-iou-thresh` | NMS 阈值 |
| `--nms-metric` | `iou` 或 `iomin` |
| `--nms-mode` | `suppress` 或 `merge` |
| `--min-area` | 最小 mask 面积 |
| `--category-name` | COCO 类名 |

非书脊目标也使用同一入口。例如训练或推理 cable 时，COCO category 建议为
`cable`，推理 prompt 也传 `--prompt "cable"`；导出的 COCO 类名可用
`--category-name cable`。

### 12.1 使用 base checkpoint 推理

```bash
cd /home/book/book01
PYTHONPATH=/home/book/sam301 \
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/sam301/sam3.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 4 \
  --output-root /home/book/book01/runs
```

注意：当前 `data/formal_book_spine_sam3_dataset/{train,val,test}` 目录只放
`annotations.json`，图片文件没有复制到 split 目录下；COCO 的 `file_name`
指向 `data/dataset_raw/.../images`。因此推理 CLI 的 `--input-dir` 要填一个
真实存在的图片目录，例如上面的 `data/dataset_raw/20260622_231208_book_spine/images`。

### 12.2 使用 inference checkpoint 推理

```bash
cd /home/book/book01
PYTHONPATH=/home/book/sam301 \
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 4 \
  --output-root /home/book/book01/runs
```

### 12.3 不要直接用 trainer checkpoint 推理

错误示例：

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 1
```

预期结果：

- 程序应拒绝；
- 错误中提示这是 trainer checkpoint；
- 错误中提示先运行 `scripts/export_sam3_inference_checkpoint.py`。

这说明保护生效。

### 12.4 推理输出

每次推理会创建新的 run：

```text
/home/book/book01/runs/inference/<run_id>/
```

常见输出：

```text
input_images/
raw_npz/
npz_nms/
raw_visualizations/
nms_visualizations/
cvat_export/
manifest.json
run_config.json
validation_report.json
errors.json
```

查看：

```bash
RUN=/home/book/book01/runs/inference/<run_id>
find "$RUN" -maxdepth 2 -type f | sort | head -100
cat "$RUN/run_config.json"
cat "$RUN/manifest.json"
```

UI 中可在：

- `历史运行记录` 查看 run 列表；
- `结果查看` 查看原图、raw/NMS 可视化、实例表、manifest、errors；
- `CVAT 导出` 重新导出 polygon / rle / both。

确认不是 silent zero-load：

- trainer checkpoint 直接推理应被拒绝；
- inference checkpoint strict load 应成功；
- 至少少量图片输出非空 mask；
- `run_config.json` 中的 `sam3_checkpoint` 应是你指定的 checkpoint；
- non-empty mask 只证明链路可用，不证明模型效果更好。

## 13. 历史 run 和产物

训练 run 命名：

```text
/home/book/book01/runs/training/YYYY-MM-DD_HH-MM-SS
```

推理 run 命名：

```text
/home/book/book01/runs/inference/<run_id>
```

训练 run 常见文件：

```text
command.txt
config/runtime_config.yaml
dataset_info.json
training_config_summary.json
provenance.json
training_summary.json
logs/<task_slug>/log.txt
checkpoints/checkpoint.pt
checkpoints/inference_model.pt
```

默认书脊任务的 `<task_slug>` 是 `book_spine`；例如 cable 任务通常是
`logs/cable/log.txt`。

区分成功和失败：

```bash
cat /home/book/book01/runs/training/<run_id>/training_summary.json
```

看字段：

- `status`
- `exit_code`
- `errors`
- `warnings`
- `discovered_checkpoint_files`

不要删除：

- 成功 run；
- 失败 run；
- `training_summary.json`；
- `command.txt`；
- `runtime_config.yaml`；
- `checkpoint.pt`；
- `inference_model.pt`；
- 日志。

这些文件用于审计和复现，但不进入 Git。

## 14. Stop / Cancel / Resume

### 14.1 训练 Stop / Cancel

已实现。

入口：

- UI `训练预检` Tab；
- 按钮：`停止训练`。

行为：

- 调用 `training_process_manager.stop()`；
- 对整个进程组先发 `SIGTERM`；
- 超时后发 `SIGKILL`；
- 状态写为 `cancelled`；
- 尽量生成 `training_summary.json`。

限制：

- `kill -9`、断电、内核崩溃无法保证清理；
- 不要杀死来源不明的其他用户进程；
- 停止后不能复用旧 preflight。

### 14.2 推理 Stop

已实现。

入口：

- UI `推理任务配置` Tab；
- 按钮：`停止`。

行为：

- 停止当前推理进程；
- 状态显示 `stopped by user`。

### 14.3 Resume

当前 UI 未实现可靠 resume 操作。

虽然 trainer checkpoint 包含 optimizer、epoch、scaler 等 resume 信息，但当前项目没有提供经过验收的 UI resume 流程，也没有声明 RNG 完整恢复。

不要手工改 runtime YAML 去做 resume，除非先完成专门审查和验收。

## 15. 常见错误与排查

### 15.1 CUDA unavailable

现象：

```text
torch.cuda.is_available() == False
```

检查：

```bash
nvidia-smi
env -u PYTHONPATH conda run -n sam301 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

安全处理：

- 如果只在 agent sandbox 中 False，去普通终端重试；
- 普通终端仍 False，不要训练；
- 不要重装 PyTorch/CUDA/driver，除非单独批准。

### 15.2 sam3 import 指向旧目录

现象：

```text
sam3: /home/book/sam3/sam3/__init__.py
```

安全处理：

- 不要训练；
- 检查 `sam301` editable install；
- 目标必须是 `/home/book/sam301/sam3/__init__.py`。

### 15.3 patch hash mismatch / UNKNOWN / UNPATCHED

检查：

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py --json status
conda run -n sam301 python scripts/manage_sam301_patch.py --json verify
```

安全处理：

- `PATCHED` 以外不要训练；
- 不要手工编辑 trainer；
- 不要强行覆盖 UNKNOWN。

### 15.4 Hydra config error

现象：

```text
MissingConfigException
hydra config validation failed
```

检查：

```bash
conda run -n sam301 python scripts/launch_sam3_training.py \
  -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1 \
  --validate-only
```

安全处理：

- 不要直接裸跑 `sam3/train/train.py`；
- 不要把 runtime YAML 复制进 `/home/book/sam301`；
- 重新 preflight。

### 15.5 missing scratch key

现象：

- `scratch.train_batch_size` 缺失；
- `scratch.gradient_accumulation_steps` 缺失；
- 类型不是正整数。

安全处理：

- preflight 应明确失败；
- 不要用空字符串或 NaN 绕过；
- 不要手改基础 YAML。

### 15.6 effective batch 大于数据量 / zero-step guard

现象：

```text
train image count (...) is smaller than the effective batch size (...)
```

原因：

- `train_batch_size x gradient_accumulation_steps x num_gpus` 大于训练图像数；
- `drop_last=True` 会导致 0 optimizer step。

安全处理：

- 对 smoke 数据使用已验收参数 `1 x 4 x 1 = 4`；
- 不要为了绕过 guard 随意改代码。

### 15.7 port already in use

现象：

```text
TCPStore port ... already in use
```

安全处理：

- 当前代码为每个 run 分配独立 port；
- 如果仍发生，保留日志；
- 不要杀未知进程；
- 重新 preflight 创建新 run，而不是复用失败 run。

### 15.8 dataset identity 未登记

现象：

```text
no dataset identity record matched...
```

含义：

- 程序不能确认该数据是人工 GT；
- 默认按未审核数据处理。

安全处理：

- 只允许 one-epoch smoke；
- 正式数据必须新增 registry entry；
- 不要直接把旧 false 改 true。

### 15.9 formal mode 被拒绝

现象：

```text
formal training mode requested, but this dataset is not marked allowed_for_formal_training=true
```

安全处理：

- 当前数据不能 formal；
- 需要人工修正 GT 后重新登记。

### 15.10 smoke epochs 超过 1

现象：

```text
max_epochs=20 exceeds the smoke-mode limit (1)
```

安全处理：

- 当前数据只允许 `max_epochs=1`；
- 不要把 `smoke` 当正式训练使用。

### 15.11 trainer checkpoint 被用于 inference

现象：

```text
is a TRAINER checkpoint ... Export it first
```

安全处理：

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

然后推理时使用 `inference_model.pt`。

### 15.12 inference checkpoint coverage 不足 / strict load 失败

现象：

- `coverage ratio ... below the minimum`
- `shape mismatch`
- `missing`
- `unexpected`
- `strict load ... failed`

安全处理：

- 不要用 `strict=False` 绕过；
- 检查 SAM301 代码版本是否变了；
- 检查 checkpoint 是否来自当前架构；
- 保留完整 exporter 输出。

### 15.13 zero predictions

现象：

- 多张图片 raw/final mask 都是 0。

检查：

- 是否误用了 trainer checkpoint；
- 是否 prompt 错误；
- threshold 是否过高；
- input image 是否正确；
- `run_config.json` 中 checkpoint 是否为预期路径。

安全处理：

- 不要把 zero predictions 当正常通过；
- 先用 base checkpoint 和 exported inference checkpoint 各跑少量图片对照。

### 15.14 checkpoint 未生成

现象：

- `status=completed` 但 `checkpoints/` 为空。

安全处理：

- 看 `training_summary.json.warnings`；
- 看 `logs/<task_slug>/log.txt`，例如默认书脊任务为 `logs/book_spine/log.txt`，
  cable 任务通常为 `logs/cable/log.txt`；
- 不要伪造 checkpoint；
- 不要把 run 标记为成功验收。

### 15.15 summary failed

现象：

- `training_summary.json.status == "failed"`；
- `exit_code != 0`。

安全处理：

- 保留 run；
- 保存 traceback；
- 不自动重试；
- 不改训练参数后直接重跑同一 run。

### 15.16 残留进程

检查：

```bash
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml|[t]orchrun"
```

安全处理：

- 不要杀来源不明的进程；
- 确认 PID/PGID 是否属于当前 run；
- 如果是 UI 管理的任务，优先用 UI `停止训练`。

## 16. 人工功能验收步骤

以下清单可直接复制到验收记录中填写。

### A. 环境检查

操作：

```bash
cd /home/book/book01
git branch --show-current
git status --short
env -u PYTHONPATH conda run -n sam301 python -c "from pathlib import Path; import sys, torch, sam3; print(sys.executable); print(torch.cuda.is_available()); print(Path(sam3.__file__).resolve())"
conda run -n sam301 python scripts/manage_sam301_patch.py --json verify
```

预期：

- branch 为 `e3-formal-training-prep`；
- Python 为 sam301；
- 普通终端 CUDA True；
- sam3 import 指向 `/home/book/sam301/sam3/__init__.py`；
- patch verify ok。

PASS / FAIL：

备注：

### B. UI 启动

操作：

```bash
cd /home/book/book01
conda run -n sam301 python /home/book/book01/app.py
```

打开：

```text
http://127.0.0.1:7860
```

预期：

- 能看到 5 个 Tab；
- 没有 import error；
- CUDA 检测信息显示在推理页。

PASS / FAIL：

备注：

### C. 数据身份显示

操作：

- 打开 `训练预检`；
- 选择当前 smoke train/val COCO；
- `training mode=smoke`；
- `max_epochs=1`；
- 点击 `运行训练预检 (不会启动训练)`。

预期：

- preflight 通过；
- 显示未人工审核数据 warning；
- JSON 中 `dataset_identity.human_reviewed=false`；
- `allowed_for_formal_training=false`。

PASS / FAIL：

备注：

### D. formal guard

操作：

- 将 `training mode` 改为 `formal`；
- 保持当前 smoke 数据；
- 点击 preflight。

预期：

- preflight 拒绝；
- 不生成可启动 token；
- 报错提到 `allowed_for_formal_training=true`。

PASS / FAIL：

备注：

### E. smoke max_epochs guard

操作：

- `training mode=smoke`；
- `max_epochs=2` 或 `20`；
- 点击 preflight。

预期：

- preflight 拒绝；
- 报错提到 smoke-mode limit 1。

PASS / FAIL：

备注：

### F. preflight

操作：

- `training mode=smoke`；
- `max_epochs=1`；
- `train_batch_size=1`；
- `gradient_accumulation_steps=4`；
- `num_gpus=1`；
- 点击 preflight。

预期：

- `errors=[]`；
- 生成新 run；
- 生成 runtime YAML、summary config、provenance、command.txt；
- 不启动训练。

PASS / FAIL：

备注：

### G. 训练状态和历史 run

如果本轮不要求重新启动训练，可用已有 run 验证：

```text
/home/book/book01/runs/training/2026-07-03_14-42-27
```

操作：

```bash
cat /home/book/book01/runs/training/2026-07-03_14-42-27/training_summary.json
ls -lh /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints
```

预期：

- `status=completed`；
- `exit_code=0`；
- `checkpoint.pt` 非 0；
- command 使用 `sam301`。

PASS / FAIL：

备注：

### H. checkpoint 导出

操作：

```bash
mkdir -p /home/book/book01/runs/review_artifacts/manual_acceptance
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt \
  --output /home/book/book01/runs/review_artifacts/manual_acceptance/inference_model.pt \
  --json
```

预期：

- `matched_tensors=1134`；
- `coverage_ratio=1.0`；
- missing/unexpected 为空；
- 输出文件存在。

PASS / FAIL：

备注：

### I. trainer checkpoint 错误类型拒绝

操作：

```bash
conda run -n sam301 python - <<'PY'
from core.sam3_adapter import Sam3Adapter
try:
    Sam3Adapter(
        checkpoint="/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt",
        device="cpu",
    )
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
PY
```

预期：

- 报错；
- 提到 `TRAINER checkpoint`；
- 提到 exporter 命令。

PASS / FAIL：

备注：

### J. inference checkpoint 加载

操作：

```bash
conda run -n sam301 python - <<'PY'
from core.checkpoint_export import load_inference_checkpoint
model, metadata = load_inference_checkpoint(
    "/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt",
    device="cpu",
)
print("loaded")
print(metadata["key_mapping"]["matched_tensors"])
print(metadata["key_mapping"]["coverage_ratio"])
PY
```

预期：

- 输出 `loaded`；
- matched tensors 为 1134；
- coverage 为 1.0。

PASS / FAIL：

备注：

### K. 4 类图片推理

操作：

用 exported inference checkpoint 跑少量图片：

```bash
PYTHONPATH=/home/book/sam301 \
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/dataset_raw/20260622_231208_book_spine/images \
  --checkpoint /home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt \
  --prompt "book spine" \
  --device cuda \
  --limit 4 \
  --output-root /home/book/book01/runs
```

预期：

- 生成新的 `runs/inference/<run_id>`；
- 有 raw/NMS 可视化；
- 有 NPZ；
- 不应全部 zero predictions。

PASS / FAIL：

备注：

### L. mask / NPZ / 可视化输出

操作：

- 打开 UI `结果查看`；
- 选择刚才的推理 run；
- 选择图片；
- 切换 `raw` / `nms`；
- 查看实例表。

预期：

- 原图显示；
- raw/NMS 可视化显示；
- 实例表包含 annotation id、score、bbox、area；
- `errors.json` 没有致命错误。

PASS / FAIL：

备注：

### M. Git 和文件完整性检查

操作：

```bash
git status --short
git ls-files | grep -E '^(runs/|data/|experiments/|test_pic/)|\.(npz|pt|pth|ckpt)$' || true
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml|[t]orchrun" || true
```

预期：

- 没有 run/checkpoint/data 被 Git 跟踪；
- 没有残留训练进程；
- 只出现你已知的未跟踪文档或 review artifacts 不显示在 Git 中。

PASS / FAIL：

备注：

## 17. 正式训练前还需要做什么

正式多 epoch 前必须完成：

1. 用 CVAT 或其他工具制作人工修正 GT；
2. 不要直接把当前 registry 的 `false` 改成 `true`；
3. 为新数据生成新的 dataset ID；
4. 重新生成 dataset manifest；
5. 重新生成 split manifest；
6. 记录新的 SHA256；
7. 标记 `human_reviewed=true`；
8. 标记 `independently_corrected_gt=true`；
9. 标记 `allowed_for_formal_training=true`；
10. 检查 exact duplicate；
11. 固定 train/val/test；
12. 确认 test 不参与训练和调参；
13. 跑原始 SAM3 baseline；
14. 增加 mask IoU、Boundary F1 等 mask 质量指标；
15. 定义 best checkpoint 策略；
16. 审批多 epoch 参数；
17. 再启动正式训练。

当前禁止：

- 用 machine pre-annotation 做 formal multi-epoch；
- 把 bbox AP 当 mask 质量；
- 把 one-epoch smoke 结果当模型效果提升。

## 18. 安全注意事项

禁止：

- `git clean`；
- `git reset --hard`；
- `git stash` 未确认用户文件；
- 删除未知 run；
- 覆盖 `/home/book/sam301/sam3.pt`；
- 覆盖历史 `checkpoint.pt`；
- 覆盖历史 `inference_model.pt`；
- 修改历史 manifest；
- 直接编辑 checkpoint；
- 杀死不属于本项目的进程；
- 使用 `sudo` 随意修改环境；
- `pip install` / `conda install` 改环境，除非单独批准；
- 提交大型 checkpoint、图片、日志、`runs/`、`data/`；
- 裸跑 `/home/book/sam301/sam3/train/train.py` 绕过正式 launcher。

安全原则：

- 新训练必须新 run；
- 失败 run 不复用；
- 失败后不自动重试；
- 所有训练从 UI 或正式 launcher 走；
- 所有推理 checkpoint 先识别类型；
- trainer checkpoint 先 export 再推理；
- 数据没有人工审核前，只做 smoke。
