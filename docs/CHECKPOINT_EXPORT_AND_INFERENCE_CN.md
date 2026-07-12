# Checkpoint 导出与推理（中文版）

> 语言版本：[English](CHECKPOINT_EXPORT_AND_INFERENCE.md) | **中文** | [日本語](CHECKPOINT_EXPORT_AND_INFERENCE_JA.md)

生成时间: 2026-07-03（中文版随英文版同步维护）

## 三种 checkpoint 类型（绝不按文件名猜测）

| 类型 | 示例路径 | 顶层结构 | 加载方式 |
|---|---|---|---|
| **Base** | `/home/book/sam301/sam3.pt` | 扁平 dict，无包装，key 带 `detector.*` / `tracker.*` 前缀（1156 个 tensor） | `sam3.model_builder.build_sam3_image_model(checkpoint_path=...)`（sam301 原版，未修改） |
| **Trainer/resume** | `<run_dir>/checkpoints/checkpoint.pt` | `{"model": {...}, "optimizer": {...}, "epoch": int, "loss": {...}, "steps": {...}, "scaler": {...}}` | 绝不直接用于推理。包含完整 resume 状态。 |
| **Inference** | `<run_dir>/checkpoints/inference_model.pt` | `{"format": "sam3_inference", "format_version": 1, "model": {...}, "metadata": {...}}` | `core.checkpoint_export.load_inference_checkpoint()` |

`core.checkpoint_export.identify_checkpoint(path)` 通过加载文件并检查真实 key 来分类——绝不看扩展名或文件名。`Sam3Adapter`（`core/sam3_adapter.py`）构造时调用它并选择正确的加载路径，否则直接拒绝。

## 为什么 trainer checkpoint 不能直接传给推理（根因）

`sam301/sam3/model_builder.py::_load_checkpoint()`（本项目未修改）的逻辑是：

```python
if "model" in ckpt and isinstance(ckpt["model"], dict):
    ckpt = ckpt["model"]
sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
```

这对 **base** checkpoint 是正确的：它的顶层 key 本来就带 `detector.` 前缀。但 **trainer** checkpoint 的 `ckpt["model"]` 子 dict 的 key 形如 `backbone.vision_backbone.trunk.pos_embed`——任何位置都没有 `"detector"` 子串，因为 `trainer.model` 由推理用的**同一个** `build_sam3_image_model` target 构建，从未包装进 `.detector` 属性（`Sam3Image.__init__` 直接设置 `self.backbone`、`self.transformer` 等）。于是 `if "detector" in k` 过滤器对 trainer checkpoint 的 1134 个 key **零匹配**，`load_state_dict({}, strict=False)` 什么都不加载，模型静默保持随机/默认初始化——这正是早前人工验收推理冒烟中 4/4 张图零预测的原因。

已针对真实 run `2026-07-03_14-42-27` 实证（2026-07-03）：

- 全新 `build_sam3_image_model(checkpoint_path=None).state_dict()`：1134 个 tensor。
- trainer checkpoint 的 `ckpt["model"]`：1134 个 tensor，**keyset 完全一致**，0 个 shape 不匹配。`model.load_state_dict(ckpt["model"], strict=True)` 直接成功——正确的映射是**恒等函数**，不是猜出来的。
- base `sam3.pt` 去掉 `detector.` 前缀后：1156 个 key（严格超集——多出 22 个 tracker/视频专用 buffer，不属于纯图像检测器）。

## 如何导出

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

选项：`--base-checkpoint <path>`（默认 `/home/book/sam301/sam3.pt`，用于下述"权重是否真的变了"检查）、`--no-base-diff` 跳过该检查、`--overwrite` 替换已存在的输出文件、`--json` 机器可读输出。

绝不覆盖：源 trainer checkpoint、base checkpoint 路径、已存在的输出文件（除非 `--overwrite`）。

## Key 映射规则（`core.checkpoint_export.build_key_mapping`）

- 对**真实**源 key 尝试一个封闭的小候选前缀集（`""`、`"module."`、`"detector."`），选择与**真实**目标模型 `state_dict()` 匹配数最多的那个。
- 拒绝猜测：如果两个不同前缀打平且产生**不同**映射，导出直接失败，绝不随便选一个。
- 两个源 key 不能映射到同一目标 key（冲突 → 硬错误）。
- 每个映射对都检查 tensor shape 和 dtype；任何 shape 不匹配都是硬错误（不静默丢弃）。
- 报告 `matched_tensors`、`matched_parameters`、`coverage_ratio`、`missing_keys`、`unexpected_keys`、`shape_mismatch`，以及按顶层模块（`backbone`/`transformer`/`segmentation_head`/`dot_prod_scoring`/`geometry_encoder`）的匹配/总数。

## Fail-loud 规则（导出时）

以下情况导出直接 raise（不写文件）：

- 输入没有被识别为 **trainer** checkpoint。
- `matched_tensors == 0`。
- `coverage_ratio < 0.98`（`MIN_COVERAGE_RATIO`）。
- 存在任何 shape 不匹配。
- 任何顶层模块匹配数为零。
- 输出路径已存在且没传 `--overwrite`，或会覆盖源/base checkpoint。
- **导出权重与 base checkpoint 在所有共同 tensor 上逐位相同**——这意味着"导出"根本不代表一个微调过的模型。

写入后，导出立即用刚写出的文件对一个**全新**模型实例做 strict-load 自校验（不是训练进程里的模型对象）——这一步不可选、不可跳过。

## strict / coverage 规则（加载时）

`core.checkpoint_export.load_inference_checkpoint(path)`：

- 拒绝 **trainer** checkpoint，报错信息指向导出命令。
- 拒绝 **base** checkpoint，报错信息指向既有的 `build_sam3_image_model(checkpoint_path=...)` 路径。
- 构建全新模型（`build_sam3_image_model(checkpoint_path=None)`）并调用 `model.load_state_dict(state_dict, strict=True)`——对真实架构该调用以**零** missing/unexpected key 成功（已验证）。`ALLOWED_MISSING_KEYS` / `ALLOWED_UNEXPECTED_KEYS` 是显式的、当前为空的白名单机制，留给未来确有正当需要的架构变更——该 loader 绝不静默回退到 `strict=False`。
- 即使 `strict=True` 没有抛错，也会拒绝加载了零个参数的 checkpoint（纵深防御）。

## 真实验收结果（2026-07-03，run `2026-07-03_14-42-27`）

- 导出 `inference_model.pt`，SHA256 `e3aea4edbadc684f7807aed0d981a481f8650e01dff3041df21d29fe09afd222`。
- `matched_tensors=1134`、`matched_parameters=841689398`、`coverage_ratio=1.0`、`missing=[]`、`unexpected=[]`。
- 对比 base：385 个 tensor 变化，749 个相同（与冻结 backbone 的训练配置一致——只训练了 `transformer`/`segmentation_head`/`dot_prod_scoring`）。
- 4 张真实冒烟图片（普通书脊、漫画/复杂图案、倾斜书脊、薄/密书架）：base checkpoint 和导出的 inference checkpoint 都在全部 4 张上产出**非空**预测，且两者的分数与 mask 数有实质差异（证明导出权重确实被使用，不是被静默忽略）。直接加载原始 trainer checkpoint 在任何推理开始前被正确**拒绝**。
- 完整数值结果：`runs/review_artifacts/checkpoint_inference_acceptance/checkpoint_inference_acceptance_results.json`（在 `runs/` 下，不进 git）。

完整验收记录见 `docs/P1_REMEDIATION_AND_INFERENCE_ACCEPTANCE.md`。

## 常见错误

- `ValueError: ... is a TRAINER checkpoint ... Export it first: ...` —— 你把推理直接指向了 `checkpoints/checkpoint.pt`；先运行导出器。
- `ValueError: ... is a BASE checkpoint ...` —— 你把 inference-checkpoint loader 指向了 `sam3.pt`；那条路径已经可以通过既有的 `build_sam3_image_model(checkpoint_path=...)` 工作，不要绕经导出器/loader。
- `coverage ratio ... below the minimum` / `N tensor shape mismatches` / `critical module ... has zero matched tensors` —— trainer checkpoint 的模型与当前 `build_sam3_image_model` 架构不再匹配（例如 SAM3 代码升级后）；先调查，不要强行绕过。
- `exported weights are BIT-IDENTICAL to the base checkpoint` —— 你导出的"trainer" checkpoint 其实从未被训练过（例如跑了 0 个 optimizer step）；这是 run 本身的真实问题，不是导出器的 bug。
