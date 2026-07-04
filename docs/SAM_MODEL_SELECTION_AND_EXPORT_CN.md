# SAM 模型选择与 Checkpoint 导出

## 1. Trainer checkpoint 与 inference model 的区别

`checkpoint_N.pt` 和 `checkpoint.pt` 是训练恢复用的 trainer checkpoint，里面包含 `model`、`optimizer`、epoch、scaler 等训练状态。它不能直接交给普通推理入口使用。

`inference_*.pt` 是由 exporter 从 trainer checkpoint 转换出的推理模型，只保存推理需要的模型权重和 metadata。普通生成初始 mask / CVAT 预标注流程只能使用原始 `/home/book/sam301/sam3.pt` 或这种 inference model。

如果直接选择 `checkpoint_35.pt` 推理，系统会拒绝并提示先到 Checkpoint 导出页面导出。

## 2. 在网页选择任意 checkpoint 导出

打开 Checkpoint 评估页面，在“手动导出任意 Trainer Checkpoint”区域：

1. 输入或选择训练 run 目录，例如 `/home/book/book01/runs/training/2026-07-04_14-28-27`。
2. 点击“刷新 Checkpoint 列表”。
3. 在 dropdown 中选择 `checkpoint_5.pt`、`checkpoint_10.pt`、`checkpoint_20.pt` 等任意 trainer checkpoint。
4. 检查选中详情：epoch、SHA256、是否 alias、checkpoint 类型、是否已有对应 inference model。
5. 修改输出目录和输出文件名。
6. 点击“导出所选 Checkpoint”。

默认输出目录是该 run 的 `checkpoints/`。默认输出名规则：

- `checkpoint_35.pt` -> `inference_checkpoint_35.pt`
- `checkpoint_40.pt` -> `inference_checkpoint_40.pt`
- `checkpoint.pt` -> `inference_checkpoint_latest.pt`

输出文件名可以手动编辑。已有文件默认拒绝覆盖；只有勾选“允许覆盖已有输出”才会覆盖。系统不会覆盖 `/home/book/sam301/sam3.pt`。

`checkpoint.pt` 如果和某个编号 checkpoint 字节相同，会显示 alias 信息。可以查看和导出，但通常不建议重复导出。

## 3. 导出验证与 metadata

导出复用 `core.checkpoint_export.export_inference_checkpoint()`，不会新增第二套 checkpoint 解析器。导出时会检查：

- 输入必须是真实 trainer checkpoint；
- key mapping 覆盖率、missing keys、unexpected keys、shape mismatch；
- 输出不是 base `sam3.pt` 的简单复制；
- 输出文件可 strict-load；
- 旁车 metadata 可写入。

每个导出模型旁边会生成：

```text
inference_checkpoint_35.pt
inference_checkpoint_35.metadata.json
```

metadata 记录 source trainer checkpoint、source SHA256、epoch、run_id、output SHA256、base model SHA256、missing/unexpected keys、coverage ratio 和 smoke/load 结果。后续不要靠文件名推断来源，以 `.metadata.json` 为准。

## 4. 生成初始 mask 时选择模型

打开初始 mask / 统一推理页面，在“模型权重选择”区域选择模型来源：

- 默认原始 SAM3：`/home/book/sam301/sam3.pt`
- 已发现 inference models：自动扫描 `/home/book/book01/runs/training/*/checkpoints/inference_*.pt` 和 `inference_best.pt`
- 手动输入目录和文件名：例如目录 `/home/book/book01/runs/training/2026-07-04_14-28-27/checkpoints`，文件名 `inference_checkpoint_35.pt`
- 手动输入完整绝对路径

页面会显示解析后的 `resolved model path`。点击“检查模型”会验证路径、文件类型、SHA256、checkpoint 类型和 `Sam3Adapter` 加载兼容性。

默认不改任何选项时仍使用 `/home/book/sam301/sam3.pt`，不会自动选择最新 checkpoint，也不会自动选择最大 epoch。

## 5. 如何确认实际使用的模型

每次生成 mask 后，run 目录中会写入：

```text
sam_model_provenance.json
run_config.json
run_summary.json
logs/run.log
```

其中 `sam_model_provenance.json` 是稳定的模型来源记录，包含：

- `sam_model_path`
- `sam_model_filename`
- `sam_model_sha256`
- `sam_model_type`
- `source_trainer_checkpoint`
- `source_epoch`
- `source_run_id`
- prompt、threshold、NMS settings
- generation timestamp

因此可以回答“这批初始 mask 是由哪个 SAM 模型生成的”。

## 6. 对比 checkpoint 35 和 40

先分别导出：

```text
checkpoint_35.pt -> inference_checkpoint_35.pt
checkpoint_40.pt -> inference_checkpoint_40.pt
```

然后在初始 mask 页面分别选择两个 inference model，输出到不同 run。比较每个 run 的 `sam_model_provenance.json`、`manifest.json`、`npz_nms/`、`coco/instances_default.json` 和可视化目录。

如果当前 run 不存在 `checkpoint_35.pt` 或 `checkpoint_40.pt`，网页列表不会显示它们；只能选择实际存在的 checkpoint。

## 7. CLI

统一推理 CLI 支持：

```bash
python scripts/run_unified_inference.py \
  --input-dir /path/to/images \
  --output-root /home/book/book01/runs \
  --model-path /home/book/book01/runs/training/<run_id>/checkpoints/inference_checkpoint_35.pt
```

`--model-path` 是 `--checkpoint` 的别名。未传入时默认 `/home/book/sam301/sam3.pt`。

## 8. 常见错误

- run 目录不存在：检查输入目录是否是 `/home/book/book01/runs/training/<run_id>`。
- `checkpoints/` 不存在：该 run 没有训练 checkpoint。
- 输出文件已存在：修改文件名，或明确勾选允许覆盖。
- 选择了 `checkpoint_N.pt` 推理：先导出为 `inference_checkpoint_N.pt`。
- 手动目录和文件名拼错：看页面的 resolved model path。
- 输入了相对路径：手动路径必须是绝对路径。
- 模型加载失败或 CUDA OOM：换 CPU 检查模型，或释放 GPU 后重试。
- metadata 写入失败：导出会失败并删除半写入输出，避免留下来源不明模型。
