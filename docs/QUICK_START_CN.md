# SAM3 Fine-tuning 工具：简明使用说明与注意事项

面向日常使用的最短路径说明。完整细节见 `docs/USER_GUIDE_ZH_CN.md`。

## 这个程序是做什么的

对 SAM3 做**单目标类别**微调（默认书脊 `book spine`，也支持 cable 等任意新目标），
并提供配套的推理、COCO/CVAT 导出、训练预检/启动/监控、checkpoint 评估与导出。
全部操作可以在本地 Web UI 完成，也有对应命令行脚本。

## 启动 UI

```bash
cd /home/book/book01
conda run -n sam301 python app.py
```

浏览器打开 `http://127.0.0.1:7860`（仅本机可访问）。
标签页：推理任务配置 / 历史运行记录 / 结果查看 / CVAT 导出 / 训练预检 / Checkpoint 评估。

## 典型流程：训练一个新目标（以 cable 为例）

1. **准备数据集**。两种方式任选：
   - 已有标注图片文件夹 → 在「训练预检」页顶部的自动划分区（或
     `scripts/build_training_dataset_split.py`）填入标注文件夹、test 文件夹、
     输出目录和 `category / training prompt = cable`，自动生成
     `train/val/test` 三个 split（每个 split 是 `images/` + `annotations.json`）；
   - 已有现成 COCO 数据集 → 保证 categories 里有目标类别，建议
     `{"id": 1, "name": "cable"}`。
2. **训练预检**。在「训练预检」页填 train/val 的 images 目录和 COCO 路径，
   `training prompt` 填 `cable`（留空则自动用 COCO 第一个 category 名），点预检。
   预检通过后会在 `runs/training/<run_id>/config/runtime_config.yaml` 生成本次
   run 的配置——**不需要也不要手动改**基础 YAML
   `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`。
3. **启动训练**，同页监控进度；训练器日志在 `<run_dir>/logs/<task_slug>/`。
4. **Checkpoint 评估**。「Checkpoint 评估」页输入 run 目录（或
   `scripts/evaluate_sam3_checkpoints.py --run-dir ... --split all --export-best`），
   自动用该 run 记录的训练 prompt 在 validation 上选出最优 checkpoint。
5. **导出推理权重**。trainer checkpoint 不能直接用于推理，先导出：
   `scripts/export_sam3_inference_checkpoint.py --input <run_dir>/checkpoints/checkpoint.pt --output <run_dir>/checkpoints/inference_model.pt`
   （评估页的 `--export-best` 会自动导出 `inference_best.pt`）。
6. **推理**。「推理任务配置」页选择模型权重、输入图片目录，`prompt` 填 `cable`、
   `category name` 填 `cable`，运行后可在「结果查看」页检查，
   并在「CVAT 导出」页导出可导入 CVAT 的 COCO 包。

## 注意事项

- **正式训练前必须登记数据集身份**。未在
  `data_manifests/dataset_identity_registry.json` 登记为
  `allowed_for_formal_training=true` 的数据集（包括所有新目标数据集），预检只允许
  smoke 模式且 `max_epochs<=1`。想跑正式多 epoch 训练，需要人工审核数据后按
  `docs/E3_DATASET_IDENTITY_ERRATUM.md` 的说明登记。这是防止拿机器预标注
  自我训练的安全设计，不要绕过。
- **基础训练 YAML 永远只读**。每次训练的所有差异（prompt、数据路径、输出目录、
  超参）都写进 per-run 的 `runtime_config.yaml`；文件名里的 book_spine 只是历史命名。
- **训练前验证 sam301 补丁**：
  `conda run -n sam301 python scripts/manage_sam301_patch.py verify` 必须输出
  PATCHED（预检和启动器也会强制检查，失败会拒绝启动）。
- **prompt 与 category 的关系**：标注按 `category_id` 匹配，prompt 是给 SAM3 的
  自由文本，两者不要求相同；不填 prompt 时自动回落为 COCO 第一个 category 名。
  一个数据集只训练 id 最小的那个类别（本工具是单目标微调）。
- **评估用的 prompt** 自动取该 run 记录的 `resolved_training_prompt`，
  一般不需要手动传 `--prompt`；手动传入时优先级最高。
- **run 目录不可复用**：输出目录已存在且非空时预检会拒绝，请让程序自动生成新
  `<run_id>`。
- **train 集不能小于有效 batch**（`train_batch_size × grad_accum × num_gpus`），
  否则一个优化步都不会跑，预检会直接拒绝。
- **环境固定**：一律通过 `conda run -n sam301 ...` 运行；SAM3 源码在
  `/home/book/sam301`，本项目不修改它（除受管补丁外）。
