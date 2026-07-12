# Checkpoint 评估与最佳模型选择（中文说明）

> 语言版本：**中文** | [日本語](CHECKPOINT_EVALUATION_JA.md)

生成时间: 2026-07-04

## 1. checkpoint.pt 与 checkpoint_N.pt 的区别

- `checkpoint_N.pt`（如 `checkpoint_5.pt`）：训练器在第 N 个 epoch 结束时按 `save_freq` 保存的**epoch 快照**，含 model + optimizer + scaler + epoch 等完整恢复状态（约 10GB）。
- `checkpoint.pt`：训练器的**最新/恢复别名**，每个 epoch 都被覆盖为最新状态。训练结束时它通常与最后一个 `checkpoint_N.pt` **字节完全相同**（评估器按 SHA256 判定：相同则记为 alias，不重复评价；不同则作为"latest"候选单独评价，epoch 从文件内部读取）。
- 两者都是 **trainer checkpoint**，不能直接用于推理（见 `docs/CHECKPOINT_EXPORT_AND_INFERENCE.md`）。

## 2. 为什么不能默认最后一个 checkpoint 最好

小数据集微调很容易过拟合：训练 loss 持续下降不代表验证集 mask 质量持续提升，后期 epoch 可能记住训练集的标注瑕疵而在验证集上边界变差。唯一可靠的判断方式是在**固定的、独立的、人工修正的验证集**上逐个实测。选择规则里最后一条平局裁决也是"更早 epoch 胜出"——绝不无依据偏向训练更久的模型。

## 3. 为什么 bbox AP 不能代表 mask 质量

本项目输出的 mask 将用于 RGB-D 对齐、RANSAC 平面拟合和机械臂抓取姿态估计。bbox AP 只衡量外接框的重合度：一个 mask 系统性外扩 10 像素、粘连相邻书脊、边界锯齿严重，它的 bbox 可能几乎不变，bbox AP 依然很高。训练日志里的 `coco_eval_bbox_AP` 只能作为训练监控信号，不能作为选择依据。本评估全部指标都在 **mask 级**计算。

## 4. validation 与 test 的区别

- **validation（验证集）**：用来做训练中决策——选最佳 checkpoint、调阈值。可以反复使用。
- **test（测试集）**：当前用于 diagnostic-only 对照。完整流程会在 validation 选定 best 后，继续把 baseline 和所有 checkpoint 放到 test 上评价，但 **test 结果绝不改变 best checkpoint**。
- 当前登记的 `book_spine_human_corrected_v1` 含 train(44)/val(8)/test(12)，`allowed_for_model_evaluation=false`。原因是本工具会展示所有 checkpoint 的 test 指标，因此该 test 后续不应再被视为完全未查看的最终盲测集。它可用于人工核实和诊断，不能作为最终模型质量结论。

## 5. 验证集守卫（为什么会拒绝评价）

评估开始前按**解析后路径**（绝不按文件名）在 `data_manifests/dataset_identity_registry.json` 中查找验证集：

- 未登记 → `blocked`，提示先登记；
- 登记为机器预标注（`human_reviewed=false`，如 `formal_book_spine_sam3_dataset` 的 val）→ `blocked`：用模型自己的预标注选 checkpoint 是自我循环，结果无意义；
- 验证集路径 == 训练集路径 → `blocked`：禁止用训练数据选 checkpoint；
- 无合法验证集时**绝不**伪造 best，`best_checkpoint.json` 写入 `status: blocked` + 原因。

## 6. 评价流程与固定条件

所有 checkpoint 使用**完全相同**的条件（写入 `evaluation/evaluation_config.yaml`）：图片与 GT、prompt（优先取该 run 记录的 `resolved_training_prompt`，例如 `book spine` 或 `cable`；run 没有记录时回落到 `config/checkpoint_evaluation.yaml` 的 `prompt`；`--prompt` 显式传入时优先级最高）、score/confidence 阈值、min_area、dtype（bf16）、device、mask 后处理（统一走 `core.sam3_adapter.Sam3Adapter.predict`）、SAM3 源码 hash、评估器版本。trainer checkpoint 通过 `core.checkpoint_export.load_trainer_checkpoint_model` **strict=True** 加载进全新模型（零静默丢权重）。

默认完整流程：

1. 检查 validation 身份与路径；
2. 对 baseline、`checkpoint_5/10/15/20` 等唯一 checkpoint 在 validation 上评价；
3. 只根据 validation 指标运行 selector，写 `best_checkpoint.json`；
4. 可选导出 `checkpoints/inference_best.pt`；
5. 检查 test 身份与路径；
6. 对 baseline 和所有唯一 checkpoint 在 test 上评价；
7. 写 `test_checkpoint_comparison.json`，字段名为 `test_highest_metric_checkpoint`，并标记 `diagnostic_only=true`。

### 实例匹配

GT × prediction 构建 IoU 矩阵，**Hungarian 算法**（`scipy.optimize.linear_sum_assignment`）最大化总 IoU 做一对一匹配；IoU=0 的配对视为未匹配。一个预测绝不同时匹配两个 GT。方法名 `hungarian_max_total_iou` 记录在所有结果文件中。

### 指标定义

- **mean_iou_all_gt（主指标）**：每个 GT 实例贡献一个 IoU，**漏检 GT 记 0**，对全部 GT 实例取平均（实例级汇总，标注多的图贡献更多实例）。matched-only 平均值同时输出但天然虚高，绝不用于选择。
- `recall_iou_T` = IoU≥T 的匹配数 / GT 总数；`precision_iou_T` = 同分子 / 预测总数（T=0.5/0.75/0.9）。
- `miss_rate_iou_50` = 1 − recall@0.5；`false_positive_count` = 预测总数 − IoU≥0.5 匹配数。
- **Boundary F1**：在**原图分辨率**计算（预测 mask 已由 adapter 还原原尺寸）；边界 = mask 与其 1px 腐蚀的差；容差默认 **2px**（可配置，实际值写入报告）；漏检 GT 记 0。
- **面积偏差**：IoU≥0.5 匹配对的 `pred_area/gt_area`，输出 mean/median/p10/p90 与 `pred_larger_than_gt_rate`，用于监控 SAM3 已知的系统性外扩。

## 7. 最佳 checkpoint 选择规则（selector.py，有单元测试）

1. 排除评价失败/NaN/无有效结果的 checkpoint；baseline 永不参与自动选择；
2. `mean_iou_all_gt` 最大者胜；
3. 差值 ≤ **0.005** 视为平局 → `mean_boundary_f1_all_gt` 更高者胜；
4. 仍平 → `miss_rate_iou_50` 更低者胜；
5. 仍平 → `false_positive_per_image` 更低者胜；
6. 仍平 → **更早 epoch** 胜。

tie tolerance、规则全文与逐步裁决轨迹写入 `best_checkpoint.json` 的 `selection_reason`。

## 8. 如何比较原始 SAM3 与微调模型

`/home/book/sam301/sam3.pt` 作为 **baseline** 每次一并评价（可 `--no-baseline` 关闭），在排名表中单独一行显示（🏁 标记），**不参与**微调内部最佳选择。`best_checkpoint.json` 的 `finetuned_improved_over_baseline` 直接回答"微调是否真的优于原始模型"（按 mean_iou_all_gt 比较）。

## 9. 输出文件位置

```
<run_dir>/evaluation/
├── evaluation_config.yaml
├── dataset_split_audit.json
├── dataset_split_audit.csv
├── best_checkpoint.json      # 只由 validation 选择
├── test_checkpoint_comparison.json
├── evaluation_summary.json
├── evaluation.log
├── validation/
│   ├── checkpoint_metrics.csv
│   ├── checkpoint_metrics.json
│   ├── per_image_metrics.csv
│   ├── per_instance_metrics.csv
│   ├── gt_snapshot.json
│   ├── human_review_index.csv
│   ├── failure_cases.csv
│   ├── raw_predictions/<checkpoint>/*.npz + *.json
│   ├── match_records/<checkpoint>/*.json
│   └── visualizations/<checkpoint>/*.png
├── test/
│   └── 同 validation，但 test 结果不参与 best selection
└── cache/
```

`training_summary.json` 会追加独立的 `checkpoint_evaluation` 块（原子写入，原字段全保留）。

缓存 key = SHA256(split name | checkpoint SHA256 | 该 split 标注 SHA256 | 该 split 图片 SHA256 聚合 | 评价配置 SHA256 | 评估器版本 | SAM3 源码 hash)。validation/test 缓存物理分开，文件名从不作为缓存依据；GT、配置或 checkpoint 任何一项变化都会重算。`--force` 强制全部重算。

`raw_predictions` 中的 NPZ 保存 mask、score、bbox、instance_id；旁边 JSON 保存图像路径、尺寸、prompt、阈值和推理耗时。`match_records` 保存 IoU matrix、Hungarian assignment、accepted matches、unmatched GT 和 unmatched prediction，可用于重新核算指标。`human_review_index.csv` 默认保留 `review_status` 和 `reviewer_notes` 空列，重评时会尽量保留用户已填写的备注。

## 10. 如何手动启动评价（命令行）

```bash
conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \
  --run-dir /home/book/book01/runs/training/<run_id> \
  --split all \
  --export-best
```

常用可选项：`--split validation|test|all`、`--test-annotations/--test-images`、`--export-best`（只导出 validation 选中的 best）、`--force`、`--no-baseline`、`--checkpoints checkpoint_5.pt checkpoint_10.pt`、`--score-threshold/--min-area/--boundary-tolerance/--prompt`（覆盖固定条件——覆盖后对所有 checkpoint 一视同仁）、`--max-images N --smoke`（冒烟；截断验证集自动强制标记 smoke，不作为正式排名）、`--device cpu`。

## 11. 网页操作

启动 UI（普通终端）：`conda run -n sam301 python /home/book/book01/app.py` → 打开 `Checkpoint 评估` 标签页：

1. 下拉框选择含 checkpoint 的训练 run（自动列出）；
2. `刷新状态/结果`：显示是否已评价、validation/test 数据身份、最佳 checkpoint、Mean IoU、是否优于 baseline、validation 表、test 表与 `best_checkpoint.json`；
3. `依次评价 Validation 和 Test`：后台子进程运行（服务端单任务守卫，重复点击会被 BLOCKED），顺序为 validation → selector → best export → test diagnostic；
4. `重新评价 Validation 和 Test（忽略缓存）`：`--split all --force`；
5. `评价 Validation 全部 Checkpoint` / `评价 Test 全部 Checkpoint`：只运行目标 split；
6. `重新评价 Validation` / `重新评价 Test`：只忽略目标 split 缓存；
7. `导出最佳推理模型`：把 validation best trainer checkpoint 导出为 `checkpoints/inference_best.pt`（走既有 export 流程，含 key 覆盖率/与 base 差异校验）。

无合法验证集时按钮不会静默成功——评价子进程立即以 `blocked` 结束并在页面显示原因。

## 12. 自动评价开关

`config/checkpoint_evaluation.yaml` 中 `auto_evaluate_after_training: false`（第一阶段默认关闭，仅手动触发）。后续打开后，训练成功且验证集守卫通过时可自动评价——钩子接口已预留（`load_defaults()` 读取该开关），本阶段未接入训练流程。

## 13. 最佳模型导出

`best_checkpoint.json` 中保存的是 **trainer checkpoint 路径引用**（不复制 10GB 文件）。导出推理模型复用既有 `scripts/export_sam3_inference_checkpoint.py`（strict key 覆盖、与 base 权重差异确认、原子写入），输出 `<run_dir>/checkpoints/inference_best.pt`，导出失败不影响评价结果本身（`export_best.status` 记录于 evaluation_summary.json）。
