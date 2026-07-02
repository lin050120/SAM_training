# E2 Gradient Accumulation Fix and Effective-Batch-4 Training Acceptance

Generated: 2026-07-03 01:00 JST

## 结果

**PASS** —— grad_accum=4（effective_batch_size=4）真实训练 1 epoch 完成：exit code 0，summary `completed`，checkpoint 生成，`sam3.pt` SHA256 前后不变，无残留进程。梯度累积经 trainer 日志证实真实生效（每 epoch 2 个 iteration，对比 accum=1 冒烟跑的 8 个）。

## 1. 代码修复（commit `5a9f585`）

依据 `docs/E2_SMOKE_AND_GRAD_ACCUM_AUDIT.md` 的审计结论，在 `core/training_runner.py::write_runtime_yaml` 中实现：当解析后的 `gradient_accumulation_steps > 1`（覆盖值或基础 YAML 值）时，runtime YAML 自动接线——

1. `scratch.collate_fn._target_` → 官方 `sam3.train.data.collator.collate_fn_api_with_chunking`（原有 `_partial_`/`dict_key`/`repeats`/`with_seg_masks` kwargs 保留，chunking 版签名是普通版超集）；
2. 新增 `scratch.collate_fn.num_chunks: ${scratch.gradient_accumulation_steps}`；
3. `trainer.data.train.batch_size` → `train_batch_size × gradient_accumulation_steps`（本例 1×4=4；`scratch.train_batch_size` 语义保持 micro-batch 不变）。

`accum == 1` 时三处均不触碰（保持基础 YAML 原样）。**未修改 `/home/book/sam301` 任何文件**（trainer、collator、基础 YAML 均原样）。

原理（对应 trainer 契约）：`trainer._run_step`（trainer.py:920-931）在累积模式要求 dataloader 每 iteration 产出恰好 accum 个 micro-batch 的 list，逐个前向/反向（micro 间 `model.no_sync()`）后一次 optimizer step；`collate_fn_api_with_chunking`（collator.py:107-134）把 DataLoader 一次 fetch 的 `micro×accum` 个样本按 `batch[i::num_chunks]` 切成 `num_chunks` 个 collated micro-batch 并返回 list。

## 2. 回归测试

新增 `tests/test_e1_training.py::GradAccumWiringTest`（2 个）：

- `test_accum_4_wires_chunking_collate_and_scaled_train_batch_size`：真实预检生成 accum=4 runtime YAML，断言 collate `_target_` 为 chunking 版、`num_chunks` 插值解析为 4、`_partial_`/`dict_key` 保留、`trainer.data.train.batch_size == 4`、`scratch.train_batch_size == 1`、`trainer.gradient_accumulation_steps == 4`、val 侧仍是普通 `collate_fn_api`（无累积）、接线后 YAML 仍能通过真实 Hydra compose、`effective_batch_size == 4`；
- `test_accum_1_leaves_collate_and_train_batch_size_untouched`：accum=1 时 collate 与 batch_size 原样、无 `num_chunks` 键。

全量测试：**151 passed, 5 warnings**（149 → 151）。

## 3. 真实训练验收（effective=4）

- **新 run**：`/home/book/book01/runs/training/2026-07-03_00-56-32`（正式 `run_training_preflight` 新建，全新 token；一次启动、无重试；未裸跑 train.py）
- **启动前闸门 17 项全过**：CUDA/RTX 5090、import guard、Hydra validation、max_epochs=1、micro batch=1、grad_accum=4、effective=4、num_gpus=1、prompt=book spine、checkpoints/logs 空、无旧 summary、command 为 sam301+wrapper，以及三项新接线断言（chunking collate、num_chunks=4、loader batch_size=4）
- **执行**：正式 `start_training(state, confirmed=True)`；PID/PGID 1439195/1439195；00:56:39 → 00:57:09 JST，duration 29.71 秒
- **结果**：exit code **0**；完成 **1 epoch**；summary status=`completed`
- **梯度累积生效证据**：trainer 日志 `Train Epoch: [0][0/2]`——8 张训练图 ÷ DataLoader batch 4 = **每 epoch 2 个 iteration**（accum=1 冒烟跑为 `[0][0/8]` 共 8 个），每 iteration 含 4 个 micro-batch 累积 + 1 次 optimizer step，即 effective batch=4、每 epoch 2 个 optimizer step，与审计预测完全一致
- **显存**：峰值 15.00 GB——与 micro-batch=1 冒烟跑相同，证实 chunking 顺序执行不叠加激活显存
- **真实损失**：`train_stats.json` 写出 `Losses/train_all_loss: 361.58` 等完整指标
- **checkpoint**：`.../2026-07-03_00-56-32/checkpoints/checkpoint.pt`，10,081,250,310 字节
- **sam3.pt 校验**：size=3450062241、mtime 不变、SHA256 前=后=`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- **清理**：无残留进程；GPU 训练后回落 866 MiB / 0%
- 冒烟 run（`2026-07-03_00-43-43`）与两个失败 run 均保持原样

## 4. 本轮边界

- 未修改 SAM3 源码/trainer.py/基础 YAML；未修改旧 run；一次启动无重试；未提交 run/日志/checkpoint；代码修复（`5a9f585`）与本报告分开提交。

## 5. E2 状态

至此 E2 的全部三个阻塞问题闭环：Hydra config 加载（`5363e75`）、grad accumulation 数据契约（`5a9f585`），以及此前的环境迁移。正式链路已在 effective_batch_size=1 和 4 两种配置下各完成一次真实 1-epoch 训练验收（completed + checkpoint + 基线 checkpoint 完整性）。后续正式微调（max_epochs>1）可按既定流程走预检 + 用户批准。
