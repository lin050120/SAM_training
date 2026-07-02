# E2 Smoke Training and Gradient Accumulation Audit

Generated: 2026-07-03 00:50 JST

## 结果总览

- **阶段 A（grad_accum=1 端到端冒烟训练）：PASS** —— 1 epoch 完成，exit code 0，summary `completed`，checkpoint 生成，`sam3.pt` 前后 SHA256 不变，无残留进程。
- **阶段 B（梯度累积只读审计）：完成** —— 根因、官方机制、精确修复方案已定位；**不需要修改 `/home/book/sam301` 源码**，仅需 book01 的 runtime YAML 生成逻辑。
- 本轮零代码/配置修改（阶段 A 通过临时驱动调用正式 launcher，阶段 B 纯只读）；SAM3 源码、trainer.py、旧失败 run 均未触碰。

## 1. 之前失败原因（run 2026-07-03_00-35-18）

`sam3/train/trainer.py:921 _run_step`：`gradient_accumulation_steps > 1` 时 trainer 要求 dataloader 每个 iteration 产出 **list**（长度 = accum 步数的 micro-batch 列表，`trainer.py:924` 还断言 `len(batch) == gradient_accumulation_steps`）；而 book_spine 配置的数据管道每步产出单个 collated batch（dict）→ `AssertionError: Expected a list of batches, got <class 'dict'>`。

## 2. 阶段 A：冒烟训练（grad_accum=1）

- **新 run**：`/home/book/book01/runs/training/2026-07-03_00-43-43`（经修复后的正式 `run_training_preflight` 创建，全新 token；旧失败 run 未复用、未修改）
- **实际参数**（`training_config_summary.json` 记录并逐项断言）：max_epochs=1、train_batch_size=1、gradient_accumulation_steps=1、effective_batch_size=1、num_gpus=1、prompt=book spine、resume_from 无、conda=sam301、Hydra validation ok、import guard ok
- **启动前闸门全过**：CUDA/RTX 5090、import guard `/home/book/sam301/sam3/__init__.py`、Hydra compose 验证、数据路径与 checkpoint 存在、checkpoints/logs 空、无残留进程
- **启动方式**：正式 `ui.training_preflight_page.start_training(state, confirmed=True)`（一次启动、无重试、未裸跑 train.py）
- **训练结果**：
  - PID/PGID：1436769 / 1436769
  - 开始 00:43:49 → 结束 00:44:20 JST，duration 30.44 秒
  - **exit code = 0，完成 1 epoch**（trainer 日志 `Train Epoch: [0][0/8]`，8 个 iteration；`train_stats.json` 写出真实损失，如 `Losses/train_all_loss: 340.65`；峰值显存 15.00 GB）
  - **summary**：`.../2026-07-03_00-43-43/training_summary.json`，status=`completed`
  - **checkpoint**：`.../2026-07-03_00-43-43/checkpoints/checkpoint.pt`，10,081,250,310 字节（≈9.4 GiB，含 model + optimizer + loss 状态，符合审计预期的默认命名 `checkpoint.pt`、`save_freq=5` 下无编号副本）
  - **sam3.pt 校验**：前后 size=3450062241、mtime 不变、SHA256 前=后=`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
  - **残留进程**：无；GPU 训练后回落 864 MiB / 0%（仅桌面）
  - 所有输出限制在新 run 目录内（14 个文件：runtime/config_resolved YAML、trainer 日志、tensorboard、train_stats、summary、checkpoint）

这证明 Hydra 修复（commit `5363e75`）之后，整条正式链路（preflight → token → import guard → Hydra 验证 → start_training → ProcessManager → 官方 trainer → summary/checkpoint 落盘）端到端工作。

## 3. 阶段 B：梯度累积数据流审计（只读）

### 数据流追踪

1. `Trainer.run_train()`（trainer.py:582-586）→ `self.train_dataset.get_loader(epoch)` → `train_epoch(dataloader)`。
2. `train_dataset` 由 `trainer.data.train` 配置实例化，为 `sam3.train.data.torch_dataset.TorchDataset`（torch_dataset.py:10）——**全 sam301 源码树中唯一的 `get_loader` 实现**，返回标准 `torch.utils.data.DataLoader(batch_size=..., collate_fn=...)`，本身永不产出 list-of-batches。
3. `train_epoch`（trainer.py:798-807）逐 iteration 取 `batch` → `_run_step`（trainer.py:920-931）：
   - `gradient_accumulation_steps > 1` → 断言 `isinstance(batch, list)` 且 `len(batch) == accum_steps`，然后 `for i, chunked_batch in enumerate(batch)` 逐 micro-batch 前向/反向，最后一个 micro-batch 之前用 `model.no_sync()` 抑制 DDP 梯度同步，之后一次 optimizer step；
   - `== 1` → `batch = [batch]` 走同一循环。
4. **产出 list 的唯一官方组件是 collate 层**：`sam3/train/data/collator.py:107` 的 **`collate_fn_api_with_chunking(batch, num_chunks, ...)`** —— 把 DataLoader 一次 fetch 的样本列表按 `batch[i::num_chunks]` 切成 `num_chunks` 个 chunk，对每个 chunk 调用普通 `collate_fn_api`，**返回 `collated_chunks` 列表**（collator.py:119-134）。这与 trainer 的 list 契约严丝合缝（`len == num_chunks == gradient_accumulation_steps`）。

### 阶段 B 六个问题的回答

1. **list of batches 应由哪个组件生成**：DataLoader 的 `collate_fn`——官方 `sam3.train.data.collator.collate_fn_api_with_chunking`。trainer 不切分、TorchDataset 不切分，切分职责在 collator。
2. **官方配置如何产生多个 micro-batch**：`trainer.data.train.batch_size` 设为 `micro_batch × gradient_accumulation_steps`（每次 fetch 一个 effective batch 的样本），`collate_fn` 用 `collate_fn_api_with_chunking(num_chunks=gradient_accumulation_steps)` 切成 N 个 collated micro-batch。注意：本 SAM3 快照自带的官方示例配置（roboflow_v100 等）全部是 `gradient_accumulation_steps: 1` + 普通 `collate_fn_api`——**仓库中没有任何现成 YAML 实际启用过 accum>1**，该路径的正确用法由 `collate_fn_api_with_chunking` 与 `trainer._run_step` 两端的代码契约互相印证。
3. **当前 book_spine YAML 缺什么**：它是从 roboflow 官方模板复制的（`collate_fn: collate_fn_api`、`batch_size: ${scratch.train_batch_size}`=1），但把 `gradient_accumulation_steps` 从 1 改成了 4（YAML 内注释"[改] 1 → 4"，是本项目早期自行修改、从未真实运行验证）——只改了 trainer 侧的 accum 步数，没有同步把 collate 换成 chunking 版本、也没有把 DataLoader batch_size 乘上 accum。两处缺失，任一都会触发断言。
4. **能否只改 book01/runtime YAML 解决**：**能**。`core/training_runner.py::write_runtime_yaml` 在生成 runtime YAML 时（当解析后的 `gradient_accumulation_steps > 1`）追加两个覆盖：
   - `scratch.collate_fn._target_ = sam3.train.data.collator.collate_fn_api_with_chunking`，并加 `num_chunks: ${scratch.gradient_accumulation_steps}`（保留 `_partial_: true` 及原有 kwargs——chunking 版签名是 collate_fn_api 的超集，仅多 `num_chunks`）；
   - `trainer.data.train.batch_size = train_batch_size × gradient_accumulation_steps`（覆盖掉 `${scratch.train_batch_size}` 插值；`scratch.train_batch_size` 语义保持 micro-batch 不变）。
   validation 集不受影响（val 无梯度累积）。
5. **是否必须修改 /home/book/sam301 源码**：**不需要**。trainer 和 collator 的官方组件已完备；缺的只是配置接线，而 runtime YAML 由 book01 生成。（可选替代是直接修 book_spine 基础 YAML——该文件本就是项目早期放入 sam301 的项目自有配置，但按既定"基础 YAML 只读"边界，推荐 runtime 层修复，不动 sam301。）
6. **正确的 effective_batch_size=4 方案**：
   - `scratch.train_batch_size=1`（micro-batch）、`scratch.gradient_accumulation_steps=4`（保持 trainer.gradient_accumulation_steps 插值）；
   - `trainer.data.train.batch_size=4`、`collate_fn_api_with_chunking(num_chunks=4)`；
   - 效果：train 集 8 张图、`drop_last=True` → 每 epoch `len(loader)=2` 个 iteration，每 iteration 4 个 micro-batch 顺序前向/反向（micro 间 `no_sync`）+ 1 次 optimizer step → effective batch = 4，每 epoch 2 个 optimizer step；
   - 显存：micro-batch 仍为 1、顺序执行，激活峰值与冒烟跑同量级（实测 15 GB / 32 GB），仅数据 fetch 侧一次持有 4 个样本；
   - 明确排除：`batch=[batch]`（trainer 对 accum=1 的内部退化路径）不是解决方案——它只有 1 个 micro-batch，等效 batch 仍是 1，不产生 4 步累积。

## 4. 推荐长期修复方案

在 `core/training_runner.py::write_runtime_yaml` 中实现第 3 节问题 4 的两条覆盖（对 `gradient_accumulation_steps > 1` 的所有 run 自动生效），并配套：
- 回归测试：生成 accum=4 的 runtime YAML 断言 collate `_target_` 为 chunking 版、`num_chunks` 正确、`trainer.data.train.batch_size == micro × accum`；accum=1 时保持原样；
- Hydra `--validate-only` 已在预检覆盖 compose 层；batch 契约层由上述 YAML 断言 + 后续一次 accum=4 真实训练验收；
- 文档更新 `docs/stage_e1_training_ui.md` 参数映射表。

该修复与本报告分开提交（本轮按指示未实施，等待批准后执行并做第二次真实训练验收）。

## 5. 本轮边界确认

- 未修改 SAM3 源码 / trainer.py / 基础 YAML / 旧 run；未自动重试；一次启动；未提交 run/日志/checkpoint；Conda/系统未动。
- 阶段 A 驱动脚本位于会话 scratchpad（临时文件），仅做 import-context 修正（`sys.path.insert(0, "/home/book/book01")`）后调用正式项目函数。
