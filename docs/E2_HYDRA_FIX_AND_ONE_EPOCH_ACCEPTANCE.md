# E2 Hydra Fix and One-Epoch Training Acceptance

Generated: 2026-07-03 00:40 JST

## Final Result

**FAIL**（Hydra 修复本身成功并已验证；单次真实训练在第一个训练 step 失败于一个与 Hydra 无关的新问题，按规则停止，未重试）

- Hydra 根因修复：**成功**——trainer 完成 Hydra compose、配置落盘、模型构建、组件初始化、移到 cuda:0、优化器构建，进程存活 15.6 秒（旧失败发生在 compose 的第 0 秒）。
- 一个 epoch 训练验收：**FAIL**——第一个训练 batch 触发 `AssertionError: Expected a list of batches, got <class 'dict'>`（`sam3/train/trainer.py:921`）。
- 只启动了一次；无自动重试；未修改训练参数；未修改 SAM3 源码。

## 1. Hydra 根因

`sam3/train/train.py` 的 `__main__` 用 `initialize_config_module("sam3.train", version_base="1.2")` 初始化 Hydra，`main()` 中 `cfg = compose(config_name=args.config)`（train.py:141,313）。因此 `-c` 是**相对 `sam3.train` 包的 Hydra config name**（官方帮助文本示例：`configs/roboflow_v100_full_ft_100_images.yaml`），搜索路径只有 `pkg://sam3.train`。book01 之前把 runtime YAML 的**绝对文件系统路径**当 config name 传入，Hydra 规范化掉前导 `/` 后在包内查找 `home/book/book01/...`，必然 `MissingConfigException`（与 2026-07-03 00:12 失败 run 的错误逐字一致，搜索路径证据：`provider=main, path=pkg://sam3.train`）。

runtime YAML 必须留在 run 目录（不得写入 `/home/book/sam301`），所以修复在 book01 侧完成。

## 2. 修复方式与修改文件

**新增 `scripts/launch_sam3_training.py`**（book01 侧包装 launcher）：
- `resolve_config_target()` 把 runtime YAML 绝对路径映射为 `(config_dir, config_name)`（目录 + stem）；
- `run_training()` 用 `initialize_config_dir(config_dir=<run>/config, version_base="1.2")` 初始化 Hydra，`register_omegaconf_resolvers()` 后调用**未修改的官方** `sam3.train.train.main()`——官方启动语义（submitit 配置、single_node_runner、`--num-gpus` 覆盖）完整保留；
- `--validate-only` 模式只做 Hydra compose 并校验 trainer/launcher/submitit 三段存在，不 import torch、不创建 trainer。

**修改文件**（commit `5363e75` Fix Hydra runtime config launch）：
- `scripts/launch_sam3_training.py`（新增，如上）
- `core/config.py`：新增 `DEFAULT_TRAINING_LAUNCHER` 常量
- `core/training_runner.py`：训练命令改为经包装 launcher；新增 `build_hydra_validation_command()` / `run_hydra_config_validation()`；预检写出 runtime YAML 后在相同 sam301 环境和子进程 env 下真实执行一次 `--validate-only`，失败直接计入 preflight errors；launcher/train_script 纳入路径存在性检查；`dataset_info.json` 和 `training_config_summary.json` 记录 `hydra_validation_ok`
- `docs/stage_e1_training_ui.md`、`README.md`：命令格式与语义说明更新
- `tests/test_e1_training.py`：新增 `HydraLaunchRegressionTest`（5 个测试）
- `tests/test_ui.py`：命令入口断言随语义更新（train.py → 包装 launcher）

回归测试覆盖（阶段二要求的 4 点）：
1. runtime YAML 不再被当作绝对 config name：`test_training_command_uses_wrapper_not_train_py_as_hydra_entry`（断言命令入口为包装 launcher，`train.py` 路径不在命令中）+ `test_resolve_config_target_maps_yaml_path_to_dir_and_stem`；
2. 生成命令可完成 Hydra 加载：`test_generated_runtime_yaml_composes_via_hydra_and_validate_only_subprocess_passes`——对真实预检产物既做进程内 compose 断言参数，又跑真实 `conda run -n sam301 ... --validate-only` 子进程断言退出码 0（并断言坏路径退出码非 0）；另有 `test_hydra_validation_failure_blocks_preflight`；
3. import guard 仍指向 `/home/book/sam301`：`test_hydra_validation_command_and_import_guard_stay_on_sam301`；
4. 旧 sam3 环境不被调用：同上断言全部命令前缀为 `conda run -n sam301`。

## 3. 测试结果

- `conda run -n sam301 python -m pytest tests/ -v`：**149 passed, 5 warnings, 15 subtests passed**（144 → 149，新增 5）
- Gradio smoke test：通过（含在全量中）
- 代码修复 commit：**`5363e75`**

## 4. 新 E2 run 与阶段四检查

- 新 run：`/home/book/book01/runs/training/2026-07-03_00-35-18`（旧失败 run `2026-07-02_19-46-53` 未删除、未修改）
- 经正式 `ui.training_preflight_page.run_training_preflight(...)` 创建；启动前检查全部通过：token 未消费、checkpoints/logs 为空、无 training_summary、command 使用 sam301 + 包装 launcher、`hydra_validation_ok: true`（预检内真实 compose 子进程）、max_epochs=1 / batch=1 / grad_accum=4 / effective=4 / num_gpus=1 / prompt=book spine、YAML 无 `/home/book/sam3/` 与旧 run 引用

## 5. 真实训练执行（阶段五）

启动前闸门（全部通过）：nvidia-smi 正常（RTX 5090，863MiB/32607MiB，0% util，无未知重型任务）；driver 进程内 `torch.cuda.is_available()=True`；官方 import guard 子进程 `sam3=/home/book/sam301/sam3/__init__.py`；无残留训练进程。

- 启动方式：**正式 `ui.training_preflight_page.start_training(preflight_state, confirmed=True)`**（launch token 被正常消费；启动时 import guard 在临界区内再次通过）
- 实际命令（ProcessManager 执行）：
  ```
  conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py -c /home/book/book01/runs/training/2026-07-03_00-35-18/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
  ```
- PID/PGID：`1435435 / 1435435`
- 开始：2026-07-03 00:35:25 JST；结束：2026-07-03 00:35:41 JST；持续 15.64 秒
- exit code：**1**；完成 epoch：**0**
- 自动重试：**无**（第一个实质错误后停止）

**取得的进展（证明 Hydra 修复生效）**：trainer 本体写出了 `config.yaml` 与 `config_resolved.yaml`（train.py `main()` 的产物，即 compose + 自定义 resolver 全链路成功）、`logs/book_spine/log.txt`（990 行：数据集加载、模型构建、"Finished setting up components"、移到 cuda:0、优化器参数组构建）和 tensorboard 事件文件——全部限制在新 run 目录内。

## 6. 第一个实质错误（新问题，非 Hydra）

```
[rank0]:   File "/home/book/sam301/sam3/train/trainer.py", line 921, in _run_step
[rank0]:     assert isinstance(batch, list), (
[rank0]: AssertionError: Expected a list of batches, got <class 'dict'>
```

完整 traceback 保存在：
- `/home/book/book01/runs/training/2026-07-03_00-35-18/training_summary.json` 的 `stdout_stderr_tail` 字段；
- trainer 自身日志 `/home/book/book01/runs/training/2026-07-03_00-35-18/logs/book_spine/log.txt`。

**机制**（只读分析，本轮未修复）：`trainer.py:921` 在 `gradient_accumulation_steps > 1` 时要求 dataloader 每步产出 **list**（长度等于 accum 步数的 micro-batch 列表）；而基础 `book_spine_finetune.yaml` 配置的数据管道（`sam3.train.data.torch_dataset.TorchDataset` + `collate_fn_api`）每步产出单个 dict batch。即基础 YAML 自身的 `gradient_accumulation_steps=4` 与其数据管道组合从未被真实验证过（此前所有轮次均未跑过真实训练）。这是 SAM3 基础配置层面的不匹配，修复要么需要调整数据管道包装（可能涉及基础 YAML 或 SAM3 用法），要么把 `gradient_accumulation_steps` 设为 1（走 `else` 分支 `batch=[batch]`）——两者都超出本轮授权（禁止修改训练参数/基础 YAML），故按"第一个实质错误停止"规则终止。

## 7. 完整性检查

- `training_summary.json`：status=`failed`，exit_code=1，conda_environment=sam301，import guard ok，discovered checkpoints `[]`
- checkpoint：无（0 个文件）
- `sam3.pt` 前后校验完全一致：
  - size `3450062241`，mtime `2026-06-15 19:22:15 +0900`
  - SHA256 前 = 后 = `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- 残留进程：无（pgrep 无匹配）；GPU 恢复 863MiB / 0%（仅桌面）
- 所有输出限制在新 run 目录内；旧 run、数据集、`/home/book/sam301` 源码、Conda 环境均未修改

## 8. 建议下一步

针对 batch-list 断言，二选一（均需用户批准）：
1. **最小验收路径**：以 `gradient_accumulation_steps=1`（effective batch=1）重新预检并做一次冒烟训练——不改任何代码，只改一次性 run 参数，先证明端到端能完成 1 epoch 并产出 checkpoint；
2. **保持 effective=4**：审计 SAM3 中产出 list-batch 的官方数据包装用法（trainer 期望的 accum-aware loader），确认基础 YAML 应如何配置——可能需要修改基础 YAML（需明确授权）。
