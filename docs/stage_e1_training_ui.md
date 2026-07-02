# Stage E1: Training Orchestration and Monitoring

生成时间: 2026-07-02

本轮在阶段 D1/D1.1 的 Gradio UI 基础上，把训练预检页面扩展为"预检 + 一键训练启动 + 训练监控"两阶段流程。本轮**没有**实际启动过 SAM3 正式训练，**没有**使用 GPU，**没有**修改 `/home/book/sam301` 里的任何文件。

## 1. 两阶段流程

页面顶部固定显示："必须先完成训练预检，预检通过后才能启动训练。"

- **阶段 A：生成并验证训练配置**（`ui/training_preflight_page.py::run_training_preflight()`）——只读取权威基础 YAML 和真实数据路径，生成一个新的 per-run runtime YAML，不启动任何进程。
- **阶段 B：启动训练**（`ui/training_preflight_page.py::start_training()`）——只有阶段 A 的结果通过（`errors` 为空）且用户勾选确认框后，才会用 `subprocess.Popen` 参数列表启动真正的官方训练入口。

预检和正式训练的区别：预检只做路径检查、COCO 摘要统计、参数解析和 YAML/命令生成，全程不导入 `torch`/`sam3` 训练器，不占用 GPU；正式训练才会真正加载模型、跑真实的 forward/backward。

## 2. 参数覆盖（真实 YAML 字段，非猜测）

以下字段路径已对照 `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml` 逐一确认，不是根据字段名猜的：

| UI 参数 | YAML 字段 | 说明 |
|---|---|---|
| `max_epochs` | `trainer.max_epochs` | 基础 YAML 当前为 20 |
| `train_batch_size` | `scratch.train_batch_size`（经 `${scratch.train_batch_size}` 插值传给 `trainer.data.train.batch_size`） | 基础 YAML 当前为 1 |
| `gradient_accumulation_steps` | `scratch.gradient_accumulation_steps`（经插值传给 `trainer.gradient_accumulation_steps`） | 基础 YAML 当前为 4 |
| `learning_rate` | `scratch.lr_transformer` | 可训练部分（transformer/decoder）的学习率，见下方"未覆盖字段"说明 |
| `num_workers` | `scratch.num_train_workers` | 只覆盖训练集 dataloader，见下方"num_workers 范围"说明 |
| `num_gpus` | 不写入 YAML，走现有 `--num-gpus` CLI 参数 | `sam3/train/train.py:173-174` 已确认 `--num-gpus` 会在运行时覆盖 `cfg.launcher.gpus_per_node`，因此预检不需要重复写 YAML |

### 2.1 `learning_rate` 覆盖范围（刻意收窄，非遗漏）

基础 YAML 里实际有三个学习率字段：

- `scratch.lr_transformer`：可训练部分的学习率，来自插值表达式 `${times:8e-4,${scratch.lr_scale}}`（用到 SAM3 自定义的 `times` OmegaConf resolver）。
- `scratch.lr_vision_backbone`：**固定为 `0.0`**（YAML 注释明确写着"冻结策略"，故意冻结视觉骨干）。
- `scratch.lr_language_backbone`：**固定为 `0.0`**（同样故意冻结，因为训练 prompt 固定不变）。

本页面的 `learning_rate` 输入**只覆盖 `scratch.lr_transformer`**，绝不触碰后两个字段——一个通用的"learning rate"输入如果连视觉骨干和文本编码器一起解冻，会破坏 YAML 作者刻意设计的冻结策略。这两个字段目前没有 UI 覆盖入口。

预检在未提供 `learning_rate` 覆盖时，会尝试读取 `scratch.lr_transformer` 的当前值用于展示；但这个字段用到的 `times` 自定义 resolver 只有在训练器真正启动、调用 SAM3 官方的 `register_omegaconf_resolvers()` 后才能求值。预检没有导入这个（较重的 torch/hydra 依赖）模块，所以未覆盖时该值在预检结果里显示为 `null`，并附一条 warning 说明原因——**不会去猜测或手算 `8e-4 * lr_scale`**。runtime YAML 里对应字段保持原始插值表达式不动，真正训练时仍会被 SAM3 自己的机制正确求值。

### 2.2 `num_workers` 只覆盖训练集

`scratch.num_train_workers`（基础 YAML 当前 10）和 `scratch.num_val_workers`（基础 YAML 当前 0）是两个独立字段。本页面的 `num_workers` 只覆盖前者；验证集固定使用基础 YAML 的值——验证集只有 2 张图（见 `docs/training_path_audit.md`），没有必要暴露单独的验证集 worker 数覆盖。

`num_workers=0` 是合法值（PyTorch DataLoader 用主进程加载数据的标准用法，基础 YAML 自己的验证集就是这么设的），不会被误判为"未设置"或报错。

### 2.3 Effective batch size

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

预检结果里同时显示 `requested_*`（用户填的原始值）和解析后的最终值（字段名不带 `requested_` 前缀，例如 `train_batch_size`/`max_epochs`/`learning_rate`/`num_workers`），方便核对"我填的" vs "实际会用的"是否一致。

### 2.4 参数留空的处理

所有覆盖参数在 UI 上都是 Textbox，留空（或只有空白字符）表示"不覆盖，回退到权威基础 YAML 的值"——不会因为 Gradio 数值组件对空值的处理不稳定（`gr.Number` 清空后可能变成 `NaN`，见 `docs/stage_d_ui.md` 的 D1.1 记录）而误写成 0 或 NaN。0（正/负数/小数/NaN/无效文本）对 `max_epochs`/`train_batch_size`/`gradient_accumulation_steps`/`learning_rate` 会被拒绝并报出明确的 validation error；`num_workers` 单独允许 0（见上）。校验逻辑同时存在于 UI 层（`ui/ui_utils.py::parse_optional_positive_int/float`）和 core 层（`core/training_runner.py::_validate_positive_override`）——core 层校验不依赖 UI 是否已经清洗过参数，因为 `scripts/training_preflight.py` 这个 CLI 入口也会直接调用同一个函数，可能被人手工传入非法值。

## 3. Runtime YAML 生成规则

- 权威基础 YAML（`/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`）永远只读，从不被修改。
- 每次预检都在 `/home/book/book01/runs/training/<run_id>/` 下创建一个全新目录（时间戳命名，`core/training_runner.py::unique_training_run_dir()`），已存在且非空则直接报错，不会覆盖旧的训练 run。预检结果同时生成一次性的启动凭证；成功启动训练后该凭证立即失效，不能再次复用同一个 `run_id`、run directory 或 runtime YAML。
- 该目录下写入：
  - `config/runtime_config.yaml`：本次实际会用的完整配置（基础 YAML 的深拷贝 + 各项覆盖）。
  - `dataset_info.json`：解析后的绝对路径、COCO 摘要、`requested_*`/resolved 的所有覆盖参数。
  - `command.txt`：最终训练命令的文本形式。
  - `logs/`、`checkpoints/`：训练器会往这两个目录写文件（本轮预检阶段只是预先创建空目录）。
- `output_root` 必须位于 `/home/book/book01` 或 `/home/book/sam301` 内部，否则报错，除非显式传 `allow_external_output=True`（目前 UI 没有暴露这个开关，等于总是要求写在工作区内）。

## 4. 如何启动训练

阶段 B 只有在以下条件**全部**满足时才会真正调用 `subprocess.Popen` 启动子进程（`ui/training_process_manager.py::validate_can_start_training()`，纯函数、可单测，是服务端强制执行的关卡，不是仅前端好看的禁用按钮）：

1. 最近一次预检通过（`preflight_state["ok"] is True`）；
2. 训练 run 目录存在；
3. runtime YAML 存在；
4. checkpoint 文件存在；
5. train/val 的图片目录和 COCO 文件都存在；
6. 当前没有其他活动训练任务；
7. 用户勾选了"我确认这将启动 GPU 训练任务。"；
8. CUDA 可用（`torch.cuda.is_available()`，在 UI 进程内检测）；
9. 请求的 `num_gpus` 不超过实际检测到的 GPU 数量；
10. 本次预检启动凭证尚未被消费；
11. run directory 中不存在 `training_summary.json`，且 `checkpoints/` 下没有已有 checkpoint 产物。

**任何一项不满足都会拒绝启动，并把所有不满足的原因一次性显示出来，不会启动一半再失败。**

启动操作在服务端临界区内完成校验和消费：一旦校验通过，启动凭证会在创建子进程之前立即标记为已消费。因此双击按钮、两个并发 callback、训练完成后再次点击、训练失败后再次点击、用户取消后再次点击，都会被服务端拒绝复用旧预检。即使 `subprocess.Popen` 自身失败，该预检也不会自动恢复；安全做法是重新执行预检，生成新的 run directory 和新的 runtime YAML 后再启动。

参数修改后旧预检立即失效：阶段 A 的每一个输入框都绑定了 `.change()` 事件（`ui/training_preflight_page.py::invalidate_preflight()`），一旦触发就把服务端保存的 `preflight_state` 直接清空（不是只改前端显示），所以哪怕用户改完参数后没重新点预检就去点"启动训练"，`validate_can_start_training()` 在服务端看到的 `preflight_ok` 已经是 `False`，照样会被拒绝——这个保证是在服务端状态层面做的，不依赖浏览器端按钮是否被正确禁用。

真正的训练命令仍然调用官方入口，是参数列表，不经过 shell：

```
conda run -n sam3 python \
  /home/book/sam301/sam3/train/train.py \
  -c <runtime_config.yaml> \
  --use-cluster 0 \
  --num-gpus <num_gpus>
```

这个命令是预检阶段（`core.training_runner.inspect_training_config()`）生成并存入 `preflight_state["command"]` 的，阶段 B 直接复用这个列表，不会重新拼接——避免"预检显示的命令"和"实际执行的命令"出现分歧。

## 5. 如何停止训练

- "停止训练"按钮调用 `training_process_manager.stop()`。
- 子进程用 `start_new_session=True` 启动（复用阶段 D1.1 已经验证过的 `ui/process_manager.py::ProcessManager`，训练页面新建了一个独立实例 `ui/training_process_manager.py::training_process_manager`，和推理页面的 `inference_process_manager`互不影响，两边可以各自最多一个活动任务）。
- 停止时对整个进程组先 `SIGTERM`，等待超时后 `SIGKILL`（`os.killpg`），`conda run` 派生的真正训练进程（孙进程）也会被一并终止。
- UI 进程正常退出时，训练进程管理器通过 `atexit.register(...)` 注册的钩子会调用 `training_process_manager.shutdown()`，尝试停止任何仍在运行的训练任务（同样复用 D1.1 的机制）。该清理先发 `SIGTERM`，等待合理超时后再发 `SIGKILL`。
- 停止后的状态标记为 `cancelled`（区别于训练自己失败退出的 `failed` 和正常结束的 `completed`），并会照常生成 `training_summary.json`。
- `kill -9`、机器断电、内核崩溃等非正常解释器退出不会执行 `atexit`，因此不保证这些情况下自动清理训练进程。

## 6. 日志位置

- 训练子进程的 stdout/stderr 会被实时捕获并显示在页面的日志框里（`ui/training_process_manager.py::training_snapshot()` 每次轮询取一次全量快照）。
- 完整原始日志会一直保留在内存快照里，训练结束（含被停止）后仍可在页面上看到全部历史输出。
- 官方训练器自己也会往 `<run_dir>/logs/book_spine/` 和 `<run_dir>/tensorboard/` 写日志（runtime YAML 里的 `trainer.logging.log_dir`/`trainer.logging.tensorboard_writer.log_dir` 已经指向本次 run 目录），这是训练器自身的行为，UI 没有改动。

## 7. Checkpoint 位置

Runtime YAML 把 `trainer.checkpoint.save_dir` 指向 `<run_dir>/checkpoints`，训练器只会往这个全新目录写 checkpoint，从不写回 `initial_checkpoint` 指向的原始文件（`/home/book/sam301/sam3.pt` 默认情况下只被读取，从未被训练流程打开写入）。`training_summary.json` 里的 `discovered_checkpoint_files` 只列出 `<run_dir>/checkpoints/` 下**真实存在**的文件，不会假设训练"应该"产生了什么文件。

## 8. cancelled / failed / completed 状态

状态判定逻辑（`ui/training_process_manager.py::training_status_label()`）：

- 进程仍在运行 → `running`
- 用户点了"停止训练" → `cancelled`
- 进程自己退出且 exit code 为 0 → `completed`
- 进程自己退出且 exit code 非 0 → `failed`

`completed` 但 `checkpoints/` 目录下没有任何文件时，`training_summary.json` 会额外记一条 warning（"训练报告 exit code 0，但 checkpoints/ 目录下没有找到任何文件"），不会因为 exit code 是 0 就默认一切正常。

## 9. 训练指标解析（best-effort，未经真实日志验证）

如果日志里出现类似 `epoch`/`iter`/`loss`/`lr`/`memory` 的文本，`ui/training_process_manager.py::parse_training_metrics()` 会尝试用正则表达式抓取最近一次出现的值展示出来。**这个解析器从未用真实 SAM3 训练日志验证过**（因为本轮完全没有跑过真实训练），只用合成的假日志文本测试过格式识别能力。解析不到时明确显示 `unavailable`，绝不编造数值；解析过程本身包在 try/except 里，任何异常都不会影响训练进程本身或让训练被误判为失败。**这是已知的未验证项，等用户做过一次真实训练后，应该回来对照真实日志核实/调整这些正则表达式。**

## 10. CUDA 不可用时的处理

训练前会检查 `torch.cuda.is_available()`（在 UI 进程内，继承启动它的终端的 GPU 可见性，见 `docs/stage_d_ui.md` 第 7 节）。因为基础 YAML 把 `trainer.accelerator` 写死为 `cuda`（训练本身就没有 CPU 模式），CUDA 不可用时会直接拒绝启动，不会回退 CPU，也不会尝试修改/重装 CUDA、驱动或 PyTorch。同时会检查请求的 `num_gpus` 是否超过实际检测到的 GPU 数量，超过同样拒绝启动。

## 11. 本轮未实际运行训练

本轮所有训练编排测试（`tests/test_e1_training.py`，当前 59 个用例）全部使用假的 `python3 -c "..."` 命令（比如 `print('epoch 1 loss 1.0')`、`sleep`、`sys.exit(N)`），从未 import 或调用 SAM3 的训练器，从未占用 GPU，从未产生真实 checkpoint。唯一一次真实调用的是 `core.training_runner.inspect_training_config()`（预检本身），它只读取真实的基础 YAML/checkpoint 路径/数据集做路径检查和 COCO 统计，不加载模型、不跑前向传播。

E1 P1 修复轮同样没有运行真实 SAM3 训练；新增测试仍只使用假短进程验证一次性 preflight、并发启动拒绝和训练退出清理。

## 12. 用户如何在普通终端做最小训练测试

本节只生成建议，本轮没有执行任何一条命令去真正训练。

### 12.1 UI 操作步骤

1. 在普通终端（不是受限 agent 沙箱）执行 `conda run -n sam3 python /home/book/book01/app.py`，浏览器打开 `http://127.0.0.1:7860`，切到"训练预检"标签页。
2. 阶段 A 保持大部分字段默认值（默认数据集本身就很小：train 8 张图 186 个标注，val 2 张图 49 个标注，见 `docs/training_path_audit.md`），只把 `max_epochs` 填 `1`，`train_batch_size` 填 `1`，`gradient_accumulation_steps` 填 `1`（缩短单次迭代时间，effective batch size 会变成 1，仅用于验证流程通不通，不代表正式训练该用这个配置）。`output_root` 保持默认 `/home/book/book01/runs/training`（每次都会新建时间戳子目录，不会覆盖任何旧 run）。
3. 点击"运行训练预检 (不会启动训练)"，确认返回的 JSON 里 `errors` 为空列表，记下 `run_dir`/`runtime_config_path`。
4. 阶段 B 勾选"我确认这将启动 GPU 训练任务。"，点击"启动训练"。
5. 观察日志框，确认第一屏日志里出现类似 "missing and/or unexpected keys" 的权重加载检查（YAML 注释里提到的验证点），且没有立刻报错退出。
6. 如果一切正常，等 1 个 epoch 跑完（因为是全量 SAM3 前向+反向，哪怕数据集很小，具体耗时取决于实际 GPU，无法在本文档给出精确预计时间，需要用户在自己机器上实测），观察状态变为 `completed`。
7. 如果想提前终止，随时点"停止训练"，状态会变为 `cancelled`。

### 12.2 等价命令行命令

预检生成的命令会显示在页面 JSON 结果的 `command` 字段和 `<run_dir>/command.txt` 里，格式固定为：

```bash
conda run -n sam3 python \
  /home/book/sam301/sam3/train/train.py \
  -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

`<run_id>` 以本次实际预检生成的时间戳目录为准，不要手动编造。也可以跳过 UI，直接用 CLI 做预检：

```bash
conda run -n sam3 python scripts/training_preflight.py \
  --max-epochs 1 --train-batch-size 1 --gradient-accumulation-steps 1
```

预检 CLI 打印出的 JSON 里同样能找到 `command` 字段，用同样的方式手动执行。

### 12.3 预计会生成的文件

在 `<run_dir>/` 下：

```
config/runtime_config.yaml
dataset_info.json
command.txt
logs/book_spine/...         (训练器写)
tensorboard/...             (训练器写)
checkpoints/...             (训练器写，save_freq=5 表示每 5 epoch 存一次；max_epochs=1 时要看训练器是否在结束时也存一份，需要实测确认)
training_summary.json       (UI 在训练进程退出后写)
```

### 12.4 中止方法

- UI 里点"停止训练"（推荐，会正确清理进程组）。
- 或者在普通终端另开一个窗口，找到 `conda run` 那一行对应的进程，`kill -TERM <pid>` 或直接在 UI 所在终端按 Ctrl+C（这会触发 `atexit` 清理逻辑，尝试自动停止活动训练任务）。

### 12.5 如何确认没有覆盖旧 checkpoint

- 训练前后对比 `/home/book/sam301/sam3.pt` 的文件大小和修改时间（`ls -la /home/book/sam301/sam3.pt`），确认没有变化——按 runtime YAML 的设计，这个文件只会被以只读方式打开加载权重，训练器的 `checkpoint.save_dir` 指向的是全新的 `<run_dir>/checkpoints/`，物理上不是同一个文件。
- 确认 `<run_dir>/checkpoints/` 下出现的是新文件（比如 `checkpoint_0.pt` 之类，具体命名需要以训练器实际产生的文件为准，本文档不编造文件名），而不是修改了 `/home/book/sam301` 目录下的任何东西。
- `training_summary.json` 里的 `discovered_checkpoint_files` 会列出这次训练 run 实际产生的 checkpoint 文件路径，全部应该在 `<run_dir>/checkpoints/` 下，不应该出现 `/home/book/sam301` 或 `/home/book/sam3` 路径。

## 13. 已知限制

- 训练指标解析（epoch/loss/lr/GPU 内存）从未用真实日志验证过，见第 9 节。
- `learning_rate` 只覆盖 `scratch.lr_transformer`，不提供 `lr_vision_backbone`/`lr_language_backbone` 的覆盖入口（刻意的，见 2.1 节）。
- `num_workers` 只覆盖训练集，不提供验证集单独覆盖（刻意的，见 2.2 节）。
- 未覆盖 `learning_rate` 时，预检展示的"当前基础值"会是 `null`（附解释性 warning），而不是真实的 `8e-5`，因为求值需要 SAM3 自定义的 OmegaConf resolver（见 2.1 节）。
- 训练页面和推理页面各自最多一个活动任务，两者互不影响（可以同时跑一个推理和一个训练——但这不代表 GPU 显存足够，如果真的同时跑，用户需要自行判断显存是否够用；本轮没有做训练/推理并发的显存测试）。
- 未做真实浏览器手动点击"启动训练"再点"停止训练"的端到端人工验证（会真正调用 SAM3，按要求本轮不执行）；所有训练进程编排逻辑只用假命令测试过。
- `training_summary.json` 的 `warnings`/`errors` 目前只覆盖"exit code 非 0"和"completed 但没发现 checkpoint 文件"两种情况，没有尝试解析训练器自身可能输出的更细粒度错误信息。
