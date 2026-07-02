# Codex 工作交接文档：阶段 E1 审查后修复与 E2 准备

生成时间: 2026-07-02
交接来源: Claude Code（本文档由 Claude 生成，供全新 Codex 会话接手）
权威审查来源: `/home/book/book01/docs/E1_CODE_REVIEW_REPORT.md`（**必须先读**，本文档中的问题摘要是该报告的摘要，不是替代品）

**交接模式：A（存在阻止 E2 的问题，必须先修复再考虑 E2）**

---

## 0. 你现在接手的是什么

你是 Codex，接手一个已经完成阶段 D1、D1.1、E1 开发，并经过独立代码审查的项目。审查发现了 2 个阻止 E2（真实训练验收）的 P1 问题和 3 个建议修复的 P2 问题。**你本轮的首要任务是修复这些问题，不是开始 E2，也不是做无关重构。**

不要因为看到"训练"两个字就想去跑真实训练。本轮任何情况下都不允许启动真实 SAM3 训练、不允许使用 GPU。

---

## 1. 项目背景（真实目标，不要误解）

这是一个**图书馆或流通公司场景下的书籍机器人识别子系统**。

- 不是开放式 OCR 项目。
- 研究目标是 **inventory-aware / closed-set target book localization**（在已知库存范围内定位目标书籍），不是通用图像理解。
- 当前阶段的核心工作是：SAM3 书脊分割的数据准备、CVAT 标注导出、本地 Gradio UI、SAM3 微调训练预检、一键训练编排、训练进程监控，以及下一步的**真实最小训练验收**。

真实主 pipeline（当前阶段只覆盖其中 SAM3 分割 + 训练编排部分，不要试图现在实现后续环节）：

```
RealSense D435i RGB-D
  → RGB/depth 对齐
  → SAM3 书脊分割
  → OCR 或 VLM 找目标书
  → 输出目标书 mask
  → RANSAC 平面拟合和抓取姿态估计
```

当前阶段**不是**在比较 OCR 和 VLM 哪个更好，那是后续阶段的问题。当前阶段做的是：SAM3 数据准备 → CVAT 导出 → Gradio UI → SAM3 微调训练预检 → 一键训练编排 → 训练进程监控 → **下一步：真实最小训练验收**。

---

## 2. 环境与路径（真实值，不要猜测或使用旧路径）

| 项目 | 路径 |
|---|---|
| 项目根目录 | `/home/book/book01` |
| SAM3 源码 | `/home/book/sam301` |
| Conda 环境 | `sam3` |
| SAM3 checkpoint | `/home/book/sam301/sam3.pt` |
| 权威训练配置 | `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml` |
| 官方训练入口 | `/home/book/sam301/sam3/train/train.py` |
| 训练输出根目录 | `/home/book/book01/runs/training` |
| 推理输出根目录 | `/home/book/book01/runs/inference` |
| 训练图片 | `/home/book/book01/data/book_spine_sam3_dataset/train/images` |
| 训练 COCO | `/home/book/book01/data/book_spine_sam3_dataset/train/annotations.json` |
| 验证图片 | `/home/book/book01/data/book_spine_sam3_dataset/val/images` |
| 验证 COCO | `/home/book/book01/data/book_spine_sam3_dataset/val/annotations.json` |

已知数据量：train 8 张图 / 186 个标注，val 2 张图 / 49 个标注。

训练 prompt：`book spine`；COCO category：`book spine`。**training prompt 和 COCO category 已经在逻辑上解耦，不得重新耦合，不得自动修改人工 COCO category。**

**重要**：`/home/book/sam3`、`/home/book/book`（不带 01）是旧路径，与本项目无关，不要读写。本项目工作目录是 `/home/book/book01`，SAM3 源码是 `/home/book/sam301`（带 01）。

---

## 3. 已完成工作

### 阶段 D1（commit `4d743f4`）
Gradio 本地 Web UI：推理任务配置、历史运行记录、结果查看、CVAT 导出、训练预检五个页面。D1.1 补丁修复了 limit 空值处理和推理子进程的进程组终止问题。**用户已完成人工浏览器验收，当时未发现问题。**

### 阶段 E1（commit `87f4ebb`）
一键训练编排：训练参数覆盖（写入 runtime YAML）、训练预检、GPU 启动前检查、用户确认后启动、stdout/stderr 实时捕获、训练状态判定（completed/failed/cancelled）、停止整个进程组、`training_summary.json` 生成。**测试全部使用假训练进程（`python3 -c "..."`），没有运行过真实 SAM3 训练。**

E1 测试结果：**100 passed**（`conda run -n sam3 python -m pytest tests/ -v`）。

**明确记录，不要假设这些已经做过**：
- 未启动真实训练。
- 未使用 GPU。
- 未修改 `/home/book/sam301`。
- 未验证真实 SAM3 训练日志的实际格式（日志指标解析器完全是猜测性的正则，从未对照真实日志调整过）。
- 未确认真实 checkpoint 文件的命名规则和实际保存路径细节。
- 未验证真实"UI 点击启动训练 → 完成"的端到端流程。

---

## 4. E1 参数映射细节（真实 YAML 字段，已核实非猜测）

| UI/core 参数 | 真实 YAML 字段 |
|---|---|
| `max_epochs` | `trainer.max_epochs` |
| `train_batch_size` | `scratch.train_batch_size` |
| `gradient_accumulation_steps` | `scratch.gradient_accumulation_steps` |
| `learning_rate` | `scratch.lr_transformer` |
| `num_workers` | `scratch.num_train_workers`（只覆盖训练集） |
| `num_gpus` | 不写入 YAML，走官方 `train.py` 已有的 `--num-gpus` CLI 参数（`train.py:173` 已确认会覆盖 `cfg.launcher.gpus_per_node`） |

计划用于真实最小训练测试的默认设定：`max_epochs=1`、`train_batch_size=1`、`gradient_accumulation_steps=4`、`num_gpus=1`、`effective_batch_size=4`、`training_prompt=book spine`。

**刻意不暴露的字段（不要"顺手"加上去）**：
- `scratch.lr_vision_backbone`、`scratch.lr_language_backbone`：基础 YAML 中被有意冻结为 `0.0`（YAML 自带注释说明"冻结策略"）。如果给一个通用的 `learning_rate` 覆盖同时改了这两个字段，会破坏这个冻结策略，因此设计上明确不开放覆盖入口。
- `scratch.num_val_workers`：验证集只有 2 张图，规模很小，沿用基础配置的值，不单独开放覆盖。

**已知限制**：`scratch.lr_transformer` 在基础 YAML 中是 `${times:8e-4,${scratch.lr_scale}}`，用到 SAM3 自定义的 OmegaConf resolver（`times`）。这个 resolver 只在训练器真正启动、调用官方 `register_omegaconf_resolvers()` 后才会注册。预检阶段没有导入这个较重的模块，所以未提供覆盖值时，预检展示的 `learning_rate` 会是 `null` 并附一条 warning 说明原因——**不要试图去猜测或手算这个值（比如 `8e-4 * 0.1`）来"修复"这个 null**，这是设计上的诚实展示，不是 bug。

---

## 5. 本次独立审查发现的问题摘要

**权威来源是 `/home/book/book01/docs/E1_CODE_REVIEW_REPORT.md`，本节只是摘要，你必须先完整阅读那份报告，不能只看这里的摘要就动手改代码。**

审查范围：`4d743f4..87f4ebb`。审查方式：只读静态审查，未修改代码，未运行 SAM3，未启动训练，未使用 GPU。

### Finding P1-1：训练完成后 preflight_state 未失效，同一 runtime YAML 可被重复启动，覆盖上一次的 checkpoint

- **严重等级**：P1
- **是否阻止 E2**：是
- **影响文件**：`ui/training_preflight_page.py`，函数 `start_training()`
- **触发条件**：预检通过 → 启动训练 → 训练完成或取消 → 不修改任何参数 → 再次点击"启动训练"
- **风险**：`preflight_state` 停留在 `ok=True`，`validate_can_start_training()` 的所有条件仍然满足，于是用同一个 `run_dir` 和同一份 runtime YAML 再次启动训练，覆盖 `<run_dir>/checkpoints/` 下已有的 checkpoint 和 `training_summary.json`。这是**正常使用流程**就能触发的路径，不是边角案例，直接破坏本阶段"不覆盖旧 training run"的核心安全承诺。
- **推荐最小修复**：训练启动成功后立即将 `preflight_state` 置为失效状态（一次预检只允许一次启动）；或者在 `validate_can_start_training()` 中新增检查——`<run_dir>/checkpoints/` 必须为空且 `training_summary.json` 不存在才允许启动。前一种方案语义更清晰，推荐优先采用。
- **推荐新增测试**：构造一次成功的 `preflight_state`，用假命令走完一次完整的 `start_training()` 生成器直到结束，然后不修改任何输入，直接再调用一次 `start_training()`，断言第二次调用被拒绝（而不是又启动了一次）。
- **修复后验证方式**：新测试通过；确认 `validate_can_start_training` 或 `preflight_state` 失效逻辑在训练结束后立刻生效，不需要用户手动改参数触发 `invalidate_preflight`。

### Finding P1-2：training_process_manager 未注册 atexit 清理，UI 退出会遗留训练进程组；文档描述了未实现的行为

- **严重等级**：P1
- **是否阻止 E2**：是
- **影响文件**：`ui/training_process_manager.py`（对照 `ui/process_manager.py` 中 `inference_process_manager` 的 atexit 注册方式）
- **触发条件**：训练运行中，用户在终端 Ctrl+C 关闭 UI，或 UI 进程正常退出
- **风险**：`ui/process_manager.py` 只对 `inference_process_manager` 注册了 `atexit.register(inference_process_manager.shutdown)`，`training_process_manager` 没有对应注册。训练子进程用 `start_new_session=True` 启动，已经脱离 UI 会话，UI 退出后该进程组会**静默存活**，成为无人监控的孤儿进程，且此时 `training_summary.json` 永远不会生成（见 P2-2）。更严重的是：`docs/stage_e1_training_ui.md` 第 5 节明确写着"UI 进程正常退出时，atexit 钩子会尝试停止任何仍在运行的训练任务"——**这行代码在仓库里根本不存在**，文档描述了一个未实现的行为，会误导后续开发者和用户。直接违反 E1 完成标准第 9 条（不产生孤儿进程）。
- **推荐最小修复**：在 `ui/training_process_manager.py` 末尾加一行 `atexit.register(training_process_manager.shutdown)`（`ProcessManager` 类已经有现成的 `shutdown()` 方法，直接复用即可，不需要新写逻辑）。
- **推荐新增测试**：断言 `ui.training_process_manager` 模块导入后，`atexit` 的退出回调列表里包含 `training_process_manager.shutdown`（或者用等价方式验证注册确实发生了，比如检查模块源码里存在这行，或者用 `atexit._exithandlers` 间接验证——具体实现方式自行判断，但必须是能捕获"漏掉这一行"这种回归的测试，不能只是 import 不报错）。
- **修复后验证方式**：新测试通过；同时更新（或至少不再自相矛盾）`docs/stage_e1_training_ui.md` 第 5 节的描述，让文档和代码一致。

### Finding P2-1：ProcessManager.start() 中 Popen 抛异常会让 manager 永久卡在 running=True

- **严重等级**：P2（建议在 E2 前修复）
- **是否阻止 E2**：否，但强烈建议修复
- **影响文件**：`ui/process_manager.py`，方法 `start()`
- **触发条件**：`self._state = ProcessState(running=True, ...)` 在 `subprocess.Popen(...)` **之前**执行；如果 Popen 本身抛异常（比如 `conda` 不在 UI 进程的 PATH 里、命令指向的文件不存在），异常会被上层页面捕获显示，但 `running=True` 永远不会被回滚（没有 reader 线程去清理它，因为进程根本没启动成功）。
- **风险**：此后 `is_running()` 永远返回 `True`，后续所有启动尝试（训练**和**推理两个实例都受影响，这是 D1 就存在的共性问题）都会被"已有任务在运行"拒绝，只能重启整个 UI 进程才能恢复。
- **推荐最小修复**：把 `subprocess.Popen(...)` 调用包进 try/except，失败时在锁内把 `running` 重置为 `False`，再重新抛出异常。
- **推荐新增测试**：用一个必然失败的命令（比如不存在的可执行文件路径）调用 `start()`，断言抛出异常后 `is_running()` 返回 `False`，且随后可以正常 `start()` 一个真实命令。
- **修复后验证方式**：新测试通过；确认失败一次后管理器状态可恢复，不需要重启 UI。

### Finding P2-2：浏览器断开/刷新会终止监控生成器，training_summary.json 在这条路径下永远不会生成

- **严重等级**：P2（建议在 E2 前修复）
- **是否阻止 E2**：否，但强烈建议修复
- **影响文件**：`ui/training_preflight_page.py`，函数 `start_training()`（`finalize_training_summary` 只在生成器自然结束时才被调用）
- **触发条件**：真实训练是数小时级任务；用户中途刷新或关闭浏览器标签页，Gradio 会取消该流式事件，生成器在 `time.sleep(1.0)` 或 yield 处被 `GeneratorExit` 中断。
- **风险**：(a) `finalize_training_summary` 永远不会执行，训练即使正常结束也不会有 `training_summary.json`；(b) 没有任何"重新连接监控"的入口——刷新页面后是空白状态，无法再看到正在运行的训练的日志和状态，"训练进程可实时监控"这个能力实际上只在最初打开的那个浏览器会话里成立。训练进程本身不受影响（已经是独立进程组），但可观测性和 summary 双双丢失。
- **推荐最小修复**：把 `training_summary.json` 的写入从"UI 生成器正常结束"这条路径挪到"进程真正退出"这个更底层、更必然发生的事件上（例如给 `ProcessManager` 加一个进程退出时的回调机制，在 reader 线程的 `finally` 块里触发 finalize）；另外加一个独立的"刷新训练状态"按钮，直接调用 `training_snapshot()`，让用户断线重连后还能看到当前状态和已有日志。
- **推荐新增测试**：模拟生成器被提前中断（比如只调用一次 `next()` 就丢弃生成器对象，触发其 `close()`/`GeneratorExit`），确认训练进程仍在跑的情况下 summary 依然会被生成（可能需要额外的轮询或回调机制配合）。
- **修复后验证方式**：新测试通过；人工确认逻辑上"进程退出"和"summary 生成"不再依赖某个特定的 Gradio 生成器还活着。

### Finding P2-3：output_root 约束过宽（允许 sam301 和 data 目录），且在路径校验之前就执行了 mkdir

- **严重等级**：P2（建议在 E2 前修复）
- **是否阻止 E2**：否，但强烈建议修复
- **影响文件**：`core/training_runner.py`，函数 `_inside_workspace()`（同时允许 `BOOK_ROOT` 和 `SAM301_ROOT` 内的路径）；函数 `unique_training_run_dir()`（第一行 `root.mkdir(parents=True, exist_ok=True)` 在路径合法性检查**之前**执行）
- **触发条件**：用户在 UI 的 output root 输入框里填了 `/home/book/sam301/xxx` 或 `/home/book/book01/data/xxx`
- **风险**：填 `/home/book/sam301/xxx` 会直接在 SAM3 源码目录里创建训练输出目录并写入 runtime YAML/日志/checkpoint，违反"原则上不修改 sam301"的边界；填 `data/xxx` 会污染数据集目录。另外，哪怕最终填的是完全在工作区外、本该被拒绝的路径，`unique_training_run_dir` 也已经在校验**之前**把这个目录 mkdir 出来了——校验失败但副作用已经发生。
- **推荐最小修复**：`inspect_training_config` 中把 `output_root` 的合法范围收紧为必须位于 `DEFAULT_TRAINING_RUN_ROOT`（即 `/home/book/book01/runs/training`）之下，除非显式传入 `allow_external_output=True`；同时把 `unique_training_run_dir` 里的 `mkdir` 调用挪到路径校验通过之后再执行。
- **推荐新增测试**：分别用指向 `/home/book/sam301/...` 和 `/home/book/book01/data/...` 的 `output_root` 调用 `inspect_training_config`，断言两者都在 `errors` 里报错，且没有在这些路径下创建任何目录。
- **修复后验证方式**：新测试通过；确认校验失败时文件系统没有任何副作用残留。

### 延后处理的问题（P3，本轮不要求修复，但如果顺手修了也不算越界）

- **P3-1**：`ProcessManager.start()` 里 `is_running()` 检查在锁外，理论上存在 check-then-act 竞态，Gradio 默认串行处理同一事件监听器所以实际很难触发。
- **P3-2**：`_log_lines` 无上限增长，`training_snapshot()` 每秒把全量日志推给浏览器，数小时真实训练下内存和带宽会持续膨胀。
- **P3-3**：core 层 `_validate_positive_override` 接受 `bool` 类型输入（`True` 被当成 `1.0` 通过），UI 层解析器显式拒绝 bool，两层口径不一致，但只有直接调用 CLI 时才可能触达，影响很小。

### 测试缺口（不算 bug，但修复上面的 finding 时应该顺带补上）

1. `start_training()` 的成功路径（生成器主循环、每秒 yield、结束后 finalize）从未被任何测试执行过——这正是 P1-1 没被自动测试拦下的根本原因。
2. 同一 `preflight_state` 二次启动的回归测试（对应 P1-1）。
3. Popen 启动失败后状态可恢复的测试（对应 P2-1）。
4. 训练管理器 atexit 注册存在性的测试（对应 P1-2）。
5. `output_root` 指向 sam301/data 应被拒绝的测试（对应 P2-3）。
6. `prompts` 的 `repr(...)` 字符串是否被官方 `COCO_FROM_JSON` 正确消费——静态审查已确认格式匹配（官方代码就是 `eval(prompts)`），但因为"测试不得 import SAM3"的约束，这一点**无法用自动化测试验证，只能留到真实训练时人工确认**，不属于你本轮需要修复的范围。

### 真实训练前必须人工确认的事项（不是代码 bug，是运行时假设，留给用户在真实训练时验证）

1. prompt override 全链路在真实训练中生效（Hydra instantiate → eval → 长度断言）。
2. checkpoint 实际产出的文件布局（`discovered_checkpoint_files` 只扫描 `checkpoints/` 目录直属文件，若 trainer 写入子目录会漏报）；`max_epochs=1` 配 `save_freq=5` 时结束时是否真的会存一份 checkpoint，需要实测确认。
3. 日志指标正则（epoch/loss/lr/显存）与真实训练日志格式的匹配度——完全未验证，解析不到会诚实显示 `unavailable`，不会伪造数值。
4. 首屏"missing and/or unexpected keys"权重加载检查。
5. `enable_segmentation=True` 在 batch 1 下的显存占用（YAML 注释里写明无实测，可能 OOM）。
6. `num_gpus > 1` 的多卡路径完全未演练过。
7. 训练前后核对 `/home/book/sam301/sam3.pt` 的大小/mtime 未变化（静态分析确认它只被读取，但应该实测一次确认）。

---

## 6. 你（Codex）本轮的任务：模式 A

**根据本次真实审查结论，你现在处于模式 A（存在阻止 E2 的 P1 问题）。**

你本轮只能：

1. 修复 P1-1 和 P1-2（阻止 E2 的问题）。
2. 强烈建议一并修复 P2-1、P2-2、P2-3。
3. 为每一个修复的问题增加对应的回归测试（能够复现原问题、且在修复前会失败、修复后会通过）。
4. 跑 CPU-only 测试全量回归。
5. 跑 Gradio import/startup smoke test。

**你本轮不能**：

- 不进入真实训练。
- 不启动 GPU。
- 不开始 E2 的任何实质工作（不生成 max_epochs=1 的正式 runtime YAML 去准备真实启动，除非你已经完成全部修复且用户明确批准进入 E2 audit 阶段）。
- 不做无关的代码风格重构。
- 不修改测试之外的其他既有稳定代码。

---

## 7. 工作边界（硬性规则，不得逾越）

### 可以读取

- `/home/book/book01`
- `/home/book/sam301`

### 优先只修改

- `/home/book/book01`

### 未经用户明确批准，不得修改

- `/home/book/sam301`
- `/home/book/sam3`（旧路径，无关）
- `/home/book/book`（旧路径，无关，注意与 `/home/book/book01` 区分）
- CUDA / NVIDIA 驱动 / PyTorch / Conda base
- 原始 checkpoint（`/home/book/sam301/sam3.pt`）
- 人工 COCO 标注文件
- 原始训练/推理图片
- 已有的 `runs/` 下的历史 run 目录

### 禁止事项

- `git push`
- 配置 GitHub remote
- 合并 master
- `git reset --hard`
- `git clean`
- 删除用户数据
- 使用 `shell=True`
- 自动启动真实训练
- 自动执行 `max_epochs > 1` 的任何配置
- 自动重跑失败的训练
- checkpoint 批量评价
- 多 checkpoint 排名
- 修改 SAM3 核心训练器代码（`/home/book/sam301` 下的文件）
- 大规模重构 Gradio UI
- 重新实现 `core/` 中已有的功能（不要写第二套 NMS、第二套 COCO exporter、第二套 process manager 等）

---

## 8. Git 规则

开始工作前必须执行并报告结果：

```bash
cd /home/book/book01
git status
git branch --show-current
git log -3 --oneline
git remote -v
```

要求：

1. 不得假设当前工作区是干净的，必须先看实际输出。
2. 不得覆盖用户未提交的修改。
3. 本次交接后，工作区里可能存在两个未提交的新文档：
   - `docs/E1_CODE_REVIEW_REPORT.md`
   - `docs/CODEX_HANDOFF_AFTER_E1_REVIEW.md`

   这两个文件是 Claude 创建的，你必须先报告它们的存在和状态，**未经用户明确要求，不得替 Claude 创建的这两个文档做 commit**。
4. 不得提交以下内容：`runs/`、`data/`、`*.pt`、`*.pth`、`*.ckpt`、`*.npz`、runtime YAML、真实训练日志、checkpoint、图片。
5. 不得执行 `git add .` 或 `git add -A`。只允许显式列出文件名添加相关的源代码、测试和文档。
6. 修复完成后，只有在用户明确要求"提交"时才创建 commit；创建 commit 前，先执行 `git status --short` 和 `git diff --stat` 展示将要提交的内容，确认没有误跟踪数据文件。

---

## 9. 环境提示

- 建议从中立工作目录启动（例如 `/home/book/claude_workspace`），显式获得对 `/home/book/book01` 和 `/home/book/sam301` 的读写授权。
- 如果你的沙箱环境中看不到 GPU（`torch.cuda.is_available()` 为 `False`，`nvidia-smi` 无法通信）：**不得据此判断宿主机没有 GPU，不得修改 CUDA 或重装 PyTorch**。GPU 检查和真实训练必须在用户的普通终端里完成。你本轮的工作范围是静态代码审查发现问题的修复、代码修改、CPU-only 测试，不涉及任何需要 GPU 的操作。

---

## 10. 验收标准

### 如果本轮做的是"修复审查问题"（模式 A 的核心任务）

至少满足：

1. 每一处代码修改都能对应到 `docs/E1_CODE_REVIEW_REPORT.md` 里的一个具体 finding（P1-1、P1-2，以及你选择一并修复的 P2）。
2. 采用最小修复，不做无关重构。
3. 为每个修复的 finding 增加至少一个能够复现原问题的回归测试（即：还原修复前的代码，该测试应该失败）。
4. 现有 100 个测试全部继续通过。
5. 新增测试全部通过。
6. Gradio import/startup smoke test 通过。
7. 不启动 SAM3。
8. 不启动真实训练。
9. 不使用 GPU。
10. 不修改 `/home/book/sam301`。
11. 没有误跟踪任何数据或 checkpoint 文件到 Git。

### 如果全部 P1/P2 已修复，且用户明确要求你准备 E2（不要自己决定进入 E2）

在获得用户明确批准之前，只能做到"准备"这一步，不能实际启动训练：

1. 完成 Git 状态检查。
2. 完成 CUDA 只读检查（只检测，不修改任何东西）。
3. 完成 resume 风险审计（确认新 runtime YAML 不会指向任何已有 checkpoint 目录，不会触发训练器的 resume 逻辑）。
4. 完成 output/checkpoint 覆盖风险审计（确认新 run 目录是全新的、唯一的，不会覆盖任何已有内容）。
5. 生成一个新的、唯一的 `max_epochs=1` runtime YAML（只生成配置，不启动进程）。
6. 展示最终会被执行的完整训练命令。
7. 展示这次预检对应的 `run_id`。
8. 展示输出路径。
9. 展示初始 checkpoint 路径。
10. 明确说明这次训练预计是否会产生 checkpoint 文件（结合 `save_freq` 和 `max_epochs` 的实际配置说明，不要模糊带过）。
11. **在真正执行训练命令之前必须停下来，等待用户的明确批准，不得自行决定启动。**

---

## 11. 建议执行顺序

1. 按顺序阅读：
   - `docs/E1_CODE_REVIEW_REPORT.md`（权威审查来源，必读）
   - `docs/CODEX_HANDOFF_AFTER_E1_REVIEW.md`（本文档）
   - `docs/stage_e1_training_ui.md`
   - `docs/training_path_audit.md`
   - `docs/category_compatibility.md`
   - `core/training_runner.py`
   - `ui/training_preflight_page.py`
   - `ui/training_process_manager.py`
   - `ui/process_manager.py`
   - `tests/test_e1_training.py`
2. 执行第 8 节的 Git 状态检查，如实报告结果。
3. 对审查报告里的每一条 finding，先在代码里定位到具体位置，确认问题确实存在（不要盲目相信摘要，去读实际代码验证），再决定是否修复以及怎么修。
4. 只修复确认存在的问题，按"最小修复"原则。
5. 运行：
   ```bash
   conda run -n sam3 python -m pytest tests/ -v
   ```
6. 运行 Gradio import/startup smoke test（参考 `tests/test_ui_smoke.py` 里已有的写法）。
7. 输出修复摘要和仍然存在的风险（如果某个 P2/P3 你选择不修，要说明原因）。
8. 如果用户明确要求你准备 E2：只做上面第 10 节里列出的"准备"工作，生成 `max_epochs=1` 的配置，展示所有关键信息后**停下来**，等待用户批准，不要自己往下走。

---

## 12. 最终汇报模板（完成本轮工作后按此格式汇报）

1. 当前 branch
2. 当前 HEAD
3. 审查 finding 总数（本文档列出的：2 个 P1 + 3 个 P2 + 3 个 P3 = 8 个，实际以你重新核实后的数字为准）
4. 已修复 finding（列出 finding ID）
5. 未修复 finding（列出 finding ID 及未修复原因）
6. 修改文件（列出所有改动的文件路径）
7. 新增测试（列出新增的测试函数名或测试点数量）
8. 测试结果（例如 "108 passed" 之类的真实数字）
9. Gradio smoke test 结果
10. 是否运行 SAM3（应为：否）
11. 是否运行真实训练（应为：否）
12. 是否使用 GPU（应为：否）
13. 是否修改 sam301（应为：否，除非用户明确批准过例外）
14. 是否生成 E2 runtime YAML（如果本轮只做修复，应为：否）
15. E2 run_id（如果没有生成，写"无"）
16. 最终训练命令（如果没有生成，写"无"）
17. 覆盖风险结论
18. resume 风险结论
19. 是否建议批准真实 `max_epochs=1` 训练（给出你的判断和理由，但不要自行启动）
20. 下一步建议
