# 阶段 E1 独立代码审查报告

- **审查日期**: 2026-07-02
- **当前分支**: claude-stage-e1
- **当前 HEAD**: 87f4ebb (Add stage E1 training orchestration and monitoring)
- **审查范围 commit**: `4d743f4..87f4ebb`
  - D1 基线: `4d743f4` Implement and stabilize stage D1 Gradio UI
  - E1: `87f4ebb` Add stage E1 training orchestration and monitoring
- **审查模式**: read-only（审查期间未修改任何代码/测试/YAML，未运行 SAM3，未启动真实训练，未使用 GPU）

以下内容为本次审查会话中实际产出的审查结论的完整存档，未做美化、未弱化任何已确认问题、未新增无代码证据支持的问题。

---

审查范围：`4d743f4..87f4ebb`，分支 `claude-stage-e1`，工作区在审查发起时干净（仅一个未跟踪的报告 md）。本轮未修改任何文件、未启动训练、未使用 GPU。

## 1. 阻止 E2 的问题

### P1-1：训练完成后 `preflight_state` 未失效，可用同一 runtime YAML 重复启动，覆盖上一次的 checkpoint

- **文件**：`ui/training_preflight_page.py`，`start_training()`（约 150–200 行）
- **触发条件**：预检通过 → 启动训练 → 训练完成（或取消）→ 不修改任何参数，直接再次点击"启动训练"。
- **后果**：`preflight_state` 仍是 `ok=True`，`validate_can_start_training()` 的全部条件再次满足（进程已不在运行、文件都存在、确认框还勾着），于是用**同一个 `run_dir` 和同一份 runtime YAML** 再启动一次训练。第二次训练把 checkpoint 写进同一个 `<run_dir>/checkpoints/`，会覆盖（或触发 trainer 对该目录已有 checkpoint 的 resume 行为，二者都违反"不覆盖旧 training run"），`training_summary.json` 也会被覆盖。这恰好破坏了本阶段最核心的安全承诺，而且是**正常使用流程**就能踩到的路径，不是刁钻边角。
- **最小修复**：启动成功后立即将 `preflight_state` 置为失效（启动即消费，一次预检只允许一次启动）；或在 `validate_can_start_training()` 增加"`<run_dir>/checkpoints/` 必须为空且 `training_summary.json` 不存在"的检查。前者语义更清晰。
- **测试为何没抓到**：`tests/test_e1_training.py` 只测了"运行中启动第二个被拒绝"，没有测"第一次结束后用同一 state 再启动"。

### P1-2：`training_process_manager` 没有注册 atexit，UI 退出会遗留训练进程组；文档与代码矛盾

- **文件**：`ui/training_process_manager.py:15`（对照 `ui/process_manager.py:163`）
- **触发条件**：训练运行中，用户在终端 Ctrl+C 关闭 UI 或 UI 正常退出。
- **后果**：atexit 只注册了 `inference_process_manager.shutdown`；训练子进程用 `start_new_session=True` 启动，已脱离 UI 会话，UI 退出后 SIGHUP 不会到达它——训练进程组**静默存活**，成为无人监控的孤儿（且此时 summary 永远不会生成，见 P2-2）。`docs/stage_e1_training_ui.md` 第 5 节声称"UI 进程正常退出时，atexit 钩子会尝试停止任何仍在运行的训练任务"，**代码里不存在这个注册**——文档描述了一个未实现的行为。直接违反 E1 完成标准第 9 条（不产生孤儿进程）。
- **最小修复**：`ui/training_process_manager.py` 末尾加 `atexit.register(training_process_manager.shutdown)` 一行。
- **测试为何没抓到**：D1 有 `shutdown()` 的单元测试但只针对独立 `ProcessManager` 实例；没有任何测试断言训练管理器模块注册了退出钩子。

## 2. 建议在 E2 前修复的问题

### P2-1：`ProcessManager.start()` 中 Popen 抛异常会把 manager 永久卡死在 running=True

- **文件**：`ui/process_manager.py:66-89`
- **触发条件**：`self._state = ProcessState(running=True, ...)` 在 `subprocess.Popen(...)` **之前**执行；若 Popen 抛异常（如 `conda` 不在 UI 进程 PATH 上、命令文件缺失），异常被页面层捕获显示，但 `running=True` 永不回滚（没有 reader 线程来清它）。
- **后果**：此后 `is_running()` 永远返回 True，所有后续启动（训练**和**推理各自的实例都受影响，这是 D1 遗留的共性 bug）都被"已有任务在运行"拒绝，只能重启 UI。
- **最小修复**：`start()` 里把 Popen 包进 try/except，失败时在锁内把 `running=False` 后重新抛出。
- **测试为何没抓到**：所有测试都用有效的 `python3` 命令，从未测试过启动失败路径。

### P2-2：浏览器断开/刷新会终止监控生成器，`training_summary.json` 在这条路径下永远不会生成

- **文件**：`ui/training_preflight_page.py::start_training()`（生成器末尾才调用 `finalize_training_summary`）
- **触发条件**：真实训练是数小时级任务；用户中途刷新或关闭浏览器标签页，Gradio 会取消该流式事件，生成器在 `time.sleep(1.0)`/yield 处被 GeneratorExit 终止。
- **后果**：(a) `finalize_training_summary` 永远不执行，训练正常结束后没有 `training_summary.json`（审查点 19 的异常路径成立）；(b) 没有任何"重新连接监控"的入口——刷新后页面空白，无法再看到运行中训练的日志/状态（"训练进程可实时监控"实际只在最初的浏览器会话内成立）。训练进程本身不受影响（独立进程组），但可观测性和 summary 双双丢失。
- **最小修复**：把 summary 的写入从 UI 生成器移到进程结束的必然路径上（例如给 `ProcessManager` 加一个进程退出回调，在 reader 线程 `finally` 里触发 finalize）；另加一个"刷新训练状态"按钮直接调 `training_snapshot()`，让断线后可重新查看。
- **测试为何没抓到**：无法在单测里模拟 Gradio 断连；且 start_training 成功路径整体未被测试（见测试缺口 1）。

### P2-3：`output_root` 约束过宽（允许 sam301 和 data 目录），且校验前就执行 mkdir

- **文件**：`core/training_runner.py::_inside_workspace()`（BOOK_ROOT 或 SAM301_ROOT 内都放行）；`unique_training_run_dir()` 第一行 `root.mkdir(parents=True, exist_ok=True)` 在 `check_path` 校验**之前**被调用
- **触发条件**：用户在 UI 的 output root 输入框填 `/home/book/sam301/xxx` 或 `/home/book/book01/data/xxx`。
- **后果**：前者直接在 sam301 里创建训练输出目录并写入 runtime YAML/日志/checkpoint（E1 明令"原则上不修改 sam301"）；后者污染数据集区域。另外即使填的是**工作区外**路径（会被报 error 拒绝），`unique_training_run_dir` 也已经先把 output_root 目录 mkdir 出来了——校验失败但副作用已发生。E1 规范第六节第 7 条要求默认限制在 `/home/book/book01/runs/training`。
- **最小修复**：`inspect_training_config` 中要求 `output_root` 位于 `DEFAULT_TRAINING_RUN_ROOT` 之下（除非 `allow_external_output=True`），并把 `unique_training_run_dir` 的 mkdir 挪到路径校验通过之后。
- **测试为何没抓到**：测试只覆盖了"工作区内通过 / 工作区外报错"两档，没有测 sam301 内、data 内这两个应拒绝的中间档。

## 3. 可以延后处理的问题

- **P3-1** `ui/process_manager.py::start()` 的 `is_running()` 检查在锁外（check-then-act 竞态）。Gradio 默认同一事件监听器串行处理，实际难触发；顺手修复方式是把检查移进锁内做原子 test-and-set。测试未覆盖并发启动。
- **P3-2** 日志无限增长 + 每秒全量传输：`_log_lines` 无上限，`training_snapshot()` 每秒 `"\n".join` 整个列表并经 Gradio 全量推给浏览器。数小时真实训练（log_freq=10）下内存和带宽都会持续膨胀，页面会越来越卡。建议 E2 时给 UI 展示加尾部截断（完整日志已由 trainer 落盘）。
- **P3-3** 校验口径不一致：core 层 `_validate_positive_override(True)` 接受 `bool`（`float(True)=1.0` → 通过），UI 层解析器显式拒绝 bool。仅 CLI 直调场景可触达，影响极小。

## 4. 测试缺口

1. **`start_training()` 成功路径从未被执行**——生成器主循环、逐秒 yield、结束后 finalize 写 summary，整条链路零覆盖。可以用手工构造的 `preflight_state`（`command` 换成假 `python3 -c` 命令、路径指向临时文件）在 CPU-only 条件下完整走通，现有测试没有这么做，这正是 P1-1 漏网的原因。
2. 同一 `preflight_state` 二次启动（P1-1 的直接回归测试）。
3. Popen 启动失败后 manager 状态可恢复（P2-1）。
4. 训练管理器 atexit 注册存在性断言（P1-2）。
5. `output_root` 指向 sam301 内 / data 内应被拒绝（P2-3）。
6. `prompts` 的 `repr(...)` 字符串被官方 `COCO_FROM_JSON` 真实消费——静态核对已确认官方代码就是 `eval(prompts)` 且 `Sam3ImageDataset` 接受 `coco_json_loader` 注入，格式匹配；但受"测试不得 import SAM3"约束无法在本轮测试中执行，归入下节人工确认。

## 5. 真实训练前必须人工确认的事项

1. **prompt override 端到端生效**：runtime YAML 中 `coco_json_loader.prompts` 的 repr 字符串经 Hydra instantiate → `eval` → `assert len(prompts)==len(categories)` 全链路在真实训练里通过（train/val COCO 均为单 category，长度断言应满足，但从未运行过）。
2. **checkpoint 实际产出布局**：`discovered_checkpoint_files` 只扫 `checkpoints/` 直下的文件（`iterdir`+`is_file`），若 trainer 写入子目录会漏报；且 `save_freq=5` 配 `max_epochs=1` 时结束时是否保存需实测。
3. 日志指标正则（epoch/loss/lr/显存）与真实 SAM3 训练日志格式的匹配度——当前完全未验证，解析不到会如实显示 unavailable，不会误报。
4. 首屏 "missing and/or unexpected keys" 权重加载检查（基础 YAML 自己标注的验证点）。
5. `enable_segmentation=True` 在 batch 1 下的显存占用（YAML 注释明示无实测，可能 OOM）。
6. `num_gpus > 1` 路径（`torch.multiprocessing` spawn 分支）完全未演练，首次多卡前单独小规模验证。
7. 训练结束后核对 `/home/book/sam301/sam3.pt` 的大小/mtime 未变（静态分析确认只读，但按文档第 12.5 节流程实测一次）。

## 6. 是否建议进入 E2

**Yes after fixes。**

必须先修复两个 P1（同一预检状态二次启动导致 checkpoint 覆盖；训练管理器缺失 atexit 且文档与代码矛盾），强烈建议一并修复三个 P2（Popen 失败卡死、断线后 summary 丢失/无法重连监控、output_root 约束）。五项的最小修复合计约几十行，且都有明确的回归测试写法（见测试缺口 1–5）。

其余核查点未发现问题：OmegaConf 字段路径全部与真实 YAML 逐行对上（含 `num_gpus` 走 `train.py:173` 已有的 CLI 覆盖）；基础 YAML 中所有 `/home/book/book`、`/home/book/sam3` 旧路径在 runtime YAML 中均被显式覆盖；`sam3.pt` 仅被读取；每次预检生成全新时间戳目录且 `exist_ok=False`，无静默覆盖、无 resume 源；effective batch size 计算正确且有测试；COCO category 与 training prompt 保持解耦；全仓库无 `shell=True`；进程组 TERM→超时→KILL 逻辑正确且孙进程终止有真实测试；completed/failed/cancelled 三态判定正确且有测试；确认框为服务端强制而非仅前端禁用；参数修改使预检失效的机制在服务端状态层面成立（但见 P1-1 的"启动后"漏洞）；测试未过度 mock（预检用真实 YAML 和真实数据集路径）。
