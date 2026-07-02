# Claude 对 Codex E1 P1 修复的独立复审报告

- **复审日期**: 2026-07-02
- **复审模式**: read-only（本轮未修改任何源代码/测试/既有文档，未创建 commit，未运行 SAM3，未启动真实训练，未使用 GPU，未修改 `/home/book/sam301`）
- **本文件是本轮唯一的文件写入。**

---

## 1. 审查范围

- 审查 commit 范围：`9baa545..ca007d1`（Codex P1 修复 commit：`ca007d1` Fix E1 training relaunch and shutdown cleanup）。
- ca007d1 修改的 6 个文件全部逐行审查：
  - `ui/training_preflight_page.py`（+44/-）
  - `ui/training_process_manager.py`（+17/-）
  - `ui/process_manager.py`（+32/-22 中的相关行）
  - `tests/test_e1_training.py`（+362）
  - `docs/stage_e1_training_ui.md`（+15/-2）
  - `docs/E1_P1_FIX_REPORT.md`（新增，35 行）
- 同时按需阅读：`app.py`、`ui/inference_page.py`、`core/training_runner.py`（`unique_training_run_dir`、`inspect_training_config` 运行时写入段）、`tests/test_ui.py`、`tests/test_ui_smoke.py`。
- 权威定义来源：`docs/E1_CODE_REVIEW_REPORT.md`（原始 P1 finding）；对照阅读了 `docs/CODEX_HANDOFF_AFTER_E1_REVIEW.md` 和 `docs/E1_P1_FIX_REPORT.md`。

## 2. Git 状态

- 工作区：`git status --short` 输出为空，**干净**，不存在需要与 commit 区分的未提交修改。
- 当前分支：`codex-e1-p1-fixes`，符合预期。
- `git log -5 --oneline`：`ca007d1` → `9baa545` → `87f4ebb` → `4d743f4` → `2e22f86`，链路符合交接文档描述。
- `git remote -v`：无 remote（符合"禁止 push/配置 remote"边界）。
- `git show --stat ca007d1`：6 files changed, 483 insertions(+), 22 deletions(-)，只触及上述 6 个文件，无越界改动。

## 3. 原始 P1-1 状态与证据

### 结论：**P1-1 closed**

原始问题：训练结束（completed/failed/cancelled）后旧 `preflight_state` 仍 `ok=True`，可用同一 `run_dir` 和同一份 runtime YAML 再次启动训练，造成 checkpoint/summary 混写覆盖。

逐项核查（编号对应复审任务书第四节）：

1. **一次性、不可预测 token**：`ui/training_preflight_page.py:151` — 每次成功预检生成 `uuid.uuid4().hex`（128-bit 随机），存入 state；预检失败/参数解析失败路径全部返回 `_empty_preflight_state()`（`launch_token=None`）。✅
2. **token 与 run_id/YAML/目录/参数绑定**：token 与 `run_dir`、`runtime_yaml`、`command`、resolved 参数同存于同一个服务端 `gr.State` dict（`training_preflight_page.py:140-153`），且任意参数 `.change()` 触发 `invalidate_preflight()` 整体清空（`:285-286`）。绑定方式是"同一服务端状态对象内共存"而非密码学绑定，但 state 不经过浏览器往返，用户无法跨 state 拼接 token，足够。✅
3. **同一临界区完成校验+消费**：`start_training()`（`:202-203`）在 `with _preflight_launch_lock:` 内调用 `_consume_preflight_for_launch()`，该函数在锁内一次完成：token 存在性检查、`consumed` 标志检查、`_consumed_preflight_tokens` 集合检查（`:169`）、`validate_can_start_training()`（含 run_dir 存在性、`training_summary.json`、`checkpoints/` 非空、already_running 等全部检查，`ui/training_process_manager.py:42-73`）、以及消费标记（`:190-192`）。✅
4. **无"先检查、释放锁、再消费"竞态**：检查和 `_consumed_preflight_tokens.add()` 在同一持锁段内，中间不释放锁。✅
5. **并发只有一个通过**：模块级 `threading.Lock` 串行化；第二个 callback 进锁时 token 已在消费集合中 → `preflight_consumed=True` → BLOCKED。有真实双线程测试（见第 6 节第 1-4 点）。✅
6. **不依赖按钮变灰**：整个门禁在服务端（锁 + token 集合 + `validate_can_start_training` 纯函数），文档明确说明按钮禁用只是 UX 提示。✅
7. **subprocess 创建前后消费时机**：token 在锁内、`Popen` 之前消费（`:190` 在 `:214` 的 `training_process_manager.start()` 之前）；`Popen` 本身在锁外执行，不会长时间持锁。✅
8. **启动失败后旧 token 禁止复用**：消费不回滚——`Popen` 异常路径（`:215-218`）只 yield 错误并 return，token 保持已消费；`_consume_preflight_for_launch` docstring 明确记录了这个策略（run dir 可能处于不确定的部分启动状态）。配套 `ProcessManager.start()` 的 running 回滚（`ui/process_manager.py:89-95`）保证 manager 不会卡死（这同时关闭了原始 P2-1）。有专项测试。✅
9-11. **completed/failed/cancelled 后不能复用**：三种终态都发生在启动之后，而 token 在启动时刻已消费，与终态无关地全部拒绝；此外磁盘层还有 `training_summary.json` 检查兜底（completed/failed/cancelled 都会写 summary）。三种状态各有测试。✅
12. **重新预检得到全新四元组**：`run_training_preflight()` 每次调用 `inspect_training_config(prepare_runtime=True)` → `unique_training_run_dir()` 生成新目录、新 `config/runtime_config.yaml`，并生成新 `uuid4` token。测试 `test_repeated_preflight_creates_new_launch_token_run_dir_and_runtime_yaml` 断言三者均不同。✅
13. **时间戳精度**：`core/training_runner.py:220-228` — 秒级时间戳，但有 `while candidate.exists(): candidate = root/f"{base}_{suffix}"` 去重循环；且 `write_runtime_yaml` 用 `mkdir(parents=True, exist_ok=False)`（`:375`）、`inspect_training_config` 有"目录已存在且非空则报 error"检查（`:557-558`）。同秒双预检得到 `_2` 后缀目录，不碰撞。✅
14. **summary 检查 TOCTOU**：磁盘检查在 `_preflight_launch_lock` 临界区内执行；同进程内所有启动路径被该锁串行化，且每个 token 对应一个全新目录、一个 token 只能通过一次——要"复用"一个目录必须持有指向它的未消费 token，而这样的 token 不存在第二个。磁盘检查只是跨进程重启场景的第二层防线，其理论 TOCTOU 窗口在本架构下没有可触发路径。✅
15. **checkpoint 产物检查覆盖面**：`any((run_dir/"checkpoints").iterdir())`（`training_process_manager.py:51`）对**任何**条目（文件、子目录、任意扩展名）都判非空，比按扩展名过滤更保守，覆盖真实 trainer 无论以何种命名/子目录结构写 checkpoint 的情况（runtime YAML 已把 `trainer.checkpoint.save_dir` 固定指向 `<run_dir>/checkpoints`）。✅
16. **不误伤 preflight-only 目录**：预检只创建**空的** `checkpoints/`、`logs/` 和 `config/runtime_config.yaml`、`dataset_info.json`、`command.txt`；空 `checkpoints/` 使 `any(iterdir())` 为 False，`training_summary.json` 不由预检生成——新预检目录可正常启动（测试 `test_new_preflight_after_consumption_can_start_...` 实际验证通过）。✅
17. **绕过路径**：UI 可达的唯一启动入口是 `start_btn.click → start_training`，全程过锁过 token。直接 Python 调用 `training_process_manager.start()` 可以绕过，但这不是 UI 可达路径，且 `ProcessManager` 自身仍保证单活动任务；原始 finding 的攻击面（正常 UI 操作流程）已全部封闭。✅（接受）
18. **UI 重启后默认安全**：`_consumed_preflight_tokens` 和 `gr.State` 同为服务端内存对象，重启后同时消失——旧 state 无法被重放（Gradio 会话状态不跨服务器重启存活）；即使假想 state 被重放，token 不在集合中会**通过** token 检查，此时磁盘层 `training_summary.json`/非空 `checkpoints/` 检查兜底拒绝已训练目录。唯一磁盘层覆盖不到的情形是"上一次训练被 kill -9 且一个 checkpoint 都没写就重启 UI"，而该情形需要 state 跨重启存活才可达，Gradio 架构下不可达（见第 11 节未验证假设第 2 条）。默认安全。✅
19. **错误信息**：BLOCKED 原因为逐条中文说明（如"这次训练预检已经被启动消费，请重新运行训练预检生成新的 run directory"）；异常路径 yield `{exc!r}`（异常 repr，非完整 traceback），完整 traceback 走 `logger.exception` 进服务端日志。✅

## 4. 原始 P1-2 状态与证据

### 结论：**P1-2 closed**

原始问题：`training_process_manager` 无 atexit 注册，UI 正常退出遗留 conda/trainer/孙进程；文档描述了未实现的行为。

逐项核查（编号对应复审任务书第五节）：

1. **atexit.register 实际存在**：`ui/training_process_manager.py:80-82`。✅
2. **注册对象即 UI 使用的实例**：注册的是包装函数 `_shutdown_training_process_manager()`（`:76-77`），调用时解析**当前**模块全局 `training_process_manager`——正是 `:16` 创建、被 `ui/training_preflight_page.py:21-26` import 使用的同一实例。包装函数写法还使 reload 后替换实例也能被正确清理。✅
3. **模块必被 app import**：`app.py:16` → `ui.training_preflight_page` → `ui.training_process_manager`（`:21`），app 启动即注册。✅
4. **无第二个未注册的 training manager**：全仓库 `ProcessManager()` 实例化仅两处——`ui/process_manager.py:165`（inference）和 `ui/training_process_manager.py:16`（training），grep 确认。✅
5-6. **不误杀推理 manager / 两者独立**：两个独立实例、两个独立 atexit 注册（`process_manager.py:171` 和 `training_process_manager.py:81`）；`ui/inference_page.py` 只使用 `inference_process_manager`。测试 `test_inference_and_training_managers_do_not_interfere` 用两个真实子进程验证 shutdown 一个不影响另一个。✅
7. **停止整个 PGID**：`shutdown() → stop() → _signal_group() → os.killpg(process.pid, sig)`（`process_manager.py:119`）；子进程 `start_new_session=True` 是自己的组长，孙进程（`conda run` 派生的真 python）同组同死。既有测试 `test_stop_kills_grandchild_process_group` 用真实三级进程验证孙进程消失。✅
8-9. **先 SIGTERM，超时后 SIGKILL**：`stop()`（`:137-143`）SIGTERM → `wait(timeout)` → `TimeoutExpired` 时 SIGKILL。两个分支各有测试（真实信号投递，只是包一层记录）。✅
10. **幂等**：第二次 `shutdown()` 时 `is_running()` 已 False → no-op；整体包 try/except 不阻塞解释器退出。测试连续调用两次验证。✅
11. **无活动进程安全**：`is_running()` False → 直接返回；`stop()` 内 `self._process is None` 也提前返回。✅
12. **子进程已自然退出安全**：reader 线程置 `running=False` 后 shutdown no-op；即使竞态进入 `stop()`，`poll()` 非 None 跳过信号，`_signal_group` 捕获 `ProcessLookupError`。✅
13. **PID/PGID 复用误杀**：reader 线程 `process.wait()` 收尸与 `killpg` 之间理论上存在纳秒级窗口，但 `killpg` 前有 `poll()` 检查、`ProcessLookupError` 被捕获，Linux 顺序分配 pid 使复用窗口实际不可达。无现实误杀风险。✅
14. **reader 线程结束**：进程组被杀 → stdout 管道 EOF → for 循环退出 → `finally` 收尸并落状态；daemon 线程不阻塞退出。✅
15. **日志尾部**：强制 SIGKILL 时子进程用户态缓冲区未 flush 的行会丢（固有限制），已写入管道的行仍被 reader 捕获；`test_shutdown_is_idempotent_and_preserves_log_and_cancelled_state` 验证已捕获日志在 shutdown 后可读。✅（固有限制，接受）
16. **三态不互相覆盖**：`stopped_by_user` 只在 `stop()` 显式置位；`training_status_label` 判定顺序 running → idle → cancelled → completed/failed，shutdown 路径产生 cancelled，不篡改先前已结束 run 的判定。✅
17. **退出清理路径的 summary**：**shutdown 路径不生成 `training_summary.json`**——summary 仍只由 UI 监控生成器自然结束时写入。这属于原始审查的 P2-2（本轮按边界未修复），`docs/E1_P1_FIX_REPORT.md` "剩余限制"一节已如实声明，不属于 P1-2 范围。⚠️ 已知开放项（见第 11 节）。
18. **重复注册**：模块级 guard `if not globals().get("_TRAINING_SHUTDOWN_REGISTERED", False)`（`:80`）；正常重复 import 走 `sys.modules` 缓存本就不重执行，`importlib.reload` 保留模块 globals 所以 guard 生效。测试用 mock 的 `atexit.register` + reload 验证"首次注册一次、再次 reload 零注册"。即使极端情况下重复注册，`shutdown()` 幂等，无害。✅
19. **Ctrl+C / 正常退出**：Ctrl+C → KeyboardInterrupt → 解释器正常退出 → atexit 执行；`demo.close()`/正常返回同理。合理。✅
20. **文档限制声明**：`docs/stage_e1_training_ui.md:111` 明确写出"`kill -9`、机器断电、内核崩溃等非正常解释器退出不会执行 `atexit`"。"机器重启"未逐字出现，但正常重启走 SIGTERM（见第 11 节第 3 条）、异常重启即断电/内核崩溃，语义已覆盖。✅

## 5. 新发现问题

按"只报告有实际代码证据、可触发的问题"标准，本轮发现 **2 个 P3**（均不阻塞 E2）：

### Finding R-1（P3）：两个不同 preflight 并发启动时，后到者的 token 被消费后才因 manager 冲突报错，报错文案有误导

- **文件/位置**：`ui/training_preflight_page.py::start_training()`（`:202-218`）
- **触发条件**：两个浏览器会话各自完成**独立**预检（两个不同的有效 token），几乎同时点击"启动训练"。会话 A 在锁内消费 token 后释放锁，但尚未执行到 `training_process_manager.start()`；会话 B 进锁时 `already_running=training_process_manager.is_running()` 仍为 False（A 的 manager.start 还没把 running 置 True），B 通过校验并消费了自己的 token；随后 A、B 都调用 `manager.start()`，`ProcessManager.start()` 的锁内 test-and-set 保证只有一个成功，另一个抛 `RuntimeError("A task is already running...")`，被显示为 "ERROR: failed to start training process"。
- **后果**：**安全性无损**——绝不会有两个训练进程，绝不会复用目录（B 用的是自己的全新目录）；损失只是 B 的预检被白白消费、报错文案说"启动失败"而非"已有任务在运行"，用户需重新预检。
- **为什么测试没捕获**：并发测试 `test_concurrent_double_start_only_one_request_succeeds` 只覆盖了"同一个 preflight 双击"（同一 token），没有覆盖"两个不同 preflight 并发"。
- **最小修复建议**：把 `training_process_manager.start(...)` 移入 `_preflight_launch_lock` 临界区（Popen 通常毫秒级，持锁可接受）；或捕获该 RuntimeError 时输出与 BLOCKED 一致的"已有训练任务在运行"文案。**不阻塞 E2**（单用户本地 UI 场景下双会话并发启动本就罕见，且失败方向是安全侧）。

### Finding R-2（P3，D1 遗留，非 ca007d1 引入）：`start()` 的 running 置位与 Popen 返回之间存在毫秒级窗口，此窗口内点"停止"会把新任务误标 cancelled

- **文件/位置**：`ui/process_manager.py::start()`（`:66-97`）与 `stop()`（`:129-134`）
- **触发条件**：同一 manager 之前跑过至少一个任务（`self._process` 非 None 的旧对象）；新一次 `start()` 已在锁内置 `running=True` 但 `Popen` 尚未返回赋值 `self._process` 的毫秒级窗口内，用户恰好点击"停止训练"——`stop()` 拿到的是**旧**进程对象，对新 `ProcessState` 置了 `stopped_by_user=True`，旧进程已退出所以不发信号；新训练照常运行，但自然结束后会被 `training_status_label` 判为 `cancelled` 而非 `completed`。
- **后果**：仅状态标签和 summary 的 `status` 字段错标；进程本身不受影响，无覆盖、无孤儿。
- **为什么测试没捕获**：需要毫秒级人为时序，单测不构造这种交错。
- **最小修复建议**：`stop()` 在锁内同时读取 `self._process` 与 `self._state`，仅当该 process 对应当前 running 状态时才置 `stopped_by_user`。**不阻塞 E2**，属可延后的健壮性打磨。

除此之外，无新的 P1/P2。命名/格式/重构类偏好未列入。

## 6. 测试质量

对照复审任务书第六节 20 点：

1. **真实线程并发**：`test_concurrent_double_start_only_one_request_succeeds` 用两个 `threading.Thread` 真并发调用 `start_training`，不是顺序两次调用。✅
2. **断言只有一个成功**：`assertEqual(sum("status=running" in s), 1)` 且 `assertEqual(sum("BLOCKED" in s), 1)`，并断言拒绝原因是"已经被启动消费"。✅
3-4. **只一个假进程/只一次启动状态**：由"恰好一个 status=running + 恰好一个 BLOCKED"间接保证（`ProcessManager` 单实例单任务，第二次 start 必抛错，且抛错会显示 ERROR 而非 status=running）；另有独立的 `test_second_start_while_running_is_rejected` 直接覆盖 manager 层。✅（间接但充分）
5-7. **completed/failed/cancelled 后拒绝**：`test_completed_failed_and_cancelled_states_do_not_allow_old_preflight_reuse` 三态齐全，cancelled 用真实 sleep 进程 + 真实 `stop()` 产生。✅
8. **subprocess 异常路径**：`test_start_failure_consumes_preflight_and_does_not_leave_manager_running`（页面层，断言 consumed=True 且 manager 可用）+ `test_start_failure_rolls_back_running_state`（manager 层，断言失败后可再次正常 start）。✅
9. **新预检新 run**：`test_repeated_preflight_creates_new_launch_token_run_dir_and_runtime_yaml`（真实预检路径）+ `test_new_preflight_after_consumption_can_start_with_new_run_dir_and_runtime_yaml`（启动路径）。✅
10-11. **已有 summary/checkpoint 目录拒绝**：`test_existing_training_summary_blocks_run_dir_reuse`、`test_existing_training_artifacts_block_run_dir_reuse`（validate 层）+ `test_existing_artifacts_and_parameter_invalidation_block_start`（页面层）。✅
12. **shutdown 用真实进程**：`test_shutdown_is_idempotent_...`、`test_inference_and_training_managers_do_not_interfere` 均启动真实 `python3 -c` 子进程，不是 mock `os.killpg`；SIGTERM/SIGKILL 分支测试对 `_signal_group` 只做记录性包装，真实信号仍被投递、真实进程真实死亡。✅
13. **孙进程消失**：`test_stop_kills_grandchild_process_group`（既有）用真实三级进程 + `os.kill(pid, 0)` 轮询验证孙进程死亡；shutdown 复用同一 `stop()` 路径。✅
14-15. **SIGTERM 成功分支 / SIGKILL fallback 分支**：`test_sigterm_success_does_not_need_sigkill`（正常 sleep 进程，断言未发 SIGKILL）、`test_sigterm_timeout_uses_sigkill`（子进程显式 `SIG_IGN` 忽略 SIGTERM，断言 SIGKILL 被发出）。✅
16-17. **重复 shutdown / 无活动任务**：`test_shutdown_is_idempotent_...` 连续两次；既有 `test_shutdown_with_no_active_task_is_noop`（tests/test_ui.py）。✅
18. **manager 隔离**：`test_inference_and_training_managers_do_not_interfere`。✅
19. **测试残留进程**：所有 setUp/tearDown 均 stop/shutdown；本轮全量跑 1 次 + 并发相关类重复跑 5 次后 `pgrep` 均为空。✅
20. **无缩短/跳过断言**：断言针对外部可观察行为（yield 的状态文本、磁盘上的 summary 文件、真实进程存活性），未见为通过测试而弱化的断言；atexit 测试还额外验证了回调分派到当前实例和 reload 防重。✅

**测试数量核对**：`tests/test_e1_training.py` 42 → 59（+17，与 Codex 报告一致）；全仓库 100 → 117（一致）。

**小瑕疵（不扣结论）**：并发测试成功一方的生成器未显式 close（依赖 GC + tearDown 的 stop 清理），5 次重复运行未见 flake 或残留；点 3/4 的"单进程"是间接断言。均不影响覆盖有效性。

## 7. 实际测试结果

- `conda run -n sam3 python -m pytest tests/ -v`：**117 passed, 5 warnings, 2 subtests passed in 6.46s**（0 failed, 0 error; warnings 均为 gradio/websockets 的 DeprecationWarning，与本次修复无关）。
- Gradio smoke test：`tests/test_ui_smoke.py::UiStartupSmokeTest::test_app_launches_and_serves_http` **PASSED**（含在全量结果中）。
- 稳定性抽查：`StartTrainingOneTimePreflightTest` 全类（含真实线程并发测试）连续运行 5 次，**5/5 全部通过**，无 flake。
- 全部测试仅使用假 `python3 -c` 短进程，CPU-only；未 import SAM3 trainer，未使用 GPU。

## 8. 残留进程检查

测试后执行 `pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml"`（全量测试后一次、5 次重复运行后再一次）：**均无匹配，无残留真实训练进程**。

## 9. 修改范围检查

- ca007d1 只修改第 1 节所列 6 个文件；未触碰 `/home/book/sam301`（其 git 工作区干净）、基础训练 YAML、人工 COCO、checkpoint、真实 `runs/`、NMS/COCO/RLE 代码。
- 未进入 E2（未生成正式 E2 runtime YAML、未启动任何真实训练）。
- diff 中无新增依赖、无 `shell=True`（grep 确认）。
- 无 GPU 工作（测试全程 CPU；`detect_cuda` 在测试中被替换为假对象）。
- 复审时工作区干净，无需要区分的未提交修改。

## 10. 数据与 checkpoint 跟踪检查

`git ls-files | grep -E '^(runs/|data/|experiments/|test_pic/)|\.(npz|pt|pth|ckpt)$'` 输出为空——**没有任何数据、run 目录或模型/checkpoint 文件被 Git 跟踪**。✅

## 11. 未验证假设

1. **Gradio `gr.State` 的原地变更持久性**：`state["consumed"]=True` 依赖 Gradio 按引用传递会话状态。即使某个 Gradio 版本改为按副本传递，模块级 `_consumed_preflight_tokens` 集合仍是独立于 state 的权威防线，安全性不受影响；未对多版本 Gradio 实测。
2. **UI 重启后 `gr.State` 不跨重启存活**：这是 Gradio 的标准行为（内存会话状态），但本轮未做真实"重启服务器 + 旧浏览器标签页重放"的端到端实验。若该假设被打破，磁盘层 summary/checkpoint 检查仍拦截已训练目录，仅"上次被 kill -9 且零 checkpoint"的目录理论上可被重放复用。
3. **`kill -TERM <UI pid>`（不带 -9）是否执行 atexit**：取决于 uvicorn/Gradio 是否安装 SIGTERM 处理器并优雅退出（主线程 launch 时通常会）。文档只承诺了 Ctrl+C/正常退出，未对 SIGTERM 逐一实测。
4. **真实 trainer 的 checkpoint 布局/命名、真实日志格式、`max_epochs=1` 时是否落 checkpoint**：本轮延续"不跑真实训练"边界，这些仍是原审查报告第 5 节列出的 E2 前人工确认项。`checkpoints/` 非空检查对任意布局（含子目录）都保守生效，不依赖这些假设。
5. **P2-2 仍开放（非本轮范围）**：浏览器断开会终止监控生成器，`training_summary.json` 在该路径及 atexit 清理路径下都不会生成。Codex 的修复报告已如实声明；真实数小时训练前建议修复（原审查已定级为"强烈建议、不阻塞"）。P2-3（output_root 过宽 + 校验前 mkdir）同样未在本轮修复、仍开放。

## 12. 是否建议进入 E2

### 结论：**Yes**

- 两个阻塞 E2 的 P1 均已按最小修复原则关闭，修复质量高：服务端锁 + 一次性 token + 磁盘产物双层防线（P1-1），atexit 包装函数 + reload guard + 真实进程组清理测试（P1-2）；顺带正确关闭了 P2-1（Popen 失败回滚）。
- 本轮新发现仅 2 个 P3（均安全侧失败、不影响单用户正常流程），不阻塞。
- 附带条件（进入 E2 前/中仍需遵守，均为已有共识而非新要求）：
  1. E2 是数小时级真实训练时，强烈建议先修 P2-2（summary 写入不应依赖浏览器会话存活），至少要接受"刷新页面即失去监控和 summary"的现状；
  2. P2-3（output_root 收紧）建议一并处理；
  3. 原审查报告第 5 节的 7 项真实训练前人工确认事项（GPU 可见性、prompt 链路、checkpoint 布局、`sam3.pt` mtime 等）全部照做；
  4. E2 只做 `max_epochs=1` 最小验收，启动前等待用户明确批准。
