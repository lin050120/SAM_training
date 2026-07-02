# E1 P1 Fix Report

生成时间: 2026-07-02

本报告记录代码审查后两个阻止 E2 的 P1 finding 的核实和最小修复。本轮没有运行 SAM3，没有启动真实训练，没有使用 GPU，没有修改 `/home/book/sam301`。

## P1-1: preflight state 可重复启动

- **是否确认存在**: 是。
- **代码证据**: `ui/training_preflight_page.py::start_training()` 在训练进程成功启动后没有消费或失效 `preflight_state`；`ui.training_process_manager.validate_can_start_training()` 只检查路径、确认框、CUDA 和是否已有活动训练任务。
- **根因**: 预检结果只在参数修改时失效，启动训练不是一次性消费操作。训练完成、失败或取消后，同一个 `run_dir` 和同一份 `runtime_config.yaml` 仍可被再次提交启动。
- **修改**: 为每次成功预检生成 `launch_token`；启动时在服务端锁内完成校验和 token 消费；token 在 `Popen` 前立即消费，启动失败也不恢复；已存在 `training_summary.json` 或 checkpoint 产物的 run directory 直接拒绝复用。
- **测试**: 新增假进程测试覆盖首次启动成功、同一 preflight 二次拒绝、completed/failed/cancelled 后拒绝、重新 preflight 后新 run 可启动、新 run_id/runtime YAML 不复用、并发双启动只有一个成功、确认框缺失拒绝、已有训练产物拒绝、参数修改后拒绝、启动失败后不可复用。

## P1-2: training process manager 缺少退出清理注册

- **是否确认存在**: 是。
- **代码证据**: `ui/process_manager.py` 只对 `inference_process_manager.shutdown` 注册了 `atexit`；`ui/training_process_manager.py` 原先没有注册训练 manager 的 shutdown。`docs/stage_e1_training_ui.md` 已声明训练 UI 正常退出会自动清理，文档与实现不一致。
- **根因**: E1 新增了训练专属 `ProcessManager` 实例，但没有像 D1 推理 manager 一样接入解释器正常退出清理。
- **修改**: `ui/training_process_manager.py` 注册训练 shutdown 包装函数；包装函数调用当前 `training_process_manager.shutdown()`。使用模块级 guard 避免 reload 时重复注册。
- **测试**: 新增测试验证 atexit 注册存在且回调会调用训练 manager shutdown；验证 shutdown 无活动任务安全、重复调用幂等、活动假进程及孙进程被终止、SIGTERM 成功时不走 SIGKILL、SIGTERM 超时后走 SIGKILL、日志和 cancelled 状态保留、推理/训练 manager 互不干扰。

## 相关最小修复

`ui/process_manager.py::ProcessManager.start()` 现在在 `Popen` 抛异常时回滚 `running=False`。这是为了满足 P1-1 的启动失败安全策略：失败的 preflight 仍被消费，但 manager 不能永久卡在 running 状态。

## 剩余限制

- `atexit` 只覆盖解释器正常退出、Ctrl+C 触发的正常清理和 `demo.close()` 一类路径；`kill -9`、断电、内核崩溃无法执行清理。
- `training_summary.json` 仍由 UI 监控生成器自然结束时写入；浏览器断开导致 summary 丢失是审查报告里的 P2-2，本轮按用户边界没有展开修复。
- 未验证真实 SAM3 日志格式、真实 checkpoint 命名和 `max_epochs=1` 时是否保存 checkpoint。

## 是否允许进入 E2

代码层面两个 P1 已修复，允许准备 E2。进入 E2 前仍需人工确认 GPU 可见性、真实训练命令、resume/覆盖风险、`/home/book/sam301/sam3.pt` mtime，以及真实训练日志和 checkpoint 产出格式。
