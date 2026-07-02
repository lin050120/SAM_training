# E1 P2 Fix Report

生成时间: 2026-07-02

本报告记录 Claude 对 Codex E1 P1 修复复审后建议在真实 `max_epochs=1` 训练前处理的两个 P2 finding。本轮没有运行 SAM3，没有启动真实训练，没有使用 GPU，没有修改 `/home/book/sam301`。

## Claude 复审结论

- P1-1: closed.
- P1-2: closed.
- P2-2: open before this fix.
- P2-3: open before this fix.
- 建议进入 E2: Yes，但建议先修 P2-2/P2-3。

## P2-2: summary finalization 依赖浏览器会话

- **是否确认存在**: 是。
- **根因**: `training_summary.json` 原先只在 `ui.training_preflight_page.start_training()` 的 Gradio streaming generator 自然结束时写入。浏览器刷新、关闭或 callback 被取消后，训练进程仍可能完成，但 finalization 不再执行。
- **修复**: `ui.process_manager.ProcessManager.start()` 增加 `on_finish` 回调；reader 线程在 stdout/stderr 读完、returncode/finished_at 写入后调用训练 summary finalizer。训练启动时注册一次 finalizer，UI polling 只负责显示状态，不再决定最终落盘。
- **安全写入**: summary 写入使用临时文件、flush/fsync 和 `os.replace()` 原子替换；finalization 幂等，已存在 summary 时直接返回磁盘内容。
- **覆盖状态**: completed、failed、cancelled 都由后台生命周期路径写入 summary。shutdown/atexit 正常停止活动训练时会尽量生成 cancelled summary；`kill -9`、断电、内核崩溃仍无法保证。
- **测试**: 新增假进程测试覆盖无 UI polling 自动生成 completed/failed/cancelled summary、stdout/stderr tail 保存、exit code/duration/checkpoint 列表、并发 finalization JSON 有效、无临时文件残留、reader thread 可回收、历史读取从磁盘恢复 summary。

## P2-3: output_root 约束过宽

- **是否确认存在**: 是。
- **根因**: `core.training_runner._inside_workspace()` 同时放行 `/home/book/book01` 和 `/home/book/sam301`；`unique_training_run_dir()` 在路径校验前创建 output root，导致非法路径也可能产生副作用。
- **修复**: 建立 canonical allowlist：训练输出必须位于 `/home/book/book01/runs/training` 本身或其真实子目录下。使用 `Path.expanduser().resolve(strict=False)` 和 `Path.relative_to()`，拒绝字符串前缀伪装、`..` 逃逸、外部 symlink、`/tmp`、`/home/book/sam301`、数据目录和 checkpoint 目录。`unique_training_run_dir()` 不再创建 root，只有校验通过后才由 runtime YAML 写入路径创建本次 run 目录。
- **启动前复核**: `validate_can_start_training()` 复核 run directory 和 runtime YAML 的 canonical 位置，runtime YAML 必须在本次 run directory 内。
- **测试**: 新增临时目录/monkeypatch 测试覆盖合法 canonical root、合法子目录、空格/日文子目录、相对路径归一化、`training_evil`、`..` 到 data、`/tmp`、`/home/book/sam301`、`/home/book/book`、数据集目录、checkpoint 目录、外部 symlink、直接 core API 绕过 UI、篡改 runtime YAML、已有 summary/checkpoint 的 run directory。

## P3 状态

- **R-1**: closed as part of P2-2/P1 launch-path tightening. `training_process_manager.start()` 现在和 token 校验/消费处于同一 `_preflight_launch_lock` 临界区，两个不同 preflight 并发启动时后到请求会看到已有训练运行并被 BLOCKED，而不是消费 token 后报启动失败。
- **R-2**: deferred. 这是 D1 遗留的 stop() 毫秒级竞态，涉及推理和训练共用 stop 语义；本轮不扩大范围。它不阻止 E2。

## 剩余真实训练验证项

- 真实 SAM3 日志格式与指标正则匹配度。
- 真实 checkpoint 文件命名和 `max_epochs=1` 是否保存 checkpoint。
- prompt override 通过 Hydra/SAM3 官方 loader 的端到端行为。
- `/home/book/sam301/sam3.pt` 训练前后 mtime/size 不变。
- GPU 可见性和最小训练显存。

## 是否建议进入 E2

Yes. E1 静态和假进程层面的阻塞问题已修复；进入 E2 前仍需人工做真实训练前检查，并在启动真实训练前等待用户明确批准。
