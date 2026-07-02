# Stage D1: Local Web UI

生成时间: 2026-07-02
最后更新: 2026-07-02 (D1.1 稳定性修复)

## 0. D1.1 修复记录 (2026-07-02)

人工浏览器检查后的稳定性修复，无新页面，未运行 SAM3 或训练：

1. **子进程孤儿问题**: `ui/process_manager.py` 现在用 `start_new_session=True` 启动子进程（Linux 下子进程成为独立会话/进程组的 leader，pgid == 子进程 pid）。停止时先对**整个进程组** `os.killpg(SIGTERM)`，等待 5 秒超时后再 `os.killpg(SIGKILL)`，因此 `conda run` 包装出来的孙进程（真正的 python 推理进程）也会被一并终止，不再遗留孤儿。已捕获的 stdout/stderr 和停止状态在停止后仍可通过 `snapshot()` 读取。UI 进程正常退出（Ctrl+C、`demo.close()`、解释器退出）时通过 `atexit` 注册的 `shutdown()` 自动清理活动任务；UI 进程自身被 `SIGKILL` 时无法拦截，属于已知限制。依旧从不使用 `shell=True`，命令参数列表未改变。
2. **limit 空值 bug**（人工测试发现：留空时报 "limit must be a positive number when set"）: 根因是 `gr.Number` 对空值不做清洗——清空输入框后后端可能收到 `NaN`（`0 <= NaN` 为 False 触发误报），且实测 Gradio 5.50.0 下 `gr.Number(precision=0)` 遇 NaN 会直接抛 `ValueError`。因此 limit 改为 `gr.Textbox(value="")`，由 `ui/ui_utils.py::parse_optional_positive_int()` 统一解析：留空/None → 不加 `--limit`（处理全部图片）；正整数（含 `10.0`、`"10"` 这类等价写法）→ `--limit <int>`；0/负数/小数/NaN/无效字符串 → 明确的 validation error，不生成任务。同时给 score_threshold / confidence_threshold / nms_iou_thresh / min_area 加了 NaN 守卫（NaN 会绕过范围比较、`int(NaN)` 会崩溃）。`core` 层"limit 必须为正数"的校验未改动，问题在 UI 输入解析层解决。训练预检页的 `num_gpus` 若收到 NaN，由页面包装函数的 try/except 转成用户可读错误，本轮未重构。
3. **source_instance_id**: 旧 run 的 NMS npz 缺 `source_instance_ids` 字段时，结果查看页该列显示 `unavailable` 而不是空白。
4. **NMS review 不再手填 ID**: CVAT 导出页的 kept/removed ID 手填框改为下拉框，选项自动来自所选 run 的 `manifest[].nms_removed`；选中 run 时自动加载并展示该 run 已有的 review 信息（对真实 run `2026-07-02_12-28-40` 自动列出 instance 3 vs 7 的 review JSON 和图片路径）。
5. **错误展示口径**: 页面上只显示简短错误信息，完整 traceback 仅写入 UI 服务端日志（`logger.exception`）。

测试: 全量 CPU-only 测试 58 个全部通过（新增进程组终止测试——含孙进程存活性验证、limit 全部 8 类输入用例、pair 下拉/解码测试、`unavailable` 渲染测试），Gradio smoke test 通过，未运行 SAM3。

## 1. 所选 UI 框架

Gradio。

## 2. 选择理由

- 本地启动只需要一行 `demo.launch()`，不需要单独跑前端构建。
- `gr.Tabs` 原生支持多页面，天然匹配本轮 5 个页面（推理任务配置 / 历史运行记录 / 结果查看 / CVAT 导出 / 训练预检）。
- `gr.Image`、`gr.Dataframe`、`gr.Code` 原生支持图片、表格、JSON 展示，不需要额外拼 HTML。
- 长任务日志可以用生成器函数（`yield`）配合 `gr.Textbox` 实现流式刷新，天然适合展示子进程 stdout/stderr；Streamlit 每次交互都会重跑整个脚本，管理"当前活动子进程 + 实时日志 + 停止按钮"这类有状态的长任务会更绕。
- 只需要引入一个新依赖（`fastapi`/`uvicorn` 作为 gradio 的间接依赖一并装好，不需要额外单独安装或自建前后端分离）。

## 3. 依赖

- 新增: `gradio` (`sam3` conda 环境)。
- 附带安装（gradio 的依赖，非独立选择）: `fastapi`, `uvicorn`, `starlette`, `pydantic`, `pandas`（升级）, `pillow`（降级）, `websockets`, `orjson`, `python-multipart` 等。

### 版本选择和一次版本调整

最初按用户批准安装 `gradio>=4.0,<5.0`，装到的最新 4.x 版本 `4.44.1` **导入失败**：

```
ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
```

原因: 当前 `sam3` 环境里 `huggingface_hub` 已经是 `1.19.0`（`sam3`/`timm` 依赖它，不应降级），而 gradio 4.x 系列的 `oauth.py` 用的是 huggingface_hub 1.0 之前就已移除的 `HfFolder` API。经用户二次批准，改为安装 `gradio>=5.0,<6.0`，实际装到 `gradio 5.50.0`，导入正常，且验证了 `torch`/`PIL`/`pandas`/`sam3`/`build_sam3_image_model` 全部仍可正常 import，`huggingface_hub` 版本未被改动。

## 4. 启动命令

```bash
conda run -n sam3 python /home/book/book01/app.py
```

默认监听 `127.0.0.1:7860`，只绑定本地回环地址，`share=False`（不会创建公网隧道）。

**必须在普通终端里启动，不要在受限的 AI coding agent 沙箱里启动**：CUDA 检测（页面一顶部的 "CUDA 检测" 提示）是在 UI 进程自身内 `import torch` 判断的，UI 进程是从哪个终端起的，就继承那个终端的 GPU 可见性。Codex/Claude 的受限执行环境曾经看不到 GPU（`torch.cuda.is_available()=False`），这不代表宿主机本身没有 GPU 或驱动损坏；本次在 Claude Code 会话内实测 `nvidia-smi` 和 `torch.cuda.is_available()` 都能看到真实 GPU（`NVIDIA GeForce RTX 5090`），但不代表所有受限执行环境都一定能看到，仍建议从普通终端启动 UI 进程以确保结果可信。

## 5. 页面说明

代码结构:

```
app.py                          # 只负责组装 gr.Tabs 和启动，不含业务逻辑
ui/
├── __init__.py
├── ui_utils.py                 # CUDA 检测、JSON 安全读取、路径校验、日志
├── run_reader.py                # 集中读取 inference run 目录（run_config/manifest/errors/...)
├── process_manager.py          # 单例，管理"当前活动推理子进程"，Popen 列表参数 + 后台读线程 + 可停止
├── inference_page.py           # 页面一
├── history_page.py             # 页面二
├── results_page.py             # 页面三
├── cvat_page.py                 # 页面四
└── training_preflight_page.py  # 页面五
```

### 页面一: 推理任务配置

- 输入: input image directory / checkpoint / prompt / device / inference threshold / confidence threshold / dtype mode / NMS threshold / NMS metric / NMS mode / min area / category name / output root / limit / segmentation format (polygon/rle/both)。
- 顶部常驻显示 CUDA 检测结果（available / torch 版本 / device 名称）。
- "开始运行" 是一个生成器函数：构建 `subprocess.Popen` 参数列表（**从不使用 `shell=True`**，从不拼接原始 shell 字符串）→ 调用现有 `scripts/run_unified_inference.py` → 后台线程读 stdout/stderr → 每 0.5s 刷新一次日志框和状态框。
- `device=cuda` 且 CUDA 不可用时**直接拒绝启动**，报错信息里明确说明"不会静默回退 CPU"，用户必须显式把 device 改成 cpu 才能继续。
- "停止" 按钮调用 `ProcessManager.stop()`：对整个进程组先 `SIGTERM`，等待超时后 `SIGKILL`（见 D1.1 修复记录第 1 条），停止状态和已产生日志一并保留在日志框里。
- `limit` 是文本输入框：留空表示处理全部图片；只接受正整数（`10.0`/`"10"` 等价写法也接受）；0/负数/小数/NaN/无效文本会得到明确的校验错误（见 D1.1 修复记录第 2 条）。
- 全局只允许一个活动任务：启动时如果已有任务在跑会直接报错，不会误开多个子进程。
- segmentation format 选 `rle`/`both` 时，推理成功结束后会在进程内（非子进程）额外调用一次 `core.cvat_export.export_cvat_package(run_dir, segmentation_format="rle")`，复用现有 CVAT 导出模块，不新写导出逻辑。

### 页面二: 历史运行记录

- 读取 `/home/book/book01/runs/inference` 下所有 run 目录，用 `ui/run_reader.py` 汇总成表格: run_id / created_at / status / 输入图片数 / 成功数 / 失败数 / raw instances / NMS instances / COCO annotations / prompt / checkpoint / device / total time / errors / segmentation formats / CVAT export 状态。
- 每个字段都优先读取 `run_config.json` / `manifest.json` / `errors.json` / `validation_report.json` / `cvat_export/validation_report.json`，缺字段显示 `unknown`/`unavailable`，单个 run 读取失败只影响那一行，不会让整页崩溃（`summarize_run()` 内部对每个 JSON 文件单独 try/except）。
- 不会一次性把所有 run 的 NPZ 读进内存；这一页只读取体积很小的 JSON 摘要文件。

### 页面三: 结果查看

- 选择 run 和图片后展示: 原图、raw/NMS 可视化（切换）、实例信息表（`annotation_id`/`source_instance_id`/`score`/`bbox`/`area`，由 COCO annotations 和该图片的 NMS npz 的 `source_instance_ids` 现场 join 得到，只加载当前选中图片的一个 NMS npz，不批量加载）、NMS removed 实例表、NMS review 图片和 JSON（自动按 `manifest[].nms_removed` 里的 kept/source id 去找 `visualizations/nms_review/instance_{kept}_vs_{removed}.png`）。
- 额外报告区: `manifest` 行、`run_config.json`、`errors.json`、`acceptance_report.json`、`validation_report.json`、polygon fidelity report、RLE validation report、旧版 `cvat_export/validation_report.json`；缺失时显示 `not available` 或具体的 "file not found" 信息，不抛异常到界面。
- 本阶段**不**支持浏览器内逐像素编辑/拆分/合并/手绘 mask（按要求明确排除）。

### 页面四: CVAT 导出

- 基于已有 inference run 调用 `core.cvat_export.export_cvat_package()` / `polygon_fidelity_report()` / `write_nms_pair_review()`，**在 UI 进程内直接调用**（这些都是纯 CPU 操作，不需要再起子进程，也不会触发 SAM3）。
- 支持 polygon / rle / both、重新导出、ZIP 打包、polygon fidelity 计算、NMS pair review 生成。NMS pair 通过下拉框选择（选项自动来自所选 run 的 manifest 记录，无需手填实例 ID）；选中 run 时自动展示已有的 review 信息。
- 页面顶部固定显示格式说明: polygon 是有损转换，不适合精确评价；RLE 与最终 NMS mask 逐像素一致，但 **CVAT 的实际人工导入仍待用户手动确认**，页面不会显示"RLE 已成功导入 CVAT"这类结论。

### 页面五: 训练预检 + 训练编排（阶段 E1 已扩展，见下方说明）

> 本节描述阶段 D1 的原始状态。阶段 E1 已经把这个页面从单阶段预检扩展为"预检 + 一键启动训练 + 训练监控"两阶段流程，并新增了 `max_epochs`/`train_batch_size`/`gradient_accumulation_steps`/`learning_rate`/`num_workers` 覆盖（此前的"已知限制"已解除）。**完整的最新说明见 `docs/stage_e1_training_ui.md`**，这里只保留 D1 阶段的历史记录。

- 页面顶部固定横幅: "本页面只生成和验证训练配置，不会启动 SAM3 训练。"（阶段 E1 已改为"必须先完成训练预检，预检通过后才能启动训练。"，因为页面现在确实提供启动入口）
- D1 阶段输入: authoritative config / checkpoint / train images / train COCO / val images / val COCO / training prompt / output root / num_gpus——这些是当时 `core.training_runner.inspect_training_config()` 实际支持的参数。
- D1 阶段已知限制（**阶段 E1 已解除**）: `max_epochs`/`batch size`/`gradient accumulation`/`learning rate`/`num_workers` 当时不能从页面覆盖。
- D1 阶段点击"运行训练预检"直接调用 `inspect_training_config(..., prepare_runtime=True)`，只生成 runtime YAML / `dataset_info.json` / `command.txt` 并展示最终训练命令，**没有任何"开始训练"按钮**（阶段 E1 新增了受服务端强制校验保护的启动按钮，见 `docs/stage_e1_training_ui.md`）。

## 6. 推理命令如何执行

`ui/inference_page.py::build_inference_command()` 生成的是参数列表（`list[str]`），交给 `subprocess.Popen(command, ...)`，不经过 shell，不做字符串拼接。子进程环境变量在 `os.environ` 基础上额外设置 `PYTHONPATH=/home/book/sam301`，保证 `import sam3` 指向 `sam301` 而不是旧的 `/home/book/sam3`。

## 7. GPU / Agent 沙箱说明

- UI 进程的 GPU 可见性 = 启动它的终端的 GPU 可见性。必须用普通终端启动 UI 才能让"CUDA 检测"栏反映真实情况。
- Codex/Claude 这类受限编码 agent 执行环境曾经看不到 `/dev/nvidia*` 设备、`torch.cuda.is_available()=False`，这不代表宿主机驱动坏了。本轮在 Claude Code 会话中实测反而看到了真实 GPU（`nvidia-smi` 正常，`torch.cuda.is_available()=True`，`NVIDIA GeForce RTX 5090`），但这只是当前 agent 沙箱配置的结果，不同受限环境的可见性可能不同，不应据此假设所有 agent 环境都能看到 GPU；仍然建议从普通终端启动本 UI。
- 本轮 UI 开发过程中没有自行运行真实 SAM3 推理，没有启动训练。

## 8. Polygon / RLE 区别

见页面四的说明和 `docs/stage_b2_real_inference.md` / `docs/current_system_analysis.md` 第 8 节，UI 只是原样展示已有报告数据，没有重新计算或改变判定口径。

## 9. CVAT 实际导入状态

自动验证过的只有: pycocotools 可解码、本项目 validator 通过。CVAT 实际导入是否成功、导入后再导出的 mask 是否保持精度，仍然需要用户手动测试，UI 不会声称"已导入成功"。

## 10. 训练预检 vs 正式训练

训练预检只生成 runtime YAML、写 `dataset_info.json`、写 `command.txt`、打印最终训练命令；本 UI 中不存在任何触发 `sam3/train/train.py` 的按钮或路径。用户如果要正式训练，需要自己手动复制预检生成的命令去普通终端执行。

## 11. 已知限制

- 训练预检页无法覆盖 `max_epochs`/`batch_size`/`gradient_accumulation_steps`/`learning_rate`/`num_workers`（见上文页面五说明）。
- 页面三的实例表格里 `source_instance_id` 依赖 NMS npz 里的 `source_instance_ids` 附加字段；对早于该字段引入之前生成的 run，这一列显示 `unavailable`（D1.1 起，不再是空白）。
- 页面二的表格一次性把所有 run 的摘要都渲染出来，run 数量非常多时（比如几千个）没有做分页；未在大量 run 场景下测试性能。
- UI 进程被 `SIGKILL`（`kill -9`）强杀时，`atexit` 清理不会执行，活动中的推理子进程组会存活；正常退出（Ctrl+C / `demo.close()`）会自动清理（见 D1.1 修复记录第 1 条）。

## 12. 未测试项目

- 没有做真实浏览器手动点击各按钮的交互测试（`/verify` 类型的端到端浏览器验证未执行）；本轮验证仅覆盖: 模块 import、`app` import、run reader 对真实 run 的读取、命令生成与校验逻辑、CUDA fail-fast 逻辑、`ProcessManager` 用 `echo`/`sleep` 等无害命令做启动/停止生命周期测试（未跑真实 SAM3 命令）、`core.cvat_export` 在临时复制的 run 目录上跑 polygon/rle/both、`core.training_runner.inspect_training_config()` 在临时 output_root 上跑（未触碰真实 `runs/training`）、`py_compile`、`pytest tests/` 全量回归、UI 用真实端口启动并用 HTTP 请求验证能正常响应后关闭。
- 没有验证多用户并发访问、长时间运行稳定性、大规模数据集下的性能。
- 没有验证 Windows/macOS 环境（仅在当前 Linux 环境验证）。

## 13. 停止进程机制

D1.1 起：子进程用 `start_new_session=True` 启动，在 Linux 下成为独立会话和进程组的 leader（pgid == 子进程 pid）。`ProcessManager.stop()` 先对整组 `os.killpg(SIGTERM)`，等待 5 秒超时后再 `os.killpg(SIGKILL)`，`conda run` 派生的孙进程也会被终止；`killpg` 因权限等原因失败时回退为只向直接子进程发信号并记录 warning。已产生的日志始终保留在内存快照里（`snapshot()`），停止后日志框仍能看到完整历史输出。UI 进程正常退出时由 `atexit` 注册的 `shutdown()` 自动停止活动任务；UI 被 `SIGKILL` 强杀时无法拦截（见"已知限制"）。测试中用 python sleep 命令验证了孙进程确实随停止一起被终止，未用真实 SAM3 命令测试。

## 14. 如何查看日志

- 推理任务的 stdout/stderr 实时显示在页面一的日志框里；同一份日志也会由 `scripts/run_unified_inference.py` 内部的 `core.run_manager.setup_file_logger()` 写到 `runs/inference/<run_id>/logs/run.log`（这是已有机制，UI 没有改动它）。
- UI 进程自身的日志（CUDA 检测、CVAT 导出异常、训练预检异常等）通过标准 `logging` 输出到启动 UI 的终端 stdout，logger 名为 `book_spine_ui`；异常发生时终端里能看到完整 traceback，但界面上只会显示简短的用户可读错误信息（不把长 traceback 泄露到普通 UI 里）。

## 15. 如何回滚或清理未完成任务

- 页面一如果任务卡住或用户想取消，点"停止"即可；已生成的 run 目录（哪怕不完整）会保留在 `runs/inference/<run_id>/` 下，不会自动删除，需要用户手动清理，UI 本身不提供删除 run 的功能（避免误删）。
- 训练预检每次都会新建一个带时间戳的 `runs/training/<run_id>/` 目录，不会覆盖旧的预检结果；如果想清理旧的预检输出，需要用户手动删除对应目录（同样，UI 不提供删除功能）。
- 如果 UI 进程本身需要强制关闭，直接在启动它的终端按 `Ctrl+C` 或 kill 该进程；如果担心遗留孤儿子进程，可以用 `ps aux | grep run_unified_inference.py` 之类的命令手动检查并处理（见"已知限制"一节）。
