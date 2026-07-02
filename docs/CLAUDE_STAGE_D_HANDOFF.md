你现在接手一个已经由 Codex 完成阶段 A、B、C 的 SAM3 书脊辅助标注、训练准备和模型评价系统。

本次任务不是从头设计项目，也不是重新实现已有流程。你必须先理解当前代码、运行结果和最新文档，然后在现有系统上继续开发阶段 D：本地 Web UI 第一版。

==================================================
0. 指令优先级
==================================================

当历史文档、旧 Prompt、代码注释和当前代码之间出现冲突时，按以下顺序判断：

1. 用户在本轮对话中的明确要求；
2. 本交接文件；
3. 当前代码和最新真实运行结果；
4. 最新阶段文档；
5. 旧分析文档和历史说明；
6. 代码中的旧注释。

不要因为旧文档中仍写有“待实现”，就重新实现已经完成并验证的功能。

不要因为某个路径、配置或功能曾经存在于旧项目，就默认当前仍然使用它。

==================================================
1. 工作目录和权限
==================================================

Claude Code 当前从中立启动目录运行：

/home/book/claude_workspace

真正允许读写的两个工作目录是：

1. 主项目：
   /home/book/book01

2. SAM3 源代码副本：
   /home/book/sam301

/home/book/claude_workspace 只是 Claude Code 的启动目录，不是项目目录。

所有项目代码、Git 命令、测试和文档操作，应在：

/home/book/book01

中进行。

SAM3 官方代码副本位于：

/home/book/sam301

本阶段原则上不需要修改 /home/book/sam301。

禁止修改：

- /home/book/sam3
- /home/book/book
- Conda base 环境
- NVIDIA 驱动
- CUDA 系统配置
- 系统 Python
- 与本项目无关的文件
- 用户密钥、SSH 配置、云服务凭证等隐私文件

未经用户批准，不得：

- 使用 sudo；
- 安装新依赖；
- 联网下载模型或数据；
- 修改两个授权目录之外的文件；
- 删除数据集；
- 删除人工标注；
- 删除 checkpoint；
- 删除已有 runs；
- 启动正式训练；
- 处理完整数据集；
- 执行长时间 GPU 任务。

==================================================
2. Git 和协作规则
==================================================

Codex 此前已经修改过大量代码。

开始工作前必须执行：

cd /home/book/book01
git status
git branch --show-current
git log -5 --oneline
git diff --stat
git diff

要求：

1. 不得执行 git reset --hard；
2. 不得执行 git clean -fd；
3. 不得丢弃未提交修改；
4. 不得覆盖 Codex 已完成的功能；
5. 不得删除不理解用途的文件；
6. 不得擅自合并、rebase 或 push；
7. 不得自动切换分支，除非用户明确要求；
8. 如果当前工作树不干净，先报告状态，再继续安全修改；
9. 不要将 runs、图片、NPZ、checkpoint 或数据集加入 Git；
10. 修改完成后只报告建议提交内容，不要自行 git push。

如果当前有 Claude 专用分支，例如：

claude-stage-d

可以继续使用。

如果不是该分支，不要擅自切换，只报告实际分支。

==================================================
3. 项目目标
==================================================

这是图书馆或流通会社场景下的书籍机器人识别子系统。

系统最终目标是：

输入书架 RGB 图像
→ 使用 SAM3 分割所有书脊
→ 输出每本书脊的实例 mask
→ 后续供目标书识别、RANSAC 平面拟合和机械臂抓取使用。

当前本阶段重点是 SAM3 辅助标注、CVAT 数据准备、训练配置准备和模型评价系统。

不要把本项目误解为普通开放式 OCR。

整体研究任务是：

inventory-aware / metadata-conditioned target book localization

但本轮 UI 开发主要围绕书脊分割、标注、训练预检和运行结果管理。

==================================================
4. 当前环境
==================================================

Conda 环境：

sam3

推荐执行 Python 的方式：

conda run -n sam3 python ...

SAM3 源代码：

/home/book/sam301

SAM3 checkpoint：

/home/book/sam301/sam3.pt

SAM3 官方训练入口：

/home/book/sam301/sam3/train/train.py

权威训练基础配置：

/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml

当前训练命令模板：

conda run -n sam3 python \
  /home/book/sam301/sam3/train/train.py \
  -c <runtime_config.yaml> \
  --use-cluster 0 \
  --num-gpus 1

不要直接使用基础 YAML 启动训练。

必须通过 training preflight 生成 runtime YAML，避免旧路径生效。

==================================================
5. CUDA 和 GPU 约束
==================================================

普通 Ubuntu 终端中，用户可以正常使用 NVIDIA GPU，并且已经成功运行真实 SAM3 单图推理。

AI coding agent 的受限执行环境可能看不到 GPU。

此前 Codex 会话中出现：

- nvidia-smi 无法与驱动通信；
- 看不到 /dev/nvidia0；
- torch.cuda.is_available() = False；
- 但 PyTorch 是 CUDA 构建，不是 CPU-only。

实际普通终端真实 SAM3 推理已经成功完成。

因此本阶段必须遵守：

1. 不重新诊断或重装 NVIDIA 驱动；
2. 不重装 CUDA；
3. 不重装 PyTorch；
4. 不在 CUDA unavailable 时回退 CPU 跑 SAM3；
5. 不自行运行真实 SAM3 推理；
6. 不自行运行训练；
7. 需要 GPU 的命令只生成给用户；
8. 用户在普通终端执行后，你可以读取产生的 run 和日志；
9. 不把 agent 环境 CUDA unavailable 误认为宿主机驱动损坏。

==================================================
6. 已完成阶段状态
==================================================

阶段 A：completed

已完成：

- 当前系统分析；
- 代码结构梳理；
- 数据流梳理；
- NPZ、COCO、NMS、训练配置检查；
- 路径审计；
- 文档记录。

阶段 B：completed

已经完成真实流程：

原始图片
→ SAM3 推理
→ raw NPZ
→ mask NMS
→ NMS NPZ
→ COCO
→ 可视化
→ manifest
→ run_config
→ errors
→ 统一 inference run 目录

阶段 C：

1. polygon mode completed with lossy mask conversion
2. exact RLE mode implemented and format/exactness validated
3. CVAT RLE 实际人工导入仍是 pending
4. 不得声称 RLE 已经在 CVAT 中实际导入成功

==================================================
7. 真实 SAM3 运行结果
==================================================

真实单图 run：

/home/book/book01/runs/inference/2026-07-02_12-28-40

输入图片：

/home/book/book01/data/book_spine_sam3_dataset/test/images/im_000001.png

图片尺寸：

1280 × 720

prompt：

book spine

checkpoint：

/home/book/sam301/sam3.pt

真实结果：

- raw masks：20
- raw scores：20
- raw mask shape：[20, 720, 1280]
- raw mask dtype：bool
- NMS 后 masks：19
- COCO annotations：19
- bbox：全部合法
- area：19/19 与最终 NMS mask 像素数一致
- annotation_id：唯一
- image_id：唯一且有效
- manifest 路径：全部存在
- errors.json：[]
- raw visualization：可读取
- NMS visualization：可读取

验收报告：

/home/book/book01/runs/inference/2026-07-02_12-28-40/acceptance_report.json

==================================================
8. Polygon 和 RLE 当前状态
==================================================

Polygon fidelity 输出：

/home/book/book01/runs/inference/2026-07-02_12-28-40/cvat_export/polygon_fidelity_report.json

/home/book/book01/runs/inference/2026-07-02_12-28-40/cvat_export/polygon_fidelity_per_instance.csv

Polygon 结果：

- instance count：19
- mean IoU：0.9731655038856
- median IoU：0.9778061224489796
- minimum IoU：0.8740740740740741
- mean Dice：0.9862458728731127
- median area ratio：0.9799950452124365
- 最大面积偏差：0.12592592592592589
- exact match：0/19

这代表：

NMS bool mask → polygon COCO 的格式转换损失

这不是：

SAM3 分割精度

RLE 输出：

/home/book/book01/runs/inference/2026-07-02_12-28-40/cvat_export/rle/instances_default.json

RLE 验证：

- ok=true
- errors=[]
- annotations=19
- exact_mask_matches=19
- exact_mask_total=19

RLE 必须继续直接从最终 NMS bool mask 生成，不能从 polygon 反推。

Both 模式目录：

cvat_export/polygon/
cvat_export/rle/

旧兼容路径：

cvat_export/annotations/instances_default.json

继续保留 polygon，不得破坏兼容性。

格式职责：

- NPZ bool mask：内部权威预测 mask
- RLE：无损 mask 交换和精确评价
- polygon：CVAT 人工编辑兼容格式
- checkpoint 评价：使用 raster GT、NPZ 或 RLE
- 不允许使用 polygon 转换后的 mask 代替原始预测 mask 做精确评价

==================================================
9. CVAT 当前状态
==================================================

当前已经自动验证：

- polygon 可被 pycocotools 解码；
- polygon 项目 validator 通过；
- RLE exact mask 19/19；
- RLE 项目 validator 通过。

尚未真实确认：

- polygon 是否已在当前 CVAT 版本实际导入；
- RLE 是否已在当前 CVAT 版本实际导入；
- RLE 在 CVAT 中是否可编辑；
- CVAT 导入后再导出是否保持 mask 精度。

UI 中必须明确区分：

1. 格式验证通过；
2. CVAT 实际导入成功。

不得把 validator 通过显示为：

“CVAT 已导入成功”。

CVAT/推理 COCO 默认 category：

book_spine

==================================================
10. Category 和 Prompt 约定
==================================================

当前存在三个相关值：

CVAT / 推理 COCO category：

book_spine

现有训练 COCO category：

book spine

SAM3 自然语言 prompt：

book spine

这三个值不要求相同。

当前训练 loader：

- 按 category_id 读取 annotations；
- category name 用于 query text；
- 不按字符串 "book_spine" 精确过滤标注。

训练 prompt 已经和 COCO category 解耦。

训练预检命令：

conda run -n sam3 python \
  scripts/training_preflight.py \
  --training-prompt "book spine"

预检保存：

- coco_category_name
- requested_training_prompt
- resolved_training_prompt
- prompt_source
- runtime YAML
- command.txt
- dataset_info.json

不得重新把 training prompt 和 COCO category name 耦合。

不得修改人工 COCO category。

==================================================
11. 训练配置当前状态
==================================================

权威基础 YAML：

/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml

已确认：

- train_batch_size = 1
- num_gpus = 1
- gradient_accumulation_steps = 4
- effective batch size = 4

真实数据路径：

训练图片：

/home/book/book01/data/book_spine_sam3_dataset/train/images

训练 COCO：

/home/book/book01/data/book_spine_sam3_dataset/train/annotations.json

验证图片：

/home/book/book01/data/book_spine_sam3_dataset/val/images

验证 COCO：

/home/book/book01/data/book_spine_sam3_dataset/val/annotations.json

数据摘要：

- train COCO：8 images，186 annotations
- val COCO：2 images，49 annotations
- missing images：0

训练输出根目录：

/home/book/book01/runs/training

每次训练预检会创建：

/home/book/book01/runs/training/<run_id>/
├── config/runtime_config.yaml
├── dataset_info.json
├── command.txt
└── 其他预检信息

基础 YAML 中可能仍保留历史路径作为模板内容。

正式训练不得绕过 training_preflight.py 直接使用基础 YAML。

==================================================
12. NMS 当前状态
==================================================

真实 run 中 NMS 删除：

- kept source instance：3
- removed source instance：7
- kept score：0.82421875
- removed score：0.462890625
- kept area：42847
- removed area：27719
- intersection：27600
- union：42966
- IoU：0.6423683843038682
- intersection / area(instance 3)：0.6441524494130277
- intersection / area(instance 7)：0.9957069158339046

诊断文件：

/home/book/book01/runs/inference/2026-07-02_12-28-40/visualizations/nms_review/instance_3_vs_7.png

/home/book/book01/runs/inference/2026-07-02_12-28-40/visualizations/nms_review/instance_3_vs_7.json

当前结论：

NMS semantic correctness: manual review required

不要自动声称删除正确。

不要在本阶段重写 NMS。

UI 中可以显示：

- kept instance
- removed instance
- score
- area
- IoU
- containment ratio
- review image
- manual review required

==================================================
13. 当前主要代码
==================================================

必须先阅读并复用。

核心模块：

/home/book/book01/core/config.py
/home/book/book01/core/run_manager.py
/home/book/book01/core/npz_io.py
/home/book/book01/core/mask_nms.py
/home/book/book01/core/coco_export.py
/home/book/book01/core/cvat_export.py
/home/book/book01/core/visualization.py
/home/book/book01/core/inference_run.py
/home/book/book01/core/sam3_adapter.py
/home/book/book01/core/training_runner.py

脚本：

/home/book/book01/scripts/run_unified_inference.py
/home/book/book01/scripts/export_cvat_package.py
/home/book/book01/scripts/accept_inference_run.py
/home/book/book01/scripts/training_preflight.py
/home/book/book01/scripts/check_cuda_environment.sh

测试：

/home/book/book01/tests/test_coco_rle.py

以及 tests/ 中其他已有测试。

文档：

/home/book/book01/docs/current_system_analysis.md
/home/book/book01/docs/stage_a_b_implementation_log.md
/home/book/book01/docs/stage_b2_real_inference.md
/home/book/book01/docs/category_compatibility.md
/home/book/book01/docs/training_path_audit.md
/home/book/book01/docs/cuda_environment_diagnosis.md
/home/book/book01/docs/sam301_changes.md
/home/book/book01/README.md

不要重新创建已有功能的平行实现。

不要新增：

- 第二套 NMS；
- 第二套 COCO exporter；
- 第二套 CVAT exporter；
- 第二套 training preflight；
- 第二套 run manager。

如果 UI 需要调用某项功能，应复用或为现有模块增加薄接口。

==================================================
14. 本轮目标：阶段 D1 本地 Web UI
==================================================

本轮只开发本地 Web UI 第一版。

不开发：

- 正式训练启动；
- checkpoint 批量评价；
- SAM3 全量推理；
- VLM；
- OCR；
- 新 NMS；
- SAM3 源代码重构；
- 浏览器内 mask 编辑器。

Web UI 的目标是把现有命令行流程变成可视化任务入口。

==================================================
15. UI 框架选择
==================================================

先检查 sam3 环境是否安装：

- gradio
- streamlit
- fastapi
- uvicorn

只检查，不安装。

需要比较：

1. 本地启动是否简单；
2. 图片展示是否方便；
3. 多页面或标签页是否清晰；
4. 长任务日志是否方便；
5. 子进程停止是否可控；
6. 历史 run 浏览是否方便；
7. 表格和 JSON 展示是否方便；
8. 文件路径输入是否方便；
9. 后续训练和评价页面扩展是否方便；
10. 是否需要引入大量新依赖。

优先选择：

- 已安装；
- 维护简单；
- 本地使用方便；
- 不需要复杂前后端分离。

如果适合的框架未安装：

停止实施，并向用户报告：

- 需要安装的包；
- 建议版本范围；
- 安装原因；
- 安装命令。

未经用户批准，不得安装。

==================================================
16. 页面一：推理任务配置
==================================================

允许输入：

- input image directory
- checkpoint
- prompt
- device
- inference threshold
- NMS threshold
- output root
- limit
- segmentation format：
  - polygon
  - rle
  - both

需要显示：

- 参数校验结果；
- 最终命令；
- 是否检测到 CUDA；
- 输出目录；
- 当前任务状态；
- stdout；
- stderr；
- 当前处理图片；
- raw mask 数；
- NMS mask 数；
- 成功数；
- 失败数。

实现要求：

1. 调用现有 scripts/run_unified_inference.py；
2. 使用 subprocess 参数列表；
3. 不拼接未经处理的 shell 命令；
4. 实时读取 stdout/stderr；
5. 支持停止子进程；
6. 停止时保存日志；
7. 不静默回退 CPU；
8. 当 device=cuda 且 CUDA 不可用时 fail fast；
9. 不覆盖旧 run；
10. 本阶段测试时不实际运行 SAM3；
11. 只验证命令生成、参数校验和进程控制结构。

==================================================
17. 页面二：历史运行记录
==================================================

读取：

/home/book/book01/runs/inference

显示：

- run_id
- 创建时间
- 状态
- 输入图片数
- 成功图片数
- 失败图片数
- raw instances
- NMS instances
- COCO annotations
- prompt
- checkpoint
- device
- total time
- errors
- segmentation format
- CVAT export 状态

优先读取：

- run_config.json
- run_summary.json
- manifest.json
- errors.json
- acceptance_report.json
- cvat_export validation report

不能只依赖文件夹名。

旧 run 缺字段时：

- 显示 unknown 或 unavailable；
- 显示 warning；
- 不得使整个页面崩溃。

不要一次性加载全部 NPZ 到内存。

==================================================
18. 页面三：结果查看
==================================================

用户选择 run 和图片后，显示：

- 原图
- raw visualization
- NMS visualization
- NMS review image
- raw instance count
- NMS instance count
- score
- bbox
- area
- source_instance_id
- annotation_id
- NMS removed instances
- NMS reason
- errors
- timings

支持：

- 图片切换；
- raw/NMS 视图切换；
- 查看实例信息；
- 查看 manifest；
- 查看 run_config；
- 查看 acceptance report；
- 查看 validation report；
- 查看 polygon fidelity；
- 查看 RLE exact match；
- 查看 NMS pair review。

本阶段不实现：

- 浏览器中逐像素编辑 mask；
- 拆分 mask；
- 合并 mask；
- 手绘 mask。

==================================================
19. 页面四：CVAT 导出
==================================================

基于已有 inference run 调用现有导出代码。

支持：

- polygon
- rle
- both
- 重新导出
- validation
- 显示输出路径
- 显示 images 数
- annotations 数
- errors
- exact match
- polygon mean/median/min IoU
- ZIP 状态，如当前已有该功能

不得重新运行 SAM3。

UI 说明必须准确：

Polygon：
- 可用于 CVAT 兼容流程；
- 属于有损 mask 转换；
- 不适合精确评价。

RLE：
- 与 NMS bool mask 逐像素一致；
- 当前项目验证 19/19 exact；
- 当前 CVAT 版本的实际人工导入仍待用户确认。

不得显示：

“RLE 已成功导入 CVAT”

除非用户以后真实确认。

==================================================
20. 页面五：训练预检
==================================================

本轮只提供训练预检，不提供“开始训练”。

必须在页面顶部明确显示：

本页面只生成和验证训练配置，不会启动 SAM3 训练。

允许输入：

- train images
- train COCO
- val images
- val COCO
- checkpoint
- authoritative config
- training prompt
- output root
- max_epochs
- batch size
- gradient accumulation
- learning rate
- num_workers
- num_gpus

调用现有 training_runner 或 training_preflight。

显示：

- resolved absolute paths
- config exists
- checkpoint exists
- train image count
- train annotation count
- val image count
- val annotation count
- missing image files
- COCO category name
- requested prompt
- resolved prompt
- prompt source
- train batch size
- num_gpus
- gradient accumulation
- effective batch size
- runtime YAML
- dataset_info.json
- command.txt
- warnings
- errors
- 最终训练命令

不得启动训练。

不得修改人工 COCO。

不得修改基础 YAML。

==================================================
21. UI 代码结构
==================================================

不要将全部逻辑写在一个文件。

可以参考：

/home/book/book01/
├── app.py
└── ui/
    ├── __init__.py
    ├── inference_page.py
    ├── history_page.py
    ├── results_page.py
    ├── cvat_page.py
    ├── training_preflight_page.py
    ├── process_manager.py
    ├── run_reader.py
    └── ui_utils.py

最终结构根据所选框架调整。

要求：

1. app.py 只负责组装页面和启动；
2. 页面逻辑拆分；
3. 读取 run 的逻辑集中管理；
4. 子进程管理集中管理；
5. 路径常量从 core.config 读取；
6. 不把绝对路径复制到每个页面；
7. UI 层不重新实现 NMS、COCO、CVAT 或 training preflight；
8. 保持类型注解；
9. 关键函数写 docstring；
10. 统一 logging；
11. 异常转换为用户可读错误；
12. 不在异常时泄露长 traceback 到普通 UI，日志中可以保留。

==================================================
22. 进程管理要求
==================================================

推理任务可能是长任务。

需要设计：

- 单个活动推理任务；
- PID 或 subprocess 句柄；
- stdout/stderr 实时缓存；
- 停止按钮；
- 正确 terminate；
- 必要时等待后 kill；
- 停止状态记录；
- UI 关闭时不要静默遗留孤儿进程。

不能使用：

shell=True

除非有明确必要并说明原因。

使用：

subprocess.Popen([...])

参数必须使用列表。

不要让 UI 自动执行 sudo、pip、conda install 或 git 命令。

==================================================
23. 健壮性要求
==================================================

1. 使用 pathlib.Path；
2. 所有输入路径先检查；
3. 所有 JSON 读取要捕获损坏文件；
4. 支持旧 run 缺字段；
5. 支持空 errors；
6. 支持未生成 CVAT 包的 run；
7. 支持无 timing 的旧 run；
8. 支持无 acceptance_report 的旧 run；
9. 不一次性加载完整数据集；
10. 不扫描 checkpoint 内部；
11. 不读取无关大型文件；
12. 图片按需加载；
13. JSON 中处理 NumPy 类型；
14. 不静默覆盖旧文件；
15. 不改变现有 manifest schema；
16. 新增字段必须向后兼容；
17. 页面错误不能导致整个应用退出；
18. 文件名支持空格、中文和日文。

==================================================
24. 测试范围
==================================================

本轮测试不得调用 GPU。

使用真实已有 run：

/home/book/book01/runs/inference/2026-07-02_12-28-40

至少测试：

1. UI 所有模块可以 import；
2. app 可以 import；
3. history reader 可读取真实 run；
4. run_config 可读取；
5. manifest 可读取；
6. acceptance_report 可读取；
7. polygon fidelity 可读取；
8. RLE validation 可读取；
9. NMS review 可读取；
10. 旧 run 缺字段时不崩溃；
11. 不完整 run 显示 warning；
12. 推理命令生成正确；
13. command 参数使用列表；
14. CUDA unavailable 时的 fail-fast 逻辑；
15. CVAT polygon 调用；
16. CVAT RLE 调用；
17. CVAT both 调用；
18. training preflight 调用；
19. path validation；
20. 文件名含空格；
21. 中文文件名；
22. 日文文件名；
23. py_compile；
24. 现有 tests 回归；
25. UI 启动 smoke test。

UI smoke test 可以短时间启动后自动停止。

不得声称浏览器交互通过，除非确实打开并测试过页面。

不得运行：

- SAM3 推理；
- 正式训练；
- checkpoint 评价；
- 完整数据集处理。

==================================================
25. 文档
==================================================

新增：

/home/book/book01/docs/stage_d_ui.md

更新：

/home/book/book01/README.md

/home/book/book01/docs/current_system_analysis.md

文档至少包括：

1. 所选 UI 框架；
2. 选择理由；
3. 依赖；
4. 启动命令；
5. 页面说明；
6. 推理命令如何执行；
7. GPU 由普通终端启动的 UI 进程使用；
8. Codex/Claude agent GPU 不可见不等于宿主机不可用；
9. polygon/RLE 区别；
10. CVAT 实际导入状态；
11. 训练预检和正式训练的区别；
12. 已知限制；
13. 未测试项目；
14. 停止进程机制；
15. 如何查看日志；
16. 如何回滚或清理未完成任务。

==================================================
26. 实施顺序
==================================================

严格按以下顺序：

1. 检查 Git 状态和分支；
2. 阅读本交接文件；
3. 阅读最新文档；
4. 阅读现有 core 和 scripts；
5. 检查 UI 框架依赖；
6. 输出简短实施计划；
7. 选择 UI 框架；
8. 如果缺依赖，停止并申请批准；
9. 如果依赖可用，实现 UI 薄层；
10. 编写 CPU-only 测试；
11. 执行现有回归测试；
12. 做 UI import 和 startup smoke test；
13. 更新文档；
14. 输出总结；
15. 停止。

不要只给计划后停止，除非：

- 需要安装依赖；
- 发现 Git 工作树存在危险冲突；
- 发现当前代码和文档严重不一致；
- 需要修改 /home/book/sam301；
- 需要 GPU；
- 需要用户做 CVAT 人工测试。

==================================================
27. 本轮明确禁止事项
==================================================

本轮不得：

- 重新执行阶段 A；
- 重写统一 run 目录；
- 重写 NPZ IO；
- 重写 NMS；
- 重写 COCO exporter；
- 重写 CVAT exporter；
- 重写 training preflight；
- 启动 SAM3；
- 启动训练；
- 进行 checkpoint 评价；
- 处理完整数据集；
- 修改 NVIDIA 环境；
- 修改 Conda base；
- 修改人工标注；
- 删除已有 run；
- 将 runs 加入 Git；
- 将 checkpoint 加入 Git；
- 将数据集加入 Git；
- 自动安装依赖；
- 自动推送 Git。

==================================================
28. 完成标准
==================================================

阶段 D1 只有在以下条件满足时才算完成：

1. 本地 Web UI 可以启动；
2. UI 模块化；
3. 可以读取历史 run；
4. 可以显示真实 run；
5. 可以查看原图、raw、NMS 可视化；
6. 可以查看 manifest 和 run config；
7. 可以查看 NMS review；
8. 可以生成推理命令；
9. 可以调用 CVAT 导出接口；
10. 可以显示 polygon fidelity；
11. 可以显示 RLE exact validation；
12. 可以执行训练预检；
13. 不会启动训练；
14. 不会静默 CPU 回退；
15. 停止子进程机制已实现；
16. CPU-only 测试通过；
17. 现有测试没有被破坏；
18. 文档完成；
19. 未测试内容被明确记录。

==================================================
29. 最终汇报格式
==================================================

完成后必须报告：

1. 当前 Git branch；
2. 开始时 Git 状态；
3. UI 框架；
4. 框架选择理由；
5. 是否安装依赖；
6. 新增文件；
7. 修改文件；
8. 删除文件；
9. UI 启动命令；
10. 测试命令；
11. 测试结果；
12. smoke test 结果；
13. 是否启动过 SAM3；
14. 是否启动过训练；
15. 是否使用 GPU；
16. 是否修改 sam301；
17. 已实现页面；
18. 未实现功能；
19. 已知问题；
20. 下一阶段建议。

不得把未运行的测试写成通过。

现在开始：

1. cd /home/book/book01
2. 检查 Git 状态和分支
3. 阅读上述文件
4. 检查 UI 框架依赖
5. 输出简短实施计划
6. 然后继续阶段 D1
