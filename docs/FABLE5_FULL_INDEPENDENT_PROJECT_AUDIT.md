# Fable5 全项目独立审查报告

- 审查日期: 2026-07-03
- 审查模式: 从零独立、只读（未修改任何源码/测试/数据/manifest/run/SAM301；未启动训练；未消费 token；只运行 CPU-only 测试、Gradio smoke、Hydra validate-only 与只读检查）
- 审查方法: 先由源码与真实产物独立得出结论，最后才与既有报告对照。本报告中所有数字均为本轮独立重算，非转述。

## 1. Executive Summary

**训练基础设施是可信的，当前"正式数据集"不是。**

- 基础设施（Hydra 包装、梯度累积数学、补丁完整性守卫、provenance、端口分配、token/进程管理）经独立验证全部正确，182 个测试通过、关键测试三连跑零 flake，最新 run 的每一项量化声明（40 outer / 160 micro / 12 val steps、loss 575.6919、AP 逐位、checkpoint 10,081,250,310 字节、sam3.pt SHA 不变）都被逐一独立证实。**可以冻结。**
- 但两项 P1 使正式多 epoch 训练在当下没有意义：
  - **P1-1**：全部 7185 个"正式"标注是 **SAM3 自己的机器预标注**（六个 COCO 的 `info.description` 均为 "book_spine SAM3 pre-annotation (polygon ~8pts, NMS)"；7185/7185 个多边形顶点数 ≤8——不存在人工修边的可能）。以此训练是对自身零样本输出的自蒸馏，无法改善项目声明要瞄准的漫画误分割/粘连/外扩问题（这些错误就在标签里），8 点八边形 GT 与"机械臂抓取需要高质量边界"直接矛盾。
  - **P1-2**：微调 checkpoint **无法被现有推理链路正确加载**——人工验收物料中的推理 smoke 4/4 张图零预测；根因已定位到 `sam301/sam3/model_builder.py::_load_checkpoint` 只保留 `"detector"` 命名空间的键，而 trainer 保存的是训练命名空间（`backbone.*`/`transformer.*`）→ 微调权重几乎零加载。训练产物当前进不了下游 OCR/RGB-D/抓取管线。
- 另有一组 P2：数据/split/manifest 生成代码完全不在仓库（一次性外部代码产物）；有效样本量与名义严重不符（train 160→120 唯一、**test 12→仅 4 张唯一照片**，各三份拷贝）；指标只有 bbox AP、没有任何 mask 指标。
- **总判定：APPROVE_WITH_FOLLOWUPS**（对基础设施与进入人工验收）；**正式多 epoch 训练不批准**；目录不建议现在重构。

## 2. 项目目标与真实架构

目标（prompt 与历史文档一致）：人工修正书脊标注微调 SAM3 → 高质量书脊实例 mask → OCR/VLM 选书 → RGB-D/RANSAC → 抓取。

真实数据流（逐步核实）：

| 步骤 | 输入 | 输出 | 负责文件 | 关键假设 | 失败处理 | 测试 |
|---|---|---|---|---|---|---|
| CVAT/COCO 源 | dataset_raw 图片 + 预标注 COCO | 6 个 `cvat_import_polygon_*.json` | （生成于早期推理+NMS 导出流程） | **假设是人工修正——实际不是（P1-1）** | 无校验 | 无 |
| 数据审计+split | 6 COCO + 图片 | 2 个 manifest + 3 个 split COCO（gitignored） | **无——代码不在仓库（P2-1）** | 一次性外部脚本正确 | 不可重跑 | 无 |
| preflight | UI/CLI 参数 + 基础 YAML | run 目录 + runtime YAML + token + provenance.json | core/training_runner.py | scratch 键有效、补丁 PATCHED、Hydra 可 compose、effective≤train 数 | 逐项 error，不写 YAML 不发 token | 充分（含真实 subprocess） |
| launch token | preflight state | 一次性消费 | ui/training_preflight_page.py | 服务端锁+集合 | BLOCKED 不消费 | 充分（真并发） |
| launcher | token+state | Popen(conda run … wrapper) | 同上 + ui/process_manager.py | 启动前重查补丁/import/端口 | fail-closed，token 不烧 | 充分 |
| wrapper | runtime YAML 绝对路径 | Hydra compose→官方 main() | scripts/launch_sam3_training.py | initialize_config_dir；末道补丁自检 | SystemExit 非零 | 充分（validate-only 真子进程） |
| SAM3 trainer | cfg | checkpoint/log/tensorboard | sam301 trainer.py（打补丁） | 补丁 loss/accum 数学 | exit≠0→summary failed | 数值等价测试（真实 _run_step） |
| summary/provenance | 进程退出 | training_summary.json（原子写） | ui/training_process_manager.py | reader 线程 on_finish | 幂等/原子 | 充分 |
| 推理使用 | checkpoint.pt | mask/NPZ/可视化 | scripts/run_unified_inference.py + core/sam3_adapter.py | **权重键命名空间兼容——实际不兼容（P1-2）** | 静默零预测 | 无（缺口） |

结论：链条到 checkpoint 落盘为止真实服务于任务；**两端**（GT 来源、checkpoint 出口）与项目核心需求脱节。中间设施没有"功能很多但脱节"的问题，防护层次（token/补丁/端口/provenance）均针对真实发生过的事故逐个建立，非过度工程。

## 3. Git 和环境基线（独立确认）

- branch `e3-formal-training-prep`，HEAD `79f727e`；tag `E2_READY_FOR_FORMAL_TRAINING_PREPARATION` 为 annotated tag（对象 4c38673）→ **commit 29b1bfa** = codex-stage-e2 尖端 = e3 分支基线。分支/tag/阶段划分正确，e3 从正确基线建立，无历史修复遗漏（e3 包含 E2 全部提交，线性历史无分叉合并）。
- 工作区仅 4 个未跟踪 docs（3 个历史评审 + HUMAN_ACCEPTANCE_CHECKLIST.md），无未提交源码。
- 77 个 tracked 文件；`git ls-files` 无 run/数据/图片/权重；.gitignore 覆盖 runs/、data/、\*.pt、图片等，无误提交临时文件。
- commit 历史"文档多"（约半数为报告），但功能/修复/测试/报告各 commit 边界清晰可区分，报告未掩盖代码变化（每个代码 commit diffstat 独立可读）。
- **只在本机有效、checkout 无法恢复的状态**（重要）：① `/home/book/sam301` 补丁（有 manifest+patch+守卫兜底 ✅）；② **3 个 split COCO（gitignored、无生成代码）**；③ 6 个源 COCO 与全部图片（数据不入 git 是合理策略，但配套的再生工具不存在，见 P2-1）；④ conda 环境（仅文档记载版本，无 lockfile 导出）。
- 环境：`import sam3` → `/home/book/sam301/sam3/__init__.py`（实测）；不会回流 `/home/book/sam3`（editable finder 指向 sam301，且守卫黑名单 /home/book/sam3）。绝对路径集中在 core/config.py——换机需改一处+重建数据，可迁移性中等（P3）。

## 4. 训练数据流与最新 run 独立验证

对 `runs/training/2026-07-03_14-42-27`（正式数据 one-epoch）逐项重验，与 Codex 声明对照：

| 声明 | 独立验证 |
|---|---|
| exit 0 / completed / 95.044s | ✅ summary 实测一致 |
| 40 optimizer steps / 160 micro / val 12 | ✅ 日志 `[0][ 0/40]`、`steps_train: 160`、`steps_val: 12` |
| final loss 575.6918604850769 | ✅ train_stats.json 逐位一致 |
| trainer log peak 23 GB | ✅ 实测日志峰值 22.26→23.00 GB（四舍五入口径） |
| val bbox AP 0.8380942391597286 / AP50 0.9771321010827602 | ✅ 日志逐位一致（另有 AP75 0.9234、AP_small 0.667、AR@100 0.880） |
| checkpoint ~10 GB | ✅ 10,081,250,310 字节 |
| sam3.pt 未修改 | ✅ 本轮重算 SHA256 = `9999e234…`，与全历史基线一致 |
| 无残留进程 | ✅ pgrep 无匹配，GPU 回落 |
| runtime YAML 指向正式 split | ✅ ann_file = formal train/val annotations，accum=4 wiring 在位 |
| provenance | ✅ 完整（commit 302b4b1、dirty=True 且如实列出 3 个未跟踪 docs、trainer/patch/manifest/runtime YAML SHA 全部与实物匹配、port 53387、python 路径） |

另发现两个未被叙述的 run：`12-34-30`（EADDRINUSE 失败——端口修复轮的诚实失败记录）与 `14-57-30`（正式 run 之后新建的 preflight-only run，从未启动，token 随 UI 会话消亡，无害残留）。

## 5. 数据集审计（本轮核心，全部独立重算）

- 通配符匹配恰 6 个文件；images/annotations 逐文件：231208=12/253、231750=4/61、183600=4/61、233120=102/4466、161819=10/235、180628=52/2109；**总计 184/7185 ✅**。categories 六文件统一 `id=1, name=book_spine` ✅。dataset_raw 有第 7 个目录（20260623_183533）无对应 COCO，被排除——未见排除理由记录。
- **标注溯源（P1-1）**：六文件 `info.description` 全部为 **"book_spine SAM3 pre-annotation (polygon ~8pts, NMS)"**；顶点分布 8:5844 / 7:1049 / 6:223 / 5:61 / 4:8（=7185，100%≤8）。人工在 CVAT 中修界不可能保持 8 点硬上限。这些文件是本项目自己的"导出给 CVAT 供人工修正"的**输入**（文件名 `cvat_import_*` 即此义），不是修正后的导出。附带发现：旧 22 图 smoke 数据集顶点同样 ≤8——连早期"人工"数据的边界也未经人工精修（可能仅做过接受/删除级审核）。
- **重复（独立聚类）**：44 组字节级重复 ✅（40 组×2 + 4 组×3），92 个文件涉及、48 张冗余 → **全集唯一图 136/184**。4 个 size-3 组恰为 231750(4 张) ≡ 183600(4 张) ≡ 233120 尾部 img_0099-0102.jpg——同一批照片以三个"批次"身份入库（且尾部 4 张是 .jpg 混在 .png 批次内），采集/整理谱系混乱的直接证据。231750 与 183600 两个"批次"整体是同一数据的重复导出（COCO 文件字节大小都相同 16041）。
- **split 独立重验**：以真实 split COCO 为准——exact 重复 **0 跨 split** ✅；本轮自算 aHash（非复用其 manifest 值）跨 split 最小汉明距离 9、≤5 的近重复对 **0** ✅。声明成立。但：**train 160 名义→120 唯一（40 张双份=25% 重复加权）；test 12 名义→4 唯一（每张 ×3，183 个标注=61×3 重复计数）；val 12 唯一**。任何报告均未披露唯一数。组隔离层面：233120 组跨 train(98)/test(4)，但 test 侧 4 张实为其他批次照片的字节拷贝，故实际泄漏为零——判断成立但依赖"尾部拷贝"这一偶然结构，非系统性组隔离。val(231208, 平均 21 标注/图) 与 train(平均 42/图) 分布差 2 倍。同一 split 内近重复 731 对（相邻帧），train 有效多样性进一步低于 120。
- **质量**：极小 mask 口径 `area_ratio<0.001`：声明 286，我算 stored-area 275 / shoelace 285——生成代码不在库无法核对精确定义（未证实的口径差，量级一致）；面积 p1=689px，无 <100px 碎片。多边形 0 越界、0 退化、bbox 全一致；stored area 与 shoelace 差 >10% 的 520 个（约 7%，预标注光栅化口径差，无实害）。184 张图 PIL 实测宽高与 COCO 全一致；184 个 manifest SHA256 与实物全一致；3 个 split COCO SHA 与 split manifest 一致；原始数据 mtime 均早于 manifest 生成时刻，未被脚本修改。
- **可重建性**：dataset manifest 含 per-image split 字段（划分被*记录*），但 split_rule 文本描述含歧义陷阱（img_0099-0102 实为 .jpg，按文本规则重实现极易出错——本审查第一次实现即因此得出假泄漏，后以真实 split 文件纠正）；**无任何工具可重建或校验**（P2-1）；数据更新后 manifest 不会自动失效（stale-manifest 风险敞开，仅靠 SHA 人工比对）。

## 6. 训练数学审计（独立复核）

- 补丁在位：trainer.py:967-968 `backward_loss = loss / accum_steps if accum_steps > 1 else loss`；文件 SHA = manifest patched 值 `bcf5d8d6…`。
- 数学：`Σ∇(Lᵢ/N)=∇(mean Lᵢ)` ✅；日志与 isfinite 用原始 loss ✅；accum=1 逐位不变 ✅；accum=2/4 等价批平均由 `tests/test_grad_accum_numerics.py` 以**真实** `Trainer._run_step`（`__new__`+最小注入，float64、1e-12）验证，非重写模拟 ✅；删除补丁会被数值等价/源码哨兵/三层 hash 守卫共同击中 ✅。
- DDP：模型无条件 DDP 包装（trainer.py:304）；DDP 对 rank 求均值与 /accum 正交，**无重复缩放**；`no_sync` 于前 N-1 micro 生效。num_gpus>1 spawn 路径从未演练（既有已知项）。
- 时序：zero_grad/scheduler(step_schedulers by `where`)/unscale+clipping/scaler.step/update 均每外层迭代一次 ✅。
- 边界：drop_last=True + preflight 拒绝 train<effective、警告不整除（160%4=0 无丢弃）✅；0-step completed 已被 R-5 守卫阻断 ✅。
- checkpoint 内容：model(unwrap DDP)/optimizer/loss/epoch/steps/scaler/best_meters(/train_dataset 状态若有)；resume 恢复上述各项、scheduler 按 `where` 位置重算无需状态；**不含 RNG 状态**（resume 后数据顺序/增广不可位重现，P3）。**book01 层 resume 实际不可用**：run 目录含 checkpoint 即被拒绝复用，且无 resume UI/CLI 入口——trainer 支持 auto-resume，但正式路径设计性禁止（当前 95s/epoch 规模无实害；长训必须先解决，P3→正式训练前 P2）。

## 7. Launcher / Hydra / 进程管理审计

- wrapper `initialize_config_dir` + 官方 `sam3.train.train.main()` 原样调用 ✅；validate-only 真实 compose（本轮实测 exit 0）✅；runtime YAML 即启动实物（sha 入 provenance 且实测匹配）✅。
- 顺序：preflight（写 YAML 前补丁守卫→import guard→YAML→Hydra 验证→端口→provenance）→ UI 发 token → launcher 临界区（reasons 全清→**端口重分配并写回 YAML→token 消费**→import guard→provenance 重算→Popen）。token 一次性（服务端锁+集合，真并发测试）；守卫失败不消费 token（专项测试断言）✅。
- 绕过路径：直接跑 command.txt/裸 train.py 可绕过 token，但 wrapper 内第三层补丁自检仍拦截未打补丁状态；数据/YAML 不再校验——单用户本机模型下可接受（文档已声明正式路径）。
- 端口：bind(0) 内核分配→[p,p] 固定→启动前再验 bindable；TOCTOU 窗口存在但被压至启动锁内、docstring 如实声明；实际发生过的 EADDRINUSE（12-34-30）被诚实记为 failed、无重试、进程清理干净——失败路径行为全部正确 ✅。并发第二个 run 被 already_running 拒绝（单活动任务设计）。
- UI 关闭/断连：summary 由 reader 线程 on_finish 原子落盘（不依赖浏览器）；atexit 停整个 PGID（SIGTERM→SIGKILL）；PID/PGID 记录于 summary；stdout/stderr 合流保存于 summary tail + trainer 自身 log 文件 ✅。误杀风险：killpg 仅对自建会话组，先 poll 后杀，无跨用户风险。stop/cancel 真实存在且经真实进程测试；**resume 不存在**（见上）。
- 遗留：R-3（BLOCKED/ERROR yield 在启动锁临界区内）自 E2 起已知未修，理论性，P3。

## 8. Patch / Provenance 审计

- manifest 三 hash（original `9c9c4159…`/patched `bcf5d8d6…`/patch `f28588dd…`）本轮独立复算全部吻合（patch -R 逆向重建原始 hash 亦验证）。
- manage 脚本语义：UNKNOWN/MISSING 一律拒绝、无 --force；apply/revert 经 dry-run→临时文件→hash 校验→os.replace 原子替换+锁文件（1355667 硬化后）；target 双重限域（必须在 expected_sam3_root 内且非 /home/book/sam3）；patch 文件限域 patches/ 且自身 SHA 校验。测试 10 项覆盖全部状态与回滚。**可靠**。
- provenance：从一次 run 可还原——book01 commit+dirty+status、trainer/patch/manifest/runtime YAML SHA、sam301 root、import path、python、MASTER_ADDR/PORT——launcher 启动时重算非复制 ✅（14-42-27 实测：dirty=True 如实记录且 dirty 内容仅为未跟踪评审文档，不影响运行代码还原）。唯一缺口：conda 环境无 lockfile 级冻结（版本仅散见文档）。

## 9. Checkpoint / Resume / Inference 审计

- **P1-2（阻断性）**：`model_builder._load_checkpoint`（sam301:539-556）先取 `ckpt["model"]` ✅，随后 `{k.replace("detector.",""): v for k,v in ckpt.items() if "detector" in k}`——只接受发布版命名空间（`detector.*`/`tracker.*`）。trainer 保存的键为训练命名空间（optimizer 日志可见 `backbone.vision_backbone.*`、`transformer.*` 等，无 detector 前缀）→ 过滤后近乎空集 → `strict=False` 静默少载 → 模型保持构造初值。实证：人工验收物料 `CHECKPOINT_INFERENCE_SUMMARY.md` 中该 checkpoint 推理 **4/4 图 raw=0 nms=0** + missing/unexpected keys 警告；而训练内 eval AP 0.84 证明权重本身正常——问题确在推理侧加载。当前**没有**：训练/推理 checkpoint 键映射工具、仅权重轻量导出（10GB trainer checkpoint 直接被当推理权重用）、checkpoint 兼容性检查、失败时的 fail-loud（零键加载应报错而非打印警告后照跑）。checkpoint 与数据版本/split 的关联靠 run 目录 provenance 尚可追溯 ✅。
- base/resume/inference checkpoint 概念区分在文档层存在，但代码层缺少导出边界（P1-2 的一部分）。
- 推理输出能力（mask/NPZ/可视化/CVAT）在旧 sam3.pt 路径上自 B/C 阶段验证过；对微调 checkpoint 因 P1-2 未真正验证。

## 10. UI 审计（以代码为准）

5 个 Tab：推理配置、历史记录、结果查看、CVAT 导出、训练预检+启动。训练页真实功能：14 输入框（参数全服务端校验，空串回退基础 YAML）、预检按钮、任意输入 .change() 服务端失效预检、确认框、启动（服务端 token+锁+三守卫）、停止、每秒快照监控、summary 展示。启动稳定（smoke PASSED 2.2s）；预检错误逐条中文一次性给出；训练失败后 UI 状态与 summary 正确恢复（多次真实失败 run 佐证）；历史 run 可查、summary 可见、checkpoint 路径在 summary 中可定位。日志框全量推送在长训下会膨胀（E1 已知 P3-2，未修——多 epoch 正式训练前应加尾部截断）。重复点击/双会话并发被服务端拒绝（非按钮态依赖）。无删除/覆盖能力→无误删风险；路径逃逸有 canonical allowlist。**缺**：run 无 smoke/formal 标记（仅时间戳目录名，混淆风险 P3）；无 resume；无 best-checkpoint/验证间隔控制；无人工验收/数据整理入口（后者也**不应**塞进训练 UI——应做独立 data tools）。过度工程：无明显项；各防护层均有真实事故对应。

## 11. 测试审计

- 实测：**182 passed, 9 warnings, 17 subtests**（~32s）；关键测试（数值等价+补丁+wiring+守卫，29 项）三连跑零 flake；Gradio smoke PASSED；Hydra validate-only exit 0；patch verify PATCHED exit 0。
- 覆盖强项：Hydra compose（进程内+真实子进程）、launcher token/并发/守卫顺序、端口分配与元数据、provenance、patch 四态与 apply/revert 回滚、梯度累积数值等价（真实 _run_step）、preflight 守卫、summary 原子/幂等、进程组终止（真实孙进程）。mock 使用克制（关键路径都有真实进程/真实文件版本）。
- 缺口（真实功能未被测试证明的区域）：**checkpoint→推理加载正确性（P1-2 恰好漏网于此）**；数据审计/split（无代码故无测试）；resume；多 GPU spawn；UI 交互层（仅 import/startup smoke）；长日志性能。
- 测试依赖本机绝对路径与真实 fixtures（sam301/数据集），干净环境不可全绿——与项目单机定位一致但限制 CI 化；对真实 trainer.py 的哨兵测试在未打补丁树上会失败（有意 fail-closed）。测试不修改真实数据（临时目录/临时树），无顺序依赖迹象。

## 12. 安全性审计

- 全库无 `rm -rf`/危险 glob/shell=True；子命令全部列表参数。原始图片/COCO/sam3.pt/旧 run 无被覆盖路径（run 目录唯一性+非空拒绝+canonical allowlist+symlink resolve）。JSON 关键写入（summary/patch 目标）原子（tmp+fsync+os.replace）；manifest/patch 双 SHA+路径限域 fail-closed。进程终止限自建 PGID。多用户实验室：GPU 抢占无协调（同机他人任务可致 OOM/EADDRINUSE——后者已真实发生并被正确处理）；无锁的并发 UI 实例风险由单活动任务+token 缓解。用户输入不进 shell。Git 操作全程显式文件名。**无 P1/P2 级安全问题。**
- 半写文件残余风险：preflight 写 dataset_info/config_summary 非原子（崩溃可留半文件）——但这些 run 会因 errors/复用拒绝而不可启动，低危 P3。

## 13. 文档审计

- 30 个 docs 中约 20 个是历史阶段报告/验收/评审存档，与 8-10 个"现行规范"混放，无 current/archive 分层，权威入口不清（README→stage_e1_training_ui→SAM301_PATCH_MANAGEMENT 是事实上的现行链，但读者无从得知）。
- README 与实现基本一致（wrapper 命令、patch guard 入口）；`current_system_analysis.md`、`stage_d_ui.md` 等仍含 `-n sam3` 旧示例（已知 O-4，未修，会误导）。
- 环境重建：SAM301_ENV_MIGRATION + PATCH_MANAGEMENT 合计基本可复现（缺 conda lockfile）。数据准备步骤**缺失**（audit/split 不可重跑）。正式训练步骤、故障恢复（patch/端口）完整。checkpoint 推理步骤存在但对微调 checkpoint 无效（P1-2）。人工验收清单存在（HUMAN_ACCEPTANCE_CHECKLIST.md，未跟踪）——但其数据部分只核对数量，**没有"标注是否人工修正"与"唯一图数"检查项**，且推理 smoke 步骤未把"零预测"定义为 FAIL 条件。
- 新成员按文档可完成：环境验证 ✅、preflight/训练 ✅、patch 管理 ✅；数据准备 ❌、resume ❌（不存在）、微调推理 ❌。

## 14. 目录结构审计（不实施）

现状问题：docs 混放（最重）；数据工具缺位导致 manifests 无源；根目录 config/ 与 core/config.py 命名易混；scripts/ 混合正式依赖（launch/manage）与工具（export/accept）；无 dead code/重复代码级问题；大文件均在 gitignored 区。**不必须立即重构。**

建议目标结构（时机：P1 数据决策落地、data tools 入库之时一并做，正式多 epoch 训练开跑前完成；顺序：先 docs 分层→再新增 data_tools/→最后小步移动 scripts；import/测试影响限于 scripts 移动，历史 run 不受影响）：

```
book01/
  core/            (保持；训练与推理核心)
  ui/              (保持)
  data_tools/      (新增：audit/split/manifest/merge/dedupe/export——先补代码再谈迁移)
  scripts/         (保持正式入口: launch/manage/preflight/inference)
  config/ + data_manifests/  (可合并为 manifests/，低收益，可缓)
  tests/           (保持)
  docs/current/    (README 链接的 8-10 份现行规范)
  docs/archive/    (全部阶段报告/验收/评审存档)
  patches/  runs/  data/  (保持)
```

每项：docs 分层——收益高/零风险/必要；data_tools 新增——收益高/零风险/必要（先有码）；scripts 拆分与 manifests 合并——收益低/小险（import 路径、文档引用）/非必要。

## 15. 未来 COCO / Data Tools 功能建议

已存在：CVAT 预标注导出（core/cvat_export）、推理→NPZ/可视化（旧权重路径）。部分存在：重复检测/审计/split（只有*结果*入库，无代码）。应增加（优先级序）：① 数据审计+split+manifest 生成工具入库（P2-1 的修复，最高）；② checkpoint 权重导出/键重映射工具（P1-2 修复的一半）；③ 可视化抽检工具（人工验收数据质量必需）；④ 标注版本/审核状态字段（区分预标注 vs 人工修正——P1-1 治理基础）；⑤ COCO 合并+ID 重映射（多批次人工标注回收时需要）；⑥ 数据质量报告导出。不应放进训练 UI：①②⑤⑥（独立 CLI/data_tools）；③④ 可考虑轻量查看页但非必须。过度工程风险：数据版本管理系统化（DVC 类）当前规模不需要。

## 16. 与已有报告对照（独立结论在先）

- **得到独立证实**：sam301 环境迁移与 editable 绑定；GPU 诊断（环境问题非驱动）；Hydra 根因与修复；grad-accum 数学与数值等价；patch guard/provenance/port 全部机制与其验收 run 数字；E3 run 全部量化数字；44 重复组/0 跨 split/0 aHash 跨 split。
- **过于乐观/关键缺漏**：E3 acceptance"PASS"与 FORMAL_DATASET_AUDIT——未提标注为机器预标注（最重）、未披露唯一图数（test=4）、AP 未标注为 bbox-only 且相对伪标签；CHECKPOINT_INFERENCE_SUMMARY 如实写了零预测与"未验证权重映射"，但没有任何报告把它升级为阻断项——**它就是阻断项**。
- **互相冲突**：无直接矛盾；但多份"PASS/APPROVE"的累积语气与真实状态（两个 P1 未被任何报告标为 P1）不成比例。
- **已过时**：STAGE_E2_PRELAUNCH_AUDIT 的 import 路径警告（已修）；current_system_analysis 的 sam3 环境（已迁移）。
- **未证实即引用**：286 极小 mask 的精确口径（生成码缺失，我算 275/285）。

## 17. Findings 表

| ID | 级 | 标题 | 证据/位置 | 影响 | 阻断验收? | 阻断正式训练? | 阻断重构? |
|---|---|---|---|---|---|---|---|
| F-P1-1 | P1 | 正式数据集 7185 标注全部为 SAM3 机器预标注（8 点多边形+NMS），非人工修正 | 6×COCO `info.description`；顶点分布 100%≤8（本报告 §5）；复现：`python3 -c "import json;print(json.load(open('data/cvat_import_polygon_20260622_231208_book_spine_iou0.5_book_spine.json'))['info'])"` | 训练=自蒸馏；无法改善目标错误类型；AP=自一致性；8 点 GT 违背抓取边界需求 | 否——验收正是要人工确认此事 | **是** | 否 |
| F-P1-2 | P1 | 微调 checkpoint 经现有推理链路加载后零预测（权重键命名空间不兼容，静默少载） | sam301 model_builder.py:539-556（`if "detector" in k` 过滤）；实证 review_artifacts/CHECKPOINT_INFERENCE_SUMMARY.md 4/4 图 raw=0；触发条件：任何 trainer checkpoint 传入 `--checkpoint` | 训练产物进不了下游；strict=False 静默失败无告警 | 是（验收清单的推理步骤将给出误导性"通过"） | **是** | 否 |
| F-P2-1 | P2 | 数据审计/split/manifest 生成代码零入库；split COCO gitignored 且不可再生 | 302b4b1 仅 JSON+md；repo 全域无 data 工具；split_rule 文本歧义（.jpg 尾部）实测足以误导重实现 | 不可复现/不可校验/stale-manifest 无防护 | 否 | **是**（作为正式流程前提） | 否 |
| F-P2-2 | P2 | 有效样本量严重低于名义：train 120/160 唯一（25% 双份加权）、test 4/12 唯一（×3 重复计数）、任何报告未披露 | 本报告 §5 独立聚类；复现：对 184 文件 sha256 聚类 | test 无统计力；train 重复加权扭曲；"12 test images"具误导性 | 是（验收须知情） | **是** | 否 |
| F-P2-3 | P2 | 指标仅 bbox AP，无 mask AP/IoU/边界指标；无 baseline 对照、无 best-checkpoint 机制 | trainer 日志仅 coco_eval_bbox_*；meters 配置无 segm；save_best_meters 未配置 | 无法评价 mask 质量=项目核心；one-epoch AP 无解释力（伪标签+bbox） | 否 | **是** | 否 |
| F-P3-1 | P3 | resume 通路被守卫设计性禁止且无入口；checkpoint 无 RNG 状态 | run_dir 有 checkpoint 即拒绝；无 resume UI/CLI；trainer 无 rng 保存 | 长训中断即整段重跑 | 否 | 多 epoch 前应决策 | 否 |
| F-P3-2 | P3 | 日志全量推送无截断（长训 UI 膨胀）；R-3 锁内 yield 遗留 | ui/process_manager `_log_lines` 无上限；tpp yield-in-lock | 多小时训练页面卡顿/内存 | 否 | 建议先修 | 否 |
| F-P3-3 | P3 | run 无 smoke/formal 阶段标记；14-57-30 preflight-only 残留未见记载 | runs/ 目录仅时间戳；provenance 无 stage 字段 | 人工易混淆 15 个 run 的性质 | 否（清单已点名具体 run） | 否 | 否 |
| F-P3-4 | P3 | 文档 30 份无 current/archive 分层；旧 `-n sam3` 示例残留；conda 环境无 lockfile | docs/ 清单；current_system_analysis.md 3 处 | 交接误导、复现缺口 | 否 | 否 | 此项即重构动机 |
| F-P3-5 | P3 | 极小 mask=286 口径不可核（我算 275/285）；20260623_183533 批次被无记录排除 | §5 | 审计可信度细节 | 否 | 否 | 否 |
| F-P3-6 | P3 | preflight 侧 dataset_info/config_summary 写入非原子；端口 TOCTOU 残窗 | training_runner 写文件处；port docstring | 崩溃留半文件（不可启动，低危）；罕见端口竞争 | 否 | 否 | 否 |
| Obs-1 | Obs | provenance 含 dirty 如实记录、launcher 重算——正面实践 | §8 | — | — | — | — |
| Obs-2 | Obs | val/train 标注密度差 2 倍、val=旧烟测同源组；多 GPU 路径未演练 | §5/§6 | 未证实风险（val 代表性、spawn 路径） | 否 | 提示 | 否 |

## 18. 人工验收前必须完成

1. 把本报告 F-P1-1/F-P1-2/F-P2-2 的事实加入验收材料：验收者须在知情"标注=机器预标注、test=4 张唯一、推理零预测"前提下检查；
2. 验收清单增补三个检查项：标注来源确认（抽看 CVAT 历史/description）、唯一图数披露、推理 smoke 的零预测=FAIL 判据；
3. 可视化抽检：用现有 gt_overlays 物料人工评定预标注 mask 质量（外扩/粘连/8 点边界是否可接受为起点）。

## 19. 正式多 epoch 训练前必须完成

1. **数据决策（P1-1）**：要么完成人工修正标注并重建数据集（推荐，符合项目目标），要么书面批准"伪标签自蒸馏"为第一阶段策略并相应重定义成功指标——二选一，不可默认；
2. **checkpoint 出口（P1-2）**：实现 trainer→推理权重导出/键重映射工具 + 加载零匹配时 fail-loud + 推理 smoke 断言预测数>0；
3. **数据工具入库（P2-1）**：audit/split/manifest 生成代码进 repo，能重建并校验现 manifest；
4. **重复治理（P2-2）**：去重或书面接受加权，重建 test（≥30-50 张唯一、覆盖困难样本分层）；
5. **指标（P2-3）**：启用 mask/segm 指标（至少 mask AP + IoU 分布 + pred/GT 面积比），跑原始 sam3.pt baseline eval 以供对照；确定 best-checkpoint/验证间隔/early-stop 策略与 epoch 数依据；
6. 建议同步：日志截断（F-P3-2）、resume 决策（F-P3-1）、run 阶段标记（F-P3-3）。

## 20. 推荐后续路线图

① 人工验收（带 §18 增补）→ ② 数据决策+人工修正批次启动（CVAT 真实修正 → 新导出 → data_tools 入库同步开发）→ ③ checkpoint 导出工具+推理 smoke 修复 → ④ 指标扩展+baseline eval → ⑤ docs 分层+轻量重构 → ⑥ 以修正数据重建 split → 正式多 epoch 训练。

## 21. 最终判定（分层）

- **A. 训练基础设施**：正确、可复现（补丁/供应链/端口/token/provenance 全验证）；**可以冻结**，仅留 F-P3-1/2 入 backlog。
- **B. 当前数据集**：来源已查明（=机器预标注）；格式正确、几何干净；**质量与身份不适合"正式训练数据"**；split 机制诚实但有效规模误导；当前 split 可作为伪标签实验基线保留，**不可**作为正式 split 沿用。
- **C. one-epoch 结果**：证明且仅证明流程端到端跑通；bbox AP 对项目目标无解释价值（伪标签自一致 + 无 mask 指标）。
- **D. 人工验收**：**可以开始**（带 §18 前提）；重点：标注来源、mask 质量抽检、重复/唯一数知情、推理零预测确认。
- **E. 正式多 epoch 训练**：**不批准**；前提见 §19（1-5 为硬条件）。
- **F. checkpoint 推理与集成**：**未打通**（F-P1-2 阻断缺口，根因已定位到行级）。
- **G. COCO/数据管理**：需增加 §15 ①-④（最小集）；⑤⑥ 随人工标注批次到来时做。
- **H. 目录结构**：不必须立即重构；时机=数据工具入库同批；目标结构见 §14。
- **I. 整体**：**APPROVE_WITH_FOLLOWUPS** —— 可进入人工验收：是；可冻结训练基础设施：是；可开始正式多 epoch 训练：**否**；建议立即重构目录：否。
