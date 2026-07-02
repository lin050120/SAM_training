# E2 GPU/NVIDIA 驱动阻塞问题诊断报告

- **诊断日期**: 2026-07-02 21:46 JST
- **诊断模式**: read-only（未启动训练、未修改系统/驱动/内核/Conda/项目代码、未 reboot、未 restart 任何服务、未加载/卸载任何内核模块）
- **最终结论**: **ENVIRONMENT_ISSUE** —— 主机 GPU/驱动栈完全健康；"CUDA 不可用"仅存在于 Codex agent 沙箱内。主机不需要任何修复。

---

## 1. 当前 branch 和 commit

- Branch: `codex-stage-e2`
- HEAD: `2e4723176253b3fab8b38ca3aecb5d1241fb7ded`（`2e47231` Document E2 one-epoch training acceptance）
- 起始 commit: `9d8e01f`；`git diff 9d8e01f..2e47231 --stat`: 仅新增 `docs/E2_REAL_ONE_EPOCH_ACCEPTANCE.md`（+239 行），零代码改动。
- `git status --short`: 仅已知未跟踪文件 `docs/CLAUDE_REVIEW_OF_SAM301_ENV_MIGRATION.md`（保持未跟踪，未动）。

## 2. Codex E2 BLOCKED 结果复核

与 Codex 报告逐项一致，未发现不一致：

1. Codex 只提交了报告文件（单文件 commit）✅
2. 未修改代码（diffstat 仅 docs）✅
3. 未修改 Conda 环境（两环境 torch/依赖仍与迁移复审时一致，见 §13-14）✅
4. 确实未启动 trainer ✅
5. `checkpoints/` 为空（0 个文件）✅
6. 无 `training_summary.json` ✅
7. 无残留训练进程（`pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml"` 无匹配）✅
8. working tree 除已知 review 文档外无其他变化 ✅

Codex 按规则在 CUDA=False 时硬停止、不重试、不改参数——**行为正确**。其报告如实声明了"from this execution environment"，且项目文档一贯警告 Codex 沙箱的 GPU 可见性不能代表主机。

## 3. 系统与 kernel 信息

- 主机名: `tai`；用户: `book`（组含 sudo、adm）
- OS: Ubuntu 22.04.3 LTS (jammy)
- Kernel: `6.8.0-124-generic`（#124~22.04.1-Ubuntu）
- 开机时间: 2026-06-23 17:48（uptime 9 天），最近一次 reboot 后未更换过内核
- 诊断时刻: 2026-07-02 21:46 JST

## 4. PCI GPU 状态

**GPU 在 PCI 层完全可见：**

```
02:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:2b85] (rev a1)
        Kernel driver in use: nvidia
        Kernel modules: nvidiafb, nouveau, nvidia_drm, nvidia
```

- PCI ID: `10de:2b85`（NVIDIA GeForce RTX 5090）
- **Kernel driver in use: nvidia** —— 设备已被 nvidia 驱动绑定

## 5. NVIDIA 设备节点状态

全部存在（自 6 月 23 日开机起），权限为全局可读写：

- `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-modeset`、`/dev/nvidia-uvm`、`/dev/nvidia-uvm-tools` 均存在（crw-rw-rw-）
- `/proc/driver/nvidia/version` 存在：`NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 595.71.05`
- `/proc/driver/nvidia/gpus/.../information`：**Model: NVIDIA GeForce RTX 5090**，GPU UUID、VBIOS 98.02.2e.80.10、GPU Firmware 595.71.05、`GPU Excluded: No` —— 驱动已实际完成 GPU 初始化

## 6. 内核模块状态

```
nvidia_uvm / nvidia_drm / nvidia_modeset / nvidia  全部已加载
nvidia refcount=376（桌面/图形正在活跃使用）
/sys/module/nvidia/version = 595.71.05
modinfo: /lib/modules/6.8.0-124-generic/updates/dkms/nvidia.ko, version 595.71.05
```

- nouveau: **未加载**，无冲突
- 模块与当前 kernel 匹配（就在当前 kernel 的 dkms 目录下）

## 7. DKMS 状态

```
nvidia/595.71.05, 6.8.0-124-generic, x86_64: installed
```

为当前 kernel 构建完整（installed，非 added/built 半成品）。另有 `backport-iwlwifi` 的 "Diff between built and installed" 警告——那是 Wi-Fi 模块，与 GPU 无关。

## 8. 安装的 NVIDIA 包

`nvidia-driver-595-open` 元包及全套组件统一为 **595.71.05-0ubuntu0.22.04.1**（libnvidia-compute-595、nvidia-utils-595、nvidia-kernel-common-595 等）。`linux-image/headers-6.8.0-124-generic` 均已安装且匹配。**不存在多套冲突 driver、不存在"更新后待重启"的版本漂移。**

## 9. Secure Boot 状态

`mokutil --sb-state`: **SecureBoot disabled** —— 不存在签名阻止加载的可能性。

## 10. NVIDIA 服务状态

`nvidia-persistenced.service`: **active (running)**，自 6 月 23 日开机运行至今。display-manager 正常（多用户图形会话在跑）。

## 11. Kernel log 关键证据

`journalctl -k -b` 中与 NVIDIA 相关的仅有两条（开机时）：

```
nvidia: module verification failed: signature and/or required key missing - tainting kernel
NVRM: loading NVIDIA UNIX Open Kernel Module for x86_64  595.71.05 ... 
```

第一条是**良性提示**：out-of-tree 未签名模块在 Secure Boot 关闭时仅 taint 内核、不阻止加载——紧接着的第二条就是模块成功加载。整个 boot 周期内**无** Xid、无 "fallen off the bus"、无 NVRM 错误、无 API mismatch、无初始化失败。

## 12. nvidia-smi / NVML 状态

**在本诊断会话（普通 shell 上下文）中 nvidia-smi 完全正常，退出码 0：**

```
NVIDIA-SMI 595.71.05    Driver Version: 595.71.05    CUDA Version: 13.2
GPU 0: NVIDIA GeForce RTX 5090   870MiB / 32607MiB   0% util
```

- `/usr/bin/nvidia-smi` 存在；NVML 用户态库 `libnvidia-ml.so.595.71.05`、`libcuda.so.595.71.05` 均存在且与内核模块 **版本完全一致（595.71.05 = 595.71.05）**，无 mismatch
- GPU 上当前只有图形进程（两个 Xorg、gnome-shell、chrome 相关，共约 870MiB），无 compute 进程

## 13. sam301 PyTorch CUDA 状态（中立目录、unset PYTHONPATH 实测）

```
sys.executable: /home/book/anaconda3/envs/sam301/bin/python
torch: 2.10.0+cu128 / torch.version.cuda: 12.8
cuda_available: True
device_count: 1
device_name: NVIDIA GeForce RTX 5090
sam3: /home/book/sam301/sam3/__init__.py
```

**E2 的全部 Stage A 前置条件在本会话中实测全部满足。**

## 14. 旧 sam3 环境 CUDA 状态

```
torch: 2.10.0+cu128 / cuda: 12.8 / cuda_available: True / device_count: 1
```

两个环境都能看到 GPU —— 排除"仅 sam301 环境问题"（类别 J）。

## 15. 虚拟化与远程会话状态

- `systemd-detect-virt`: **none**（裸机）；`/proc/1/cgroup` 正常（非容器）
- 会话：`nedo-pana`（tty2/:1，6 月 23 日起）、`tai`（tty3/:2，6 月 24 日起）、`book`（chrome-remote-desktop）——**机器上有其他用户的活跃图形会话**
- 本用户经 Chrome Remote Desktop（DISPLAY=:20, x11）接入；远程桌面**没有**影响 GPU 访问（本会话 nvidia-smi/torch 均正常，直接反证）

## 16. 根因分类

**类别 I 的变体：Codex agent 沙箱没有 GPU 设备访问权（非虚拟机/容器整机问题，而是 agent 会话级别的设备隔离）。**

逐类排除：

| 类别 | 判定 | 证据 |
|---|---|---|
| A. PCI 不可见 | 排除 | lspci 可见 10de:2b85，driver in use: nvidia |
| B. 内核模块未加载 | 排除 | lsmod 四个 nvidia 模块全在，refcount 376 |
| C. DKMS 无匹配模块 | 排除 | nvidia/595.71.05 对 6.8.0-124 为 installed |
| D. Secure Boot 阻止 | 排除 | SecureBoot disabled；模块已加载 |
| E. nouveau 冲突 | 排除 | nouveau 未加载 |
| F. 内核/NVML 版本不匹配 | 排除 | 595.71.05 == 595.71.05，nvidia-smi 正常 |
| G. 更新后待重启 | 排除 | 包、DKMS、已加载模块三者版本一致；kernel 未更换 |
| H. GPU 初始化失败/Xid | 排除 | /proc 信息完整，日志无 Xid/NVRM 错误 |
| I. 无 GPU passthrough 的受限环境 | **命中（Codex 沙箱）** | 同一主机、同一命令：Codex 沙箱=失败，普通 shell 上下文=成功 |
| J. 仅 sam301 环境问题 | 排除 | 两个 Conda 环境 CUDA 均为 True |

## 17. 证据与把握程度

- **事实**：Codex 于 21:33 在其沙箱得到 `nvidia-smi` 无法通信 + CUDA=False；本诊断于 21:46（13 分钟后，期间无 reboot、无驱动/包变更——包安装时间 5 月 1 日、模块加载时间 6 月 23 日）在普通 shell 上下文对**完全相同的检查**全部通过。
- **推断**：差异只能来自执行环境本身。agent 沙箱普遍通过设备隔离屏蔽 `/dev/nvidia*`，与 Codex 现象（driver 通信失败而非版本 mismatch 报错）吻合；项目文档从 D1 阶段起就多次预告"Codex sandbox GPU visibility 不代表主机"。
- **把握程度**：**确凿（很高）**。这是同机、近同时、同命令的直接对照实验，不依赖任何间接推理。
- **未完全排除的解释**：无法直接检视 Codex 沙箱的隔离配置细节（bwrap/landlock/设备 cgroup 等），故"沙箱屏蔽设备"的具体机制是合理推断；但"主机健康、故障仅在该会话内"这一结论不受此影响。

## 18. 建议修复方案（按风险排序）

### 方案 1（最小风险、最可能有效，推荐）：不修复任何东西，在普通终端执行 E2

- **为什么**：主机无故障，无可修复对象。项目文档本来就要求真实训练"在普通终端（不是受限 agent 沙箱）执行"。
- **步骤**：用户在普通终端（本地 tty 或 Chrome Remote Desktop 内的终端均可——本诊断已证明该上下文 GPU 可用）执行 §21 的验收命令，确认后按既定流程启动 E2（等用户明确批准后执行 command.txt）。
- sudo：不需要。reboot：不需要。中断其他用户：否。修改 driver/kernel/CUDA/Conda：否。
- **风险**：无。**回滚**：不适用。
- **验收**：§21 三条命令全部符合预期。

### 方案 2（如果希望 agent 会话内可见 GPU）：调整 agent 沙箱的设备访问配置

- **为什么**：若希望 Codex/agent 直接监控或驱动训练，需要其沙箱放行 `/dev/nvidia*`。
- **步骤**：在 agent 工具自身的沙箱/权限设置中允许 GPU 设备（具体取决于 Codex 的沙箱实现；这是 agent 配置，不是系统修改）。或折中：用户在终端手启训练，agent 只读监控日志文件。
- sudo：不需要（系统层无改动）。reboot：不需要。影响其他用户：否。
- **风险**：扩大 agent 权限面（策略性考虑，非技术风险）。**回滚**：恢复原沙箱配置。
- **验收**：agent 会话内 `nvidia-smi` 退出码 0。

### 方案 3（仅在方案 1 的普通终端验收也失败时才考虑——当前证据表明不会发生）

- 若普通终端也复现失败，则重新采集 §11 内核日志再诊断；当前驱动栈三层版本一致，**不建议**任何驱动重装/DKMS 重建/重启类操作。若最终确需 reboot 或驱动操作：**需要用户确认机器当前无人使用后才能执行**（现有 nedo-pana、tai 两位用户的活跃会话会被中断）。
- sudo：需要。reboot：可能。影响其他用户：**是**。

## 19. 是否需要重启

**不需要。** 驱动包(595.71.05)=DKMS(595.71.05)=已加载模块(595.71.05)=NVML(595.71.05) 四层一致，不存在需要重启才能对齐的状态。

## 20. 对其他用户的影响

本轮诊断零影响。方案 1/2 亦零影响。机器上有 `nedo-pana`、`tai` 两个其他用户的长期图形会话（GPU 上有其 Xorg/gnome-shell 进程）——任何未来涉及 reboot/display-manager/驱动的操作都必须先经用户确认机器无人使用（当前无此必要）。

## 21. GPU 恢复后的最小验收命令

（GPU 从未损坏；以下是启动 E2 前在普通终端的标准确认，均只读）

```bash
nvidia-smi
env -u PYTHONPATH conda run -n sam301 python -c "from pathlib import Path; import sys, torch, sam3; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('sam3:', Path(sam3.__file__).resolve())"
pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml" || echo "no residual trainer"
```

预期：driver 595.71.05 / RTX 5090；cuda: True；sam3 → `/home/book/sam301/sam3/__init__.py`；无残留进程。本诊断会话中这三项已全部实测通过。

## 22. E2 run 复用还是新建

当前 run `2026-07-02_19-46-53` 状态：只经过 preflight（4 个 preflight 文件 + 空 logs/checkpoints），trainer 从未创建，无 summary，launch token 从未被消费（token 只存在于 UI 会话内；本次 BLOCKED 走的是终端 Stage A 检查，未触碰 UI 启动路径），runtime YAML/dataset/基础 checkpoint 自预检以来均未变化（sam3.pt SHA256 有 Codex 前后比对记录）。

- **复用（工程上安全，轻微推荐）**：优点——preflight 产物完全有效且零污染，避免多一个空 run 目录；BLOCKED 记录已由 `E2_REAL_ONE_EPOCH_ACCEPTANCE.md`（commit 2e47231）永久保存，run 目录本身没有承载 BLOCKED 状态、无需保留为"失败现场"。缺点——同一 run 目录对应两次启动尝试，审计粒度略粗。
- **新建**：优点——一次尝试一个 run 的审计最干净，旧 run 目录原样封存。缺点——多一个未使用目录，且需重跑预检（成本极低）。

两种均可接受；**不在本轮执行任何一种**，由用户决定。若复用，启动前应重新确认：`checkpoints/` 仍为空、无 `training_summary.json`、`sam3.pt` size/mtime 未变。

## 23. 本次没有启动训练

确认：未执行 command.txt，未创建 trainer 进程，未消费 launch token，未写 checkpoints，未生成 training_summary.json，run 目录未被修改（诊断中 torch 检查仅短暂初始化 CUDA 上下文查询设备名，未加载任何模型）。

## 24. 本次没有修改系统、驱动、Conda 或项目代码

确认：全部命令为只读查询；未 install/uninstall 任何包，未碰内核模块，未 restart 服务，未改 /etc，未改两个 Conda 环境，未改 book01 与 sam301 源码。本报告是唯一新增文件，并按指示单独提交。
