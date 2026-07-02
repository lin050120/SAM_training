# CUDA Environment Diagnosis

生成时间: 2026-07-02

## 已确认事实

- 当前 Codex 会话中 `nvidia-smi` 可执行，但无法与 NVIDIA driver 通信:
  `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`
- 当前 Codex 会话中可见空的 `/dev/nvidia-caps` 目录，但未发现 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 等可用 NVIDIA 设备节点。
- `CUDA_VISIBLE_DEVICES` 和 `NVIDIA_VISIBLE_DEVICES` 均为空，未看到由这两个变量主动隐藏 GPU 的证据。
- Conda 环境 `sam3` 存在，Python 路径为 `/home/book/anaconda3/envs/sam3/bin/python`，Python 版本为 `3.12.13`。
- PyTorch 来自 `/home/book/anaconda3/envs/sam3/lib/python3.12/site-packages/torch/__init__.py`。
- PyTorch 版本为 `2.10.0+cu128`，`torch.version.cuda` 为 `12.8`。
- `torch.__config__.show()` 显示 `USE_CUDA=ON`、`CUDA_VERSION=12.8`、`CUDNN_VERSION=9.10.2`，因此当前 PyTorch 不是 CPU-only 构建。
- 当前 Codex 会话中 `torch.cuda.is_available()` 为 `False`，`torch.cuda.device_count()` 为 `0`，并出现 `Can't initialize NVML` warning。
- 当前用户为 `book`，`uid=1001(book) gid=1001(book) groups=1001(book),65534(nogroup)`。
- 使用 `PYTHONPATH=/home/book/sam301` 时，`sam3` import 来源为 `/home/book/sam301/sam3/__init__.py`。

## 推断

- 真实单图 SAM3 推理尚未执行的直接原因是当前 Codex 会话中 CUDA 不可用，不是因为正常 GPU 推理本身预计很慢。
- 当前 PyTorch 是 CUDA 构建，CUDA unavailable 的主要问题不在于安装了 CPU-only PyTorch。
- 由于 `nvidia-smi` 无法通信且没有可用 NVIDIA 设备节点，可能原因包括:
  - NVIDIA driver 未加载或当前环境不可访问 driver；
  - `/dev/nvidia*` 设备没有暴露给当前 Codex 执行环境；
  - Codex 沙箱/执行环境与普通终端的 GPU 可见性不同；
  - 当前用户或容器环境没有设备访问权限。

## 尚未确认内容

- 普通终端中是否能看到 GPU。
- 普通终端中 `torch.cuda.is_available()` 是否为 `True`。
- 这是主机 driver 问题、设备权限问题，还是仅 Codex 执行环境/沙箱可见性问题。

## Codex 内外对比脚本

已创建只读诊断脚本:

```bash
bash /home/book/book01/scripts/check_cuda_environment.sh
```

该脚本只输出环境信息，不修改系统配置。请在普通终端中执行同一脚本，用于和 Codex 内结果对比。

## 推荐下一步

1. 在普通终端执行 `/home/book/book01/scripts/check_cuda_environment.sh`。
2. 如果普通终端中 CUDA 可用，而 Codex 中不可用，则记录为 Codex 执行环境或沙箱 GPU 可见性问题，不应重装 PyTorch。
3. 如果普通终端中同样 CUDA 不可用，再根据 `nvidia-smi`、`/dev/nvidia*` 和 PyTorch 输出判断是否需要 driver、设备权限或环境修复。
4. 未经批准，不安装、升级或重装 PyTorch、CUDA、driver 或 Conda 环境。
