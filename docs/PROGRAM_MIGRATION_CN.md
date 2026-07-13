# 程序迁移说明（中文）

本文说明如何在另一台电脑上部署当前这套 SAM3 训练程序。GitHub 里的
`book01` 仓库只包含业务代码、配置、manifest 和文档；训练数据、图片、权重、
历史 run、SAM3 源码和 conda 环境需要另外准备。

## 1. 推荐目录结构

最稳妥的方式是在新电脑保持和当前机器相同的路径：

```text
/home/book/book01
/home/book/sam301
```

当前项目的关键路径写在：

```text
/home/book/book01/core/config.py
```

其中：

```python
BOOK_ROOT = Path("/home/book/book01")
SAM301_ROOT = Path("/home/book/sam301")
```

如果新电脑路径不同，至少要修改 `core/config.py`。另外
`config/sam301_patch_manifest.json` 里也记录了 `/home/book/sam301`，路径变化时
需要同步处理，否则 patch guard 会失败。

## 2. 从 GitHub 下载项目代码

建议拉当前通用单目标训练分支：

```bash
git clone -b codex-stage-e5-general-target-training \
  git@github.com:lin050120/SAM_training.git \
  /home/book/book01
```

如果已经 clone 过：

```bash
cd /home/book/book01
git checkout codex-stage-e5-general-target-training
git pull
```

## 3. 准备 conda 环境

当前所有命令默认使用 conda 环境：

```text
sam301
```

建议在旧机器导出环境：

```bash
conda env export -n sam301 > sam301_environment.yml
```

把 `sam301_environment.yml` 复制到新机器后创建环境：

```bash
conda env create -f sam301_environment.yml
```

之后确认 Python 可用：

```bash
conda run -n sam301 python --version
```

## 4. 准备 SAM3 源码和权重

GitHub 的 `book01` 仓库不包含 SAM3 源码。你需要把 SAM3 放在：

```text
/home/book/sam301
```

该目录至少需要包含：

```text
/home/book/sam301/sam3/
/home/book/sam301/sam3.pt
/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml
```

可以从旧机器复制 `/home/book/sam301`，也可以重新安装 SAM3 后补齐本项目需要的
`sam3.pt` 和训练 YAML。

## 5. 安装 SAM3 包

让 `sam301` 环境从 `/home/book/sam301` import SAM3：

```bash
cd /home/book/sam301
conda run -n sam301 pip install -e ".[train,dev]"
```

确认 import 路径正确：

```bash
env -u PYTHONPATH conda run -n sam301 python -c \
  "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
```

必须输出类似：

```text
/home/book/sam301/sam3/__init__.py
```

不能指向旧目录，例如 `/home/book/sam3`。

## 6. 应用并验证 trainer patch

本项目要求 `/home/book/sam301/sam3/train/trainer.py` 带有梯度累积 loss scaling
补丁。训练预检、启动器和训练子进程都会 fail closed 检查这个补丁。

在 `/home/book/book01` 下执行：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/manage_sam301_patch.py status
conda run -n sam301 python scripts/manage_sam301_patch.py apply
conda run -n sam301 python scripts/manage_sam301_patch.py verify
```

`verify` 必须退出码为 0，并显示 PATCHED。  
如果 `status` 是 UNKNOWN，不要强行覆盖，需要先人工检查 SAM3 文件来源。

## 7. 复制数据、模型和历史 run

`.gitignore` 排除了数据和大文件，例如：

```text
data/
runs/
*.pt
*.jpg
*.png
*.npz
```

所以 GitHub 下载后不会自动包含：

- 训练数据集
- 原始图片
- 推理/训练历史 run
- checkpoint / inference model
- `.npz` 中间结果

如果你要继续使用现有数据和模型，需要手动复制：

```text
/home/book/book01/data/
/home/book/book01/runs/          # 需要历史 run 时复制
/home/book/sam301/sam3.pt
```

`data_manifests/dataset_identity_registry.json` 在 GitHub 里，但它登记的
`annotations_path` 指向的数据文件必须在新机器真实存在。

## 8. 数据集身份登记

正式训练或多 epoch 训练要求数据集已经登记为：

```json
"allowed_for_formal_training": true
```

如果是新目标，例如 cable，可以在 UI 的 `数据集登记` 页面登记。目录结构应类似：

```text
data/cable_sam3_dataset/
  train/images/
  train/annotations.json
  val/images/
  val/annotations.json
  test/images/              # 可选
  test/annotations.json     # 可选
```

登记后再回到 `训练预检` 页面，选择同一套 train/val COCO，把
`training mode` 设为 `formal`。

## 9. 检查 GPU / CUDA

新电脑必须有可用 NVIDIA GPU、驱动和与 PyTorch 兼容的 CUDA。检查命令：

```bash
conda run -n sam301 python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

如果输出 `False`，UI 会拒绝启动 CUDA 推理和真实训练。

## 10. 启动 UI

```bash
cd /home/book/book01
conda run -n sam301 python app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

UI 只监听本机 `127.0.0.1`，不会默认公开到外网。

## 11. 最小验收检查

迁移后建议执行：

```bash
cd /home/book/book01
conda run -n sam301 python scripts/manage_sam301_patch.py verify
env -u PYTHONPATH conda run -n sam301 python -c \
  "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
conda run -n sam301 pytest -q tests/test_sam301_patch.py tests/test_ui.py::UiImportsTest
```

如果要完整回归：

```bash
conda run -n sam301 pytest -q tests
```

## 12. 迁移要点总结

只下载 GitHub 代码不够。新电脑还需要：

- `sam301` conda 环境
- `/home/book/sam301` SAM3 源码
- `/home/book/sam301/sam3.pt`
- SAM3 editable install
- trainer patch 验证通过
- 训练/推理数据和模型文件
- GPU/CUDA 正常
- 如路径不同，修改 `core/config.py` 和 patch manifest 相关路径

