# 程序迁移说明（中文）

本文说明如何在另一台电脑上部署当前这套 SAM3 训练程序。GitHub 里的
`book01` 仓库只包含业务代码、配置、manifest 和文档；训练数据、图片、权重、
历史 run、SAM3 源码和 conda 环境需要另外准备。

## 1. 自动路径配置

新电脑上的目录不需要和旧电脑相同。准备好 `book01`、`sam301` 和名为
`sam301` 的 conda 环境后，在新 `book01` 目录运行：

```bash
conda run -n sam301 python scripts/migrate_environment.py
```

脚本会弹出两个文件夹选择窗口，依次选择新的 `book01` 和 `sam301`。确认后脚本会：

- 验证项目代码、SAM3 源码、训练 YAML、BPE 和 `sam3.pt`；
- 生成本机专用的 `config/local_paths.json`；
- 检查 `import sam3` 是否来自新目录，错误时询问是否修复 editable install；
- 检查 trainer patch，UNPATCHED 时询问是否应用，UNKNOWN 时停止自动修改；
- 检查 CUDA，并把结果写入 `config/migration_report.json`。

这两个文件都被 Git 忽略，不会把一台电脑的绝对路径上传到 GitHub。已有配置覆盖前会
生成 `config/local_paths.<时间>.bak.json`。

注意：在新位置的 `book01` 中，如果还没有运行迁移向导（即没有
`config/local_paths.json`），程序会直接报错并提示先运行
`scripts/migrate_environment.py`，而不是继续使用旧电脑的路径。

也可以直接使用命令行：

```bash
conda run -n sam301 python scripts/migrate_environment.py \
  --book-root "/新的路径/book01" \
  --sam301-root "/新的路径/sam301"
```

建议先预演，预演不会写文件、重装包或应用 patch：

```bash
conda run -n sam301 python scripts/migrate_environment.py \
  --book-root "/新的路径/book01" \
  --sam301-root "/新的路径/sam301" \
  --dry-run
```

无人值守模式必须显式提供两个目录。需要允许修复时再增加
`--repair-install` 和 `--apply-patch`：

```bash
conda run -n sam301 python scripts/migrate_environment.py \
  --book-root "/新的路径/book01" \
  --sam301-root "/新的路径/sam301" \
  --non-interactive --repair-install --apply-patch
```

程序没有本机配置时仍兼容以下旧默认目录：

```text
/home/book/book01
/home/book/sam301
```

不要手工修改 `core/config.py`。生产 patch manifest 使用相对 SAM301 根目录的 trainer
路径，也不需要因电脑路径变化而修改。

## 2. 从 GitHub 下载项目代码

建议拉当前分支（含通用单目标训练，以及 2026-07-24 的两个 SAM3 数值修复：
bf16 GradScaler underflow 与 Triton focal-loss `gamma=0` 反向 NaN。更早的
stage 分支缺少这两个补丁清单，训练到第 6~7 epoch 会全 NaN 崩溃）：

```bash
git clone -b codex-stage-e8-nan-fix-and-test-loss-20260725 \
  git@github.com:lin050120/SAM_training.git \
  /home/book/book01
```

如果已经 clone 过：

```bash
cd /home/book/book01
git checkout codex-stage-e8-nan-fix-and-test-loss-20260725
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

自动迁移脚本会执行这项检查，并在路径错误时先征得确认再运行无依赖的 editable
install。以下命令用于需要手工处理时：

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

自动迁移脚本会检查该状态，并只在状态为 UNPATCHED 且得到确认时应用。UNKNOWN
表示文件既不匹配原始哈希也不匹配补丁哈希，脚本一定不会覆盖。

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
- 运行 `scripts/migrate_environment.py` 生成本机路径配置
