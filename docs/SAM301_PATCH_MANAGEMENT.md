# SAM301 补丁管理（trainer 梯度累积 loss 缩放）

生成时间: 2026-07-03

## 1. 为什么需要补丁

`/home/book/sam301/sam3/train/trainer.py` 的 `_run_step` 在 `gradient_accumulation_steps > 1` 时对每个 micro-batch 直接 `backward(loss)`，梯度是**求和**而非平均——不变学习率下不等价于真实大 batch。补丁（唯一改动点）：

```python
backward_loss = loss / accum_steps if accum_steps > 1 else loss
self.scaler.scale(backward_loss).backward()
```

只缩放 backward 输入；日志/isfinite 用原始 loss；optimizer/scheduler/clipping 时序不变；accum=1 逐位不变。数值等价性由 `tests/test_grad_accum_numerics.py` 用真实 `_run_step`（float64、1e-12 容差）持续验证。

**问题**：`/home/book/sam301` 不是 git 仓库，重装/重拷源码树会静默丢失补丁。因此用 git 管理的 manifest + 全量 SHA256 + 三层启动守卫做 fail-closed 固化。

## 2. Manifest 字段（`config/sam301_patch_manifest.json`）

| 字段 | 含义 |
|---|---|
| `patch_id` / `version` | 补丁标识与版本 |
| `purpose` | 补丁用途完整说明 |
| `target_file` | 目标文件规范绝对路径；loader 使用 `Path.resolve()` 后要求它位于 `expected_sam3_root` 内，并拒绝任何指向 `/home/book/sam3` 或通过 symlink 逃逸的 target |
| `patch_file` | 补丁 diff（相对 book01 根）；解析后必须位于 `/home/book/book01/patches` 内，并由 `patch_file_sha256` 校验其完整性 |
| `original_sha256` | 未修补 trainer.py 的**完整** SHA256（`9c9c4159…248d`，三方独立验证：修补前备份、`patch -R` 逆向重建、Codex 审查记录） |
| `patched_sha256` | 修补后**完整** SHA256（`bcf5d8d6…9ec2`，Codex 独立复现） |
| `expected_sam3_root` | 预期 SAM3 import 根 `/home/book/sam301` |

## 3. 四个命令（均在 book01 根目录执行）

```bash
# 状态：PATCHED / UNPATCHED / UNKNOWN / MISSING
conda run -n sam301 python scripts/manage_sam301_patch.py status
conda run -n sam301 python scripts/manage_sam301_patch.py --json status   # 机器可读

# 验证（正式训练闸门）：只有 PATCHED 退出码为 0，其余一律非零
conda run -n sam301 python scripts/manage_sam301_patch.py verify

# 应用：仅当当前 hash == original 时允许；先 dry-run，在同目录临时文件上 patch，hash 必须 == patched 后原子替换
conda run -n sam301 python scripts/manage_sam301_patch.py apply

# 回滚：仅当当前 hash == patched 时允许；在同目录临时文件上反向 patch，hash 必须 == original 后原子替换
conda run -n sam301 python scripts/manage_sam301_patch.py revert
```

**UNKNOWN 状态（hash 与两个已知值都不符）绝对禁止强制覆盖**——apply/revert 都会拒绝，此工具没有 --force。此时必须人工检查文件（可能有第三方修改需要保留），确认后手动恢复到已知状态再操作。

apply/revert 持有目标文件同目录的锁文件，复制原文件到同目录临时文件，对临时文件执行 patch 或 reverse patch，校验最终 SHA256，保留原文件权限，fsync 临时文件，使用 `os.replace` 原子替换，并 fsync 父目录。任意可检测失败都不会替换真实 trainer。

## 4. 环境重建后的正确顺序

1. 恢复/重装 `/home/book/sam301` 源码树与 `sam301` conda 环境（editable install 指向 `/home/book/sam301`）；
2. `manage_sam301_patch.py status` —— 预期 `UNPATCHED`；
3. `manage_sam301_patch.py apply`；
4. `manage_sam301_patch.py verify` —— 必须退出码 0；
5. `conda run -n sam301 python -m pytest tests/test_grad_accum_numerics.py tests/test_sam301_patch.py -v`；
6. 正常走训练预检。

## 5. 正式训练前必须执行

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py verify   # 必须 exit 0 / PATCHED
env -u PYTHONPATH conda run -n sam301 python -c "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
# 必须输出 /home/book/sam301/sam3/__init__.py（不得指回 /home/book/sam3）
sha256sum /home/book/sam301/sam3/train/trainer.py
# 必须等于 manifest 的 patched_sha256: bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2
```

即使忘记手动执行，训练链路也会 fail closed —— 三层自动守卫：

| 层 | 位置 | 行为 |
|---|---|---|
| 1. preflight | `core/training_runner.py::inspect_training_config`（prepare_runtime 时，写 runtime YAML 之前） | 非 PATCHED → 预检 error，不写 YAML、不产生可启动 run、UI 不发 token |
| 2. launcher | `ui/training_preflight_page.py::_consume_preflight_for_launch`（消费 token 之前） | 预检后文件被替换 → BLOCKED，**token 不被消费**，无子进程 |
| 3. 训练子进程 | `scripts/launch_sam3_training.py::run_training`（调用官方 main 之前，stdlib 自校验） | 最后防线，hash 不符直接退出非零 |

三层与既有的 sam3 import guard（预检 + 启动时验证 import 解析到 `/home/book/sam301/sam3/__init__.py`、editable binding 未指回 `/home/book/sam3`）并行生效。

每次可启动 preflight 和正式 launcher 都会记录机器可读 provenance：

- `provenance.json`
- `dataset_info.json` 的 `training_provenance`
- `training_config_summary.json` 的 `training_provenance`
- `training_summary.json` 的 `training_provenance`

字段包括 book01 Git commit/dirty 状态、manifest SHA256、patch SHA256、trainer.py SHA256、SAM301 root、sam3 import path、Python executable 和 runtime YAML SHA256。launcher 启动前会重新计算这些值，不只复制 preflight 结果。
