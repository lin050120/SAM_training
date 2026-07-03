# E2 Loss Scaling and Preflight Fix Acceptance

Generated: 2026-07-03 10:20 JST

## 结果

**PASS** —— P2（梯度累积 loss 缩放）与两个 P3（R-4 缺键 TypeError、R-5 零步假 completed）全部修复；数值等价测试通过（float64、1e-12 容差）；全量 160 passed；修复后 accum=4 真实训练一次通过（2 outer iterations × 4 micro-batches、8 图全覆盖、exit 0、completed、checkpoint 生成、sam3.pt 不变、无残留）。

## 1. P2 根因与修复公式

### 根因

`sam3/train/trainer.py::_run_step` 在 `gradient_accumulation_steps > 1` 时对每个 micro-batch 直接 `self.scaler.scale(loss).backward()`，无任何 1/accum 缩放——N 个 micro-batch 的梯度是**求和**而非平均，等效于把梯度放大约 N 倍，在不变学习率下不等价于真实 batch=N。

### 修复公式（sam301 最小补丁，唯一改动点 trainer.py:961）

```python
backward_loss = loss / accum_steps if accum_steps > 1 else loss
self.scaler.scale(backward_loss).backward()
```

数学语义：`Σ_i ∇(L_i / N) = ∇((1/N) Σ_i L_i)` —— N 个 micro-batch 累积梯度 = 这 N 个 micro-batch **loss 均值**的梯度，即任务要求的语义。缩放因子用 `accum_steps = len(batch)`（**实际** micro-batch 数量），其上方两行既有断言保证它恒等于配置的 `gradient_accumulation_steps`——"实际数量缩放"与"固定配置"由 trainer 自身断言强制一致。

### backward loss 与 logging loss 的区别

- **backward**：`loss / accum_steps`（仅 accum>1 时）；
- **logging/指标**：`loss_mts[loss_key].update(loss.item(), batch_size)` 与 isfinite 检查均使用**原始未缩放** `loss`——补丁未触碰这两处（实证见第 6 节：修复前后 epoch 日志 loss 完全一致，而 checkpoint 权重不同）。

### optimizer / scheduler / clipping 调用频率（未改变）

补丁只改 backward 输入，`train_epoch` 中的时序原样：每个**外层** dataloader 迭代一次——`optim.step_schedulers`（scheduler）→ `scaler.unscale_` + gradient clipping → `scaler.step(optimizer)` → `scaler.update()`；`zero_grad` 每外层迭代一次（`_run_step` 开头）。optimizer.step 次数不变（本验收 epoch 为 2 次）。accum=1 路径 `backward_loss is loss`，行为逐位不变（有精确相等测试）。

### sam301 提交方式说明

`/home/book/sam301` **不是 git 仓库**，无法在其内提交。处理：补丁前备份原文件生成精确 diff，以 `patches/sam301_trainer_grad_accum_loss_scaling.patch` 形式在 book01 **单独提交**（commit `81f79d5`）存档；修改前 trainer.py SHA256 `9c9c4159d2d5…` 已记录。补丁为最小改动（1 行替换 + 注释），无无关重构；`/home/book/sam3` 未触碰。

## 2. P3 两项修复（book01，commit `74ec46f`）

### R-4：scratch 键缺失的干净报错

- `inspect_training_config`：在 resolved 值计算后显式校验 `scratch.train_batch_size` 与 `scratch.gradient_accumulation_steps`（存在、整数、≥1；覆盖值已有独立正数校验），缺失/非法 → 追加清晰 error，不再走到 `int(None)`；
- `write_runtime_yaml`：防御性检查，两键缺失时抛带明确信息的 `ValueError`（保护直接调用方），替代原 `TypeError`。

### R-5：train size < effective batch 拒绝启动

- `inspect_training_config` 在 COCO 摘要后计算 `effective = train_batch_size × gradient_accumulation_steps × num_gpus`：
  - `train 图片数 < effective` → **error 拒绝**（即预计 outer steps < 1；drop_last=True 下将产生 0 optimizer step 的假 completed），且 errors 非空使 runtime YAML 不写、不可启动；
  - 不整除 → **warning**（注明每 epoch 丢弃 N 张图），允许继续。

## 3. 测试

### 数值等价测试（`tests/test_grad_accum_numerics.py`，5 个）

驱动**真实打补丁后的** `sam3.train.trainer.Trainer._run_step`（`Trainer.__new__` + 最小属性注入，非重实现；AMP autocast/GradScaler enabled=False，CPU-only，float64）：

1. **核心等价**：同一初始权重（deepcopy）+ 同一批确定性数据（固定种子 4 样本），batch=4/accum=1 vs micro=1/accum=4，各做一次 `_run_step` + SGD step 后**逐参数比较**——`allclose(rtol=1e-12, atol=1e-12)` 通过（float64 下属严格容差）；
2. **accum=1 回归**：梯度与裸 `loss.backward()` **逐位相等**（`torch.equal`），且 logged loss = 原始 loss（15 位小数）；
3. **accum=2/4 缩放**：N 个相同 micro-batch 的累积梯度 = 单个 micro loss 的梯度（均值语义，1e-12），且 N 条 logged loss 全部为未缩放原值；
4. **契约未弱化**：accum=4 传 dict 或长度 3 的 list 仍触发原有断言；
5. **补丁在位哨兵**：源码级断言补丁行存在于 `/home/book/sam301/sam3/train/trainer.py`（sam301 无版本控制，此测试可捕获文件被还原）。

### 守卫测试（`tests/test_e1_training.py::PreflightGuardsTest`，4 个）

缺两个 scratch 键 → 预检干净报错（两条消息齐全，无异常）；`write_runtime_yaml` 直接调用 → 清晰 ValueError；accum=16（effective 16 > 8 图）→ 预检拒绝且 runtime YAML 未写；accum=3（8 % 3 ≠ 0）→ warning + 正常放行。

### 既有测试适配（1 处，非弱化）

`test_overrides_are_written_to_runtime_yaml` 原用 batch=2×accum=8（effective 16 > 8 图），被新 R-5 守卫**正确拒绝**——参数改为 2×4（effective 8），拒绝场景由 PreflightGuardsTest 专门覆盖。

### 运行结果

- 新增测试（numerics + guards + 既有 wiring）重复 **3 次**：11 passed × 3，无 flake；
- 全量：**160 passed, 9 warnings**（151 → 160，+9；新增 warnings 为 import sam3.train.trainer 带入的 pkg_resources/timm/torch.jit Deprecation 类，与本修复无关）；
- Gradio smoke test：通过（含在全量中）；
- Hydra wrapper `--validate-only`：既有真实子进程测试通过，且新 run 预检内真实执行通过（`hydra_validation_ok: true`）。

## 4. 修改与 commit

| Commit | 内容 |
|---|---|
| `74ec46f`（book01） | `core/training_runner.py`（R-4 预检校验 + R-5 effective batch 守卫 + write_runtime_yaml 防御性 ValueError）、`tests/test_e1_training.py`（PreflightGuardsTest 4 个 + 1 处既有测试参数适配） |
| `81f79d5`（book01，单独提交 sam301 相关） | `patches/sam301_trainer_grad_accum_loss_scaling.patch`（sam301 补丁精确 diff 存档）、`tests/test_grad_accum_numerics.py` |
| sam301 实际修改 | `/home/book/sam301/sam3/train/trainer.py` 一处（如上）；sam301 非 git 仓库，无法产生 commit，以补丁文件替代审计记录 |

## 5. 修复后 accum=4 真实训练

- **新 run**：`/home/book/book01/runs/training/2026-07-03_10-17-13`（正式 `run_training_preflight` 新建；旧 run 全部未动；一次启动、无重试；经正式 `start_training(state, confirmed=True)`）
- 参数：max_epochs=1、train_batch_size=1、gradient_accumulation_steps=4、effective_batch_size=4、num_gpus=1、prompt=book spine、resume_from=None
- 启动前 17 项闸门全过（含 chunking collate / num_chunks=4 / loader batch_size=4 / hydra validated / import guard）
- **结果**：
  - PID/PGID 1496106/1496106；10:17:19 → 10:17:50 JST，29.65 秒
  - exit code **0**，完成 **1 epoch**，summary status=`completed`
  - **2 个 outer iterations**（`Train Epoch: [0][0/2]`）；**每次 4 个 micro-batches**、共 `Trainer/steps_train: 8` 个 micro `_step`——**8 张训练图全覆盖**（2×4=8）；**optimizer step 次数 = 2**（每外层迭代一次，时序未变）
  - checkpoint：`checkpoints/checkpoint.pt`，10,081,250,310 字节
  - `sam3.pt`：size=3450062241、mtime 不变、SHA256 前=后=`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
  - 无残留进程；GPU 回落 804 MiB / 0%

## 6. 缩放生效的实证（修复前后对照）

- **日志 loss 相同**：本 run 与修复前 accum=4 run（`2026-07-03_00-56-32`）的 epoch 平均 `train_all_loss` 完全一致（361.57754135131836）——**这正是设计要求**（logging 用原始未缩放 loss；两 run 数据顺序一致，首迭代前向相同）；
- **checkpoint 不同**：两 run 的 `checkpoint.pt` 字节级比较在偏移 570386 处即分歧——**参数更新确实因 backward 缩放而改变**；
- 数学正确性的权威证据是第 3 节的数值等价测试（真实 `_run_step`、float64、1e-12）——单 epoch 2 步的真实训练指标本身无法区分求和/平均语义，故以单测为准、以 checkpoint 差异为实训佐证。

## 7. 边界确认

未修改 `/home/book/sam3`；未删除旧 run；无自动重试；未改 batch/学习率等无关参数（既有测试参数适配已说明）；无 `git add .`；run/日志/checkpoint 未进 Git。
