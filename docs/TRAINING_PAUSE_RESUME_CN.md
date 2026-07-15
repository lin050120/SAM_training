# SAM3 训练暂停与恢复

## 功能含义

该功能是持久化暂停：停止训练进程并释放 GPU 显存，之后从原训练 run 的
`checkpoints/checkpoint.pt` 恢复。它可以跨 UI 重启和电脑重启使用。

SAM3 在每个完整 epoch 结束后更新 `checkpoint.pt`。checkpoint 包含模型、optimizer、
epoch、训练步数、loss 状态和 AMP scaler。暂停发生在 epoch 中途时，最近 checkpoint
之后的本 epoch 进度不会保留。

## 暂停训练

1. 打开 `训练预检` 页面。
2. 在监控信息中确认已经生成 `<run_dir>/checkpoints/checkpoint.pt`。
3. 点击 `暂停训练并释放显存`。
4. 等待任务状态显示 `paused`。此时训练进程已经退出，GPU 显存已释放。

如果第一个 epoch 还未完成，UI 会拒绝暂停，避免产生无法恢复的任务。若正在写
`checkpoint.pt.tmp`，请等待保存完成后再点击暂停。

## 恢复训练

1. 打开同一页面的 `阶段 C: 从最近完整 Checkpoint 恢复`。
2. 点击 `刷新可恢复 run`。
3. 选择原来的训练 run。
4. 查看 `恢复预检信息`，确认 `resumable=true`、数据路径、prompt、max epochs 和
   `resume_checkpoint` 正确。
5. 勾选恢复确认框，点击 `恢复训练`。

恢复会复用原 run 的 runtime YAML、输出目录和 `checkpoint.pt`，并分配新的分布式端口。
无需重新运行新训练预检，也不要把 trainer checkpoint 填进 `initial checkpoint`。

## 自动守卫

恢复前，服务端会重新检查：

- 原 run、runtime YAML 和非空 `checkpoint.pt` 均存在；
- 原 run 使用的基础 `sam3.pt` 和 BPE 文件仍然存在；
- runtime YAML 的 checkpoint 输出仍指向原 run；
- train/val 数据路径存在，当前 dataset registry 仍允许原训练模式；
- CUDA 数量满足原 run 要求；
- SAM301 trainer patch 和 SAM3 import 路径正确；
- 没有活动进程正在使用同一个 runtime YAML；
- 已完成的 run 不会被误恢复。

`training_summary.json` 顶层显示最近一次启动结果，`attempts` 数组保存首次启动、暂停和
每次恢复的历史。每次恢复都继续写入同一 run，不创建第二套模型训练目录。

## 限制

- 暂停不能保留尚未完成的 epoch；最多损失从最近完整 checkpoint 到点击暂停之间的进度。
- 不能在第一个 checkpoint 生成前暂停。
- 恢复不用于修改 max epochs、batch size、learning rate、prompt 或数据集。需要修改这些参数时，
  应重新预检并创建新 run。
- 源代码更新不会热加载到已经运行的 UI。更新后需要等当前训练结束，再重启 UI 才能看到新按钮；
  关闭旧 UI 会终止由它管理的活动训练。
