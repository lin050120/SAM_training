# Fable5 审查行动清单

日期: 2026-07-03 · 依据: docs/FABLE5_FULL_INDEPENDENT_PROJECT_AUDIT.md · 总判定: APPROVE_WITH_FOLLOWUPS（基础设施冻结+进入人工验收）；正式多 epoch 训练不批准

## 立即阻断项（正式训练的硬前提，非验收阻断）

1. **F-P1-1 数据决策**：确认并处置"7185 标注全部为 SAM3 机器预标注"——人工修正标注重建数据集，或书面批准伪标签自蒸馏策略并重定义指标。二选一，不可默认沿用。
2. **F-P1-2 checkpoint 出口**：trainer→推理权重导出/键重映射工具；`_load_checkpoint` 零匹配时 fail-loud；推理 smoke 必须断言预测数>0（当前 4/4 图零预测）。

## 人工验收前（本周即可做）

3. 验收材料补充三项事实：标注=机器预标注；test=4 张唯一照片（×3 拷贝）、train=120 唯一（40 张双份）；微调 checkpoint 推理零预测。
4. HUMAN_ACCEPTANCE_CHECKLIST 增补检查项：标注来源确认、唯一图数披露、推理零预测=FAIL 判据。
5. 用现有 gt_overlays 做 mask 质量人工抽检（8 点边界/外扩/粘连是否可作起点）。

## 正式训练前（硬条件）

6. **F-P2-1**：数据审计/split/manifest 生成代码入库，可重建并校验现有 manifest（消除 stale-manifest 与文本规则歧义——.jpg 尾部陷阱实测存在）。
7. **F-P2-2**：重复治理（去重或书面接受加权）；重建 test 集（≥30-50 唯一图、困难样本分层：漫画/倾斜/细书脊/粘连/反光）。
8. **F-P2-3**：启用 mask/segm 指标（mask AP + IoU 分布 + pred/GT 面积比至少）；跑原始 sam3.pt 的 baseline eval 作对照；确定 best-checkpoint/验证间隔/early-stop/epoch 数依据。
9. 决策 resume 策略（F-P3-1：当前守卫禁止 resume 且无入口，长训中断即整段重跑）。
10. UI 日志尾部截断（F-P3-2：全量推送在多小时训练下膨胀）。

## 正式训练后

11. 微调 vs baseline 的 per-image paired 对比与困难样本分层评估。
12. checkpoint 与数据版本/split 关联的正式记录（provenance 增 stage/dataset 字段）。

## 可延期

13. run 目录 smoke/formal 阶段标记（F-P3-3）；清理 14-57-30 类 preflight-only 残留的说明。
14. conda 环境 lockfile 导出；`current_system_analysis.md` 等旧 `-n sam3` 示例清理（F-P3-4）。
15. R-3 锁内 yield、preflight JSON 原子写、端口 TOCTOU 残窗（F-P3-6）——理论性。
16. 极小 mask 口径统一（随 #6 工具入库自然解决）。

## 不建议实施

17. DVC 类系统化数据版本管理（当前规模过度工程）。
18. 把 COCO 合并/审计/split 工具塞进训练 UI（应为独立 data_tools CLI）。
19. 立即目录大重构（见下）。
20. 对 sam301 做 git 化或进一步补丁扩展（现 manifest+守卫已足；补丁面越小越好）。

## 目录重构建议（时机：data_tools 入库同批、正式训练开跑前）

- 第一步（零风险）：docs/ 分层为 docs/current/（8-10 份现行规范，README 给出唯一入口链）与 docs/archive/（全部阶段报告/验收/评审）。
- 第二步：新增 data_tools/（audit/split/manifest/merge/dedupe/export 代码——先有码后有目录）。
- 第三步（可选低收益）：config/ 与 data_manifests/ 合并为 manifests/；scripts/ 保持正式入口不动。
- 不移动：core/、ui/、tests/、patches/、runs/、data/。历史 run 与 import 路径不受第一、二步影响。
