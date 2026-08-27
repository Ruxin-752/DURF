# Submission Status

## 结论

当前状态：`engineering_ready_human_evaluation_pending`。核心工程链路已经具备，尚不能替代的缺口是真人五分类/temporal-credit gold 与真人偏好对照实验。

## 已完成并可报告

- 原论文五分类随机行协议复现：Accuracy `87.16%`，Macro-F1 `75.25%`。
- 五分类严格 task/text-disjoint 暴露回归：Accuracy `77.36%`，Macro-F1 `72.52%`。
- Route 1 phrase grounding 合成 untouched：Micro-F1 `75.59%`，Macro-F1 `58.99%`。
- Route 1 的 53 维现已全部审计：`33` 维可改变候选排序，`6` 维仅为共享上下文，`14` 维当前不支持；后两类显式 mask/拒识。
- Route 2 冻结合成测试：`8513` 条，53维 MSE `0.012748`，变化维 MSE `0.025101`，最近奖励配置 `99.99%`。
- Route 2 trajectory-required 反事实测试：`60` 条，full/text-only 变化维 MSE `0.054841` / `0.512521`，full 相对改善 `89.30%`；最近目标为 `100.00%` / `10.00%`。
- 40 回合同 seed 合成在线适应：偏好满足率由 `10.0%` 提升至 `77.5%`，regret 由 `23.6` 降至 `4.6`，安全违规由 `14` 降至 `3`。这只是系统诊断，不是真人结论。
- 无界面 Route 2 启动、v3 状态保存和精确恢复 smoke 已通过；恢复前后状态 SHA 完全一致。
- 辅助三分类：`89` 条人工确认的 AI 候选语言，Accuracy `64.04%`，Macro-F1 `63.06%`；它不是论文 87% 指标。

## 验证

- 自动测试共 `239` 项：adapted `196/196`、DURF baseline `35/35`、UI `8/8`，失败和错误均为 0。
- `90` 个 Python 文件语法检查通过，`git diff --check` 通过。

## 尚需真人完成

- 五分类真人 gold：当前 `0` 条。
- 真人语句对应的 actor、目标事件/时间窗、目标 feature 与 polarity 标注。
- 新真人局中的真实语言更新与行为变化验收（自动保存/恢复链路已通过）。
- frozen H0 与 online-adaptive 的盲测，报告偏好满足率、regret、任务回报和安全违规。

## 口径

论文 87% 只对应五分类原协议复现；Route 2 高分只说明合成完整奖励任务成功。未采集上述真人证据前，不声明真人语言 accuracy 或真人 preference improvement。
