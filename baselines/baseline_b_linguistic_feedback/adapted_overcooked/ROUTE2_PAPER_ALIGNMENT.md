# Route 2 论文对齐状态

更新日期：2026-08-10。对应论文：*Learning Rewards from Linguistic Feedback*（Sumers et al., AAAI 2021）。完整审计见 [ROUTE2_AUDIT_REPORT.md](ROUTE2_AUDIT_REPORT.md)。

## 当前实现

Route 2 执行：

```text
(自然语言反馈, 轨迹特征) -> 完整教师奖励向量 w_hat
(mu, tau) <- GaussianUpdate((mu, tau), observation=w_hat, precision=2)
a* = argmax_a w · phi(state, a)
```

- 网络骨架与原代码一致：`EmbeddingBag(30)`，拼接轨迹特征，隐藏层 128 + ReLU，再线性输出奖励向量。
- 原任务的奖励输出为 9 维；Overcooked 使用统一的 53 维奖励 schema，其中当前 36 个合成配置有 26 个变化维、27 个常量维。
- 行为层没有把语言写成固定 subgoal。语言只更新奖励权重，候选行为仍由 `w · phi` 排序；奖励平局时才使用 H0 回退。

## 数据与十折协议

正式语料为 `route2_teacher_feedback.paper_v5.synthetic.json`：

- 83,592 条可识别的合成完整奖励样本；
- 12 个独立、稳定的合成作者 ID；作者跨奖励配置、reference type 和 speech act；
- 36 个完整奖励配置；
- 每条基础反馈跨全部 36 个奖励配置扩展；
- 相同 `(text, trajectory_features)` 若对应不同完整奖励向量，数据构建会 fail closed；当前冲突数为 0；
- teacher–reward 二部图为一个连通分量。

十折仍按原论文的旋转方式：

```text
第 i 折：dev  = fold i
         test = fold (i + 1) mod 10
         train = 其余 8 folds
```

每折同时隔离 `teacher_id` 和 `reward_config_id`，词表只由该折训练文本建立。teacher/reward 轴落在不同分区的混合样本从该折排除。最终 83,592 条中有 8,513 条进入一次且仅一次的 test，覆盖率为 10.184%；这是双轴交叉验证的结构性结果，不能写成“全语料测试”。

超参数候选只在每折 train/dev 上选择。选择视图物理删除 test 样本；10 折赢家为 9 个固定预算 Adam、1 个变化维加权的 paper-SGD 候选。`split_seed=137` 是同一合成语料的新冻结划分，不是新的独立语言数据集。

## 一次性正式测试

所有模型在 test 前先完成以下冻结：

- selection、CV、ensemble manifest 自哈希校验；
- 10 个 winner checkpoint 的 SHA256 与 checkpoint metadata 校验；
- test 读取前原子创建 receipt，防止崩溃或并发导致重复测试；
- reward tie 一律按错误处理，不把稳定排序的 fallback 当正确票。

正式目录已生成 `test_evaluation_receipt.json`，因此 evaluator 会拒绝再次打开这份 test。

### Held-out micro 结果（8,513 条）

| 指标 | Route 2 |
|---|---:|
| 完整 53 维 MSE | 0.012748 |
| 26 个变化维 MSE | 0.025101 |
| 平均 cosine similarity | 0.997620 |
| active-sign accuracy | 99.9969% |
| 最近奖励配置 accuracy | 8,512 / 8,513 = 99.9883% |
| 偏好敏感 reward-context majority | 1,194 / 1,224 = 97.5490% |
| 空手烹饪偏好敏感 majority | 101 / 108 = 93.5185% |

未加权 10 折 macro 为：变化维 MSE `0.021359`、最近奖励配置 accuracy `99.9935%`、偏好敏感 majority `97.6471%`、空手烹饪 majority `93.6111%`。

### 常数与输入消融

| 系统 | 变化维 MSE | 最近配置 accuracy | 空手烹饪 majority |
|---|---:|---:|---:|
| Route 2 | 0.025101 | 99.9883% | 93.5185% |
| fold-train mean 常数 | 1.468598 | 2.2789% | 31.4815% |
| canonical reward 常数 | 1.418902 | 2.2789% | 35.1852% |
| text-only 输入 | 0.037102 | 99.9178% | 91.6667% |
| trajectory-only 输入 | 8.108210 | 0% | 50.0000% |

严格 validity gate 要求模型同时胜过两个常数基线的变化维 MSE、最近配置 accuracy 和偏好敏感烹饪行为；本次三项均通过。

## 结论边界

这次结果证明的是：网络可以在**可识别的合成完整奖励任务**中，对未见 synthetic author 与未见 reward configuration 做组合泛化。

它还不能证明真人语言理解：

- text-only 与完整输入非常接近，说明当前 synthetic full-w 任务主要依赖显式可识别的偏好语义，轨迹只提供小幅回归增益；
- trajectory-only 完全失败，不能据此宣称已经完成真人 temporal credit assignment；
- 4 条独立手写的真人风格局部语言探针只有 `1/4 = 25%`；
- 当前没有人工标注的 Overcooked 完整奖励测试集。

因此，历史 Route 2 目录中的行为数字不再是当前主结果，也不得用来支持真人泛化。真人评论在没有独立完整奖励标签时必须保留为 unlabeled；不能把模型预测或恢复后的 posterior 反写成监督真值。

## 正式文件

- 语料：`data/route2_teacher_feedback.paper_v5.synthetic.json`
- dev-only 选择：`outputs/route2/paper_aligned_v5_seed137_selected/selection_manifest.json`
- 冻结部署：`outputs/route2/paper_aligned_v5_seed137_selected/ensemble_manifest.json`
- 一次性测试：`outputs/route2/paper_aligned_v5_seed137_selected/evaluation_report.json`
- 测试 receipt：`outputs/route2/paper_aligned_v5_seed137_selected/test_evaluation_receipt.json`

## 后续真人验证

1. 冻结 participant/game-disjoint 真人 test；训练、调参和阈值选择只能看真人 train/dev。
2. 人工标注 phrase reference、actor、时间窗口和目标特征，分别量化 reference accuracy、grounding accuracy、窗口命中率及端到端 credit F1。
3. 用盲测比较 frozen/H0 与语言自适应条件，报告偏好满足率、任务回报、后悔值和安全约束违反率。
4. 只有独立 elicitation 得到完整人类奖励配置时，真人样本才可进入 Route 2 监督训练；普通自然语言只用于在线 posterior 更新与独立行为评估。

指涉分类器的独立结果见 [REFERENCE_CLASSIFIER_REPORT.md](REFERENCE_CLASSIFIER_REPORT.md)。
