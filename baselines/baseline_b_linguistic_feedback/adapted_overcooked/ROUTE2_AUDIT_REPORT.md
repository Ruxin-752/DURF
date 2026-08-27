# Route 2 严格审计报告（seed 137）

日期：2026-08-10。

## 审计结论

Route 2 已通过冻结的 synthetic full-reward 测试，但尚未通过真人局部语言测试。

- 通过：未见 synthetic author + 未见 reward configuration 的完整 53 维奖励推断。
- 未通过：4 条独立手写局部语言探针仅 `1/4 = 25%`。
- 因此当前结果是合成系统上限检查，不能外推为“已经理解真人偏好”。

## 严格协议

- 数据：83,592 条合成样本、12 个稳定 synthetic authors、36 个完整奖励配置。
- 输出：53 维奖励向量，其中 26 维随配置变化。
- 十折：fold `i` 做 dev，fold `(i+1)%10` 做 test，其余 8 折训练；teacher 与 reward config 同时隔离。
- 选择：每折只看 train/dev，test 样本从候选选择视图中物理删除。
- 冻结：selection/CV/ensemble 自哈希、10 个 checkpoint SHA256 和 metadata 全部绑定。
- 测试：先创建一次性 receipt，再读取 test；当前 receipt 已完成，禁止重跑。
- test：8,513 条，占完整语料 10.184%；其余为双轴不匹配而排除的组合。

## 最终指标

| 指标 | micro | 10-fold macro |
|---|---:|---:|
| 变化维 MSE | 0.025101 | 0.021359 |
| 最近奖励配置 accuracy | 99.9883% | 99.9935% |
| 偏好敏感行为 majority | 97.5490% | 97.6471% |
| 空手烹饪偏好 majority | 93.5185% | 93.6111% |

补充：完整向量 MSE `0.012748`，cosine `0.997620`，active-sign accuracy `99.9969%`。

常数基线明显更差：canonical 的变化维 MSE 为 `1.418902`、最近配置 accuracy 为 `2.2789%`、空手烹饪 majority 为 `35.1852%`。严格三项 validity gate 均通过。

## 最重要的消融

| 输入 | 变化维 MSE | 最近配置 accuracy |
|---|---:|---:|
| language + trajectory | 0.025101 | 99.9883% |
| language only | 0.037102 | 99.9178% |
| trajectory only | 8.108210 | 0% |

这说明完整模型虽优于 text-only，但主要信号仍来自合成文本中的可识别奖励语义。当前结果不能证明轨迹 credit assignment 已在真人语言上正确。

## 可以与不可以声称的内容

可以声称：

- 网络结构、完整奖励目标和 teacher/reward 双轴十折协议已实现；
- 合成完整奖励任务显著胜过常数基线；
- test 没有参与候选选择，并由 checkpoint/manifest/receipt 冻结链保护。

不可以声称：

- 真人 Overcooked 语言 accuracy 达到上述数字；
- 历史 Route 2 行为数字仍代表当前严格结论；
- 真人 temporal/actor/feature credit 已验证；
- 普通真人评论可以作为完整 53 维监督标签。

## 下一步

1. 收集真人语言，同时保留未标注池；不得用模型预测或 posterior 充当 gold。
2. 另行标注 reference、actor、时间窗口和局部目标特征，测试 credit assignment。
3. 仅对有独立完整 reward elicitation 的真人样本训练 Route 2。
4. 冻结 participant/game-disjoint 真人 test，盲测 frozen/H0 与语言自适应 agent 的偏好满足率和任务回报。

详细实现、常数基线与文件位置见 [ROUTE2_PAPER_ALIGNMENT.md](ROUTE2_PAPER_ALIGNMENT.md)。
