# Hu-v0: 条件化 Subgoal 偏好重排序器

> 状态说明：本文保留单层 Hu-v0 的设计背景。当前实现已经升级为
> Task / Coordination 双头结构，最新工程口径与命令见
> [`hierarchical_hu_runtime.md`](hierarchical_hu_runtime.md)。

本文档说明当前 Hu-v0 的真实定位、训练标签、模型形式和运行方法。这里特意不再使用“PPO + Hu”的说法，因为当前实际可用 backbone 已经转向：

```text
state -> rule/planner subgoal -> learned low-level executor action
```

Hu-v0 应该接在 **subgoal 选择层**，而不是接在 PPO reward、PPO logits 或低层动作策略上。

## 0. 当前代码真实情况

当前在线 agent 的 high-level 决策已经开始从硬编码 `return` 过渡到“候选 subgoal + task_score”结构。核心入口仍然是：

```text
durf.baseline.collect_rule_teacher_dataset.rule_teacher_decision
```

旧版本是一串硬编码优先级：

```text
如果手里有原料 -> 尝试放锅
如果手里有盘子且汤好了 -> 去取汤
如果汤好了 -> 去拿盘子
如果汤在煮 -> 去拿盘子或准备下一轮食材
如果锅还缺原料 -> 去拿缺的原料
否则 -> WAIT
```

新版本保留旧接口，但内部已经变成：

```text
candidate subgoals -> score -> sort -> choose
```

也就是说，`rule_teacher_decision` 仍然返回：

```text
(chosen_subgoal, planner_action)
```

但现在它是通过：

```text
rule_teacher_candidates(state)
-> [CandidateSubgoal(subgoal, task_score, action, reason), ...]
-> choose_task_candidate(...)
```

选出来的。

这一步完成了基础任务模型和 Hu 的接口对齐：基础模型也开始表达为 `condition/subgoal -> task_score`。Hu-v0 已经可以离线学习 `condition + subgoal -> preference_score`，下一步要做的是在线合并：

```text
rule_task_score + lambda * Hu_score
```

这样 Hu 才能真正改变 “WAIT / GET_DISH / YIELD / GET_USEFUL_INGREDIENT” 等 subgoal 的选择。

每个候选还会记录 `reason`，例如 `soup_ready_get_dish` 或 `pot_needs_ingredient`。这个字段只解释 task scorer 为什么给出该候选和分数，属于 debug / audit 信息，不进入 Hu 训练。Hu 训练仍只使用：

```text
user_id + condition_features + preferred_subgoal/rejected_subgoal
```

这能避免 Hu 学到规则命名本身，而不是学真实条件化偏好。

## 1. Hu-v0 学什么

Hu-v0 学的是：

```text
在某个用户、某组条件下，人类更偏好哪个 subgoal，而不是哪个 subgoal。
```

训练样本来自：

```text
outputs/human_ai_sessions/<session_id>/hu_subgoal_preferences.jsonl
```

一条训练样本长这样：

```json
{
  "record_type": "hu_pairwise_subgoal_preference",
  "user_id": "PILOT01",
  "layout": "ring_tomato_onion_10x6_h0_full_task",
  "condition_features": {
    "pot_cooking_or_ready": true,
    "human_has_dish": true,
    "ai_empty_handed": true
  },
  "preferred_subgoal": "GET_USEFUL_INGREDIENT",
  "rejected_subgoal": "WAIT",
  "source_event": "AI_failed_to_prepare_ingredient_while_waiting",
  "source_feedback_id": "feedback_001"
}
```

这条样本表达：

```text
当锅在煮/快好、人类拿着盘子、AI 空手时，
这个用户更希望 AI 去准备有用食材，而不是原地等待。
```

## 2. Hu-v0 不学什么

Hu-v0 不学习：

- 怎么走路；
- 怎么避开墙；
- 怎么拿番茄、拿洋葱、端汤；
- 怎么从 observation 直接输出 Overcooked 低层动作；
- 怎么改环境 reward；
- 怎么替代当前 subgoal executor。

这些仍然由 rule/planner 和 learned low-level executor 负责。

Hu-v0 只做：

```text
candidate subgoals -> preference scores -> reranked subgoals
```

## 3. 模型形式

当前实现：

- 文件：[durf/hu/subgoal_reranker.py](/C:/Users/my185/Desktop/研究/durf/DURF/durf/hu/subgoal_reranker.py:1)
- 模型：线性 pairwise ranking model
- 输入：`user_id + condition_features + subgoal`
- 输出：`preference_score`

分数分解：

```text
score(user, condition, subgoal)
= global_subgoal_bias[subgoal]
+ user_subgoal_bias[user, subgoal]
+ sum_k condition_weight[k, subgoal] * condition[k]
```

其中 condition 值编码为：

```text
True  ->  1
False -> -1
None  ->  0
```

这样做的好处是：

- 数据少时也能训练；
- 每个 condition 对每个 subgoal 的影响可以解释；
- 可以自然加入 user-specific 偏好；
- 未来可以平滑升级到小 MLP，但第一版不被黑盒复杂度绑架。

## 4. 训练目标

每条样本有：

```text
preferred_subgoal > rejected_subgoal
```

训练目标：

```text
Hu(user, condition, preferred_subgoal)
>
Hu(user, condition, rejected_subgoal)
```

损失函数：

```text
loss = -log sigmoid(
  score(preferred_subgoal) - score(rejected_subgoal)
)
```

这和 preference learning / DPO 的核心思想类似：不用要求人类给绝对分数，而是从“这个比那个更好”的偏好对中学习。

## 5. 和当前 executor 怎么结合

当前执行链应理解为：

```text
当前游戏状态
-> condition_features
-> 规则/程序生成候选 subgoals
-> rule/planner 给出 task_score
-> Hu-v0 给出 hu_score
-> final_score = task_score + lambda * hu_score
-> 选择最终 subgoal
-> low-level executor + MotionPlanner 执行具体动作
```

概念公式：

```text
final_score(subgoal)
= executor_task_score(subgoal)
+ lambda * Hu(user, condition_features, subgoal)
```

当前代码已经把 Hu 接入在线游戏决策，并支持默认 shadow scoring 与
`--hu-apply` 实际重排。Task 和 Coordination 使用独立分数头。

## 6. 运行方式

### 6.1 先生成 Hu 训练标签

```powershell
cd "C:\Users\my185\Desktop\研究\durf\DURF"

python -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs\human_ai_sessions\<session_id> `
  --user-id PILOT01 `
  --lookback-steps 30 `
  --use-llm
```

输出：

```text
outputs/human_ai_sessions/<session_id>/hu_subgoal_preferences.jsonl
```

如果这局没有自然语言反馈，或者反馈无法可靠归因，训练样本可能是 0，这是正常保护机制。

### 6.2 训练 Hu-v0

```powershell
python -m durf.hu.train_subgoal_reranker `
  --dataset outputs\human_ai_sessions\<session_id> `
  --output-dir outputs\hu_models\hu_v0_pilot01 `
  --epochs 200 `
  --learning-rate 0.05
```

输出：

```text
outputs/hu_models/hu_v0_pilot01/hierarchical_hu.json
outputs/hu_models/hu_v0_pilot01/metadata.json
```

`metadata.json` 中最重要的是：

- `train_metrics.pairwise_accuracy`
- `validation_metrics.pairwise_accuracy`
- `top_condition_weights`

`top_condition_weights` 可以帮助我们解释：

```text
哪个条件提升/压低了哪个 subgoal 的偏好分数？
```

### 6.2.1 每个参与者独立 train/test

评估协议是：同一参与者先玩一局收集训练数据 -> 训练 -> 同一人再玩一局，用第二局的数据做 test：

```powershell
python -m durf.hu.train_subgoal_reranker `
  --dataset outputs\human_ai_sessions\<第一局 session> `
  --test-dataset outputs\human_ai_sessions\<第二局 session> `
  --output-dir outputs\hu_models\pilot01_eval `
  --epochs 200
```

`--test-dataset` 强制 per-participant 协议：

- test 里出现没有训练数据的 user 会直接报错（每个 test session 必须是同一参与者的后续对局）；
- 如果某个 user 的 test 反馈时间早于其训练反馈时间，会在 `protocol_checks.temporal_warnings` 警告（很可能 `--dataset` / `--test-dataset` 传反了）；
- 与训练数据复用同一 `source_feedback_id` 的样本会被剔除，不进入 test 指标（防止同一条反馈泄漏）。

`metadata.json` 中除聚合的 `test_metrics_by_level` 外，还按参与者拆分输出 `test_metrics_by_user`，以及协议检查结果 `protocol_checks`（`protocol_ok`、`test_users_without_training_data`、`temporal_warnings`）。

### 6.3 用 Hu-v0 给候选 subgoals 打分

推荐使用 `--condition-file`，避免 PowerShell JSON 引号问题：

```json
{
  "pot_cooking_or_ready": true,
  "human_has_dish": true,
  "ai_empty_handed": true
}
```

```powershell
python -m durf.hu.score_subgoals `
  --model outputs\hu_models\hu_v0_pilot01\hierarchical_hu.json `
  --decision-level task `
  --user-id PILOT01 `
  --condition-file outputs\hu_models\hu_v0_pilot01\example_condition.json `
  --candidate-subgoals GET_USEFUL_INGREDIENT WAIT GET_DISH YIELD
```

输出示例：

```json
[
  {
    "subgoal": "GET_USEFUL_INGREDIENT",
    "preference_score": 2.1
  },
  {
    "subgoal": "GET_DISH",
    "preference_score": 0.4
  },
  {
    "subgoal": "YIELD",
    "preference_score": -0.2
  },
  {
    "subgoal": "WAIT",
    "preference_score": -1.8
  }
]
```

## 7. 当前边界

Hu-v0 目前只能学到训练标签中出现过的条件组合和 subgoal 对比。它不能凭空学会没有标注过的偏好。

当前最需要的数据不是“大量普通 gameplay”，而是高质量的：

```text
自然语言反馈
-> 可信归因
-> condition_features
-> preferred/rejected subgoal
```

也就是说，下一步最重要的是采集几局带反馈的 H0 session，并人工检查 `hu_attribution_provenance.jsonl`。

## 8. 后续升级路线

优先级建议：

1. Hu-v0 离线训练和解释稳定。
2. 写人工审核工具，允许修正 `preferred/rejected_subgoal`。
3. 将 Hu-v0 接入在线 subgoal 排序，但先只做 logging / shadow mode。
4. 再开启实际影响行为的 `lambda`，从很小值开始。
5. 数据量足够后，考虑 Hu-v1 小 MLP。
