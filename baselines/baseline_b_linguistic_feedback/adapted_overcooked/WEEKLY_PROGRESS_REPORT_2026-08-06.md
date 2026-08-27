# 本周进展汇报：从语言反馈学习 Overcooked 协作偏好

日期：2026-08-06

> **历史报告说明（2026-08-10）**：本文记录 8 月 6 日当时的实现。Route 2 的旧
> 多种子、text-only 与行为数字已被冻结的 seed-137 teacher/reward 双轴测试取代；
> 最新指标和真人语言限制见 [ROUTE2_AUDIT_REPORT.md](ROUTE2_AUDIT_REPORT.md)。

## 1. 一句话说明现在做到了什么

我们已经完成一个可运行的闭环：

```text
人类自然语言反馈
    -> 判断反馈在说哪段行为、哪些特征、是表扬还是批评
    -> 更新该用户的奖励权重 w
    -> 对 H0 提供的可执行候选计算 w · phi(state, candidate)
    -> 选择得分最高的行为
```

关键点是：语言不能直接触发固定动作。它只能改变奖励权重；“是否提前拿原料、是否等待、是否让路”仍由统一的 `w · phi` 比较产生。

目前工程闭环和合成数据验证已经完成，项目处于“可以开始真人 pilot，但还不能宣称已经理解真人偏好”的阶段。

## 2. 整体架构

系统分成三层：

1. **Credit assignment 层**：理解一句反馈在评价谁、哪段轨迹、哪个动作或特征，以及正负方向。
2. **Reward learning 层**：Route 1 或 Route 2 把反馈转化为 53 维奖励权重 `w`。
3. **Behavior 层**：H0 枚举当前可执行的候选，统一使用 `w · phi` 排序，再由 H0 执行选中的 subgoal。

Route 1 和 Route 2 的最终输出相同，都是同一套 53 维奖励；它们的区别是“如何从语言得到奖励”。

## 3. 当前 credit assignment 实际是怎么做的

当前 credit assignment 不是一个单独的端到端模型，而是“固定近期窗口 + 轨迹特征提取 + 语言 reference grounding”。Route 1 显式使用这些结果；Route 2 只使用近期轨迹向量，让神经网络隐式完成语言与奖励的对应。

### 3.1 先保留最近 25 个 step

游戏每一步记录：

- AI 和人类动作；
- step 前后的状态；
- 双方位置和持有物变化；
- 锅的状态；
- 环境奖励；
- 当时选择的 subgoal、候选及其 `phi`。

系统滚动保留最近 25 个实际 transition 和最近 25 个 subgoal decision。进入新 episode 时清空这个窗口，避免把上一局行为归因给下一局反馈。

玩家暂停并提交反馈后，当前实现不是根据句子预测精确的 `[start_step, end_step]`，而是先把最近最多 25 个 step 聚合成一个轨迹特征向量。当前能够从状态变化检测：

- 拿番茄、洋葱、盘子或汤；
- 把原料放入锅；
- 完成送餐；
- 锅为空、烹饪中或汤已完成；
- AI 挡住人类、让人类等待、抢占下一格或产生碰撞风险。

如果实际 transition 没提取到有效事件，就回退到最近已选择 subgoal 的 `phi` 并进行合并。

因此，当前 temporal credit assignment 的真实含义是“反馈指向最近 25 步的聚合行为”，还不是一个经过真人 gold window 验证的精确时间定位器。这是目前最需要通过真人数据改进的部分。

### 3.2 Route 1 的显式语言归因

Route 1 收到反馈后依次执行：

1. **反馈形式分类**：用规则区分 evaluative、imperative 和 descriptive。
2. **短语切分**：按有限标点规则把一句话拆成短语。
3. **Reference type 分类**：TF-IDF + Logistic Regression 判断短语属于：
   - trajectory；
   - feature；
   - action_spatial；
   - action_behavioral；
   - other。
4. **Grounding 到 53 维特征**：
   - trajectory / action_behavioral：使用最近 25 步聚合的轨迹特征；
   - action_spatial / imperative：先在当前可执行候选里识别语言指向的动作，再使用该候选的 `phi`；
   - feature：使用 TF-IDF 多标签 grounding 模型预测特征，并用当前候选中实际存在的特征做 mask；
   - 模型低置信或不可用时，回退到可审核的关键词 grounding；
   - other：放弃更新。
5. **Valence 提取**：使用修改后的 VADER；`do not / stop / avoid` 等禁止表达强制按负向处理，命令某个可执行动作默认表示正向期望。
6. **置信度门控**：reference、grounding 或 valence 低于阈值时拒绝更新；强 grounding 可以挽救保守的 reference abstention。

当前 live feedback 默认使用更高 observation precision，但还会乘以输入、reference、grounding 和 valence 的置信度。系统还会拒绝近期语义重复反馈，并限制单次 posterior 的最大权重变化和 KL，防止一句误判造成过大更新。

当前没有独立的 AI / Human / Team actor 分类器。在线更新默认把反馈当作对 AI 行为的指导；人类动作只用于判断 AI 是否挡路或发生协调冲突。这一点也需要在真人 credit annotation 中补齐。

### 3.3 Route 1 如何使用 credit

Route 1 会为每个成功归因的短语生成：

```text
target feature vector x
valence y
observation precision beta
```

然后将其作为高斯奖励观察更新 `p(w_user)`。也就是说，Route 1 的 credit assignment 是显式、可检查的：日志能看到 reference type、grounding source、目标特征、valence、有效 precision，以及哪些奖励权重被改变。

### 3.4 Route 2 如何使用 credit

当前 Route 2 不经过 Route 1 的反馈形式分类、reference classifier 或显式 feature grounding。它的输入是：

```text
原始反馈文本 + 最近 25 步的归一化轨迹特征向量
```

神经网络直接输出完整 `w_hat`，再与当前用户权重混合。因此 Route 2 的 temporal credit 仍来自同一个固定近期窗口，但语言具体对应哪些特征是网络内部隐式学习的。

这使 Route 2 更容易泛化到不同措辞，但可解释性较低；目前只能从 `top_changes` 看到哪些权重变化，不能像 Route 1 一样给出完整的 reference/grounding 中间证据。

### 3.5 一个具体例子

如果玩家在 AI 提前拿洋葱后说：`Do not grab an onion this early.`

- 轨迹窗口会包含 AI 拿洋葱产生的 `ingredient_onion / pick_onion` 等特征；
- Route 1 会把 `do not` 识别为负向，把 onion/action grounding 到当前候选特征，再对这些特征执行负向贝叶斯更新；
- Route 2 会把整句话和同一近期轨迹向量输入网络，得到准备类权重为负的 `w_hat`；
- 两条路线最后都重新计算所有候选的 `w · phi`；只有当 `WAIT` 的奖励确实更高时才选择等待。

这个例子中语言没有直接把 subgoal 设置成 `WAIT`，它只改变了奖励。
## 4. Route 1：可解释的在线贝叶斯更新

### 4.1 设计

Route 1 不直接训练一个端到端奖励网络。它先把自然语言拆成可审核的奖励证据：

```text
feedback + recent trajectory
    -> feedback form
    -> temporal/reference grounding
    -> referenced feature vector x
    -> valence y
    -> Bayesian update of p(w)
```

每个用户维护一个奖励分布：

```text
w_user ~ Normal(mu_user, Sigma_user)
```

一条反馈形成近似观察：

```text
y = x^T w_user + noise
```

然后更新 `mu_user` 和 `Sigma_user`。当前实现支持论文中的两种方式：

- **Literal**：主要更新语言明确提到的特征；
- **PseudoPragmatic**：除明确特征外，对未提到特征加入较弱的反向信息。

### 4.2 Route 1 的用处

- 适合真人在线适应：一条反馈就能更新，不需要先收集大量该用户数据；
- 每次更新都能解释：归因到哪段轨迹、哪些特征、权重改变多少；
- 能保存每个用户独立的 posterior，并在下一局继续；
- 最接近原论文“语言形成奖励观察，再更新奖励信念”的机制。

### 4.3 Route 1 的限制

- 性能高度依赖 credit assignment；如果窗口、对象或正负方向错了，贝叶斯更新会稳定地朝错误方向累积；
- 语言 grounding 目前主要由合成数据和规则验证，尚未在足够真人语言上验证；
- Reward learner 本身不需要神经网络训练，但它前面的 reference/grounding 模块仍需要真人标注改进。

## 5. Route 2：从语言和轨迹推断完整奖励的神经网络

### 5.1 设计

Route 2 使用与原论文代码相同的网络骨架：

```text
EmbeddingBag(vocab, 30)
+ normalized trajectory features
-> Linear(..., 128)
-> ReLU
-> complete reward vector w_hat
```

原论文任务输出 9 维奖励；迁移到 Overcooked 后输出同一特征空间中的 53 维奖励。网络结构没有改变，只改变了领域特征维度。

训练目标已从旧版的“反馈指向的局部特征 × 情感”改为：

```text
(language, trajectory) -> complete 53-dimensional teacher reward
```

为解决只有一个 `gold_comfort_weights.json` 的问题，我们自动生成了 36 个隐藏奖励配置，包括：

- 更重视让路和路径协调；
- 更重视分工或目标尊重；
- 不同任务效率偏好；
- 谨慎、中性、主动准备三类预取偏好。

奖励配置、语言风格和场景在 train/dev/test 之间隔离。

### 5.2 Route 2 的用处

- 学习语言表达之间的泛化，不必为每个短语手写规则；
- 可以从 population 数据给新用户一个奖励估计或先验；
- 能把复杂语言和轨迹联合映射到同一个奖励空间；
- 适合在积累多名真人数据后做离线训练和跨用户泛化。

### 5.3 Route 2 的限制

一句局部反馈通常不能唯一决定完整 53 维奖励。例如“不要太早拿洋葱”说明了准备偏好，却没有说明用户对让路、抢目标、重复劳动等特征的看法。

因此 Route 2 可以学习相关权重的方向，但不能把单句输出的完整 `w_hat` 当作该用户的绝对真值。拿到真人数据后，Route 2 更适合作为 population prior 或低置信度奖励观察，而不是每句话都覆盖整个用户奖励。

## 6. Route 1 和 Route 2 的关系

| 项目 | Route 1 | Route 2 |
|---|---|---|
| 核心方法 | 显式 grounding + 贝叶斯更新 | 神经网络直接预测奖励向量 |
| 是否需要大量训练数据 | 较少 | 需要多教师、多语言数据 |
| 在线个性化 | 强 | 当前使用混合更新，仍较粗糙 |
| 可解释性 | 高 | 较低 |
| 跨表达泛化 | 依赖 grounding | 更强 |
| 共同输出 | 53 维奖励 `w` | 53 维奖励 `w` |
| 共同决策 | `argmax w · phi` | `argmax w · phi` |

当前运行时把它们作为两个可对照模式，避免同一句反馈被重复计算两次。

拿到真人数据后的推荐关系是：

```text
Route 2 学 population prior / 提供 grounding 建议
                    +
Route 1 维护每个真人的 posterior，并逐条累积反馈
                    ->
个性化 reward w_user
```

这样既利用神经网络理解多样语言，又保留论文式、可解释的小样本在线更新。

## 7. Agent 目前训练到了什么程度

### 7.1 已完成的部分

- Route 1 在线 posterior 更新、日志、保存和恢复已经接入游戏；
- Route 2 已使用 36 个合成奖励配置完成多随机种子训练；
- 烹饪期 `GET_TOMATO / GET_ONION / GET_DISH / WAIT` 都是中性候选；
- 主策略严格使用 `w · phi`，不再使用手工 `lambda` 或固定 WAIT guard 改写结果；
- 精确奖励平局时才回退到 H0 原决策；
- UI 已支持暂停后输入自然语言、中文输入法、粘贴和在线更新提示；
- 自动测试通过：迁移基线 106/106、在线代理 29/29、UI 4/4。

### 7.2 当前模型结果

| Route 2 模型 | 目的 | 奖励 MSE | cosine | subgoal accuracy | 真人式预取措辞 |
|---|---|---:|---:|---:|---:|
| 完整风格模型 | 验证网络能否恢复完整隐藏奖励 | 0.003014 | 0.999577 | 114/114 | 2/4 |
| local-only 模型 | 当前真人聊天默认模型 | 0.529602 | 0.920745 | 111/114 | 4/4 |

local-only 模型在固定烹饪状态中能够做到：

- “利用烹饪时间准备下一轮” -> 提高准备特征权重并选择预取；
- “现在太早，保持空手” -> 降低准备特征权重并选择 `WAIT`。

候选和 `phi` 在两次测试中完全相同，唯一变化的是语言产生的 `w`。

### 7.3 现在可以和不可以下的结论

现在可以说：

- 模型结构、奖励更新和行为选择链已经跑通；
- 合成多教师条件下能推断不同奖励并改变行为；
- 没有再通过 subgoal 规则写死预取偏好。

现在还不能说：

- 已经理解未见真人的自然语言；
- 已经学到稳定的个人偏好；
- 与 H0 相比，人类实际更满意；
- 4 条措辞测试可以代表真实人类泛化。

因此当前科学状态是 **synthetic proof-of-concept / human-pilot ready**。

## 8. 有了真人玩家和真人数据后，如何更新模型

真人仍然只需要提供自然语言，不需要提供 53 维权重，也不需要知道 reward configuration ID。

### 8.1 需要采集两类数据

第一类是自然游戏反馈：

- feedback text；
- episode、反馈时刻和之前的轨迹窗口；
- 更新前后的候选、`phi`、`w`、ranking 和实际动作；
- participant/session ID，只用于分组和保存个体 posterior，不直接触发行为。

第二类是少量 preference probe：

- 在固定状态中给出两个或多个可执行候选；
- 让参与者选择 preferred / rejected；
- probe 分为适应集和最终 held-out 测试集。

这些 probe 不是要求用户填写权重，而是把自然偏好转换为可验证的成对约束。

### 8.2 真人在线更新：优先更新个人奖励，不立即微调整个网络

对每条反馈：

1. 找到被评价的时间窗口和 actor；
2. 判断是评价已发生行为，还是要求未来采取某种行为；
3. grounding 到特征证据 `x`，而不是直接输出最终 subgoal；
4. 得到 valence 和置信度；
5. 使用 Route 1 更新该用户的 `mu_user, Sigma_user`；
6. 用更新后的 `w_user` 重新计算所有候选的 `w · phi`。

例如：

- “刚才太早拿洋葱”归因到实际拿取轨迹，对相关准备特征形成负奖励观察；
- “下次可以提前准备洋葱”归因到语言所指的候选特征，形成正奖励观察；
- “拿洋葱而不是盘子”使用 `phi(onion) - phi(dish)` 形成成对偏好证据；
- 模糊反馈降低 observation precision 或请求澄清。

这里识别“洋葱”只是确定反馈在谈哪些事实特征，不代表系统直接执行 `GET_ONION`。最终仍要让所有可执行候选参与 `w · phi` 竞争。

### 8.3 真人离线更新：怎样真正改进 Route 2

只有自然语言而没有奖励真值时，不能直接把每句话当作完整 53 维监督目标。合理流程是：

1. 研究者审核一部分反馈的时间窗口、reference、feature 和 valence；
2. 根据同一用户的多条语言观察和 preference probes，联合估计该用户的 latent reward `w_user`；
3. 使用该用户估计出的 `w_user`，为其多条 `(language, trajectory)` 提供一致的教师奖励目标；
4. 离线微调 Route 2，使其从未见用户语言预测 population-level reward；
5. 按 participant 划分 train/dev/test，不能把同一用户的相邻反馈拆到不同集合；
6. 新模型建立新版本和模型哈希，旧在线状态不能直接混用。

Route 2 更新的是跨用户语言理解能力；Route 1 更新的是当前用户的个性化奖励。两者分工不同。

### 8.4 如何保证有效但不写死

- H0 只枚举可执行候选并执行动作；
- `phi` 只描述事实，不包含“这个动作一定好”；
- 自然语言只形成 reward evidence；
- 所有候选统一用 `w · phi` 排序；
- 不使用 `if text contains onion: GET_ONION`；
- 不使用用户 ID 直接查固定策略；
- 训练期间可按原论文从 posterior 采样 `w` 探索；
- 最终测试使用 posterior mean，保证可复现；
- 更新幅度由 grounding 置信度或 observation precision 控制，而不是永远固定 `alpha`。

## 9. 如何量化 linguistic credit assignment 是否正确

每条真人反馈需要人工 gold annotation：

- gold time window；
- gold actor；
- gold reference type；
- gold subgoal / feature set；
- gold valence；
- 是否歧义。

其中 20%–30% 由第二名标注者独立审核，先确认人工 gold 本身可靠。

### 9.1 分项指标

1. **Temporal IoU**：预测窗口与 gold window 的交并比；同时报告 start/end MAE。
2. **Actor Accuracy**：AI / Human / Team 是否正确。
3. **Reference Accuracy**：trajectory / action / feature / other 是否正确。
4. **Grounded Feature F1**：归因到的特征集合是否正确。
5. **Valence Accuracy**：正、负、不明确是否正确。

必须分别报告 evaluative、imperative、正反馈、负反馈、清晰反馈和模糊反馈，不能只给一个总体平均。

### 9.2 一个端到端 credit 指标

定义 **End-to-End Linguistic Credit Accuracy（ELCA）**。一条反馈只有同时满足以下条件才算正确：

```text
Temporal IoU >= 0.5
actor 正确
reference type 正确
grounded feature F1 >= 0.5，或 referenced candidate Top-1 正确
valence 正确
```

```text
ELCA = 端到端归因正确的反馈数 / 可审核反馈总数
```

它回答“这句话最终有没有被记到正确的人、时间和奖励特征上”。

### 9.3 检查更新方向是否正确

对于有 gold preferred/rejected 的反馈：

```text
margin_before = w_before · (phi_preferred - phi_rejected)
margin_after  = w_after  · (phi_preferred - phi_rejected)
Delta margin  = margin_after - margin_before
```

定义 **Credit-Consistent Update Rate（CCUR）**：

```text
CCUR = Delta margin > 0 的反馈数 / 可审核反馈总数
```

ELCA 测语言理解和归因，CCUR 测归因结果是否真的让奖励朝正确方向变化。两者必须一起报告。

### 9.4 必要消融

- **Oracle credit + reward update**：人工提供正确 grounding；
- **Automatic credit + reward update**：系统自动 grounding；
- **No update**：不更新奖励。

如果 oracle 好而 automatic 差，问题在 credit assignment；如果 oracle 也差，问题在 reward update、`phi` 或候选设计。

## 10. 如何量化 human preference 是否被尊重

credit assignment 正确不等于最终行为一定符合人类偏好，因此需要独立的行为实验。

### 10.1 实验流程

每位参与者经历：

1. **Baseline**：H0，不学习反馈；
2. **Shadow**：系统更新 `w`，但不改变行为，用于检查模型是否学到了偏好；
3. **Apply**：更新后的 `w` 真正通过 `w · phi` 改变选择。

顺序随机或交叉平衡。适应阶段允许反馈；最终测试阶段冻结权重，不再接收反馈，并使用未在适应阶段出现的 probe 状态。

### 10.2 主要行为指标

定义 **Held-out Personalized Preferred-Choice Rate（HPCR）**：

```text
HPCR = agent 在 held-out probe 中选择该参与者 preferred 候选的次数
       / preferred 候选可执行的 probe 总数
```

最重要的比较是：

```text
Preference Gain = HPCR_apply - HPCR_baseline
```

只有 Preference Gain 在参与者层面的置信区间明显高于 0，才能说学习后的 agent 更尊重人类偏好。

同时报告：

- Rejected-choice Rate；
- `w · phi(preferred) - w · phi(rejected)` 的平均 margin；
- 正 margin 的 probe 比例；
- 不同参与者之间相反偏好能否得到相反 ranking。

### 10.3 任务能力约束

偏好改善不能靠停止做任务换来。需要同时验证：

- Delivery Success Rate 相对 H0 的下降不超过预先规定的 10%；
- Deliveries per Episode；
- Time to First Delivery；
- Idle Rate；
- Blocking/冲突率。

### 10.4 人类主观结果

每局结束后可使用 1–7 分量表：

- AI 是否理解了我的反馈；
- AI 是否适应了我的协作偏好；
- 行为变化是否符合预期；
- 合作是否流畅；
- 是否愿意继续指导该 AI。

主观量表只能作为行为指标的补充，不能替代 held-out preference choice。

### 10.5 统计单位

统计单位必须是参与者，而不是 timestep。报告参与者内 paired difference、bootstrap 95% CI 或配对检验，避免把大量相邻步骤当成独立样本制造虚假显著性。

## 11. 接下来最重要的工作

1. 建立真人 feedback gold annotation：window、actor、reference、feature、valence。
2. 建立适应集和 held-out preference probe，不用同一 probe 同时训练和测试。
3. 先做 6–10 人 pilot，优先计算 ELCA 和 CCUR，定位 credit assignment 问题。
4. 用 preference probes 和语言观察估计每个用户的 latent `w_user`。
5. 在 participant-disjoint 数据上更新 Route 2，并保留 Route 1 个体 posterior。
6. 正式比较 Baseline、Shadow、Apply，报告 HPCR、Preference Gain 和任务非劣性。

## 12. 给导师的当前结论

本周完成的是奖励学习机制和可运行系统，而不是最终真人效果证明。Route 1 提供与原论文一致、可解释的在线奖励 posterior；Route 2 提供从语言和轨迹推断奖励的跨表达能力；二者最终进入同一个 `w · phi` 行为选择层，所以没有把人类语言写成固定 subgoal 规则。

下一阶段需要用真人数据分别回答两个问题：

1. **Linguistic credit assignment 对不对？** 用 ELCA、Temporal IoU、Feature F1 和 CCUR 回答。
2. **Human preference 有没有被尊重？** 用 held-out HPCR、Preference Gain 和任务非劣性回答。

只有这两组结果同时成立，才能合理声称 agent 不仅“听懂了反馈”，而且“在不明显损害任务能力的前提下，行为更符合该用户的偏好”。
