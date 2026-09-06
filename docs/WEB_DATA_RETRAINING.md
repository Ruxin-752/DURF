# Web 玩家数据与离线重训练

2026-09-06 线上已发布第 6 版 `durf-synthetic-v3-20260906`，默认 Route1。三个模型文件哈希与验收版本一致，授权管理员的生产 D1 隔离写读删通过；未授权 export/diagnostic 返回 401。详见 `SYNTHETIC_RELEASE_2026-09-06.md`。这证明隔离数据链路可用；现有玩家预测仍不是训练真值，本轮及后续当前方案继续仅使用合成数据。

> 当前约束（2026-09-06）：用户明确要求只使用合成数据。本轮训练与验收未使用后台玩家数据，也不启动下文历史人工标注流程。后台接口仍用于经同意的研究记录及运维检查；数据连通不等于已有可用于训练的真值。当前合成训练入口为 `scripts/train_synthetic_feedback_v3.py`。线上部署尚未执行，最终工程测试与线上结果由发布流程补充。

v3 第三轮合成验收为 speech 143/150、grounding 146/150、混合句组件集合 18/20；去精确重叠后分别为 142/149、145/149、17/19，全部预设门槛仍通过。Evaluative + feature 的 speech 仅 15/20，仍是弱项。去重分析沿用原预测，不是新盲测；AI 编写标签的合成成绩不是人工验证的玩家准确率。详见 `SYNTHETIC_RELEASE_2026-09-06.md`。

两个模型采用 word TF-IDF unigram/bigram + logistic regression，词表只拟合训练集，模型选择使用开发集。前两轮失败样本已转为曝光开发数据，第三轮封存集不参与选择。0.55 是预先固定的拒绝阈值，未按 dev/test 准确率调优；T=1 原始分数未独立校准。

## 当前数据存在哪里

浏览器先把事件放在当前页面的内存队列中，断网会重试；关闭页面后，尚未上传的数据可能丢失。匿名参与者 UUID 保存在浏览器 `localStorage`。同步成功的数据写入线上 Cloudflare D1 的 `DB` binding，不写入 Git 仓库。

上传队列按实际 UTF-8 字节切批到 256 KiB 请求上限以内，单事件 payload 仍限 16,384 字符。共享 trace 编码减少重复，不截断或取整特征值；不可容纳的单事件不会通过裁剪伪装成完整记录。页面关闭时 keepalive 另有更小限制，内存队列不等于可靠离线持久化。

D1 的研究数据表是：

- `sessions`：匿名参与者、会话、同意版本、客户端和 schema 版本；
- `events`：有序的游戏动作与事件。`durf-web-event-v2` 的 `tick_summary` 每 0.5 秒记录前后状态、请求/执行的双方动作、reward、done 和结构化事件；
- `feedback`：玩家原话、三分类概率、逐短语预测、模型 hash、`route` 和 `route_trace`；
- `research_session_tokens`、`research_rate_limits`、`research_security_secrets`：鉴权与防滥用数据，不进入研究导出。

网页隐藏 Route1/Route2 选择器。新的本地实现默认执行 Route1：`speech_act` 三分类用于玩家显示，独立 `grounding` 决定 action / feature / trajectory，再进入 Pragmatic 奖励更新。任一短句无法可靠定位时，全句不提交更新，并保留近期轨迹。Route2 保留为显式部署配置，其已有模型来自早前合成语料。实际线上版本必须以部署报告核实，不能由本地默认值推断。

`speech_act` 分别表示评价、要求/建议与事实陈述。grounding 的 action 指明确动作要求或反事实，feature 指状态属性/规则/偏好，trajectory 指已发生行为或对其评价，二者没有一一硬映射。纯状态描述即使预测为 feature，也可能因没有可操作奖励目标而拒绝更新。

Route1 的 VADER、中性 +15、compound ×30、precision=2 和完整未指及特征补集遵循原文计算。`paper-full-feature-complement-v1` 恢复完整 53 维补集，修复了可行特征补集缩小后负面评价反向提升 onion 的问题，已有同 prior、同快照的真实执行回归。厨房绑定、适用性拒绝、全句事务及明确禁止句 -30/no-complement 分支仍是迁移规则，不能称为原文实验的完整复现。

研究表不主动保存姓名、邮箱或原始 IP，也不会发送邮件。托管平台仍可能保留必要的运维日志。当前尚无自动过期、玩家删除接口和自动备份；正式招募前必须确定保留期限、删除流程与备份策略。

## 管理员导出

管理员通过 `GET /api/research/export` 导出 JSONL，并提供 `Authorization: Bearer <ADMIN_EXPORT_TOKEN>`。令牌只放在部署 secret 和管理员本机环境变量中，不能提交 Git。每行 `recordType` 为 `session`、`event` 或 `feedback`。

导出文件包含原话和研究标识，必须放在未跟踪的私有目录。当前接口会在内存中一次拼出整个 JSONL；数据规模增加前应改为分页或流式导出。

## 当前后台验证范围

管理员 `/api/research/diagnostic` 在独立诊断表用合成值执行写入、读回、删除，不创建参与者、同意、session、event 或 feedback 记录。本地 Miniflare 实际 D1 引擎 roundtrip 已通过：HTTP 200、读回匹配、清理后 0 行、未创建研究表，证据为 `artifacts/synthetic-acceptance-20260906/local-d1-roundtrip.json`。

该结果仅覆盖本地数据库运输检查，不代表生产 D1、参与者采集链路或训练标签质量已经通过。`/model-check` 只在浏览器内显示模型结果，不提交研究接口、不代替用户同意。旧线上记录只做过结构审计，本轮未创建研究记录冒充真实玩家验证，也未使用其文本训练。线上 manifest 与生产隔离诊断仍待部署后记录。

## 1. 历史能力：生成盲标任务和固定切分（本轮不执行）

从仓库根目录运行：

```bash
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/prepare_web_retraining_data.py prepare \
  --export private/web-export.jsonl \
  --output-dir private/web-retraining/run-YYYYMMDD \
  --split-registry private/web-retraining/previous-run/split_registry.json \
  --filter-consent-version durf-anonymous-research-en-2026-08-27-v3
```

第一次建库时，必须把上面命令中的 `--split-registry ...` 替换为显式的 `--initialize-split-registry`。以后每次必须传上一次成功输出的 `--split-registry PATH`，并写到新的 run 目录。两个参数必须二选一，不能同时传，也不能都省略；输入 registry 和输出目录不能指向同一个文件。

同意版本有两种互斥模式：`--filter-consent-version VERSION` 显式保留该版本的完整 session 及其关联 event/feedback，并在 manifest 的 `consent_selection` 中记录原始数量、排除数量和所选记录的 SHA-256；`--consent-version VERSION` 保持原有严格检查，遇到其他版本的反馈就失败，不会自动过滤。过滤前仍检查关联，不能用过滤掩盖跨会话的错误引用。省略两者不会筛选同意版本；不得把结构审计的无筛选结果直接当作允许训练的研究数据。

该命令会校验记录类型、ID、会话关联、事件序号、schema、同意版本和模型 hash，并生成：

- `feedback_annotation_tasks.jsonl`：给标注员的盲标文件，只含文本、任务 ID 和空标签，不含线上预测、概率、route、用户或会话；
- `feedback_annotation_join.jsonl`：私有回连表，保存任务 SHA-256、原记录、预测来源和固定 split；不能交给盲标员；
- `route2_unlabeled.json`：可回放的 Route2 候选池，明确标为无监督目标，不能直接训练；
- `split_registry.json`：追加式私有切分登记表，固定 participant、session 和规范化文本 SHA-256 的 split；
- `prepare_manifest.json`：输入/输出 SHA-256、数量与切分审计。

2026-09-06 的转换器修复通过 payload 的 `classifierModelHash` 和 `updaterModelHash` 分别校验 feedback 与 event 的 hash，要求小写 64 位 SHA-256。Route2 允许双模型来源；新 Route1 仅在 `feedbackSemanticsVersion=speech-act-grounding-v1` 且 `groundingModelHash=updaterModelHash` 时允许两个不同 hash。部分声明或不匹配时拒绝，`selectedRoute` 存在时也必须匹配 feedback.route。回连表保留这些来源与语义版本。完全无这两个字段的旧记录仍要求共享 hash（或原先允许的空 event hash），不会猜测 updater。模型预测不是训练真值；payload 的 `trainingScope=synthetic_only` 指模型训练来源，不表示该玩家输入是合成数据。

新 Route1 完整逐句定位保存在原始 event payload，编码由 `web/lib/grounding-research-trace.ts` 定义：

| 字段 | 含义 |
|---|---|
| `groundingVectorEncoding` | 固定为 `indexed-dense-v1` |
| `groundingFeatureNames` | 同一 payload 共享、有序的特征名数组 |
| `groundingFeatureVectors` | 引用名到等长数值/`null` 数组的表；`null` 为缺失，显式 0 保留 |
| `groundingModelContexts` | 上下文引用名到 `{modelHash, threshold, adaptation}` 的表 |
| `groundingPredictions[].targetFeaturesRef` | 目标特征向量引用 |
| `groundingPredictions[].pragmaticAlternativesRef` | 补集向量引用 |
| `groundingPredictions[].contextRef` | 模型 hash、阈值和方法版本的共享上下文引用 |

逐句还保留 phrase、label、confidence、status、reason、selectedSubgoal、directivePolarity、rawSentiment、effectiveValence、valenceSource 和最终 `committed`。`status='updated'` 可以仅表示候选计算成功，整句提交后 `committed` 才为 true；任一句失败时不提交 posterior。

消费者应使用 `restoreGroundingFeatureVector` 和 `restoreGroundingModelContext` 或等价严格逻辑，检查编码、长度、引用和有限数值。不能把 `null` 转为 0，也不能只导出引用字符串；旧键值向量表与新 indexed 编码要按版本区分。原始关联 event 与全部共享表必须一起保存。当前盲标转换器保留分类来源和回连 ID，定位向量需从原始关联 event 恢复，不能当成额外监督 gold。

策略执行回执比较同一前状态、同一人类动作下的实际执行与更新前权重反事实。posterior 变化、候选计划变化与已经执行动作变化是不同事实，分析时不能互换。

轨迹优先使用 `gameSnapshot.recentTrajectoryFeatures`，即当前客户端实际送入 Route2 的向量；空向量也是有效的实际输入，不能退回单帧状态。非法值直接报错。旧记录仍可读取 `gameSnapshot.features` 或前一条 `tick_summary.featureCounts`，但 `trajectory_is_exact_model_input=false`，不得当作已恢复实际模型输入。此修复不生成三分类人工真值，也不生成 Route2 的独立奖励监督。

真实后台审计见 [WEB_BACKEND_AUDIT_2026-09-06.md](WEB_BACKEND_AUDIT_2026-09-06.md)。当日当前同意版本只有 1 个匿名参与者、14 条反馈，稳定切分全部进入 train，没有 dev 或 frozen-test；这些数据适合盲标和错误分析，尚不足以批准重新训练的模型上线。

新登记项按稳定的 80% train、10% dev、10% frozen-test 分配。单位不是单句话，而是“同一匿名参与者组成的连通组”；如果不同参与者提交了完全相同的规范化文本，也会合并到同一组。已经登记的 participant、session 或文本永不重新哈希或迁移；后来的同文本参与者继承原 split。如果一批新数据把两个不同的固定 split 连在一起，命令会明确失败，等待人工调查，不会移动 frozen 数据。小样本可能暂时没有某个 split，应继续收集，不能手工搬运样本填满。

## 2. 历史能力：人工标注和质检（本轮不执行）

历史工具要求标注员只能看到盲标任务，并提供 `Evaluative`、`Imperative`、`Descriptive` 标签。本轮新模型将日常 speech act 与独立 action/feature/trajectory grounding 分开；旧五类 reference 的折叠标签不能自动作为新 speech_act 真值。下面示例仅说明已有工具接口，不表示当前已授权人工标注。

审核后的 JSONL 每行格式为：

```json
{"annotation_id":"web_annotation_...","classification_label":"Imperative","review_status":"approved","annotator_id":"reviewer-01","annotation_revision":1}
```

`approved` 应同时表示标签复核和隐私检查已经完成。分歧样本先裁决再批准。禁止把线上模型预测复制为人工标签。

完成标注后运行：

```bash
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/prepare_web_retraining_data.py finalize-classifier \
  --tasks private/web-retraining/run-YYYYMMDD/feedback_annotation_tasks.jsonl \
  --join private/web-retraining/run-YYYYMMDD/feedback_annotation_join.jsonl \
  --annotations private/web-retraining/run-YYYYMMDD/reviewed_annotations.jsonl \
  --prepare-manifest private/web-retraining/run-YYYYMMDD/prepare_manifest.json \
  --output-dir private/web-retraining/run-YYYYMMDD/classifier
```

该步骤先用 `prepare_manifest.json` 验证完整 tasks/join 文件的 SHA-256，再要求 task、join、annotation 的 ID 完全一致并复验逐任务 hash。任何 split 或 provenance 篡改都会在读取标签前失败。它只接受人工批准的规范标签，并输出：

- `human_feedback_form_web_train.json`；
- `human_feedback_form_web_dev.json`；
- `human_feedback_form_web_frozen_test.jsonl`；
- `classifier_split_manifest.json`。

## 3. 历史能力：人类三分类训练与冻结评测（本轮不执行）

训练只能读取 train/dev：

```bash
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/train_feedback_form_classifier.py \
  --human-train private/web-retraining/run-YYYYMMDD/classifier/human_feedback_form_web_train.json \
  --human-dev private/web-retraining/run-YYYYMMDD/classifier/human_feedback_form_web_dev.json \
  --output private/web-retraining/models/feedback-form-YYYYMMDD.joblib \
  --report private/web-retraining/models/feedback-form-YYYYMMDD.report.json
```

模型、阈值和校准只在 dev 上选择。选择完成并冻结模型后，才封存和评估 frozen-test：

```bash
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/prepare_human_feedback_form_holdout.py \
  --input private/web-retraining/run-YYYYMMDD/classifier/human_feedback_form_web_frozen_test.jsonl \
  --model private/web-retraining/models/feedback-form-YYYYMMDD.joblib \
  --test-output private/web-retraining/run-YYYYMMDD/classifier/frozen-test.json \
  --sidecar private/web-retraining/run-YYYYMMDD/classifier/frozen-test.sidecar.json \
  --text-origin human_authored \
  --confirm-human-labels

python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_human_feedback_form_test.py \
  --model private/web-retraining/models/feedback-form-YYYYMMDD.joblib \
  --test private/web-retraining/run-YYYYMMDD/classifier/frozen-test.json \
  --sidecar private/web-retraining/run-YYYYMMDD/classifier/frozen-test.sidecar.json \
  --report private/web-retraining/run-YYYYMMDD/classifier/frozen-evaluation.json
```

发布门槛至少检查 accuracy、每类 recall、macro-F1、ECE、低置信覆盖率，以及 Python/浏览器概率一致性。现有通用 holdout sidecar 会写明它自身没有 teacher/session identity；本转换器的私有 split manifest 能证明参与者与会话隔离，但评测脚本还没有把这份证明绑定进同一份签名报告，这是下一步审计缺口。

## 4. Route2 训练所需的独立目标

Web 轨迹和当前模型算出的 53 维权重都不是 Route2 的监督真值。`route2_unlabeled.json` 只能用于待标注池和行为分析。

Route2 的每个监督样本还必须独立取得：

- 完整 53 维 `teacher_reward_weights`；
- `teacher_id` 与 `reward_config_id`；
- target provenance 只能是 `experiment_assigned_reward_config` 或 `independent_reward_annotation`。

补齐独立 reward assignment 后，才能使用 `prepare_route2_dataset.py` 和 `train_route2_paper_crossval.py`。现有脚本会拒绝 prediction、pseudo 或 recovered posterior 作为 reward gold，并按 `teacher_id` 和 `reward_config_id` 做论文式 cross-validation。十折训练还要求足够多且可切分的教师/奖励配置。

当前只合成的授权下，这些目标应来自受控合成奖励配置；不能把玩家 posterior 或模型预测当 gold，也不启动人工奖励标注。本轮保留的 Route2 artifact 已追溯到此前合成语料，不是本次重新训练的模型。

训练完成后依次用 `finalize_route2_selection.py` 冻结选择、`evaluate_route2_paper_crossval.py` 做一次性评测，再进入浏览器导出。当前尚缺“Web D1 候选 → 独立奖励分配 → 完整 Route2 监督 corpus”的转换器；在实验协议确定前不能自动补造目标。

## 5. 发布与回滚

本轮使用 `scripts/promote_synthetic_web.py` 校验通过的报告、模型/数据/预测/源码 hash 和 stateful 拒绝证据，生成带版本的 speech/grounding artifact 及 `manifest-synthetic-v1.json`，保留旧 `manifest.json`。新加载器核验模型字节 hash；旧的 `export_web_models.py` 固定文件名流程不代表本轮发布路径。

构建和打包收据绑定 Git HEAD、源码文件 SHA 与产物 SHA，源码或产物变化后需要重建。本轮线上部署尚未执行，不能将本地默认模型或诊断结果写成生产成功记录。

安全发布顺序是：

1. 在 staging 目录导出新 artifact，保存训练报告、数据 manifest 和 SHA-256；
2. 运行 Python/浏览器一致性测试、冻结评测和行为验证，按用户已确认的发布范围执行；
3. 使用不可变的版本文件名先发布 speech/grounding artifact，并保留所引用的 Route2 artifact；
4. 最后原子更新本轮的 `/models/manifest-synthetic-v1.json` 版本指针；
5. 观察错误率和数据质量；需要回滚时恢复上一份 manifest 和对应 artifact。

历史 `export_web_models.py` 仍默认写固定文件名 `feedback-form-v3.json` 与 `route2-v5.json`；当前合成发布脚本已采用版本文件名与独立 manifest。回滚需要恢复已保存的发布版本、manifest 及对应 artifact，不把历史固定文件名工具当作新发布入口。网页不会边玩边自动替换训练好的生产模型。

当前允许的闭环为：

`新合成文本及明确生成标签 → train/dev 隔离与重叠审计 → 独立 speech/grounding 训练 → 冻结候选 → 全新封存合成验收 → 奖励行为验证 → 版本化 artifact + manifest → 受控发布/回滚`

现存玩家记录在当前授权下不进入训练。未来若改变数据范围，仍需适用同意、独立标签、质量复核和 participant/session/text 分组隔离，不能复制在线预测成为真值。最终全量测试、线上 manifest、页面及授权管理员隔离 D1 诊断结果由本轮发布记录补充。
