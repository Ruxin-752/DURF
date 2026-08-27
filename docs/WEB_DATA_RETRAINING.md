# Web 玩家数据与离线重训练

## 当前数据存在哪里

浏览器先把事件放在当前页面的内存队列中，断网会重试；关闭页面后，尚未上传的数据可能丢失。匿名参与者 UUID 保存在浏览器 `localStorage`。同步成功的数据写入线上 Cloudflare D1 的 `DB` binding，不写入 Git 仓库。

D1 的研究数据表是：

- `sessions`：匿名参与者、会话、同意版本、客户端和 schema 版本；
- `events`：有序的游戏动作与事件。`durf-web-event-v2` 的 `tick_summary` 每 0.5 秒记录前后状态、请求/执行的双方动作、reward、done 和结构化事件；
- `feedback`：玩家原话、三分类概率、逐短语预测、模型 hash、`route` 和 `route_trace`；
- `research_session_tokens`、`research_rate_limits`、`research_security_secrets`：鉴权与防滥用数据，不进入研究导出。

网页现在隐藏 Route1/Route2 选择器，当前实验默认执行 Route1；Route1/Route2 的模型 artifact、算法代码以及导出字段仍保留。以后如需 A/B 实验，应由服务端给会话分组并把分组写入数据，不能由玩家自行切换，也不能重写旧会话。

研究表不主动保存姓名、邮箱或原始 IP，也不会发送邮件。托管平台仍可能保留必要的运维日志。当前尚无自动过期、玩家删除接口和自动备份；正式招募前必须确定保留期限、删除流程与备份策略。

## 管理员导出

管理员通过 `GET /api/research/export` 导出 JSONL，并提供 `Authorization: Bearer <ADMIN_EXPORT_TOKEN>`。令牌只放在部署 secret 和管理员本机环境变量中，不能提交 Git。每行 `recordType` 为 `session`、`event` 或 `feedback`。

导出文件包含原话和研究标识，必须放在未跟踪的私有目录。当前接口会在内存中一次拼出整个 JSONL；数据规模增加前应改为分页或流式导出。

## 1. 生成盲标任务和固定切分

从仓库根目录运行：

```bash
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/prepare_web_retraining_data.py prepare \
  --export private/web-export.jsonl \
  --output-dir private/web-retraining/run-YYYYMMDD \
  --split-registry private/web-retraining/previous-run/split_registry.json \
  --consent-version durf-anonymous-research-en-2026-08-27-v3
```

第一次建库时，必须把上面命令中的 `--split-registry ...` 替换为显式的 `--initialize-split-registry`。以后每次必须传上一次成功输出的 `--split-registry PATH`，并写到新的 run 目录。两个参数必须二选一，不能同时传，也不能都省略；输入 registry 和输出目录不能指向同一个文件。

该命令会校验记录类型、ID、会话关联、事件序号、schema、同意版本和模型 hash，并生成：

- `feedback_annotation_tasks.jsonl`：给标注员的盲标文件，只含文本、任务 ID 和空标签，不含线上预测、概率、route、用户或会话；
- `feedback_annotation_join.jsonl`：私有回连表，保存任务 SHA-256、原记录、预测来源和固定 split；不能交给盲标员；
- `route2_unlabeled.json`：可回放的 Route2 候选池，明确标为无监督目标，不能直接训练；
- `split_registry.json`：追加式私有切分登记表，固定 participant、session 和规范化文本 SHA-256 的 split；
- `prepare_manifest.json`：输入/输出 SHA-256、数量与切分审计。

新登记项按稳定的 80% train、10% dev、10% frozen-test 分配。单位不是单句话，而是“同一匿名参与者组成的连通组”；如果不同参与者提交了完全相同的规范化文本，也会合并到同一组。已经登记的 participant、session 或文本永不重新哈希或迁移；后来的同文本参与者继承原 split。如果一批新数据把两个不同的固定 split 连在一起，命令会明确失败，等待人工调查，不会移动 frozen 数据。小样本可能暂时没有某个 split，应继续收集，不能手工搬运样本填满。

## 2. 人工标注和质检

标注员只能看到盲标任务，按原论文的三个外部类别标注：`Evaluative`、`Imperative`、`Descriptive`。五类 reference classifier 是内部路径，不是新的人工 gold 类别。

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

## 3. 三分类训练与冻结评测

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

## 4. Route2 训练

Web 轨迹和当前模型算出的 53 维权重都不是 Route2 的监督真值。`route2_unlabeled.json` 只能用于待标注池和行为分析。

Route2 的每个监督样本还必须独立取得：

- 完整 53 维 `teacher_reward_weights`；
- `teacher_id` 与 `reward_config_id`；
- target provenance 只能是 `experiment_assigned_reward_config` 或 `independent_reward_annotation`。

补齐独立 reward assignment 后，才能使用 `prepare_route2_dataset.py` 和 `train_route2_paper_crossval.py`。现有脚本会拒绝 prediction、pseudo 或 recovered posterior 作为 reward gold，并按 `teacher_id` 和 `reward_config_id` 做论文式 cross-validation。十折训练还要求足够多且可切分的教师/奖励配置。

训练完成后依次用 `finalize_route2_selection.py` 冻结选择、`evaluate_route2_paper_crossval.py` 做一次性评测，再进入浏览器导出。当前尚缺“Web D1 候选 → 独立奖励分配 → 完整 Route2 监督 corpus”的转换器；在实验协议确定前不能自动补造目标。

## 5. 发布与回滚

`export_web_models.py` 会导出三分类和 Route2 浏览器 artifact 以及 `manifest.json`。网页从 `/models/manifest.json` 读取两个 artifact 路径，并在加载前验证各自 SHA-256；任一 hash 不匹配就拒绝使用。

安全发布顺序是：

1. 在 staging 目录导出新 artifact，保存训练报告、数据 manifest 和 SHA-256；
2. 运行 Python/浏览器一致性测试及冻结评测，由人工批准；
3. 使用不可变的版本文件名先发布两个 artifact；
4. 最后原子更新 `/models/manifest.json`，把它作为唯一生产版本指针；
5. 观察错误率和数据质量；需要回滚时恢复上一份 manifest 和对应 artifact。

当前 `export_web_models.py` 仍默认写固定文件名 `feedback-form-v3.json` 与 `route2-v5.json`，尚未实现不可变文件命名、发布历史或一键回滚命令。这些是部署自动化的剩余工作；在完成前应在私有发布记录中保存每一版 manifest，并手工审批更新。网页不会边玩边自动替换生产模型。

完整闭环为：

`D1 导出 → 校验/脱敏 → 盲标与质检 → participant/session-disjoint split → 三分类或 Route2 离线训练 → frozen eval → 人工审批 → 版本化 artifact + manifest → 受控发布/回滚`
