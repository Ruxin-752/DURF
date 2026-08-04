# Important File Map

## Contents

1. Repository entry points
2. Task backbone
3. Feedback attribution
4. Hu model
5. Tests
6. Project documents
7. Data artifacts
8. Legacy and historical material

All paths are relative to the repository root.

## 1. Repository Entry Points

| Path | Responsibility |
|---|---|
| `README.md` | Top-level setup, Pygame entry points, and active research direction. |
| `pyproject.toml` | Python 3.10-only package and dependency contract. |
| `durf/group_a/play_with_baseline.py` | Main human-AI Pygame runtime; logs sessions; combines Task, Recovery, Coordination, Safety, chat, and Hu. |
| `durf/group_a/deepseek_chat.py` | DeepSeek API client. Reads `DEEPSEEK_API_KEY`, optional model and URL variables. |
| `src/overcooked_ai_py/data/layouts/ring_tomato_onion_10x6_h0_full_task.layout` | Active full-task map, if present under this exact layout path. Confirm with `rg --files src | rg h0_full_task`. |

## 2. Task Backbone

| Path | Responsibility |
|---|---|
| `durf/baseline/collect_rule_teacher_dataset.py` | Rule teacher, candidate Task subgoals, task scores, rollout collection, and target-position metadata. |
| `durf/baseline/task_logic.py` | Shared task/recovery logic, including stale or unneeded ingredient decisions. |
| `durf/baseline/coordination.py` | Explicit conflict classification, Coordination candidates, bounded option state, yield cooldown, and decision logs. |
| `durf/baseline/train_subgoal_executor.py` | Train low-level Keras executor from subgoal-labelled examples. |
| `durf/baseline/evaluate_subgoal_executor.py` | Evaluate task/subgoal executor behavior. |
| `durf/baseline/runtime.py` | Environment, agent, and layout loading utilities. |
| `durf/baseline/train_rllib_agent.py` | Historical/current PPO training utility; not the current ours backbone identity. |
| `durf/baseline/build_human_demo_dataset.py` | Convert successful human sessions into BC data; historical exploration, not current required path. |
| `durf/baseline/collect_recovery_dataset.py` | Collect failure/recovery examples. |
| `durf/baseline/build_reviewed_recovery_dataset.py` | Build reviewed recovery data. |

Current executor model default is referenced inside `play_with_baseline.py`; inspect the CLI default before claiming an exact path. At the handoff snapshot it pointed to:

```text
outputs/subgoal_executors/h0_p0_rule_executor_v5_fast_two_delivery_recovery/executor.keras
```

## 3. Feedback Attribution

| Path | Responsibility |
|---|---|
| `durf/feedback_attribution/schemas.py` | Canonical trajectory, feedback, candidate-event, and attribution record constructors. |
| `durf/feedback_attribution/session_converter.py` | Convert raw session CSV into `trajectory.jsonl` and `feedback_events.jsonl`. |
| `durf/feedback_attribution/condition_features.py` | Fixed Hu condition schema plus rich provenance context; Task/Coordination feature subsets. |
| `durf/feedback_attribution/event_detectors.py` | Deterministic event detector library; facts only, no semantic interpretation. |
| `durf/feedback_attribution/generate_candidate_events.py` | Session-level candidate-event generation command. |
| `durf/feedback_attribution/feedback_type_router.py` | Simple evaluative/imperative/descriptive baseline classifier. |
| `durf/feedback_attribution/sample_builder.py` | Deterministic attribution baseline and candidate alignment without future leakage. |
| `durf/feedback_attribution/llm_attributor.py` | LLM semantic attribution, clarification, schema proposals, audit record. |
| `durf/feedback_attribution/hu_dataset_builder.py` | Build provenance and pairwise labels; consume approved review decisions; reject mixed domains. |
| `durf/feedback_attribution/subgoal_preferences.py` | Canonical Task/Coordination vocabularies, aliases, and conservative event-to-pair suggestions. |
| `durf/feedback_attribution/probe_state_detector.py` | Detect future-recurring state patterns and record candidate rankings for evaluation. |
| `durf/feedback_attribution/review_io.py` | Build compact review items and save decisions separately. |
| `durf/feedback_attribution/review_session.py` | Tkinter review UI and validation rules. |
| `durf/feedback_attribution/review_replay.py` | Map-based trajectory replay inside review. |
| `durf/feedback_attribution/demo_offline_attribution.py` | One-command CSV-to-attribution-to-Hu-label-to-probe pipeline. |

## 4. Hu Model

| Path | Responsibility |
|---|---|
| `durf/hu/subgoal_reranker.py` | `PairwiseSample`, linear reranker, `HierarchicalHu`, training/evaluation, legacy loading. |
| `durf/hu/train_subgoal_reranker.py` | Load reviewed datasets, split by decision domain, train two heads, save model/metrics. |
| `durf/hu/score_subgoals.py` | Offline Task or Coordination candidate scoring. |

## 5. Tests

| Path | Coverage |
|---|---|
| `testing/feedback_attribution_test.py` | Conditions, events, preferences, reviewed overrides, hierarchical Hu, split behavior. |
| `testing/coordination_test.py` | Conflict detection, default yield, Hu override, expiry/cooldown. |
| `testing/baseline_task_logic_test.py` | Task recovery logic. |
| `testing/review_replay_test.py` | Review replay window selection. |
| `testing/overcooked_test.py` and others | Upstream environment tests; heavier than the focused suite. |

## 6. Project Documents

| Path | Use |
|---|---|
| `docs/hierarchical_hu_runtime.md` | Latest Task/Coordination architecture and commands. Read first for Hu/runtime work. |
| `docs/hu_feedback_data_flow.md` | Full attribution and provenance data flow. |
| `docs/candidate_event_taxonomy.md` | Current event definitions, actor/valence, detector limitations, schema gaps. |
| `docs/probe_state_detector.md` | Probe definitions and before/after ranking evaluation. |
| `docs/review_window.md` | Review interface and gold-label boundary. |
| `docs/schema_update_review_policy.md` | Dynamic event/condition schema governance. |
| `docs/hu_v0_subgoal_reranker.md` | Original single-head rationale; latest runtime supersedes stale sections. |
| `docs/day1_pilot_data_flow.md` | Historical detailed logging documentation. |

## 7. Data Artifacts

Each session lives under:

```text
outputs/human_ai_sessions/<session_id>/
```

Important files:

| File | Meaning |
|---|---|
| `session_metadata.json` | Build, agent, layout, timing, Hu settings. |
| `trajectory.csv` | Raw step log including before/after state and runtime decisions. |
| `chat_messages.csv` | Human language and LLM conversation. |
| `trajectory.jsonl` | Normalized step records. |
| `feedback_events.jsonl` | Normalized human feedback. |
| `candidate_events.jsonl` | Program-detected candidate facts. |
| `attribution_preview.jsonl` | Automatic rule/LLM attribution output. |
| `llm_attribution_audit.jsonl` | Prompt/raw/parsed LLM audit. |
| `review_items.jsonl` | Review package. |
| `review_decisions.jsonl` | Human gold/revision decisions; separate from automatic output. |
| `hu_attribution_provenance.jsonl` | Full explanation/audit representation. |
| `hu_subgoal_preferences.jsonl` | Minimal trainable pairwise samples. |
| `schema_updates.jsonl` | Reviewable schema proposals. |
| `probe_hits.jsonl` | Future-detectable probe matches and candidate rankings. |

Hu models live under:

```text
outputs/hu_models/<model_id>/hierarchical_hu.json
outputs/hu_models/<model_id>/metadata.json
```

Most `outputs/` artifacts are intentionally ignored by Git. Package only deliberate representative artifacts.

## 8. Legacy and Historical Material

- `archive/legacy_scalar_feedback/`: J/K scalar feedback and examples.
- `archive/legacy_notebooks/`: old tutorial notebook.
- `models/rllib_agents/`: historical PPO checkpoints and selected packaged agents.
- PPO curriculum/recovery scripts under `durf/baseline/`: preserve for reproducibility, but do not confuse them with the current ours method.

