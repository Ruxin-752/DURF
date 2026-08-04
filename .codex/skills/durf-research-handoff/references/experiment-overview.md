# DURF Experiment Overview

## Contents

1. Objective
2. Research contribution
3. Current system
4. Feedback attribution contract
5. Hu model
6. Comparison design
7. Evaluation layers
8. Interpretation constraints

## 1. Objective

DURF studies whether natural-language feedback can help a human teammate express diverse, conditional collaboration preferences to an adaptive agent in Overcooked-AI.

The central claim is not merely that an LLM can parse text. The proposed contribution is an evidence-grounded pipeline that connects:

```text
unconstrained language
+ recent trajectory
+ program-detected candidate events
+ condition schema
-> attributable pairwise preferences
-> observable changes in future collaboration behavior
```

Humans are simultaneously teammates and teachers, generally ordinary participants rather than expert demonstrators. Feedback is collected online during or between play, and the desired product is an agent that preserves task competence while adapting collaboration preferences.

## 2. Research Contribution

The strongest intended contribution is conditional feedback attribution in a cooperative sequential task:

- Temporal attribution: which recent interval the feedback refers to.
- Behavioral attribution: which event or behavior pattern is targeted.
- Semantic attribution: what the user's language refers to.
- Conditional attribution: under which state facts the preference should hold.

Secondary contributions include clarification questions for ambiguity, reviewable dynamic schema proposals, personalization, feedback burden analysis, and task-performance constraints.

## 3. Current System

The active experimental map is `ring_tomato_onion_10x6_h0_full_task`, with an empty-pot full-task start and recipe `tomato + tomato + onion`.

The current best practical task backbone is not pure PPO:

```text
state
-> feasible candidate subgoals
-> rule/planner task scores
-> selected subgoal
-> learned Keras low-level subgoal executor / MotionPlanner fallback
```

Historical PPO agents, curriculum scripts, TAMER-style scalar feedback, and Docker demo assets remain useful baselines or archives but are not the active ours pipeline.

## 4. Feedback Attribution Contract

Roles:

- Program: record complete state/action trajectories; detect facts and candidate events; compute fixed condition features; detect reusable probe states.
- LLM/DeepSeek: classify feedback semantics, select among candidate events, identify relevant conditions, request clarification, and propose reviewable schema extensions.
- Human reviewer: construct pilot gold labels, correct attribution, approve pairwise preferences, and approve/reject schema changes.

Representations must remain distinct:

```text
event = what behavior happened
condition = state facts under which the preference applies
preference = preferred candidate > rejected candidate
probe = a future-detectable state pattern for evaluation
```

## 5. Hu Model

Current Hu is a condition-aware, user-aware, hierarchical linear pairwise ranking model with two independent heads:

```text
Hu_task(user, condition, task_subgoal)
Hu_coord(user, condition, coordination_option)
```

Score form:

```text
global candidate bias
++ user-candidate bias
++ sum(condition value * condition-candidate weight)
```

Training objective:

```text
score(preferred) > score(rejected)
loss = -log sigmoid(score(preferred) - score(rejected))
```

Runtime composition:

```text
final_score = base_score + lambda * Hu_preference_score
```

Hu does not modify environment reward, output primitive movement directly, or replace task safety.

## 6. Comparison Design

The planned non-LLM comparison is a localized version of *Learning Rewards from Linguistic Feedback*:

```text
language
-> feedback type
-> feature/subgoal grounding
-> update feature/subgoal weights
```

Localization may use the same subgoal task backbone and fixed Overcooked schema. Fair comparison requires the same map, task capability, trajectory evidence available before feedback, candidate vocabulary where appropriate, and evaluation states. Do not require the localized baseline to reproduce the original paper's open-domain data volume because this project fixes the map, task, subgoal vocabulary, event schema, and collaboration domain.

Recommended conditions for formal analysis:

- Localized Linguistic baseline.
- Ours without optional clarification/dynamic schema, if an ablation is needed.
- Full ours with evidence-grounded LLM attribution and approved schema handling.

Shadow mode is a diagnostic condition, not a behavior-changing treatment.

## 7. Evaluation Layers

Use five layers and keep their evidence separate:

1. Design audit: zero cross-domain labels, schema consistency, deterministic tests.
2. Attribution audit: temporal IoU, event accuracy, candidate recall, condition F1, clarification quality against human gold.
3. Hu audit: pairwise accuracy and preference margin, separately for Task and Coordination.
4. Behavioral audit: preferred-choice rate, rejected-choice rate, rank/margin changes on recurring probes.
5. Task/human audit: delivery success, reward, completion time, idle/blocking rates, perceived adaptation, workload, feedback burden, LLM latency/failure.

Primary outcomes should remain compact:

- Attribution accuracy.
- Preference compliance on probe states.
- Task non-inferiority under a predeclared margin.

## 8. Interpretation Constraints

- Human review is gold construction and quality control, not evidence of automatic model capability.
- Candidate recall upper-bounds LLM event-selection accuracy.
- A Hu that always yields has not learned conditional preference.
- Task success alone does not establish preference adaptation.
- Preference compliance alone is insufficient if delivery performance collapses.
- Personalization cannot be claimed until held-out per-user data support it.
- Formal participant sample size must follow pilot effect estimates and power analysis, not an arbitrary fixed number.

