# Baseline B Overcooked Migration Notes

> **Current status (2026-08-10):** Route 1 is implemented, and synthetic Route 2
> passes the frozen complete-reward teacher/config holdout. This is not a human
> language result: the separate local-language probe is only 1/4. Route 2 uses
> 36 hidden configurations, the paper's ten-fold rotation, and a hash-frozen
> ten-model deployment ensemble; see
> [ROUTE2_AUDIT_REPORT.md](ROUTE2_AUDIT_REPORT.md).
> Every deviation from the original paper/experiment, and the explicit
> "stop-before-training" boundary, is catalogued in
> [`DIFFERENCES_FROM_PAPER.md`](DIFFERENCES_FROM_PAPER.md). Sentiment is now
> NLTK VADER (English-only); the feedback corpus is English-only.

## What Was Migrated

The original *Learning Rewards from Linguistic Feedback* pipeline uses natural language to update beliefs over reward features. This Overcooked adaptation keeps the core structure but changes the task domain:

```text
human linguistic feedback
-> feedback form classification
-> sentiment extraction (valence)
-> grounding to Overcooked features (reference vector)
-> Bayesian conjugate update of a Gaussian reward-weight belief
-> probe state evaluation (posterior mean + sampled policy)
```

The adaptation supports both offline evaluation and live Pygame feedback. It
does not change PPO checkpoints: language updates reward weights, those weights
rerank H0-valid subgoals by `w · phi`, and H0 still executes the selected
subgoal. Controlled probes and paper-style held-out folds evaluate the same
reward-to-choice path used at runtime.

## Key Migration Decisions

The original source remains under `original_rewards_repo/` and is treated as reference code only. Its features and pretrained models are specific to the original colored-shape task, so they are not imported into the Overcooked runtime.

The active Overcooked code lives in `adapted_overcooked/`:

```text
adapted_overcooked/
  data/
    overcooked_features.json
    initial_weights.json
    feedback_examples.json
    probe_states.json
  src/
    feature_schema.py
    feedback_form_classifier.py
    sentiment_extractor.py
    overcooked_grounding.py
    belief_model.py           # GaussianBelief: MVN prior + conjugate update
    observations.py           # feedback -> Gaussian observation(s)
    reward_weight_model.py    # BayesianRewardLearner (literal / pseudopragmatic)
    subgoal_schema.py         # H0 8-subgoal vocabulary (shared with h0_planner)
    subgoal_featurizer.py     # (state, subgoal) -> shared reward features
    subgoal_reranker.py       # re-rank H0 subgoals with the learned reward
    trajectory_featurizer.py
    probe_evaluator.py
  scripts/
    evaluate_probe_states.py
    evaluate_subgoal_reranking.py
    run_baseline_b_pipeline.py
```

The first probe set targets `ring_tomato_onion_10x6` and focuses on human comfort:

- do not take a dish the human is going to use
- move away when blocking the human
- choose a complementary task
- avoid cutting into the human's shortest path
- do not interrupt the human's plan
- help without interfering

## How The Pieces Map

| Original idea | Overcooked adaptation |
|---|---|
| MDP features `phi(s,a)` | Hand-authored Overcooked action features in `probe_states.json` |
| Gaussian belief over reward weights `w` (`beliefs.py`) | `GaussianBelief` in `belief_model.py`, same `N(0, 25)` prior |
| Conjugate update (`MultivariateNormalLearner`) | `BayesianRewardLearner.update` via `update_from_observation` |
| Literal / PseudoPragmatic learners | `--mode literal` / `--mode pseudopragmatic` |
| Sampled action selection (`execute_trajectories`) | `probe_evaluator.evaluate_probes_sampled` |
| Evaluative feedback | Updates recent or provided trajectory features |
| Imperative feedback | Grounds to an intended action's feature vector |
| Descriptive feedback | Grounds by keywords to mentioned behavior features |
| Original task score | Probe accuracy |

## Subgoal Compatibility (H0 integration)

The reward learner is now **subgoal-compatible**: the weights learned from
language feedback can re-rank H0's 8 subgoals (`GET_TOMATO`, `PUT_TOMATO_IN_POT`,
`GET_ONION`, `PUT_ONION_IN_POT`, `GET_DISH`, `PICKUP_SOUP`, `SERVE_SOUP`,
`WAIT`). This works because training and runtime share ONE feature schema:

```text
language feedback --(train)--> reward belief w over overcooked_features.json
                                              |
state + candidate subgoal --(subgoal_featurizer)--> phi(state, subgoal)
                                              |
                     re-rank H0's feasible subgoals by  task_score + lambda * comfort_score
```

- `subgoal_featurizer.featurize_subgoal(state, subgoal)` produces the same
  53-dim `phi` used in learning (task features from recipe/pot progress,
  coordination features from the human's current target).
- `subgoal_reranker.choose_subgoal(w, state, feasible_subgoals)` picks the best
  of the **task-valid** subgoals H0 proposes; task-validity (the safety floor)
  stays with H0, comfort re-ranking is added on top.
- H0 owns stall-prevention (e.g. cap consecutive `WAIT`s); once the human stops
  contesting a resource the comfort bonus for waiting disappears.

Smoke check (learned vs zero vs hand-authored-prior weights on subgoal probes):

```powershell
python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_subgoal_reranking.py --mode literal
```

Current result: learned 5/5, hand-authored-prior 5/5, zero-weights 0/5 (all
ties), i.e. the language feedback — not just the prior — drives the subgoal
choice.

## Configuration

No API key is required for the minimum runnable baseline. DeepSeek chat remains part of the pygame interface, but this Baseline B version uses deterministic rules so it can run without network access.

Run from the DURF repository root. On Windows PowerShell:

```powershell
cd /d E:\Research\DURF
$env:PYTHONPATH="$PWD;$PWD\src"
```

If using `uv`, keep the existing Python 3.10 environment:

```powershell
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python --version
```

## Tests And Smoke Runs

Static probe evaluation:

```powershell
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_probe_states.py
```

Expected shape:

```text
Probe Accuracy: 6/6 = 100.0%
[OK] P01_do_not_take_human_dish_or_block_serving_route: ...
```

Full feedback-to-weights pipeline:

```powershell
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/run_baseline_b_pipeline.py
```

Expected outputs:

```text
Updated feature weights:
  blocks_human_path: ...
  respects_human_intent: ...

Probe Accuracy: ...
Result JSON: baselines\baseline_b_linguistic_feedback\adapted_overcooked\outputs\baseline_b_pipeline_result.json
```

## Next Migration Steps

The next milestone is to connect real pygame session logs:

1. Run `durf.group_a.play_with_baseline` and collect `outputs/human_ai_sessions/<session_id>`.
2. Use `durf.feedback_attribution.session_converter` to create `trajectory.jsonl` and `feedback_events.jsonl`.
3. Extend `trajectory_featurizer.py` to compute richer `nphi(trajectory)` from `state_before_json` and `state_after_json`.
4. Feed those trajectory features into the same `RewardWeightModel`.

This keeps data collection, attribution, and reward-weight evaluation separate, which makes the baseline easier to debug.
