# Baseline B Overcooked Migration Notes

## What Was Migrated

The original *Learning Rewards from Linguistic Feedback* pipeline uses natural language to update beliefs over reward features. This Overcooked adaptation keeps the core structure but changes the task domain:

```text
human linguistic feedback
-> feedback form classification
-> sentiment extraction
-> grounding to Overcooked features
-> linear reward weight update
-> probe state evaluation
```

The adapted version is intentionally offline. It does not change PPO checkpoints or the pygame agent policy. It evaluates whether learned feature weights prefer the expected action in controlled probe states.

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
    reward_weight_model.py
    trajectory_featurizer.py
    probe_evaluator.py
  scripts/
    evaluate_probe_states.py
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
| Reward feature weights `w` | JSON/dict weights in `RewardWeightModel` |
| Evaluative feedback | Updates recent or provided trajectory features |
| Imperative feedback | Grounds to an intended action's feature vector |
| Descriptive feedback | Grounds by keywords to mentioned behavior features |
| Original task score | Probe accuracy |

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
