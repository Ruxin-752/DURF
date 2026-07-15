# Adapted Overcooked Baseline B

This folder contains the Overcooked adaptation of Baseline B from *Learning Rewards from Linguistic Feedback*.

The first runnable version is intentionally offline and deterministic:

```text
linguistic feedback
-> feedback type classification
-> sentiment extraction
-> Overcooked feature grounding
-> feature weight update
-> probe state evaluation
```

It does not modify the PPO policy or the live pygame game loop. It learns a small feature-weight reward model and evaluates that model on hand-authored Overcooked probe states.

The authored seed set currently contains 88 examples. The pipeline validates
IDs, feedback types, features, numeric values, target actions, and grounding
sources before learning. Probe reports include zero-weight, initial-weight,
learned-weight, and leave-one-probe-out results. Tied action scores are failures
rather than being resolved by JSON action order.

## Run

From the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_probe_states.py
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/run_baseline_b_pipeline.py
```

If the environment is already active, the scripts can also be run with plain `python`.

## Tests

```powershell
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python -B -m unittest discover `
  -s baselines/baseline_b_linguistic_feedback/adapted_overcooked/tests `
  -v
```

## Convert an attributed session

First produce normalized trajectory, feedback, candidate-event, and attribution
JSONL files:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -B -m durf.feedback_attribution.demo_offline_attribution `
  --session outputs/human_ai_sessions/<session_id>
```

Then convert attributed windows to provenance-rich Baseline B examples:

```powershell
python -B baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/convert_session_feedback.py `
  --session outputs/human_ai_sessions/<session_id>
```

The default output is
`baseline_b_feedback_examples.json` in the session directory. Ambiguous
attributions are excluded unless `--include-ambiguous` is passed. Converted
records retain the session, timestamp, step, layout, source, attribution
window, and confidence. They can be passed to the normal pipeline with
`--feedback`.

Current ring-layout session logs contain no human language feedback, so a ring
conversion can legitimately produce zero examples. This is a data-collection
gap, not a conversion failure.
