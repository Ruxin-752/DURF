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

## Run

From the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/evaluate_probe_states.py
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/run_baseline_b_pipeline.py
```

If the environment is already active, the scripts can also be run with plain `python`.
