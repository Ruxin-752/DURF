# Baseline B Linguistic Feedback

This folder keeps two related code areas:

- `original_rewards_repo/`: reference source code from *Learning Rewards from Linguistic Feedback*. It is kept as a historical algorithm reference and is not imported directly by the Overcooked pipeline.
- `adapted_overcooked/`: the runnable DURF Overcooked adaptation. This is the active implementation for the `ring_tomato_onion_10x6` probe-state baseline.

Start with:

```powershell
cd /d E:\Research\DURF
$env:PYTHONPATH="$PWD;$PWD\src"
uv run --python "E:/miniconda/envs/pantheonrl_env/python.exe" python baselines/baseline_b_linguistic_feedback/adapted_overcooked/scripts/run_baseline_b_pipeline.py
```
