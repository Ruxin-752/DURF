# Archived SB3 PPO Baseline

This module evaluates the 1M-step no-feedback model preserved under
`archive_local/`.

The original run used:

- PantheonRL `OvercookedMultiEnv-v0`
- Stable-Baselines3 PPO 2.8.0
- `cramped_room`
- a newly initialized PPO partner policy
- 1,001,472 recorded timesteps

Run commands from the repository root with the existing environment:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
conda run -n pantheonrl_env python -m durf.baseline.watch_baseline
```

Fixed-seed evaluation:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
conda run -n pantheonrl_env python -m durf.baseline.evaluate_baseline `
  --episodes 20 `
  --output outputs\baseline_evaluation.json
```

The model should only be promoted to the official baseline after both visual
inspection and quantitative evaluation.
