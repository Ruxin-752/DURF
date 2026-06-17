# Archived RLlib PPO Baseline

This module runs the trained RLlib PPO agents restored from commit
`b1e6c627`. The default pygame baseline uses:

```text
models/rllib_agents/RllibCrampedRoomSP
```

Use it only with `cramped_room`; other restored agents are map-specific assets.

Run commands from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.watch_baseline --layout cramped_room
```

Fixed-seed evaluation:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.evaluate_baseline `
  --episodes 20 `
  --output outputs\baseline_evaluation.json
```

The viewer keeps the original pygame pause/reset controls:

```text
Space = pause or resume
R     = reset
Q/Esc = quit
```
