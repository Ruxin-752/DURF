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

## Training a New Map Agent

The new map `ring_tomato_onion_10x6` is present locally, but its PPO agent is
not committed yet. Train a local self-play PPO agent with:

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.train_rllib_agent `
  --layout ring_tomato_onion_10x6 `
  --agent-name RllibRingTomatoOnion10x6SP `
  --iterations 2 `
  --num-workers 0 `
  --ray-local-mode `
  --overwrite
```

This installs the latest checkpoint under:

```text
models/rllib_agents/RllibRingTomatoOnion10x6SP/agent
```

After a smoke run works, increase `--iterations` for a stronger policy.
