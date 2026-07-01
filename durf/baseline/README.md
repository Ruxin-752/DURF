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

To continue improving the installed `ring_tomato_onion_10x6` agent instead of
restarting from scratch, use the stronger ring-map profile:

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.baseline.train_rllib_agent `
  --layout ring_tomato_onion_10x6 `
  --agent-name RllibRingTomatoOnion10x6SP `
  --profile paper-ring `
  --resume-installed `
  --iterations 300 `
  --num-workers 0 `
  --ray-local-mode `
  --overwrite
```

Then evaluate the installed checkpoint with:

```powershell
python -m durf.baseline.evaluate_baseline `
  --agent RllibRingTomatoOnion10x6SP `
  --layout ring_tomato_onion_10x6 `
  --episodes 10 `
  --seed 200 `
  --output outputs\baseline_eval_ring_latest.json
```

If the sparse reward stays near zero on the target map, use curriculum training
so the policy first learns easier versions of the same 10x6 task before moving
back to the mixed tomato/onion order:

```powershell
conda activate durf310
cd "C:\Users\my185\Desktop\研究\durf\DURF"
.\scripts\train_ring_curriculum.ps1
```

The curriculum stages are now deliberately fine-grained. Each stage evaluates
the installed checkpoint before moving on, so a failed transfer is visible
early instead of after a long run:

0. `ring_tomato_onion_10x6_curriculum_micro`: same 10x6 observation size, tiny
   teaching kitchen, tomato-only soup, with temporary tomato pickup shaping.
1. `ring_tomato_onion_10x6_curriculum_micro_delivery`: tiny teaching kitchen
   with dish/server close to the pot, so the policy can learn the full
   potting-to-serving chain.
2. `ring_tomato_onion_10x6_curriculum_open_delivery`: bridge map where tomato
   moves away from the pot, while dish/server stay close enough to preserve
   the serving skill.
3. `ring_tomato_onion_10x6_curriculum_open`: open kitchen with the tomato
   dispenser close enough that the micro policy has a gentler transfer.
4. `ring_tomato_onion_10x6_curriculum_easy`: same 10x6 observation size, more
   open kitchen, tomato dispenser farther away.
5. `ring_tomato_onion_10x6_curriculum_corridor`: partial ring/corridor
   geometry, still tomato-only.
6. `ring_tomato_onion_10x6`: target ring geometry, target
   tomato-tomato-onion soup.

`ring_tomato_onion_10x6_curriculum_tomato` is kept as an older tomato-only
target-geometry diagnostic map, but it is no longer in the default curriculum.

The intended learning path is:

1. learn tomato pickup/potting;
2. learn dish/soup serving;
3. preserve those skills while increasing distance and corridor structure;
4. add onion only after the tomato-only chain survives transfer.

If you want a tomato-only target-geometry diagnostic after stage 4, run:

```powershell
python -m durf.baseline.evaluate_baseline `
  --agent RllibRingTomatoOnion10x6SP `
  --layout ring_tomato_onion_10x6_curriculum_tomato `
  --episodes 10 `
  --seed 200 `
  --output outputs\baseline_eval_ring_tomato_diag.json
```

For a shorter smoke test, pass smaller iteration counts:

```powershell
.\scripts\train_ring_curriculum.ps1 `
  -Stage0Iterations 20 `
  -Stage1Iterations 20 `
  -Stage2Iterations 20 `
  -Stage3Iterations 20 `
  -Stage4Iterations 20 `
  -Stage5Iterations 20 `
  -EvalEpisodes 3
```

If a nested PowerShell loses the conda environment, pass the Python executable
explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_ring_curriculum.ps1 `
  -PythonExe "C:\Users\my185\Miniconda3\envs\durf310\python.exe"
```
