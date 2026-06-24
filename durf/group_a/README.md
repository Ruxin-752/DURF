# Group A: Human + PPO Pygame Baseline

Run the archived RLlib PPO as the blue agent and control the green agent:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m durf.group_a.play_with_baseline --layout cramped_room
```

Controls:

- `WASD` or arrow keys: move the green human agent
- `Space`: interact
- `J`: record positive feedback (`+1`)
- `K`: record negative feedback (`-1`)
- `P`, `Tab`, `F1`, or the on-screen button: pause or resume
- `R`: reset the episode
- `Q` or `Esc`: quit

Session data is written to `outputs/human_ai_sessions/<timestamp>/`.

The J/K keys are currently retained as lightweight inspection logs only. They
do not update PPO and are not the main feedback interface for the new
LLM-assisted attribution study.

The default interaction rate is two environment steps per second with a
three-second preparation countdown. To slow it further:

```powershell
python -m durf.group_a.play_with_baseline `
  --step-hz 1.5 --start-delay 5
```

The older standalone scalar keyboard listener has been archived under
`archive/legacy_scalar_feedback/`.
