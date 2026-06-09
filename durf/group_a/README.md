# Group A: Scalar Feedback

## Verified component

`keyboard_listener.py` is the currently trusted component.

Run from the repository root:

```powershell
python durf/group_a/keyboard_listener.py
```

Controls:

- `J`: write `+1`
- `K`: write `-1`
- `Space`: write `0`
- `Q`: exit

Use `--terminal-only` if global keyboard listening is unavailable:

```powershell
python durf/group_a/keyboard_listener.py --terminal-only
```

By default, runtime output is written to `reward_signal.txt` and
`feedback_log.csv` in the current directory. Both are ignored by Git.

## Not yet verified

The keyboard signal has not yet been connected to a proven online PPO training
loop. A valid end-to-end implementation must demonstrate that:

1. feedback changes the reward consumed by PPO;
2. PPO actually performs parameter updates;
3. training and evaluation logs record the code version and configuration;
4. pause behavior is identical across experimental groups.

# Human + PPO collaboration

Run the archived 1M-step PPO as the blue agent and control the green agent:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
conda run -n pantheonrl_env python -m durf.group_a.play_with_baseline
```

Controls:

- `WASD` or arrow keys: move the green human agent
- `Space`: interact
- `J`: record positive feedback (`+1`)
- `K`: record negative feedback (`-1`)
- `P`: pause or resume
- `R`: reset the episode
- `Q` or `Esc`: quit

Session data is written to `outputs/human_ai_sessions/<timestamp>/`. The J/K
signals are logged for inspection but do not update PPO yet.

The default interaction rate is two environment steps per second with a
three-second preparation countdown. To slow it further:

```powershell
conda run -n pantheonrl_env python -m durf.group_a.play_with_baseline `
  --step-hz 1.5 --start-delay 5
```
