"""
Human-feedback 可视化采集入口。

默认模式下，玩家用 WASD/方向键/空格控制 ego 厨师，另一个厨师由
PantheonRL partner AI 控制；Q/E/C 反馈会按 TAMER 训练需要写入日志。
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame
import yaml
from stable_baselines3 import PPO
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from src.env.factory import make_overcooked_env
from src.feedback.keyboard_listener import feedback_event_from_key

WINDOW_SIZE = (900, 680)
STAY_ACTION = 4
INTERACT_ACTION = 5
ACTION_NAMES = ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"]
ACTION_SYMBOLS = ["↑", "↓", "→", "←", "stay", "interact"]
SUPPORTED_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
]

PLAYER_ACTION_KEYS = {
    pygame.K_w: 0,
    pygame.K_UP: 0,
    pygame.K_s: 1,
    pygame.K_DOWN: 1,
    pygame.K_d: 2,
    pygame.K_RIGHT: 2,
    pygame.K_a: 3,
    pygame.K_LEFT: 3,
    pygame.K_SPACE: INTERACT_ACTION,
}

PLAYER_HOLD_PRIORITY = [
    pygame.K_SPACE,
    pygame.K_w,
    pygame.K_UP,
    pygame.K_s,
    pygame.K_DOWN,
    pygame.K_d,
    pygame.K_RIGHT,
    pygame.K_a,
    pygame.K_LEFT,
]

FEEDBACK_CSV_FIELDS = [
    "timestamp_ms",
    "participant_id",
    "layout_name",
    "global_step",
    "event_type",
    "signal_value",
    "source",
    "control_mode",
    "player_action",
    "paused",
]

TRAJECTORY_FIELDS = [
    "timestamp_iso",
    "monotonic_ms",
    "global_step",
    "episode_step",
    "episode_count",
    "participant_id",
    "layout_name",
    "control_mode",
    "action",
    "action_name",
    "overcooked_score",
    "env_reward",
    "tamer_feedback",
    "tamer_feedback_total",
    "training_reward",
    "done",
    "feedback_count",
    "observation",
]

PYNPUT_KEY_TO_PYGAME = {
    "w": pygame.K_w,
    "a": pygame.K_a,
    "s": pygame.K_s,
    "d": pygame.K_d,
    "up": pygame.K_UP,
    "down": pygame.K_DOWN,
    "left": pygame.K_LEFT,
    "right": pygame.K_RIGHT,
    "space": pygame.K_SPACE,
    "p": pygame.K_p,
    "q": pygame.K_q,
    "e": pygame.K_e,
    "c": pygame.K_c,
    "esc": pygame.K_ESCAPE,
}


def _unwrap_base_env(env):
    unwrapped = env
    while True:
        if hasattr(unwrapped, "base_env"):
            return unwrapped.base_env
        if not hasattr(unwrapped, "env"):
            raise RuntimeError("Could not unwrap Overcooked base_env")
        unwrapped = unwrapped.env


def _reset_env(env):
    obs = env.reset()
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _to_jsonable(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _current_player_action(held_keys: set[int]) -> int:
    for key in PLAYER_HOLD_PRIORITY:
        if key in held_keys:
            return PLAYER_ACTION_KEYS[key]
    return STAY_ACTION


def _pynput_key_name(key) -> str | None:
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    name = getattr(key, "name", None)
    if name:
        return name.lower()
    return None


def _start_global_keyboard_listener(input_queue: Queue):
    from pynput import keyboard

    def on_press(key):
        name = _pynput_key_name(key)
        if name in PYNPUT_KEY_TO_PYGAME:
            input_queue.put(("down", PYNPUT_KEY_TO_PYGAME[name], name))

    def on_release(key):
        name = _pynput_key_name(key)
        if name in PYNPUT_KEY_TO_PYGAME:
            input_queue.put(("up", PYNPUT_KEY_TO_PYGAME[name], name))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def _draw_scene(
    screen,
    font,
    viz,
    state,
    grid,
    overcooked_score: float,
    tamer_feedback_total: float,
    participant: str,
    step_count: int,
    total_timesteps: int,
    feedback_count: int,
    control_mode: str,
    action: int,
    paused: bool,
    focused: bool,
    last_key: str,
    last_event: str,
    input_source: str,
) -> None:
    hud_data = StateVisualizer.default_hud_data(state, score=overcooked_score)
    surface = viz.render_state(state=state, hud_data=hud_data, grid=grid)
    surf_rect = surface.get_rect()

    if surf_rect.width > WINDOW_SIZE[0] or surf_rect.height > WINDOW_SIZE[1]:
        scale = min(WINDOW_SIZE[0] / surf_rect.width, WINDOW_SIZE[1] / surf_rect.height) * 0.82
        surface = pygame.transform.scale(
            surface,
            (int(surf_rect.width * scale), int(surf_rect.height * scale)),
        )
        surf_rect = surface.get_rect()

    screen.fill((30, 30, 30))
    screen.blit(
        surface,
        (
            (WINDOW_SIZE[0] - surf_rect.width) // 2,
            (WINDOW_SIZE[1] - surf_rect.height) // 2 - 35,
        ),
    )

    info_lines = [
        f"Participant: {participant}",
        f"Mode: {control_mode}    Input: {input_source}    {'PAUSED' if paused else 'RUNNING'}",
        f"Pygame focus: {'YES' if focused else 'NO'}",
        f"Step: {step_count}/{total_timesteps}",
        f"Overcooked score: {overcooked_score:.1f}",
        f"TAMER feedback sum: {tamer_feedback_total:+.1f}    Feedback count: {feedback_count}",
        f"Player action: {ACTION_SYMBOLS[int(action)]}",
        f"Last key: {last_key or '-'}    Last event: {last_event or '-'}",
        "",
        "Keys are captured globally with pynput by default.",
        "WASD/Arrows = move    Space = interact    P = pause/resume",
        "Q = +1 good    E = -1 bad    C = coach pause    ESC = exit",
    ]
    for i, line in enumerate(info_lines):
        if "PAUSED" in line:
            color = (255, 220, 120)
        elif "focus: NO" in line:
            color = (255, 150, 150)
        elif "WASD" in line or "Q =" in line:
            color = (180, 220, 180)
        else:
            color = (220, 220, 220)
        text = font.render(line, True, color)
        screen.blit(text, (12, WINDOW_SIZE[1] - 28 - (len(info_lines) - i) * 22))

    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Group A visual TAMER data collection")
    parser.add_argument("--participant", default="P01")
    parser.add_argument(
        "--layout",
        choices=SUPPORTED_LAYOUTS,
        default=None,
        help="地图名；不传则使用 config/default.yaml 里的 env.layout_name",
    )
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--input-source",
        choices=["pynput", "pygame", "both"],
        default="pynput",
        help="pynput captures global keyboard input; pygame uses window key events",
    )
    parser.add_argument(
        "--control-mode",
        choices=["player", "agent"],
        default="player",
        help="player: WASD controls ego chef; agent: PPO controls ego chef",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    layout_name = args.layout or config["env"]["layout_name"]
    alpha = config["feedback"]["alpha"]
    device = config.get("device", "cpu")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "logs" / f"human-feedback_{layout_name}_{args.participant}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    feedback_csv_path = run_dir / "tamer_feedback.csv"
    trajectory_path = run_dir / "tamer_trajectory.jsonl"

    print(f"\n{'=' * 60}")
    print("  Group A Baseline - Visual TAMER Collection")
    print(f"  Participant : {args.participant}")
    print(f"  Layout      : {layout_name}")
    print(f"  Control     : {args.control_mode}")
    print(f"  Input       : {args.input_source}")
    print(f"  Steps       : {args.total_timesteps:,}")
    print(f"  Log dir     : {run_dir}")
    print(f"{'=' * 60}\n")

    feedback_queue = multiprocessing.Queue()
    env = make_overcooked_env(
        layout_name=layout_name,
        feedback_queue=feedback_queue,
        alpha=alpha,
        seed=args.seed,
    )
    base_env = _unwrap_base_env(env)
    grid = base_env.mdp.terrain_mtx

    model = None
    if args.control_mode == "agent":
        model = PPO(
            config["ppo"]["policy"],
            env,
            seed=args.seed,
            device=device,
            verbose=0,
            **{k: v for k, v in config["ppo"].items() if k != "policy"},
        )

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(f"Group A Baseline - {args.participant}")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    viz = StateVisualizer()

    print("Controls: WASD/Arrows=Move  Space=Interact  P=Pause  Q/E/C=Feedback  ESC=Exit\n")

    obs = _reset_env(env)
    session_start_ms = int(time.monotonic() * 1000)
    overcooked_score = 0.0
    tamer_feedback_total = 0.0
    step_count = 0
    episode_step = 0
    episode_count = 0
    feedback_count = 0
    running = True
    paused = False
    action = STAY_ACTION
    held_action_keys: set[int] = set()
    focused = pygame.key.get_focused()
    last_key = ""
    last_event = "started"
    input_queue: Queue = Queue()
    global_listener = None
    if args.input_source in ("pynput", "both"):
        global_listener = _start_global_keyboard_listener(input_queue)
        last_event = "pynput ready"

    with open(feedback_csv_path, "w", newline="", encoding="utf-8") as feedback_file, open(
        trajectory_path, "w", encoding="utf-8", buffering=1
    ) as trajectory_file:
        feedback_writer = csv.DictWriter(feedback_file, fieldnames=FEEDBACK_CSV_FIELDS)
        feedback_writer.writeheader()

        def handle_key_down(key: int, key_name: str, source: str) -> None:
            nonlocal action, feedback_count, last_event, last_key, paused, running
            last_key = key_name
            if key == pygame.K_ESCAPE:
                running = False
                last_event = f"{source} exit"
            elif key == pygame.K_p:
                paused = not paused
                last_event = f"{source} {'pause' if paused else 'resume'}"
                print(f"  [{'PAUSE' if paused else 'RESUME'}] Step {step_count}")
            elif key in PLAYER_ACTION_KEYS:
                held_action_keys.add(key)
                action = _current_player_action(held_action_keys)
                last_event = f"{source} action {ACTION_NAMES[int(action)]}"
                print(f"  [ACTION] Step {step_count}: {key_name} -> {ACTION_NAMES[int(action)]}")
            else:
                feedback = feedback_event_from_key(
                    key,
                    session_start_ms,
                    source=source,
                    metadata={
                        "global_step": step_count,
                        "episode_step": episode_step,
                        "control_mode": args.control_mode,
                        "player_action": int(action),
                        "paused": paused,
                    },
                )
                if feedback is not None:
                    feedback_event, label = feedback
                    feedback_queue.put(feedback_event)
                    feedback_count += 1
                    last_event = f"{source} feedback {feedback_event.event_type}"
                    feedback_writer.writerow(
                        {
                            "timestamp_ms": feedback_event.timestamp_ms,
                            "participant_id": args.participant,
                                    "layout_name": layout_name,
                            "global_step": step_count,
                            "event_type": feedback_event.event_type,
                            "signal_value": feedback_event.signal_value,
                            "source": feedback_event.source,
                            "control_mode": args.control_mode,
                            "player_action": int(action),
                            "paused": paused,
                        }
                    )
                    feedback_file.flush()
                    print(f"  [{label}] Step {step_count}")
                    if feedback_event.event_type == "coach_pause":
                        paused = True
                else:
                    last_event = f"{source} unmapped {key_name}"

        def handle_key_up(key: int, key_name: str, source: str) -> None:
            nonlocal action, last_event, last_key
            last_key = key_name
            if key in PLAYER_ACTION_KEYS:
                held_action_keys.discard(key)
                action = _current_player_action(held_action_keys)
                last_event = f"{source} keyup {key_name}"

        while running and step_count < args.total_timesteps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.WINDOWFOCUSGAINED:
                    focused = True
                    last_event = "focus gained"
                    print("  [FOCUS] Pygame window focused")
                elif event.type == pygame.WINDOWFOCUSLOST:
                    focused = False
                    held_action_keys.clear()
                    action = STAY_ACTION
                    last_event = "focus lost"
                    print("  [FOCUS] Pygame window lost focus")
                elif event.type == pygame.KEYUP and args.input_source in ("pygame", "both"):
                    handle_key_up(event.key, pygame.key.name(event.key), "pygame")
                elif event.type == pygame.KEYDOWN and args.input_source in ("pygame", "both"):
                    handle_key_down(event.key, pygame.key.name(event.key), "pygame")

            while True:
                try:
                    kind, key, key_name = input_queue.get_nowait()
                except Empty:
                    break
                if kind == "down":
                    handle_key_down(key, key_name, "pynput")
                else:
                    handle_key_up(key, key_name, "pynput")

            if args.control_mode == "player":
                action = _current_player_action(held_action_keys)

            if not paused:
                if args.control_mode == "agent":
                    action, _ = model.predict(obs, deterministic=False)
                    action = int(action)

                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result

                overcooked_reward = float(info.get("env_reward", reward))
                tamer_step_feedback = float(info.get("human_reward", 0.0))
                training_reward = float(info.get("total_reward", reward))
                overcooked_score += overcooked_reward
                tamer_feedback_total += tamer_step_feedback
                step_count += 1
                episode_step += 1

                record = {
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                    "monotonic_ms": int(time.monotonic() * 1000) - session_start_ms,
                    "global_step": step_count,
                    "episode_step": episode_step,
                    "episode_count": episode_count,
                    "participant_id": args.participant,
                    "layout_name": layout_name,
                    "control_mode": args.control_mode,
                    "action": int(action),
                    "action_name": ACTION_NAMES[int(action)],
                    "overcooked_score": overcooked_score,
                    "env_reward": overcooked_reward,
                    "tamer_feedback": tamer_step_feedback,
                    "tamer_feedback_total": tamer_feedback_total,
                    "training_reward": training_reward,
                    "done": bool(done),
                    "feedback_count": feedback_count,
                    "observation": _to_jsonable(obs),
                }
                trajectory_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if done:
                    print(
                        f"  Episode done! Episode: {episode_count}, "
                        f"Steps: {episode_step}, Overcooked score: {overcooked_score:.1f}, "
                        f"TAMER feedback: {tamer_feedback_total:+.1f}"
                    )
                    obs = _reset_env(env)
                    overcooked_score = 0.0
                    tamer_feedback_total = 0.0
                    episode_step = 0
                    episode_count += 1

            _draw_scene(
                screen=screen,
                font=font,
                viz=viz,
                state=base_env.state,
                grid=grid,
                overcooked_score=overcooked_score,
                tamer_feedback_total=tamer_feedback_total,
                participant=args.participant,
                step_count=step_count,
                total_timesteps=args.total_timesteps,
                feedback_count=feedback_count,
                control_mode=args.control_mode,
                action=action,
                paused=paused,
                focused=focused,
                last_key=last_key,
                last_event=last_event,
                input_source=args.input_source,
            )
            clock.tick(args.fps)

    if global_listener is not None:
        global_listener.stop()

    if model is not None:
        save_path = run_dir / "final_model"
        model.save(str(save_path))
        print(f"\nModel saved to: {save_path}.zip")

    print(f"\nFeedback log: {feedback_csv_path}")
    print(f"Trajectory log: {trajectory_path}")
    print(f"Total steps: {step_count}, Feedback given: {feedback_count}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
