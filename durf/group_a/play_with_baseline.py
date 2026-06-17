"""Play Overcooked as the green agent beside the archived blue RLlib PPO."""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pygame
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from durf.baseline.runtime import (
    DEFAULT_AGENT_NAME,
    DEFAULT_PLAYABLE_LAYOUTS,
    REPO_ROOT,
    ensure_agent_layout,
    load_rllib_agent,
    make_direct_multi_env,
    resolve_agent_dir,
    rllib_action_index,
)


STAY = 4
INTERACT = 5
ACTION_NAMES = ("north", "south", "east", "west", "stay", "interact")
PAUSE_KEYS = (pygame.K_p, pygame.K_TAB, pygame.K_F1)
PAUSE_BUTTON = pygame.Rect(790, 620, 130, 42)
PAUSE_DEBOUNCE_MS = 300
LAYOUT_SWITCH_DEBOUNCE_MS = 300
BUILD_ID = "pause-v4-fast-input-maps"
LAYOUT_NUMBER_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4)
WINDOWS_LAYOUT_NUMBER_KEYS = (0x31, 0x32, 0x33, 0x34)
MOTION_KEY_ACTIONS = {
    pygame.K_UP: 0,
    pygame.K_w: 0,
    pygame.K_DOWN: 1,
    pygame.K_s: 1,
    pygame.K_RIGHT: 2,
    pygame.K_d: 2,
    pygame.K_LEFT: 3,
    pygame.K_a: 3,
}
MOTION_SCANCODE_ACTIONS = {
    pygame.KSCAN_UP: 0,
    pygame.KSCAN_W: 0,
    pygame.KSCAN_DOWN: 1,
    pygame.KSCAN_S: 1,
    pygame.KSCAN_RIGHT: 2,
    pygame.KSCAN_D: 2,
    pygame.KSCAN_LEFT: 3,
    pygame.KSCAN_A: 3,
}
MOTION_CHAR_ACTIONS = {
    "w": 0,
    "s": 1,
    "d": 2,
    "a": 3,
}
WINDOWS_VK_ACTIONS = {
    0x26: 0,  # Up
    0x57: 0,  # W
    0x28: 1,  # Down
    0x53: 1,  # S
    0x27: 2,  # Right
    0x44: 2,  # D
    0x25: 3,  # Left
    0x41: 3,  # A
}
WINDOWS_VK_NAMES = {
    0x09: "Tab",
    0x20: "Space",
    0x25: "Left",
    0x26: "Up",
    0x27: "Right",
    0x28: "Down",
    0x41: "A",
    0x44: "D",
    0x46: "F",
    0x4A: "J",
    0x4B: "K",
    0x4D: "M",
    0x4E: "N",
    0x50: "P",
    0x52: "R",
    0x53: "S",
    0x57: "W",
    0x31: "1",
    0x32: "2",
    0x33: "3",
    0x34: "4",
    0x70: "F1",
}
WINDOWS_WATCHED_KEYS = set(WINDOWS_VK_NAMES) | set(WINDOWS_VK_ACTIONS)
WINDOWS_KEYBOARD_AVAILABLE = os.name == "nt"
if WINDOWS_KEYBOARD_AVAILABLE:
    _get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
else:
    _get_async_key_state = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT_NAME,
        help="RLlib agent name or path; defaults to RllibCrampedRoomSP",
    )
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=list(DEFAULT_PLAYABLE_LAYOUTS),
        help="Layouts available for hot switching with 1-4 or N/M.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--step-hz",
        type=float,
        default=8.0,
        help="Environment decisions per second. Higher values reduce input latency.",
    )
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "human_ai_sessions"),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Exit after this many total steps; useful for smoke tests.",
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_session_dir(output_dir: str | Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(output_dir).resolve() / run_id
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def event_motion_action(event) -> int | None:
    if event.key in MOTION_KEY_ACTIONS:
        return MOTION_KEY_ACTIONS[event.key]
    scancode = getattr(event, "scancode", None)
    if scancode in MOTION_SCANCODE_ACTIONS:
        return MOTION_SCANCODE_ACTIONS[scancode]
    unicode_key = getattr(event, "unicode", "").lower()
    if unicode_key in MOTION_CHAR_ACTIONS:
        return MOTION_CHAR_ACTIONS[unicode_key]
    return None


def human_motion_action(held_motion_actions: set[int]) -> int:
    if 0 in held_motion_actions:
        return 0
    if 1 in held_motion_actions:
        return 1
    if 2 in held_motion_actions:
        return 2
    if 3 in held_motion_actions:
        return 3
    return STAY


def windows_pressed_keys() -> set[int]:
    if not WINDOWS_KEYBOARD_AVAILABLE:
        return set()
    return {
        vk
        for vk in WINDOWS_WATCHED_KEYS
        if _get_async_key_state(vk) & 0x8000
    }


def windows_motion_actions(pressed_keys: set[int]) -> set[int]:
    return {
        action
        for vk, action in WINDOWS_VK_ACTIONS.items()
        if vk in pressed_keys
    }


def windows_key_name(vk: int) -> str:
    return WINDOWS_VK_NAMES.get(vk, f"VK{vk}")


def describe_key_event(event) -> str:
    key_name = pygame.key.name(event.key) if event.key is not None else "?"
    return (
        f"key={key_name} code={event.key} "
        f"scan={getattr(event, 'scancode', '?')} "
        f"char={getattr(event, 'unicode', '')!r}"
    )


def write_row(writer: csv.DictWriter, handle, row: dict, flush: bool = True) -> None:
    writer.writerow(row)
    if flush:
        handle.flush()


def write_crash_log(exc: BaseException) -> Path:
    crash_dir = REPO_ROOT / "outputs" / "crash_logs"
    crash_dir.mkdir(parents=True, exist_ok=True)
    crash_path = crash_dir / f"play_with_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    crash_path.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )
    return crash_path


def render_game_surface(visualizer, env, episode_reward):
    state = env.base_env.state
    hud_data = StateVisualizer.default_hud_data(state, score=episode_reward)
    surface = visualizer.render_state(
        state=state,
        hud_data=hud_data,
        grid=env.base_env.mdp.terrain_mtx,
    )
    scale = min(900 / surface.get_width(), 590 / surface.get_height(), 1.0)
    if scale < 1:
        surface = pygame.transform.smoothscale(
            surface,
            (
                int(surface.get_width() * scale),
                int(surface.get_height() * scale),
            ),
        )
    return surface


def main() -> int:
    args = parse_args()
    if args.step_hz <= 0:
        raise ValueError("--step-hz must be greater than zero")
    if args.render_fps <= 0:
        raise ValueError("--render-fps must be greater than zero")
    if args.start_delay < 0:
        raise ValueError("--start-delay cannot be negative")

    layouts = list(dict.fromkeys([args.layout, *args.layouts]))
    for layout_name in layouts:
        ensure_agent_layout(args.agent, layout_name)
    layout_index = layouts.index(args.layout)
    current_layout = args.layout

    agent_dir = resolve_agent_dir(args.agent)
    ai_agent = load_rllib_agent(args.agent, agent_index=0)
    env = make_direct_multi_env(current_layout, args.seed)
    envs_by_layout = {current_layout: env}
    session_dir = new_session_dir(args.output_dir)

    trajectory_path = session_dir / "trajectory.csv"
    feedback_path = session_dir / "feedback.csv"
    trajectory_handle = trajectory_path.open("w", newline="", encoding="utf-8")
    feedback_handle = feedback_path.open("w", newline="", encoding="utf-8")
    trajectory_fields = [
        "timestamp_utc",
        "episode",
        "episode_step",
        "total_step",
        "layout",
        "ai_action",
        "ai_action_name",
        "human_action",
        "human_action_name",
        "environment_reward",
        "episode_reward",
        "predict_ms",
        "environment_step_ms",
        "done",
    ]
    feedback_fields = [
        "timestamp_utc",
        "episode",
        "episode_step",
        "total_step",
        "feedback",
        "last_ai_action",
        "last_ai_action_name",
    ]
    pause_path = session_dir / "pause_events.csv"
    pause_handle = pause_path.open("w", newline="", encoding="utf-8")
    pause_fields = [
        "timestamp_utc",
        "event",
        "episode",
        "episode_step",
        "total_step",
        "layout",
        "pygame_ticks",
    ]
    trajectory_writer = csv.DictWriter(trajectory_handle, fieldnames=trajectory_fields)
    feedback_writer = csv.DictWriter(feedback_handle, fieldnames=feedback_fields)
    pause_writer = csv.DictWriter(pause_handle, fieldnames=pause_fields)
    trajectory_writer.writeheader()
    feedback_writer.writeheader()
    pause_writer.writeheader()

    pygame.init()
    pygame.key.set_repeat()
    screen = pygame.display.set_mode((940, 740))
    pygame.display.set_caption(f"DURF Human + PPO Baseline [{BUILD_ID}]")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)
    small_font = pygame.font.Font(None, 22)
    visualizer = StateVisualizer()

    ai_obs, _ = env.multi_reset()
    episode = 1
    episode_step = 0
    total_step = 0
    episode_reward = 0.0
    paused = False
    running = True
    quit_reason = "unknown"
    pending_interact = False
    pending_motion = None
    held_motion_actions: set[int] = set()
    previous_windows_keys: set[int] = set()
    last_key_debug = "no key yet"
    last_ai_action = STAY
    last_predict_ms = 0.0
    last_environment_step_ms = 0.0
    feedback_count = {1: 0, -1: 0}
    feedback_flash = ""
    feedback_flash_frames = 0
    step_interval_ms = max(1, round(1000 / args.step_hz))
    countdown_until = pygame.time.get_ticks() + round(args.start_delay * 1000)
    next_step_at = countdown_until
    last_pause_toggle_at = -PAUSE_DEBOUNCE_MS
    last_layout_switch_at = -LAYOUT_SWITCH_DEBOUNCE_MS
    game_surface = render_game_surface(visualizer, env, episode_reward)

    def reset_episode(new_layout_index: int | None = None) -> None:
        nonlocal ai_obs
        nonlocal countdown_until
        nonlocal current_layout
        nonlocal env
        nonlocal episode
        nonlocal episode_reward
        nonlocal episode_step
        nonlocal game_surface
        nonlocal held_motion_actions
        nonlocal layout_index
        nonlocal next_step_at
        nonlocal pending_interact
        nonlocal pending_motion

        if new_layout_index is not None and new_layout_index != layout_index:
            layout_index = new_layout_index
            current_layout = layouts[layout_index]
            if current_layout not in envs_by_layout:
                envs_by_layout[current_layout] = make_direct_multi_env(
                    current_layout,
                    args.seed,
                )
            env = envs_by_layout[current_layout]

        ai_obs, _ = env.multi_reset()
        ai_agent.reset()
        episode += 1
        episode_step = 0
        episode_reward = 0.0
        pending_interact = False
        pending_motion = None
        held_motion_actions.clear()
        countdown_until = pygame.time.get_ticks() + round(args.start_delay * 1000)
        next_step_at = countdown_until
        game_surface = render_game_surface(visualizer, env, episode_reward)

    def request_layout_switch(new_layout_index: int) -> bool:
        nonlocal last_layout_switch_at
        now = pygame.time.get_ticks()
        if not 0 <= new_layout_index < len(layouts):
            return False
        if now - last_layout_switch_at < LAYOUT_SWITCH_DEBOUNCE_MS:
            return False
        last_layout_switch_at = now
        reset_episode(new_layout_index)
        return True

    print(f"Agent: {agent_dir}")
    print(f"Interface build: {BUILD_ID}")
    print(f"Script: {Path(__file__).resolve()}")
    print(f"Session logs: {session_dir}")
    print("Human is green; PPO is blue.")
    print(
        f"Timing: {args.step_hz:g} environment steps/s, "
        f"{args.render_fps} render FPS, {args.start_delay:g}s start delay"
    )
    print(
        "Controls: WASD/Arrows=Move, Space=Interact, J=+1, K=-1, "
        "P/Tab/F1=Pause, R=Reset, 1-4/N/M=Switch map, Esc=Quit"
    )

    try:
        while running:
            pause_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_reason = "window close button"
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit_reason = "pygame Esc"
                        running = False
                    elif (
                        event.key in PAUSE_KEYS
                        or getattr(event, "unicode", "").lower() == "p"
                    ):
                        pause_requested = True
                    elif event.key == pygame.K_SPACE and not paused:
                        pending_interact = True
                        last_key_debug = f"{describe_key_event(event)} -> interact"
                    elif event.key in LAYOUT_NUMBER_KEYS:
                        requested_index = LAYOUT_NUMBER_KEYS.index(event.key)
                        if request_layout_switch(requested_index):
                            last_key_debug = f"map -> {current_layout}"
                    elif event.key == pygame.K_n:
                        if request_layout_switch((layout_index + 1) % len(layouts)):
                            last_key_debug = f"next map -> {current_layout}"
                    elif event.key == pygame.K_m:
                        if request_layout_switch((layout_index - 1) % len(layouts)):
                            last_key_debug = f"previous map -> {current_layout}"
                    else:
                        motion_action = event_motion_action(event)
                        if motion_action is not None:
                            last_key_debug = (
                                f"{describe_key_event(event)} -> "
                                f"{ACTION_NAMES[motion_action]}"
                            )
                            if not paused:
                                held_motion_actions.add(motion_action)
                                pending_motion = motion_action
                        elif event.key == pygame.K_r:
                            reset_episode()
                        elif event.key in (pygame.K_j, pygame.K_k):
                            feedback = 1 if event.key == pygame.K_j else -1
                            feedback_count[feedback] += 1
                            write_row(
                                feedback_writer,
                                feedback_handle,
                                {
                                    "timestamp_utc": utc_timestamp(),
                                    "episode": episode,
                                    "episode_step": episode_step,
                                    "total_step": total_step,
                                    "feedback": feedback,
                                    "last_ai_action": last_ai_action,
                                    "last_ai_action_name": ACTION_NAMES[last_ai_action],
                                },
                            )
                            feedback_flash = f"Recorded feedback: {feedback:+d}"
                            feedback_flash_frames = args.render_fps
                elif event.type == pygame.KEYUP:
                    motion_action = event_motion_action(event)
                    if motion_action is not None:
                        held_motion_actions.discard(motion_action)
                        last_key_debug = (
                            f"released {describe_key_event(event)} -> "
                            f"{ACTION_NAMES[motion_action]}"
                        )
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and PAUSE_BUTTON.collidepoint(event.pos)
                ):
                    pause_requested = True

            windows_keys = windows_pressed_keys()
            windows_pressed_now = windows_keys - previous_windows_keys
            previous_windows_keys = windows_keys
            windows_new_actions = windows_motion_actions(windows_pressed_now)
            if windows_new_actions and not paused:
                pending_motion = human_motion_action(windows_new_actions)
                last_key_debug = (
                    f"windows press "
                    f"{'+'.join(windows_key_name(vk) for vk in sorted(windows_pressed_now & set(WINDOWS_VK_ACTIONS)))}"
                    f" -> {ACTION_NAMES[pending_motion]}"
                )
            if 0x20 in windows_pressed_now and not paused:  # Space
                pending_interact = True
                last_key_debug = "windows Space -> interact"
            if windows_pressed_now & {0x50, 0x09, 0x70}:  # P, Tab, F1
                pause_requested = True
                last_key_debug = "windows pause key"
            for key_index, vk in enumerate(WINDOWS_LAYOUT_NUMBER_KEYS):
                if vk in windows_pressed_now and request_layout_switch(key_index):
                    last_key_debug = f"windows map -> {current_layout}"
            if 0x4E in windows_pressed_now:  # N
                if request_layout_switch((layout_index + 1) % len(layouts)):
                    last_key_debug = f"windows next map -> {current_layout}"
            if 0x4D in windows_pressed_now:  # M
                if request_layout_switch((layout_index - 1) % len(layouts)):
                    last_key_debug = f"windows previous map -> {current_layout}"
            if 0x52 in windows_pressed_now:  # R
                reset_episode()
                last_key_debug = "windows R -> reset"
            if windows_pressed_now & {0x4A, 0x4B}:  # J, K
                feedback = 1 if 0x4A in windows_pressed_now else -1
                feedback_count[feedback] += 1
                write_row(
                    feedback_writer,
                    feedback_handle,
                    {
                        "timestamp_utc": utc_timestamp(),
                        "episode": episode,
                        "episode_step": episode_step,
                        "total_step": total_step,
                        "feedback": feedback,
                        "last_ai_action": last_ai_action,
                        "last_ai_action_name": ACTION_NAMES[last_ai_action],
                    },
                )
                feedback_flash = f"Recorded feedback: {feedback:+d}"
                feedback_flash_frames = args.render_fps
                last_key_debug = f"windows {'J' if feedback > 0 else 'K'} -> feedback"

            now = pygame.time.get_ticks()
            if (
                pause_requested
                and now - last_pause_toggle_at >= PAUSE_DEBOUNCE_MS
            ):
                paused = not paused
                last_pause_toggle_at = now
                pending_interact = False
                pending_motion = None
                held_motion_actions.clear()
                if not paused:
                    next_step_at = now + step_interval_ms
                    countdown_until = 0
                pause_event = "paused" if paused else "resumed"
                write_row(
                    pause_writer,
                    pause_handle,
                    {
                        "timestamp_utc": utc_timestamp(),
                        "event": pause_event,
                        "episode": episode,
                        "episode_step": episode_step,
                        "total_step": total_step,
                        "layout": current_layout,
                        "pygame_ticks": now,
                    },
                )
                print(
                    f"{pause_event.upper()} at episode={episode}, "
                    f"step={episode_step}, total_step={total_step}"
                )

            if not paused and running and now >= next_step_at:
                if pending_interact:
                    human_action = INTERACT
                elif pending_motion is not None:
                    human_action = pending_motion
                else:
                    human_action = STAY
                pending_interact = False
                pending_motion = None
                predict_started = time.perf_counter()
                ai_action_raw = rllib_action_index(ai_agent, env.base_env.state)
                last_predict_ms = (time.perf_counter() - predict_started) * 1000
                ai_action = int(ai_action_raw)
                step_started = time.perf_counter()
                (ai_obs, _), (reward, _), done, _ = env.multi_step(
                    ai_action,
                    human_action,
                )
                last_environment_step_ms = (
                    time.perf_counter() - step_started
                ) * 1000
                last_ai_action = ai_action
                episode_step += 1
                total_step += 1
                episode_reward += float(reward)
                write_row(
                    trajectory_writer,
                    trajectory_handle,
                    {
                        "timestamp_utc": utc_timestamp(),
                        "episode": episode,
                        "episode_step": episode_step,
                        "total_step": total_step,
                        "layout": current_layout,
                        "ai_action": ai_action,
                        "ai_action_name": ACTION_NAMES[ai_action],
                        "human_action": human_action,
                        "human_action_name": ACTION_NAMES[human_action],
                        "environment_reward": float(reward),
                        "episode_reward": episode_reward,
                        "predict_ms": round(last_predict_ms, 3),
                        "environment_step_ms": round(last_environment_step_ms, 3),
                        "done": bool(done),
                    },
                    flush=total_step % 10 == 0 or bool(done),
                )
                game_surface = render_game_surface(
                    visualizer,
                    env,
                    episode_reward,
                )
                # Schedule from the completed step so a slow frame never causes catch-up.
                next_step_at = pygame.time.get_ticks() + step_interval_ms

                if args.max_steps is not None and total_step >= args.max_steps:
                    quit_reason = "max steps reached"
                    running = False
                elif done:
                    print(
                        f"Episode {episode}: reward={episode_reward:.1f}, "
                        f"steps={episode_step}"
                    )
                    reset_episode()

            screen.fill((28, 30, 34))
            screen.blit(game_surface, game_surface.get_rect(center=(470, 305)))
            button_color = (100, 145, 105) if paused else (150, 105, 70)
            pygame.draw.rect(screen, button_color, PAUSE_BUTTON, border_radius=7)
            button_text = small_font.render(
                "RESUME" if paused else "PAUSE",
                True,
                (255, 255, 255),
            )
            screen.blit(button_text, button_text.get_rect(center=PAUSE_BUTTON.center))
            if paused:
                overlay = pygame.Surface((940, 610), pygame.SRCALPHA)
                overlay.fill((10, 12, 16, 150))
                screen.blit(overlay, (0, 0))
                pause_label = pygame.font.Font(None, 72).render(
                    "PAUSED",
                    True,
                    (255, 225, 120),
                )
                screen.blit(pause_label, pause_label.get_rect(center=(470, 280)))
                resume_label = font.render(
                    "Press P / Tab / F1 or click RESUME",
                    True,
                    (245, 245, 245),
                )
                screen.blit(
                    resume_label,
                    resume_label.get_rect(center=(470, 335)),
                )
            countdown_ms = max(0, countdown_until - pygame.time.get_ticks())
            if paused:
                run_status = "PAUSED"
            elif countdown_ms > 0:
                run_status = f"STARTING IN {countdown_ms / 1000:.1f}s"
            else:
                run_status = "RUNNING"
            status = (
                f"Blue: PPO | Green: YOU | Episode {episode} | Step {episode_step} | "
                f"Reward {episode_reward:.1f} | Map {layout_index + 1}/{len(layouts)}: "
                f"{current_layout} | {run_status}"
            )
            controls = (
                "WASD/Arrows Move | Space Interact | P/Tab/F1 Pause | "
                "R Reset | 1-4/N/M Map | Esc Quit"
            )
            feedback_status = (
                f"J +1 ({feedback_count[1]}) | K -1 ({feedback_count[-1]}) | "
                "LOGGING ONLY - MODEL IS NOT UPDATING"
            )
            timing_status = (
                f"{BUILD_ID} | Decision rate: {args.step_hz:g}/s | "
                f"AI predict: {last_predict_ms:.1f} ms | "
                f"Environment: {last_environment_step_ms:.1f} ms"
            )
            input_status = f"Last input: {last_key_debug}"
            screen.blit(font.render(status, True, (235, 235, 235)), (20, 630))
            screen.blit(small_font.render(timing_status, True, (165, 190, 235)), (20, 655))
            screen.blit(small_font.render(controls, True, (170, 210, 180)), (20, 680))
            screen.blit(
                small_font.render(feedback_status, True, (245, 200, 105)),
                (20, 705),
            )
            screen.blit(
                small_font.render(input_status, True, (180, 220, 245)),
                (470, 705),
            )
            if feedback_flash_frames > 0:
                screen.blit(
                    font.render(feedback_flash, True, (120, 225, 135)),
                    (690, 655),
                )
                feedback_flash_frames -= 1
            pygame.display.flip()
            clock.tick(args.render_fps)
    finally:
        trajectory_handle.flush()
        feedback_handle.flush()
        pause_handle.flush()
        trajectory_handle.close()
        feedback_handle.close()
        pause_handle.close()
        for cached_env in set(envs_by_layout.values()):
            cached_env.close()
        pygame.quit()

    print(f"Trajectory: {trajectory_path}")
    print(f"Feedback: {feedback_path}")
    print(f"Pause events: {pause_path}")
    print(f"Exit reason: {quit_reason}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        crash_path = write_crash_log(exc)
        print(f"CRASH: {exc}", file=sys.stderr)
        print(f"Crash log: {crash_path}", file=sys.stderr)
        raise
