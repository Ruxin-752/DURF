"""Play Overcooked as the green agent beside the archived blue PPO baseline."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pygame
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3 import PPO

from durf.baseline.runtime import (
    REPO_ROOT,
    make_direct_multi_env,
    resolve_model_path,
)


STAY = 4
INTERACT = 5
ACTION_NAMES = ("north", "south", "east", "west", "stay", "interact")
PAUSE_KEYS = (pygame.K_p, pygame.K_TAB, pygame.K_F1)
PAUSE_BUTTON = pygame.Rect(790, 620, 130, 42)
PAUSE_DEBOUNCE_MS = 300
BUILD_ID = "pause-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="SB3 PPO .zip; defaults to the archived 1M model")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--step-hz",
        type=float,
        default=2.0,
        help="Environment decisions per second. Lower values give humans more time.",
    )
    parser.add_argument("--render-fps", type=int, default=30)
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
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_session_dir(output_dir: str | Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(output_dir).resolve() / run_id
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def human_motion_action() -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 0
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return 1
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 2
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 3
    return STAY


def write_row(writer: csv.DictWriter, handle, row: dict, flush: bool = True) -> None:
    writer.writerow(row)
    if flush:
        handle.flush()


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

    model_path = resolve_model_path(args.model)
    model = PPO.load(model_path, device="cpu")
    env = make_direct_multi_env(args.layout, args.seed)
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
    pending_interact = False
    pending_motion = None
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
    game_surface = render_game_surface(visualizer, env, episode_reward)

    print(f"Model: {model_path}")
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
        "P/Tab/F1=Pause, R=Reset"
    )

    try:
        while running:
            pause_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif (
                        event.key in PAUSE_KEYS
                        or getattr(event, "unicode", "").lower() == "p"
                    ):
                        pause_requested = True
                    elif event.key == pygame.K_SPACE and not paused:
                        pending_interact = True
                    elif not paused and event.key in (pygame.K_UP, pygame.K_w):
                        pending_motion = 0
                    elif not paused and event.key in (pygame.K_DOWN, pygame.K_s):
                        pending_motion = 1
                    elif not paused and event.key in (pygame.K_RIGHT, pygame.K_d):
                        pending_motion = 2
                    elif not paused and event.key in (pygame.K_LEFT, pygame.K_a):
                        pending_motion = 3
                    elif event.key == pygame.K_r:
                        ai_obs, _ = env.multi_reset()
                        episode += 1
                        episode_step = 0
                        episode_reward = 0.0
                        pending_interact = False
                        pending_motion = None
                        countdown_until = (
                            pygame.time.get_ticks() + round(args.start_delay * 1000)
                        )
                        next_step_at = countdown_until
                        game_surface = render_game_surface(
                            visualizer,
                            env,
                            episode_reward,
                        )
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
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and PAUSE_BUTTON.collidepoint(event.pos)
                ):
                    pause_requested = True

            now = pygame.time.get_ticks()
            if (
                pause_requested
                and now - last_pause_toggle_at >= PAUSE_DEBOUNCE_MS
            ):
                paused = not paused
                last_pause_toggle_at = now
                pending_interact = False
                pending_motion = None
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
                    human_action = human_motion_action()
                pending_interact = False
                pending_motion = None
                predict_started = time.perf_counter()
                ai_action_raw, _ = model.predict(
                    ai_obs,
                    deterministic=args.deterministic,
                )
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
                    running = False
                elif done:
                    print(
                        f"Episode {episode}: reward={episode_reward:.1f}, "
                        f"steps={episode_step}"
                    )
                    ai_obs, _ = env.multi_reset()
                    episode += 1
                    episode_step = 0
                    episode_reward = 0.0
                    countdown_until = (
                        pygame.time.get_ticks() + round(args.start_delay * 1000)
                    )
                    next_step_at = countdown_until
                    game_surface = render_game_surface(
                        visualizer,
                        env,
                        episode_reward,
                    )

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
                f"Reward {episode_reward:.1f} | {run_status}"
            )
            controls = (
                "WASD/Arrows Move | Space Interact | P/Tab/F1 Pause | "
                "R Reset | Q/Esc Quit"
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
            screen.blit(font.render(status, True, (235, 235, 235)), (20, 630))
            screen.blit(small_font.render(timing_status, True, (165, 190, 235)), (20, 655))
            screen.blit(small_font.render(controls, True, (170, 210, 180)), (20, 680))
            screen.blit(
                small_font.render(feedback_status, True, (245, 200, 105)),
                (20, 705),
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
        env.close()
        pygame.quit()

    print(f"Trajectory: {trajectory_path}")
    print(f"Feedback: {feedback_path}")
    print(f"Pause events: {pause_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
