"""Play Overcooked as the green agent beside the archived blue PPO baseline."""

from __future__ import annotations

import argparse
import csv
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="SB3 PPO .zip; defaults to the archived 1M model")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
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


def write_row(writer: csv.DictWriter, handle, row: dict) -> None:
    writer.writerow(row)
    handle.flush()


def main() -> int:
    args = parse_args()
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
    trajectory_writer = csv.DictWriter(trajectory_handle, fieldnames=trajectory_fields)
    feedback_writer = csv.DictWriter(feedback_handle, fieldnames=feedback_fields)
    trajectory_writer.writeheader()
    feedback_writer.writeheader()

    pygame.init()
    screen = pygame.display.set_mode((940, 740))
    pygame.display.set_caption("DURF Human + PPO Baseline")
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
    last_ai_action = STAY
    feedback_count = {1: 0, -1: 0}
    feedback_flash = ""
    feedback_flash_frames = 0

    print(f"Model: {model_path}")
    print(f"Session logs: {session_dir}")
    print("Human is green; PPO is blue.")
    print("Controls: WASD/Arrows=Move, Space=Interact, J=+1, K=-1, P=Pause, R=Reset")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_SPACE and not paused:
                        pending_interact = True
                    elif event.key == pygame.K_r:
                        ai_obs, _ = env.multi_reset()
                        episode += 1
                        episode_step = 0
                        episode_reward = 0.0
                        pending_interact = False
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
                        feedback_flash_frames = args.fps

            if not paused and running:
                human_action = INTERACT if pending_interact else human_motion_action()
                pending_interact = False
                ai_action_raw, _ = model.predict(
                    ai_obs,
                    deterministic=args.deterministic,
                )
                ai_action = int(ai_action_raw)
                (ai_obs, _), (reward, _), done, _ = env.multi_step(
                    ai_action,
                    human_action,
                )
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
                        "done": bool(done),
                    },
                )

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

            screen.fill((28, 30, 34))
            screen.blit(surface, surface.get_rect(center=(470, 305)))
            status = (
                f"Blue: PPO | Green: YOU | Episode {episode} | Step {episode_step} | "
                f"Reward {episode_reward:.1f} | {'PAUSED' if paused else 'RUNNING'}"
            )
            controls = "WASD/Arrows Move | Space Interact | P Pause | R Reset | Q/Esc Quit"
            feedback_status = (
                f"J +1 ({feedback_count[1]}) | K -1 ({feedback_count[-1]}) | "
                "LOGGING ONLY - MODEL IS NOT UPDATING"
            )
            screen.blit(font.render(status, True, (235, 235, 235)), (20, 630))
            screen.blit(small_font.render(controls, True, (170, 210, 180)), (20, 665))
            screen.blit(
                small_font.render(feedback_status, True, (245, 200, 105)),
                (20, 695),
            )
            if feedback_flash_frames > 0:
                screen.blit(
                    font.render(feedback_flash, True, (120, 225, 135)),
                    (690, 665),
                )
                feedback_flash_frames -= 1
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        trajectory_handle.close()
        feedback_handle.close()
        env.close()
        pygame.quit()

    print(f"Trajectory: {trajectory_path}")
    print(f"Feedback: {feedback_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
