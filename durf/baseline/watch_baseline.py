"""Watch the archived 1M-step SB3 PPO baseline in a Pygame window."""

from __future__ import annotations

import argparse
import sys
import time

import pygame
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from stable_baselines3 import PPO

from durf.baseline.runtime import (
    get_base_env,
    make_baseline_env,
    reset_env,
    resolve_model_path,
    step_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="SB3 PPO .zip; defaults to the archived 1M model")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Exit after this many environment steps; useful for smoke tests.",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = PPO.load(model_path, device="cpu")
    env = make_baseline_env(args.layout, args.seed)
    base_env = get_base_env(env)

    pygame.init()
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("DURF Baseline Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 26)
    visualizer = StateVisualizer()

    obs = reset_env(env)
    episode = 1
    episode_reward = 0.0
    episode_steps = 0
    paused = False
    running = True

    print(f"Model: {model_path}")
    print("Controls: Space=Pause/Resume, R=Reset, Esc/Q=Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs = reset_env(env)
                    episode_reward = 0.0
                    episode_steps = 0

        if not paused:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, _ = step_env(env, action)
            episode_reward += reward
            episode_steps += 1

            if args.max_steps is not None and episode_steps >= args.max_steps:
                running = False

            if done:
                print(
                    f"Episode {episode}: reward={episode_reward:.1f}, "
                    f"steps={episode_steps}"
                )
                episode += 1
                time.sleep(0.5)
                obs = reset_env(env)
                episode_reward = 0.0
                episode_steps = 0

        state = base_env.state
        hud_data = StateVisualizer.default_hud_data(state, score=episode_reward)
        surface = visualizer.render_state(
            state=state,
            hud_data=hud_data,
            grid=base_env.mdp.terrain_mtx,
        )

        available_width, available_height = 880, 610
        scale = min(
            available_width / surface.get_width(),
            available_height / surface.get_height(),
            1.0,
        )
        if scale < 1:
            surface = pygame.transform.smoothscale(
                surface,
                (
                    int(surface.get_width() * scale),
                    int(surface.get_height() * scale),
                ),
            )

        screen.fill((28, 30, 34))
        screen.blit(surface, surface.get_rect(center=(450, 320)))
        status = (
            f"Episode {episode} | Step {episode_steps} | "
            f"Reward {episode_reward:.1f} | "
            f"{'PAUSED' if paused else 'RUNNING'}"
        )
        screen.blit(font.render(status, True, (235, 235, 235)), (20, 650))
        screen.blit(
            font.render("Space Pause | R Reset | Q/Esc Quit", True, (170, 210, 180)),
            (20, 675),
        )
        pygame.display.flip()
        clock.tick(args.fps)

    env.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
