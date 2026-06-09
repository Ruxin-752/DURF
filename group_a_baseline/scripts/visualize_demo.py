"""
Visual demo script - Load trained model and watch the agent play in a window.

Usage:
    conda activate pantheonrl_env
    cd C:/Users/my185/Desktop/研究/durf/overcooked_ai/group_a_baseline
    python scripts/visualize_demo.py

Controls:
    ESC / Q  - Quit
    Space    - Pause/Resume
    R        - Reset episode
"""
import sys
import os
import time
import argparse
from pathlib import Path

# Fix Windows GBK encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure src module is findable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gym
import overcookedgym  # noqa: F401 - register OvercookedMultiEnv-v0
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pantheonrl.common.agents import OnPolicyAgent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


def main():
    parser = argparse.ArgumentParser(description="Visualize trained Overcooked agent")
    parser.add_argument(
        "--model",
        type=str,
        default="logs/no-feedback_P00_20260602_224619/final_model.zip",
        help="Path to trained model (.zip)"
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="cramped_room",
        help="Overcooked layout name"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Render FPS (default 10, higher = faster)"
    )
    args = parser.parse_args()

    # Resolve model path relative to project root (not cwd)
    project_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model)
    if not model_path.exists():
        model_path = project_root / args.model
    if not model_path.exists():
        print(f"[ERROR] Model not found: {args.model}")
        print(f"  Tried: {Path(args.model)}")
        print(f"  Tried: {project_root / args.model}")
        print("Run training first or specify correct path")
        return

    print(f"Loading model: {model_path}")
    model = PPO.load(str(model_path), device="auto")

    # Create environment
    print(f"Creating environment: {args.layout}")
    env = gym.make("OvercookedMultiEnv-v0", layout_name=args.layout)

    # Create random PPO partner
    dummy_alt = env.getDummyEnv(1)
    partner_model = PPO("MlpPolicy", DummyVecEnv([lambda: dummy_alt]), verbose=0)
    partner = OnPolicyAgent(partner_model)
    env.add_partner_agent(partner)

    # Get the base Overcooked environment for state access
    base_env = env.base_env

    # Get grid layout for visualizer
    grid = base_env.mdp.terrain_mtx

    # Initialize StateVisualizer
    viz = StateVisualizer()

    # Initialize pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Create a visible window
    WINDOW_SIZE = (800, 600)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Overcooked Demo (Press ESC to quit)")

    print("\n" + "=" * 50)
    print("  Controls: ESC/Q=Quit  Space=Pause  R=Reset")
    print("=" * 50 + "\n")

    obs = env.reset()
    total_reward = 0.0
    step_count = 0
    paused = False
    running = True
    episode_count = 0

    while running:
        # Handle keyboard events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("[PAUSED]" if paused else "[RESUMED]")
                elif event.key == pygame.K_r:
                    print("[RESET] Resetting episode")
                    obs = env.reset()
                    total_reward = 0.0
                    step_count = 0

        if paused:
            clock.tick(args.fps)
            continue

        # Agent decision
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        # Handle both old gym (4 values) and new gymnasium (5 values) APIs
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        total_reward += reward
        step_count += 1

        # Render using StateVisualizer - get a pygame surface
        state = base_env.state
        hud_data = StateVisualizer.default_hud_data(state, score=total_reward)
        surface = viz.render_state(
            state=state,
            hud_data=hud_data,
            grid=grid
        )

        # Blit the surface onto our window, centered
        surf_rect = surface.get_rect()
        if surf_rect.width > 0 and surf_rect.height > 0:
            # Scale if too big
            if surf_rect.width > WINDOW_SIZE[0] or surf_rect.height > WINDOW_SIZE[1]:
                scale = min(WINDOW_SIZE[0] / surf_rect.width, WINDOW_SIZE[1] / surf_rect.height) * 0.9
                new_w = int(surf_rect.width * scale)
                new_h = int(surf_rect.height * scale)
                surface = pygame.transform.scale(surface, (new_w, new_h))
                surf_rect = surface.get_rect()

            screen.fill((30, 30, 30))
            screen.blit(surface, (
                (WINDOW_SIZE[0] - surf_rect.width) // 2,
                (WINDOW_SIZE[1] - surf_rect.height) // 2
            ))
            pygame.display.flip()

        # Status info
        if step_count % 50 == 0:
            print(f"  Step {step_count:>5d} | Reward: {total_reward:>+6.1f}")

        # Episode end
        if terminated or truncated:
            episode_count += 1
            print(f"\n[DONE] Episode {episode_count} finished! Steps: {step_count}, Total Reward: {total_reward:.1f}\n")
            time.sleep(1)
            obs = env.reset()
            total_reward = 0.0
            step_count = 0

        clock.tick(args.fps)

    env.close()
    pygame.quit()
    print("Exited.")


if __name__ == "__main__":
    main()
