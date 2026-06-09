"""
Human-Feedback 训练 + 实时游戏画面

让你在训练过程中看到游戏画面，同时按 Q/E 给 AI 反馈。

用法:
    conda activate pantheonrl_env
    cd C:/Users/my185/Desktop/研究/durf/overcooked_ai/group_a_baseline
    python scripts/train_human_feedback_visual.py --participant P01

按键:
    Q  → +1 奖励 (Good)
    E  → -1 奖励 (Bad)
    ESC → 退出训练
"""
import sys
import os
import time
import argparse
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pantheonrl.common.agents import OnPolicyAgent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from src.env.factory import make_overcooked_env
from src.feedback.protocol import FeedbackEvent


def main():
    parser = argparse.ArgumentParser(description="Human-Feedback Training with Visual")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # Load config
    with open(Path(__file__).resolve().parent.parent / args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    layout_name = config["env"]["layout_name"]
    alpha = config["feedback"]["alpha"]

    print(f"\n{'='*60}")
    print(f"  Human-Feedback Training (with Visual)")
    print(f"  Participant: {args.participant}")
    print(f"  Layout: {layout_name}")
    print(f"  Steps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")

    # Create feedback queue for human feedback
    feedback_queue = multiprocessing.Queue()

    # Create environment with feedback queue
    env = make_overcooked_env(
        layout_name=layout_name,
        feedback_queue=feedback_queue,
        alpha=alpha,
        seed=args.seed,
    )

    # Get base env for state access (unwrap through Monitor, GymV21CompatibilityV0, HumanFeedbackEnvWrapper)
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    base_env = unwrapped.base_env
    grid = base_env.mdp.terrain_mtx

    # Create PPO model
    model = PPO(
        config["ppo"]["policy"],
        env,
        seed=args.seed,
        device="auto",
        verbose=0,
        **{k: v for k, v in config["ppo"].items() if k != "policy"},
    )

    # Initialize visualizer
    viz = StateVisualizer()
    pygame.init()
    clock = pygame.time.Clock()

    WINDOW_SIZE = (800, 600)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(f"Human-Feedback Training - {args.participant}")

    # Font for HUD
    font = pygame.font.Font(None, 24)

    print("Controls: Q=+1(Good)  E=-1(Bad)  ESC=Exit\n")

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Gymnasium API: (obs, info)
    total_reward = 0.0
    step_count = 0
    running = True
    feedback_count = 0

    while running and step_count < args.total_timesteps:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_q:
                    # Positive feedback
                    feedback = FeedbackEvent(
                        timestamp_ms=int(time.time() * 1000),
                        event_type="positive",
                        signal_value=1.0,
                        source="keyboard",
                    )
                    feedback_queue.put(feedback)
                    feedback_count += 1
                    print(f"  [+] Step {step_count}: +1 reward given")
                elif event.key == pygame.K_e:
                    # Negative feedback
                    feedback = FeedbackEvent(
                        timestamp_ms=int(time.time() * 1000),
                        event_type="negative",
                        signal_value=-1.0,
                        source="keyboard",
                    )
                    feedback_queue.put(feedback)
                    feedback_count += 1
                    print(f"  [-] Step {step_count}: -1 reward given")

        # Agent step
        action, _ = model.predict(obs, deterministic=False)
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        total_reward += reward
        step_count += 1

        # Render
        state = base_env.state
        hud_data = StateVisualizer.default_hud_data(state, score=total_reward)
        surface = viz.render_state(state=state, hud_data=hud_data, grid=grid)

        surf_rect = surface.get_rect()
        if surf_rect.width > 0 and surf_rect.height > 0:
            if surf_rect.width > WINDOW_SIZE[0] or surf_rect.height > WINDOW_SIZE[1]:
                scale = min(WINDOW_SIZE[0] / surf_rect.width, WINDOW_SIZE[1] / surf_rect.height) * 0.85
                new_w = int(surf_rect.width * scale)
                new_h = int(surf_rect.height * scale)
                surface = pygame.transform.scale(surface, (new_w, new_h))
                surf_rect = surface.get_rect()

            screen.fill((30, 30, 30))
            screen.blit(surface, (
                (WINDOW_SIZE[0] - surf_rect.width) // 2,
                (WINDOW_SIZE[1] - surf_rect.height) // 2 - 20
            ))

            # HUD overlay
            info_lines = [
                f"Participant: {args.participant}",
                f"Step: {step_count}/{args.total_timesteps}",
                f"Score: {total_reward:.1f}",
                f"Feedback given: {feedback_count}",
                "",
                "Q = +1 (Good)    E = -1 (Bad)    ESC = Exit"
            ]
            for i, line in enumerate(info_lines):
                color = (180, 220, 180) if "Q" in line or "E" in line else (200, 200, 200)
                text = font.render(line, True, color)
                screen.blit(text, (10, WINDOW_SIZE[1] - 30 - (len(info_lines) - i) * 22))

            pygame.display.flip()

        # Episode end
        if terminated or truncated:
            print(f"  Episode done! Steps: {step_count}, Score: {total_reward:.1f}")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward = 0.0

        clock.tick(args.fps)

    # Save model
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(__file__).resolve().parent.parent / "logs" / f"human-feedback_{args.participant}_{ts}" / "final_model"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"\nModel saved to: {save_path}.zip")
    print(f"Total steps: {step_count}, Feedback given: {feedback_count}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
