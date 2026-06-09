"""
Play with AI — 你控制一个厨师，AI 控制另一个（Group A 模式）

用法:
    conda activate pantheonrl_env
    cd C:/Users/my185/Desktop/研究/durf/overcooked_ai/group_a_baseline
    python scripts/play_with_ai.py --model "logs/no-feedback_P00_20260602_224619/final_model.zip"

按键:
    WASD / 方向键  → 移动
    Space          → 交互（拿/放食材、上菜）
    Q              → +1 奖励 (Good)
    E              → -1 奖励 (Bad)
    ESC            → 退出
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
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from src.env.factory import make_overcooked_env
from src.feedback.protocol import FeedbackEvent


def main():
    parser = argparse.ArgumentParser(description="Play with AI (Group A mode)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained PPO model for partner AI")
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    # Load the trained PPO model for the partner AI
    print("Loading PPO model...")
    partner_model = PPO.load(args.model)
    print("PPO model loaded!")

    # Create feedback queue for Q/E feedback
    feedback_queue = multiprocessing.Queue()

    # Create environment using factory
    print("Creating environment...")
    env = make_overcooked_env(
        layout_name=args.layout,
        feedback_queue=feedback_queue,
    )
    print("Environment created!")

    # Get base env for rendering
    # env is Monitor -> GymV21CompatibilityV0 -> HumanFeedbackEnvWrapper -> OvercookedMultiEnv
    base_env = env.env.env.base_env
    grid = base_env.mdp.terrain_mtx

    # Initialize pygame
    viz = StateVisualizer()
    pygame.init()
    clock = pygame.time.Clock()

    WINDOW_SIZE = (800, 600)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Overcooked - Group A: Play with AI + Q/E Feedback")
    font = pygame.font.Font(None, 24)

    # Action mapping
    KEY_TO_ACTION = {
        pygame.K_UP:    0,  # NORTH
        pygame.K_DOWN:  1,  # SOUTH
        pygame.K_LEFT:  2,  # WEST
        pygame.K_RIGHT: 3,  # EAST
        pygame.K_w:     0,  # NORTH
        pygame.K_s:     1,  # SOUTH
        pygame.K_a:     2,  # WEST
        pygame.K_d:     3,  # EAST
        pygame.K_SPACE: 4,  # STAY/INTERACT
    }

    print(f"\n{'='*60}")
    print(f"  Overcooked - Group A: Play with AI + Q/E Feedback")
    print(f"  Layout: {args.layout}")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")
    print(f"  Controls:")
    print(f"    WASD/Arrows  → Move your chef")
    print(f"    Space        → Interact (pick up/put down/serve)")
    print(f"    Q            → +1 reward (Good!)")
    print(f"    E            → -1 reward (Bad!)")
    print(f"    ESC          → Exit")
    print(f"{'='*60}\n")

    # Reset environment
    print("Resetting environment...")
    obs_dict = env.reset()
    # Gymnasium compatibility: obs is a dict with 'observation' key
    if isinstance(obs_dict, dict):
        obs = obs_dict["observation"]
    else:
        obs = obs_dict
    state = base_env.state
    print(f"Environment ready! obs type: {type(obs)}, obs[0] shape: {obs[0].shape if hasattr(obs[0], 'shape') else 'N/A'}")

    total_reward = 0.0
    step_count = 0
    running = True
    last_human_action = 4  # STAY
    feedback_count = 0

    while running:
        # Get human action from keyboard
        human_action = last_human_action  # default: repeat last action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_TO_ACTION:
                    human_action = KEY_TO_ACTION[event.key]
                    last_human_action = human_action
                    print(f"  [ACTION] Step {step_count}: Human action = {human_action}")
                elif event.key == pygame.K_q:
                    feedback_queue.put(FeedbackEvent(
                        timestamp_ms=int(time.time() * 1000),
                        event_type="positive",
                        signal_value=1.0,
                        source="keyboard",
                    ))
                    feedback_count += 1
                    print(f"  [+] Step {step_count}: +1 reward given")
                elif event.key == pygame.K_e:
                    feedback_queue.put(FeedbackEvent(
                        timestamp_ms=int(time.time() * 1000),
                        event_type="negative",
                        signal_value=-1.0,
                        source="keyboard",
                    ))
                    feedback_count += 1
                    print(f"  [-] Step {step_count}: -1 reward given")

        # Get AI action from the trained model
        # obs is (2, 96) numpy array - agent 0 and agent 1 observations
        ai_obs = obs[1]  # AI controls agent 1
        ai_action, _ = partner_model.predict(ai_obs, deterministic=False)

        # Step the environment with both actions
        # human controls agent 0, AI controls agent 1
        actions = np.array([human_action, ai_action])
        step_result = env.step(actions)
        if len(step_result) == 5:
            # Gymnasium API: obs, reward, terminated, truncated, info
            obs_dict, rewards, terminated, truncated, infos = step_result
            dones = np.logical_or(terminated, truncated)
        else:
            # Gym old API: obs, reward, done, info
            obs_dict, rewards, dones, infos = step_result
        if isinstance(obs_dict, dict):
            obs = obs_dict["observation"]
        else:
            obs = obs_dict

        # Poll feedback queue
        human_signal = 0.0
        while True:
            try:
                event = feedback_queue.get_nowait()
                human_signal += event.signal_value
            except:
                break

        # Total reward = env reward + human feedback
        step_reward = rewards[0] + human_signal
        total_reward += step_reward
        step_count += 1

        # Get updated state
        state = base_env.state

        # Render
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

            # HUD
            action_names = ["↑", "↓", "←", "→", "⏎", "✋"]
            info_lines = [
                f"Step: {step_count}",
                f"Score: {total_reward:.1f}",
                f"Your action: {action_names[human_action]}",
                f"AI action: {action_names[ai_action]}",
                f"Feedback given: {feedback_count}",
                "",
                "WASD=Move  Space=Interact  Q=+1  E=-1  ESC=Exit"
            ]
            for i, line in enumerate(info_lines):
                color = (180, 220, 180) if "WASD" in line else (200, 200, 200)
                text = font.render(line, True, color)
                screen.blit(text, (10, WINDOW_SIZE[1] - 30 - (len(info_lines) - i) * 22))

            pygame.display.flip()

        # Episode end
        if dones[0]:
            print(f"  Episode done! Steps: {step_count}, Score: {total_reward:.1f}, Feedback: {feedback_count}")
            obs_dict = env.reset()
            if isinstance(obs_dict, dict):
                obs = obs_dict["observation"]
            else:
                obs = obs_dict
            state = base_env.state
            total_reward = 0.0
            step_count = 0

        clock.tick(args.fps)

    env.close()
    pygame.quit()
    print(f"\nGame ended. Total steps: {step_count}, Feedback given: {feedback_count}")


if __name__ == "__main__":
    main()
