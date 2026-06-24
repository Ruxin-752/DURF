"""Watch the archived RLlib PPO baseline in a Pygame window."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime

import pygame
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from durf.baseline.runtime import (
    DEFAULT_AGENT_NAME,
    DEFAULT_PLAYABLE_LAYOUTS,
    REPO_ROOT,
    ensure_agent_layout,
    filter_compatible_layouts,
    load_rllib_agent,
    make_baseline_env,
    resolve_agent_dir,
    rllib_action_index,
)


def write_crash_log(exc: BaseException) -> str:
    crash_dir = REPO_ROOT / "outputs" / "crash_logs"
    crash_dir.mkdir(parents=True, exist_ok=True)
    crash_path = crash_dir / f"watch_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    crash_path.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )
    return str(crash_path)


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
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Exit after this many environment steps; useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_layouts = list(dict.fromkeys([args.layout, *args.layouts]))
    ensure_agent_layout(args.agent, args.layout)
    layouts = filter_compatible_layouts(args.agent, requested_layouts)
    if args.layout not in layouts:
        layouts.insert(0, args.layout)
    layout_index = layouts.index(args.layout)
    current_layout = args.layout
    agent_dir = resolve_agent_dir(args.agent)
    agent0 = load_rllib_agent(args.agent, agent_index=0)
    agent1 = load_rllib_agent(args.agent, agent_index=1)
    env = make_baseline_env(current_layout, args.seed)
    envs_by_layout = {current_layout: env}
    base_env = env.base_env

    pygame.init()
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("DURF Baseline Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 26)
    visualizer = StateVisualizer()

    env.multi_reset()
    episode = 1
    episode_reward = 0.0
    episode_steps = 0
    paused = False
    running = True
    quit_reason = "unknown"

    def reset_episode(new_layout_index: int | None = None) -> None:
        nonlocal base_env
        nonlocal current_layout
        nonlocal env
        nonlocal episode_steps
        nonlocal episode_reward
        nonlocal layout_index

        if new_layout_index is not None:
            if not 0 <= new_layout_index < len(layouts):
                return
            if new_layout_index != layout_index:
                layout_index = new_layout_index
                current_layout = layouts[layout_index]
                if current_layout not in envs_by_layout:
                    envs_by_layout[current_layout] = make_baseline_env(
                        current_layout,
                        args.seed,
                    )
                env = envs_by_layout[current_layout]
                base_env = env.base_env

        env.multi_reset()
        agent0.reset()
        agent1.reset()
        episode_reward = 0.0
        episode_steps = 0

    print(f"Agent: {agent_dir}")
    print("Controls: Space=Pause/Resume, R=Reset, 1-4/N/M=Switch map, Esc=Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_reason = "window close button"
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_reason = "pygame Esc"
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_episode()
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    reset_episode((pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4).index(event.key))
                elif event.key == pygame.K_n:
                    reset_episode((layout_index + 1) % len(layouts))
                elif event.key == pygame.K_m:
                    reset_episode((layout_index - 1) % len(layouts))

        if not paused:
            state = base_env.state
            action0 = rllib_action_index(agent0, state)
            action1 = rllib_action_index(agent1, state)
            _, rewards, done, _ = env.multi_step(action0, action1)
            episode_reward += float(rewards[0])
            episode_steps += 1

            if args.max_steps is not None and episode_steps >= args.max_steps:
                quit_reason = "max steps reached"
                running = False

            if done:
                print(
                    f"Episode {episode}: reward={episode_reward:.1f}, "
                    f"steps={episode_steps}"
                )
                episode += 1
                time.sleep(0.5)
                reset_episode()

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
            f"Map {layout_index + 1}/{len(layouts)}: {current_layout} | "
            f"{'PAUSED' if paused else 'RUNNING'}"
        )
        screen.blit(font.render(status, True, (235, 235, 235)), (20, 650))
        screen.blit(
            font.render("Space Pause | R Reset | 1-4/N/M Map | Esc Quit", True, (170, 210, 180)),
            (20, 675),
        )
        pygame.display.flip()
        clock.tick(args.fps)

    for cached_env in set(envs_by_layout.values()):
        cached_env.close()
    pygame.quit()
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
