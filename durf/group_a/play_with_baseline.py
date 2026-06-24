"""Play Overcooked as the green agent beside the archived blue RLlib PPO."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import queue
import sys
import threading
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
from durf.group_a.deepseek_chat import DeepSeekChatError, chat_once


STAY = 4
INTERACT = 5
ACTION_NAMES = ("north", "south", "east", "west", "stay", "interact")
PAUSE_KEYS = (pygame.K_p, pygame.K_TAB, pygame.K_F1)
PAUSE_BUTTON = pygame.Rect(790, 620, 130, 42)
CHAT_BUTTON = pygame.Rect(650, 620, 120, 42)
PAUSE_DEBOUNCE_MS = 300
LAYOUT_SWITCH_DEBOUNCE_MS = 300
BUILD_ID = "pause-v5-chat"
LAYOUT_NUMBER_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4)
WINDOWS_LAYOUT_NUMBER_KEYS = (0x31, 0x32, 0x33, 0x34)
WINDOWS_CHAT_CHAR_KEYS = {
    **{vk: chr(vk).lower() for vk in range(0x41, 0x5B)},
    **{vk: chr(vk) for vk in range(0x30, 0x3A)},
    0x20: " ",
    0xBA: ";",
    0xBB: "=",
    0xBC: ",",
    0xBD: "-",
    0xBE: ".",
    0xBF: "/",
    0xC0: "`",
    0xDB: "[",
    0xDC: "\\",
    0xDD: "]",
    0xDE: "'",
}
WINDOWS_SHIFT_CHAT_CHARS = {
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    ";": ":",
    "=": "+",
    ",": "<",
    "-": "_",
    ".": ">",
    "/": "?",
    "`": "~",
    "[": "{",
    "\\": "|",
    "]": "}",
    "'": '"',
}
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
    0x08: "Backspace",
    0x0D: "Enter",
}
WINDOWS_WATCHED_KEYS = (
    set(WINDOWS_VK_NAMES) | set(WINDOWS_VK_ACTIONS) | set(WINDOWS_CHAT_CHAR_KEYS)
)
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


def windows_chat_text(pressed_now: set[int], pressed_keys: set[int]) -> str:
    shift_pressed = 0x10 in pressed_keys or 0xA0 in pressed_keys or 0xA1 in pressed_keys
    chars: list[str] = []
    for vk in sorted(pressed_now):
        char = WINDOWS_CHAT_CHAR_KEYS.get(vk)
        if not char:
            continue
        if shift_pressed:
            if "a" <= char <= "z":
                char = char.upper()
            else:
                char = WINDOWS_SHIFT_CHAT_CHARS.get(char, char)
        chars.append(char)
    return "".join(chars)


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


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def object_summary(obj):
    if obj is None:
        return None
    summary = {
        "type": type(obj).__name__,
        "name": getattr(obj, "name", None),
        "position": to_jsonable(getattr(obj, "position", None)),
    }
    for attr in (
        "ingredients",
        "cooking_tick",
        "is_cooking",
        "is_ready",
        "is_idle",
        "is_full",
    ):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            summary[attr] = to_jsonable(value() if callable(value) else value)
    return {key: value for key, value in summary.items() if value is not None}


def player_summary(player):
    return {
        "position": to_jsonable(getattr(player, "position", None)),
        "orientation": to_jsonable(getattr(player, "orientation", None)),
        "held_object": object_summary(getattr(player, "held_object", None)),
    }


def terrain_rows(mdp) -> list[str]:
    rows = getattr(mdp, "terrain_mtx", [])
    return ["".join(row) for row in rows]


def pot_state_summary(mdp, state):
    if not hasattr(mdp, "get_pot_states"):
        return None
    try:
        return to_jsonable(mdp.get_pot_states(state))
    except Exception as exc:
        return {"error": f"get_pot_states failed: {exc}"}


def state_facts(env) -> dict:
    state = env.base_env.state
    mdp = env.base_env.mdp
    players = list(getattr(state, "players", []))
    ai_player = players[0] if len(players) > 0 else None
    human_player = players[1] if len(players) > 1 else None
    objects = getattr(state, "objects", {})
    return {
        "ai_pos": to_jsonable(getattr(ai_player, "position", None)),
        "human_pos": to_jsonable(getattr(human_player, "position", None)),
        "ai_held_object": object_summary(getattr(ai_player, "held_object", None)),
        "human_held_object": object_summary(getattr(human_player, "held_object", None)),
        "players": [player_summary(player) for player in players],
        "objects": [
            {
                "position": to_jsonable(position),
                "object": object_summary(obj),
            }
            for position, obj in getattr(objects, "items", lambda: [])()
        ],
        "pot_states": pot_state_summary(mdp, state),
        "layout_features": {
            "layout_name": getattr(env, "layout_name", None),
            "terrain": terrain_rows(mdp),
        },
    }


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        words = raw_line.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
            while font.size(current)[0] > max_width and len(current) > 1:
                cut = len(current)
                while cut > 1 and font.size(current[:cut])[0] > max_width:
                    cut -= 1
                lines.append(current[:cut])
                current = current[cut:]
        lines.append(current)
    return lines


def render_chat_panel(
    screen,
    font,
    small_font,
    chat_messages: list[dict[str, str]],
    chat_input: str,
    chat_pending: bool,
    chat_status: str,
) -> None:
    panel = pygame.Rect(80, 80, 780, 500)
    pygame.draw.rect(screen, (20, 24, 32), panel, border_radius=8)
    pygame.draw.rect(screen, (115, 150, 190), panel, width=2, border_radius=8)
    screen.blit(font.render("DeepSeek Chat", True, (245, 245, 245)), (105, 100))
    hint = "Enter Send | Esc Close | Describe why the agent behavior is bad"
    screen.blit(small_font.render(hint, True, (170, 210, 180)), (105, 126))

    y = 155
    left = 105
    right_padding = 28
    content_gap = 8
    history = chat_messages[-8:]
    for message in history:
        role = message["role"].upper()
        color = (180, 220, 255) if message["role"] == "user" else (245, 220, 150)
        prefix = f"{role}: "
        prefix_width = small_font.size(prefix)[0] + content_gap
        content_x = left + prefix_width
        max_text_width = panel.right - content_x - right_padding
        prefix_surface = small_font.render(prefix, True, color)
        for idx, line in enumerate(
            wrap_text(message["content"], small_font, max_text_width)
        ):
            if idx == 0:
                screen.blit(prefix_surface, (left, y))
            screen.blit(small_font.render(line, True, color), (content_x, y))
            y += 22
            if y > 455:
                break
        if y > 455:
            break
        y += 6

    status = "Thinking..." if chat_pending else chat_status
    if status:
        screen.blit(small_font.render(status, True, (245, 200, 105)), (105, 460))

    input_rect = pygame.Rect(105, 490, 730, 42)
    pygame.draw.rect(screen, (35, 40, 50), input_rect, border_radius=5)
    pygame.draw.rect(screen, (120, 140, 170), input_rect, width=1, border_radius=5)
    cursor = "|" if pygame.time.get_ticks() // 500 % 2 == 0 else ""
    input_text = f"{chat_input}{cursor}" if chat_input else f"Type message...{cursor}"
    input_color = (235, 235, 235) if chat_input else (130, 140, 155)
    screen.blit(small_font.render(input_text[-120:], True, input_color), (118, 503))


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
    chat_path = session_dir / "chat_messages.csv"
    trajectory_handle = trajectory_path.open("w", newline="", encoding="utf-8")
    chat_handle = chat_path.open("w", newline="", encoding="utf-8")
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
        "state_before_json",
        "state_after_json",
    ]
    chat_fields = [
        "timestamp_utc",
        "episode",
        "episode_step",
        "total_step",
        "layout",
        "role",
        "content",
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
    chat_writer = csv.DictWriter(chat_handle, fieldnames=chat_fields)
    pause_writer = csv.DictWriter(pause_handle, fieldnames=pause_fields)
    trajectory_writer.writeheader()
    chat_writer.writeheader()
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
    chat_open = False
    chat_input = ""
    chat_messages: list[dict[str, str]] = []
    chat_pending = False
    chat_status = "Set DEEPSEEK_API_KEY to enable model replies."
    chat_result_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    last_textinput_at = -1000
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

    def log_chat_message(role: str, content: str) -> None:
        write_row(
            chat_writer,
            chat_handle,
            {
                "timestamp_utc": utc_timestamp(),
                "episode": episode,
                "episode_step": episode_step,
                "total_step": total_step,
                "layout": current_layout,
                "role": role,
                "content": content,
            },
        )

    def start_chat_request(prompt: str) -> None:
        nonlocal chat_pending
        nonlocal chat_status

        chat_messages.append({"role": "user", "content": prompt})
        log_chat_message("user", prompt)
        chat_pending = True
        chat_status = ""
        history = [
            {
                "role": "system",
                "content": (
                    "You are helping a human evaluate a cooperative Overcooked "
                    "PPO agent. Reply concisely and ask clarifying questions when needed."
                ),
            },
            *chat_messages[-10:],
        ]

        def worker() -> None:
            try:
                reply = chat_once(history)
                chat_result_queue.put(("assistant", reply))
            except DeepSeekChatError as exc:
                chat_result_queue.put(("error", str(exc)))
            except Exception as exc:
                chat_result_queue.put(("error", f"Chat failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

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
        "Controls: WASD/Arrows=Move, Space=Interact, "
        "P/Tab/F1=Pause, Chat=Language feedback, R=Reset, "
        "1-4/N/M=Switch map, Esc=Quit"
    )

    try:
        while running:
            while not chat_result_queue.empty():
                role, content = chat_result_queue.get()
                chat_pending = False
                if role == "assistant":
                    chat_messages.append({"role": "assistant", "content": content})
                    log_chat_message("assistant", content)
                    chat_status = ""
                else:
                    chat_status = content
                    chat_messages.append(
                        {"role": "assistant", "content": f"[Error] {content}"}
                    )
                    log_chat_message("assistant", f"[Error] {content}")

            pause_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_reason = "window close button"
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if chat_open:
                        if event.key == pygame.K_ESCAPE:
                            chat_open = False
                            chat_input = ""
                            pygame.key.stop_text_input()
                        elif event.key == pygame.K_RETURN:
                            prompt = chat_input.strip()
                            if prompt and not chat_pending:
                                chat_input = ""
                                start_chat_request(prompt)
                        elif event.key == pygame.K_BACKSPACE:
                            chat_input = chat_input[:-1]
                        elif event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL:
                            # Clipboard paste is not portable in pygame without extra deps.
                            chat_status = "Paste is not supported here; type your message."
                    elif event.key == pygame.K_ESCAPE:
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
                elif event.type == pygame.TEXTINPUT and chat_open:
                    chat_input += event.text
                    last_textinput_at = pygame.time.get_ticks()
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
                ):
                    if CHAT_BUTTON.collidepoint(event.pos):
                        chat_open = not chat_open
                        if chat_open:
                            paused = True
                            pygame.key.start_text_input()
                            pending_interact = False
                            pending_motion = None
                            held_motion_actions.clear()
                            chat_status = (
                                "" if os.getenv("DEEPSEEK_API_KEY")
                                else "Set DEEPSEEK_API_KEY to enable model replies."
                            )
                        else:
                            pygame.key.stop_text_input()
                    elif PAUSE_BUTTON.collidepoint(event.pos):
                        pause_requested = True

            windows_keys = windows_pressed_keys()
            windows_pressed_now = windows_keys - previous_windows_keys
            previous_windows_keys = windows_keys
            windows_new_actions = windows_motion_actions(windows_pressed_now)
            if chat_open:
                if 0x08 in windows_pressed_now:  # Backspace
                    chat_input = chat_input[:-1]
                if 0x0D in windows_pressed_now:  # Enter
                    prompt = chat_input.strip()
                    if prompt and not chat_pending:
                        chat_input = ""
                        start_chat_request(prompt)
                fallback_text = windows_chat_text(
                    windows_pressed_now - {0x08, 0x0D},
                    windows_keys,
                )
                if fallback_text and pygame.time.get_ticks() - last_textinput_at > 50:
                    chat_input += fallback_text
            if windows_new_actions and not paused and not chat_open:
                pending_motion = human_motion_action(windows_new_actions)
                last_key_debug = (
                    f"windows press "
                    f"{'+'.join(windows_key_name(vk) for vk in sorted(windows_pressed_now & set(WINDOWS_VK_ACTIONS)))}"
                    f" -> {ACTION_NAMES[pending_motion]}"
                )
            if 0x20 in windows_pressed_now and not paused and not chat_open:  # Space
                pending_interact = True
                last_key_debug = "windows Space -> interact"
            if windows_pressed_now & {0x50, 0x09, 0x70} and not chat_open:  # P, Tab, F1
                pause_requested = True
                last_key_debug = "windows pause key"
            if not chat_open:
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
                state_before = state_facts(env)
                step_started = time.perf_counter()
                (ai_obs, _), (reward, _), done, _ = env.multi_step(
                    ai_action,
                    human_action,
                )
                state_after = state_facts(env)
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
                        "state_before_json": json_dumps(state_before),
                        "state_after_json": json_dumps(state_after),
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
            chat_button_color = (70, 115, 155) if chat_open else (75, 95, 125)
            pygame.draw.rect(screen, chat_button_color, CHAT_BUTTON, border_radius=7)
            chat_button_text = small_font.render(
                "CHAT",
                True,
                (255, 255, 255),
            )
            screen.blit(
                chat_button_text,
                chat_button_text.get_rect(center=CHAT_BUTTON.center),
            )
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
                "Chat | R Reset | N/M Map | Esc Quit"
            )
            feedback_status = "Language feedback: click Chat, type comment, press Enter"
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
            if chat_open:
                render_chat_panel(
                    screen,
                    font,
                    small_font,
                    chat_messages,
                    chat_input,
                    chat_pending,
                    chat_status,
                )
            pygame.display.flip()
            clock.tick(args.render_fps)
    finally:
        trajectory_handle.flush()
        chat_handle.flush()
        pause_handle.flush()
        trajectory_handle.close()
        chat_handle.close()
        pause_handle.close()
        for cached_env in set(envs_by_layout.values()):
            cached_env.close()
        pygame.key.stop_text_input()
        pygame.quit()

    print(f"Trajectory: {trajectory_path}")
    print(f"Chat messages: {chat_path}")
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
