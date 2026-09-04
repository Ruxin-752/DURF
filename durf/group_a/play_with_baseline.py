"""Play Overcooked as the green human beside a blue AI teammate."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pygame
import tensorflow as tf
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from durf.baseline.action_prior import StepPrefixPrior, available_priors
from durf.baseline.coordination import (
    choose_reroute_action,
    CONTINUE_CURRENT_SUBGOAL,
    YIELD,
    CoordinationController,
    build_coordination_candidates,
    path_conflict_type,
)
from durf.baseline.collect_rule_teacher_dataset import (
    SUBGOAL_TO_INDEX,
    SUBGOALS,
    counters_for_put_down,
    first_action_to_feature,
    make_motion_planner,
    pots_needing_ingredient,
    choose_task_candidate,
    rule_teacher_candidates,
)
from durf.baseline.task_cost import attach_step_costs, build_feature_map
from durf.baseline.runtime import (
    DEFAULT_AGENT_NAME,
    DEFAULT_PLAYABLE_LAYOUTS,
    REPO_ROOT,
    ensure_agent_layout,
    filter_compatible_layouts,
    load_rllib_agent,
    make_direct_multi_env,
    resolve_agent_dir,
    rllib_action_index,
)
from durf.feedback_attribution.condition_features import extract_condition_features
from durf.group_a.deepseek_chat import DeepSeekChatError, chat_once
from durf.hu.subgoal_reranker import (
    COORDINATION_DECISION_LEVEL,
    TASK_DECISION_LEVEL,
    load_runtime_hu,
    runtime_hu_score,
    runtime_hu_has_support,
    warn_unknown_runtime_subgoal,
)


STAY = 4
INTERACT = 5
ACTION_NAMES = ("north", "south", "east", "west", "stay", "interact")
PAUSE_KEYS = (pygame.K_p, pygame.K_TAB, pygame.K_F1)
PAUSE_BUTTON = pygame.Rect(790, 620, 130, 42)
CHAT_BUTTON = pygame.Rect(650, 620, 120, 42)
PAUSE_DEBOUNCE_MS = 300
LAYOUT_SWITCH_DEBOUNCE_MS = 300
BUILD_ID = "hierarchical-hu-v1"
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
    parser.add_argument(
        "--ai-mode",
        choices=("ppo", "random", "subgoal_executor"),
        default="ppo",
        help=(
            "Use the selected PPO agent, a uniformly random collaborator, or "
            "the rule-planner + learned subgoal executor backbone."
        ),
    )
    parser.add_argument(
        "--load-retired-executor",
        action="store_true",
        help=(
            "Load the retired learned subgoal executor. Nothing calls it -- the "
            "motion planner executes every subgoal -- so this exists only for "
            "reproducing older runs."
        ),
    )
    parser.add_argument(
        "--subgoal-executor",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs"
            / "subgoal_executors"
            / "h0_p0_rule_executor_v5_fast_two_delivery_recovery"
            / "executor.keras"
        ),
        help="Keras executor used when --ai-mode subgoal_executor.",
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
        default=2.0,
        help=(
            "Environment decisions per second. Default 2.0 means one timestep "
            "every 0.5 seconds."
        ),
    )
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument(
        "--horizon",
        type=int,
        default=800,
        help="Maximum environment timesteps per episode. Default 800 for H0 demos.",
    )
    parser.add_argument(
        "--ai-action-prior",
        choices=available_priors(),
        default=None,
        help=(
            "Optional deterministic AI prefix before PPO takes over. "
            "Useful for diagnosing brittle opening skills."
        ),
    )
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
    parser.add_argument(
        "--annotation-mode",
        action="store_true",
        help=(
            "Auto-pause every --annotation-interval environment steps and "
            "record human correction text without sending it to the LLM."
        ),
    )
    parser.add_argument(
        "--annotation-interval",
        type=int,
        default=8,
        help="In annotation mode, pause for human feedback every N environment steps.",
    )
    parser.add_argument(
        "--hu-model",
        type=Path,
        default=None,
        help=(
            "Optional Hu-v0 subgoal reranker JSON. By default it is used in "
            "shadow logging only unless --hu-apply is set."
        ),
    )
    parser.add_argument("--hu-user-id", default="PILOT01")
    parser.add_argument(
        "--hu-step-tolerance",
        type=float,
        default=None,
        help=(
            "Estimated extra STEPS to the next delivery the agent may spend "
            "to honour the learned preference (durf/baseline/task_cost.py). "
            "When set (> 0) it replaces the task-point band: candidates whose "
            "estimated steps-to-delivery are within this many steps of the "
            "task optimum's form the acceptable set. The task optimum itself "
            "is still the frozen backbone's pick. Unset/0 = task-point rule."
        ),
    )
    parser.add_argument(
        "--hu-task-tolerance",
        type=float,
        default=0.0,
        help=(
            "Task points the agent may give up for the learned preference. "
            "Candidates within this many points of the task optimum form the "
            "acceptable set; the user's preference orders that set. 0 = frozen "
            "task backbone. This is the same quantity as the non-inferiority "
            "margin used to accept the task-competence result."
        ),
    )
    parser.add_argument(
        "--hu-coordination-tolerance",
        type=float,
        default=0.0,
        help=(
            "Prior points the coordination prior may give up for the learned "
            "preference. The prior is two-level (0/1), so this is a switch: "
            "0 keeps the prior's yield/continue pick; >= 1 lets the user's "
            "preference decide, with the prior as tie-break. Replaces "
            "--hu-coordination-lambda: the prior and Hu are never added."
        ),
    )
    parser.add_argument("--hu-coordination-lambda", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--hu-apply",
        action="store_true",
        help="Actually apply Hu reranking. Default is shadow logging only.",
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


def held_name(facts: dict, key: str) -> str:
    obj = facts.get(key)
    if not obj:
        return "-"
    return obj.get("name") or obj.get("type") or "object"


def pos_text(value) -> str:
    if value is None:
        return "?"
    return ",".join(str(part) for part in value)


def replay_line(step_record: dict) -> str:
    after = step_record["state_after"]
    return (
        f"t-{step_record['age']}: "
        f"AI {step_record['ai_action_name']} "
        f"@{pos_text(after.get('ai_pos'))}({held_name(after, 'ai_held_object')}) | "
        f"YOU {step_record['human_action_name']} "
        f"@{pos_text(after.get('human_pos'))}({held_name(after, 'human_held_object')}) | "
        f"r={step_record['reward']:.1f}"
    )


def replay_messages(recent_steps: list[dict], window: int) -> list[dict[str, str]]:
    if not recent_steps:
        return [
            {
                "role": "assistant",
                "content": "No recent steps yet.",
            }
        ]
    window_steps = recent_steps[-window:]
    total = len(window_steps)
    messages = [
        {
            "role": "assistant",
            "content": (
                "Recent action replay. Use prefixes if helpful: "
                "now:, last3:, last8:, event:"
            ),
        }
    ]
    for index, record in enumerate(window_steps):
        record = dict(record)
        record["age"] = total - index
        messages.append({"role": "assistant", "content": replay_line(record)})
    return messages


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
    title: str = "DeepSeek Chat",
    hint: str = "Enter Send | Esc Close | Describe why the agent behavior is bad",
) -> None:
    panel = pygame.Rect(80, 80, 780, 500)
    pygame.draw.rect(screen, (20, 24, 32), panel, border_radius=8)
    pygame.draw.rect(screen, (115, 150, 190), panel, width=2, border_radius=8)
    screen.blit(font.render(title, True, (245, 245, 245)), (105, 100))
    screen.blit(small_font.render(hint, True, (170, 210, 180)), (105, 126))

    left = 105
    right_padding = 28
    content_gap = 8
    line_height = 22
    history_top = 155
    history_bottom = 448
    history_y = history_bottom
    for message in reversed(chat_messages):
        role = message["role"].upper()
        color = (180, 220, 255) if message["role"] == "user" else (245, 220, 150)
        prefix = f"{role}: "
        prefix_width = small_font.size(prefix)[0] + content_gap
        content_x = left + prefix_width
        max_text_width = panel.right - content_x - right_padding
        prefix_surface = small_font.render(prefix, True, color)
        lines = wrap_text(message["content"], small_font, max_text_width)
        block_height = len(lines) * line_height + 6
        if history_y - block_height < history_top:
            visible_count = max(0, (history_y - history_top - 6) // line_height)
            if visible_count == 0:
                break
            first_visible_index = max(0, len(lines) - visible_count)
            lines = lines[first_visible_index:]
            block_height = len(lines) * line_height
            continuation = first_visible_index > 0
        else:
            first_visible_index = 0
            continuation = False

        block_y = history_y - block_height
        for idx, line in enumerate(lines):
            source_index = first_visible_index + idx
            line_y = block_y + idx * line_height
            if source_index == 0:
                screen.blit(prefix_surface, (left, line_y))
            elif continuation and idx == 0:
                cont_prefix = f"{role} ... "
                screen.blit(small_font.render(cont_prefix, True, color), (left, line_y))
            screen.blit(small_font.render(line, True, color), (content_x, line_y))
        history_y = block_y - 6
        if history_y <= history_top:
            break

    status = "Thinking..." if chat_pending else chat_status
    if status:
        status_lines = wrap_text(status, small_font, 730)
        for idx, line in enumerate(status_lines[:2]):
            screen.blit(
                small_font.render(line, True, (245, 200, 105)),
                (105, 458 + idx * 18),
            )

    input_rect = pygame.Rect(105, 490, 730, 72)
    pygame.draw.rect(screen, (35, 40, 50), input_rect, border_radius=5)
    pygame.draw.rect(screen, (120, 140, 170), input_rect, width=1, border_radius=5)
    cursor = "|" if pygame.time.get_ticks() // 500 % 2 == 0 else ""
    input_text = f"{chat_input}{cursor}" if chat_input else f"Type message...{cursor}"
    input_color = (235, 235, 235) if chat_input else (130, 140, 155)
    input_lines = wrap_text(input_text, small_font, input_rect.width - 26)
    for idx, line in enumerate(input_lines[-3:]):
        screen.blit(
            small_font.render(line, True, input_color),
            (118, 500 + idx * 20),
        )


def render_annotation_panel(
    screen,
    font,
    small_font,
    replay_snapshot: list[dict],
    replay_surfaces: list[pygame.Surface],
    chat_input: str,
    chat_status: str,
) -> None:
    panel = pygame.Rect(45, 55, 850, 625)
    pygame.draw.rect(screen, (18, 22, 30), panel, border_radius=8)
    pygame.draw.rect(screen, (145, 180, 220), panel, width=2, border_radius=8)
    screen.blit(
        font.render(
            f"Annotation Replay: last {len(replay_surfaces)} steps",
            True,
            (245, 245, 245),
        ),
        (70, 75),
    )
    hint = "Enter Save | Esc Skip | Prefix: now:, last3:, last8:, event:"
    screen.blit(small_font.render(hint, True, (170, 210, 180)), (70, 101))

    thumb_w = 185
    thumb_h = 118
    gap_x = 16
    gap_y = 38
    start_x = 70
    start_y = 132
    max_items = min(len(replay_surfaces), 8)
    first_index = len(replay_surfaces) - max_items
    shown_surfaces = replay_surfaces[first_index:]
    shown_records = replay_snapshot[first_index:]

    for idx, surface in enumerate(shown_surfaces):
        row = idx // 4
        col = idx % 4
        x = start_x + col * (thumb_w + gap_x)
        y = start_y + row * (thumb_h + gap_y)
        rect = pygame.Rect(x, y, thumb_w, thumb_h)
        pygame.draw.rect(screen, (35, 40, 52), rect, border_radius=5)
        pygame.draw.rect(screen, (90, 115, 145), rect, width=1, border_radius=5)
        scale = min(thumb_w / surface.get_width(), thumb_h / surface.get_height())
        scaled = pygame.transform.smoothscale(
            surface,
            (
                max(1, int(surface.get_width() * scale)),
                max(1, int(surface.get_height() * scale)),
            ),
        )
        screen.blit(scaled, scaled.get_rect(center=rect.center))
        record = shown_records[idx] if idx < len(shown_records) else {}
        age = max_items - idx
        label = (
            f"t-{age} AI:{record.get('ai_action_name', '?')} "
            f"YOU:{record.get('human_action_name', '?')} "
            f"r={float(record.get('reward', 0.0)):.0f}"
        )
        screen.blit(
            small_font.render(label, True, (235, 225, 170)),
            (x, y + thumb_h + 4),
        )

    if chat_status:
        for idx, line in enumerate(wrap_text(chat_status, small_font, 780)[:2]):
            screen.blit(
                small_font.render(line, True, (245, 200, 105)),
                (70, 448 + idx * 18),
            )

    input_rect = pygame.Rect(70, 500, 800, 95)
    pygame.draw.rect(screen, (35, 40, 50), input_rect, border_radius=5)
    pygame.draw.rect(screen, (120, 140, 170), input_rect, width=1, border_radius=5)
    cursor = "|" if pygame.time.get_ticks() // 500 % 2 == 0 else ""
    input_text = f"{chat_input}{cursor}" if chat_input else f"Type annotation...{cursor}"
    input_color = (235, 235, 235) if chat_input else (130, 140, 155)
    input_lines = wrap_text(input_text, small_font, input_rect.width - 26)
    for idx, line in enumerate(input_lines[-4:]):
        screen.blit(
            small_font.render(line, True, input_color),
            (83, 512 + idx * 20),
        )


def main() -> int:
    args = parse_args()
    if args.step_hz <= 0:
        raise ValueError("--step-hz must be greater than zero")
    if args.render_fps <= 0:
        raise ValueError("--render-fps must be greater than zero")
    if args.start_delay < 0:
        raise ValueError("--start-delay cannot be negative")
    if args.horizon <= 0:
        raise ValueError("--horizon must be greater than zero")
    if args.annotation_interval <= 0:
        raise ValueError("--annotation-interval must be greater than zero")

    requested_layouts = list(dict.fromkeys([args.layout, *args.layouts]))
    if args.ai_mode == "ppo":
        ensure_agent_layout(args.agent, args.layout)
        layouts = filter_compatible_layouts(args.agent, requested_layouts)
    elif args.ai_mode == "subgoal_executor":
        if not args.subgoal_executor.exists():
            raise FileNotFoundError(f"Subgoal executor not found: {args.subgoal_executor}")
        # The learned executor is trained for the H0 ring observation/subgoal space.
        # Keep hot-switching disabled in this mode so a test cannot silently jump
        # to an incompatible layout.
        layouts = [args.layout]
    else:
        layouts = requested_layouts
    if args.layout not in layouts:
        layouts.insert(0, args.layout)
    layout_index = layouts.index(args.layout)
    current_layout = args.layout

    agent_dir = resolve_agent_dir(args.agent) if args.ai_mode == "ppo" else None
    ai_agent = load_rllib_agent(args.agent, agent_index=0) if args.ai_mode == "ppo" else None
    # The learned executor is retired: every subgoal is executed by the motion
    # planner.  The file is only loaded when explicitly asked for, so a run no
    # longer pays the TensorFlow import cost for a model nothing calls.
    subgoal_model = (
        tf.keras.models.load_model(args.subgoal_executor)
        if args.ai_mode == "subgoal_executor" and args.load_retired_executor
        else None
    )
    hu_model = load_runtime_hu(args.hu_model) if args.hu_model else None
    if args.hu_task_tolerance < 0:
        raise ValueError("--hu-task-tolerance cannot be negative")
    if args.hu_step_tolerance is not None and args.hu_step_tolerance < 0:
        raise ValueError("--hu-step-tolerance cannot be negative")
    if args.hu_coordination_lambda is not None:
        raise ValueError(
            "--hu-coordination-lambda is gone: the coordination prior and Hu are no "
            "longer added. Use --hu-coordination-tolerance (0 = prior only, "
            ">= 1 = the preference decides)."
        )
    if args.hu_coordination_tolerance < 0:
        raise ValueError("--hu-coordination-tolerance cannot be negative")
    coordination_controller = CoordinationController(
        min_commit_steps=1,
        max_option_steps=3,
        yield_cooldown_steps=2,
    )
    ai_action_prior = StepPrefixPrior.from_name(args.ai_action_prior)
    env = make_direct_multi_env(current_layout, args.seed, horizon=args.horizon)
    envs_by_layout = {current_layout: env}
    motion_planners_by_layout = (
        {current_layout: make_motion_planner(current_layout, args.seed, args.horizon)}
        if args.ai_mode == "subgoal_executor"
        else {}
    )
    task_features_by_layout = {
        layout: build_feature_map(planner.mdp, planner)
        for layout, planner in motion_planners_by_layout.items()
    }
    session_dir = new_session_dir(args.output_dir)

    trajectory_path = session_dir / "trajectory.csv"
    chat_path = session_dir / "chat_messages.csv"
    annotation_path = session_dir / "annotations.csv"
    metadata_path = session_dir / "session_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "build_id": BUILD_ID,
                "agent": str(agent_dir) if agent_dir else None,
                "ai_mode": args.ai_mode,
                "subgoal_executor": (
                    str(args.subgoal_executor) if args.ai_mode == "subgoal_executor" else None
                ),
                "layout": args.layout,
                "layouts": layouts,
                "seed": args.seed,
                "step_hz": args.step_hz,
                "horizon": args.horizon,
                "human_player_index": 1,
                "ai_player_index": 0,
                "hu_model": str(args.hu_model) if args.hu_model else None,
                "hu_user_id": args.hu_user_id,
                "hu_task_tolerance": args.hu_task_tolerance,
                "hu_step_tolerance": args.hu_step_tolerance,
                "hu_coordination_tolerance": args.hu_coordination_tolerance,
                "hu_apply": args.hu_apply,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    trajectory_handle = trajectory_path.open("w", newline="", encoding="utf-8")
    chat_handle = chat_path.open("w", newline="", encoding="utf-8")
    annotation_handle = annotation_path.open("w", newline="", encoding="utf-8")
    trajectory_fields = [
        "timestamp_utc",
        "episode",
        "episode_step",
        "total_step",
        "layout",
        "ai_action",
        "ai_action_name",
        "ai_subgoal",
        "ai_condition_features_json",
        "ai_subgoal_candidates_json",
        "task_decision_json",
        "coordination_decision_json",
        "ai_event",
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
    annotation_fields = [
        "timestamp_utc",
        "episode",
        "episode_step",
        "total_step",
        "layout",
        "annotation_scope_hint",
        "annotation_text",
        "last_ai_action",
        "last_ai_action_name",
        "episode_reward",
        "state_json",
        "recent_replay_json",
    ]
    trajectory_writer = csv.DictWriter(trajectory_handle, fieldnames=trajectory_fields)
    chat_writer = csv.DictWriter(chat_handle, fieldnames=chat_fields)
    pause_writer = csv.DictWriter(pause_handle, fieldnames=pause_fields)
    annotation_writer = csv.DictWriter(annotation_handle, fieldnames=annotation_fields)
    trajectory_writer.writeheader()
    chat_writer.writeheader()
    pause_writer.writeheader()
    annotation_writer.writeheader()

    pygame.init()
    pygame.key.set_repeat()
    screen = pygame.display.set_mode((940, 740))
    mode_label = {
        "ppo": "PPO Baseline",
        "random": "Random Partner",
        "subgoal_executor": "Subgoal Executor",
    }[args.ai_mode]
    pygame.display.set_caption(f"DURF Human + {mode_label} [{BUILD_ID}]")
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
    last_ai_subgoal = ""
    last_ai_subgoal_candidates: list[dict] = []
    last_ai_event = ""
    last_predict_ms = 0.0
    last_environment_step_ms = 0.0
    chat_open = False
    chat_input = ""
    chat_messages: list[dict[str, str]] = []
    chat_pending = False
    chat_status = "Set DEEPSEEK_API_KEY to enable model replies."
    chat_result_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    annotation_pending = False
    annotation_prompt_step = 0
    annotation_replay_snapshot: list[dict] = []
    annotation_replay_surfaces: list[pygame.Surface] = []
    recent_steps: list[dict] = []
    recent_step_surfaces: list[pygame.Surface] = []
    last_textinput_at = -1000
    step_interval_ms = max(1, round(1000 / args.step_hz))
    countdown_until = pygame.time.get_ticks() + round(args.start_delay * 1000)
    next_step_at = countdown_until
    last_pause_toggle_at = -PAUSE_DEBOUNCE_MS
    last_layout_switch_at = -LAYOUT_SWITCH_DEBOUNCE_MS
    game_surface = render_game_surface(visualizer, env, episode_reward)

    def serialize_subgoal_candidates(candidates) -> list[dict]:
        return [
            {
                "subgoal": candidate.subgoal,
                "task_score": candidate.task_score,
                "hu_score": candidate.hu_score,
                "final_score": candidate.final_score,
                "reason": candidate.reason,
                "feasible": candidate.feasible,
                "metadata": candidate.metadata,
                "step_cost": candidate.step_cost,
                "hu_supported": candidate.hu_supported,
            }
            for candidate in candidates
        ]

    def live_condition_features(human_action: int) -> dict:
        facts = state_facts(env)
        return extract_condition_features(
            {
                "state_facts": facts,
                "extra": {"state_before": facts},
                "human_action_name": ACTION_NAMES[int(human_action)],
            }
        )

    def apply_hu_shadow_scores(candidates, condition_features: dict) -> None:
        if hu_model is None:
            return
        for candidate in candidates:
            candidate.hu_supported = runtime_hu_has_support(
                hu_model, TASK_DECISION_LEVEL, candidate.subgoal
            )
            try:
                candidate.hu_score = runtime_hu_score(
                    hu_model,
                    TASK_DECISION_LEVEL,
                    args.hu_user_id,
                    condition_features,
                    candidate.subgoal,
                )
            except KeyError:
                candidate.hu_score = 0.0
                warn_unknown_runtime_subgoal(TASK_DECISION_LEVEL, candidate.subgoal)

    def subgoal_executor_action(
        human_action: int,
    ) -> tuple[int, str, list[dict], dict, dict]:
        motion_planner = motion_planners_by_layout[current_layout]
        candidates = rule_teacher_candidates(env.base_env.state, motion_planner, 0)
        attach_step_costs(
            candidates,
            features=task_features_by_layout[current_layout],
            motion_planner=motion_planner,
            player=env.base_env.state.players[0],
            state=env.base_env.state,
            mdp=env.base_env.mdp,
        )
        condition_features = live_condition_features(human_action)
        apply_hu_shadow_scores(candidates, condition_features)
        chosen = choose_task_candidate(
            candidates,
            task_tolerance=args.hu_task_tolerance if args.hu_apply else 0.0,
            step_tolerance=args.hu_step_tolerance if args.hu_apply else None,
        )
        subgoal_name = chosen.subgoal
        planner_action = chosen.action
        target_positions = [
            tuple(position)
            for position in chosen.metadata.get("target_positions") or []
            if isinstance(position, (list, tuple)) and len(position) == 2
        ]
        ai_pos = tuple(env.base_env.state.players[0].position)
        condition_features["ai_current_subgoal"] = subgoal_name
        condition_features["ai_adjacent_to_current_subgoal_target"] = any(
            manhattan(ai_pos, position) == 1
            for position in target_positions
        )
        serialized_candidates = serialize_subgoal_candidates(candidates)
        task_decision = {
            "record_type": "runtime_decision",
            "decision_id": f"task:e{episode}:t{total_step + 1}",
            "decision_level": TASK_DECISION_LEVEL,
            "current_timestep": total_step + 1,
            "condition_at_decision": condition_features,
            "candidate_set": [
                candidate.get("subgoal")
                for candidate in serialized_candidates
            ],
            "candidates": serialized_candidates,
            "selected": subgoal_name,
            "selected_action": int(planner_action),
            "hu_applied": bool(
                args.hu_apply
                and (args.hu_task_tolerance > 0.0 or (args.hu_step_tolerance or 0.0) > 0.0)
            ),
            "hu_task_tolerance": args.hu_task_tolerance,
            "hu_step_tolerance": args.hu_step_tolerance,
        }
        # Every subgoal the generator can produce is executed by the motion
        # planner.  The learned executor used to sit behind this point as a
        # fallback, but the whitelist it guarded had grown to cover the whole
        # vocabulary, so the network was never called in any recorded session
        # while the docs still described the backbone as
        # "state -> subgoal -> learned low-level executor action".  Both methods
        # under comparison execute through the planner, so the planner is the
        # backbone and the claim now matches the code.
        return (
            int(planner_action),
            subgoal_name,
            serialized_candidates,
            condition_features,
            task_decision,
        )

    def motion_target(position, action_index: int):
        action_name = ACTION_NAMES[int(action_index)]
        if action_name == "north":
            return [position[0], position[1] - 1]
        if action_name == "south":
            return [position[0], position[1] + 1]
        if action_name == "east":
            return [position[0] + 1, position[1]]
        if action_name == "west":
            return [position[0] - 1, position[1]]
        return list(position)

    def action_moves(action_index: int) -> bool:
        return ACTION_NAMES[int(action_index)] in {"north", "south", "east", "west"}

    def choose_yield_action(ai_pos, human_pos, human_target) -> int | None:
        valid_positions = set(env.base_env.mdp.get_valid_player_positions())
        avoid = {tuple(human_pos), tuple(human_target)}
        best_action = None
        best_score = None
        for action_index, action_name in enumerate(ACTION_NAMES[:4]):
            candidate = motion_target(ai_pos, action_index)
            candidate_tuple = tuple(candidate)
            if candidate_tuple not in valid_positions or candidate_tuple in avoid:
                continue
            score = (
                abs(candidate[0] - human_pos[0])
                + abs(candidate[1] - human_pos[1])
                + abs(candidate[0] - human_target[0])
                + abs(candidate[1] - human_target[1])
            )
            if best_score is None or score > best_score:
                best_action = action_index
                best_score = score
        return best_action

    def manhattan(a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def detect_subgoal_issue(state, subgoal_name: str) -> str:
        # AI_HELD_DISH_BEFORE_SOUP_READY and AI_HELD_UNNEEDED_INGREDIENT used
        # to live here and be force-corrected downstream in
        # recovery_action_override, unconditionally overriding whatever the
        # task/Hu layer had already chosen. Both states are now covered by
        # real, scored candidates in generate_candidate_subgoals (WAIT_NEAR_POT
        # / PUT_DOWN_OBJECT / WAIT), reproducing the exact same H0 action at
        # hu_task_tolerance=0 -- so Hu can finally have a say here instead of
        # being silently overruled. Only a genuinely stale committed choice
        # (the world changed since this subgoal was picked) still needs a
        # post-hoc recovery step.
        if args.ai_mode != "subgoal_executor":
            return ""
        if subgoal_name in ("PUT_TOMATO_IN_POT", "PUT_ONION_IN_POT"):
            ingredient = "tomato" if subgoal_name == "PUT_TOMATO_IN_POT" else "onion"
            if not pots_needing_ingredient(state, env.base_env.mdp, ingredient):
                return "STALE_PUT_INGREDIENT_SUBGOAL"
        return ""

    def put_down_unneeded_object_action(state) -> int:
        motion_planner = motion_planners_by_layout[current_layout]
        action = first_action_to_feature(
            motion_planner,
            state.players[0],
            counters_for_put_down(
                state, env.base_env.mdp, motion_planner, state.players[0]
            ),
            {state.players[1].position},
        )
        return int(action) if action is not None else STAY

    def recovery_action_override(
        proposed_ai_action: int,
        subgoal_name: str,
    ) -> tuple[int, str]:
        if args.ai_mode != "subgoal_executor":
            return proposed_ai_action, ""
        state = env.base_env.state
        issue = detect_subgoal_issue(state, subgoal_name)
        if issue:
            return put_down_unneeded_object_action(state), issue
        return proposed_ai_action, ""

    def coordination_action_decision(
        proposed_ai_action: int,
        human_action: int,
        subgoal_name: str,
        condition_features: dict,
        task_target_positions: list | None = None,
    ) -> tuple[int, str, dict]:
        if args.ai_mode != "subgoal_executor":
            coordination_controller.clear_if_no_conflict()
            return proposed_ai_action, "", {}

        state = env.base_env.state
        ai_pos = tuple(state.players[0].position)
        human_pos = tuple(state.players[1].position)
        ai_target = tuple(motion_target(ai_pos, proposed_ai_action))
        human_target = tuple(motion_target(human_pos, human_action))
        conflict_type = path_conflict_type(
            ai_pos=ai_pos,
            human_pos=human_pos,
            ai_target=ai_target,
            human_target=human_target,
            ai_is_moving=action_moves(proposed_ai_action),
            human_is_moving=action_moves(human_action),
        )
        if conflict_type is None:
            coordination_controller.clear_if_no_conflict()
            return proposed_ai_action, "", {}

        condition_features = dict(condition_features)
        if action_moves(human_action):
            human_delta = (
                human_target[0] - human_pos[0],
                human_target[1] - human_pos[1],
            )
            route_blocked = human_target == ai_pos or (
                human_target[0] + human_delta[0],
                human_target[1] + human_delta[1],
            ) == ai_pos
        else:
            route_blocked = False
        condition_features["human_trying_to_pass"] = route_blocked
        condition_features["ai_on_human_path"] = route_blocked
        # WAIT / BACK_OFF / REROUTE are execution-level refinements of a
        # single Hu-scored YIELD decision, not options Hu chooses between:
        # REROUTE (keep making task progress on a path that avoids the
        # human) beats BACK_OFF (retreat to the most separating open tile)
        # beats WAIT (stay put) when the stronger options aren't available.
        # This applies to every conflict type -- a dynamic simultaneous-move
        # conflict deserves the same repertoire as a static tile conflict.
        reroute_action = choose_reroute_action(
            motion_planners_by_layout[current_layout],
            env.base_env.state.players[0],
            task_target_positions,
            human_pos,
            human_target,
            proposed_ai_action,
        )
        if reroute_action is not None:
            yield_action = reroute_action
            yield_mode = "reroute"
        else:
            back_off_action = choose_yield_action(ai_pos, human_pos, human_target)
            if back_off_action is not None:
                yield_action = back_off_action
                yield_mode = "back_off"
            else:
                yield_action = STAY
                yield_mode = "wait"
        candidates = build_coordination_candidates(
            proposed_action=proposed_ai_action,
            yield_action=yield_action,
            stay_action=STAY,
            conflict_type=conflict_type,
            ai_adjacent_to_current_subgoal_target=bool(
                condition_features.get(
                    "ai_adjacent_to_current_subgoal_target"
                )
            ),
            yield_mode=yield_mode,
        )

        def coordination_hu_score(option: str) -> float:
            if hu_model is None:
                return 0.0
            return runtime_hu_score(
                hu_model,
                COORDINATION_DECISION_LEVEL,
                args.hu_user_id,
                condition_features,
                option,
            )

        result = coordination_controller.resolve(
            timestep=total_step + 1,
            episode=episode,
            task_subgoal=subgoal_name,
            conflict_type=conflict_type,
            ai_pos=ai_pos,
            human_pos=human_pos,
            candidates=candidates,
            condition_features=condition_features,
            hu_score=coordination_hu_score if hu_model is not None else None,
            hu_tolerance=args.hu_coordination_tolerance,
            apply_hu=args.hu_apply,
            hu_supported=(
                (lambda option: runtime_hu_has_support(hu_model, COORDINATION_DECISION_LEVEL, option))
                if hu_model is not None
                else None
            ),
        )
        if result is None:
            return proposed_ai_action, "", {}
        return result.action, result.event, result.decision

    def hard_safety_guard(action: int) -> tuple[int, str]:
        if 0 <= int(action) < len(ACTION_NAMES):
            return int(action), ""
        return STAY, "AI_SAFETY_REJECTED_INVALID_ACTION"

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
                    horizon=args.horizon,
                )
                if args.ai_mode == "subgoal_executor":
                    motion_planners_by_layout[current_layout] = make_motion_planner(
                        current_layout,
                        args.seed,
                        args.horizon,
                    )
                    task_features_by_layout[current_layout] = build_feature_map(
                        motion_planners_by_layout[current_layout].mdp,
                        motion_planners_by_layout[current_layout],
                    )
            env = envs_by_layout[current_layout]

        ai_obs, _ = env.multi_reset()
        if ai_agent is not None:
            ai_agent.reset()
        if ai_action_prior:
            ai_action_prior.reset()
        coordination_controller.reset()
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

    def log_annotation(content: str) -> None:
        scope_hint = "event"
        text = content.strip()
        if ":" in text:
            prefix = text.split(":", 1)[0].strip().lower()
            if prefix in {"now", "last1", "last3", "last8", "event"}:
                scope_hint = prefix
        write_row(
            annotation_writer,
            annotation_handle,
            {
                "timestamp_utc": utc_timestamp(),
                "episode": episode,
                "episode_step": episode_step,
                "total_step": total_step,
                "layout": current_layout,
                "annotation_scope_hint": scope_hint,
                "annotation_text": content,
                "last_ai_action": last_ai_action,
                "last_ai_action_name": ACTION_NAMES[last_ai_action],
                "episode_reward": episode_reward,
                "state_json": json_dumps(state_facts(env)),
                "recent_replay_json": json_dumps(annotation_replay_snapshot),
            },
        )
        log_chat_message("human_annotation", content)

    def open_annotation_prompt() -> None:
        nonlocal annotation_pending
        nonlocal annotation_prompt_step
        nonlocal annotation_replay_snapshot
        nonlocal annotation_replay_surfaces
        nonlocal chat_input
        nonlocal chat_open
        nonlocal chat_status
        nonlocal paused
        nonlocal pending_interact
        nonlocal pending_motion
        nonlocal recent_step_surfaces
        nonlocal recent_steps

        annotation_pending = True
        annotation_prompt_step = total_step
        annotation_replay_snapshot = list(recent_steps[-args.annotation_interval :])
        annotation_replay_surfaces = [
            surface.copy()
            for surface in recent_step_surfaces[-args.annotation_interval :]
        ]
        paused = True
        chat_open = True
        chat_input = ""
        pending_interact = False
        pending_motion = None
        recent_steps.clear()
        recent_step_surfaces.clear()
        held_motion_actions.clear()
        pygame.key.start_text_input()
        chat_status = (
            "Annotation checkpoint. Write now:/last3:/last8:/event: then feedback. "
            "Enter saves/resumes; Esc skips."
        )

    def close_annotation_prompt(resume: bool) -> None:
        nonlocal annotation_pending
        nonlocal annotation_replay_snapshot
        nonlocal annotation_replay_surfaces
        nonlocal chat_input
        nonlocal chat_open
        nonlocal chat_status
        nonlocal next_step_at
        nonlocal paused

        annotation_pending = False
        annotation_replay_snapshot = []
        annotation_replay_surfaces = []
        chat_input = ""
        chat_open = False
        pygame.key.stop_text_input()
        chat_status = ""
        if resume:
            paused = False
            next_step_at = pygame.time.get_ticks() + step_interval_ms

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
                    "You are a player-facing feedback intake assistant for a "
                    "human-AI Overcooked pilot study. The person chatting with you "
                    "is a player, not a developer. Your job is to record and clarify "
                    "what behavior the AI agent did well or poorly. Do not ask about "
                    "reward functions, observation spaces, training setup, algorithms, "
                    "or implementation details. When feedback is ambiguous, ask one "
                    "short player-answerable question about the recent behavior, such "
                    "as what the AI did, what the player wanted instead, or when it "
                    "happened. Keep replies brief."
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
    print(f"AI action prior: {args.ai_action_prior or 'none'}")
    print(f"Interface build: {BUILD_ID}")
    print(f"Script: {Path(__file__).resolve()}")
    print(f"Session logs: {session_dir}")
    print(f"Human is green; {mode_label} is blue.")
    print(
        f"Timing: {args.step_hz:g} environment steps/s, "
        f"{args.render_fps} render FPS, {args.start_delay:g}s start delay"
    )
    print(
        "Controls: WASD/Arrows=Move, Space=Interact, "
        "P/Tab/F1=Pause, Chat=Language feedback, R=Reset, "
        "1-4/N/M=Switch map, Esc=Quit"
    )
    if args.annotation_mode:
        print(
            "Annotation mode: auto-pauses every "
            f"{args.annotation_interval} environment steps. "
            "Type feedback and press Enter to save/resume; Esc skips."
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
                            if annotation_pending:
                                close_annotation_prompt(resume=True)
                                last_key_debug = "annotation skipped"
                            else:
                                chat_open = False
                                chat_input = ""
                                pygame.key.stop_text_input()
                        elif event.key == pygame.K_RETURN:
                            prompt = chat_input.strip()
                            if annotation_pending:
                                if prompt:
                                    log_annotation(prompt)
                                    chat_messages.append(
                                        {
                                            "role": "user",
                                            "content": f"[Annotation] {prompt}",
                                        }
                                    )
                                    last_key_debug = "annotation saved"
                                else:
                                    last_key_debug = "annotation empty -> skipped"
                                close_annotation_prompt(resume=True)
                            elif prompt and not chat_pending:
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
                        if annotation_pending:
                            continue
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
                prior_action = (
                    ai_action_prior.next_action() if ai_action_prior else None
                )
                if prior_action is None:
                    if args.ai_mode == "random":
                        ai_action_raw = random.randrange(len(ACTION_NAMES))
                        current_ai_subgoal = ""
                        current_ai_condition_features = {}
                        current_ai_subgoal_candidates = []
                        current_task_decision = {}
                    elif args.ai_mode == "subgoal_executor":
                        (
                            ai_action_raw,
                            current_ai_subgoal,
                            current_ai_subgoal_candidates,
                            current_ai_condition_features,
                            current_task_decision,
                        ) = subgoal_executor_action(human_action)
                    else:
                        ai_action_raw = rllib_action_index(ai_agent, env.base_env.state)
                        current_ai_subgoal = ""
                        current_ai_condition_features = {}
                        current_ai_subgoal_candidates = []
                        current_task_decision = {}
                else:
                    ai_action_raw = prior_action
                    current_ai_subgoal = "action_prior"
                    current_ai_condition_features = {}
                    current_ai_subgoal_candidates = []
                    current_task_decision = {}
                last_predict_ms = (time.perf_counter() - predict_started) * 1000
                state_before = state_facts(env)
                recovered_action, recovery_event = recovery_action_override(
                    int(ai_action_raw),
                    current_ai_subgoal,
                )
                (
                    coordinated_action,
                    coordination_event,
                    current_coordination_decision,
                ) = coordination_action_decision(
                    recovered_action,
                    human_action,
                    current_ai_subgoal,
                    current_ai_condition_features,
                    current_task_decision.get("selected_target_positions"),
                )
                ai_action, safety_event = hard_safety_guard(coordinated_action)
                current_ai_event = (
                    safety_event
                    or coordination_event
                    or recovery_event
                )
                if current_task_decision and recovery_event:
                    current_task_decision["recovery_event"] = recovery_event
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
                last_ai_subgoal = current_ai_subgoal
                last_ai_subgoal_candidates = current_ai_subgoal_candidates
                last_ai_event = current_ai_event
                episode_step += 1
                total_step += 1
                episode_reward += float(reward)
                recent_steps.append(
                    {
                        "episode": episode,
                        "episode_step": episode_step,
                        "total_step": total_step,
                        "layout": current_layout,
                        "ai_action": ai_action,
                        "ai_action_name": ACTION_NAMES[ai_action],
                        "ai_subgoal": current_ai_subgoal,
                        "ai_condition_features": current_ai_condition_features,
                        "ai_subgoal_candidates": current_ai_subgoal_candidates,
                        "task_decision": current_task_decision,
                        "coordination_decision": current_coordination_decision,
                        "ai_event": current_ai_event,
                        "human_action": human_action,
                        "human_action_name": ACTION_NAMES[human_action],
                        "reward": float(reward),
                        "episode_reward": episode_reward,
                        "state_before": state_before,
                        "state_after": state_after,
                    }
                )
                if len(recent_steps) > max(args.annotation_interval, 16):
                    del recent_steps[: len(recent_steps) - max(args.annotation_interval, 16)]
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
                        "ai_subgoal": current_ai_subgoal,
                        "ai_condition_features_json": json_dumps(
                            current_ai_condition_features
                        ),
                        "ai_subgoal_candidates_json": json_dumps(
                            current_ai_subgoal_candidates
                        ),
                        "task_decision_json": json_dumps(current_task_decision),
                        "coordination_decision_json": json_dumps(
                            current_coordination_decision
                        ),
                        "ai_event": current_ai_event,
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
                recent_step_surfaces.append(game_surface.copy())
                max_recent_visuals = max(args.annotation_interval, 16)
                if len(recent_step_surfaces) > max_recent_visuals:
                    del recent_step_surfaces[
                        : len(recent_step_surfaces) - max_recent_visuals
                    ]
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
                elif (
                    args.annotation_mode
                    and not annotation_pending
                    and total_step > 0
                    and total_step % args.annotation_interval == 0
                    and total_step != annotation_prompt_step
                ):
                    open_annotation_prompt()

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
                f"Blue: {mode_label} | Green: YOU | Episode {episode} | Step {episode_step} | "
                f"Reward {episode_reward:.1f} | {run_status}"
            )
            controls = (
                "WASD/Arrows Move | Space Interact | P/Tab/F1 Pause | "
                "Chat | R Reset | N/M Map | Esc Quit"
            )
            if annotation_pending:
                feedback_status = (
                    "ANNOTATION: type what AI should do, Enter save/resume, Esc skip"
                )
            elif args.annotation_mode:
                feedback_status = (
                    f"Annotation mode: auto-pause every {args.annotation_interval} steps"
                )
            else:
                feedback_status = "Language feedback: click Chat, type comment, press Enter"
            timing_status = (
                f"{BUILD_ID} | Decision rate: {args.step_hz:g}/s | "
                f"AI predict: {last_predict_ms:.1f} ms | "
                f"Environment: {last_environment_step_ms:.1f} ms"
            )
            subgoal_status = f"AI subgoal: {last_ai_subgoal or 'n/a'}"
            if last_ai_event:
                subgoal_status = f"{subgoal_status} | event: {last_ai_event}"
            input_status = f"{subgoal_status} | Last input: {last_key_debug}"
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
                if annotation_pending:
                    render_annotation_panel(
                        screen,
                        font,
                        small_font,
                        annotation_replay_snapshot,
                        annotation_replay_surfaces,
                        chat_input,
                        chat_status,
                    )
                else:
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
        annotation_handle.flush()
        trajectory_handle.close()
        chat_handle.close()
        pause_handle.close()
        annotation_handle.close()
        for cached_env in set(envs_by_layout.values()):
            cached_env.close()
        pygame.key.stop_text_input()
        pygame.quit()

    print(f"Trajectory: {trajectory_path}")
    print(f"Chat messages: {chat_path}")
    print(f"Pause events: {pause_path}")
    print(f"Annotations: {annotation_path}")
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
