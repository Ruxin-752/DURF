"""Play Overcooked as the green human beside the blue H0 or PPO agent."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# A previously editable-installed Overcooked package may point at a moved
# checkout. Prefer this workspace's source and bundled layouts for the GUI.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_SRC = _WORKSPACE_ROOT / "src"
_ADAPTED_FEEDBACK_ROOT = (
    _WORKSPACE_ROOT
    / "baselines"
    / "baseline_b_linguistic_feedback"
    / "adapted_overcooked"
)
for _import_root in (_WORKSPACE_ROOT, _WORKSPACE_SRC, _ADAPTED_FEEDBACK_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

import pygame
from durf.group_a.pixel_kitchen_visualizer import PixelKitchenVisualizer as StateVisualizer

from durf.baseline.h0_planner import (
    H0_MODEL_SUBGOALS,
    SUBGOALS,
    first_action_to_feature,
    make_motion_planner,
    pots_needing_ingredient,
    rule_teacher_decision,
)
from durf.baseline.runtime import (
    DEFAULT_LAYOUT_NAME,
    DEFAULT_PLAYABLE_LAYOUTS,
    REPO_ROOT,
    RING_TOMATO_ONION_H0_LAYOUT,
    default_agent_for_layout,
    ensure_agent_layout,
    filter_compatible_layouts,
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

# The UI is rendered on one fixed logical canvas. ``pygame.SCALED`` maps this
# canvas to the physical window, so resizing or high-DPI scaling cannot make
# independently positioned panels drift over one another.
WINDOW_SIZE = (1440, 900)
GAME_VIEW_RECT = pygame.Rect(24, 24, 950, 700)
GAME_INFO_RECT = pygame.Rect(24, 740, 950, 136)
GAME_STATUS_RECT = pygame.Rect(42, 753, 914, 38)
GAME_TIMING_RECT = pygame.Rect(42, 800, 914, 32)
GAME_CONTROLS_RECT = pygame.Rect(42, 842, 914, 24)
SIDEBAR_RECT = pygame.Rect(994, 24, 422, 852)
CHAT_PANEL_RECT = pygame.Rect(1010, 128, 390, 668)
CHAT_ANALYSIS_RECT = pygame.Rect(1024, 210, 362, 132)
CHAT_HISTORY_RECT = pygame.Rect(1018, 370, 374, 218)
CHAT_STATUS_RECT = pygame.Rect(1024, 598, 362, 42)
CHAT_INPUT_RECT = pygame.Rect(1024, 650, 362, 132)
CHAT_BUTTON = pygame.Rect(1010, 812, 185, 48)
PAUSE_BUTTON = pygame.Rect(1215, 812, 185, 48)
MAX_CHAT_INPUT_CHARS = 1000
PAUSE_DEBOUNCE_MS = 300
LAYOUT_SWITCH_DEBOUNCE_MS = 300
BUILD_ID = "human-feedback-ui-v7"

FEEDBACK_FORM_KEYS = ("evaluative", "imperative", "descriptive")
FEEDBACK_FORM_LABELS = {
    "evaluative": "Evaluative / 评价",
    "imperative": "Imperative / 指令",
    "descriptive": "Descriptive / 描述",
}
FEEDBACK_FORM_COLORS = {
    "evaluative": (215, 103, 151),
    "imperative": (240, 172, 72),
    "descriptive": (78, 174, 224),
}
COMFORT_FEEDBACK_MODES = (
    "frozen",
    "route1-literal",
    "route1-pseudopragmatic",
    "route2",
)
DEFAULT_H0_EXECUTOR = (
    REPO_ROOT
    / "models"
    / "subgoal_executors"
    / "h0_rule_executor_v5"
    / "executor.keras"
)
DEFAULT_LIVE_STATE_DIR = (
    REPO_ROOT
    / "baselines"
    / "baseline_b_linguistic_feedback"
    / "adapted_overcooked"
    / "outputs"
    / "live"
)
DEFAULT_LEARNER_STATE = DEFAULT_LIVE_STATE_DIR / "route1_state.json"
DEFAULT_ROUTE2_LEARNER_STATE = DEFAULT_LIVE_STATE_DIR / "route2_state_v5.json"
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


def default_learner_state_for_teacher(
    feedback_mode: str,
    teacher_id: str,
    *,
    session_id: str,
) -> Path:
    """Return a stable, privacy-preserving learner state namespace."""

    normalized_teacher = str(teacher_id or "").strip()
    if normalized_teacher and normalized_teacher.lower() != "anonymous":
        identity = "teacher_" + hashlib.sha256(
            normalized_teacher.encode("utf-8")
        ).hexdigest()[:12]
    else:
        identity = "session_" + str(session_id).strip()[:12]
    template = (
        DEFAULT_ROUTE2_LEARNER_STATE
        if feedback_mode == "route2"
        else DEFAULT_LEARNER_STATE
    )
    return template.with_name(f"{template.stem}_{identity}{template.suffix}")
WINDOWS_KEYBOARD_AVAILABLE = os.name == "nt"
if WINDOWS_KEYBOARD_AVAILABLE:
    _get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
else:
    _get_async_key_state = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ai-mode",
        choices=("ppo", "subgoal_executor", "comfort_subgoal"),
        default="comfort_subgoal",
        help=(
            "Blue-agent backend. 'subgoal_executor' is the bundled H0 rule "
            "policy; 'comfort_subgoal' is H0 with subgoals re-ranked by the "
            "reward learned from language feedback (Path A)."
        ),
    )
    parser.add_argument(
        "--comfort-weights",
        type=Path,
        default=None,
        help=(
            "Optional initial reward weights. Route 2 defaults to a neutral "
            "zero vector so H0 acts until language supplies a preference."
        ),
    )
    parser.add_argument(
        "--route2-model",
        type=Path,
        default=None,
        help=(
            "Optional Route 2 checkpoint or ensemble manifest. The default is "
            "the selected paper-style ten-fold complete-reward ensemble when "
            "available."
        ),
    )
    parser.add_argument(
        "--comfort-lambda",
        type=float,
        choices=(1.0,),
        default=1.0,
        help="Paper-aligned score is fixed to w dot phi; only 1.0 is valid.",
    )
    parser.add_argument(
        "--comfort-feedback-mode",
        choices=COMFORT_FEEDBACK_MODES,
        default="route2",
        help=(
            "How chat feedback changes comfort weights. Route 1 modes run the "
            "paper-style classify -> ground -> Bayesian update loop; route2 "
            "uses the ten-fold neural reward estimate and the paper's Gaussian "
            "belief update; frozen records but ignores feedback."
        ),
    )
    parser.add_argument(
        "--route1-prior",
        choices=("zero", "frozen"),
        default="zero",
        help=(
            "Route 1 Gaussian prior mean: paper-faithful zero, or the exported "
            "frozen weights as a warm start."
        ),
    )
    parser.add_argument(
        "--route1-lookback",
        type=int,
        default=25,
        help=(
            "Recent selected-subgoal decisions used to ground evaluative Route "
            "1 feedback as a trajectory reference."
        ),
    )
    parser.add_argument(
        "--route2-blend",
        type=float,
        default=None,
        help=(
            "Legacy EMA alpha for Route 2. Omit it to use the paper's Gaussian "
            "belief update."
        ),
    )
    parser.add_argument(
        "--route2-observation-precision",
        type=float,
        default=2.0,
        help="Precision of the ensemble reward observation (paper default: 2).",
    )
    parser.add_argument(
        "--human-feedback-precision",
        type=float,
        default=4.0,
        help="Route 1 precision multiplier for live human comments.",
    )
    parser.add_argument(
        "--learner-state",
        type=Path,
        default=None,
        help=(
            "Versioned learner checkpoint written after each accepted comment. "
            "Defaults to a mode-specific Route 1 or Route 2 state file."
        ),
    )
    parser.add_argument(
        "--resume-learner-state",
        action="store_true",
        help="Resume the exact Route 1 posterior or Route 2 weights from --learner-state.",
    )

    parser.add_argument(
        "--agent",
        default=None,
        help="RLlib agent name or path; defaults based on the requested layout.",
    )
    parser.add_argument("--layout", default=None)
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=list(DEFAULT_PLAYABLE_LAYOUTS),
        help="Layouts available for hot switching with 1-4 or N/M.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument(
        "--subgoal-executor",
        type=Path,
        default=DEFAULT_H0_EXECUTOR,
        help="Bundled Keras H0 executor used by --ai-mode subgoal_executor.",
    )
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
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "human_ai_sessions"),
    )
    parser.add_argument(
        "--teacher-id",
        default=os.getenv("DURF_TEACHER_ID", "anonymous"),
        help=(
            "Stable pseudonymous participant ID used for provenance, a "
            "participant-specific learner-state namespace, and future "
            "teacher-disjoint evaluation; do not put a real name here."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Exit after this many total steps; useful for smoke tests.",
    )
    args = parser.parse_args()
    if args.layout is None:
        args.layout = (
            RING_TOMATO_ONION_H0_LAYOUT
            if args.ai_mode in ("subgoal_executor", "comfort_subgoal")
            else DEFAULT_LAYOUT_NAME
        )
    if args.ai_mode == "ppo" and args.agent is None:
        args.agent = default_agent_for_layout(args.layout)
    return args


def load_h0_executor(path: str | Path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"H0 subgoal executor not found: {path}")
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for --ai-mode subgoal_executor"
        ) from exc
    model = load_model(path, compile=False)
    expected_inputs = [(None, 10, 6, 26), (None, len(H0_MODEL_SUBGOALS))]
    input_shapes = [tuple(shape) for shape in model.input_shape]
    output_shape = tuple(model.output_shape)
    if input_shapes != expected_inputs or output_shape != (None, len(ACTION_NAMES)):
        raise ValueError(
            f"Unexpected H0 model contract: inputs={input_shapes}, "
            f"output={output_shape}"
        )
    return model


def h0_executor_action(base_env, model, motion_planner) -> tuple[int, str]:
    """Reproduce the bundled H0 playtest policy from commit 4ccb410."""

    if model is None:
        raise RuntimeError("H0 bundled model was not loaded")
    subgoal, planner_action = rule_teacher_decision(
        base_env.state,
        motion_planner,
        player_index=0,
    )
    return int(planner_action), subgoal


def motion_target(position, action_index: int) -> list[int]:
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


def choose_yield_action(base_env, ai_pos, human_pos, human_target) -> int | None:
    valid_positions = set(base_env.mdp.get_valid_player_positions())
    avoid = {tuple(human_pos), tuple(human_target)}
    best_action = None
    best_score = None
    for action_index in range(4):
        candidate = motion_target(ai_pos, action_index)
        if tuple(candidate) not in valid_positions or tuple(candidate) in avoid:
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


def detect_subgoal_issue(base_env, state, subgoal_name: str) -> str:
    held_name = getattr(state.players[0].held_object, "name", None)
    if held_name == "dish" and not base_env.mdp.get_ready_pots(
        base_env.mdp.get_pot_states(state)
    ):
        return "AI_HELD_DISH_BEFORE_SOUP_READY"
    if held_name in ("tomato", "onion") and not pots_needing_ingredient(
        state,
        base_env.mdp,
        held_name,
    ):
        return "AI_HELD_UNNEEDED_INGREDIENT"
    if subgoal_name in ("PUT_TOMATO_IN_POT", "PUT_ONION_IN_POT"):
        ingredient = (
            "tomato" if subgoal_name == "PUT_TOMATO_IN_POT" else "onion"
        )
        if not pots_needing_ingredient(state, base_env.mdp, ingredient):
            return "STALE_PUT_INGREDIENT_SUBGOAL"
    return ""


def empty_counter_locations(base_env, state, motion_planner) -> list[tuple[int, int]]:
    mdp = base_env.mdp
    if hasattr(mdp, "get_counter_locations"):
        counters = list(mdp.get_counter_locations())
    else:
        valid_positions = set(mdp.get_valid_player_positions())
        counters = []
        for y, row in enumerate(getattr(mdp, "terrain_mtx", [])):
            for x, terrain in enumerate(row):
                pos = (x, y)
                if pos in valid_positions or terrain != "X":
                    continue
                adjacent = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                if any(candidate in valid_positions for candidate in adjacent):
                    counters.append(pos)
    feature_positions = set()
    for getter_name in (
        "get_pot_locations",
        "get_serving_locations",
        "get_dish_dispenser_locations",
        "get_tomato_dispenser_locations",
        "get_onion_dispenser_locations",
    ):
        getter = getattr(mdp, getter_name, None)
        if getter is not None:
            feature_positions.update(getter())
    occupied = set(getattr(state, "objects", {}).keys())
    motion_goal_positions = set(getattr(motion_planner, "motion_goals_for_pos", {}))
    return [
        tuple(position)
        for position in counters
        if tuple(position) not in feature_positions
        and tuple(position) not in occupied
        and tuple(position) in motion_goal_positions
    ]


def put_down_unneeded_object_action(base_env, state, motion_planner) -> int:
    action = first_action_to_feature(
        motion_planner,
        state.players[0],
        empty_counter_locations(base_env, state, motion_planner),
        {state.players[1].position},
    )
    return int(action) if action is not None else STAY


def cooperative_action_wrapper(
    base_env,
    motion_planner,
    proposed_ai_action: int,
    human_action: int,
    subgoal_name: str,
) -> tuple[int, str]:
    state = base_env.state
    ai_pos = list(state.players[0].position)
    human_pos = list(state.players[1].position)
    ai_target = motion_target(ai_pos, proposed_ai_action)
    human_target = motion_target(human_pos, human_action)
    issue = detect_subgoal_issue(base_env, state, subgoal_name)
    if issue:
        if issue == "AI_HELD_DISH_BEFORE_SOUP_READY":
            pot_states = base_env.mdp.get_pot_states(state)
            if pot_states.get("cooking"):
                return STAY, issue
        return (
            put_down_unneeded_object_action(base_env, state, motion_planner),
            issue,
        )
    if action_moves(proposed_ai_action) and ai_target == human_pos:
        return STAY, "AI_WAITED_FOR_HUMAN_BLOCK"
    if action_moves(human_action) and human_target == ai_pos:
        yield_action = choose_yield_action(
            base_env,
            ai_pos,
            human_pos,
            human_target,
        )
        if yield_action is not None:
            return yield_action, "AI_YIELDED_TO_HUMAN_PATH"
        return STAY, "AI_COULD_NOT_YIELD_TO_HUMAN_PATH"
    return proposed_ai_action, ""


def resolve_executed_ai_action(
    ai_mode: str,
    base_env,
    motion_planner,
    proposed_ai_action: int,
    human_action: int,
    subgoal_name: str,
) -> tuple[int, str]:
    """Keep paper-aligned comfort actions unchanged by legacy rule wrappers."""

    if ai_mode == "subgoal_executor":
        return cooperative_action_wrapper(
            base_env,
            motion_planner,
            proposed_ai_action,
            human_action,
            subgoal_name,
        )
    return int(proposed_ai_action), ""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_session_dir(output_dir: str | Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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


def apply_live_feedback_update(
    comfort_agent,
    prompt: str,
    feedback_event_id: str,
    feedback_form_prediction: dict | None = None,
) -> dict:
    """Forward one UI ``f_G`` result into the idempotent online learner."""

    return comfort_agent.update_from_feedback(
        prompt,
        feedback_event_id=feedback_event_id,
        feedback_form_prediction=feedback_form_prediction,
        source="human_live",
        confidence=1.0,
    )


def feedback_update_record(
    trace: dict,
    *,
    timestamp_utc: str,
    episode: int,
    episode_step: int,
    total_step: int,
    session_id: str,
    teacher_id: str,
    feedback_event_id: str,
) -> dict:
    """Attach stable session provenance to a learner audit trace."""

    record = {
        "timestamp_utc": timestamp_utc,
        "episode": episode,
        "episode_step": episode_step,
        "total_step": total_step,
        "session_id": session_id,
        "teacher_id": teacher_id,
        **trace,
        # The UI-generated identifier is authoritative even if a future learner
        # trace accidentally omits or changes it.
        "feedback_event_id": feedback_event_id,
    }
    record.setdefault("source", "human_live")
    record["learning_applied"] = record.get("status") == "updated"
    record["online_learning_enabled"] = record.get("mode") not in {
        None,
        "frozen",
        "ppo",
        "subgoal_executor",
    }
    if record.get("mode") == "frozen" and record.get("status") == "ignored":
        record.setdefault("rejection_reason", "frozen_control_mode")
    return record


def online_learning_enabled(ai_mode: str, feedback_mode: str | None) -> bool:
    """Return whether language feedback can update the live learner."""

    return ai_mode == "comfort_subgoal" and feedback_mode != "frozen"


def feedback_mode_banner(ai_mode: str, feedback_mode: str | None) -> str:
    if online_learning_enabled(ai_mode, feedback_mode):
        return f"{str(feedback_mode).upper()} | ONLINE LEARNING"
    if ai_mode == "comfort_subgoal" and feedback_mode == "frozen":
        return "FROZEN CONTROL | RECORD ONLY"
    return "RECORD ONLY | NO ONLINE LEARNING"


def feedback_supervision_metadata(ai_mode: str, feedback_mode: str | None) -> dict:
    if not online_learning_enabled(ai_mode, feedback_mode):
        note = (
            "Feedback is logged for the frozen control and never updates the learner."
            if ai_mode == "comfort_subgoal" and feedback_mode == "frozen"
            else "Feedback is logged only; this AI mode has no online learner."
        )
        return {
            "kind": "record_only_no_learning",
            "teacher_reward_weights": None,
            "reward_config_id": None,
            "note": note,
        }
    return {
        "kind": "unlabeled_human_language",
        "teacher_reward_weights": None,
        "reward_config_id": None,
        "note": (
            "Natural language updates the online posterior, but is not a supervised "
            "full-reward target without an independent label."
        ),
    }


def configured_learner_state_path(
    feedback_mode: str,
    requested_path: str | Path | None,
    teacher_id: str,
    *,
    session_id: str,
) -> Path | None:
    """Frozen controls have no resumable live learner state."""

    if feedback_mode == "frozen":
        return None
    if requested_path is not None:
        return Path(requested_path)
    return default_learner_state_for_teacher(
        feedback_mode,
        teacher_id,
        session_id=session_id,
    )


def validate_feedback_configuration(
    ai_mode: str,
    feedback_mode: str | None,
    resume_learner_state: bool,
) -> None:
    if (
        ai_mode == "comfort_subgoal"
        and feedback_mode == "frozen"
        and resume_learner_state
    ):
        raise ValueError(
            "--resume-learner-state is not supported in frozen control mode; "
            "use --comfort-feedback-mode route2 for online learning"
        )


def no_learning_reply(ai_mode: str, feedback_mode: str | None) -> str:
    if ai_mode == "comfort_subgoal" and feedback_mode == "frozen":
        return (
            "Your feedback was recorded in frozen control mode, so reward "
            "weights and agent behavior were not updated. Restart with "
            "--comfort-feedback-mode route2 to enable online learning."
        )
    return (
        "Your feedback was recorded. This AI mode does not update reward "
        "weights or agent behavior online."
    )


_DIRECT_COMMAND_HEAD = re.compile(
    r"^(?:please\s+)?(?:(?:do\s+not|don't|never)\s+)?"
    r"(?:avoid|bring|circle|clear|do|fetch|focus|get|give|go|grab|head|hold|keep|"
    r"leave|make|move|pick|place|put|return|serve|set|stand|start|stay|step|"
    r"stop|switch|take|transfer|use|wait|walk)\b",
    re.IGNORECASE,
)
_DIRECT_REQUEST_HEAD = re.compile(
    r"^(?:(?:can|could|would|will)\s+you\b|why\s+(?:don't|do\s+not|can't|"
    r"cannot|won't)\s+you\b)",
    re.IGNORECASE,
)
_SECOND_PERSON_DIRECTIVE = re.compile(
    r"\byou\s+(?:should|need\s+to|must|have\s+to|ought\s+to)\b",
    re.IGNORECASE,
)
_COUNTERFACTUAL_EVALUATION = re.compile(
    r"\b(?:should|could|would)\s+(?:not\s+)?have\b|"
    r"\b(?:wish|regret)\b",
    re.IGNORECASE,
)
_PAST_OR_OUTCOME_CUE = re.compile(
    r"\b(?:last|previous|just|that\s+(?:move|route|choice|attempt)|turned\s+out|"
    r"was|were|hurt|cost|slowed|delayed|wasted)\b",
    re.IGNORECASE,
)
_EVALUATION_CUE = re.compile(
    r"\b(?:bad|good|great|poor|wrong|right|excellent|efficient|inefficient|"
    r"helpful|unhelpful|unnecessary|satisfied|happy|unhappy|worse|better)\b",
    re.IGNORECASE,
)
_STATE_CLAUSE_HEAD = re.compile(
    r"^(?:there\s+(?:is|are)|the\b|this\b|that\b|an?\b|you\s+(?:are|can)\b|"
    r"we\s+have\b|when\b)",
    re.IGNORECASE,
)


def split_feedback_form_phrases(text: str, *, limit: int = 4) -> list[str]:
    """Split a chat message into clauses shared by the UI and Route 1.

    The complete, unmodified message still goes to Route 1/Route 2.  Route 1
    consumes these exact clause predictions so a mixed message is not forced
    into one three-way label.
    """

    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    clauses: list[str] = []
    clause_start = (
        r"(?:you\s+(?:should|need\s+to|must|have\s+to|can\s+)|please\b|"
        r"(?:can|could|would|will)\s+you\b|why\s+(?:don't|do\s+not)\s+you\b|"
        r"(?:don't|do\s+not|never|stop|move|take|get|put|serve|grab|fetch|pick|"
        r"bring|clear|avoid|the|this|that|there|when|what|why)\b)"
    )
    boundary = re.compile(rf"\s*[,;:]\s+(?={clause_start})", re.IGNORECASE)
    # Without punctuation, split only before an unmistakable multiword
    # directive.  Broad heads such as "put" or "the" would otherwise split a
    # perfectly normal clause at nearly every word.
    embedded_directive = re.compile(
        r"\s+(?=(?:you\s+(?:should|need\s+to|must|have\s+to)|please\b|"
        r"(?:can|could|would|will)\s+you\b|why\s+(?:don't|do\s+not)\s+you\b|"
        r"don't\b|do\s+not\b|never\b))",
        re.IGNORECASE,
    )
    for sentence in sentences:
        for piece in boundary.split(sentence):
            # Human feedback often omits punctuation before "you should ...".
            # Split only when a meaningful clause already precedes the cue.
            subpieces = embedded_directive.split(piece)
            if len(subpieces) > 1 and (
                len(subpieces[0].strip()) < 8
                or len(subpieces[0].strip().split()) <= 3
            ):
                subpieces = [piece]
            for clause in subpieces:
                clause = clause.strip(" ,;:")
                if clause:
                    clauses.append(clause)
                if len(clauses) >= limit:
                    return clauses
    return clauses or [cleaned]


def _syntax_feedback_form(phrase: str) -> dict | None:
    """Return an explainable rule match without inventing a probability.

    These rules are useful as a second opinion and as an offline fallback, but
    their historic 0.97/0.91/0.88 constants were never calibrated.  A rule
    match therefore carries a label and explanation only.
    """

    lowered = phrase.strip().lower()
    if not lowered:
        return None
    if _COUNTERFACTUAL_EVALUATION.search(lowered):
        label, reason = "Evaluative", "past/counterfactual evaluation"
    elif (
        _DIRECT_COMMAND_HEAD.search(lowered)
        or _DIRECT_REQUEST_HEAD.search(lowered)
        or _SECOND_PERSON_DIRECTIVE.search(lowered)
        or (
            lowered.endswith("?")
            and re.match(
                r"^you\s+can\s+(?:go|grab|take|get|put|serve|move|fetch|pick|"
                r"bring|clear|avoid|leave|make|place|walk)\b",
                lowered,
            )
        )
    ):
        label, reason = "Imperative", "request/instruction syntax"
    elif (
        (_PAST_OR_OUTCOME_CUE.search(lowered) and _EVALUATION_CUE.search(lowered))
        or re.search(r"\bwhich\s+(?:is|was)\s+(?:bad|good|wrong|helpful)\b", lowered)
    ):
        label, reason = "Evaluative", "judgment/outcome syntax"
    elif _STATE_CLAUSE_HEAD.search(lowered) or re.match(r"^[a-z]+ing\b", lowered):
        label, reason = "Descriptive", "state/property syntax"
    else:
        return None
    return {
        "feedback_type": label.lower(),
        "label": label,
        "confidence": None,
        "probabilities": {},
        "classifier": "speech_form_syntax_v2",
        "abstained": False,
        "score_kind": "rule_match_no_probability",
        "calibrated": False,
        "reason": reason,
    }


def normalize_feedback_probabilities(value: object) -> dict[str, float]:
    """Return a finite three-class distribution that sums to one."""

    if not isinstance(value, dict):
        return {}
    clean: dict[str, float] = {}
    for key in FEEDBACK_FORM_KEYS:
        try:
            score = float(value.get(key, 0.0))
        except (TypeError, ValueError):
            score = 0.0
        clean[key] = score if math.isfinite(score) and score >= 0 else 0.0
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {key: clean[key] / total for key in FEEDBACK_FORM_KEYS}


def _classify_feedback_form_phrase_for_ui(text: str) -> dict:
    syntax = _syntax_feedback_form(text)
    try:
        from src.feedback_form_classifier import predict_feedback_form

        raw = dict(predict_feedback_form(text))
    except Exception as exc:
        if syntax is not None:
            return {
                **syntax,
                "text": text,
                "model_error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "feedback_type": "unknown",
            "label": "Unknown",
            "confidence": None,
            "probabilities": {},
            "classifier": "unavailable",
            "abstained": True,
            "score_kind": "unavailable",
            "calibrated": False,
            "error": f"{type(exc).__name__}: {exc}",
            "text": text,
        }

    classifier = str(raw.get("classifier") or "unknown")
    # A missing artifact falls back to hand-authored scores. Do not show those
    # numbers as confidence. Prefer an explainable syntax label when possible.
    if classifier == "fallback_rules":
        if syntax is not None:
            return {
                **syntax,
                "text": text,
                "fallback_audit_feedback_type": raw.get("feedback_type"),
            }
        feedback_type = str(raw.get("feedback_type") or "unknown").lower()
        canonical = {
            "evaluative": "Evaluative",
            "imperative": "Imperative",
            "descriptive": "Descriptive",
        }
        return {
            "feedback_type": feedback_type,
            "label": canonical.get(feedback_type, "Uncertain"),
            "confidence": None,
            "probabilities": {},
            "classifier": classifier,
            "abstained": True,
            "score_kind": "heuristic_no_probability",
            "calibrated": False,
            "reason": "model artifact unavailable; heuristic label only",
            "text": text,
        }

    feedback_type = str(raw.get("feedback_type") or "unknown").lower()
    canonical = {
        "evaluative": "Evaluative",
        "imperative": "Imperative",
        "descriptive": "Descriptive",
    }
    probabilities = normalize_feedback_probabilities(raw.get("probabilities"))
    confidence = probabilities.get(feedback_type)
    if confidence is None:
        try:
            fallback_confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            fallback_confidence = math.nan
        confidence = fallback_confidence if math.isfinite(fallback_confidence) else None
    syntax_label = syntax.get("label") if syntax is not None else None
    model_label = str(
        raw.get("classification_label") or canonical.get(feedback_type, "Unknown")
    )
    calibrated = bool(raw.get("calibrated", False))
    return {
        "feedback_type": feedback_type,
        "label": model_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "classifier": classifier,
        "abstained": bool(raw.get("abstained", False)),
        "confidence_threshold": raw.get("confidence_threshold"),
        "score_kind": (
            "model_probability_calibrated"
            if calibrated
            else "model_probability_uncalibrated"
        ),
        "calibrated": calibrated,
        "calibration_version": raw.get("calibration_version"),
        "classification_target": "paper_feedback_strategy/reference_collapsed",
        "model_path": raw.get("model_path"),
        "syntax_label": syntax_label,
        "syntax_agreement": (
            syntax_label == model_label if syntax_label is not None else None
        ),
        "text": text,
        "reason": "learned three-class model; syntax is an audit-only second opinion",
    }


def classify_feedback_form_for_ui(text: str) -> dict:
    """Return the phrase-level ``f_G`` result shared with Route 1."""

    phrases = [
        _classify_feedback_form_phrase_for_ui(phrase)
        for phrase in split_feedback_form_phrases(text)
    ]
    if not phrases:
        return {
            "feedback_type": "unknown",
            "label": "Unknown",
            "confidence": None,
            "probabilities": {},
            "classifier": "unavailable",
            "abstained": True,
            "score_kind": "unavailable",
            "calibrated": False,
            "phrases": [],
        }
    predicted_labels = {
        str(row.get("label") or "Unknown")
        for row in phrases
        if row.get("label") not in (None, "Unknown", "Uncertain")
    }
    if len(predicted_labels) > 1:
        label, feedback_type = "Mixed", "mixed"
    elif len(predicted_labels) == 1:
        label = next(iter(predicted_labels))
        feedback_type = label.lower()
    else:
        label, feedback_type = "Uncertain", "unknown"
    distributions = [
        normalize_feedback_probabilities(row.get("probabilities"))
        for row in phrases
    ]
    distributions = [row for row in distributions if row]
    probabilities = (
        {
            key: sum(row[key] for row in distributions) / len(distributions)
            for key in FEEDBACK_FORM_KEYS
        }
        if distributions
        else {}
    )
    confidence = probabilities.get(feedback_type) if feedback_type in probabilities else None
    thresholds = [
        float(row["confidence_threshold"])
        for row in phrases
        if row.get("confidence_threshold") is not None
    ]
    aggregate_calibrated = (
        len(distributions) == len(phrases)
        and bool(distributions)
        and all(bool(row.get("calibrated")) for row in phrases)
    )
    calibration_versions = {
        str(row["calibration_version"])
        for row in phrases
        if row.get("calibration_version")
    }
    return {
        "feedback_type": feedback_type,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_label": (
            max(probabilities, key=probabilities.get) if probabilities else feedback_type
        ),
        "classifier": "phrase_level_feedback_strategy",
        "abstained": all(bool(row.get("abstained")) for row in phrases),
        "has_uncertain_phrase": any(row.get("abstained") for row in phrases),
        "confidence_threshold": max(thresholds) if thresholds else None,
        "score_kind": (
            "mean_clause_model_probability_calibrated"
            if len(distributions) > 1 and aggregate_calibrated
            else "mean_clause_model_probability_uncalibrated"
            if len(distributions) > 1
            else "model_probability_calibrated"
            if distributions and aggregate_calibrated
            else "model_probability_uncalibrated"
            if distributions
            else "rule_match_no_probability"
        ),
        "calibrated": aggregate_calibrated,
        "calibration_version": (
            next(iter(calibration_versions))
            if len(calibration_versions) == 1
            else "mixed"
            if calibration_versions
            else None
        ),
        "classification_target": "paper_feedback_strategy/reference_collapsed",
        "ui_only": False,
        "route1_control_input": True,
        "phrases": phrases,
    }


def _classifier_display_name(classifier: str) -> str:
    names = {
        "speech_form_syntax_v1": "legacy syntax rules",
        "speech_form_syntax_v2": "syntax rule",
        "phrase_level_speech_form_hybrid": "legacy phrase hybrid",
        "phrase_level_feedback_strategy": "phrase-level strategy model",
        "tfidf_logistic_regression": "TF-IDF LR",
        "tfidf_logistic_regression_low_confidence": "TF-IDF LR",
        "fallback_rules": "fallback rules",
        "unavailable": "classifier unavailable",
    }
    return names.get(str(classifier), str(classifier).replace("_", " "))


def feedback_update_metadata(trace: dict, classification: dict) -> str:
    """Build compact factual metadata rendered below an agent answer."""

    mode = str(trace.get("mode") or "unknown")
    phrases = list(classification.get("phrases") or [])
    role_note = (
        "Route 1 f_G learning gate"
        if mode.startswith("route1")
        else "UI audit only, not learning gate"
    )
    lines = [f"Strategy: {classification.get('label', 'Unknown')} | {role_note}"]
    for index, phrase in enumerate(phrases[:3], start=1):
        score = phrase.get("confidence")
        score_text = (
            f"model score {float(score) * 100:.0f}% (uncalibrated)"
            if score is not None and phrase.get("probabilities")
            else "rule match; no numeric probability"
        )
        uncertain = " uncertain" if phrase.get("abstained") else ""
        lines.append(
            f"P{index} {phrase.get('label', 'Unknown')} | {score_text}{uncertain} "
            f"[{_classifier_display_name(str(phrase.get('classifier', 'unknown')))}]"
        )
    if not phrases:
        score = classification.get("confidence")
        score_text = (
            f"model score {float(score) * 100:.1f}% (uncalibrated)"
            if score is not None and classification.get("probabilities")
            else "numeric score unavailable"
        )
        lines.append(
            f"Prediction {classification.get('label', 'Unknown')} | {score_text} | "
            f"{_classifier_display_name(str(classification.get('classifier', 'unknown')))}"
        )

    status = str(trace.get("status") or "unknown")
    update_rule = str(trace.get("update_rule") or trace.get("interpretation") or mode)
    update_rule = {
        "paper_gaussian_precision": "Gaussian precision update",
        "paper_independent_gaussian": "Gaussian precision update",
        "legacy_exponential_blend": "legacy EMA update",
        "neural": "neural reward inference",
    }.get(update_rule, update_rule.replace("_", " "))
    before_subgoal = trace.get("before_subgoal")
    after_subgoal = trace.get("after_subgoal")
    ranking = (
        f" | {before_subgoal or 'unknown'} -> "
        f"{after_subgoal or before_subgoal or 'unknown'}"
        if before_subgoal or after_subgoal
        else ""
    )
    if status == "updated":
        lines.append(f"Updated: {update_rule}{ranking}")
    else:
        lines.append(f"Update: {status.upper()} | {mode} | {update_rule}{ranking}")

    changes = list(trace.get("top_changes") or [])[:2]
    if changes:
        summaries = []
        for change in changes:
            feature = str(change.get("feature") or "unknown_feature")
            before = float(change.get("before", 0.0))
            after = float(change.get("after", before))
            delta = float(change.get("delta", after - before))
            summaries.append(f"{feature} {delta:+.2f}")
        lines.append("Weight delta: " + "; ".join(summaries))
    else:
        unchanged = "unchanged" if status != "updated" else "no numeric delta reported"
        lines.append(f"Weights: {unchanged}")
    return "\n".join(lines)


def weight_update_reply(trace: dict) -> str:
    """Truthful live-learning acknowledgement without an action promise."""

    status = str(trace.get("status") or "unknown")
    if status != "updated":
        if status == "error":
            return (
                "I could not apply this feedback. The error was logged and "
                "the weights were not changed."
            )
        return (
            f"I recorded your feedback, but its update status is {status}. "
            "No reward-weight change was applied."
        )
    before = trace.get("before_subgoal")
    after = trace.get("after_subgoal")
    if before and after and before != after:
        current = f" Current top: {before} -> {after}."
    elif after:
        current = f" Current top remains {after}."
    else:
        current = " Feasible subgoals were rescored."
    return (
        "Reward weights updated."
        + current
        + " Actions still use live-state w·φ; no rule was hard-coded."
    )


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
    scale = min(
        (GAME_VIEW_RECT.width - 20) / surface.get_width(),
        (GAME_VIEW_RECT.height - 20) / surface.get_height(),
        1.0,
    )
    if scale < 1:
        # Keep pixel-art edges crisp even for unusually large layouts.
        surface = pygame.transform.scale(
            surface,
            (
                int(surface.get_width() * scale),
                int(surface.get_height() * scale),
            ),
        )
    return surface


def load_ui_font(size: int, *, bold: bool = False) -> pygame.font.Font:
    """Load a Unicode font so committed Chinese IME text is visible."""

    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = (
        windows_dir / "Fonts" / "Noto Sans SC.ttf",
        windows_dir / "Fonts" / "msyh.ttc",
        windows_dir / "Fonts" / "simhei.ttf",
    )
    for candidate in candidates:
        if candidate.is_file():
            font = pygame.font.Font(str(candidate), size)
            font.set_bold(bold)
            return font
    matched = pygame.font.match_font(
        "noto sans sc,microsoft yahei,simhei,arial unicode ms",
        bold=bold,
    )
    return pygame.font.Font(matched, size) if matched else pygame.font.Font(None, size)


def insert_chat_text(
    value: str,
    cursor: int,
    addition: str,
    *,
    limit: int = MAX_CHAT_INPUT_CHARS,
) -> tuple[str, int]:
    """Insert committed IME/clipboard text at the active cursor."""

    cursor = max(0, min(cursor, len(value)))
    addition = addition.replace("\r\n", "\n").replace("\r", "\n")
    addition = "".join(char for char in addition if char == "\n" or char >= " ")
    addition = addition[: max(0, limit - len(value))]
    return value[:cursor] + addition + value[cursor:], cursor + len(addition)


def clipboard_text() -> str:
    """Read Unicode clipboard text, with a reliable Windows implementation."""

    if os.name == "nt":
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.OpenClipboard.argtypes = [ctypes.c_void_p]
        user32.GetClipboardData.argtypes = [ctypes.c_uint]
        user32.GetClipboardData.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        if user32.OpenClipboard(None):
            try:
                handle = user32.GetClipboardData(13)  # CF_UNICODETEXT
                if handle:
                    pointer = kernel32.GlobalLock(handle)
                    if pointer:
                        try:
                            return ctypes.wstring_at(pointer)
                        finally:
                            kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
    try:
        raw = pygame.scrap.get(pygame.SCRAP_TEXT)
    except (pygame.error, AttributeError):
        raw = None
    if isinstance(raw, bytes):
        for encoding in ("utf-8", "utf-16-le", "mbcs"):
            try:
                return raw.decode(encoding).rstrip("\x00")
            except (UnicodeDecodeError, LookupError):
                continue
    return raw if isinstance(raw, str) else ""

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


def draw_wrapped_text(
    screen,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    *,
    line_height: int,
    max_lines: int,
) -> int:
    # Some CJK fonts report a substantially taller line box than their nominal
    # point size. Never compress below that value, and clip to the allocated
    # rectangle so glyphs cannot spill into a neighboring panel.
    actual_line_height = max(int(line_height), int(font.get_linesize()))
    height_limit = max(0, rect.height // max(1, actual_line_height))
    visible_limit = max(0, min(int(max_lines), height_limit))
    lines = clipped_wrapped_lines(text, font, rect.width, visible_limit)
    previous_clip = screen.get_clip()
    screen.set_clip(previous_clip.clip(rect))
    for index, line in enumerate(lines):
        screen.blit(
            font.render(line, True, color),
            (rect.x, rect.y + index * actual_line_height),
        )
    screen.set_clip(previous_clip)
    return len(lines)


def clipped_wrapped_lines(
    text: str,
    font: pygame.font.Font,
    max_width: int,
    max_lines: int,
) -> list[str]:
    """Wrap text and mark a visual truncation without exceeding its box."""

    if max_lines <= 0 or max_width <= 0:
        return []
    lines = wrap_text(text, font, max_width)
    if len(lines) <= max_lines:
        return lines
    visible = lines[:max_lines]
    suffix = "..."
    last = visible[-1].rstrip()
    while last and font.size(last + suffix)[0] > max_width:
        last = last[:-1].rstrip()
    visible[-1] = (last + suffix) if last else suffix
    return visible


def draw_panel(
    screen,
    rect: pygame.Rect,
    *,
    fill: tuple[int, int, int] = (28, 32, 41),
    border: tuple[int, int, int] = (75, 87, 106),
    radius: int = 10,
) -> None:
    """Draw a warm pixel-game card with a restrained drop shadow."""

    shadow = rect.move(0, 4)
    pygame.draw.rect(screen, (12, 15, 21), shadow, border_radius=radius)
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)
    pygame.draw.line(
        screen,
        tuple(min(255, value + 18) for value in border),
        (rect.x + radius, rect.y + 1),
        (rect.right - radius, rect.y + 1),
    )


def draw_button(screen, rect: pygame.Rect, label: str, font, color) -> None:
    hovered = rect.collidepoint(pygame.mouse.get_pos())
    display_color = (
        tuple(min(255, value + 16) for value in color) if hovered else color
    )
    pygame.draw.rect(screen, (12, 15, 21), rect.move(0, 4), border_radius=8)
    pygame.draw.rect(screen, display_color, rect, border_radius=8)
    pygame.draw.rect(screen, (165, 180, 198), rect, width=1, border_radius=8)
    pygame.draw.line(
        screen,
        tuple(min(255, value + 35) for value in display_color),
        (rect.x + 8, rect.y + 2),
        (rect.right - 8, rect.y + 2),
    )
    label_surface = font.render(label, True, (255, 255, 255))
    screen.blit(label_surface, label_surface.get_rect(center=rect.center))


def latest_feedback_classification(
    chat_messages: list[dict[str, object]],
) -> dict[str, object] | None:
    for message in reversed(chat_messages):
        classification = message.get("classification")
        if isinstance(classification, dict):
            return classification
    return None


def render_feedback_analysis(
    screen,
    body_font: pygame.font.Font,
    meta_font: pygame.font.Font,
    classification: dict[str, object] | None,
) -> None:
    """Render all three strategy scores in a dedicated non-overlapping band."""

    rect = CHAT_ANALYSIS_RECT
    probabilities = normalize_feedback_probabilities(
        classification.get("probabilities") if classification else None
    )
    feedback_type = str(
        (classification or {}).get("feedback_type") or "unknown"
    ).lower()
    accent = FEEDBACK_FORM_COLORS.get(feedback_type, (112, 126, 148))
    draw_panel(screen, rect, fill=(23, 29, 38), border=accent, radius=7)

    title = meta_font.render("3-WAY FEEDBACK STRATEGY", True, (198, 208, 222))
    screen.blit(title, (rect.x + 10, rect.y + 7))
    if classification and (
        classification.get("abstained")
        or classification.get("has_uncertain_phrase")
    ):
        badge_text = (
            "LOW SCORE"
            if classification.get("abstained")
            else "PARTLY LOW"
        )
        badge = meta_font.render(badge_text, True, (255, 211, 122))
        badge_rect = badge.get_rect(topright=(rect.right - 10, rect.y + 7))
        screen.blit(badge, badge_rect)

    if not classification:
        draw_wrapped_text(
            screen,
            "Submit feedback to see Evaluative, Imperative, and Descriptive scores.",
            meta_font,
            (135, 153, 176),
            pygame.Rect(rect.x + 10, rect.y + 36, rect.width - 20, 74),
            line_height=meta_font.get_linesize(),
            max_lines=4,
        )
        return

    if not probabilities:
        label = str(classification.get("label") or "Uncertain")
        screen.blit(
            body_font.render(f"Rule result: {label}", True, accent),
            (rect.x + 10, rect.y + 36),
        )
        draw_wrapped_text(
            screen,
            "The learned model is unavailable, so no percentage is shown. Rule matches are not calibrated confidence.",
            meta_font,
            (158, 174, 194),
            pygame.Rect(rect.x + 10, rect.y + 66, rect.width - 20, 54),
            line_height=meta_font.get_linesize(),
            max_lines=3,
        )
        return

    calibrated = bool(classification.get("calibrated"))
    threshold = classification.get("confidence_threshold")
    caption = "CALIBRATED MODEL PROBABILITY" if calibrated else "MODEL SCORES · NOT CALIBRATED"
    if threshold is not None:
        caption += f" · LOW < {float(threshold) * 100:.0f}%"
    screen.blit(
        meta_font.render(caption, True, (126, 144, 166)),
        (rect.x + 10, rect.y + 27),
    )
    label_names = {
        "evaluative": "EVALUATIVE",
        "imperative": "IMPERATIVE",
        "descriptive": "DESCRIPTIVE",
    }
    row_y = rect.y + 50
    for index, key in enumerate(FEEDBACK_FORM_KEYS):
        value = probabilities[key]
        color = FEEDBACK_FORM_COLORS[key]
        y = row_y + index * 24
        label_surface = meta_font.render(label_names[key], True, color)
        screen.blit(label_surface, (rect.x + 10, y - 3))
        bar_rect = pygame.Rect(rect.x + 112, y, 184, 10)
        pygame.draw.rect(screen, (42, 49, 61), bar_rect, border_radius=3)
        fill_width = max(2, int(round(bar_rect.width * value))) if value > 0 else 0
        if fill_width:
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_rect.height),
                border_radius=3,
            )
        percentage = meta_font.render(f"{value * 100:5.1f}%", True, (225, 230, 238))
        screen.blit(percentage, percentage.get_rect(topright=(rect.right - 10, y - 3)))


def render_feedback_home_panel(
    screen,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    meta_font: pygame.font.Font,
    *,
    chat_status: str,
    episode: int,
    episode_step: int,
    last_ai_subgoal: str,
    last_ai_event: str,
    learning_enabled: bool,
) -> None:
    """Render the collapsed sidebar with three clear, separated type cards."""

    panel = CHAT_PANEL_RECT
    draw_panel(screen, panel, fill=(20, 25, 33), border=(73, 92, 115), radius=10)
    screen.blit(
        font.render("FEEDBACK GUIDE / 反馈指南", True, (245, 245, 245)),
        (panel.x + 14, panel.y + 13),
    )
    draw_wrapped_text(
        screen,
        "The classifier reports one paper-facing strategy label and all three model scores.",
        meta_font,
        (143, 162, 184),
        pygame.Rect(panel.x + 14, panel.y + 48, panel.width - 28, 44),
        line_height=meta_font.get_linesize(),
        max_lines=2,
    )
    screen.blit(
        meta_font.render("THE THREE STRATEGIES", True, (190, 200, 214)),
        (panel.x + 14, panel.y + 99),
    )
    descriptions = {
        "evaluative": "Judges a recent outcome or behavior.",
        "imperative": "Requests the next action or change.",
        "descriptive": "States an observed behavior or property.",
    }
    card_y = panel.y + 125
    for index, key in enumerate(FEEDBACK_FORM_KEYS):
        color = FEEDBACK_FORM_COLORS[key]
        card = pygame.Rect(panel.x + 14, card_y + index * 69, panel.width - 28, 58)
        pygame.draw.rect(screen, (27, 33, 43), card, border_radius=6)
        pygame.draw.rect(screen, (60, 70, 84), card, width=1, border_radius=6)
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(card.x, card.y, 4, card.height),
            border_radius=2,
        )
        screen.blit(
            small_font.render(FEEDBACK_FORM_LABELS[key], True, color),
            (card.x + 12, card.y + 5),
        )
        screen.blit(
            meta_font.render(descriptions[key], True, (172, 183, 198)),
            (card.x + 12, card.y + 34),
        )

    flow_y = panel.y + 344
    screen.blit(
        meta_font.render("PLAYTEST FLOW", True, (190, 200, 214)),
        (panel.x + 14, flow_y),
    )
    flow = (
        "1  Play and observe the AI\n"
        "2  Pause at the behavior you noticed\n"
        "3  Write specific feedback in English\n"
        + (
            "4  Review the update, then resume"
            if learning_enabled
            else "4  Review the record, then resume"
        )
    )
    draw_wrapped_text(
        screen,
        flow,
        meta_font,
        (192, 203, 218),
        pygame.Rect(panel.x + 14, flow_y + 25, panel.width - 28, 94),
        line_height=meta_font.get_linesize() + 2,
        max_lines=5,
    )

    status_y = panel.y + 476
    screen.blit(
        meta_font.render("LATEST STATUS", True, (190, 200, 214)),
        (panel.x + 14, status_y),
    )
    draw_wrapped_text(
        screen,
        chat_status or "No feedback submitted yet.",
        meta_font,
        (245, 200, 105),
        pygame.Rect(panel.x + 14, status_y + 24, panel.width - 28, 54),
        line_height=meta_font.get_linesize(),
        max_lines=3,
    )
    context = (
        f"EP {episode} · STEP {episode_step}   |   "
        f"SUBGOAL {last_ai_subgoal}\nEVENT {last_ai_event or 'none'}"
    )
    draw_wrapped_text(
        screen,
        context,
        meta_font,
        (132, 151, 174),
        pygame.Rect(panel.x + 14, panel.bottom - 70, panel.width - 28, 50),
        line_height=meta_font.get_linesize(),
        max_lines=3,
    )


def render_chat_panel(
    screen,
    font,
    small_font,
    meta_font,
    chat_messages: list[dict[str, object]],
    chat_input: str,
    chat_cursor: int,
    chat_composition: str,
    chat_pending: bool,
    chat_status: str,
    mode_banner: str,
    learning_enabled: bool,
    chat_scroll: int = 0,
) -> None:
    panel = CHAT_PANEL_RECT
    draw_panel(
        screen,
        panel,
        fill=(20, 25, 33),
        border=(95, 126, 160),
        radius=10,
    )
    screen.blit(
        font.render("FEEDBACK LAB / 反馈台", True, (245, 245, 245)),
        (panel.x + 14, panel.y + 12),
    )
    draw_wrapped_text(
        screen,
        (
            "Paused · English feedback updates the reward model."
            if learning_enabled
            else f"{mode_banner} · submissions are recorded only."
        ),
        meta_font,
        (170, 210, 180) if learning_enabled else (255, 175, 105),
        pygame.Rect(panel.x + 14, panel.y + 47, panel.width - 28, 27),
        line_height=meta_font.get_linesize(),
        max_lines=1,
    )

    render_feedback_analysis(
        screen,
        small_font,
        meta_font,
        latest_feedback_classification(chat_messages),
    )

    conversation_y = CHAT_HISTORY_RECT.y - meta_font.get_linesize() - 4
    screen.blit(
        meta_font.render("CONVERSATION", True, (125, 145, 170)),
        (CHAT_HISTORY_RECT.x + 6, conversation_y),
    )
    scroll = max(0, min(int(chat_scroll), max(0, len(chat_messages) - 1)))
    scroll_hint = f"{scroll} newer - wheel down" if scroll else "mouse wheel to review"
    hint_surface = meta_font.render(scroll_hint, True, (105, 125, 150))
    screen.blit(
        hint_surface,
        hint_surface.get_rect(
            topright=(CHAT_HISTORY_RECT.right - 6, conversation_y)
        ),
    )

    body_line_height = small_font.get_linesize()
    meta_line_height = meta_font.get_linesize()
    role_line_height = meta_font.get_linesize()
    history_y = CHAT_HISTORY_RECT.bottom
    previous_clip = screen.get_clip()
    screen.set_clip(CHAT_HISTORY_RECT)
    rendered_cards = 0
    for message in list(reversed(chat_messages))[scroll:]:
        is_user = message.get("role") == "user"
        role = "YOU" if is_user else "AGENT"
        color = (180, 220, 255) if is_user else (245, 220, 150)
        card_color = (25, 34, 45) if is_user else (39, 36, 31)
        content_limit = 2 if (not is_user and message.get("meta")) else 3
        content_width = CHAT_HISTORY_RECT.width - 30
        content_lines = clipped_wrapped_lines(
            str(message.get("content") or ""),
            small_font,
            content_width,
            content_limit,
        )
        meta_lines: list[str] = []
        if not is_user and message.get("meta"):
            meta_lines = clipped_wrapped_lines(
                str(message["meta"]), meta_font, content_width, 3
            )
        block_height = 7 + role_line_height + 3 + len(content_lines) * body_line_height
        if meta_lines:
            block_height += 8 + len(meta_lines) * meta_line_height
        block_height += 7
        available = history_y - CHAT_HISTORY_RECT.top
        if block_height > available:
            # Older cards stay whole; the mouse wheel exposes the next card.
            break
        block_y = history_y - block_height
        card_rect = pygame.Rect(
            CHAT_HISTORY_RECT.x + 3,
            block_y,
            CHAT_HISTORY_RECT.width - 6,
            block_height,
        )
        pygame.draw.rect(screen, card_color, card_rect, border_radius=6)
        pygame.draw.rect(
            screen, (57, 68, 82), card_rect, width=1, border_radius=6
        )
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(card_rect.x, block_y, 3, block_height),
            border_radius=2,
        )
        role_y = block_y + 7
        content_x = card_rect.x + 10
        screen.blit(meta_font.render(role, True, color), (content_x, role_y))
        content_y = role_y + role_line_height + 3
        for index, line in enumerate(content_lines):
            screen.blit(
                small_font.render(line, True, color),
                (content_x, content_y + index * body_line_height),
            )
        if meta_lines:
            meta_y = content_y + len(content_lines) * body_line_height + 8
            pygame.draw.line(
                screen,
                (82, 88, 98),
                (content_x, meta_y - 4),
                (card_rect.right - 10, meta_y - 2),
                width=1,
            )
            for index, line in enumerate(meta_lines):
                screen.blit(
                    meta_font.render(line, True, (145, 160, 180)),
                    (content_x, meta_y + index * meta_line_height),
                )
        rendered_cards += 1
        history_y = block_y - 7

    if not chat_messages:
        empty = meta_font.render(
            "No feedback yet. Your latest exchange appears here.",
            True,
            (105, 125, 150),
        )
        screen.blit(empty, empty.get_rect(center=CHAT_HISTORY_RECT.center))
    elif not rendered_cards:
        screen.blit(
            meta_font.render(
                "Message available - use the mouse wheel.",
                True,
                (145, 160, 180),
            ),
            (
                CHAT_HISTORY_RECT.x + 8,
                CHAT_HISTORY_RECT.bottom - meta_line_height - 4,
            ),
        )
    screen.set_clip(previous_clip)

    status = (
        f"{chat_status} | Preparing agent response..."
        if chat_pending and chat_status
        else "Preparing agent response..."
        if chat_pending
        else chat_status
    )
    if status:
        old_clip = screen.get_clip()
        screen.set_clip(CHAT_STATUS_RECT)
        draw_wrapped_text(
            screen,
            status,
            meta_font,
            (245, 200, 105),
            CHAT_STATUS_RECT,
            line_height=meta_font.get_linesize(),
            max_lines=2,
        )
        screen.set_clip(old_clip)

    pygame.draw.rect(screen, (35, 40, 50), CHAT_INPUT_RECT, border_radius=5)
    pygame.draw.rect(
        screen,
        (130, 175, 220),
        CHAT_INPUT_RECT,
        width=2,
        border_radius=5,
    )
    # Use a narrow ASCII caret; it renders consistently across fallback fonts.
    cursor_marker = "|" if pygame.time.get_ticks() // 500 % 2 == 0 else " "
    if chat_input or chat_composition:
        display_text = (
            chat_input[:chat_cursor]
            + chat_composition
            + cursor_marker
            + chat_input[chat_cursor:]
        )
        input_color = (235, 235, 235)
    else:
        display_text = "Type feedback... Enter to submit; Shift+Enter for a new line."
        input_color = (130, 145, 165)
    input_lines = wrap_text(display_text, small_font, CHAT_INPUT_RECT.width - 20)
    input_line_height = small_font.get_linesize()
    old_clip = screen.get_clip()
    screen.set_clip(CHAT_INPUT_RECT.inflate(-10, -8))
    for index, line in enumerate(input_lines[-3:]):
        screen.blit(
            small_font.render(line, True, input_color),
            (
                CHAT_INPUT_RECT.x + 10,
                CHAT_INPUT_RECT.y + 5 + index * input_line_height,
            ),
        )
    screen.set_clip(old_clip)

def main() -> int:
    args = parse_args()
    live_learning_enabled = online_learning_enabled(
        args.ai_mode,
        args.comfort_feedback_mode if args.ai_mode == "comfort_subgoal" else None,
    )
    mode_banner = feedback_mode_banner(
        args.ai_mode,
        args.comfort_feedback_mode if args.ai_mode == "comfort_subgoal" else None,
    )
    if args.step_hz <= 0:
        raise ValueError("--step-hz must be greater than zero")
    if args.render_fps <= 0:
        raise ValueError("--render-fps must be greater than zero")
    if args.start_delay < 0:
        raise ValueError("--start-delay cannot be negative")
    if args.horizon <= 0:
        raise ValueError("--horizon must be greater than zero")
    validate_feedback_configuration(
        args.ai_mode,
        args.comfort_feedback_mode if args.ai_mode == "comfort_subgoal" else None,
        args.resume_learner_state,
    )
    if (
        args.ai_mode == "comfort_subgoal"
        and args.resume_learner_state
        and args.learner_state is None
        and str(args.teacher_id or "").strip().lower() in {"", "anonymous"}
    ):
        raise ValueError(
            "--resume-learner-state requires a stable --teacher-id or an "
            "explicit --learner-state; anonymous sessions are isolated"
        )
    session_dir = new_session_dir(args.output_dir)
    session_id = uuid.uuid4().hex
    session_started_utc = utc_timestamp()

    subgoal_model = None
    motion_planner = None
    comfort_agent = None
    if args.ai_mode == "subgoal_executor":
        if args.layout != RING_TOMATO_ONION_H0_LAYOUT:
            raise ValueError(
                "--ai-mode subgoal_executor requires layout "
                f"{RING_TOMATO_ONION_H0_LAYOUT!r}"
            )
        layouts = [RING_TOMATO_ONION_H0_LAYOUT]
        agent_dir = Path(args.subgoal_executor).resolve()
        ai_agent = None
        subgoal_model = load_h0_executor(args.subgoal_executor)
        motion_planner = make_motion_planner(
            args.layout,
            args.seed,
            args.horizon,
        )
    elif args.ai_mode == "comfort_subgoal":
        if args.layout != RING_TOMATO_ONION_H0_LAYOUT:
            raise ValueError(
                "--ai-mode comfort_subgoal requires layout "
                f"{RING_TOMATO_ONION_H0_LAYOUT!r}"
            )
        from durf.baseline.comfort_subgoal_agent import ComfortSubgoalAgent

        layouts = [RING_TOMATO_ONION_H0_LAYOUT]
        agent_dir = "comfort_subgoal (w dot phi)"
        ai_agent = None
        motion_planner = make_motion_planner(args.layout, args.seed, args.horizon)
        learner_state_path = configured_learner_state_path(
            args.comfort_feedback_mode,
            args.learner_state,
            str(args.teacher_id),
            session_id=session_id,
        )
        comfort_agent = ComfortSubgoalAgent(
            motion_planner,
            weights_path=args.comfort_weights,
            model_path=args.route2_model,
            lambda_pref=args.comfort_lambda,
            ai_index=0,
            feedback_mode=args.comfort_feedback_mode,
            route1_prior=args.route1_prior,
            route1_lookback=args.route1_lookback,
            online_blend=args.route2_blend,
            route2_observation_precision=args.route2_observation_precision,
            human_feedback_precision=args.human_feedback_precision,
            learner_state_path=learner_state_path,
            resume_learner_state=args.resume_learner_state,
        )
    else:
        requested_layouts = list(dict.fromkeys([args.layout, *args.layouts]))
        ensure_agent_layout(args.agent, args.layout)
        layouts = filter_compatible_layouts(args.agent, requested_layouts)
        if args.layout not in layouts:
            layouts.insert(0, args.layout)
        agent_dir = resolve_agent_dir(args.agent)
        ai_agent = load_rllib_agent(args.agent, agent_index=0)
    layout_index = layouts.index(args.layout)
    current_layout = args.layout

    env = make_direct_multi_env(current_layout, args.seed, horizon=args.horizon)
    envs_by_layout = {current_layout: env}

    trajectory_path = session_dir / "trajectory.csv"
    chat_path = session_dir / "chat_messages.csv"
    feedback_updates_path = session_dir / "feedback_updates.jsonl"
    session_manifest_path = session_dir / "session_manifest.json"
    initial_learner_state_path = session_dir / "initial_learner_state.json"
    final_learner_state_path = session_dir / "final_learner_state.json"
    trajectory_handle = trajectory_path.open("w", newline="", encoding="utf-8")
    chat_handle = chat_path.open("w", newline="", encoding="utf-8")
    feedback_updates_handle = feedback_updates_path.open("w", encoding="utf-8")
    trajectory_fields = [
        "timestamp_utc",
        "episode",
        "episode_step",
        "total_step",
        "layout",
        "ai_mode",
        "comfort_feedback_mode",
        "ai_subgoal",
        "ai_event",
        "ai_proposed_action",
        "ai_proposed_action_name",
        "ai_action",
        "ai_action_name",
        "policy_override",
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
        "feedback_event_id",
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

    session_manifest = {
        "schema_version": 2,
        "session_id": session_id,
        "teacher_id": str(args.teacher_id),
        "started_utc": session_started_utc,
        "build_id": BUILD_ID,
        "layout": current_layout,
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "ai_mode": args.ai_mode,
        "feedback_mode": (
            args.comfort_feedback_mode if args.ai_mode == "comfort_subgoal" else None
        ),
        "feedback_role": "adaptive" if live_learning_enabled else "control",
        "online_learning_enabled": live_learning_enabled,
        "resume_learner_state_requested": bool(args.resume_learner_state),
        "score_formula": (
            "w_dot_phi" if args.ai_mode == "comfort_subgoal" else None
        ),
        "route2_observation_precision": (
            float(args.route2_observation_precision)
            if args.ai_mode == "comfort_subgoal"
            and args.comfort_feedback_mode == "route2"
            else None
        ),
        "supervision": feedback_supervision_metadata(
            args.ai_mode,
            args.comfort_feedback_mode
            if args.ai_mode == "comfort_subgoal"
            else None,
        ),
        "files": {
            "trajectory": trajectory_path.name,
            "chat_messages": chat_path.name,
            "feedback_updates": feedback_updates_path.name,
            "initial_learner_state": (
                initial_learner_state_path.name if comfort_agent is not None else None
            ),
            "final_learner_state": (
                final_learner_state_path.name if comfort_agent is not None else None
            ),
        },
    }
    if comfort_agent is not None:
        initial_learner_state = comfort_agent.learner_state_dict()
        initial_learner_state_path.write_text(
            json.dumps(initial_learner_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        session_manifest["learner_state_path"] = (
            str(learner_state_path) if learner_state_path is not None else None
        )
        session_manifest["model_path"] = initial_learner_state.get("model_path")
        session_manifest["model_sha256"] = initial_learner_state.get("model_sha256")
    session_manifest_path.write_text(
        json.dumps(session_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pygame.init()
    pygame.key.set_repeat(350, 35)
    try:
        screen = pygame.display.set_mode(
            WINDOW_SIZE,
            flags=pygame.SCALED | pygame.RESIZABLE,
            vsync=1,
        )
    except pygame.error:
        # Dummy video drivers and older SDL builds may not support SCALED.
        screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(f"DURF Human Feedback Lab [{BUILD_ID}]")
    try:
        pygame.scrap.init()
    except pygame.error:
        pass
    clock = pygame.time.Clock()
    large_font = load_ui_font(42, bold=True)
    font = load_ui_font(21, bold=True)
    small_font = load_ui_font(17)
    meta_font = load_ui_font(13)
    visualizer = StateVisualizer(
        max_size=(GAME_VIEW_RECT.width - 24, GAME_VIEW_RECT.height - 24)
    )

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
    last_ai_subgoal = "N/A"
    last_ai_event = ""
    last_predict_ms = 0.0
    last_environment_step_ms = 0.0
    chat_open = False
    chat_input = ""
    chat_cursor = 0
    chat_composition = ""
    chat_messages: list[dict[str, object]] = []
    chat_scroll = 0
    chat_pending = False
    chat_status = (
        "Online learning is active: submitted feedback updates reward weights."
        if live_learning_enabled
        else f"{mode_banner}. Feedback will not change the current agent."
    )
    feedback_submission_count = 0
    feedback_update_status_counts: dict[str, int] = {}
    learner_updates_applied = 0
    chat_result_queue: queue.Queue[tuple[str, str]] = queue.Queue()
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
        nonlocal last_ai_event
        nonlocal last_ai_subgoal
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
            env = envs_by_layout[current_layout]

        ai_obs, _ = env.multi_reset()
        if ai_agent is not None:
            ai_agent.reset()
        if comfort_agent is not None:
            # Keep the learned posterior across episodes, but never attribute
            # new feedback to a trajectory from the previous episode.
            comfort_agent.reset_context()
        episode += 1
        episode_step = 0
        episode_reward = 0.0
        last_ai_event = ""
        last_ai_subgoal = "N/A"
        pending_interact = False
        pending_motion = None
        held_motion_actions.clear()
        countdown_until = pygame.time.get_ticks() + round(args.start_delay * 1000)
        next_step_at = countdown_until
        game_surface = render_game_surface(visualizer, env, episode_reward)

    def request_layout_switch(new_layout_index: int) -> bool:
        nonlocal last_layout_switch_at
        if args.ai_mode in ("subgoal_executor", "comfort_subgoal"):
            return False
        now = pygame.time.get_ticks()
        if not 0 <= new_layout_index < len(layouts):
            return False
        if now - last_layout_switch_at < LAYOUT_SWITCH_DEBOUNCE_MS:
            return False
        last_layout_switch_at = now
        reset_episode(new_layout_index)
        return True

    def log_chat_message(
        role: str,
        content: str,
        *,
        feedback_event_id: str = "",
    ) -> None:
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
                "feedback_event_id": feedback_event_id,
                "content": content,
            },
        )

    def log_feedback_update(trace: dict, feedback_event_id: str) -> dict:
        nonlocal learner_updates_applied

        record = feedback_update_record(
            trace,
            timestamp_utc=utc_timestamp(),
            episode=episode,
            episode_step=episode_step,
            total_step=total_step,
            session_id=session_id,
            teacher_id=str(args.teacher_id),
            feedback_event_id=feedback_event_id,
        )
        feedback_updates_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        feedback_updates_handle.flush()
        status = str(record.get("status") or "unknown")
        feedback_update_status_counts[status] = (
            feedback_update_status_counts.get(status, 0) + 1
        )
        if record["learning_applied"]:
            learner_updates_applied += 1
        return record

    def set_paused(should_pause: bool, event_label: str) -> bool:
        nonlocal countdown_until
        nonlocal held_motion_actions
        nonlocal next_step_at
        nonlocal paused
        nonlocal pending_interact
        nonlocal pending_motion

        if paused == should_pause:
            return False
        paused = should_pause
        pending_interact = False
        pending_motion = None
        held_motion_actions.clear()
        now = pygame.time.get_ticks()
        if not paused:
            next_step_at = now + step_interval_ms
            countdown_until = 0
        write_row(
            pause_writer,
            pause_handle,
            {
                "timestamp_utc": utc_timestamp(),
                "event": event_label,
                "episode": episode,
                "episode_step": episode_step,
                "total_step": total_step,
                "layout": current_layout,
                "pygame_ticks": now,
            },
        )
        print(
            f"{event_label.upper()} at episode={episode}, "
            f"step={episode_step}, total_step={total_step}"
        )
        return True

    def open_feedback() -> None:
        nonlocal chat_composition
        nonlocal chat_cursor
        nonlocal chat_open
        nonlocal chat_status

        set_paused(True, "feedback_paused")
        chat_open = True
        chat_cursor = len(chat_input)
        chat_composition = ""
        if not chat_messages:
            chat_status = (
                "Game paused. Submit feedback to update the reward model."
                if live_learning_enabled
                else f"{mode_banner}. Submissions are written to the log only."
            )
        pygame.key.start_text_input()
        pygame.key.set_text_input_rect(CHAT_INPUT_RECT)

    def close_feedback() -> None:
        nonlocal chat_composition
        nonlocal chat_open

        chat_open = False
        chat_composition = ""
        pygame.key.stop_text_input()

    def submit_feedback() -> None:
        nonlocal chat_composition
        nonlocal chat_cursor
        nonlocal chat_input
        nonlocal chat_status

        if chat_composition:
            chat_status = "Confirm the current IME composition, then press Enter again."
            return
        prompt = chat_input.strip()
        if not prompt:
            chat_status = "Please enter specific feedback before submitting."
            return
        chat_input = ""
        chat_cursor = 0
        chat_composition = ""
        start_chat_request(prompt)

    def start_chat_request(prompt: str) -> None:
        nonlocal chat_pending
        nonlocal chat_scroll
        nonlocal chat_status
        nonlocal feedback_submission_count

        feedback_event_id = uuid.uuid4().hex
        feedback_submission_count += 1
        form_classification = classify_feedback_form_for_ui(prompt)
        chat_messages.append(
            {
                "role": "user",
                "content": prompt,
                "classification": form_classification,
            }
        )
        chat_scroll = 0
        log_chat_message(
            "user",
            prompt,
            feedback_event_id=feedback_event_id,
        )
        # Path A: make the selected Route 1/Route 2 update explicit and auditable.
        trace = None
        if args.ai_mode == "comfort_subgoal" and comfort_agent is not None:
            try:
                trace = apply_live_feedback_update(
                    comfort_agent,
                    prompt,
                    feedback_event_id,
                    form_classification,
                )
                trace["feedback_form_classification"] = dict(form_classification)
                log_feedback_update(trace, feedback_event_id)
                if trace["status"] == "updated" and trace["mode"].startswith("route1"):
                    change = (trace.get("top_changes") or [{}])[0].get("feature", "no feature")
                    switch_note = (
                        "policy held (score margin)"
                        if trace.get("accepted_but_no_policy_switch")
                        else f"{trace.get('before_subgoal')} -> {trace.get('after_subgoal')}"
                    )
                    chat_status = (
                        f"{trace['mode']} | {trace['feedback_type']} | {change} | "
                        f"p={trace.get('source_precision_multiplier', 1):g} | "
                        f"{switch_note}"
                    )
                elif trace["status"] == "updated":
                    chat_status = (
                        f"{trace['mode']} updated | "
                        f"{trace.get('before_subgoal')} -> {trace.get('after_subgoal')}"
                    )
                elif trace["status"] == "rejected_low_confidence":
                    stages = ", ".join(
                        sorted(
                            {
                                str(reason.get("stage"))
                                for reason in trace.get("rejection_reasons", [])
                                if reason.get("stage")
                            }
                        )
                    )
                    chat_status = (
                        "Feedback not applied: low confidence"
                        + (f" ({stages})" if stages else "")
                        + ". Please be more specific."
                    )
                elif trace["status"] == "rejected_duplicate":
                    chat_status = "Duplicate feedback ignored; weights unchanged."
                elif trace.get("mode") == "frozen":
                    chat_status = no_learning_reply(args.ai_mode, trace.get("mode"))
                else:
                    chat_status = f"{trace['mode']}: {trace['status']} feedback."
            except Exception as exc:
                error_trace = {
                    "timestamp_utc": utc_timestamp(),
                    "episode": episode,
                    "episode_step": episode_step,
                    "total_step": total_step,
                    "session_id": session_id,
                    "teacher_id": str(args.teacher_id),
                    "feedback_event_id": feedback_event_id,
                    "status": "error",
                    "mode": args.comfort_feedback_mode,
                    "text": prompt,
                    "source": "human_live",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                trace = error_trace
                trace["feedback_form_classification"] = dict(form_classification)
                log_feedback_update(error_trace, feedback_event_id)
                chat_status = f"Comfort update failed and was logged: {exc}"
        else:
            trace = {
                "status": "ignored",
                "mode": args.ai_mode,
                "text": prompt,
                "source": "human_live",
                "rejection_reason": "online_learning_not_supported",
                "feedback_form_classification": dict(form_classification),
            }
            log_feedback_update(trace, feedback_event_id)
            chat_status = no_learning_reply(args.ai_mode, None)
        if not live_learning_enabled:
            reply = no_learning_reply(
                args.ai_mode,
                args.comfort_feedback_mode
                if args.ai_mode == "comfort_subgoal"
                else None,
            )
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "meta": feedback_update_metadata(trace, form_classification),
                    "classification": form_classification,
                }
            )
            log_chat_message(
                "assistant",
                reply,
                feedback_event_id=feedback_event_id,
            )
            chat_pending = False
            return
        if trace is not None:
            # A generic chat model cannot know which action w·phi will select
            # on a future state.  Use the actual learner trace so the UI never
            # promises a hard-coded action that the policy did not choose.
            reply = weight_update_reply(trace)
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                    "meta": feedback_update_metadata(trace, form_classification),
                    "classification": form_classification,
                }
            )
            log_chat_message(
                "assistant",
                reply,
                feedback_event_id=feedback_event_id,
            )
            chat_pending = False
            return
        if not os.getenv("DEEPSEEK_API_KEY"):
            chat_pending = False
            return
        chat_pending = True
        if not chat_status:
            chat_status = ""
        history = [
            {
                "role": "system",
                "content": (
                    "You are helping a human evaluate a cooperative Overcooked "
                    "AI agent. Reply concisely and ask clarifying questions when needed."
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

    print(f"AI mode: {args.ai_mode}")
    if args.ai_mode == "comfort_subgoal":
        route2_update = (
            f"legacy EMA alpha={args.route2_blend:g}"
            if args.route2_blend is not None
            else f"paper Gaussian precision={args.route2_observation_precision:g}"
        )
        print(
            f"Feedback learner: {args.comfort_feedback_mode} "
            f"(Route1 prior={args.route1_prior}, lookback={args.route1_lookback}, "
            f"Route2 update={route2_update})"
        )
        print(
            f"Learner state: {learner_state_path} "
            f"(resume={args.resume_learner_state})"
        )
    print(f"Feedback status: {mode_banner}")
    if not live_learning_enabled:
        print("WARNING: submitted feedback is logged only; it will not update the Agent.")
    print(f"Agent: {agent_dir}")
    print(f"Interface build: {BUILD_ID}")
    print(f"Script: {Path(__file__).resolve()}")
    print(f"Session logs: {session_dir}")
    print("Human is green; AI is blue.")
    print(
        f"Timing: {args.step_hz:g} environment steps/s, "
        f"{args.render_fps} render FPS, {args.start_delay:g}s start delay, "
        f"horizon={args.horizon}"
    )
    print(
        "Controls: WASD/Arrows=Move, Space=Interact, "
        "P/Tab/F1=Pause, Chat=Language feedback, R=Reset, "
        "1-4/N/M=Switch map (PPO only), Esc=Quit"
    )

    try:
        while running:
            while not chat_result_queue.empty():
                role, content = chat_result_queue.get()
                chat_pending = False
                if role == "assistant":
                    chat_messages.append({"role": "assistant", "content": content})
                    chat_scroll = 0
                    log_chat_message("assistant", content)
                    # Keep the local learning result visible; model chat is a
                    # separate, optional channel and must not erase it.
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
                        modifiers = getattr(event, "mod", pygame.key.get_mods())
                        if event.key == pygame.K_ESCAPE:
                            close_feedback()
                        elif event.key == pygame.K_v and modifiers & pygame.KMOD_CTRL:
                            pasted = clipboard_text()
                            if pasted:
                                chat_input, chat_cursor = insert_chat_text(
                                    chat_input,
                                    chat_cursor,
                                    pasted,
                                )
                                chat_composition = ""
                                chat_status = "Clipboard text pasted."
                            else:
                                chat_status = "The clipboard does not contain usable text."
                        elif event.key == pygame.K_RETURN:
                            if modifiers & pygame.KMOD_SHIFT:
                                chat_input, chat_cursor = insert_chat_text(
                                    chat_input,
                                    chat_cursor,
                                    "\n",
                                )
                            else:
                                submit_feedback()
                        elif event.key == pygame.K_BACKSPACE:
                            if chat_cursor > 0:
                                chat_input = (
                                    chat_input[: chat_cursor - 1]
                                    + chat_input[chat_cursor:]
                                )
                                chat_cursor -= 1
                            chat_composition = ""
                        elif event.key == pygame.K_DELETE:
                            if chat_cursor < len(chat_input):
                                chat_input = (
                                    chat_input[:chat_cursor]
                                    + chat_input[chat_cursor + 1 :]
                                )
                            chat_composition = ""
                        elif event.key == pygame.K_LEFT:
                            chat_cursor = max(0, chat_cursor - 1)
                            chat_composition = ""
                        elif event.key == pygame.K_RIGHT:
                            chat_cursor = min(len(chat_input), chat_cursor + 1)
                            chat_composition = ""
                        elif event.key == pygame.K_HOME:
                            chat_cursor = 0
                            chat_composition = ""
                        elif event.key == pygame.K_END:
                            chat_cursor = len(chat_input)
                            chat_composition = ""
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
                    chat_input, chat_cursor = insert_chat_text(
                        chat_input,
                        chat_cursor,
                        event.text,
                    )
                    chat_composition = ""
                elif event.type == pygame.TEXTEDITING and chat_open:
                    chat_composition = event.text
                elif event.type == pygame.MOUSEWHEEL and chat_open:
                    # Wheel up reviews older cards; wheel down returns to now.
                    chat_scroll = max(
                        0,
                        min(
                            max(0, len(chat_messages) - 1),
                            chat_scroll + int(event.y),
                        ),
                    )
                elif event.type == pygame.KEYUP:
                    motion_action = event_motion_action(event)
                    if motion_action is not None:
                        held_motion_actions.discard(motion_action)
                        last_key_debug = (
                            f"released {describe_key_event(event)} -> "
                            f"{ACTION_NAMES[motion_action]}"
                        )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if CHAT_BUTTON.collidepoint(event.pos):
                        if chat_open:
                            close_feedback()
                        else:
                            open_feedback()
                    elif PAUSE_BUTTON.collidepoint(event.pos):
                        if chat_open:
                            close_feedback()
                        pause_requested = True
                    elif chat_open and CHAT_INPUT_RECT.collidepoint(event.pos):
                        pygame.key.start_text_input()
                        pygame.key.set_text_input_rect(CHAT_INPUT_RECT)

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
            if pause_requested and now - last_pause_toggle_at >= PAUSE_DEBOUNCE_MS:
                last_pause_toggle_at = now
                pause_event = "resumed" if paused else "paused"
                set_paused(not paused, pause_event)
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
                if args.ai_mode == "subgoal_executor":
                    if subgoal_model is None or motion_planner is None:
                        raise RuntimeError("H0 runtime was not initialized")
                    ai_action_raw, last_ai_subgoal = h0_executor_action(
                        env.base_env,
                        subgoal_model,
                        motion_planner,
                    )
                elif args.ai_mode == "comfort_subgoal":
                    if comfort_agent is None:
                        raise RuntimeError("comfort_subgoal runtime was not initialized")
                    ai_action_raw, last_ai_subgoal = comfort_agent.act(
                        env.base_env.state,
                        human_action=human_action,
                    )
                else:
                    ai_action_raw = rllib_action_index(ai_agent, env.base_env.state)
                    last_ai_subgoal = "PPO"
                last_predict_ms = (time.perf_counter() - predict_started) * 1000
                state_before = state_facts(env)
                if args.ai_mode in ("subgoal_executor", "comfort_subgoal"):
                    ai_action, last_ai_event = resolve_executed_ai_action(
                        args.ai_mode,
                        env.base_env,
                        motion_planner,
                        int(ai_action_raw),
                        human_action,
                        last_ai_subgoal,
                    )
                else:
                    ai_action = int(ai_action_raw)
                    last_ai_event = ""
                step_started = time.perf_counter()
                (ai_obs, _), (reward, _), done, _ = env.multi_step(
                    ai_action,
                    human_action,
                )
                state_after = state_facts(env)
                if args.ai_mode == "comfort_subgoal" and comfort_agent is not None:
                    comfort_agent.record_transition(
                        state_before=state_before,
                        state_after=state_after,
                        ai_action_name=ACTION_NAMES[ai_action],
                        human_action_name=ACTION_NAMES[human_action],
                        environment_reward=float(reward),
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
                        "ai_mode": args.ai_mode,
                        "comfort_feedback_mode": (
                            args.comfort_feedback_mode
                            if args.ai_mode == "comfort_subgoal"
                            else ""
                        ),
                        "ai_subgoal": last_ai_subgoal,
                        "ai_event": last_ai_event,
                        "ai_proposed_action": int(ai_action_raw),
                        "ai_proposed_action_name": ACTION_NAMES[int(ai_action_raw)],
                        "ai_action": ai_action,
                        "ai_action_name": ACTION_NAMES[ai_action],
                        "policy_override": bool(int(ai_action) != int(ai_action_raw)),
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

            screen.fill((17, 21, 28))
            # A subtle tiled backdrop visually connects the crisp pixel kitchen
            # with the modern research controls without reducing readability.
            for tile_y in range(0, WINDOW_SIZE[1], 32):
                for tile_x in range(0, WINDOW_SIZE[0], 32):
                    if (tile_x // 32 + tile_y // 32) % 2:
                        pygame.draw.rect(
                            screen,
                            (19, 24, 32),
                            pygame.Rect(tile_x, tile_y, 32, 32),
                        )
            draw_panel(
                screen,
                GAME_VIEW_RECT,
                fill=(14, 18, 24),
                border=(72, 84, 102),
                radius=11,
            )
            old_clip = screen.get_clip()
            screen.set_clip(old_clip.clip(GAME_VIEW_RECT.inflate(-12, -12)))
            screen.blit(
                game_surface,
                game_surface.get_rect(center=GAME_VIEW_RECT.center),
            )
            screen.set_clip(old_clip)

            draw_panel(
                screen,
                GAME_INFO_RECT,
                fill=(27, 32, 41),
                border=(68, 81, 100),
                radius=9,
            )
            draw_panel(
                screen,
                SIDEBAR_RECT,
                fill=(25, 30, 39),
                border=(78, 92, 113),
                radius=11,
            )

            if paused:
                overlay = pygame.Surface(GAME_VIEW_RECT.size, pygame.SRCALPHA)
                overlay.fill((10, 12, 16, 155))
                screen.blit(overlay, GAME_VIEW_RECT.topleft)
                pause_label = large_font.render("GAME PAUSED", True, (255, 225, 120))
                screen.blit(
                    pause_label,
                    pause_label.get_rect(
                        center=(GAME_VIEW_RECT.centerx, GAME_VIEW_RECT.centery - 32)
                    ),
                )
                draw_wrapped_text(
                    screen,
                    (
                        "Submit feedback, review the learning result, then resume."
                        if live_learning_enabled
                        else "FROZEN: feedback is recorded and does not update the agent."
                    ),
                    small_font,
                    (245, 245, 245),
                    pygame.Rect(
                        GAME_VIEW_RECT.x + 80,
                        GAME_VIEW_RECT.centery + 15,
                        GAME_VIEW_RECT.width - 160,
                        55,
                    ),
                    line_height=small_font.get_linesize(),
                    max_lines=2,
                )

            countdown_ms = max(0, countdown_until - pygame.time.get_ticks())
            if paused:
                run_status = "PAUSED"
            elif countdown_ms > 0:
                run_status = f"STARTING IN {countdown_ms / 1000:.1f}s"
            else:
                run_status = "RUNNING"
            agent_mode = args.ai_mode + (
                f"/{args.comfort_feedback_mode}"
                if args.ai_mode == "comfort_subgoal"
                else ""
            )
            status = (
                f"{run_status}   ·   YOU green + AI blue   ·   "
                f"EP {episode} / STEP {episode_step}   ·   "
                f"REWARD {episode_reward:.1f}   ·   "
                f"LEARNING {'ON' if live_learning_enabled else 'OFF'}"
            )
            timing_status = (
                f"AI {agent_mode}   ·   {args.step_hz:g} step/s   ·   "
                f"predict {last_predict_ms:.1f} ms / env {last_environment_step_ms:.1f} ms   ·   "
                f"subgoal {last_ai_subgoal}   ·   event {last_ai_event or 'none'}"
            )
            controls = (
                "WASD / ARROWS  Move     SPACE  Interact     P  Pause     "
                "R  Restart     ESC  Quit"
            )
            draw_wrapped_text(
                screen,
                status,
                small_font,
                (235, 238, 242),
                GAME_STATUS_RECT,
                line_height=small_font.get_linesize(),
                max_lines=2,
            )
            draw_wrapped_text(
                screen,
                timing_status,
                small_font,
                (147, 184, 228),
                GAME_TIMING_RECT,
                line_height=small_font.get_linesize(),
                max_lines=2,
            )
            draw_wrapped_text(
                screen,
                controls,
                meta_font,
                (163, 205, 176),
                GAME_CONTROLS_RECT,
                line_height=meta_font.get_linesize(),
                max_lines=1,
            )

            screen.blit(
                font.render("DURF KITCHEN LAB", True, (245, 245, 245)),
                (SIDEBAR_RECT.x + 20, SIDEBAR_RECT.y + 17),
            )
            sidebar_state = (
                "FEEDBACK OPEN · GAME PAUSED"
                if chat_open
                else "PAUSED · READY FOR FEEDBACK"
                if paused
                else "RUNNING · PAUSE TO GIVE FEEDBACK"
            )
            draw_wrapped_text(
                screen,
                f"{mode_banner}\n{sidebar_state}",
                meta_font,
                (
                    (255, 175, 105)
                    if not live_learning_enabled
                    else (150, 205, 245)
                    if paused
                    else (170, 210, 180)
                ),
                pygame.Rect(
                    SIDEBAR_RECT.x + 20,
                    SIDEBAR_RECT.y + 53,
                    SIDEBAR_RECT.width - 40,
                    48,
                ),
                line_height=meta_font.get_linesize(),
                max_lines=3,
            )

            if chat_open:
                render_chat_panel(
                    screen,
                    font,
                    small_font,
                    meta_font,
                    chat_messages,
                    chat_input,
                    chat_cursor,
                    chat_composition,
                    chat_pending,
                    chat_status,
                    mode_banner,
                    live_learning_enabled,
                    chat_scroll,
                )
            else:
                render_feedback_home_panel(
                    screen,
                    font,
                    small_font,
                    meta_font,
                    chat_status=chat_status,
                    episode=episode,
                    episode_step=episode_step,
                    last_ai_subgoal=last_ai_subgoal,
                    last_ai_event=last_ai_event,
                    learning_enabled=live_learning_enabled,
                )

            draw_button(
                screen,
                CHAT_BUTTON,
                (
                    "Close Feedback"
                    if chat_open
                    else "Pause & Feedback"
                    if live_learning_enabled
                    else "Pause & Record"
                ),
                small_font,
                (70, 115, 155) if chat_open else (75, 105, 135),
            )
            draw_button(
                screen,
                PAUSE_BUTTON,
                "Resume Game" if paused else "Pause Game",
                small_font,
                (85, 140, 95) if paused else (145, 100, 68),
            )
            pygame.display.flip()
            clock.tick(args.render_fps)
    finally:
        try:
            if comfort_agent is not None:
                final_learner_state = comfort_agent.learner_state_dict()
                final_learner_state_path.write_text(
                    json.dumps(final_learner_state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            session_manifest.update(
                {
                    "ended_utc": utc_timestamp(),
                    "exit_reason": quit_reason,
                    "episodes_started": int(episode),
                    "total_steps": int(total_step),
                    "human_feedback_events": int(feedback_submission_count),
                    "human_feedback_submissions": int(feedback_submission_count),
                    "feedback_update_status_counts": dict(
                        sorted(feedback_update_status_counts.items())
                    ),
                    "learner_updates_applied": int(learner_updates_applied),
                }
            )
            session_manifest_path.write_text(
                json.dumps(session_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"Session manifest finalization failed: {exc}", file=sys.stderr)
        trajectory_handle.flush()
        chat_handle.flush()
        pause_handle.flush()
        feedback_updates_handle.flush()
        trajectory_handle.close()
        chat_handle.close()
        pause_handle.close()
        feedback_updates_handle.close()
        for cached_env in set(envs_by_layout.values()):
            cached_env.close()
        pygame.key.stop_text_input()
        pygame.quit()

    print(f"Trajectory: {trajectory_path}")
    print(f"Chat messages: {chat_path}")
    print(f"Feedback updates: {feedback_updates_path}")
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
