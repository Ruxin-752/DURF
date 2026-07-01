"""Runtime helpers for the archived RLlib PPO agents used by pygame."""

from __future__ import annotations

import random
from pathlib import Path

import dill
from human_aware_rl.rllib.rllib import load_agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "models" / "rllib_agents"
DEFAULT_AGENT_NAME = "RllibCrampedRoomSP"
CRAMPED_ROOM_COMPATIBLE_LAYOUTS = (
    "cramped_room",
    "cramped_room_wide",
    "cramped_room_corridor",
    "cramped_room_two_pots",
)
RING_TOMATO_ONION_LAYOUT = "ring_tomato_onion_10x6"
RING_TOMATO_ONION_CURRICULUM_LAYOUTS = (
    "ring_tomato_onion_10x6_curriculum_micro",
    "ring_tomato_onion_10x6_curriculum_micro_delivery",
    "ring_tomato_onion_10x6_curriculum_cooking_serving_spawn_clean",
    "ring_tomato_onion_10x6_curriculum_cooking_serving_spawn_near",
    "ring_tomato_onion_10x6_curriculum_ready_serving",
    "ring_tomato_onion_10x6_curriculum_ready_serving_spawn_clean",
    "ring_tomato_onion_10x6_curriculum_ready_serving_spawn_near",
    "ring_tomato_onion_10x6_curriculum_tomato_gap",
    "ring_tomato_onion_10x6_curriculum_final_tomato_held_spawn_near",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_spawn_near",
    "ring_tomato_onion_10x6_curriculum_final_onion_pickup_spawn_near",
    "ring_tomato_onion_10x6_curriculum_soup_held_at_service_target_geometry",
    "ring_tomato_onion_10x6_curriculum_soup_held_target_geometry",
    "ring_tomato_onion_10x6_curriculum_dish_held_ready_soup_target_geometry",
    "ring_tomato_onion_10x6_curriculum_empty_ready_soup_target_geometry",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_pot_near",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_top_left",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_mid_left_2",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_mid_left_3",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_bottom_mid_2",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_bottom_mid_3",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_bottom_left",
    "ring_tomato_onion_10x6_curriculum_final_onion_held_target_bottom",
    "ring_tomato_onion_10x6_curriculum_final_onion_pickup_target_geometry",
    "ring_tomato_onion_10x6_curriculum_second_tomato_held_spawn_near",
    "ring_tomato_onion_10x6_curriculum_one_tomato_finish_spawn_clean",
    "ring_tomato_onion_10x6_curriculum_one_tomato_finish_spawn_near",
    "ring_tomato_onion_10x6_curriculum_two_tomato_finish_spawn_near",
    "ring_tomato_onion_10x6_curriculum_tomato_down_potting",
    "ring_tomato_onion_10x6_curriculum_tomato_down_spawn_near",
    "ring_tomato_onion_10x6_curriculum_tomato_down_near",
    "ring_tomato_onion_10x6_curriculum_tomato_down_mid",
    "ring_tomato_onion_10x6_curriculum_open_delivery",
    "ring_tomato_onion_10x6_curriculum_open",
    "ring_tomato_onion_10x6_curriculum_easy",
    "ring_tomato_onion_10x6_curriculum_corridor",
    "ring_tomato_onion_10x6_curriculum_tomato",
)
DEFAULT_PLAYABLE_LAYOUTS = (
    *CRAMPED_ROOM_COMPATIBLE_LAYOUTS,
    *RING_TOMATO_ONION_CURRICULUM_LAYOUTS,
    RING_TOMATO_ONION_LAYOUT,
)
DEFAULT_MDP_PARAMS = {
    "old_dynamics": True,
    "rew_shaping_params": {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    },
}
AGENT_LAYOUTS = {
    "RllibAsymmetricAdvantagesBC": "asymmetric_advantages",
    "RllibAsymmetricAdvantagesSP": "asymmetric_advantages",
    "RllibCoordinationRingBC": "coordination_ring",
    "RllibCoordinationRingSP": "coordination_ring",
    "RllibCounterCircuit1OrderBC": "counter_circuit_o_1order",
    "RllibCounterCircuit1OrderSP": "counter_circuit_o_1order",
    "RllibCrampedRoomBC": "cramped_room",
    "RllibCrampedRoomSP": "cramped_room",
    "RllibForcedCoordinationBC": "forced_coordination",
    "RllibForcedCoordinationSP": "forced_coordination",
}
AGENT_COMPATIBLE_LAYOUTS = {
    "RllibCrampedRoomBC": CRAMPED_ROOM_COMPATIBLE_LAYOUTS,
    "RllibCrampedRoomSP": CRAMPED_ROOM_COMPATIBLE_LAYOUTS,
    "RllibRingTomatoOnion10x6SP": (
        *RING_TOMATO_ONION_CURRICULUM_LAYOUTS,
        RING_TOMATO_ONION_LAYOUT,
    ),
}


def resolve_agent_dir(agent: str | Path | None) -> Path:
    """Resolve an RLlib agent name or directory to its checkpoint directory."""
    agent_ref = Path(agent) if agent else Path(DEFAULT_AGENT_NAME)
    path = agent_ref if agent_ref.is_absolute() else AGENT_ROOT / agent_ref
    path = path.resolve()
    agent_dir = path / "agent" if path.is_dir() and (path / "agent").exists() else path
    if not agent_dir.exists():
        raise FileNotFoundError(f"RLlib agent not found: {agent_dir}")
    return agent_dir


def agent_name_from_dir(agent_dir: Path) -> str:
    return agent_dir.parent.name if agent_dir.name == "agent" else agent_dir.name


def expected_layout_for_agent(agent: str | Path | None) -> str | None:
    name = agent_name_from_dir(resolve_agent_dir(agent))
    return AGENT_LAYOUTS.get(name)


def compatible_layouts_for_agent(agent: str | Path | None) -> tuple[str, ...]:
    agent_dir = resolve_agent_dir(agent)
    name = agent_name_from_dir(agent_dir)
    compatible_layouts = AGENT_COMPATIBLE_LAYOUTS.get(name)
    if compatible_layouts:
        return compatible_layouts
    trained_layout = trained_layout_for_agent_dir(agent_dir)
    if trained_layout:
        return (trained_layout,)
    expected = AGENT_LAYOUTS.get(name)
    return (expected,) if expected else ()


def trained_layout_for_agent_dir(agent_dir: Path) -> str | None:
    for config_path in (agent_dir / "config.pkl", agent_dir.parent / "config.pkl"):
        if not config_path.exists():
            continue
        try:
            with config_path.open("rb") as handle:
                config = dill.load(handle)
        except Exception:
            continue
        return (
            config.get("environment_params", {})
            .get("mdp_params", {})
            .get("layout_name")
        )
    return None


def filter_compatible_layouts(
    agent: str | Path | None,
    layout_names: list[str],
) -> list[str]:
    compatible_layouts = compatible_layouts_for_agent(agent)
    if not compatible_layouts:
        return layout_names
    return [layout for layout in layout_names if layout in compatible_layouts]


def ensure_agent_layout(agent: str | Path | None, layout_name: str) -> None:
    compatible_layouts = compatible_layouts_for_agent(agent)
    if compatible_layouts and layout_name not in compatible_layouts:
        raise ValueError(
            f"{agent or DEFAULT_AGENT_NAME} is configured for "
            f"{', '.join(repr(layout) for layout in compatible_layouts)}, "
            f"but requested {layout_name!r}"
        )


def load_rllib_agent(agent: str | Path | None = None, agent_index: int = 0):
    loaded = load_agent(str(resolve_agent_dir(agent)), agent_index=agent_index)
    loaded.reset()
    return loaded


def rllib_action_index(agent, state) -> int:
    action, _ = agent.action(state)
    return int(Action.ACTION_TO_INDEX[action])


class PygameOvercookedEnv:
    """Small adapter exposing the multi-agent methods used by pygame scripts."""

    def __init__(self, layout_name: str = "cramped_room", seed: int = 42):
        self.layout_name = layout_name
        self.base_env = OvercookedEnv.from_mdp(
            OvercookedGridworld.from_layout_name(
                layout_name,
                **DEFAULT_MDP_PARAMS,
            ),
            horizon=400,
            info_level=0,
        )
        self.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)

    def multi_reset(self):
        self.base_env.reset()
        return None, None

    def multi_step(self, action0: int, action1: int):
        joint_action = (
            Action.INDEX_TO_ACTION[int(action0)],
            Action.INDEX_TO_ACTION[int(action1)],
        )
        _, reward, done, info = self.base_env.step(joint_action)
        shared_reward = float(reward)
        return (None, None), (shared_reward, shared_reward), bool(done), info

    def close(self) -> None:
        return None


def make_baseline_env(layout_name: str = "cramped_room", seed: int = 42):
    """Create an environment whose two actions are supplied by RLlib agents."""
    return make_direct_multi_env(layout_name, seed)


def make_direct_multi_env(layout_name: str = "cramped_room", seed: int = 42):
    """Create an environment whose two actions are supplied by the caller."""
    return PygameOvercookedEnv(layout_name, seed)
