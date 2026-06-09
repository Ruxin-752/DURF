"""Runtime helpers for the archived PantheonRL + SB3 baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gym
import overcookedgym  # noqa: F401 - registers OvercookedMultiEnv-v0
from pantheonrl.common.agents import OnPolicyAgent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO_ROOT
    / "archive_local"
    / "group_a_baseline_failed"
    / "logs"
    / "no-feedback_P00_20260602_224619"
    / "final_model.zip"
)


def resolve_model_path(model_path: str | Path | None) -> Path:
    path = Path(model_path) if model_path else DEFAULT_MODEL
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return path


def make_baseline_env(layout_name: str = "cramped_room", seed: int = 42):
    """Create the same ego-training environment used by the archived run."""
    env = gym.make("OvercookedMultiEnv-v0", layout_name=layout_name)

    # The archived training run used a newly initialized PPO policy as partner.
    dummy_partner_env = env.getDummyEnv(1)
    partner_model = PPO(
        "MlpPolicy",
        DummyVecEnv([lambda: dummy_partner_env]),
        seed=seed,
        device="cpu",
        verbose=0,
    )
    env.add_partner_agent(OnPolicyAgent(partner_model))
    env.seed(seed)
    return env


def make_direct_multi_env(layout_name: str = "cramped_room", seed: int = 42):
    """Create an environment whose two actions are supplied by the caller."""
    wrapped_env = gym.make("OvercookedMultiEnv-v0", layout_name=layout_name)
    env = wrapped_env.unwrapped
    env.seed(seed)
    return env


def reset_env(env) -> Any:
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, float(reward), bool(done), info


def get_base_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current.base_env
