"""Env factory — 统一的 env 构造入口。

Wrapper 顺序（由内到外）:
    OvercookedMultiEnv          ← PantheonRL multi-agent env
    HumanFeedbackEnvWrapper     ← 我们的注入层（仅 feedback_queue 非 None 时启用）
    Monitor                     ← SB3 内置，必须在最外层（SB3 issue #146）

运行环境要求: conda activate pantheonrl_env
"""
from __future__ import annotations

import gym
from pathlib import Path
from typing import Any, Optional

import overcookedgym  # noqa: F401 — 注册 OvercookedMultiEnv-v0
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from pantheonrl.common.agents import OnPolicyAgent

from src.env.human_feedback_wrapper import HumanFeedbackEnvWrapper


def make_overcooked_env(
    layout_name: str = "cramped_room",
    feedback_queue: Optional[Any] = None,
    alpha: float = 1.0,
    seed: int = 42,
    monitor_path: Optional[str | Path] = None,
) -> gym.Env:
    """构造完整的 Overcooked training env。

    Parameters
    ----------
    layout_name    : Overcooked 地图名，默认 'cramped_room'
    feedback_queue : multiprocessing.Queue，None 表示 no-feedback 模式
    alpha          : human reward 缩放系数
    seed           : 随机种子
    monitor_path   : Monitor 日志目录；None 则不写磁盘
    """
    # 1. 创建 multi-agent base env
    env = gym.make("OvercookedMultiEnv-v0", layout_name=layout_name)

    # 2. 为第二个 agent 创建随机 PPO partner（初始权重，未训练）
    #    getDummyEnv(1) 返回只有 observation_space / action_space 的哑 env
    dummy_alt = env.getDummyEnv(1)
    partner_model = PPO("MlpPolicy", DummyVecEnv([lambda: dummy_alt]), verbose=0, device="cpu")
    partner = OnPolicyAgent(partner_model)
    env.add_partner_agent(partner)

    # 3. 注入 human feedback（feedback_queue 存在时才包裹）
    if feedback_queue is not None:
        env = HumanFeedbackEnvWrapper(env, feedback_queue, alpha=alpha)

    # 4. Monitor 必须在最外层，否则记录的是注入前的 reward（SB3 issue #146）
    # 当前 PantheonRL 环境使用 SB3 1.7.x，它期望旧版 gym spaces；
    # 这里不能包 shimmy/GymV21CompatibilityV0，否则 action_space 会变成 gymnasium.spaces。
    filename = str(monitor_path) if monitor_path else None
    env = Monitor(env, filename=filename)

    # 5. 设置 seed
    if seed is not None:
        env.seed(seed)
        env.reset()

    return env
