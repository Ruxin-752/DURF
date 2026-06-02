"""Unit tests for HumanFeedbackEnvWrapper (Phase 2 验收标准).

使用纯 mock env，不依赖 PantheonRL / Overcooked，可在任何含 gym 的环境中运行。
"""
import queue
import pytest
import numpy as np

# 兼容 gym（旧）和 gymnasium（新）：优先用 gym，没有则用 gymnasium
try:
    import gym
    import gym.spaces as spaces
    _GymBase = gym.Env
except ModuleNotFoundError:
    import gymnasium as gym          # type: ignore[no-redef]
    import gymnasium.spaces as spaces  # type: ignore[assignment]
    _GymBase = gym.Env               # type: ignore[misc]

from src.feedback.protocol import FeedbackEvent
from src.env.human_feedback_wrapper import HumanFeedbackEnvWrapper


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _ConstantRewardEnv(_GymBase):
    """每 step 返回固定 env_reward，done 由构造参数控制。"""

    def __init__(self, env_reward: float = 1.0, episode_len: int = 5):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._env_reward = env_reward
        self._episode_len = episode_len
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return self.observation_space.sample()

    def step(self, action):
        self._t += 1
        done = self._t >= self._episode_len
        return self.observation_space.sample(), self._env_reward, done, {}

    def seed(self, seed=None):
        pass


def _make_env(env_reward=1.0, episode_len=5, alpha=1.0, max_signal=5.0):
    q = queue.Queue()
    env = HumanFeedbackEnvWrapper(
        _ConstantRewardEnv(env_reward=env_reward, episode_len=episode_len),
        feedback_queue=q,
        alpha=alpha,
        max_signal_per_step=max_signal,
    )
    return env, q


def _put(q, signal_value, source="keyboard"):
    event = FeedbackEvent(
        timestamp_ms=0, event_type="positive" if signal_value > 0 else "negative",
        signal_value=signal_value, source=source,
    )
    q.put(event)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_feedback_passes_through():
    """没有反馈时，total_reward == env_reward。"""
    env, _ = _make_env(env_reward=2.0)
    env.reset()
    _, r, _, info = env.step(0)
    assert r == pytest.approx(2.0)
    assert info["env_reward"] == pytest.approx(2.0)
    assert info["human_reward"] == pytest.approx(0.0)
    assert info["total_reward"] == pytest.approx(2.0)


def test_positive_feedback_adds_to_reward():
    """注入 +1 时，total_reward = env_reward + 1。"""
    env, q = _make_env(env_reward=1.0)
    env.reset()
    _put(q, +1.0)
    _, r, _, info = env.step(0)
    assert r == pytest.approx(2.0)
    assert info["human_reward"] == pytest.approx(1.0)


def test_negative_feedback_subtracts_reward():
    """注入 -1 时，total_reward = env_reward - 1。"""
    env, q = _make_env(env_reward=1.0)
    env.reset()
    _put(q, -1.0)
    _, r, _, info = env.step(0)
    assert r == pytest.approx(0.0)
    assert info["human_reward"] == pytest.approx(-1.0)


def test_alpha_scaling():
    """alpha=2.0 时，human 贡献 = 2 * signal_value。"""
    env, q = _make_env(env_reward=0.0, alpha=2.0)
    env.reset()
    _put(q, 1.0)
    _, r, _, info = env.step(0)
    assert r == pytest.approx(2.0)
    assert info["human_reward"] == pytest.approx(1.0)   # raw signal
    assert info["total_reward"] == pytest.approx(2.0)  # alpha * signal


def test_max_signal_clipping():
    """累积 signal 超过 max_signal_per_step 时被裁剪。"""
    env, q = _make_env(env_reward=0.0, max_signal=3.0)
    env.reset()
    for _ in range(10):
        _put(q, 1.0)  # 累积 10，但上限 3
    _, r, _, info = env.step(0)
    assert r == pytest.approx(3.0)
    assert info["human_reward"] == pytest.approx(3.0)


def test_info_dict_has_breakdown():
    """info 字典包含 env_reward / human_reward / total_reward 三个字段。"""
    env, _ = _make_env()
    env.reset()
    _, _, _, info = env.step(0)
    for key in ("env_reward", "human_reward", "total_reward"):
        assert key in info, f"缺少 info['{key}']"


def test_step_count_increments():
    """每次 step，info['step_count'] 递增。"""
    env, _ = _make_env()
    env.reset()
    for expected in range(3):
        _, _, _, info = env.step(0)
        assert info["step_count"] == expected


def test_episode_human_reward_resets():
    """episode 结束时，episode_human_reward 被写入 info，下轮重置为 0。"""
    env, q = _make_env(env_reward=0.0, episode_len=2)
    env.reset()
    _put(q, 1.0)
    env.step(0)           # step 1, not done
    _put(q, 1.0)
    _, _, done, info = env.step(0)   # step 2, done=True
    assert done
    assert info["episode_human_reward"] == pytest.approx(2.0)
    # 重置后内部计数归零
    env.reset()
    assert env._episode_human_reward == pytest.approx(0.0)
