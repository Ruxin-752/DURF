"""Human feedback reward injection wrapper.

Wraps any Gym-compatible env (specifically the Overcooked PantheonRL env).
At each step, polls a multiprocessing.Queue for FeedbackEvent objects and
adds their signal_value (scaled by alpha) to the env reward.

Records env_reward / human_reward / total_reward separately in info dict
so Monitor / callbacks can log them.

兼容性说明:
  - 支持 4-tuple (obs, r, done, info) — gym 0.25.x 旧接口
  - 支持 5-tuple (obs, r, terminated, truncated, info) — gymnasium 新接口
  - Queue 接受 multiprocessing.Queue 或 queue.Queue（接口一致）
"""
from __future__ import annotations

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym  # type: ignore[no-redef]
from queue import Empty
from typing import Any


class HumanFeedbackEnvWrapper(gym.Wrapper):
    """奖励注入 wrapper：把来自 feedback_queue 的人类信号叠加到 env reward。

    Parameters
    ----------
    env           : 被包裹的 gym.Env（PantheonRL multi-agent env 经 frame_wrap 后的版本）
    feedback_queue: multiprocessing.Queue（或任何实现 get_nowait / Empty 接口的 Queue）
    alpha         : human reward 缩放系数，默认 1.0
    max_signal_per_step : 单步累积 human signal 的上限（防极端反馈）
    """

    def __init__(
        self,
        env: gym.Env,
        feedback_queue: Any,
        alpha: float = 1.0,
        max_signal_per_step: float = 5.0,
    ) -> None:
        super().__init__(env)
        self.feedback_queue = feedback_queue
        self.alpha = alpha
        self.max_signal_per_step = max_signal_per_step

        self._step_count: int = 0
        self._episode_human_reward: float = 0.0

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def step(self, action):
        result = self.env.step(action)
        obs, env_reward, done, info = self._normalize_step_return(result)

        human_signal = self._poll_feedback()
        human_signal = max(
            -self.max_signal_per_step,
            min(self.max_signal_per_step, human_signal),
        )

        total_reward = float(env_reward) + self.alpha * human_signal

        # 写入 info，让 Monitor / TrajectoryLoggerCallback 可见
        info = dict(info)  # 防止上游 in-place 修改
        info["env_reward"] = float(env_reward)
        info["human_reward"] = float(human_signal)
        info["total_reward"] = float(total_reward)
        info["step_count"] = self._step_count

        self._step_count += 1
        self._episode_human_reward += human_signal

        if done:
            info["episode_human_reward"] = self._episode_human_reward
            self._episode_human_reward = 0.0
            self._step_count = 0

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        self._step_count = 0
        self._episode_human_reward = 0.0
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poll_feedback(self) -> float:
        """一次性消费 queue 中所有积压的 FeedbackEvent，返回累积 signal。"""
        signal = 0.0
        while True:
            try:
                event = self.feedback_queue.get_nowait()
                signal += event.signal_value
            except Empty:
                break
        return signal

    @staticmethod
    def _normalize_step_return(result):
        """兼容 4-tuple 和 5-tuple step() 返回值。"""
        if len(result) == 5:
            obs, r, terminated, truncated, info = result
            return obs, r, terminated or truncated, info
        return result  # 4-tuple: obs, r, done, info
