"""Comfort-shaped RLlib env creator for PPO training.

Wraps ``OvercookedMultiAgent`` so each step adds the learned comfort reward to
both agents' shaped reward. In self-play there is no separate human, so each
agent is rewarded for being considerate of *the other* player (symmetric
comfort), annealed together with the existing dense-reward curriculum via
``reward_shaping_factor``.

Comfort is configured through an extra ``comfort_shaping`` key in ``env_config``
which ``OvercookedMultiAgent.from_config`` ignores, so plain task training is
untouched unless the flag is set.
"""

from __future__ import annotations


def _wrap_step_with_comfort(env, comfort):
    """Wrap ``env.step`` so both agents get ``comfort.shaping`` added.

    ``comfort`` is any object exposing ``shaping(state, mdp, ai_index)`` (a
    :class:`~durf.baseline.comfort_reward.ComfortReward` at runtime, a stub in
    tests), which keeps this wrapper unit-testable without ray/overcooked.
    """

    original_step = env.step  # capture the original bound method

    def step(action_dict):
        obs, rewards, dones, infos = original_step(action_dict)
        try:
            mdp = env.base_env.mdp
            state = env.base_env.state  # state after the joint action
            factor = float(getattr(env, "reward_shaping_factor", 1.0))
            comfort0 = factor * comfort.shaping(state, mdp, ai_index=0)
            comfort1 = factor * comfort.shaping(state, mdp, ai_index=1)
            rewards[env.curr_agents[0]] += comfort0
            rewards[env.curr_agents[1]] += comfort1
            for agent, value in (
                (env.curr_agents[0], comfort0),
                (env.curr_agents[1], comfort1),
            ):
                if agent in infos and isinstance(infos[agent], dict):
                    infos[agent]["comfort_shaping"] = value
        except Exception as exc:  # never let shaping crash a rollout
            print(f"[comfort_env] shaping skipped this step: {exc}")
        return obs, rewards, dones, infos

    env.step = step
    return env


def comfort_env_creator(env_config):
    """RLlib env creator that optionally applies learned comfort shaping."""

    from human_aware_rl.rllib.rllib import OvercookedMultiAgent

    env = OvercookedMultiAgent.from_config(env_config)
    comfort_cfg = env_config.get("comfort_shaping") or {}
    if comfort_cfg.get("enabled"):
        from durf.baseline.comfort_reward import ComfortReward

        comfort = ComfortReward(
            weights_path=comfort_cfg.get("weights_path"),
            coeff=float(comfort_cfg.get("coeff", 0.5)),
        )
        env = _wrap_step_with_comfort(env, comfort)
    return env
