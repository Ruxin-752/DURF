"""Train a self-play RLlib PPO agent for a DURF Overcooked layout.

This is a small command-line wrapper around the original HARL RLlib trainer
utilities. It keeps training outputs in `outputs/rllib_training/` and copies
the latest checkpoint into `models/rllib_agents/<agent-name>/agent` so the
pygame scripts can load it.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "models" / "rllib_agents"


DEFAULT_LAYOUT = "ring_tomato_onion_10x6"
DEFAULT_AGENT_NAME = "RllibRingTomatoOnion10x6SP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=800)
    parser.add_argument("--sgd-minibatch-size", type=int, default=800)
    parser.add_argument("--num-sgd-iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--old-dynamics", action="store_true", default=True)
    parser.add_argument("--no-old-dynamics", dest="old_dynamics", action="store_false")
    parser.add_argument("--use-phi", action="store_true", default=True)
    parser.add_argument("--no-use-phi", dest="use_phi", action="store_false")
    parser.add_argument(
        "--results-dir",
        default=str(REPO_ROOT / "outputs" / "rllib_training"),
    )
    parser.add_argument(
        "--ray-temp-dir",
        default=str(REPO_ROOT / "outputs" / "ray_tmp"),
        help="Ray session directory; kept inside the repo outputs folder by default.",
    )
    parser.add_argument(
        "--ray-local-mode",
        action="store_true",
        help="Run Ray tasks in the driver process; slower but useful for Windows smoke tests.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Train only; do not copy latest checkpoint into models/rllib_agents.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing installed agent directory.",
    )
    return parser.parse_args()


def build_training_params(args: argparse.Namespace, results_dir: Path, ray_temp_dir: Path):
    """Build the HARL/RLlib params dict without going through Sacred."""
    from human_aware_rl.imitation.behavior_cloning_tf2 import (
        BC_SAVE_DIR,
        BehaviorCloningPolicy,
    )
    from human_aware_rl.ppo.ppo_rllib import RllibLSTMPPOModel, RllibPPOModel
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent

    def env_creator(env_config):
        from human_aware_rl.rllib.rllib import OvercookedMultiAgent

        return OvercookedMultiAgent.from_config(env_config)

    model_params = {
        "use_lstm": False,
        "NUM_HIDDEN_LAYERS": 3,
        "SIZE_HIDDEN_LAYERS": 64,
        "NUM_FILTERS": 25,
        "NUM_CONV_LAYERS": 3,
        "CELL_SIZE": 256,
        "D2RL": False,
    }
    training_params = {
        "num_workers": args.num_workers,
        "train_batch_size": args.train_batch_size,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "rollout_fragment_length": 400,
        "num_sgd_iter": args.num_sgd_iter,
        "lr": args.lr,
        "lr_schedule": None,
        "grad_clip": 0.1,
        "gamma": 0.99,
        "lambda": 0.98,
        "vf_share_layers": True,
        "vf_loss_coeff": 1e-4,
        "kl_coeff": 0.2,
        "clip_param": 0.05,
        "num_gpus": 0,
        "seed": args.seed,
        "evaluation_interval": max(args.iterations + 1, 10),
        "entropy_coeff_schedule": [(0, 0.2), (3e5, 0.1)],
        "eager_tracing": False,
        "log_level": "ERROR",
    }
    environment_params = {
        "mdp_params": {
            "layout_name": args.layout,
            "rew_shaping_params": {
                "PLACEMENT_IN_POT_REW": 3,
                "DISH_PICKUP_REWARD": 3,
                "SOUP_PICKUP_REWARD": 5,
                "DISH_DISP_DISTANCE_REW": 0,
                "POT_DISTANCE_REW": 0,
                "SOUP_DISTANCE_REW": 0,
            },
            "old_dynamics": args.old_dynamics,
        },
        "env_params": {"horizon": 400},
        "multi_agent_params": {
            "reward_shaping_factor": 1.0,
            "reward_shaping_horizon": float("inf"),
            "use_phi": args.use_phi,
            "bc_schedule": OvercookedMultiAgent.self_play_bc_schedule,
        },
    }
    return {
        "model_params": model_params,
        "training_params": training_params,
        "environment_params": environment_params,
        "bc_params": {
            "bc_policy_cls": BehaviorCloningPolicy,
            "bc_config": {
                "model_dir": str(Path(BC_SAVE_DIR) / "default"),
                "stochastic": True,
                "eager": False,
            },
        },
        "shared_policy": True,
        "num_training_iters": args.iterations,
        "evaluation_params": {
            "ep_length": 400,
            "num_games": 1,
            "display": False,
        },
        "experiment_name": f"PPO_{args.layout}_durf_sp",
        "save_every": args.save_every,
        "results_dir": str(results_dir),
        "ray_params": {
            "custom_model_id": "MyPPOModel",
            "custom_model_cls": (
                RllibLSTMPPOModel if model_params["use_lstm"] else RllibPPOModel
            ),
            "temp_dir": str(ray_temp_dir),
            "local_mode": args.ray_local_mode,
            "env_creator": env_creator,
        },
        "resume_checkpoint_path": None,
        "verbose": False,
    }


def train(params: dict) -> dict:
    import ray
    from human_aware_rl.rllib.rllib import gen_trainer_from_params, save_trainer

    original_ray_init = ray.init

    def init_with_local_mode(*args, **kwargs):
        kwargs.setdefault("local_mode", params["ray_params"].get("local_mode", False))
        return original_ray_init(*args, **kwargs)

    ray.init = init_with_local_mode
    trainer = gen_trainer_from_params(params)
    result = {}
    try:
        for iteration in range(params["num_training_iters"]):
            print(f"Starting training iteration {iteration + 1}/{params['num_training_iters']}")
            result = trainer.train()
            if iteration % params["save_every"] == 0:
                save_path = save_trainer(trainer, params)
                print(f"Saved checkpoint: {save_path}")
        save_path = save_trainer(trainer, params)
        print(f"Saved final checkpoint: {save_path}")
        return result
    finally:
        ray.init = original_ray_init
        trainer.stop()


def latest_run_dir(results_dir: Path, experiment_name: str) -> Path:
    runs = sorted(
        results_dir.glob(f"{experiment_name}*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No training run directory found for {experiment_name}")
    return runs[0]


def latest_checkpoint_dir(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.glob("checkpoint_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        checkpoints = sorted(
            run_dir.glob("checkpoint-*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    return checkpoints[0]


def install_run(run_dir: Path, checkpoint_path: Path, agent_name: str, overwrite: bool) -> Path:
    install_root = AGENT_ROOT / agent_name
    agent_dir = install_root / "agent"
    if agent_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Agent already exists: {agent_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(agent_dir)
    install_root.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    shutil.copy2(config_path, install_root / "config.pkl")

    agent_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.is_dir():
        shutil.copytree(checkpoint_path, agent_dir / checkpoint_path.name)
    else:
        shutil.copy2(checkpoint_path, agent_dir / checkpoint_path.name)
        for metadata_name in (".is_checkpoint", ".tune_metadata"):
            metadata_path = checkpoint_path.parent / metadata_name
            if metadata_path.exists():
                shutil.copy2(metadata_path, agent_dir / metadata_name)
    return agent_dir


def main() -> int:
    args = parse_args()
    # Must be set before importing ppo_rllib_client so its local-testing
    # defaults use small CPU-friendly values.
    os.environ.setdefault("RUN_ENV", "local")

    import ray

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    ray_temp_dir = Path(args.ray_temp_dir).resolve()
    ray_temp_dir.mkdir(parents=True, exist_ok=True)

    params = build_training_params(args, results_dir, ray_temp_dir)
    experiment_name = params["experiment_name"]

    print(f"Training layout: {args.layout}")
    print(f"Iterations: {args.iterations}")
    print(f"Results dir: {results_dir}")
    print(f"Ray temp dir: {ray_temp_dir}")
    print(f"Install agent name: {args.agent_name}")
    result = train(params)
    print(f"Training result: {result}")

    run_dir = latest_run_dir(results_dir, experiment_name)
    checkpoint_path = latest_checkpoint_dir(run_dir)
    print(f"Latest run: {run_dir}")
    print(f"Latest checkpoint: {checkpoint_path}")
    if not args.no_install:
        agent_dir = install_run(run_dir, checkpoint_path, args.agent_name, args.overwrite)
        print(f"Installed agent: {agent_dir}")

    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
