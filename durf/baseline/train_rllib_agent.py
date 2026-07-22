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
    parser.add_argument(
        "--profile",
        choices=("default", "paper-ring"),
        default="default",
        help=(
            "Training profile. 'paper-ring' uses PPO hyperparameters close to "
            "the original coordination_ring reproduction settings."
        ),
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=800)
    parser.add_argument("--sgd-minibatch-size", type=int, default=800)
    parser.add_argument("--num-sgd-iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--vf-loss-coeff", type=float, default=1e-4)
    parser.add_argument("--kl-coeff", type=float, default=0.2)
    parser.add_argument("--clip-param", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--reward-shaping-horizon",
        type=float,
        default=float("inf"),
        help="Timestep where dense reward shaping linearly anneals to zero.",
    )
    parser.add_argument(
        "--tomato-pickup-reward",
        type=float,
        default=0.0,
        help="Curriculum-only dense reward for picking up a tomato.",
    )
    parser.add_argument(
        "--placement-in-pot-reward",
        type=float,
        default=3.0,
        help="Curriculum dense reward for placing a valid ingredient in a pot.",
    )
    parser.add_argument(
        "--tomato-disp-distance-reward",
        type=float,
        default=0.0,
        help="Curriculum reward for moving closer to a tomato dispenser while empty-handed.",
    )
    parser.add_argument(
        "--tomato-to-pot-distance-reward",
        type=float,
        default=0.0,
        help="Curriculum reward for moving closer to a pot while holding tomato.",
    )
    parser.add_argument(
        "--onion-to-pot-distance-reward",
        type=float,
        default=0.0,
        help="Curriculum reward for moving closer to a pot while holding onion.",
    )
    parser.add_argument(
        "--onion-pickup-reward",
        type=float,
        default=0.0,
        help="Curriculum-only dense reward for picking up an onion.",
    )
    parser.add_argument(
        "--dish-pickup-reward",
        type=float,
        default=3.0,
        help="Curriculum dense reward for picking up a dish.",
    )
    parser.add_argument(
        "--ready-dish-pickup-reward",
        type=float,
        default=0.0,
        help=(
            "Curriculum dense reward for picking up a dish only when a ready "
            "soup is available."
        ),
    )
    parser.add_argument(
        "--soup-pickup-reward",
        type=float,
        default=5.0,
        help="Curriculum dense reward for picking up a ready soup.",
    )
    parser.add_argument(
        "--dish-disp-distance-reward",
        type=float,
        default=0.0,
        help="Curriculum reward for moving closer to a dish dispenser when a dish is useful.",
    )
    parser.add_argument(
        "--pot-distance-reward",
        type=float,
        default=0.0,
        help="Curriculum reward for moving closer to a pot while holding a dish.",
    )
    parser.add_argument(
        "--soup-distance-reward",
        type=float,
        default=0.0,
        help="Curriculum reward for moving closer to a serving location while holding soup.",
    )
    parser.add_argument(
        "--comfort-shaping-weights",
        type=Path,
        default=None,
        help=(
            "Path to learned comfort weights JSON (from "
            "evaluate_route2_subgoal.py). When set, PPO reward is shaped by the "
            "learned comfort reward; omit for plain task training."
        ),
    )
    parser.add_argument(
        "--comfort-shaping-coeff",
        type=float,
        default=0.5,
        help="Coefficient beta on the comfort shaping term.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--old-dynamics", action="store_true", default=True)
    parser.add_argument("--no-old-dynamics", dest="old_dynamics", action="store_false")
    parser.add_argument("--use-phi", action="store_true", default=None)
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
        "--resume-from",
        type=Path,
        help="Checkpoint file or checkpoint_*/ directory to continue training from.",
    )
    parser.add_argument(
        "--resume-installed",
        action="store_true",
        help="Continue from models/rllib_agents/<agent-name>/agent latest checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing installed agent directory.",
    )
    args = parser.parse_args()
    if args.profile == "paper-ring":
        args.lr = 2.4e-4
        args.gamma = 0.977
        args.gae_lambda = 0.6
        args.vf_loss_coeff = 0.024
        args.kl_coeff = 0.176
        args.clip_param = 0.063
        args.grad_clip = 0.254
        args.reward_shaping_horizon = 4_500_000
        if args.use_phi is None:
            args.use_phi = False
    elif args.use_phi is None:
        args.use_phi = True
    return args


def build_training_params(args: argparse.Namespace, results_dir: Path, ray_temp_dir: Path):
    """Build the HARL/RLlib params dict without going through Sacred."""
    from human_aware_rl.imitation.behavior_cloning_tf2 import (
        BC_SAVE_DIR,
        BehaviorCloningPolicy,
    )
    from human_aware_rl.ppo.ppo_rllib import RllibLSTMPPOModel, RllibPPOModel
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent

    comfort_enabled = args.comfort_shaping_weights is not None
    if comfort_enabled:
        from durf.baseline.comfort_env import comfort_env_creator as env_creator
    else:
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
        "rollout_fragment_length": args.train_batch_size,
        "num_sgd_iter": args.num_sgd_iter,
        "lr": args.lr,
        "lr_schedule": None,
        "grad_clip": args.grad_clip,
        "gamma": args.gamma,
        "lambda": args.gae_lambda,
        "vf_share_layers": True,
        "vf_loss_coeff": args.vf_loss_coeff,
        "kl_coeff": args.kl_coeff,
        "clip_param": args.clip_param,
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
                "PLACEMENT_IN_POT_REW": args.placement_in_pot_reward,
                "TOMATO_PICKUP_REWARD": args.tomato_pickup_reward,
                "TOMATO_DISP_DISTANCE_REWARD": args.tomato_disp_distance_reward,
                "TOMATO_TO_POT_DISTANCE_REWARD": args.tomato_to_pot_distance_reward,
                "ONION_TO_POT_DISTANCE_REWARD": args.onion_to_pot_distance_reward,
                "ONION_PICKUP_REWARD": args.onion_pickup_reward,
                "DISH_PICKUP_REWARD": args.dish_pickup_reward,
                "READY_DISH_PICKUP_REWARD": args.ready_dish_pickup_reward,
                "SOUP_PICKUP_REWARD": args.soup_pickup_reward,
                "DISH_DISP_DISTANCE_REW": args.dish_disp_distance_reward,
                "POT_DISTANCE_REW": args.pot_distance_reward,
                "SOUP_DISTANCE_REW": args.soup_distance_reward,
            },
            "old_dynamics": args.old_dynamics,
        },
        "env_params": {"horizon": args.horizon},
        "multi_agent_params": {
            "reward_shaping_factor": 1.0,
            "reward_shaping_horizon": args.reward_shaping_horizon,
            "use_phi": args.use_phi,
            "bc_schedule": OvercookedMultiAgent.self_play_bc_schedule,
        },
    }
    if comfort_enabled:
        # Ignored by OvercookedMultiAgent.from_config; read by comfort_env_creator.
        environment_params["comfort_shaping"] = {
            "enabled": True,
            "weights_path": str(Path(args.comfort_shaping_weights).resolve()),
            "coeff": float(args.comfort_shaping_coeff),
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
        kwargs.setdefault("num_gpus", 0)
        kwargs.setdefault("include_dashboard", False)
        return original_ray_init(*args, **kwargs)

    ray.init = init_with_local_mode
    trainer = gen_trainer_from_params(params)
    result = {}
    try:
        resume_checkpoint_path = params.get("resume_checkpoint_path")
        if resume_checkpoint_path:
            print(f"Restoring checkpoint: {resume_checkpoint_path}")
            trainer.restore(str(resume_checkpoint_path))
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


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from and args.resume_installed:
        raise ValueError("Use only one of --resume-from or --resume-installed.")
    if args.resume_from:
        checkpoint_path = args.resume_from.resolve()
    elif args.resume_installed:
        checkpoint_path = latest_checkpoint_dir(AGENT_ROOT / args.agent_name / "agent")
    else:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


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
    params["resume_checkpoint_path"] = resolve_resume_checkpoint(args)
    experiment_name = params["experiment_name"]

    print(f"Training layout: {args.layout}")
    print(f"Iterations: {args.iterations}")
    print(f"Profile: {args.profile}")
    print(f"Results dir: {results_dir}")
    print(f"Ray temp dir: {ray_temp_dir}")
    print(f"Install agent name: {args.agent_name}")
    if params["resume_checkpoint_path"]:
        print(f"Resume checkpoint: {params['resume_checkpoint_path']}")
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
