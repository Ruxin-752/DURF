"""Quick environment verification script"""
import sys
import os

# Get project root (scripts/../ = group_a_baseline/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

print(f"Python: {sys.version}")

# Test yaml
import yaml
print(f"yaml: {yaml.__version__}")

# Test gym
import gym
print(f"gym: {gym.__version__}")

# Test stable-baselines3
import stable_baselines3
print(f"stable-baselines3: {stable_baselines3.__version__}")

# Test torch
import torch
print(f"torch: {torch.__version__}")

# Test config loading (use utf-8 encoding)
with open('config/default.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(f"Config: layout={cfg['env']['layout']}, algo={cfg['training']['algorithm']}")

# Test local imports
from src.feedback.protocol import FeedbackEvent, FeedbackSource
print(f"FeedbackEvent: OK")
from src.envs.wrapper import HumanFeedbackWrapper
print(f"HumanFeedbackWrapper: OK")
from src.utils.repro import set_seed
print(f"set_seed: OK")

print("\n✅ All imports successful! Environment is ready.")
