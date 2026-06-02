@echo off
REM Evaluate a trained model
REM Usage: evaluate.bat <model_path> [n_episodes]
REM Example: evaluate.bat logs\no-feedback_P00_20260602_221108\final_model.zip 10
cd /d "%~dp0.."
C:\Users\my185\miniconda3\envs\pantheonrl_env\python.exe -c "
import sys
sys.path.insert(0, '.')
from src.eval.evaluator import evaluate_model, save_evaluation_results

model_path = sys.argv[1] if len(sys.argv) > 1 else 'logs\\no-feedback_P00_20260602_221108\\final_model.zip'
n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10

metrics = evaluate_model(model_path, n_episodes=n_episodes)
save_evaluation_results(metrics, model_path.replace('.zip', '_eval.json'))
" %*
pause
