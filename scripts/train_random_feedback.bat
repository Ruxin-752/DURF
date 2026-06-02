@echo off
REM random-feedback: auto-inject random signals, no human needed
REM Usage: train_random_feedback.bat [participant_id]
cd /d "%~dp0.."
C:\Users\my185\miniconda3\envs\pantheonrl_env\python.exe -m src.train.trainer ^
    --mode random-feedback ^
    --total-timesteps 1000000 ^
    --participant %~1 ^
    --seed 42 ^
    --config config/default.yaml
pause
