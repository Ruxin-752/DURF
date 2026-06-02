@echo off
REM no-feedback baseline: pure environment reward, no human input
REM Usage: train_no_feedback.bat [participant_id]
cd /d "%~dp0.."
C:\Users\my185\miniconda3\envs\pantheonrl_env\python.exe -m src.train.trainer ^
    --mode no-feedback ^
    --total-timesteps 1000000 ^
    --participant %~1 ^
    --seed 42 ^
    --config config/default.yaml
pause
