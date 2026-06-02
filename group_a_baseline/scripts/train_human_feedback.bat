@echo off
REM human-feedback: real-time keyboard interaction (requires display)
REM Usage: train_human_feedback.bat P01
REM Keys: Q=+1(Good) / E=-1(Bad) / C=Coach-Pause / ESC=Exit
cd /d "%~dp0.."
C:\Users\my185\miniconda3\envs\pantheonrl_env\python.exe -m src.train.trainer ^
    --mode human-feedback ^
    --total-timesteps 100000 ^
    --participant %~1 ^
    --seed 42 ^
    --config config/default.yaml
pause
