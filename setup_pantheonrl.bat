@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo   PantheonRL + Overcooked-AI 一键环境搭建
echo ============================================
echo.

REM ---- 配置变量 ----
set ENV_NAME=pantheonrl_env
set PYTHON_VER=3.10
REM PantheonRL clone 到 overcooked_ai 的同级目录
set PANTHEON_DIR=%~dp0..\PantheonRL

REM ---- 校验脚本位置 ----
if not exist "%~dp0pyproject.toml" (
    echo [错误] 请在 overcooked_ai 根目录下运行此脚本!
    echo        当前识别目录: %~dp0
    pause & exit /b 1
)

REM ---- Step 0: 创建 conda 环境 (若已存在则跳过) ----
echo [0/5] 创建 conda 环境: %ENV_NAME% (Python %PYTHON_VER%)
conda info --envs 2>nul | findstr /C:"%ENV_NAME%" >nul
if not errorlevel 1 (
    echo      环境 %ENV_NAME% 已存在, 跳过创建
) else (
    call conda create -n %ENV_NAME% python=%PYTHON_VER% -y
    if errorlevel 1 (
        echo [错误] 创建 conda 环境失败! 请确保已安装 Miniconda/Anaconda
        pause & exit /b 1
    )
)

REM ---- 后续命令直接在目标环境中执行，避免非交互终端激活失败 ----
set RUN=conda run -n %ENV_NAME%

REM ---- Step 1: 安装 PyTorch (自动检测 GPU) ----
echo [1/5] 检测 GPU 并安装 PyTorch...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo      未检测到 NVIDIA GPU, 安装 CPU 版 PyTorch
    %RUN% python -m pip install torch torchvision torchaudio
) else (
    echo      检测到 NVIDIA GPU, 安装 CUDA 12.1 版 PyTorch
    %RUN% python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)
if errorlevel 1 (
    echo [错误] 安装 PyTorch 失败! & pause & exit /b 1
)

REM ---- Step 2: 安装 overcooked_ai ----
echo [2/5] 安装 overcooked_ai (editable mode)
%RUN% python -m pip install -e "%~dp0."
if errorlevel 1 (
    echo [错误] 安装 overcooked_ai 失败! & pause & exit /b 1
)

REM ---- Step 3: Clone 并安装 PantheonRL ----
echo [3/5] 安装 PantheonRL...
if not exist "%PANTHEON_DIR%" (
    git clone https://github.com/Stanford-ILIAD/PantheonRL.git "%PANTHEON_DIR%"
    if errorlevel 1 (
        echo [错误] Clone PantheonRL 失败! 请检查网络连接
        pause & exit /b 1
    )
)
%RUN% python -m pip install -e "%PANTHEON_DIR%"
if errorlevel 1 (
    echo [错误] 安装 PantheonRL 失败! & pause & exit /b 1
)

REM ---- Step 4: 安装 gym + NumPy + shimmy ----
echo [4/5] 安装 gym 0.25.2 / numpy^<2.0 / shimmy...
REM 固定 numpy<2.0: np.bool8 在 NumPy 2.0 中被移除, 锁版本比打 patch 更可靠
%RUN% python -m pip install "gym<0.26" "numpy<2.0" "shimmy>=2.0"
if errorlevel 1 (
    echo [错误] 安装依赖失败! & pause & exit /b 1
)

REM ---- Step 5: 应用 PantheonRL patch ----
echo [5/5] 应用 PantheonRL 代码修复 (overcooked.py + agents.py)
cd /d "%PANTHEON_DIR%"
REM patch 内路径为 a/PantheonRL/..., 需要 -p2 去掉两层前缀
git apply -p2 "%~dp0patches\pantheonrl_overcooked.patch" 2>nul
if errorlevel 1 (
    echo      注意: git apply 返回非零 (可能已应用过或冲突), 检查实际改动...
    git diff --stat
    echo      若上方有改动输出则 patch 已部分应用, 若无输出请手动核查:
    echo        %PANTHEON_DIR%\overcookedgym\overcooked.py
    echo        %PANTHEON_DIR%\pantheonrl\common\agents.py
) else (
    echo      PantheonRL patch 应用成功
)

REM ---- 验证 ----
echo.
echo 验证安装...
%RUN% python -c "import torch, gym, overcooked_ai_py, stable_baselines3; print('OK  torch', torch.__version__, '| gym', gym.__version__)"
if errorlevel 1 (
    echo [错误] 验证失败, 某些包未正确安装
    pause & exit /b 1
)

echo.
echo ============================================
echo  环境搭建完成!
echo.
echo  运行训练:
echo    conda activate %ENV_NAME%
echo    cd /d "%PANTHEON_DIR%"
echo    python trainer.py OvercookedMultiEnv-v0 PPO PPO ^
echo        --env-config "{\"layout_name\":\"cramped_room\"}" ^
echo        --ego-config "{\"verbose\":1}" ^
echo        --total-timesteps 10000 --seed 42
echo ============================================
pause
