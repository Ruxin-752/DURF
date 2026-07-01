param(
    [int]$Stage0Iterations = 100,
    [int]$Stage1Iterations = 120,
    [int]$Stage2Iterations = 180,
    [int]$Stage3Iterations = 220,
    [int]$Stage4Iterations = 300,
    [int]$Stage5Iterations = 500,
    [int]$EvalEpisodes = 10,
    [int]$Seed = 0,
    [int]$StartStage = 0,
    [int]$MaxStage = 5,
    [int]$SaveEvery = 20,
    [string]$PythonExe = "",
    [switch]$SkipEval,
    [switch]$StrictProgressChecks
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONPATH = "$PWD;$PWD\src"

if ($PythonExe) {
    $PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
} elseif ($env:CONDA_PREFIX) {
    $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    $PythonExe = "python"
}

$AgentName = "RllibRingTomatoOnion10x6SP"
$CommonArgs = @(
    "--agent-name", $AgentName,
    "--profile", "paper-ring",
    "--num-workers", "0",
    "--ray-local-mode",
    "--use-phi",
    "--overwrite",
    "--save-every", "$SaveEvery",
    "--seed", "$Seed"
)

function Assert-LastCommandSucceeded {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Stop-RayIfAvailable {
    # Ray 2.2's Windows CLI can crash while importing click in this env.
    # Keep this as a no-op and isolate stages with fresh temp dirs instead.
}

Write-Host "Checking Python environment..." -ForegroundColor DarkCyan
& $PythonExe -c "import sys; print(sys.executable); import dill, ray; print('dill', dill.__version__); print('ray', ray.__version__)"
Assert-LastCommandSucceeded "Python dependency check"

function Train-Stage {
    param(
        [string]$Name,
        [string]$Layout,
        [int]$Iterations,
        [bool]$Resume,
        [double]$TomatoPickupReward = 0.0,
        [double]$OnionPickupReward = 0.0,
        [double]$DishPickupReward = 3.0,
        [double]$ReadyDishPickupReward = 0.0,
        [double]$SoupPickupReward = 5.0,
        [string]$ProgressEvent = "",
        [int]$MinProgressEventCount = 1
    )

    Write-Host ""
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    Write-Host "Layout: $Layout"
    Write-Host "Iterations: $Iterations"
    Write-Host "Resume installed checkpoint: $Resume"
    Write-Host "Pickup shaping: tomato=$TomatoPickupReward onion=$OnionPickupReward"
    Write-Host "Serving shaping: dish=$DishPickupReward ready_dish=$ReadyDishPickupReward soup=$SoupPickupReward"

    Stop-RayIfAvailable
    $RayTempDir = "outputs\ray_tmp\curriculum_${Layout}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

    $Args = @(
        "-m", "durf.baseline.train_rllib_agent",
        "--layout", $Layout,
        "--iterations", "$Iterations",
        "--tomato-pickup-reward", "$TomatoPickupReward",
        "--onion-pickup-reward", "$OnionPickupReward",
        "--dish-pickup-reward", "$DishPickupReward",
        "--ready-dish-pickup-reward", "$ReadyDishPickupReward",
        "--soup-pickup-reward", "$SoupPickupReward",
        "--ray-temp-dir", $RayTempDir
    ) + $CommonArgs

    if ($Resume) {
        $Args += "--resume-installed"
    }

    & $PythonExe @Args
    Assert-LastCommandSucceeded "Training $Layout"

    if (-not $SkipEval) {
        $EvalOut = "outputs\baseline_eval_${Layout}_latest.json"
        Write-Host ""
        Write-Host "Evaluating $Layout -> $EvalOut" -ForegroundColor DarkCyan
        Stop-RayIfAvailable
        & $PythonExe -m durf.baseline.evaluate_baseline `
            --agent $AgentName `
            --layout $Layout `
            --episodes $EvalEpisodes `
            --seed 200 `
            --output $EvalOut
        Assert-LastCommandSucceeded "Evaluation $Layout"

        $EvalJson = Get-Content -Raw -Encoding UTF8 $EvalOut | ConvertFrom-Json
        Write-Host ""
        Write-Host "Progress summary for $Layout" -ForegroundColor Yellow
        Write-Host "  mean_reward: $($EvalJson.mean_reward)"
        Write-Host "  success_rate: $($EvalJson.success_rate)"
        Write-Host "  tomato_pickup: $($EvalJson.event_counts.tomato_pickup)"
        Write-Host "  potting_tomato: $($EvalJson.event_counts.potting_tomato)"
        Write-Host "  onion_pickup: $($EvalJson.event_counts.onion_pickup)"
        Write-Host "  potting_onion: $($EvalJson.event_counts.potting_onion)"
        Write-Host "  dish_pickup: $($EvalJson.event_counts.dish_pickup)"
        Write-Host "  soup_pickup: $($EvalJson.event_counts.soup_pickup)"
        Write-Host "  soup_delivery: $($EvalJson.event_counts.soup_delivery)"

        if ($ProgressEvent) {
            $Observed = 0
            if ($EvalJson.event_counts.PSObject.Properties.Name -contains $ProgressEvent) {
                $Observed = [int]$EvalJson.event_counts.$ProgressEvent
            }
            if ($Observed -lt $MinProgressEventCount) {
                $Message = "Progress check failed for ${Layout}: $ProgressEvent=$Observed, expected >= $MinProgressEventCount"
                if ($StrictProgressChecks) {
                    throw $Message
                }
                Write-Host "WARNING: $Message" -ForegroundColor Red
            } else {
                Write-Host "Progress check passed: $ProgressEvent=$Observed" -ForegroundColor Green
            }
        }
    }
}

if ($StartStage -le 0 -and $MaxStage -ge 0) {
    Train-Stage `
        -Name "Stage 0 / micro kitchen, learn tomato pickup and potting" `
        -Layout "ring_tomato_onion_10x6_curriculum_micro" `
        -Iterations $Stage0Iterations `
        -Resume $false `
        -TomatoPickupReward 1.0 `
        -OnionPickupReward 0.0 `
        -DishPickupReward 0.0 `
        -ReadyDishPickupReward 0.0 `
        -SoupPickupReward 5.0 `
        -ProgressEvent "potting_tomato" `
        -MinProgressEventCount 1
}

if ($StartStage -le 1 -and $MaxStage -ge 1) {
    Train-Stage `
        -Name "Stage 1 / micro kitchen, learn dish pickup and delivery chain" `
        -Layout "ring_tomato_onion_10x6_curriculum_micro_delivery" `
        -Iterations $Stage1Iterations `
        -Resume $true `
        -TomatoPickupReward 0.5 `
        -OnionPickupReward 0.0 `
        -DishPickupReward 0.0 `
        -ReadyDishPickupReward 6.0 `
        -SoupPickupReward 10.0 `
        -ProgressEvent "soup_delivery" `
        -MinProgressEventCount 1
}

if ($StartStage -le 2 -and $MaxStage -ge 2) {
    Train-Stage `
        -Name "Stage 2 / open delivery bridge, preserve full serving chain" `
        -Layout "ring_tomato_onion_10x6_curriculum_open_delivery" `
        -Iterations $Stage2Iterations `
        -Resume $true `
        -TomatoPickupReward 0.5 `
        -OnionPickupReward 0.0 `
        -DishPickupReward 0.0 `
        -ReadyDishPickupReward 6.0 `
        -SoupPickupReward 10.0 `
        -ProgressEvent "soup_pickup" `
        -MinProgressEventCount 1
}

if ($StartStage -le 3 -and $MaxStage -ge 3) {
    Train-Stage `
        -Name "Stage 3 / easy kitchen, dispenser farther away" `
        -Layout "ring_tomato_onion_10x6_curriculum_easy" `
        -Iterations $Stage3Iterations `
        -Resume $true `
        -TomatoPickupReward 0.8 `
        -OnionPickupReward 0.0 `
        -DishPickupReward 0.0 `
        -ReadyDishPickupReward 6.0 `
        -SoupPickupReward 6.0 `
        -ProgressEvent "potting_tomato" `
        -MinProgressEventCount 1
}

if ($StartStage -le 4 -and $MaxStage -ge 4) {
    Train-Stage `
        -Name "Stage 4 / partial corridor, tomato-only recipe" `
        -Layout "ring_tomato_onion_10x6_curriculum_corridor" `
        -Iterations $Stage4Iterations `
        -Resume $true `
        -TomatoPickupReward 0.6 `
        -OnionPickupReward 0.0 `
        -DishPickupReward 0.0 `
        -ReadyDishPickupReward 6.0 `
        -SoupPickupReward 6.0 `
        -ProgressEvent "potting_tomato" `
        -MinProgressEventCount 1
}

if ($StartStage -le 5 -and $MaxStage -ge 5) {
    Train-Stage `
        -Name "Stage 5 / target ring geometry, tomato-tomato-onion recipe" `
        -Layout "ring_tomato_onion_10x6" `
        -Iterations $Stage5Iterations `
        -Resume $true `
        -TomatoPickupReward 0.2 `
        -OnionPickupReward 0.4 `
        -DishPickupReward 0.0 `
        -ReadyDishPickupReward 6.0 `
        -SoupPickupReward 6.0 `
        -ProgressEvent "potting_onion" `
        -MinProgressEventCount 1
}

Write-Host ""
Write-Host "Curriculum training finished. Final installed agent:" -ForegroundColor Green
Write-Host "models\rllib_agents\$AgentName\agent"
Write-Host ""
Write-Host "Final manual evaluation command:"
Write-Host "python -m durf.baseline.evaluate_baseline --agent $AgentName --layout ring_tomato_onion_10x6 --episodes 20 --seed 300 --output outputs\baseline_eval_ring_curriculum_final.json"
