param(
    [string]$SourceAgentName = "RllibRingCurrentBestPurePpo30Pct",
    [string]$CandidateAgentName = "",
    [string]$Layout = "ring_tomato_onion_10x6_curriculum_final_onion_counter_pickup_pot_adjacent_top",
    [int]$Iterations = 120,
    [int]$EvalEpisodes = 20,
    [int[]]$EvalSeeds = @(2300, 3201, 3601),
    [int]$ScanEpisodes = 10,
    [int]$ScanSeed = 9300,
    [int]$ScanEveryNthCheckpoint = 1,
    [int]$TrainSeed = 8800,
    [int]$SaveEvery = 20,
    [int]$NumWorkers = 0,
    [string]$RayBaseTempDir = "",
    [string[]]$TrainPolicies = @("ppo_0"),
    [switch]$TrainBothPolicies,
    [double]$LrOverride = 0.0,
    [double]$EntropyStart = 0.2,
    [double]$EntropyEnd = 0.1,
    [double]$OnionDropPenalty = 0.0,
    [double]$DishDropPenalty = 0.0,
    [double]$SoupDropPenalty = 0.0,
    [switch]$SharedPolicy,
    [string]$PythonExe = "",
    [switch]$SkipSourceEval,
    [switch]$UseRayLocalMode
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONPATH = "$PWD;$PWD\src"

if (-not $CandidateAgentName) {
    $CandidateAgentName = "RllibRingBackbonePpoCandidate_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

if ($PythonExe) {
    $PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
} elseif ($env:CONDA_PREFIX) {
    $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    $PythonExe = "python"
}

if (-not $RayBaseTempDir) {
    $RayBaseTempDir = Join-Path $env:TEMP "durf_ray_tmp"
}
New-Item -ItemType Directory -Force -Path $RayBaseTempDir | Out-Null

if ($TrainBothPolicies) {
    $TrainPolicies = @("ppo_0", "ppo_1")
}

function Assert-LastCommandSucceeded {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Get-LatestCheckpoint {
    param([string]$AgentName)
    $AgentDir = Join-Path $RepoRoot "models\rllib_agents\$AgentName\agent"
    if (-not (Test-Path -LiteralPath $AgentDir)) {
        throw "Installed agent directory not found: $AgentDir"
    }
    $Checkpoint = Get-ChildItem -LiteralPath $AgentDir -Directory |
        Where-Object { $_.Name -like "checkpoint_*" -or $_.Name -like "checkpoint-*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $Checkpoint) {
        throw "No checkpoint_* directory found under $AgentDir"
    }
    return $Checkpoint.FullName
}

function Evaluate-Agent {
    param(
        [string]$AgentName,
        [string]$Tag
    )
    foreach ($Seed in $EvalSeeds) {
        $Out = "outputs\baseline_eval_${Tag}_${Seed}.json"
        Write-Host ""
        Write-Host "Evaluating $AgentName on seed $Seed -> $Out" -ForegroundColor DarkCyan
        & $PythonExe -m durf.baseline.evaluate_baseline `
            --agent $AgentName `
            --layout $Layout `
            --episodes $EvalEpisodes `
            --seed $Seed `
            --output $Out `
            --skip-layout-check
        Assert-LastCommandSucceeded "Evaluation $AgentName seed $Seed"
    }
}

function Read-EvalSuccessRate {
    param([string]$Path)
    $EvalJson = Get-Content -Raw -Encoding UTF8 $Path | ConvertFrom-Json
    return [double]$EvalJson.success_rate
}

function New-CheckpointEvalAgent {
    param(
        [System.IO.DirectoryInfo]$RunDir,
        [System.IO.DirectoryInfo]$Checkpoint,
        [string]$CandidateAgentName
    )
    $EvalRoot = Join-Path $RepoRoot "outputs\checkpoint_scan_agents\$CandidateAgentName\$($Checkpoint.Name)"
    $EvalAgentDir = Join-Path $EvalRoot "agent"
    if (Test-Path -LiteralPath $EvalRoot) {
        Remove-Item -LiteralPath $EvalRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $EvalAgentDir | Out-Null

    Copy-Item -LiteralPath (Join-Path $RunDir.FullName "config.pkl") -Destination (Join-Path $EvalRoot "config.pkl") -Force
    Copy-Item -LiteralPath $Checkpoint.FullName -Destination (Join-Path $EvalAgentDir $Checkpoint.Name) -Recurse -Force
    return $EvalAgentDir
}

Write-Host "Checking Python environment..." -ForegroundColor DarkCyan
& $PythonExe -c "import sys; print(sys.executable); import dill, ray; print('dill', dill.__version__); print('ray', ray.__version__)"
Assert-LastCommandSucceeded "Python dependency check"

$SourceCheckpoint = Get-LatestCheckpoint -AgentName $SourceAgentName
Write-Host ""
Write-Host "PPO backbone training candidate" -ForegroundColor Cyan
Write-Host "  source agent:    $SourceAgentName"
Write-Host "  source ckpt:     $SourceCheckpoint"
Write-Host "  candidate agent: $CandidateAgentName"
Write-Host "  layout:          $Layout"
Write-Host "  iterations:      $Iterations"
Write-Host "  train seed:      $TrainSeed"
Write-Host "  num workers:     $NumWorkers"
Write-Host "  ray local mode:  $UseRayLocalMode"
Write-Host "  ray base temp:   $RayBaseTempDir"
Write-Host "  shared policy:   $SharedPolicy"
Write-Host "  train policies:  $($TrainPolicies -join ', ')"
Write-Host "  lr override:     $LrOverride"
Write-Host "  entropy:         $EntropyStart -> $EntropyEnd"
Write-Host "  drop penalties:  onion=$OnionDropPenalty dish=$DishDropPenalty soup=$SoupDropPenalty"
Write-Host "  eval seeds:      $($EvalSeeds -join ', ')"

if (-not $SkipSourceEval) {
    Evaluate-Agent -AgentName $SourceAgentName -Tag "source_${SourceAgentName}"
}

$RayTempDir = Join-Path $RayBaseTempDir "ppo_backbone_${CandidateAgentName}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Write-Host ""
Write-Host "Training candidate PPO backbone..." -ForegroundColor Cyan
$TrainArgs = @(
    "-m", "durf.baseline.train_rllib_agent",
    "--layout", $Layout,
    "--agent-name", $CandidateAgentName,
    "--profile", "paper-ring",
    "--iterations", "$Iterations",
    "--num-workers", "$NumWorkers",
    "--resume-from", $SourceCheckpoint,
    "--save-every", "$SaveEvery",
    "--seed", "$TrainSeed",
    "--entropy-start", "$EntropyStart",
    "--entropy-end", "$EntropyEnd",
    "--onion-drop-penalty", "$OnionDropPenalty",
    "--dish-drop-penalty", "$DishDropPenalty",
    "--soup-drop-penalty", "$SoupDropPenalty",
    "--overwrite",
    "--ray-temp-dir", $RayTempDir
)
if ($LrOverride -gt 0) {
    $TrainArgs += "--lr-override"
    $TrainArgs += "$LrOverride"
}
if (-not $SharedPolicy) {
    $TrainArgs += "--separate-policies"
}
if ($TrainPolicies -and $TrainPolicies.Count -gt 0) {
    $TrainArgs += "--train-policies"
    $TrainArgs += $TrainPolicies
}
if ($UseRayLocalMode) {
    $TrainArgs += "--ray-local-mode"
}
& $PythonExe @TrainArgs
Assert-LastCommandSucceeded "Candidate PPO training"

$RunDir = Get-ChildItem -LiteralPath "outputs\rllib_training" -Directory |
    Where-Object { $_.Name -like "PPO_${Layout}_durf_sp*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $RunDir) {
    throw "Could not find latest PPO run directory for $Layout"
}

Write-Host ""
Write-Host "Scanning saved checkpoints from $($RunDir.FullName)" -ForegroundColor Cyan
$Checkpoints = Get-ChildItem -LiteralPath $RunDir.FullName -Directory |
    Where-Object { $_.Name -like "checkpoint_*" -or $_.Name -like "checkpoint-*" } |
    Sort-Object Name

if (-not $Checkpoints) {
    throw "No checkpoints found under $($RunDir.FullName)"
}

$BestCheckpoint = $null
$BestSuccessRate = -1.0
$Index = 0
foreach ($Checkpoint in $Checkpoints) {
    $Index += 1
    if ($ScanEveryNthCheckpoint -gt 1 -and (($Index - 1) % $ScanEveryNthCheckpoint) -ne 0 -and $Index -ne $Checkpoints.Count) {
        continue
    }
    $EvalAgentDir = New-CheckpointEvalAgent -RunDir $RunDir -Checkpoint $Checkpoint -CandidateAgentName $CandidateAgentName
    $Out = "outputs\baseline_eval_scan_${CandidateAgentName}_$($Checkpoint.Name)_seed${ScanSeed}.json"
    Write-Host ""
    Write-Host "Quick scan $($Checkpoint.Name) -> $Out" -ForegroundColor DarkCyan
    & $PythonExe -m durf.baseline.evaluate_baseline `
        --agent $EvalAgentDir `
        --layout $Layout `
        --episodes $ScanEpisodes `
        --seed $ScanSeed `
        --output $Out `
        --skip-layout-check
    Assert-LastCommandSucceeded "Checkpoint scan $($Checkpoint.Name)"
    $SuccessRate = Read-EvalSuccessRate -Path $Out
    Write-Host "  success_rate=$SuccessRate"
    if ($SuccessRate -gt $BestSuccessRate) {
        $BestSuccessRate = $SuccessRate
        $BestCheckpoint = $Checkpoint
    }
}

if ($BestCheckpoint) {
    Write-Host ""
    Write-Host "Best scanned checkpoint: $($BestCheckpoint.FullName) success_rate=$BestSuccessRate" -ForegroundColor Green
    Write-Host "Installing best scanned checkpoint as $CandidateAgentName" -ForegroundColor Cyan
    & $PythonExe -m durf.baseline.train_rllib_agent `
    --layout $Layout `
    --agent-name $CandidateAgentName `
    --profile paper-ring `
        --iterations 0 `
    --num-workers 0 `
    --ray-local-mode `
        --separate-policies `
        --train-policies $TrainPolicies `
        --resume-from $BestCheckpoint.FullName `
        --entropy-start $EntropyStart `
        --entropy-end $EntropyEnd `
        --onion-drop-penalty $OnionDropPenalty `
        --dish-drop-penalty $DishDropPenalty `
        --soup-drop-penalty $SoupDropPenalty `
        --save-every $SaveEvery `
    --seed $TrainSeed `
    --overwrite `
        --ray-temp-dir (Join-Path $RayBaseTempDir "ppo_backbone_install_${CandidateAgentName}_$(Get-Date -Format 'yyyyMMdd_HHmmss')")
    Assert-LastCommandSucceeded "Install best scanned checkpoint"
}

Evaluate-Agent -AgentName $CandidateAgentName -Tag "candidate_${CandidateAgentName}"

Write-Host ""
Write-Host "Done. Candidate installed at:" -ForegroundColor Green
Write-Host "models\rllib_agents\$CandidateAgentName\agent"
Write-Host ""
Write-Host "Compare the JSON files under outputs\baseline_eval_source_* and outputs\baseline_eval_candidate_*."
Write-Host "Promote this candidate only if it beats $SourceAgentName on matched seeds."
