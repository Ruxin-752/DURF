param(
  [int]$Iterations = 180
)

$ErrorActionPreference = "Continue"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Python = "C:\Users\my185\Miniconda3\envs\durf310\python.exe"
$LogDir = Join-Path $Repo "outputs\background_training"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "bottom_mid2_$Stamp.log"

& $Python -m durf.baseline.train_rllib_agent `
  --layout ring_tomato_onion_10x6_curriculum_final_onion_held_target_bottom_mid_2 `
  --agent-name RllibRingFinalOnionHeldTargetBottomMid2Candidate `
  --iterations $Iterations `
  --horizon 160 `
  --num-workers 0 `
  --ray-local-mode `
  --overwrite `
  --save-every 20 `
  --seed 740 `
  --resume-from models\rllib_agents\RllibRingFinalOnionHeldTargetMidLeft3Candidate\agent\checkpoint_002800 `
  --lr 0.000025 `
  --gamma 0.96 `
  --gae-lambda 0.6 `
  --vf-loss-coeff 0.02 `
  --kl-coeff 0.1 `
  --clip-param 0.08 `
  --grad-clip 0.5 `
  --reward-shaping-horizon 4500000 `
  --no-use-phi `
  --tomato-disp-distance-reward 0 `
  --tomato-to-pot-distance-reward 0 `
  --onion-to-pot-distance-reward 35 `
  --dish-disp-distance-reward 18 `
  --pot-distance-reward 25 `
  --soup-distance-reward 20 `
  --placement-in-pot-reward 100 `
  --tomato-pickup-reward 0 `
  --onion-pickup-reward 0 `
  --dish-pickup-reward 30 `
  --ready-dish-pickup-reward 50 `
  --soup-pickup-reward 80 `
  *> $LogPath

exit $LASTEXITCODE
