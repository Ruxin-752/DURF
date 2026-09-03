# Sim session batch generator for Hu_general data collection.
# Runs 4 personas x 20 seeds = 80 sessions.
# Training seeds: 0-15 (64 sessions), Test seeds: 20-23 (16 sessions)

$ErrorActionPreference = "Continue"
$condaActivate = "conda activate durf310"
$venv = '$env:PYTHONPATH="$PWD;$PWD\src"'

$personas = @("cooperative", "selfish", "polite", "lenient")
$trainSeeds = 0..15
$testSeeds = 20..23
$allSeeds = $trainSeeds + $testSeeds

$total = $personas.Count * $allSeeds.Count
$done = 0
$failed = @()

Write-Host "=== Starting $total sim sessions ==="
Write-Host "Personas: $personas"
Write-Host "Train seeds: $trainSeeds"
Write-Host "Test seeds: $testSeeds"
Write-Host ""

foreach ($seed in $allSeeds) {
    foreach ($persona in $personas) {
        $done++
        Write-Host "[$done/$total] persona=$persona seed=$seed ..." -NoNewline
        
        $cmd = "conda activate durf310; `$env:PYTHONPATH='$PWD;$PWD\src'; python -m durf.group_a.sim_session --sim-human $persona --seed $seed --horizon 800 --episodes 2 --sim-seed $seed --sim-feedback-min-gap 15 2>&1 | Out-Null; exit `$LASTEXITCODE"
        
        $result = powershell -Command $cmd 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host " FAILED"
            $failed += "persona=$persona seed=$seed"
        } else {
            Write-Host " OK"
        }
    }
}

Write-Host ""
Write-Host "=== Complete: $done sessions ==="
if ($failed.Count -gt 0) {
    Write-Host "FAILED sessions:"
    $failed | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "All sessions succeeded."
}
