param(
    [string]$RepoRoot = "",
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"

function Find-RepoRoot {
    param([string]$StartPath)
    $current = (Resolve-Path -LiteralPath $StartPath).Path
    while ($true) {
        if (
            (Test-Path -LiteralPath (Join-Path $current "pyproject.toml")) -and
            (Test-Path -LiteralPath (Join-Path $current "durf")) -and
            (Test-Path -LiteralPath (Join-Path $current "src"))
        ) {
            return $current
        }
        $parent = Split-Path -Parent $current
        if (-not $parent -or $parent -eq $current) {
            throw "Could not locate the DURF repository root from $StartPath"
        }
        $current = $parent
    }
}

if (-not $RepoRoot) {
    $RepoRoot = Find-RepoRoot -StartPath $PSScriptRoot
} else {
    $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
}

$safeDirectory = $RepoRoot.Replace("\", "/")

function Invoke-RepoGit {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & git -c "safe.directory=$safeDirectory" -C $RepoRoot @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: $($Arguments -join ' ')"
    }
}

Write-Host "=== DURF project snapshot ===" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host "Timestamp:  $([DateTimeOffset]::Now.ToString('o'))"

Write-Host "`n--- Git ---" -ForegroundColor Cyan
Invoke-RepoGit branch --show-current
Invoke-RepoGit log -5 --oneline --decorate
Invoke-RepoGit status --short

Write-Host "`n--- Python contract ---" -ForegroundColor Cyan
Select-String -LiteralPath (Join-Path $RepoRoot "pyproject.toml") -Pattern "requires-python"

Write-Host "`n--- Latest human-AI sessions ---" -ForegroundColor Cyan
$sessions = Join-Path $RepoRoot "outputs\human_ai_sessions"
if (Test-Path -LiteralPath $sessions) {
    Get-ChildItem -LiteralPath $sessions -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 8 Name, LastWriteTime
} else {
    Write-Host "No outputs/human_ai_sessions directory."
}

Write-Host "`n--- Latest Hu models ---" -ForegroundColor Cyan
$huModels = Join-Path $RepoRoot "outputs\hu_models"
if (Test-Path -LiteralPath $huModels) {
    Get-ChildItem -LiteralPath $huModels -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 8 Name, LastWriteTime
} else {
    Write-Host "No outputs/hu_models directory."
}

if ($RunTests) {
    Write-Host "`n--- Focused tests ---" -ForegroundColor Cyan
    $pythonCandidates = @(
        (Join-Path $env:USERPROFILE "Miniconda3\envs\durf310\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\envs\durf310\python.exe")
    )
    $python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $python) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {
            throw "Could not find Python. Activate the durf310 environment first."
        }
        $python = $pythonCommand.Source
    }

    $oldPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = "$RepoRoot;$RepoRoot\src"
        Push-Location $RepoRoot
        & $python -m unittest `
            testing.feedback_attribution_test `
            testing.coordination_test `
            testing.baseline_task_logic_test `
            testing.review_replay_test
        if ($LASTEXITCODE -ne 0) {
            throw "Focused tests failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
        $env:PYTHONPATH = $oldPythonPath
    }
}

Write-Host "`nSnapshot complete." -ForegroundColor Green
