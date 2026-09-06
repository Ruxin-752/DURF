# Batch offline attribution pipeline for sim sessions.
# For each sim session directory, runs:
#   1. session_converter  (CSV -> trajectory.jsonl + feedback_events.jsonl)
#   2. generate_candidate_events
#   3. LLM semantic attribution (DeepSeek), rule baseline as labelled stand-in
#   4. hu_dataset_builder + probe_state_detector
#
# 2026-09-05: the LLM is ON.  Until now this script ran the keyword baseline
# only ("sim feedback is template-based with known keywords"), which is why
# 4 538 of the 4 554 pairwise labels in the corpus never touched the LLM the
# paper is about.  Every label now carries an `attributor` field; to run the
# baseline deliberately, pass --no-llm to demo_offline_attribution and expect
# attributor=rule_baseline everywhere.  Requires DEEPSEEK_API_KEY.

$ErrorActionPreference = "Continue"

$sessionBase = "outputs/human_ai_sessions"
$allDirs = Get-ChildItem -Path $sessionBase -Directory | Sort-Object Name

$total = 0
$done = 0
$failed = @()
$withFallbacks = @()

# Find sim sessions from the batch run (data_source = synthetic_sim_human)
Write-Host "Scanning for sim sessions ..."
foreach ($dir in $allDirs) {
    $metaFile = Join-Path $dir.FullName "session_metadata.json"
    if (-not (Test-Path $metaFile)) { continue }
    try {
        $meta = Get-Content $metaFile -Raw | ConvertFrom-Json
        if ($meta.data_source -eq "synthetic_sim_human") {
            # Check if already processed
            if (Test-Path (Join-Path $dir.FullName "hu_subgoal_preferences.jsonl")) {
                continue
            }
            $total++
        }
    } catch { }
}

if ($total -eq 0) {
    Write-Host "All sim sessions already have hu_subgoal_preferences.jsonl. Nothing to do."
    exit 0
}

Write-Host "Processing $total unprocessed sim sessions ..."
Write-Host ""

foreach ($dir in $allDirs) {
    $metaFile = Join-Path $dir.FullName "session_metadata.json"
    if (-not (Test-Path $metaFile)) { continue }
    $meta = Get-Content $metaFile -Raw | ConvertFrom-Json
    if ($meta.data_source -ne "synthetic_sim_human") { continue }
    
    # Skip if already done
    $prefsFile = Join-Path $dir.FullName "hu_subgoal_preferences.jsonl"
    if (Test-Path $prefsFile) { continue }
    
    $done++
    $sessionId = $dir.Name
    Write-Host "[$done/$total] $sessionId ..." -NoNewline

    $result = & python -m durf.feedback_attribution.demo_offline_attribution `
        --session $dir.FullName `
        --user-id "PILOT01" `
        --lookback-steps 30 `
        2>&1

    if ($LASTEXITCODE -eq 0) {
        # Also run probe state detector
        & python -m durf.feedback_attribution.probe_state_detector `
            --session $dir.FullName --no-convert 2>&1 | Out-Null
        # An LLM outage is not a success: surface it and count it.
        $warn = $result | Select-String -Pattern "WARNING: LLM failed"
        if ($warn) {
            Write-Host " OK (with LLM fallbacks)"
            Write-Host "    $($warn.Line)"
            $withFallbacks += $sessionId
        } else {
            Write-Host " OK"
        }
    } else {
        Write-Host " FAILED"
        $failed += $sessionId
    }
}

Write-Host ""
Write-Host "=== Attribution pipeline complete ==="
Write-Host "Processed: $done sessions"
if ($failed.Count -gt 0) {
    Write-Host "FAILED: $($failed.Count) sessions"
    foreach ($sid in $failed) {
        Write-Host "  $sid"
    }
} else {
    Write-Host "All sessions succeeded."
}
if ($withFallbacks.Count -gt 0) {
    Write-Host "LLM FALLBACKS in $($withFallbacks.Count) sessions (labels marked rule_fallback_after_llm_error):"
    foreach ($sid in $withFallbacks) {
        Write-Host "  $sid"
    }
}
