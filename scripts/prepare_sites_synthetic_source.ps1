param([string]$WorkspaceRoot = (Join-Path $PSScriptRoot '..'))
$ErrorActionPreference = 'Stop'
$workspacePath = [System.IO.Path]::GetFullPath($WorkspaceRoot).TrimEnd('\')
$webSource = Join-Path $workspacePath 'web'
$releaseSource = Join-Path $workspacePath 'outputs/sites-synthetic-release/source'
if (-not (Test-Path -LiteralPath (Join-Path $releaseSource '.git'))) {
    throw 'Clone the existing Sites source repository before preparing a release.'
}
$sourceFolders = @('app', 'components', 'lib', 'migrations', 'public', 'tests')
$sourceFiles = @('.env.example', '.gitignore', '.openai/hosting.json', 'THIRD_PARTY_NOTICES.md',
    'package.json', 'package-lock.json', 'next.config.ts', 'tsconfig.json',
    'vite.config.ts', 'vitest.config.ts', 'worker-configuration.d.ts')
$preservedRoots = @('.git', 'node_modules', 'dist', '.wrangler', '.next', '.vinext', 'out', 'coverage')
$preservedFiles = @('next-env.d.ts', 'tsconfig.tsbuildinfo')

function Assert-NoReparse([string]$LiteralPath) {
    $item = Get-Item -LiteralPath $LiteralPath -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Linked input is forbidden: $LiteralPath"
    }
}
function Assert-ReleasePath([string]$LiteralPath) {
    $absolute = [IO.Path]::GetFullPath($LiteralPath)
    if (-not $absolute.StartsWith($releaseSource + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Release write/delete escaped the isolated checkout: $absolute"
    }
    $candidate = Split-Path $absolute -Parent
    while ($candidate.Length -ge $workspacePath.Length) {
        if (Test-Path -LiteralPath $candidate) { Assert-NoReparse $candidate }
        if ($candidate -eq $workspacePath) { break }
        $candidate = Split-Path $candidate -Parent
    }
}
function Assert-SafeName([string]$Relative) {
    $parts = $Relative -split '[/\\]'
    foreach ($part in $parts) {
        if (($part.StartsWith('.env') -and $part -ne '.env.example') -or
            $part.StartsWith('.dev.vars') -or $part -in @('private', 'node_modules', '.git', '.wrangler') -or
            $part -match '\.(sqlite|db|sqlite3)(-|$)') {
            throw "Unsafe source input: $Relative"
        }
    }
}
function Get-SafeFiles([string]$Folder) {
    Assert-NoReparse $Folder
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push($Folder)
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $current -Force) {
            Assert-NoReparse $item.FullName
            if ($item.PSIsContainer) { $pending.Push($item.FullName) } else { $item }
        }
    }
}

# Freeze an allowlist before any mutation, and reject secrets rather than copying them.
$expected = @{}
foreach ($folderName in $sourceFolders) {
    foreach ($file in Get-SafeFiles (Join-Path $webSource $folderName)) {
        $relative = $file.FullName.Substring($webSource.Length + 1)
        Assert-SafeName $relative
        $expected[$relative] = @{ source = $file.FullName; hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash }
    }
}
foreach ($relative in $sourceFiles) {
    $from = Join-Path $webSource $relative
    Assert-NoReparse $from
    Assert-SafeName $relative
    $expected[$relative.Replace('/', '\')] = @{ source = $from; hash = (Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash }
}
Assert-ReleasePath (Join-Path $releaseSource 'package.json')
$existing = @()
foreach ($item in Get-ChildItem -LiteralPath $releaseSource -Force) {
    if ($item.Name -in $preservedRoots -or $item.Name -in $preservedFiles) { continue }
    Assert-NoReparse $item.FullName
    $items = if ($item.PSIsContainer) { @(Get-SafeFiles $item.FullName) } else { @($item) }
    foreach ($file in $items) {
        $relative = $file.FullName.Substring($releaseSource.Length + 1)
        Assert-SafeName $relative
        Assert-ReleasePath $file.FullName
        $existing += $file
    }
}
# Exact synchronization removes stale source files one at a time. There is no
# recursive deletion, and preserved junctions/generated trees are never walked.
foreach ($file in $existing) {
    $relative = $file.FullName.Substring($releaseSource.Length + 1)
    if (-not $expected.ContainsKey($relative)) {
        Assert-ReleasePath $file.FullName
        Remove-Item -LiteralPath $file.FullName -Force
    }
}
foreach ($relative in $expected.Keys) {
    $target = Join-Path $releaseSource $relative
    Assert-ReleasePath $target
    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $expected[$relative].source -Destination $target -Force
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $expected[$relative].hash -or
        (Get-FileHash -LiteralPath $expected[$relative].source -Algorithm SHA256).Hash -ne $expected[$relative].hash) {
        throw "Source changed during preparation: $relative"
    }
}
$dependencyLink = Join-Path $releaseSource 'node_modules'
if (-not (Test-Path -LiteralPath $dependencyLink)) {
    New-Item -ItemType Junction -Path $dependencyLink -Target (Join-Path $webSource 'node_modules') | Out-Null
}
$manifest = @{
    schema_version = 'durf-prepared-source-v1'
    source = $releaseSource
    files = @($expected.Keys | Sort-Object | ForEach-Object { @{ path = $_; sha256 = $expected[$_].hash.ToLowerInvariant() } })
}
$manifestPath = Join-Path (Split-Path $releaseSource -Parent) 'prepared-source-manifest.json'
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Output "Prepared exact isolated Sites source: $releaseSource"

