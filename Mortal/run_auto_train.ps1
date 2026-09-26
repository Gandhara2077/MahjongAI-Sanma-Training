#requires -Version 5.1
<#
.SYNOPSIS
    Launch the Mortal automation pipeline using the active training profile.

.DESCRIPTION
    The profile selects the variant and automation TOML. -Variant and
    -ConfigPath remain available as explicit overrides for advanced use.
#>
param(
    [ValidateSet('yonma', 'sanma')]
    [string]$Variant,

    [string]$ProfilePath,
    [string]$ConfigPath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArguments
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$repoRoot = Split-Path $root -Parent
$script = Join-Path $root 'mortal\automation_pipeline.py'

$pythonCandidates = @()
if ($env:MORTAL_PYTHON) {
    $pythonCandidates += $env:MORTAL_PYTHON
}
$pythonCandidates += (Join-Path $repoRoot '.venv-rocm\Scripts\python.exe')
$pythonCandidates += (Join-Path $repoRoot '.venv\Scripts\python.exe')
$pythonCandidates += (Join-Path $root '.venv\Scripts\python.exe')

$python = $pythonCandidates |
    Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
    Select-Object -First 1
if (-not $python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $python = $command.Source }
}
if (-not $python) {
    throw 'Python was not found. Set MORTAL_PYTHON or create the repository .venv-rocm.'
}

$version = & $python -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
if ($LASTEXITCODE -ne 0 -or $version.Trim() -ne '3.12') {
    throw "Python 3.12 is required, got: $($version.Trim())"
}
$python = (Resolve-Path -LiteralPath $python).Path

if (-not $ProfilePath) {
    $ProfilePath = Join-Path $repoRoot 'config\training-profile.toml'
}
if (-not [System.IO.Path]::IsPathRooted($ProfilePath)) {
    $ProfilePath = Join-Path $repoRoot $ProfilePath
}
if (-not $Variant) {
    $resolver = Join-Path $repoRoot 'scripts\resolve_profile.py'
    $resolvedJson = & $python $resolver --profile $ProfilePath --repo-root $repoRoot 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw (($resolvedJson | Out-String).Trim())
    }
    $resolved = (($resolvedJson -join [Environment]::NewLine) | ConvertFrom-Json)
    $Variant = [string]$resolved.variant
    if (-not $ConfigPath) { $ConfigPath = [string]$resolved.automation_config }
}
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $root "automation\$Variant.toml"
}
$ConfigPath = (Resolve-Path -LiteralPath $ConfigPath -ErrorAction Stop).Path

$env:PYTHONUTF8 = '1'
$env:MORTAL_PYTHON = $python
$env:PYO3_PYTHON = $python
$pythonDir = Split-Path $python -Parent
if (($env:PATH -split ';') -notcontains $pythonDir) {
    $env:PATH = "$pythonDir;$env:PATH"
}
$env:MORTAL_PROFILE = (Resolve-Path -LiteralPath $ProfilePath -ErrorAction Stop).Path
$env:MORTAL_CFG = $ConfigPath
$cacheRoot = Join-Path $repoRoot '.cache\miopen'
$env:MIOPEN_USER_DB_PATH = Join-Path $cacheRoot 'db'
$env:MIOPEN_CUSTOM_CACHE_DIR = Join-Path $cacheRoot 'cache'
New-Item -ItemType Directory -Force -Path $env:MIOPEN_USER_DB_PATH | Out-Null
New-Item -ItemType Directory -Force -Path $env:MIOPEN_CUSTOM_CACHE_DIR | Out-Null

& (Join-Path $root 'tools\use-variant.ps1') -Variant $Variant -PythonExe $python
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location $root
try {
    & $python $script --config $ConfigPath @PipelineArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
