#requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ConfigPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputRoot,

    [int[]]$WorkerCounts = @(16, 32, 64),
    [int]$GamesPerIter = 600,
    [string]$PythonExe,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

if ($GamesPerIter -le 0 -or ($GamesPerIter % 3) -ne 0) {
    throw 'GamesPerIter must be positive and divisible by 3.'
}
if (-not $WorkerCounts -or @($WorkerCounts | Where-Object { $_ -le 0 }).Count -gt 0) {
    throw 'WorkerCounts must contain only positive integers.'
}
if (@($WorkerCounts | Select-Object -Unique).Count -ne $WorkerCounts.Count) {
    throw 'WorkerCounts must not contain duplicates.'
}

$config = (Resolve-Path -LiteralPath $ConfigPath -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "ConfigPath is not a file: $ConfigPath"
}

if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $benchmarkRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}
else {
    $benchmarkRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $OutputRoot))
}
if (Test-Path -LiteralPath $benchmarkRoot -PathType Leaf) {
    throw "OutputRoot is a file: $benchmarkRoot"
}
if (Test-Path -LiteralPath $benchmarkRoot -PathType Container) {
    if (@(Get-ChildItem -LiteralPath $benchmarkRoot -Force).Count -gt 0) {
        throw "OutputRoot must be new or empty; refusing to overwrite: $benchmarkRoot"
    }
}

$context = Get-TrainingContext -PythonExe $PythonExe
if ($context.Players -ne 3 -or $context.NativeVariant -ne 'sanma') {
    throw 'The active training profile must select the sanma native variant.'
}
$env:MORTAL_CFG = $config
Use-ActiveVariant -Context $context -DryRun:$DryRun
if (-not $DryRun) {
    Write-Host "ROCM_OK device=$(Assert-Rocm -Context $context)"
    if (-not (Test-Path -LiteralPath $benchmarkRoot -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $benchmarkRoot | Out-Null
    }
}

$evaluator = Join-Path $context.MortalRoot 'mortal\one_vs_two.py'
$results = @()
foreach ($workerCount in $WorkerCounts) {
    $logDir = Join-Path $benchmarkRoot ("workers-{0}" -f $workerCount)
    $arguments = @(
        '--max-workers', [string]$workerCount,
        '--games-per-iter', [string]$GamesPerIter,
        '--iters', '1',
        '--log-dir', $logDir
    )

    if ($DryRun) {
        Write-TrainingDryRun "$($context.Python) $evaluator $($arguments -join ' ') MORTAL_CFG=$config"
        continue
    }

    if (Test-Path -LiteralPath $logDir -PathType Leaf) {
        throw "Worker log path is a file: $logDir"
    }
    if (Test-Path -LiteralPath $logDir -PathType Container) {
        if (@(Get-ChildItem -LiteralPath $logDir -Force).Count -gt 0) {
            throw "Worker log directory must be new or empty: $logDir"
        }
    }
    else {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = $null
    Push-Location $context.MortalRoot
    try {
        & $context.Python $evaluator @arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $stopwatch.Stop()
        Pop-Location
    }

    $actualLogs = if (Test-Path -LiteralPath $logDir -PathType Container) {
        @(Get-ChildItem -LiteralPath $logDir -Filter '*.json.gz' -File).Count
    }
    else {
        0
    }
    if ($exitCode -ne 0 -or $actualLogs -ne $GamesPerIter) {
        throw "Worker count $workerCount failed: exit_code=$exitCode expected_logs=$GamesPerIter actual_logs=$actualLogs"
    }

    $results += [pscustomobject]@{
        worker_count    = [int]$workerCount
        elapsed_seconds = [double]$stopwatch.Elapsed.TotalSeconds
        expected_logs   = [int]$GamesPerIter
        actual_logs     = [int]$actualLogs
        exit_code       = [int]$exitCode
    }
}

if ($DryRun) {
    return
}

$csvPath = Join-Path $benchmarkRoot 'benchmark.csv'
$results | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "EVAL_WORKER_BENCHMARK_OK path=$csvPath"
