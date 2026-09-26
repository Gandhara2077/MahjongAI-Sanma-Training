#requires -Version 5.1

param(
    [string]$ProfilePath,
    [string]$PythonExe,
    [string]$StartDate,
    [string]$EndDate,
    [double]$Delay = -1,
    [switch]$Force,
    [switch]$Quiet,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$context = Get-TrainingContext -ProfilePath $ProfilePath -PythonExe $PythonExe -SkipBaseModelCheck

if (-not $StartDate) { $StartDate = $context.StartDate }
if (-not $EndDate) { $EndDate = $context.EndDate }
if ($StartDate -notmatch '^\d{8}$' -or $EndDate -notmatch '^\d{8}$') {
    throw 'StartDate and EndDate must use YYYYMMDD.'
}
if ([DateTime]::ParseExact($EndDate, 'yyyyMMdd', $null) -lt
    [DateTime]::ParseExact($StartDate, 'yyyyMMdd', $null)) {
    throw 'EndDate must not be earlier than StartDate.'
}
if ($Delay -lt 0) { $Delay = $context.Delay }

$downloadArguments = @(
    'koromo/download_tenhou.py',
    '--variant', $context.Variant,
    '--start-date', $StartDate,
    '--end-date', $EndDate,
    '--output', $context.DownloadRoot,
    '--delay', $Delay.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--workers', [string]$context.DownloadWorkers
)
if ($Quiet) { $downloadArguments += '--quiet' }
Invoke-RootPython -Context $context -Script $downloadArguments[0] -Arguments $downloadArguments[1..($downloadArguments.Count - 1)] -DryRun:$DryRun

$convertArguments = @(
    'koromo/convert_mjlog_to_mjson.py',
    '--variant', $context.Variant,
    '--input', $context.RawRoot,
    '--output', $context.MjsonRoot
)
if ($Force) { $convertArguments += '--force' }
if ($Quiet) { $convertArguments += '--quiet' }
Invoke-RootPython -Context $context -Script $convertArguments[0] -Arguments $convertArguments[1..($convertArguments.Count - 1)] -DryRun:$DryRun

$manifest = Join-Path $context.ManifestRoot "$($context.Variant)-$StartDate-$EndDate.json"
$checkArguments = @(
    'checks/data/sanma_data_check.py',
    '--variant', $context.Variant,
    '--raw-root', $context.RawRoot,
    '--mjson-root', $context.MjsonRoot,
    '--start-date', $StartDate,
    '--end-date', $EndDate,
    '--manifest', $manifest
)
Invoke-RootPython -Context $context -Script $checkArguments[0] -Arguments $checkArguments[1..($checkArguments.Count - 1)] -DryRun:$DryRun
if (-not $DryRun) {
    Write-Host "DATA_FETCH_OK variant=$($context.Variant) range=$StartDate..$EndDate manifest=$manifest"
}
