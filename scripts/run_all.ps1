#requires -Version 5.1

param(
    [string]$ProfilePath,
    [string]$PythonExe,
    [ValidateSet('prepare', 'data', 'validate', 'grp', 'offline', 'evaluate', 'export')]
    [string]$FromStep = 'prepare',
    [switch]$SkipPrepare,
    [switch]$SkipData,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$stageNames = @('prepare', 'data', 'validate', 'grp', 'offline', 'evaluate', 'export')
$stageFiles = @{
    prepare = '01_prepare.ps1'
    data = '02_fetch_data.ps1'
    validate = '03_validate.ps1'
    grp = '04_train_grp.ps1'
    offline = '05_train_offline.ps1'
    evaluate = '06_evaluate.ps1'
    export = '07_export.ps1'
}
$startIndex = [Array]::IndexOf($stageNames, $FromStep)
if ($startIndex -lt 0) { throw "Unknown start step: $FromStep" }

for ($index = $startIndex; $index -lt $stageNames.Count; $index++) {
    $stage = $stageNames[$index]
    if (($stage -eq 'prepare' -and $SkipPrepare) -or
        ($stage -eq 'data' -and $SkipData)) {
        Write-Host "SKIP $stage"
        continue
    }
    $stagePath = Join-Path $PSScriptRoot $stageFiles[$stage]
    $stageParameters = @{}
    if ($ProfilePath) { $stageParameters['ProfilePath'] = $ProfilePath }
    if ($PythonExe) { $stageParameters['PythonExe'] = $PythonExe }
    if ($DryRun) { $stageParameters['DryRun'] = $true }
    Write-Host "STAGE $stage"
    & $stagePath @stageParameters
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $stage failed: $LASTEXITCODE"
    }
}
Write-Host "RUN_ALL_OK from=$FromStep"
