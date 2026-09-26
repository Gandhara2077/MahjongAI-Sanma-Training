#requires -Version 5.1

param(
    [string]$ProfilePath,
    [string]$PythonExe,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$context = Get-TrainingContext -ProfilePath $ProfilePath -PythonExe $PythonExe -DryRun:$DryRun
Use-ActiveVariant -Context $context -DryRun:$DryRun
if (-not $DryRun) { Write-Host "ROCM_OK device=$(Assert-Rocm -Context $context)" }
Invoke-MortalPython -Context $context -Script 'mortal/train.py' -DryRun:$DryRun
if (-not $DryRun) { Write-Host "OFFLINE_TRAIN_OK variant=$($context.Variant)" }
