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
$evaluationScript = if ($context.Players -eq 3) {
    'mortal/one_vs_two.py'
}
else {
    'mortal/one_vs_three.py'
}
Invoke-MortalPython -Context $context -Script $evaluationScript -DryRun:$DryRun
if (-not $DryRun) { Write-Host "EVALUATE_OK variant=$($context.Variant)" }
