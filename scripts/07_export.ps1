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
if (-not $context.StateFile -or -not $context.DeploymentFile) {
    throw 'Selected training config must define control.state_file and control.deployment_file.'
}
$exportArguments = @('mortal/export_model.py', $context.StateFile, $context.DeploymentFile)
Invoke-MortalPython -Context $context -Script $exportArguments[0] -Arguments $exportArguments[1..($exportArguments.Count - 1)] -DryRun:$DryRun
if (-not $DryRun) { Write-Host "EXPORT_OK path=$($context.DeploymentFile)" }
