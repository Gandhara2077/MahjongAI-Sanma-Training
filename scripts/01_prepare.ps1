#requires -Version 5.1

param(
    [string]$ProfilePath,
    [string]$PythonExe,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$context = Get-TrainingContext -ProfilePath $ProfilePath -PythonExe $PythonExe -DryRun:$DryRun
Write-Host "PROFILE variant=$($context.Variant) players=$($context.Players)"
Write-Host "TRAINING_CONFIG $($context.TrainingConfig)"
Write-Host "BASE_MODEL $($context.BaseModel)"

$buildScript = Join-Path $PSScriptRoot 'build_native.ps1'
& $buildScript -ProfilePath $context.ProfilePath -PythonExe $context.Python -DryRun:$DryRun
if ($LASTEXITCODE -ne 0) {
    throw "native build failed: $LASTEXITCODE"
}
Use-ActiveVariant -Context $context -DryRun:$DryRun
if (-not $DryRun) {
    Write-Host "PREPARE_OK variant=$($context.Variant)"
}
