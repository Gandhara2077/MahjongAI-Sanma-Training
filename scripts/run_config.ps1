#requires -Version 5.1

# Generic config launcher: runs a Mortal entry script (train.py,
# one_vs_two.py, ...) against a config file with the project environment
# (MIOpen cache, native variant, venv).

param(
    [string]$Cfg = 'config/sanma-calib-64x4.toml',
    [string]$Script = 'mortal/train.py'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$context = Get-TrainingContext -SkipBaseModelCheck
Use-ActiveVariant -Context $context
$env:MORTAL_CFG = $Cfg
Write-Host "ROCM_OK device=$(Assert-Rocm -Context $context)"
Write-Host "CALIB_CFG=$Cfg SCRIPT=$Script"
Invoke-MortalPython -Context $context -Script $Script
if ($LASTEXITCODE -ne 0) { throw "entry script exited with $LASTEXITCODE" }
Write-Host 'CALIB_DONE'
