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
$profileCheckArguments = @(
    'checks/abi/training_profile_check.py',
    '--profile', $context.ProfilePath,
    '--repo-root', $context.Root,
    '--generic'
)
Invoke-RootPython -Context $context -Script $profileCheckArguments[0] -Arguments $profileCheckArguments[1..($profileCheckArguments.Count - 1)] -DryRun:$DryRun

if ($DryRun) {
    Write-TrainingDryRun "ROCm probe for $($context.Python)"
}
else {
    $device = Assert-Rocm -Context $context
    Write-Host "ROCM_OK device=$device"
}

if ($context.Variant -eq 'sanma') {
    foreach ($script in @(
        'checks/abi/sanma_setup_check.py',
        'checks/smoke/native_sanma_smoke.py',
        'checks/abi/sanma_abi_check.py',
        'checks/smoke/sanma_training_smoke.py'
    )) {
        Invoke-RootPython -Context $context -Script $script -DryRun:$DryRun
    }
}
if (-not $DryRun) {
    Write-Host "VALIDATE_OK variant=$($context.Variant)"
}
