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

# Backward-compatible name. The active profile selects sanma or yonma.
& (Join-Path $PSScriptRoot '02_fetch_data.ps1') @PSBoundParameters
exit $LASTEXITCODE
