#requires -Version 5.1

param(
    [string]$ProfilePath,
    [string]$PythonExe,
    [switch]$DryRun
)

# Backward-compatible name. The active profile, not this filename, selects the
# variant and native feature.
& (Join-Path $PSScriptRoot 'build_native.ps1') @PSBoundParameters
exit $LASTEXITCODE
