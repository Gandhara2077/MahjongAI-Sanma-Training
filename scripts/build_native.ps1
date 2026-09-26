#requires -Version 5.1

param(
    [string]$ProfilePath,
    [string]$PythonExe,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$context = Get-TrainingContext -ProfilePath $ProfilePath -PythonExe $PythonExe -DryRun:$DryRun
$env:PYO3_ENVIRONMENT_SIGNATURE = "$($context.Variant)-py312-$($context.Python)"

$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if ($cargo) {
    $cargoExe = $cargo.Source
}
else {
    $cargoExe = Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe'
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $cargoExe -PathType Leaf)) {
    throw "cargo was not found: $cargoExe"
}

$cargoArguments = @('build', '--locked', '-p', 'libriichi', '--release')
if ($context.Variant -eq 'sanma') {
    $cargoArguments += @('--features', 'sanma')
}
$display = "$cargoExe $($cargoArguments -join ' ')"
$tag = & $context.Python -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw 'could not determine the Python extension tag'
}
$destinationDir = Join-Path $context.Root "Mortal\native\$($context.NativeVariant)"
$destination = Join-Path $destinationDir "libriichi-$($tag.Trim()).pyd"
if ($DryRun) {
    Write-TrainingDryRun "$display -> $destination"
}
else {
    Push-Location $context.MortalRoot
    try {
        & $cargoExe @cargoArguments
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

    $source = Join-Path $context.MortalRoot 'target\release\riichi.dll'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "cargo did not produce the expected extension: $source"
    }
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
    Write-Host "NATIVE_BUILD_OK variant=$($context.Variant) path=$destination"
}
