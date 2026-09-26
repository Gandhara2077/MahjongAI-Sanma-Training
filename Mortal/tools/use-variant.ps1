#requires -Version 5.1
<#
.SYNOPSIS
    Activate one libriichi variant (yonma/sanma) for the Mortal Python stack.

.DESCRIPTION
    Copies the prebuilt native extension Mortal/native/<variant>/libriichi-py<XY>.pyd
    to mortal/libriichi.pyd, which is the single importable
    extension slot. Only one variant can be active in a working tree at a time;
    every variant-dependent Python code path keys off this file.

    Without -Variant the script only prints the current activation state
    (SHA-256 comparison against both native builds).

.EXAMPLE
    .\tools\use-variant.ps1 -Variant yonma
    .\tools\use-variant.ps1 -Variant sanma -PythonExe ..\.venv-rocm\Scripts\python.exe
    .\tools\use-variant.ps1            # status only
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('yonma', 'sanma')]
    [string]$Variant,

    # Python interpreter whose version selects the libriichi-py<XY>.pyd tag.
    # Defaults to MORTAL_PYTHON, then the repository ROCm venv, then PATH.
    [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent          # Mortal/
$destination = Join-Path $root 'mortal\libriichi.pyd'

function Get-Sha256([string]$Path) {
    # .NET implementation: Get-FileHash is not available in every local
    # PowerShell installation (it lives in Microsoft.PowerShell.Utility).
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $bytes = $sha.ComputeHash($stream)
            return ([System.BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Resolve-Python {
    if ($PythonExe) {
        return $PythonExe
    }
    $candidates = @()
    if ($env:MORTAL_PYTHON) {
        $candidates += $env:MORTAL_PYTHON
    }
    $repoRoot = Split-Path $root -Parent
    $candidates += (Join-Path $repoRoot '.venv-rocm\Scripts\python.exe')
    $candidates += (Join-Path $repoRoot '.venv\Scripts\python.exe')
    $candidates += (Join-Path $root '.venv\Scripts\python.exe')
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        $candidates += $command.Source
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    return $null
}

function Get-PythonTag([string]$Exe) {
    $output = & $Exe -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $output -match '^py\d+$') {
        return $Matches[0]
    }
    return $null
}

function Get-ActiveVariant {
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        return 'none (mortal/libriichi.pyd does not exist)'
    }
    $activeHash = Get-Sha256 $destination
    foreach ($name in @('yonma', 'sanma')) {
        $nativeDir = Join-Path $root "native\$name"
        if (-not (Test-Path -LiteralPath $nativeDir -PathType Container)) {
            continue
        }
        foreach ($build in Get-ChildItem -LiteralPath $nativeDir -Filter 'libriichi-py*.pyd') {
            if ((Get-Sha256 $build.FullName) -eq $activeHash) {
                return "$name ($($build.Name))"
            }
        }
    }
    return 'unknown (hash matches neither native build; rebuilt locally?)'
}

Write-Host "Active variant: $(Get-ActiveVariant)"

if (-not $Variant) {
    exit 0
}

$python = Resolve-Python
$tag = $null
if ($python) {
    $tag = Get-PythonTag $python
}

$nativeDir = Join-Path $root "native\$Variant"
$source = $null
if ($tag) {
    $candidate = Join-Path $nativeDir "libriichi-$tag.pyd"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $source = $candidate
    }
}
if (-not $source) {
    # Fall back to a single available build for this variant.
    $builds = @(Get-ChildItem -LiteralPath $nativeDir -Filter 'libriichi-py*.pyd' -ErrorAction SilentlyContinue)
    if ($tag -and $builds.Count -gt 0) {
        $searched = Join-Path $nativeDir "libriichi-$tag.pyd"
        throw "Native extension for Python tag $tag not found: $searched (available: $($builds.Name -join ', ')). Build it per README (PYO3_PYTHON must point at the matching interpreter)."
    }
    if ($builds.Count -eq 1) {
        $source = $builds[0].FullName
    }
    elseif ($builds.Count -eq 0) {
        throw "No prebuilt $Variant native extension under $nativeDir. Build it per README (cargo build -p libriichi --lib --release [--features sanma])."
    }
    else {
        throw "Multiple $Variant builds under $nativeDir and no Python interpreter found to select one; pass -PythonExe."
    }
}

if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
    (Get-Sha256 $source) -eq (Get-Sha256 $destination)) {
    Write-Host "Already active: $Variant ($([IO.Path]::GetFileName($source))); no copy needed."
}
else {
    # Copy via a temp file + replacement: the replacement is atomic on NTFS,
    # so python can never observe a partially written libriichi.pyd.
    $tempDestination = "$destination.tmp-$PID"
    $backupDestination = "$destination.backup-$PID"
    try {
        Copy-Item -LiteralPath $source -Destination $tempDestination -Force
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            [System.IO.File]::Replace($tempDestination, $destination, $backupDestination, $false)
        }
        else {
            [System.IO.File]::Move($tempDestination, $destination)
        }
    }
    finally {
        if (Test-Path -LiteralPath $tempDestination -PathType Leaf) {
            Remove-Item -LiteralPath $tempDestination -Force
        }
        if (Test-Path -LiteralPath $backupDestination -PathType Leaf) {
            Remove-Item -LiteralPath $backupDestination -Force -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath $backupDestination -PathType Leaf) {
                Write-Warning "Previous extension is still in use; deferred cleanup: $backupDestination"
            }
        }
    }
    Write-Host "Activated: $Variant -> mortal/libriichi.pyd (from $source)"
}
exit 0
