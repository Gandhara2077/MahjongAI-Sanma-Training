param(
    [string]$MahjongCopilotRoot = '',
    [string]$PythonVersion = '3.12'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($MahjongCopilotRoot)) {
    $downloadRoot = 'C:\Users\Administrator\Downloads'
    $candidate = Get-ChildItem -LiteralPath $downloadRoot -Directory |
        Where-Object { $_.Name -like 'MahjongCopilot*' } |
        Select-Object -First 1
    if ($null -eq $candidate) {
        throw "MahjongCopilot directory not found under $downloadRoot; pass -MahjongCopilotRoot explicitly"
    }
    $MahjongCopilotRoot = $candidate.FullName
}

$modelSource = Join-Path $MahjongCopilotRoot 'models\mortal.pth'
$referenceFilename = "libriichi3p-$PythonVersion-x86_64-pc-windows-msvc.pyd"
$referenceSource = Join-Path $MahjongCopilotRoot (Join-Path 'libriichi3p' $referenceFilename)
$modelPath = Join-Path $root '.cache\models\mortal3p.pth'
$referencePath = Join-Path $root (Join-Path '.cache\libriichi3p' $referenceFilename)
$manifestPath = Join-Path $root 'artifacts\sanma-assets.json'

foreach ($path in @($modelSource, $referenceSource)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required sanma asset not found: $path"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $modelPath), (Split-Path $referencePath), (Split-Path $manifestPath) | Out-Null
Copy-Item -LiteralPath $modelSource -Destination $modelPath -Force
Copy-Item -LiteralPath $referenceSource -Destination $referencePath -Force

function Get-AssetRecord([string]$path) {
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $path
    [ordered]@{
        path = $path.Substring($root.Length + 1).Replace('\', '/')
        bytes = (Get-Item -LiteralPath $path).Length
        sha256 = $hash.Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    variant = 'sanma'
    reference = [ordered]@{
        python = $PythonVersion
        filename = (Split-Path -Leaf $referencePath)
    }
    abi = [ordered]@{
        version = 4
        obs_shape = @(775, 34)
        action_space = 44
    }
    model = Get-AssetRecord $modelPath
    reference_extension = Get-AssetRecord $referencePath
}

$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Output "SANMA_ASSETS_READY $manifestPath"
