#requires -Version 5.1

Set-StrictMode -Version Latest
$TrainingRepoRoot = Split-Path $PSScriptRoot -Parent

function Resolve-TrainingPython {
    param([string]$PythonExe)

    $candidates = @()
    if ($PythonExe) { $candidates += $PythonExe }
    if ($env:MORTAL_PYTHON) { $candidates += $env:MORTAL_PYTHON }
    $candidates += (Join-Path $TrainingRepoRoot '.venv-rocm\Scripts\python.exe')
    $candidates += (Join-Path $TrainingRepoRoot '.venv\Scripts\python.exe')
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }

    foreach ($candidate in $candidates) {
        if (-not ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf))) {
            continue
        }
        $version = & $candidate -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))' 2>$null
        if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq '3.12') {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw 'Python 3.12 was not found. Create .venv-rocm or set MORTAL_PYTHON.'
}

function Set-TrainingEnvironment {
    param([pscustomobject]$Context)

    $env:MORTAL_PYTHON = $Context.Python
    $env:PYO3_PYTHON = $Context.Python
    $pythonDir = Split-Path $Context.Python -Parent
    if (($env:PATH -split ';') -notcontains $pythonDir) {
        $env:PATH = "$pythonDir;$env:PATH"
    }
    $cacheRoot = Join-Path $Context.Root '.cache\miopen'
    $env:MIOPEN_USER_DB_PATH = Join-Path $cacheRoot 'db'
    $env:MIOPEN_CUSTOM_CACHE_DIR = Join-Path $cacheRoot 'cache'
    New-Item -ItemType Directory -Force -Path $env:MIOPEN_USER_DB_PATH | Out-Null
    New-Item -ItemType Directory -Force -Path $env:MIOPEN_CUSTOM_CACHE_DIR | Out-Null
    $env:MORTAL_CFG = $Context.TrainingConfig
    $env:MORTAL_PROFILE = $Context.ProfilePath
}

function Get-TrainingContext {
    param(
        [string]$ProfilePath,
        [string]$PythonExe,
        [switch]$SkipBaseModelCheck,
        [switch]$DryRun
    )

    $python = Resolve-TrainingPython -PythonExe $PythonExe
    if (-not $ProfilePath) {
        $ProfilePath = Join-Path $TrainingRepoRoot 'config\training-profile.toml'
    }
    elseif (-not [System.IO.Path]::IsPathRooted($ProfilePath)) {
        $ProfilePath = Join-Path $TrainingRepoRoot $ProfilePath
    }
    $profile = (Resolve-Path -LiteralPath $ProfilePath -ErrorAction Stop).Path
    $resolver = Join-Path $TrainingRepoRoot 'scripts\resolve_profile.py'
    $output = & $python $resolver --profile $profile --repo-root $TrainingRepoRoot 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw (($output | Out-String).Trim())
    }
    $resolved = (($output -join [Environment]::NewLine) | ConvertFrom-Json)
    $context = [pscustomobject]@{
        Root = $TrainingRepoRoot
        ProfilePath = $resolved.profile
        Python = $python
        Variant = [string]$resolved.variant
        NativeVariant = [string]$resolved.native_variant
        Players = [int]$resolved.players
        MortalRoot = [string]$resolved.mortal_root
        TrainingConfig = [string]$resolved.training_config
        AutomationConfig = [string]$resolved.automation_config
        BaseModel = [string]$resolved.base_model
        StateFile = [string]$resolved.state_file
        DeploymentFile = [string]$resolved.deployment_file
        GrpStateFile = [string]$resolved.grp_state_file
        LogsRoot = [string]$resolved.logs_root
        DownloadRoot = [string]$resolved.download_root
        RawRoot = [string]$resolved.raw_root
        MjsonRoot = [string]$resolved.mjson_root
        ManifestRoot = [string]$resolved.manifest_root
        StartDate = [string]$resolved.start_date
        EndDate = [string]$resolved.end_date
        Delay = [double]$resolved.delay
        DownloadWorkers = [int]$resolved.download_workers
    }
    if ($context.Players -notin @(3, 4)) {
        throw "Unsupported player count in profile: $($context.Players)"
    }
    if ($context.NativeVariant -notin @('sanma', 'yonma')) {
        throw "Unsupported native variant in profile: $($context.NativeVariant)"
    }
    Set-TrainingEnvironment -Context $context
    # A dry run only previews the pipeline, so it must not require external
    # weights that are never redistributed with the repository.
    if (-not $SkipBaseModelCheck -and -not $DryRun -and -not (Test-Path -LiteralPath $context.BaseModel -PathType Leaf)) {
        throw "Configured base model does not exist: $($context.BaseModel)"
    }
    return $context
}

function Write-TrainingDryRun {
    param([string]$Message)
    Write-Host "[dry-run] $Message"
}

function Use-ActiveVariant {
    param(
        [pscustomobject]$Context,
        [switch]$DryRun
    )

    $script = Join-Path $Context.MortalRoot 'tools\use-variant.ps1'
    if ($DryRun) {
        Write-TrainingDryRun "$script -Variant $($Context.NativeVariant) -PythonExe $($Context.Python)"
        return
    }
    & $script -Variant $Context.NativeVariant -PythonExe $Context.Python
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate native variant $($Context.NativeVariant): $LASTEXITCODE"
    }
}

function Assert-Rocm {
    param([pscustomobject]$Context)

    $device = & $Context.Python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'; print(torch.cuda.get_device_name(0))" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw (($device | Out-String).Trim())
    }
    return ($device | Select-Object -Last 1).ToString().Trim()
}

function Invoke-RootPython {
    param(
        [pscustomobject]$Context,
        [string]$Script,
        [string[]]$Arguments,
        [switch]$DryRun
    )

    $scriptPath = Join-Path $Context.Root $Script
    $display = "$($Context.Python) $scriptPath $($Arguments -join ' ')"
    if ($DryRun) {
        Write-TrainingDryRun $display
        return
    }
    Push-Location $Context.Root
    try {
        & $Context.Python $scriptPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed ($LASTEXITCODE): $display"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-MortalPython {
    param(
        [pscustomobject]$Context,
        [string]$Script,
        [string[]]$Arguments,
        [switch]$DryRun
    )

    $scriptPath = Join-Path $Context.MortalRoot $Script
    $display = "$($Context.Python) $scriptPath $($Arguments -join ' ')"
    if ($DryRun) {
        Write-TrainingDryRun $display
        return
    }
    Push-Location $Context.MortalRoot
    try {
        & $Context.Python $scriptPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed ($LASTEXITCODE): $display"
        }
    }
    finally {
        Pop-Location
    }
}
