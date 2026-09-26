#requires -Version 5.1
# Online closed-loop smoke: server + client + trainer for a fixed step budget.
# Kills every process when the trainer exits; exit code mirrors the trainer's.
param(
    [Parameter(Mandatory)][string]$Config,
    [int]$TimeoutSeconds = 3600
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot  # repo root (script lives in scripts/)
$mortal = Join-Path $root 'Mortal'
$python = Join-Path $root '.venv-rocm\Scripts\python.exe'
$absCfg = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $root $Config }

$env:MORTAL_CFG = $absCfg
$env:PYTHONUNBUFFERED = '1'
Set-Location $mortal

$logDir = Join-Path $root 'online_smoke_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$server = Start-Process -FilePath $python -ArgumentList 'mortal/server.py' `
    -WorkingDirectory $mortal -RedirectStandardOutput (Join-Path $logDir 'server.log') `
    -RedirectStandardError (Join-Path $logDir 'server.err.log') -PassThru -NoNewWindow
$client = Start-Process -FilePath $python -ArgumentList 'mortal/client.py' `
    -WorkingDirectory $mortal -RedirectStandardOutput (Join-Path $logDir 'client.log') `
    -RedirectStandardError (Join-Path $logDir 'client.err.log') -PassThru -NoNewWindow

try {
    $trainer = Start-Process -FilePath $python -ArgumentList 'mortal/train.py' `
        -WorkingDirectory $mortal -RedirectStandardOutput (Join-Path $logDir 'trainer.log') `
        -RedirectStandardError (Join-Path $logDir 'trainer.err.log') -PassThru -NoNewWindow
    if (-not $trainer.WaitForExit($TimeoutSeconds * 1000)) {
        throw "trainer timed out after ${TimeoutSeconds}s"
    }
    exit $trainer.ExitCode
}
finally {
    foreach ($p in @($client, $server)) {
        try {
            if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        } catch {}
    }
    # DataLoader workers are children of the trainer; the trainer is already
    # gone, so sweep any leftover python workers spawned for this run.
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -notmatch 'hermes'
    } | ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
    }
}
