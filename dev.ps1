<#
    Starts the Next backend and frontend together.

        .\dev.ps1

    Backend  -> http://localhost:8000  (docs at /docs)
    Frontend -> http://localhost:5173
#>
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$Perf
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv-next\Scripts\python.exe'
$node = Join-Path $root '..\TopologicStudio\node-v24.11.1-win-x64'

if (-not (Test-Path $python)) {
    Write-Host 'No .venv-next found. Create it with:' -ForegroundColor Yellow
    Write-Host '  python -m venv .venv-next'
    Write-Host '  .venv-next\Scripts\python -m pip install -r topologicpy-web-backend\requirements-dev.txt'
    exit 1
}

# Fail early and loudly if the native topologic backend is missing: importing
# topologicpy succeeds without it, then every geometry call dies.
& $python -c "from app.graphs.topologic_engine import version_info; import sys; i = version_info(); print('topologicpy', i['topologicpy'], '| core', i['topologic_core'], '| usable', i['usable']); sys.exit(0 if i['usable'] else 1)" 2>&1 |
    ForEach-Object { Write-Host "  $_" }
if ($LASTEXITCODE -ne 0) {
    Write-Host 'topologicpy is installed but its native backend is not usable.' -ForegroundColor Red
    Write-Host 'Install it with:  .venv-next\Scripts\python -m pip install topologic-core==8.0.4'
    exit 1
}

if (Test-Path $node) { $env:PATH = "$node;$env:PATH" }
if ($Perf) { $env:PERF_LOG = '1' }

$backend = Start-Process -PassThru -NoNewWindow -FilePath $python `
    -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--reload', '--port', $BackendPort) `
    -WorkingDirectory (Join-Path $root 'topologicpy-web-backend')

Write-Host "backend  -> http://localhost:$BackendPort" -ForegroundColor Green
Write-Host "frontend -> http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host 'Ctrl+C stops both.'

try {
    Push-Location (Join-Path $root 'topologicpy-web-frontend')
    if (-not (Test-Path 'node_modules')) { npm install }
    npm run dev -- --port $FrontendPort
}
finally {
    Pop-Location
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
}
