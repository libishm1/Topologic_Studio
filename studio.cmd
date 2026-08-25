@echo off
setlocal EnableDelayedExpansion
REM ---------------------------------------------------------------------------
REM  Topologic Studio Next - local launcher
REM
REM    studio          start backend + frontend, open the browser
REM    studio stop     stop both
REM    studio status   show what is running
REM    studio test     run the browser test suite against the running app
REM
REM  Backend and frontend each get their own window; closing a window stops
REM  that service. This file is the canonical launcher; the copy on PATH is a
REM  one-line shim that calls it.
REM ---------------------------------------------------------------------------

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BACKEND=%ROOT%\topologicpy-web-backend"
set "FRONTEND=%ROOT%\topologicpy-web-frontend"
set "PY=%ROOT%\.venv-next\Scripts\python.exe"
set "NODEDIR=%ROOT%\..\TopologicStudio\node-v24.11.1-win-x64"
set "APPURL=http://localhost:5173"
set "APIURL=http://127.0.0.1:8000"

if /i "%~1"=="stop"   goto :stop
if /i "%~1"=="status" goto :status
if /i "%~1"=="test"   goto :test
if /i "%~1"=="help"   goto :usage
if /i "%~1"=="/?"     goto :usage
if not "%~1"=="" goto :usage

REM --------------------------------------------------------------- preflight

if not exist "%PY%" (
  echo [X] No Python environment at %PY%
  echo     Create it with:
  echo       cd /d "%ROOT%"
  echo       python -m venv .venv-next
  echo       .venv-next\Scripts\python -m pip install -r topologicpy-web-backend\requirements-dev.txt
  exit /b 1
)

if not exist "%FRONTEND%\node_modules" (
  echo [X] Frontend dependencies are not installed.
  echo     Run:  cd /d "%FRONTEND%" ^&^& npm install
  exit /b 1
)

REM topologicpy imports fine without its native core and then dies on the first
REM geometry call, so check the backend is genuinely usable before starting it.
echo Checking the TopologicPy backend...
pushd "%BACKEND%"
"%PY%" -c "from app.graphs.topologic_engine import version_info; import sys; i=version_info(); print('    topologicpy', i['topologicpy'], '| core', i['topologic_core'], '| usable', i['usable']); sys.exit(0 if i['usable'] else 1)"
if errorlevel 1 (
  popd
  echo [X] topologicpy is installed but its native backend is not usable.
  echo     Fix:  "%PY%" -m pip install topologic-core==8.0.4
  exit /b 1
)
popd

REM ----------------------------------------------------------------- start up

call :isup "%APIURL%/health" && (
  echo [=] Backend already running.
) || (
  echo [+] Starting backend...
  start "Topologic Studio - backend" /D "%BACKEND%" cmd /c ""%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
)

call :isup "%APPURL%" && (
  echo [=] Frontend already running.
) || (
  echo [+] Starting frontend...
  if exist "%NODEDIR%\node.exe" set "PATH=%NODEDIR%;%PATH%"
  start "Topologic Studio - frontend" /D "%FRONTEND%" cmd /c "npx vite dev --port 5173 --strictPort"
)

echo.
echo Waiting for both services...
set /a tries=0
:wait
set /a tries+=1
if %tries% gtr 60 (
  echo [X] Timed out. Check the two service windows for errors.
  exit /b 1
)
powershell -NoProfile -Command "try{$null=iwr '%APIURL%/health' -UseBasicParsing -TimeoutSec 2; $null=iwr '%APPURL%' -UseBasicParsing -TimeoutSec 2; exit 0}catch{exit 1}" >nul 2>&1
if errorlevel 1 (
  REM ping, not timeout: timeout refuses to run when stdin is redirected.
  ping -n 2 127.0.0.1 >nul
  goto :wait
)

echo.
echo   Topologic Studio is running
echo     app  %APPURL%
echo     api  %APIURL%/docs
echo.
echo   Stop with:  studio stop   (or close the two service windows)
echo.

start "" "%APPURL%"
exit /b 0

REM ------------------------------------------------------------------ actions

:stop
echo Stopping Topologic Studio...
powershell -NoProfile -Command ^
  "foreach($p in 8000,5173){ $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if($c){ foreach($x in $c){ $pr = Get-Process -Id $x.OwningProcess -ErrorAction SilentlyContinue; if($pr){ Write-Host ('  stopped ' + $pr.ProcessName + ' on port ' + $p); Stop-Process -Id $pr.Id -Force } } } else { Write-Host ('  nothing on port ' + $p) } }"
exit /b 0

:status
powershell -NoProfile -Command ^
  "foreach($s in @(@{n='backend ';u='%APIURL%/health'},@{n='frontend';u='%APPURL%'})){ try{ $r=iwr $s.u -UseBasicParsing -TimeoutSec 3; Write-Host ('  ' + $s.n + ' up   ' + $s.u) }catch{ Write-Host ('  ' + $s.n + ' DOWN ' + $s.u) } }"
exit /b 0

:test
call :isup "%APPURL%" || (
  echo [X] The app is not running. Start it with:  studio
  exit /b 1
)
if exist "%NODEDIR%\node.exe" set "PATH=%NODEDIR%;%PATH%"
pushd "%ROOT%"
node tools\browser-test.mjs %2 %3 %4 %5
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%

:usage
echo Topologic Studio Next - local launcher
echo.
echo   studio          start backend + frontend and open the browser
echo   studio stop     stop both services
echo   studio status   show what is running
echo   studio test     run the browser test suite against the running app
exit /b 0

REM ----------------------------------------------------------------- helpers

:isup
powershell -NoProfile -Command "try{$null=iwr '%~1' -UseBasicParsing -TimeoutSec 2; exit 0}catch{exit 1}" >nul 2>&1
exit /b %ERRORLEVEL%
