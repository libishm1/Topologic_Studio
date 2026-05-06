# OneDrive Windows Development Notes

The workspace lives under:

```text
C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp
```

This creates practical constraints:

- spaces in paths
- a hyphenated organization folder
- OneDrive sync latency
- globally unavailable `npm`
- multiple Python environments
- bundled Node runtime checked into the project

## Use Explicit Paths

Always quote paths in PowerShell:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio"
```

Avoid relying on global tools unless already configured. The verification run found:

- global `npm` is not recognized.
- global Python 3.13 does not have FastAPI.
- workspace `.venv` has FastAPI but not TopologicPy.
- backend `.venv` has FastAPI, Uvicorn, Pydantic, and TopologicPy.

## Bundled Node Runtime

Use the bundled Node path before invoking npm scripts:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$nodeDir = Resolve-Path "..\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

Why both `$env:Path` and `npm.cmd` matter:

- calling `npm.cmd` directly starts npm.
- npm scripts then invoke `node`.
- without `$nodeDir` on `PATH`, script execution fails with `"node" is not recognized`.

## Backend Virtual Environment

Use the backend `.venv`, not the workspace `.venv`, when testing backend imports:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The backend `.venv` currently has `topologicpy 0.8.93`.

## Build Artifacts

For documentation verification, build output was redirected under the wiki:

```powershell
& "$nodeDir\npm.cmd" run build -- --outDir ../wiki/verification/frontend-dist --emptyOutDir
```

This avoids touching the normal frontend `dist/` output while still proving Vite can build.

## OneDrive-Specific Hygiene

- Keep generated caches out of git where possible: `node_modules`, `__pycache__`, logs, and local build output.
- Avoid long-running dev servers while OneDrive is syncing huge dependency folders.
- Prefer the bundled Node path in README/developer docs because it avoids admin installs.
- Keep `.env` files local and out of documentation.
- Do not commit ZIP copies of Node runtimes unless the project explicitly wants portable offline setup.

## Known Local Runtime Commands

Backend:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$nodeDir = Resolve-Path "..\node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

