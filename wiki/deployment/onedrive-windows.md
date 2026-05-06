# OneDrive Windows Development

The project is inside:

```text
C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio
```

The path contains spaces and is synced by OneDrive. Treat shell commands and generated files accordingly.

## Practical Rules

- Always quote full paths in PowerShell.
- Prefer running commands from the project subdirectory instead of passing long paths.
- Use `-LiteralPath` for paths with spaces when reading files in PowerShell.
- Avoid writing build outputs into source folders unless needed. For verification, use `wiki/verification/`.
- Avoid editing or deleting OneDrive placeholder/runtime files unless the task explicitly asks for cleanup.

## Bundled Node Runtime

Global `npm` may not be available. Use the bundled Node folder:

```powershell
$env:PATH=(Resolve-Path ..\node-v24.11.1-win-x64).Path + ';' + $env:PATH
npm.cmd run build
```

From the repo root:

```powershell
$env:PATH=(Resolve-Path .\node-v24.11.1-win-x64).Path + ';' + $env:PATH
```

## Backend Local Runtime

Use the backend virtual environment:

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If port `8000` is already occupied, choose another port and set `VITE_API_BASE` accordingly.

## Logs

Local logs present at repo root:

- `frontend-dev.log`: Vite dev server started on `http://localhost:5173/`.
- `uvicorn-out.log`: health request succeeded and an old `/upload-topology` request returned `500`.
- `uvicorn-err.log`: includes an older stack trace for TopologicPy JSON parsing.

Do not treat old logs as current behavior without rerunning the route.

