# Verification Status

Last checked: 2026-05-06.

## Required Browser Evidence For Future Runs

Any future browser verification must record:

- headless or headful mode
- tested URL
- browser console summary
- page errors
- failed requests
- API statuses
- screenshot paths
- human review decision if visual or research-validity judgment was needed

Use [headless browser console testing](../testing/headless-browser-console-testing.md).

## Backend Import And Routes

Command:

```powershell
cd TopologicStudio\topologicpy-web-backend
.\.venv\Scripts\python.exe -c "from app.main import app; print(app.title, app.version); [print(r.path, sorted(list(getattr(r,'methods',[])))) for r in app.routes if hasattr(r,'methods')]"
```

Result:

- Import succeeded.
- App title/version: `TopologicPy Web Backend 0.1.0`.
- Expected routes are registered.

## Backend Versions

Command:

```powershell
.\.venv\Scripts\python.exe -c "import fastapi, uvicorn, pydantic, topologicpy; ..."
```

Result:

- FastAPI `0.124.0`
- Uvicorn `0.38.0`
- Pydantic `2.12.5`
- TopologicPy `0.8.93`

## Frontend Lint

Command:

```powershell
cd TopologicStudio\topologicpy-web-frontend
$env:PATH=(Resolve-Path ..\node-v24.11.1-win-x64).Path + ';' + $env:PATH
npm.cmd run lint
```

Result:

- Failed with 9 errors and 4 warnings.
- See [open risks](open-risks.md).

## Frontend Build

Command:

```powershell
npm.cmd run build -- --outDir ../wiki/verification/frontend-dist --emptyOutDir
```

Result:

- Build succeeded.
- 89 modules transformed.
- Output written to `wiki/verification/frontend-dist/`.
- Large chunk warning remains.

## PDF Parse

Command:

```powershell
python -c "import fitz; ..."
```

Result:

- `the synergy of non manifold topology.pdf` has 10 pages and 0 extractable text characters.
- OCR required.
